"""Blocks, statements, chain assertions and guarded asserts."""

from __future__ import annotations

from dataclasses import field, fields, replace
from math import inf

from optimization_bridge import LoopReplay
from shape_domain import REPEAT, LinkGraph
from value_domain import Domain

from .syntax import Node
from .typesys import (AliasResult, Array, BitWritePlan, BOOL, BorrowArray, BorrowRecord, BorrowScalar,
                      CopyArray, CopyRecord, DynamicBitWritePlan, HeapArray, HeapObject, HeapRecord,
                      Interface, InterfaceResult, is_callback_enum, Record, RecordLink, Scalar, Slice,
                      TaggedResult, unqualified, ZERO)
from .domain_utils import container, domain_union, plain_integer_parameter
from .state import PathState
from .c_emit import indent

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class BlockMixin:
    def trace_stack_exit(self, paths, depth, edge):
        entities = []

        def owned(sym, name):
            if (not isinstance(sym.ty, (Scalar, Array, Record)) or sym.heap_owner or sym.static_storage
                    or sym.borrowed or sym.bit_overlay or sym.projection is not None
                    or sym.uid in self.alias_members):
                return
            entities.append((sym.uid, name))
            if isinstance(sym.ty, Record):
                for field in sym.fields:
                    owned(field, f"{name}.{field.name}")

        for scope in self.scope[depth:]:
            for sym in scope.values():
                owned(sym, sym.name)
        for path in paths:
            guard = path.guard_label()
            for uid, name in entities:
                self.trace.append(f"  mark tombstone id#{uid} {name} [{guard}]: stack {edge}")

    def propagation_nodes(self, root, *, composite_call=False, composite_literal=False, condition=False):
        span_calls, projected_calls, interface_calls = set(), set(), set()

        def truth_operand(node):
            if node.kind == "call":
                span_calls.add(id(node))
            if node.kind == "slice":
                return Node(node.token, "length", (node,))
            return node

        pending = [truth_operand(root) if condition else root]
        nodes = set()
        while pending:
            part = pending.pop()
            if part is None:
                continue
            if part.kind in ("binary", "rank_compare"):
                ranked = part.kind == "rank_compare"
                operands = part.args[2:] if ranked else part.args[1:]
                if not ranked:
                    if part.args[0] in ("&&", "||"):
                        pending.extend(map(truth_operand, operands))
                    else:
                        pending.extend(operands)
                if ranked or part.args[0] in ("==", "!=", "===", "!=="):
                    projection_kinds = ("index", "member", "slice") if ranked else ("slice",)
                    for operand in operands:
                        while operand.kind in projection_kinds:
                            if operand.kind == "index":
                                pending.append(operand.args[1])
                            elif operand.kind == "slice":
                                pending.extend(operand.args[1:3])
                            operand = operand.args[0]
                        if operand.kind == "call":
                            pending.append(truth_operand(operand))
            elif part.kind in ("index", "member"):
                host = part.args[0]
                if host.kind == "call":
                    projected_calls.add(id(host))
                pending.append(truth_operand(host))
                if part.kind == "index":
                    pending.append(part.args[1])
            elif part.kind == "slice":
                pending.extend(part.args[1:3])
                if part.args[0].kind in ("index", "member", "propagate"):
                    pending.append(part.args[0])
            elif part.kind == "length":
                operand = part.args[0]
                while operand.kind == "slice":
                    pending.extend(operand.args[1:3])
                    operand = operand.args[0]
                if operand.kind == "call":
                    projected_calls.add(id(operand))
                if operand.kind in ("call", "index", "member", "cast", "propagate"):
                    pending.append(truth_operand(operand))
            elif part.kind == "unary":
                op, operand = part.args
                pending.append(truth_operand(operand) if op == "!" else operand)
            elif part.kind == "cast":
                operand, strategy, _ = part.args
                pending.append(truth_operand(operand) if strategy in ("view", "trap") else operand)
            elif part.kind in ("in", "propagate"):
                pending.append(part.args[0])
            elif part.kind == "storage_count":
                pending.append(truth_operand(part.args[0]))
            elif part.kind == "call":
                if self.is_free_call(part) and len(part.args[1]) == 1:
                    operand = part.args[1][0]
                    projected_calls.add(id(operand))
                    pending.append(operand)
                signature = self.composite_call_signature(part)
                callee = part.args[0]
                if callee.kind != "name" or not self.named_function(callee.args[0]):
                    pending.append(callee)
                initializer = composite_call and part is root
                if signature is not None and (signature.result is None or isinstance(signature.result, (Scalar, TaggedResult))
                        or id(part) in span_calls and isinstance(signature.result, Slice)
                        or id(part) in projected_calls and isinstance(
                            signature.result, (Record, CopyArray, HeapRecord, HeapObject, HeapArray))
                        or composite_literal and isinstance(signature.result, AliasResult)
                        or id(part) in interface_calls and isinstance(signature.result, InterfaceResult)
                        or initializer and isinstance(signature.result, (Record, HeapRecord, HeapObject, HeapArray, CopyArray, Slice, AliasResult, InterfaceResult))):
                    for arg, parameter in zip(part.args[1], signature.params):
                        ty = unqualified(parameter)
                        if isinstance(ty, (Scalar, BorrowScalar, BorrowRecord, CopyRecord)):
                            pending.append(arg)
                        elif isinstance(ty, Interface):
                            interface_calls.add(id(arg))
                            pending.append(arg)
                        elif isinstance(ty, (Slice, BorrowArray, CopyArray)):
                            pending.append(truth_operand(arg))
            elif composite_literal and part.kind == "copy":
                pending.append(part.args[0])
            elif composite_literal and part.kind == "record_initializer":
                pending.extend(value for _, value in part.args[0])
            elif composite_literal and part.kind == "list":
                pending.extend(part.args[0])
            elif composite_literal and part.kind == "repeat":
                pending.append(part.args[0])
            else:
                continue
            nodes.add(id(part))
        return nodes

    def block(self, stmts, paths):
        with self.code_block_budget(stmts):
            return self.check_block(stmts, paths)

    def check_block(self, stmts, paths):
        self.scope.append({})
        lines, pending_writes = [], []
        pending_dynamic = False

        def flush_writes():
            nonlocal pending_dynamic
            if pending_writes:
                if pending_dynamic:
                    if len(pending_writes) == 1:
                        lines.extend(pending_writes[0].fallback)
                    else:
                        group = self.dynamic_bit_fusion_group(pending_writes)
                        layout = self.dynamic_bit_fusion_base(group)
                        self.trace.append(f"  fuse {len(group)} dynamic bit writes")
                        lines.extend(self.fuse_dynamic_bit_writes(group, layout))
                else:
                    lines.extend(self.fuse_bit_writes(pending_writes))
                pending_writes.clear()
                pending_dynamic = False

        for index, stmt in enumerate(stmts):
            if not paths:
                break
            self.budget_paths(paths, stmt.token)
            self.spend(stmt.token, len(paths))
            self.trace.append(f"line {stmt.token.line}:{stmt.token.col} {stmt.kind}")
            self.states("before", paths)
            prior_returns = len(self.return_states)
            previous_property_context = self.property_codegen_context
            previous_relation_codegen = self.property_relation_codegen
            previous_propagating_nodes = self.propagating_expression_nodes
            value = stmt.args[-1] if stmt.kind in ("define", "return", "assign", "eval") else None
            condition = stmt.args[0] if stmt.kind in ("if", "while") else None
            self.propagating_expression_nodes = self.propagation_nodes(
                condition or value, composite_call=stmt.kind == "define", condition=condition is not None)
            property_root = condition if stmt.kind == "if" else value
            if stmt.kind == "assign" and (stmt.args[1].text != "="
                                          or self.has_effects(stmt.args[0], allow_debug_checks=True)):
                property_root = None
            elif stmt.kind == "eval":
                property_root = None
            if stmt.kind not in ("define", "return"):
                value = None
            pure_value = value is not None and not self.has_effects(value, allow_property_calls=True)
            self.property_relation_codegen = value if pure_value and len(paths) == 1 else None
            self.property_codegen_context = self.shared_property_context(property_root, paths)
            previous_active_paths = getattr(self, "_active_paths", ())
            self._active_paths = tuple(paths)
            rest = stmts[index + 1:]
            if rest:
                self._future_stmts.append(rest)
            try:
                paths, code = self.stmt(stmt, paths, rest if stmt.kind == "while" else ())
            finally:
                if rest:
                    self._future_stmts.pop()
                self._active_paths = previous_active_paths
                self.property_codegen_context = previous_property_context
                self.property_relation_codegen = previous_relation_codegen
                self.propagating_expression_nodes = previous_propagating_nodes
            if stmt.kind == "return":
                returned = (path for path in self.return_states[prior_returns:]
                            if not path.guards or not path.guards[-1].startswith("error@"))
                self.trace_stack_exit(returned, 0, "return")
            if len(code) == 1 and isinstance(code[0], (BitWritePlan, DynamicBitWritePlan)):
                write = code[0]
                dynamic_write = isinstance(write, DynamicBitWritePlan)
                if pending_writes:
                    first = pending_writes[0]
                    incompatible = first.address != write.address
                    if not incompatible and (dynamic_write or pending_dynamic):
                        group = self.dynamic_bit_fusion_group([*pending_writes, write]) if len(pending_writes) < 16 else None
                        incompatible = group is None or self.dynamic_bit_fusion_base(group) is None
                    elif not incompatible:
                        # Keep the integer mask proportional to the touched span.
                        incompatible = abs(first.offset - write.offset) > 64
                    if incompatible:
                        flush_writes()
                pending_writes.append(write)
                pending_dynamic |= dynamic_write
            else:
                flush_writes()
                lines.extend(code)
            self.states("after", paths)
        flush_writes()
        if stmts:
            depth = len(self.scope) - 1
            self.check_heap_exit(paths, depth, stmts[-1].token)
            self.trace_stack_exit(paths, depth, "scope exit")
        locals_ = list(self.scope.pop().values())
        local_ids = set()
        while locals_:
            sym = locals_.pop()
            local_ids.add(sym.uid)
            locals_.extend(sym.fields)
        # A local heap node that an outer link already points at is owned by
        # that chain: its fields must survive the block so `next` can still be
        # followed through it.  Only live link targets are kept, so ordinary
        # locals still disappear at the closing brace.
        retained = set()
        for state in paths:
            pending = [target for uid, owners in state.borrows.items() if uid not in local_ids
                       for target in owners
                       if target.uid in local_ids and state.heap.get(target.uid) == frozenset(("Alive",))]
            while pending:
                target = pending.pop()
                if target.uid in retained:
                    continue
                retained.add(target.uid)
                retained.update(field.uid for field in target.fields)
                for field in target.fields:
                    pending.extend(item for item in state.borrows.get(field.uid, ())
                                   if item.uid in local_ids
                                   and state.heap.get(item.uid) == frozenset(("Alive",)))
        local_ids -= retained
        projected = [PathState({uid: d for uid, d in p.values.items() if uid not in local_ids}, p.guards,
                               {uid: facts for uid, facts in p.bits.items() if uid not in local_ids},
                               {uid: form for uid, form in p.forms.items() if uid not in local_ids},
                               p.zone.project(p.values.keys() - local_ids),
                               {uid: owners for uid, owners in p.borrows.items() if uid not in local_ids},
                               {uid: states for uid, states in p.heap.items() if uid not in local_ids},
                               {uid: content for uid, content in p.arrays.items() if uid not in local_ids},
                               {uid: positions for uid, positions in p.regions.items() if uid not in local_ids},
                               {key: value for key, value in p.predicates.items()
                                if not (key[:1] in (("array_untouched",), ("storage_count",), ("storage_count_owner",),
                                                    ("property_value",), ("slice_row",), ("array_element",),
                                                    ("array_window_post",))
                                        and key[1] in local_ids)},
                               {uid: epoch for uid, epoch in p.epochs.items() if uid not in local_ids},
                               {key: value for key, value in p.bit_values.items() if key[0] not in local_ids},
                               row_writes={key: bits for key, bits in p.row_writes.items() if key[1][0].uid not in local_ids},
                               starts={uid: start for uid, start in p.starts.items() if uid not in local_ids},
                               linear={uid: form for uid, form in p.linear.items() if uid not in local_ids},
                               fresh_links=frozenset(edge for edge in p.fresh_links if edge[0] not in local_ids))
                     for p in paths]
        return PathState.fold(projected, *self.current_pinned_baseline()), lines

    @staticmethod
    def pure_guard(node):
        if node.kind in ("call", "propagate") or node.kind == "cast" and node.args[1].endswith(("trap", "sat")):
            return False
        return all(Compiler.pure_guard(arg) for arg in node.args if isinstance(arg, Node))

    def chain_anchor(self, node, state):
        if node.kind == "null":
            return self.null_symbol
        sym = self.place(node)
        if sym is None or not isinstance(sym.ty, (Record, RecordLink)):
            node.token.error("chain anchor must name a record, record link or null")
        if isinstance(sym.ty, RecordLink):
            self.require_live(sym, state, node.token)
            targets = self.link_targets(sym, state)
        else:
            if sym.link_access is not None:
                self.require_live(sym, state, node.token)
            targets = self.physical_places(sym, state, allow_null=True)
        if len(targets) != 1:
            node.token.error("cannot prove a unique chain anchor")
        return next(iter(targets))

    def records_may_overlap(self, left, right, *, use_index_forms=False):
        a = left.projection.root if left.projection is not None else left
        b = right.projection.root if right.projection is not None else right
        if a.uid != b.uid:
            if self.may_alias(a, b):
                return True
            def contains(owner, uid):
                return any(field.uid == uid or contains(field, uid) for field in owner.fields)

            return contains(a, b.uid) or contains(b, a.uid)
        first = left.projection.steps if left.projection is not None else ()
        second = right.projection.steps if right.projection is not None else ()
        for x, y in zip(first, second):
            if isinstance(x, str) and isinstance(y, str) and x != y:
                return False
            if isinstance(x, Domain) and isinstance(y, Domain) and not (x & y):
                return False
        if use_index_forms:
            for x, y in zip(left.projection.index_forms, right.projection.index_forms):
                bounds = self.affine_difference_bounds(x, y)
                if bounds is not None and (bounds[0] > 0 or bounds[1] < 0):
                    return False
        return True

    @staticmethod
    def record_location(sym):
        projection = sym.projection
        return sym.uid if projection is None else (projection.root.uid, projection.steps)

    def verify_chain(self, head, tail, path, token):
        if tail.projection is not None and any(not isinstance(step, str) and
                                              (not isinstance(step, Domain) or step.lo != step.hi)
                                              for step in tail.projection.steps):
            token.error("cannot prove a unique projected chain tail")
        if not head.uid:
            token.error("chain head must be a live record")
        edges, unbounded = {}, set()
        graph = LinkGraph()
        tail_location = self.record_location(tail)
        pending, seen = [head], set()
        while pending:
            node = pending.pop()
            location = self.record_location(node)
            if location in seen:
                continue
            seen.add(location)
            self.spend(token)
            if node.uid:
                self.require_target_live(node, path, token)
            graph.add(location)
            if node.chain_unbounded:
                if tail.uid:
                    token.error("cannot prove a particular tail inside an unbounded chain")
                unbounded.add(location)
                continue
            if location == tail_location and not node.chain_summary:
                continue
            if not node.uid:
                token.error("chain reaches null before the declared tail")
            next_type = unqualified(dict(node.ty.fields).get("next"))
            if not isinstance(next_type, RecordLink):
                token.error("chain requires a next link on every node")
            field = self.child_place(node, ("next",))
            targets = self.link_targets(field, path)
            if not targets:
                token.error("cannot prove the next targets of the chain")
            edges[location] = targets
            for target in targets:
                if not target.uid:
                    continue
                graph.link(location, (self.record_location(target),),
                           fresh=(field.uid, target.uid) in path.fresh_links)
            pending.extend(target for target in sorted(targets, key=lambda target: target.uid)
                           if target.uid and self.record_location(target) not in seen)
        if tail.uid and tail_location not in seen:
            token.error("chain does not reach the declared tail")
        def anchor(location):
            if isinstance(location, int):
                return f"id#{location}"
            uid, fields = location
            suffix = "".join(f".{step}" if isinstance(step, str) else f"[{step}]" for step in fields)
            return f"id#{uid}{suffix}"

        head_location = self.record_location(head)
        summary = graph.summary(head_location)
        if summary.cycle and summary.shape.kind != REPEAT:
            loop = " -> ".join([anchor(item) for item in summary.cycle] + [anchor(summary.cycle[0])])
            token.error(f"cannot prove chain terminates: next links form a cycle ({loop})")
        if summary.shape.kind == REPEAT:
            # Every turn allocates a new node, so the chain is finite even
            # though the abstract edge points back at the allocation site.
            maximum = summary.shape.length_for(0) + summary.shape.period
        elif unbounded & summary.reach:
            maximum = inf
        else:
            maximum = summary.distances[head_location]
        guard = path.guard_label()
        self.trace.append(f"  verify chain(head: {anchor(head_location)} -> tail: {anchor(tail_location)}) "
                          f"[{guard}] => proven")
        for location in sorted(summary.distances, key=anchor):
            if location in edges:
                successors = {anchor(self.record_location(target)) for target in edges[location]}
                self.trace.append(f"    alive {anchor(location)}; next in {{{', '.join(sorted(successors))}}}")
            elif location in unbounded:
                self.trace.append(f"    alive {anchor(location)}; borrowed suffix terminates at id#0")
            else:
                self.trace.append(f"    tail {anchor(location)}: alive")
        if summary.shape.kind == REPEAT:
            shape = summary.shape
            self.trace.append(f"    chain length {shape.prefix} + {shape.period}*rounds "
                              f"(fresh allocation each iteration)")
        elif maximum == inf:
            self.trace.append("    chain length has no static upper bound")
        else:
            self.trace.append(f"    chain length {maximum} derived from the link meta-graph")
        return maximum

    def assert_chain(self, stmt, paths):
        head_node, tail_node = stmt.args
        for path in paths:
            head, tail = self.chain_anchor(head_node, path), self.chain_anchor(tail_node, path)
            self.verify_chain(head, tail, path, stmt.token)
        return paths, []

    def guarded_assert(self, stmt, paths):
        subject, rows = stmt.args
        sym = self.place(subject)
        if sym is None or not isinstance(sym.ty, Scalar):
            subject.token.error("guarded assertion requires a scalar variable or field")
        if is_callback_enum(sym.ty.enum):
            subject.token.error("function values cannot be used in numeric assertions")
        for path in paths:
            self.require_live(sym, path, subject.token)
        result, proved_domains = [], []
        for marker, guard, claim in rows:
            if guard is not None and (not self.pure_guard(guard) or self.debug_bounds and self.has_effects(guard)):
                guard.token.error("guarded assertion predicates cannot have side effects")
            target = None
            if claim is not None:
                if sym.ty.floating:
                    target = self.float_set(claim, sym.ty.bits)
                else:
                    ty = self.resolve(claim)
                    if not isinstance(ty, Scalar) or ty.floating:
                        claim.token.error("guarded assertion requires a matching scalar value set")
                    target = ty.domain
            hit, miss = [], []
            for path in paths:
                if guard is None:
                    hit.append(path)
                else:
                    yes, no, _ = self.condition(guard, path)
                    hit.extend(yes)
                    miss.extend(no)
            self.budget_paths(result + hit + miss, marker)
            if claim is None and hit:
                marker.error("cannot prove guarded assertion path is unreachable")
            for path in hit:
                domain = self.storage_value(sym, path).domain
                if self.proven_presence(domain - target, marker):
                    marker.error(f"guarded assertion cannot prove {domain} fits {target}")
                self.trace.append(f"  assert id#{sym.uid}: {domain} subset {target} => proven")
                result.append(path.guard(f"assert@{marker.line}:{marker.col}"))
                proved_domains.append(domain)
            paths = miss
        if paths:
            stmt.token.error("guarded assertion does not cover all reachable paths")
        code = []
        if proved_domains and sym.projection is None and sym.link_access is None and self.pure_guard(subject):
            domain = domain_union(proved_domains)
            if sym.ty.floating:
                predicates = self.float_range_predicates(subject.token, sym.c, domain)
            else:
                predicates = self.integer_range_predicates(subject.token, sym.c, domain, container(sym.ty.bits, sym.ty.signed))
            if predicates:
                assume = self.helper("contract_assume", None)
                code = [f"{assume}({predicate});" for predicate in predicates]
                self.trace.extend(f"  assume assertion {predicate}" for predicate in predicates)
        return result, code

    @staticmethod
    def _collect_ast_names(items):
        names = set()
        stack = list(items)
        while stack:
            item = stack.pop()
            if isinstance(item, Node):
                if item.kind in ("name", "type_name") and item.args and isinstance(item.args[0], str):
                    names.add(item.args[0])
                stack.extend(item.args)
                if item.state_guard is not None:
                    stack.append(item.state_guard)
            elif isinstance(item, (tuple, list)):
                stack.extend(item)
        return names

    def dead_scalar_uids(self, paths):
        """Pure scalar locals/value-parameters never referenced after the current statement."""
        future_names = self._contract_live_names | self._collect_ast_names(self._future_stmts)
        entry_origins = set(getattr(self, "entry_offset_origins", ()))
        dead = set()
        for scope in self.scope:
            for sym in scope.values():
                if (not isinstance(sym.ty, Scalar) or sym.borrowed or sym.heap_owner
                        or sym.static_storage or sym.bit_overlay or sym.projection is not None
                        or sym.uid in self.alias_members or sym.uid in self.param_places
                        or sym.name in future_names):
                    continue
                uid = sym.uid
                if any(
                    uid not in p.values or uid in p.borrows or uid in p.heap
                    or uid in p.arrays or uid in p.regions or uid in p.bits or uid in p.starts
                    or any(k[0] == uid for k in p.bit_values)
                    or any(k[1][0].uid == uid for k in p.row_writes)
                    or any(isinstance(k, tuple) and len(k) >= 2
                           and k[0] in ("storage_count", "storage_count_owner", "array_element",
                                        "array_untouched", "array_window_post", "slice_row")
                           and k[1] == uid for k in p.predicates)
                    or (p.forms.get(uid) is not None and p.forms[uid].origin in entry_origins)
                    for p in paths
                ):
                    continue
                dead.add(uid)
        return dead

    def current_pinned_baseline(self):
        return getattr(self, "pinned_baseline", (frozenset(), frozenset()))

    def compact_paths(self, paths, token, limit=None, baseline=None):
        """Fold paths that already agree before giving up on any of them.

        Two paths that differ only in their guard labels describe exactly the
        same state, so keeping one of them loses nothing.  States that really
        differ are kept as long as the hard limit allows, because joining them
        throws away the relations between the variables they describe.
        """
        if len(paths) < 2:
            return paths
        limit = limit or self.PATH_LIMIT
        base_epochs, base_origins = baseline if baseline is not None else (None, None)
        unique = PathState.fold(paths, base_epochs, base_origins)
        folded = len(paths) - len(unique)
        while len(unique) > limit:
            unique = [PathState.join(unique[index:index + 2]) if index + 1 < len(unique)
                      else unique[index] for index in range(0, len(unique), 2)]
        if len(unique) < len(paths):
            self.trace.append(f"  path budget {len(paths)} -> {len(unique)} at {token.line}:{token.col}"
                              f"{f' ({folded} identical)' if folded else ''}")
        label = f"merge@{token.line}:{token.col}"
        return [replace(path, guards=(label,)) for path in unique]

    def stmt(self, stmt, paths, following):
        paths = [replace(path) for path in paths]
        k, a, t = stmt.kind, stmt.args, stmt.token
        if k == "define":
            return self.define(stmt, paths)
        if k == "assign":
            return self.assign(stmt, paths)
        if k == "chain_assert":
            return self.assert_chain(stmt, paths)
        if k == "guard_assert":
            return self.guarded_assert(stmt, paths)
        if k == "while":
            self._future_stmts.append((stmt,))
            try:
                return self.loop(stmt, paths, following)
            finally:
                self._future_stmts.pop()
        if k == "interation":
            self._future_stmts.append((stmt,))
            try:
                return self.interation(stmt, paths)
            finally:
                self._future_stmts.pop()
        if k == "continue":
            edges = self.loops[-1]
            self.check_heap_exit(paths, edges.depth, t)
            outgoing = edges.continues
            leaving = [path.guard(f"{k}@{t.line}:{t.col}") for path in paths]
            self.trace_stack_exit(leaving, edges.depth, k)
            outgoing.extend(leaving)
            self.budget_paths(outgoing, t)
            return [], [f"goto {edges.continue_label};"]
        if k == "if":
            condition, yes, no = a
            replay = getattr(self, "optimization_replay", None)
            path_limit = self.PATH_LIMIT
            if replay is not None and not isinstance(replay, LoopReplay) and replay.branches >= 16:
                path_limit = min(self.PATH_LIMIT, 16)
            side_limit = path_limit // 2
            self.trace.append(f"  branch cond [{t.line}:{t.col}] ({self.conditions[t]})")
            baseline = self.current_pinned_baseline()
            base_epochs, base_origins = baseline
            true_paths, false_paths = [], []
            compacted = False
            continuing_test = error_test = None
            for path in paths:
                hit, miss, code = self.condition(condition, path)
                if hit or miss:
                    continuing_test = code
                elif error_test is None or self.propagation_code_rank(code) > self.propagation_code_rank(error_test):
                    error_test = code
                label = f"{t.line}:{t.col}"
                true_paths.extend(p.guard(label + "=true") for p in hit)
                false_paths.extend(p.guard(label + "=false") for p in miss)
                if len(true_paths) + len(false_paths) > path_limit:
                    true_paths = PathState.fold(true_paths, base_epochs, base_origins)
                    false_paths = PathState.fold(false_paths, base_epochs, base_origins)
                if len(true_paths) + len(false_paths) > path_limit:
                    true_paths = self.compact_paths(true_paths, t, side_limit, baseline)
                    false_paths = self.compact_paths(false_paths, t, side_limit, baseline)
                    compacted = True
            true_paths = PathState.fold(true_paths, base_epochs, base_origins)
            false_paths = PathState.fold(false_paths, base_epochs, base_origins)
            if compacted:
                true_paths = self.compact_paths(true_paths, t, side_limit, baseline)
                false_paths = self.compact_paths(false_paths, t, side_limit, baseline)
            code = continuing_test or error_test
            self.states("true", true_paths)
            self.states("false", false_paths)
            prior_returns = len(self.return_states)
            prior_continues = len(self.loops[-1].continues) if self.loops else 0
            true_out, true_code = self.block(yes, true_paths)
            false_out, false_code = self.block(no, false_paths)
            exiting = true_out + false_out
            pure_check = (
                bool(true_paths) and bool(false_paths) and not compacted
                and len(self.return_states) == prior_returns
                and (not self.loops or len(self.loops[-1].continues) == prior_continues)
                and not self.has_effects(condition, allow_property_calls=True)
                and len(true_out) == len(true_paths) and len(false_out) == len(false_paths)
                and {p.canonical_key(*baseline) for p in true_out} == {p.canonical_key(*baseline) for p in true_paths}
                and {p.canonical_key(*baseline) for p in false_out} == {p.canonical_key(*baseline) for p in false_paths}
            )
            if pure_check:
                joined_out = paths
            else:
                if not true_out or not false_out:
                    resolved_guards = {f"{t.line}:{t.col}=true", f"{t.line}:{t.col}=false"}
                    exiting = [replace(p, guards=tuple(g for g in p.guards if g not in resolved_guards))
                               if any(g in resolved_guards for g in p.guards) else p
                               for p in exiting]
                joined_out = PathState.fold(exiting, base_epochs, base_origins)
                if len(joined_out) >= 2:
                    dead_uids = self.dead_scalar_uids(joined_out)
                    joined_out = PathState.fold_dead(joined_out, dead_uids, base_epochs, base_origins)
                if len(joined_out) >= 2:
                    pinned_uids = {uid for _, uid, _, _ in getattr(self, "guard_parameters", ())}
                    pinned_uids.update(
                        uid for _, uid, ep, _ in getattr(self, "scalar_return_inputs", ())
                        if all(p.epochs.get(uid) is ep for p in joined_out)
                    )
                    for scope in self.scope:
                        for sym in scope.values():
                            if (sym.uid in self.param_places
                                    or isinstance(sym.ty, Scalar) and (sym.ty.result_arms is not None or sym.ty.floating)):
                                pinned_uids.add(sym.uid)
                    joined_out = PathState.fold_independent_scalar(
                        joined_out, dead_uids - pinned_uids, pinned_uids, base_epochs, base_origins)
            if not self.has_effects(condition) and (not true_paths or not false_paths):
                reachable = true_code if true_paths else false_code
                return joined_out, ["{", *indent(reachable), "}"]
            if not true_paths:
                true_code = ["__builtin_unreachable();"]
            if not false_paths:
                false_code = ["__builtin_unreachable();"]
            suffix = f"{t.line}_{t.col}"
            false_label, join_label = (f"lan_branch_{part}_{suffix}" for part in ("false", "join"))
            lowered = [f"if (!({code})) goto {false_label};", "{", *indent(true_code), "}"]
            if true_out:
                lowered.append(f"goto {join_label};")
            lowered.extend([f"{false_label}: ;", "{", *indent(false_code), "}"])
            if true_out:
                lowered.append(f"{join_label}: ;")
            labels = f"{false_label}, {join_label}" if true_out else false_label
            self.trace.append(f"  lower branch@{t.line}:{t.col}: {labels}")
            return joined_out, lowered
        if k == "return":
            value, = a
            if self.inferred_returns is not None:
                return self.infer_return(t, value, paths)
            ty = self.functions[self.current][1].result
            if (value is None) != (ty is None):
                t.error("return does not match the function's result contract")
            if isinstance(ty, TaggedResult):
                return self.return_tagged(value, ty, paths)
            if isinstance(ty, (HeapRecord, HeapObject, HeapArray)):
                return self.return_owned(value, ty, paths)
            consumes_owner = self.expression_consumes_owner(value)
            if not consumes_owner:
                self.check_heap_exit(paths, 0, t)
            if value is None:
                self.return_states.extend(replace(path) for path in paths)
                return [], ["return;"]
            if isinstance(ty, Record):
                return self.return_record(value, ty, paths)
            if isinstance(ty, CopyArray):
                return self.return_array(value, ty, paths)
            if isinstance(ty, Slice):
                return self.return_slice(value, ty, paths)
            if isinstance(ty, AliasResult):
                return self.return_alias(value, ty, paths)
            if isinstance(ty, InterfaceResult):
                return self.return_interface(value, ty, paths)
            continuing_code = error_code = None
            for path in paths:
                ret = self.expr(value, path)
                if ret.always_returns:
                    if error_code is None or self.propagation_code_rank(ret.code) > self.propagation_code_rank(error_code):
                        error_code = ret.code
                    continue
                if consumes_owner:
                    self.check_heap_exit([path], 0, t)
                ret = self.coerce_storage(value.token, ret, ty)
                if plain_integer_parameter(ty):
                    ret = self.ssa_narrow(stmt, ret, ty.domain, "returns", "return")
                self.range_check(value.token, ret, ty)
                self.check_result_bounds(value.token, ret, path)
                if self.functions[self.current][1].idempotent_property:
                    self.idempotent_return_domains.append(ret.domain)
                case_paths = ((ret.domain, path),)
                if self.guard_parameters:
                    if ret.domain == BOOL and not self.has_effects(value):
                        yes, no, _ = self.condition(value, path)
                        self.guard_return_observations.extend((True, state) for state in yes)
                        self.guard_return_observations.extend((False, state) for state in no)
                        case_paths = (*((Domain.point(1), state) for state in yes), *((ZERO, state) for state in no))
                    else:
                        self.guard_return_observations.extend((outcome, replace(path)) for outcome in (True, False)
                                                              if int(outcome) in (ret.domain.lo, ret.domain.hi))
                self.observe_scalar_return(ret, case_paths)
                if ty.result_arms is not None and not ty.floating:
                    self.result_exit_domains.append(ret.domain)
                self.trace.append(f"  return {ret.domain} [{path.guard_label()}]")
                self.return_states.append(replace(path))
                continuing_code = ret.code
            return [], [f"return {continuing_code or error_code};"]
        if k == "eval":
            if self.is_free_call(a[0]):
                return self.free_heap(a[0], paths)
            continuing = []
            continuing_code = error_code = None
            for path in paths:
                if a[0].kind == "call":
                    early_return = []
                    result, code = self.call(a[0], path, capture_early_return=early_return)
                    if isinstance(result, TaggedResult):
                        t.error("composite result must be unwrapped or returned")
                    if not early_return and isinstance(result, (HeapRecord, HeapObject, HeapArray)):
                        t.error("owning result must be bound or returned")
                else:
                    value = self.expr(a[0], path)
                    code = value.code
                    early_return = [value] if value.always_returns else []
                if early_return:
                    if error_code is None or self.propagation_code_rank(code) > self.propagation_code_rank(error_code):
                        error_code = code
                    continue
                continuing.append(path)
                continuing_code = code
            lines = [f"(void)({continuing_code or error_code});"]
            if not continuing:
                lines.append("__builtin_unreachable();")
            return continuing, lines
        raise AssertionError(k)
