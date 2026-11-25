"""Checking function bodies and parameters."""

from __future__ import annotations

import copy
import json
from dataclasses import field, fields, replace

from difference_domain import Difference
from optimization_bridge import replay_function
from return_reasoning import ReturnCase, ReturnObservation, return_implications
from semantic_ir import Proposition, Theorem, TheoremExchange
from array_domain import ArrayDomain
from value_domain import Domain, DomainComplexity
from table_interface import encode_domain

from .errors import LanError
from .syntax import Node
from .typesys import (AliasResult, Array, BorrowArray, BorrowRecord, BorrowScalar, CopyArray, CopyRecord,
                      ExternCallable, HeapArray, HeapObject, HeapRecord, Interface, ReadOnly, Record, RecordLink,
                      Scalar, Slice, Span, success_type, Symbol, unqualified, Value, WIDE, ZERO)
from .domain_utils import c_int, container, domain_union, plain_integer_parameter, replay_scalar
from .state import PathState
from .interface_types import (ALIAS_PATH_LIMIT, array_mutation_projection, array_mutation_step,
                              CHAIN_ALIAS_FALLBACK_DEPTH, chain_attachments, CHAIN_REACHABLE,
                              dependent_source_domain, encode_interface_type, GRAPH_ALIAS_FALLBACK_DEPTH,
                              linked_owned_record, mutable_chain, mutable_chain_record, mutable_graph,
                              offset_expression_lengths, offset_expression_parameters, OWNED_CHAIN_LONG,
                              slice_offset_terms, valid_interface_rebind_effects)
from .summaries import graph_write_path, graph_write_program, guard_parameter_kind, small_summary_domain
from .c_emit import indent

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("join_result_contents", "result_content_size")
join_result_contents = None
result_content_size = None


class FunctionMixin:
    def check_result_bounds(self, token, value, state):
        for source, op, delta in self.result_bound_sources:
            bound = self.storage_value(source, state) if isinstance(source, Symbol) else source
            limit = self.binary(token, "+", bound, Value(Domain.point(delta), c_int(delta)), state)
            if not self.prove_order(token, value, op, limit, state):
                token.error("cannot prove dependent result bound")

    def initialize_record_parameter(self, root, path, graph_nodes=None):
        for child in self.storage_leaves(root):
            if isinstance(child.ty, (Scalar, Slice)):
                path = path.update(child, child.ty.domain)
            elif isinstance(child.ty, RecordLink) and graph_nodes is not None:
                path = path.update(child, ZERO)
                targets = frozenset((self.null_symbol, graph_nodes[child.ty.identity]))
                path = replace(path, borrows=path.borrows | {child.uid: targets})
            elif isinstance(child.ty, Array) and not child.ty.packed_bits:
                contents = (self.graph_array_contents(child.ty, graph_nodes) if graph_nodes is not None
                            and self.contains_record_links(child.ty) else
                            ArrayDomain.unknown(child.ty.length, child.ty.element.domain))
                path = path.with_array(child, contents)
        return path

    def trace_record_parameter(self, root, path):
        for field in self.storage_leaves(root):
            label = f"id#{field.uid}"
            ty = field.ty
            if isinstance(ty, (Scalar, Slice)):
                self.trace_content(label, ty, path.values[field.uid])
            elif isinstance(ty, Array):
                self.trace_array_contents(label, ty, path, field)
            elif isinstance(ty, RecordLink) and field.uid in path.borrows:
                self.trace.append(f"    {label} targets := {self.trace_alias_targets(path.borrows[field.uid])}")

    def trace_array_contents(self, label, ty, path, sym):
        if ty.packed_bits:
            self.trace.append(f"    {label}.len := {{{ty.length}}}")
            self.trace.append(f"    {label}[0, {ty.length}) := {ty.element.domain}")
        else:
            self.trace_content(label, ty, self.actual_array(path, sym))

    def initialize_graph_parameter(self, root, path, token, payload_index=None, return_prefix=0, return_paths=()):
        nodes = {}
        pending = [root.ty]
        while pending:
            ty = pending.pop()
            if ty.identity in nodes:
                continue
            self.spend(token)
            if payload_index is not None and not mutable_chain_record(ty):
                token.error("writable graph nodes require scalar payload fields")
            symbol = Symbol("graph_node", len(self.symbols) + 1, ty, root.depth,
                            readonly=root.readonly, borrowed=True, chain_summary=True)
            self.symbols.append(symbol)
            self.record_symbols(symbol, token, zero=False)
            nodes[ty.identity] = symbol
            pending.extend(self.link_record(field, token) for field in self.linked_field_types(ty))
        for symbol in (root, *nodes.values()):
            path = self.initialize_record_parameter(symbol, path, graph_nodes=nodes)
            if payload_index is not None:
                self.register_graph_payload(symbol, payload_index)
        origin = self.param_places.get(root.uid)
        if (return_prefix or return_paths) and root.readonly and origin is not None:
            prefix = origin[1]
            expanded = {prefix: root}

            def expand(edge_path, target):
                nonlocal path
                parent = expanded[edge_path[:-1]]
                self.spend(token)
                successor = Symbol("graph_edge", len(self.symbols) + 1, target.ty, root.depth,
                                   readonly=True, borrowed=True, chain_summary=True)
                self.symbols.append(successor)
                self.record_symbols(successor, token, zero=False)
                self.param_places[successor.uid] = origin[0], edge_path
                path = self.initialize_record_parameter(successor, path, graph_nodes=nodes)
                link = self.child_place(parent, (edge_path[-1],))
                path = replace(path, borrows=path.borrows | {link.uid: frozenset((self.null_symbol, successor))})
                expanded[edge_path] = successor

            for route in return_paths:
                edge_path = prefix
                for name in route:
                    parent = expanded[edge_path]
                    field = unqualified(dict(parent.ty.fields).get(name))
                    if not isinstance(field, RecordLink):
                        break
                    edge_path = (*edge_path, name)
                    if edge_path not in expanded:
                        expand(edge_path, nodes[field.identity])

            pending, remaining = [prefix], 64
            for parent_path in pending:
                if not remaining:
                    break
                if len(parent_path) - len(prefix) >= min(return_prefix, GRAPH_ALIAS_FALLBACK_DEPTH):
                    continue
                for name, field in expanded[parent_path].ty.fields:
                    field = unqualified(field)
                    if not isinstance(field, RecordLink):
                        continue
                    edge_path = (*parent_path, name)
                    if edge_path not in expanded:
                        if not remaining:
                            break
                        remaining -= 1
                        expand(edge_path, nodes[field.identity])
                    pending.append(edge_path)
        return path

    def graph_nodes(self, roots, state, token):
        pending, visited = list(roots), set()
        while pending:
            root = pending.pop()
            location = self.record_location(root)
            if not root.uid or location in visited:
                continue
            visited.add(location)
            self.spend(token)
            self.require_target_live(root, state, token)
            pending.extend(self.record_link_targets(root, state))
            yield root

    def verify_live_graph(self, roots, state, token):
        visited = sorted(root.uid for root in self.graph_nodes(roots, state, token))
        self.trace.append(f"  verify live graph nodes {visited} => proven")

    def initialize_chain_parameter(self, root, limit, path, token, shared=False, payload_index=None, return_prefix=0):
        if payload_index is not None:
            self.register_chain_payload(root, payload_index)
        current = root
        origin = self.param_places.get(root.uid)
        prefix = (return_prefix + 1 if return_prefix and root.readonly and origin is not None else 0) if limit is True else limit
        for depth in range(prefix):
            self.spend(token)
            targets = {self.null_symbol}
            if depth + 1 < prefix:
                successor = Symbol("chain_node", len(self.symbols) + 1, root.ty, root.depth,
                                   readonly=root.readonly, borrowed=True, chain_summary=shared or limit is True)
                self.symbols.append(successor)
                self.record_symbols(successor, token, zero=False)
                if root.readonly and origin is not None:
                    self.param_places[successor.uid] = origin[0], (*origin[1], *("next",) * (depth + 1))
                path = self.initialize_record_parameter(successor, path)
                if payload_index is not None:
                    self.register_chain_payload(successor, payload_index)
                targets.add(successor)
            field = self.child_place(current, ("next",))
            if payload_index is not None:
                self.chain_distances[field.uid] = depth + 1
            path = path.update(field, ZERO)
            path = replace(path, borrows=path.borrows | {field.uid: frozenset(targets)})
            if depth + 1 < prefix:
                current = successor
        if limit is True:
            successor = Symbol("chain_suffix", len(self.symbols) + 1, root.ty, root.depth,
                               readonly=root.readonly, borrowed=True, chain_summary=True, chain_unbounded=True)
            self.symbols.append(successor)
            self.record_symbols(successor, token, zero=False)
            if root.readonly and origin is not None:
                self.param_places[successor.uid] = origin[0], (CHAIN_REACHABLE,)
            path = self.initialize_record_parameter(successor, path)
            if payload_index is not None:
                self.register_chain_payload(successor, payload_index)
            targets = frozenset((self.null_symbol, successor))
            for parent in (current, successor):
                field = self.child_place(parent, ("next",))
                path = path.update(field, ZERO)
                path = replace(path, borrows=path.borrows | {field.uid: targets})
        return path

    def property_returns(self, statements, paths, token):
        returns = []
        parameters = dict(self.scope[-1])
        copies = {}
        for name, source in parameters.items():
            copy = replace(source, name=name, uid=len(self.symbols) + 1)
            self.symbols.append(copy)
            copies[name] = copy
            initialized = []
            for path in paths:
                value = self.storage_value(source, path)
                initialized.append(path.update(copy, value.domain, value.affine, value.difference, value.linear))
            paths = initialized

        def branches(condition, paths):
            yes, no = [], []
            for path in paths:
                hit, miss, _ = self.condition(condition, path)
                yes.extend(hit)
                no.extend(miss)
            self.budget_paths(yes + no, condition.token)
            return yes, no

        def walk(body, paths, edges=None):
            self.scope.append({})
            try:
                for stmt in body:
                    if not paths:
                        break
                    self.spend(stmt.token, len(paths))
                    if stmt.kind == "return":
                        returns.extend((stmt.token, self.expr(stmt.args[0], path), path) for path in paths)
                        self.budget_paths(returns, stmt.token)
                        return []
                    if stmt.kind == "define":
                        if stmt not in self.property_local_types:
                            stmt.token.error("algebraic proof cannot replay an unchecked local declaration")
                        paths, _ = self.define(stmt, paths)
                        continue
                    if stmt.kind == "assign":
                        paths, _ = self.assign(stmt, [replace(path) for path in paths])
                        continue
                    if stmt.kind == "guard_assert":
                        paths, _ = self.guarded_assert(stmt, paths)
                        continue
                    if stmt.kind == "continue":
                        edges.extend(paths)
                        return []
                    if stmt.kind == "while":
                        condition, loop_body = stmt.args[:2]
                        finished = []
                        for iteration in range(self.LOOP_LIMIT + 1):
                            active, done = branches(condition, paths)
                            finished.extend(done)
                            if not active:
                                break
                            if iteration == self.LOOP_LIMIT:
                                stmt.token.error("algebraic loop proof exceeds the iteration budget")
                            continued = []
                            paths = walk(loop_body, active, continued) + continued
                            self.budget_paths(paths + finished, stmt.token)
                            if not paths:
                                break
                        paths = finished
                        continue
                    condition, yes, no = stmt.args
                    true_paths, false_paths = branches(condition, paths)
                    paths = walk(yes, true_paths, edges) + walk(no, false_paths, edges)
                return paths
            finally:
                self.scope.pop()

        self.scope.append(copies)
        try:
            if walk(statements, paths):
                token.error("algebraic property requires a result on every path")
            return returns
        finally:
            self.scope.pop()

    def verified_guard_outcome(self, outcome):
        paths = [state for truth, state in self.guard_return_observations if truth == outcome]
        if not paths:
            return "bool", False
        claims = []
        stable = []
        for operand, uid, epoch, declared in self.guard_parameters[:12]:
            if all(state.epochs.get(uid) is epoch for state in paths):
                domain = Domain.union(state.values[uid] for state in paths)
                if operand[0] == "bool_param":
                    if domain.lo == domain.hi:
                        claims.append(operand if domain.lo else ("!", operand))
                else:
                    stable.append((operand, uid, domain))
                    if domain != declared:
                        claims.append(("in", operand, domain))
        for i, (left, uid, domain) in enumerate(stable):
            for right, other_uid, other in stable[i + 1:]:
                if len(claims) >= 10:
                    break
                difference = Difference(uid, other_uid)
                ranges = [state.zone.range(difference) for state in paths]
                if any(bounds is None for bounds in ranges):
                    continue
                lower = min(lo for lo, _ in ranges)
                upper = max(hi for _, hi in ranges)
                relation = ("arith", "-", left, right)
                if upper < domain.hi - other.lo and WIDE.lo <= upper <= WIDE.hi:
                    claims.append(("<=", relation, ("const", upper)))
                if lower > domain.lo - other.hi and WIDE.lo <= lower <= WIDE.hi:
                    claims.append((">=", relation, ("const", lower)))
        guard = ("bool", True)
        for claim in claims[:10]:
            guard = claim if guard == ("bool", True) else ("&&", guard, claim)
        return guard

    def link_return_paths(self, node, contract):
        def origins(result):
            if not isinstance(result, AliasResult):
                return ()
            return tuple((source, path) for source, path in result.origins()
                         if all(isinstance(step, str) and step != CHAIN_REACHABLE for step in path))

        if contract.result is not None:
            routes = origins(success_type(contract.result))
            return max((len(path) for _, path in routes), default=0), routes
        records = (*self.record_types.values(), *self.types.values())
        fields = {"next"} | {name for record in records if isinstance(record, Record)
                             for name, field in record.fields if isinstance(unqualified(field), RecordLink)}
        returns, bindings = [], {}
        pending = list(node.args[3])
        while pending:
            part = pending.pop()
            if isinstance(part, (list, tuple)):
                pending.extend(part)
            elif isinstance(part, Node):
                if part.kind == "return":
                    if part.args[0] is not None:
                        returns.append((part.args[0], (), frozenset()))
                elif part.kind == "define":
                    name, _, initializer = part.args
                    if initializer is not None:
                        bindings.setdefault(name.text, []).append(initializer)
                elif part.kind == "assign":
                    target, op, value = part.args
                    if target.kind == "name" and op.text == "=":
                        bindings.setdefault(target.args[0], []).append(value)
                else:
                    pending.extend(part.args)

        parameters = {param.text: index for index, param in enumerate(node.args[2])}
        depth, routes, visited = 0, set(), set()
        while returns:
            value, path, seen = returns.pop()
            key = id(value), path, seen
            if id(value) in seen or key in visited or len(path) > ALIAS_PATH_LIMIT:
                continue
            visited.add(key)
            self.spend(value.token)
            seen = seen | {id(value)}
            while value.kind == "member" and value.args[1].text in fields:
                path = (value.args[1].text, *path)
                value = value.args[0]
            result = self.composite_call_type(value)
            depth = max(depth, len(path))
            for source, prefix in origins(result):
                arguments = value.args[1]
                if source < len(arguments):
                    returns.append((arguments[source], (*prefix, *path), seen))
            if value.kind == "name":
                name = value.args[0]
                if path and name in parameters:
                    routes.add((parameters[name], path))
                returns.extend((initializer, path, seen) for initializer in bindings.get(name, ()))
        return depth, tuple(sorted(routes))

    def function(self, name, node, contract):
        previous = self.ambient_gas, self.budget_block, self.block_gas
        self.ambient_gas, self.budget_block, self.block_gas = self.GAS_LIMIT, None, {}
        try:
            candidates = tuple(index for index, ty in enumerate(contract.params) if linked_owned_record(ty))
            if (candidates and name not in self.recursive_groups and not self.inferring_results
                    and OWNED_CHAIN_LONG ** min(len(candidates), 8) <= self.PATH_LIMIT):
                saved = self.save_analysis_state()
                self.owned_chain_parameters[name] = candidates
                try:
                    return self.check_function(name, node, contract)
                except LanError:
                    # The legacy contract supplies a terminal node. Keep it
                    # when the same body cannot handle every finite chain.
                    self.restore_analysis_state(saved)
            return self.check_function(name, node, contract)
        finally:
            self.ambient_gas, self.budget_block, self.block_gas = previous

    def check_function(self, name, node, contract):
        if name in self.pending_recursive_results and not self.inferring_results:
            self.infer_recursive_results(self.recursive_groups[name])
            node, contract = self.functions[name]
            if name in self.pending_recursive_results:
                self.pending_recursive_results.difference_update(self.recursive_groups[name])
                self.prove_recursive_termination(self.recursive_groups[name])
        if name in self.recursive_groups and name not in self.recursive_summaries:
            self.seed_recursive_summaries(name)
        property_has_loop = False
        if (contract.identity_property or contract.idempotent_property or contract.commutative_property
                or contract.diagonal_property):
            pending = list(node.args[3])
            while pending:
                stmt = pending.pop()
                if stmt.kind == "define":
                    _, annotation, initializer = stmt.args
                    if ((annotation is None or plain_integer_parameter(self.resolve(annotation)))
                            and not self.has_effects(initializer, allow_property_calls=True)):
                        continue
                if stmt.kind == "assign":
                    target, _, value = stmt.args
                    if target.kind == "name" and not self.has_effects(value, allow_property_calls=True):
                        continue
                if stmt.kind == "continue":
                    continue
                if stmt.kind == "guard_assert" and stmt.args[0].kind == "name":
                    # The assertion verifier checks purity and proves every reachable row.
                    continue
                if stmt.kind not in ("if", "return", "while") or self.has_effects(stmt.args[0], allow_property_calls=True):
                    stmt.token.error("algebraic property requires an effect-free body with scalar local updates")
                if stmt.kind == "while":
                    property_has_loop = True
                    pending.extend(stmt.args[1])
                if stmt.kind == "if":
                    pending.extend(stmt.args[1])
                    pending.extend(stmt.args[2])
        self.current = name
        self.property_local_types = {} if (contract.identity_property or contract.idempotent_property
                                           or contract.commutative_property or contract.diagonal_property) else None
        self.param_places, self.borrow_ancestry = {}, {}
        self.parameter_uids = {}
        self.opaque_interface_targets = set()
        self.owned_interface_targets = set()
        self.consumed_parameters = {}
        self.chain_fields = {}
        self.chain_payloads = {}
        # (link field uid, target uid) edges whose target is a fresh heap
        # object each iteration; function-scoped because the allocation site
        # is a static property of the loop, not of one analysis path.
        self.chain_spans = {}
        self.attach_heads = {}
        self.chain_distances = {}
        self.attach_protected = set()
        self.alias_types, self.alias_members = {}, {}
        self.writes, self.return_states = set(), []
        self.rebind_order = {}
        self.interface_before_writes = set()
        self.interface_after_writes = set()
        self.idempotent_return_domains = []
        self.guard_return_observations = []
        self.scalar_return_cases = []
        self.result_exit_domains = []
        self.scalar_return_inputs = ()
        self.guard_parameters = ()
        self.record_return_fields = []
        self.owned_chain_return_lengths = set()
        self.array_return_contents = []
        self.array_results.pop(name, None)
        ty_node = node.args[1]
        self.inferred_returns = [] if ty_node.kind == "contract" and ty_node.args[1] is None else None
        self.inferred_slice_paths = []
        self.scope = [{}]
        pending = list(node.args[3])
        self.rebound_slices = set()
        while pending:
            stmt = pending.pop()
            if stmt.kind == "assign":
                dst, _, src = stmt.args
                if dst.kind == "name" and not self.is_copy(src):
                    self.rebound_slices.add(dst.args[0])
            elif stmt.kind == "while":
                pending.extend(stmt.args[1])
            elif stmt.kind == "interation":
                pending.extend(stmt.args[-1])
            elif stmt.kind == "if":
                pending.extend(stmt.args[1])
                pending.extend(stmt.args[2])
        path = self.module_const_entry_state()
        link_return_prefix, link_return_paths = (
            self.link_return_paths(node, contract)
            if any(isinstance(ty, BorrowRecord) and ty.readonly and (ty.chain_limit is True or ty.graph)
                   for ty in contract.params) else (0, ()))
        params, parameter_symbols, entry_code = [], [], []
        self.trace.append(f"\n@function {name}")
        for index, (param, ty) in enumerate(zip(node.args[2], contract.params)):
            record_param = isinstance(ty, (CopyRecord, BorrowRecord))
            interface_param = isinstance(ty, Interface)
            scalar_alias = isinstance(ty, BorrowScalar)
            array_param = isinstance(ty, (BorrowArray, CopyArray))
            borrowed_param = isinstance(ty, (BorrowRecord, BorrowScalar, BorrowArray))
            heap_param = isinstance(ty, (HeapRecord, HeapObject, HeapArray))
            storage = (ty.record if record_param else ty.scalar if scalar_alias
                       else ty.array if array_param
                       else ty.record if isinstance(ty, HeapRecord)
                       else ty.base if isinstance(ty, HeapObject) else unqualified(ty))
            frozen = isinstance(ty, ReadOnly) or (record_param or scalar_alias or array_param or interface_param) and ty.readonly
            sym = self.declare(param, storage, readonly=frozen, borrowed=borrowed_param,
                               heap=isinstance(ty, (HeapRecord, HeapObject)))
            cname = sym.c_var if borrowed_param or heap_param else sym.c
            params.append(f"{ty.c} {cname}")
            parameter_symbols.append(sym)
            if heap_param:
                self.parameter_uids[sym.uid] = index
                path = replace(path, heap=path.heap | {sym.uid: frozenset(("Alive",))})
                if isinstance(ty, HeapArray):
                    path = path.update(sym, ty.domain).with_array(
                        sym, ArrayDomain.unknown(ty.domain.hi, ty.element.domain))
                elif isinstance(ty, HeapRecord):
                    path = path.update(sym, ZERO)
                    self.record_symbols(sym, param, zero=False)
                    path = self.initialize_record_parameter(sym, path)
                    path = self.initialize_interface_parameters(sym, path, param, owning=True)
                    if linked_owned_record(ty):
                        path = self.initialize_terminal_owner(sym, path)
                elif isinstance(ty.base, Scalar):
                    path = path.update(sym, ty.base.domain)
                elif isinstance(ty.base, Array) and not ty.base.packed_bits:
                    path = path.update(sym, ZERO).with_array(
                        sym, ArrayDomain.unknown(ty.base.length, ty.base.element.domain))
                else:
                    path = path.update(sym, ZERO)
                self.trace.append(f"  heap id#{sym.uid} {sym.name}: Alive, owning parameter")

            if isinstance(ty, (BorrowScalar, BorrowArray)) and ty.nullable:
                self.alias_types[sym.uid], self.alias_members[sym.uid] = ty, (sym.uid, ())
                targets = frozenset((sym, self.null_symbol))
                path = replace(path.update(sym, ZERO), borrows=path.borrows | {sym.uid: targets})
            if heap_param:
                pass
            elif isinstance(ty, ExternCallable):
                self.alias_types[sym.uid], self.alias_members[sym.uid] = ty, (sym.uid, ())
                targets = frozenset((sym, self.null_symbol)) if ty.nullable else frozenset((sym,))
                path = replace(path.update(sym, ZERO), borrows=path.borrows | {sym.uid: targets})
                self.trace.append(f"  param id#{sym.uid} {sym.name}: extern callable {ty.c}"
                                  f"{' nullable' if ty.nullable else ''}")
            elif array_param:
                if isinstance(ty, CopyArray):
                    sym = replace(sym, c_place=f"{cname}.data")
                    self.symbols[sym.uid - 1] = sym
                    self.scope[-1][sym.name] = sym
                else:
                    self.register_borrow(sym, index)
                label = f"id#{sym.uid}"
                mode = "alias" if borrowed_param else "copy"
                immutable = ", immutable" if ty.readonly else ""
                nullable = ", nullable (contents when nonnull)" if borrowed_param and ty.nullable else ""
                self.trace.append(f"  param {label} {sym.name}: array pass {mode} len {storage.length}"
                                  f"{immutable}{nullable}")
                if not storage.packed_bits:
                    contents = ArrayDomain.unknown(storage.length, storage.element.domain)
                    path = path.with_array(sym, contents)
                self.trace_array_contents(label, storage, path, sym)
            elif record_param or interface_param:
                if borrowed_param:
                    self.alias_types[sym.uid], self.alias_members[sym.uid] = ty, (sym.uid, ())
                    targets = frozenset((sym, self.null_symbol)) if ty.nullable else frozenset((sym,))
                    path = replace(path.update(sym, ZERO), borrows=path.borrows | {sym.uid: targets})
                self.record_symbols(sym, param, zero=False)
                if borrowed_param or interface_param:
                    self.register_borrow(sym, index)
                if isinstance(ty, BorrowRecord) and ty.graph:
                    path = self.initialize_graph_parameter(sym, path, param,
                                                           payload_index=index if mutable_graph(ty) else None,
                                                           return_prefix=link_return_prefix,
                                                           return_paths=tuple(route for source, route in link_return_paths
                                                                              if source == index))
                else:
                    path = self.initialize_record_parameter(sym, path)
                if isinstance(ty, BorrowRecord) and ty.chain_limit is not None:
                    chain_prefix = min(link_return_prefix, CHAIN_ALIAS_FALLBACK_DEPTH)
                    if ty.chain_limit is True and ty.readonly:
                        known_prefix = max((len(route) for source, route in link_return_paths
                                            if source == index and all(step == "next" for step in route)),
                                           default=0)
                        chain_prefix = max(chain_prefix, known_prefix)
                    path = self.initialize_chain_parameter(sym, ty.chain_limit, path, param,
                                                           payload_index=index if mutable_chain(ty) else None,
                                                           return_prefix=chain_prefix)
                    if ty.attach is not None:
                        self.attach_heads[self.child_place(sym, ("next",)).uid] = index
                if any(donor == index for _, donor in chain_attachments(contract.params)):
                    roots = (self.graph_nodes((sym,), path, param) if ty.graph
                             else self.chain_nodes(path, (sym,), param))
                    for root in roots:
                        self.attach_protected.update(field.uid for field in root.fields)
                if interface_param:
                    self.trace.append(f"  define entity id#{sym.uid} {sym.name} [entry]: interface borrowed parameter")
                else:
                    nullable = ", nullable (fields when nonnull)" if borrowed_param and ty.nullable else ""
                    self.trace.append(f"  param id#{sym.uid} {sym.name}: record pass "
                                      f"{'alias' if borrowed_param else 'copy'}{' immutable' if ty.readonly else ''}{nullable}")
                self.trace_record_parameter(sym, path)
                path = self.initialize_interface_parameters(sym, path, param)
            elif isinstance(ty, Slice):
                storage_owner = sym
                if param.text in self.rebound_slices:
                    # The C descriptor is local, while its incoming storage
                    # remains a distinct source after descriptor rebinding.
                    sym = replace(sym, depth=len(self.scope) + 1)
                    self.symbols[sym.uid - 1] = sym
                    self.scope[-1][sym.name] = sym
                    parameter_symbols[-1] = sym
                    storage_owner = Symbol(param.text + "$storage", len(self.symbols) + 1, ty,
                                           depth=sym.depth if ty.copy else 0, c_place=sym.c,
                                           readonly=ty.readonly, borrowed=not ty.copy)
                    self.symbols.append(storage_owner)
                if ty.copy:
                    owner = f"lan_copy_{sym.uid}"
                    entry_code.extend([
                        self.slice_clone(ty, sym.c, owner),
                        f"{sym.c}.data = {owner};",
                    ])
                else:
                    self.register_borrow(storage_owner, index)
                span = Span(ty.element, Value(ty.domain, f"(lan_int){sym.c}.len"), sym.depth if ty.copy else 0, code=sym.c,
                            borrowed_from=frozenset((storage_owner,)), readonly=ty.readonly, text=ty.text,
                            origins=((storage_owner, ZERO),))
                path = path.bind_span(sym, span)
                # Binding the incoming descriptor does not modify its backing bytes.
                path = replace(path, epochs={uid: epoch for uid, epoch in path.epochs.items() if uid != sym.uid})
                if ty.graph:
                    if mutable_graph(ty):
                        self.chain_fields[storage_owner.uid] = index
                        self.chain_spans[index] = storage_owner
                    path, contents = self.initialize_graph_span(storage_owner, path, param,
                                                               payload_index=index if mutable_graph(ty) else None)
                elif ty.chain_limit is not None:
                    if mutable_chain(ty):
                        self.chain_fields[storage_owner.uid] = index
                        self.chain_spans[index] = storage_owner
                    path, contents = self.initialize_chain_span(storage_owner, path, param,
                                                               payload_index=index if mutable_chain(ty) else None)
                else:
                    contents = (ArrayDomain.unknown(ty.domain.hi, ty.element.domain) if ty.content_bound is None
                                else ArrayDomain.filled(ty.domain.hi, ty.element.domain, ty.content_bound))
                path = path.with_array(storage_owner, contents)
                self.trace_slice_binding("define entity", sym, span, path)
            else:
                if scalar_alias:
                    self.register_borrow(sym, index)
                path = path.update(sym, storage.domain)
                self.trace.append(f"  param id#{sym.uid} {sym.name}: {ty.c} & {storage.domain}")
        self.result_bound_sources = []
        self.result_offset_snapshots = {}
        payload = success_type(contract.result)
        if isinstance(payload, Slice):
            indices = {index for factors, _ in slice_offset_terms(payload) for index in factors}
            if payload.offset_expression is not None:
                indices |= offset_expression_parameters(payload.offset_expression)
            indices = sorted(indices)
            for index in indices:
                param = parameter_symbols[index]
                snapshot = Symbol("entry_offset", len(self.symbols) + 1, param.ty, readonly=True)
                self.symbols.append(snapshot)
                path = path.update(snapshot, path.values[param.uid], path.forms.get(param.uid), Difference(param.uid))
                self.result_offset_snapshots[index] = snapshot
            lengths = (offset_expression_lengths(payload.offset_expression)
                       if payload.offset_expression is not None else ())
            for index in sorted(lengths):
                param, ty = parameter_symbols[index], contract.params[index]
                snapshot = Symbol("entry_length", len(self.symbols) + 1, self.types["uint32"], readonly=True)
                self.symbols.append(snapshot)
                if isinstance(ty, Slice):
                    path = path.update(snapshot, path.values[param.uid], path.forms.get(param.uid), Difference(param.uid))
                else:
                    path = path.update(snapshot, Domain.point(ty.array.length))
                self.result_offset_snapshots[("len", index)] = snapshot
        assumptions = []
        for target, source, length, op, delta in contract.bounds:
            if target == len(contract.params):
                domain = dependent_source_domain(contract.params[source], length)
                if domain.lo == domain.hi:
                    snapshot = Value(domain, c_int(domain.lo))
                else:
                    param = parameter_symbols[source]
                    snapshot = Symbol("entry_bound", len(self.symbols) + 1, self.types["uint64"] if length else param.ty)
                    self.symbols.append(snapshot)
                    path = path.update(snapshot, path.values[param.uid], path.forms.get(param.uid),
                                       Difference(param.uid))
                self.result_bound_sources.append((snapshot, op, delta))
                continue
            left = parameter_symbols[target].uid
            right = parameter_symbols[source].uid
            if length and isinstance(contract.params[source], (BorrowArray, CopyArray)):
                delta += contract.params[source].array.length
                right = 0
            difference = Difference(left, right, -delta)
            if op == ">=":
                difference = difference.negated()
            path = path.with_zone(path.zone.constrain(difference, 0))
            if path is None:
                node.token.error("dependent parameter bounds have no possible input")
            if right:
                source_code = parameter_symbols[source].c + (".len" if length else "")
                target_code = parameter_symbols[target].c
                bound = f"(lan_int){source_code}"
                if delta:
                    bound = f"({bound} + {c_int(delta)})"
                assumptions.append(f"(lan_int){target_code} {op} {bound}")
        def entry_expression_constants_fit(expr):
            if expr[0] == "binary":
                return all(entry_expression_constants_fit(operand) for operand in expr[2:])
            if expr[0] == "cast":
                return entry_expression_constants_fit(expr[5])
            return expr[0] != "const" or WIDE.lo <= expr[1] <= WIDE.hi

        entry_expressions = (*contract.parameter_expressions,
                             *((None, op, expr) for op, expr in contract.result_expressions))
        for target, op, expression in entry_expressions:
            inputs = {index: self.storage_value(parameter_symbols[index], path)
                      for index in offset_expression_parameters(expression)}
            for index in offset_expression_lengths(expression):
                param, ty = parameter_symbols[index], contract.params[index]
                if isinstance(ty, Slice):
                    value = Value(path.values[param.uid], f"{param.c}.len", affine=path.forms.get(param.uid),
                                  difference=Difference(param.uid), linear=path.linear.get(param.uid))
                    inputs["len", index] = self.capture_value(value, path)
                else:
                    inputs["len", index] = Value(Domain.point(ty.array.length), c_int(ty.array.length))
            limit = self.integer_expression_value(node.token, expression, inputs, path)
            if target is None:
                self.result_bound_sources.append((self.capture_value(limit, path), op, 0))
            else:
                value = self.storage_value(parameter_symbols[target], path)
                path = self.assume_dependent_expression(node.token, path, value, op, limit)
                if entry_expression_constants_fit(expression):
                    assumptions.append(f"(lan_int)({value.code}) {op} (lan_int)({limit.code})")
        range_assumptions = []
        for ty, sym in zip(contract.params, parameter_symbols):
            storage = unqualified(ty)
            if isinstance(ty, (BorrowRecord, BorrowScalar, BorrowArray, ExternCallable)) and not ty.nullable:
                range_assumptions.append(f"{sym.c_var} != NULL")
            elif isinstance(ty, Interface):
                range_assumptions.append(f"{sym.c}.data != NULL")
            if isinstance(storage, Scalar) and storage.floating:
                range_assumptions.extend(self.float_range_predicates(node.token, sym.c, path.values[sym.uid]))
                continue
            if isinstance(storage, Scalar):
                domain = path.values[sym.uid]
                physical_domain = container(storage.bits, storage.signed)
                value = sym.c
            elif isinstance(ty, Slice):
                domain = path.values[sym.uid]
                physical_domain = container(64, False)
                value = f"{sym.c}.len"
                range_assumptions.extend(self.slice_data_predicates(sym.c, domain))
            else:
                continue
            range_assumptions.extend(self.integer_range_predicates(node.token, value, domain, physical_domain))
        assumptions = range_assumptions + assumptions
        if assumptions:
            assume = self.helper("contract_assume", None)
            entry_code[:0] = [f"{assume}({predicate});" for predicate in assumptions]
            self.trace.extend(f"  assume entry {predicate}" for predicate in assumptions)
        self.entry_offset_origins = {
            path.forms[sym.uid].origin: (("len", index, 32) if isinstance(ty, Slice) else ("param", index))
            for index, (sym, ty) in enumerate(zip(parameter_symbols, contract.params))
            if (plain_integer_parameter(ty) or isinstance(ty, Slice)) and sym.uid in path.forms
        }
        # Parameters and top-level locals share the function's scope.
        local_names = {s.args[0].text for s in node.args[3] if s.kind == "define"}
        for param in node.args[2]:
            if param.text in local_names:
                param.error(f"local definition repeats parameter {param.text!r}")
        if (name not in self.recursive_functions and isinstance(contract.result, Scalar) and contract.result.boolean
                and all(guard_parameter_kind(ty) for ty in contract.params)
                and self.function_guard(node, contract) is None):
            guards = []
            for index, (param, ty) in enumerate(zip(parameter_symbols, contract.params)):
                kind = guard_parameter_kind(ty)
                if kind == "array_param":
                    continue
                operand = ("len" if kind == "slice_param" else kind, index)
                path = path.touch(param.uid)
                guards.append((operand, param.uid, path.epochs[param.uid], unqualified(ty).domain))
            self.guard_parameters = tuple(guards)
        if not self.inferring_results and (contract.result is None or replay_scalar(contract.result)):
            self.scalar_return_inputs = tuple(
                (index, sym.uid, path.epochs.get(sym.uid), path.values[sym.uid])
                for index, (sym, ty) in enumerate(zip(parameter_symbols, contract.params))
                if replay_scalar(unqualified(ty)))[:8]
        identity_entry = replace(path) if contract.identity_property and property_has_loop else None
        replay = None
        exchange = None
        if ((contract.result is None or replay_scalar(contract.result))
                and all(replay_scalar(ty) for ty in contract.params)
                and not (contract.bounds or contract.result_expressions or contract.parameter_expressions
                         or contract.identity_property or contract.idempotent_property
                         or contract.commutative_property or contract.diagonal_property)):
            def shape(ty):
                return ty.boolean, ty.domain.lo, ty.domain.hi

            inputs = {param.text: shape(ty) for param, ty in zip(node.args[2], contract.params)}
            scalar_types = {key: shape(ty) for key, ty in self.types.items() if replay_scalar(ty)}
            exchange = TheoremExchange(name)
            replay = replay_function(inputs, node.args[3], scalar_types,
                                     self.scalar_replay_signature, exchange=exchange)
        self.optimization_replay = replay
        outer_pinned = self.current_pinned_baseline()
        outer_future = self._future_stmts
        outer_contract_live = self._contract_live_names
        self._future_stmts = []
        self._contract_live_names = self._collect_ast_names((node.args[1],))
        pinned_epochs = {item[2] for item in getattr(self, "guard_parameters", ())}
        pinned_epochs.update(item[2] for item in getattr(self, "scalar_return_inputs", ()) if item[2] is not None)
        pinned_origins = set(getattr(self, "entry_offset_origins", ()))
        if identity_entry is not None:
            id_ep, id_org = PathState.baseline((identity_entry,))
            pinned_epochs |= id_ep
            pinned_origins |= id_org
        self.pinned_baseline = (outer_pinned[0] | frozenset(pinned_epochs),
                                outer_pinned[1] | frozenset(pinned_origins))
        try:
            entries = [path]
            for index in self.owned_chain_parameters.get(name, ()):
                nodes = self.owned_chain_nodes(parameter_symbols[index], OWNED_CHAIN_LONG, node.args[2][index])
                entries = [self.restore_owned_chain(nodes, length, entry)
                           for entry in entries for length in range(1, OWNED_CHAIN_LONG + 1)]
            fallthrough, code = self.block(node.args[3], entries)
        finally:
            self.optimization_replay = None
            self.pinned_baseline = outer_pinned
            self._future_stmts = outer_future
            self._contract_live_names = outer_contract_live
        if exchange is not None and exchange.accepted and not self.inferring_results:
            self.proof_exchanges[name] = exchange
        self.trace_stack_exit(fallthrough, 0, "implicit return")
        for path in fallthrough:
            for uid, index in sorted(self.parameter_uids.items(), key=lambda item: item[1]):
                if path.heap.get(uid) == frozenset(("Alive",)):
                    node.args[2][index].error(
                        f"heap leak: {self.symbols[uid - 1].name!r} leaves the function before free")
        self.return_states.extend(fallthrough)
        if self.inferred_returns is not None:
            result = self.inferred_result(node.args[0], fallthrough)
            contract = replace(contract, result=result)
            self.functions[name] = node, contract
            encoded = json.dumps(encode_interface_type(contract), sort_keys=True, ensure_ascii=True)
            self.trace.append(f"  implicit contract {name} {encoded}")
        self.inferred_returns = None
        linkage = "static __attribute__((unused)) " if self.internal_linkage else ""
        head = (f"{linkage}{contract.result.c if contract.result else 'void'} "
                f"{self.function_c_name(name)}({', '.join(params) or 'void'})")
        mutations = []
        places = {place: self.symbols[uid - 1] for uid, place in self.param_places.items()}
        slice_posts = {}
        for index in {index for index, _ in self.writes}:
            ty = contract.params[index]
            if isinstance(ty, Slice) and isinstance(ty.element, Scalar):
                sym = places[index, ()]
                bounds = []
                for path in self.return_states:
                    # A rebound parameter has a separate owner without a descriptor value.
                    length_domain = path.values.get(sym.uid, ty.domain)
                    if length_domain.hi == 0:
                        continue
                    indices = Domain.of((0, length_domain.hi - 1))
                    contents = self.actual_array(path, sym).read(indices)
                    length = None
                    if sym.uid in path.values:
                        length = Value(length_domain, "", affine=path.forms.get(sym.uid),
                                       linear=path.linear.get(sym.uid))
                    key = ("array_element", sym.uid, (frozenset(), 0))
                    bounds.append(self.known_array_window(path, key, indices, contents, span_length=length))
                post = domain_union(bounds)
                if post and post != ty.element.domain:
                    slice_posts[index] = post
        aggregate = {index for index, post in slice_posts.items()
                     if (index, ()) in self.writes or contract.params[index].content_bound is None
                     or not contract.params[index].content_bound.within(post)}
        writes = set()
        for index, names in self.writes:
            if (index, names) in self.interface_before_writes:
                names = ("@interface_before", *names)
            elif (index, names) in self.interface_after_writes:
                names = ("@interface_after", *names)
            writes.add((index, () if index in aggregate else names))
        self.normalize_interface_rebinds(writes, node.token)
        rebinds = {(index, names[1]) for index, names in writes
                   if names and names[0] == "@interface_rebind"}
        if rebinds and not valid_interface_rebind_effects(contract.params, writes):
            node.token.error("borrowed interface rebinding requires distinct fields per holder and supported record writes")
        for index, names in sorted(writes):
            if names and names[0] in ("@interface_before", "@interface_after"):
                sym = places.get((index, names[1:]))
                post = (domain_union(p.values[sym.uid] for p in self.return_states)
                        if isinstance(sym.ty, Scalar) and self.return_states else None)
                mutations.append((index, names, post))
                when = "before" if names[0] == "@interface_before" else "after"
                self.trace.append(f"  mutation param#{index}.{'.'.join(names[1:])} => "
                                  f"{post or 'content'} ({when} rebind)")
                continue
            if names and names[0] == "@interface_rebind":
                if contract.result is not None and not isinstance(contract.result, Scalar):
                    node.token.error("borrowed interface rebinding requires a scalar or void result")
                mutations.append((index, names, None))
                if len(names) == 6:
                    guard = f" when param#{names[3]} == {names[4]} else param#{names[5]}"
                elif len(names) >= 5:
                    # Odd lengths beyond five are conjunctions of literals.
                    guard = " when " + " and ".join(
                        f"param#{names[position]} == {names[position + 1]}"
                        for position in range(3, len(names) - 1, 2))
                else:
                    guard = ""
                self.trace.append(f"  mutation param#{index}.{names[1]}.target => param#{names[2]}{guard}")
                continue
            if names and names[0] in ("@graph", "@chain", "@prune", "@skip", "@attach", "@attach_deep", "@attach_at", "@attach_graph"):
                mutations.append((index, names, None))
                self.trace.append(f"  mutation param#{index} => {names[0][1:]}")
                continue
            if any(map(array_mutation_step, names)):
                post = slice_posts.get(index)
                ty = contract.params[index]
                if rebinds:
                    # The array footprint identifies possible writes; avoid
                    # promising a fill when rebind receivers may share donors.
                    post = None
                elif isinstance(ty, Slice) and isinstance(ty.element, Scalar):
                    sym = places[index, ()]
                    (indices,), _ = array_mutation_projection(ty, names)
                    if self.return_states and all(
                            indices.hi < path.values.get(sym.uid, ty.domain).lo for path in self.return_states):
                        values = domain_union(self.actual_array(path, sym).read(indices) for path in self.return_states)
                        if values and values != ty.element.domain:
                            post = values
                elif array_mutation_step(names[-1]) and (sym := places.get((index, names[:-1]))) is not None:
                    storage = sym.ty
                    if isinstance(storage, Array) and isinstance(storage.element, Scalar) and self.return_states:
                        (indices,), _ = array_mutation_projection(storage, names[-1:])
                        values = domain_union(self.actual_array(path, sym).read(indices) for path in self.return_states)
                        if values and values != storage.element.domain:
                            post = values
                mutations.append((index, names, post))
                self.trace.append(f"  mutation param#{index}.{'.'.join(names)} => {post or 'content'}")
                continue
            sym = places[index, names]
            post = (domain_union(p.values[sym.uid] for p in self.return_states)
                    if isinstance(sym.ty, Scalar) and self.return_states else None)
            if isinstance(sym.ty, Slice):
                post = slice_posts.get(index)
            mutations.append((index, names, post))
            self.trace.append(f"  mutation param#{index}.{'.'.join(names)} => {post or 'content'}")
        if self.return_states:
            for index, ty in enumerate(contract.params):
                if not (isinstance(ty, BorrowRecord) and mutable_graph(ty)):
                    continue
                parameter = node.args[2][index].text
                if ty.attach is not None:
                    coarse = {(index, (tag,), None) for tag in ("@attach_graph", "@prune")
                              if (index, (tag,)) in writes}
                    if not coarse:
                        continue
                    donor = node.args[2][ty.attach].text
                    donor_record = contract.params[ty.attach].record
                    effects, payload = graph_write_program(
                        node.args[3], parameter,
                        lambda call: self.forward_graph_writes(call, parameter, donor),
                        donor=donor, allow_cut=ty.prune, record=ty.record)
                    if effects and all(graph_write_path(ty.record, route, donor_record if operation == "attach" else None)
                                       for operation, _, route in effects):
                        mutations = [effect for effect in mutations if effect not in coarse]
                        if payload and (index, ("@graph",), None) not in mutations:
                            mutations.append((index, ("@graph",), None))
                        for position, (operation, mode, route) in enumerate(effects):
                            path = (("@graph_path_attach", mode, *route) if len(effects) == 1 and operation == "attach"
                                    else ("@graph_path_edit", str(position), operation, mode, *route))
                            mutations.append((index, path, None))
                            detail = f"attach param#{ty.attach}" if operation == "attach" else "null"
                            if mode == "may":
                                detail = f"possibly {detail}"
                            self.trace.append(f"  mutation param#{index}.{'.'.join(route)} => {detail}")
                    continue
                if not ty.prune or (index, ("@prune",)) not in writes:
                    continue
                root = parameter_symbols[index]
                effects, payload = graph_write_program(
                    node.args[3], parameter,
                    lambda call: self.forward_graph_writes(call, parameter), record=ty.record)
                if effects and all(graph_write_path(ty.record, route) for _, _, route in effects):
                    mutations.remove((index, ("@prune",), None))
                    if payload and (index, ("@graph",), None) not in mutations:
                        mutations.append((index, ("@graph",), None))
                    for position, (_, mode, route) in enumerate(effects):
                        mutations.append((index, ("@graph_path_prune", str(position), mode, *route), None))
                        detail = "possibly null" if mode == "may" else "null when reachable" if len(route) > 1 else "null"
                        self.trace.append(f"  mutation param#{index}.{'.'.join(route)} => {detail}")
                    continue
                for name_of_field, field_type in ty.record.fields:
                    if not isinstance(field_type, RecordLink):
                        continue
                    field = self.child_place(root, (name_of_field,))
                    if all(self.link_targets(field, state) == frozenset((self.null_symbol,))
                           for state in self.return_states):
                        mutations.append((index, ("@graph_root_null", name_of_field), None))
                        self.trace.append(f"  mutation param#{index}.{name_of_field} => null")
        if name in self.recursive_functions:
            # Calls inside the group were analysed against the fixed point of the
            # group's write summaries, so the exported summary cannot be narrower.
            declared = self.recursive_summaries.get(name, ())
            recorded = {(index, path) for index, path, _ in mutations}
            mutations.extend(entry for entry in declared if (entry[0], entry[1]) not in recorded)
        self.mutations[name] = tuple(mutations)
        self.owning[name] = tuple(sorted(self.consumed_parameters.items()))
        if self.owning[name]:
            labels = ", ".join(f"param#{index} {node.args[2][index].text} ({method})"
                               for index, method in self.owning[name])
            self.trace.append(f"  owning parameters {name}: {labels}")
        if self.guard_return_observations and not mutations:
            outcomes = [self.verified_guard_outcome(outcome) for outcome in (True, False)]
            if outcomes != [("bool", True), ("bool", True)]:
                self.verified_guards[name] = ("outcomes", *outcomes)
        payload = success_type(contract.result)
        if self.owned_chain_return_lengths - {1}:
            self.owned_chain_results[name] = tuple(sorted(self.owned_chain_return_lengths))
            self.trace.append(f"  owning result {name}: chain lengths {self.owned_chain_results[name]}")
        else:
            self.owned_chain_results.pop(name, None)
        if (self.array_return_contents and isinstance(payload, CopyArray)
                and not self.contains_record_links(payload.array)
                and all(value is not None for value in self.array_return_contents)):
            contents = join_result_contents(self.array_return_contents)
            if result_content_size(contents) <= 4096:
                self.array_results[name] = contents
        if self.record_return_fields:
            common = set.intersection(*(set(fields) for fields in self.record_return_fields))
            budget = 4096
            fields = []
            candidates = [(path, join_result_contents(state[path] for state in self.record_return_fields))
                          for path in sorted(common)]
            for path, value in sorted(candidates, key=lambda item: result_content_size(item[1])):
                cost = result_content_size(value)
                if cost <= budget:
                    fields.append((path, value))
                    budget -= cost
            self.record_results[name] = tuple(sorted(fields))
        if contract.result:
            if fallthrough:
                node.args[0].error("a reachable path does not return a value")
            code.append("__builtin_unreachable();")
        if contract.idempotent_property or identity_entry is not None:
            parameter = parameter_symbols[0]
            if identity_entry is not None:
                fixed_point_entry = identity_entry
            else:
                reachable_results = Domain.union(self.idempotent_return_domains) or contract.result.domain
                fixed_point_entry = PathState({}).update(parameter, reachable_results)
            symbol_count, trace_count = len(self.symbols), len(self.trace)
            rhs_codes, helpers = self.rhs_codes.copy(), self.helpers.copy()
            try:
                for token, value, path in self.property_returns(node.args[3], [fixed_point_entry], node.token):
                    entry = self.storage_value(parameter, path)
                    if not self.prove_order(token, value, "==", entry, path):
                        if contract.identity_property:
                            token.error("cannot prove identity property on the declared parameter domain")
                        token.error("cannot prove idempotent property on the reachable result domain")
            finally:
                del self.symbols[symbol_count:]
                del self.trace[trace_count:]
                self.rhs_codes, self.helpers = rhs_codes, helpers
        if contract.identity_property:
            self.trace.append(f"  theorem property {name}: forall(argument 0) => result == argument 0")
        if contract.idempotent_property:
            self.trace.append(f"  theorem property {name}: forall(x) => {name}({name}(x)) == {name}(x)")
        if contract.commutative_property:
            a, b = parameter_symbols
            entry = PathState.with_independent_values({a.uid: a.ty.domain, b.uid: b.ty.domain})
            symbol_count, trace_count = len(self.symbols), len(self.trace)
            rhs_codes, helpers = self.rhs_codes.copy(), self.helpers.copy()
            original = self.scope[-1]
            try:
                first_returns = self.property_returns(node.args[3], [entry], node.token)
                self.scope[-1] = original | {a.name: b, b.name: a}
                for _, first, first_path in first_returns:
                    for token, second, common in self.property_returns(node.args[3], [replace(first_path)], node.token):
                        if not self.prove_order(token, first, "==", second, common):
                            token.error("cannot prove commutative property")
            finally:
                self.scope[-1] = original
                del self.symbols[symbol_count:]
                del self.trace[trace_count:]
                self.rhs_codes, self.helpers = rhs_codes, helpers
            self.trace.append(f"  theorem property {name}: forall(x,y) => {name}(x,y) == {name}(y,x)")
        if contract.diagonal_property:
            first, second = parameter_symbols
            entry = PathState.with_independent_values({first.uid: first.ty.domain})
            shared = self.storage_value(first, entry)
            entry = entry.update(second, shared.domain, shared.affine, shared.difference, shared.linear)
            symbol_count, trace_count = len(self.symbols), len(self.trace)
            rhs_codes, helpers = self.rhs_codes.copy(), self.helpers.copy()
            try:
                for token, value, path in self.property_returns(node.args[3], [entry], node.token):
                    if not self.prove_order(token, value, "==", shared, path):
                        token.error("cannot prove diagonal property")
            finally:
                del self.symbols[symbol_count:]
                del self.trace[trace_count:]
                self.rhs_codes, self.helpers = rhs_codes, helpers
            self.trace.append(f"  theorem property {name}: forall(x) => {name}(x,x) == x")
        if not self.inferring_results:
            self.collect_storage_observations(name, contract)
            self.storage_counts[name] = self.inferred_storage_counts(name, contract)
            if (name not in self.recursive_functions and self.result_exit_domains
                    and len(self.result_exit_domains) == len(self.return_states)):
                try:
                    actual = domain_union(self.result_exit_domains)
                    if actual and actual != contract.result.domain and actual.within(contract.result.domain):
                        self.result_domains[name] = actual
                except DomainComplexity:
                    pass
            if (replay_scalar(contract.result) and self.scalar_return_inputs
                    and len(self.scalar_return_cases) == len(self.return_states)):
                cases = tuple(dict.fromkeys(case for group in self.scalar_return_cases for case in group))
                if 2 <= len(cases) <= 32:
                    indices = tuple(spec[0] for spec in self.scalar_return_inputs)
                    observation = ReturnObservation(indices, cases)
                    self.semantic_module.return_observations[name] = observation
                    if (any(row.result != cases[0].result for row in cases)
                            and any(row.arguments != cases[0].arguments for row in cases)
                            and all(small_summary_domain(domain)
                                    for row in cases for domain in (row.result, *row.arguments))):
                        self.call_return_observations[name] = observation
        return head, [f"{head} {{", *indent(entry_code + code), "}"]

    def observe_scalar_return(self, value, case_paths):
        if (not self.scalar_return_inputs or value.floating or value.enum or value.character
                or value.result_arms is not None or len(self.scalar_return_cases) >= 128):
            return
        cases = []
        for result, state in case_paths:
            summaries = self.saved_return_cases(state, value)
            alternatives = [(result, state)]
            if summaries and value.affine.scale:
                form, rows, places = summaries[0]
                try:
                    alternatives = []
                    for row in rows:
                        compatible = result & form.preimage(row.result).affine(value.affine.scale, value.affine.offset)
                        if compatible:
                            refined = self.constrain_call_arguments(state, row.arguments, places)
                            if refined is not None:
                                alternatives.append((compatible, refined))
                except DomainComplexity:
                    alternatives = [(result, state)]
            cases.extend(ReturnCase(selected, tuple(
                path.values.get(uid, initial) if path.epochs.get(uid) is epoch else initial
                for _, uid, epoch, initial in self.scalar_return_inputs)) for selected, path in alternatives)
        if cases:
            self.scalar_return_cases.append(tuple(cases))

    def derive_return_theorems(self):
        for name, observation in self.semantic_module.return_observations.items():
            try:
                conclusions = tuple(return_implications(observation))
            except DomainComplexity:
                continue
            if not conclusions:
                continue
            node, _ = self.functions[name]
            evidence = len(self.return_evidence)
            self.return_evidence.append((name, node.semantic_anchor, observation))
            for premises, (target, domain) in conclusions:
                predicate = Proposition("return_implication",
                                        (tuple((key, encode_domain(values)) for key, values in premises),
                                         (target, encode_domain(domain))))
                self.source_theorems.append(Theorem(predicate, node.semantic_anchor, "return_case",
                                                    (node.token.line, node.token.col), function=name,
                                                    evidence=evidence))
