"""Alias binding, liveness and heap allocation / release."""

from __future__ import annotations

from dataclasses import field, fields, replace
from itertools import count

from shape_domain import CHAIN, DAG, REPEAT, TREE, LinkGraph
from array_domain import ArrayDomain
from value_domain import Domain
from table_interface import c_field

from .errors import DANGLING_RELEASE_HEADLINE, LanError, OracleRequest, OracleRequestError
from .syntax import Node
from .typesys import (AliasIndexExpression, AliasParamIndex, AliasResult, Array, BitView, BOOL, BorrowRecord,
                      BorrowScalar, HeapArray, HeapObject, HeapRecord, Interface, Record, RecordLink, Slice,
                      SliceEndIndex, Span, unqualified, Value, WIDE, ZERO)
from .domain_utils import c_int, plain_integer_parameter, zero_domain
from .interface_types import (alias_graph_records, chain_reachable_path, linked_alias_path,
                              linked_owned_record, offset_expression_domain)

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class AliasHeapMixin:
    def alias_node(self, node):
        sym = self.place(node)
        return sym if node.kind == "name" and sym is not None and sym.uid in self.alias_types else None

    @staticmethod
    def alias_storage(ty):
        if isinstance(ty, BorrowRecord):
            return ty.record
        return ty.scalar if isinstance(ty, BorrowScalar) else ty.array

    def alias_source(self, node, state, ty, token, *, capture_exit=None):
        if self.contains_interface_storage(ty):
            token.error("a record with interface fields cannot itself be borrowed")
        if node is None or node.kind == "null":
            targets, code = frozenset((self.null_symbol,)), "NULL"
        elif isinstance(self.composite_call_type(node), AliasResult):
            arguments, actual_types = [], []
            result, code = self.call(node, state, arguments, capture_argument_types=actual_types,
                                     capture_early_return=capture_exit)
            if capture_exit:
                return frozenset(), code
            if self.alias_storage(result.alias) != self.alias_storage(ty) or result.alias.readonly and not ty.readonly:
                token.error("returned alias requires the same target type and compatible mutability")
            def actual_step(step, value):
                if isinstance(step, AliasParamIndex):
                    domain = unqualified(actual_types[step.parameter]).domain.affine(1, step.offset)
                elif isinstance(step, AliasIndexExpression):
                    try:
                        domain = offset_expression_domain(step.expression, actual_types)
                    except ValueError:
                        domain = WIDE
                else:
                    return step
                if value is not None:
                    domain = (domain & value.domain) or domain
                linear = self.linear_value(value, prefer_symbolic=True) if value is not None else None
                if linear is not None and not linear[0]:
                    domain = (domain & Domain.point(linear[1])) or domain
                return domain.lo if domain.lo == domain.hi else domain

            def actual_index(step):
                if isinstance(step, AliasIndexExpression):
                    operands = [replace(value, width=unqualified(parameter).bits,
                                        signed=unqualified(parameter).signed)
                                if plain_integer_parameter(parameter) else value
                                for value, parameter in zip(arguments, actual_types)]
                    try:
                        return self.integer_expression_value(token, step.expression, operands, state,
                                                             refine_bounds=True)
                    except (LanError, ValueError):
                        # The validated domain remains usable without a symbolic relation.
                        return None
                if not isinstance(step, AliasParamIndex):
                    return None
                value = arguments[step.parameter]
                if not step.offset:
                    return value
                offset = Value(Domain.point(step.offset), c_int(step.offset))
                return self.binary(token, "+", value, offset)

            parameters = self.composite_call_signature(node).params
            graph_records = alias_graph_records(parameters, result.alias, result.graph_records)
            targets = set()
            for source, path in result.origins():
                index_values = tuple(actual_index(step) for step in path)
                path = tuple(actual_step(step, value) for step, value in zip(path, index_values))
                argument = arguments[source]
                if isinstance(argument, Span):
                    step, *tail = path
                    index_value, index_shift = index_values[0], 0
                    if isinstance(step, SliceEndIndex):
                        index_value, index_shift = argument.length, -step.distance
                        indices = index_value.domain.affine(1, index_shift)
                    else:
                        indices = Domain.point(step) if isinstance(step, int) else step
                    indices &= Domain.of((0, argument.length.domain.hi - 1))
                    if not indices:
                        continue
                    elements = self.span_element_places(argument, indices, state,
                                                        index_value=index_value, index_shift=index_shift)
                    targets.update(self.alias_result_projection(target, tail, index_values[1:]) for target in elements)
                elif chain_reachable_path(parameters[source], path):
                    roots = self.physical_places(argument, state, allow_null=True)
                    for target in self.chain_nodes(state, roots, token):
                        self.require_target_live(target, state, token)
                        targets.add(target)
                elif linked_alias_path(parameters[source], path, graph_records):
                    reachable = set(self.physical_places(argument, state, allow_null=True))
                    for step in path:
                        successors = set()
                        for target in reachable:
                            if target.uid:
                                self.require_target_live(target, state, token)
                                successors.update(self.link_targets(self.child_place(target, (step,)), state))
                            else:
                                successors.add(target)
                        reachable = successors
                    if not result.alias.nullable:
                        reachable.discard(self.null_symbol)
                    targets.update(reachable)
                else:
                    targets.update(self.alias_result_projection(target, path, index_values)
                                   for target in self.physical_places(argument, state) if target.uid)
            targets = frozenset(targets)
            if result.alias.nullable:
                targets |= {self.null_symbol}
        elif isinstance(ty, BorrowRecord) and (source := self.place(node)) is not None and isinstance(source.ty, RecordLink):
            if source.ty.identity != ty.record.identity or (source.readonly or source.ty.readonly) and not ty.readonly:
                token.error("record link requires a compatible record alias")
            propagate = capture_exit is not None and self.has_propagation(node)
            source = self.comparison_place(source, state, allow_effectful_links=propagate, propagate_exit=propagate)
            if source.projection is not None and source.projection.always_returns:
                capture_exit.append(Value(ZERO, source.c, effectful=True, always_returns=True))
                return frozenset(), source.c
            self.require_live(source, state, token)
            targets, code = self.link_targets(source, state), source.c
        else:
            source = self.place(node)
            if (source is None or source.projection is not None or source.bit_overlay
                    or source.ty != self.alias_storage(ty)):
                token.error("alias requires addressable storage of the same declared type")
            if source.readonly and not ty.readonly:
                token.error("cannot remove an immutable qualification through an alias")
            if source.link_access is not None and capture_exit is not None and self.has_propagation(node):
                source = self.comparison_place(source, state, allow_effectful_links=True, propagate_exit=True)
                if source.projection is not None and source.projection.always_returns:
                    capture_exit.append(Value(ZERO, source.c, effectful=True, always_returns=True))
                    return frozenset(), source.c
            if node.kind != "name":
                self.require_live(source, state, token)
            targets = frozenset(self.physical_places(source, state, allow_null=True))
            if source.uid in self.alias_types:
                code = source.c_var
            else:
                code = source.c if isinstance(source.ty, Array) else f"&({source.c})"
        if not ty.nullable and self.null_symbol in targets:
            token.error("non-null alias requires an initializer proved non-null")
        return targets, code

    def alias_binding_assumptions(self, sym, paths):
        if any(self.null_symbol in path.borrows[sym.uid] for path in paths):
            return []
        predicate = f"{sym.c_var} != NULL"
        self.trace.append(f"  assume alias binding {predicate}")
        assume = self.helper("contract_assume", None)
        return [f"{assume}({predicate});"]

    def binding_initializer_paths(self, node, paths, resolve):
        self.propagating_expression_nodes |= self.propagation_nodes(node, composite_call=True)
        continuing, bindings = [], []
        exit_code = None
        for path in paths:
            early = []
            binding = resolve(path, early)
            if early:
                candidate = f"(void)({early[0].code});"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            continuing.append(path)
            bindings.append(binding)
        return continuing, bindings, exit_code

    def define_alias(self, name, ty, node, paths):
        paths, bindings, exit_code = self.binding_initializer_paths(
            node, paths, lambda path, early: self.alias_source(node, path, ty, name, capture_exit=early))
        if not paths:
            return [], [exit_code, "__builtin_unreachable();"]
        sym = self.declare(name, self.alias_storage(ty), readonly=ty.readonly, borrowed=True)
        self.alias_types[sym.uid], self.alias_members[sym.uid] = ty, (sym.uid, ())
        if isinstance(ty, BorrowRecord):
            self.record_symbols(sym, name, zero=False)
        result = []
        for path, (targets, _) in zip(paths, bindings):
            if any(target.depth > sym.depth and not self.is_whole_heap_object(target)
                   for target in targets):
                name.error("alias would outlive its target storage")
            bound = replace(path.update(sym, ZERO), borrows=path.borrows | {sym.uid: targets})
            result.append(self.note_loop_cursor_binding(bound, sym, node, targets))
            self.trace.append(f"  alias id#{sym.uid} -> {self.trace_alias_targets(targets)}")
        return result, [f"{ty.c} {sym.c_var} = {bindings[0][1]};",
                        *self.alias_binding_assumptions(sym, result)]

    def rebind_alias(self, sym, op, node, paths):
        if op.text != "=":
            op.error("alias rebind uses '='")
        ty = self.alias_types[sym.uid]
        paths, bindings, exit_code = self.binding_initializer_paths(
            node, paths, lambda path, early: self.alias_source(node, path, ty, op, capture_exit=early))
        if not paths:
            return [], [exit_code, "__builtin_unreachable();"]
        self.note_transition_write(sym, alias_binding=True)
        result = []
        for path, (targets, code) in zip(paths, bindings):
            if any(target.depth > sym.depth and not self.is_whole_heap_object(target)
                   for target in targets):
                op.error("alias rebind would outlive its target storage")
            bound = replace(path.touch(sym.uid), borrows=path.borrows | {sym.uid: targets})
            result.append(self.note_loop_cursor_binding(bound, sym, node, targets))
            guard = path.guard_label()
            self.trace.append(f"  rebind alias id#{sym.uid} -> {self.trace_alias_targets(targets)} [{guard}]")
        return result, [f"{sym.c_var} = {code};", *self.alias_binding_assumptions(sym, result)]

    def link_storage_paths(self, sym, state, repeated):
        if sym.projection is not None:
            yield sym, state
        elif sym.link_access is not None:
            parent, name = sym.link_access
            for field, path in self.link_storage_paths(parent, state, repeated):
                if isinstance(parent.ty, RecordLink):
                    indirect = field.chain_summary or field.projection is not None or field.heap_owner in repeated
                    choices = self.link_targets(parent, path) if indirect else path.borrows[field.uid]
                    for owner in choices - {self.null_symbol}:
                        narrowed = path if indirect else replace(
                            path, borrows=path.borrows | {field.uid: frozenset((owner,))})
                        yield self.child_place(owner, (name,)), narrowed
                else:
                    yield self.child_place(field, (name,)), path
        elif sym.uid in self.alias_members:
            root, names = self.alias_members[sym.uid]
            for owner in state.borrows[root] - {self.null_symbol}:
                narrowed = replace(state, borrows=state.borrows | {root: frozenset((owner,))})
                yield self.child_place(owner, names), narrowed
        else:
            yield sym, state

    @staticmethod
    def link_address_domain(left, right, repeated=()):
        if not left or not right:
            return BOOL
        left, right = ({target if target is not None and target.uid else None for target in side}
                       for side in (left, right))
        if len(left) == len(right) == 1 and left == right:
            target = next(iter(left))
            if target is None or not target.chain_summary and target.projection is None and target.uid not in repeated:
                return Domain.point(1)
        if not any(a == b or (a is not None and b is not None and (
                       a.borrowed and b.borrowed
                       or a.uid == b.uid and (a.projection is not None or b.projection is not None)))
                   for a in left for b in right):
            return ZERO
        return BOOL

    def link_condition(self, node, state):
        subject, inverse = node, False
        if node.kind == "binary" and node.args[0] in ("==", "!="):
            op, left, right = node.args
            if left.kind == "null":
                left, right = right, left
            if right.kind != "null":
                a, b = self.place(left), self.place(right)
                if a is None or b is None or not any(isinstance(sym.ty, RecordLink) for sym in (a, b)):
                    return None
                if not all(isinstance(sym.ty, (Record, RecordLink)) for sym in (a, b)) or a.ty.identity != b.ty.identity:
                    node.token.error("record link comparison requires the same target type")
                addresses, targets = [], []
                for sym in (a, b):
                    place = self.comparison_place(sym, state, allow_effectful_links=True, propagate_exit=True)
                    if place.projection is not None and place.projection.always_returns:
                        parts = [f"(void)({address})" for address in addresses] + [f"(void)({place.c})", "0"]
                        return [], [], f"({', '.join(parts)})"
                    if isinstance(sym.ty, RecordLink):
                        self.require_live(place, state, node.token)
                        addresses.append(place.c)
                        targets.append(self.link_targets(place, state))
                    else:
                        if sym.link_access is not None:
                            self.require_live(place, state, node.token)
                        addresses.append(sym.c_var if sym.uid in self.alias_types
                                         else f"&({place.c})")
                        targets.append(frozenset(self.physical_places(place, state, allow_null=True)))
                domain = (Domain.point(1) if addresses[0] == addresses[1] else self.link_address_domain(
                    *targets, self.repeated_allocation_uids(state, node.token)))
                yes, no = bool(domain.hi), not domain.lo
                if op == "!=":
                    yes, no = no, yes
                code = f"({addresses[0]} {op} {addresses[1]})"
                if self.has_effects(node):
                    helper = self.helper("binary_eval", None, ("uint8_t", "const void *", "const void *", f"(a {op} b)"))
                    code = f"{helper}({addresses[0]}, {addresses[1]})"
                    if self.has_effects(node, allow_debug_checks=True):
                        yes = no = True
                return [state] if yes else [], [state] if no else [], code
            subject, inverse = left, op == "=="
        elif node.kind == "in" and node.args[1].kind == "type_name" and node.args[1].args[0] == "null":
            subject, inverse = node.args[0], True
        sym = self.place(subject)
        if sym is None or not isinstance(sym.ty, RecordLink):
            return None
        if sym.projection is not None and self.has_effects(subject, allow_debug_checks=True):
            sym = self.comparison_place(sym, state, allow_effectful_links=True, propagate_exit=True)
            if sym.projection.always_returns:
                return [], [], sym.c
        self.require_live(sym, state, node.token)
        yes, no = [], []
        repeated = self.repeated_allocation_uids(state, node.token)
        for field, path in self.link_storage_paths(sym, state, repeated):
            indirect = field.chain_summary or field.projection is not None
            targets = self.link_targets(sym, path) if indirect else path.borrows[field.uid]
            for branch, remaining, nonnull in ((yes, targets - {self.null_symbol}, True),
                                               (no, targets & {self.null_symbol}, False)):
                if remaining:
                    if field.heap_owner in repeated:
                        # This guard observes one runtime instance of the site.
                        branch.append(path)
                    elif indirect:
                        key = (self.projected_link_guard_key(sym, field, path) if field.projection is not None
                               else self.chain_access_key("chain_link", sym, path))
                        branch.append(replace(path, predicates=path.predicates | {key: nonnull}))
                    else:
                        branch.append(replace(path, borrows=path.borrows | {field.uid: frozenset(remaining)}))
            self.budget_paths(yes + no, node.token)
        address = self.comparison_place(sym, state).c
        code = f"!({address})" if inverse else f"({address} != NULL)"
        return (no, yes, code) if inverse else (yes, no, code)

    def alias_condition(self, node, state):
        linked = self.link_condition(node, state)
        if linked is not None:
            return linked
        sym = self.alias_node(node)
        inverse = False
        if sym is None and node.kind == "binary" and node.args[0] in ("==", "!="):
            op, left, right = node.args
            a, b = self.alias_node(left), self.alias_node(right)
            if a is not None and b is not None and not (isinstance(a.ty, Record) and isinstance(b.ty, Record)):
                if a.ty != b.ty:
                    node.token.error("alias comparison requires the same target type")
                x, y = state.borrows[a.uid], state.borrows[b.uid]
                equal = a.uid == b.uid or (len(x) == len(y) == 1 and x == y
                                           and not next(iter(x)).chain_summary and next(iter(x)).projection is None)
                distinct = not any(p == q or p.borrowed and q.borrowed
                                   or p.uid == q.uid and (p.projection is not None or q.projection is not None)
                                   for p in x for q in y)
                yes, no = not distinct, not equal
                if op == "!=":
                    yes, no = no, yes
                code = f"({a.c_var} {op} {b.c_var})"
                return [state] if yes else [], [state] if no else [], code
        if sym is None and node.kind == "in" and node.args[1].kind == "type_name" and node.args[1].args[0] == "null":
            sym, inverse = self.alias_node(node.args[0]), True
        if sym is None and node.kind == "binary" and node.args[0] in ("==", "!="):
            op, left, right = node.args
            if left.kind == "null":
                left, right = right, left
            if right.kind == "null":
                sym, inverse = self.alias_node(left), op == "=="
        if sym is None:
            return None
        targets = state.borrows[sym.uid]
        nonnull, null = targets - {self.null_symbol}, targets & {self.null_symbol}
        yes = [replace(state, borrows=state.borrows | {sym.uid: frozenset(nonnull)})] if nonnull else []
        no = [replace(state, borrows=state.borrows | {sym.uid: frozenset(null)})] if null else []
        code = f"({sym.c_var} {'==' if inverse else '!='} NULL)"
        return (no, yes, code) if inverse else (yes, no, code)

    def require_target_live(self, target, state, token, access=None):
        if not target.uid:
            token.error("cannot dereference an alias that may be null")
        owner = target.heap_owner or target.uid
        statuses = state.heap.get(owner, frozenset(("Alive",)))
        if statuses != frozenset(("Alive",)):
            access = access or target
            entity = self.symbols[owner - 1]
            error = OracleRequestError(f"{token.line}:{token.col}: heap entity {target.name!r} "
                                       "may be a Tombstone (use after free or ownership transfer)")
            error.request = OracleRequest(
                self.source_path, self.source_digest, token.line, token.col,
                (f"Access: id#{access.uid} {access.name}",
                 f"Backing heap: id#{owner} {entity.name}",
                 f"Known states: {' | '.join(sorted(statuses))}",
                 "Required state: Alive"),
                "Heap lifetime proof failed at this access.")
            raise error

    def require_live(self, sym, state, token):
        if sym.link_access is not None:
            self.require_live(sym.link_access[0], state, token)
        if sym.heap_owner and sym.uid in self.alias_members:
            # The physical target may be on the stack, but reaching it still
            # reads the interface descriptor stored in this heap allocation.
            self.require_target_live(sym, state, token)
        for target in self.physical_places(sym, state, allow_null=True):
            self.require_target_live(target, state, token, access=sym)

    def check_heap_exit(self, paths, depth, token):
        alive = frozenset(("Alive",))
        for path in paths:
            leaving = [uid for uid, statuses in path.heap.items()
                       if "Alive" in statuses and self.symbols[uid - 1].depth > depth]
            if not leaving:
                continue
            # Only a live link that survives this exit can retain an allocation.
            # Follow its descendants through the leaving objects, including
            # chains that contain several allocations from one iteration.
            pending, descendants = [], {}
            for uid, targets in path.borrows.items():
                link = self.symbols[uid - 1]
                if not isinstance(link.ty, RecordLink) or len(targets) != 1:
                    continue
                target, = targets
                if path.heap.get(target.uid) != alive or not self.is_whole_heap_object(target):
                    continue
                if link.heap_owner:
                    if path.heap.get(link.heap_owner) != alive:
                        continue
                    if link.depth > depth:
                        descendants.setdefault(link.heap_owner, []).append(target.uid)
                        continue
                elif link.depth > depth and self.attachment_parameter(link) is None:
                    continue
                pending.append(target.uid)
            retained = set()
            while pending:
                uid = pending.pop()
                if uid not in retained:
                    retained.add(uid)
                    pending.extend(descendants.get(uid, ()))
            for uid in leaving:
                if uid in self.parameter_uids or uid not in retained:
                    sym = self.symbols[uid - 1]
                    token.error(f"heap leak: {sym.name!r} leaves scope before free")

    def define_heap_array(self, name, ty, node, paths, readonly=False):
        if isinstance(self.composite_call_type(node), HeapArray):
            if ty.count is not None:
                node.token.error("bind a returned heap array using an owned slice type or inferred type")
            paths, actual, code = self.initializer_calls(node, paths)
            if not self.owned_result_fits(actual, ty):
                node.token.error("owned array result does not fit the declared element and length contract")
            if not paths:
                return [], [f"(void)({code});", "__builtin_unreachable();"]
            sym = self.declare(name, ty, readonly=readonly, ownership_origin="owned_result")
            result = []
            for path in paths:
                state = path.update(sym, actual.domain)
                state = state.with_array(sym, ArrayDomain.unknown(ty.domain.hi, ty.element.domain))
                result.append(replace(state, heap=state.heap | {sym.uid: frozenset(("Alive",))}))
            self.trace.append(f"  define entity id#{sym.uid} {sym.name}: owned result, Alive, len {actual.domain}")
            code = self.assume_slice_result(node.token, actual.domain, code)
            return result, [f"lan_slice {sym.c} = {code};"]
        if ty.count is None:
            name.error("an owned slice declaration requires an owning array result")
        previous = self.propagating_expression_nodes
        self.propagating_expression_nodes = self.propagation_nodes(ty.count)
        try:
            evaluated = [(path, self.expr(ty.count, path)) for path in paths]
        finally:
            self.propagating_expression_nodes = previous
        continuing = [(path, value) for path, value in evaluated if not value.always_returns]
        if not continuing:
            code = max((value.code for _, value in evaluated), key=self.propagation_code_rank)
            return [], [f"(void)({code});", "__builtin_unreachable();"]
        paths, lengths = map(list, zip(*continuing))
        for index, length in enumerate(lengths):
            integer = not (length.floating or length.boolean or length.enum or length.character)
            if integer and length.domain.within(ty.domain):
                continue
            valid = length.domain & ty.domain if integer and self.debug_bounds else Domain.of()
            if not valid:
                ty.count.token.error(f"cannot prove heap length {length.domain} fits {ty.domain}")
            self.trace.append(f"  debug check heap length {length.domain} fits {ty.domain}")
            lengths[index] = replace(length, domain=valid)
        if node is not None:
            node.token.error("heap array declaration uses zero initialization")
        zero = Value(zero_domain(ty.element), "0", boolean=ty.element.boolean, enum=ty.element.enum,
                     character=ty.element.character)
        self.range_check(name, zero, ty.element)
        sym = self.declare(name, ty, readonly=readonly, ownership_origin="allocate")
        out = []
        for p, length in zip(paths, lengths):
            state = p.update(sym, length.domain, length.affine, length.difference, length.linear)
            state = state.with_array(sym, ArrayDomain.zero(ty.domain.hi, ty.element.domain))
            out.append(replace(state, heap=state.heap | {sym.uid: frozenset(("Alive",))}))
            self.trace.append(f"  heap id#{sym.uid} {name.text}: Alive, len {length.domain}")
        alloc = self.allocating_helper("heap_array")
        count = lengths[0].code
        if self.debug_bounds:
            membership = self.membership(ty.count.token, ty.domain)
            check = self.helper("debug_heap_length", None, membership)
            count = f"{check}({count})"
        code = f"lan_slice {sym.c} = {alloc}((size_t)({count}), sizeof({ty.element.c}));"
        return out, [code]

    @staticmethod
    def is_free_call(node):
        return (node.kind == "call" and node.args[0].kind == "name"
                and node.args[0].args[0] == "free")

    def dangling_array_links(self, state, target, roots):
        from composite_array import RecordValue, TreeArray

        def visit(value, name):
            if isinstance(value, TreeArray):
                for part in value.segments:
                    label = (f"{name}[{part.start}]" if part.stop == part.start + 1
                             else f"{name}[{part.start},{part.stop})")
                    yield from visit(part.domain, label)
            elif isinstance(value, RecordValue):
                for field, content in value.fields:
                    yield from visit(content, f"{name}.{field}")
            elif isinstance(value, frozenset) and any(
                    owner is not None and (owner == target or owner.heap_owner == target.uid)
                    for owner in value):
                yield name

        def fields(sym, name):
            if (sym.heap_owner or sym.uid) == target.uid:
                return
            if state.heap.get(sym.heap_owner or sym.uid) == frozenset(("Tombstone",)):
                return
            content = state.arrays.get(sym.uid)
            if isinstance(content, TreeArray) and self.contains_record_links(sym.ty):
                for label in visit(content, name):
                    yield sym.uid, label
            for field in sym.fields:
                yield from fields(field, f"{name}.{field.name}")

        for symbol, name in roots:
            yield from fields(symbol, name)

    def free_temporary(self, node, paths):
        """Release an owning call result that never became a binding."""
        actual = self.composite_call_type(node)
        if not isinstance(actual, (HeapRecord, HeapObject, HeapArray)):
            node.token.error("free requires an owning heap entity")
        paths, _, code = self.initializer_calls(node, paths)
        if not paths:
            return self.early_composite_return(code)
        self.memory_effects.add(self.current)
        self.trace.append(f"  free unnamed result [{node.token.line}:{node.token.col}]: Tombstone")
        if isinstance(actual, HeapArray):
            code = f"({code}).data"
        elif linked_owned_record(actual) and self.owned_chain_call_lengths(node) != (1,):
            return paths, self.free_walk_code(f"temp_{node.token.line}_{node.token.col}", code,
                                             ((actual.record, "next", actual.record),),
                                             None, None, (), node.token)
        return paths, [f"free((void *)({code}));"]

    @staticmethod
    def heap_record(symbol):
        ty = symbol.ty
        return ty.record if isinstance(ty, HeapRecord) else ty

    def link_fields(self, symbol):
        """The link fields one node owns, taken from that node's own type."""
        record = self.heap_record(symbol)
        if not isinstance(record, Record):
            return ()
        return tuple(name for name, field in record.fields
                     if isinstance(unqualified(field), RecordLink))

    def free_closure(self, root, state, token):
        """Every live node the released structure owns, with its derived shape.

        The meta-graph already knows the shape, so `free` can prove that a
        traversal releases each node exactly once instead of leaking the tail.
        A summary node stands for the nodes a loop allocated, which is why the
        same traversal also covers the repeated section.  Each node contributes
        the links of its own record type, so one structure may mix types: a
        chain can alternate between two records and a tree can hold children of
        another record without either leaking.
        """
        graph = LinkGraph()
        nodes, pending, seen = {}, [root], set()
        stops = set()
        edges = {}
        link_names = set()
        # The meta-graph folds two links to the same node into one edge, so it
        # cannot tell a chain from a node reached twice.  Count every field
        # write instead: a node reached through two links would be freed twice.
        parents = {}
        while pending:
            node = pending.pop()
            location = self.record_location(node)
            if location in seen:
                continue
            seen.add(location)
            nodes[location] = node
            graph.add(location)
            owned = []
            for name in self.link_fields(node):
                link_names.add(name)
                field = self.child_place(node, (name,))
                targets = self.link_targets(field, state)
                if targets is None:
                    token.error(f"cannot prove the {name} targets of the released structure")
                for target in targets:
                    if not target.uid:
                        continue
                    if self.is_whole_heap_object(target):
                        owned.append(target)
                        fresh = (field.uid, target.uid) in state.fresh_links
                        # A loop-carried edge reaches the next instance of the
                        # same allocation site, not a second parent of this one.
                        if not fresh and self.record_location(target) != location:
                            parents[self.record_location(target)] = parents.get(
                                self.record_location(target), 0) + 1
                        edges.setdefault(name, {}).setdefault(location, []).append(
                            (self.record_location(target), fresh))
                        graph.link(location, (self.record_location(target),),
                                   fresh=(field.uid, target.uid) in state.fresh_links)
                    else:
                        if (target.heap_owner or target.projection is not None
                                or target.link_access is not None
                                or target.uid in self.alias_members or target.depth > len(self.scope)):
                            token.error("cannot prove the address of the non-owning structure boundary")
                        stops.add(f"&({target.c})")
            pending.extend(target for target in sorted(owned, key=lambda target: target.uid)
                           if self.record_location(target) not in seen)
        self.spend(token)
        shared = any(count > 1 for count in parents.values())
        return (list(nodes.values()), graph.summary(self.record_location(root)), stops,
                link_names, shared, edges, nodes)

    def free_walk(self, root, nodes, edges):
        """The one order in which a single pass releases every node, if any.

        A doubly linked list looks like a cycle, but only one field owns its
        nodes: the others point at neighbours the pass already reaches (the
        predecessor, and for a repeated section the previous instance).  Once
        that is proven, one walk releases every node exactly once and the extra
        fields need no traversal of their own.

        One field name usually carries the whole walk, which is also what tells
        two links that both point back into it apart.  When the record types of
        a structure each name their successor differently, the forward link is
        chosen per node instead.
        """
        locations = set(nodes)
        for name in sorted(edges):
            found = self.free_named_walk(root, locations, name, edges)
            if found is not None:
                return found
        return self.free_mixed_walk(root, locations, edges)

    def free_named_walk(self, root, locations, name, edges):
        """The walk one field carries, or None when that field is not a spine."""
        order, steps, current, seen = [], [], self.record_location(root), set()
        wrap = None
        while current in locations:
            if current in seen:
                # The walk comes back to a node it already released without a
                # loop turn that explains it.
                return None
            seen.add(current)
            order.append(current)
            found = edges.get(name, {}).get(current, ())
            forward = [(target, fresh) for target, fresh in found if target != current]
            own = [fresh for target, fresh in found if target == current]
            if any(own):
                # A loop allocates a new node on this edge each round, so the
                # repeated section ends here and starts over on the next turn.
                steps.append((name, current, True))
                wrap = current
                break
            if own:
                # The node links to itself within one round: the walk would
                # never reach the end.
                return None
            if not forward:
                break
            if len(forward) > 1:
                return None
            steps.append((name, forward[0][0], forward[0][1]))
            current = forward[0][0]
        return self.free_walk_result(root, order, steps, wrap, edges, locations)

    def free_mixed_walk(self, root, locations, edges):
        """The walk whose every node picks its own forward link."""
        forward = {}
        for name, by_source in edges.items():
            for source, targets in by_source.items():
                for target, fresh in targets:
                    forward.setdefault(source, []).append((name, target, fresh))
        order, steps, current, seen = [], [], self.record_location(root), {}
        wrap = None
        while current in locations:
            if current in seen:
                return None
            seen[current] = len(order)
            order.append(current)
            here = forward.get(current, ())
            ahead = [(name, target, fresh) for name, target, fresh in here
                     if target not in seen]
            if len(ahead) > 1:
                # A branch needs the post-order walk, not one pass.
                return None
            if ahead:
                name, target, fresh = ahead[0]
                steps.append((name, target, fresh))
                current = target
                continue
            closing = [(name, target) for name, target, fresh in here
                       if fresh and target in seen]
            if len(closing) > 1:
                return None
            if closing:
                steps.append((closing[0][0], closing[0][1], True))
                wrap = closing[0][1]
            break
        return self.free_walk_result(root, order, steps, wrap, edges, locations)

    def free_walk_result(self, root, order, steps, wrap, edges, locations):
        """Check that every other link only reaches a node the walk releases."""
        if not steps or set(order) != locations:
            return None
        predecessor = {order[index]: order[index - 1] for index in range(1, len(order))}
        if wrap is not None:
            predecessor[order[0]] = order[-1]
        used = {(order[index], name) for index, (name, _, _) in enumerate(steps)}
        forward_fields = {name for name, _, _ in steps}
        # A node the walk reaches through a loop turn stands for every instance
        # of that turn, so its "self" link is the instance before it.
        repeated = {source for by_source in edges.values()
                    for source, targets in by_source.items()
                    for target, fresh in targets if target == source and fresh}
        for name, by_source in edges.items():
            for source, targets in by_source.items():
                for target, fresh in targets:
                    if (source, name) in used:
                        continue
                    if name in forward_fields:
                        # The walk already follows this field somewhere, so a
                        # second edge of the same kind would make the source
                        # look like the end of the structure while it is not.
                        return None
                    if target == predecessor.get(source):
                        continue
                    if target == source and source in repeated:
                        continue
                    return None
        exact = LinkGraph()
        for index, location in enumerate(order):
            exact.add(location)
            if index < len(steps):
                name, target, fresh = steps[index]
                exact.link(location, (target,), fresh=fresh)
        summary = exact.summary(self.record_location(root))
        if summary.shape.cyclic:
            # The walk returns to a node it already released, so one pass over
            # it would never finish.  A loop turn that allocates a new node is
            # a repeat, not a cycle, and stays bounded.
            return None
        return order, steps, wrap, summary

    def free_heap(self, node, paths):
        _, args = node.args
        if len(args) != 1:
            node.token.error("free requires one owning heap entity")
        sym = self.place(args[0])
        if sym is None and args[0].kind in ("call", "propagate"):
            return self.free_temporary(args[0], paths)
        if sym is None or sym.bit_overlay:
            args[0].token.error("free requires an owning heap entity")
        self.memory_effects.add(self.current)
        def field_names(symbol, name):
            yield symbol.uid, name
            if isinstance(symbol.ty, (Record, Interface)):
                for field in symbol.fields:
                    yield from field_names(field, f"{name}.{field.name}")

        innermost = {name: symbol for scope in self.scope for name, symbol in scope.items()}
        roots = [(symbol, name if innermost[name].uid == symbol.uid else f"{name} (outer id#{symbol.uid})")
                 for scope in self.scope for name, symbol in scope.items()]
        names = dict(entry for symbol, name in roots for entry in field_names(symbol, name))
        location = node.token.line, node.token.col
        out = []
        chained = None
        tree = None
        chain_stops = set()

        def release(state, target, repeated):
            if state.heap.get(target.uid) != frozenset(("Alive",)):
                node.token.error(f"heap entity {target.name!r} may already be freed or moved")
            self.consume_parameter(target)
            self.note_transition_write(target, lifetime=True)
            notes = self.release_notes.setdefault(location, set())
            target_name = names.get(target.uid, target.name)
            # Two kinds of vertex: one the compiler knows individually, and one
            # standing for the nodes a loop allocates (an id set that differs
            # only by identity).  Both end up Tombstone; the label says which
            # proof discharged them.
            label = "id set, one node per round" if repeated else "special vertex"
            self.trace.append(f"  mark tombstone id#{target.uid} {target.name}: Tombstone ({label})")
            aliases = dict(state.borrows)
            for scope in self.scope:
                for visible in scope.values():
                    if isinstance(visible.ty, Slice) and visible.ty.projection is not None:
                        aliases[visible.uid] = visible.ty.projection.owners
                    elif isinstance(visible.ty, BitView):
                        host = visible.ty.host
                        aliases[visible.uid] = state.borrows.get(host.uid, frozenset((host,)))
            for uid, owners in aliases.items():
                if uid in names and (target in owners or any(owner.heap_owner == target.uid for owner in owners)):
                    alias = self.symbols[uid - 1]
                    backing = alias.heap_owner or alias.uid
                    if backing == target.uid or state.heap.get(backing) == frozenset(("Tombstone",)):
                        continue
                    label = names[uid]
                    self.trace.append(f"  view id#{uid} may dangle after free [{guard}]: {label}")
                    notes.add(f"{label} may dangle when {target_name} becomes Tombstone.")
            for uid, label in self.dangling_array_links(state, target, roots):
                self.trace.append(f"  link id#{uid} {label} may dangle after free [{guard}]")
                notes.add(f"{label} may dangle when {target_name} becomes Tombstone.")
            state = replace(state, heap=state.heap | {target.uid: frozenset(("Tombstone",))})
            if sym.uid in self.alias_types:
                state = replace(state, borrows=state.borrows | {sym.uid: frozenset((target,))})
            return state

        for path in paths:
            targets = self.physical_places(sym, path, allow_null=True)
            repeated_allocations = self.repeated_allocation_uids(path, node.token)
            guard = path.guard_label()
            for target in targets:
                if not target.uid or not (isinstance(target.ty, HeapArray) or target.heap_owner == target.uid):
                    args[0].token.error("free requires an owning heap entity proved non-null")
                if path.heap.get(target.uid) != frozenset(("Alive",)):
                    node.token.error(f"heap entity {target.name!r} may already be freed or moved")
                if target.uid in repeated_allocations:
                    node.token.error("cannot free an individual instance inside a symbolic repeated allocation group")
                owned = [target]
                repeated = frozenset()
                record = target.ty.record if isinstance(target.ty, HeapRecord) else target.ty
                if isinstance(record, Record) and Compiler.contains_record_links(record):
                    (owned, summary, stops, links, shared,
                     edges, nodes) = self.free_closure(target, path, node.token)
                    shape = summary.shape
                    # A lone node needs no walk at all; anything larger, and
                    # any structure that stops at storage it does not own, does.
                    walking = len(owned) > 1 or stops or shape.kind == REPEAT
                    walk = (self.free_walk(target, nodes, edges)
                            if walking else None)
                    if walk is not None:
                        order, steps, wrap, exact = walk
                        plan = [(self.heap_record(nodes[order[index]]), name,
                                 self.heap_record(nodes[target_location]))
                                for index, (name, target_location, _) in enumerate(steps)]
                        fields = []
                        for _, name, _ in plan:
                            if name not in fields:
                                fields.append(name)
                        self.trace.append(
                            f"  free spine over {', '.join(f'f_{name}' for name in fields)}: "
                            f"{len(order)} nodes, every other link only points back at a neighbour")
                        owned, chained = [nodes[location] for location in order], (plan, wrap, exact.shape)
                        chain_stops.update(stops)
                        if exact.shape.kind == REPEAT:
                            repeated = frozenset(exact.cycle)
                        state = path
                        for item in owned:
                            state = release(state, item,
                                            self.record_location(item) in repeated)
                        out.append(state)
                        continue
                    if shape.kind not in (CHAIN, REPEAT, TREE, DAG):
                        node.token.error(
                            f"free cannot release a {shape.describe()}: break it into a chain first")
                    if shape.kind == DAG:
                        node.token.error(
                            f"free cannot release a {shape.describe()}: "
                            "a node with two parents would be freed twice")
                    if shared:
                        node.token.error(
                            "free cannot release a structure whose node is reached by two links: "
                            "it would be freed twice")
                    root_links = self.link_fields(target)
                    if walking and (shape.kind == TREE or root_links != ("next",)):
                        # Post-order recursion is the only walk that can reach
                        # every branch of a tree, and it also covers a chain
                        # carried by a link other than `next`.
                        if stops:
                            node.token.error("free cannot stop a tree at a non-owning boundary")
                        tree = record, root_links
                    elif walking:
                        chained = (((record, "next", record),), None, shape)
                        chain_stops.update(stops)
                    if shape.kind == REPEAT:
                        repeated = frozenset(summary.cycle)
                state = path
                for item in owned:
                    state = release(state, item, self.record_location(item) in repeated)
                out.append(state)
        if isinstance(sym.ty, HeapArray):
            pointer = f"{sym.c}.data"
        elif isinstance(sym.ty, Array) and sym.uid in self.alias_types:
            pointer = sym.c
        else:
            pointer = f"&{sym.c}"
        entries = sorted(self.release_notes.get(location, ()))
        comments = ["/**", *(f" * {entry}" for entry in entries), " */"] if entries else []
        if chained is not None:
            plan, wrap, exact = chained
            return out, [*comments, *self.free_walk_code(sym.uid, pointer, plan, wrap,
                                                         exact, chain_stops, node.token)]
        if tree is not None:
            record, links = tree
            helper = self.free_tree_helper(record, node.token)
            fields = ", ".join(f"f_{name}" for name in links)
            self.trace.append(f"  free tree from id#{sym.uid}: post-order walk over {fields}")
            return out, [*comments, f"{helper}({pointer});"]
        return out, [*comments, f"free((void *){pointer});"]

    def free_walk_code(self, uid, pointer, plan, wrap, shape, stops, token):
        """C for one release pass over a spine of nodes.

        A walk whose steps all share one record type and one field is the
        ordinary cursor loop.  Anything else unrolls the type cycle of the
        repeated section: the C type of the cursor changes from step to step,
        so no single typed cursor can follow the whole walk, and a static walk
        that never repeats needs no loop at all.
        """
        cursor = f"lan_free_{uid}"
        record, field, _ = plan[0]
        if all(item[0].c == record.c and item[1] == field and item[2].c == record.c
               for item in plan):
            condition = cursor + "".join(
                f" && (const void *){cursor} != (const void *){stop}" for stop in sorted(stops))
            self.trace.append(f"  free chain from id#{uid}: one pass over the link meta-graph")
            return [f"{{ {record.c} *{cursor} = {pointer}; while ({condition}) {{ "
                    f"{record.c} *lan_free_next_{uid} = ({record.c} *){cursor}->{c_field(field)}; "
                    f"free((void *){cursor}); {cursor} = lan_free_next_{uid}; }} }}"]
        if stops:
            # A boundary the walk must not release can sit anywhere in the
            # cycle, and the unrolled body has no single cursor to compare.
            token.error("free cannot stop a chain that changes record type")
        if wrap is None:
            self.trace.append(f"  free chain from id#{uid}: {len(plan)} typed steps, no repeat")
            return self.free_straight_code(uid, pointer, plan, cursor)
        prefix, period = shape.prefix, shape.period
        if not period or prefix + period != len(plan):
            token.error("cannot prove the repeated section of the released structure")
        self.trace.append(
            f"  free chain from id#{uid}: {prefix} fixed nodes then a type cycle of {period}")
        lines = ["{", f"    {record.c} *{cursor} = {pointer};"]
        if prefix:
            lines.append(f"    if (!{cursor}) goto {cursor}_done;")
            for index in range(prefix):
                _, name, target = plan[index]
                current = cursor if index == 0 else f"{cursor}_fixed{index - 1}"
                step = f"{cursor}_fixed{index}"
                lines.append(f"    {target.c} *{step} = {current}->{c_field(name)};")
                lines.append(f"    free((void *){current});")
                lines.append(f"    if (!{step}) goto {cursor}_done;")
            seed = f"{cursor}_fixed{prefix - 1}"
        else:
            seed = cursor
        loop = f"{cursor}_turn"
        lines.append(f"    {plan[prefix][0].c} *{loop} = {seed};")
        lines.append(f"    while ({loop}) {{")
        for index in range(prefix, prefix + period):
            _, name, target = plan[index]
            current = loop if index == prefix else f"{cursor}_round{index - 1}"
            step = f"{cursor}_round{index}"
            lines.append(f"        {target.c} *{step} = {current}->{c_field(name)};")
            lines.append(f"        free((void *){current});")
            lines.append(f"        if (!{step}) break;")
        if prefix + period - 1 != prefix:
            lines.append(f"        {loop} = {cursor}_round{prefix + period - 1};")
        lines.append("    }")
        lines.append(f"    {cursor}_done:;")
        lines.append("}")
        return lines

    @staticmethod
    def free_straight_code(uid, pointer, plan, cursor):
        """C for a walk over statically known nodes that never repeats."""
        lines = ["{", f"    {plan[0][0].c} *{cursor} = {pointer};",
                 f"    if (!{cursor}) goto {cursor}_done;"]
        for index, (_, name, target) in enumerate(plan):
            current = cursor if index == 0 else f"{cursor}_step{index - 1}"
            step = f"{cursor}_step{index}"
            lines.append(f"    {target.c} *{step} = {current}->{c_field(name)};")
            lines.append(f"    free((void *){current});")
            lines.append(f"    if (!{step}) goto {cursor}_done;")
        last = f"{cursor}_step{len(plan) - 1}" if plan else cursor
        lines.append(f"    free((void *){last});")
        lines.append(f"    {cursor}_done:;")
        lines.append("}")
        return lines

    def free_tree_helper(self, record, token):
        """Register the post-order release helper of every type a tree reaches.

        One helper per record type, each calling the helper of the type its own
        field declares, so a tree may mix record types without a visited set.
        """
        plans = getattr(self, "free_tree_plans", None)
        if plans is None:
            plans = self.free_tree_plans = {}
        pending = [record]
        while pending:
            current = pending.pop()
            if current.identity in plans:
                continue
            entries = []
            for name, field in current.fields:
                field = unqualified(field)
                if isinstance(field, RecordLink):
                    entries.append((name, self.link_record(field, token)))
            plans[current.identity] = (current, tuple(entries))
            pending.extend(child for _, child in entries)
        names = {}
        for identity, (parent, entries) in list(plans.items()):
            names[identity] = self.helper("free_tree", parent, entries)
        return names[record.identity]

    def expression_consumes_owner(self, node):
        """True when evaluating this expression hands a heap owner to a callee."""
        if isinstance(node, (tuple, list)):
            return any(self.expression_consumes_owner(child) for child in node)
        if not isinstance(node, Node):
            return False
        if node.kind in ("call", "propagate"):
            contract = self.composite_call_signature(node)
            if contract is not None and any(isinstance(ty, (HeapRecord, HeapObject, HeapArray))
                                            for ty in contract.params):
                return True
        return any(self.expression_consumes_owner(child) for child in node.args)

    def consume_parameter(self, symbol, method="free"):
        index = self.parameter_uids.get(symbol.uid)
        if index is None or index in self.consumed_parameters:
            return
        self.consumed_parameters[index] = method
        self.trace.append(f"  consume param#{index} {symbol.name}: "
                          f"{'freed in body' if method == 'free' else 'transferred by move'}")

    def dangling_release_requests(self, *, include_resolved=False):
        return [OracleRequest(self.source_path, self.source_digest, line, column,
                              tuple(sorted(entries)), DANGLING_RELEASE_HEADLINE)
                for (line, column), entries in sorted(self.release_notes.items())
                if entries or include_resolved]
