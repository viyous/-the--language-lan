"""Record definition, copying and comparison."""

from __future__ import annotations

import copy
from dataclasses import field, fields, replace
from itertools import count

from array_domain import ArrayDomain
from value_domain import Domain, DomainComplexity
from table_interface import c_field

from .syntax import Node
from .typesys import (AffineForm, AliasResult, Array, BOOL, BorrowRecord, Contract, CopyArray, CopyRecord,
                      ExternCallable, HeapRecord, Interface, Record, RecordLink, Scalar, Slice, storage_size,
                      unqualified, Value, ValueOrigin, ZERO)
from .domain_utils import zero_domain
from .c_emit import c_predicate_sequence

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class RecordMixin:
    @staticmethod
    def record_field_types(ty):
        for name, field in ty.fields:
            field = unqualified(field)
            if isinstance(field, Record):
                yield from Compiler.record_field_types(field)
            else:
                yield name, field

    def define_record(self, name, ty, node, paths, heap=False, readonly=False, static=False):
        interface_record = self.contains_interface_storage(ty)
        if interface_record and (static or node is None or node.kind not in ("record_initializer", "copy")):
            name.error("a record with interface fields requires a local field initializer")
        if heap and interface_record and (node.kind != "record_initializer" or self.contains_record_links(ty)):
            name.error("a heap record with interface fields requires a link-free type and a field initializer")
        if isinstance(self.composite_call_type(node), (Record, HeapRecord)):
            return self.define_record_call(name, ty, node, paths, heap, readonly)
        if node is None and any(isinstance(field, Slice) for _, field in self.record_field_types(ty)):
            node = Node(name, "record_initializer", ((),))
        if node is not None and node.kind == "record_initializer":
            return self.define_initialized_record(name, ty, node, paths, heap, readonly, static)
        source = None
        if node is not None:
            if node.kind != "copy":
                node.token.error("record initialization requires an explicit as: copy")
            source = self.record_place(node.args[0])
            if source is None or source.ty != ty:
                node.token.error("record copy requires the same record type")
            for state in paths:
                self.require_live(source, state, node.token)
        sym = self.declare(name, ty, heap=heap, readonly=readonly, static=static, allocated_record=heap,
                           ownership_origin="allocate" if heap else None)
        self.trace.append(f"  define entity id#{sym.uid} {sym.name}: {ty.c}, "
                          f"layout {', '.join(field for field, _ in ty.layout)}")
        self.record_symbols(sym, name, zero=source is None)
        if interface_record:
            return self.copy_interface_record(sym, source, paths, node.token, declare=True)
        if source is not None:
            paths = self.copy_record_states(sym, source, paths)
        for child in self.storage_leaves(sym):
            if source is not None:
                continue
            if isinstance(child.ty, Scalar):
                paths = [p.update(child, zero_domain(child.ty)) for p in paths]
            elif isinstance(child.ty, RecordLink):
                paths = [self.store_link(p, child, frozenset((self.null_symbol,)), name) for p in paths]
            elif isinstance(child.ty, Slice):
                paths = [p.update(child, ZERO) for p in paths]
            elif child.ty.packed_bits and child.ty.length <= 4096:
                paths = [p.with_bits(child, ((1 << child.ty.length) - 1, 0)) for p in paths]
            elif not child.ty.packed_bits:
                paths = [p.with_array(child, ArrayDomain.zero(child.ty.length, child.ty.element.domain)) for p in paths]
        if heap:
            paths = [replace(p.update(sym, ZERO), heap=p.heap | {sym.uid: frozenset(("Alive",))})
                     for p in paths]
            pointer = sym.c_var
            alloc = self.allocating_helper("heap_array")
            self.trace.append(f"  heap id#{sym.uid} {sym.name}: Alive")
            code = [f"{ty.c} *{pointer} = ({ty.c} *){alloc}(1, sizeof({ty.c})).data;"]
            if source is not None:
                code.append(f"{sym.c} = {source.c};")
            return paths, code
        initializer = source.c if source is not None else "{0}"
        prefix = "static const " if static else ""
        return paths, [f"{prefix}{ty.c} {sym.c} = {initializer};"]

    def copy_record_states(self, dst, src, paths):
        result = []
        for original in paths:
            state = original
            for left in self.storage_leaves(dst):
                state = self.write_storage(state, left)
            for left, right in zip(self.storage_leaves(dst), self.storage_leaves(src)):
                if isinstance(left.ty, Scalar):
                    state = self.store_value(state, left, self.storage_value(right, original))
                elif isinstance(left.ty, RecordLink):
                    state = self.store_link(state, left, self.link_targets(right, original), self.nodes[0].token)
                elif isinstance(left.ty, Slice):
                    state = state.update(left, self.record_text_domain(right, original))
                elif left.ty.packed_bits:
                    state = self.store_bits(state, left, self.storage_bits(original, right))
                else:
                    state = self.store_array(state, left, self.storage_array(original, right))
            result.append(state)
        return result

    def copy_interface_record(self, dst, src, paths, token, *, declare=False):
        if dst.heap_owner:
            token.error("an interface record copy requires a stack destination")
        if (dst.uid in self.alias_members or dst.projection is not None or dst.link_access is not None
                or src.projection is not None or src.link_access is not None):
            token.error("a record with interface fields can only be copied into a direct local binding")
        destinations = [field for field in dst.fields if isinstance(field.ty, Interface)]
        if not declare and (dst.readonly or any(leaf.readonly for leaf in self.storage_leaves(dst))
                            or any(field.readonly for field in destinations)):
            token.error("cannot modify an immutable record or field")
        for state in paths:
            self.require_live(src, state, token)
            if not declare:
                self.require_live(dst, state, token)
        prefix = f"{dst.ty.c} " if declare else ""
        code = [f"{prefix}{dst.c} = {src.c};"]
        result = self.copy_record_states(dst, src, paths)
        for field in destinations:
            source = self.child_place(src, (field.name,))
            if source.readonly and not field.readonly:
                token.error("cannot copy an immutable interface into a writable field")
            for state in paths:
                self.require_live(source, state, token)
            if declare:
                code.append(f"size_t {field.interface_offsets}[{len(field.ty.fields)}];")
            else:
                self.note_transition_write(field, alias_binding=True)
            result = self.bind_interface("define entity" if declare else "update entity",
                                         field, token, [(source, "")] * len(result), result)
            code.append(f"{field.c} = {self.interface_storage_code(field, source.c)};")
        return result, code

    def assign_record_copy(self, target, op, node, paths):
        dst = self.record_place(target)
        src = self.record_place(node.args[0])
        if op.text != "=" or src is None or dst.ty != src.ty:
            op.error("record copy requires '=' and the same record type")
        if self.contains_interface_storage(dst.ty):
            return self.copy_interface_record(dst, src, paths, op)
        if dst.readonly or any(leaf.readonly for leaf in self.storage_leaves(dst)):
            op.error("cannot modify an immutable record or field")
        for state in paths:
            self.require_live(dst, state, op)
            self.require_live(src, state, node.token)
        self.trace.append(f"  copy record id#{src.uid} into id#{dst.uid}")
        if all(self.independent_bit_storage(dst, src, path) for path in paths):
            copy = self.helper("disjoint_object_copy", None)
            code = f"{copy}(&({dst.c}), &({src.c}), sizeof({dst.c}));"
        else:
            code = f"{dst.c} = {src.c};"
        return self.copy_record_states(dst, src, paths), [code]

    def comparison_place(self, sym, state, *, allow_effectful_links=False, require_proven=False):
        return sym

    def value_equality_statements(self, ty, left, right, labels=None, rank=0):
        ty = unqualified(ty)
        if labels is None:
            labels = count()
        if isinstance(ty, Scalar):
            return [f"if (!(({left}) == ({right}))) goto lan_unequal;"]
        if isinstance(ty, RecordLink):
            if rank <= 1:
                return [f"if (({left}) != ({right})) goto lan_unequal;"]
            record = self.link_record(ty, self.nodes[0].token)
            helper = self.helpers["record_rank_equal", record, rank - 1]
            return [f"if (!{helper}({left}, {right})) goto lan_unequal;"]
        if isinstance(ty, Record):
            return [line for name, field in ty.fields for line in
                    self.value_equality_statements(field, f"({left}).{c_field(name)}", f"({right}).{c_field(name)}", labels, rank)]
        label = next(labels)
        end = f"lan_compare_end_{label}"
        if isinstance(ty, Slice):
            return [f"if (({left}).len != ({right}).len) goto lan_unequal;",
                    f"if (!({left}).len) goto {end};",
                    f"if (memcmp(({left}).data, ({right}).data, ({left}).len)) goto lan_unequal;",
                    f"{end}: ;"]
        if ty.packed_bits:
            return [f"if (memcmp({left}, {right}, {ty.c_length}) != 0) goto lan_unequal;"]
        index, head = f"lan_compare_index_{label}", f"lan_compare_head_{label}"
        body = self.value_equality_statements(ty.element, f"({left})[{index}]", f"({right})[{index}]", labels, rank)
        return [f"size_t {index} = 0;", f"{head}: ;",
                f"if ({index} >= {ty.length}) goto {end};", *body,
                f"++{index};", f"goto {head};", f"{end}: ;"]

    def linked_field_types(self, ty):
        ty = unqualified(ty)
        if isinstance(ty, RecordLink):
            yield ty
        elif isinstance(ty, Array):
            yield from self.linked_field_types(ty.element)
        elif isinstance(ty, Record):
            for _, field in ty.fields:
                yield from self.linked_field_types(field)

    def ranked_helper(self, ty, rank, token):
        pending = [(ty, rank, False)]
        while pending:
            record, depth, ready = pending.pop()
            key = ("record_rank_equal", record, depth)
            if key in self.helpers:
                continue
            if ready or depth <= 1:
                self.spend(token)
                self.helper(*key)
                continue
            pending.append((record, depth, True))
            fields = self.linked_field_types(record)
            pending.extend((self.link_record(field, token), depth - 1, False)
                           for field in reversed(tuple(fields)))
            self.spend(token)
        return self.helpers["record_rank_equal", ty, rank]

    def check_ranked_links(self, targets, rank, state, token, seen):
        pending = [(target, rank) for target in targets]
        while pending:
            target, depth = pending.pop()
            identity = (target.projection or target.uid, depth)
            if not target.uid or identity in seen:
                continue
            seen.add(identity)
            self.spend(token)
            self.require_target_live(target, state, token)
            if depth and self.contains_record_links(target.ty):
                pending.extend((link, depth - 1) for link in self.record_link_targets(target, state))

    def compare_ranked_spans(self, node, state):
        op, rank, left, right = node.args
        if self.span_kind(left) != "ordinary" or self.span_kind(right) != "ordinary":
            return None
        a, b, exit_value = self.comparison_spans(left, right, state)
        if exit_value is not None:
            return exit_value
        if a.element != b.element:
            node.token.error("ranked slice comparison requires the same element type")
        if not self.contains_record_links(a.element):
            comparison = Node(node.token, "binary", (op, left, right))
            return self.compare_ordinary_spans(comparison, state, a, b)
        common_lengths = a.length.domain & b.length.domain
        stable = not any(self.has_effects(operand, allow_debug_checks=True) for operand in (left, right))
        if (common_lengths and stable
                and self.prove_order(node.token, a.length, "!=", b.length, state)):
            common_lengths = Domain()
        equal = BOOL
        if not common_lengths:
            equal = ZERO
        elif a.length.domain == b.length.domain == ZERO:
            equal = Domain.point(1)
        if common_lengths and common_lengths.hi > 0:
            seen = set()
            indices = Domain.of((0, common_lengths.hi - 1))
            comparable = (a, b)
            if stable:
                comparable = tuple(self.span_at_comparable_lengths(span, common_lengths, state)
                                   for span in comparable)
            for original, span in zip((a, b), comparable):
                end = self.span_end_domain(node.token, original, state, original.positions()) if stable else None
                targets = self.span_link_targets(span, state, indices, absolute_limit=end.hi if end else None)
                self.check_ranked_links(targets, rank - 1, state, node.token, seen)
            if stable:
                same_length = self.prove_order(node.token, a.length, "==", b.length, state)
                if same_length or common_lengths.lo > 0:
                    x, y = (replace(span, length=replace(span.length, domain=common_lengths))
                            for span in comparable)
                    contents = self.compare_span_contents(x, y, state, node.token, rank=rank)
                    if same_length or contents == ZERO:
                        equal = contents
        element_helper = self.ranked_helper(a.element, rank, node.token)
        assume = self.helper("contract_assume", None)
        helper = self.helper("slice_rank_equal", a.element, (assume, element_helper))
        args = [a.c_value(), b.c_value()]
        code = (self.ordered_call(node.token, helper, ["lan_slice", "lan_slice"], args)
                if self.has_effects(node) else f"{helper}({', '.join(args)})")
        domain = equal.affine(-1, 1) if op == "!==" else equal
        return Value(domain, f"!({code})" if op == "!==" else code, 8, True)

    def span_at_comparable_lengths(self, span, lengths, state):
        positions = span.positions()
        start = self.span_start(span)
        if len(positions) != 1 or start is None:
            return span
        owner, offset = positions[0]
        if offset is None:
            return span
        if self.related(start, span.length):
            form = start.affine
            inputs = form.origin.domain & span.length.affine.preimage(lengths) & form.preimage(offset)
            narrowed = offset & inputs.affine(form.scale, form.offset)
        else:
            start_form = self.linear_value(start, prefer_symbolic=True)
            length_form = self.linear_value(span.length, prefer_symbolic=True)
            narrowed = offset
            if start_form is not None and length_form is not None and length_form[0]:
                start_terms, start_bias = start_form
                length_terms, length_bias = length_form
                factors, denominator = next(iter(length_terms))
                numerator = dict(start_terms).get(factors, 0)
                if numerator and ({(term, scale * denominator) for term, scale in start_terms}
                                  == {(term, scale * numerator) for term, scale in length_terms}):
                    # denominator * start = numerator * length + bias
                    bias = denominator * start_bias - numerator * length_bias
                    scaled = lengths.affine(numerator, bias)
                    narrowed &= AffineForm(ValueOrigin(offset), denominator, 0).preimage(scaled)
            if narrowed == offset:
                difference = self.dependent_difference_domain(start, span.length, state)
                if difference is not None:
                    narrowed &= self.sum_index_domains(lengths, difference)
        return replace(span, origins=((owner, narrowed),)) if narrowed else span

    def compare_ranked(self, node, state):
        op, rank, left, right = node.args
        spans = self.compare_ranked_spans(node, state)
        if spans is not None:
            return spans
        a, b = self.place(left), self.place(right)
        if a is None or b is None or not all(isinstance(sym.ty, (Record, RecordLink)) for sym in (a, b)):
            node.token.error("ranked comparison requires records of the same type")
        records = [self.link_record(sym.ty, node.token) if isinstance(sym.ty, RecordLink) else sym.ty
                   for sym in (a, b)]
        if records[0] != records[1]:
            node.token.error("ranked comparison requires records of the same type")
        record = records[0]
        if (self.contains_record_links(record)
                or any(isinstance(sym.ty, RecordLink) for sym in (a, b))
                or self.has_propagation(node)):
            args, checks, projections, operands = [], [], [], []
            for sym in (a, b):
                if sym.projection is not None and isinstance(sym.ty, Record):
                    projected = self.comparison_place(sym, state, allow_effectful_links=True,
                                                       propagate_exit=True)
                    if projected.projection.always_returns:
                        parts = [f"(void)({arg})" for arg in args] + [f"(void)({projected.c})", "0"]
                        code = f"({', '.join(parts)})"
                        return Value(BOOL, code, 8, True, effectful=True, always_returns=True)
                    projections.append(projected)
                    operands.append((projected,))
                    args.append(f"&({projected.c})")
                    continue
                if isinstance(sym.ty, RecordLink):
                    self.require_live(sym, state, node.token)
                    targets, address = self.link_targets(sym, state), self.comparison_place(sym, state).c
                else:
                    if sym.link_access is not None:
                        self.require_live(sym, state, node.token)
                    targets = self.physical_places(sym, state, allow_null=True)
                    place = self.comparison_place(sym, state)
                    address = (sym.c_var if sym.uid in self.alias_types and sym.link_access is None
                               else f"&({place.c})")
                checks.append((targets, rank))
                operands.append(tuple(targets))
                args.append(address)
            # The helper reads both records after evaluating both addresses.
            for projected in projections:
                if projected.projection.captured is not None:
                    self.require_live(projected, state, node.token)
                checks.append((self.record_link_targets(projected, state), rank - 1))
            seen = set()
            for targets, depth in checks:
                self.check_ranked_links(targets, depth, state, node.token, seen)
            helper = self.ranked_helper(record, rank, node.token)
            pointer = f"const {record.c} *"
            effectful = self.has_effects(node)
            code = (self.ordered_call(node.token, helper, [pointer, pointer], args)
                    if effectful else f"{helper}({', '.join(args)})")
            equal = BOOL
            if not self.has_effects(node, allow_debug_checks=True):
                equal = self.compare_ranked_targets(*operands, state, rank, node.token)
            domain = equal.affine(-1, 1) if op == "!==" else equal
            return Value(domain, f"!({code})" if op == "!==" else code, 8, True)
        # Ordinary record comparison also proves values from the current state.
        if not any(getattr(self.alias_types.get(sym.uid), "nullable", False) for sym in (a, b)):
            comparison = Node(node.token, "binary", (op, left, right))
            return self.compare_records(comparison, state)
        a, b = self.comparison_place(a, state), self.comparison_place(b, state)
        for symbol in (a, b):
            for target in self.physical_places(symbol, state):
                self.require_target_live(target, state, node.token)
        args = [symbol.c_var if symbol.uid in self.alias_types
                else f"&({symbol.c})" for symbol in (a, b)]
        helper = self.helper("record_value_equal", a.ty, "nullable")
        pointer = f"const {a.ty.c} *"
        code = (self.ordered_call(node.token, helper, [pointer, pointer], args)
                if self.has_effects(node) else f"{helper}({', '.join(args)})")
        return Value(BOOL, f"!({code})" if op == "!==" else code, 8, True)

    def projected_address_indices(self, a, b, *, captured=False):
        first, second = a.projection, b.projection
        if (first is None or second is None or first.root.uid != second.root.uid
                or len(first.steps) != len(second.steps) or a.ty != b.ty):
            return None
        root = first.root
        if captured:
            if len(first.captured or ()) != 1 or len(second.captured or ()) != 1:
                return None
            if isinstance(root.ty, Slice):
                span = root.ty.projection
                host = span.host if span is not None else None
                fixed_host = (host is not None and isinstance(host.ty, Array) and host.projection is None
                              and host.uid not in self.alias_members and host.link_access is None)
                same_epoch = first.descriptor_epoch is not None and first.descriptor_epoch is second.descriptor_epoch
                if not (fixed_host or same_epoch):
                    return None
            elif not isinstance(root.ty, Array):
                return None
        if (not first.resolved or not second.resolved
                or not captured and (first.captured is not None or second.captured is not None)
                or not isinstance(root.ty, (Array, Slice)) or root.uid in self.alias_members
                or root.link_access is not None or root.bit_overlay or not storage_size(a.ty)
                or len(first.index_values) != len(first.steps) or len(second.index_values) != len(second.steps)):
            return None
        pairs = []
        for left, right, x, y in zip(first.steps, second.steps, first.index_values, second.index_values):
            if isinstance(left, str) or isinstance(right, str):
                if left != right:
                    return None
                continue
            if x is None or y is None:
                return None
            pairs.append((x, y))
        return tuple(pairs) or None

    def shallow_projection_domain(self, node, a, b, state):
        _, left, right = node.args
        key = self.predicate_key(left, state)
        if key is not None and key == self.predicate_key(right, state):
            return Domain.point(1)
        indices = self.projected_address_indices(a, b)
        if indices is not None:
            equality = [self.offset_equality_domain(node.token, x, y, state) for x, y in indices]
            if ZERO in equality:
                return ZERO
            return Domain.point(1) if all(domain.lo == 1 for domain in equality) else BOOL
        if a.projection is None or b.projection is None:
            return BOOL
        first, second = a.projection, b.projection
        for projection in (first, second):
            root = projection.root
            if (not projection.resolved or projection.captured is not None or not isinstance(root.ty, Array)
                    or root.borrowed or root.uid in self.alias_members or root.link_access is not None
                    or root.bit_overlay):
                return BOOL
        if (first.root.uid == second.root.uid and first.steps == second.steps
                and all(isinstance(step, str) or step.lo == step.hi for step in first.steps)):
            return Domain.point(1)
        if storage_size(a.ty) and storage_size(b.ty):
            try:
                if not self.records_may_overlap(a, b):
                    return ZERO
            except DomainComplexity:
                pass
        return BOOL

    def compare_records(self, node, state):
        op, lhs, rhs = node.args
        a, b = self.record_place(lhs), self.record_place(rhs)
        if a is None and b is None:
            return None
        if a is None or b is None or a.ty != b.ty:
            node.token.error("record comparison requires two records of the same type")
        if self.contains_interface_storage(a.ty):
            node.token.error("record comparison does not support interface fields")
        if op not in ("==", "!=", "===", "!=="):
            node.token.error("records only support equality comparison")
        if op in ("===", "!==") and self.contains_record_links(a.ty):
            node.token.error("records containing aliases require an explicit comparison rank")
        for sym in (a, b):
            if sym.link_access is not None or sym.heap_owner and sym.uid in self.alias_members:
                self.require_live(sym, state, node.token)
        projected = a.projection is not None or b.projection is not None
        if projected:
            a = self.comparison_place(a, state, propagate_exit=True)
            if a.projection is not None and a.projection.always_returns:
                return Value(BOOL, a.c, 8, True, effectful=True, always_returns=True)
            b = self.comparison_place(b, state, propagate_exit=True)
            if b.projection is not None and b.projection.always_returns:
                code = f"((void)(&({a.c})), (void)({b.c}), 0)"
                return Value(BOOL, code, 8, True, effectful=True, always_returns=True)
        if op in ("==", "!=") and projected:
            code = f"(&({a.c}) {op} &({b.c}))"
            if self.has_effects(node):
                helper = self.helper("binary_eval", None, ("uint8_t", f"const {a.ty.c} *", f"const {b.ty.c} *", f"(a {op} b)"))
                code = f"{helper}(&({a.c}), &({b.c}))"
                if self.has_effects(node, allow_debug_checks=True):
                    return Value(BOOL, code, 8, True)
            equal = self.shallow_projection_domain(node, a, b, state)
            return Value(equal if op == "==" else equal.affine(-1, 1), code, 8, True)
        if op in ("==", "!="):
            left = self.physical_places(a, state, allow_null=True)
            right = self.physical_places(b, state, allow_null=True)
            equal = a.c == b.c or (len(left) == len(right) == 1 and left == right
                                   and not left[0].chain_summary and left[0].projection is None
                                   and left[0].uid not in self.repeated_allocation_uids(state, node.token))
            distinct = not any(x == y or x.borrowed and y.borrowed
                               or x.uid == y.uid and (x.projection is not None or y.projection is not None)
                               for x in left for y in right)
            domain = Domain.point(equal if op == "==" else not equal) if equal or distinct else BOOL
            return Value(domain, f"(&{a.c} {op} &{b.c})", 8, True)
        self.require_live(a, state, node.token)
        self.require_live(b, state, node.token)
        effects = self.has_effects(node)
        stable = not self.has_effects(node, allow_debug_checks=True)
        guarded_equal = stable and self.guarded_value_equality(lhs, rhs, state)
        if projected or any(not isinstance(field.ty, Scalar) for field in self.storage_leaves(a)):
            helper = self.helper("record_value_equal", a.ty)
            pointer = f"const {a.ty.c} *"
            args = [f"&({a.c})", f"&({b.c})"]
            code = (self.ordered_call(node.token, helper, [pointer, pointer], args)
                    if effects else f"{helper}({', '.join(args)})")
            domain = (Domain.point(1) if guarded_equal else
                      self.compare_record_contents(a, b, state) if stable else BOOL)
            return Value(domain.affine(-1, 1) if op == "!==" else domain,
                         f"!({code})" if op == "!==" else code, 8, True)
        tests = []
        for x, y in zip(self.storage_leaves(a), self.storage_leaves(b)):
            if not isinstance(x.ty, Scalar):
                node.token.error("record value equality currently requires scalar leaves")
            vx, vy = self.storage_value(x, state), self.storage_value(y, state)
            tests.append(self.binary(node.token, "==", vx, vy, state))
        code = c_predicate_sequence([test.code for test in tests], conjunction=True)
        domain = (Domain.point(1) if guarded_equal else
                  ZERO if any(test.domain == ZERO for test in tests) else
                  Domain.point(1) if all(test.domain == Domain.point(1) for test in tests) else BOOL)
        return Value(domain if op == "===" else domain.affine(-1, 1),
                     code if op == "===" else "!" + code, 8, True)

    def record_code(self):
        emitted = set()
        lines = []

        def add(ty):
            ty = unqualified(ty)
            if isinstance(ty, ExternCallable):
                if ty.c not in emitted:
                    add(ty.contract)
                    emitted.add(ty.c)
                    guard = ty.c.upper() + "_DEFINED"
                    lines.extend((f"#ifndef {guard}", f"#define {guard}",
                                  Compiler.c_fnptr_typedef(ty.c, ty.contract, foreign=True), "#endif"))
                return
            if isinstance(ty, CopyArray):
                if ty.c not in emitted:
                    emitted.add(ty.c)
                    guard = ty.c.upper() + "_DEFINED"
                    element = "uint8_t" if ty.array.packed_bits else ty.array.element.c
                    lines.extend((f"#ifndef {guard}", f"#define {guard}",
                                  f"typedef struct {{ {element} data[{ty.array.c_length}]; }} {ty.c};", "#endif"))
                return
            if isinstance(ty, AliasResult):
                add(ty.alias)
                for record in ty.graph_records:
                    add(record)
                return
            if isinstance(ty, (CopyRecord, BorrowRecord)):
                ty = ty.record
            if isinstance(ty, Contract):
                for param in ty.params:
                    add(param)
                add(ty.result)
                return
            if not isinstance(ty, Record) or ty.c in emitted:
                return
            emitted.add(ty.c)
            for _, field in ty.fields:
                add(field)
            guard = ty.c.upper() + "_DEFINED"
            lines.extend((f"#ifndef {guard}", f"#define {guard}", f"typedef struct {ty.c} {{"))
            for name, field in ty.layout:
                field = unqualified(field)
                if isinstance(field, Array):
                    lines.append(f"    {field.element.c} {c_field(name)}[{field.c_length}];")
                else:
                    lines.append(f"    {field.c} {c_field(name)};")
            lines.extend((f"}} {ty.c};", "#endif"))

        for ty in self.types.values():
            add(ty)
        for sym in self.symbols:
            add(sym.ty)
        for _, contract in self.functions.values():
            add(contract)
        for binding in (*self.imported_functions.values(), *self.externs.values()):
            add(binding.contract)
        return lines
