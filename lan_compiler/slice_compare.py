"""Slice snapshots, disjointness and comparisons."""

from __future__ import annotations

from dataclasses import field
from itertools import count
from math import gcd

from array_domain import ArrayDomain
from float_domain import FloatDomain
from value_domain import Domain

from .syntax import Node
from .typesys import AffineForm, Array, BOOL, Record, Scalar, Slice, unqualified, Value, WIDE, ZERO

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class SliceComparisonMixin:
    def span_snapshot(self, state, span, count):
        positions = span.positions()
        if len(positions) != 1:
            return None
        owner, offset = positions[0]
        if offset is None or offset.lo != offset.hi:
            return None
        contents = self.actual_array(state, owner)
        if not 0 <= offset.lo <= contents.length - count:
            return None
        return contents.slice(offset.lo, offset.lo + count)

    def refine_span_snapshot(self, state, span, source):
        if source is None:
            return [state]
        result = []
        for owner, offset in span.positions():
            if (owner.static_storage and isinstance(owner.ty, Slice)) or offset is None or offset.lo != offset.hi:
                return [state]
            contents = self.actual_array(state, owner)
            if offset.lo < 0 or offset.lo + source.length > contents.length:
                continue
            window = contents.slice(offset.lo, offset.lo + source.length)
            narrowed = self.refine_equal_array(span.element, window, source)
            if narrowed is not None:
                updated = contents.copy_from(narrowed, dst_start=offset.lo, src_start=0, count=source.length)
                result.append(state.with_array(owner, updated))
        return result

    def split_slice_start(self, span, state):
        start = self.span_start(span)
        return [state] if start is None else self.split_affine_value(start.domain, start.affine, state)

    def split_affine_value(self, domain, form, state):
        if form is None or not form.scale or domain.lo == domain.hi:
            return [state]
        for uid, variable in state.forms.items():
            if variable.origin is not form.origin or not variable.scale:
                continue
            original = form.origin.domain & form.preimage(domain) & variable.preimage(state.values[uid])
            if not original or original.hi - original.lo >= 8:
                continue
            symbol = self.symbols[uid - 1]
            candidates = []
            for value in range(original.lo, original.hi + 1):
                if original & Domain.point(value):
                    narrowed = Domain.point(variable.scale * value + variable.offset)
                    candidate = state.narrow(symbol, narrowed)
                    if candidate is not None:
                        candidates.append(candidate)
            return candidates
        return [state]

    def span_has_effects(self, node, span=None):
        if self.debug_bounds and node.kind == "slice":
            return ((span is not None and not span.bounds_proven)
                    or any(self.has_effects(part) for part in node.args if isinstance(part, Node)))
        return self.has_effects(node)

    def compare_correlated_span_contents(self, token, left, right, state, a, b, remaining=2):
        if (not a.length.domain & b.length.domain
                or self.prove_order(token, a.length, "!=", b.length, state)):
            return ZERO
        equal = self.compare_span_contents(a, b, state, token)
        if equal != BOOL:
            return equal
        prefix = min(8, a.length.domain.lo, b.length.domain.lo)
        if prefix > 0 and isinstance(a.element, Scalar) and a.element == b.element:
            complete = a.length.domain.hi == b.length.domain.hi == prefix
            equal = Domain.point(1) if complete else BOOL
            for index in range(prefix):
                offset = Domain.point(index)
                relative = Value(offset, "")
                values = [self.known_array_element(state, self.array_element_key(span, relative),
                                                   self.read_span_contents(state, span, offset))
                          for span in (a, b)]
                left_value, right_value = values
                if a.element.floating:
                    comparison = left_value.compare("==", right_value)
                    if comparison == ZERO:
                        return ZERO
                    if comparison != Domain.point(1):
                        equal = BOOL
                else:
                    if not left_value & right_value:
                        return ZERO
                    if left_value != right_value or left_value.lo != left_value.hi:
                        equal = BOOL
            if equal != BOOL:
                return equal
        if not remaining:
            return BOOL
        for value in (self.span_start(a), self.span_start(b), a.length, b.length):
            if value is None:
                continue
            candidates = self.split_affine_value(value.domain, value.affine, state)
            if not candidates or len(candidates) == 1 and candidates[0] is state:
                continue
            result = Domain()
            for candidate in candidates:
                x = self.ordinary_span(left, candidate)
                y = self.ordinary_span(right, candidate)
                result |= self.compare_correlated_span_contents(
                    token, left, right, candidate, x, y, remaining - 1)
                if result == BOOL:
                    break
            return result
        return BOOL

    def span_equality_windows(self, left, right, state, remaining=2):
        a, b = self.ordinary_span(left, state), self.ordinary_span(right, state)
        lengths = a.length.domain & b.length.domain
        if not lengths:
            return
        snapshot_a = self.span_snapshot(state, a, lengths.lo)
        snapshot_b = self.span_snapshot(state, b, lengths.lo)
        if remaining:
            for span, snapshot in ((a, snapshot_a), (b, snapshot_b)):
                if snapshot is not None:
                    continue
                candidates = self.split_slice_start(span, state)
                if len(candidates) == 1 and candidates[0] is state:
                    continue
                for candidate in candidates:
                    yield from self.span_equality_windows(left, right, candidate, remaining - 1)
                return
        yield state, a, b, snapshot_a, snapshot_b

    def span_equality_paths(self, node, state):
        _, left, right = node.args
        left_kind, right_kind = self.span_kind(left), self.span_kind(right)
        if self.has_effects(node, allow_debug_checks=True):
            return [state]
        if left_kind == right_kind == "packed":
            return self.packed_equality_paths(left, right, state)
        if left_kind != "ordinary" or right_kind != "ordinary":
            return self.aggregate_equality_paths(left, right, state)
        a, b = self.ordinary_span(left, state), self.ordinary_span(right, state)
        paths = self.span_equal_length_paths(node, state, left, right, a, b)
        result = []
        for path in paths:
            for candidate, a, b, snapshot_a, snapshot_b in self.span_equality_windows(left, right, path):
                first = self.refine_span_snapshot(candidate, a, snapshot_b)
                result.extend(p for one in first for p in self.refine_span_snapshot(one, b, snapshot_a))
        return result

    def span_equal_length_paths(self, node, state, left, right, a, b):
        x, y = self.fresh_value(a.length, state), self.fresh_value(b.length, state)
        length = x.domain & y.domain
        state = self.assume_difference(state, self.linear_difference("-", x, y), "==", node.token)
        if state is None:
            return []
        paths = self.restrict(state, Node(left.token, 'length', (left,)), length)
        return [p for path in paths for p in self.restrict(path, Node(right.token, 'length', (right,)), length)]

    def packed_equality_paths(self, left, right, state):
        places = []
        for node in (left, right):
            symbol = self.place(node)
            if symbol is None or symbol.bit_overlay:
                return [state]
            if symbol.projection is not None:
                return self.aggregate_equality_paths(left, right, state, packed=True)
            targets = self.physical_places(symbol, state, allow_null=True)
            if len(targets) != 1 or not targets[0].uid:
                return [state]
            if targets[0].projection is not None:
                return self.aggregate_equality_paths(left, right, state, packed=True)
            places.append(targets[0])
        a, b = places
        if a.ty.length != b.ty.length:
            return []
        a_mask, a_bits = self.storage_bits(state, a)
        b_mask, b_bits = self.storage_bits(state, b)
        if a_mask & b_mask & (a_bits ^ b_bits):
            return []
        facts = a_mask | b_mask, (a_bits & a_mask) | (b_bits & b_mask)
        return [state.with_bits(a, facts).with_bits(b, facts)]

    def packed_equality(self, left, right, state):
        a_mask, a_bits = self.storage_bits(state, left)
        b_mask, b_bits = self.storage_bits(state, right)
        common = a_mask & b_mask
        width = left.ty.length
        if common.bit_length() > width:
            common &= (1 << width) - 1
        if common & (a_bits ^ b_bits):
            return ZERO
        return Domain.point(1) if common.bit_count() == width else BOOL

    @staticmethod
    def contains_float(ty):
        ty = unqualified(ty)
        if isinstance(ty, Scalar):
            return ty.floating
        if isinstance(ty, Record):
            return any(Compiler.contains_float(field) for _, field in ty.fields)
        return isinstance(ty, Array) and Compiler.contains_float(ty.element)

    def same_fixed_slice_owner(self, left, right):
        if left.uid <= 0 or left.uid != right.uid:
            return False
        a, b = left.projection, right.projection
        if a is None or b is None:
            return a is b and left == right
        return (a.resolved and b.resolved and a.captured is None and b.captured is None
                and a.root == b.root and isinstance(a.root.ty, Array) and a.root.link_access is None
                and a.root.uid not in self.alias_members and a.steps == b.steps
                and all(isinstance(step, str) or isinstance(step, Domain) and step.lo == step.hi
                        for step in a.steps))

    @staticmethod
    def row_index_value(step):
        if isinstance(step, tuple):
            return Value(WIDE, "", linear=step)
        if isinstance(step, AffineForm):
            return Value(WIDE, "", affine=step)
        return Value(Domain.point(step), "") if isinstance(step, int) else None

    def same_captured_row(self, a, b, state):
        left, right = a.row_identity, b.row_identity
        if left is None or right is None or left[:2] != right[:2] or len(left[2]) != len(right[2]):
            return False
        for first, second in zip(left[2], right[2]):
            if first == second:
                continue
            x, y = self.row_index_value(first), self.row_index_value(second)
            if x is None or y is None or self.dependent_difference_domain(x, y, state) != ZERO:
                return False
        return True

    def shared_span_starts(self, a, b, state):
        left, right = a.positions(), b.positions()
        if len(left) == len(right) == 1:
            same_row = self.same_captured_row(a, b, state)
            if same_row or self.same_fixed_slice_owner(left[0][0], right[0][0]):
                start_a, start_b = self.span_start(a), self.span_start(b)
                if start_a is not None and start_b is not None:
                    return start_a, start_b
        return None

    def disjoint_captured_rows(self, a, b, state):
        left, right = a.row_identity, b.row_identity
        if left is None or right is None or left[:2] != right[:2]:
            return False
        for x, y in zip(left[2], right[2]):
            if isinstance(x, AffineForm) and isinstance(y, AffineForm):
                bounds = self.affine_difference_bounds(x, y)
                if bounds is not None and (bounds[0] > 0 or bounds[1] < 0):
                    return True
            if isinstance(x, (tuple, AffineForm)) or isinstance(y, (tuple, AffineForm)):
                first, second = self.row_index_value(x), self.row_index_value(y)
                if first is not None and second is not None:
                    difference = self.dependent_difference_domain(first, second, state)
                    if difference is not None and not difference & ZERO:
                        return True
                    form = self.combine_linear("-", first, second)
                    if not self.refine_symbolic_congruence(ZERO, form):
                        return True
            elif type(x) is type(y) and x != y:
                return True
        return False

    @staticmethod
    def refine_offset_distance(distance, left, right):
        left_base, left_step = left.domain.congruence()
        right_base, right_step = right.domain.congruence()
        distance = distance.with_congruence(right_base - left_base, gcd(left_step, right_step))

        def residue(value):
            if value.domain.lo == value.domain.hi:
                return value.domain.lo, 0
            if value.affine is not None:
                return value.affine.offset, abs(value.affine.scale)
            return 0, 1

        left_base, left_step = residue(left)
        right_base, right_step = residue(right)
        return distance.with_congruence(right_base - left_base, gcd(left_step, right_step))

    @staticmethod
    def refine_symbolic_congruence(domain, form):
        if form is None:
            return domain
        terms, constant = form
        if not terms:
            return domain & Domain.point(constant)
        modulus = gcd(*(coefficient for _, coefficient in terms))
        return domain.with_congruence(constant, modulus)

    def disjoint_captured_windows(self, token, a, b, state, starts):
        if self.disjoint_captured_rows(a, b, state):
            return True
        if starts is None:
            return False
        left, right = starts
        end_a = self.binary(token, "+", left, a.length, state)
        end_b = self.binary(token, "+", right, b.length, state)
        if (self.prove_order(token, end_a, "<=", right, state)
                or self.prove_order(token, end_b, "<=", left, state)):
            return True
        # The windows may change order. Only distances in this open interval
        # can overlap, even when the lengths and starts vary together.
        overlap = Domain.of((1 - b.length.domain.hi, a.length.domain.hi - 1))
        return not self.offset_distance_may_fit(token, left, right, state, overlap)

    def offset_equality_domain(self, token, left, right, state):
        equal = self.binary(token, "==", left, right, state).domain
        if equal == BOOL and not self.offset_distance_may_fit(token, left, right, state, ZERO):
            return ZERO
        return equal

    def offset_distance_may_fit(self, token, left, right, state, target):
        delta = self.binary(token, "-", right, left, state)
        distance = self.refine_offset_distance(delta.domain, left, right)
        distance = self.refine_symbolic_congruence(distance, self.linear_value(delta))
        possible = distance & target
        return bool(possible and any(possible.with_congruence(residue, 8)
                                     for residue in self.integer_residues_mod8(delta)))

    def slice_address_starts(self, left, right, a, b, state, *, allow_debug_checks=False):
        if (left.kind == right.kind == "slice" and a.relative_bounds is not None
                and b.relative_bounds is not None):
            key = self.predicate_key(left.args[0], state, allow_debug_checks=allow_debug_checks)
            if key is not None and key == self.predicate_key(right.args[0], state,
                                                            allow_debug_checks=allow_debug_checks):
                return a.relative_bounds[0], b.relative_bounds[0]
        return self.shared_span_starts(a, b, state)

    def slice_address_fact_key(self, node, state):
        op, left, right = node.args
        if self.has_effects(node, allow_debug_checks=True):
            return None

        def base(operand):
            while operand.kind == "slice":
                parent, lower, _, _ = operand.args
                if lower is not None and not (lower.kind == "number" and type(lower.args[0]) is int
                                              and lower.args[0] == 0):
                    break
                operand = parent
            return operand

        comparison = Node(node.token, "binary", (op, base(left), base(right)))
        return self.predicate_key(comparison, state, allow_debug_checks=True)

    def slice_address_domain(self, token, left, right, a, b, state):
        for first, second in ((left, right), (right, left)):
            for op in ("==", "!="):
                comparison = Node(token, "binary", (op, first, second))
                known = state.predicates.get(self.slice_address_fact_key(comparison, state))
                if known is not None:
                    return Domain.point(known == (op == "=="))
        key = self.predicate_key(left, state)
        if key is not None and key == self.predicate_key(right, state):
            return Domain.point(1)
        starts = self.slice_address_starts(left, right, a, b, state)
        if starts is not None:
            return self.offset_equality_domain(token, *starts, state)
        # Empty windows can carry a one-past pointer or a null pointer.
        if not (a.text or b.text) and a.length.domain.lo > 0 and b.length.domain.lo > 0:
            if self.disjoint_spans(a, b) or self.disjoint_captured_rows(a, b, state):
                return ZERO
        return BOOL

    def shallow_address_paths(self, node, state, yes, no):
        op, left, right = node.args
        if self.has_effects(node, allow_debug_checks=True):
            return yes, no
        ordinary = self.span_kind(left) == self.span_kind(right) == "ordinary"
        if ordinary:
            a, b = self.ordinary_span(left, state), self.ordinary_span(right, state)
            starts = self.slice_address_starts(left, right, a, b, state, allow_debug_checks=True)
            indices = (starts,) if starts is not None else ()
        else:
            a, b = self.place(left), self.place(right)
            if a is None or b is None or a.projection is None or b.projection is None:
                return yes, no
            a, b = self.comparison_place(a, state), self.comparison_place(b, state)
            indices = self.projected_address_indices(a, b)
        if not indices:
            return yes, no
        differences = []
        for x, y in indices:
            x, y = self.fresh_value(x, state), self.fresh_value(y, state)
            if not self.prove_order(node.token, x, "==", y, state):
                differences.append(self.linear_difference("-", x, y))

        def narrow(paths, equal):
            if not equal and 2 * len(differences) * len(paths) > self.PATH_LIMIT:
                return paths
            narrowed = []
            for path in paths:
                for difference in differences:
                    if not equal and difference is None:
                        narrowed.append(path)
                        break
                    if not equal:
                        for order in ("<", ">"):
                            branch = self.assume_difference(path, difference, order, node.token)
                            if branch is not None:
                                narrowed.append(branch)
                    path = self.assume_difference(path, difference, "==", node.token)
                    if path is None:
                        break
                else:
                    if equal:
                        narrowed.append(path)
            return narrowed

        return narrow(yes, op == "=="), narrow(no, op == "!=")

    def same_slice_window(self, token, left, right, a, b, state):
        return (self.prove_order(token, a.length, "==", b.length, state)
                and self.slice_address_domain(token, left, right, a, b, state) == Domain.point(1))

    def reflexive_slice_contents(self, token, span, state):
        if not self.contains_float(span.element) or span.length.domain.hi == 0:
            return True
        indices = Domain.of((0, span.length.domain.hi - 1))
        if isinstance(span.element, Scalar):
            numeric = span.element.domain.without_nan()
            return self.read_span_contract_contents(token, state, span, indices, numeric).within(numeric)

        from composite_array import RecordValue, TextContent, TreeArray

        pending = []
        for owner, offset in span.positions():
            contents = self.actual_array(state, owner)
            selected = self.array_indices(contents, offset, indices)
            if selected:
                pending.append(contents.read(selected))
        if not pending:
            return False
        budget = 256
        while pending:
            budget -= 1
            if budget < 0:
                return False
            value = pending.pop()
            if isinstance(value, FloatDomain):
                if value.nan:
                    return False
            elif isinstance(value, RecordValue):
                pending.extend(field for _, field in value.fields)
            elif isinstance(value, (ArrayDomain, TreeArray)):
                pending.extend(segment.domain for segment in value.segments)
            elif not isinstance(value, (Domain, TextContent)):
                return False
        return True

    def compare_spans(self, node, state):
        record = self.compare_records(node, state)
        if record is not None:
            return record
        op, left, right = node.args
        if op in ("==", "!=", "===", "!=="):
            a, b = self.place(left), self.place(right)
            packed = lambda sym: sym is not None and isinstance(sym.ty, Array) and sym.ty.packed_bits
            if packed(a) and packed(b):
                a = self.comparison_place(a, state, propagate_exit=True)
                if a.projection is not None and a.projection.always_returns:
                    return Value(BOOL, a.c, 8, True, effectful=True, always_returns=True)
                b = self.comparison_place(b, state, propagate_exit=True)
                if b.projection is not None and b.projection.always_returns:
                    code = f"((void)({a.c}), (void)({b.c}), 0)"
                    return Value(BOOL, code, 8, True, effectful=True, always_returns=True)
                self.require_live(a, state, node.token)
                self.require_live(b, state, node.token)
                effects = self.has_effects(node)
                stable = not self.has_effects(node, allow_debug_checks=True)
                if op in ("==", "!="):
                    test = f"((const void *)({a.c}) {op} (const void *)({b.c}))"
                    if effects:
                        helper = self.helper("binary_eval", None, ("uint8_t", "const void *", "const void *", f"(a {op} b)"))
                        test = f"{helper}({a.c}, {b.c})"
                    if not stable:
                        domain = BOOL
                    elif self.independent_bit_storage(a, b, state):
                        domain = Domain.point(op == "!=")
                    else:
                        equal = self.shallow_projection_domain(node, a, b, state)
                        if equal == BOOL and not (a.bit_overlay or b.bit_overlay):
                            left_places = self.physical_places(a, state)
                            right_places = self.physical_places(b, state)
                            if (len(left_places) == len(right_places) == 1 and left_places == right_places
                                    and left_places[0].projection is None and not left_places[0].bit_overlay
                                    and not left_places[0].chain_summary):
                                equal = Domain.point(1)
                        domain = equal if op == "==" else equal.affine(-1, 1)
                    return Value(domain, test, 8, True)
                helper = self.helper("slice_equal", None, self.helper("contract_assume", None))
                args = [f"((lan_slice){{{sym.c}, {sym.ty.c_length}}})" for sym in (a, b)] + ["1"]
                test = (self.ordered_call(node.token, helper, ["lan_slice", "lan_slice", "size_t"], args)
                        if effects else f"{helper}({', '.join(args)})")
                equal = (ZERO if a.ty.length != b.ty.length else BOOL if not stable else
                         Domain.point(1) if self.guarded_value_equality(left, right, state) else
                         self.packed_equality(a, b, state))
                return Value(equal.affine(-1, 1) if op == "!==" else equal,
                             f"!({test})" if op == "!==" else test, 8, True)
        if self.span_kind(left) != "ordinary" and self.span_kind(right) != "ordinary":
            return None
        if op not in ("==", "!=", "===", "!=="):
            node.token.error("ordinary arrays and slices only support equality comparison")
        a, b, exit_value = self.comparison_spans(left, right, state)
        if exit_value is not None:
            return exit_value
        return self.compare_ordinary_spans(node, state, a, b)

    def compare_ordinary_spans(self, node, state, a, b):
        op, left, right = node.args
        if a.element != b.element and not (a.text and b.text):
            node.token.error("slice comparison requires the same element type")
        debug_pure = (self.debug_bounds and not self.has_effects(left, allow_debug_checks=True)
                      and not self.has_effects(right, allow_debug_checks=True))
        checked_key = self.predicate_key(left, state, allow_debug_checks=True) if debug_pure else None
        same_checked_window = (checked_key is not None
                               and checked_key == self.predicate_key(right, state, allow_debug_checks=True))
        domain = BOOL
        if op in ("==", "!="):
            code = f"(({a.c_value()}).data {op} ({b.c_value()}).data)"
            if same_checked_window:
                domain = Domain.point(op == "==")
            elif ((not self.span_has_effects(left, a) and not self.span_has_effects(right, b))
                  or debug_pure and (self.shared_span_starts(a, b, state) is not None
                                     or self.same_captured_row(a, b, state)
                                     or self.disjoint_captured_rows(a, b, state))):
                equal = self.slice_address_domain(node.token, left, right, a, b, state)
                domain = equal if op == "==" else equal.affine(-1, 1)
        else:
            if self.contains_record_links(a.element):
                node.token.error("arrays containing aliases require explicit per-record ranked comparison")
            if not isinstance(a.element, Scalar) or a.element.floating:
                helper = self.helper("slice_value_equal", a.element, self.helper("contract_assume", None))
            else:
                helper = self.helper("slice_equal", None, self.helper("contract_assume", None))
            code = f"{'!' if op == '!==' else ''}{helper}({a.c_value()}, {b.c_value()}, sizeof({a.element.c}))"
            left_bytes, right_bytes = self.literal_bytes(a), self.literal_bytes(b)
            if not (a.length.domain & b.length.domain):
                domain = Domain.point(op == "!==")
            elif left_bytes is not None and right_bytes is not None:
                domain = Domain.point((left_bytes == right_bytes) == (op == "==="))
            elif ((a.element == b.element or a.text and b.text)
                    and not self.span_has_effects(left, a)
                    and not self.span_has_effects(right, b)):
                if (self.guarded_value_equality(left, right, state)
                        or (self.same_slice_window(node.token, left, right, a, b, state)
                            and self.reflexive_slice_contents(node.token, a, state))):
                    equal = Domain.point(1)
                else:
                    equal = self.compare_correlated_span_contents(node.token, left, right, state, a, b)
                domain = equal.affine(-1, 1) if op == "!==" else equal
            elif (debug_pure and (same_checked_window or self.same_captured_row(a, b, state)
                                  and self.same_slice_window(node.token, left, right, a, b, state))
                  and self.reflexive_slice_contents(node.token, a, state)):
                domain = Domain.point(op == "===")
            elif debug_pure:
                # Use checked span facts without rematerializing endpoints on speculative paths.
                equal = self.compare_correlated_span_contents(node.token, left, right, state, a, b, remaining=0)
                domain = equal.affine(-1, 1) if op == "!==" else equal
        if self.has_effects(left) or self.has_effects(right):
            if op in ("==", "!="):
                test = f"(a.data {op} b.data)"
            else:
                test = f"{'!' if op == '!==' else ''}{helper}(a, b, sizeof({a.element.c}))"
            eval_helper = self.helper("binary_eval", None, ("uint8_t", "lan_slice", "lan_slice", test))
            code = f"{eval_helper}({a.c_value()}, {b.c_value()})"
        return Value(domain, code, 8, True)
