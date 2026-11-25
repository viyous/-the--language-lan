"""Bit regions, bit facts, bit reads/writes and write fusion."""

from __future__ import annotations

import copy
from dataclasses import replace
from itertools import count
from math import gcd

from float_domain import FloatDomain
from value_domain import Domain

from .syntax import Node
from .typesys import (Array, bit_literal_bytes, bit_literal_value, BitView, BitWritePlan, BOOL, BorrowRecord,
                      DynamicBitWritePlan, require_bit_slice_literal_width, Scalar, Value, ZERO)
from .domain_utils import c_int, container, physical
from .c_emit import indent


class BitMixin:
    def bit_region(self, node, state, *, capture=False, require_proven=False, propagate_exit=False):
        host, start, stop, closed = node.args
        symbol = self.comparison_place(self.packed(host), state, propagate_exit=propagate_exit,
                                       allow_effectful_links=propagate_exit and self.has_propagation(host))
        if symbol.projection is not None and symbol.projection.always_returns:
            return Value(ZERO, symbol.c, effectful=True, always_returns=True)
        self.require_live(symbol, state, node.token)
        ordered = any(self.has_effects(part) for part in (host, start, stop) if part is not None)
        lower = self.slice_endpoint(start, state, "lower bit") if start is not None else Value(ZERO, "0")
        if propagate_exit and lower.always_returns:
            return replace(lower, code=f"((void)({symbol.c}), ({lower.code}))")
        limit = Value(Domain.point(symbol.ty.length), c_int(symbol.ty.length))
        upper = self.slice_endpoint(stop, state, "upper bit") if stop is not None else limit
        if propagate_exit and upper.always_returns:
            return replace(upper, code=f"((void)({symbol.c}), (void)({lower.code}), ({upper.code}))")
        if any(v.floating or v.boolean or v.enum or v.character for v in (lower, upper)):
            node.token.error("bit slice endpoints require integers")
        relation = "<=" if closed else "<"
        limit_relation = "<" if closed else "<="
        lower_domain, upper_domain = lower.domain, upper.domain
        nonnegative = lower_domain.lo >= 0
        upper_bounded = self.prove_order(node.token, upper, limit_relation, limit, state)
        bounded = nonnegative and upper_bounded
        if not bounded and (require_proven or not self.debug_bounds):
            node.token.error("cannot prove dynamic bit slice bounds")
        needs_order_check = (self.debug_bounds and not require_proven
                             and not self.prove_order(node.token, lower, relation, upper, state))
        checked = self.debug_bounds and not require_proven and (not bounded or needs_order_check)
        width = None
        if checked:
            valid_hi = upper.domain & Domain.of((0, symbol.ty.length - int(closed)))
            valid_lo = lower.domain & Domain.of((0, valid_hi.hi - int(not closed))) if valid_hi else Domain.of()
            if not valid_lo:
                node.token.error("bit slice endpoints cannot form a nonempty in-bounds range")
            lower = replace(lower, domain=valid_lo)
            upper = replace(upper, domain=valid_hi)
            self.trace.append(f"  debug check bit slice id#{symbol.uid} at {node.token.line}:{node.token.col}")
        if not ordered and not checked and start is not None and stop is not None and stop.kind == "binary" and stop.args[0] == "+":
            _, left, right = stop.args
            key = self.predicate_key(start, state)
            if key == self.predicate_key(left, state):
                width = self.expr(right, state)
            elif key == self.predicate_key(right, state):
                width = self.expr(left, state)
        if width is None:
            width = self.binary(node.token, "-", upper, lower, state)
        if closed:
            width = self.binary(node.token, "+", width, Value(Domain.point(1), "1"), state)
        if checked:
            valid_width = width.domain & Domain.of((1, symbol.ty.length))
            if not valid_width:
                node.token.error("bit slice endpoints cannot form a nonempty in-bounds range")
            width = replace(width, domain=valid_width)
            lower = replace(lower, domain=lower.domain & Domain.of((0, symbol.ty.length - valid_width.lo)))
            if not lower.domain:
                node.token.error("bit slice endpoints cannot form a nonempty in-bounds range")
            upper = replace(upper, domain=upper.domain & Domain.of(
                (lower.domain.lo + valid_width.lo - int(closed), symbol.ty.length - int(closed))))
            if not upper.domain:
                node.token.error("bit slice endpoints cannot form a nonempty in-bounds range")
        if width.domain.lo < 1:
            node.token.error("cannot prove bit slice is nonempty")
        guard = state.guard_label()
        self.trace.append(f"  verify bit_slice_bounds id#{symbol.uid} [{guard}] at {node.token.line}:{node.token.col}")
        for condition, established in (
                (f"lower {lower_domain} >= 0", nonnegative),
                (f"lower {lower_domain} {relation} upper {upper_domain}", not needs_order_check),
                (f"upper {upper_domain} {limit_relation} bit len {limit.domain}", upper_bounded)):
            status = "proven" if established else "requires Debug check"
            self.trace.append(f"    check {condition} => {status}")
        self.trace.append(f"    result width {width.domain}" + (" after Debug checks" if checked else ""))
        setup = []
        if ordered or capture or self.debug_bounds:
            suffix = f"{node.token.line}_{node.token.col}"
            data, first, last = (f"lan_region_{part}_{suffix}" for part in ("data", "first", "last"))
            qualifier = "const " if symbol.readonly and not symbol.checked_overlay else ""
            self.helper("global_temp", None, (f"{qualifier}uint8_t *", data))
            self.helper("global_temp", None, ("lan_int", first))
            self.helper("global_temp", None, ("lan_int", last))
            setup = [f"{data} = {symbol.c};", f"{first} = {lower.code};", f"{last} = {upper.code};"]
            if self.debug_bounds:
                check = self.helper("debug_bit_range", None)
                setup.append(f"{check}({first}, {last}, UINT64_C({symbol.ty.length}), {int(closed)});")
            else:
                assume = self.helper("contract_assume", None)
                setup.extend((f"{assume}({first} >= 0);", f"{assume}({first} {relation} {last});",
                              f"{assume}({last} {limit_relation} (lan_int)UINT64_C({symbol.ty.length}));"))
            symbol = replace(symbol, c_place=data)
            lower = replace(lower, code=first)
            width = replace(width, code=f"({last} - {first} + {int(closed)})")
        return symbol, lower, width, setup

    def bit_window_candidates(self, start, width):
        if self.related(start, width):
            first, count = start.affine, width.affine
            inputs = first.origin.domain & first.preimage(start.domain) & count.preimage(width.domain)
            if not inputs:
                return []
            values = self.bounded_window_values(inputs, 64) or []
            pairs = [(first.scale * n + first.offset, count.scale * n + count.offset) for n in values]
            return [(offset, size) for offset, size in pairs
                    if start.domain & Domain.point(offset) and width.domain & Domain.point(size)]

        sizes = self.bounded_window_values(width.domain, 64)
        if not sizes:
            return []
        offsets = start.domain
        if start.affine is not None:
            offsets = offsets.with_congruence(start.affine.offset, abs(start.affine.scale))
        positions = self.bounded_window_values(offsets, 64 // len(sizes)) or []
        return [(offset, size) for offset in positions for size in sizes]

    @staticmethod
    def bit_offset_mask(offsets):
        residue, modulus = offsets.congruence()
        step = max(1, modulus)
        possible = 0
        for low, high in offsets.ranges:
            first = low + (residue - low) % step
            if first > high:
                continue
            count = (high - first) // step + 1
            progression = 1 if count == 1 else ((1 << (count * step)) - 1) // ((1 << step) - 1)
            possible |= progression << first
        return possible

    def bit_window_read_domain(self, state, sym, start, width):
        domains = (self.bit_read_domain(state, sym, offset, size)
                   for offset, size in self.bit_window_candidates(start, width)
                   if offset + size <= sym.ty.length)
        known = Domain.union(domains)
        if known:
            return known
        size = width.domain.hi
        if sym.ty.length <= 4096:
            offsets = start.domain & Domain.of((0, sym.ty.length - width.domain.lo))
            if start.affine is not None:
                offsets = offsets.with_congruence(start.affine.offset, abs(start.affine.scale))
            possible = self.bit_offset_mask(offsets)
            mask, data = self.storage_bits(state, sym)
            if possible and mask:
                zeros = mask & ~data
                storage = (1 << sym.ty.length) - 1
                known_mask = known_data = 0
                for bit in range(size):
                    positions = (possible << bit) & storage
                    if positions and positions & data == positions:
                        known_mask |= 1 << bit
                        known_data |= 1 << bit
                    elif positions & zeros == positions:
                        known_mask |= 1 << bit
                if known_mask == (1 << size) - 1:
                    # Each width truncates the same known bit pattern.
                    sizes = self.bounded_window_values(width.domain, 64)
                    if sizes is not None:
                        return Domain.union(Domain.point(known_data & ((1 << count) - 1))
                                            for count in sizes)
                # A shorter window contributes zero above its width.
                known_mask &= ~(known_data & ~((1 << width.domain.lo) - 1))
                known_data &= known_mask
                return self.masked_bit_domain(known_mask, known_data, size)
        return container(width.domain.hi, False)

    def assumed_scalar_bit_width(self, token, code):
        width = f"lan_scalar_bit_width_{token.line}_{token.col}"
        # This bound holds for every path that can emit the scalar bit operation.
        return self.assumed_result("lan_int", width, code, [f"{width} >= 1", f"{width} <= 64"])

    def read_variable_bits(self, node, state):
        region = self.bit_region(node, state, capture=True, propagate_exit=True)
        if isinstance(region, Value):
            return region
        symbol, start, width, setup = region
        if width.domain.hi > 64:
            node.token.error("integer bit slice conversion requires at most 64 bits")
        domain = container(width.domain.hi, False)
        if start.domain.lo == start.domain.hi and width.domain.lo == width.domain.hi:
            domain = self.bit_read_domain(state, symbol, start.domain.lo, width.domain.lo)
        elif all(not self.has_effects(part) for part in node.args if isinstance(part, Node)):
            domain = self.bit_window_read_domain(state, symbol, start, width)
        load = self.helper("bit_load", None)
        count = self.assumed_scalar_bit_width(node.token, width.code)
        args = [symbol.c, f"(uint64_t)({start.code})", f"(unsigned)({count})", "0"]
        code = self.ordered_call(node.token, load, ["const uint8_t *", "uint64_t", "unsigned", "int"], args)
        if setup:
            steps = ", ".join(s.rstrip(";") for s in setup)
            code = f"({steps}, {code})"
        return Value(domain, code, 64)

    def dynamic_bit_window(self, node, state):
        region = self.bit_region(node, state, capture=True, propagate_exit=True)
        if isinstance(region, Value):
            return region
        symbol, lower, width, setup = region
        size = width.domain
        if size.lo != size.hi or not 1 <= size.lo <= 64:
            node.token.error("dynamic scalar bit slice requires a proven fixed width from 1 to 64")
        widths = getattr(self, "dynamic_bit_widths", None)
        if widths is None:
            self.dynamic_bit_widths = widths = {}
        if widths.setdefault(node, size.lo) != size.lo:
            node.token.error("dynamic scalar bit slice width differs between paths")
        if setup:
            assume = self.helper("contract_assume", None)
            setup.append(f"{assume}(({width.code}) == {size.lo});")
        return symbol, lower, size.lo, setup

    def dynamic_bit_slice(self, node):
        return self.has_effects(node.args[0]) or any(
            endpoint is not None and not self.literal_endpoint(endpoint) for endpoint in node.args[1:3])

    def bit_span(self, node):
        host, start, stop, closed = node.args
        sym = self.packed(host)
        lo = self.const_int(start, ()) if start else 0
        hi = self.const_int(stop, ()) + closed if stop else sym.ty.length
        if not 0 <= lo < hi <= sym.ty.length:
            node.token.error(f"bit slice [{lo}, {hi}) must be nonempty and inside [0, {sym.ty.length})")
        self.trace.append(f"  verify bit slice id#{sym.uid}: [{lo}, {hi}) inside [0, {sym.ty.length})")
        return sym, lo, hi - lo

    def bit_field_target(self, node):
        host, name = node.args
        if host.kind != "name":
            node.token.error("a bit field requires a named view")
        sym = self.lookup(host)
        if not isinstance(sym.ty, BitView) or name.text not in sym.ty.fields:
            name.error(f"unknown bit field {name.text!r}")
        return sym.ty.host, sym.ty.fields[name.text]

    def bit_endian(self, endian):
        if endian == "native":
            return self.helper("native_big", None) + "()"
        return "1" if endian == "be" else "0"

    @staticmethod
    def bit_order(value, width, endian):
        if endian == "le" or width == 8:
            return value
        flipped = int.from_bytes(value.to_bytes(width // 8, "little"), "big")
        if endian == "native" and flipped != value:
            return None
        return flipped

    def known_bits(self, state, sym, offset, width, endian):
        if offset is None or sym.ty.length > 4096:
            return None
        mask, data = self.storage_bits(state, sym)
        required = (1 << width) - 1
        if (mask >> offset) & required != required:
            return None
        return self.bit_order((data >> offset) & required, width, endian)

    @staticmethod
    def bit_raw_domain(domain, width):
        return (domain & container(width, False)) | (domain & Domain.of((-(1 << width), -1))).affine(1, 1 << width)

    @staticmethod
    def domain_bit_facts(domain, width):
        mask, data = (1 << width) - 1, None
        for lo, hi in domain.ranges:
            common = ((1 << width) - 1) ^ ((1 << (lo ^ hi).bit_length()) - 1)
            mask &= common
            if data is not None:
                mask &= ~(data ^ lo)
            data = lo
        residue, modulus = domain.congruence()
        # The power-of-two factor fixes the low bits of every candidate.
        low = ((modulus & -modulus) - 1) & ((1 << width) - 1) if modulus else 0
        known = ((data or 0) & mask) | (residue & low)
        return mask | low, known

    @staticmethod
    def masked_bit_domain(mask, data, width):
        free = ((1 << width) - 1) ^ mask
        # Free low bits already form contiguous intervals. Only enumerate
        # the remaining choices, with a bound on the number of intervals.
        low = (free ^ (free + 1)) >> 1
        high = free ^ low
        if high.bit_count() > 7:
            fixed_low = (mask ^ (mask + 1)) >> 1
            return Domain.of((data, data | free)).with_congruence(data, fixed_low + 1)
        starts = [data]
        while high:
            bit = high & -high
            starts.extend([start | bit for start in starts])
            high ^= bit
        return Domain.of(*((start, start | low) for start in starts))

    def swapped_bit_domain(self, domain, width):
        values = self.bounded_window_values(domain, 64)
        if values is not None:
            return Domain.union(Domain.point(self.bit_order(value, width, "be")) for value in values)
        full = (1 << width) - 1
        known_mask, known_data = self.domain_bit_facts(domain, width)
        ranges = []
        for lo, hi in domain.ranges:
            while lo <= hi:
                # Cover the interval with aligned blocks before reversing bytes.
                # Each block fixes its prefix and can also carry known low bits.
                size = 1 << ((hi - lo + 1).bit_length() - 1)
                if lo:
                    size = min(size, lo & -lo)
                prefix = full ^ (size - 1)
                if not (lo ^ known_data) & prefix & known_mask:
                    swapped_mask = self.bit_order(prefix | known_mask, width, "be")
                    swapped_data = self.bit_order(lo | known_data, width, "be")
                    ranges.append(self.masked_bit_domain(swapped_mask, swapped_data, width))
                lo += size
        return Domain.union(ranges)

    @staticmethod
    def bit_subdomain(domain, shift, width):
        modulus = 1 << width
        if not domain or (shift == 0 and 0 <= domain.lo <= domain.hi < modulus):
            return domain
        ranges = []
        for lo, hi in domain.ranges:
            first, last = lo >> shift, hi >> shift
            if last - first >= modulus - 1:
                ranges = [(0, modulus - 1)]
                break
            first_block, first_value = divmod(first, modulus)
            last_block, last_value = divmod(last, modulus)
            if first_block == last_block:
                ranges.append((first_value, last_value))
            else:
                ranges.extend(((0, last_value), (first_value, modulus - 1)))
        projected = Domain.of(*ranges)
        residue, step = domain.congruence()
        if step and step % (1 << shift) == 0:
            # Dropping low bits divides the stride. Wrapping can only retain
            # the factor shared with the window's modulus.
            stride = step >> shift
            block = domain.lo >> (shift + width)
            if block != domain.hi >> (shift + width):
                stride = gcd(stride, modulus)
            projected = projected.with_congruence((residue >> shift) - block * modulus, stride)
        return projected

    def project_bit_fact(self, domain, order, size, relative, width, endian):
        if isinstance(domain, FloatDomain):
            domain = domain.raw_bits()
            order = order.removeprefix("float:")

        def project(source, destination):
            if source == "be" and (relative % 8 or width % 8):
                little = self.swapped_bit_domain(domain, size)
                part = self.bit_subdomain(little, relative, width)
                return part if destination == "le" else self.swapped_bit_domain(part, width)
            shift = size - relative - width if source == "be" else relative
            part = self.bit_subdomain(domain, shift, width)
            return part if source == destination else self.swapped_bit_domain(part, width)

        orders = ("le", "be") if "native" in (order, endian) else ("le",)
        return Domain.union(project(native if order == "native" else order,
                                    native if endian == "native" else endian)
                            for native in orders)

    def bit_window_fact(self, state, target, offset, width, endian):
        result = container(width, False)
        for (uid, start, size, order), domain in state.bit_values.items():
            if uid == target.uid and start <= offset and offset + width <= start + size:
                result &= self.project_bit_fact(domain, order, size, offset - start, width, endian)
        return result

    def bit_fact_locations(self, state, sym, offset):
        if offset is None:
            return ()
        if not sym.bit_overlay or sym.uid not in state.regions:
            return tuple((target, offset) for target in self.physical_places(sym, state))
        positions = state.regions[sym.uid]
        if len(positions) == 1:
            owner, index = positions[0]
            if index is not None and index.lo == index.hi:
                return ((owner, index.lo * sym.ty.length + offset),)
        return ()

    def bit_read_domain(self, state, sym, offset, width, endian='le', signed=False):
        window = (1 << width) - 1
        if endian == "native":
            little = self.bit_read_domain(state, sym, offset, width, "le")
            big = self.bit_read_domain(state, sym, offset, width, "be")
            raw = little | big
        else:
            mask, data = self.storage_bits(state, sym)
            mask, data = (mask >> offset) & window, (data >> offset) & window
            mask, data = self.bit_order(mask, width, endian), self.bit_order(data, width, endian)
            raw = self.masked_bit_domain(mask, data, width)
        facts = []
        for target, start in self.bit_fact_locations(state, sym, offset):
            facts.append(self.bit_window_fact(state, target, start, width, endian))
        if facts:
            raw &= Domain.union(facts)
        if sym.bit_overlay and offset == 0 and endian == "native":
            if sym.uid in state.regions and width == sym.ty.length:
                values = []
                for owner, position in state.regions[sym.uid]:
                    element = owner.ty.element
                    if not isinstance(element, Scalar) or element.bits != width:
                        break
                    contents = self.actual_array(state, owner)
                    indices = self.array_indices(contents, position, ZERO)
                    if indices:
                        values.append(contents.read(indices))
                else:
                    if values:
                        raw &= Domain.union(value.raw_bits() if isinstance(value, FloatDomain)
                                            else self.bit_raw_domain(value, width) for value in values)
            elif sym.uid not in state.regions:
                components = [self.storage_value(target, state) if isinstance(target.ty, Scalar)
                              else self.scalar_component_value(target, state)
                              for target in self.physical_places(sym, state)]
                if components and all(value is not None and value.width == width
                                      for value in components):
                    raw &= Domain.union(value.domain.raw_bits() if value.floating
                                        else self.bit_raw_domain(value.domain, width) for value in components)
        if signed:
            positive = raw & container(width - 1, False)
            negative = (raw & Domain.of((1 << (width - 1), window))).affine(1, -(1 << width))
            return positive | negative
        return raw

    def restrict_bit_read(self, state, sym, offset, width, domain, endian='le'):
        locations = self.bit_fact_locations(state, sym, offset)
        if not locations or any(target.chain_summary or target.projection is not None for target, _ in locations):
            return [state]
        if sym.bit_overlay and sym.uid not in state.regions and not all(
                isinstance(target.ty, (Scalar, Array)) for target, _ in locations):
            return [state]
        floating = isinstance(domain, FloatDomain)
        if floating:
            mask = data = None
        else:
            domain = self.bit_raw_domain(domain, width)
            mask, data = self.domain_bit_facts(domain, width)
            mask, data = self.bit_order(mask, width, endian), self.bit_order(data, width, endian)
        result = []
        for target, start in locations:
            key = (target.uid, start, width, 'float:' + endian if floating else endian)
            top = FloatDomain.top(width) if floating else container(width, False)
            refined = domain & state.bit_values.get(key, top)
            if not refined:
                continue
            path = replace(state, bit_values=state.bit_values | {key: refined})
            if sym.uid in self.alias_members and (not sym.bit_overlay or sym.uid not in state.regions):
                root, names = self.alias_members[sym.uid]
                owners = frozenset(owner for owner in path.borrows[root]
                                   if owner.uid and self.child_place(owner, names).uid == target.uid)
                path = replace(path, borrows=path.borrows | {root: owners})
            if not sym.bit_overlay and mask is not None and data is not None and sym.ty.length <= 4096:
                old_mask, old_data = state.bits.get(target.uid, (0, 0))
                if ((old_data >> offset) ^ data) & (old_mask >> offset) & mask:
                    continue
                path = path.with_bits(target, (old_mask | (mask << offset), old_data | (data << offset)))
            result.append(path)
        return result

    def preserve_bit_ranges(self, state, sym, offset, width):
        if offset is None or not state.bit_values:
            return {}
        locations = self.bit_fact_locations(state, sym, 0)
        if not locations:
            return {}
        starts = offset if isinstance(offset, Domain) else Domain.point(offset)
        touched = Domain.of(*((lo, hi + width - 1) for lo, hi in starts.ranges))
        written_by_target = {target.uid: touched.affine(1, base) for target, base in locations}
        return self.untouched_bit_ranges(state, written_by_target)

    def untouched_bit_ranges(self, state, written_by_target):
        result = {}
        for key, value in state.bit_values.items():
            uid, start, size, order = key
            written = written_by_target.get(uid)
            if written is None:
                continue
            window = Domain.of((start, start + size - 1))
            if not written & window:
                result[key] = value
                continue
            for first, last in (window - written).ranges:
                width_left = last - first + 1
                part = self.project_bit_fact(value, order, size, first - start, width_left, "le")
                if part != container(width_left, False):
                    part_key = uid, first, width_left, "le"
                    result[part_key] = result.get(part_key, container(width_left, False)) & part
        return result

    def update_dynamic_bit_window(self, state, sym, offset, width, value_at=None, *,
                                  value_domain=None, same_storage_source_at=None):
        if value_at is None and value_domain is not None and sym.projection is None and sym.link_access is None:
            value_at = lambda start, size: value_domain
        starts = offset.domain
        if offset.affine is not None:
            form = offset.affine
            starts = starts.with_congruence(form.offset, abs(form.scale))
        first = offset.domain.lo
        last = min(sym.ty.length, offset.domain.hi + width.domain.hi)
        positions = self.bounded_window_values(starts, 64)
        written = (Domain.of(*((start, min(last, start + width.domain.hi) - 1) for start in positions))
                   if positions else None)
        candidates = ([(start, size) for start, size in self.bit_window_candidates(offset, width)
                       if start + size <= sym.ty.length] if value_at is not None else [])
        if candidates:
            written = Domain.of(*((start, start + size - 1) for start, size in candidates))
        elif positions is None and sym.ty.length <= 4096:
            possible = self.bit_offset_mask(starts & Domain.of((0, sym.ty.length - width.domain.lo)))
            touched = 0
            for bit in range(width.domain.hi):
                touched |= possible << bit
            touched &= (1 << sym.ty.length) - 1
            runs = []
            remaining = touched
            while remaining:
                start = (remaining & -remaining).bit_length() - 1
                tail = remaining >> start
                count = (tail ^ (tail + 1)).bit_length() - 1
                runs.append((start, start + count - 1))
                remaining &= ~(((1 << count) - 1) << start)
            if runs:
                written = Domain.of(*runs)
        preserved = (self.preserve_bit_ranges(state, sym, written, 1) if written is not None
                     else self.preserve_bit_ranges(state, sym, starts, width.domain.hi))
        invalidated = written if written is not None else Domain.of((first, last - 1))
        updated = self.update_bits(state, sym, first, last - first, BOOL, written=invalidated)
        updated = replace(updated, bit_values=updated.bit_values | preserved)
        if value_at is None or sym.bit_overlay or sym.ty.length > 4096:
            return updated
        target = self.direct_bit_place(state, sym)
        if target is None:
            return updated

        # Keep only facts that agree after every possible write.
        old_mask, old_data = self.storage_bits(state, sym)
        mask, data = (1 << sym.ty.length) - 1, None
        enclosing = None
        for start, size in candidates:
            fitting = value_at(start, size) & container(size, False)
            if not fitting:
                continue
            if enclosing is None or enclosing:
                origin = same_storage_source_at(start, size) if same_storage_source_at is not None else None
                rewritten = self.rewrite_enclosing_bit_ranges(
                    state, target, start, size, fitting, "le", source_offset=origin)
                enclosing = (rewritten if enclosing is None else
                             {key: enclosing[key] | rewritten[key] for key in enclosing.keys() & rewritten.keys()})
            window = (1 << size) - 1
            value_mask, value_data = self.domain_bit_facts(fitting, size)
            touched = window << start
            candidate = (old_data & ~touched) | (value_data << start)
            mask &= (old_mask & ~touched) | (value_mask << start)
            if data is None:
                data = candidate
            else:
                mask &= ~(data ^ candidate)
        if data is None:
            possible = self.bit_offset_mask(starts & Domain.of((0, sym.ty.length - width.domain.lo)))
            fitting = value_domain & container(width.domain.hi, False) if value_domain is not None else None
            value_mask, value_data = self.domain_bit_facts(fitting, width.domain.hi) if fitting else (0, 0)
            invalidated = 0
            for bit in range(width.domain.hi):
                positions = possible << bit
                if value_mask & (1 << bit):
                    matching = old_data if value_data & (1 << bit) else old_mask & ~old_data
                    positions &= ~matching
                invalidated |= positions
            mask, data = old_mask & ~invalidated, old_data
            if starts:
                # A run of equal source bits proves the overlap of its
                # translated windows, even when the candidates are too many.
                prefix = (1 << width.domain.lo) - 1
                for bit_value, runs in ((0, value_mask & ~value_data), (1, value_mask & value_data)):
                    runs &= prefix
                    while runs:
                        low = (runs & -runs).bit_length() - 1
                        tail = runs >> low
                        size = (tail ^ (tail + 1)).bit_length() - 1
                        runs &= ~(((1 << size) - 1) << low)
                        begin = starts.hi + low
                        end = starts.lo + low + size
                        if begin < end:
                            common = ((1 << (end - begin)) - 1) << begin
                            mask |= common
                            data = (data | common) if bit_value else (data & ~common)
        if enclosing:
            updated = replace(updated, bit_values=updated.bit_values | enclosing)
        return self.store_bits(updated, sym, (mask, data & mask), record_write=False)

    def rewrite_enclosing_bit_ranges(self, state, target, offset, width, domain, endian, *, source_offset=None):
        result = {}
        incoming = None
        for (uid, start, size, order), previous in state.bit_values.items():
            if (uid != target.uid or size <= width or size > 64
                    or start > offset or offset + width > start + size):
                continue
            correlated = source_offset is not None and start <= source_offset <= start + size - width
            if not correlated and incoming is None:
                raw = self.bit_raw_domain(domain, width)
                physical = self.project_bit_fact(raw, endian, width, 0, width, "le")
                incoming = self.bounded_window_values(physical, 64) or []
            if not correlated and not incoming:
                continue
            budget = 64 if correlated else 64 // len(incoming)
            physical = self.project_bit_fact(previous, order, size, 0, size, "le")
            values = self.bounded_window_values(physical, budget)
            if not values:
                continue
            shift = offset - start
            mask = (1 << width) - 1
            keep = ~(mask << shift)
            if correlated:
                # Read the source from the same value, before overwriting any bits.
                source_shift = source_offset - start
                outcomes = {(old & keep) | (((old >> source_shift) & mask) << shift) for old in values}
            else:
                outcomes = {(old & keep) | (new << shift) for old in values for new in incoming}
            updated = Domain.of(*((value, value) for value in outcomes))
            key = uid, start, size, "le"
            result[key] = result[key] & updated if key in result else updated
        return result

    def update_bits(self, state, sym, offset, width, domain, endian="le", *, floating=None, written=None,
                    retain_domain=False):
        window = "unknown offset" if offset is None else f"[{offset},{offset + width})"
        self.trace.append(f"  update entity id#{sym.uid} bits {window}: write {domain}")
        preserved = self.preserve_bit_ranges(state, sym, offset, width)
        if written is None and offset is not None:
            indices = Domain.of((offset, offset + width - 1))
        else:
            indices = written
        if written is None and offset is not None and not sym.bit_overlay and (
                state.bit_values or floating is not None or retain_domain):
            targets = self.physical_places(sym, state)
            if len(targets) == 1:
                target = targets[0]
                if target.projection is None and not target.bit_overlay and not target.chain_summary:
                    preserved.update(self.rewrite_enclosing_bit_ranges(state, target, offset, width, domain, endian))
                    if floating is not None or retain_domain:
                        key = (target.uid, offset, width, "float:" + endian if floating is not None else endian)
                        preserved[key] = floating if floating is not None else self.bit_raw_domain(domain, width)
        state = self.write_storage(state, sym, indices)
        state = replace(state, bit_values=state.bit_values | preserved)
        # Larger bit arrays still get checked and lowered. Keep compile-time
        # masks small enough to use predictably across every abstract path.
        if sym.ty.length > 4096:
            return state
        if offset is None:
            return self.store_bits(state, sym, (0, 0), record_write=False)
        mask, data = self.storage_bits(state, sym)
        mask_width = (1 << width) - 1
        touched = (mask_width << offset if written is None else
                   sum(((1 << (hi - lo + 1)) - 1) << lo for lo, hi in written.ranges))
        mask, data = mask & ~touched, data & ~touched
        if written is None:
            known, value = self.domain_bit_facts(self.bit_raw_domain(domain, width), width)
            if endian == "be":
                known = self.bit_order(known, width, "be")
                value = self.bit_order(value, width, "be")
            elif endian == "native":
                swapped_mask = self.bit_order(known, width, "be")
                swapped_value = self.bit_order(value, width, "be")
                known &= swapped_mask & ~(value ^ swapped_value)
            mask |= known << offset
            data |= (value & known) << offset
        return self.store_bits(state, sym, (mask, data), record_write=False)

    def bit_read_code(self, sym, offset, width, ty, endian):
        order = self.bit_endian(endian)
        helper = self.helper("bit_load", None)
        offset_code = f"(uint64_t)({offset.code})" if isinstance(offset, Value) else f"UINT64_C({offset})"
        code = f"{helper}({sym.c}, {offset_code}, {width}, {order})"
        if ty is not None and ty.signed:
            return f"{self.helper('bit_sign', None)}({code}, {width})"
        return code

    def read_bits(self, token, state, sym, offset, width, ty=None, endian="le"):
        self.require_live(sym, state, token)
        if width > 64:
            token.error("bit slices wider than 64 bits require a bit array container")
        code = self.bit_read_code(sym, offset, width, ty, endian)
        if isinstance(offset, Value):
            offset = offset.domain.lo if offset.domain.lo == offset.domain.hi else None
        signed = ty is not None and ty.signed
        domain = container(width, signed) if offset is None else self.bit_read_domain(state, sym, offset, width, endian, signed)
        result = Value(domain, code, ty.bits if ty else width, ty.boolean if ty else False,
                       enum=ty.enum if ty else None, character=ty.character if ty else None, signed=signed)
        if ty is not None and ty.floating:
            physical = FloatDomain.top(width)
            value = self.float_cast(token, result, "view", replace(ty, domain=physical))
            facts = [state.bit_values.get((target.uid, start, width, 'float:' + endian), physical)
                     for target, start in self.bit_fact_locations(state, sym, offset)]
            if facts:
                value = replace(value, domain=value.domain & FloatDomain.union(facts))
            self.range_check(token, value, ty)
            return value
        if ty is not None:
            self.range_check(token, result, ty)
        return result

    def independent_bit_rhs(self, node):
        if node.kind in ("number", "bit_literal", "boolean", "float_number"):
            return True
        if node.kind == "name":
            sym = self.lookup(node)
            return isinstance(sym.ty, Scalar) and not sym.borrowed and sym.uid not in self.alias_members
        if node.kind == "cast":
            source, strategy, annotation = node.args
            return (strategy in ("safe", "view") and isinstance(self.resolve(annotation), Scalar)
                    and self.independent_bit_rhs(source))
        if node.kind in ("unary", "binary"):
            return all(self.independent_bit_rhs(part) for part in node.args[1:])
        return False

    def direct_bit_fusion_host(self, symbol):
        return (not (symbol.bit_overlay or symbol.borrowed)
                and symbol.projection is None and symbol.link_access is None
                and symbol.uid not in self.alias_members)

    def bit_write_plan(self, code, sym, offset, width, value, endian, target, op, source):
        if (op.text != "=" or offset is None or sym.projection is not None or self.has_effects(target)
                or value.effectful):
            return code
        if source.kind in ("boolean", "float_number") or self.literal_endpoint(source) or bit_literal_value(source) is not None:
            if value.domain.lo == value.domain.hi:
                data = self.bit_order(value.domain.lo & ((1 << width) - 1), width, endian)
                if data is not None:
                    return BitWritePlan(sym.c, offset, width, data, code,
                                        dynamic_safe=self.direct_bit_fusion_host(sym))
        # Ordinary packed storage cannot overlap a scalar local or a by-value
        # parameter. Raw object views and borrowed scalar RHSs can overlap.
        if not sym.bit_overlay and self.independent_bit_rhs(source):
            return BitWritePlan(sym.c, offset, width, None, code, value.code, self.bit_endian(endian),
                                dynamic_safe=self.direct_bit_fusion_host(sym))
        return code

    @staticmethod
    def fuse_bit_writes(writes):
        if len(writes) == 1:
            return [writes[0].fallback]
        base = min(write.offset for write in writes) // 8
        end = max(write.offset + write.width for write in writes)
        address = writes[0].address
        setup = [f"uint64_t lan_fused_{i} = (uint64_t)({write.expression});"
                 for i, write in enumerate(writes) if write.expression is not None]
        code = []
        for byte in range(base, (end + 7) // 8):
            covered = data = 0
            terms = []
            for i in range(len(writes) - 1, -1, -1):
                write = writes[i]
                first = max(byte * 8, write.offset)
                stop = min(byte * 8 + 8, write.offset + write.width)
                if stop <= first:
                    continue
                shift, within = first - write.offset, first - byte * 8
                mask = (((1 << (stop - first)) - 1) << within) & ~covered
                covered |= mask
                if not mask:
                    continue
                if write.data is None:
                    source_shift = str(shift)
                    if write.byte_order != "0":
                        # Big and native endian fields are byte aligned.
                        reversed_shift = write.width - 8 - shift
                        source_shift = (str(reversed_shift) if write.byte_order == "1" else
                                        f"({shift} + !!({write.byte_order}) * ({reversed_shift - shift}))")
                    terms.append(f"(((lan_fused_{i} >> {source_shift}) << {within}) & UINT8_C({mask}))")
                else:
                    data |= ((write.data >> shift) << within) & mask
            if covered:
                cell = f"({address})[{byte}]"
                if covered != 255:
                    terms.insert(0, f"({cell} & UINT8_C({255 ^ covered}))")
                if data or not terms:
                    terms.append(f"UINT8_C({data})")
                code.append(f"{cell} = (uint8_t)({' | '.join(terms)});")
        return ["{", *indent([*setup, *code]), "}"] if setup else code

    @staticmethod
    def dynamic_bit_fusion_group(writes):
        group = []
        for write in writes:
            if isinstance(write, DynamicBitWritePlan):
                group.append(write)
                continue
            if not write.dynamic_safe:
                return None
            setup = ()
            expression = write.expression
            if write.data is not None:
                expression = f"UINT64_C({write.data})"
            elif write.byte_order != "0":
                saved = "lan_fused_ordered"
                setup = (f"uint64_t {saved} = (uint64_t)({expression});",)
                reversed_value = f"(__builtin_bswap64({saved}) >> {64 - write.width})"
                expression = f"({write.byte_order} ? {reversed_value} : {saved})"
            group.append(DynamicBitWritePlan(
                write.address, (write.offset, write.offset), write.width, None, setup,
                f"UINT64_C({write.offset})", str(write.width), expression, (write.fallback,)))
        return group

    @staticmethod
    def dynamic_bit_fusion_base(writes):
        first = min(write.offset_bounds[0] for write in writes) // 8 * 8
        last = max(write.offset_bounds[1] + write.width_bound for write in writes)
        if last - first <= 64:
            return first, 0, (last - first + 7) // 8

        anchor = writes[0].affine
        if anchor is None or not all(write.affine is not None and write.affine.origin is anchor.origin
                                     and write.affine.scale == anchor.scale for write in writes):
            return None
        first = min(write.affine.offset for write in writes)
        last = max(write.affine.offset + write.width_bound for write in writes)
        # Aligning the moving start may add up to seven bits to the span.
        if anchor.scale % 8 == 0:
            span = last - first // 8 * 8
        else:
            span = last - first + 7
        if span <= 64:
            return None, anchor.offset - first, (span + 7) // 8
        return None

    def fuse_dynamic_bit_writes(self, writes, layout):
        fixed_base, anchor_delta, byte_count = layout
        first_write = writes[0]
        assume = self.helper("contract_assume", None)
        statements = [
            f"uint8_t *lan_fused_bytes = {first_write.address};",
            "uint64_t lan_fused_mask = 0, lan_fused_data = 0;",
            "uint64_t lan_fused_base;" if fixed_base is None else
            f"uint64_t lan_fused_base = UINT64_C({fixed_base});",
        ]
        for i, write in enumerate(writes):
            setup = list(write.setup)
            if i == 0 and fixed_base is None:
                setup.append(f"lan_fused_base = ({write.position} - UINT64_C({anchor_delta})) & ~UINT64_C(7);")
            setup.extend((
                f"uint64_t lan_fused_shift = {write.position} - lan_fused_base;",
                f"{assume}(lan_fused_shift <= 64u - {write.width});",
                f"uint64_t lan_fused_low = UINT64_MAX >> (64 - {write.width});",
                "uint64_t lan_fused_part = lan_fused_low << lan_fused_shift;",
                f"uint64_t lan_fused_rhs = (uint64_t)({write.expression});",
                "lan_fused_data = (lan_fused_data & ~lan_fused_part)"
                " | ((lan_fused_rhs & lan_fused_low) << lan_fused_shift);",
                "lan_fused_mask |= lan_fused_part;",
            ))
            statements.extend(("{", *indent(setup), "}"))
        for byte in range(byte_count):
            mask = f"lan_fused_byte_mask_{byte}"
            value = f"(uint8_t)(lan_fused_data >> {byte * 8})"
            cell = f"lan_fused_bytes[lan_fused_base / 8 + {byte}]"
            statements.extend((
                f"uint8_t {mask} = (uint8_t)(lan_fused_mask >> {byte * 8});",
                f"if ({mask}) {{",
                *indent([f"{cell} = {mask} == UINT8_MAX ? {value} : "
                         f"(uint8_t)(({cell} & (uint8_t)~{mask}) | {value});"]),
                "}",
            ))
        return ["{", *indent(statements), "}"]

    def write_bits(self, sym, offset, width, value, endian):
        order = self.bit_endian(endian)
        helper = self.helper("bit_store", None)
        code = f"(uint64_t)({offset.code})" if isinstance(offset, Value) else f"UINT64_C({offset})"
        return f"{helper}({sym.c}, {code}, {width}, (uint64_t)({value.code}), {order});"

    def independent_bit_storage(self, dst, src, state):
        def direct(symbol):
            return (symbol.uid and symbol.projection is None and symbol.link_access is None
                    and not symbol.bit_overlay)

        def automatic(symbol):
            return not (symbol.borrowed or symbol.static_storage or symbol.heap_owner)

        def owned_allocation(symbol):
            return not symbol.borrowed and symbol.heap_owner == symbol.uid

        def distinct_record_fields(a, b):
            if not self.borrow_ancestry.get(a.uid) or not self.borrow_ancestry.get(b.uid):
                return False
            source_a, fields_a = self.param_places[a.uid]
            source_b, fields_b = self.param_places[b.uid]
            if source_a != source_b or not isinstance(self.functions[self.current][1].params[source_a], BorrowRecord):
                return False
            return any(left != right for left, right in zip(fields_a, fields_b))

        destinations = self.physical_places(dst, state)
        sources = self.physical_places(src, state)
        return bool(destinations and sources) and all(
            direct(a) and direct(b) and a.uid != b.uid
            and (automatic(a) or automatic(b) or (owned_allocation(a) and owned_allocation(b))
                 or distinct_record_fields(a, b))
            for a in destinations for b in sources)

    def copy_bits(self, dst, offset, src, start, width, *, disjoint=False):
        if offset % 8 == start % 8 == width % 8 == 0:
            strategy = "memcpy" if disjoint else "memmove"
            self.trace.append(f"  copy byte-aligned bit slices: {strategy}")
            return f"{strategy}(({dst.c}) + {offset // 8}, ({src.c}) + {start // 8}, {width // 8});"
        load = self.helper("bit_load", None)
        store = self.helper("bit_store", None)
        helper = self.helper("bit_copy", disjoint or None, (load, store))
        if disjoint:
            self.trace.append("  copy bit slices with disjoint byte storage: restrict")
        return (f"{helper}({dst.c}, UINT64_C({offset}), {src.c}, "
                f"UINT64_C({start}), UINT64_C({width}));")

    @staticmethod
    def bit_literal_contents(value, width):
        from composite_array import TreeArray

        return TreeArray.from_values(width, BOOL, (Domain.point(int(bit)) for bit in reversed(f"{value:0{width}b}")))

    def write_bit_literal(self, dst, offset, value, width):
        if isinstance(offset, int) and offset % 8 == width % 8 == 0:
            return (f"memcpy(({dst.c}) + {offset // 8}, "
                    f"(const uint8_t[]){bit_literal_bytes(value, width)}, {width // 8});")
        load, store = self.helper("bit_load", None), self.helper("bit_store", None)
        copy = self.helper("bit_copy", True, (load, store))
        start = offset.code if isinstance(offset, Value) else str(offset)
        return (f"{copy}({dst.c}, (uint64_t)({start}), "
                f"(const uint8_t[]){bit_literal_bytes(value, width)}, 0, UINT64_C({width}));")

    def assign_wide_bit_literal(self, target, op, node, paths, value, width):
        states = []
        error = None
        for path in paths:
            region = self.bit_region(target, path, capture=True, propagate_exit=True)
            if isinstance(region, Value):
                candidate = f"(void)({region.code});"
                if error is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(error):
                    error = candidate
                continue
            symbol, start, count, setup = region
            if symbol.readonly and not symbol.checked_overlay:
                target.token.error("cannot modify an immutable bit array")
            require_bit_slice_literal_width(node.token, count.domain, width)
            fixed = start.domain.lo == start.domain.hi
            if op.text == "=":
                if symbol.checked_overlay:
                    candidates = self.bit_window_candidates(start, count)
                    windows = self.checked_bit_windows(op, symbol, start.domain, count.domain, candidates)
                    changed = self.checked_overlay_write(
                        op, path, symbol, [(offset, size, Domain.point(value)) for offset, size in windows], "le")
                elif fixed:
                    changed = self.update_bits(path, symbol, start.domain.lo, width, Domain.point(value))
                else:
                    changed = self.update_dynamic_bit_window(path, symbol, start, count, value_domain=Domain.point(value))
                lower = target.args[1]
                offset = start.domain.lo if lower is None or self.literal_endpoint(lower) else start
                stores = [self.write_bit_literal(symbol, offset, value, width)]
            else:
                if symbol.checked_overlay:
                    op.error("wide compound bit writes through a checked overlay require smaller windows")
                stores, changed = [], path
                for shift in range(0, width, 64):
                    size = min(64, width - shift)
                    part = (value >> shift) & ((1 << size) - 1)
                    offset = Value(start.domain.affine(1, shift), f"({start.code} + {shift})")
                    previous = self.read_bits(op, path, symbol, offset, size)
                    updated = self.binary(op, op.text[0], previous, Value(Domain.point(part), c_int(part)), path)
                    stores.append(self.write_bits(symbol, offset, size, updated, "le"))
                    if fixed:
                        changed = self.update_bits(changed, symbol, offset.domain.lo, size, updated.domain)
                if not fixed:
                    changed = self.update_dynamic_bit_window(path, symbol, start, count)
            states.append(changed)
            code = ["{", *indent([*setup, *stores]), "}"]
        return (states, code) if states else ([], [error, "__builtin_unreachable();"])

    def translated_bit_ranges(self, state, dst, src, first, last, shift):
        source_places, destination_places = self.physical_places(src, state), self.physical_places(dst, state)
        if src.bit_overlay or dst.bit_overlay or len(source_places) != 1 or len(destination_places) != 1:
            return {}
        source, destination = source_places[0], destination_places[0]
        if any(place.projection is not None or place.chain_summary or place.bit_overlay
               for place in (source, destination)):
            return {}
        result = {}
        for (uid, bit, size, endian), value in state.bit_values.items():
            start, stop = max(first, bit), min(last, bit + size)
            if uid != source.uid or start >= stop:
                continue
            width = stop - start
            position = start + shift
            if width != size or position % 8 and endian.removeprefix("float:") != "le":
                value = self.project_bit_fact(value, endian, size, start - bit, width, "le")
                endian = "le"
                if value == container(width, False):
                    continue
            key = destination.uid, position, width, endian
            result[key] = result[key] & value if key in result else value
        return result

    def direct_bit_place(self, state, sym):
        if not sym.bit_overlay:
            places = self.physical_places(sym, state)
            if len(places) == 1:
                place = places[0]
                if place.projection is None and not place.bit_overlay and not place.chain_summary:
                    return place
        return None

    def copied_bit_state(self, state, dst, offset, src, start, width):
        self.trace.append(f"  update entity id#{dst.uid} bits [{offset},{offset + width}): "
                          f"copy id#{src.uid}[{start},{start + width})")
        source_mask, source_data = self.storage_bits(state, src)
        preserved_ranges = self.preserve_bit_ranges(state, dst, offset, width)
        preserved_ranges.update(self.translated_bit_ranges(state, dst, src, start, start + width, offset - start))
        if state.bit_values and width < 64:
            target = self.direct_bit_place(state, dst)
            if target is not None:
                source = self.direct_bit_place(state, src)
                same_storage = source is not None and source.uid == target.uid
                domain = self.bit_read_domain(state, src, start, width)
                preserved_ranges.update(self.rewrite_enclosing_bit_ranges(
                    state, target, offset, width, domain, "le", source_offset=start if same_storage else None))
        state = self.write_storage(state, dst, Domain.of((offset, offset + width - 1)))
        state = replace(state, bit_values=state.bit_values | preserved_ranges)
        if dst.ty.length > 4096:
            return state
        dest_mask, dest_data = self.storage_bits(state, dst)
        window = (1 << width) - 1
        preserved = ~(window << offset)
        return self.store_bits(state, dst, (
            (dest_mask & preserved) | (((source_mask >> start) & window) << offset),
            (dest_data & preserved) | (((source_data >> start) & window) << offset),
        ), record_write=False)
