"""Bit copies, overlay writes and bit-span assignment."""

from __future__ import annotations

import copy
from dataclasses import field, replace
from itertools import combinations, count

from bit_domain import bitwise_domain
from value_domain import Domain

from .syntax import Node
from .typesys import (Array, bit_literal_value, bit_literal_width, ReadOnly, Record, RecordLink,
                      require_bit_literal_width, Scalar, Slice, storage_size, unqualified, Value,
                      ValueOrigin, ZERO)
from .domain_utils import container, domain_union
from .c_emit import indent


class BitAssignmentMixin:
    @staticmethod
    def bit_byte_residue(value):
        residue, modulus = value.domain.congruence()
        if modulus % 8 == 0:
            return residue % 8
        if value.affine is not None:
            form = value.affine
            residue, modulus = form.origin.domain.congruence()
            if form.scale * modulus % 8 == 0:
                return (form.scale * residue + form.offset) % 8
        return None

    def integer_residues_mod8(self, value):
        fixed = self.bit_byte_residue(value)
        if fixed is not None:
            return {fixed}

        def residues(domain):
            return {r for r in range(8) if domain.with_congruence(r, 8)}

        possible = residues(value.domain)
        form = self.linear_value(value, prefer_symbolic=True)
        if form is None:
            return possible
        terms, constant = form
        result = {constant % 8}
        for factors, coefficient in terms:
            if coefficient % 8 == 0:
                continue
            product = {coefficient % 8}
            for origin, power in factors:
                inputs = residues(origin.domain) if isinstance(origin, ValueOrigin) else range(8)
                product = {(a * pow(b, power, 8)) % 8 for a in product for b in inputs}
            result = {(a + b) % 8 for a in result for b in product}
        return possible & result

    def disjoint_bit_copy_bytes(self, start, width, origin, count, path):
        dst_end = (start.domain.hi + width.domain.hi + 7) // 8
        src_end = (origin.domain.hi + count.domain.hi + 7) // 8
        if dst_end <= origin.domain.lo // 8 or src_end <= start.domain.lo // 8:
            return True
        low = start.domain.lo - origin.domain.hi
        high = start.domain.hi - origin.domain.lo
        affine = self.affine_difference_bounds(start.affine, origin.affine)
        difference = self.linear_difference("-", start, origin)
        zone = path.zone.range(difference) if difference is not None else None
        for bound in (affine, zone):
            if bound is not None:
                low, high = max(low, bound[0]), min(high, bound[1])
        distance = self.refine_offset_distance(Domain.of((low, high)), origin, start)
        distance = self.refine_symbolic_congruence(distance, self.combine_linear("-", start, origin))
        possible_overlap = distance & Domain.of((-width.domain.hi - 6, count.domain.hi + 6))
        if not possible_overlap:
            return True

        source_residues = self.integer_residues_mod8(origin)
        for dst_residue in self.integer_residues_mod8(start):
            dst_coverage = (dst_residue + width.domain.hi + 7) // 8 * 8
            for src_residue in source_residues:
                src_coverage = (src_residue + count.domain.hi + 7) // 8 * 8
                adjustment = dst_residue - src_residue
                overlap = possible_overlap & Domain.of(
                    (adjustment + 1 - dst_coverage, adjustment + src_coverage - 1))
                if overlap.with_congruence(adjustment, 8):
                    return False
        return True

    def copy_bit_regions(self, target, source, op, paths):
        result = []
        exit_code = None

        def remember_exit(value, prefix=()):
            nonlocal exit_code
            candidate = "{ " + " ".join((*prefix, f"(void)({value.code});")) + " }"
            if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                exit_code = candidate

        byte_aligned = byte_disjoint = True
        for path in paths:
            destination = self.bit_region(target, path, propagate_exit=True)
            if isinstance(destination, Value):
                remember_exit(destination)
                continue
            dst, start, width, dst_setup = destination
            if dst.readonly and not dst.checked_overlay:
                op.error("cannot modify an immutable bit array")
            source_region = self.bit_region(source, path, propagate_exit=True)
            if isinstance(source_region, Value):
                remember_exit(source_region, (*dst_setup, f"(void)({dst.c});",
                                              f"(void)({start.code});", f"(void)({width.code});"))
                continue
            src, origin, count, src_setup = source_region
            if not self.prove_order(op, width, "==", count, path):
                common = width.domain & count.domain
                if not self.debug_bounds or not common:
                    op.error("cannot prove bit slice copy widths are equal")
                self.trace.append("  debug check bit slice copy widths")
                width = replace(width, domain=common)
                count = replace(count, domain=common)
            byte_aligned = byte_aligned and all(
                self.integer_residues_mod8(value) == {0} for value in (start, origin, width))
            pure_windows = all(not self.has_effects(part) for window in (target, source)
                               for part in window.args if isinstance(part, Node))
            if byte_disjoint:
                host_key = self.predicate_key(target.args[0], path) if pure_windows else None
                same_host = dst.c == src.c or (host_key is not None
                    and host_key == self.predicate_key(source.args[0], path))
                projections = (dst.projection, src.projection)
                projected_disjoint = (
                    pure_windows and any(projections)
                    and all(not symbol.bit_overlay for symbol in (dst, src))
                    and all(projection is None or not projection.root.bit_overlay for projection in projections)
                    and self.disjoint_projections(dst, src, path, op))
                byte_disjoint = pure_windows and (
                    self.independent_bit_storage(dst, src, path)
                    or projected_disjoint
                    or same_host and self.disjoint_bit_copy_bytes(
                        start, width, origin, count, path))
            if dst.checked_overlay:
                candidates = self.bit_window_candidates(start, width) if pure_windows else ()
                source_at = None
                if pure_windows:
                    shift = self.binary(op, "-", start, origin, path).domain
                    source_at = self.bit_copy_source_position(start, origin, count, shift)
                result.append(self.checked_bit_copy(
                    op, path, dst, start.domain, width.domain & count.domain, src, origin.domain,
                    candidates, source_at=source_at))
            elif all(value.domain.lo == value.domain.hi for value in (start, origin, width)):
                result.append(self.copied_bit_state(path, dst, start.domain.lo,
                                                    src, origin.domain.lo, width.domain.lo))
            else:
                copied = {}
                value_at = value_domain = same_storage_source_at = None
                track_values = max(dst.ty.length, src.ty.length) <= 4096
                if pure_windows and (path.bit_values or track_values):
                    shift = self.binary(op, "-", start, origin, path).domain
                    if path.bit_values and shift.lo == shift.hi:
                        end = self.binary(op, "+", origin, count, path).domain
                        copied = self.translated_bit_ranges(path, dst, src, origin.domain.hi, end.lo, shift.lo)
                    if track_values:
                        # The shared source facts also cover copies above the candidate budget.
                        value_domain = self.bit_window_read_domain(path, src, origin, count)
                        source_at = self.bit_copy_source_position(start, origin, count, shift)
                        if source_at is None:
                            value_at = lambda position, size: value_domain
                        else:
                            source_place = self.direct_bit_place(path, src)
                            target_place = self.direct_bit_place(path, dst)
                            if (source_place is not None and target_place is not None
                                    and source_place.uid == target_place.uid):
                                same_storage_source_at = source_at

                            def value_at(position, size):
                                source_position = source_at(position, size)
                                if (source_position is None or source_position < 0
                                        or source_position + size > src.ty.length):
                                    return value_domain
                                return self.bit_read_domain(path, src, source_position, size)
                updated = self.update_dynamic_bit_window(
                    path, dst, start, width, value_at, value_domain=value_domain,
                    same_storage_source_at=same_storage_source_at)
                result.append(replace(updated, bit_values=updated.bit_values | copied))
            suffix = f"{op.line}_{op.col}"
            d, i, n, s, j = (f"lan_bit_{part}_{suffix}" for part in ("dst", "start", "count", "src", "origin"))
            if self.debug_bounds:
                check = self.helper("debug_bit_copy_width", None)
                checks = [f"{check}({n}, (uint64_t)({count.code}));"]
            else:
                assume = self.helper("contract_assume", None)
                source_count = f"lan_bit_source_count_{suffix}"
                checks = [f"uint64_t {source_count} = (uint64_t)({count.code});",
                          f"{assume}({n} == {source_count});", f"{assume}({n} > 0);"]
                for position, length in ((i, dst.ty.length), (j, src.ty.length)):
                    checks.extend((f"{assume}({n} <= UINT64_C({length}));",
                                   f"{assume}({position} <= UINT64_C({length}) - {n});"))

            code = ["{", *indent([*dst_setup, f"uint8_t *{d} = {dst.c};",
                                  f"uint64_t {i} = (uint64_t)({start.code});",
                                  f"uint64_t {n} = (uint64_t)({width.code});",
                                  *src_setup, f"const uint8_t *{s} = {src.c};",
                                  f"uint64_t {j} = (uint64_t)({origin.code});",
                                  *checks])]
        if not result:
            return [], [exit_code, "__builtin_unreachable();"]
        if byte_aligned:
            strategy = "memcpy" if byte_disjoint else "memmove"
            self.trace.append(f"  copy byte-aligned dynamic bit slices: {strategy}")
            assume = self.helper("contract_assume", None)
            transfer = [f"{assume}((({i} | {j} | {n}) & 7) == 0);",
                        f"{strategy}({d} + {i} / 8, {s} + {j} / 8, {n} / 8);"]
        else:
            load, store = self.helper("bit_load", None), self.helper("bit_store", None)
            copy = self.helper("bit_copy", byte_disjoint or None, (load, store))
            if byte_disjoint:
                self.trace.append("  copy dynamic bit slices with disjoint byte storage: restrict")
            transfer = [f"{copy}({d}, {i}, {s}, {j}, {n});"]
        return result, [*code, *indent(transfer), "}"]

    def checked_overlay_write(self, token, state, sym, updates, endian, *, floating=None):
        from composite_array import read_path, write_path

        if not updates:
            token.error("bit write has no in-bounds window")

        def updated_domain(old, ty, field_offset, offset, width, value):
            self.spend(token)
            if ty.floating and floating is not None and endian == "native" and offset == field_offset and width == ty.bits:
                return floating
            if ty.floating:
                raw = self.float_cast(token, Value(old, "0", ty.bits), "view", self.types[f"uint{ty.bits}"]).domain
            else:
                raw = self.bit_raw_domain(old, ty.bits)
            new = self.bit_raw_domain(value, width)
            combinations = (sum(hi - lo + 1 for lo, hi in raw.ranges)
                            * sum(hi - lo + 1 for lo, hi in new.ranges))
            results = []
            for big in (False, True):
                reverse = endian == "be" or endian == "native" and big
                if offset == field_offset and width == ty.bits and reverse == big:
                    results.append(new)
                    continue
                mask = 0
                chunks = []
                done = 0
                while done < width:
                    byte, start = divmod(offset + done, 8)
                    take = min(width - done, 8 - start)
                    shift = width - done - take if reverse else done
                    byte -= field_offset // 8
                    if 0 <= byte < ty.bits // 8:
                        position = ((ty.bits // 8 - 1 - byte) if big else byte) * 8 + start
                        chunk_mask = (1 << take) - 1
                        chunks.append((shift, position, chunk_mask))
                        mask |= chunk_mask << position
                    done += take
                keep = ((1 << ty.bits) - 1) ^ mask
                if combinations <= 128:
                    words = {sum(((word >> shift) & bits) << position for shift, position, bits in chunks)
                             for lo, hi in new.ranges for word in range(lo, hi + 1)}
                    results.append(Domain.of(*((old_word & keep | word, old_word & keep | word)
                                               for lo, hi in raw.ranges for old_word in range(lo, hi + 1)
                                               for word in words)))
                    continue
                replacement = ZERO
                for shift, position, bits in chunks:
                    chunk = Domain.of(*((lo >> shift, hi >> shift) for lo, hi in new.ranges))
                    chunk = bitwise_domain("&", chunk, Domain.point(bits))[0]
                    replacement = bitwise_domain("|", replacement, chunk.affine(1 << position, 0))[0]
                preserved = bitwise_domain("&", raw, Domain.point(keep))[0]
                results.append(bitwise_domain("|", preserved, replacement)[0])
            domain = Domain.union(results)
            if ty.floating:
                return self.float_cast(token, Value(domain, "0", ty.bits), "view", self.types[f"float{ty.bits}"]).domain
            if ty.signed:
                sign = 1 << (ty.bits - 1)
                domain = (domain & Domain.of((0, sign - 1))) | (
                    domain & Domain.of((sign, sign * 2 - 1))).affine(1, -sign * 2)
            return domain

        def verified(domain, ty):
            if self.proven_presence(domain - ty.domain, token):
                token.error(f"bit write may violate backing value range {ty.domain}: {domain}")
            return domain

        def write_content(ty, old, first, candidates, weak):
            if isinstance(ty, (RecordLink, Slice)):
                token.error("bit write may overlap a protected reference or text field")
            if isinstance(ty, ReadOnly):
                token.error("cannot modify immutable backing field")
            if isinstance(ty, Array):
                return write_array(ty, old, first, candidates, weak)[0]
            if isinstance(ty, Record):
                result = old
                for names, byte in self.checked_overlay_fields(ty):
                    field = ty
                    readonly = False
                    for name in names:
                        field = dict(field.fields)[name]
                        readonly |= isinstance(field, ReadOnly)
                        field = unqualified(field)
                    start = first + byte * 8
                    stop = start + storage_size(field) * 8
                    affected = [(offset, width, value) for offset, width, value in candidates
                                if offset < stop and offset + width > start]
                    if not affected:
                        continue
                    if readonly:
                        token.error("cannot modify immutable backing field")
                    value = write_content(field, read_path(old, names), start, affected,
                                          weak or len(affected) < len(candidates))
                    result = write_path(result, names, value)
                return result
            result = verified(domain_union(updated_domain(old, ty, first, offset, width, value)
                                           for offset, width, value in candidates), ty)
            return result | old if weak else result

        def write_packed(contents, first, candidates, weak):
            by_bit = {}
            for offset, width, value in candidates:
                raw = self.bit_raw_domain(value, width)
                for done in range(width):
                    index = offset + done - first
                    if not 0 <= index < contents.length:
                        continue
                    self.spend(token)
                    shifts = []
                    if endian in ("le", "native"):
                        shifts.append(done)
                    if endian in ("be", "native"):
                        shifts.append(width - 8 - (done // 8) * 8 + done % 8)
                    domain = Domain.union(bitwise_domain("&", Domain.of(*(
                        (lo >> shift, hi >> shift) for lo, hi in raw.ranges)), Domain.point(1))[0]
                                          for shift in shifts)
                    by_bit.setdefault(index, []).append(domain)
            changed = contents
            for index, domains in by_bit.items():
                value = Domain.union(domains)
                if weak or len(domains) < len(candidates):
                    value |= contents.read(index)
                changed = changed.write(index, value)
            return changed, Domain.of(*((index, index) for index in by_bit))

        def write_array(ty, contents, first, candidates, weak):
            if ty.packed_bits:
                return write_packed(contents, first, candidates, weak)
            stride = storage_size(ty.element) * 8
            by_index = {}
            for offset, width, value in candidates:
                lo = max(0, (offset - first) // stride)
                hi = min(ty.length - 1, (offset + width - 1 - first) // stride)
                self.spend(token, hi - lo + 1)
                for index in range(lo, hi + 1):
                    by_index.setdefault(index, []).append((offset, width, value))
            changed = contents
            for index, affected in by_index.items():
                start = first + index * stride
                old = contents.read(index)
                partial = weak or len(affected) < len(candidates)
                value = write_content(ty.element, old, start, affected, partial)
                changed = changed.write(index, value)
            return changed, Domain.of(*((index, index) for index in by_index))

        if sym.uid in state.regions:
            positions = state.regions[sym.uid]
            for owner, offset in positions:
                self.require_target_live(owner, state, token, access=sym)
                if owner.readonly:
                    token.error("cannot modify immutable backing storage")
                contents = self.actual_array(state, owner)
                selected = self.array_indices(contents, offset, ZERO)
                if not selected:
                    continue
                exact = len(positions) == 1 and selected.lo == selected.hi
                changed = contents
                for segment in contents.segments:
                    indices = selected & Domain.of((segment.start, segment.stop - 1))
                    if not indices:
                        continue
                    ty = owner.ty.element
                    domain = verified(domain_union(updated_domain(segment.domain, ty, 0, start, width, value)
                                                   for start, width, value in updates), ty)
                    changed = changed.write(indices, domain) if exact else changed.weak_write(indices, domain)
                state = self.write_actual_storage(state, owner, selected).with_array(owner, changed)
            return state

        targets = self.physical_places(sym, state)
        writes, array_writes = [], []
        for target in targets:
            self.require_target_live(target, state, token, access=sym)
            if target.readonly:
                token.error("cannot modify immutable backing storage")
            for names, byte in self.checked_overlay_fields(target.ty):
                field = self.child_place(target, names)
                first, last = byte * 8, (byte + storage_size(field.ty)) * 8
                touched = [(offset, width, value) for offset, width, value in updates
                           if offset < last and offset + width > first]
                if not touched:
                    continue
                if isinstance(field.ty, (RecordLink, Slice)):
                    token.error("bit write may overlap a protected reference or text field")
                if field.readonly:
                    token.error("cannot modify immutable backing field")
                if isinstance(field.ty, Array):
                    contents = self.record_value(field, state)
                    changed, indices = write_array(field.ty, contents, first, touched,
                                                   len(targets) > 1 or len(touched) < len(updates))
                    array_writes.append((field, changed, indices))
                    continue
                old = self.storage_value(field, state).domain
                domain = domain_union(updated_domain(old, field.ty, first, offset, width, value)
                                      for offset, width, value in touched)
                verified(domain, field.ty)
                if len(targets) > 1 or len(touched) < len(updates):
                    domain |= old
                writes.append((field, domain))
        for target, domain in writes:
            ty = target.ty
            value = Value(domain, "0", ty.bits, ty.boolean, enum=ty.enum, character=ty.character, signed=ty.signed)
            state = self.store_value(state, target, value)
        for target, contents, indices in array_writes:
            if target.projection is not None:
                state = self.write_storage(state, target)
            else:
                state = self.write_actual_storage(state, target, indices)
            state = self.restore_record_array(target, state, contents)
        return state

    def assign_bit_span(self, target, op, node, paths):
        self.propagating_expression_nodes |= self.propagation_nodes(target)
        literal = bit_literal_value(node)
        if target.kind == "slice" and literal is not None and literal[1] > 64 and op.text in ("=", "&=", "|=", "^="):
            return self.assign_wide_bit_literal(target, op, node, paths, *literal)
        if (target.kind == "slice" and node.kind == "slice" and op.text == "="
                and (self.dynamic_bit_slice(target) or self.dynamic_bit_slice(node)
                     or self.has_effects(target) or self.has_effects(node))):
            return self.copy_bit_regions(target, node, op, paths)
        if target.kind == "slice" and self.dynamic_bit_slice(target):
            return self.assign_dynamic_bits(target, op, node, paths)
        ty, endian = None, "le"
        if target.kind == "member":
            sym, bit = self.bit_field_target(target)
            offset, width, ty, endian = bit.offset, bit.width, bit.ty, bit.endian
        else:
            sym, offset, width = self.bit_span(target)
            if node.kind == "slice" and op.text == "=":
                if sym.checked_overlay:
                    return self.copy_bit_regions(target, node, op, paths)
                src, start, count = self.bit_span(node)
                if sym.readonly:
                    target.token.error("cannot modify an immutable bit array")
                for state in paths:
                    self.require_live(sym, state, target.token)
                    self.require_live(src, state, node.token)
                if count != width:
                    node.token.error("bit slice copy requires equal widths")
                states = [self.copied_bit_state(p, sym, offset, src, start, width) for p in paths]
                disjoint = (sym.c == src.c and ((offset + width + 7) // 8 <= start // 8
                                                   or (start + width + 7) // 8 <= offset // 8)
                            or all(self.independent_bit_storage(sym, src, path) for path in paths))
                return states, [self.copy_bits(sym, offset, src, start, width, disjoint=disjoint)]
        if sym.readonly and not sym.checked_overlay:
            target.token.error("cannot modify an immutable bit field")
        literal_width = literal[1] if literal is not None else bit_literal_width(node)
        if literal_width is not None:
            require_bit_literal_width(node.token, width, literal_width)
        if width > 64:
            target.token.error("writes wider than 64 bits require a same-width bit slice source")
        bounds = container(width, ty.signed if ty else False)
        result = []
        error_code = None
        for path in paths:
            self.require_live(sym, path, target.token)
            old = self.read_bits(target.token, path, sym, offset, width, ty, endian) if op.text != "=" else None
            value = self.expr(node, path)
            if value.always_returns:
                address = old.code if old is not None else sym.c
                candidate = f"{{ (void)({address}); (void)({value.code}); }}"
                if error_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(error_code):
                    error_code = candidate
                continue
            if op.text != "=":
                value = self.binary(op, op.text[:-1], old, value)
            floating = None
            if ty:
                value = self.coerce_storage(op, value, ty)
                self.range_check(op, value, ty)
                if ty.floating:
                    floating = value.domain
                    value = self.float_cast(op, value, "view", Scalar(bounds, width))
            elif value.boolean or value.floating or value.enum or value.character:
                op.error("bit slice assignment requires an integer or same-width slice")
            if self.proven_presence(value.domain - bounds, op):
                op.error(f"cannot prove {value.domain} fits the {width}-bit field range {bounds}")
            if sym.checked_overlay:
                result.append(self.checked_overlay_write(
                    op, path, sym, ((offset, width, value.domain),), endian, floating=floating))
            else:
                result.append(self.update_bits(path, sym, offset, width, value.domain, endian, floating=floating,
                                               retain_domain=ty is not None))
            code = self.write_bits(sym, offset, width, value, endian)
            plan = self.bit_write_plan(code, sym, offset, width, value, endian, target, op, node)
        if not result and error_code is not None:
            return [], [error_code, "__builtin_unreachable();"]
        return result, [plan]

    def budget_paths(self, paths, token):
        if len(paths) > self.PATH_HARD_LIMIT:
            token.error("too many live paths; safety could not be proved within the analysis budget")
