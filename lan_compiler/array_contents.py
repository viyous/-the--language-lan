"""Counting windows and known array element contents."""

from __future__ import annotations

from dataclasses import field, replace
from itertools import count

from storage_reasoning import CountWindow, StorageExit, StorageObservation, WindowCount, count_bounds
from array_domain import ArrayDomain
from value_domain import Domain, DomainComplexity
from table_interface import c_field

from .errors import ALIVE
from .typesys import (Array, HeapArray, is_callback_enum, Scalar, Slice, Span, storage_size, Symbol, Value,
                      WIDE, ZERO)
from .domain_utils import domain_union
from .summaries import count_column, STORAGE_SUMMARY_LIMIT
from .c_emit import c_counted_loop

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class ArrayContentMixin:
    @staticmethod
    def window_counts(state, uid):
        if ("storage_count_owner", uid) in state.predicates:
            for key, fact in state.predicates.items():
                if key[:2] == ("storage_count", uid):
                    yield key, fact

    def count_window(self, state, span, column, relative, length, token):
        positions = span.positions()
        if len(positions) != 1:
            return None
        owner, offset = positions[0]
        if (offset is None or owner.uid <= 0 or owner.projection is not None
                or owner.link_access is not None or owner.bit_overlay or owner.chain_summary
                or not isinstance(owner.ty, (Array, Slice, HeapArray))
                or state.heap.get(owner.heap_owner or owner.uid, ALIVE) != ALIVE):
            return None
        if owner.heap_owner and owner.heap_owner in self.repeated_allocation_uids(state, token):
            return None
        if count_column(owner.ty.element, column) is None:
            return None
        start = self.span_start(span)
        form = self.linear_value(start, prefer_symbolic=True) if start is not None else None
        if offset.lo == offset.hi:
            form = frozenset(), offset.lo
        if form is None:
            return None
        region = CountWindow((form[0], form[1] + relative), length, column)
        footprint = Domain.of((offset.lo + relative, offset.hi + relative + length - 1))
        return owner.uid, region, footprint

    def capture_count_updates(self, state, facts, arguments, token):
        updates = []
        for fact in facts:
            source = arguments[fact.parameter]
            split = fact.path.index(None) if None in fact.path else len(fact.path)
            prefix, column = fact.path[:split], fact.path[split + 1:]
            if isinstance(source, Symbol):
                owners = self.physical_places(source, state, allow_null=True)
                if len(owners) != 1 or owners[0].projection is not None or not owners[0].uid:
                    continue
                field = self.child_place(owners[0], prefix)
                span = Span(field.ty.element, Value(Domain.point(field.ty.length), ""), field.depth, host=field)
            else:
                span = source
            window = self.count_window(state, span, column, fact.start, fact.length, token)
            if window is not None:
                uid, region, footprint = window
                updates.append((("storage_count", uid, region, fact.selected),
                                WindowCount(footprint, fact.bounds)))
        return updates

    def remember_count_updates(self, state, updates):
        if not updates:
            return
        predicates = state.predicates.copy()
        retained = sum(key[:1] == ("storage_count",) for key in predicates)
        for key, fact in updates:
            uid = key[1]
            owner = self.symbols[uid - 1]
            if state.heap.get(owner.heap_owner or uid, ALIVE) != ALIVE:
                continue
            current = predicates.get(key)
            if current is not None:
                low, high = max(fact.bounds[0], current.bounds[0]), min(fact.bounds[1], current.bounds[1])
                if low <= high:
                    fact = replace(fact, bounds=(low, high))
            elif retained >= STORAGE_SUMMARY_LIMIT:
                continue
            else:
                retained += 1
            predicates[key] = fact
            predicates["storage_count_owner", uid] = True
        state.predicates = predicates

    def count_in_span(self, node, state):
        from composite_array import TreeArray, project_scalar_column

        host, annotation, column = node.args
        if self.span_kind(host) != "ordinary":
            host.token.error(".count requires an ordinary array or slice")
        span = self.ordinary_span(host, state)
        if span.length.always_returns:
            return replace(span.length, code=f"((void)({span.c_value()}), 0)", effectful=True)
        for owner, _ in span.positions():
            self.require_target_live(owner, state, host.token)
        element = count_column(span.element, column)
        if element is None or is_callback_enum(element.enum):
            node.token.error(".count needs an integer or boolean element or scalar field")
        selected = self.resolve(annotation)
        if (not isinstance(selected, Scalar) or not isinstance(selected.domain, Domain)
                or is_callback_enum(selected.enum)):
            annotation.token.error(".count needs an integer value set")
        selection = selected.domain
        if element.enum is not None and selected.enum is not None and element.enum != selected.enum:
            annotation.token.error(".count value set has a different enum type")
        low, high = 0, span.length.domain.hi
        membership = self.membership(node.token, selection & WIDE)
        access = "data[lan_i]" + "".join(f".{c_field(name)}" for name in column)
        helper = self.helper("count_in_span", span.element.c, (membership, access))
        code = f"{helper}({span.c_value()})"
        try:
            if not (element.domain & selection):
                high = 0
            elif element.domain.within(selection):
                low = span.length.domain.lo
            elif high and high == span.length.domain.lo:
                window = self.count_window(state, span, column, 0, high, node.token)
                if window is not None:
                    uid, region, _ = window
                    if not region.start[0]:
                        owner = self.symbols[uid - 1]
                        contents = self.actual_array(state, owner)
                        start, stop = region.start[1], region.start[1] + region.length
                        if 0 <= start <= stop <= contents.length:
                            if isinstance(contents, TreeArray):
                                contents = project_scalar_column(contents, column)
                            keys, separated = self.storage_key_observations(state, uid, start, stop)
                            observation = StorageObservation(
                                0, (StorageExit(contents.slice(start, stop), keys, separated),))
                            low, high = count_bounds(observation, selection) or (low, high)
                    for (_, _, other, values), fact in self.window_counts(state, uid):
                        if region.length != other.length or region.column != other.column:
                            continue
                        if region.start != other.start:
                            a = Value(WIDE, "", linear=region.start)
                            b = Value(WIDE, "", linear=other.start)
                            if self.dependent_difference_domain(a, b, state) != ZERO:
                                continue
                        if values.within(selection):
                            low = max(low, fact.bounds[0])
                        if (selection & element.domain).within(values):
                            high = min(high, fact.bounds[1])
                        if not (selection & values):
                            high = min(high, region.length - fact.bounds[0])
                        if element.domain.within(selection | values):
                            low = max(low, region.length - fact.bounds[1])
        except DomainComplexity:
            pass
        if low > high:
            node.token.error("conflicting proofs for storage count")
        return Value(Domain.of((low, high)), code, 32)

    def storage_array(self, state, sym):
        return ArrayDomain.join_all(self.actual_array(state, target) for target in self.physical_places(sym, state))

    def store_array(self, state, sym, contents):
        targets = self.physical_places(sym, state)
        for target in targets:
            value = contents if len(targets) == 1 else self.actual_array(state, target).join(contents)
            state = state.with_array(target, value)
        return state

    @staticmethod
    def array_indices(contents, offset, indices, *, preserve_gaps=False):
        # A span is proved within its owner when it is created. Correlation lost
        # between offset and length may overapproximate indices beyond this cap;
        # those combinations cannot denote an actual element of this span.
        bounds = Domain.of((0, contents.length - 1))
        if not indices:
            return Domain()
        if offset is None:
            return bounds
        if preserve_gaps:
            if offset.predicates or indices.predicates:
                selected = Compiler.sum_index_domains(offset, indices) & bounds
                if selected and selected.predicates and selected.hi - selected.lo < 64:
                    return Domain.of(*((i, i) for i in range(selected.lo, selected.hi + 1)
                                       if Domain.point(i).within(selected)))
                return selected
            starts, selections = offset.ranges, indices.ranges
            if len(starts) * len(selections) > 64:
                starts = ((offset.lo, offset.hi),)
            return Domain.of(*((first + lo, last + hi)
                               for first, last in starts for lo, hi in selections)) & bounds
        return Domain.of((offset.lo + indices.lo, offset.hi + indices.hi)) & bounds

    def refined_span_offset(self, state, span, offset):
        start = self.span_start(span)
        if offset is None or offset.lo == offset.hi or start is None or start.affine is None:
            return offset
        form = start.affine
        original = form.preimage(start.domain)
        for uid, other in state.forms.items():
            if other.origin is form.origin:
                original &= other.preimage(state.values[uid])
        return (offset & original.affine(form.scale, form.offset)) or offset

    def refine_span_position(self, state, span):
        positions = span.positions()
        if len(positions) == 1:
            owner, offset = positions[0]
            narrowed = self.refined_span_offset(state, span, offset)
            if narrowed != offset:
                start = self.span_start(span)
                return replace(span, origins=((owner, narrowed),), start=replace(start, domain=narrowed))
        return span

    def array_element_key(self, span, index):
        positions = span.positions()
        if len(positions) != 1:
            return None
        owner, _ = positions[0]
        valid_owner = isinstance(owner.ty, (Array, HeapArray, Slice))
        if not valid_owner or owner.projection is not None or owner.bit_overlay or owner.chain_summary:
            return None
        start = self.span_start(span)
        if start is None:
            return None
        location = self.combine_linear("+", start, index)
        return ("array_element", owner.uid, location) if location is not None else None

    def known_array_window(self, state, key, indices, domain, *, span_length=None, span_start=None):
        if key is None:
            return domain
        fact = state.predicates.get(("array_window_post", key[1]))
        if fact is None:
            return domain
        epoch, start, minimum, length, start_domain, end, value, _ = fact
        if state.epochs.get(key[1]) is not epoch:
            return domain
        terms, offset = key[2]
        if terms == start[0]:
            relative = offset - start[1]
            if 0 <= relative + indices.lo and relative + indices.hi < minimum:
                return domain & value
            if (relative >= 0 and span_length is not None and length is not None
                    and 0 <= indices.lo <= indices.hi < span_length.domain.hi):
                candidate_length = Compiler.linear_value(span_length, prefer_symbolic=True)
                if (candidate_length is not None and candidate_length[0] == length[0]
                        and relative + candidate_length[1] <= length[1]):
                    return domain & value
        if (span_start is not None and span_length is not None and end is not None
                and span_start.domain.lo >= start_domain.hi
                and 0 <= indices.lo <= indices.hi < span_length.domain.hi):
            candidate_end = self.combine_linear("+", span_start, span_length)
            if candidate_end is not None and candidate_end[0] == end[0] and candidate_end[1] <= end[1]:
                return domain & value
        return domain

    def remember_uniform_array_window(self, state, span, domain):
        key = self.array_element_key(span, Value(ZERO, ""))
        if key is None:
            return state
        owner, location = key[1:]
        length = self.linear_value(span.length, prefer_symbolic=True)
        start = self.span_start(span)
        end = self.combine_linear("+", start, span.length)
        footprint = Domain.of((start.domain.lo, start.domain.hi + span.length.domain.hi - 1))
        fact = (state.epochs.get(owner), location, span.length.domain.lo, length, start.domain, end, domain, footprint)
        return replace(state, predicates=state.predicates | {("array_window_post", owner): fact})

    def known_array_element(self, state, key, domain):
        if key is not None:
            direct = state.predicates.get(key)
            if direct is not None and direct[0] is state.epochs.get(key[1]):
                domain &= direct[1]
            elif domain and domain.lo != domain.hi:
                for index, (other, fact) in enumerate(self.current_array_element_facts(state, key[1])):
                    if index == 8:
                        break
                    if self.array_key_difference(state, key, other) == ZERO:
                        domain &= fact[1]
                        self.trace.append(f"  reuse keyed element id#{key[1]} at a proved equal offset")
        return self.known_array_window(state, key, ZERO, domain)

    @staticmethod
    def current_array_element_facts(state, uid):
        epoch = state.epochs.get(uid)
        for other, fact in state.predicates.items():
            if (isinstance(other, tuple) and len(other) == 3
                    and other[:2] == ("array_element", uid) and fact[0] is epoch):
                yield other, fact

    def array_key_difference(self, state, left, right):
        # Keys are absolute offsets in one backing object. Their terms use
        # captured value origins, so rebinding an index cannot rename a key.
        lhs = Value(WIDE, "", linear=left[2])
        rhs = Value(WIDE, "", linear=right[2])
        return self.dependent_difference_domain(lhs, rhs, state)

    def remember_array_element(self, before, after, key, span, indices, domain):
        uid = key[1]
        _, offset = span.positions()[0]
        footprint = self.sum_index_domains(offset, indices) if offset is not None else None
        epoch = after.epochs[uid]
        facts = {name: fact for name, fact in after.predicates.items()
                 if not (isinstance(name, tuple) and len(name) == 3 and name[:2] == key[:2])}
        retained = 0
        for name, (_, previous_domain, previous_footprint) in self.current_array_element_facts(before, uid):
            if name == key or retained == 7:
                continue
            same_terms = name[2][0] == key[2][0]
            disjoint = (same_terms and name[2][1] != key[2][1]
                        or footprint is not None and previous_footprint is not None
                        and not footprint & previous_footprint)
            if not disjoint:
                difference = self.array_key_difference(before, name, key)
                disjoint = difference is not None and not (difference & ZERO)
                if disjoint:
                    self.trace.append(f"  preserve keyed element id#{uid} at a proved disjoint offset")
            if disjoint:
                facts[name] = epoch, previous_domain, previous_footprint
                retained += 1
        facts[key] = epoch, domain, footprint
        return replace(after, predicates=facts)

    def correlated_span_element(self, state, span, indices, index):
        positions = span.positions()
        if len(positions) != 1:
            return span, indices
        owner, _ = positions[0]
        start = self.span_start(span)
        subscript = replace(index, domain=indices)
        if start is not None and self.related(start, subscript):
            form = self.combine_forms("+", start, subscript)
            absolute = self.correlated_domain(form, start, subscript) if form is not None else None
            if absolute:
                return replace(span, host=None, borrowed_from=span.owners,
                               origins=((owner, ZERO),), start=None), absolute
        return self.refine_span_position(state, span), indices

    @staticmethod
    def untouched_array_fact(state, owner):
        fact = state.predicates.get(("array_untouched", owner.uid))
        if owner.projection is None and fact is not None and state.epochs.get(owner.uid) is fact[0]:
            return fact[1:]
        return None

    def read_array_contents(self, state, owner, selected):
        contents = self.actual_array(state, owner)
        untouched = self.untouched_array_fact(state, owner)
        if untouched is not None:
            before, written = untouched
            try:
                changed = selected & written
                unchanged = selected - written if changed else selected
            except DomainComplexity:
                pass
            else:
                if unchanged:
                    original = before.read(unchanged)
                    return original | contents.read(changed) if changed else original
        return contents.read(selected)

    def read_span_contents(self, state, span, indices, *, absolute_limit=None):
        values = []
        for owner, offset in span.positions():
            contents = self.actual_array(state, owner)
            selected = (self.array_indices(contents, offset, indices) if offset is None else
                        self.sum_index_domains(offset, indices) & Domain.of((0, contents.length - 1)))
            if absolute_limit is not None:
                selected &= Domain.of((0, absolute_limit - 1))
            if selected:
                values.append(self.read_array_contents(state, owner, selected))
        result = domain_union(values)
        return result or span.element.domain

    def known_span_window(self, state, span, indices, domain):
        key = self.array_element_key(span, Value(ZERO, ""))
        return self.known_array_window(state, key, indices, domain, span_length=span.length,
                                       span_start=self.span_start(span))

    def read_span_contract_contents(self, token, state, span, indices, required):
        end_domain = self.span_end_domain(token, span, state, span.positions())
        actual = self.read_span_contents(state, span, indices,
                                         absolute_limit=end_domain.hi if end_domain else None)
        key = self.array_element_key(span, Value(ZERO, ""))
        actual = self.known_array_window(state, key, indices, actual, span_length=span.length,
                                         span_start=self.span_start(span))
        if actual.within(required):
            return actual
        offsets = self.bounded_window_values(indices, 8)
        if offsets:
            facts = []
            for offset in offsets:
                key = self.array_element_key(span, Value(Domain.point(offset), ""))
                if key is None or key not in state.predicates:
                    break
                epoch, value, _ = state.predicates[key]
                if state.epochs.get(key[1]) is not epoch:
                    break
                facts.append(value)
            else:
                return actual & domain_union(facts)
        windows = self.span_candidate_windows(token, span, state)
        if not windows:
            return actual
        values = []
        for owner, start, end in windows:
            relative = indices & Domain.of((0, end - start - 1))
            if relative:
                values.append(self.read_array_contents(state, owner, relative.affine(1, start)))
        return domain_union(values)

    def write_span_contents(self, state, span, indices, value, fill=False, bulk=False, *,
                            element_key=None, write_window=None):
        positions = span.positions()
        state = self.write_span(state, span, indices, element_key=element_key, write_window=write_window)
        for owner, offset in positions:
            contents = self.actual_array(state, owner)
            selected = self.array_indices(contents, offset, indices, preserve_gaps=True)
            if not selected:
                continue
            exact = len(positions) == 1 and offset is not None and offset.lo == offset.hi
            strong_fill = fill and exact and not selected.predicates
            single_write = exact and not bulk and not fill
            if strong_fill:
                updated = contents
                for lo, hi in selected.ranges:
                    updated = updated.fill(lo, hi + 1, value)
            elif single_write:
                updated = contents.write(selected, value)
            else:
                updated = contents.weak_write(selected, value)
            state = state.with_array(owner, updated)
            mode = "fill" if strong_fill else "write" if single_write else "may write"
            self.trace.append(f"  update entity id#{owner.uid} indices {selected}: {mode} {value}")
        return state

    def remember_array_window(self, state, span, values):
        positions = span.positions()
        if len(positions) != 1:
            return state
        _, start = positions[0]
        if start is None:
            return state
        facts = {}
        for index, value in enumerate(values):
            relative = Domain.point(index)
            key = self.array_element_key(span, Value(relative, ""))
            if key is None:
                return state
            facts[key] = state.epochs[key[1]], value, self.sum_index_domains(start, relative)
        if not facts:
            return state
        owner = next(iter(facts))[1]
        retained = {key: fact for key, fact in state.predicates.items()
                    if not (isinstance(key, tuple) and key[:2] == ("array_element", owner))}
        return replace(state, predicates=retained | facts)

    def fill_span_contents(self, state, span, value, end):
        positions = span.positions()
        end_key = self.array_element_key(span, span.length)
        end_fact = state.predicates.get(end_key) if end_key is not None else None
        if end_fact is not None and state.epochs.get(end_key[1]) is not end_fact[0]:
            end_fact = None
        if end is not None and len(positions) == 1 and positions[0][1] is not None:
            owner, start = positions[0]
            contents = self.actual_array(state, owner)
            possible = Domain.of((max(0, start.lo), min(contents.length, end.hi) - 1))
            absolute = replace(span, host=None, borrowed_from=span.owners, origins=((owner, ZERO),), start=None)
            state = self.write_span_contents(state, absolute, possible, value, bulk=True, write_window=span)
            first, last = max(0, start.hi), min(contents.length, end.lo)
            if first < last:
                state = state.with_array(owner, self.actual_array(state, owner).fill(first, last, value))
        else:
            length = span.length.domain
            state = self.write_span_contents(state, span, Domain.of((0, length.hi - 1)), value,
                                             fill=length.lo == length.hi, bulk=True)
        if end_fact is not None:
            preserved = (state.epochs[end_key[1]], *end_fact[1:])
            state = replace(state, predicates=state.predicates | {end_key: preserved})
        return state

    def havoc_span_contents(self, state, span):
        count = span.length.domain
        return self.write_span_contents(state, span, Domain.of((0, count.hi - 1)), span.element.domain,
                                        fill=count.lo == count.hi, bulk=True)

    def copy_span_bit_facts(self, state, source, start, target, offset, count):
        if not state.bit_values or not count or any(
                owner.projection is not None or not isinstance(owner.ty.element, Scalar)
                for owner in (source, target)):
            return {}
        stride = storage_size(source.ty.element) * 8
        if stride != storage_size(target.ty.element) * 8:
            return {}
        start_bit, stop_bit = start * stride, (start + count) * stride
        shift = (offset - start) * stride
        copied = {}
        for (uid, base, width, order), domain in state.bit_values.items():
            if uid != source.uid:
                continue
            first, stop = max(base, start_bit), min(base + width, stop_bit)
            if first >= stop:
                continue
            if first != base or stop != base + width:
                domain = self.project_bit_fact(domain, order, width, first - base, stop - first, "le")
                order = "le"
            key = target.uid, first + shift, stop - first, order
            copied[key] = copied[key] & domain if key in copied else domain
        return copied

    def copy_untouched_array(self, state, target, offset, count, source_fact, start):
        source_before, source_written = source_fact
        window = Domain.of((offset, offset + count - 1))
        written = (source_written & Domain.of((start, start + count - 1))).affine(1, offset - start)
        if not window - written:
            return state
        key = ("array_untouched", target.uid)
        epoch = state.epochs.get(target.uid)
        previous = state.predicates.get(key)
        if previous is not None and previous[0] is epoch:
            before = previous[1]
            written |= previous[2] & Domain.of((0, offset - 1), (offset + count, before.length - 1))
        else:
            before = self.actual_array(state, target)
        before = before.copy_from(source_before, dst_start=offset, src_start=start, count=count)
        return replace(state, predicates=state.predicates | {key: (epoch, before, written)})

    def copy_span_contents(self, state, dst, src, token, *, correlated=False):
        count = dst.length.domain
        positions = src.positions()
        destinations = dst.positions()
        if len(positions) == 1 and destinations:
            owner, start = positions[0]
            if (start is not None and start.lo == start.hi
                    and count.hi <= self.actual_array(state, owner).length - start.lo
                    and all(offset is not None and offset.lo == offset.hi
                            and count.hi <= self.actual_array(state, target).length - offset.lo
                            for target, offset in destinations)):
                source = self.actual_array(state, owner).slice(start.lo, start.lo + count.hi)
                source_fact = self.untouched_array_fact(state, owner)
                copied_bits = {}
                if len(destinations) == 1:
                    target, offset = destinations[0]
                    copied_bits = self.copy_span_bit_facts(state, owner, start.lo, target, offset.lo, count.lo)
                state = self.write_span(state, dst, Domain.of((0, count.hi - 1)))
                if copied_bits:
                    state = replace(state, bit_values=state.bit_values | copied_bits)
                for target, offset in destinations:
                    contents = self.actual_array(state, target)
                    updated = contents.copy_from(source, dst_start=offset.lo, src_start=0, count=count.hi)
                    if len(destinations) != 1 or count.lo != count.hi:
                        updated = updated.join(contents)
                    if len(destinations) == 1 and 0 < count.lo < count.hi:
                        updated = updated.copy_from(source, dst_start=offset.lo, src_start=0, count=count.lo)
                    state = state.with_array(target, updated)
                    if source_fact is not None and count.lo and len(destinations) == 1 and target.projection is None:
                        state = self.copy_untouched_array(state, target, offset.lo, count.lo,
                                                          source_fact, start.lo)
                    if count.hi:
                        mode = "copy" if len(destinations) == 1 and count.lo == count.hi else "may copy"
                        self.trace.append(f"  update entity id#{target.uid} indices [{offset.lo},{offset.lo + count.hi}): "
                                          f"{mode} id#{owner.uid}[{start.lo},{start.lo + count.hi})")
                return state
        dest_end = self.span_end_domain(token, dst, state, destinations) if correlated else None
        if correlated and len(positions) == len(destinations) == 1:
            source_start, dest_start = self.span_start(src), self.span_start(dst)
            if source_start is not None and dest_start is not None and dest_end is not None:
                shift = self.binary(token, "-", dest_start, source_start, state).domain
                if shift.lo == shift.hi:
                    delta = shift.lo
                    owner = positions[0][0]
                    source_array = self.actual_array(state, owner)
                    source_fact = self.untouched_array_fact(state, owner)
                    target = destinations[0][0]
                    target_length = self.actual_array(state, target).length
                    low = max(0, dest_start.domain.lo, delta)
                    high = min(target_length, dest_end.hi, source_array.length + delta)
                    if low < high:
                        snapshot = source_array.slice(low - delta, high - delta)
                        first, last = max(low, dest_start.domain.hi), min(high, dest_end.lo)
                        copied_bits = {}
                        if first < last:
                            copied_bits = self.copy_span_bit_facts(
                                state, owner, first - delta, target, first, last - first)
                        absolute = replace(dst, host=None, borrowed_from=dst.owners,
                                           origins=((target, ZERO),), start=None)
                        state = self.write_span(state, absolute, Domain.of((low, high - 1)), write_window=dst)
                        if copied_bits:
                            state = replace(state, bit_values=state.bit_values | copied_bits)
                        contents = self.actual_array(state, target)
                        copied = contents.copy_from(snapshot, dst_start=low, src_start=0, count=high - low)
                        updated = contents.join(copied)
                        if first < last:
                            updated = updated.copy_from(snapshot, dst_start=first, src_start=first - low,
                                                        count=last - first)
                        state = state.with_array(target, updated)
                        if first < last and source_fact is not None and target.projection is None:
                            state = self.copy_untouched_array(state, target, first, last - first,
                                                              source_fact, first - delta)
                        return state

        indices = Domain.of((0, count.hi - 1))
        source_end = self.span_end_domain(token, src, state, positions) if correlated else None
        value = self.read_span_contents(state, src, indices, absolute_limit=source_end.hi if source_end else None)
        fill = count.lo == count.hi
        write_window = None
        if dest_end is not None and destinations[0][1] is not None:
            target, offset = destinations[0]
            if offset.lo != offset.hi:
                write_window = dst
                dst = replace(dst, host=None, borrowed_from=dst.owners, origins=((target, ZERO),), start=None)
                indices = Domain.of((offset.lo, dest_end.hi - 1))
                fill = False
        return self.write_span_contents(state, dst, indices, value, fill=fill, bulk=True,
                                        write_window=write_window)
