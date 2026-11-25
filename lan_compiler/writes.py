"""Write sets, alias invalidation, graph attachment and call mutations."""

from __future__ import annotations

from dataclasses import field, fields, replace
from math import gcd

from value_domain import Domain, DomainComplexity
from table_interface import c_field

from .typesys import (Array, BorrowArray, BorrowRecord, BorrowScalar, HeapArray, Interface, ReadOnly, Record,
                      RecordLink, Scalar, Slice, Span, storage_size, unqualified, Value, WIDE, ZERO)
from .domain_utils import physical
from .interface_types import (array_mutation_projection, array_mutation_step, encode_array_mutation_step,
                              mutable_chain, mutable_graph)

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class WriteMixin:
    @staticmethod
    def type_leaves(ty, prefix=()):
        if isinstance(ty, ReadOnly):
            return
        if isinstance(ty, Record):
            for name, child in ty.fields:
                yield from Compiler.type_leaves(child, prefix + (name,))
        else:
            yield prefix, ty

    @staticmethod
    def coarse_writes(index, ty):
        """The summary of writing everything `ty` lets a callee write.

        This is the vocabulary a native declaration reuses: a graph or chain
        borrow is written through its own coarse effect, a record through every
        leaf it exposes, and a by-value parameter not at all.
        """
        if mutable_graph(ty):
            effect = "@attach_graph" if ty.attach is not None else "@graph"
            result = [(index, (effect,), None)]
            if ty.prune:
                result.append((index, ("@prune",), None))
            return result
        if isinstance(ty, (BorrowRecord, Slice)) and ty.attach is not None:
            # Without a declared bound the donor may join at any interior
            # link, so the summary must cover every suffix position.
            return [(index, ("@attach_deep",), None)]
        if mutable_chain(ty):
            result = [(index, ("@chain",), None)]
            if ty.prune:
                result.append((index, ("@skip",), None))
            return result
        if isinstance(ty, (BorrowRecord, Interface)) and not ty.readonly:
            return [(index, path, leaf.domain if isinstance(leaf, Scalar) else None)
                    for path, leaf in Compiler.type_leaves(
                        ty.record if isinstance(ty, BorrowRecord) else ty.schema)]
        if isinstance(ty, BorrowArray) and not ty.readonly:
            return [(index, (), None)]
        if isinstance(ty, BorrowScalar) and not ty.readonly:
            return [(index, (), ty.scalar.domain)]
        if isinstance(ty, Slice) and not ty.readonly and not ty.copy:
            return [(index, (), None)]
        return []

    @staticmethod
    def unknown_mutations(contract):
        result = []
        for index, ty in enumerate(contract.params):
            result.extend(Compiler.coarse_writes(index, ty))
        return tuple(result)

    def declared_write_set(self, node, contract, params):
        """Turn `writes { ... }` into the mutation summary a caller applies.

        A native function has no body, so its effect on the caller cannot be
        inferred; the declaration is the whole summary.  `memory` records that
        the call may allocate or release memory the caller did not hand in.
        """
        whole, fields, memory, external_effect = set(), {}, False, False
        indices = {name.text: index for index, (name, _) in enumerate(params)}
        for entry in node.args[0]:
            path = entry.args[0]
            head = path[0]
            if head in ("memory", "all") or (head == "external" and head not in indices):
                if len(path) > 1:
                    entry.token.error(f"`{head}` does not name a field")
                if head == "all":
                    whole.update(range(len(contract.params)))
                memory |= head in ("memory", "all")
                external_effect |= head in ("external", "all")
                continue
            index = indices.get(head)
            if index is None:
                names = ", ".join(name.text for name, _ in params) or "no parameters"
                entry.token.error(f"the write set names unknown parameter {head!r} ({names})")
            fields.setdefault(index, []).append((tuple(path[1:]), entry))
        mutations = []
        for index in sorted(whole | set(fields)):
            ty = contract.params[index]
            if index in whole:
                selected = self.coarse_writes(index, ty)
            elif fields.get(index):
                selected = [item for path, entry in fields[index]
                            for item in self.field_writes(index, ty, path, entry)]
            else:
                selected = []
            for item in selected:
                if item not in mutations:
                    mutations.append(item)
        return tuple(mutations), memory, external_effect

    def field_writes(self, index, ty, path, entry):
        """The summary of writing one field path below a borrowed parameter."""
        name = entry.token.text
        if isinstance(ty, ReadOnly) or getattr(ty, "readonly", False):
            entry.token.error(f"parameter {name!r} is immutable: "
                              "declare it with {immutable: false} to allow writes")
        if not path:
            return self.coarse_writes(index, ty) or self._value_error(name, entry)
        current, offset = ty, 0
        for step in path:
            base = unqualified(current)
            if isinstance(base, (BorrowRecord, Interface)):
                record = base.record if isinstance(base, BorrowRecord) else base.schema
                entries = dict(record.fields)
            else:
                entry.token.error(f"parameter {name!r} has no field {'.'.join(path[:offset + 1])!r}")
            if step not in entries:
                entry.token.error(f"parameter {name!r} has no field {'.'.join(path[:offset + 1])!r}")
            current = entries[step]
            offset += 1
            if isinstance(current, ReadOnly) or getattr(current, "readonly", False):
                entry.token.error(f"field {'.'.join(path[:offset])!r} of {name!r} is immutable")
        leaf = unqualified(current)
        if isinstance(leaf, Scalar):
            return [(index, path, leaf.domain)]
        return [(index, path, None)]

    def _value_error(self, name, entry):
        entry.token.error(f"parameter {name!r} is passed by value and cannot be written: "
                          "a callee only shares a parameter declared as a borrow")

    def register_borrow(self, sym, index, path=(), ancestors=()):
        self.param_places[sym.uid] = (index, path)
        self.borrow_ancestry[sym.uid] = ancestors
        if isinstance(sym.ty, (Record, Interface)):
            for child in sym.fields:
                inherited = tuple((ty, names + (child.name,)) for ty, names in ancestors)
                self.register_borrow(child, index, path + (child.name,),
                                     inherited + ((sym.ty, (child.name,)),))

    @staticmethod
    def same_scalar_storage(left, right):
        return (left.bits, left.signed, left.boolean, left.enum, left.character, left.floating) == (
            right.bits, right.signed, right.boolean, right.enum, right.character, right.floating)

    def may_alias(self, left, right):
        if left.uid == right.uid:
            return True
        a, b = self.param_places.get(left.uid), self.param_places.get(right.uid)
        if a is None or b is None or a[0] == b[0]:
            return False
        params = self.functions[self.current][1].params
        if isinstance(params[a[0]], Interface) or isinstance(params[b[0]], Interface):
            return True
        # A packed borrow can be an object representation of another argument.
        if any(isinstance(sym.ty, Array) and sym.ty.packed_bits for sym in (left, right)):
            return True
        if any(isinstance(sym.ty, (Array, Slice)) and not isinstance(sym.ty.element, Scalar)
               for sym in (left, right)):
            return True
        if isinstance(left.ty, Scalar) and isinstance(right.ty, (Array, Slice)):
            return not getattr(right.ty, "packed_bits", False) and self.same_scalar_storage(left.ty, right.ty.element)
        if isinstance(right.ty, Scalar) and isinstance(left.ty, (Array, Slice)):
            return not getattr(left.ty, "packed_bits", False) and self.same_scalar_storage(right.ty, left.ty.element)
        if isinstance(left.ty, (Array, Slice)) and isinstance(right.ty, (Array, Slice)):
            if isinstance(left.ty, Slice) or isinstance(right.ty, Slice):
                return (not getattr(left.ty, "packed_bits", False) and not getattr(right.ty, "packed_bits", False)
                        and left.ty.element.bits == right.ty.element.bits)
            if not self.borrow_ancestry[left.uid] or not self.borrow_ancestry[right.uid]:
                return left.ty.element == right.ty.element and left.ty.packed_bits == right.ty.packed_bits
        if isinstance(left.ty, Scalar) and isinstance(right.ty, Scalar):
            if not self.borrow_ancestry[left.uid] or not self.borrow_ancestry[right.uid]:
                return (left.ty.bits, left.ty.signed, left.ty.boolean, left.ty.enum, left.ty.character) == (
                    right.ty.bits, right.ty.signed, right.ty.boolean, right.ty.enum, right.ty.character)
        return any(item in self.borrow_ancestry[right.uid] for item in self.borrow_ancestry[left.uid])

    def invalidate_aliases(self, state, written):
        for candidate in self.symbols:
            if candidate.uid == written.uid or not self.may_alias(written, candidate):
                continue
            self.note_transition_write(candidate)
            state = state.touch(candidate.uid).forget_bit_values(candidate.uid)
            if isinstance(candidate.ty, Scalar) and candidate.uid in state.values:
                state = state.update(candidate, candidate.ty.domain)
            elif isinstance(candidate.ty, Array) and candidate.ty.packed_bits:
                state = state.with_bits(candidate, (0, 0))
            elif isinstance(candidate.ty, (Array, Slice)) and candidate.uid in state.arrays:
                state = state.with_array(candidate, state.arrays[candidate.uid].havoc())
        return state

    def note_transition_write(self, symbol, indices=None, *, descriptor=False, bit_view=None,
                              row=None, field_path=None, scalar=False, lifetime=False, link=False,
                              alias_binding=False):
        for frame in getattr(self, "transition_frames", ()):
            if lifetime:
                frame[symbol, "heap_status"] = None
            elif alias_binding:
                frame[symbol, "alias_binding"] = None
            elif descriptor:
                frame[symbol, "length"] = None
                frame[symbol, "slice_target"] = None
                frame[symbol, "slice_start"] = None
            elif field_path is not None:
                frame[symbol, "field_path", field_path] = None
            elif scalar:
                frame[symbol, "scalar_write"] = None
            elif link:
                frame[symbol, "link_write"] = None
            elif indices is None:
                frame[symbol] = None
            else:
                key = row if row is not None else (symbol, "bits", bit_view) if bit_view is not None else symbol
                if frame.get(key, ()) is not None:
                    frame.setdefault(key, []).extend(indices.ranges)

    def record_transition_region(self, state, region, indices):
        tokens = getattr(self, "row_transition_tokens", ())
        if not tokens:
            return state
        writes = state.row_writes | {(token, region): state.row_writes.get((token, region), Domain.of()) | indices
                                    for token in tokens}
        return replace(state, row_writes=writes)

    def record_link_write(self, state, field):
        self.note_transition_write(field, link=True)
        return self.record_transition_region(state, (field, "link_write"), ZERO)

    def register_graph_payload(self, root, index):
        payload = self.chain_payloads.setdefault(index, [])
        for field in root.fields:
            self.chain_fields[field.uid] = index
            if isinstance(field.ty, Scalar):
                payload.append(field)

    def register_chain_payload(self, root, index):
        self.chain_fields[self.child_place(root, ("next",)).uid] = index
        payload = self.chain_payloads.setdefault(index, [])
        for field in root.fields:
            if isinstance(field.ty, Scalar):
                self.chain_fields[field.uid] = index
                payload.append(field)

    def physical_array_row(self, symbol, state):
        if symbol.uid in self.alias_members:
            targets = self.physical_places(symbol, state, allow_null=True)
            if len(targets) != 1 or not targets[0].uid:
                return None
            symbol = targets[0]
        projection = symbol.projection
        if projection is not None and projection.resolved and isinstance(projection.root.ty, Slice):
            targets = self.captured_projection_owners(symbol, state)
            if len(targets) != 1:
                return None
            symbol = targets[0]
        return symbol if self.array_row_write_key(symbol) is not None else None

    def array_row_write_key(self, symbol):
        projection = symbol.projection
        if (projection is None or not projection.resolved or projection.captured is not None
                or not isinstance(symbol.ty, Array) or not isinstance(symbol.ty.element, Scalar)
                or not isinstance(projection.root.ty, Array)
                or projection.root.uid in self.alias_members or projection.root.link_access is not None):
            return None
        steps = []
        for step in projection.steps:
            if isinstance(step, str):
                steps.append(step)
            elif isinstance(step, Domain) and step.lo == step.hi:
                steps.append(step.lo)
            else:
                return None
        return projection.root, "bit_row" if symbol.ty.packed_bits else "array_row", tuple(steps)

    def write_actual_storage(self, state, target, indices=None, *, bit_view=None, field_path=None,
                             mutation_steps=None, array_footprint=None, count_start=None, count_indices=None):
        count_footprint = array_footprint
        if (count_footprint is None and target.projection is None and bit_view is None and field_path is None
                and isinstance(target.ty, (Array, Slice, HeapArray)) and not getattr(target.ty, "packed_bits", False)):
            count_footprint = indices
        if indices is not None and target.projection is not None and isinstance(target.projection.root.ty, Slice):
            physical = self.physical_array_row(target, state)
            if physical is None:
                root = target.projection.root
                span = root.ty.projection
                for owner in span.owners if span is not None else state.borrows.get(root.uid, (root,)):
                    state = self.write_actual_storage(state, owner)
                return state
            target = physical
        row = self.array_row_write_key(target) if indices is not None else None
        scalar = isinstance(target.ty, Scalar) and target.projection is None and field_path is None
        link = isinstance(target.ty, RecordLink) and target.projection is None and field_path is None
        if target.projection is not None:
            if mutation_steps is not None:
                mutation_steps = (*target.projection.steps, *mutation_steps)
            target = target.projection.root
            if row is None:
                indices = None
        if target.uid in self.attach_protected:
            self.nodes[0].token.error("cannot modify the read-only attachment source")
        if target.uid in self.owned_interface_targets:
            self.nodes[0].token.error(
                "cannot write through the interface field of an owning parameter")
        self.note_transition_write(target, indices, bit_view=bit_view, row=row, field_path=field_path,
                                   scalar=scalar, link=link)
        region = row
        if scalar:
            region, indices = (target, "scalar_write"), ZERO
        if link:
            region, indices = (target, "link_write"), ZERO
        if field_path is not None:
            region, indices = (target, "field_path", field_path), ZERO
        if region is None and indices is not None:
            region = (target, "indexed_write", bit_view)
        if region is not None:
            state = self.record_transition_region(state, region, indices)
        if target.uid in self.chain_fields:
            index = self.chain_fields[target.uid]
            param = self.functions[self.current][1].params[index]
            if param.attach is not None and Compiler.contains_record_links(target.ty):
                # A payload write leaves the topology alone, so an attachment
                # contract still holds: only the link storage is fixed.
                self.nodes[0].token.error("attachment contracts permit only the declared head link write")
            self.writes.add((index, ("@graph" if param.graph else "@chain",)))
            if index in self.chain_spans:
                root = self.chain_spans[index]
                state = self.havoc_chain_span_storage(state, root)
            for field in self.chain_payloads[index]:
                self.note_transition_write(field)
                state = state.touch(field.uid).forget_bit_values(field.uid).update(field, field.ty.domain)
            return replace(state, predicates={key: value for key, value in state.predicates.items()
                                              if key[0] in ("chain_link", "projected_link")})
        place = self.param_places.get(target.uid)
        if place is not None:
            index, names = place
            param = self.functions[self.current][1].params[index]
            steps = mutation_steps if mutation_steps is not None else field_path
            if array_footprint and (isinstance(param, (BorrowArray, BorrowRecord)) and isinstance(target.ty, Array)
                                    or isinstance(param, Slice) and isinstance(target.ty, Slice)):
                length = target.ty.length if isinstance(target.ty, Array) else target.ty.domain.hi
                ranges = array_footprint.ranges
                lo, hi = array_footprint.lo, array_footprint.hi
                residue, stride = array_footprint.congruence() if array_footprint.predicates else (0, 1)
                first = lo + (residue - lo) % stride if stride > 1 else lo
                if 1 < stride <= hi - first:
                    self.writes.add((index, names + (f"@stride:{first}:{hi + 1}:{stride}",)))
                else:
                    if len(ranges) > 8:
                        ranges = ((lo, hi),)
                    for first, last in ranges:
                        step = f"@index:{first}" if first == last else f"@range:{first}:{last + 1}"
                        suffix = (step,) if first == last or first > 0 or last + 1 < length else ()
                        self.writes.add((index, names + suffix))
            else:
                if steps and isinstance(param, (BorrowRecord, BorrowArray, Interface, Slice)):
                    projected = tuple(map(encode_array_mutation_step, steps))
                    if array_mutation_projection(target.ty, projected) is not None:
                        place = index, names + projected
                self.writes.add(place)
        preserved = self.untouched_counts(state, target, count_footprint, count_start, count_indices,
                                          field_path if field_path is not None else mutation_steps)
        state = self.invalidate_aliases(state.touch(target.uid).forget_bit_values(target.uid), target)
        if preserved:
            state = replace(state, predicates=state.predicates | preserved | {("storage_count_owner", target.uid): True})
        return state

    def untouched_counts(self, state, target, footprint, start, indices, steps):
        if ("storage_count_owner", target.uid) not in state.predicates:
            return {}
        retained = {}
        column = None
        if steps and isinstance(steps[0], (int, Domain)) and all(isinstance(name, str) for name in steps[1:]):
            footprint = Domain.point(steps[0]) if type(steps[0]) is int else steps[0]
            column = steps[1:]
        for key, fact in self.window_counts(state, target.uid):
            region = key[2]
            try:
                separate = (column and region.column and column[:len(region.column)] != region.column
                            and region.column[:len(column)] != column)
                if not separate and footprint is not None:
                    separate = not (footprint & fact.footprint)
                if not separate and start is not None and indices is not None:
                    anchor = Value(WIDE, "", linear=region.start)
                    difference = self.dependent_difference_domain(start, anchor, state)
                    if difference is not None:
                        terms, bias = self.combine_linear("-", start, anchor)
                        stride = gcd(*(scale for _, scale in terms))
                        if stride > 1:
                            difference = difference.with_congruence(bias, stride)
                        overlap = Domain.of(*((-hi, region.length - 1 - lo) for lo, hi in indices.ranges))
                        separate = not (difference & overlap)
                if separate:
                    retained[key] = fact
            except DomainComplexity:
                continue
        return retained

    def write_storage(self, state, sym, indices=None):
        if sym.bit_overlay and sym.uid in state.regions:
            positions = state.regions[sym.uid]
            for owner, _ in positions:
                state = self.write_actual_storage(state, owner)
            for owner, offset in positions:
                content = self.actual_array(state, owner)
                selected = self.array_indices(content, offset, ZERO)
                state = state.with_array(owner, content.havoc_indices(selected))
            return state
        for target in self.physical_places(sym, state):
            if not sym.bit_overlay:
                captured = (sym if indices is not None and sym.uid in self.alias_members
                            and sym.uid not in self.alias_types and target.projection is None
                            and isinstance(sym.ty, Array) and sym.ty.packed_bits else None)
                state = self.write_actual_storage(state, target, indices, bit_view=captured)
                continue
            if target.projection is not None:
                state = self.havoc_projected_storage(state, target)
                continue
            for leaf in self.storage_leaves(target):
                state = self.write_actual_storage(state, leaf)
                if isinstance(leaf.ty, Scalar):
                    state = state.update(leaf, leaf.ty.domain)
                elif isinstance(leaf.ty, Array) and leaf.ty.packed_bits:
                    state = state.with_bits(leaf, (0, 0))
                else:
                    state = state.with_array(leaf, self.actual_array(state, leaf).havoc())
        return state

    @staticmethod
    def affine_difference_bounds(left, right):
        if left is None or right is None or left.origin is not right.origin:
            return None
        scale = left.scale - right.scale
        shift = left.offset - right.offset
        domain = left.origin.domain
        if type(domain.lo) is not int or type(domain.hi) is not int:
            return None
        ends = (scale * domain.lo + shift, scale * domain.hi + shift)
        return min(ends), max(ends)

    def slice_ends_before(self, token, left, right, state):
        _, end, inclusive = left.relative_bounds
        start, _, _ = right.relative_bounds
        return self.prove_order(token, end, "<" if inclusive else "<=", start, state)

    def disjoint_spans(self, left, right):
        for a, a_offset in left.positions():
            for b, b_offset in right.positions():
                if a.uid == b.uid:
                    if (a.projection is not None and b.projection is not None
                            and a.projection.resolved and b.projection.resolved
                            and a.projection.captured is None and b.projection.captured is None
                            and a.projection.root.uid == b.projection.root.uid == a.uid
                            and a.uid not in self.alias_members):
                        try:
                            if not self.records_may_overlap(a, b, use_index_forms=left.host is a and right.host is b):
                                continue
                        except DomainComplexity:
                            pass  # An unknown overlap still uses memmove.
                    if (a != b or a_offset is None or b_offset is None
                            or not (a_offset.hi + left.length.domain.hi <= b_offset.lo
                                    or b_offset.hi + right.length.domain.hi <= a_offset.lo)):
                        return False
                elif a.uid in self.param_places and b.uid in self.param_places:
                    root_a = a.projection.root if a.projection is not None else a
                    root_b = b.projection.root if b.projection is not None else b
                    if isinstance(root_a.ty, Slice) or isinstance(root_b.ty, Slice) or self.may_alias(root_a, root_b):
                        return False
        return bool(left.positions() and right.positions())

    def write_span(self, state, span, indices, *, element_key=None, write_window=None):
        if not indices:
            return state
        selected = {}
        track_writes = getattr(self, "transition_frames", ())
        keyed_owners = {key[1] for key in state.predicates
                        if isinstance(key, tuple) and len(key) == 3 and key[0] == "array_element"}
        # Some writes use an absolute range for updating element domains. Keep
        # the original slice too so its offset can still prove disjointness.
        count_window = span if write_window is None else write_window
        count_start = self.span_start(count_window) if len(count_window.positions()) == 1 else None
        count_indices = indices if write_window is None else Domain.of((0, write_window.length.domain.hi - 1))
        for owner, offset in span.positions():
            if ((track_writes or state.bit_values or owner.uid in self.param_places
                 or owner.uid in keyed_owners or ("storage_count_owner", owner.uid) in state.predicates
                 or indices.predicates or offset is not None and offset.predicates
                 or ("array_window_post", owner.uid) in state.predicates
                 or ("array_untouched", owner.uid) in state.predicates)
                    and isinstance(owner.ty, (Array, Slice, HeapArray)) and isinstance(owner.ty.element, Scalar)
                    and not getattr(owner.ty, "packed_bits", False)
                    and (owner.projection is None or self.physical_array_row(owner, state) is not None)):
                contents = self.actual_array(state, owner)
                selected[owner] = selected.get(owner, Domain.of()) | self.array_indices(
                    contents, offset, indices, preserve_gaps=True)
        for owner in span.owners:
            key = ("array_window_post", owner.uid)
            fact = state.predicates.get(key)
            footprint = selected.get(owner)
            keep = False
            if fact is not None and state.epochs.get(owner.uid) is fact[0]:
                keep = footprint is not None and not (footprint & fact[-1])
                if not keep and element_key is not None and element_key[1] == owner.uid:
                    _, start, _, _, _, end, _, _ = fact
                    terms, offset = element_key[2]
                    keep = (terms == start[0] and offset < start[1]
                            or end is not None and terms == end[0] and offset >= end[1])
            preserved = {}
            if state.bit_values and footprint is not None and owner.projection is None:
                stride = storage_size(owner.ty.element) * 8
                touched = Domain.of(*((lo * stride, (hi + 1) * stride - 1) for lo, hi in footprint.ranges))
                preserved = self.untouched_bit_ranges(state, {owner.uid: touched})
                if footprint.predicates:
                    for key, value in state.bit_values.items():
                        uid, start, width, _ = key
                        if uid == owner.uid and not (
                                footprint & Domain.of((start // stride, (start + width - 1) // stride))):
                            preserved[key] = value
            untouched_key = ("array_untouched", owner.uid)
            untouched = None
            if footprint and owner.projection is None:
                previous = state.predicates.get(untouched_key)
                if previous is not None and state.epochs.get(owner.uid) is previous[0]:
                    untouched = previous[1], previous[2] | footprint
                elif footprint.predicates:
                    untouched = self.actual_array(state, owner), footprint
            untouched_elements = {}
            if footprint is not None and owner.uid in keyed_owners and owner.projection is None:
                for name, (_, value, previous) in self.current_array_element_facts(state, owner.uid):
                    if previous is not None and not (footprint & previous):
                        untouched_elements[name] = value, previous
            plain = (owner.projection is None and isinstance(owner.ty, (Array, Slice, HeapArray))
                     and not getattr(owner.ty, "packed_bits", False)
                     and ("storage_count_owner", owner.uid) in state.predicates)
            state = self.write_actual_storage(state, owner, footprint if track_writes else None,
                                              array_footprint=footprint if owner.projection is None else None,
                                              count_start=count_start if plain else None,
                                              count_indices=count_indices if plain else None)
            if untouched_elements:
                epoch = state.epochs[owner.uid]
                state = replace(state, predicates=state.predicates | {
                    name: (epoch, value, previous) for name, (value, previous) in untouched_elements.items()})
            if untouched is not None:
                fact_value = (state.epochs[owner.uid], *untouched)
                state = replace(state, predicates=state.predicates | {untouched_key: fact_value})
            if preserved:
                state = replace(state, bit_values=state.bit_values | preserved)
            if keep:
                state = replace(state, predicates=state.predicates | {key: (state.epochs[owner.uid], *fact[1:])})
        return state

    @staticmethod
    def child_place(sym, path):
        for name in path:
            if sym.captured_places is not None:
                field = dict(sym.ty.fields)[name]
                places = tuple(Compiler.child_place(owner, (name,)) if owner.uid else owner
                               for owner in sym.captured_places)
                sym = replace(sym, name=name, ty=unqualified(field), c_place=f"({sym.c}).{c_field(name)}",
                              readonly=sym.readonly or isinstance(field, ReadOnly), captured_places=places)
            elif sym.projection is not None:
                field = dict(sym.ty.fields)[name]
                sym = replace(sym, name=name, ty=unqualified(field), c_place=f"({sym.c}).{c_field(name)}",
                              readonly=sym.readonly or isinstance(field, ReadOnly),
                              projection=replace(sym.projection, steps=(*sym.projection.steps, name),
                                  captured=(tuple(Compiler.child_place(owner, (name,))
                                                  for owner in sym.projection.captured)
                                            if sym.projection.captured is not None else None)))
            else:
                sym = next(child for child in sym.fields if child.name == name)
        return sym

    def attachment_parameter(self, storage):
        index = self.chain_fields.get(storage.uid)
        if index is None:
            return None
        ty = self.functions[self.current][1].params[index]
        return index if ty.attach is not None else None

    def note_attachment(self, state, field, targets, token, head=False):
        if field.uid in self.attach_protected:
            token.error("cannot modify the read-only attachment source")
        index = self.chain_fields.get(field.uid)
        if index is not None:
            ty = self.functions[self.current][1].params[index]
            head |= self.attach_heads.get(field.uid) == index
            donor = ty.attach
            if not targets or any(self.param_places.get(target.uid) != (donor, ()) for target in targets):
                token.error("attachment must use the declared source chain parameter")
            names = ("@attach_graph",) if ty.graph else ("@attach",) if head else ("@attach_deep",)
            self.writes.add((index, names))
        return replace(self.record_link_write(state, field).touch(field.uid), predicates={})

    def attach_next(self, state, field, targets, token, weak):
        if field.projection is not None:
            if weak:
                targets |= self.link_targets(field, state)
            return self.store_link(state, field, targets, token)
        state = self.note_attachment(state, field, targets, token)
        if weak:
            targets |= state.borrows[field.uid]
        return self.update_link_targets(state, field, targets, token)

    def note_chain_prune(self, state, storage, token, effect="@prune"):
        if storage.uid in self.attach_protected:
            token.error("cannot modify the read-only attachment source")
        index = self.chain_fields.get(storage.uid)
        if index is not None:
            ty = self.functions[self.current][1].params[index]
            if not ty.prune:
                token.error("cannot infer a safe link-removal summary for this write")
            if ty.graph and effect != "@prune":
                token.error("a graph prune may only set a link to null")
            self.writes.add((index, (effect,)))
        return replace(self.record_link_write(state, storage).touch(storage.uid), predicates={})

    def prune_next(self, state, field, token, weak):
        targets = frozenset((self.null_symbol,))
        if field.projection is not None:
            if weak:
                targets |= self.link_targets(field, state)
            return self.store_link(state, field, targets, token)
        state = self.note_chain_prune(state, field, token)
        if weak:
            targets |= state.borrows[field.uid]
        return self.update_link_targets(state, field, targets, token)

    def chain_nodes(self, state, roots, token):
        seen = set()
        pending = list(roots)
        while pending:
            root = pending.pop()
            location = self.record_location(root)
            if not root.uid or location in seen:
                continue
            seen.add(location)
            self.spend(token)
            pending.extend(self.link_targets(self.child_place(root, ("next",)), state))
            yield root

    def havoc_chain_skips(self, state, roots, token):
        updates = []
        for root in self.chain_nodes(state, roots, token):
            field = self.child_place(root, ("next",))
            children = self.link_targets(field, state)
            suffix = frozenset(self.chain_nodes(state, children, token))
            updates.append((field, suffix | {self.null_symbol}))
        for field, targets in updates:
            if field.projection is not None:
                state = self.store_link(state, field, targets, token, skip=True)
            else:
                state = self.note_chain_prune(state, field, token, effect="@skip")
                state = self.update_link_targets(state, field, targets, token)
        return state

    def disjoint_projected_records(self, left, right, state, token):
        if isinstance(left, Span):
            return False
        pairs = self.projected_address_indices(left, right, captured=True)
        if not pairs:
            a, b = left.projection, right.projection
            if (a is None or b is None or a.absolute_index is None or b.absolute_index is None
                    or a.captured[0].projection.root.uid != b.captured[0].projection.root.uid):
                return False
            pairs = ((Value(WIDE, "", linear=a.absolute_index), Value(WIDE, "", linear=b.absolute_index)),)
        # Comparison refreshes captured versions before consulting current difference bounds.
        return any(self.prove_order(token, x, "!=", y, state)
                   or not self.refine_symbolic_congruence(ZERO, self.combine_linear("-", x, y))
                   for x, y in pairs)

    def attachment_descendants_overlap(self, left_root, right_root, left_children, right_children):
        # Keep descendants in both checks even when a summarized projection looks like a root.
        return (any(self.records_may_overlap(left, right_root) for left in left_children)
                or any(self.records_may_overlap(left_root, right) for right in right_children)
                or any(self.records_may_overlap(left, right) for left in left_children for right in right_children))

    def check_graph_attachment(self, receiver, source, element, roots, donors, state, token):
        separate_roots = len(roots) == len(donors) == 1 and self.disjoint_projected_records(receiver, source, state, token)
        if separate_roots:
            first_children = set(self.graph_nodes(self.record_link_targets(roots[0], state), state, token))
            second_children = set(self.graph_nodes(self.record_link_targets(donors[0], state), state, token))
            first = first_children | {roots[0]}
            overlap = self.attachment_descendants_overlap(roots[0], donors[0], first_children, second_children)
        else:
            first = set(self.graph_nodes(roots, state, token))
            if isinstance(receiver, Span) and receiver.length.domain.hi:
                first.update(self.span_element_places(receiver, Domain.of((0, receiver.length.domain.hi - 1)), state))
            second = set(self.graph_nodes(donors, state, token))
            overlap = any(self.records_may_overlap(left, right) for left in first for right in second)
        if overlap:
            token.error("attachment requires disjoint graphs")
        identity = donors[0].ty.identity
        sites = [(unqualified(ty), root.depth) for root in first for _, ty in root.ty.fields
                 if isinstance(unqualified(ty), RecordLink) and unqualified(ty).identity == identity]
        if isinstance(receiver, Span):
            sites.extend((unqualified(field), owner.depth) for owner in receiver.owners
                         for _, field in element.fields if isinstance(unqualified(field), RecordLink)
                         and unqualified(field).identity == identity)
        if any(target.depth > depth for _, depth in sites for target in donors):
            token.error("attached graph would outlive its target storage")
        if any(not field.readonly for field, _ in sites) and any(target.readonly for target in donors):
            # A direct forwarding call uses the donor from its own attachment
            # contract. The outer caller has already checked the underlying
            # storage; the formal's readonly qualification protects it only
            # while that contract is running.
            head = self.param_places.get(receiver.uid) if not isinstance(receiver, Span) else None
            donor = self.param_places.get(source.uid)
            params = self.functions[self.current][1].params
            forwarded = (tuple(roots) == (receiver,) and tuple(donors) == (source,)
                         and head is not None and not head[1] and donor is not None and not donor[1]
                         and params[head[0]].attach == donor[0])
            if not forwarded:
                token.error("attachment cannot create a writable link to an immutable record")

    def attach_graph_links(self, state, roots, targets, token, *, entry=None):
        nodes = tuple(self.graph_nodes(roots, state if entry is None else entry, token))
        identity = next(iter(targets)).ty.identity
        for root in nodes:
            for name, ty in root.ty.fields:
                if not root.readonly and isinstance(ty, RecordLink) and ty.identity == identity:
                    state = self.attach_next(state, self.child_place(root, (name,)), targets, token, weak=True)
        return state

    def prune_graph_links(self, state, roots, token, *, entry=None):
        nodes = tuple(self.graph_nodes(roots, state if entry is None else entry, token))
        for root in nodes:
            for name, ty in root.ty.fields:
                if isinstance(unqualified(ty), RecordLink) and not isinstance(ty, ReadOnly) and not root.readonly:
                    state = self.prune_next(state, self.child_place(root, (name,)), token, weak=True)
        return state

    def forward_graph_writes(self, call, parameter, donor=None):
        callee, args = call.args
        if callee.kind != "name" or any(self.has_effects(arg) for arg in args):
            return None
        name = callee.args[0]
        if name in self.functions:
            mutations = self.mutations.get(name)
            contract = self.functions[name][1]
        else:
            binding = self.imported_functions.get(name)
            mutations = binding.mutations if binding is not None and not binding.foreign else None
            contract = binding.contract if binding is not None else None
        if mutations is None:
            return None
        effects = []
        for index, names, _ in mutations:
            arg = args[index]
            if arg.kind != "name" or arg.args[0] != parameter or not names:
                return None
            if names[0] == "@graph_path_prune":
                effects.append(("cut", names[2], names[3:]))
            elif names[0] == "@graph_path_null":
                effects.append(("cut", "must", names[1:]))
            elif names[0] in ("@graph_path_edit", "@graph_path_attach"):
                donor_index = contract.params[index].attach
                if (donor is None or donor_index is None or args[donor_index].kind != "name"
                        or args[donor_index].args[0] != donor):
                    return None
                if names[0] == "@graph_path_edit":
                    effects.append((names[2], names[3], names[4:]))
                else:
                    effects.append(("attach", names[1], names[2:]))
            else:
                return None
        return effects

    def apply_graph_path_write(self, state, roots, route, required, token, *, entry=None, donors=None, allowed=None):
        # Ordered edits can traverse previously written links, but cannot
        # write into donor nodes introduced by a preceding possible attach.
        traversal = state if entry is None or allowed is not None else entry
        owners = set(roots)
        for depth, name in enumerate(route, 1):
            if self.null_symbol in owners:
                owners.remove(self.null_symbol)
                required = False
            if allowed is not None:
                permitted = {owner for owner in owners if self.record_location(owner) in allowed}
                required = required and owners == permitted
                owners = permitted
            if not owners:
                return state
            if len(owners) > 128 or any(node.projection is not None or node.chain_summary for node in owners):
                return (self.prune_graph_links(state, roots, token, entry=entry) if donors is None else
                        self.attach_graph_links(state, roots, donors, token, entry=entry))
            fields = [self.child_place(node, (name,)) for node in sorted(owners, key=lambda node: node.uid)]
            self.spend(token, len(fields))
            if depth == len(route):
                weak = not required or len(fields) != 1
                for field in fields:
                    state = (self.prune_next(state, field, token, weak=weak) if donors is None else
                             self.attach_next(state, field, donors, token, weak=weak))
                return state
            owners = {owner for field in fields for owner in self.link_targets(field, traversal)}
        return state

    def havoc_chain_payload(self, state, roots, token, graph=False):
        nodes = self.graph_nodes(roots, state, token) if graph else self.chain_nodes(state, roots, token)
        for root in nodes:
            for name, ty in root.ty.fields:
                if isinstance(ty, Scalar) and not root.readonly:
                    field = self.child_place(root, (name,))
                    if field.projection is not None:
                        state = self.store_value(state, field, Value(ty.domain, ""))
                    else:
                        state = self.write_actual_storage(state, field).update(field, ty.domain)
        return state

    def chain_nodes_at(self, state, roots, hops, token):
        roots = frozenset(root for root in roots if root.uid)
        for _ in range(hops):
            self.spend(token, len(roots))
            roots = frozenset(target for root in roots
                              for target in self.link_targets(self.child_place(root, ("next",)), state) if target.uid)
        return roots

    def apply_interface_write(self, state, source, names, post, token):
        """Resolve a write that goes through an interface reference.

        The callee records the path it can see -- the interface field, a ``*``
        step for the selection, then the fields of the selected record.  The
        caller resolves that step against the objects its own reference points
        to, so only those objects are invalidated.
        """
        split = names.index("*")
        field = self.child_place(source, names[:split])
        rest = names[split + 1:]
        owners = self.physical_places(field, state)
        if not owners and not isinstance(source, Span):
            # The reference selects nothing this analysis can name: fall back to
            # the whole argument rather than dropping the effect.
            return self.havoc_interface_place(state, source, None)
        places = [self.child_place(owner, rest) if rest else owner for owner in owners]
        effects = [(place, post | state.values.get(place.uid, place.ty.domain)
                    if len(places) > 1 and post is not None and isinstance(place.ty, Scalar) else post)
                   for place in places]
        for place, effect in effects:
            state = self.havoc_interface_place(state, place, effect)
            self.trace.append(f"  apply interface write id#{place.uid} at {token.line}:{token.col}: "
                              f"'{'.'.join(names)}'")
        return state

    def havoc_interface_place(self, state, place, post):
        state = self.write_storage(state, place)
        if isinstance(place.ty, Scalar):
            return state.update(place, post or place.ty.domain)
        if isinstance(place.ty, Array):
            return (state.with_bits(place, (0, 0)) if place.ty.packed_bits
                    else state.with_array(place, self.actual_array(state, place).havoc()))
        for leaf in self.storage_leaves(place):
            state = self.write_storage(state, leaf)
            if isinstance(leaf.ty, Scalar):
                state = state.update(leaf, leaf.ty.domain)
            elif isinstance(leaf.ty, Array):
                state = (state.with_bits(leaf, (0, 0)) if leaf.ty.packed_bits
                         else state.with_array(leaf, self.actual_array(state, leaf).havoc()))
        return state

    @staticmethod
    def interface_rebind_choices(names, arguments, args):
        if len(names) == 3:
            return [(arguments[int(names[2])], None)], False
        if len(names) == 6:
            guard_index = int(names[3])
            domain = arguments[guard_index].domain
            expected = int(names[4])
            arms = [(expected, names[2]), (1 - expected, names[5])]
            keep_old = False
        else:
            # A conjunction: the donor is selected only while every literal
            # holds.  A literal the guard domain already satisfies everywhere
            # carries no condition, and an arm the domain rules out selects
            # nothing at this call site.
            arms = []
            conditions = []
            for position in range(3, len(names), 2):
                guard_index = int(names[position])
                outcome = int(names[position + 1])
                enabled = Domain.point(outcome)
                domain = arguments[guard_index].domain
                if not domain & enabled:
                    return [], True
                if domain.within(enabled):
                    continue
                conditions.append((guard_index, outcome))
            if not conditions:
                return [(arguments[int(names[2])], None)], False
            # Keeping one literal is conservative: the arm still covers every
            # execution the full conjunction covers.
            return ([(arguments[int(names[2])], (args[conditions[0][0]], str(conditions[0][1])))],
                    True)
        choices = []
        for outcome, donor in arms:
            enabled = Domain.point(outcome)
            if domain & enabled:
                guard = None if domain.within(enabled) else (args[guard_index], str(outcome))
                choices.append((arguments[int(donor)], guard))
        return choices, keep_old

    def check_interface_rebind_holders(self, state, effects, arguments, token):
        indices = sorted({index for index, _ in effects})
        if len(indices) < 2:
            return
        active = {index for (index, _), (choices, _) in effects.items() if choices}
        fields = {}
        for index, names in effects:
            if names and names[0] == "@interface_rebind":
                fields.setdefault(index, set()).add(names[1])

        def same_storage(left, right):
            lefts, rights = self.physical_places(left, state), self.physical_places(right, state)
            return (len(lefts) == len(rights) == 1
                    and lefts[0].uid == rights[0].uid
                    and lefts[0].projection == rights[0].projection)

        holders = []
        for index in indices:
            holder = arguments[index]
            self.check_interface_rebind_holder(state, holder, token)
            origin = self.param_places.get(holder.uid)
            for previous_index, previous in holders:
                if self.records_may_overlap(holder, previous):
                    # Passing the same record to two receivers is safe when
                    # each receiver rebinds a different field.  With one field
                    # the per-parameter summaries cannot express their mutual
                    # order, so that case stays rejected.
                    if not same_storage(holder, previous):
                        token.error("multiple interface rebind receivers must be disjoint")
                    if fields.get(index, set()) & fields.get(previous_index, set()):
                        token.error("overlapping interface rebind receivers must be disjoint and write distinct fields")
                    continue
                if holder.borrowed and previous.borrowed:
                    other = self.param_places.get(previous.uid)
                    # Only a direct wrapper can pass on the separation obligation.
                    # Both effects must survive into its exported summary.
                    if (origin is None or other is None or origin[1] or other[1] or origin[0] == other[0]
                            or index not in active or previous_index not in active):
                        token.error("cannot prove interface rebind receivers are disjoint")
            holders.append((index, holder))

    def check_interface_rebind_holder(self, state, holder, token):
        if (holder.projection is not None or holder.link_access is not None
                or holder.heap_owner not in (0, holder.uid)
                or self.physical_places(holder, state) != (holder,)):
            token.error("interface rebinding requires a direct holder argument")
        self.require_live(holder, state, token)

    def apply_mutations(self, state, mutations, arguments, token, contract, args):
        pending = {}
        attachment_entry = replace(state)
        # A coarse payload havoc has to run before the ordered link edits: it
        # must still cover nodes a later edit detaches, which the caller may
        # hold an alias to.  Applying it afterwards would leave them unchanged.
        ordered = {index for index, names, _ in mutations
                   if names and names[0] in ("@graph_path_edit", "@graph_path_prune",
                                             "@graph_path_attach")}
        if ordered:
            first = [entry for entry in mutations
                     if entry[1] == ("@graph",) and entry[0] in ordered]
            mutations = first + [entry for entry in mutations if entry not in first]
        interface_effects = {(index, names): self.interface_rebind_choices(names, arguments, args)
                             for index, names, _ in mutations if names and names[0] == "@interface_rebind"}
        if interface_effects:
            self.check_interface_rebind_holders(state, interface_effects, arguments, token)
        # Alternate guarded arms of one field describe one may-set, so they are
        # applied together: applying them one by one would let an earlier arm
        # overwrite the seed the later arm still needs.
        rebind_arms = {}
        for (index, names), (choices, keep_old) in interface_effects.items():
            rebind_arms.setdefault((index, names[1]), []).append((choices, keep_old))
        rebind_done = set()
        # A write through the old interface target must happen before the
        # rebind that replaces that target.
        for index, names, post in mutations:
            if names and names[0] == "@interface_before":
                state.absorb(self.apply_interface_write(
                    state, arguments[index], names[1:], post, token))
        for index, names, post in mutations:
            source = arguments[index]
            if names and names[0] in ("@interface_before", "@interface_after"):
                continue
            if names and names[0] == "@interface_rebind":
                key = (index, names[1])
                if key in rebind_done:
                    continue
                rebind_done.add(key)
                choices, keep_old = [], False
                for arm_choices, arm_keeps_old in rebind_arms[key]:
                    choices.extend(arm_choices)
                    keep_old = keep_old or arm_keeps_old
                if choices:
                    state.absorb(self.apply_interface_rebind(state, source, names[1], choices, token,
                                                             keep_old=keep_old))
                continue
            if names == ("@prune",) and mutable_graph(contract.params[index]):
                if isinstance(source, Span):
                    state.absorb(self.prune_graph_span(state, source, token))
                else:
                    entry = attachment_entry if contract.params[index].attach is not None else state
                    roots = self.physical_places(source, entry)
                    state.absorb(self.prune_graph_links(state, roots, token, entry=entry))
                continue
            if names and names[0] == "@graph_root_null":
                roots = self.physical_places(source, state)
                if len(roots) == 1 and roots[0].projection is None and not roots[0].chain_summary:
                    field = self.child_place(roots[0], (names[1],))
                    state.absorb(self.prune_next(state, field, token, weak=False))
                continue
            if names and names[0] == "@graph_path_null":
                roots = self.physical_places(source, attachment_entry, allow_null=True)
                state.absorb(self.apply_graph_path_write(state, roots, names[1:], True, token,
                                                        entry=attachment_entry))
                continue
            if names and names[0] == "@graph_path_prune":
                roots = self.physical_places(source, state, allow_null=True)
                state.absorb(self.apply_graph_path_write(state, roots, names[3:], names[2] == "must", token))
                continue
            if names and names[0] == "@graph_path_attach":
                roots = self.physical_places(source, attachment_entry, allow_null=True)
                donor = arguments[contract.params[index].attach]
                targets = frozenset(self.physical_places(donor, attachment_entry))
                state.absorb(self.apply_graph_path_write(state, roots, names[2:], names[1] == "must", token,
                                                        entry=attachment_entry, donors=targets))
                continue
            if names and names[0] == "@graph_path_edit":
                roots = self.physical_places(source, attachment_entry, allow_null=True)
                allowed = frozenset(self.record_location(root)
                                    for root in self.graph_nodes(roots, attachment_entry, token))
                donor = arguments[contract.params[index].attach]
                targets = (frozenset(self.physical_places(donor, attachment_entry))
                           if names[2] == "attach" else None)
                state.absorb(self.apply_graph_path_write(state, roots, names[4:], names[3] == "must", token,
                                                        entry=attachment_entry, donors=targets, allowed=allowed))
                continue
            if names == ("@graph",):
                if isinstance(source, Span):
                    state.absorb(self.havoc_chain_span(state, source, token, graph=True))
                else:
                    roots = self.physical_places(source, state)
                    state.absorb(self.havoc_chain_payload(state, roots, token, graph=True))
                continue
            if names == ("@attach_graph",):
                donor = arguments[contract.params[index].attach]
                targets = frozenset(self.physical_places(donor, state))
                if isinstance(source, Span):
                    state.absorb(self.attach_graph_span(state, source, targets, token))
                else:
                    roots = self.physical_places(source, attachment_entry)
                    state.absorb(self.attach_graph_links(state, roots, targets, token, entry=attachment_entry))
                continue
            if names and names[0] in ("@attach", "@attach_deep", "@attach_at"):
                donor = arguments[contract.params[index].attach]
                targets = frozenset(self.physical_places(donor, state))
                distance = None if names == ("@attach_deep",) else int(names[1]) if len(names) == 2 else 1
                if isinstance(source, Span):
                    state.absorb(self.attach_chain_span(state, attachment_entry, source, targets, token, distance))
                else:
                    roots = self.physical_places(source, attachment_entry)
                    if distance is None:
                        roots = tuple(self.chain_nodes(attachment_entry, roots, token))
                    else:
                        roots = self.chain_nodes_at(attachment_entry, roots, distance - 1, token)
                    for root in roots:
                        state.absorb(self.attach_next(state, self.child_place(root, ("next",)), targets, token, weak=True))
                continue
            if names in (("@chain",), ("@prune",), ("@skip",)) and isinstance(source, Span):
                changed = (self.havoc_chain_span(state, source, token) if names == ("@chain",)
                           else self.prune_chain_span(state, source, token, skip=names == ("@skip",)))
                state.absorb(changed)
                continue
            if names in (("@chain",), ("@prune",), ("@skip",)):
                roots = self.physical_places(source, state)
                if names == ("@chain",):
                    state.absorb(self.havoc_chain_payload(state, roots, token))
                elif names == ("@skip",):
                    state.absorb(self.havoc_chain_skips(state, roots, token))
                else:
                    for root in tuple(self.chain_nodes(state, roots, token)):
                        field = self.child_place(root, ("next",))
                        state.absorb(self.prune_next(state, field, token, weak=True))
                continue
            if names and "*" in names:
                state.absorb(self.apply_interface_write(state, source, names, post, token))
                continue
            if isinstance(source, Span):
                if names and array_mutation_step(names[0]):
                    param = contract.params[index]
                    storage = param.array if isinstance(param, BorrowArray) else param
                    steps, leaf = array_mutation_projection(storage, names)
                    indices = steps[0] & Domain.of((0, source.length.domain.hi - 1))
                    if len(steps) == 1 and isinstance(source.element, Scalar):
                        if post is None:
                            state.absorb(self.write_span_contents(state, source, indices, source.element.domain))
                        else:
                            definite = indices & Domain.of((0, source.length.domain.lo - 1))
                            possible = indices - definite
                            if possible:
                                state.absorb(self.write_span_contents(state, source, possible, post, bulk=True))
                            if definite:
                                state.absorb(self.write_span_contents(state, source, definite, post, fill=True))
                        continue
                    for element in self.span_element_places(source, indices, state):
                        state.absorb(self.havoc_array_field(state, element, steps[1:], leaf))
                    continue
                if post is None:
                    changed = self.havoc_span_contents(state, source)
                elif isinstance(contract.params[index], Slice):
                    changed = self.fill_span_contents(state, source, post, None)
                    changed = self.remember_uniform_array_window(changed, source, post)
                else:
                    key = self.array_element_key(source, Value(ZERO, ""))
                    changed = self.write_span_contents(state, source, ZERO, post, element_key=key)
                    if key is not None:
                        changed = self.remember_array_element(state, changed, key, source, ZERO, post)
                state.absorb(changed)
                continue
            if not source.uid:
                continue
            element = next((i for i, step in enumerate(names) if array_mutation_step(step)), len(names))
            target = self.child_place(source, names[:element])
            if element < len(names):
                projection = array_mutation_projection(target.ty, names[element:])
                if projection is not None:
                    steps, leaf = projection
                    if len(steps) == 1 and isinstance(target.ty, Array) and isinstance(target.ty.element, Scalar):
                        owners = frozenset(self.physical_places(target, state))
                        if not owners:
                            continue
                        if all(owner.projection is None for owner in owners):
                            span = Span(leaf, Value(Domain.point(target.ty.length), ""), target.depth,
                                        borrowed_from=owners, origins=tuple((owner, ZERO) for owner in owners))
                            state.absorb(self.write_span_contents(
                                state, span, steps[0], post or leaf.domain, fill=post is not None))
                            continue
                    state.absorb(self.havoc_array_field(state, target, steps, leaf))
                    continue
            targets = self.physical_places(target, state, allow_null=True)
            if any(place.projection is not None for place in targets):
                state.absorb(self.write_storage(state, target))
                if isinstance(target.ty, Scalar):
                    state.absorb(self.store_value(state, target, Value(target.ty.domain, "")))
                elif isinstance(target.ty, Array) and target.ty.packed_bits:
                    state.absorb(self.store_bits(state, target, (0, 0)))
                elif isinstance(target.ty, Array):
                    state.absorb(self.store_array(state, target, self.storage_array(state, target).havoc()))
                continue
            state.absorb(self.write_storage(state, target))
            weak = len(targets) > 1
            for target in targets:
                if not target.uid:
                    continue
                effect = post
                if weak and effect is not None:
                    effect |= state.values.get(target.uid, target.ty.domain)
                previous_weak = False
                if target.uid in pending:
                    _, old, previous_weak = pending[target.uid]
                    if old is None or effect is None:
                        effect = None
                    elif weak or previous_weak:
                        # Different possible addresses give alternative writes.
                        effect |= old
                    else:
                        effect &= old
                pending[target.uid] = (target, effect, weak or previous_weak)
        # Each postcondition describes the state at return, including possible
        # aliases between parameters. Apply all facts after invalidating aliases.
        for target, post, _ in pending.values():
            if isinstance(target.ty, Scalar) and not target.chain_summary:
                state.absorb(state.update(target, post or target.ty.domain))
            elif isinstance(target.ty, Array) and target.ty.packed_bits:
                state.absorb(state.with_bits(target, (0, 0)))
            elif isinstance(target.ty, Array):
                state.absorb(state.with_array(target, self.actual_array(state, target).havoc()))
            self.trace.append(f"  apply mutation id#{target.uid} at {token.line}:{token.col}: {post or 'content'}")
        # A write through the interface target after the rebinds sees whichever
        # donor was selected by the last rebinding.
        for index, names, post in mutations:
            if names and names[0] == "@interface_after":
                state.absorb(self.apply_interface_write(
                    state, arguments[index], names[1:], post, token))
