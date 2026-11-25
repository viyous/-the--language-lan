"""Storage observations and derived theorems."""

from __future__ import annotations

from dataclasses import field, fields
from itertools import combinations

from semantic_ir import Proposition, Theorem
from storage_reasoning import KeyObservation, StorageCount, StorageExit, StorageObservation, value_counts, value_set_counts
from value_domain import Domain, DomainComplexity
from table_interface import encode_domain

from .typesys import Array, BorrowArray, BorrowRecord, Record, Scalar, Slice, unqualified, ZERO
from .summaries import small_summary_domain, storage_count_target, STORAGE_SUMMARY_LIMIT


class TheoremMixin:
    def storage_key_observations(self, path, uid, start, stop):
        window = Domain.of((start, stop - 1))
        entries = [(key, KeyObservation(footprint.affine(1, -start), values))
                   for key, (_, values, footprint) in self.current_array_element_facts(path, uid)
                   if footprint is not None and isinstance(values, Domain) and footprint.within(window)][:8]
        separated = []
        for a, b in combinations(range(len(entries)), 2):
            try:
                difference = self.array_key_difference(path, entries[a][0], entries[b][0])
                if difference is not None and not (difference & ZERO):
                    separated.append((a, b))
            except DomainComplexity:
                continue
        return tuple(item for _, item in entries), tuple(separated)

    def collect_storage_observations(self, name, contract):
        from composite_array import TreeArray, project_scalar_column

        if not self.return_states or len(self.return_states) > self.PATH_LIMIT:
            return
        observed = {uid for path in self.return_states for uid in path.arrays}
        observed.update(key[1] for path in self.return_states for key, fact in path.predicates.items()
                        if isinstance(key, tuple) and len(key) == 3 and key[0] == "array_element"
                        and fact[0] is path.epochs.get(key[1]))
        observations = []
        remaining_segments = 32_768
        for uid, (index, fields) in self.param_places.items():
            param = contract.params[index]
            owner = self.symbols[uid - 1]
            ty = owner.ty
            if owner.chain_summary or owner.link_access is not None:
                continue
            if not fields and isinstance(param, Slice) and param.domain.lo == param.domain.hi:
                length, element = param.domain.lo, param.element
            elif (isinstance(ty, Array) and not ty.packed_bits
                  and isinstance(param, (BorrowArray, BorrowRecord))
                  and not param.nullable and "*" not in fields):
                # Field paths may go through inline records only. A graph
                # summary can represent many objects at a single analysis ID.
                leaf_type = param.record if isinstance(param, BorrowRecord) else param.array
                for field in fields:
                    leaf_type = (unqualified(dict(leaf_type.fields).get(field))
                                 if isinstance(leaf_type, Record) else None)
                if leaf_type != ty:
                    continue
                length, element = ty.length, ty.element
            else:
                continue
            if not length or uid not in observed:
                continue
            if isinstance(element, Record):
                columns = self.storage_count_columns(element)
            elif isinstance(element, Scalar) and isinstance(element.domain, Domain):
                columns = ((),)
            else:
                continue
            if not columns:
                continue
            exits = {column: [] for column in columns}
            for path in self.return_states:
                contents = self.actual_array(path, owner)
                if contents.length != length:
                    break
                remaining_segments -= len(columns) * len(contents.segments)
                if remaining_segments < 0:
                    break
                if isinstance(contents, TreeArray):
                    for column, states in exits.items():
                        try:
                            projected = project_scalar_column(contents, column)
                        except DomainComplexity:
                            continue
                        if projected is not None:
                            states.append(StorageExit(projected, ()))
                    continue
                try:
                    keys, separated = self.storage_key_observations(path, uid, 0, length)
                except DomainComplexity:
                    keys, separated = (), ()
                exits[()].append(StorageExit(contents, keys, separated))
            else:
                for column, states in exits.items():
                    if len(states) == len(self.return_states):
                        route = fields + ((None, *column) if column else ())
                        observations.append(StorageObservation(index, tuple(dict.fromkeys(states)), route))
            if remaining_segments < 0:
                break
        if observations:
            self.semantic_module.storage_observations[name] = tuple(observations)

    def inferred_storage_counts(self, name, contract):
        facts = {}
        for observation in self.semantic_module.storage_observations.get(name, ()):
            if len(facts) >= STORAGE_SUMMARY_LIMIT:
                break
            path, index = observation.path, observation.parameter
            if len(path) > 64 or storage_count_target(contract, index, path) is None:
                continue
            length = observation.exits[0].contents.length
            singleton = ((Domain.point(n), bounds) for n, bounds in value_counts(observation))
            for selected, bounds in (*singleton, *value_set_counts(observation)):
                if small_summary_domain(selected):
                    facts[index, path, 0, length, selected] = bounds
                if len(facts) >= STORAGE_SUMMARY_LIMIT:
                    break
        if self.return_states and len(facts) < STORAGE_SUMMARY_LIMIT:
            # Forward facts established by callees on all reachable returns.
            for key in self.return_states[0].predicates:
                if key[:1] != ("storage_count",):
                    continue
                _, uid, region, selected = key
                place = self.param_places.get(uid)
                if place is None or region.start[0]:
                    continue
                index, path = place
                if region.column:
                    path = (*path, None, *region.column)
                target = storage_count_target(contract, index, path)
                start, length = region.start[1], region.length
                if (len(path) > 64 or target is None or not 0 <= start < start + length <= target[2]
                        or not selected.within(target[3].domain)):
                    continue
                alternatives = [exit_state.predicates.get(key) for exit_state in self.return_states]
                if not all(alternatives):
                    continue
                low = min(item.bounds[0] for item in alternatives)
                high = max(item.bounds[1] for item in alternatives)
                fact_key = index, path, start, length, selected
                old_low, old_high = facts.get(fact_key, (0, length))
                bounds = max(low, old_low), min(high, old_high)
                if bounds[0] <= bounds[1]:
                    facts[fact_key] = bounds
                if len(facts) >= STORAGE_SUMMARY_LIMIT:
                    break
        return tuple(StorageCount(*key, bounds) for key, bounds in facts.items())

    @staticmethod
    def storage_count_columns(record):
        pending, result = [((), record)], []
        while pending and len(result) < 32:
            prefix, ty = pending.pop()
            if isinstance(ty, Record):
                pending.extend(((*prefix, name), unqualified(field)) for name, field in reversed(ty.fields))
            elif isinstance(ty, Scalar) and isinstance(ty.domain, Domain):
                result.append(prefix)
        return tuple(result)

    def derive_storage_theorems(self):
        for name, observations in self.semantic_module.storage_observations.items():
            node, _ = self.functions[name]
            anchor = node.semantic_anchor
            location = node.token.line, node.token.col
            for observation in observations:
                conclusions = tuple(value_counts(observation))
                set_conclusions = tuple(value_set_counts(observation))
                if not conclusions and not set_conclusions:
                    continue
                evidence = len(self.storage_evidence)
                self.storage_evidence.append((name, anchor, observation))
                place = (observation.parameter, observation.path) if observation.path else observation.parameter
                label = f"param#{observation.parameter}" + "".join(
                    "[*]" if field is None else f".{field}" for field in observation.path)
                for value, (lo, hi) in conclusions:
                    self.source_theorems.append(Theorem(
                        Proposition("storage_value_count", (place, value, lo, hi)),
                        anchor, "storage_count", location, function=name, evidence=evidence))
                    self.trace.append(f"  theorem {name} {label} "
                                      f"count({value}) in [{lo}, {hi}]")
                for selected, (lo, hi) in set_conclusions:
                    self.source_theorems.append(Theorem(
                        Proposition("storage_set_count", (place, encode_domain(selected), lo, hi)),
                        anchor, "storage_set_count", location, function=name, evidence=evidence))
