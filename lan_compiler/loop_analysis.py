"""Fixed-point analysis of loops."""

from __future__ import annotations

from dataclasses import field, replace

from difference_domain import Zone
from value_domain import Domain

from .errors import ProofBudgetError
from .syntax import Node
from .typesys import (Array, BitField, BorrowRecord, BorrowScalar, HeapArray, Interface, Scalar, Slice,
                      Symbol, ValueOrigin, WIDE)
from .domain_utils import integer_states
from .state import LoopEdges, PathState
from .c_emit import indent


class LoopAnalysisMixin:
    def analyze_loop(self, stmt, paths, *, transition_claims=(), post_guards=(), counted=None):
        condition, body = stmt.args[:2]
        literal_true = condition.kind == "boolean" and condition.args[0] is True
        invariants, measure = stmt.args[2:4] if len(stmt.args) > 2 else ((), None)
        claims = self.loop_value_claims(invariants, "invariant")
        scalar_claims = {sym.uid: domain for sym, domain, _, index in claims.values()
                         if index is None and sym.uid not in self.alias_members and sym.link_access is None}
        accelerated = False
        suffix = f"{stmt.token.line}_{stmt.token.col}"
        head_label, back_label, exit_label = (f"lan_loop_{part}_{suffix}" for part in ("head", "back", "exit"))
        self.check_loop_claims(claims, paths, "invariant", "entry")
        recurrence_bounds = self.loop_induction_bounds(stmt, paths, counted)

        def restrict_recurrence(states, *, header=False):
            for uid, (body_bound, header_bound) in recurrence_bounds.items():
                sym = self.symbols[uid - 1]
                domain = header_bound if header else body_bound
                states = [narrowed for state in states if (narrowed := state.narrow(sym, domain)) is not None]
            return states

        snapshot = None
        if measure is not None:
            if self.has_effects(measure):
                measure.token.error("loop measure must be side-effect free")
            snapshot = Symbol("loop_measure", len(self.symbols) + 1, Scalar(WIDE, 64, True))
            self.symbols.append(snapshot)

        def measure_value(path):
            value = self.expr(measure, path)
            if value.floating or value.boolean or value.character or value.enum:
                measure.token.error("loop measure must be an integer")
            return value

        ids = tuple(sorted(paths[0].values))
        storage_ids = frozenset(paths[0].arrays)
        bit_scope = len(self.symbols)
        slice_claims = {sym.uid: (sym, token) for sym, _, token, index in transition_claims
                        if isinstance(sym.ty, Slice) and isinstance(index, (tuple, Node))}
        for sym, _, token, index in transition_claims:
            field = self.fixed_projected_field(sym, symbolic=True) if index is None else None
            root = field[0] if field is not None else sym.projection.root if sym.projection is not None else None
            if root is not None and isinstance(root.ty, Slice):
                slice_claims[root.uid] = root, token
            if sym.link_access is not None:
                for place in self.loop_link_projections(sym):
                    root = place.projection.root
                    if isinstance(root.ty, Slice):
                        slice_claims[root.uid] = root, token
        slice_choices = slice_claims.keys()
        for sym, token in slice_claims.values():
            expanded = [candidate for path in paths for candidate in self.split_slice_windows(sym, token, path)]
            if expanded and len(expanded) <= self.PATH_LIMIT:
                paths = expanded
        field_indices = self.loop_field_indices(
            (sym for sym, _, _, index in transition_claims
             if (sym.link_access is not None
                 or index is None and self.fixed_projected_field(sym, symbolic=True) is not None
                 or isinstance(index, (tuple, Node)) and sym.projection is not None)),
            (endpoint for _, _, _, index in transition_claims if isinstance(index, Node)
             for endpoint in index.args[1:3]))
        if counted is not None and counted.limit <= 8:
            counters = [sym for scope in self.scope for sym in scope.values()
                        if sym.name.startswith("$interation_turn_")
                        and all(sym.uid in path.values and path.values[sym.uid].lo == path.values[sym.uid].hi
                                and path.values[sym.uid].hi <= 8 for path in paths)]
            if len(counters) <= 2:
                field_indices.update((counter.uid, counter) for counter in counters)
        if condition.kind == "name":
            control = self.lookup(condition)
            if (isinstance(control.ty, Scalar) and control.ty.boolean
                    and control.uid in ids and control.uid not in self.alias_members):
                # Keep the write facts from a completed boolean-controlled
                # iteration separate from the entry state that still runs it.
                field_indices[control.uid] = control
        # A small number of boolean selectors preserves correlations between
        # distinct writes and their respective exit obligations.
        selectors = {}
        for guard in post_guards:
            if guard is not None and guard.kind == "unary" and guard.args[0] == "!":
                guard = guard.args[1]
            if guard is not None and guard.kind == "name":
                sym = self.lookup(guard)
                if (isinstance(sym.ty, Scalar) and sym.ty.boolean
                        and sym.uid in ids and sym.uid not in self.alias_members):
                    selectors[sym.uid] = sym
        if len(selectors) <= 2:
            field_indices.update(selectors)
        paths = self.split_loop_field_indices(field_indices, paths)
        storage_choices = tuple(sorted(uid for uid in paths[0].borrows
                                       if uid in slice_choices
                                       or uid in self.alias_types or (uid in self.alias_members
                                           and (isinstance(self.symbols[uid - 1].ty, Interface)
                                                or isinstance(self.symbols[uid - 1].ty, Array)
                                                and self.symbols[uid - 1].ty.packed_bits))))
        label = f"loop@{stmt.token.line}:{stmt.token.col}"
        thresholds = self.loop_thresholds(condition)
        if counted is not None:
            thresholds.update((counted.limit - 1, counted.limit))
        forgotten = {}
        entry_relations = {key for path in paths for key in path.predicates
                           if isinstance(key, tuple) and key
                           and key[0] in ("dependent_upper", "dependent_excluded")}
        entry_origins = {form.origin for path in paths for form in path.forms.values()} if entry_relations else set()
        entry_elements = {key for path in paths for key in path.predicates
                          if isinstance(key, tuple) and len(key) == 3 and key[0] == "array_element"
                          and 0 < key[1] <= bit_scope}
        entry_counts = {key for path in paths for key in path.predicates
                        if key[:1] == ("storage_count",) and key[1] in storage_ids}
        element_origins = set()
        pending = [key[2] for key in entry_elements] + [key[2].start for key in entry_counts]
        while pending:
            term = pending.pop()
            if isinstance(term, ValueOrigin):
                element_origins.add(term)
            elif isinstance(term, (tuple, frozenset)):
                pending.extend(term)
        element_bindings = {
            uid: (paths[0].epochs.get(uid), form, paths[0].linear.get(uid))
            for uid, form in paths[0].forms.items()
            if form.origin in element_origins
            and all(path.forms.get(uid) == form
                    and path.epochs.get(uid) is paths[0].epochs.get(uid)
                    and path.linear.get(uid) == paths[0].linear.get(uid) for path in paths)
        }
        entry_properties = {key for path in paths for key in path.predicates
                            if isinstance(key, tuple) and key[0] == "property_value"}
        entry_epochs = {}
        if entry_properties:
            storage = set(ids) | storage_ids | set().union(*(path.epochs for path in paths))
            for path in paths:
                for uid in storage:
                    entry_epochs.setdefault(uid, set()).add(path.epochs.get(uid))

        def loop_relations(path):
            facts = {key: path.predicates[key] for key in entry_relations if key in path.predicates}
            forms = {uid: form for uid, form in path.forms.items() if form.origin in entry_origins} if facts else {}
            return forms, facts

        def loop_properties(path):
            return {key: path.predicates[key] for key in entry_properties
                    if key in path.predicates and path.epochs.get(key[1]) is key[2]}

        def loop_elements(path):
            return {key: path.predicates[key] for key in entry_elements
                    if key in path.predicates and path.predicates[key][0] is path.epochs.get(key[1])}

        def element_values(path):
            return {key: fact[1:] for key, fact in loop_elements(path).items()}

        def loop_counts(path):
            return {key: path.predicates[key] for key in entry_counts if key in path.predicates}

        def element_forms(path):
            return {uid: form for uid, (epoch, form, _) in element_bindings.items()
                    if path.epochs.get(uid) is epoch}

        def property_versions(path):
            # Track loss of entry versions without comparing fresh epochs from later rounds.
            return {uid: path.epochs.get(uid) for uid, epochs in entry_epochs.items()
                    if path.epochs.get(uid) in epochs}

        heap_lengths = tuple(uid for uid in ids if isinstance(self.symbols[uid - 1].ty, HeapArray))
        length_origins = {path.forms[uid].origin for path in paths for uid in heap_lengths if uid in path.forms}
        pending = [path.linear[uid] for path in paths for uid in heap_lengths if uid in path.linear]
        while pending:
            part = pending.pop()
            if isinstance(part, ValueOrigin):
                length_origins.add(part)
            elif isinstance(part, (tuple, frozenset)):
                pending.extend(part)

        def window_forms(path):
            origins = {start.affine.origin for start in path.starts.values() if start.affine is not None}
            return {uid: form for uid, form in path.forms.items() if form.origin in origins or form.origin in length_origins}

        element_aliases = {sym.uid: sym for sym, _, _, index in transition_claims
                           if index is None and isinstance(self.alias_types.get(sym.uid), BorrowScalar)}

        for uid in sorted(element_aliases):
            expanded = []
            for path in paths:
                targets = path.borrows.get(uid, ())
                if (1 < len(targets) <= 8 and all(
                        not target.uid or (target.projection is None and target.link_access is None
                                           and not target.chain_summary)
                        or self.projected_scalar_region(target) is not None
                        or self.fixed_projected_field(target) is not None for target in targets)):
                    expanded.extend(replace(path, borrows=path.borrows | {uid: frozenset((target,))})
                                    for target in sorted(targets, key=lambda target: target.c))
                else:
                    expanded.append(path)
            if len(expanded) <= self.PATH_LIMIT:
                paths = expanded

        def exact_slice_window(state, uid):
            positions = state.regions.get(uid, ())
            if len(positions) == 1:
                owner, offset = positions[0]
                if offset is not None and offset.lo == offset.hi:
                    return owner.uid, offset.lo
            return None

        row_aliases = {self.alias_members[sym.uid][0]: sym for sym, _, _, index in transition_claims
                       if sym.uid in self.alias_members and isinstance(index, (tuple, Node, BitField))}
        tracked_windows = {uid: set() for uid in sorted(set(slice_choices) | element_aliases.keys() | row_aliases.keys())}

        tracked_indices = {uid: set() for uid in sorted(field_indices)}

        linked_targets = {sym: set() for sym, _, _, _ in transition_claims if sym.link_access is not None}

        def partition(states):
            groups = {}
            for state in states:
                owners = tuple((uid, tuple(sorted(owner.uid for owner in state.borrows[uid])))
                               for uid in storage_choices)
                windows = []
                for uid, known in tracked_windows.items():
                    if uid in element_aliases:
                        targets = self.physical_places(element_aliases[uid], state, allow_null=True)
                        window = None
                        if len(targets) == 1:
                            region = self.projected_scalar_region(targets[0])
                            field = self.fixed_projected_field(targets[0])
                            if region is not None:
                                window = region[0].uid, region[1]
                            elif field is not None:
                                window = field[0].uid, field[1]
                    elif uid in row_aliases:
                        target = self.physical_array_row(row_aliases[uid], state)
                        row = self.array_row_write_key(target) if target is not None else None
                        window = (row[0].uid, row[2]) if row is not None else None
                    else:
                        window = exact_slice_window(state, uid)
                    if window is not None and len(known) < 8:
                        known.add(window)
                    windows.append(window if window in known else None)
                indices = []
                for uid, known in tracked_indices.items():
                    domain = state.values[uid]
                    value = domain.lo if domain.lo == domain.hi else None
                    if value is not None and len(known) < 8:
                        known.add(value)
                    indices.append(value if value in known else None)
                links = []
                for sym, known in linked_targets.items():
                    targets = self.physical_places(sym, state, allow_null=True)
                    target = targets[0].uid if len(targets) == 1 else None
                    if target is not None and len(known) < 8:
                        known.add(target)
                    links.append(target if target in known else None)
                groups.setdefault((owners, tuple(windows), tuple(indices), tuple(links)), []).append(state)
            self.budget_paths(list(groups), stmt.token)
            return {key: PathState.join(group) for key, group in groups.items()}

        def same(left, right):
            return (left.values == right.values and left.zone == right.zone and left.bits == right.bits
                    and left.borrows == right.borrows and left.heap == right.heap
                    and left.arrays == right.arrays and left.regions == right.regions and left.starts == right.starts and left.linear == right.linear
                    and left.bit_values == right.bit_values and left.row_writes == right.row_writes
                    and left.fresh_links == right.fresh_links
                    and left.epochs.keys() == right.epochs.keys()
                    and window_forms(left) == window_forms(right)
                    and loop_relations(left) == loop_relations(right)
                    and loop_properties(left) == loop_properties(right)
                    and element_values(left) == element_values(right)
                    and loop_counts(left) == loop_counts(right)
                    and element_forms(left) == element_forms(right)
                    and property_versions(left) == property_versions(right))

        def forget_induction_details(states):
            counter = counted.induction
            result = {}
            for key, path in states.items():
                domain = path.values[counter.uid]
                result[key] = replace(
                    path, values=path.values | {counter.uid: Domain.of((domain.lo, domain.hi)) & counter.ty.domain},
                    forms={uid: form for uid, form in path.forms.items() if uid != counter.uid},
                    linear={uid: form for uid, form in path.linear.items() if uid != counter.uid},
                    bits={uid: bits for uid, bits in path.bits.items() if uid != counter.uid})
            return result

        heads = partition(paths)
        for iteration in range(1, self.LOOP_LIMIT + 1):
            widen_induction = (counted is not None and counted.induction is not None
                               and iteration >= self.LOOP_WIDEN_AFTER)
            if widen_induction:
                heads = forget_induction_details(heads)
            self.spend(stmt.token)
            symbol_count, trace_count = len(self.symbols), len(self.trace)
            exit_facts = self.loop_exit_facts.copy()
            recurrences = self.loop_recurrences.copy()
            helpers, rhs_codes = self.helpers.copy(), self.rhs_codes.copy()
            keyed_access_types = self.keyed_access_types.copy()
            alias_types, alias_members = self.alias_types.copy(), self.alias_members.copy()
            calls = self.calls[self.current].copy()
            writes, returns = self.writes.copy(), len(self.return_states)
            rebind_order = self.rebind_order.copy()
            before_writes = self.interface_before_writes.copy()
            after_writes = self.interface_after_writes.copy()
            record_returns = len(self.record_return_fields)
            array_returns = len(self.array_return_contents)
            guard_returns = len(self.guard_return_observations)
            scalar_returns = len(self.scalar_return_cases)
            result_exits = len(self.result_exit_domains)
            idempotent_returns = len(self.idempotent_return_domains)
            inferred = len(self.inferred_returns) if self.inferred_returns is not None else 0
            inferred_slices = len(self.inferred_slice_paths)
            yes, no = [], []
            continuing_test = error_test = None
            outer_replay = getattr(self, "optimization_replay", None)
            header_replay = self.loop_replay(condition, body, tuple(heads.values()))
            property_context = None
            if entry_properties:
                proof_heads = [replace(head, predicates=loop_properties(head)) for head in heads.values()]
                property_context = self.shared_property_context(condition, proof_heads)
            for head in heads.values():
                state = PathState.with_independent_values(head.values, (label,), head.zone, head.borrows, head.heap, head.arrays, head.regions)
                state = replace(state, bits=head.bits, bit_values=head.bit_values, epochs=head.epochs).with_zone(head.zone)
                if state is None:
                    state = PathState.with_independent_values(head.values, (label,),
                              Zone.independent(integer_states(head.values)), head.borrows, head.heap, head.arrays, head.regions)
                    state = replace(state, epochs=head.epochs)
                forms, predicates = loop_relations(head)
                stable_elements = element_forms(head)
                counts = loop_counts(head)
                linear = {uid: value for uid, value in head.linear.items() if uid not in stable_elements}
                linear.update({uid: element_bindings[uid][2] for uid in stable_elements
                               if element_bindings[uid][2] is not None})
                state = replace(state, starts=head.starts, linear=linear, row_writes=head.row_writes,
                                fresh_links=head.fresh_links,
                                forms=state.forms | forms | window_forms(head) | stable_elements,
                                predicates=predicates | loop_properties(head) | loop_elements(head) | counts
                                | {("storage_count_owner", key[1]): True for key in counts})
                for offset_snapshot in self.result_offset_snapshots.values():
                    if offset_snapshot.uid in head.forms:
                        state = state.bind_equal(offset_snapshot, head.forms[offset_snapshot.uid])
                self.states("loop head", [state])
                previous_context = self.property_codegen_context
                self.property_codegen_context = property_context
                self.optimization_replay = header_replay
                try:
                    hit, miss, test = self.condition(condition, state, require_boolean=True)
                finally:
                    self.property_codegen_context = previous_context
                    self.optimization_replay = outer_replay
                if hit or miss:
                    continuing_test = test
                elif error_test is None or self.propagation_code_rank(test) > self.propagation_code_rank(error_test):
                    error_test = test
                yes.extend(hit)
                no.extend(miss)
            test = continuing_test or error_test
            self.budget_paths(yes + no, stmt.token)
            body_paths = [path.guard("condition=true") for path in yes]
            if counted is not None and counted.induction is not None:
                counter, bounds = counted.induction, counted.body_domain
                body_paths = [narrowed for path in body_paths if (narrowed := path.narrow(counter, bounds)) is not None]
            if recurrence_bounds:
                body_paths = restrict_recurrence(body_paths)
            body_replay = header_replay
            if body_replay is None and body_paths:
                # An opaque condition may have effects. Start the body from the
                # checked true states rather than replaying from before the test.
                body_replay = self.loop_replay(None, body, body_paths)
            measure_before = measure_after = None
            if snapshot is not None:
                for path in body_paths:
                    value = measure_value(path)
                    if value.domain.lo < 0:
                        measure.token.error("cannot prove loop measure is nonnegative on entry")
                    path.absorb(path.update(snapshot, value.domain, value.affine, value.difference))
                    measure_before = value.code
            self.states("loop body", body_paths)
            edges = LoopEdges(len(self.scope), [], back_label, symbol_cutoff=bit_scope)
            for path in body_paths:
                previous = frozenset(uid for uid, statuses in path.heap.items()
                                     if uid > bit_scope and "Alive" in statuses)
                edges.previous_allocations |= previous
                path.predicates = path.predicates | {(edges.entry_alias_marker, "previous"): previous} | {
                    (edges.entry_alias_marker, uid): path.epochs.get(uid)
                    for uid in path.borrows if isinstance(self.alias_types.get(uid), BorrowRecord)
                }
            outer_pinned = self.current_pinned_baseline()
            loop_ep, loop_org = PathState.baseline(paths)
            body_ep, body_org = PathState.baseline(body_paths)
            self.pinned_baseline = (outer_pinned[0] | loop_ep | body_ep,
                                    outer_pinned[1] | loop_org | body_org)
            self.loops.append(edges)
            self.optimization_replay = body_replay
            try:
                normal, code = self.block(body, body_paths)
            finally:
                self.optimization_replay = outer_replay
                self.pinned_baseline = outer_pinned
                self.loops.pop()
            back = normal + edges.continues
            self.check_loop_cursors(edges, back, stmt.token)
            latch = []
            if counted is not None and back:
                for update in counted.updates:
                    back, emitted = self.assign(update, back)
                    latch.extend(emitted)
            self.check_loop_claims(claims, back, "invariant", "back edge")
            if snapshot is not None:
                for path in back:
                    before = self.storage_value(snapshot, path)
                    after = measure_value(path)
                    measure_after = after.code
                    if not self.prove_order(measure.token, after, "<", before, path):
                        measure.token.error("cannot prove loop measure strictly decreases on back edge")
                    remaining = path.values.keys() - {snapshot.uid}
                    path.absorb(replace(path,
                        values={uid: path.values[uid] for uid in remaining},
                        forms={uid: form for uid, form in path.forms.items() if uid in remaining},
                        zone=path.zone.project(remaining)))
            self.budget_paths(back, stmt.token)
            recurrent = restrict_recurrence(back, header=True) if recurrence_bounds else back
            joined = partition([*heads.values(), *recurrent])
            if widen_induction:
                # A range cut can change the symbolic origin or congruence
                # even when the numeric invariant is stable. Forget those
                # facts on both sides of the fixed-point comparison.
                joined = forget_induction_details(joined)
            if joined.keys() == heads.keys() and all(same(heads[key], value) for key, value in joined.items()):
                if counted is None and not literal_true and yes:
                    if not no:
                        condition.token.error("while condition is always true")
                    if self.loop_back_cannot_exit(condition, back, no):
                        condition.token.error("while cannot reach a false condition from its legal back-edge states")
                exits = [p.guard("condition=false") for p in no]
                self.budget_paths(exits, stmt.token)
                exits = [p.project(ids, storage_ids, bit_scope) for p in exits]
                self.trace.append(f"  loop fixed point after {iteration} analysis round(s)")
                self.trace_state_claims(invariants, "verified invariant")
                if accelerated:
                    self.trace.append("  loop invariant accelerated scalar widening")
                for uid, (body_bound, head_bound) in recurrence_bounds.items():
                    name = self.symbols[uid - 1].name
                    self.trace.append(f"  loop recurrence {name}: body entry {body_bound}; header {head_bound}")
                self.states("loop back edge tail", [p.project(ids, storage_ids, bit_scope) for p in normal])
                self.states("loop back edge continue tail", [p.project(ids, storage_ids, bit_scope) for p in edges.continues])
                self.states("loop exit condition return", exits)
                failed_test = exit_label if no else f"lan_loop_unreachable_{suffix}"
                lowered = [f"{head_label}: ;", *self.oracle_checks(claims, paths[0])]
                if not literal_true:
                    lowered.append(f"if (!({test})) goto {failed_test};")
                if measure_before is not None:
                    metric = f"lan_loop_measure_{suffix}"
                    check = self.helper("oracle_check", None)
                    lowered.extend([f"lan_int {metric} = {measure_before};", f"{check}({metric} >= 0);"])
                lowered.extend(["{", *indent(code), "}"])
                if back:
                    if edges.continues:
                        lowered.append(f"{back_label}: ;")
                    lowered.extend(latch)
                    if measure_after is not None:
                        lowered.append(f"{check}(({measure_after}) < {metric});")
                    lowered.append(f"goto {head_label};")
                if not no and not literal_true:
                    lowered.extend([f"{failed_test}: ;", "__builtin_unreachable();"])
                if exits:
                    lowered.append(f"{exit_label}: ;")
                labels = [head_label]
                if edges.continues:
                    labels.append(back_label)
                if not no and not literal_true:
                    labels.append(failed_test)
                if exits:
                    labels.append(exit_label)
                self.trace.append(f"  lower {label}: {', '.join(labels)}")
                return exits, ["{", *indent(lowered), "}"]

            del self.symbols[symbol_count:]
            del self.trace[trace_count:]
            self.loop_exit_facts = exit_facts
            self.loop_recurrences = recurrences
            self.helpers, self.rhs_codes = helpers, rhs_codes
            self.keyed_access_types = keyed_access_types
            self.alias_types, self.alias_members = alias_types, alias_members
            self.calls[self.current] = calls
            self.writes = writes
            self.rebind_order = rebind_order
            self.interface_before_writes = before_writes
            self.interface_after_writes = after_writes
            del self.return_states[returns:]
            del self.record_return_fields[record_returns:]
            del self.array_return_contents[array_returns:]
            del self.guard_return_observations[guard_returns:]
            del self.scalar_return_cases[scalar_returns:]
            del self.result_exit_domains[result_exits:]
            del self.idempotent_return_domains[idempotent_returns:]
            if self.inferred_returns is not None:
                del self.inferred_returns[inferred:]
                del self.inferred_slice_paths[inferred_slices:]
            ordinary_widening = iteration >= self.LOOP_WIDEN_AFTER
            if ordinary_widening or scalar_claims and iteration >= 2:
                for key, state in list(joined.items()):
                    if key not in heads:
                        continue
                    head = heads[key]
                    values = self.widen_loop_head(head.values, state.values, thresholds)
                    if counted is not None and counted.induction is not None:
                        values[counted.induction.uid] &= counted.head_domain
                    if not ordinary_widening:
                        values = state.values | {uid: values[uid] for uid in scalar_claims}
                    for uid, domain in scalar_claims.items():
                        values[uid] &= domain
                    if ordinary_widening:
                        for uid, (_, domain) in recurrence_bounds.items():
                            values[uid] &= domain
                    if not ordinary_widening and values == state.values:
                        continue
                    accelerated |= not ordinary_widening
                    self.spend(stmt.token, len(state.zone.ids) ** 3)
                    zone = head.zone.widen(state.zone, values, forgotten.setdefault(key, set()))
                    joined[key] = replace(state, values=values, zone=zone)
            # A loop body can leave values for its own heap nodes behind so
            # links through them stay readable.  They are not loop-carried
            # variables: project the joined state back onto the entry ids
            # before it becomes the next head, otherwise a later iteration
            # widens a uid whose symbol table entry has already been rolled
            # back.
            heads = {key: state.project(ids, storage_ids, bit_scope)
                     for key, state in joined.items()}
        token = stmt.token
        raise ProofBudgetError(f"{token.line}:{token.col}: loop analysis did not converge; safety could not be proved")
