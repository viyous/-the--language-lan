"""Physical storage values and recursive link targets."""

from __future__ import annotations

from dataclasses import field, replace

from difference_domain import Difference
from size_change import strongly_connected_components

from .typesys import BorrowRecord, RecordLink, Value, ZERO
from .domain_utils import domain_union
from .interface_types import mutable_graph


class StorageLinkMixin:
    def physical_places(self, sym, state, allow_null=False):
        if sym.captured_places is not None:
            return tuple(place for place in sym.captured_places if place.uid or allow_null)
        if sym.link_access is not None:
            parent, name = sym.link_access
            owners = (self.link_targets(parent, state) if isinstance(parent.ty, RecordLink)
                      else self.physical_places(parent, state, allow_null=True))
            return tuple(self.child_place(owner, (name,)) if owner.uid else owner for owner in owners
                         if owner.uid or allow_null)
        if sym.uid not in self.alias_members:
            return (sym,)
        root, names = self.alias_members[sym.uid]
        result = []
        for owner in sorted(state.borrows[root], key=lambda item: item.uid):
            if owner.uid == 0:
                if allow_null:
                    result.append(owner)
                continue
            result.append(self.child_place(owner, names))
        return tuple(result)

    def storage_value(self, sym, state):
        targets = self.physical_places(sym, state)
        domain = domain_union(state.values.get(target.uid, target.ty.domain) for target in targets)
        form = state.forms.get(targets[0].uid) if targets else None
        if any(state.forms.get(target.uid) != form for target in targets):
            form = None
        difference = Difference(targets[0].uid) if len(targets) == 1 and targets[0].uid in state.values else None
        linear = state.linear.get(targets[0].uid) if len(targets) == 1 and not targets[0].chain_summary else None
        if any(target.chain_summary for target in targets):
            key = self.chain_access_key("chain_value", sym, state)
            domain &= state.predicates.get(key, domain)
            form, difference = None, None
        return self.capture_value(Value(domain, sym.c, sym.ty.bits, sym.ty.boolean, form, difference,
                                        sym.ty.enum, sym.ty.character, signed=sym.ty.signed,
                                        result_arms=sym.ty.result_arms, linear=linear,
                                        result_error_enums=sym.ty.result_error_enums), state)

    def store_value(self, state, sym, value):
        targets = self.physical_places(sym, state)
        state = self.write_storage(state, sym)
        value = self.fresh_value(value, state)
        for target in targets:
            # A loop-local node retained by its links can outlive its precise
            # scalar state. A write through that site changes only one instance.
            if target.chain_summary:
                continue
            if target.heap_owner and target.uid not in state.values:
                owner = self.symbols[target.heap_owner - 1]
                if (owner not in self.initializing_heap_records
                        and not any(scope.get(owner.name) == owner for scope in self.scope)):
                    continue
            if len(targets) == 1:
                state = state.update(target, value.domain, value.affine, value.difference, value.linear)
            else:
                state = state.update(target, state.values[target.uid] | value.domain)
        return state

    def storage_bits(self, state, sym):
        if sym.bit_overlay:
            return 0, 0
        targets = self.physical_places(sym, state)
        facts = [state.bits.get(target.uid, (0, 0)) for target in targets]
        if not facts:
            return 0, 0
        mask, data = facts[0]
        for other_mask, other_data in facts[1:]:
            mask &= other_mask & ~(data ^ other_data)
        return mask, data & mask

    def store_bits(self, state, sym, facts, *, record_write=True):
        if sym.bit_overlay:
            return state
        targets = self.physical_places(sym, state)
        for target in targets:
            if target.heap_owner and target.uid not in state.values:
                continue
            mask, data = facts
            if len(targets) > 1:
                old_mask, old_data = state.bits.get(target.uid, (0, 0))
                mask &= old_mask & ~(data ^ old_data)
            state = state.with_bits(target, (mask, data & mask))
        return state

    def chain_access_key(self, kind, sym, state):
        root = sym
        while root.link_access is not None:
            root = root.link_access[0]
        uid = self.alias_members.get(root.uid, (root.uid, ()))[0]
        return (kind, sym.c, uid, state.epochs.get(uid))

    def link_targets(self, sym, state):
        targets = set()
        for field in self.physical_places(sym, state, allow_null=True):
            if not field.uid:
                targets.add(self.null_symbol)
            else:
                choices = state.borrows.get(field.uid, frozenset())
                if field.chain_summary:
                    known = state.predicates.get(self.chain_access_key("chain_link", sym, state))
                    if known is not None:
                        choices = (choices - {self.null_symbol} if known
                                   else choices & {self.null_symbol})
                targets.update(choices)
        return frozenset(targets)

    def link_initializer(self, ty, node, state, token, *, capture_exit=None):
        alias = BorrowRecord(self.link_record(ty, token), readonly=ty.readonly, nullable=True)
        targets, code = self.alias_source(node, state, alias, token, capture_exit=capture_exit)
        for target in targets - {self.null_symbol}:
            self.require_target_live(target, state, token)
        return targets, code

    def check_loop_link_rewrite(self, state, field, targets, token):
        old = state.borrows.get(field.uid, ())
        live = {target.uid for target in old if "Alive" in state.heap.get(target.uid, ())}
        if self.loops:
            for loop in self.loops:
                previous = state.predicates.get((loop.entry_alias_marker, "previous"), loop.previous_allocations)
                if field.uid <= loop.symbol_cutoff and live & previous:
                    token.error("cannot overwrite a link to live allocations from previous loop iterations")
                for cursor in loop.append_cursors:
                    key = (loop.entry_alias_marker, "append", cursor)
                    append = state.predicates.get(key)
                    if append is not None and append[0] == field.uid and append[:2] in state.fresh_links:
                        if append[2] and targets == frozenset((self.null_symbol,)):
                            return False
                        token.error("cannot rewrite the current link of a symbolic loop cursor after appending")
        # Mutating the representative of several allocations changes only
        # one runtime object. A new pointer into a repeat may likewise refer
        # to an interior instance and cannot stand for its complete ownership.
        live.update(target.uid for target in targets if "Alive" in state.heap.get(target.uid, ()))
        live.add(field.heap_owner)
        if self.loops:
            cutoff = self.loops[-1].symbol_cutoff
            live = {uid for uid in live if 0 < uid <= cutoff}
        if live and live & self.repeated_allocation_uids(state, token):
            token.error("cannot rewrite a link into or inside a symbolic repeated allocation group")
        return True

    def repeated_allocation_uids(self, state, token):
        if not state.fresh_links:
            return frozenset()
        # The fixed point retains symbols from an earlier analysis round
        # after the symbol table has rolled back. Read the owners from the
        # path's targets so their fields still identify the correct sites.
        live_sites = {uid for _, uid in state.fresh_links if "Alive" in state.heap.get(uid, ())}
        link_ids = {field for field, uid in state.fresh_links if uid in live_sites}
        owners = {target for targets in state.borrows.values() for target in targets
                  if target.uid in live_sites}
        field_owners = {}
        for owner in owners:
            for leaf in self.storage_leaves(owner):
                if leaf.uid in link_ids:
                    field_owners.setdefault(leaf.uid, set()).add(owner.uid)
        arcs = set()
        vertices = set()
        for link_uid, target_uid in state.fresh_links:
            if target_uid not in live_sites:
                continue
            for source_uid in field_owners.get(link_uid, ()):
                arcs.add((source_uid, target_uid))
                vertices.update((source_uid, target_uid))
        self.spend(token, len(arcs))
        components = strongly_connected_components(vertices, arcs)
        return frozenset(uid for source, target in arcs if components[source] == components[target]
                         for uid in (source, target))

    def check_loop_link_instance(self, state, field, destination, token):
        alias = None
        if destination is not None and destination.link_access is None and destination.projection is None:
            alias = self.alias_members.get(destination.uid)
            if destination == field and alias is None:
                return
        for loop in self.loops:
            marker = loop.entry_alias_marker
            previous = state.predicates.get((marker, "previous"), loop.previous_allocations)
            if field.heap_owner not in previous:
                continue
            if alias is not None:
                root, _ = alias
                binding = state.predicates.get((marker, "current", root))
                if binding == (field.heap_owner, state.epochs.get(root)):
                    continue
                append = state.predicates.get((marker, "append", root))
                if (append is not None and append[0] == field.uid and (marker, root) in state.predicates
                        and state.predicates[marker, root] is state.epochs.get(root)):
                    continue
            token.error("cannot prove a link write selects a distinct allocation in this loop iteration")

    def update_link_targets(self, state, field, targets, token, *, destination=None):
        self.check_loop_link_instance(state, field, destination, token)
        if not self.check_loop_link_rewrite(state, field, targets, token):
            return state
        guard = state.guard_label()
        self.trace.append(f"  update entity id#{field.uid} {field.name} [{guard}]: "
                          f"link targets {self.trace_alias_targets(targets)}")
        return replace(state, borrows=state.borrows | {field.uid: targets},
                       fresh_links=frozenset(edge for edge in state.fresh_links if edge[0] != field.uid))

    def store_link(self, state, sym, targets, token, places=None, skip=False):
        for target in targets - {self.null_symbol}:
            self.require_target_live(target, state, token)
        if places is None:
            places = self.physical_places(sym, state)
        for field in places:
            if field.uid in self.chain_fields:
                param = self.functions[self.current][1].params[self.chain_fields[field.uid]]
                if mutable_graph(param) and targets != frozenset((self.null_symbol,)):
                    token.error("cannot rewrite a graph link to a non-null target")
                if targets != frozenset((self.null_symbol,)) and not skip:
                    token.error("cannot rewrite a chain link to a non-null target outside its existing suffix")
                effect = "@skip" if skip and targets != frozenset((self.null_symbol,)) else "@prune"
                state = self.note_chain_prune(state, field, token, effect=effect)
                if len(places) != 1 or field.chain_summary:
                    bound = state.borrows[field.uid] | targets
                else:
                    bound = targets
                state = self.update_link_targets(state, field, bound, token, destination=sym)
                continue
            # Heap storage outlives the block that allocated it: the link only
            # has to keep the target alive until free, which the tombstone
            # state machine checks.  Stack targets still have to outlive the
            # field that points at them.
            if any(target.depth > field.depth and not self.is_whole_heap_object(target)
                   for target in targets):
                token.error("record link would outlive its target storage")
            state = self.write_actual_storage(state, field)
            old = state.borrows.get(field.uid, frozenset())
            bound = targets if len(places) == 1 else old | targets
            state = state.update(field, ZERO)
            state = self.update_link_targets(state, field, bound, token, destination=sym)
        return state

    def link_assignment_code(self, sym, op, node, code, *, nonnull):
        if not nonnull and not self.has_effects(node):
            return f"{sym.c} = {code};"
        name = f"lan_link_dst_{op.line}_{op.col}"
        assignment = f"{sym.ty.c} *{name} = &({sym.c}); *{name} = {code};"
        if nonnull:
            predicate = f"*{name} != NULL"
            self.trace.append(f"  assume link assignment {predicate}")
            assume = self.helper("contract_assume", None)
            assignment += f" {assume}({predicate});"
        return f"{{ {assignment} }}"

    def skips_successor(self, field, node, state):
        if node.kind != "member" or self.has_effects(node):
            return False
        source = self.place(node)
        if source is None or not isinstance(source.ty, RecordLink) or source.link_access is None:
            return False
        parent, name = source.link_access
        return name == "next" and self.comparison_place(parent, state).c == field.c

    def loop_allocation_edge(self, state, sym, places, node, targets):
        if (not self.loops or node.kind != "name" or len(places) != 1 or len(targets) != 1
                or sym.link_access is not None or sym.projection is not None):
            return None
        loop = self.loops[-1]
        target, = targets
        field, = places
        if (target != self.place(node) or not target.allocated_record or target.uid <= loop.symbol_cutoff
                or field.heap_owner <= loop.symbol_cutoff or field.projection is not None):
            return None
        alias = self.alias_members.get(sym.uid)
        if alias is not None:
            root, steps = alias
            marker = (loop.entry_alias_marker, root)
            if len(steps) == 1 and marker in state.predicates and state.predicates[marker] is state.epochs.get(root):
                return field.uid, target.uid
        elif sym == field and field.heap_owner < target.uid and self.symbols[field.heap_owner - 1].allocated_record:
            return field.uid, target.uid
        return None

    def note_loop_append(self, state, sym, places, node, targets, token):
        if not self.loops or not places:
            return state
        loop = self.loops[-1]
        marker = loop.entry_alias_marker
        alias = self.alias_members.get(sym.uid) if sym.link_access is None and sym.projection is None else None
        root = alias[0] if alias is not None and alias[1] == ("next",) else None
        entry = root is not None and (marker, root) in state.predicates and (
            state.predicates[marker, root] is state.epochs.get(root))
        direct = node.kind == "name" and len(targets) == 1 and next(iter(targets)) == self.place(node)
        fresh = direct and any(target.allocated_record and target.uid > loop.symbol_cutoff for target in targets)
        appending = entry and len(places) == 1 and fresh
        if not appending:
            return state
        place, = places
        target, = targets
        key = (marker, "append", root)
        if key in state.predicates:
            token.error("cannot append twice through the same loop cursor before advancing it")
        loop.append_cursors.add(root)
        saved_null = state.borrows.get(place.uid) == frozenset((self.null_symbol,))
        return replace(state, predicates=state.predicates | {key: (place.uid, target.uid, saved_null)})

    def note_loop_cursor_binding(self, state, sym, node, targets):
        if not self.loops or node.kind != "name" or len(targets) != 1:
            return state
        loop = self.loops[-1]
        target, = targets
        if target.allocated_record and target.uid > loop.symbol_cutoff and target == self.place(node):
            key = (loop.entry_alias_marker, "current", sym.uid)
            return replace(state, predicates=state.predicates | {key: (target.uid, state.epochs.get(sym.uid))})
        return state

    def check_loop_cursors(self, loop, paths, token):
        marker = loop.entry_alias_marker
        for path in paths:
            for root in loop.append_cursors:
                append = path.predicates.get((marker, "append", root))
                targets = path.borrows.get(root, ())
                current_epoch = path.epochs.get(root)
                unchanged = (marker, root) in path.predicates and path.predicates[marker, root] is current_epoch
                if append is None and (unchanged or all(target.uid <= loop.symbol_cutoff for target in targets)):
                    continue
                binding = path.predicates.get((marker, "current", root))
                if len(targets) == 1 and binding is not None:
                    target, = targets
                    if binding == (target.uid, current_epoch) and path.heap.get(target.uid) == frozenset(("Alive",)):
                        next_link = next((field for field in target.fields if field.name == "next"), None)
                        if next_link is not None and (
                                path.borrows.get(next_link.uid) == frozenset((self.null_symbol,)) or
                                append is not None and append[0] == next_link.uid and append[2]
                                and append[:2] in path.fresh_links):
                            continue
                token.error("cannot prove the loop cursor advances to a fresh tail on every back edge")

    def assign_link(self, sym, op, node, paths, *, propagate_target=False):
        if op.text != "=" or sym.readonly:
            op.error("record link assignment requires '=' and mutable storage")
        result = []
        nonnull = True
        exit_code = None
        for path in paths:
            field = self.comparison_place(sym, path, allow_effectful_links=propagate_target,
                                          propagate_exit=propagate_target)
            if field.projection is not None and field.projection.always_returns:
                candidate = f"(void)({field.c});"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            self.require_live(field, path, op)
            places = self.physical_places(field, path)
            early = []
            attaching = node.kind != "null" and any(self.attachment_parameter(place) is not None for place in places)
            if attaching:
                alias = BorrowRecord(self.link_record(sym.ty, op), readonly=True)
                targets, code = self.alias_source(node, path, alias, op, capture_exit=early)
            else:
                targets, code = self.link_initializer(sym.ty, node, path, op, capture_exit=early)
            if early:
                candidate = f"{{ (void)(&({field.c})); (void)({early[0].code}); }}"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            if attaching:
                code = f"({sym.ty.c})({code})"
                for place in places:
                    path = self.attach_next(path, place, targets, op, weak=len(places) != 1)
                result.append(path)
            elif not self.has_effects(node) and self.place(node) == sym:
                result.append(path)
            else:
                skip = any(place.uid in self.chain_fields for place in places) and self.skips_successor(field, node, path)
                edge = self.loop_allocation_edge(path, sym, places, node, targets)
                state = self.note_loop_append(path, sym, places, node, targets, op)
                state = self.store_link(state, sym, targets, op, places, skip=skip)
                if edge is not None:
                    state = replace(state, fresh_links=state.fresh_links | {edge})
                result.append(state)
            nonnull = nonnull and bool(targets) and self.null_symbol not in targets
            emitted = field, code
        if not result:
            return [], [exit_code, "__builtin_unreachable();"]
        field, code = emitted
        return result, [self.link_assignment_code(field, op, node, code, nonnull=nonnull)]
