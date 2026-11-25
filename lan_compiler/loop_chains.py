"""Chain materialization and indexed release inside loops."""

from __future__ import annotations

from dataclasses import field, replace
from itertools import count

from value_domain import Domain

from .typesys import BitView, BOOL, BorrowRecord, HeapRecord, Record, Scalar, Slice, ZERO
from .domain_utils import plain_integer_parameter
from .interface_types import graph_or_chain_parameter, mutable_chain_record, valid_chain_record
from .c_emit import indent


class LoopChainMixin:
    @staticmethod
    def chain_prefix_demand(statements):
        """Find nearby releases of aliases captured from the first few links."""
        rounds = 0

        def scan(body, aliases):
            nonlocal rounds
            for stmt in body:
                if stmt.kind == "define":
                    name, ty, source = stmt.args
                    aliases.pop(name.text, None)
                    if ty is None or ty.kind != "slice_qualified" or ty.args[3] != "alias":
                        continue
                    distance = 0
                    while source is not None and source.kind == "member" and source.args[1].text == "next":
                        source = source.args[0]
                        distance += 1
                    if source is not None and source.kind == "name" and 1 <= distance <= 4:
                        aliases[name.text] = distance
                elif stmt.kind == "if":
                    scan(stmt.args[1], aliases.copy())
                    scan(stmt.args[2], aliases.copy())
                elif stmt.kind == "eval" and stmt.args[0].kind == "call":
                    callee, args = stmt.args[0].args
                    if callee.kind == "name" and callee.args[0] == "free" and len(args) == 1 and args[0].kind == "name":
                        rounds = max(rounds, aliases.get(args[0].args[0], 0))

        scan(statements, {})
        # Preserve the detached node's immediate successor as well, so that
        # unlinking it does not modify a pointer into a summarized group.
        return rounds + 1 if rounds else 0

    def chain_loop_materialization(self, stmt, paths, following):
        """Recognize a counting loop whose heap allocations can get individual IDs."""
        if self.loops or len(stmt.args) != 2 or not paths or len(paths) > 8:
            return 0, 0
        condition, body = stmt.args
        if condition.kind != "binary" or not 1 < len(body) <= 24:
            return 0, 0
        operator, left, right = condition.args
        if (operator not in ("<", "<=", ">", ">=") or left.kind != "name"
                or right.kind != "number" or type(right.args[0]) is not int
                or body[-1].kind != "assign"):
            return 0, 0
        counter = self.lookup(left)
        if (not plain_integer_parameter(counter.ty) or counter.uid in self.alias_members
                or counter.heap_owner or counter.borrowed or counter.readonly):
            return 0, 0
        target, step_op, step_node = body[-1].args
        if (target.kind != "name" or target.args != left.args or step_op.text not in ("+=", "-=")
                or step_node.kind != "number" or type(step_node.args[0]) is not int
                or step_node.args[0] <= 0):
            return 0, 0

        initial = paths[0].values.get(counter.uid)
        bound = right.args[0]
        step = step_node.args[0] * (1 if step_op.text == "+=" else -1)
        if initial is None or (step > 0) != (operator in ("<", "<=")):
            return 0, 0
        count = 0
        if initial.lo == initial.hi and all(path.values.get(counter.uid) == initial for path in paths):
            value = initial.lo
            check = {"<": value < bound, "<=": value <= bound, ">": value > bound, ">=": value >= bound}
            if not check[operator]:
                return 0, 0
            distance = abs(bound - value) + (operator in ("<=", ">="))
            trips = (distance + abs(step) - 1) // abs(step)
            if trips <= 16:
                count = trips
        prefix = self.chain_prefix_demand(following) if not count else 0
        if not count and not prefix:
            return 0, 0

        allocations = {}
        for item in body[:-1]:
            if item.kind == "define":
                name, ty, initializer = item.args
                if (name.text == counter.name or ty is None or ty.kind != "storage"
                        or ty.args[1].text != "heap" or ty.args[0].kind != "type_name"
                        or initializer is not None and (initializer.kind != "record_initializer"
                            or any(value.kind not in ("name", "number", "boolean", "null")
                                   for _, value in initializer.args[0]))):
                    return 0, 0
                storage = self.resolve(ty)
                if not isinstance(storage, HeapRecord) or not valid_chain_record(storage.record):
                    return 0, 0
                allocations[name.text] = storage.record
            elif item.kind == "assign":
                place, op, value = item.args
                if op.text != "=" or value.kind not in ("name", "null"):
                    return 0, 0
                if place.kind == "member":
                    if place.args[0].kind != "name" or place.args[1].text != "next":
                        return 0, 0
                    parent = place.args[0]
                    record = allocations.get(parent.args[0])
                    if record is None:
                        record = self.lookup(parent).ty
                    if not isinstance(record, Record) or not valid_chain_record(record):
                        return 0, 0
                elif place.kind == "name" and place.args[0] not in allocations:
                    alias = self.place(place)
                    if alias is None or not isinstance(self.alias_types.get(alias.uid), BorrowRecord):
                        return 0, 0
                else:
                    return 0, 0
            else:
                return 0, 0
        return (count, prefix) if allocations else (0, 0)

    def materialize_chain_prefix(self, stmt, paths, count):
        condition, body = stmt.args
        exits, code = [], []
        label = f"lan_chain_prefix_exit_{stmt.token.line}_{stmt.token.col}"
        for _ in range(count):
            active, inactive = [], []
            for path in paths:
                hit, miss, test = self.condition(condition, path)
                active.extend(hit)
                inactive.extend(miss)
            exits.extend(inactive)
            if inactive:
                code.append(f"if (!({test})) goto {label};")
            if not active:
                paths = []
                break
            paths, body_code = self.block(body, active)
            code.extend(["{", *indent(body_code), "}"])
        skipped = bool(exits)
        if paths:
            remaining, loop_code = self.analyze_loop(stmt, paths)
            exits.extend(remaining)
            code.extend(loop_code)
        if skipped:
            code.append(f"{label}: ;")
        self.trace.append(f"  materialize up to {count} chain prefix iterations")
        self.budget_paths(exits, stmt.token)
        return exits, code

    def indexed_release_has_capture(self, path, site, roots, links, token):
        pending = [target for uid, refs in path.borrows.items() if uid not in links for target in refs]
        pending.extend(target for positions in path.regions.values() for target, _ in positions)
        pending.extend(sym for sym, _ in roots)
        seen = {}
        while pending:
            symbol = pending.pop()
            if symbol.uid == site.uid or symbol.heap_owner == site.uid:
                return True
            if id(symbol) in seen:
                continue
            seen[id(symbol)] = symbol
            self.spend(token)
            pending.extend(symbol.fields)
            pending.extend(symbol.captured_places or ())
            if symbol.link_access is not None:
                pending.extend(self.physical_places(symbol, path, allow_null=True))
            if symbol.projection is not None:
                pending.append(symbol.projection.root)
                pending.extend(symbol.projection.captured or ())
            if isinstance(symbol.ty, BitView):
                pending.append(symbol.ty.host)
            elif isinstance(symbol.ty, Slice) and symbol.ty.projection is not None:
                pending.extend(symbol.ty.projection.owners)
        arrays = [(symbol, symbol.name) for symbol in seen.values() if symbol.uid in path.arrays]
        return any(self.dangling_array_links(path, site, arrays))

    def release_indexed_chain_node(self, stmt, paths):
        """Recognize and prove one guarded unlink-and-free traversal."""
        if self.loops or len(stmt.args) != 2:
            return None
        condition, body = stmt.args
        if (len(body) != 1 or body[0].kind != "if" or condition.kind != "binary"
                or condition.args[0] != "&&" or condition.args[1].kind != "name"):
            return None
        _, running_node, condition = condition.args
        running = self.lookup(running_node)
        if (not isinstance(running.ty, Scalar) or not running.ty.boolean or running.readonly
                or running.borrowed or running.heap_owner or running.uid in self.alias_members
                or any(path.values.get(running.uid) != Domain.point(1) for path in paths)):
            return None

        def access(node):
            steps = 0
            while node.kind == "member" and node.args[1].text == "next":
                steps += 1
                node = node.args[0]
            return (node.args[0], steps) if node.kind == "name" else None

        def nonnull(node):
            if node.kind == "binary" and node.args[0] == "!=" and node.args[2].kind == "null":
                return access(node.args[1])
            return None

        def assign(node, target, operator, source):
            dst, op, value = node.args
            return access(dst) == target and op.text == operator and access(value) == source

        first_guard = condition
        keep_suffix = condition.kind == "binary" and condition.args[0] == "&&"
        if keep_suffix:
            first_guard, suffix_guard = condition.args[1:]
        first = nonnull(first_guard)
        if first is None or first[1] != 1 or (keep_suffix and nonnull(suffix_guard) != (first[0], 2)):
            return None
        cursor_name = first[0]
        test, hit, miss = body[0].args
        if (tuple(item.kind for item in hit) != ("define", "assign", "assign", "eval", "assign")
                or tuple(item.kind for item in miss) != ("assign", "assign")
                or test.kind != "binary" or test.args[0] != "=="
                or test.args[1].kind != "name" or test.args[2].kind != "number"
                or type(test.args[2].args[0]) is not int or test.args[2].args != (0,)):
            return None
        victim, victim_type, source = hit[0].args
        counter_node = test.args[1]
        counter_name = counter_node.args[0]
        step, operator, amount = miss[1].args
        call = hit[3].args[0]
        stop_target, stop_op, stop_value = hit[4].args
        if (len({cursor_name, victim.text, counter_name, running.name}) != 4
                or victim_type is None or source is None or access(source) != first
                or not assign(hit[1], first, "=", (victim.text, 1))
                or not assign(hit[2], (victim.text, 1), "=", None) or hit[2].args[2].kind != "null"
                or not self.is_free_call(call) or len(call.args[1]) != 1 or access(call.args[1][0]) != (victim.text, 0)
                or not assign(miss[0], (cursor_name, 0), "=", first)
                or access(stop_target) != (running.name, 0) or stop_op.text != "="
                or stop_value.kind != "boolean" or stop_value.args != (False,)
                or access(step) != (counter_name, 0) or operator.text != "-="
                or amount.kind != "number" or type(amount.args[0]) is not int or amount.args != (1,)):
            return None

        cursor = self.lookup(first_guard.args[1].args[0])
        counter = self.lookup(counter_node)
        alias = self.alias_types.get(cursor.uid)
        if (not isinstance(alias, BorrowRecord) or cursor.readonly or alias.readonly or alias.nullable
                or graph_or_chain_parameter(alias) or cursor.uid in self.param_places
                or not valid_chain_record(alias.record) or not mutable_chain_record(alias.record)
                or not plain_integer_parameter(counter.ty) or not counter.ty.bits or counter.ty.result_arms is not None
                or counter.uid in self.alias_members or counter.readonly or counter.borrowed or counter.heap_owner):
            return None
        link = self.child_place(cursor, ("next",))
        if link.readonly or link.ty.readonly:
            return None

        # The only incoming link to the summarized family must be the head's
        # own link. Captured pointers into the family could dangle after an
        # arbitrary deletion and cannot be treated as aliases of the survivors.
        roots = [(sym, name) for scope in self.scope for name, sym in scope.items()]
        plans = []
        alive = frozenset(("Alive",))
        for path in paths:
            owners = self.physical_places(cursor, path, allow_null=True)
            domain = path.values.get(counter.uid)
            if (len(owners) != 1 or domain is None or domain.lo < 0
                    or not Domain.of((0, domain.hi)).within(counter.ty.domain)):
                return None
            head, = owners
            if (not self.is_whole_heap_object(head) or head.ty != alias.record or head.readonly
                    or path.heap.get(head.uid) != alive or head.chain_summary):
                return None
            entry = self.child_place(head, ("next",))
            targets = path.borrows.get(entry.uid, ())
            if len(targets) != 1:
                return None
            site, = targets
            if not site.uid:
                plans.append((path, head, entry, None, None, domain))
                continue
            if (site == head or site.ty != alias.record or not self.is_whole_heap_object(site)
                    or site.readonly or site.chain_summary or path.heap.get(site.uid) != alive):
                return None
            successor = self.child_place(site, ("next",))
            if (path.borrows.get(successor.uid) != frozenset((site, self.null_symbol))
                    or (successor.uid, site.uid) not in path.fresh_links
                    or any(field.uid in path.values for field in site.fields if isinstance(field.ty, Scalar))):
                return None
            if self.indexed_release_has_capture(path, site, roots, (entry.uid, successor.uid), stmt.token):
                stmt.token.error("cannot release an indexed node while another reference into the repeated chain remains")
            plans.append((path, head, entry, site, successor, domain))
        if not any(site is not None for _, _, _, site, _, _ in plans) or self.resolve(victim_type) != alias:
            return None

        result = []
        for path, head, entry, site, successor, domain in plans:
            if site is not None:
                path = self.write_actual_storage(path, running).update(running, BOOL)
                path = self.write_actual_storage(path, entry)
                path = self.write_actual_storage(path, successor)
                self.note_transition_write(site, lifetime=True)
                for field in site.fields:
                    if isinstance(field.ty, Scalar):
                        path = path.forget_bit_values(field.uid).with_bits(field, (0, 0))
                path = replace(path, predicates={})
                if not keep_suffix and domain & ZERO:
                    # Removing the family's sole node leaves the cursor on
                    # the head with index zero and no live summarized objects.
                    empty = path.narrow(counter, ZERO)
                    if empty is not None:
                        result.append(replace(empty, borrows=empty.borrows | {entry.uid: frozenset((self.null_symbol,))},
                                              heap=empty.heap | {site.uid: frozenset(("Tombstone",))}))
                if domain.hi:
                    self.note_transition_write(cursor, alias_binding=True)
                    path = replace(path.touch(cursor.uid), borrows=path.borrows | {cursor.uid: frozenset((head, site))})
                    path = self.write_actual_storage(path, counter)
                    path = path.update(counter, Domain.of((0, domain.hi))).with_bits(counter, (0, 0))
                # Any remaining members are still covered by the same
                # complete family. The detached pointer cannot escape.
            result.append(path)

        ptr = cursor.c_var
        removed = f"lan_removed_{stmt.token.line}_{stmt.token.col}"
        guard = f"{ptr}->f_next" + (f" && {ptr}->f_next->f_next" if keep_suffix else "")
        self.memory_effects.add(self.current)
        code = [f"while ({guard}) {{",
                f"    if ({counter.c} == 0) {{",
                f"        {alias.record.c} *{removed} = {ptr}->f_next;",
                f"        {ptr}->f_next = {removed}->f_next;",
                f"        {removed}->f_next = NULL;",
                f"        free((void *){removed});",
                f"        {running.c} = 0;",
                "        break;",
                "    }",
                f"    {ptr} = {ptr}->f_next;",
                f"    --{counter.c};",
                "}"]
        self.trace.append("  release indexed interior chain node; repeated suffix stays Alive" if keep_suffix else
                          "  release indexed chain node; remaining family is live or empty")
        self.budget_paths(result, stmt.token)
        return result, code
