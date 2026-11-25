"""Interface arguments, binding and rebinding."""

from __future__ import annotations

from dataclasses import field, replace

from .typesys import BorrowRecord, Interface, InterfaceResult, Record, Value, ZERO
from .interface_types import (compose_interface_rebind, direct_scalar_record_write, encode_interface_rebind,
                              interface_mismatch, interface_rebind_array_write, interface_rebind_expression,
                              interface_rebind_guard, valid_interface_rebind)


class InterfaceMixin:
    def interface_argument(self, node, state, ty, *, capture_exit=None):
        result = self.composite_call_type(node)
        data = None
        if isinstance(result, InterfaceResult):
            arguments = []
            _, data = self.call(node, state, arguments, capture_early_return=capture_exit)
            if capture_exit:
                return None, data
            source = arguments[result.source]
            shape = result.interface
        else:
            source = self.place(node)
            if source is None or not isinstance(source.ty, (Record, Interface)):
                node.token.error("interface argument requires an addressable record or interface")
            if source.projection is not None:
                source = self.capture_projected_argument(source, state, node.token,
                                                         propagate_exit=capture_exit is not None)
                if source.projection.always_returns:
                    capture_exit.append(Value(ZERO, source.c, effectful=True, always_returns=True))
                    return None, source.c
            self.require_live(source, state, node.token)
            shape = source.ty
        readonly = source.readonly or (isinstance(shape, Interface) and shape.readonly)
        mismatch = interface_mismatch(shape.fields, readonly, ty)
        if mismatch is not None:
            node.token.error(mismatch)
        if isinstance(source.ty, Interface):
            indices = {name: index for index, (name, _) in enumerate(source.ty.fields)}
            offsets = [indices[name] for name, _ in ty.fields]
            if offsets == list(range(len(offsets))):
                table = f"{source.c}.offsets"
            else:
                entries = ", ".join(f"{source.c}.offsets[{index}]" for index in offsets)
                table = f"(const size_t[]){{{entries}}}"
            data = data if data is not None else f"{source.c}.data"
        else:
            table = self.helper("interface_offsets", source.ty, ty)
            data = data if data is not None else f"&({source.c})"
        return source, f"((lan_interface){{{data}, {table}}})"

    def known_interface_layout(self, node, source):
        if node.kind == "call":
            callee = node.args[0]
            if (callee.kind != "name" or callee.args[0] in self.externs
                    or not isinstance(self.composite_call_type(node), InterfaceResult)):
                return None
        return source.ty if isinstance(source.ty, Record) else source.interface_layout

    def interface_field_targets(self, source, state, depth, token):
        targets = frozenset(self.physical_places(source, state))
        if any(target.depth > depth for target in targets):
            token.error("interface would outlive its target storage")
        if not targets or any(target.projection is not None or target.heap_owner
                              or target.uid in state.heap for target in targets):
            token.error("interface field requires a direct non-heap record target")
        return targets

    def bind_interface(self, action, sym, token, bindings, paths):
        result = []
        for path, (source, _) in zip(paths, bindings):
            if sym.interface_offsets is not None:
                targets = self.interface_field_targets(source, path, sym.depth, token)
            else:
                targets = frozenset(self.physical_places(source, path))
                if any(target.depth > sym.depth for target in targets):
                    token.error("interface would outlive its target storage")
            guard = path.guard_label()
            self.trace.append(f"  {action} id#{sym.uid} {sym.name} [{guard}]: "
                              f"interface targets {self.trace_alias_targets(targets)}")
            bound = path.touch(sym.uid).update(sym, ZERO)
            result.append(replace(bound, borrows=bound.borrows | {sym.uid: targets}))
        return result

    def interface_storage_code(self, sym, code):
        helper = self.helper("interface_rebind", None)
        offsets = sym.interface_offsets or f"lan_offsets_{sym.uid}"
        if (sym.borrowed or sym.heap_owner) and sym.interface_offsets is not None:
            # The existing descriptor identifies its mutable offset buffer.
            offsets = f"(size_t *){sym.c}.offsets"
        return f"{helper}({code}, {offsets}, {len(sym.ty.fields)})"

    @staticmethod
    def interface_rebind_arms(stmt):
        condition, yes, no = stmt.args
        if condition.kind == "unary" and condition.args[0] == "!":
            return condition.args[1], ((yes, "0"), (no, "1"))
        return condition, ((yes, "1"), (no, "0"))

    def interface_rebind_site(self, stmt, token, forwarded_guard, literals=()):
        """Guard literals between the enclosing block and this rebind.

        ``None`` means the statement does not contain the rebind at all, while
        an empty tuple means it happens unconditionally.  Nested guards are
        collected along the way, so ``if (a) { if (b) { h.view = x; } }``
        yields ``((a, 1), (b, 1))``: both literals must hold.
        """
        if stmt.kind == "assign" and stmt.args[1] is token:
            return tuple(literals)
        if stmt.kind == "if":
            condition, arms = self.interface_rebind_arms(stmt)
            for arm, outcome in arms:
                for item in arm:
                    found = self.interface_rebind_site(item, token, forwarded_guard,
                                                       literals + ((condition, outcome),))
                    if found is not None:
                        return found
            return None
        if stmt.kind in ("eval", "return"):
            site = stmt.args[0]
            if (site is not None and site.kind == "call" and site.token is token
                    and all(arg.kind in ("name", "boolean") for arg in site.args[1])):
                return tuple(literals) + ((forwarded_guard,) if forwarded_guard else ())
        return None

    def interface_rebind_block_ok(self, stmt, params, indices):
        """Every statement below this guard is a companion write or a guard."""
        if stmt.kind == "if":
            _, arms = self.interface_rebind_arms(stmt)
            return all(self.interface_rebind_block_ok(item, params, indices)
                       for arm, _ in arms for item in arm)
        return self.interface_rebind_companion_write(stmt, params, indices)

    def interface_rebind_companion_write(self, stmt, params, indices):
        if stmt.kind != "assign":
            return False
        target, op, value = stmt.args
        names = []
        if target.kind == "index":
            target, subscript = target.args
            if self.has_effects(subscript, allow_debug_checks=True):
                return False
            names.append("@element")
        while target.kind == "member":
            names.append(target.args[1].text)
            target = target.args[0]
        if target.kind != "name" or target.args[0] not in indices:
            return False
        index = indices[target.args[0]]
        param, route = params[index], names[::-1]
        if direct_scalar_record_write(param, route) or interface_rebind_array_write(param, route):
            return not self.has_effects(value, allow_debug_checks=True)
        if len(route) > 1 and isinstance(param, BorrowRecord):
            view = dict(param.record.fields).get(route[0])
            if isinstance(view, Interface) and not view.readonly:
                # Body analysis must resolve the write to a donor parameter.
                # An unresolved `*` mutation cannot be combined with a rebind.
                viewed = replace(param, record=view.schema)
                if (direct_scalar_record_write(viewed, route[1:])
                        or interface_rebind_array_write(viewed, route[1:])):
                    return not self.has_effects(value, allow_debug_checks=True)
        return (len(names) == 1 and op.text == "=" and value.kind == "name"
                and valid_interface_rebind(params, index, names[0], indices.get(value.args[0])))

    def note_parameter_interface_rebind(self, field, source, token, *, forwarded_guard=None):
        declaration, contract = self.functions[self.current]
        receiver = self.param_places.get(field.uid)
        donor = self.param_places.get(source.uid) if source is not None else None
        if (receiver is None or receiver[1] != (field.name,) or donor is None or donor[1]
                or not valid_interface_rebind(contract.params, receiver[0], field.name, donor[0])):
            token.error("rebinding a borrowed interface field requires a compatible borrowed record parameter as its source")
        body = declaration.args[3]
        indices = {param.text: index for index, param in enumerate(declaration.args[2])}
        site_error = ("rebinding a borrowed interface field requires direct field writes, "
                      "boolean-guarded field writes or a direct forwarding call")
        test = None
        sites = [position for position, stmt in enumerate(body)
                 if self.interface_rebind_site(stmt, token, forwarded_guard) is not None]
        first_site = min(sites) if sites else None
        last_site = max(sites) if sites else None

        def mark(stmt, destination):
            if stmt.kind != "assign":
                return
            target = stmt.args[0]
            names = []
            while target.kind == "member":
                names.append(target.args[1].text)
                target = target.args[0]
            if target.kind != "name":
                return
            owner = indices.get(target.args[0])
            route = tuple(reversed(names))
            if (owner == receiver[0] and len(route) > 1 and route[0] == field.name
                    and self.interface_rebind_companion_write(stmt, contract.params, indices)):
                key = (owner, (field.name, "*", *route[1:]))
                # A write that sits between two guarded arms belongs to neither
                # side: it targets the value selected so far, which the summary
                # cannot name.  Reporting it as "before" would claim a write to
                # the original target and narrow its value wrongly.
                other = (self.interface_after_writes if destination is self.interface_before_writes
                         else self.interface_before_writes)
                if key in other:
                    token.error("an interface write between two guarded rebinds cannot be ordered")
                destination.add(key)

        for position, stmt in enumerate(body):
            site = self.interface_rebind_site(stmt, token, forwarded_guard)
            if first_site is not None and position < first_site:
                mark(stmt, self.interface_before_writes)
            elif last_site is not None and position > last_site:
                mark(stmt, self.interface_after_writes)
            if site is not None:
                test = site
                if stmt.kind != "if":
                    continue
            if stmt.kind == "if":
                condition, _ = self.interface_rebind_arms(stmt)
                guard = indices.get(condition.args[0], -1) if condition.kind == "name" else -1
                if (interface_rebind_guard(contract.params, guard)
                        and self.interface_rebind_block_ok(stmt, contract.params, indices)):
                    continue
            if stmt is body[-1] and stmt.kind == "return" and not self.has_effects(
                    stmt.args[0], allow_debug_checks=True):
                continue
            if self.interface_rebind_companion_write(stmt, contract.params, indices):
                continue
            token.error(site_error)
        if test is None or self.current in self.recursive_functions:
            token.error(site_error)
        guarded = ()
        if test:
            literals = {}
            for condition, expected in test:
                guard = -1
                for index, param in enumerate(declaration.args[2]):
                    if (condition.kind == "name" and param.text == condition.args[0]
                            and interface_rebind_guard(contract.params, index)):
                        guard = index
                        break
                if guard < 0:
                    token.error("conditional interface rebinding requires a boolean parameter guard")
                if literals.setdefault(guard, expected) != expected:
                    token.error("conditional interface rebinding repeats a guard with both outcomes")
            guarded = tuple(str(part) for guard in sorted(literals)
                            for part in (guard, literals[guard]))
        entry = (receiver[0], ("@interface_rebind", field.name, str(donor[0]), *guarded))
        self.writes.add(entry)
        self.rebind_order[entry] = (token.line, token.col)

    def normalize_interface_rebinds(self, writes, token):
        """Fold repeated writes to one interface field into a target expression.

        A setter may assign ``holder.view`` more than once.  The final target
        is still a function of the boolean guards and the original target, so
        the compact summary only needs the composed expression.  Different
        guards would require a decision tree the summary cannot express yet;
        those are rejected rather than silently losing an earlier write.
        """
        groups = {}
        for entry in writes:
            index, names = entry
            if names and names[0] == "@interface_rebind":
                groups.setdefault((index, names[1]), []).append(entry)
        for (index, field), entries in groups.items():
            # Complementary guarded writes cover both arms, so their order is
            # irrelevant.  Keep the historical encoding for that case.
            ordered = sorted(entries, key=lambda entry: entry[1])
            if len(ordered) == 2:
                first, second = ordered[0][1], ordered[1][1]
                if (len(first) == len(second) == 5 and first[3] == second[3]
                        and first[4] != second[4]):
                    for entry in entries:
                        writes.discard(entry)
                    writes.add((index, (*first, second[2])))
                    continue
            entries.sort(key=lambda entry: self.rebind_order.get(entry, (1 << 30, 1 << 30)))
            expression = ("old",)
            for entry in entries:
                expression = compose_interface_rebind(
                    expression, interface_rebind_expression(entry[1]))
                if expression is None:
                    # Distinct guards cannot collapse into one expression.  Keep
                    # one leaf per arm instead: the summary is a may-set, so the
                    # union of the arms is exactly the set of possible targets.
                    break
            else:
                encoded = encode_interface_rebind(field, expression)
                if encoded is not None:
                    for entry in entries:
                        writes.discard(entry)
                    writes.add((index, encoded))

    def apply_interface_rebind(self, state, holder, name, choices, token, *, keep_old):
        self.check_interface_rebind_holder(state, holder, token)
        field = self.child_place(holder, (name,))
        if field.readonly:
            token.error("cannot rebind an immutable interface field")
        targets = state.borrows[field.uid] if keep_old else frozenset()
        for source, guard in choices:
            candidates = self.interface_field_targets(source, state, field.depth, token)
            readonly = source.readonly or any(target.readonly for target in candidates)
            mismatch = interface_mismatch(source.ty.fields, readonly, field.ty)
            if mismatch is not None:
                token.error(mismatch)
            if field.borrowed:
                self.note_parameter_interface_rebind(field, source, token, forwarded_guard=guard)
            targets |= candidates
        self.note_transition_write(field, alias_binding=True)
        guard_label = state.guard_label()
        self.trace.append(f"  update entity id#{field.uid} {field.name} [{guard_label}]: "
                          f"interface targets {self.trace_alias_targets(targets)}")
        bound = state.touch(field.uid).update(field, ZERO)
        return replace(bound, borrows=bound.borrows | {field.uid: targets})

    def define_interface(self, name, ty, node, paths):
        if node is None:
            name.error("interface binding requires a record or interface initializer")
        paths, bindings, exit_code = self.binding_initializer_paths(
            node, paths, lambda path, early: self.interface_argument(node, path, ty, capture_exit=early))
        if not paths:
            return [], [exit_code, "__builtin_unreachable();"]
        layout = None
        if name.text not in self.rebound_slices:
            layouts = {self.known_interface_layout(node, source) for source, _ in bindings}
            if len(layouts) == 1:
                layout = next(iter(layouts))
        sym = self.declare(name, ty, readonly=ty.readonly, interface_layout=layout)
        self.alias_members[sym.uid] = (sym.uid, ())
        self.record_symbols(sym, name, zero=False)
        # A source descriptor can be rebound later while this binding still
        # points at the original object. Snapshot the layout with the pointer.
        lines = [f"size_t lan_offsets_{sym.uid}[{len(ty.fields)}];"]
        code = self.interface_storage_code(sym, bindings[0][1])
        result = self.bind_interface("define entity", sym, name, bindings, paths)
        return result, [*lines, f"lan_interface {sym.c} = {code};"]

    def rebind_interface(self, sym, op, node, paths):
        if op.text != "=" or sym.uid not in self.alias_members:
            op.error("local interface rebinding requires '='")
        if sym.interface_offsets is not None and sym.readonly:
            op.error("cannot rebind an immutable interface field")
        if sym.heap_owner in self.parameter_uids:
            # The record is only borrowed from the caller for the duration of
            # the call.  Pointing its interface at a callee-local object would
            # let a later write touch that object without a caller-visible
            # summary, so the field stays read-only in the callee.
            op.error("cannot rebind the interface field of an owning parameter")
        if sym.heap_owner:
            for path in paths:
                self.require_target_live(sym, path, op)
        if sym.borrowed and sym.interface_offsets is not None:
            self.note_parameter_interface_rebind(sym, self.record_place(node) if node.kind == "name" else None, op)
        paths, bindings, exit_code = self.binding_initializer_paths(
            node, paths, lambda path, early: self.interface_argument(node, path, sym.ty, capture_exit=early))
        if not paths:
            return [], [exit_code, "__builtin_unreachable();"]
        self.note_transition_write(sym, alias_binding=True)
        result = self.bind_interface("update entity", sym, op, bindings, paths)
        code = self.interface_storage_code(sym, bindings[0][1])
        return result, [f"{sym.c} = {code};"]

    def return_interface(self, node, ty, paths, inferred_view=None):
        view = ty.interface if ty is not None else inferred_view
        for path in paths:
            source, code = self.interface_argument(node, path, view)
            targets = self.physical_places(source, path)
            places = {self.param_places.get(target.uid) for target in targets}
            if ty is None:
                if len(places) != 1 or None in places or next(iter(places))[1]:
                    node.token.error("inferred interface must refer to a borrowed parameter")
                index, _ = next(iter(places))
                self.inferred_returns.append((node.token, InterfaceResult(view, index)))
            elif places != {(ty.source, ())}:
                node.token.error("returned interface must refer to the declared borrowed parameter")
            self.check_heap_exit([path], 0, node.token)
            guard = path.guard_label()
            self.trace.append(f"  return interface {self.trace_alias_targets(targets)} [{guard}]")
            self.return_states.append(replace(path))
        return [], [f"return ({code}).data;"]
