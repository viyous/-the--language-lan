"""Native call checks, error propagation and function calls."""

from __future__ import annotations

import copy
from dataclasses import field, fields, replace
from itertools import count
from math import inf

from value_domain import Domain, DomainComplexity
from table_interface import c_field

from .encoding import TextEncoding
from .syntax import Node
from .typesys import (AliasResult, Array, BorrowArray, BorrowRecord, BorrowScalar, CopyArray, CopyRecord,
                      ExternCallable, FunctionBinding, HeapArray, HeapObject, HeapRecord, Interface, InterfaceResult,
                      Record, RecordLink, Scalar, Slice, Span, success_type, Symbol, TaggedResult, unqualified,
                      Value, ZERO)
from .domain_utils import c_int, container, domain_union, plain_integer_parameter
from .interface_types import (alias_graph_records, chain_attachments, chain_reachable_path,
                              dependent_source_domain, encode_interface_type, graph_scalar_borrows, linked_alias_path,
                              linked_owned_record, offset_expression_lengths, offset_expression_parameters,
                              slice_offset_terms)
from .summaries import interface_accessor_member, INTERFACE_FIELD_COMPARISONS
from .c_emit import c_predicate_choice, c_predicate_sequence


class CallMixin:
    def native_alias_origin_predicate(self, source, path, snapshot, result, graph_records=()):
        checks = []
        if chain_reachable_path(source, path):
            helper = self.helper("chain_reachable", source.c)
            return f"{helper}({snapshot}, {result})"
        if linked_alias_path(source, path, graph_records):
            address = snapshot
            for step in path:
                checks.append(f"{address} != NULL")
                address = f"({address})->{c_field(step)}"
            checks.append(f"(const void *){result} == (const void *)({address})")
            return c_predicate_sequence(checks, conjunction=True)
        if isinstance(source, Slice):
            step = path[0]
            checks.extend((f"{snapshot}.data != NULL", f"{snapshot}.len > {step}"))
            place = f"((const {source.element.c} *){snapshot}.data)"
        else:
            if source.nullable:
                checks.append(f"{snapshot} != NULL")
            place = f"(*{snapshot})" if isinstance(source, BorrowRecord) else snapshot
        for step in path:
            place = f"({place})[{step}]" if isinstance(step, int) else f"({place}).{c_field(step)}"
        address = f"&({place})" if path else snapshot
        checks.append(f"(const void *){result} == (const void *)({address})")
        return c_predicate_sequence(checks, conjunction=True)

    def native_slice_origin_checks(self, source, result, snapshot, descriptor, offset):
        if isinstance(source, Slice):
            base = f"((const {source.element.c} *){snapshot}.data)"
            length = f"{snapshot}.len"
        elif isinstance(source, BorrowArray):
            base, length = snapshot, str(source.array.length)
        else:
            field = source.record
            base = f"(*{snapshot})"
            for name in result.source_field:
                field = unqualified(dict(field.fields)[name])
                base = f"({base}).{c_field(name)}"
            length = str(field.length)
        if offset is None:
            check = self.helper("debug_native_slice_window", None)
            return [f"{check}({base}, {length}, {descriptor}, sizeof({result.element.c}))"]
        checks = [f"{offset} >= 0", f"(lan_int){length} >= {offset}"]
        if result.text == "c":
            data = f"(const void *){descriptor}"
        else:
            checks.append(f"{descriptor}.len <= {length} - (size_t){offset}")
            data = f"{descriptor}.data"
        same = f"{data} == (const void *)({base})"
        shifted = f"{data} == (const void *)(({base}) + (size_t){offset})"
        checks.append(c_predicate_choice(f"{offset} == 0", same, shifted))
        return checks

    def native_writeback_check(self, token, ty, snapshot):
        if isinstance(ty, Slice):
            checked = self.checked_native_span_content(token, replace(ty, content_bound=None), snapshot)
            return f"({checked}, 1)" if checked != snapshot else None
        if isinstance(ty, BorrowScalar):
            predicate = self.membership(token, ty.scalar.domain)
            check = f"{predicate}(*{snapshot})"
        elif isinstance(ty, BorrowRecord):
            predicate = self.native_record_validator(token, ty.record)
            if predicate is None:
                return None
            check = f"{predicate}({snapshot})"
        else:
            predicate = self.native_array_validator(token, ty.array)
            if predicate is None:
                return None
            check = f"{predicate}((const {ty.array.c} *){snapshot})"
        return c_predicate_choice(f"!{snapshot}", "1", check) if ty.nullable else check

    def consume_temporary(self, arg, ty, index, callee, state, arguments, codes, *, accepts_chain):
        """Hand an owning call result straight to a consuming parameter.

        The object never acquires a binding, so no later statement can observe
        it and there is no tombstone to report; the callee's own exit check
        still decides whether the object is released or passed on.
        """
        actual = self.composite_call_type(arg)
        if not isinstance(actual, (HeapRecord, HeapObject, HeapArray)):
            arg.token.error("a consuming parameter requires a named heap owner or an owning result")
        if isinstance(ty, HeapArray):
            if (not isinstance(actual, HeapArray) or actual.element != ty.element
                    or not actual.domain.within(ty.domain)):
                arg.token.error("a consuming array parameter requires a matching owning array")
        elif actual != ty:
            arg.token.error("a consuming parameter requires the same heap type")
        if linked_owned_record(ty) and not accepts_chain and self.owned_chain_call_lengths(arg) != (1,):
            arg.token.error("cannot pass an owning chain to a parameter that requires next to be null")
        type_again, value = self.call(arg, state)
        arguments.append(Symbol("#temporary", 0, ty, 0, "", 0, False, False))
        codes.append(value)
        guard = state.guard_label()
        self.trace.append(f"  transfer ownership of an unnamed result "
                          f"[{arg.token.line}:{arg.token.col} {guard}] to param#{index} {callee}")
        return value

    def callback_binding(self, node, ty, selected):
        if self.current in self.recursive_groups:
            node.token.error("indirect calls are not supported inside recursive functions")
        contract = replace(ty.contract, identity_property=False, idempotent_property=False,
                           commutative_property=False, diagonal_property=False)
        writes = set()
        for candidate in ty.candidates:
            self.callback_edges.setdefault(self.current, {})[candidate] = node.token
            summary = self.mutations.get(candidate)
            if summary is None:
                summary = self.whole_parameter_summary(contract)
            for index, path, _ in summary:
                writes.add((index, path))
        mutations = tuple((index, path, None) for index, path in sorted(writes))
        selector = f"lan_callback_selector_{node.token.line}_{node.token.col}"
        pointer = f"lan_callback_{node.token.line}_{node.token.col}"
        pointer_type = self.helper("callback_fnptr", None, self.functions[ty.candidates[0]][1])
        self.helper("global_temp", None, ("uint16_t", selector))
        self.helper("global_temp", None, (pointer_type, pointer))
        choice = f"(__builtin_trap(), ({pointer_type})0)"
        for candidate in reversed(ty.candidates):
            number = self.callable_selectors[candidate]
            choice = f"{selector} == {number} ? &{self.function_c_name(candidate)} : {choice}"
        prefix = f"{selector} = {selected.code}, {pointer} = ({choice})"
        chain_parameters = set.intersection(*(set(self.owned_chain_parameters.get(name, ()))
                                              for name in ty.candidates))
        return FunctionBinding(contract, pointer, mutations=mutations,
                               owned_chain_parameters=tuple(sorted(chain_parameters))), prefix

    def propagated_error_paths(self, token, paths):
        if self.initializing_heap_records:
            freed = {sym.uid: frozenset(("Tombstone",)) for sym in self.initializing_heap_records}
            paths = [replace(path, heap=path.heap | freed) for path in paths]
        self.check_heap_exit(paths, 0, token)
        return paths

    def propagation_cleanup(self):
        return "".join(f"free({sym.c_var}); "
                       for sym in reversed(self.initializing_heap_records))

    def propagate_composite_call(self, node, state, capture_arguments, capture_argument_types, capture_early_return):
        operand, = node.args
        early = []
        result, code = self.call(operand, state, capture_arguments,
                                 capture_argument_types=capture_argument_types, capture_early_return=early)
        if early:
            if capture_early_return is None:
                node.token.error("this expression cannot propagate an error here")
            capture_early_return.extend(early)
            return result.success, code
        token = node.token
        target = self.functions[self.current][1].result
        error = result.error
        temporary = f"lan_result_{token.line}_{token.col}"
        expression = f"{temporary}.payload.error"
        if isinstance(target, TaggedResult):
            if target.error.enum.identity != error.enum.identity or not error.domain.within(target.error.domain):
                token.error("propagated error must fit the caller's named error arm")
            expression = target.wrap(expression, success=False)
        elif (not isinstance(target, Scalar) or target.result_arms is None or target.floating
              or error.enum.identity not in target.result_error_enums
              or not error.domain.within(target.result_arms[1])):
            token.error("composite '?' requires an explicit result contract with a compatible named error arm")
        paths = self.propagated_error_paths(token, [state])
        if isinstance(target, Scalar):
            value = Value(error.domain, expression, error.bits, signed=error.signed)
            for path in paths:
                self.check_result_bounds(token, value, path)
                self.result_exit_domains.append(error.domain)
        leaving = [path.guard(f"error@{token.line}:{token.col}") for path in paths]
        self.trace_stack_exit(leaving, 0, "error return")
        self.return_states.extend(leaving)
        self.trace.append(f"  propagate tagged error {error.domain}; continue with {result.success.c}")
        location = f"@{token.line}:{token.col}"
        self.states(f"propagate {location} return error {error.domain}", paths)
        self.states(f"propagate {location} continue with composite success", [state])
        code = (f"({{ {result.c} {temporary} = {code}; if (!{temporary}.ok) "
                f"{{ {self.propagation_cleanup()}return {expression}; }} {temporary}.payload.value; }})")
        return result.success, self.assume_composite_result(operand, result.success, code)

    def call(self, node, state, capture_arguments=None, *, capture_result_relation=None, guard_places=None,
             capture_argument_types=None, capture_early_return=None, capture_return_cases=None):
        if node.kind == "propagate":
            return self.propagate_composite_call(node, state, capture_arguments,
                                                 capture_argument_types, capture_early_return)
        callee, args = node.args
        name = callee.args[0] if callee.kind == "name" else "<callback>"
        if name == "free":
            node.token.error("free is a statement and does not produce a value")
        callback = self.callback_type(callee) if not self.named_function(name) else None
        prefix = ""
        foreign = False
        binding = None
        if isinstance(callback, ExternCallable):
            name = "<extern_callback>"
            place = self.place(callee)
            if place is None:
                callee.token.error("extern callable invocation requires a parameter, variable or field")
            if callback.nullable and self.null_symbol in state.borrows.get(place.uid, (self.null_symbol,)):
                callee.token.error(f"cannot prove nullable extern callable {place.name!r} is non-null before calling")
            binding = FunctionBinding(
                callback.contract, place.c, foreign=True,
                mutations=callback.mutations, memory=callback.memory,
                external_effect=callback.external_effect,
            )
            contract, c_name, foreign, mutations = binding.contract, binding.c_name, True, binding.mutations
        elif callback is not None:
            name = "<callback>"
            selected = self.expr(callee, state)
            if selected.always_returns:
                if capture_early_return is None:
                    callee.token.error("this callback selection cannot propagate an error here")
                capture_early_return.append(selected)
                return callback.contract.result, selected.code
            self.range_check(callee.token, selected, callback)
            binding, prefix = self.callback_binding(node, callback, selected)
            contract, c_name, mutations = binding.contract, binding.c_name, binding.mutations
        elif any(name in scope for scope in self.scope):
            callee.token.error(f"local {name!r} is not callable")
        elif name in self.functions:
            contract, c_name = self.functions[name][1], self.function_c_name(name)
            self.calls[self.current].add(name)
            mutations = self.mutations[name]
        elif self.named_function(name):
            binding = self.externs.get(name) or self.imported_functions[name]
            contract, c_name, foreign = binding.contract, binding.c_name, binding.foreign
            mutations = binding.mutations if binding.mutations is not None else self.unknown_mutations(contract)
        elif callee.kind == "name":
            callee.token.error(f"unknown function {name!r}; it must be declared or imported")
        else:
            callee.token.error("expected a declared function or a callback value")
        same_group = name in self.recursive_groups and self.current in self.recursive_groups[name]
        chain_parameters = (binding.owned_chain_parameters if binding is not None
                            else self.owned_chain_parameters.get(name, ()))
        if binding is not None:
            storage_counts, return_observation = binding.storage_counts, binding.return_observation
        elif not same_group:
            storage_counts = self.storage_counts.get(name, ())
            return_observation = self.call_return_observations.get(name)
        else:
            # Summary facts must not justify the checks that establish them.
            storage_counts, return_observation = (), None
        if capture_return_cases is not None and return_observation is not None and guard_places is None:
            guard_places = {}
        if binding is not None and self.current:
            if binding.memory:
                self.memory_effects.add(self.current)
            if binding.external_effect:
                self.external_effects.add(self.current)
        if len(args) != len(contract.params):
            node.token.error(f"{name!r} expects {len(contract.params)} arguments")
        codes, arguments, argument_values = [], [], {}
        checked_argument_lengths = False
        changed_parameters = {index for index, _, _ in mutations}
        result_type = contract.result
        result_domain = binding.result_domain if binding is not None else self.result_domains.get(name)
        if result_domain is not None:
            result_type = result_type.narrow_domain(result_domain)
        payload = success_type(result_type)
        if isinstance(payload, Slice) and not payload.readonly and payload.source_param is not None:
            changed_parameters.add(payload.source_param)
        if isinstance(result_type, InterfaceResult) and not result_type.interface.readonly:
            changed_parameters.add(result_type.source)
        alias_sources = {source for source, _ in payload.origins()} if isinstance(payload, AliasResult) else set()
        if alias_sources and not payload.alias.readonly:
            changed_parameters.update(alias_sources)
        def argument_exit(value):
            parts = [part.rstrip(";") for part in (prefix, *(f"(void)({code})" for code in codes)) if part]
            value = replace(value, code=f"({', '.join(parts)}, ({value.code}))") if parts else value
            capture_early_return.append(value)
            return result_type, value.code

        def span_argument_exit(span):
            return argument_exit(replace(span.length, code=f"((void)({span.c_value()}), 0)",
                                         effectful=True))

        def address_argument(source, arg):
            propagate = capture_early_return is not None
            effectful_link = propagate and self.has_propagation(arg)
            if source.projection is not None:
                return self.capture_projected_argument(source, state, arg.token,
                                                       allow_effectful_links=effectful_link,
                                                       propagate_exit=propagate)
            if source.link_access is not None:
                return self.comparison_place(source, state, allow_effectful_links=effectful_link,
                                             propagate_exit=effectful_link)
            return source

        def capture_linked_place(source):
            if source.link_access is not None:
                targets = self.physical_places(source, state, allow_null=True)
                return self.capture_alias_argument(source, targets, f"&({source.c})")
            return source

        alias_exit = [] if capture_early_return is not None else None
        handoffs = {}
        unnamed_owner = False
        for index, (arg, ty) in enumerate(zip(args, contract.params)):
            if unnamed_owner and self.has_propagation(arg):
                arg.token.error("argument can propagate before an unnamed heap result is handed to the callee")
            if isinstance(ty, (HeapRecord, HeapObject, HeapArray)):
                if arg.kind != "name":
                    self.consume_temporary(arg, ty, index, name, state, arguments, codes,
                                           accepts_chain=index in chain_parameters)
                    unnamed_owner = True
                    continue
                source = self.place(arg)
                if (source is None or not self.is_whole_heap_object(source)
                        or source.readonly or source.uid in self.alias_members):
                    arg.token.error("a consuming parameter requires a live mutable heap owner")
                if source.uid in handoffs or state.heap.get(source.uid) != frozenset(("Alive",)):
                    arg.token.error(f"heap entity {source.name!r} may already be freed or moved")
                if isinstance(ty, HeapArray):
                    if not isinstance(source.ty, HeapArray) or source.ty.element != ty.element:
                        arg.token.error("a consuming array parameter requires a matching owning array")
                    if not state.values[source.uid].within(ty.domain):
                        arg.token.error("cannot prove the consumed array length fits its contract")
                    code = source.c
                else:
                    expected = ty.record if isinstance(ty, HeapRecord) else ty.base
                    if source.ty != expected or source.heap_owner != source.uid:
                        arg.token.error("a consuming parameter requires the same heap type")
                    code = f"&({source.c})"
                arguments.append(source)
                codes.append(code)
                handoffs[source.uid] = index, source
                continue
            record_source = self.place(arg) if isinstance(ty, BorrowRecord) and arg.kind != "null" else None
            record_link = record_source is not None and isinstance(record_source.ty, RecordLink)
            if (index in alias_sources and not isinstance(ty, Slice)) or record_link:
                source = (record_source if record_link else self.place(arg)) if arg.kind != "null" else None
                if (arg.kind != "null" and source is None and not (
                        isinstance(ty, BorrowScalar) and arg.kind == "index"
                        and self.span_kind(arg.args[0]) == "ordinary")):
                    arg.token.error("alias result source argument requires addressable storage")
                parameter = replace(ty, readonly=True) if index not in changed_parameters else ty
                targets, code = self.alias_source(arg, state, parameter, arg.token, capture_exit=alias_exit)
                if alias_exit:
                    return argument_exit(alias_exit[0])
                for target in targets - {self.null_symbol}:
                    self.require_target_live(target, state, arg.token)
                arguments.append(self.null_symbol if arg.kind == "null"
                                 else self.capture_alias_argument(source, targets, code))
                codes.append(f"({ty.c})({code})")
            elif isinstance(ty, (BorrowScalar, BorrowArray)) and ty.nullable:
                if (arg.kind != "null" and self.place(arg) is None and not (
                        isinstance(ty, BorrowScalar) and arg.kind == "index"
                        and self.span_kind(arg.args[0]) == "ordinary")):
                    arg.token.error("bind an alias result before passing it as an argument")
                parameter = replace(ty, readonly=True) if index not in changed_parameters else ty
                targets, code = self.alias_source(arg, state, parameter, arg.token, capture_exit=alias_exit)
                if alias_exit:
                    return argument_exit(alias_exit[0])
                for target in targets - {self.null_symbol}:
                    self.require_target_live(target, state, arg.token)
                source = (self.null_symbol if arg.kind == "null"
                          else self.capture_alias_argument(self.place(arg), targets, code))
                arguments.append(source)
                codes.append(f"({ty.c})({code})")
            elif isinstance(ty, Interface):
                early = [] if capture_early_return is not None else None
                source, code = self.interface_argument(arg, state, ty, capture_exit=early)
                if early:
                    return argument_exit(early[0])
                arguments.append(source)
                codes.append(code)
            elif isinstance(ty, (CopyRecord, CopyArray)) and arg.kind == "cast" and arg.args[1] == "view":
                operand, _, annotation = arg.args
                target = ty.record if isinstance(ty, CopyRecord) else ty.array
                if self.resolve(annotation) != target:
                    arg.token.error("composite bit view argument requires the declared parameter type")
                early = [] if capture_early_return is not None else None
                source, address, _ = self.composite_view_source(target, operand, [state], arg.token,
                                                                capture_exit=early)
                if early:
                    return argument_exit(early[0][1])
                destination = "temp.data" if isinstance(ty, CopyArray) else "temp"
                pointer = destination if isinstance(ty, CopyArray) else f"&{destination}"
                copy = self.helper("disjoint_object_copy", None)
                helper = self.helper("copy_view_arg", ty.c, (source.c, destination, pointer, copy))
                code = f"{helper}({address})"
                arguments.append(None)
                codes.append(code)
            elif isinstance(ty, CopyArray) and isinstance(self.composite_call_type(arg), CopyArray):
                source, paths = self.composite_call_view_source(arg, [state])
                actual = source.ty
                target = ty.array
                self.check_array_call_argument(arg, actual, target)
                state.absorb(paths[0])
                length = Value(Domain.point(actual.length), c_int(actual.length))
                arguments.append(source if actual.packed_bits else Span(
                    actual.element, length, source.depth, host=source, readonly=True))
                copy = self.helper("disjoint_object_copy", None)
                helper = self.helper("copy_array_arg", ty.c, copy)
                codes.append(f"{helper}({source.c})")
            elif isinstance(ty, (CopyRecord, BorrowRecord)):
                if arg.kind == "null":
                    if not isinstance(ty, BorrowRecord) or not ty.nullable:
                        arg.token.error("null requires a nullable alias parameter")
                    arguments.append(self.null_symbol)
                    codes.append("NULL")
                    continue
                source = self.record_place(arg)
                if source is None or source.ty != ty.record:
                    arg.token.error("record argument requires the same record type")
                source = address_argument(source, arg)
                if source.projection is not None and source.projection.always_returns:
                    return argument_exit(Value(ZERO, source.c, effectful=True, always_returns=True))
                if not isinstance(ty, BorrowRecord) or not ty.nullable:
                    self.require_live(source, state, arg.token)
                else:
                    for target in self.physical_places(source, state):
                        self.require_target_live(target, state, arg.token, access=source)
                if source.readonly and index in changed_parameters:
                    arg.token.error("cannot pass an immutable record to a mutating parameter")
                captured = source
                if isinstance(ty, BorrowRecord):
                    captured = capture_linked_place(source)
                    alias = self.alias_members.get(source.uid)
                    if (captured.captured_places is None and captured.projection is None and alias is not None
                            and isinstance(self.symbols[alias[0] - 1].ty, Interface)):
                        # Later arguments or the callee can rebind the interface
                        # without changing the address passed for this argument.
                        captured = replace(captured, captured_places=self.physical_places(source, state, allow_null=True))
                arguments.append(captured)
                codes.append(f"({ty.c})&({source.c})" if isinstance(ty, BorrowRecord) else source.c)
            elif isinstance(ty, (BorrowArray, CopyArray)):
                if ty.array.packed_bits:
                    source = self.place(arg)
                    if source is None or source.ty != ty.array:
                        arg.token.error("packed array parameter requires the same fixed bit array type")
                    source = address_argument(source, arg)
                    if source.projection is not None and source.projection.always_returns:
                        return argument_exit(Value(ZERO, source.c, effectful=True, always_returns=True))
                    self.require_live(source, state, arg.token)
                    code, readonly = source.c, source.readonly
                    if isinstance(ty, BorrowArray):
                        source = capture_linked_place(source)
                else:
                    source = self.ordinary_span(arg, state)
                    if source.length.always_returns and capture_early_return is not None:
                        return span_argument_exit(source)
                    readonly = source.readonly
                    array_contract = Slice(ty.array.element, length_bound=Domain.point(ty.array.length), readonly=True)
                    if self.debug_bounds:
                        source = self.checked_slice_length(arg.token, source, array_contract, "array argument")
                        checked_argument_lengths = True
                    source = self.check_span_contract(arg.token, source, array_contract)
                    source = replace(source, readonly=readonly)
                    code = source.c_data()
                if readonly and index in changed_parameters:
                    arg.token.error("cannot pass an immutable array to a mutating parameter")
                arguments.append(source)
                if isinstance(ty, CopyArray):
                    copy = self.helper("disjoint_object_copy", None)
                    helper = self.helper("copy_array_arg", ty.c, copy)
                    code = f"{helper}({code})"
                else:
                    code = f"({ty.c})({code})"
                codes.append(code)
            elif isinstance(ty, BorrowScalar):
                source = self.place(arg)
                if source is not None and isinstance(source.ty, Scalar):
                    source = address_argument(source, arg)
                    if source.projection is not None and source.projection.always_returns:
                        return argument_exit(Value(ZERO, source.c, effectful=True, always_returns=True))
                    self.require_live(source, state, arg.token)
                    value = (self.storage_value(source, state)
                             if source.projection is not None or source.link_access is not None
                             else self.expr(arg, state))
                    element, readonly, code = source.ty, source.readonly, f"&({source.c})"
                    source = capture_linked_place(source)
                elif arg.kind == "index" and self.span_kind(arg.args[0]) == "ordinary":
                    index_values = []
                    access = self.subscript(arg, state, capture_index=index_values,
                                            propagate_exit=capture_early_return is not None)
                    if isinstance(access, Value):
                        return argument_exit(access)
                    source, position, indices = access
                    element, readonly = source.element, source.readonly
                    code = f"&({source.c_data()}[{position}])"
                    host, subscript = arg.args
                    pure_index = not (self.span_has_effects(host, source) or self.has_effects(subscript))
                    start = self.add_span_start(source, index_values[0], state, arg.token) if pure_index else None
                    source = replace(source, origins=self.shift_positions(source, indices), start=start,
                                     length=Value(Domain.point(1), "1"))
                    value = Value(element.domain, "", element.bits, element.boolean,
                                  enum=element.enum, character=element.character)
                else:
                    arg.token.error("scalar alias argument requires a variable, field or ordinary array element")
                if index in changed_parameters and (readonly or element != ty.scalar):
                    arg.token.error("writable scalar alias requires mutable storage with the same scalar contract")
                if not element.domain.within(ty.scalar.domain):
                    arg.token.error("scalar alias must accept the storage's full declared domain")
                if (element.bits, element.signed, element.floating) != (ty.scalar.bits, ty.scalar.signed, ty.scalar.floating):
                    arg.token.error("scalar alias requires the same physical representation")
                self.range_check(arg.token, value, ty.scalar)
                arguments.append(source)
                codes.append(f"({ty.c})({code})")
            elif isinstance(ty, ExternCallable):
                if arg.kind == "null":
                    if not ty.nullable:
                        arg.token.error("null requires a nullable extern callable parameter")
                    arguments.append(self.null_symbol)
                    codes.append("NULL")
                elif arg.kind == "callback_ref":
                    fn_token = arg.args[0]
                    fn_name = fn_token.text
                    if fn_name in self.functions:
                        fn_contract = self.functions[fn_name][1]
                        if any(self.c_param_type(p, True) != self.c_param_type(p, False) for p in fn_contract.params):
                            arg.token.error("Lan function with slice parameter cannot be passed directly as a C function pointer")
                        if fn_contract.result is not None and isinstance(fn_contract.result, Slice) and fn_contract.result.text == "c":
                            arg.token.error("Lan function returning c_string slice cannot be passed directly as a C function pointer")
                        fn_writes = self.mutations.get(fn_name, ())
                        fn_mem, fn_ext = self.function_effects(fn_name)
                        fn_c = f"&{self.function_c_name(fn_name)}"
                    elif fn_name in self.externs:
                        ext = self.externs[fn_name]
                        fn_contract = ext.contract
                        fn_writes = ext.mutations or ()
                        fn_mem, fn_ext = ext.memory, ext.external_effect
                        fn_c = f"&{ext.c_name}"
                    else:
                        fn_token.error(f"unknown callback target {fn_name!r}")
                    if encode_interface_type(fn_contract) != encode_interface_type(ty.contract):
                        arg.token.error("callback contract does not match extern callable parameter")
                    allowed_writes = {(i, p) for i, p, _ in ty.mutations}
                    actual_writes = {(i, p) for i, p, _ in fn_writes}
                    if not actual_writes.issubset(allowed_writes) or (fn_mem and not ty.memory) or (fn_ext and not ty.external_effect):
                        arg.token.error("callback write set exceeds extern callable parameter effects")
                    arguments.append(Symbol("#fnptr", 0, ty, 0, fn_c))
                    codes.append(fn_c)
                else:
                    source = self.place(arg)
                    if source is None or not isinstance(source.ty, ExternCallable):
                        arg.token.error("extern callable argument requires a matching extern callable, callback(fn), or null")
                    if encode_interface_type(source.ty.contract) != encode_interface_type(ty.contract):
                        arg.token.error("extern callable contract does not match parameter")
                    allowed_writes = {(i, p) for i, p, _ in ty.mutations}
                    actual_writes = {(i, p) for i, p, _ in source.ty.mutations}
                    if (not actual_writes.issubset(allowed_writes)
                            or (source.ty.memory and not ty.memory)
                            or (source.ty.external_effect and not ty.external_effect)):
                        arg.token.error("extern callable write set exceeds parameter effects")
                    if not ty.nullable and source.ty.nullable and self.null_symbol in state.borrows.get(source.uid, (self.null_symbol,)):
                        arg.token.error(f"cannot prove nullable extern callable {source.name!r} is non-null")
                    arguments.append(source)
                    codes.append(source.c)
            elif isinstance(ty, Slice):
                expected_enc = ty.text if isinstance(ty.text, TextEncoding) else None
                span = self.ordinary_span(arg, state, expected_encoding=expected_enc)
                if span.length.always_returns and capture_early_return is not None:
                    return span_argument_exit(span)
                if self.debug_bounds and ty.length_bound is not None:
                    span = self.checked_slice_length(arg.token, span, ty, "argument")
                    checked_argument_lengths = True
                # The descriptor is captured now, but its contents are read at the call.
                descriptor = replace(ty, readonly=ty.readonly or index not in changed_parameters)
                span = self.check_span_contract(arg.token, span, descriptor, state, descriptor_only=True)
                arguments.append(span)
                code = span.c_data() if foreign and not ty.copy else span.c_value()
                codes.append(f"(const char *)({code})" if foreign and ty.text == "c" and not ty.copy else code)
            else:
                value = self.expr(arg, state)
                if (value.always_returns and capture_early_return is not None
                        and id(node) in self.propagating_expression_nodes):
                    return argument_exit(value)
                value = self.coerce_storage(arg.token, value, unqualified(ty))
                if plain_integer_parameter(ty):
                    value = self.ssa_narrow(arg, value, unqualified(ty).domain, "operands", "argument")
                self.range_check(arg.token, value, unqualified(ty))
                argument_values[index] = value
                if guard_places is not None and arg.kind == "name":
                    place = self.place(arg)
                    if (place is not None and isinstance(place.ty, Scalar) and place.uid not in self.alias_members
                            and place.projection is None and place.link_access is None and not place.bit_overlay):
                        guard_places[index] = (place.uid, state.epochs.get(place.uid))
                arguments.append(value)
                codes.append(value.code)
        # The callee acquires its owners only after every argument has been
        # evaluated. A propagating argument can return before this point.
        for index, source in handoffs.values():
            if state.heap.get(source.uid) != frozenset(("Alive",)):
                args[index].token.error(f"heap entity {source.name!r} may already be freed or moved")
            ty = contract.params[index]
            if index in chain_parameters:
                owners, _ = self.owned_chain_transfer(ty, source, state, args[index].token)
            else:
                self.require_terminal_owner(ty, source, state, args[index].token)
                owners = (source,)
            guard = state.guard_label()
            self.trace.append(f"  transfer ownership id#{source.uid} {source.name} [{guard}] "
                              f"to param#{index} {name}")
            for owner in owners:
                self.trace.append(f"  mark tombstone id#{owner.uid} {owner.name} [{guard}]: handed to callee")
                self.consume_parameter(owner, "move")
                self.note_transition_write(owner, lifetime=True)
            state.absorb(replace(state, heap=state.heap | {
                owner.uid: frozenset(("Tombstone",)) for owner in owners}))
        checked_slice_elements = []
        for index, (arg, ty, argument) in enumerate(zip(args, contract.params, arguments)):
            if isinstance(ty, (BorrowScalar, BorrowArray, BorrowRecord, Slice, Interface)):
                owners = argument.owners if isinstance(argument, Span) else self.physical_places(argument, state)
                for owner in owners:
                    if owner.uid:
                        self.require_target_live(owner, state, arg.token)
            if isinstance(ty, Slice) and ty.graph:
                self.verify_live_graph(self.span_link_targets(argument, state), state, arg.token)
            projected_record = isinstance(ty, BorrowRecord) and argument.projection is not None
            if isinstance(ty, BorrowRecord) and ty.graph:
                roots = (self.record_link_targets(argument, state) if projected_record
                         else self.physical_places(argument, state))
                self.verify_live_graph(roots, state, arg.token)
            if isinstance(ty, BorrowRecord) and ty.chain_limit is not None:
                roots = (self.record_link_targets(argument, state) if projected_record
                         else self.physical_places(argument, state, allow_null=True))
                for target in roots:
                    self.verify_chain(target, self.null_symbol, state, arg.token)
            if isinstance(ty, Slice) and ty.chain_limit is not None:
                for target in self.span_link_targets(argument, state):
                    if not target.uid:
                        continue
                    self.verify_chain(target, self.null_symbol, state, arg.token)
            if isinstance(ty, Slice) and (ty.content_bound is not None or ty.text == 'c'):
                runtime_elements = self.debug_bounds and ty.content_bound is not None
                checked = self.check_span_contract(arg.token, argument, replace(ty, readonly=True), state,
                                                   runtime_elements=runtime_elements)
                arguments[index] = replace(checked, readonly=argument.readonly)
                if runtime_elements:
                    checked_slice_elements.append(index)
                    self.trace.append(f"  debug check slice argument elements at {arg.token.line}:{arg.token.col}")
        graph_parameters, scalar_borrows = graph_scalar_borrows(contract.params)
        if scalar_borrows:
            graph_places = []
            for index in graph_parameters:
                argument = arguments[index]
                roots = self.physical_places(argument, state)
                if argument.projection is not None:
                    roots += tuple(self.record_link_targets(argument, state))
                graph_places.extend(self.graph_nodes(roots, state, args[index].token))
            for index in scalar_borrows:
                argument = arguments[index]
                places = (argument.owners if isinstance(argument, Span)
                          else self.physical_places(argument, state))
                for graph in graph_places:
                    root = graph.projection.root if graph.projection is not None else graph
                    for place in places:
                        base = place.projection.root if place.projection is not None else place
                        if (not base.uid or self.records_may_overlap(graph, place)
                                or ((root.borrowed or root.uid in self.param_places)
                                    and (base.borrowed or base.uid in self.param_places))):
                            args[index].token.error("cannot prove scalar borrow is disjoint from the borrowed graph")
        for head, donor in chain_attachments(contract.params):
            receiver, source = arguments[head], arguments[donor]
            head_type = contract.params[head]
            if ((source.projection is not None and source.projection.captured is None)
                    or not isinstance(receiver, Span) and receiver.projection is not None
                    and receiver.projection.captured is None):
                node.token.error("attachment requires captured addressable chains")
            if isinstance(receiver, Span):
                element = head_type.element
                receivers = self.span_link_targets(receiver, state)
                depths = [owner.depth for owner in receiver.owners]
            else:
                element = head_type.record
                receivers = self.physical_places(receiver, state)
                depths = [root.depth for root in receivers]
            donors = self.physical_places(source, state)
            if head_type.graph:
                self.check_graph_attachment(receiver, source, element, receivers, donors, state, node.token)
                continue
            separate_roots = (
                len(receivers) == len(donors) == 1 and self.disjoint_projected_records(receiver, source, state, node.token))
            if separate_roots:
                left_root, right_root = receivers[0], donors[0]
                left_links = self.link_targets(self.child_place(left_root, ("next",)), state)
                right_links = self.link_targets(self.child_place(right_root, ("next",)), state)
                left_children = set(self.chain_nodes(state, left_links, node.token))
                right_children = set(self.chain_nodes(state, right_links, node.token))
                first = left_children | {left_root}
                overlap = self.attachment_descendants_overlap(left_root, right_root, left_children, right_children)
            else:
                first = set(self.chain_nodes(state, receivers, node.token))
                second = set(self.chain_nodes(state, donors, node.token))
                heads = (self.span_element_places(receiver, Domain.of((0, receiver.length.domain.hi - 1)), state)
                         if isinstance(receiver, Span) and receiver.length.domain.hi else ())
                overlap = any(self.records_may_overlap(left, right) for left in (*first, *heads) for right in second)
            if overlap:
                node.token.error("attachment requires disjoint chains")
            for index, names, _ in mutations:
                if index != head or not names:
                    continue
                if names == ("@attach_deep",):
                    depths.extend(root.depth for root in first)
                elif names[0] == "@attach_at":
                    hops = int(names[1]) - (2 if isinstance(receiver, Span) else 1)
                    depths.extend(root.depth for root in self.chain_nodes_at(state, receivers, hops, node.token))
            if depths and any(target.depth > min(depths) for target in donors):
                node.token.error("attached chain would outlive its target storage")
            if not unqualified(dict(element.fields)["next"]).readonly and any(target.readonly for target in donors):
                node.token.error("attachment cannot create a writable link to an immutable record")
        result = result_type
        result_limits = {}
        checked_bounds = tuple(bound for bound in contract.bounds if bound[0] < len(args)) if self.debug_bounds else ()
        checked_expressions = contract.parameter_expressions if self.debug_bounds else ()
        for target, source, length, op, delta in contract.bounds:
            if length:
                argument = arguments[source]
                bound = (argument.length if isinstance(argument, Span) else
                         Value(dependent_source_domain(contract.params[source], True), "0"))
            else:
                bound = argument_values[source]
            limit = self.binary(node.token, "+", bound, Value(Domain.point(delta), c_int(delta)), state)
            if target == len(contract.params):
                bound_domain = (Domain.of((limit.domain.lo, inf)) if op == ">=" else
                                Domain.of((-inf, limit.domain.hi)))
                domain = result.domain & bound_domain
                if not domain:
                    node.token.error("dependent result interval is empty for this call")
                result = replace(result, domain=domain)
                if capture_result_relation is not None:
                    result_limits[source, length, delta, op] = self.capture_value(limit, state)
            elif not self.prove_order(node.token, argument_values[target], op, limit, state):
                if length or not self.ssa_call_order(
                        node, target, op, ("binary", "+", ("param", source), ("const", delta))):
                    if not self.debug_bounds:
                        args[target].token.error("cannot prove dependent parameter bound")
        inputs = dict(enumerate(arguments))
        for target, op, expression in contract.parameter_expressions:
            limit = self.integer_expression_value(node.token, expression, inputs, state)
            if not self.prove_order(node.token, argument_values[target], op, limit, state):
                if not self.ssa_call_order(node, target, op, expression):
                    if not self.debug_bounds:
                        args[target].token.error("cannot prove dependent parameter expression")
        for op, expression in contract.result_expressions:
            limit = self.integer_expression_value(node.token, expression, inputs, state)
            bound = Domain.of((limit.domain.lo, inf)) if op == ">=" else Domain.of((-inf, limit.domain.hi))
            domain = result.domain & bound
            if not domain:
                node.token.error("dependent result interval is empty for this call")
            result = replace(result, domain=domain)
            if capture_result_relation is not None:
                result_limits[expression, None, 0, op] = self.capture_value(limit, state)
        result_expression_state = (replace(state) if contract.result_expressions or checked_expressions
                                   or foreign and self.debug_bounds and isinstance(result, Slice) else None)
        if capture_arguments is not None:
            capture_arguments.extend(arguments)
        if capture_argument_types is not None:
            for index, parameter in enumerate(contract.params):
                if index in argument_values and plain_integer_parameter(parameter):
                    parameter = replace(unqualified(parameter), domain=argument_values[index].domain)
                elif isinstance(parameter, Slice):
                    parameter = replace(parameter, length_bound=arguments[index].length.domain)
                capture_argument_types.append(parameter)
        count_updates = self.capture_count_updates(state, storage_counts, arguments, node.token)
        self.apply_mutations(state, mutations, arguments, node.token, contract, args)
        self.remember_count_updates(state, count_updates)
        if return_observation is not None:
            try:
                rows = tuple(row for row in return_observation.cases if all(
                    argument_values[index].domain & requirement
                    for index, requirement in zip(return_observation.parameters, row.arguments)))
                narrowed = result.domain & domain_union(row.result for row in rows)
                if narrowed:
                    result = result.narrow_domain(narrowed)
                    if capture_return_cases is not None:
                        places = tuple(guard_places.get(index) for index in return_observation.parameters)
                        if any(place is not None for place in places):
                            capture_return_cases.append((rows, places))
            except DomainComplexity:
                pass
        if capture_result_relation is not None and result_limits:
            exact = next((self.fresh_value(limit, state)
                          for (source, length, delta, op), limit in result_limits.items()
                          if op == ">=" and (source, length, delta, "<=") in result_limits), None)
            bounds = [(op, self.fresh_value(limit, state))
                      for (_, _, _, op), limit in result_limits.items()]
            capture_result_relation.append((exact, bounds))
        # Chain length is derived, never declared, so an attachment summary
        # cannot exceed a source-level bound; only reachability is checked.
        diagonal = False
        if contract.diagonal_property and not any(self.has_effects(arg, allow_property_calls=True) for arg in args):
            equal_on_path = self.prove_order(node.token, argument_values[0], "==", argument_values[1], state)
            diagonal = self.property_relation_codegen is node and equal_on_path
            if not diagonal:
                shared = self.shared_property_state(node)
                terms_state = shared if shared is not None else state
                first = self.property_term(args[0], terms_state, [], use_saved=shared is not None)
                diagonal = (first is not None
                            and first == self.property_term(args[1], terms_state, [], use_saved=shared is not None))
            if equal_on_path and not diagonal:
                # A path-specific proof may refine analysis without removing
                # the call from C shared by paths with unequal arguments.
                result = replace(result, domain=result.domain & argument_values[0].domain)
        if contract.identity_property or diagonal:
            argument = self.fresh_value(argument_values[0], state)
            result = replace(result, domain=result.domain & argument.domain)
            if capture_result_relation is not None:
                if capture_result_relation:
                    _, bounds = capture_result_relation[0]
                    capture_result_relation[0] = (argument, bounds)
                else:
                    capture_result_relation.append((argument, []))
            self.trace.append(f"  apply property {name}: result == argument 0")
            return result, f"(({result.c})({codes[0]}))"
        accessor = (binding.interface_accessor if binding is not None else
                    self.function_interface_accessor(*self.functions[name], mutations))
        if accessor is not None:
            index, path, *arithmetic = accessor
            selector = arithmetic[3] if arithmetic and arithmetic[0] == "choose" else None
            selection = None
            if type(selector) is int:
                selection = argument_values[selector].domain
            elif isinstance(selector, tuple):
                selector_index, selector_path = selector
                source = arguments[selector_index]
                if (args[selector_index].kind == "name" and isinstance(source.ty, Record)
                        and source.uid not in self.alias_members and source.projection is None
                        and all(isinstance(step, str) for step in selector_path)):
                    field = self.child_place(source, selector_path)
                    selection = state.values.get(field.uid)
            selected = None
            if selection and selection.lo == selection.hi:
                selected = bool(selection.lo)
                if not selected:
                    index, path = arithmetic[2], arithmetic[1]
                arithmetic = []
                selector = None
            fields = [(index, path)]
            if arithmetic:
                other_index = arithmetic[2] if len(arithmetic) >= 3 else index
                fields.append((other_index, arithmetic[1]))
            if isinstance(selector, tuple):
                fields.append(selector)
            layouts = {i: self.known_interface_layout(args[i], arguments[i]) for i, _ in fields}
            if all(layout is not None for layout in layouts.values()):
                label = ".".join(map(str, path))
                branch = "" if selected is None else f" on known {'true' if selected else 'false'} branch"
                self.trace.append(f"  inline interface accessor {name}: field {label}{branch}")
                address = f"lan_inline_interface_{node.token.line}_{node.token.col}"
                indices = set()
                for _, steps in fields:
                    for step in steps:
                        if not isinstance(step, tuple):
                            continue
                        if step[0] == "from_end":
                            step = step[1]
                        if isinstance(step, tuple) and step[0] == "param":
                            indices.add(step[1])
                        elif isinstance(step, tuple) and step[0] == "affine":
                            indices.update(i for i, _ in step[1])
                subscripts = {i: f"{address}_index_{i}" for i in indices}

                def index_operand(step):
                    if not isinstance(step, tuple):
                        return str(step)
                    if step[0] == "affine":
                        terms = [f"(lan_int){subscripts[i]} * {c_int(scale)}" for i, scale in step[1]]
                        return " + ".join((*terms, c_int(step[2])))
                    operand = subscripts[step[1]]
                    if len(step) == 4:
                        return f"(lan_int){operand} * {c_int(step[3])} + {c_int(step[2])}"
                    if len(step) == 3:
                        return f"(lan_int){operand} + {c_int(step[2])}"
                    return operand

                def access_path(base, source_index, steps):
                    components = [base]
                    for depth, step in enumerate(steps):
                        if step == "@len":
                            container = interface_accessor_member(contract, source_index, steps[:depth])
                            if isinstance(container, Array):
                                components = [c_int(container.length)]
                            else:
                                components.append(".len")
                        elif isinstance(step, str):
                            components.append(f".{c_field(step)}")
                        else:
                            container = interface_accessor_member(contract, source_index, steps[:depth])
                            if isinstance(step, tuple) and step[0] == "from_end":
                                text = "".join(components)
                                distance = step[1]
                                offset = index_operand(distance) if isinstance(distance, tuple) else c_int(distance)
                                if isinstance(distance, tuple) and len(distance) > 2:
                                    offset = f"({offset})"
                                helper = self.helper("slice_from_end", container.element.c)
                                components = [f"{helper}({text}, {offset})"]
                                continue
                            if isinstance(container, Slice):
                                data = "".join(components)
                                components = [f"((const {container.element.c} *)({data}).data)"]
                            operand = index_operand(step)
                            if isinstance(container, Array) and container.packed_bits:
                                helper = self.helper("bit_read", None)
                                data = "".join(components)
                                components = [f"{helper}({data}, {operand})"]
                            else:
                                components.append(f"[{operand}]")
                    return "".join(components)

                def convert_field(code, member):
                    if member.c == result.c:
                        return code
                    if result.floating:
                        convert = self.helper("float_convert", None, (member.c, result.c))
                        return f"{convert}({code})"
                    return f"(({result.c})({code}))"

                setup, bases = [], {}
                flag = f"{address}_flag"
                capture = len(args) > 1 or bool(arithmetic) or "@len" in path
                for i, code in enumerate(codes):
                    if i in layouts:
                        source, layout = arguments[i], layouts[i]
                        concrete = isinstance(source.ty, Record) and args[i].kind != "call"
                        pointer = f"&({source.c})" if concrete else f"(const {layout.c} *)({code}).data"
                        if capture:
                            name_at = f"{address}_source_{i}"
                            self.helper("global_temp", None, (f"const {layout.c} *", name_at))
                            setup.append(f"{name_at} = {pointer}")
                            bases[i] = f"(*{name_at})"
                        else:
                            bases[i] = f"({source.c})" if concrete else f"(*({pointer}))"
                    elif i in subscripts:
                        self.helper("global_temp", None, (contract.params[i].c, subscripts[i]))
                        setup.append(f"{subscripts[i]} = {code}")
                    elif i == selector:
                        self.helper("global_temp", None, ("int", flag))
                        setup.append(f"{flag} = !!({code})")
                    else:
                        setup.append(f"(void)({code})")
                access = access_path(bases[index], index, path)
                member = interface_accessor_member(contract, index, path)
                if arithmetic:
                    other_access = access_path(bases[other_index], other_index, arithmetic[1])
                    op = INTERFACE_FIELD_COMPARISONS.get(arithmetic[0], arithmetic[0])
                    right = interface_accessor_member(contract, other_index, arithmetic[1])
                    if op == "choose":
                        condition = access_path(bases[selector[0]], selector[0], selector[1]) if isinstance(selector, tuple) else flag
                        access = convert_field(access, member)
                        other_access = convert_field(other_access, right)
                        access = f"(({condition}) ? ({access}) : ({other_access}))"
                    elif op in ("&&", "||"):
                        access = c_predicate_sequence((access, other_access), conjunction=op == "&&")
                    elif op == "^^":
                        access = f"((!!({access})) != (!!({other_access})))"
                    elif member.floating or right.floating:
                        if arithmetic[0] in INTERFACE_FIELD_COMPARISONS:
                            access = f"(({access}) {op} ({other_access}))"
                        else:
                            bits = max(ty.bits for ty in (member, right) if ty.floating)
                            helper = self.helper("float_binary", None, (op, bits))
                            access = f"(({result.c}){helper}({access}, {other_access}))"
                    elif op in ("<<", ">>"):
                        helper = self.helper("shift_left" if op == "<<" else "shift_right", None)
                        count = self.assumed_shift_count(node.token, other_access)
                        access = f"(({result.c}){helper}({access}, (unsigned)({count})))"
                    elif op in ("/", "%"):
                        divisor = self.assumed_divisor(node.token, other_access)
                        access = f"(({result.c})((lan_int)({access}) {op} ({divisor})))"
                    else:
                        access = f"(({result.c})((lan_int)({access}) {op} (lan_int)({other_access})))"
                else:
                    access = convert_field(access, member)
                return result, f"({', '.join(setup)}, {access})" if setup else access
        if contract.idempotent_property:
            inner = args[0]
            while inner.kind in ("call", "cast"):
                if inner.kind == "cast":
                    operand, strategy, annotation = inner.args
                    if strategy != "safe" or not plain_integer_parameter(self.resolve(annotation)):
                        break
                    inner = operand
                    continue
                inner_callee, inner_args = inner.args
                key = self.algebraic_function_key(inner_callee)
                if key == (c_name, contract):
                    self.trace.append(f"  apply property {name}: collapse nested call")
                    code = codes[0] if inner is args[0] else f"(({result.c})({codes[0]}))"
                    return result, code
                if key is None or not key[1].identity_property:
                    break
                inner = inner_args[0]
            if inner.kind == "call" and not self.has_effects(inner, allow_property_calls=True):
                term = self.property_term(inner, state, [])
                if term is not None and term[:2] == ("property", (c_name, contract)):
                    self.trace.append(f"  apply property {name}: collapse nested call")
                    return result, f"(({result.c})({codes[0]}))"
            shared = self.shared_property_state(node)
            if shared is not None and not self.has_effects(args[0], allow_property_calls=True):
                term = self.property_term(args[0], shared, [], use_saved=True)
                while term is not None and term[0] == "cast":
                    term = term[2]
                if term is not None and term[:2] == ("property", (c_name, contract)):
                    self.trace.append(f"  apply property {name}: reuse idempotent result")
                    return result, f"(({result.c})({codes[0]}))"
        if contract.commutative_property and not any(self.has_effects(arg) for arg in args):
            if codes[1] < codes[0]:
                codes.reverse()
                self.trace.append(f"  apply property {name}: canonicalize argument order")
        foreign_copies = {i for i, ty in enumerate(contract.params) if foreign and isinstance(ty, Slice) and ty.copy}
        result_bounds = [
            (source, length, op, delta)
            for target, source, length, op, delta in contract.bounds
            if target == len(contract.params)
        ]
        result_expressions = contract.result_expressions
        length_sources = {source for source, length, _, _ in result_bounds if length}
        length_sources.update(source for _, source, length, _, _ in checked_bounds if length)
        for _, expression in result_expressions:
            length_sources.update(offset_expression_lengths(expression))
        for _, _, expression in checked_expressions:
            length_sources.update(offset_expression_lengths(expression))
        alias_origins = ()
        if foreign and self.debug_bounds and isinstance(result, AliasResult):
            origins = result.origins()
            if all(isinstance(step, (str, int)) for _, path in origins for step in path):
                alias_origins = origins
                length_sources.update(source for source, _ in origins)
        slice_origin = (result.source_param if foreign and self.debug_bounds and isinstance(result, Slice) else None)
        if slice_origin is not None:
            length_sources.add(slice_origin)
            if result.offset_expression is not None:
                length_sources.update(offset_expression_lengths(result.offset_expression))
        writebacks = tuple(index for index, ty in enumerate(contract.params)
                           if foreign and self.debug_bounds and isinstance(ty, (BorrowScalar, BorrowRecord, BorrowArray, Slice))
                           and not ty.readonly and not (isinstance(ty, Slice) and ty.copy))
        length_sources.update(writebacks)
        length_sources.update(checked_slice_elements)
        foreign_lengths = {index for index in length_sources
                           if foreign and isinstance(contract.params[index], Slice) and index not in foreign_copies}
        if (prefix or foreign_copies or checked_slice_elements or checked_bounds or checked_expressions
                or result_bounds or result_expressions
                or alias_origins or slice_origin is not None
                or writebacks or len(args) > 1 and (
                    checked_argument_lengths or any(self.has_effects(arg) for arg in args))):
            temporaries, declarations, copies, snapshots = [], [], [], []
            for i, (ty, code) in enumerate(zip(contract.params, codes)):
                temp = f"lan_arg_{node.token.line}_{node.token.col}_{i}"
                snapshots.append(temp)
                if i in foreign_copies:
                    declarations.append(f"lan_slice {temp} = {code};")
                    owner = f"{temp}_copy"
                    copies.append(self.slice_clone(ty, temp, owner))
                    temporaries.append(f"({self.c_param_type(ty, foreign)}){owner}")
                elif i in foreign_lengths:
                    declarations.append(f"lan_slice {temp} = {arguments[i].c_value()};")
                    temporaries.append(f"({self.c_param_type(ty, foreign)}){temp}.data")
                else:
                    declarations.append(f"{self.c_param_type(ty, foreign)} {temp} = {code};")
                    temporaries.append(temp)
            expression = f"{c_name}({', '.join(temporaries)})"
            checks = [check for index in writebacks
                      if (check := self.native_writeback_check(node.token, contract.params[index], snapshots[index]))]
            if checks:
                saved = f"lan_native_writeback_{node.token.line}_{node.token.col}"
                if result is None:
                    expression = f"({expression}, 0)"
                    expression = self.assumed_result("int", saved, expression, checks, checked=True)
                    expression = f"((void){expression})"
                else:
                    native_type = "const char *" if isinstance(result, Slice) and result.text == "c" else result.c
                    expression = self.assumed_result(native_type, saved, expression, checks, checked=True)
            def length_code(index):
                ty = contract.params[index]
                return f"{snapshots[index]}.len" if isinstance(ty, Slice) else c_int(ty.array.length)

            def captured_dependency(dependency):
                runtime_inputs = {
                    index: replace(argument_values[index], code=snapshots[index], effectful=False)
                    for index in offset_expression_parameters(dependency)
                }
                for index in offset_expression_lengths(dependency):
                    original = arguments[index]
                    domain = dependent_source_domain(contract.params[index], True)
                    value = original.length if isinstance(original, Span) else Value(domain, "0")
                    runtime_inputs["len", index] = replace(value, code=length_code(index), effectful=False)
                return self.integer_expression_value(
                    node.token, dependency, runtime_inputs, result_expression_state).code

            if result_bounds or result_expressions:
                result_name = f"lan_dependent_result_{node.token.line}_{node.token.col}"
                predicates = []
                for source, length, op, delta in result_bounds:
                    source_code = length_code(source) if length else snapshots[source]
                    predicates.append(f"(lan_int){result_name} {op} (lan_int)({source_code}) + {c_int(delta)}")
                for op, dependency in result_expressions:
                    bound = captured_dependency(dependency)
                    predicates.append(f"(lan_int){result_name} {op} (lan_int)({bound})")
                expression = self.assumed_result(result.c, result_name, expression, predicates,
                                                 checked=foreign and self.debug_bounds)
            if alias_origins:
                pointer = f"lan_native_alias_{node.token.line}_{node.token.col}"
                graph_records = alias_graph_records(contract.params, result.alias, result.graph_records)
                tests = [self.native_alias_origin_predicate(contract.params[source], path, snapshots[source], pointer, graph_records)
                         for source, path in alias_origins]
                if result.alias.nullable:
                    tests.insert(0, f"{pointer} == NULL")
                predicate = c_predicate_sequence(tests, conjunction=False)
                expression = self.assumed_result(result.c, pointer, expression, [predicate], checked=True)
            if slice_origin is not None:
                descriptor = f"lan_native_origin_{node.token.line}_{node.token.col}"
                offset = f"lan_native_offset_{node.token.line}_{node.token.col}"
                if result.source_offset is not None:
                    offset_code = c_int(result.source_offset)
                elif slice_offset_terms(result) or result.offset_expression is not None:
                    inputs = {index: replace(value, code=snapshots[index], effectful=False)
                              for index, value in argument_values.items()}
                    if result.offset_expression is not None:
                        for index in offset_expression_lengths(result.offset_expression):
                            original = arguments[index]
                            length = (replace(original.length, code=length_code(index), effectful=False)
                                      if isinstance(original, Span) else
                                       Value(dependent_source_domain(contract.params[index], True),
                                             length_code(index)))
                            inputs["len", index] = length
                    offset_code = self.slice_offset_value(node.token, result, inputs, result_expression_state).code
                else:
                    offset = None
                checks = self.native_slice_origin_checks(contract.params[slice_origin], result,
                                                         snapshots[slice_origin], descriptor, offset)
                if result.text == "c":
                    expression = self.assumed_result("const char *", descriptor, expression, checks, checked=True)
                    available = f"{snapshots[slice_origin]}.len - (size_t){offset}"
                    expression = self.checked_native_c_string(node.token, result, expression, available)
                else:
                    expression = self.assumed_result(result.c, descriptor, expression, checks, checked=True)
                if offset is not None:
                    self.helper("global_temp", None, ("lan_int", offset))
                    expression = f"({offset} = {offset_code}, {expression})"
            prechecks = []
            if checked_slice_elements or checked_bounds or checked_expressions:
                check = self.helper("oracle_check", None)
                for index in checked_slice_elements:
                    ty = contract.params[index]
                    snapshot = snapshots[index]
                    predicate = self.membership(args[index].token, ty.content_bound)
                    scan = self.helper("debug_native_elements", ty.element.c, (predicate, False))
                    prechecks.extend([f"{check}({snapshot}.len == 0 || {snapshot}.data != NULL);",
                                      f"(void){scan}({snapshot});"])
                for target, source, length, op, delta in checked_bounds:
                    bound = length_code(source) if length else snapshots[source]
                    prechecks.append(f"{check}((lan_int){snapshots[target]} {op} "
                                     f"(lan_int)({bound}) + {c_int(delta)});")
                for target, op, dependency in checked_expressions:
                    bound = captured_dependency(dependency)
                    prechecks.append(f"{check}((lan_int){snapshots[target]} {op} (lan_int)({bound}));")
                if checked_bounds or checked_expressions:
                    self.trace.append(f"  debug check dependent arguments at {node.token.line}:{node.token.col}")
            steps = []
            if prefix:
                steps.append(prefix)
            for i, (ty, code) in enumerate(zip(contract.params, codes)):
                temp = snapshots[i]
                if i in foreign_copies:
                    self.helper("global_temp", None, ("lan_slice", temp))
                    steps.append(f"/*lan_slice*/ {temp} = {code}")
                elif i in foreign_lengths:
                    self.helper("global_temp", None, ("lan_slice", temp))
                    steps.append(f"/*lan_slice*/ {temp} = {arguments[i].c_value()}")
                else:
                    c_ty = self.c_param_type(ty, foreign)
                    self.helper("global_temp", None, (c_ty, temp))
                    steps.append(f"/*{c_ty}*/ {temp} = {code}")
            steps.extend(p.rstrip(";") for p in prechecks)
            steps.extend(c.rstrip(";") for c in copies)
            if foreign_copies:
                call_res = f"lan_call_copy_{node.token.line}_{node.token.col}"
                if result is None:
                    steps.append(expression)
                else:
                    ret_c = "const char *" if isinstance(result, Slice) and result.text == "c" else result.c
                    self.helper("global_temp", None, (ret_c, call_res))
                    steps.append(f"{call_res} = {expression}")
                for i in sorted(foreign_copies):
                    steps.append(f"free({snapshots[i]}_copy)")
                if result is not None:
                    steps.append(call_res)
            else:
                steps.append(expression)
            code = f"({', '.join(steps)})"
            if slice_origin is not None and result.text == "c":
                return result, code
            return result, self.assume_composite_result(node, result_type, code, foreign=foreign)
        code = f"{c_name}({', '.join(codes)})"
        return result, self.assume_composite_result(node, result_type, code, foreign=foreign)

    def has_effects(self, node, *, allow_property_calls=False, allow_debug_checks=False):
        pending = [node]
        while pending:
            value = pending.pop()
            if isinstance(value, Node):
                if value.kind == "call":
                    callee, _ = value.args
                    if not allow_property_calls or self.algebraic_contract(callee) is None:
                        return True
                if (value.kind == "propagate"
                        or self.debug_bounds and not allow_debug_checks and value.kind in ("index", "slice")):
                    return True
                if value.kind == "cast":
                    _, strategy, annotation = value.args
                    if strategy.endswith(("trap", "sat")):
                        return True
                    if (self.debug_bounds and not allow_debug_checks and strategy == "view"
                            and isinstance(self.resolve(annotation), Slice)):
                        return True
                pending.extend(value.args)
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return False
