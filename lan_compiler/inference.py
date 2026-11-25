"""Parameter, return and result inference."""

from __future__ import annotations

from dataclasses import field, fields, replace
from itertools import count

from float_domain import FloatDomain, FloatDomainError
from value_domain import Domain

from .syntax import Node
from .typesys import (AffineForm, AliasResult, Array, BorrowRecord, Contract, CopyArray, HeapArray,
                      HeapObject, HeapRecord, Interface, InterfaceResult, is_callback_enum, ReadOnly, Record,
                      RecordLink, Scalar, Slice, unqualified, Value, WIDTHS)
from .domain_utils import container, domain_union, physical, replay_scalar
from .interface_types import (alias_graph_intermediates, alias_graph_records, graph_or_chain_parameter,
                              linked_alias_path, linked_owned_record, mutable_chain_record,
                              slice_offset_terms, valid_alias_provenance, valid_chain_record,
                              valid_interface_provenance, valid_slice_provenance)
from .summaries import graph_write_path


class InferenceMixin:
    def infer_release_parameters(self, definitions, annotations):
        """Infer incoming allocation requirements before checking call bodies."""
        owned_types = (HeapArray, HeapObject, HeapRecord)
        requirements, forwards = {}, []

        def binding(node):
            anchor = getattr(node, "semantic_anchor", None)
            return None if anchor is None else self.semantic_module.operations[anchor].binding

        def is_explicit_owner(ty):
            if ty is None:
                return False
            while ty.kind == "type_group":
                ty = ty.args[0]
            if ty.kind == "storage":
                return ty.args[1].text == "heap"
            return ty.kind == "type_name" and isinstance(unqualified(self.resolve(ty)), owned_types)

        for name, node in definitions.items():
            contract = annotations[name]
            if contract.kind != "contract" or len(contract.args[0]) != len(node.args[2]):
                continue
            parameters = {self.semantic_module.declarations[token]: index
                          for index, token in enumerate(node.args[2])}
            operations = tuple(self.body_operations(node.args[3], live_only=True))
            reassigned = {binding(op.args[0]) for op in operations
                          if op.kind == "assign" and op.args[0].kind == "name"}
            aliases = {}
            for op in operations:
                if op.kind != "define" or binding(op) in reassigned:
                    continue
                _, annotation, initial = op.args
                if (annotation is not None and annotation.kind == "slice_qualified"
                        and annotation.args[3] == "alias" and initial is not None and initial.kind == "name"):
                    aliases[binding(op)] = binding(initial)

            def parameter_index(argument):
                if argument.kind != "name":
                    return None
                source = binding(argument)
                while source is not None and source in aliases:
                    source = aliases[source]
                return parameters.get(source)

            consumed = {index for index, (_, ty) in enumerate(contract.args[0]) if is_explicit_owner(ty)}
            requirements[name] = consumed
            for op in operations:
                if op.kind != "call":
                    continue
                callee, arguments = op.args
                if callee.kind != "name" or binding(callee) is not None:
                    continue
                target = callee.args[0]
                for position, argument in enumerate(arguments):
                    index = parameter_index(argument)
                    if index is None:
                        continue
                    if target == "free" and len(arguments) == 1:
                        consumed.add(index)
                    elif target in definitions:
                        # Calls take an actual owning handle. A local alias can
                        # identify the free target but cannot forward ownership.
                        if binding(argument) in parameters:
                            forwards.append((name, index, target, position))
                    elif (binding(argument) in parameters and target in self.imported_functions
                          and position in dict(self.imported_functions[target].owning)):
                        consumed.add(index)

        changed = True
        while changed:
            changed = False
            for name, index, target, position in forwards:
                if position in requirements.get(target, ()) and index not in requirements[name]:
                    requirements[name].add(index)
                    changed = True

        for name, indices in requirements.items():
            annotation = annotations[name]
            declarations = list(annotation.args[0])
            for index in sorted(indices):
                parameter, ty = declarations[index]
                if ty is None or is_explicit_owner(ty) or ty.kind == "storage":
                    continue
                if ty.kind == "slice_qualified" and (ty.args[2] is True or ty.args[3] is not None):
                    continue
                # This is an internal ABI choice. The ordinary ownership
                # checker still validates releases and all returning paths.
                storage = replace(ty.token, text="heap")
                declarations[index] = parameter, Node(ty.token, "storage", (ty, storage))
                self.trace.append(f"  inferred consuming parameter {name}.{parameter.text}")
            if tuple(declarations) != annotation.args[0]:
                annotations[name] = replace(annotation, args=(tuple(declarations), *annotation.args[1:]))

    @staticmethod
    def body_operations(body, *, live_only=False):
        pending = [body]
        while pending:
            item = pending.pop()
            if isinstance(item, (tuple, list)):
                if live_only and isinstance(item, list):
                    for index, statement in enumerate(item):
                        if isinstance(statement, Node) and statement.kind in ("return", "continue"):
                            item = item[:index + 1]
                            break
                pending.extend(reversed(item))
            elif isinstance(item, Node):
                yield item
                if live_only and item.kind == "if" and item.args[0].kind == "boolean":
                    condition, yes, no = item.args
                    pending.append(yes if condition.args[0] else no)
                elif not (live_only and item.kind == "while" and item.args[0].kind == "boolean"
                          and item.args[0].args[0] is False):
                    pending.extend(reversed(item.args))

    def recursive_chain_parameters(self, definitions, components):
        needed, incoming = set(), {}
        for name, node in definitions.items():
            _, _, params, body = node.args
            operations = list(self.body_operations(body, live_only=True))
            shadowed = {item.args[0].text for item in operations if item.kind == "define"}
            sources = {param.text: index for index, param in enumerate(params) if param.text not in shadowed}
            if body and body[0].kind == "chain_assert":
                head, tail = body[0].args
                if head.kind == "name" and tail.kind == "null":
                    needed.update((name, index) for index, param in enumerate(params) if param.text == head.args[0])
            for item in operations:
                if item.kind != "call":
                    continue
                callee, args = item.args
                if callee.kind != "name":
                    continue
                target = callee.args[0]
                if target in sources or target in shadowed or components.get(target) != components[name]:
                    continue
                for index, arg in enumerate(args):
                    follows_next = False
                    while arg.kind == "member" and arg.args[1].text == "next":
                        arg = arg.args[0]
                        follows_next = True
                    if arg.kind != "name" or arg.args[0] not in sources:
                        continue
                    source, destination = (name, sources[arg.args[0]]), (target, index)
                    incoming.setdefault(destination, set()).add(source)
                    if follows_next:
                        needed.update((source, destination))

        pending = list(needed)
        while pending:
            for source in incoming.get(pending.pop(), ()):
                if source not in needed:
                    needed.add(source)
                    pending.append(source)
        return needed

    def infer_chain_parameters(self, node, annotation, chain_parameters):
        if annotation.kind != "contract":
            return annotation
        function, _, params, _ = node.args
        declarations = annotation.args[0]
        if len(params) != len(declarations):
            return annotation
        needed = {index for index in range(len(params)) if (function.text, index) in chain_parameters}
        inferred = list(declarations)
        for index in sorted(needed):
            name, declared = declarations[index]
            if declared is None or declared.kind != "type_name":
                continue
            record = self.resolve(declared)
            if isinstance(record, Record) and valid_chain_record(record):
                chain = BorrowRecord(record, readonly=True, chain_limit=True)
                inferred[index] = (name, Node(declared.token, "inferred_parameter_type", (chain,)))
                self.trace.append(f"  inferred chain parameter {function.text}.{params[index].text}")
        if inferred == list(declarations):
            return annotation
        return replace(annotation, args=(tuple(inferred), *annotation.args[1:]))

    def infer_graph_parameters(self, node, annotation, *, recursive=False):
        if annotation.kind != "contract":
            return annotation
        function, _, parameters, body = node.args
        declarations = annotation.args[0]
        if len(parameters) != len(declarations):
            return annotation

        operations = list(self.body_operations(body))
        shadowed = {item.args[0].text for item in operations if item.kind == "define"}
        candidates = {}
        for index, (_, declared) in enumerate(declarations):
            if declared is None or parameters[index].text in shadowed:
                continue
            base = declared
            if declared.kind == "slice_qualified":
                base = declared.args[0]
                if base.kind not in ("type_name", "array", "slice_type") or declared.args[3] is not None:
                    continue
                if any(value is not None for value in declared.args[4:7]) or declared.args[7]:
                    continue
            elif base.kind not in ("type_name", "array", "slice_type"):
                continue
            if base.kind in ("array", "slice_type"):
                element = base.args[0]
                if (element.kind != "type_name" or element.args == ("bit",)
                        or not isinstance(self.resolve(element), Record)):
                    continue
            ty = self.resolve(declared)
            readonly = isinstance(ty, ReadOnly) or isinstance(ty, Slice) and ty.readonly
            ty = unqualified(ty)
            if isinstance(ty, Array) and isinstance(ty.element, Record):
                ty = Slice(ty.element, length_bound=Domain.point(ty.length))
            if isinstance(ty, Record) and self.contains_record_links(ty):
                lifted = BorrowRecord(ty, readonly=True, graph=True)
            elif (isinstance(ty, Slice) and isinstance(ty.element, Record)
                  and self.contains_record_links(ty.element) and not ty.copy):
                lifted = replace(ty, readonly=True, graph=True)
            else:
                continue
            candidates[parameters[index].text] = index, lifted, readonly
        if not candidates:
            return annotation

        def member_path(expr):
            fields = []
            while expr.kind == "member":
                expr, field = expr.args
                fields.append(field.text)
            indexed = expr.kind == "index"
            if indexed:
                expr = expr.args[0]
            if (fields and expr.kind == "name" and expr.args[0] in candidates
                    and indexed == isinstance(candidates[expr.args[0]][1], Slice)):
                return expr.args[0], tuple(reversed(fields))
            return None, ()

        def record(ty):
            return ty.element if isinstance(ty, Slice) else ty.record

        forwarded = {id(arg) for operation in operations if operation.kind == "call" for arg in operation.args[1]}
        used = {item.args[0] for item in operations if item.kind == "name"
                and (recursive or id(item) not in forwarded) and item.args[0] in candidates}
        writes, cuts, skips = set(), set(), set()
        donors = {}
        for operation in operations:
            if operation.kind == "member":
                name, _ = member_path(operation)
                if name is not None:
                    used.add(name)
            if operation.kind != "assign":
                continue
            target, op, value = operation.args
            name, route = member_path(target)
            if name is None:
                continue
            index, ty, readonly = candidates[name]
            if readonly:
                continue
            writes.add(name)
            if op.text != "=" or not graph_write_path(record(ty), route):
                continue
            if value.kind == "null":
                cuts.add(name)
            elif (valid_chain_record(record(ty)) and route[-1] == "next"
                  and member_path(value) == (name, (*route, "next"))):
                cuts.add(name)
                skips.add(name)
            elif value.kind == "name" and value.args[0] in candidates:
                donor = value.args[0]
                donor_index, donor_type, _ = candidates[donor]
                if (donor_index != index and isinstance(donor_type, BorrowRecord)
                        and graph_write_path(record(ty), route, donor_type.record)):
                    donors.setdefault(name, set()).add(donor)

        attachments = {name: next(iter(sources)) for name, sources in donors.items()
                       if len(sources) == 1 and next(iter(sources)) not in writes}
        used.update(attachments.values())
        inferred = list(declarations)
        for name in sorted(used):
            index, ty, _ = candidates[name]
            token, declared = declarations[index]
            donor = attachments.get(name)
            if donor is not None:
                _, donor_type, _ = candidates[donor]
                if ty.graph != donor_type.graph:
                    donor = None
            proposed = replace(ty, readonly=ty.readonly and name not in writes,
                               prune=ty.prune or name in cuts,
                               attach=ty.attach if ty.attach is not None or donor is None else candidates[donor][0])
            if name in skips and proposed.attach is None:
                proposed = replace(proposed, graph=False, chain_limit=True)
            if not proposed.readonly and not mutable_chain_record(record(proposed)):
                continue
            if isinstance(proposed, Slice) and proposed.prune and proposed.attach is not None:
                continue
            inferred[index] = token, Node(declared.token, "inferred_parameter_type", (proposed,))
            shape = "graph" if proposed.graph else "chain"
            self.trace.append(f"  inferred {shape} parameter {function.text}.{name}")
        if inferred == list(declarations):
            return annotation
        return replace(annotation, args=(tuple(inferred), *annotation.args[1:]))

    def infer_graph_forwarding_parameters(self, node, annotation, declared_functions):
        if annotation.kind != "contract":
            return annotation
        function, _, parameters, body = node.args
        declarations = annotation.args[0]
        if len(parameters) != len(declarations) or len(body) != 1 or body[0].kind not in ("return", "eval"):
            return annotation
        call = body[0].args[0]
        if call is None or call.kind != "call":
            return annotation
        callee, args = call.args
        local_names = {parameter.text for parameter in parameters}
        if callee.kind != "name" or callee.args[0] in local_names | {function.text}:
            return annotation
        if any(arg.kind not in ("name", "number", "float_number", "character", "boolean", "null") for arg in args):
            return annotation
        forwarded = {}
        for index, arg in enumerate(args):
            if arg.kind == "name":
                forwarded.setdefault(arg.args[0], []).append(index)
        candidates = {}
        for index, (_, declared) in enumerate(declarations):
            if (declared is not None and declared.kind == "type_name"
                    and parameters[index].text in forwarded):
                record = self.resolve(declared)
                if isinstance(record, Record) and self.contains_record_links(record):
                    candidates[index] = record
        if not candidates:
            return annotation

        name = callee.args[0]
        source = declared_functions.get(name)
        binding = self.imported_functions.get(name)
        if source is not None and source.kind == "contract":
            expected = tuple(ty for _, ty in source.args[0])
        else:
            contract = self.resolve(source) if source is not None else binding.contract if binding else None
            if not isinstance(contract, Contract):
                return annotation
            expected = contract.params

        def graph_type(ty):
            if not isinstance(ty, Node):
                return ty
            while ty.kind == "type_group":
                ty = ty.args[0]
            if ty.kind == "inferred_parameter_type":
                return ty.args[0]
            if ty.kind == "type_name":
                return self.resolve(ty)
            return None

        if len(args) != len(expected):
            return annotation
        inferred = list(declarations)
        for index, record in candidates.items():
            name, declared = declarations[index]
            desired = None
            for position in forwarded[parameters[index].text]:
                candidate = graph_type(expected[position])
                if not graph_or_chain_parameter(candidate) or not isinstance(candidate, BorrowRecord) or candidate.record != record:
                    break
                if candidate.attach is not None:
                    if not 0 <= candidate.attach < len(args):
                        break
                    donor = args[candidate.attach]
                    donor_index = next((i for i, param in enumerate(parameters)
                                        if donor.kind == "name" and param.text == donor.args[0]), None)
                    if donor_index is None or donor_index == index:
                        break
                    candidate = replace(candidate, attach=donor_index)
                if desired is not None and candidate != desired:
                    break
                desired = candidate
            else:
                inferred[index] = name, Node(declared.token, "inferred_parameter_type", (desired,))
                shape = "graph" if desired.graph else "chain"
                self.trace.append(f"  inferred {shape} parameter {function.text}.{parameters[index].text}")
        if inferred == list(declarations):
            return annotation
        return replace(annotation, args=(tuple(inferred), *annotation.args[1:]))

    def infer_parameter_annotations(self, node, contract, explicit):
        params, result = contract.args[:2]
        candidates = {name.text: [] for name, annotation in params if annotation is None}

        def add(value, annotation, visible):
            if annotation is None or value is None or value.kind != "name" or value.args[0] not in visible:
                return
            inferred = (annotation.args[0] if annotation.kind == "inferred_parameter_type"
                        else self.resolve(annotation))
            if replay_scalar(inferred) and not inferred.computed:
                candidates[value.args[0]].append((annotation, physical(inferred, annotation.token)))

        def scan_calls(expr, visible, local_names):
            pending = [expr]
            while pending:
                item = pending.pop()
                if isinstance(item, (tuple, list)):
                    pending.extend(reversed(item))
                    continue
                if not isinstance(item, Node):
                    continue
                if item.kind == "call":
                    callee, args = item.args
                    if callee.kind == "name" and callee.args[0] not in local_names:
                        name = callee.args[0]
                        if name in explicit:
                            called = explicit[name]
                        else:
                            binding = self.externs.get(name) or self.imported_functions.get(name)
                            called = binding.contract if binding is not None else None
                        if called is not None and len(args) == len(called.params):
                            for arg, parameter in zip(args, map(unqualified, called.params)):
                                if replay_scalar(parameter):
                                    annotation = Node(arg.token, "inferred_parameter_type", (parameter,))
                                    add(arg, annotation, visible)
                pending.extend(reversed(item.args))

        def scan(statements, visible, local_names):
            visible = set(visible)
            local_names = set(local_names)
            for stmt in statements:
                if stmt.kind == "define":
                    name, annotation, value = stmt.args
                    add(value, annotation, visible)
                    scan_calls(value, visible, local_names)
                    visible.discard(name.text)
                    local_names.add(name.text)
                elif stmt.kind == "return":
                    add(stmt.args[0], result, visible)
                    scan_calls(stmt.args[0], visible, local_names)
                elif stmt.kind == "if":
                    condition, yes, no = stmt.args
                    scan_calls(condition, visible, local_names)
                    scan(yes, visible, local_names)
                    scan(no, visible, local_names)
                elif stmt.kind == "while":
                    scan_calls(stmt.args[0], visible, local_names)
                    scan(stmt.args[1], visible, local_names)
                elif stmt.kind == "interation":
                    name, initial, _, count, step, loop_body = stmt.args
                    for operand in (initial, count, step):
                        scan_calls(operand, visible, local_names)
                    scan(loop_body, visible - {name.text}, local_names | {name.text})
                else:
                    scan_calls(stmt, visible, local_names)

        scan(node.args[3], candidates, {name.text for name, _ in params})
        resolved = []
        for name, annotation in params:
            if annotation is None:
                choices = candidates[name.text]
                if not choices:
                    name.error(f"cannot infer parameter {name.text!r}; supply its type annotation")
                annotation, inferred = choices[0]
                if any(other != inferred for _, other in choices[1:]):
                    name.error(f"conflicting inferred types for parameter {name.text!r}; supply its type annotation")
                self.trace.append(f"  inferred parameter {node.args[0].text}.{name.text}: {inferred.c} & {inferred.domain}")
            resolved.append((name, annotation))
        return replace(contract, args=(tuple(resolved), *contract.args[1:]))

    def complete_alias_graph_records(self, params, result):
        if not isinstance(result, AliasResult):
            return result
        available = [record for record in (*self.record_types.values(), *self.types.values())
                     if isinstance(record, Record)]
        return replace(result, graph_records=alias_graph_intermediates(params, result, available))

    def inferred_link_alias(self, source, token):
        root, tail = source, []
        while root.link_access is not None:
            root, name = root.link_access
            tail.append(name)
        origin = self.param_places.get(root.uid)
        if origin is None:
            return None
        index, path = origin
        params = self.functions[self.current][1].params
        alias = BorrowRecord(self.link_record(source.ty, token), readonly=True, nullable=True)
        path = (*path, *reversed(tail))
        result = self.complete_alias_graph_records(params, AliasResult(alias, index, path))
        if linked_alias_path(params[index], path, alias_graph_records(params, alias, result.graph_records)):
            return alias
        return None

    def infer_return(self, token, node, paths):
        if node is None:
            self.inferred_returns.append((token, None))
            self.check_heap_exit(paths, 0, token)
            self.return_states.extend(replace(path) for path in paths)
            return [], ["return;"]
        if node.kind == "null":
            self.inferred_returns.append((token, self.null_symbol))
            self.check_heap_exit(paths, 0, token)
            self.return_states.extend(replace(path) for path in paths)
            return [], ["return NULL;"]
        result = self.composite_call_type(node)
        operand = node.args[0] if self.is_copy(node) else node
        source = self.place(operand)
        if result is None and source is not None:
            alias = self.alias_types.get(source.uid)
            owns_heap = self.is_whole_heap_object(source)
            if operand is node and alias is not None and not source.readonly:
                candidates = tuple(target for path in paths
                                   for target in self.physical_places(source, path, allow_null=True))
                owns_heap = bool(candidates) and all(self.is_whole_heap_object(target) and not target.readonly
                                                     for target in candidates)
            if owns_heap and operand is node:
                if isinstance(source.ty, HeapArray):
                    length = domain_union(path.values[source.uid] for path in paths)
                    result = HeapArray(source.ty.element, length_bound=length)
                elif isinstance(source.ty, Record):
                    result = HeapRecord(source.ty)
                else:
                    result = HeapObject(source.ty)
            elif operand is not node:
                if isinstance(source.ty, Record):
                    result = source.ty
                elif isinstance(source.ty, Array):
                    result = CopyArray(source.ty)
        if result is None and source is not None and operand is node:
            alias = self.alias_types.get(source.uid) if node.kind == "name" else None
            if alias is None and isinstance(source.ty, Record) and source.uid in self.param_places:
                alias = BorrowRecord(source.ty, readonly=source.readonly)
            if alias is None and isinstance(source.ty, RecordLink):
                alias = self.inferred_link_alias(source, token)
            if alias is not None:
                return self.return_alias(node, None, paths, alias)
        if isinstance(result, Slice) or result is None and self.span_kind(node) == "ordinary":
            return self.return_slice(node, None, paths)
        if isinstance(result, AliasResult):
            return self.return_alias(node, None, paths, result.alias)
        if isinstance(result, InterfaceResult):
            return self.return_interface(node, None, paths, result.interface)
        if result is None and source is not None and isinstance(source.ty, Interface):
            return self.return_interface(node, None, paths, source.ty)
        if result is not None:
            self.inferred_returns.append((token, result))
            if isinstance(result, Record):
                return self.return_record(node, result, paths)
            if isinstance(result, CopyArray):
                return self.return_array(node, result, paths)
            return self.return_owned(node, result, paths)
        continuing_code = error_code = None
        for path in paths:
            value = self.expr(node, path)
            if value.always_returns:
                if error_code is None or self.propagation_code_rank(value.code) > self.propagation_code_rank(error_code):
                    error_code = value.code
                continue
            if is_callback_enum(value.enum):
                token.error("a callback return requires an explicit callable result type")
            self.check_heap_exit([path], 0, token)
            self.inferred_returns.append((token, value))
            self.observe_scalar_return(value, ((value.domain, path),))
            self.trace.append(f"  return {value.domain} [{path.guard_label()}]")
            self.return_states.append(replace(path))
            continuing_code = value.code
        return [], [f"return {continuing_code or error_code};"]

    def matches_slice_offset(self, token, candidate, start, path):
        if start is None:
            return False
        inputs = {}
        for origin, atom in self.entry_offset_origins.items():
            domain = origin.domain
            for uid, form in path.forms.items():
                if form.origin is origin:
                    domain &= form.preimage(path.values[uid])
            if not domain:
                return False
            key = atom[1] if atom[0] == "param" else ("len", atom[1])
            inputs[key] = Value(domain, "0", affine=AffineForm(origin))
        try:
            offset = self.slice_offset_value(token, candidate, inputs, path, require_total=True)
        except ValueError:
            return False
        return self.prove_order(token, start, "==", offset, path)

    def inferred_result(self, name, fallthrough):
        values = [value for _, value in self.inferred_returns if value is not None]
        if not values:
            return None
        if fallthrough or len(values) != len(self.inferred_returns):
            name.error("a reachable path does not return an inferred value")
        nullable = any(value is self.null_symbol for value in values)
        if nullable:
            values = [value for value in values if value is not self.null_symbol]
            if not values or not all(isinstance(value, AliasResult) for value in values):
                name.error("cannot infer an alias type from null alone or a non-alias return")
        first = values[0]
        if isinstance(first, AliasResult):
            if not all(isinstance(value, AliasResult) and
                       self.alias_storage(value.alias) == self.alias_storage(first.alias)
                       for value in values):
                name.error("inferred aliases require a common borrowed parameter or field")
            origins = self.inferred_alias_origins(
                {origin for value in values for origin in value.origins()}, name)
            source, field_path = origins[0]
            result = replace(first, source=source, field_path=field_path, alternatives=origins[1:],
                             alias=replace(first.alias,
                             nullable=nullable or any(value.alias.nullable for value in values),
                             readonly=any(value.alias.readonly for value in values)))
            if isinstance(result.alias, BorrowRecord) and result.alias.readonly and any(
                    replace(value.alias, nullable=result.alias.nullable, readonly=True) != result.alias
                    for value in values):
                result = replace(result, alias=BorrowRecord(result.alias.record, readonly=True,
                                                            nullable=result.alias.nullable))
            contract = self.functions[self.current][1]
            params = contract.params
            result = self.complete_alias_graph_records(params, result)
            if not valid_alias_provenance(params, result, contract.bounds, contract.parameter_expressions):
                name.error("cannot infer a valid alias return provenance")
            return result
        if isinstance(first, Slice) and all(isinstance(value, Slice) for value in values):
            domain = domain_union(value.domain for value in values)
            result = replace(first, length_bound=domain, readonly=any(value.readonly for value in values))
            if any((value.element, value.text, value.source_param, value.source_field) !=
                   (first.element, first.text, first.source_param, first.source_field) for value in values):
                name.error("inferred slices require a common element type and storage origin")
            def offset_key(value):
                return value.source_offset, slice_offset_terms(value), value.offset_bias, value.offset_expression

            if len({offset_key(value) for value in values}) > 1:
                candidates = {}
                for value, (_, alternative, _) in zip(values, self.inferred_slice_paths):
                    for candidate in (value, alternative):
                        if candidate is not None and (slice_offset_terms(candidate) or
                                                      candidate.offset_expression is not None):
                            candidates.setdefault(offset_key(candidate), candidate)
                result = replace(result, source_offset=None, offset_terms=(), offset_products=(),
                                 offset_bias=0, offset_expression=None)
                for candidate in candidates.values():
                    if all(offset_key(value) == offset_key(candidate) or
                           self.matches_slice_offset(name, candidate, start, path)
                           for value, (path, _, start) in zip(values, self.inferred_slice_paths)):
                        result = replace(candidate, length_bound=domain, readonly=result.readonly)
                        break
            if not valid_slice_provenance(self.functions[self.current][1].params, result):
                name.error("cannot infer a valid slice return provenance")
            return result
        if isinstance(first, HeapArray) and all(isinstance(value, HeapArray) and value.element == first.element
                                               for value in values):
            return replace(first, length_bound=domain_union(value.domain for value in values))
        if not isinstance(first, Value):
            if any(value != first for value in values):
                name.error("incompatible inferred return types")
            if self.contains_record_links(first) and not linked_owned_record(first):
                name.error("record links across function results require a lifetime contract")
            if not valid_interface_provenance(self.functions[self.current][1].params, first):
                name.error("cannot infer a valid interface return provenance")
            return first
        if any(not isinstance(value, Value) or (value.boolean, value.enum, value.character) !=
               (first.boolean, first.enum, first.character) for value in values):
            name.error("incompatible inferred return types")
        minimum = max(value.width for value in values)
        if any(value.floating for value in values):
            if any(value.boolean or value.enum or value.character or value.result_error_enums for value in values):
                name.error("incompatible inferred return types")
            minimum = max(minimum, *(value.domain.bits for value in values if value.floating))

            def convert(domain, bits):
                if isinstance(domain, FloatDomain):
                    return domain.convert(bits, "exact")
                return FloatDomain.from_integers(domain, bits, exact=True)

            for bits in (32, 64):
                if bits < minimum:
                    continue
                try:
                    success = FloatDomain.union(
                        (convert(value.result_arms[0] if value.result_arms is not None else value.domain, bits)
                         for value in values), bits)
                    errors = FloatDomain.union(
                        (convert(value.result_arms[1], bits) for value in values if value.result_arms is not None), bits)
                except FloatDomainError as error:
                    if bits == 64:
                        name.error(f"inferred floating result requires exact conversion: {error}")
                    continue
                if success & errors:
                    name.error("inferred success and error values overlap; declare an explicit result type")
                return Scalar(success | errors, bits, floating=True,
                              result_arms=(success, errors) if any(v.result_arms is not None for v in values) else None)
            name.error("no supported floating precision fits the inferred result")
        domain = domain_union(value.domain for value in values)
        arms = None
        if any(value.result_arms is not None for value in values):
            success = domain_union(value.result_arms[0] if value.result_arms is not None else value.domain
                                   for value in values)
            errors = domain_union(value.result_arms[1] for value in values if value.result_arms is not None)
            if self.proven_presence(success & errors, name):
                name.error("inferred success and error values overlap; declare an explicit result type")
            domain = success | errors
            if success or errors:
                arms = success, errors
        error_enums = (tuple(sorted({identity for value in values for identity in value.result_error_enums}))
                       if arms and arms[1] else ())
        signed = domain.lo < 0 or any(value.signed for value in values)
        for bits in WIDTHS:
            if bits >= minimum and domain.within(container(bits, signed)):
                return Scalar(domain, bits, signed, first.boolean, enum=first.enum,
                              result_arms=arms, character=first.character, result_error_enums=error_enums)
        if domain.lo >= 0 and domain.within(container(64, False)) and first.enum is None:
            return Scalar(domain, 64, result_arms=arms, character=first.character, result_error_enums=error_enums)
        name.error(f"no fixed-width integer container fits inferred result {domain}")

    def slice_clone(self, ty, descriptor, owner):
        allocate = self.allocating_helper("slice_clone")
        extra = " + 1" if ty.text == "c" else ""
        self.helper("global_temp", None, ("void *", owner))
        return f"{owner} = {allocate}({descriptor}.data, ({descriptor}.len{extra}) * sizeof({ty.element.c}))"
