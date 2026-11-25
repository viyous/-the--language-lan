"""Name registration, imports, exports and oracles."""

from __future__ import annotations

from dataclasses import replace

from semantic_ir import Proposition, Theorem
from size_change import strongly_connected_components
from value_domain import Domain, IntersectionOracles, IntersectionRule
from table_interface import c_ident, digest, encode_domain

from .encoding import EncodingTag, JsonMap, TextEncoding
from .errors import LanError
from .syntax import Node, TYPE_OBJECT_MARKERS, TypeObject
from .typesys import (AliasResult, Array, BorrowArray, BorrowRecord, BorrowScalar, contains_callable, Contract,
                      CopyArray, CopyRecord, EnumInfo, FunctionBinding, HeapArray, HeapObject, HeapRecord,
                      InterfaceTypeFunction, ModuleConst, Record, Scalar, Slice, Span, success_type, Symbol,
                      TaggedResult, TypeFunctionBinding, unqualified, Value, ZERO)
from .interface_types import alias_graph_records, encode_interface_type, linked_owned_record
from .summaries import valid_foreign_symbol


class RegistrationMixin:
    def register_names(self):
        self.callable_enum = EnumInfo(digest(f"{self.module_path}:{self.source_digest}:callback"),
                                      "<callback>", False, ())
        names = set(self.types) | self.import_names
        for node in self.nodes:
            if node.kind not in ("type", "string_encoding", "type_function", "function", "extern", "const"):
                continue
            name = node.args[0]
            if name.text == "free":
                name.error("free is a built-in heap operation")
            if name.text in names:
                name.error(f"duplicate top-level name {name.text!r}")
            names.add(name.text)
            if node.kind == "type":
                self.type_nodes[name.text] = node.args[1]
            elif node.kind == "string_encoding":
                self.string_encoding_nodes[name.text] = node
            elif node.kind == "const":
                self.const_nodes[name.text] = node
            elif node.kind == "function":
                self.callable_signatures[name.text] = node.args[1]
                self.callable_selectors[name.text] = len(self.callable_selectors) + 1
            elif node.kind == "type_function":
                self.type_functions[name.text] = node
                params = set()
                for param, _ in node.args[1]:
                    if param.text in params:
                        param.error("duplicate type function parameter")
                    params.add(param.text)

    def callable_target(self, token):
        name = token.text
        annotation = self.callable_signatures.get(name)
        if annotation is None:
            token.error(f"callable target {name!r} must be a declared function in this module")
        if self.callable_selectors[name] > 65535:
            token.error("callable target exceeds the module's function selector capacity")
        if name in self.callable_options:
            return self.callable_options[name][0]
        if annotation.kind == "contract" and (annotation.args[1] is None or
                any(parameter is None for _, parameter in annotation.args[0])):
            token.error(f"callable target {name!r} requires explicit parameter and result types")
        if name in self.resolving_callables:
            token.error(f"cyclic callable contract involving {name!r}")
        self.resolving_callables.add(name)
        try:
            contract = self.functions[name][1] if name in self.functions else self.resolve(annotation)
        finally:
            self.resolving_callables.remove(name)
        if not isinstance(contract, Contract):
            token.error(f"callable target {name!r} requires an explicit function contract")
        parameter_kinds = (Scalar, BorrowScalar, BorrowRecord, BorrowArray, Slice, CopyRecord, CopyArray,
                           HeapRecord, HeapObject, HeapArray)
        if (contract.result is not None and not isinstance(contract.result, Scalar)
                or any(not isinstance(unqualified(param), parameter_kinds) for param in contract.params)
                or any(getattr(param, "nullable", False) or self.contains_interface_storage(param)
                       or self.contains_record_links(param) for param in contract.params)
                or contains_callable(contract)):
            token.error(f"callable target {name!r} requires a scalar or void result and supported parameter types")
        self.callable_options[name] = contract, token
        return contract

    def prepare_type_objects(self):
        """Give the checker its existing storage and qualification nodes."""
        owner_anchors = set()
        converted = {}

        def normalize(value):
            if isinstance(value, (tuple, list)):
                return type(value)(map(normalize, value))
            if not isinstance(value, Node) or value.kind == "type_function":
                return value
            if value in converted:
                return converted[value]
            node = value
            if value.kind in ("type_merge", "type_object"):
                node = self.lower_type_object(value)
                if node.kind == "type_object":
                    converted[value] = node
                    return node
            node = replace(node, args=normalize(node.args), state_guard=normalize(node.state_guard))
            if node.kind == "define":
                annotation = node.args[1]
                if annotation is not None and annotation.kind == "storage" and annotation.args[1].text == "heap":
                    owner_anchors.add(node.semantic_anchor)
            converted[value] = node
            return node

        self.nodes = normalize(self.nodes)
        self.type_nodes = {node.args[0].text: node.args[1] for node in self.nodes if node.kind == "type"}
        self.string_encoding_nodes = {node.args[0].text: node for node in self.nodes if node.kind == "string_encoding"}
        self.const_nodes = {node.args[0].text: node for node in self.nodes if node.kind == "const"}
        if owner_anchors:
            self.semantic_module.operations = tuple(
                replace(operation, opcode="bind_owner") if operation.anchor in owner_anchors else operation
                for operation in self.semantic_module.operations
            )

    def bind_import(self, node, dependency):
        for original, local in node.args[0]:
            name = local.text
            if name in self.import_names or name in self.types:
                local.error(f"duplicate imported name {name!r}")
            if original.text not in dependency.exports:
                original.error(f"module {dependency.filename!r} does not export {original.text!r}")
            binding = dependency.exports[original.text]
            if isinstance(binding, InterfaceTypeFunction):
                original.error(f"type function {binding.name!r} needs its .lang implementation at compile time")
            self.import_names.add(name)
            if isinstance(binding, FunctionBinding):
                self.imported_functions[name] = binding
                result = success_type(binding.contract.result)
                records = tuple(ty.record for ty in (*binding.contract.params, result) if linked_owned_record(ty))
                if isinstance(result, AliasResult) and any(
                        isinstance(param, BorrowRecord) and param.graph for param in binding.contract.params):
                    records += tuple(alias_graph_records(binding.contract.params, result.alias, result.graph_records))
                for record in records:
                    previous = self.record_types.setdefault(record.identity, record)
                    if previous != record:
                        local.error("conflicting linked record definitions")
            elif isinstance(binding, TypeFunctionBinding):
                self.imported_type_functions[name] = binding
            elif isinstance(binding, ModuleConst):
                if binding.source_only:
                    original.error(f"non-scalar module constant {binding.name!r} needs its .lang source at compile time")
                if isinstance(binding.ty, JsonMap):
                    self.const_compile_values[name] = (binding.ty, None)
                    self.module_consts[name] = replace(binding, name=name)
                    self.trace.append(f"import {original.text} as {name} from {ascii(dependency.filename)}")
                    continue
                uid = len(self.symbols) + 1
                if binding.sym is not None:
                    c_name = binding.sym.c
                else:
                    prefix = "lan_const" if self.module_id is None else f"lan_m{self.module_id}_const"
                    c_name = f"{prefix}_imp_{uid}_{c_ident(name)}"
                    self.module_const_code.append(f"static const {binding.ty.c} {c_name} = {binding.c_init};")
                sym = Symbol(name, uid, binding.ty, 0, c_name, 0, readonly=True, borrowed=False, static_storage=True)
                self.symbols.append(sym)
                if isinstance(binding.ty, Record):
                    self.record_symbols(sym, local, zero=False)
                span = (replace(binding.span, code=c_name, borrowed_from=frozenset((sym,)), origins=((sym, ZERO),))
                        if binding.span is not None else None)
                self.module_consts[name] = replace(binding, sym=sym, span=span, name=name)
                if binding.scalar_value is not None:
                    self.const_compile_values[name] = (binding.scalar_value, binding.ty)
            else:
                self.types[name] = binding
            self.trace.append(f"import {original.text} as {name} from {ascii(dependency.filename)}")

    def function_effects(self, name):
        """Summarize allocator and external effects through local calls."""
        pending, seen = [name], set()
        memory, external = False, False
        while pending:
            callee = pending.pop()
            if callee not in seen:
                seen.add(callee)
                memory |= callee in self.memory_effects
                external |= callee in self.external_effects
                if memory and external:
                    break
                pending.extend(self.calls.get(callee, ()))
                pending.extend(self.callback_edges.get(callee, ()))
        return memory, external

    def publish_exports(self):
        for node in self.nodes:
            if node.kind != "export":
                continue
            for local, public in node.args[0]:
                name, exported = local.text, public.text
                if exported in self.exports:
                    public.error(f"duplicate export {exported!r}")
                if name in self.functions:
                    memory, external_effect = self.function_effects(name)
                    binding = FunctionBinding(self.functions[name][1], self.function_c_name(name),
                                              mutations=self.mutations[name],
                                              result_fields=self.record_results.get(name, ()),
                                              result_array=self.array_results.get(name),
                                              guard=self.function_guard(*self.functions[name]) or self.verified_guards.get(name),
                                              interface_accessor=self.function_interface_accessor(
                                                  *self.functions[name], self.mutations[name]),
                                              owning=self.owning.get(name, ()),
                                              memory=memory, result_domain=self.result_domains.get(name),
                                              external_effect=external_effect,
                                              storage_counts=self.storage_counts.get(name, ()),
                                              return_observation=self.call_return_observations.get(name),
                                              owned_chain_lengths=self.owned_chain_results.get(name, ()),
                                              owned_chain_parameters=self.owned_chain_parameters.get(name, ()))
                elif name in self.externs:
                    binding = self.externs[name]
                elif name in self.type_functions:
                    binding = TypeFunctionBinding(self, name)
                elif name in self.imported_functions:
                    binding = self.imported_functions[name]
                elif name in self.imported_type_functions:
                    binding = self.imported_type_functions[name]
                elif name in self.module_consts:
                    binding = self.module_consts[name]
                elif name in self.type_nodes or name in self.string_encoding_nodes or name in self.import_names:
                    binding = self.types[name]
                    if isinstance(binding, TypeObject):
                        binding = self.materialize_type_object(binding)
                else:
                    local.error(f"cannot export undeclared name {name!r}")
                if not isinstance(binding, (TypeFunctionBinding, ModuleConst)):
                    ty = binding.contract if isinstance(binding, FunctionBinding) else binding
                    if contains_callable(ty, self.record_types):
                        local.error("callable values are local to their Lan module and cannot be exported")
                elif isinstance(binding, ModuleConst):
                    if contains_callable(binding.ty, self.record_types):
                        local.error("callable values are local to their Lan module and cannot be exported")
                self.exports[exported] = binding
                self.trace.append(f"export {name} as {exported}")

    def materialize_type_object(self, ty, env=()):
        """Capture property values before leaving their scope."""
        entries = []
        for key, value in ty.entries:
            if value.kind == "object_value":
                constant = self.const_expr(value.args[0], env)
                if type(constant) not in (int, bool):
                    value.token.error("type object constants must be integers or booleans")
                literal = Node(value.token, "boolean" if type(constant) is bool else "number", (constant,))
                value = replace(value, args=(literal,))
            elif value.kind == "type_name" and key.text == "field":
                value = replace(value, kind="object_path", args=(value.args,))
            elif value.kind not in TYPE_OBJECT_MARKERS:
                resolved = self.resolve(value, env)
                if isinstance(resolved, TypeObject):
                    resolved = self.materialize_type_object(resolved, env)
                if resolved is None:
                    value.token.error("void cannot be a property value in a type object")
                value = replace(value, kind="resolved_type", args=(resolved,))
            entries.append((key, value))
        return TypeObject(tuple(entries))

    @staticmethod
    def polynomial_clause(domain):
        if domain.intervals or len(domain.predicates) != 1:
            return None
        clause = domain.predicates[0]
        if len(clause.require) != 1 or clause.exclude:
            return None
        return clause

    def register_oracles(self):
        originals = self.types.copy()
        oracles = IntersectionOracles({}, {}, lambda rule: self.trace.append(f"  apply oracle {rule.location}"))
        for node in self.nodes:
            if node.kind == "theorem":
                intersection = self.check_source_theorem(node)
                if intersection is None:
                    continue
                left, right, result = intersection
            elif node.kind == "oracle":
                try:
                    left, right, result = (self.numeric_type(item).domain for item in node.args)
                except LanError as exc:
                    node.token.error(f"invalid intersection theorem: {exc}")
            else:
                continue
            t = node.token
            key = frozenset((left, right))
            previous = oracles.exact.get(key)
            if previous is not None and previous.result != result:
                if previous.result - result or result - previous.result:
                    t.error(f"conflicting oracle results for the same intersection (previous at {previous.location})")
            actual = left & right
            missing, invented = actual - result, result - actual
            missed = missing.witness()
            if missed is not None:
                t.error(f"oracle contradicts the intersection: {missed} is a common value missing from its result")
            extra = invented.witness()
            if extra is not None:
                t.error(f"oracle contradicts the intersection: {extra} is not a common value")
            location = f"{t.line}:{t.col}"
            rule = IntersectionRule(result, location)
            oracles.exact[key] = rule
            a, b = self.polynomial_clause(left), self.polynomial_clause(right)
            if a is not None and b is not None and a.require != b.require:
                pair = frozenset(a.require + b.require)
                lower, upper = max(a.lo, b.lo), min(a.hi, b.hi)
                rule = IntersectionRule(result, location, lower, upper)
                rules = oracles.polynomial_pairs.setdefault(pair, [])
                for old in rules:
                    overlap = Domain.of((max(lower, old.lower), min(upper, old.upper)))
                    old_part, new_part = old.result & overlap, result & overlap
                    if (old_part - new_part).witness() is not None or (new_part - old_part).witness() is not None:
                        t.error(f"conflicting oracle results on overlapping bounds (previous at {old.location})")
                rules.append(rule)
            status = "trusted" if missing or invented else "verified"
            if node.kind == "oracle":
                self.source_theorems.append(Theorem(
                    Proposition("equal", tuple(map(encode_domain, (actual, result)))),
                    node.semantic_anchor, "declaration", (t.line, t.col),
                    "assumed" if status == "trusted" else status))
            self.trace.append(f"theorem oracle {location} ({status}): ({left}) & ({right}) => {result}")
        # Type aliases used to validate oracles were resolved without any rules.
        # Recompute them in the real compilation under the complete rule set.
        self.types = originals
        self.type_cache.clear()
        self.callable_options.clear()
        return oracles

    def check_source_theorem(self, node):
        relation, operands, axiom = node.args
        if relation.text not in ("equal", "subset", "disjoint") or len(operands) != 2:
            relation.error("expected equal, subset or disjoint with two integer-set operands")
        left, right = (self.numeric_type(operand).domain for operand in operands)
        if relation.text == "equal":
            difference = (left - right) | (right - left)
        elif relation.text == "subset":
            difference = left - right
        else:
            difference = left & right
        if difference:
            witness = difference.witness()
            if witness is not None:
                relation.error(f"false theorem: counterexample {witness}")
            if not axiom:
                relation.error("cannot prove theorem; use an explicit axiom to supply an external fact")
        token = node.token
        self.source_theorems.append(Theorem(Proposition(relation.text, tuple(map(encode_domain, (left, right)))),
                                           node.semantic_anchor, "declaration", (token.line, token.col),
                                           "assumed" if axiom else "verified"))
        if relation.text == "disjoint":
            return left, right, Domain.of()
        if relation.text == "equal":
            for candidate, result in zip(operands, (right, left)):
                while candidate.kind == "type_group":
                    candidate = candidate.args[0]
                if candidate.kind == "type_op" and candidate.token.text == "&":
                    a, b = (self.numeric_type(item).domain for item in candidate.args)
                    return a, b, result
        return None

    def eval_string_encoding(self, node):
        name_tok, map_expr, mode = node.args
        if name_tok.text in self.types:
            return self.types[name_tok.text]
        val = self.const_expr(map_expr, ())
        if not isinstance(val, JsonMap):
            map_expr.token.error("type string requires a compile-time JSON mapping object")
        enc = val.to_encoding(name_tok.text, name_tok)
        ty = EncodingTag(enc) if mode == "tag" else Slice(self.types["uint8"], readonly=True, text=enc)
        self.types[name_tok.text] = ty
        return ty

    def register(self):
        # Phase 1: Synchronous top-down compile-time phase (const, expand, string_encoding, type)
        for node in self.nodes:
            if node.kind == "const":
                self.eval_top_const(node)
            elif node.kind == "expand":
                self.eval_expand(node)
            elif node.kind == "string_encoding":
                self.eval_string_encoding(node)
            elif node.kind == "type":
                name = node.args[0]
                resolved = self.resolve(Node(name, "type_name", (name.text,)))
                self.check_intersection_type(resolved, name)

        # Phase 1.5: Final materialization of all module constants using their final values
        for node in self.nodes:
            if node.kind == "const":
                self.resolve_module_const(node.args[0].text)

        pending = []
        function_names = []
        definitions = {node.args[0].text: node for node in self.nodes if node.kind == "function"}
        declared_functions = {node.args[0].text: node.args[1] for node in self.nodes
                              if node.kind in ("function", "extern")}
        self.infer_release_parameters(definitions, declared_functions)
        calls = [(name, item.args[0].args[0]) for name, node in definitions.items()
                 for item in self.body_operations(node.args[3])
                 if item.kind == "call" and item.args[0].kind == "name"
                 and item.args[0].args[0] in definitions]
        components = strongly_connected_components(definitions, calls)
        chain_parameters = self.recursive_chain_parameters(definitions, components)
        recursive = {caller for caller, callee in calls if components[caller] == components[callee]}
        for name, node in definitions.items():
            annotation = self.infer_chain_parameters(node, declared_functions[name], chain_parameters)
            declared_functions[name] = self.infer_graph_parameters(node, annotation, recursive=name in recursive)
        while True:
            forwarded = {name: self.infer_graph_forwarding_parameters(node, declared_functions[name],
                                                                      declared_functions)
                         for name, node in definitions.items()}
            if all(annotation is declared_functions[name] for name, annotation in forwarded.items()):
                break
            declared_functions.update(forwarded)

        def register_function(node, annotation):
            name = node.args[0]
            if annotation is not node.args[1]:
                node = replace(node, args=(name, annotation, *node.args[2:]))
            contract = self.resolve(annotation)
            if not isinstance(contract, Contract):
                annotation.token.error("function definition requires a contract")
            if len(node.args[2]) != len(contract.params):
                name.error(f"function contract requires {len(contract.params)} parameters")
            self.functions[name.text] = node, contract
            self.calls[name.text] = set()

        # Phase 2: Externs and runtime function signatures
        for node in self.nodes:
            if node.kind not in ("function", "extern"):
                continue
            name, ty = node.args[:2]
            if node.kind == "extern":
                if not valid_foreign_symbol(name.text):
                    name.error("external symbol conflicts with the C backend")
                self.native_signature = True
                try:
                    contract = self.resolve(ty)
                finally:
                    self.native_signature = False
                if contains_callable(contract, self.record_types):
                    name.error("callable values are local to their Lan module")
                if any(isinstance(param, (HeapRecord, HeapObject, HeapArray)) for param in contract.params):
                    name.error("a native function cannot take ownership of a Lan heap object")
                if isinstance(contract.result, TaggedResult):
                    name.error("composite result unions are only supported by Lan functions")
                if linked_owned_record(contract.result):
                    name.error("a native function cannot return an owning linked record")
                writes, memory, external_effect = self.declared_write_set(node.args[2], contract, ty.args[0])
                self.externs[name.text] = FunctionBinding(
                    contract, name.text, external=True, foreign=True,
                    mutations=writes, memory=memory, external_effect=external_effect)
                self.trace.append(f"extern C {name.text}: trusted implementation contract")
                detail = "nothing" if not writes else ", ".join(
                    "#%d%s" % (index, "." + ".".join(names) if names else "")
                    for index, names, _ in writes)
                self.trace.append(f"  writes {name.text}: {detail}"
                                  + ("; may allocate or release memory" if memory else "")
                                  + ("; may affect external state" if external_effect else ""))
            elif node.kind == "function":
                function_names.append(name.text)
                ty = declared_functions[name.text]
                if ty.kind == "contract" and any(annotation is None for _, annotation in ty.args[0]):
                    pending.append(node)
                else:
                    register_function(node, ty)
        if pending:
            explicit = {name: contract for name, (_, contract) in self.functions.items()}
            for node in pending:
                name = node.args[0]
                annotation = self.infer_parameter_annotations(node, declared_functions[name.text], explicit)
                register_function(node, annotation)
            self.functions = {name: self.functions[name] for name in function_names}
        for name, (contract, token) in self.callable_options.items():
            if encode_interface_type(self.functions[name][1]) != encode_interface_type(contract):
                token.error(f"callable target {name!r} has an inferred contract; declare the complete contract explicitly")
