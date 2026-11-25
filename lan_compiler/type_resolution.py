"""Resolving type syntax, type objects and parameter bounds."""

from __future__ import annotations

import json
import math
from dataclasses import field, fields, is_dataclass, replace
from fractions import Fraction
from itertools import count
from math import inf, lcm

from float_domain import FloatDomain, FloatDomainError, quantize, next_float, is_representable
from value_domain import Clause, Domain, DomainComplexity, Polynomial, evaluate
from table_interface import InterfaceError, digest

from .encoding import ConstList, EncodingTag, JsonMap, TextEncoding
from .errors import FloatConstant, LanError
from .lexer import Token
from .syntax import (Node, REMOVED_TYPE_QUALIFIERS, type_object_kind, TYPE_OBJECT_MARKERS,
                     TYPE_OBJECT_SCALARS, TYPE_OBJECT_WORDS, TYPE_QUALIFIERS, TypeArgument, TypeObject)
from .typesys import (AliasResult, Array, BorrowArray, BorrowRecord, BorrowScalar, CallableType, Contract,
                      CopyArray, CopyRecord, EnumInfo, ExternCallable, HeapArray, HeapObject, HeapRecord, Interface,
                      InterfaceResult, is_callback_enum, NullType, PACKED_BIT, ReadOnly, Record,
                      record_text_type, RecordLink, Scalar, Slice, StaticReadOnly, storage_size,
                      success_type, tagged_error_type, TaggedResult, unqualified, WIDTHS)
from .domain_utils import (closed_integer_property, commutative_integer_signature, container,
                           diagonal_integer_signature, float_literal_domain, physical,
                           plain_integer_parameter)
from .interface_types import (combine_offset_polynomial, decode_offset_expression,
                              dependent_expression_domain, dependent_source_domain, encode_interface_type,
                              graph_or_chain_parameter, linked_owned_record, OFFSET_BINARY_OPS,
                              offset_expression_domain, offset_expression_parameters, offset_length_width,
                              valid_alias_provenance, valid_chain_parameters, valid_interface_provenance,
                              valid_slice_provenance)

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class TypeResolutionMixin:
    @staticmethod
    def type_object_value(value, *, resolved=False):
        """Compare object properties and resolved types independently of source location."""
        def part(item):
            return Compiler.type_object_value(item, resolved=resolved)

        if resolved and isinstance(value, TypeObject):
            return TypeObject, tuple((key.text, part(item)) for key, item in value.entries)
        if isinstance(value, Node):
            operator = value.token.text if value.kind in ("bound", "type_op") else None
            return value.kind, operator, tuple(map(part, value.args))
        if isinstance(value, Token):
            return value.text
        if isinstance(value, (list, tuple)):
            return tuple(map(part, value))
        if resolved and is_dataclass(value) and not isinstance(value, (Domain, FloatDomain)):
            return type(value), tuple(part(getattr(value, item.name)) for item in fields(value))
        return (type(value), value)

    def type_property_is_constant(self, value, env):
        if value.kind != "type_name":
            return False
        name, = value.args
        binding = self.compile_time_binding(name, env)
        if binding is not None:
            return not isinstance(binding[name][0], TypeArgument)
        if getattr(self, "_active_paths", ()) and self.scope:
            if any(name in scope for scope in reversed(self.scope)):
                return self._active_runtime_constant(replace(value, kind="name")) is not None
        return self._singleton_type_constant(name) is not None

    def type_object_parts(self, node, env=(), *, seen=frozenset(), preserve_record=False):
        """Read an object or an alias to it without choosing a physical type."""
        if node.kind == "type_group":
            return self.type_object_parts(node.args[0], env, seen=seen, preserve_record=preserve_record)
        if node.kind == "type_merge":
            right_base, right = self.type_object_parts(node.args[1], env, seen=seen)
            if right_base is not None:
                node.token.error("right operand of · must be a type object or an alias to one")
            qualifying = all(key.text != "repr" and key.text in (*TYPE_QUALIFIERS, *REMOVED_TYPE_QUALIFIERS)
                             for key, _ in right)
            qualifying = qualifying and any(value.kind in TYPE_OBJECT_MARKERS for _, value in right)
            left_base, left = self.type_object_parts(node.args[0], env, seen=seen,
                                                     preserve_record=qualifying)
            return left_base, (*left, *right)
        if node.kind == "record" and preserve_record:
            return node, ()
        if node.kind in ("record", "type_object"):
            entries = []
            for key, value in node.args[0]:
                if self.type_property_is_constant(value, env):
                    value = Node(value.token, "object_value", (replace(value, kind="name"),))
                entries.append((key, value))
            return None, tuple(entries)
        if node.kind == "set" and not node.args[0]:
            return None, ()
        if node.kind == "type_name":
            name, = node.args
            parameter = self.type_parameter_binding(name, env)
            if parameter is not None:
                binding = parameter.value
            else:
                if name in TYPE_OBJECT_WORDS["kind"]:
                    return None, (type_object_kind(node.token, name),)
                if name in seen:
                    node.token.error(f"cyclic type definition involving {name!r}")
                if name in self.type_object_nodes:
                    target = self.type_object_nodes[name]
                    if target.kind == "record" and preserve_record:
                        return node, ()
                    if target.kind in ("type_name", "type_merge", "type_object", "record", "type_group",
                                       "array", "type_apply"):
                        base, entries = self.type_object_parts(target, seen=seen | {name})
                        if preserve_record and base is None and entries and all(
                                value.kind not in TYPE_OBJECT_MARKERS or key.text == "repr" for key, value in entries):
                            return node, ()
                        return (node, entries) if base is not None and not entries else (base, entries)
                binding = self.types.get(name)
        elif node.kind in ("type_apply", "resolved_type"):
            binding = self.resolve(node, env)
        else:
            binding = None
        if isinstance(binding, TypeObject):
            return None, binding.entries
        if self.full_integer_type(binding):
            token = node.token
            value = Node(token, "number", (binding.bits,))
            return None, (type_object_kind(token, "int" if binding.signed else "uint"),
                          (replace(token, text="bits"), Node(token, "object_value", (value,))))
        if isinstance(binding, Array):
            elem = (Node(node.token, "type_name", ("bit",)) if binding.packed_bits
                    else Node(node.token, "resolved_type", (binding.element,)))
            length = Node(node.token, "number", (binding.length,))
            return self.type_object_parts(Node(node.token, "array", (elem, length)), env, seen=seen)
        if isinstance(binding, Record) and binding.key_type is None and not preserve_record:
            entries = tuple((replace(node.token, text=field), Node(node.token, "resolved_type", (ty,)))
                            for field, ty in binding.fields)
            if binding.repr_c:
                token = replace(node.token, text="repr")
                entries += ((token, Node(token, "object_word", ("c",))),)
            return None, entries
        if node.kind == "array":
            element, length = node.args
            token = node.token
            return None, (type_object_kind(token, "array"),
                          (replace(token, text="elem"), element),
                          (replace(token, text="len"), Node(length.token, "object_value", (length,))))
        return node, ()

    @staticmethod
    def full_integer_type(ty):
        return (isinstance(ty, Scalar) and ty.bits in WIDTHS
                and not (ty.floating or ty.boolean or ty.character or ty.enum or ty.computed)
                and ty.result_arms is None and ty.domain == container(ty.bits, ty.signed))

    def lower_type_object(self, node, env=()):
        def expression(value):
            if value.kind == "object_value":
                return value.args[0]
            if value.kind == "type_name":
                return replace(value, kind="name")
            return value

        base, entries = self.type_object_parts(node, env)
        properties = {}
        for key, value in entries:
            if key.text in properties:
                previous = properties[key.text][1]
                same = self.type_object_value(previous) == self.type_object_value(value)
                if not same and (previous.kind == value.kind == "object_value"
                                 or key.text in TYPE_OBJECT_SCALARS and
                                 {previous.kind, value.kind} == {"object_value", "type_name"}):
                    try:
                        left = self.const_expr(expression(previous), env)
                        right = self.const_expr(expression(value), env)
                        same = type(left) is type(right) and left == right
                    except LanError:
                        pass
                if not same and previous.kind not in TYPE_OBJECT_MARKERS and value.kind not in TYPE_OBJECT_MARKERS:
                    try:
                        left, right = self.resolve(previous, env), self.resolve(value, env)
                        same = left == right
                        if isinstance(left, TypeObject) and isinstance(right, TypeObject):
                            same = self.type_object_value(left, resolved=True) == self.type_object_value(
                                right, resolved=True)
                    except LanError:
                        pass
                if not same:
                    key.error(f"conflicting type object property {key.text!r}")
            else:
                properties[key.text] = key, value

        def take(name):
            entry = properties.pop(name, None)
            return entry[1] if entry else None

        def word(value, name):
            if value.kind != "object_word":
                value.token.error(f"type object property {name!r} requires a name")
            return value.args[0]

        kind_entry = properties.get("kind")
        kind = word(kind_entry[1], "kind") if kind_entry is not None and kind_entry[1].kind == "object_word" else None
        if base is None and kind in ("int", "uint", "array"):
            if kind == "array":
                if "elem" not in properties or "len" not in properties:
                    return replace(node, kind="type_object", args=(tuple(properties.values()),))
                take("kind")
                element, length = take("elem"), take("len")
                if length.kind not in ("object_value", "type_name"):
                    length.token.error("array len requires an integer length")
                base = Node(node.token, "array", (element, expression(length)))
            else:
                if "bits" not in properties:
                    return replace(node, kind="type_object", args=(tuple(properties.values()),))
                take("kind")
                width = take("bits")
                bits = self.const_int(expression(width), env)
                if bits not in WIDTHS:
                    width.token.error("integer bits must be 8, 16, 32 or 64")
                base = Node(node.token, "type_name", (f"{kind}{bits}",))

        if base is None:
            attributes = {name for name, (_, value) in properties.items()
                          if value.kind in TYPE_OBJECT_MARKERS}
            if attributes - {"repr"} or attributes == properties.keys() or (
                    node.kind == "type_object" and any(key.text in ("len", "elem")
                    and value.kind in ("set", "interval", "bound") for key, value in properties.values())):
                return replace(node, kind="type_object", args=(tuple(properties.values()),))
            fields = tuple((key, value) for name, (key, value) in properties.items() if name not in attributes)
            repr_c = "repr" in attributes
            if repr_c:
                word(take("repr"), "repr")
            for key, _ in fields:
                take(key.text)
            base = Node(node.token, "record", (fields, repr_c))

        if unknown := next((key for key in properties if key not in TYPE_QUALIFIERS), None):
            if unknown in REMOVED_TYPE_QUALIFIERS:
                properties[unknown][0].error(f"removed shape annotation {unknown!r}; the compiler infers it from pointer operations")
            properties[unknown][0].error(f"unknown type object property {unknown!r}")
        length, contents, readonly, passing, storage, source, offset, field, repr_value = map(take, TYPE_QUALIFIERS)
        if repr_value is not None:
            repr_value.token.error("repr requires a record object")
        if readonly is not None:
            marker = readonly
            readonly = self.const_expr(expression(marker), env)
            if type(readonly) is not bool:
                marker.token.error("immutable must be boolean")
        if passing is not None:
            marker = passing
            passing = word(marker, "pass")
            if passing not in ("copy", "alias"):
                marker.token.error("expected alias or copy")
        if length is not None and length.kind == "object_value":
            length = Node(length.token, "set", ((expression(length),),))
        if storage is not None:
            if length is not None and all(value is None for value in (contents, source, passing, offset, field)):
                base = Node(node.token, "slice_qualified", (base, length, readonly, None, None, None, None, ()))
                length = readonly = None
            base = Node(node.token, "storage", (base, replace(storage.token, text=word(storage, "storage"))))
        if source is not None:
            source = self.const_int(expression(source), env)
        if field is not None:
            if field.kind == "object_path":
                field = field.args[0]
            elif field.kind == "type_name":
                field = field.args
            else:
                field.token.error("field requires a record field path")
        if any(value is not None for value in (length, contents, readonly, passing, source, offset, field)):
            if source is not None and passing is not None:
                base = Node(node.token, "slice_qualified",
                            (base, length, readonly, passing, contents, None, None, ()))
                length = readonly = passing = contents = None
            return replace(node, kind="slice_qualified",
                           args=(base, length, readonly, passing, contents, source,
                                 expression(offset) if offset is not None else None, field or ()))
        return replace(node, token=base.token, kind=base.kind, args=base.args)

    def _try_const_merge(self, node, env):
        """Handle `·` in type_expr context when operands are compile-time JsonMap or ConstList values."""
        if node.kind != "type_merge":
            return None
        left_node, right_node = node.args

        def eval_side(part):
            if part.kind == "type_merge":
                res = self._try_const_merge(part, env)
                if res is not None:
                    return res
            if part.kind == "json_map":
                return self.const_expr(part, env)
            if part.kind == "set" and not part.args[0]:
                return JsonMap(())
            if part.kind == "type_name":
                name = part.args[0]
                scope = self.compile_time_binding(name, env)
                if scope is not None:
                    bound = scope[name][0]
                    inner = bound.value if isinstance(bound, TypeArgument) else bound
                    if isinstance(inner, (JsonMap, ConstList)):
                        return inner
                if name in self.const_compile_values:
                    inner = self.const_compile_values[name][0]
                    if isinstance(inner, (JsonMap, ConstList)):
                        return inner
            if part.kind == "type_apply":
                res = self.resolve(part, env)
                if isinstance(res, (JsonMap, ConstList)):
                    return res
            return None

        left = eval_side(left_node)
        if isinstance(left, JsonMap):
            right = eval_side(right_node)
            if not isinstance(right, JsonMap):
                right_node.token.error("right operand of '·' must be a mapping object")
            return left.merge(right)
        if isinstance(left, ConstList):
            right = eval_side(right_node)
            if right is None and right_node.kind == "interval" and not right_node.args[2] and not right_node.args[3]:
                right = ConstList((self.const_expr(right_node.args[0], env), self.const_expr(right_node.args[1], env)))
            if not isinstance(right, ConstList):
                right_node.token.error("right operand of '·' must be a list")
            return left.concat(right)
        return None

    def resolve(self, node, env=()):
        self.spend(node.token)
        k, a, t = node.kind, node.args, node.token
        if k == "resolved_type":
            return a[0]
        if k == "json_map":
            return self.const_expr(node, env)
        if k == "type_merge" and (merged := self._try_const_merge(node, env)) is not None:
            return merged
        if k == "callable_type":
            candidates = a[0]
            if not candidates:
                t.error("callable requires at least one local function")
            names = tuple(candidate.text for candidate in candidates)
            if len(set(names)) != len(names):
                t.error("duplicate function in callable candidate list")
            contracts = [self.callable_target(candidate) for candidate in candidates]
            signature = encode_interface_type(contracts[0])
            if any(encode_interface_type(contract) != signature for contract in contracts[1:]):
                t.error("callable candidates require identical parameter and result contracts")
            selectors = Domain.of(*((self.callable_selectors[name],) * 2 for name in names))
            return CallableType(selectors, 16, enum=self.callable_enum,
                                candidates=tuple(sorted(names)), contract=contracts[0])
        if k == "extern_callable":
            params_ast, result_ast, writes_ast = a
            contract_node = Node(t, "contract", (params_ast, result_ast))
            saved_native = self.native_signature
            self.native_signature = True
            try:
                contract = self.resolve(contract_node, env)
            finally:
                self.native_signature = saved_native
            if any(isinstance(param, (HeapRecord, HeapObject, HeapArray)) for param in contract.params):
                t.error("an extern callable cannot take ownership of a Lan heap object")
            if isinstance(contract.result, TaggedResult):
                t.error("composite result unions are only supported by Lan functions")
            if linked_owned_record(contract.result):
                t.error("an extern callable cannot return an owning linked record")
            writes, memory, external_effect = self.declared_write_set(writes_ast, contract, params_ast)
            return ExternCallable(contract, mutations=writes, memory=memory, external_effect=external_effect)
        if k in ("type_object", "type_merge") or (k == "record" and env and any(
                self.type_property_is_constant(value, env) for _, value in a[0])):
            lowered = self.lower_type_object(node, env)
            return TypeObject(lowered.args[0]) if lowered.kind == "type_object" else self.resolve(lowered, env)
        if k == "record_key_type":
            owner = self.resolve(a[0], env)
            if not isinstance(owner, Record) or owner.key_type is None:
                t.error(".key requires a type declared with 'type enum'")
            return owner.key_type
        if k == "type_name":
            name, = a
            parameter = self.type_parameter_binding(name, env)
            if parameter is not None:
                return parameter.value
            if name == "bit":
                t.error("bit is a logical unit; use bit[N] with a positive multiple of 8 for storage")
            if name not in self.types:
                if name in self.string_encoding_nodes:
                    return self.eval_string_encoding(self.string_encoding_nodes[name])
                if name not in self.type_nodes:
                    t.error(f"unknown type {name!r}")
                if name in self.resolving:
                    t.error(f"cyclic type definition involving {name!r}")
                self.resolving.add(name)
                try:
                    self.types[name] = self.resolve(self.type_nodes[name])
                finally:
                    self.resolving.remove(name)
            return self.types[name]
        if k == "enum":
            name, annotation, members = a
            base = self.scalar_type(annotation, env)
            if base.boolean or base.enum or base.character or base.domain != container(base.bits, base.signed):
                annotation.token.error("enum representation requires an unrefined fixed-width integer type")
            if not 0 < len(members) <= 4096:
                t.error("enum must have between 1 and 4096 members")
            names, values, result = set(), set(), []
            value = -1
            for member, initializer in members:
                if member.text in names:
                    member.error(f"duplicate enum member {member.text!r}")
                value = self.const_int(initializer, env) if initializer is not None else value + 1
                if value in values:
                    member.error(f"duplicate enum value {value}")
                if not base.domain.lo <= value <= base.domain.hi:
                    member.error(f"enum value {value} exceeds {base.domain}")
                names.add(member.text)
                values.add(value)
                result.append((member.text, value))
            domain = Domain.of(*((value, value) for value in values))
            identity = digest(f"{self.module_path}:{t.line}:{t.col}:{self.source_digest}")
            return replace(base, domain=domain, enum=EnumInfo(identity, name.text, base.signed, tuple(result)))
        if k in ("record", "interface", "keyed_record"):
            key_type = None
            if k == "keyed_record":
                record_name, fields = a
                repr_c = False
            else:
                fields, repr_c = a
            names, result = set(), []
            if not fields or len(fields) > 4096:
                t.error("record or interface must declare between 1 and 4096 fields")
            for name, annotation in fields:
                if name.text in names:
                    name.error(f"duplicate record field {name.text!r}")
                names.add(name.text)
                ty = self.resolve_record_field(annotation, env) if k == "record" else self.resolve(annotation, env)
                readonly = isinstance(ty, ReadOnly)
                ty = unqualified(ty)
                if k == "keyed_record" and not isinstance(ty, Scalar):
                    annotation.token.error("enum record fields currently require scalar types")
                if isinstance(ty, Scalar):
                    self.check_intersection_type(ty, name)
                    ty = physical(ty, name)
                elif isinstance(ty, Slice) and record_text_type(ty):
                    readonly = True
                elif isinstance(ty, Interface) and k == "record" and not repr_c:
                    pass
                elif not isinstance(ty, (Array, Record, RecordLink)):
                    name.error("record fields require a scalar, fixed array, record or immutable string type")
                elif self.contains_interface_storage(ty):
                    name.error("a record with interface fields cannot be nested")
                result.append((name.text, ReadOnly(ty) if readonly else ty))
            identity = digest(f"{self.module_path}:{t.line}:{t.col}:{self.source_digest}")
            if k == "keyed_record":
                members = tuple((name, index) for index, (name, _) in enumerate(result))
                bits = 8 if len(members) <= 256 else 16
                key_type = Scalar(Domain.of((0, len(members) - 1)), bits,
                                  enum=EnumInfo(digest(f"{identity}:key"), f"{record_name.text}.key", False, members))
            elif env:
                layout = [(name, encode_interface_type(field)) for name, field in result]
                identity = digest(f"{identity}:{repr_c}:{json.dumps(layout, sort_keys=True)}")
            ty = Record(identity, tuple(result), repr_c, key_type)
            if ty.leaves > 4096 or ty.max_bytes >= 1 << 31:
                t.error("record must have at most 4096 fields and occupy fewer than 2^31 bytes")
            if k == "interface" and self.contains_record_links(ty):
                t.error("interface fields containing record links require a lifetime contract")
            if k != "interface":
                self.record_types[identity] = ty
            return Interface(ty) if k == "interface" else ty
        if k == "interval":
            lo, hi, open_left, open_right = a
            low, high = self.const_expr(lo, env), self.const_expr(hi, env)
            if isinstance(low, (Fraction,FloatConstant)) or isinstance(high, (Fraction,FloatConstant)):
                return Scalar(self.float_set(node, 64, env), 64, floating=True)
            return Scalar(Domain.of((self.const_int(lo, env) + open_left, self.const_int(hi, env) - open_right)))
        if k == "type_group":
            return self.resolve(a[0], env)
        if k == "storage":
            base, storage = a
            if storage.text == "heap" and base.kind == "array" and not (
                    base.args[0].kind == "type_name" and base.args[0].args == ("bit",)):
                element, count = base.args
                return HeapArray(self.scalar_type(element, env), count)
            ty = self.resolve(base, env)
            frozen = isinstance(ty, ReadOnly)
            plain = unqualified(ty)
            if isinstance(plain, Slice) and storage.text == "heap":
                if (plain.text is not None or not isinstance(plain.element, Scalar)
                        or plain.content_bound is not None or plain.source_param is not None):
                    storage.error("owned array result requires an ordinary scalar slice")
                result = HeapArray(plain.element)
                domain = result.domain & plain.domain
                if not domain:
                    storage.error("owned array length must be positive")
                result = replace(result, length_bound=domain)
                return ReadOnly(result) if plain.readonly else result
            if isinstance(plain, Record) and storage.text == "heap":
                result = HeapRecord(plain)
                return ReadOnly(result) if frozen else result
            if isinstance(plain, Array) and storage.text == "heap" and not plain.packed_bits:
                result = HeapArray(plain.element, Node(t, "number", (plain.length,)))
                return ReadOnly(result) if frozen else result
            if storage.text == "heap" and (isinstance(plain, Scalar)
                                          or isinstance(plain, Array) and plain.packed_bits):
                result = HeapObject(physical(plain, t) if isinstance(plain, Scalar) else plain)
                return ReadOnly(result) if frozen else result
            if isinstance(plain, (Scalar, Array, Record)) and storage.text == "rodata":
                return StaticReadOnly(plain)
            if isinstance(plain, (Scalar, Array, Record)) and storage.text == "stack":
                return ty
            storage.error("unsupported storage qualifier for this type")
        if k == "bound":
            limit = self.const_expr(a[0], env)
            if isinstance(limit, (Fraction,FloatConstant)):
                return Scalar(self.float_set(node, 64, env), 64, floating=True)
            limit = self.const_int(a[0], env)
            if t.text in ("<", "<="):
                return Scalar(Domain.of((-inf, limit - (t.text == "<"))))
            return Scalar(Domain.of((limit + (t.text == ">"), inf)))
        if k == "set":
            values = [self.const_expr(item, env) for item in a[0]]
            if any(isinstance(value, (Fraction, FloatConstant)) for value in values):
                literals = tuple(str(value) if isinstance(value, FloatConstant) and not math.isfinite(value)
                                 else Fraction(value) for value in values)
                return Scalar(float_literal_domain(literals, 64), 64, floating=True, float_literals=literals)
            return Scalar(Domain.of(*((n, n) for n in (self.const_int(item, env) for item in a[0]))))
        if k == "poly":
            terms = self.polynomial(a[0], env)
            denominator = self.limit_constant(t, lcm(*(c.denominator for c in terms)))
            coefficients = tuple(self.limit_constant(t, c.numerator * (denominator // c.denominator)) for c in terms)
            if any(evaluate(coefficients, n) % denominator for n in range(len(coefficients))):
                t.error("poly must produce integers for every natural-number input")
            if len(coefficients) == 1:
                return Scalar(Domain.point(coefficients[0] // denominator))
            poly = Polynomial.reduced(coefficients, denominator)
            return Scalar(Domain.from_clauses([Clause(-inf, inf, (poly,))]))
        if k == "type_apply":
            base, args, end = a
            if (base.kind == "type_name"
                    and self.type_parameter_binding(base.args[0], env) is None
                    and (base.args[0] in self.type_functions or base.args[0] in self.imported_type_functions)):
                if end != ")":
                    t.error("type function arguments must end with ')'")
                return self.compute_type(base.args[0], args, env, t)
            if len(args) != 2:
                t.error("expected two interval bounds or a declared type function")
            bounds = tuple(self.value_argument(arg) for arg in args)
            bound = Node(t, "interval", (*bounds, True, end == ")"))
            return self.resolve(Node(t, "refine", (base, bound)), env)
        if k == "refine":
            base = self.numeric_type(a[0], env)
            if base.floating:
                return base.narrow_domain(base.domain & self.float_set(a[1], base.bits, env))
            bound = self.numeric_type(a[1], env)
            return base.narrow_domain(base.domain & bound.domain)
        if k == "type_op":
            left = self.resolve(a[0], env)
            op = t.text
            if isinstance(left, Scalar) and is_callback_enum(left.enum):
                t.error("callable types cannot be used in numeric set operations")
            if isinstance(left, Scalar) and left.floating:
                operand = a[1]
                while operand.kind == "type_group":
                    operand = operand.args[0]
                named = self.numeric_type(operand, env) if operand.kind == "type_name" and not env else None
                other = named
                if operand.kind == "type_op":
                    carrier_node = operand
                    while carrier_node.kind in ("type_op", "type_group"):
                        carrier_node = carrier_node.args[0]
                    carrier = self.resolve(carrier_node, env)
                    if isinstance(carrier, Scalar) and carrier.floating and not carrier.float_literals:
                        other = self.resolve(operand, env)
                if operand.kind == "type_op" and other is not None:
                    if other.bits != left.bits:
                        t.error("floating set operations require the same physical representation")
                    right = other.domain
                else:
                    right = self.float_set(a[1], left.bits, env)
                domain = left.domain | right if op == "|" else left.domain & right if op == "&" else left.domain - right
                classified = other is not None and other.floating and other.result_arms is not None
                if op != "|":
                    if op == "&" and classified:
                        if left.result_arms is None:
                            return other.narrow_domain(domain)
                        arms = self.intersect_result_arms(t, left.result_arms, other.result_arms, domain)
                        return replace(left, domain=domain, result_arms=arms, float_literals=())
                    return left.narrow_domain(domain)
                arms = None
                if classified:
                    left_arms = left.result_arms or (left.domain, left.domain - left.domain)
                    arms = self.merge_result_arms(t, left_arms, other.result_arms)
                elif named is not None and left.domain and right:
                    discrete = (not named.bits and not named.computed if not named.floating else
                                not right.filters and all(lo == hi for lo, hi in right.ranges))
                    if discrete:
                        success, errors = left.result_arms or (left.domain, left.domain - left.domain)
                        arms = self.merge_result_arms(t, (success, errors), (right - right, right))
                if arms is None and left.result_arms is not None:
                    arms = self.merge_result_arms(t, left.result_arms, (right, right - right))
                return replace(left, domain=domain, result_arms=arms, float_literals=())
            right = self.resolve(a[1], env)
            if isinstance(right, Scalar) and is_callback_enum(right.enum):
                t.error("callable types cannot be used in numeric set operations")
            if isinstance(right, Scalar) and right.floating:
                t.error("write the physical floating type on the left of a numeric set operation")
            if op == "|" and isinstance(left, NullType) and isinstance(right, (BorrowRecord, BorrowScalar, BorrowArray, ExternCallable)):
                return replace(right, nullable=True)
            if op == "|" and isinstance(right, NullType) and isinstance(left, (BorrowRecord, BorrowScalar, BorrowArray, ExternCallable)):
                return replace(left, nullable=True)
            if op == "|" and isinstance(left, (Record, Array, Slice, HeapRecord, HeapObject, HeapArray, AliasResult)):
                if not tagged_error_type(right):
                    a[1].token.error("a composite result requires a named error enum")
                if isinstance(left, Array):
                    left = CopyArray(left)
                if isinstance(left, HeapArray) and left.count is not None:
                    length = Domain.point(self.const_int(left.count, env))
                    if not length.within(left.domain):
                        a[0].token.error("owned array result length does not fit its contract")
                    left = replace(left, count=None, length_bound=length)
                return TaggedResult(left, right)
            if not all(isinstance(item, Scalar) and not item.boolean for item in (left, right)):
                t.error("set operation requires numeric sets or an alias union with null")
            self.spend(t, len(left.domain.ranges) * len(right.domain.ranges))
            domain = {"|": Domain.__or__, "&": Domain.__and__, "\\": Domain.__sub__}[op](left.domain, right.domain)
            if left.character:
                if op in ("&", "\\") or left.character == right.character:
                    return replace(left, domain=domain)
                t.error("character unions require a single encoding")
            if right.character:
                t.error("character types cannot be combined with numeric types on the left")
            if left.enum and (op in ("&", "\\") or left.enum == right.enum):
                return replace(left, domain=domain)
            result_type = right if op == "&" and left.result_arms is None else left
            narrowed = result_type.narrow_domain(domain) if op != "|" else None
            arms = narrowed.result_arms if narrowed is not None else None
            error_enums = narrowed.result_error_enums if narrowed is not None else ()
            if op == "&" and left.result_arms is not None and right.result_arms is not None:
                arms = self.intersect_result_arms(t, left.result_arms, right.result_arms, domain)
                if arms is not None and arms[1]:
                    error_enums = tuple(sorted(set(left.result_error_enums) | set(right.result_error_enums)))
                else:
                    error_enums = ()
            operand = a[1]
            while operand.kind == "type_group":
                operand = operand.args[0]
            if op == "|" and right.result_arms is not None:
                left_arms = left.result_arms or (left.domain, Domain.of())
                arms = self.merge_result_arms(t, left_arms, right.result_arms)
                error_enums = tuple(sorted(set(left.result_error_enums) | set(right.result_error_enums)))
            elif (op == "|" and operand.kind == "type_name"
                    and (right.enum is not None or not right.bits and not right.computed)
                    and left.domain and right.domain and domain.lo != -inf and domain.hi != inf and not env):
                success, errors = left.result_arms or (left.domain, Domain.of())
                arms = self.merge_result_arms(t, (success, errors), (Domain.of(), right.domain))
                identities = set(left.result_error_enums) if left.result_arms else set()
                if right.enum is not None:
                    identities.add(right.enum.identity)
                error_enums = tuple(sorted(identities))
            elif op == "|" and left.result_arms is not None:
                arms = self.merge_result_arms(t, left.result_arms, (right.domain, Domain.of()))
                error_enums = left.result_error_enums
            if left.computed or right.computed:
                return Scalar(domain, computed=True)
            if env:
                return Scalar(domain)
            if domain and (domain.lo == -inf or domain.hi == inf):
                return Scalar(domain)
            carriers = [s for s in (left, right) if s.bits] if op == "|" else [left] if op == "\\" else [
                s for s in (left, right) if s.bits
            ]
            if carriers:
                minimum = max(s.bits for s in carriers) if op == "|" else min(s.bits for s in carriers)
                for bits in WIDTHS:
                    if bits >= minimum:
                        for signed in (any(s.signed for s in carriers), not any(s.signed for s in carriers)):
                            if domain.within(container(bits, signed)):
                                return Scalar(domain, bits, signed, result_arms=arms, result_error_enums=error_enums)
                t.error(f"no fixed-width integer container fits {domain}")
            return Scalar(domain, result_arms=arms, result_error_enums=error_enums)
        if k in ("slice_type", "slice_qualified"):
            element, length, readonly = a[:3]
            passing = a[3] if k == "slice_qualified" else None
            contents = a[4] if k == "slice_qualified" and len(a) > 4 else None
            source = a[5] if k == "slice_qualified" and len(a) > 5 else None
            offset = a[6] if k == "slice_qualified" and len(a) > 6 else None
            field_path = a[7] if k == "slice_qualified" and len(a) > 7 else ()
            if field_path and source is None:
                t.error("field requires from in the same qualifier")
            if offset is not None and source is None:
                t.error("offset requires from in the same qualifier")
            if k == "slice_type":
                elem = self.resolve(element, env)
                if isinstance(elem, EncodingTag):
                    return Slice(self.types["uint8"], readonly=True, text=elem.encoding)
                if self.contains_interface_storage(elem):
                    t.error("slices of records with interface fields are not supported")
                if isinstance(elem, Scalar):
                    elem = physical(elem, t)
                elif not isinstance(elem, (Array, Record)):
                    t.error("slice element requires a scalar, fixed array or record")
                ty = Slice(elem)
            else:
                ty = self.resolve(element, env)
                if source is not None and isinstance(ty, Interface):
                    if source < 0 or field_path or any(value is not None for value in (offset, length, contents, passing)):
                        t.error("interface result requires a nonnegative from parameter index")
                    return InterfaceResult(replace(ty, readonly=ty.readonly or bool(readonly)), source)
                if source is not None and isinstance(ty, (BorrowRecord, BorrowScalar, BorrowArray)):
                    if source < 0 or any(value is not None for value in (offset, length, contents, passing, readonly)):
                        t.error("alias result requires only a nonnegative from parameter index")
                    return AliasResult(ty, source, field_path)
                if field_path and not isinstance(ty, Slice):
                    t.error("field applies only to a borrowed alias or slice result")
                if any(isinstance(step, int) for step in field_path):
                    t.error("indexed field paths require a borrowed alias result")
                if source is not None and not isinstance(ty, Slice):
                    t.error("from requires a borrowed alias or slice result type")
                if contents is not None and not isinstance(ty, Slice):
                    contents.token.error('elem predicate requires a slice or string type')
                if isinstance(ty, ReadOnly):
                    if readonly is False:
                        t.error("cannot remove an immutable qualification")
                    ty, readonly = ty.base, True
                if isinstance(unqualified(ty), HeapArray) and length is not None:
                    if isinstance(ty, ReadOnly):
                        if readonly is False:
                            t.error("cannot remove an immutable qualification")
                        ty, readonly = ty.base, True
                    if contents is not None or source is not None or passing is not None:
                        t.error("owned array length cannot be combined with borrow qualifiers")
                    bound = self.numeric_type(length, env)
                    if bound.floating or not (domain := bound.domain & ty.domain):
                        length.token.error("invalid owned array length predicate")
                    ty = replace(ty, length_bound=domain)
                    return ReadOnly(ty) if readonly else ty
                if isinstance(ty, StaticReadOnly):
                    if readonly is False or passing is not None or length is not None:
                        t.error("rodata storage is immutable and cannot have parameter qualifiers")
                    return ty
                if isinstance(ty, (Scalar, Record, Array, HeapRecord, HeapArray, HeapObject)) and passing is None and readonly is False:
                    return ty
                if isinstance(ty, (Scalar, Record, Array, HeapRecord, HeapArray, HeapObject)) and passing in (None, "copy") and readonly is True:
                    if passing is None or isinstance(ty, Scalar):
                        return ReadOnly(physical(ty, t) if isinstance(ty, Scalar) else ty)
                if isinstance(ty, Interface) and passing in (None, "alias") and length is None:
                    if readonly is False and ty.readonly:
                        t.error("cannot remove an interface's immutable qualification")
                    return replace(ty, readonly=ty.readonly or bool(readonly))
                if isinstance(ty, Record) and passing in (None, "alias") and length is None:
                    return BorrowRecord(ty, readonly=bool(readonly))
                if passing == "copy" and isinstance(ty, Record) and length is None:
                    return CopyRecord(ty, readonly=bool(readonly))
                if isinstance(ty, Array) and length is None:
                    return (CopyArray if passing == "copy" else BorrowArray)(ty, bool(readonly))
                if passing == "alias" and isinstance(ty, Scalar) and length is None:
                    return BorrowScalar(physical(ty, t), bool(readonly))
                if passing == "copy" and isinstance(ty, Scalar):
                    if length is not None or readonly is not None:
                        t.error("copy parameters with extra qualifiers are not yet supported")
                    return CopyRecord(ty) if isinstance(ty, Record) else ty
                if not isinstance(ty, Slice):
                    t.error("slice options require a string or slice type")
            if contents is not None:
                if not isinstance(ty.element, Scalar):
                    contents.token.error("elem predicate requires scalar elements")
                bound = (self.float_set(contents, ty.element.bits, env) if ty.element.floating
                         else self.numeric_type(contents, env).domain)
                bound &= ty.element.domain
                if ty.content_bound is not None:
                    bound &= ty.content_bound
                if not bound:
                    contents.token.error('empty slice element predicate')
                ty = replace(ty, content_bound=bound)
            if source is not None:
                if source < 0:
                    t.error("invalid slice return provenance")
                if isinstance(offset, Node):
                    if not self.const_int_endpoint(offset, env):
                        offset.token.error("symbolic offset requires a function result contract")
                    offset = self.const_int(offset, env)
                if offset is not None and not 0 <= offset <= ty.capacity.hi:
                    t.error("invalid slice return offset")
                ty = replace(ty, source_param=source, source_offset=offset, source_field=field_path)
            if passing == "copy":
                ty = replace(ty, copy=True)
            if readonly is not None:
                if ty.readonly and not readonly:
                    t.error("cannot remove a slice type's immutable qualification")
                ty = replace(ty, readonly=readonly)
            if length is not None:
                length_type = self.numeric_type(length, env)
                if length_type.floating:
                    length.token.error('slice length predicate requires integers')
                domain = length_type.domain & ty.domain
                if not domain or not domain.within(ty.domain):
                    length.token.error(f"slice length contract must fit {ty.domain}")
                ty = replace(ty, length_bound=domain)
            return ty
        if k == "array":
            elem, count = a
            count = self.const_int(count, env)
            if elem.kind == "type_name" and elem.args[0] == "bit":
                if count <= 0 or count % 8:
                    t.error("bit storage length must be a positive multiple of 8")
                if count // 8 >= 1 << 31:
                    t.error("bit storage must occupy fewer than 2^31 bytes")
                return Array(PACKED_BIT, count, packed_bits=True)
            elem = self.resolve(elem, env)
            if isinstance(elem, EncodingTag) or (isinstance(elem, Slice) and isinstance(elem.text, TextEncoding)):
                enc = elem.encoding if isinstance(elem, EncodingTag) else elem.text
                if not 0 < count <= (1 << 31) - 1:
                    t.error("encoded buffer must have positive length and occupy fewer than 2^31 bytes")
                return Array(self.types["uint8"], count, encoding=enc)
            if self.contains_interface_storage(elem):
                t.error("arrays of records with interface fields are not supported")
            if isinstance(elem, Scalar):
                self.check_intersection_type(elem, t)
                elem = physical(elem, t)
            elif not isinstance(elem, (Array, Record)):
                t.error("array element requires a scalar, fixed array or record")
            if not 0 < count <= ((1 << 31) - 1) // storage_size(elem):
                t.error("array must have positive length and occupy fewer than 2^31 bytes")
            return Array(elem, count)
        if k == "seal":
            base, candidates, max_bits = a
            computed = self.numeric_type(base, env)
            if not computed.domain:
                t.error("an empty value set has no physical storage")
            if max_bits <= 0:
                t.error("physical seal requires a positive maximum bit width")
            compatible = []
            for candidate in candidates:
                ty = self.numeric_type(Node(candidate, "type_name", (candidate.text,)), env)
                if not ty.bits or ty.computed:
                    candidate.error("physical seal candidates must have a fixed numeric container")
                if ty.bits > max_bits:
                    continue
                try:
                    if computed.floating:
                        if ty.floating:
                            domain = computed.domain.convert(ty.bits,"exact")
                        elif computed.domain.integer_only:
                            domain = computed.domain.exact_integer_values()
                        else:
                            continue
                    else:
                        domain = FloatDomain.from_integers(computed.domain,ty.bits) if ty.floating else computed.domain
                except FloatDomainError:
                    continue
                if domain.within(ty.domain):
                    compatible.append(replace(ty,domain=domain))
            if not compatible:
                t.error(f"no physical seal candidate up to {max_bits} bits contains {computed.domain}")
            ty = min(compatible, key=lambda ty: ty.bits)
            return replace(ty,computed=False)
        if k == "contract":
            params, result = a[:2]
            property_kind = a[3] if len(a) > 2 else None
            identity_property = property_kind == "identity"
            idempotent_property = property_kind == "idempotent"
            commutative_property = property_kind == "commutative"
            diagonal_property = property_kind == "diagonal"
            if property_kind and result is None:
                a[2].error("algebraic property requires an explicit result type")
            names = set()
            for name, _ in params:
                if name.text in names:
                    name.error("duplicate contract parameter")
                names.add(name.text)
            parameters, bounds, names = [], [], {}
            parameter_expressions = []
            for name, param in params:
                expressions = []
                ty = self.resolve_parameter_bound(param, env, names, parameters, bounds, expressions)
                if self.contains_interface_storage(ty) and not isinstance(ty, (Record, ReadOnly, HeapRecord)):
                    param.token.error("a record with interface fields is local-only")
                if expressions and (not plain_integer_parameter(ty) or not unqualified(ty).domain):
                    param.token.error("invalid dependent integer parameter expression")
                parameter_expressions.extend((len(parameters), op, expr) for op, expr in expressions)
                if len(parameter_expressions) > 64:
                    param.token.error("too many dependent parameter expressions")
                if isinstance(ty, ReadOnly) and isinstance(ty.base, Record):
                    ty = BorrowRecord(ty.base, readonly=True)
                elif isinstance(ty, ReadOnly) and isinstance(ty.base, Array):
                    ty = BorrowArray(ty.base, readonly=True)
                if isinstance(ty, (HeapObject, HeapArray)) and self.contains_interface_storage(ty):
                    param.token.error("a record with interface fields is local-only")
                if isinstance(ty, Record):
                    ty = BorrowRecord(ty)
                if isinstance(ty, Array):
                    ty = BorrowArray(ty)
                if isinstance(ty, ReadOnly) and isinstance(ty.base, (HeapRecord, HeapObject, HeapArray)):
                    param.token.error("an owning heap parameter cannot be immutable")
                if isinstance(ty, (HeapRecord, HeapObject, HeapArray)):
                    record = ty.record if isinstance(ty, HeapRecord) else None
                    if record is not None and self.contains_record_links(record) and not linked_owned_record(ty):
                        param.token.error("an owning linked record requires one self-referential next link")
                    if isinstance(ty, HeapArray) and ty.count is not None:
                        length = self.const_int(ty.count, env)
                        if not Domain.point(length).within(ty.domain):
                            param.token.error("owned array parameter length does not fit its contract")
                if not isinstance(ty, (Scalar, Slice, CopyRecord, BorrowRecord, BorrowScalar, BorrowArray, CopyArray, ReadOnly, Interface, HeapRecord, HeapObject, HeapArray, ExternCallable)):
                    param.token.error("function parameters require a scalar or borrowed slice")
                if isinstance(ty, Scalar):
                    self.check_intersection_type(ty, param.token)
                    ty = physical(ty, param.token)
                if isinstance(ty, ReadOnly):
                    self.check_intersection_type(ty.base, param.token)
                    ty = ReadOnly(physical(ty.base, param.token))
                if self.contains_record_links(ty) and not (graph_or_chain_parameter(ty) or linked_owned_record(ty)):
                    if self.native_signature:
                        # Shape, borrowing and mutability all come from the
                        # pointer operations in a body, so there is nothing to
                        # infer from for a symbol the C side implements.
                        param.token.error(
                            "a native function cannot take a linked record: "
                            "its borrow shape is inferred from the body, and a native "
                            "has none; pass a slice or a fixed array instead")
                    param.token.error(
                        "cannot infer a lifetime contract for this linked record parameter: "
                        "leave the type bare and use the link in the body "
                        "(an explicit 'pass' qualifier keeps it out of inference)")
                names[name.text] = len(parameters)
                parameters.append(ty)
            if not valid_chain_parameters(parameters):
                t.error("a chain contract requires non-null chains and no other writable borrows; writable chains require exclusive borrowing")
            if result is None:
                return Contract(tuple(parameters), None, tuple(bounds), parameter_expressions=tuple(parameter_expressions))
            expression_bounds = []
            offset = result.args[6] if result.kind == "slice_qualified" else None
            if isinstance(offset, Node) and not self.const_int_endpoint(offset, env):
                expression = self.slice_offset_expression(offset, names, env, parameters)
                terms, bias = ((), 0) if expression is not None else self.slice_offset_reference(offset, names, env)
                args = (*result.args[:6], None, *result.args[7:])
                ret = unqualified(self.resolve_parameter_bound(replace(result, args=args), env, names, parameters, bounds))
                if not isinstance(ret, Slice) or ret.source_param is None:
                    offset.token.error("symbolic offset requires a borrowed slice result")
                linear = tuple((factors[0], scale) for factors, scale in terms if len(factors) == 1)
                products = tuple((factors, scale) for factors, scale in terms if len(factors) > 1)
                if expression is not None:
                    indices = offset_expression_parameters(expression)
                    if all(plain_integer_parameter(parameters[index]) for index in indices):
                        try:
                            offset_expression_domain(expression, parameters)
                        except (ValueError, DomainComplexity, ZeroDivisionError) as error:
                            offset.token.error(str(error))
                    ret = replace(ret, offset_expression=expression)
                elif len(linear) == 1 and linear[0][1] == 1:
                    ret = replace(ret, offset_param=linear[0][0], offset_products=products, offset_bias=bias)
                else:
                    ret = replace(ret, offset_terms=linear, offset_products=products, offset_bias=bias)
            else:
                ret = unqualified(self.resolve_parameter_bound(result, env, names, parameters, bounds, expression_bounds))
                if expression_bounds and (not plain_integer_parameter(ret) or not ret.domain or len(expression_bounds) > 64):
                    result.token.error("invalid dependent integer result expression")
            tagged = ret if isinstance(ret, TaggedResult) else None
            ret = success_type(ret)
            if isinstance(ret, Array):
                ret = CopyArray(ret)
            if isinstance(ret, HeapArray) and ret.count is not None:
                length = Domain.point(self.const_int(ret.count, env))
                if not length.within(ret.domain):
                    result.token.error("owned array result length does not fit its contract")
                ret = HeapArray(ret.element, length_bound=length)
            if isinstance(ret, Slice) and not ret.readonly and ret.source_param is None:
                result.token.error("slice return requires a read-only static view")
            if ret is not None and not isinstance(ret, (Scalar, Record, CopyArray, Slice, HeapRecord, HeapObject, HeapArray, AliasResult, InterfaceResult, ExternCallable)):
                result.token.error("unsupported function result type")
            self.check_intersection_type(ret, result.token)
            if self.contains_interface_storage(ret):
                result.token.error("a record with interface fields is local-only")
            if self.contains_record_links(ret) and not linked_owned_record(ret):
                result.token.error("record links across function results require a lifetime contract")
            if not valid_interface_provenance(parameters, ret):
                result.token.error("interface from must select a compatible non-null borrowed parameter")
            ret = self.complete_alias_graph_records(parameters, ret)
            if not valid_alias_provenance(parameters, ret, bounds, parameter_expressions):
                result.token.error("alias from must select a compatible borrowed parameter")
            if not valid_slice_provenance(parameters, ret):
                result.token.error("slice from must select a compatible borrowed parameter")
            if tagged is not None:
                ret = replace(tagged, success=ret)
            if identity_property:
                if not (plain_integer_parameter(parameters[0]) and plain_integer_parameter(ret)):
                    a[2].error("identity property requires an integer parameter and integer result")
                bounds.extend(((1, 0, False, ">=", 0), (1, 0, False, "<=", 0)))
            ret = physical(ret, result.token) if isinstance(ret, Scalar) else ret
            if idempotent_property and not closed_integer_property(parameters[0], ret):
                a[2].error("idempotent property requires an integer result that fits its parameter")
            if commutative_property and (bounds or expression_bounds or parameter_expressions
                    or not commutative_integer_signature(parameters, ret)):
                a[2].error("commutative property requires two matching integer parameters and an independent integer result")
            if diagonal_property and (bounds or expression_bounds or parameter_expressions
                    or not diagonal_integer_signature(parameters, ret)):
                a[2].error("diagonal property requires two matching integer parameters and a result accepting their domain")
            return Contract(tuple(parameters), ret, tuple(bounds), tuple(expression_bounds),
                            tuple(parameter_expressions), identity_property, idempotent_property, commutative_property,
                            diagonal_property)
        raise AssertionError(k)

    def resolve_record_field(self, node, env):
        if node.kind == "type_group":
            return self.resolve_record_field(node.args[0], env)
        if node.kind == "type_op" and node.token.text == "|":
            left, right = node.args
            if left.kind == "type_name" and left.args == ("null",):
                left, right = right, left
            if right.kind == "type_name" and right.args == ("null",):
                while left.kind == "type_group":
                    left = left.args[0]
                if left.kind == "type_name" and self.type_parameter_binding(left.args[0], env) is None:
                    alias = self.type_nodes.get(left.args[0])
                    if alias is not None and alias.kind == "slice_qualified":
                        left = alias
                readonly = False
                if left.kind == "slice_qualified":
                    base, length, readonly, passing, elem, source, offset, field = left.args
                    if passing != "alias" or any(value is not None for value in (length, elem, source, offset)) or field:
                        return self.resolve(node, env)
                    left = base
                if left.kind == "type_name":
                    name = left.args[0]
                    if self.type_parameter_binding(name, env) is None:
                        target = self.type_nodes.get(name)
                        if target is not None and target.kind in ("record", "keyed_record"):
                            token = target.token
                            identity = digest(f"{self.module_path}:{token.line}:{token.col}:{self.source_digest}")
                            return RecordLink(identity, bool(readonly))
                        if isinstance(self.types.get(name), Record):
                            return RecordLink(self.types[name].identity, bool(readonly))
        ty = self.resolve(node, env)
        if isinstance(ty, BorrowRecord) and ty.nullable:
            return RecordLink(ty.record.identity, ty.readonly)
        return ty

    @staticmethod
    def contains_interface_storage(ty):
        ty = success_type(unqualified(ty))
        if isinstance(ty, StaticReadOnly):
            return Compiler.contains_interface_storage(ty.base)
        if isinstance(ty, (BorrowRecord, CopyRecord, HeapRecord)):
            ty = ty.record
        if isinstance(ty, (BorrowArray, CopyArray)):
            ty = ty.array
        if isinstance(ty, (Array, Slice, HeapArray)):
            return Compiler.contains_interface_storage(ty.element)
        if isinstance(ty, AliasResult):
            return Compiler.contains_interface_storage(ty.alias)
        return isinstance(ty, Record) and any(isinstance(unqualified(field), Interface)
                                             for _, field in ty.fields)

    @staticmethod
    def contains_record_links(ty):
        ty = success_type(unqualified(ty))
        if isinstance(ty, RecordLink):
            return True
        if isinstance(ty, (BorrowRecord, CopyRecord, HeapRecord)):
            ty = ty.record
        if isinstance(ty, (Record, Interface)):
            return any(Compiler.contains_record_links(field) for _, field in ty.fields)
        if isinstance(ty, (BorrowArray, CopyArray)):
            ty = ty.array
        return isinstance(ty, (Array, Slice, HeapArray)) and Compiler.contains_record_links(ty.element)

    def link_record(self, ty, token):
        if ty.identity in self.record_types:
            return self.record_types[ty.identity]
        for candidate in self.types.values():
            if isinstance(candidate, Record) and candidate.identity == ty.identity:
                return candidate
        token.error("record link target type must be declared or imported")

    def slice_offset_expression(self, node, names, env, parameters, *, force=False):
        def nonlinear(expr):
            if expr.kind == "binary":
                return (expr.args[0] in OFFSET_BINARY_OPS and expr.args[0] not in ("+", "-", "*")
                        or nonlinear(expr.args[1]) or nonlinear(expr.args[2]))
            return expr.kind in ("unary", "cast", "length")
        if not force and not nonlinear(node):
            return None
        def parse(expr):
            if expr.kind == "length" and expr.args[0].kind == "name" and expr.args[0].args[0] in names:
                index = names[expr.args[0].args[0]]
                param = parameters[index]
                width = offset_length_width(param)
                return ("len", index, width), width, False
            if expr.kind == "name" and expr.args[0] in names:
                index = names[expr.args[0]]
                if not plain_integer_parameter(parameters[index]):
                    expr.token.error("symbolic offset requires integer entry parameters")
                ty = unqualified(parameters[index])
                return ("param", index), ty.bits, ty.signed
            if self.const_int_endpoint(expr, env):
                value = self.const_int(expr, env)
                return ("const", value), 0, value < 0
            if expr.kind == "cast":
                operand, strategy, annotation = expr.args
                target = self.scalar_type(annotation, env)
                if target.boolean or target.floating or target.enum or target.character:
                    expr.token.error("slice offset conversion requires an integer target")
                if strategy not in ("safe", "trunc", "wrap", "sat", "trap", "view"):
                    expr.token.error("unsupported integer offset conversion")
                value, width, signed = parse(operand)
                return ("cast", strategy, target, width, signed, value), target.bits, target.signed
            if expr.kind == "unary" and expr.args[0] in ("+", "-", "~"):
                op, operand = expr.args
                value, width, signed = parse(operand)
                if op == "+":
                    return value, width, signed
                if op == "~" and width and not signed:
                    domain = offset_expression_domain(value, parameters)
                    if not domain.within(container(width, False)):
                        expr.token.error("complement operand exceeds its physical width")
                mask = (-1 if signed or not width else (1 << width) - 1) if op == "~" else 0
                return ("binary", "-", ("const", mask), value), width, signed or op == "-" or not width
            if expr.kind == "binary" and expr.args[0] in OFFSET_BINARY_OPS:
                op, left, right = expr.args
                a, width_a, signed_a = parse(left)
                b, width_b, signed_b = parse(right)
                return ("binary", op, a, b), max(width_a, width_b), signed_a or signed_b
            expr.token.error("symbolic offset requires integer entry parameters and integer arithmetic or bit operations")
        try:
            return decode_offset_expression(parse(node)[0])
        except (InterfaceError, ValueError, DomainComplexity, ZeroDivisionError) as error:
            node.token.error(str(error))

    def slice_offset_reference(self, expr, names, env):
        def parse(node):
            if node.kind == "name" and node.args[0] in names:
                return {frozenset(((names[node.args[0]], 1),)): 1}
            if self.const_int_endpoint(node, env):
                return {frozenset(): self.const_int(node, env)}
            if node.kind == "binary" and node.args[0] in ("+", "-", "*"):
                op, left, right = node.args
                result = combine_offset_polynomial(op, parse(left), parse(right))
                if result is not None:
                    return result
                node.token.error("symbolic offset exceeds polynomial expression limits")
            node.token.error("symbolic offset requires integer entry parameters, constants, and +, - or *")

        polynomial = parse(expr)
        bias = polynomial.pop(frozenset(), 0)
        result = tuple(sorted((tuple(sorted(index for index, power in factors for _ in range(power))), scale)
                              for factors, scale in polynomial.items()))
        if not 0 < len(result) <= 64 or any(abs(scale) > 64 for _, scale in result) or not -(1 << 64) <= bias <= 1 << 64:
            expr.token.error("symbolic offset exceeds the supported polynomial expression limits")
        return result, bias

    def resolve_parameter_bound(self, node, env, names, parameters, bounds, expression_bounds=None):
        if node.kind == "inferred_parameter_type":
            self.spend(node.token)
            return node.args[0]
        relations = []

        def reference(expr):
            length = expr.kind == "length"
            subject = expr.args[0] if length else expr
            if subject.kind == "name" and subject.args[0] in names:
                return names[subject.args[0]], length, 0
            if expr.kind == "binary" and expr.args[0] in ("+", "-"):
                op, left, right = expr.args
                if op == "+" and self.const_int_endpoint(left, env):
                    left, right = right, left
                ref = reference(left)
                if ref is not None and self.const_int_endpoint(right, env):
                    delta = self.const_int(right, env)
                    return ref[0], ref[1], ref[2] + (delta if op == "+" else -delta)
            return None

        def endpoint(expr, op, adjustment=0):
            ref = reference(expr)
            if ref is None:
                def references_parameter(part):
                    return (part.kind == "name" and part.args[0] in names
                            or any(references_parameter(child) for child in part.args if isinstance(child, Node)))

                if expression_bounds is None or not references_parameter(expr):
                    return expr
                expression = self.slice_offset_expression(expr, names, env, parameters, force=True)
                try:
                    domain = dependent_expression_domain(expression, parameters)
                    if adjustment:
                        expression = decode_offset_expression(("binary", "+", expression, ("const", adjustment)))
                        dependent_expression_domain(expression, parameters)
                except (ValueError, InterfaceError, DomainComplexity) as exc:
                    expr.token.error(str(exc))
                expression_bounds.append((op, expression))
                edge = domain.lo if op == ">=" else domain.hi
                return Node(expr.token, "number", (edge,))
            source, length, offset = ref
            domain = dependent_source_domain(parameters[source], length)
            if domain is None or not -(1 << 64) <= offset + adjustment <= 1 << 64:
                expr.token.error("invalid dependent parameter bound")
            relations.append((len(parameters), source, length, op, offset + adjustment))
            edge = domain.lo if op == ">=" else domain.hi
            return Node(expr.token, "number", (edge + offset,))

        def envelope(item):
            if item.kind == "interval":
                lower, upper, open_left, open_right = item.args
                args = (endpoint(lower, ">=", int(open_left)), endpoint(upper, "<=", -int(open_right)),
                        open_left, open_right)
            elif item.kind == "bound":
                op = item.token.text
                args = (endpoint(item.args[0], "<=" if op.startswith("<") else ">=",
                                 -1 if op == "<" else int(op == ">")),)
            elif item.kind in ("refine", "type_group") or item.kind == "type_op" and item.token.text == "&":
                args = tuple(envelope(arg) for arg in item.args)
            elif item.kind == "slice_qualified":
                args = (envelope(item.args[0]), *item.args[1:])
            else:
                return item
            return replace(item, args=args)

        ty = self.resolve(envelope(node), env)
        if relations:
            if not plain_integer_parameter(ty):
                node.token.error("dependent bounds require an integer parameter or result")
            if not unqualified(ty).domain:
                node.token.error("dependent integer interval is empty")
            bounds.extend(relations)
            if len(bounds) > 8192:
                node.token.error("too many dependent bounds in one contract")
        return ty

    def float_set(self, node, bits, env=()):
        kind, args, token = node.kind, node.args, node.token
        if kind == "type_group":
            return self.float_set(args[0], bits, env)
        if kind == "interval":
            left, right, open_left, open_right = args
            lo, hi = self.const_expr(left, env), self.const_expr(right, env)
            first, last = quantize(lo, bits, "up"), quantize(hi, bits, "down")
            if open_left and is_representable(lo, bits):
                first = next_float(first, bits, 1)
            if open_right and is_representable(hi, bits):
                last = next_float(last, bits, -1)
            return FloatDomain.interval(first, last, bits) if lo <= hi else FloatDomain(bits)
        if kind == "bound":
            value = self.const_expr(args[0], env)
            bound = quantize(value, bits, "down" if token.text in ("<", "<=") else "up")
            if token.text in ("<", ">") and is_representable(value, bits):
                bound = next_float(bound, bits, -1 if token.text == "<" else 1)
            return FloatDomain.interval(-inf, bound, bits) if token.text in ("<", "<=") else FloatDomain.interval(bound, inf, bits)
        if kind == "set":
            result = FloatDomain(bits)
            for item in args[0]:
                value = self.const_expr(item, env)
                result |= FloatDomain(bits, zeros=3) if value == 0 else FloatDomain.point(value, bits)
            return result
        if kind == "type_op":
            left_type,right_type = (self.resolve(item,env) for item in args)
            if all(isinstance(ty,Scalar) and not ty.floating for ty in (left_type,right_type)):
                return FloatDomain.integer_set(self.resolve(node,env).domain,bits)
            a, b = (self.float_set(item, bits, env) for item in args)
            return a | b if token.text == "|" else a & b if token.text == "&" else a - b
        ty = self.resolve(node, env)
        if not isinstance(ty, Scalar) or ty.boolean or ty.enum or ty.character:
            token.error("floating refinement requires a numeric value set")
        if ty.floating:
            if ty.bits != bits:
                if ty.float_literals:
                    return float_literal_domain(ty.float_literals, bits)
                token.error("floating set operations require the same physical representation")
            return ty.domain
        return FloatDomain.integer_set(ty.domain,bits)

    def intersect_result_arms(self, token, left, right, domain):
        arms = self.merge_result_arms(token, tuple(arm & domain for arm in left),
                                      tuple(arm & domain for arm in right))
        return arms if any(arms) else None

    def merge_result_arms(self, token, left, right):
        success = left[0] | right[0]
        errors = left[1] | right[1]
        if self.proven_presence(success & errors, token):
            token.error("success and error values must be disjoint for a zero-tag result union")
        return success, errors

    def numeric_type(self, node, env=()):
        ty = self.resolve(node, env)
        if isinstance(ty, Scalar) and is_callback_enum(ty.enum):
            node.token.error("callable types cannot be used as numeric types")
        if not isinstance(ty, Scalar) or ty.boolean:
            node.token.error("expected an integer type or set")
        return ty

    def scalar_type(self, node, env=()):
        ty = self.resolve(node, env)
        if not isinstance(ty, Scalar):
            node.token.error("expected a scalar type")
        self.check_intersection_type(ty, node.token)
        return physical(ty, node.token)

    def polynomial(self, node, env):
        t, kind, args = node.token, node.kind, node.args
        self.spend(t)
        if kind == "name" and args[0] == "n":
            return 0, 1
        if kind == "unary" and args[0] in ("+", "-"):
            result = self.polynomial(args[1], env)
            return result if args[0] == "+" else tuple(-c for c in result)
        if kind != "binary" or args[0] not in ("+", "-", "*", "/"):
            if self.uses_poly_generator(node):
                t.error("expressions involving the poly generator n support +, -, * and division by constants")
            return (self.const_int(node, env),)
        op, lhs, rhs = args
        left, right = self.polynomial(lhs, env), self.polynomial(rhs, env)
        if op == "/":
            if len(right) != 1 or not right[0]:
                t.error("poly division requires a nonzero constant divisor")
            result = [Fraction(c) / right[0] for c in left]
        elif op == "*":
            result = [0] * (len(left) + len(right) - 1)
            if len(result) > 65:
                t.error("poly degree exceeds the computation limit of 64")
            self.spend(t, len(left) * len(right))
            for i, x in enumerate(left):
                for j, y in enumerate(right):
                    result[i + j] += x * y
        else:
            result = [0] * max(len(left), len(right))
            for i, x in enumerate(left):
                result[i] += x
            for i, y in enumerate(right):
                result[i] += y if op == "+" else -y
        while len(result) > 1 and not result[-1]:
            result.pop()
        for c in result:
            self.limit_constant(t, c.numerator)
            self.limit_constant(t, c.denominator)
        return tuple(result)

    @staticmethod
    def uses_poly_generator(node):
        pending = [node]
        while pending:
            item = pending.pop()
            if isinstance(item, Node):
                if item.kind == "name" and item.args[0] == "n":
                    return True
                pending.extend(item.args)
            elif isinstance(item, tuple):
                pending.extend(item)
        return False
