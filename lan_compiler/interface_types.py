"""Type encoding and decoding for .table interfaces, together with the alias, chain,
interface-rebind and offset-expression provenance checks the decoder relies on."""

from __future__ import annotations

import re
from dataclasses import field, fields, replace
from fractions import Fraction
from math import gcd, inf

from bit_domain import bitwise_domain
from float_domain import FloatDomain, FloatDomainError
from value_domain import Domain, DomainComplexity
from table_interface import InterfaceError, decode_domain, encode_domain, fields as interface_fields

from .encoding import EncodingTag, TextEncoding
from .lexer import CHARACTERS, NAME, RESERVED, Token
from .syntax import Node, TYPE_OBJECT_WORDS, TypeObject
from .typesys import (AliasIndexExpression, AliasParamIndex, AliasResult, Array, BOOL, BorrowArray,
                      BorrowRecord, BorrowScalar, CallableType, contains_callable, Contract, CopyArray,
                      CopyRecord, EnumInfo, ExternCallable, HeapArray, HeapObject, HeapRecord, Interface, InterfaceResult,
                      NullType, PACKED_BIT, ReadOnly, Record, record_text_type, RecordLink, Scalar, Slice,
                      SliceEndIndex, storage_size, success_type, tagged_error_type, TaggedResult,
                      unqualified, ValueOrigin, WIDE, WIDTHS, ZERO)
from .domain_utils import (closed_integer_property, commutative_integer_signature, container,
                           diagonal_integer_signature, float_literal_domain, plain_integer_parameter,
                           trunc_div)
from .state import PathState

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


def encode_alias_path(path):
    return [encode_domain(step) if isinstance(step, Domain) else
            {"end": step.distance} if isinstance(step, SliceEndIndex) else
            {"param": step.parameter, "offset": step.offset} if isinstance(step, AliasParamIndex) else
            {"expr": encode_offset_expression(step.expression)} if isinstance(step, AliasIndexExpression) else step
            for step in path]


def encode_text_encoding(enc: TextEncoding):
    return {"identity": enc.identity, "name": enc.name, "terminator": enc.terminator.hex(), "pad": enc.pad}


def decode_text_encoding(raw) -> TextEncoding:
    if not isinstance(raw, dict):
        raise InterfaceError("invalid custom text encoding in interface")
    interface_fields(raw, ("identity", "name", "terminator", "pad"), "text encoding")
    identity, name, terminator_hex, pad = raw["identity"], raw["name"], raw["terminator"], raw["pad"]
    if (not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity)
            or not isinstance(name, str) or not name
            or not isinstance(terminator_hex, str) or len(terminator_hex) % 2 != 0
            or type(pad) is not int or not 0 <= pad <= 255):
        raise InterfaceError("invalid custom text encoding in interface")
    try:
        terminator = bytes.fromhex(terminator_hex)
    except ValueError as exc:
        raise InterfaceError("invalid terminator hex in custom text encoding") from exc
    return TextEncoding(identity=identity, name=name, terminator=terminator, pad=pad)


def encode_interface_type(ty):
    if isinstance(ty, EncodingTag):
        return {"kind": "encoding_tag", "encoding": encode_text_encoding(ty.encoding)}
    if isinstance(ty, TaggedResult):
        return {"kind": "tagged_result", "success": encode_interface_type(ty.success),
                "error": encode_interface_type(ty.error)}
    if isinstance(ty, CallableType):
        # Internal type identity for records and arrays. This kind is not
        # accepted by the table importer.
        return {"kind": "callable", "module": ty.enum.identity, "selectors": encode_domain(ty.domain),
                "candidates": list(ty.candidates), "contract": encode_interface_type(ty.contract)}
    if isinstance(ty, ExternCallable):
        result = {"kind": "extern_callable", "contract": encode_interface_type(ty.contract)}
        if ty.mutations:
            result["writes"] = [[index, list(path)] for index, path, _ in ty.mutations]
        if ty.memory:
            result["memory"] = True
        if ty.external_effect:
            result["external"] = True
        if ty.nullable:
            result["nullable"] = True
        return result
    if isinstance(ty, TypeObject):
        entries = []
        for name, value in ty.entries:
            if value.kind == "object_value":
                property_value = {"value": value.args[0].args[0]}
            elif value.kind == "object_word":
                property_value = {"word": value.args[0]}
            elif value.kind == "object_path":
                property_value = {"path": list(value.args[0])}
            else:
                property_value = {"type": encode_interface_type(value.args[0])}
            entries.append([name.text, property_value])
        return {"kind": "type_object", "properties": entries}
    if isinstance(ty, NullType):
        return {"kind": "null"}
    if isinstance(ty, ReadOnly):
        return {"kind": "immutable", "base": encode_interface_type(ty.base)}
    if ty is None:
        return {"kind": "void"}
    if isinstance(ty, Scalar):
        if ty.result_arms is not None:
            good, error = ty.result_arms
            encode = FloatDomain.to_dict if ty.floating else encode_domain
            result = {"kind": "result_union", "base": encode_interface_type(replace(ty, result_arms=None)),
                      "success": encode(good), "error": encode(error)}
            if ty.result_error_enums:
                result["error_enums"] = list(ty.result_error_enums)
            return result
        if ty.floating:
            result = {"kind": "float", "bits": ty.bits, "domain": ty.domain.to_dict()}
            if (0 < len(ty.float_literals) <= 4096 and all(
                    isinstance(value, str) or max(value.numerator.bit_length(), value.denominator.bit_length()) <= 4096
                    for value in ty.float_literals)):
                result["literals"] = [value if isinstance(value, str) else [value.numerator, value.denominator]
                                      for value in ty.float_literals]
            return result
        if ty.character is not None:
            return {"kind": "character", "encoding": ty.character, "domain": encode_domain(ty.domain)}
        if ty.enum is not None:
            return {"kind": "enum", "identity": ty.enum.identity, "name": ty.enum.name,
                    "members": [list(member) for member in ty.enum.members],
                    "base": encode_interface_type(replace(ty, enum=None))}
        return {"kind": "scalar", "bits": ty.bits, "signed": ty.signed,
                "boolean": ty.boolean, "computed": ty.computed, "domain": encode_domain(ty.domain)}
    if isinstance(ty, Array):
        result = {"kind": "array", "element": encode_interface_type(ty.element),
                  "length": ty.length, "packed": ty.packed_bits}
        if ty.encoding is not None:
            result["encoding"] = encode_text_encoding(ty.encoding)
        return result
    if isinstance(ty, Slice):
        result = {"kind": "slice", "element": encode_interface_type(ty.element),
                  "len": encode_domain(ty.domain), "readonly": ty.readonly}
        if ty.chain_limit is not None:
            result["chain"] = ty.chain_limit
        if ty.prune:
            result["prune"] = True
        if ty.attach is not None:
            result["attach"] = ty.attach
        if ty.graph:
            result["graph"] = True
        if ty.copy:
            result["copy"] = True
        if ty.text is not None:
            result["text"] = encode_text_encoding(ty.text) if isinstance(ty.text, TextEncoding) else ty.text
        if ty.source_param is not None:
            result['from'] = ty.source_param
        if ty.source_field:
            result["field"] = list(ty.source_field)
        if ty.source_offset is not None:
            result['offset'] = ty.source_offset
        if ty.offset_param is not None:
            result["offset_param"] = ty.offset_param
        if ty.offset_terms:
            result["offset_terms"] = [list(term) for term in ty.offset_terms]
        if ty.offset_products:
            result["offset_products"] = [[list(factors), scale] for factors, scale in ty.offset_products]
        if ty.offset_bias:
            result["offset_bias"] = ty.offset_bias
        if ty.offset_expression is not None:
            result["offset_expression"] = encode_offset_expression(ty.offset_expression)
        if ty.content_bound is not None:
            result['elem'] = encode_interface_type(replace(ty.element, domain=ty.content_bound))
        return result
    if isinstance(ty, AliasResult):
        result = {"kind": "alias_result", "alias": encode_interface_type(ty.alias), "from": ty.source}
        if ty.field_path:
            result["field"] = encode_alias_path(ty.field_path)
        if ty.alternatives:
            result["alternatives"] = [
                [source, encode_alias_path(path)]
                for source, path in ty.alternatives]
        if ty.graph_records:
            result["graph_records"] = [encode_interface_type(record) for record in ty.graph_records]
        return result
    if isinstance(ty, HeapRecord):
        return {"kind": "record_owner", "record": encode_interface_type(ty.record)}
    if isinstance(ty, HeapObject):
        return {"kind": "object_owner", "base": encode_interface_type(ty.base)}
    if isinstance(ty, HeapArray) and ty.count is None:
        return {"kind": "array_owner", "element": encode_interface_type(ty.element),
                "len": encode_domain(ty.domain)}
    if isinstance(ty, CopyRecord):
        result = {"kind": "record_copy", "record": encode_interface_type(ty.record)}
        if ty.readonly:
            result["readonly"] = True
        return result
    if isinstance(ty, (BorrowArray, CopyArray)):
        result = {"kind": "array_borrow" if isinstance(ty, BorrowArray) else "array_copy",
                  "array": encode_interface_type(ty.array), "readonly": ty.readonly}
        if isinstance(ty, BorrowArray) and ty.nullable:
            result["nullable"] = True
        return result
    if isinstance(ty, BorrowScalar):
        result = {"kind": "scalar_borrow", "scalar": encode_interface_type(ty.scalar), "readonly": ty.readonly}
        if ty.nullable:
            result["nullable"] = True
        return result
    if isinstance(ty, BorrowRecord):
        result = {"kind": "record_borrow", "record": encode_interface_type(ty.record), "readonly": ty.readonly}
        if ty.chain_limit is not None:
            result["chain"] = ty.chain_limit
        if ty.prune:
            result["prune"] = True
        if ty.attach is not None:
            result["attach"] = ty.attach
        if ty.graph:
            result["graph"] = True
        if ty.nullable:
            result["nullable"] = True
        return result
    if isinstance(ty, RecordLink):
        return {"kind": "record_link", "identity": ty.identity, "readonly": ty.readonly}
    if isinstance(ty, InterfaceResult):
        return {"kind": "duck_result", "view": encode_interface_type(ty.interface), "from": ty.source}
    if isinstance(ty, Interface):
        return {"kind": "duck_interface", "schema": encode_interface_type(ty.schema), "readonly": ty.readonly}
    if isinstance(ty, Record):
        result = {"kind": "record", "identity": ty.identity, "repr_c": ty.repr_c,
                  "fields": [[name, encode_interface_type(field)] for name, field in ty.fields]}
        if ty.key_type is not None:
            result["key_type"] = encode_interface_type(ty.key_type)
        return result
    if isinstance(ty, Contract):
        result = {"kind": "contract", "params": [encode_interface_type(p) for p in ty.params],
                  "result": encode_interface_type(ty.result)}
        if ty.bounds:
            result["bounds"] = [list(bound) for bound in ty.bounds]
        if ty.result_expressions:
            result["result_expressions"] = [[op, encode_offset_expression(expr)]
                                            for op, expr in ty.result_expressions]
        if ty.parameter_expressions:
            result["parameter_expressions"] = [[index, op, encode_offset_expression(expr)]
                                               for index, op, expr in ty.parameter_expressions]
        if ty.identity_property:
            result["identity_property"] = True
        if ty.idempotent_property:
            result["idempotent_property"] = True
        if ty.commutative_property:
            result["commutative_property"] = True
        if ty.diagonal_property:
            result["diagonal_property"] = True
        return result
    raise InterfaceError("type cannot be represented in a .table interface")


def dependent_source_domain(ty, length):
    if length:
        if isinstance(ty, Slice):
            return ty.domain
        if isinstance(ty, (BorrowArray, CopyArray)):
            return Domain.point(ty.array.length)
        return None
    return unqualified(ty).domain if plain_integer_parameter(ty) else None


def dependent_expression_domain(expr, params, *, refine=None):
    def validate(part):
        if part[0] == "binary":
            for operand in part[2:]:
                validate(operand)
        elif part[0] == "cast":
            validate(part[5])
            if part[1] == "trap" and not offset_expression_domain(part[5], params, refine=refine).within(part[2].domain):
                raise ValueError("dependent expression conversion may trap")
        elif part[0] == "param":
            if not plain_integer_parameter(params[part[1]]):
                raise ValueError("dependent expressions require integer parameters")
        elif part[0] not in ("len", "const"):
            raise ValueError("unsupported dependent expression")

    validate(expr)
    return offset_expression_domain(expr, params, refine=refine)


ALIAS_PATH_LIMIT = 4096
CHAIN_REACHABLE = "@chain_reachable"
CHAIN_ALIAS_FALLBACK_DEPTH = 8
GRAPH_ALIAS_FALLBACK_DEPTH = 4


def chain_alias_path(source, path):
    if not isinstance(source, BorrowRecord) or not source.readonly or source.chain_limit is None:
        return False
    return 0 < len(path) <= ALIAS_PATH_LIMIT and all(step == "next" for step in path)


def chain_reachable_path(source, path):
    return (path == (CHAIN_REACHABLE,) and isinstance(source, BorrowRecord)
            and source.readonly and source.chain_limit is True)


def graph_alias_link(source, path, records=(), *, intermediates=None):
    if (not isinstance(source, BorrowRecord) or not source.readonly or not source.graph
            or not 0 < len(path) <= ALIAS_PATH_LIMIT):
        return None
    available = {record.identity: record for record in records}
    available[source.record.identity] = source.record
    record = source.record
    visited = []
    for index, step in enumerate(path):
        fields = dict(record.fields)
        field = unqualified(fields.get(step)) if isinstance(step, str) else None
        if not isinstance(field, RecordLink):
            return None
        if index + 1 == len(path):
            if intermediates is not None:
                intermediates.update((item.identity, item) for item in visited)
            return field
        record = available.get(field.identity)
        if record is None:
            return None
        visited.append(record)
    return None


def alias_graph_records(params, alias, extra=()):
    records = [ty.record for ty in params if isinstance(ty, BorrowRecord)]
    if isinstance(alias, BorrowRecord):
        records.append(alias.record)
    return [*extra, *records]


def alias_graph_intermediates(params, result, available):
    known = alias_graph_records(params, result.alias)
    records = {}
    for index, path in result.origins():
        if 0 <= index < len(params):
            graph_alias_link(params[index], path, [*available, *known], intermediates=records)
    for record in known:
        records.pop(record.identity, None)
    return tuple(records.values())


def linked_alias_path(source, path, records=()):
    return chain_alias_path(source, path) or graph_alias_link(source, path, records) is not None


def valid_dependent_alias_index(expression, params, bounds, parameter_expressions, length, slice_source=None):
    limit = Domain.of((0, length - 1))
    try:
        domain = dependent_expression_domain(expression, params)
    except ValueError:
        if not (bounds or parameter_expressions):
            raise
        domain = None
    if domain is not None and domain.within(limit):
        return bool(domain)
    if not (bounds or parameter_expressions or slice_source is not None):
        return False
    origins = {}

    def polynomial(part):
        kind = part[0]
        if kind == "const":
            return {frozenset(): part[1]}
        if kind in ("param", "len"):
            value = offset_expression_domain(part, params)
            if value.lo == value.hi:
                return {frozenset(): value.lo}
            origin = origins.setdefault(part, ValueOrigin(value))
            return {frozenset(((origin, 1),)): 1}
        if kind == "binary" and part[1] in ("+", "-", "*", "&", "/", "%", "<<", ">>"):
            left, right = map(polynomial, part[2:])
            if left is not None and right is not None:
                op = part[1]
                if op in ("&", "/", "%", "<<", ">>"):
                    operands = tuple((frozenset((term, scale) for term, scale in form.items() if term),
                                      form.get(frozenset(), 0)) for form in (left, right))
                    atom = (("integer-op", op, frozenset(operands)) if op == "&"
                            else ("integer-op", op, *operands))
                    return {frozenset(((atom, 1),)): 1}
                return combine_offset_polynomial(op, left, right)
        return None

    form = polynomial(expression)
    if form is None:
        return False
    facts = {}
    relations = list(parameter_expressions)
    for target, source, length, op, delta in bounds:
        if target < len(params):
            base = ("len", source, offset_length_width(params[source])) if length else ("param", source)
            relations.append((target, op, ("binary", "+", base, ("const", delta))))
    for target, op, right in relations:
        left = ("param", target)
        if op == ">=":
            left, right = right, left
        constraint = polynomial(("binary", "-", left, right))
        if constraint is None:
            continue
        upper = -constraint.pop(frozenset(), 0)
        if constraint:
            scale = gcd(*constraint.values())
            terms = frozenset((term, coefficient // scale) for term, coefficient in constraint.items())
            key = ("dependent_upper", terms)
            facts[key] = min(facts.get(key, inf), upper // scale)
    state = PathState({}, predicates=facts)

    def upper_bound(poly, sign=1):
        terms = frozenset((term, sign * scale) for term, scale in poly.items() if term)
        bias = sign * poly.get(frozenset(), 0)
        return bias + (Compiler.dependent_upper_bound(terms, state) if terms else 0)

    if domain is None:
        def refine(part, possible):
            subexpression = polynomial(part)
            if subexpression is None:
                return possible
            lo, hi = -upper_bound(subexpression, -1), upper_bound(subexpression)
            return (possible & Domain.of((lo, hi))) or possible

        domain = dependent_expression_domain(expression, params, refine=refine)
    if not domain:
        return False
    if domain.within(limit):
        return True
    if domain.lo < 0 and upper_bound(form, -1) > 0:
        return False
    if upper_bound(form) < length:
        return True
    if slice_source is None:
        return False
    source_length = polynomial(("len", slice_source, offset_length_width(params[slice_source])))
    difference = combine_offset_polynomial("-", form, source_length)
    return difference is not None and upper_bound(difference) < 0


def valid_alias_provenance(params, result, bounds=(), parameter_expressions=()):
    if not isinstance(result, AliasResult):
        return True
    if result.alternatives:
        return all(valid_alias_provenance(params, replace(result, source=source, field_path=path, alternatives=()),
                                          bounds, parameter_expressions)
                   for source, path in result.origins())
    if not 0 <= result.source < len(params):
        return False
    source = params[result.source]
    records = alias_graph_records(params, result.alias, result.graph_records)
    if chain_reachable_path(source, result.field_path):
        return (isinstance(result.alias, BorrowRecord) and result.alias.readonly
                and result.alias.record.identity == source.record.identity)
    if linked_alias_path(source, result.field_path, records):
        link = graph_alias_link(source, result.field_path, records)
        identity = link.identity if link is not None else source.record.identity
        return (isinstance(result.alias, BorrowRecord) and result.alias.record.identity == identity
                and result.alias.readonly)
    if result.field_path:
        if isinstance(source, Slice):
            if source.copy or source.text is not None:
                return False
            field = source
        elif isinstance(source, BorrowRecord):
            field = source.record
        elif isinstance(source, BorrowArray):
            field = source.array
        else:
            return False
        readonly = source.readonly
        for name in result.field_path:
            if isinstance(name, SliceEndIndex):
                if (field is not source or not isinstance(source, Slice)
                        or not 1 <= name.distance <= source.domain.hi
                        or not valid_dependent_alias_index(("const", name.distance - 1), params, bounds,
                                                            parameter_expressions, source.domain.lo,
                                                            result.source)):
                    return False
                child = source.element
            elif isinstance(name, (int, Domain, AliasParamIndex, AliasIndexExpression)):
                if field is source and isinstance(source, Slice):
                    maximum_length = (source.domain.lo if isinstance(name, (AliasParamIndex, AliasIndexExpression))
                                      else source.domain.hi)
                elif isinstance(field, Array) and not field.packed_bits:
                    maximum_length = field.length
                else:
                    return False
                if isinstance(name, AliasIndexExpression):
                    slice_source = result.source if field is source and isinstance(source, Slice) else None
                    try:
                        if not valid_dependent_alias_index(name.expression, params, bounds, parameter_expressions,
                                                           maximum_length, slice_source):
                            return False
                    except (IndexError, ValueError):
                        return False
                elif isinstance(name, AliasParamIndex):
                    if not 0 <= name.parameter < len(params) or not plain_integer_parameter(params[name.parameter]):
                        return False
                    indices = params[name.parameter].domain.affine(1, name.offset)
                else:
                    indices = Domain.point(name) if isinstance(name, int) else name
                if not isinstance(name, AliasIndexExpression):
                    if not indices or indices.lo < 0 or indices.hi >= maximum_length:
                        return False
                    if (field is source and isinstance(source, Slice) and not result.alias.nullable
                            and indices.lo >= source.domain.lo
                            and not valid_dependent_alias_index(("const", indices.lo), params, bounds,
                                                                parameter_expressions, source.domain.lo,
                                                                result.source)):
                        return False
                child = field.element
            elif isinstance(field, Record):
                child = dict(field.fields).get(name)
                if child is None:
                    return False
            else:
                return False
            readonly |= isinstance(child, ReadOnly)
            field = unqualified(child)
        target = (result.alias.record if isinstance(result.alias, BorrowRecord) else
                  result.alias.scalar if isinstance(result.alias, BorrowScalar) else result.alias.array)
        return (field == target and (result.alias.readonly or not readonly)
                and (result.alias.nullable or not getattr(source, "nullable", False)))
    if (isinstance(source, BorrowRecord) and source.readonly
            and isinstance(result.alias, BorrowRecord) and result.alias.readonly
            and result.alias == BorrowRecord(source.record, readonly=True, nullable=result.alias.nullable)):
        return result.alias.nullable or not source.nullable
    return (type(source) is type(result.alias)
            and replace(source, readonly=result.alias.readonly, nullable=result.alias.nullable) == result.alias
            and (result.alias.readonly or not source.readonly)
            and (result.alias.nullable or not source.nullable))


def interface_mismatch(actual_fields, readonly, view):
    fields = dict(actual_fields)
    for name, field in view.fields:
        expected = unqualified(field)
        actual = fields.get(name)
        if actual is None:
            return f"interface argument is missing field {name!r}"
        frozen = view.readonly or isinstance(field, ReadOnly)
        if not frozen and (readonly or isinstance(actual, ReadOnly)):
            return f"interface field {name!r} requires mutable storage"
        actual = unqualified(actual)
        if contains_callable(actual) and not contains_callable(expected):
            return f"interface field {name!r} has an incompatible type"
        if actual == expected:
            continue
        if (isinstance(actual, Slice) and isinstance(expected, Slice)
                and encode_interface_type(actual) == encode_interface_type(expected)):
            continue
        if (frozen and isinstance(actual, Scalar) and isinstance(expected, Scalar)
                and replace(actual, domain=expected.domain) == expected and actual.domain.within(expected.domain)):
            continue
        return f"interface field {name!r} has an incompatible type"
    return None


def valid_interface_provenance(params, result):
    if not isinstance(result, InterfaceResult):
        return True
    if not 0 <= result.source < len(params):
        return False
    source = params[result.source]
    if isinstance(source, BorrowRecord) and not source.nullable:
        fields = source.record.fields
    elif isinstance(source, Interface):
        fields = source.fields
    else:
        return False
    return interface_mismatch(fields, source.readonly, result.interface) is None


def valid_interface_rebind(params, index, name, donor):
    if donor is None or index == donor:
        return False
    receiver, source = params[index], params[donor]
    if (not isinstance(receiver, BorrowRecord) or receiver.readonly or receiver.nullable
            or not isinstance(source, BorrowRecord) or source.nullable
            or graph_or_chain_parameter(receiver) or graph_or_chain_parameter(source)):
        return False
    view = dict(receiver.record.fields).get(name)
    return (isinstance(view, Interface) and not view.readonly
            and interface_mismatch(source.record.fields, source.readonly, view) is None)


def interface_rebind_guard(params, index):
    return (0 <= index < len(params) and isinstance(ty := unqualified(params[index]), Scalar)
            and ty.boolean)


def direct_scalar_record_write(param, names):
    if (not names or not isinstance(param, BorrowRecord) or param.readonly or param.nullable
            or graph_or_chain_parameter(param)):
        return False
    ty = param.record
    for name in names:
        if not isinstance(ty, Record):
            return False
        ty = dict(ty.fields).get(name)
    return isinstance(ty, Scalar)


def interface_rebind_array_write(param, names):
    return (bool(names) and isinstance(param, BorrowRecord) and not param.readonly and not param.nullable
            and not graph_or_chain_parameter(param) and array_mutation_step(names[-1])
            and not any(map(array_mutation_step, names[:-1]))
            and array_mutation_projection(param.record, names) is not None)


def valid_interface_rebind_effects(params, writes):
    # Guarded arms of one field may repeat: the summary is a may-set, so the
    # caller unions the arms instead of picking one.  An unguarded rebind
    # overwrites whatever came before it, so pairing one with another arm would
    # need an order the summary does not carry.
    fields = {}
    for index, names in writes:
        if names and names[0] in ("@interface_before", "@interface_after"):
            route = names[1:]
            param = params[index] if 0 <= index < len(params) else None
            if (not isinstance(param, BorrowRecord) or "*" not in route
                    or len(route) < 2):
                return False
            continue
        if names and names[0] == "@interface_rebind":
            if len(names) < 3:
                return False
            fields.setdefault((index, names[1]), []).append(names)
        elif not (direct_scalar_record_write(params[index], names)
                  or interface_rebind_array_write(params[index], names)):
            return False
    if any(len(entries) > 1 and any(len(entry) == 3 for entry in entries)
           for entries in fields.values()):
        return False
    return bool(fields)


def interface_rebind_expression(names):
    """Parse the compact summary back into a target expression.

    ``("old",)`` is the value the field had before this setter ran; a donor is
    a final target and a condition selects between two expressions.
    """
    try:
        if len(names) == 3:
            return "donor", int(names[2])
        if len(names) == 5:
            return "cond", int(names[3]), int(names[4]), ("donor", int(names[2])), ("old",)
        if len(names) == 6:
            return ("cond", int(names[3]), int(names[4]),
                    ("donor", int(names[2])), ("donor", int(names[5])))
    except (TypeError, ValueError):
        return None
    return None


def compose_interface_rebind(current, effect):
    """Apply one rebind after ``current`` and keep the final target expression."""
    if effect is None:
        return None
    if effect[0] == "donor":
        return effect
    if not effect or effect[0] != "cond":
        return None
    _, guard, expected, when_true, when_false = effect
    if when_false != ("old",):
        # An exhaustive two-arm rebind overwrites both possible old values.
        return effect
    if current == ("old",) or current[0] == "donor":
        return "cond", guard, expected, when_true, current
    if current[0] != "cond" or current[1] != guard:
        return None
    if current[2] == expected:
        return "cond", guard, expected, when_true, current[4]
    # The new single-arm write covers the branch this condition did not.
    return "cond", guard, expected, when_true, current[3]


def encode_interface_rebind(field, expression):
    if expression[0] == "donor":
        return "@interface_rebind", field, str(expression[1])
    if expression[0] != "cond":
        return None
    _, guard, expected, when_true, when_false = expression
    if when_true[0] != "donor":
        return None
    if when_false == ("old",):
        return "@interface_rebind", field, str(when_true[1]), str(guard), str(expected)
    if when_false[0] != "donor":
        return None
    if when_true[1] == when_false[1]:
        return "@interface_rebind", field, str(when_true[1])
    return ("@interface_rebind", field, str(when_true[1]), str(guard), str(expected),
            str(when_false[1]))


def valid_chain_record(record):
    next_field = unqualified(dict(record.fields).get("next"))
    return (isinstance(next_field, RecordLink) and next_field.identity == record.identity
            and all(name == "next" or not Compiler.contains_record_links(field) for name, field in record.fields))


def linked_owned_record(ty):
    return (isinstance(ty, HeapRecord) and valid_chain_record(ty.record)
            and not Compiler.contains_interface_storage(ty.record))


# Lengths before this sentinel are exact. The final alternative represents
# a concrete prefix followed by a nonempty, finite group of allocations.
OWNED_CHAIN_LONG = 5


def chain_api_signature(signature, result_lengths, parameters):
    # Link an ownership summary to the implementation that proved it.
    # In particular, an older callee may release only the head allocation.
    if result_lengths or parameters:
        return {"contract": signature, "owned_chain_lengths": result_lengths,
                "owned_chain_parameters": parameters}
    return signature


def mutable_chain_record(record):
    return all(isinstance(unqualified(field), (Scalar, RecordLink)) for _, field in record.fields)


def valid_chain_limit(limit):
    # A chain length is a property of the link meta-graph, so only the
    # "unbounded chain" marker remains in source.  Numeric bounds are gone:
    # the shape analysis derives the length and reports it in .table.
    return limit is True


def mutable_chain(ty):
    return isinstance(ty, (BorrowRecord, Slice)) and ty.chain_limit is not None and not ty.readonly


def mutable_graph(ty):
    return isinstance(ty, (BorrowRecord, Slice)) and ty.graph and not ty.readonly


def graph_or_chain_parameter(ty):
    return (isinstance(ty, (BorrowRecord, Slice)) and ty.chain_limit is not None
            or isinstance(ty, (BorrowRecord, Slice)) and ty.graph)


def chain_attachments(params):
    return [(index, ty.attach) for index, ty in enumerate(params)
            if isinstance(ty, (BorrowRecord, Slice)) and ty.attach is not None]


def graph_scalar_borrows(params):
    """Graph contracts for which a call can check extra scalar borrows."""
    graphs = [i for i, ty in enumerate(params) if isinstance(ty, BorrowRecord) and ty.graph]
    scalars = [i for i, ty in enumerate(params) if isinstance(ty, BorrowScalar)]
    if not graphs or not scalars:
        return (), ()
    if all(params[i].readonly for i in (*graphs, *scalars)):
        return (), ()
    if any(isinstance(ty, (BorrowRecord, BorrowArray, Slice, Interface))
           and i not in graphs for i, ty in enumerate(params)):
        return (), ()
    return graphs, scalars


def valid_chain_parameters(params):
    chains = [ty for ty in params if isinstance(ty, (BorrowRecord, Slice)) and ty.chain_limit is not None]
    graphs = [ty for ty in params if isinstance(ty, (BorrowRecord, Slice)) and ty.graph]
    if not chains and not graphs:
        return True
    borrows = [ty for ty in params if isinstance(ty, (BorrowRecord, BorrowScalar, BorrowArray, Slice, Interface))
               and not (isinstance(ty, Slice) and ty.copy)]
    if any(isinstance(ty, BorrowRecord) and ty.nullable for ty in (*chains, *graphs)):
        return False
    _, separate_scalars = graph_scalar_borrows(params)
    if separate_scalars:
        borrows = [ty for ty in borrows if not isinstance(ty, BorrowScalar)]
    attachments = chain_attachments(params)
    if attachments:
        index, source = attachments[0]
        if len(attachments) != 1 or len(borrows) != 2 or not 0 <= source < len(params) or source == index:
            return False
        head, donor = params[index], params[source]
        if head.graph:
            return isinstance(donor, BorrowRecord) and donor.graph and donor.readonly
        element = head.element if isinstance(head, Slice) else head.record
        # Attachment only needs an acyclic donor; the resulting chain length is
        # derived from the link meta-graph, so no numeric bound is compared.
        return (isinstance(donor, BorrowRecord) and donor.readonly and donor.chain_limit is not None
                and element == donor.record)
    if any(mutable_graph(ty) for ty in graphs):
        return len(borrows) == 1
    if graphs and any(not ty.readonly for ty in borrows):
        return False
    if any(mutable_chain(ty) for ty in chains):
        return len(borrows) == 1
    return all(ty.readonly for ty in borrows)


def combine_offset_polynomial(op, left, right):
    if op == "*":
        result = {}
        for x, a in left.items():
            for y, b in right.items():
                powers = dict(x)
                for origin, exponent in y:
                    powers[origin] = powers.get(origin, 0) + exponent
                if sum(powers.values()) > 8:
                    return None
                factors = frozenset(powers.items())
                result[factors] = result.get(factors, 0) + a * b
    else:
        result = left.copy()
        sign = 1 if op == "+" else -1
        for factors, coefficient in right.items():
            result[factors] = result.get(factors, 0) + sign * coefficient
    result = {factors: coefficient for factors, coefficient in result.items() if coefficient}
    if len(result) > 65 or any(abs(coefficient) > 1 << 127 for coefficient in result.values()):
        return None
    return result


def slice_offset_terms(ty):
    linear = ((ty.offset_param, 1),) if ty.offset_param is not None else ty.offset_terms
    return tuple(((index,), scale) for index, scale in linear) + ty.offset_products


OFFSET_BINARY_OPS = ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>")

def encode_offset_expression(expr):
    if expr[0] == "cast":
        return ["cast", expr[1], encode_interface_type(expr[2]), expr[3], expr[4],
                encode_offset_expression(expr[5])]
    if expr[0] == "binary":
        return ["binary", expr[1], encode_offset_expression(expr[2]), encode_offset_expression(expr[3])]
    return list(expr)

def integer_offset_cast_domain(domain, strategy, ty, width):
    if strategy in ("safe", "trunc"):
        if not domain.within(ty.domain):
            raise ValueError("cannot prove offset conversion fits its target type")
        return domain
    if strategy == "trap":
        result = domain & ty.domain
        if not result:
            raise ValueError("offset trap conversion cannot produce a target value")
        return result
    if strategy == "view":
        if width != ty.bits or not any(domain.within(container(width, signed)) for signed in (False, True)):
            raise ValueError("offset integer view requires the same fixed width")
    if strategy == "sat":
        if ty.domain.predicates or len(ty.domain.ranges) != 1:
            raise ValueError("offset saturation requires a contiguous target range")
        lo, hi = ty.domain.lo, ty.domain.hi
        return Domain.of(*((min(max(a, lo), hi), min(max(b, lo), hi)) for a, b in domain.ranges))
    full = container(ty.bits, ty.signed)
    lo, modulus = full.lo, 1 << ty.bits
    spans = []
    for a, b in domain.ranges:
        q0, r0 = divmod(a - lo, modulus)
        q1, r1 = divmod(b - lo, modulus)
        if q0 == q1:
            spans.append((lo + r0, lo + r1))
        elif q1 == q0 + 1:
            spans.extend(((lo + r0, full.hi), (lo, lo + r1)))
        else:
            spans.extend(full.ranges)
    result = domain if domain.within(full) else Domain.of(*spans)
    if not result.within(ty.domain):
        raise ValueError("offset conversion can produce values outside its target type")
    return result

def decode_offset_expression(raw, depth=0, budget=None):
    budget = [129] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 16 or not isinstance(raw, (list, tuple)) or not raw:
        raise InterfaceError("invalid or excessive slice offset expression")
    if raw[0] == "param" and len(raw) == 2 and type(raw[1]) is int and raw[1] >= 0:
        return tuple(raw)
    if (raw[0] == "len" and len(raw) == 3 and type(raw[1]) is int and raw[1] >= 0
            and type(raw[2]) is int and raw[2] in (0,32)):
        return tuple(raw)
    if raw[0] == "const" and len(raw) == 2 and type(raw[1]) is int and abs(raw[1]) <= 1 << 64:
        return tuple(raw)
    if len(raw) == 6 and raw[0] == "cast":
        if (raw[1] not in ("safe", "trunc", "wrap", "sat", "trap", "view")
                or type(raw[3]) is not int or raw[3] not in (0, *WIDTHS) or type(raw[4]) is not bool):
            raise InterfaceError("invalid slice offset conversion")
        target = raw[2]
        if isinstance(target, dict) and target.get("kind") == "scalar":
            target = decode_interface_type(target)
        if (not isinstance(target, Scalar) or target.bits not in WIDTHS
                or target.boolean or target.floating or target.enum or target.character):
            raise InterfaceError("slice offset conversion requires a fixed-width integer target")
        return ("cast", raw[1], target, raw[3], raw[4], decode_offset_expression(raw[5], depth + 1, budget))
    if len(raw) == 4 and raw[0] == "binary" and raw[1] in OFFSET_BINARY_OPS:
        return ("binary", raw[1], decode_offset_expression(raw[2], depth + 1, budget),
                decode_offset_expression(raw[3], depth + 1, budget))
    raise InterfaceError("invalid slice offset expression")

def offset_expression_parameters(expr):
    if expr[0] == "param":
        return {expr[1]}
    if expr[0] in ("const", "len"):
        return set()
    if expr[0] == "cast":
        return offset_expression_parameters(expr[5])
    return offset_expression_parameters(expr[2]) | offset_expression_parameters(expr[3])

def offset_expression_lengths(expr):
    if expr[0] == "len":
        return {expr[1]}
    if expr[0] in ("const", "param"):
        return set()
    if expr[0] == "cast":
        return offset_expression_lengths(expr[5])
    return offset_expression_lengths(expr[2]) | offset_expression_lengths(expr[3])

def offset_length_width(param):
    if isinstance(param, Slice):
        return 32
    if isinstance(param, CopyArray) or isinstance(param, BorrowArray) and not param.nullable:
        return 0
    raise ValueError("offset length requires a non-null array or slice parameter")

def offset_expression_width(expr, params):
    if expr[0] == "param":
        return unqualified(params[expr[1]]).bits
    if expr[0] == "const":
        return 0
    if expr[0] == "len":
        return offset_length_width(params[expr[1]])
    if expr[0] == "cast":
        return expr[2].bits
    return max(offset_expression_width(expr[2], params), offset_expression_width(expr[3], params))

def product_congruence(left, right):
    """The residue class every product of two operands shares.

    With ``a = ra + ma*k`` and ``b = rb + mb*j`` each product is ``ra*rb``
    plus a combination of ``ra*mb``, ``rb*ma`` and ``ma*mb``, so all of them
    sit in the class ``ra*rb`` modulo the gcd of those three steps.  A
    modulus of zero or one says nothing: a zero step is an exact operand and
    a one step is every integer.
    """
    ra, ma = left.congruence()
    rb, mb = right.congruence()
    if not ma and not mb:
        return None
    modulus = gcd(gcd(ra * mb, rb * ma), ma * mb)
    if modulus <= 1:
        return None
    return (ra * rb) % modulus, modulus


def square_domain(domain):
    return Domain.of(*((0 if lo <= 0 <= hi else min(lo * lo, hi * hi),
                        max(lo * lo, hi * hi)) for lo, hi in domain.ranges))

def offset_expression_domain(expr, params, *, refine=None):
    domain = _offset_expression_domain(expr, params, refine)
    return refine(expr, domain) if refine is not None else domain


def _offset_expression_domain(expr, params, refine):
    if expr[0] == "len":
        param = params[expr[1]]
        width = offset_length_width(param)
        if expr[2] != width:
            raise ValueError("offset length width does not match its parameter")
        return param.domain if width else Domain.point(param.array.length)
    if expr[0] == "param":
        domain = unqualified(params[expr[1]]).domain
        if type(domain.lo) is not int or type(domain.hi) is not int:
            raise ValueError("slice offset requires bounded integer parameters")
        return domain
    if expr[0] == "const":
        return Domain.point(expr[1])
    if expr[0] == "cast":
        if expr[3] != offset_expression_width(expr[5], params):
            raise ValueError("offset conversion source width does not match its expression")
        return integer_offset_cast_domain(offset_expression_domain(expr[5], params, refine=refine), expr[1], expr[2], expr[3])
    op = expr[1]
    a, b = (offset_expression_domain(part, params, refine=refine) for part in expr[2:])
    if expr[2] == expr[3]:
        if op in ("-", "^"):
            return ZERO
        if op in ("&", "|"):
            if not WIDE.lo <= a.lo <= a.hi <= WIDE.hi:
                raise ValueError("slice offset exceeds intermediate width")
            return a
        if op == "*":
            squared = square_domain(a)
            if squared.hi > WIDE.hi:
                raise ValueError("slice offset exceeds intermediate width")
            return squared
    if op in ("/", "%") and b & ZERO:
        raise ValueError("slice offset divisor may be zero")
    if op in ("/", "%") and expr[2] == expr[3]:
        if not WIDE.lo <= a.lo <= a.hi <= WIDE.hi:
            raise ValueError("slice offset exceeds intermediate width")
        return Domain.point(int(op == "/"))
    if op == "%" and a & Domain.point(WIDE.lo) and b & Domain.point(-1):
        raise ValueError("slice offset remainder would overflow intermediate division")
    if op in ("&", "|", "^"):
        domain, _ = bitwise_domain(op, a, b)
        if not WIDE.lo <= domain.lo <= domain.hi <= WIDE.hi:
            raise ValueError("slice offset exceeds intermediate width")
        return domain
    if op in ("<<", ">>") and not 0 <= b.lo <= b.hi < 128:
        raise ValueError("slice offset shift count must be in [0, 128)")
    if op == ">>" and expr[2][:2] == ("binary", "<<") and expr[2][3] == expr[3]:
        return offset_expression_domain(expr[2][2], params, refine=refine)
    if (op == "<<" and a.hi << b.hi > WIDE.hi
            and expr[2][:2] == ("binary", ">>") and expr[2][3] == expr[3]):
        original = offset_expression_domain(expr[2][2], params, refine=refine)
        if original.lo >= 0 and original.hi <= WIDE.hi:
            return Domain.of((0, original.hi)).with_congruence(0, 1 << b.lo)
    divisor_ranges = b.ranges
    if op in ("/", "%"):
        divisor_ranges = tuple((start, stop) for lo, hi in b.ranges
                               for start, stop in ((lo, min(hi, -1)), (max(lo, 1), hi))
                               if start <= stop)
    spans = []
    for lo, hi in a.ranges:
        for low, high in divisor_ranges:
            if op == "+":
                span = lo + low, hi + high
            elif op == "-":
                span = lo - high, hi - low
            elif op in ("*", "/"):
                fn = (lambda x, y: x * y) if op == "*" else trunc_div
                values = [fn(x, y) for x in (lo, hi) for y in (low, high)]
                span = min(values), max(values)
            elif op in ("<<", ">>"):
                values = [x << y if op == "<<" else x >> y for x in (lo, hi) for y in (low, high)]
                span = min(values), max(values)
            elif lo == hi and low == high:
                value = lo - trunc_div(lo, low) * low
                span = value, value
            else:
                limit = max(abs(low), abs(high)) - 1
                span = max(lo, -limit) if lo < 0 else 0, min(hi, limit) if hi > 0 else 0
            if not WIDE.lo <= span[0] <= span[1] <= WIDE.hi:
                raise ValueError("slice offset exceeds intermediate width")
            spans.append(span)
    domain = Domain.of(*spans)
    subtrahend = expr[3]
    if op == "-" and a.lo >= 0 and subtrahend[0] == "binary":
        operator, left, right = subtrahend[1:]
        if (operator == "&" and expr[2] in (left, right)
                or operator in ("/", "%", ">>") and expr[2] == left
                and offset_expression_domain(right, params, refine=refine).lo >= (0 if operator == ">>" else 1)):
            domain &= Domain.of((0, a.hi))
    return domain

def valid_slice_provenance(params, result):
    if any(isinstance(param, Slice) and (param.source_param is not None or slice_offset_terms(param) or param.offset_expression is not None) for param in params):
        return False
    if not isinstance(result, Slice):
        return True
    if result.copy or result.chain_limit is not None:
        return False
    if result.source_param is None:
        return not result.source_field and not slice_offset_terms(result) and result.offset_expression is None
    if not 0 <= result.source_param < len(params):
        return False
    source = params[result.source_param]
    if result.text == "c" and result.source_offset is None:
        return False
    if result.source_field:
        if not isinstance(source, BorrowRecord) or source.nullable:
            return False
        field, readonly = source.record, source.readonly
        for name in result.source_field:
            if not isinstance(field, Record) or (child := dict(field.fields).get(name)) is None:
                return False
            readonly |= isinstance(child, ReadOnly)
            field = unqualified(child)
        if not isinstance(field, Array):
            return False
        source = BorrowArray(field, readonly=readonly)
    if isinstance(source, Slice) and not source.copy:
        element = source.element
    elif isinstance(source, BorrowArray) and not source.nullable and not source.array.packed_bits:
        element = source.array.element
    else:
        return False
    source_max = source.domain.hi if isinstance(source, Slice) else source.array.length
    if result.offset_expression is not None:
        indices = offset_expression_parameters(result.offset_expression)
        if any(not 0 <= index < len(params) or not plain_integer_parameter(params[index]) for index in indices):
            return False
        lengths = offset_expression_lengths(result.offset_expression)
        if any(not 0 <= index < len(params) or not isinstance(params[index], (Slice, BorrowArray, CopyArray))
               for index in lengths):
            return False
        try:
            domain = offset_expression_domain(result.offset_expression, params)
        except (ValueError, DomainComplexity, ZeroDivisionError):
            return False
        if not domain or not 0 <= domain.lo <= domain.hi <= source_max:
            return False
    if terms := slice_offset_terms(result):
        low = high = result.offset_bias
        for factors, coefficient in terms:
            lower = upper = coefficient
            for index in set(factors):
                if not 0 <= index < len(params) or not plain_integer_parameter(params[index]):
                    return False
                domain = unqualified(params[index]).domain
                exponent = factors.count(index)
                endpoints = [domain.lo ** exponent, domain.hi ** exponent]
                if exponent % 2 == 0 and domain.lo <= 0 <= domain.hi:
                    endpoints.append(0)
                products = [a * b for a in (lower, upper) for b in (min(endpoints), max(endpoints))]
                lower, upper = min(products), max(products)
            low += lower
            high += upper
        if not 0 <= low <= high <= source_max:
            return False
    if result.source_offset is not None and result.source_offset + result.domain.hi > source_max:
        return False
    return (element == result.element and (result.readonly or not source.readonly)
            and (result.text is None or isinstance(source, Slice) and source.text == result.text))


def decode_interface_type(raw):
    if not isinstance(raw, dict):
        raise InterfaceError("invalid interface type")
    kind = raw.get("kind")
    if kind == "callable":
        raise InterfaceError("callable values are local to their Lan module")
    if kind == "extern_callable":
        keys = ("kind", "contract") + tuple(k for k in ("writes", "memory", "external", "nullable") if k in raw)
        interface_fields(raw, keys, "extern callable")
        contract = decode_interface_type(raw["contract"])
        if not isinstance(contract, Contract):
            raise InterfaceError("extern callable requires a contract")
        writes = []
        for entry in raw.get("writes", ()):
            if (not isinstance(entry, list) or len(entry) != 2 or type(entry[0]) is not int
                    or not isinstance(entry[1], list) or not all(isinstance(step, str) for step in entry[1])):
                raise InterfaceError("invalid extern callable write entry")
            writes.append((entry[0], tuple(entry[1]), None))
        return ExternCallable(contract, mutations=tuple(writes),
                              memory=bool(raw.get("memory", False)),
                              external_effect=bool(raw.get("external", False)),
                              nullable=bool(raw.get("nullable", False)))
    if kind == "type_object":
        interface_fields(raw, ("kind", "properties"), "type object")
        properties = raw["properties"]
        if not isinstance(properties, list) or len(properties) > 4096:
            raise InterfaceError("invalid type object properties")
        entries = {}
        for entry in properties:
            if not isinstance(entry, list) or len(entry) != 2:
                raise InterfaceError("invalid type object property")
            name, value = entry
            if (not isinstance(name, str) or not NAME.fullmatch(name) or name in entries
                    or not isinstance(value, dict) or len(value) != 1):
                raise InterfaceError("invalid type object property")
            key = Token(name, 0, 0)
            tag, payload = next(iter(value.items()))
            if tag == "value" and (type(payload) is bool or type(payload) is int and payload.bit_length() <= 4096):
                constant = Node(key, "boolean" if type(payload) is bool else "number", (payload,))
                property_node = Node(key, "object_value", (constant,))
            elif tag == "word" and payload in TYPE_OBJECT_WORDS.get(name, ()):
                property_node = Node(key, "object_word", (payload,))
            elif tag == "path" and name == "field" and isinstance(payload, list) and 1 <= len(payload) <= 128 and all(
                    isinstance(step, str) and NAME.fullmatch(step) or type(step) is int and 0 <= step < 1 << 31
                    for step in payload):
                property_node = Node(key, "object_path", (tuple(payload),))
            elif tag == "type":
                ty = decode_interface_type(payload)
                if ty is None:
                    raise InterfaceError("invalid void property in a type object")
                property_node = Node(key, "resolved_type", (ty,))
            else:
                raise InterfaceError("invalid type object property value")
            entries[name] = key, property_node
        return TypeObject(tuple(entries.values()))
    if kind == "void":
        interface_fields(raw, ("kind",), "void type")
        return None
    if kind == "float":
        keys = ("kind", "bits", "domain") + (("literals",) if "literals" in raw else ())
        interface_fields(raw, keys, "floating type")
        try:
            domain = FloatDomain.from_dict(raw["domain"])
        except FloatDomainError as exc:
            raise InterfaceError(str(exc)) from exc
        if raw["bits"] not in (32, 64) or domain.bits != raw["bits"] or not domain:
            raise InterfaceError("invalid floating domain")
        encoded = raw.get("literals")
        literals = []
        if "literals" in raw:
            if raw["bits"] != 64 or not isinstance(encoded, list) or not 1 <= len(encoded) <= 4096:
                raise InterfaceError("invalid floating literal set")
            for value in encoded:
                if isinstance(value, str) and value in ("inf", "-inf", "nan"):
                    literals.append(value)
                elif (isinstance(value, list) and len(value) == 2
                      and all(type(part) is int and part.bit_length() <= 4096 for part in value)
                      and value[1] > 0):
                    literals.append(Fraction(*value))
                else:
                    raise InterfaceError("invalid floating literal set")
            if float_literal_domain(literals, 64) != domain:
                raise InterfaceError("floating literals do not match their declared domain")
        return Scalar(domain, raw["bits"], floating=True, float_literals=tuple(literals))
    if kind == "character":
        interface_fields(raw, ("kind", "encoding", "domain"), "character type")
        encoding = raw["encoding"]
        if not isinstance(encoding, str) or encoding not in CHARACTERS:
            raise InterfaceError("invalid character encoding")
        bits, allowed = CHARACTERS[encoding]
        domain = decode_domain(raw["domain"])
        if not domain or not domain.within(allowed):
            raise InterfaceError("invalid character domain")
        return Scalar(domain, bits, character=encoding)
    if kind == "scalar":
        interface_fields(raw, ("kind", "bits", "signed", "boolean", "computed", "domain"), "scalar type")
        bits, signed, boolean, computed = (raw[key] for key in ("bits", "signed", "boolean", "computed"))
        if (type(bits) is not int or bits not in (0, *WIDTHS)
                or any(type(flag) is not bool for flag in (signed, boolean, computed))
                or (computed and bits) or (boolean and (bits != 8 or signed))):
            raise InterfaceError("invalid scalar representation in interface")
        domain = decode_domain(raw["domain"])
        if boolean and (not domain or not domain.within(BOOL)):
            raise InterfaceError("invalid boolean domain in interface")
        if bits and not domain.within(container(bits, signed)):
            raise InterfaceError("interface scalar domain exceeds its physical container")
        return Scalar(domain, bits, signed, boolean, computed)
    if kind == "tagged_result":
        interface_fields(raw, ("kind", "success", "error"), "tagged result")
        success, error = decode_interface_type(raw["success"]), decode_interface_type(raw["error"])
        if (not isinstance(success, (Record, CopyArray, Slice, HeapRecord, HeapObject, HeapArray, AliasResult))
                or not tagged_error_type(error) or Compiler.contains_interface_storage(success)):
            raise InterfaceError("invalid tagged result type")
        return TaggedResult(success, error)
    if kind == "result_union":
        optional = ("error_enums",) if "error_enums" in raw else ()
        interface_fields(raw, ("kind", "base", "success", "error", *optional), "result union type")
        base = decode_interface_type(raw["base"])
        if (not isinstance(base, Scalar) or base.enum or base.character or base.boolean
                or base.result_arms is not None):
            raise InterfaceError("invalid result union in interface")
        decode = FloatDomain.from_dict if base.floating else decode_domain
        try:
            good, error = decode(raw["success"]), decode(raw["error"])
            if not (good or error) or good & error or good | error != base.domain:
                raise InterfaceError("invalid result union in interface")
        except FloatDomainError as exc:
            raise InterfaceError(str(exc)) from exc
        enums = raw.get("error_enums", [])
        if (not isinstance(enums, list) or len(enums) > 4096 or (base.floating or not error) and enums
                or any(not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity) for identity in enums)
                or len(set(enums)) != len(enums)):
            raise InterfaceError("invalid error enum identities in result union")
        return replace(base, result_arms=(good, error), result_error_enums=tuple(sorted(enums)))
    if kind == "enum":
        interface_fields(raw, ("kind", "identity", "name", "members", "base"), "enum type")
        identity, name, members = raw["identity"], raw["name"], raw["members"]
        base = decode_interface_type(raw["base"])
        if (not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity)
                or not isinstance(name, str) or not (
                    (NAME.fullmatch(name) and name not in RESERVED)
                    or (name.endswith(".key") and NAME.fullmatch(name[:-4]) and name[:-4] not in RESERVED))
                or not isinstance(members, list) or not 0 < len(members) <= 4096
                or not isinstance(base, Scalar) or not base.bits or base.boolean or base.enum or base.character or not base.domain):
            raise InterfaceError("invalid enum representation in interface")
        names, values, decoded = set(), set(), []
        full = container(base.bits, base.signed)
        for item in members:
            if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                    or not NAME.fullmatch(item[0]) or item[0] in names or item[0] in RESERVED
                    or type(item[1]) is not int or item[1] in values or not full.lo <= item[1] <= full.hi):
                raise InterfaceError("invalid enum member in interface")
            names.add(item[0])
            values.add(item[1])
            decoded.append(tuple(item))
        if not base.domain.within(Domain.of(*((n, n) for n in values))):
            raise InterfaceError("enum domain contains undeclared values")
        return replace(base, enum=EnumInfo(identity, name, base.signed, tuple(decoded)))
    if kind == "encoding_tag":
        interface_fields(raw, ("kind", "encoding"), "encoding tag type")
        return EncodingTag(decode_text_encoding(raw["encoding"]))
    if kind == "array":
        keys = ("kind", "element", "length", "packed")
        if "encoding" in raw:
            keys += ("encoding",)
        interface_fields(raw, keys, "array type")
        element = decode_interface_type(raw["element"])
        length, packed = raw["length"], raw["packed"]
        encoding = decode_text_encoding(raw["encoding"]) if "encoding" in raw else None
        if (not isinstance(element, (Scalar, Array, Record))
                or isinstance(element, Scalar) and (not element.bits or not element.domain)
                or type(length) is not int or length <= 0 or type(packed) is not bool):
            raise InterfaceError("invalid array representation in interface")
        if packed and (element != PACKED_BIT or length % 8):
            raise InterfaceError("invalid packed bit array in interface")
        ty = Array(element, length, packed, encoding=encoding)
        if storage_size(ty) >= 1 << 31:
            raise InterfaceError("interface array exceeds the supported storage size")
        return ty
    if kind == "slice":
        keys = ("kind", "element", "len", "readonly")
        optional = ("text", "elem", "from", "field", "offset", "offset_param", "offset_terms", "offset_products", "offset_bias", "offset_expression", "copy", "chain", "prune", "attach", "graph")
        interface_fields(raw, keys + tuple(key for key in optional if key in raw), "slice type")
        element = decode_interface_type(raw["element"])
        if (not isinstance(element, (Scalar, Array, Record))
                or isinstance(element, Scalar) and (not element.bits or not element.domain)
                or type(raw["readonly"]) is not bool):
            raise InterfaceError("invalid slice representation in interface")
        text = raw.get("text")
        if "text" in raw:
            if isinstance(text, dict):
                text = decode_text_encoding(text)
                if (not raw["readonly"] or not isinstance(element, Scalar)
                        or element.bits != 8 or element.signed or element.boolean or element.enum is not None):
                    raise InterfaceError("invalid custom string view representation")
            elif (text not in ("ascii", "utf8", "c") or not raw["readonly"]
                  or not isinstance(element, Scalar) or element.character != ("ascii" if text == "ascii" else "utf8")):
                raise InterfaceError("invalid string view representation")
        if "copy" in raw and raw["copy"] is not True:
            raise InterfaceError("invalid slice copy qualifier")
        chain = raw.get("chain")
        if "chain" in raw and (not valid_chain_limit(chain) or not isinstance(element, Record)
                               or not valid_chain_record(element) or not raw["readonly"] and not mutable_chain_record(element)
                               or any(key in raw for key in ("text", "elem", "copy", "from"))):
            raise InterfaceError("invalid chain slice parameter")
        prune = raw.get("prune", False)
        if "prune" in raw and (prune is not True or chain is None and raw.get("graph") is not True or raw["readonly"]):
            raise InterfaceError("chain pruning requires a writable chain slice")
        attach = raw.get("attach")
        if "attach" in raw and (type(attach) is not int or attach < 0 or chain is None and raw.get("graph") is not True or prune or raw["readonly"]):
            raise InterfaceError("invalid chain slice attachment parameter")
        graph = raw.get("graph", False)
        if "graph" in raw and (graph is not True or not isinstance(element, Record)
                               or not raw["readonly"] and not mutable_chain_record(element)
                               or not Compiler.contains_record_links(element)
                               or any(key in raw for key in ("text", "elem", "copy", "from", "chain"))):
            raise InterfaceError("invalid recursive graph slice parameter")
        ty = Slice(element, length_bound=decode_domain(raw["len"]), readonly=raw["readonly"], text=text,
                   copy=raw.get("copy", False), chain_limit=chain, prune=prune, attach=attach, graph=graph)
        if 'elem' in raw:
            content = decode_interface_type(raw['elem'])
            if (not isinstance(element, Scalar) or not isinstance(content, Scalar)
                    or replace(content, domain=element.domain) != element
                    or not content.domain or not content.domain.within(element.domain)):
                raise InterfaceError('invalid slice element predicate')
            ty = replace(ty, content_bound=content.domain)
        if "from" in raw:
            source = raw["from"]
            if type(source) is not int or source < 0:
                raise InterfaceError("invalid slice return provenance")
            ty = replace(ty, source_param=source)
        if "field" in raw:
            names = raw["field"]
            if (ty.source_param is None or not isinstance(names, list) or not names
                    or len(names) > 4096
                    or not all(isinstance(name, str) and NAME.fullmatch(name) for name in names)):
                raise InterfaceError("invalid slice source field")
            ty = replace(ty, source_field=tuple(names))
        if "offset" in raw:
            offset = raw["offset"]
            if ty.source_param is None or type(offset) is not int or not 0 <= offset <= ty.capacity.hi:
                raise InterfaceError("invalid slice return offset")
            ty = replace(ty, source_offset=offset)
        if "offset_param" in raw:
            index = raw["offset_param"]
            if ty.source_param is None or ty.source_offset is not None or type(index) is not int or index < 0:
                raise InterfaceError("invalid symbolic slice return offset")
            ty = replace(ty, offset_param=index)
        if "offset_terms" in raw:
            terms = raw["offset_terms"]
            if (ty.source_param is None or ty.source_offset is not None or ty.offset_param is not None
                    or not isinstance(terms, list) or not 0 < len(terms) <= 64):
                raise InterfaceError("invalid symbolic slice offset terms")
            previous = -1
            for term in terms:
                if (not isinstance(term, list) or len(term) != 2 or type(term[0]) is not int
                        or type(term[1]) is not int or term[0] <= previous or not 0 < abs(term[1]) <= 64):
                    raise InterfaceError("invalid symbolic slice offset term")
                previous = term[0]
            ty = replace(ty, offset_terms=tuple(map(tuple, terms)))
        if "offset_products" in raw:
            products = raw["offset_products"]
            if (ty.source_param is None or ty.source_offset is not None or not isinstance(products, list)
                    or not 0 < len(products) <= 64 - len(slice_offset_terms(ty))):
                raise InterfaceError("invalid polynomial slice offset")
            decoded, previous = [], ()
            for item in products:
                if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], list)
                        or not 2 <= len(item[0]) <= 8 or any(type(index) is not int or index < 0 for index in item[0])
                        or item[0] != sorted(item[0]) or type(item[1]) is not int or not 0 < abs(item[1]) <= 64):
                    raise InterfaceError("invalid polynomial slice offset term")
                factors = tuple(item[0])
                if factors <= previous:
                    raise InterfaceError("duplicate or unordered polynomial slice offset terms")
                previous = factors
                decoded.append((factors, item[1]))
            ty = replace(ty, offset_products=tuple(decoded))
        if "offset_bias" in raw:
            bias = raw["offset_bias"]
            if not slice_offset_terms(ty) or type(bias) is not int or not -(1 << 64) <= bias <= 1 << 64:
                raise InterfaceError("invalid symbolic slice offset bias")
            ty = replace(ty, offset_bias=bias)
        if "offset_expression" in raw:
            if ty.source_param is None or ty.source_offset is not None or slice_offset_terms(ty) or ty.offset_bias:
                raise InterfaceError("conflicting slice offset expression")
            ty = replace(ty, offset_expression=decode_offset_expression(raw["offset_expression"]))
        if not ty.domain or not ty.domain.within(ty.capacity):
            raise InterfaceError("invalid slice length contract in interface")
        return ty
    if kind in ("array_borrow", "array_copy"):
        nullable = "nullable" in raw
        keys = ("kind", "array", "readonly") + (("nullable",) if nullable else ())
        interface_fields(raw, keys, "array parameter")
        if nullable and (kind != "array_borrow" or raw["nullable"] is not True):
            raise InterfaceError("invalid nullable array alias")
        array = decode_interface_type(raw["array"])
        if not isinstance(array, Array) or type(raw["readonly"]) is not bool:
            raise InterfaceError("invalid array parameter")
        return BorrowArray(array, raw["readonly"], nullable) if kind == "array_borrow" else CopyArray(array, raw["readonly"])
    if kind == "immutable":
        interface_fields(raw, ("kind", "base"), "immutable type")
        base = decode_interface_type(raw["base"])
        if not (isinstance(base, (Scalar, Array, Record)) or isinstance(base, Slice) and record_text_type(base)):
            raise InterfaceError("invalid immutable storage type")
        return ReadOnly(base)
    if kind == "alias_result":
        keys = ("kind", "alias", "from") + tuple(key for key in ("field", "alternatives", "graph_records") if key in raw)
        interface_fields(raw, keys, "borrowed alias result")
        alias = decode_interface_type(raw["alias"])

        def decode_origin(source, fields):
            if type(source) is not int or source < 0:
                raise InterfaceError("invalid borrowed alias result")
            if (not isinstance(fields, list) or len(fields) > ALIAS_PATH_LIMIT
                    or any(not (isinstance(name, str) and (NAME.fullmatch(name) or name == CHAIN_REACHABLE)
                                or type(name) is int and name >= 0 or isinstance(name, list)
                                or isinstance(name, dict) and set(name) == {"end"}
                                and type(name["end"]) is int and name["end"] > 0
                                or isinstance(name, dict) and set(name) == {"param", "offset"}
                                and all(type(value) is int for value in name.values())
                                or isinstance(name, dict) and set(name) == {"expr"}) for name in fields)):
                raise InterfaceError("invalid alias result field path")
            return source, tuple(decode_domain(step) if isinstance(step, list) else
                                 SliceEndIndex(step["end"]) if isinstance(step, dict) and "end" in step else
                                 AliasIndexExpression(decode_offset_expression(step["expr"])) if isinstance(step, dict) and "expr" in step else
                                 AliasParamIndex(step["param"], step["offset"]) if isinstance(step, dict) else step
                                 for step in fields)

        if not isinstance(alias, (BorrowRecord, BorrowScalar, BorrowArray)):
            raise InterfaceError("invalid borrowed alias result")
        source, path = decode_origin(raw["from"], raw.get("field", []))
        alternatives = raw.get("alternatives", [])
        if (not isinstance(alternatives, list) or len(alternatives) > 4096
                or any(not isinstance(item, list) or len(item) != 2 for item in alternatives)):
            raise InterfaceError("invalid alias result alternatives")
        encoded_records = raw.get("graph_records", [])
        if not isinstance(encoded_records, list) or len(encoded_records) > 4096:
            raise InterfaceError("invalid alias graph records")
        records = tuple(decode_interface_type(record) for record in encoded_records)
        if (any(not isinstance(record, Record) for record in records)
                or len({record.identity for record in records}) != len(records)):
            raise InterfaceError("invalid alias graph records")
        return AliasResult(alias, source, path, tuple(decode_origin(*item) for item in alternatives), records)
    if kind == "array_owner":
        interface_fields(raw, ("kind", "element", "len"), "owned array")
        element = decode_interface_type(raw["element"])
        length = decode_domain(raw["len"])
        if not isinstance(element, Scalar) or not element.bits or not element.domain:
            raise InterfaceError("invalid owned array element")
        result = HeapArray(element, length_bound=length)
        if not length or not length.within(result.capacity):
            raise InterfaceError("invalid owned array length")
        return result
    if kind == "object_owner":
        interface_fields(raw, ("kind", "base"), "owned object")
        base = decode_interface_type(raw["base"])
        if not (isinstance(base, Scalar) and base.bits and base.domain or
                isinstance(base, Array) and base.packed_bits):
            raise InterfaceError("invalid owned object representation")
        return HeapObject(base)
    if kind == "record_owner":
        interface_fields(raw, ("kind", "record"), "owned record")
        record = decode_interface_type(raw["record"])
        if not isinstance(record, Record):
            raise InterfaceError("invalid owned record representation")
        return HeapRecord(record)
    if kind == "null":
        interface_fields(raw, ("kind",), "null type")
        return NullType()
    if kind == "scalar_borrow":
        nullable = "nullable" in raw
        keys = ("kind", "scalar", "readonly") + (("nullable",) if nullable else ())
        interface_fields(raw, keys, "scalar parameter")
        if nullable and raw["nullable"] is not True:
            raise InterfaceError("invalid nullable scalar alias")
        scalar = decode_interface_type(raw["scalar"])
        if not isinstance(scalar, Scalar) or type(raw["readonly"]) is not bool or not scalar.bits:
            raise InterfaceError("invalid scalar borrow parameter")
        return BorrowScalar(scalar, raw["readonly"], nullable)
    if kind in ("record_copy", "record_borrow"):
        keys = ("kind", "record") + (("readonly",) if "readonly" in raw else ()) + (("nullable",) if "nullable" in raw else ())
        interface_fields(raw, keys + tuple(key for key in ("chain", "prune", "attach", "graph") if key in raw), "record parameter")
        if "nullable" in raw and (kind != "record_borrow" or raw["nullable"] is not True):
            raise InterfaceError("invalid nullable record alias")
        if "readonly" in raw and type(raw["readonly"]) is not bool:
            raise InterfaceError("invalid record parameter qualification")
        record = decode_interface_type(raw["record"])
        if not isinstance(record, Record):
            raise InterfaceError("copy parameter requires a record type")
        chain = raw.get("chain")
        if "chain" in raw and (kind != "record_borrow" or chain is not True
                               or raw.get("nullable") or not valid_chain_record(record)
                               or not raw.get("readonly") and not mutable_chain_record(record)):
            raise InterfaceError("invalid chain parameter")
        prune = raw.get("prune", False)
        if "prune" in raw and (prune is not True or chain is None and raw.get("graph") is not True or raw.get("readonly") is not False):
            raise InterfaceError("chain pruning requires a writable record chain")
        attach = raw.get("attach")
        if "attach" in raw and (type(attach) is not int or attach < 0
                                or chain is None and raw.get("graph") is not True
                                or prune and raw.get("graph") is not True or raw.get("readonly") is not False):
            raise InterfaceError("invalid chain attachment parameter")
        graph = raw.get("graph", False)
        if "graph" in raw and (graph is not True or kind != "record_borrow" or "readonly" not in raw
                               or not raw["readonly"] and not mutable_chain_record(record)
                               or chain is not None or raw.get("nullable") or not Compiler.contains_record_links(record)):
            raise InterfaceError("invalid recursive graph parameter")
        return (BorrowRecord(record, raw.get("readonly", True), raw.get("nullable", False), chain, prune, attach, graph) if kind == "record_borrow"
                else CopyRecord(record, raw.get("readonly", False)))
    if kind == "record":
        keys = ("kind", "identity", "repr_c", "fields")
        interface_fields(raw, (*keys, "key_type") if "key_type" in raw else keys, "record type")
        identity, repr_c, fields = raw["identity"], raw["repr_c"], raw["fields"]
        if (not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity)
                or type(repr_c) is not bool or not isinstance(fields, list) or not 0 < len(fields) <= 4096):
            raise InterfaceError("invalid record representation in interface")
        names, decoded = set(), []
        for field in fields:
            if (not isinstance(field, list) or len(field) != 2 or not isinstance(field[0], str)
                    or not NAME.fullmatch(field[0]) or field[0] in names or field[0] in RESERVED):
                raise InterfaceError("invalid record field in interface")
            name, encoded = field
            ty = decode_interface_type(encoded)
            storage = unqualified(ty)
            if (not isinstance(storage, (Scalar, Array, Record, Slice, RecordLink, Interface))
                    or isinstance(storage, Slice) and (not isinstance(ty, ReadOnly) or not record_text_type(storage))
                    or isinstance(storage, Scalar) and (not storage.bits or not storage.domain)):
                raise InterfaceError("invalid record field type in interface")
            decoded.append((name, ty))
            names.add(name)
        key_type = decode_interface_type(raw["key_type"]) if "key_type" in raw else None
        if "key_type" in raw and (not Record.valid_keys(key_type, decoded) or repr_c
                                   or any(not isinstance(unqualified(field), Scalar) for _, field in decoded)):
            raise InterfaceError("invalid keyed record layout in interface")
        ty = Record(identity, tuple(decoded), repr_c, key_type)
        if ty.leaves > 4096 or ty.max_bytes >= 1 << 31:
            raise InterfaceError("record exceeds interface layout limits")
        return ty
    if kind == "record_link":
        interface_fields(raw, ("kind", "identity", "readonly"), "record link")
        identity, readonly = raw["identity"], raw["readonly"]
        if not isinstance(identity, str) or not re.fullmatch("[0-9a-f]{64}", identity) or type(readonly) is not bool:
            raise InterfaceError("invalid record link")
        return RecordLink(identity, readonly)
    if kind == "duck_result":
        interface_fields(raw, ("kind", "view", "from"), "structural interface result")
        view, source = decode_interface_type(raw["view"]), raw["from"]
        if not isinstance(view, Interface) or type(source) is not int or source < 0:
            raise InterfaceError("invalid structural interface result")
        return InterfaceResult(view, source)
    if kind == "duck_interface":
        interface_fields(raw, ("kind", "schema", "readonly"), "structural interface")
        schema = decode_interface_type(raw["schema"])
        if (not isinstance(schema, Record) or schema.repr_c or type(raw["readonly"]) is not bool
                or Compiler.contains_record_links(schema)):
            raise InterfaceError("invalid structural interface representation")
        return Interface(schema, raw["readonly"])
    if kind == "contract":
        optional = ("bounds", "result_expressions", "parameter_expressions", "identity_property", "idempotent_property",
                    "commutative_property", "diagonal_property")
        keys = ("kind", "params", "result") + tuple(key for key in optional if key in raw)
        interface_fields(raw, keys, "function contract")
        if not isinstance(raw["params"], list) or len(raw["params"]) > 4096:
            raise InterfaceError("invalid interface parameters")
        params = tuple(decode_interface_type(item) for item in raw["params"])
        result = decode_interface_type(raw["result"])
        payload = success_type(result)
        if (any(not isinstance(ty, (Slice, CopyRecord, BorrowRecord, BorrowScalar, BorrowArray, CopyArray, Interface,
                                    HeapRecord, HeapObject, HeapArray, ExternCallable))
                and (not isinstance(unqualified(ty), Scalar) or not unqualified(ty).bits or not unqualified(ty).domain)
                for ty in params) or result is not None
                and not isinstance(payload, (Record, CopyArray, HeapRecord, HeapObject, HeapArray, AliasResult, InterfaceResult, ExternCallable))
                and not (isinstance(payload, Slice) and (payload.readonly or payload.source_param is not None))
                and (not isinstance(result, Scalar) or not result.bits or not result.domain)):
            raise InterfaceError("interface function contracts contain an unsupported parameter or result type")
        if (not valid_chain_parameters(params)
                or Compiler.contains_record_links(payload) and not linked_owned_record(payload)
                or any(Compiler.contains_record_links(ty)
                       and not (graph_or_chain_parameter(ty) or linked_owned_record(ty))
                       for ty in params)):
            raise InterfaceError("record links across function contracts require a lifetime contract")
        if (isinstance(payload, AliasResult) and payload.graph_records
                and set(payload.graph_records) != set(alias_graph_intermediates(params, payload, payload.graph_records))):
            raise InterfaceError("unused or redundant alias graph records")
        if not valid_slice_provenance(params, payload) or not valid_interface_provenance(params, payload):
            raise InterfaceError("invalid slice return provenance")
        bounds = raw.get("bounds", [])
        if not isinstance(bounds, list) or len(bounds) > 8192:
            raise InterfaceError("invalid dependent parameter bounds")
        checked = []
        for bound in bounds:
            if not isinstance(bound, list) or len(bound) != 5:
                raise InterfaceError("invalid dependent parameter bound")
            target, source, length, op, delta = bound
            if (type(target) is not int or type(source) is not int or not 0 <= source < target <= len(params)
                    or type(length) is not bool or op not in ("<=", ">=")
                    or type(delta) is not int or not -(1 << 64) <= delta <= 1 << 64
                    or not plain_integer_parameter(result if target == len(params) else params[target])
                    or dependent_source_domain(params[source], length) is None):
                raise InterfaceError("invalid dependent parameter bound")
            checked.append(tuple(bound))
        expressions = raw.get("result_expressions", [])
        if not isinstance(expressions, list) or len(expressions) > 64:
            raise InterfaceError("invalid dependent result expressions")
        decoded = []
        for item in expressions:
            if (not plain_integer_parameter(result) or not isinstance(item, list)
                    or len(item) != 2 or item[0] not in ("<=", ">=")):
                raise InterfaceError("invalid dependent result expression")
            op, expression = item
            expression = decode_offset_expression(expression)
            try:
                dependent_expression_domain(expression, params)
            except (ValueError, DomainComplexity, IndexError) as exc:
                raise InterfaceError("invalid dependent result expression") from exc
            decoded.append((op, expression))
        parameter_expressions = raw.get("parameter_expressions", [])
        if not isinstance(parameter_expressions, list) or len(parameter_expressions) > 64:
            raise InterfaceError("invalid dependent parameter expressions")
        decoded_parameters = []
        for item in parameter_expressions:
            if (not isinstance(item, list) or len(item) != 3 or type(item[0]) is not int
                    or not 0 <= item[0] < len(params) or item[1] not in ("<=", ">=")
                    or not plain_integer_parameter(params[item[0]])):
                raise InterfaceError("invalid dependent parameter expression")
            target, op, expression = item
            expression = decode_offset_expression(expression)
            try:
                dependent_expression_domain(expression, params[:target])
            except (ValueError, DomainComplexity, IndexError) as exc:
                raise InterfaceError("invalid dependent parameter expression") from exc
            decoded_parameters.append((target, op, expression))
        if not valid_alias_provenance(params, payload, checked, decoded_parameters):
            raise InterfaceError("invalid slice return provenance")
        identity = raw.get("identity_property", False)
        if type(identity) is not bool or identity and (
                len(params) != 1 or not plain_integer_parameter(params[0]) or not plain_integer_parameter(result)
                or (1, 0, False, ">=", 0) not in checked or (1, 0, False, "<=", 0) not in checked):
            raise InterfaceError("invalid identity property in interface")
        idempotent = raw.get("idempotent_property", False)
        if type(idempotent) is not bool or idempotent and (identity or len(params) != 1
                or not closed_integer_property(params[0], result)):
            raise InterfaceError("invalid idempotent property in interface")
        commutative = raw.get("commutative_property", False)
        if type(commutative) is not bool or commutative and (identity or idempotent or checked or decoded
                or decoded_parameters or not commutative_integer_signature(params, result)):
            raise InterfaceError("invalid commutative property in interface")
        diagonal = raw.get("diagonal_property", False)
        if type(diagonal) is not bool or diagonal and (identity or idempotent or commutative or checked or decoded
                or decoded_parameters or not diagonal_integer_signature(params, result)):
            raise InterfaceError("invalid diagonal property in interface")
        return Contract(params, result, tuple(checked), tuple(decoded), tuple(decoded_parameters),
                        identity, idempotent, commutative, diagonal)
    raise InterfaceError("unknown interface type kind")


def array_mutation_step(name):
    return name == "@element" or name.startswith(("@index:", "@range:", "@stride:"))


def encode_array_mutation_step(step):
    if isinstance(step, Domain):
        if not step or step.lo != step.hi:
            return "@element"
        step = step.lo
    return f"@index:{step}" if isinstance(step, int) else step


def array_mutation_projection(ty, names):
    steps = []
    for name in names:
        if array_mutation_step(name):
            if isinstance(ty, Slice) and not steps:
                length = ty.domain.hi
            elif isinstance(ty, Array) and not ty.packed_bits:
                length = ty.length
            else:
                return None
            if name == "@element":
                steps.append(Domain.of((0, length - 1)))
            elif name.startswith("@range:"):
                interval = re.fullmatch(r"@range:(0|[1-9][0-9]{0,18}):(0|[1-9][0-9]{0,18})", name)
                if interval is None:
                    return None
                start, stop = map(int, interval.groups())
                if not 0 <= start < stop <= length:
                    return None
                steps.append(Domain.of((start, stop - 1)))
            elif name.startswith("@stride:"):
                interval = re.fullmatch(
                    r"@stride:(0|[1-9][0-9]{0,18}):(0|[1-9][0-9]{0,18}):([1-9][0-9]{0,18})", name)
                if interval is None:
                    return None
                start, stop, stride = map(int, interval.groups())
                if not 0 <= start < stop <= length or not 1 < stride < stop - start:
                    return None
                steps.append(Domain.of((start, stop - 1)).with_congruence(start, stride))
            else:
                index = name.removeprefix("@index:")
                if not re.fullmatch(r"0|[1-9][0-9]{0,18}", index) or int(index) >= length:
                    return None
                steps.append(Domain.point(int(index)))
            ty = ty.element
        else:
            if not isinstance(ty, Record):
                return None
            ty = dict(ty.fields).get(name)
            steps.append(name)
        if isinstance(ty, ReadOnly):
            return None
    return (tuple(steps), ty) if isinstance(ty, Scalar) else None
