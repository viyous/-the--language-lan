"""Mutation, guard, accessor and storage summaries carried by .table interfaces."""

from __future__ import annotations

from dataclasses import field, fields, replace

from return_reasoning import ReturnCase, ReturnObservation
from storage_reasoning import StorageCount
from bit_domain import bitwise_domain
from float_domain import FloatDomain, FloatDomainError
from value_domain import Domain
from table_interface import InterfaceError, decode_domain, encode_domain, fields as interface_fields

from .lexer import C_KEYWORDS, NAME, RESERVED
from .syntax import Node
from .typesys import (Array, BorrowArray, BorrowRecord, BorrowScalar, CopyArray, HeapArray, HeapObject,
                      HeapRecord, Interface, ReadOnly, Record, record_text_type, RecordLink, Scalar, Slice,
                      unqualified, WIDE, WIDTHS, ZERO)
from .domain_utils import plain_integer_parameter, replay_scalar, trunc_div
from .interface_types import (array_mutation_projection, array_mutation_step, encode_interface_type,
                              graph_or_chain_parameter, interface_rebind_guard, mutable_chain, mutable_graph,
                              valid_interface_rebind, valid_interface_rebind_effects)

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("decode_result_content",)
decode_result_content = None


def decode_mutation_postcondition(encoded, scalar):
    if scalar.floating:
        if not isinstance(encoded, dict) or set(encoded) != {"float"}:
            raise InterfaceError("invalid floating mutation postcondition")
        try:
            post = FloatDomain.from_dict(encoded["float"])
        except FloatDomainError as exc:
            raise InterfaceError(f"invalid floating mutation postcondition: {exc}") from exc
        if post.bits != scalar.bits:
            raise InterfaceError("floating mutation postcondition has the wrong width")
    else:
        post = decode_domain(encoded)
    if not post or not post.within(scalar.domain):
        raise InterfaceError("mutation postcondition exceeds its field contract")
    return post


def graph_write_path(record, path, donor=None):
    if not 1 <= len(path) <= 32:
        return False
    fields = dict(record.fields)
    for name in path[:-1]:
        field = fields.get(name)
        if not isinstance(field, RecordLink) or field.readonly or field.identity != record.identity:
            return False
    field = fields.get(path[-1])
    return isinstance(field, RecordLink) and (donor is None or field.identity == donor.identity)


def graph_write_program(body, parameter, forward, *, donor=None, allow_cut=True, record=None):
    def member_path(node):
        parts = []
        while node.kind == "member":
            node, field = node.args
            parts.append(field.text)
        return tuple(reversed(parts)) if node.kind == "name" and node.args[0] == parameter else ()

    def nonnull_checks(condition):
        checks, other = set(), False
        guards = [condition]
        while guards:
            guard = guards.pop()
            if guard.kind != "binary":
                other = True
                continue
            op, left, right = guard.args
            if op == "&&":
                guards.extend((left, right))
                continue
            if left.kind == "null":
                left, right = right, left
            path = member_path(left)
            if op not in ("!=", "!==") or right.kind != "null" or not path:
                other = True
            else:
                checks.add(path)
        return checks, other

    def pure(node):
        if isinstance(node, Node):
            return node.kind not in ("call", "propagate") and all(map(pure, node.args))
        return not isinstance(node, (list, tuple)) or all(map(pure, node))

    effects = []
    payload = [False]

    def walk(statements, checks, optional, *, top=False, in_loop=False, flags=()):
        flags = set(flags)
        for offset, statement in enumerate(statements):
            if len(effects) >= 64:
                return False
            if statement.kind == "if":
                condition, yes, no = statement.args
                if not pure(condition):
                    return False
                required, other = nonnull_checks(condition)
                if (not walk(yes, checks | required, optional or other, in_loop=in_loop, flags=flags)
                        or not walk(no, checks, True, in_loop=in_loop, flags=flags)):
                    return False
            elif statement.kind == "while" and donor is None:
                condition, repeated = statement.args[:2]
                # Repeated null-only effects cannot add reachable targets.
                # One weak pass already includes every possible cut.
                if not pure(condition) or not walk(repeated, checks, True, in_loop=True, flags=flags):
                    return False
            elif statement.kind == "continue" and in_loop:
                continue
            elif statement.kind == "define":
                # A local binding writes nothing.  A later write through it is a
                # separate statement this walk judges on its own, so keeping the
                # binding here cannot hide a graph write.
                if not pure(statement):
                    return False
                name, annotation, value = statement.args
                flags.discard(name.text)
                if (value is not None and value.kind == "boolean"
                        and (annotation is None
                             or annotation.kind == "type_name" and annotation.args == ("boolean",))):
                    flags.add(name.text)
            elif statement.kind == "assign":
                target, op, value = statement.args
                if target.kind == "name" and target.args[0] in flags and op.text == "=" and value.kind == "boolean":
                    continue
                path = member_path(target)
                cutting = value.kind == "null" and allow_cut
                attaching = donor is not None and value.kind == "name" and value.args[0] == donor
                if op.text != "=" or not path:
                    return False
                if not (cutting or attaching):
                    # A write that does not end on a link only changes payloads,
                    # which the coarse `@graph` effect already covers.  Keeping
                    # the program precise here is what stops an unrelated scalar
                    # write from degrading every link write to "anywhere".
                    if (record is not None and pure(value)
                            and not isinstance(dict(record.fields).get(path[-1]), RecordLink)):
                        payload[0] = True
                        continue
                    return False
                # A null prefix prevents the write and a null final edge is
                # already in the required state for a cut.
                prefixes = len(path) + int(cutting)
                guaranteed = not optional and checks <= {path[:i] for i in range(1, prefixes)}
                effects.append(("cut" if cutting else "attach", "must" if guaranteed else "may", path))
            elif statement.kind in ("eval", "return"):
                value, = statement.args
                if statement.kind == "return" and (not top or offset != len(statements) - 1):
                    return False
                if value is not None and value.kind == "call":
                    forwarded = forward(value)
                    if forwarded is None:
                        return False
                    for operation, mode, path in forwarded:
                        prefixes = len(path) + int(operation == "cut")
                        guaranteed = not optional and mode == "must" and checks <= {
                            path[:i] for i in range(1, prefixes)}
                        effects.append((operation, "must" if guaranteed else "may", path))
                elif not pure(value):
                    return False
            else:
                return False
        return True

    if not (walk(body, set(), False, top=True) and len(effects) <= 64):
        return (), False
    return effects, payload[0]


def decode_owning(raw, contract):
    if not isinstance(raw, list) or len(raw) > len(contract.params):
        raise InterfaceError("invalid owning parameter summary")
    result, seen = [], set()
    for item in raw:
        if (not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int
                or item[1] not in ("free", "move") or not 0 <= item[0] < len(contract.params)):
            raise InterfaceError("invalid owning parameter summary entry")
        index, method = item
        if index in seen or not isinstance(contract.params[index], (HeapRecord, HeapObject, HeapArray)):
            raise InterfaceError("invalid owning parameter summary entry")
        seen.add(index)
        result.append((index, method))
    return tuple(sorted(result))


def decode_mutations(raw, contract):
    if not isinstance(raw, list) or len(raw) > 8192:
        raise InterfaceError("invalid mutation summary")
    result, seen, ordered_prunes, precise_attachments, ordered_edits = [], set(), {}, set(), {}
    for item in raw:
        if (not isinstance(item, list) or len(item) != 3 or type(item[0]) is not int
                or not 0 <= item[0] < len(contract.params) or not isinstance(item[1], list)
                or any(not isinstance(name, str) for name in item[1])):
            raise InterfaceError("invalid mutation summary entry")
        index, names, encoded = item
        key = index, tuple(names)
        if key in seen:
            raise InterfaceError("duplicate mutation summary entry")
        seen.add(key)
        ty = contract.params[index]
        if names and names[0] in ("@interface_before", "@interface_after"):
            field = (dict(ty.record.fields).get(names[1])
                     if isinstance(ty, BorrowRecord) and not ty.readonly and len(names) > 2 else None)
            if not isinstance(field, Interface) or "*" not in names[2:]:
                raise InterfaceError("invalid interface write around rebind")
            leaf = field.schema
            for step in names[2:]:
                if step == "*":
                    continue
                leaf = dict(leaf.fields).get(step) if isinstance(leaf, Record) else None
                if leaf is None:
                    break
            post = None
            if encoded is not None:
                if not isinstance(leaf, Scalar):
                    raise InterfaceError("invalid interface write before rebind")
                post = decode_domain(encoded)
                if not post or not post.within(leaf.domain):
                    raise InterfaceError("invalid interface write before rebind")
            result.append((index, tuple(names), post))
            continue
        if names and names[0] == "@interface_rebind":
            parameter_indices = {str(i): i for i in range(len(contract.params))}

            def parameter(name):
                return parameter_indices.get(name, -1)

            def literal(position):
                return (interface_rebind_guard(contract.params, parameter(names[position]))
                        and names[position + 1] in ("0", "1"))

            # Length 3 is an unguarded rebind, 5 one guarded arm, 7/9/... a
            # conjunction of literals, and 6 the exhaustive two-arm form.
            conjunction = len(names) >= 5 and len(names) % 2 == 1
            exhaustive = len(names) == 6
            donor = parameter(names[2]) if len(names) >= 3 else -1
            valid = (donor >= 0 and encoded is None
                     and (contract.result is None or isinstance(contract.result, Scalar))
                     and valid_interface_rebind(contract.params, index, names[1], donor)
                     and (len(names) == 3 or conjunction or exhaustive))
            if valid and exhaustive:
                valid = (literal(3)
                         and valid_interface_rebind(contract.params, index, names[1],
                                                    parameter(names[5])))
            if valid and conjunction:
                seen = {}
                for position in range(3, len(names) - 1, 2):
                    guard = parameter(names[position])
                    bit = names[position + 1]
                    if not literal(position) or seen.setdefault(guard, bit) != bit:
                        valid = False
                        break
            if not valid:
                raise InterfaceError("invalid interface rebind mutation summary")
            result.append((index, tuple(names), None))
            continue
        if mutable_graph(ty):
            # A payload write never moves a link, so `@graph` describes an
            # attachment parameter just as precisely as a plain one: the
            # attachment is only ever about the topology.
            allowed = (names == ["@graph"]
                       or ty.attach is not None and names == ["@attach_graph"]
                       or ty.prune and names == ["@prune"])
            if (isinstance(ty, BorrowRecord) and ty.prune and ty.attach is None
                    and len(names) == 2 and names[0] == "@graph_root_null"):
                allowed = isinstance(dict(ty.record.fields).get(names[1]), RecordLink)
            if (isinstance(ty, BorrowRecord) and ty.prune and ty.attach is None
                    and names and names[0] == "@graph_path_null"):
                allowed = len(names) >= 3 and graph_write_path(ty.record, names[1:])
            if (isinstance(ty, BorrowRecord) and ty.prune and ty.attach is None
                    and len(names) >= 4 and names[0] == "@graph_path_prune"):
                position = ordered_prunes.get(index, 0)
                allowed = (position < 64 and names[1] == str(position) and names[2] in ("must", "may")
                           and graph_write_path(ty.record, names[3:]))
                if allowed:
                    ordered_prunes[index] = position + 1
            if (isinstance(ty, BorrowRecord) and ty.attach is not None
                    and len(names) >= 3 and names[0] == "@graph_path_attach"):
                donor = contract.params[ty.attach]
                allowed = (index not in precise_attachments and names[1] in ("must", "may")
                           and isinstance(donor, BorrowRecord)
                           and graph_write_path(ty.record, names[2:], donor.record))
                if allowed:
                    precise_attachments.add(index)
            if (isinstance(ty, BorrowRecord) and ty.attach is not None
                    and len(names) >= 5 and names[0] == "@graph_path_edit"):
                position = ordered_edits.get(index, 0)
                operation = names[2]
                donor = contract.params[ty.attach]
                if operation == "cut" and ty.prune:
                    route_ok = graph_write_path(ty.record, names[4:])
                elif operation == "attach" and isinstance(donor, BorrowRecord):
                    route_ok = graph_write_path(ty.record, names[4:], donor.record)
                else:
                    route_ok = False
                allowed = (position < 64 and names[1] == str(position) and names[3] in ("must", "may") and route_ok)
                if allowed:
                    ordered_edits[index] = position + 1
            if not allowed or encoded is not None:
                raise InterfaceError("invalid graph mutation summary")
            result.append((index, tuple(names), None))
            continue
        if mutable_chain(ty):
            # `@chain` widens payloads only, exactly like `@graph`, so an
            # attachment contract may still declare its topology edit beside it.
            allowed = names == ["@chain"]
            if ty.attach is not None:
                allowed = (allowed or names == ["@attach"]
                           or ty.chain_limit is True and names == ["@attach_deep"])
                if ty.chain_limit is not True and len(names) == 2 and names[0] == "@attach_at":
                    donor = contract.params[ty.attach]
                    maximum = ty.chain_limit - donor.chain_limit
                    allowed = names[1] in {str(depth) for depth in range(2, maximum + 1)}
            else:
                allowed = allowed or ty.prune and names in (["@prune"], ["@skip"])
            if not allowed or encoded is not None:
                raise InterfaceError("invalid chain mutation summary")
            result.append((index, tuple(names), None))
            continue
        if (not isinstance(ty, (BorrowRecord, BorrowScalar, BorrowArray, Slice, Interface)) or ty.readonly
                or isinstance(ty, Slice) and ty.copy):
            raise InterfaceError("mutation summary requires a writable borrowed parameter")
        if isinstance(ty, (BorrowRecord, Interface)):
            ty = ty.record if isinstance(ty, BorrowRecord) else ty.schema
        elif isinstance(ty, BorrowScalar):
            ty = ty.scalar
        elif isinstance(ty, BorrowArray):
            ty = ty.array
        if any(map(array_mutation_step, names)):
            if not isinstance(contract.params[index], (BorrowRecord, BorrowArray, Interface, Slice)):
                raise InterfaceError("invalid array field mutation summary")
            projection = array_mutation_projection(ty, names)
            if projection is None:
                raise InterfaceError("invalid array field mutation path")
            post = None
            if encoded is not None:
                steps, leaf = projection
                if (not isinstance(contract.params[index], (BorrowRecord, BorrowArray, Slice))
                        or not array_mutation_step(names[-1])
                        or not all(isinstance(step, str) for step in steps[:-1])):
                    raise InterfaceError("invalid array field mutation summary")
                post = decode_mutation_postcondition(encoded, leaf)
            result.append((index, tuple(names), post))
            continue
        for name in names:
            if name == "*" and isinstance(ty, Interface):
                ty = ty.schema
                continue
            if not isinstance(ty, Record) or name not in dict(ty.fields):
                raise InterfaceError("invalid mutation field path")
            ty = dict(ty.fields)[name]
            if isinstance(ty, ReadOnly):
                raise InterfaceError("mutation summary cannot modify an immutable field")
        if (isinstance(ty, Slice) and encoded is not None
                and ty.domain.hi > 0):
            ty = ty.element
        if isinstance(ty, Scalar):
            post = decode_mutation_postcondition(encoded, ty) if encoded is not None else ty.domain
        elif isinstance(ty, (Array, Slice)) and encoded is None:
            post = None
        else:
            raise InterfaceError("mutation summary must identify storage leaves")
        result.append((index, tuple(names), post))
    if any(names and names[0] == "@interface_rebind" for _, names, _ in result):
        if (not valid_interface_rebind_effects(contract.params, ((index, names) for index, names, _ in result))
                or any(post is not None and names and array_mutation_step(names[-1])
                       for _, names, post in result)):
            raise InterfaceError("invalid interface rebind mutation summary")
    def sole_payload(names):
        # `@graph` only widens scalar payloads; it never moves a link, so the
        # ordered program still describes the topology exactly.  Anything else
        # (a coarse attach or prune) would contradict that description.
        return names == ("@graph",)

    if any(index in ordered_prunes and not sole_payload(names)
           and (not names or names[0] != "@graph_path_prune") for index, names, _ in result):
        raise InterfaceError("ordered graph pruning cannot be combined with other graph effects")
    if any(index in precise_attachments and not sole_payload(names)
           and (not names or names[0] != "@graph_path_attach") for index, names, _ in result):
        raise InterfaceError("precise graph attachment cannot be combined with other graph effects")
    if any(index in ordered_edits and not sole_payload(names)
           and (not names or names[0] != "@graph_path_edit") for index, names, _ in result):
        raise InterfaceError("ordered graph edits cannot be combined with other graph effects")
    return tuple(result)


def decode_result_fields(raw, result):
    ty = result.record if isinstance(result, HeapRecord) else result
    if not isinstance(ty, Record) or not isinstance(raw, list) or len(raw) > ty.leaves:
        raise InterfaceError("invalid record result fields")
    fields = {}
    budget = [4096]
    for item in raw:
        if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], list)
                or not item[0] or any(not isinstance(name, str) for name in item[0])):
            raise InterfaceError("invalid record result field entry")
        names, encoded = item
        field = ty
        for name in names:
            if not isinstance(field, Record) or name not in dict(field.fields):
                raise InterfaceError("invalid record result field path")
            field = unqualified(dict(field.fields)[name])
        if not isinstance(field, (Scalar, Slice, Array)) or tuple(names) in fields:
            raise InterfaceError("invalid or duplicate record result field")
        fields[tuple(names)] = decode_result_content(field, encoded, budget)
    return tuple(sorted(fields.items()))


def guard_boolean_parameter(ty):
    ty = unqualified(ty)
    return isinstance(ty, Scalar) and ty.boolean


def guard_parameter_kind(ty):
    if plain_integer_parameter(ty):
        return "param"
    if guard_boolean_parameter(ty):
        return "bool_param"
    if isinstance(ty, Slice) and not graph_or_chain_parameter(ty):
        return "slice_param"
    if isinstance(ty, CopyArray) or (isinstance(ty, BorrowArray) and not ty.nullable
                                     and not graph_or_chain_parameter(ty)):
        return "array_param"
    return None


GUARD_COMPARISONS = ("==", "!=", "===", "!==", "<", "<=", ">", ">=")
GUARD_BITWISE = ("&", "|", "^")
GUARD_SHIFTS = ("<<", ">>")
GUARD_DIVISION = ("/", "%")
GUARD_ARITHMETIC = ("+", "-", "*") + GUARD_DIVISION + GUARD_BITWISE + GUARD_SHIFTS


def guard_operand_bounds(expression, contract):
    kind, item = expression[:2]
    if kind == "const":
        return item, item
    if kind == "param":
        domain = contract.params[item].domain
        return (domain.lo, domain.hi) if domain else None
    if kind != "arith" or item not in ("+", "-", "*", *GUARD_BITWISE):
        return None
    left, right = (guard_operand_bounds(operand, contract) for operand in expression[2:])
    if left is None or right is None:
        return None
    if item == "+":
        lo, hi = left[0] + right[0], left[1] + right[1]
    elif item == "-":
        lo, hi = left[0] - right[1], left[1] - right[0]
    elif item in GUARD_BITWISE:
        domain, _ = bitwise_domain(item, Domain.of(left), Domain.of(right))
        lo, hi = domain.lo, domain.hi
    else:
        products = [a * b for a in left for b in right]
        lo, hi = min(products), max(products)
    return (lo, hi) if WIDE.lo <= lo <= hi <= WIDE.hi else None


def valid_guard_shift_count(value, contract):
    limits = guard_operand_bounds(value, contract)
    return limits is not None and 0 <= limits[0] <= limits[1] < 128


def valid_guard_divisor(value, contract):
    kind, item = value[:2]
    if kind == "const":
        return item != 0
    if kind == "param":
        domain = contract.params[item].domain
        return domain is not None and not (domain & Domain.point(0))
    limits = guard_operand_bounds(value, contract)
    return limits is not None and (limits[0] > 0 or limits[1] < 0)


def encode_guard_operand(value):
    if value[0] == "arith":
        return [value[0], value[1], *(encode_guard_operand(part) for part in value[2:])]
    return list(value)


def encode_guard(guard):
    kind, *parts = guard
    if kind in ("&&", "||", "!", "choose", "outcomes"):
        return [kind, *(encode_guard(part) for part in parts)]
    if kind in ("bool", "bool_param"):
        return [kind, parts[0]]
    if kind == "in":
        return [kind, encode_guard_operand(parts[0]), encode_domain(parts[1])]
    return [kind, *(encode_guard_operand(part) for part in parts)]


def decode_guard(raw, contract, mutations):
    if (mutations != () or not isinstance(contract.result, Scalar) or not contract.result.boolean
            or not all(guard_parameter_kind(ty) for ty in contract.params)):
        raise InterfaceError("guard summary requires a boolean result and supported unmodified parameters")
    budget = [128]

    def operand(value, depth):
        budget[0] -= 1
        if budget[0] < 0 or depth > 32 or not isinstance(value, list) or not value:
            raise InterfaceError("invalid or oversized guard operand")
        if value[0] == "arith" and len(value) == 4 and value[1] in GUARD_ARITHMETIC:
            op = value[1]
            left, right = (operand(part, depth + 1) for part in value[2:])
            if op in GUARD_SHIFTS and not valid_guard_shift_count(right, contract):
                raise InterfaceError("invalid guard shift count")
            if op in GUARD_DIVISION and not valid_guard_divisor(right, contract):
                raise InterfaceError("invalid guard divisor")
            return "arith", op, left, right
        if len(value) != 2 or type(value[1]) is not int:
            raise InterfaceError("invalid guard operand")
        kind, index = value
        if kind == "param" and 0 <= index < len(contract.params) and plain_integer_parameter(contract.params[index]):
            return kind, index
        if kind == "len" and 0 <= index < len(contract.params) and guard_parameter_kind(contract.params[index]) == "slice_param":
            return kind, index
        if kind == "const" and WIDE.lo <= index <= WIDE.hi:
            return kind, index
        raise InterfaceError("invalid guard operand")

    def predicate(value, depth):
        budget[0] -= 1
        if budget[0] < 0 or depth > 32 or not isinstance(value, list) or not value:
            raise InterfaceError("invalid or oversized guard summary")
        kind, *parts = value
        if (kind in ("&&", "||") and len(parts) == 2) or (kind == "choose" and len(parts) == 3) or (
                kind == "outcomes" and depth == 0 and len(parts) == 2):
            return kind, *(predicate(part, depth + 1) for part in parts)
        if kind == "bool" and len(parts) == 1 and type(parts[0]) is bool:
            return kind, parts[0]
        if (kind == "bool_param" and len(parts) == 1 and type(parts[0]) is int
                and 0 <= parts[0] < len(contract.params) and guard_boolean_parameter(contract.params[parts[0]])):
            return kind, parts[0]
        if kind == "!" and len(parts) == 1:
            return kind, predicate(parts[0], depth + 1)
        if kind == "in" and len(parts) == 2:
            return kind, operand(parts[0], depth + 1), decode_domain(parts[1])
        if kind in GUARD_COMPARISONS and len(parts) == 2:
            return kind, *(operand(part, depth + 1) for part in parts)
        raise InterfaceError("invalid guard predicate")

    return predicate(raw, 0)


def interface_index_domain(contract, step):
    if len(step) in (2, 3, 4) and step[0] == "param":
        bias = step[2] if len(step) >= 3 else 0
        terms = ((step[1], step[3] if len(step) == 4 else 1),)
    elif len(step) == 3 and step[0] == "affine":
        _, terms, bias = step
        if not isinstance(terms, tuple) or not 2 <= len(terms) <= 16:
            return None
    else:
        return None
    if type(bias) is not int or not WIDE.lo <= bias <= WIDE.hi:
        return None
    domain = ZERO
    for term in terms:
        if not isinstance(term, tuple) or len(term) != 2:
            return None
        index, scale = term
        if (type(index) is not int or not 0 <= index < len(contract.params)
                or not plain_integer_parameter(contract.params[index])
                or type(scale) is not int or not WIDE.lo <= scale <= WIDE.hi):
            return None
        scaled = contract.params[index].domain.affine(scale, 0)
        if not scaled or not scaled.within(WIDE):
            return None
        if len(terms) == 1:
            return scaled.affine(1, bias)
        domain = Domain.of((domain.lo + scaled.lo, domain.hi + scaled.hi))
        if not domain.within(WIDE):
            return None
    return domain.affine(1, bias)


def interface_index_step(terms, bias):
    if not terms:
        return bias
    if len(terms) == 1:
        (parameter, scale), = terms.items()
        if scale != 1:
            return "param", parameter, bias, scale
        return ("param", parameter, bias) if bias else ("param", parameter)
    return "affine", tuple(terms.items()), bias


def interface_accessor_member(contract, index, path):
    member = contract.params[index]
    for step in path:
        if step == "@len":
            if isinstance(member, Array):
                length = member.length
                width = next(bits for bits in WIDTHS if length < 1 << bits)
                member = Scalar(Domain.point(length), width)
            elif isinstance(member, Slice) and record_text_type(member):
                member = Scalar(member.domain, 32)
            else:
                return None
        elif isinstance(step, str) and isinstance(member, (Interface, Record)):
            member = unqualified(dict(member.fields).get(step))
        elif isinstance(member, Array) or isinstance(member, Slice) and record_text_type(member):
            if type(step) is int:
                domain = Domain.point(step)
            elif isinstance(step, tuple) and len(step) == 2 and step[0] == "from_end":
                if not isinstance(member, Slice):
                    return None
                distance = step[1]
                if type(distance) is int:
                    domain = Domain.point(distance)
                elif isinstance(distance, tuple):
                    domain = interface_index_domain(contract, distance)
                else:
                    return None
                if not domain or not domain.within(Domain.of((1, member.domain.lo))):
                    return None
                member = unqualified(member.element)
                continue
            elif isinstance(step, tuple):
                domain = interface_index_domain(contract, step)
            else:
                return None
            length = member.domain.lo if isinstance(member, Slice) else member.length
            if not domain or not domain.within(Domain.of((0, length - 1))):
                return None
            member = unqualified(member.element)
        else:
            return None
    return member


INTERFACE_FIELD_COMPARISONS = {"==": "==", "!=": "!=", "===": "==", "!==": "!=",
                               "<": "<", "<=": "<=", ">": ">", ">=": ">="}
INTERFACE_FIELD_LOGIC = ("&&", "||", "^^")
INTERFACE_FIELD_OPERATORS = (*INTERFACE_FIELD_COMPARISONS, *INTERFACE_FIELD_LOGIC,
                             "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>")


def valid_interface_accessor(contract, accessor, mutations):
    index, path, *arithmetic = accessor
    if (type(index) is not int or not 0 <= index < len(contract.params)
            or not isinstance(path, tuple) or not 1 <= len(path) <= 16
            or not isinstance(path[0], str) or mutations != ()
            or not isinstance(contract.params[index], Interface) or not isinstance(contract.result, Scalar)
            or not all(isinstance(ty, (Scalar, Interface)) for ty in contract.params)
            or contract.bounds or contract.parameter_expressions or contract.result_expressions):
        return False
    member = interface_accessor_member(contract, index, path)
    if not isinstance(member, Scalar):
        return False
    result = contract.result
    if arithmetic:
        op, other, *source = arithmetic
        other_index = source[0] if source else index
        if ((op == "choose" and len(source) != 2)
                or (op != "choose" and (op not in INTERFACE_FIELD_OPERATORS or len(source) > 1))
                or not 1 <= len(other) <= 16 or not isinstance(other[0], str)
                or type(other_index) is not int or not 0 <= other_index < len(contract.params)
                or not isinstance(contract.params[other_index], Interface)):
            return False
        right = interface_accessor_member(contract, other_index, other)
        if op == "choose":
            selector = source[1]
            if isinstance(selector, tuple) and len(selector) == 2:
                selector_index, selector_path = selector
                if (type(selector_index) is not int or not 0 <= selector_index < len(contract.params)
                        or not isinstance(contract.params[selector_index], Interface)
                        or not 1 <= len(selector_path) <= 16 or not isinstance(selector_path[0], str)):
                    return False
                selector_type = interface_accessor_member(contract, selector_index, selector_path)
            elif type(selector) is int and 0 <= selector < len(contract.params):
                selector_type = contract.params[selector]
            else:
                return False
            return (guard_boolean_parameter(selector_type)
                    and result.result_arms is None
                    and all(isinstance(ty, Scalar) and ty.result_arms is None
                            and interface_field_fits_result(ty, result) for ty in (member, right)))
        if op in INTERFACE_FIELD_LOGIC:
            if not all(isinstance(ty, Scalar) and ty.boolean and ty.result_arms is None
                       for ty in (member, right, result)):
                return False
            operation = {"&&": int.__and__, "||": int.__or__, "^^": int.__xor__}[op]
            outcomes = {operation(a, b) for a in (0, 1) for b in (0, 1)
                        if member.domain & Domain.point(a) and right.domain & Domain.point(b)}
            return bool(outcomes) and Domain.of(*((value, value) for value in outcomes)).within(result.domain)
        if isinstance(right, Scalar) and (member.floating or right.floating):
            operands = member, right
            bits = max(ty.bits for ty in operands if ty.floating)
            if (result.result_arms is not None
                    or any(ty.result_arms is not None or not ty.domain or ty.bits > bits
                           or not (ty.floating or plain_integer_parameter(ty)) for ty in operands)):
                return False
            try:
                a, b = (ty.domain.convert(bits) if ty.floating
                        else FloatDomain.from_integers(ty.domain, bits, exact=True) for ty in operands)
            except FloatDomainError:
                return False
            if op in INTERFACE_FIELD_COMPARISONS:
                return result.boolean and a.compare(INTERFACE_FIELD_COMPARISONS[op], b).within(result.domain)
            if op not in ("+", "-", "*", "/") or not result.floating or result.bits < bits:
                return False
            return a.binary(op, b).convert(result.bits).within(result.domain)
        if not isinstance(right, Scalar) or member.result_arms is not None or right.result_arms is not None:
            return False
        a, b = member.domain, right.domain
        if not a or not b:
            return False
        if op in INTERFACE_FIELD_COMPARISONS:
            if (not result.boolean or result.result_arms is not None or member.boolean != right.boolean
                    or member.enum != right.enum
                    or (member.character != right.character and not (
                        member.character and right.character and "ascii" in (member.character, right.character)))):
                return False
            comparison = INTERFACE_FIELD_COMPARISONS[op]
            if comparison in ("==", "!="):
                true_possible = bool(a & b)
                false_possible = a.lo != a.hi or b.lo != b.hi or a.lo != b.lo
                if comparison == "!=":
                    true_possible, false_possible = false_possible, true_possible
            else:
                if comparison in (">", ">="):
                    a, b = b, a
                if comparison in ("<=", ">="):
                    true_possible, false_possible = a.lo <= b.hi, a.hi > b.lo
                else:
                    true_possible, false_possible = a.lo < b.hi, a.hi >= b.lo
            domain = Domain.of(*((value, value) for value, possible in enumerate((false_possible, true_possible))
                                 if possible))
            return domain.within(result.domain)
        if (not all(plain_integer_parameter(ty) for ty in (member, right, result))
                or result.result_arms is not None or max(member.bits, right.bits) > result.bits):
            return False
        if op in ("<<", ">>"):
            if not 0 <= b.lo <= b.hi < 128:
                return False
            shifts = [x << y if op == "<<" else x >> y
                      for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
            return Domain.of((min(shifts), max(shifts))).within(result.domain)
        if op in ("&", "|", "^"):
            if len(a.ranges) * len(b.ranges) > 64:
                a, b = Domain.of((a.lo, a.hi)), Domain.of((b.lo, b.hi))
            domain, _ = bitwise_domain(op, a, b)
            return domain.within(result.domain)
        if op in ("/", "%") and b & ZERO:
            return False
        if op == "/":
            spans = []
            for divisor in (b & Domain.of((b.lo, -1)), b & Domain.of((1, b.hi))):
                if divisor:
                    quotients = [trunc_div(x, y) for x in (a.lo, a.hi) for y in (divisor.lo, divisor.hi)]
                    spans.append((min(quotients), max(quotients)))
            return Domain.of(*spans).within(result.domain)
        if op == "%":
            limit = max(abs(b.lo), abs(b.hi)) - 1
            lo, hi = max(min(a.lo, 0), -limit), min(max(a.hi, 0), limit)
        elif op == "*":
            products = [x * y for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
            lo, hi = min(products), max(products)
        else:
            lo, hi = (a.lo + b.lo, a.hi + b.hi) if op == "+" else (a.lo - b.hi, a.hi - b.lo)
        return Domain.of((lo, hi)).within(result.domain)
    return interface_field_fits_result(member, result)


def interface_field_fits_result(member, result):
    if encode_interface_type(member) == encode_interface_type(result):
        return True
    if (type(member.domain) is type(result.domain) and replace(member, domain=result.domain) == result):
        return member.domain.within(result.domain)
    if (member.bits > result.bits or member.result_arms is not None or result.result_arms is not None
            or member.boolean or result.boolean or member.enum is not None or result.enum is not None
            or member.character is not None or result.character is not None
            or (member.floating and not result.floating)):
        return False
    if result.floating:
        try:
            domain = (member.domain.convert(result.bits, "exact") if member.floating else
                      FloatDomain.from_integers(member.domain, result.bits, exact=True))
            return domain.within(result.domain)
        except FloatDomainError:
            return False
    return member.domain.within(result.domain)


def valid_foreign_symbol(name):
    return (isinstance(name, str) and name.isascii() and NAME.fullmatch(name) and name not in RESERVED
            and name not in C_KEYWORDS and name not in ("main", "calloc", "malloc", "free", "abort")
            and not name.startswith(("lan_", "lh_")))


STORAGE_SUMMARY_LIMIT = 64
SUMMARY_INTERVAL_LIMIT = 32


def count_column(ty, column):
    for name in column:
        ty = unqualified(dict(ty.fields).get(name)) if isinstance(ty, Record) else None
    return ty if isinstance(ty, Scalar) and isinstance(ty.domain, Domain) else None


def storage_count_target(contract, parameter, path):
    """Resolve the inline storage and scalar column named by a count fact."""
    if type(parameter) is not int or not 0 <= parameter < len(contract.params):
        return None
    param = contract.params[parameter]
    if isinstance(param, Slice) and not (param.copy or param.graph or param.chain_limit is not None):
        storage, length = param, param.domain.lo
    elif isinstance(param, (BorrowArray, BorrowRecord)) and not param.nullable:
        storage = param.record if isinstance(param, BorrowRecord) else param.array
        length = None
    else:
        return None
    split = path.index(None) if None in path else len(path)
    prefix, column = path[:split], path[split + 1:]
    if split < len(path) and not column:
        return None
    for field in prefix:
        storage = unqualified(dict(storage.fields).get(field)) if isinstance(storage, Record) else None
    if isinstance(storage, Array) and not storage.packed_bits:
        length = storage.length
    elif not isinstance(storage, Slice) or prefix:
        return None
    scalar = count_column(storage.element, column)
    if scalar is None:
        return None
    return prefix, column, length, scalar


def small_summary_domain(domain):
    return (isinstance(domain, Domain) and bool(domain) and not domain.predicates
            and len(domain.intervals) <= SUMMARY_INTERVAL_LIMIT
            and type(domain.lo) is int and type(domain.hi) is int)


def decode_summary_domain(raw, container_domain, label):
    if (not isinstance(raw, list) or not 1 <= len(raw) <= SUMMARY_INTERVAL_LIMIT
            or any(not isinstance(piece, list) or len(piece) != 4
                   or piece[2:] != [[], []] or any(type(n) is not int for n in piece[:2])
                   for piece in raw)):
        raise InterfaceError(f"invalid domain in {label}")
    domain = decode_domain(raw)
    if not small_summary_domain(domain) or not domain.within(container_domain):
        raise InterfaceError(f"invalid domain in {label}")
    return domain


def decode_storage_counts(raw, contract):
    if not isinstance(raw, list) or len(raw) > STORAGE_SUMMARY_LIMIT:
        raise InterfaceError("invalid storage count summary")
    results = []
    seen = set()
    for item in raw:
        if not isinstance(item, list) or len(item) != 7:
            raise InterfaceError("invalid storage count entry")
        index, path, start, length, selected, lo, hi = item
        if (not isinstance(path, list) or len(path) > 64
                or any(field is not None and (not isinstance(field, str) or not NAME.fullmatch(field))
                       for field in path) or path.count(None) > 1
                or any(type(n) is not int for n in (start, length, lo, hi))):
            raise InterfaceError("invalid storage count region or bounds")
        path = tuple(path)
        target = storage_count_target(contract, index, path)
        if (target is None or not 0 <= start < start + length <= target[2]
                or not 0 <= lo <= hi <= length):
            raise InterfaceError("invalid storage count region or bounds")
        selected = decode_summary_domain(selected, target[3].domain, "storage count")
        key = index, path, start, length, selected
        if key in seen:
            raise InterfaceError("duplicate storage count")
        seen.add(key)
        results.append(StorageCount(index, path, start, length, selected, (lo, hi)))
    return tuple(results)


def decode_return_observation(raw, contract):
    interface_fields(raw, ("parameters", "cases"), "return observation")
    indices, rows = raw["parameters"], raw["cases"]
    if (not replay_scalar(contract.result) or not isinstance(indices, list) or not 1 <= len(indices) <= 8
            or any(type(index) is not int or not 0 <= index < len(contract.params)
                   or not replay_scalar(unqualified(contract.params[index])) for index in indices)
            or sorted(set(indices)) != indices or not isinstance(rows, list) or not 2 <= len(rows) <= 32):
        raise InterfaceError("invalid scalar return observation")
    cases = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(indices) + 1:
            raise InterfaceError("invalid scalar return case")
        result = decode_summary_domain(row[0], contract.result.domain, "scalar return case")
        arguments = tuple(decode_summary_domain(domain, unqualified(contract.params[index]).domain,
                                                "scalar return argument")
                          for index, domain in zip(indices, row[1:]))
        cases.append(ReturnCase(result, arguments))
    return ReturnObservation(tuple(indices), tuple(cases))
