"""Small helpers over abstract domains and C literals."""

from __future__ import annotations

import struct

from float_domain import FloatDomain
from value_domain import Domain

from .errors import FloatConstant
from .typesys import Scalar, unqualified, WIDE, WIDTHS, ZERO


def known_float_bit_domain(mask, data, offset, width, endian):
    window = (1 << width) - 1
    if (mask >> offset) & window != window:
        return None
    raw = ((data >> offset) & window).to_bytes(width // 8, "little")
    orders = ("<", ">") if endian == "native" else ((">",) if endian == "be" else ("<",))
    fmt = "f" if width == 32 else "d"
    return FloatDomain.union(FloatDomain.point(struct.unpack(order + fmt, raw)[0], width) for order in orders)


def domain_union(domains):
    domains = tuple(domains)
    return FloatDomain.union(domains) if domains and isinstance(domains[0], FloatDomain) else Domain.union(domains)


def integer_states(values):
    return {uid: domain for uid, domain in values.items() if not isinstance(domain, FloatDomain)}


def zero_domain(ty):
    return FloatDomain.point(0, ty.bits) if ty.floating else ZERO


def container(bits, signed):
    return Domain.of((-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1))


def physical(ty, token):
    if ty.computed:
        token.error("a computed type requires a physical seal: in {uint16, uint32} 32")
    if not ty.domain:
        token.error("an empty value set has no physical storage")
    if ty.floating:
        return ty
    if ty.bits or ty.boolean:
        return ty
    signed = ty.domain.lo < 0
    for bits in WIDTHS:
        if ty.domain.within(container(bits, signed)):
            return Scalar(ty.domain, bits, signed)
    token.error(f"no fixed-width integer container fits {ty.domain}")


def ascii_table_trace(lines):
    return "\n".join(lines).encode("ascii", "backslashreplace").decode("ascii") + "\n"


def c_int(n):
    if n < 0:
        if n < -(1 << 63):
            return "(-((lan_int)UINT64_C(9223372036854775807)) - 1)"
        return f"(-{c_int(-n - 1)} - 1)"
    if n < (1 << 63):
        return f"((lan_int)UINT64_C({n}))"
    if n < (1 << 64):
        return f"((lan_uint)UINT64_C({n}))"
    return "((lan_uint)UINT64_C(18446744073709551615))"


def c_uint(n):
    if n < 0:
        return f"((lan_uint){c_int(n)})"
    if n < (1 << 63):
        return f"((lan_uint){c_int(n)})"
    if n < (1 << 64):
        return f"((lan_uint)UINT64_C({n}))"
    return "((lan_uint)UINT64_C(18446744073709551615))"


def trunc_div(a, b):
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def float_literal_domain(literals, bits):
    result = FloatDomain(bits)
    for literal in literals:
        value = FloatConstant(float(literal)) if isinstance(literal, str) else literal
        result |= FloatDomain(bits, zeros=3) if value == 0 else FloatDomain.point(value, bits)
    return result


def plain_integer_parameter(ty):
    ty = unqualified(ty)
    return isinstance(ty, Scalar) and not (ty.floating or ty.boolean or ty.enum or ty.character)


def replay_scalar(ty):
    return isinstance(ty, Scalar) and not (
        ty.floating or ty.enum or ty.character or ty.result_arms is not None)


def commutative_integer_signature(params, result):
    return (len(params) == 2 and plain_integer_parameter(params[0]) and params[0] == params[1]
            and plain_integer_parameter(result))


def diagonal_integer_signature(params, result):
    return (commutative_integer_signature(params, result)
            and params[0].domain.within(result.domain))


def closed_integer_property(parameter, result):
    if not (plain_integer_parameter(parameter) and plain_integer_parameter(result)):
        return False
    parameter = unqualified(parameter)
    return result.bits <= parameter.bits and result.domain.within(parameter.domain)
