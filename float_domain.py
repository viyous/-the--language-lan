"""IEEE binary32/binary64 abstract values and checked numeric conversions.

This module is independent of the parser and C emitter.  Endpoints are host
binary64 values, but every endpoint is representable in the selected format.
All rounding decisions use integers/Fraction, never a preliminary host float
conversion.  Thus decimal literals and integer conversions do not double-round.
Arithmetic models one IEEE round-to-nearest, ties-to-even operation; exact
rational extrema are rounded once, which encloses every concrete result.

Finite nonzero intervals, signed zeros, infinities and NaN are tracked
separately.  Set operations preserve zero signs; numeric guards should use
without_zero()/only_zero(), because +0 == -0.  NaN is an abstract class: payload
bits and signaling-NaN exceptions belong to a separate bit-view analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import struct

from value_domain import Domain
from table_interface import encode_domain, decode_domain, InterfaceError

POS_ZERO = 1
NEG_ZERO = 2
BOTH_ZEROS = POS_ZERO | NEG_ZERO


class FloatDomainError(ValueError):
    pass


class FloatConversionError(FloatDomainError):
    pass


@dataclass(frozen=True)
class BinaryFormat:
    bits: int
    precision: int
    emin: int
    emax: int

    @property
    def subnormal_exponent(self):
        return self.emin - self.precision + 1

    @property
    def min_subnormal(self):
        return math.ldexp(1.0, self.subnormal_exponent)

    @property
    def max_finite(self):
        return math.ldexp(float((1 << self.precision) - 1), self.emax - self.precision + 1)

    @property
    def max_integer_exact_prefix(self):
        return 1 << self.precision


FORMATS = {32: BinaryFormat(32, 24, -126, 127),
           64: BinaryFormat(64, 53, -1022, 1023)}


def binary_format(bits):
    if type(bits) is not int:
        raise FloatDomainError("floating representation width must be an integer")
    try:
        return FORMATS[bits]
    except (KeyError, TypeError):
        raise FloatDomainError("floating representation must be binary32 or binary64") from None


def _negative(value):
    if isinstance(value, float):
        return math.copysign(1.0, value) < 0
    if isinstance(value, str):
        return value.strip().startswith("-")
    return value < 0


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise FloatDomainError("boolean is not a floating numeric literal")
    try:
        return Fraction(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise FloatDomainError("expected a finite numeric value") from exc


def _special(value):
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("nan", "+nan", "-nan"):
            return math.nan
        if lower in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if lower in ("-inf", "-infinity"):
            return -math.inf
    if isinstance(value, float) and not math.isfinite(value):
        return value
    return None


def _pow2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _floor_log2(numerator, denominator):
    exponent = numerator.bit_length() - denominator.bit_length()
    if (numerator < denominator << exponent if exponent >= 0
            else numerator << -exponent < denominator):
        exponent -= 1
    return exponent


def quantize(value, bits=64, mode="nearest"):
    """Round an exact value to a format, preserving signed zero.

    mode is nearest (ties even), down (-infinity), up (+infinity), or zero.
    int, Fraction and decimal strings are exact inputs.  A float denotes its
    already-rounded host binary64 value.  Nonfinite inputs remain nonfinite.
    """
    fmt = binary_format(bits)
    if mode not in ("nearest", "down", "up", "zero"):
        raise FloatDomainError("unknown binary rounding mode")
    special = _special(value)
    if special is not None:
        return special
    exact = _fraction(value)
    negative = exact < 0 or not exact and _negative(value)
    if not exact:
        return -0.0 if negative else 0.0
    magnitude = abs(exact)
    exponent = _floor_log2(magnitude.numerator, magnitude.denominator)
    step = max(exponent - fmt.precision + 1, fmt.subnormal_exponent)
    scaled = magnitude / _pow2(step)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if mode == "nearest":
        quotient += (2 * remainder > scaled.denominator or
                     2 * remainder == scaled.denominator and quotient % 2 == 1)
    elif mode == "up" and not negative or mode == "down" and negative:
        quotient += bool(remainder)
    if quotient == 0:
        return -0.0 if negative else 0.0
    # Comparing exact integers avoids overflowing a host operation prematurely.
    rounded = quotient * _pow2(step)
    if rounded > Fraction.from_float(fmt.max_finite):
        toward_finite = mode == "zero" or mode == "down" and not negative or mode == "up" and negative
        result = fmt.max_finite if toward_finite else math.inf
    else:
        result = math.ldexp(float(quotient), step)
    return -result if negative else result


def is_representable(value, bits=64):
    special = _special(value)
    if special is not None:
        return True
    rounded = quantize(value, bits)
    return math.isfinite(rounded) and Fraction.from_float(rounded) == _fraction(value)


def next_float(value, bits=64, direction=1):
    """Next representable number; -0 and +0 compare as one numeric value."""
    binary_format(bits)
    if direction not in (-1, 1):
        raise FloatDomainError("next-float direction must be -1 or 1")
    value = quantize(value, bits)
    if math.isnan(value):
        return value
    if value == math.inf and direction > 0 or value == -math.inf and direction < 0:
        return value
    if value == 0:
        return math.copysign(binary_format(bits).min_subnormal, direction)
    pack, word = ("f", "I") if bits == 32 else ("d", "Q")
    encoded = struct.unpack(">" + word, struct.pack(">" + pack, value))[0]
    encoded += direction if value > 0 else -direction
    return struct.unpack(">" + pack, struct.pack(">" + word, encoded))[0]


def _merge(intervals, bits):
    result = []
    normalized = []
    for lo, hi in intervals:
        if not is_representable(lo, bits) or not is_representable(hi, bits):
            raise FloatDomainError("interval endpoint is outside its floating format")
        normalized.append((quantize(lo, bits), quantize(hi, bits)))
    for lo, hi in sorted(normalized):
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi or lo <= 0 <= hi:
            raise FloatDomainError("internal finite intervals must exclude zero")
        if not is_representable(lo, bits) or not is_representable(hi, bits):
            raise FloatDomainError("interval endpoint is outside its floating format")
        if result and (result[-1][1] < 0) == (lo < 0) and next_float(result[-1][1], bits) >= lo:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    if len(result) > 4096:
        raise FloatDomainError("floating domain exceeded the interval budget")
    return tuple(result)


def _integer_round(value, mode):
    value = _fraction(value)
    if mode == "floor":
        return value.numerator // value.denominator
    if mode == "ceil":
        return -((-value.numerator) // value.denominator)
    if mode == "trunc":
        return int(value)
    if mode == "round":
        # Explicit integer 'round' uses C round/roundf semantics: ties away.
        sign = -1 if value < 0 else 1
        value = abs(value)
        return sign * ((2 * value.numerator + value.denominator) // (2 * value.denominator))
    raise FloatConversionError("float-to-integer conversion requires trunc, floor, ceil or round")


@dataclass(frozen=True)
class IntegerConversion:
    domain: Domain
    trap_possible: bool = False

    @property
    def definitely_traps(self):
        return self.trap_possible and not self.domain



@dataclass(frozen=True)
class FloatPiece:
    """Finite interval, either integer-only or excluding an integer value set.

    Keeping integer predicates in mathematical coordinates avoids enumerating
    up to 2**64 integers and preserves exact complements containing fractions.
    """
    lo: float
    hi: float
    integer: Domain | None = None
    excluded: Domain = Domain()


def representable_integers(domain, bits=64):
    """Exact integer subset representable in IEEE format, in exponent bands."""
    fmt = binary_format(bits)
    maximum = int(fmt.max_finite)
    domain = domain & Domain.of((-maximum, maximum))
    result = domain & Domain.of((-(1 << fmt.precision), 1 << fmt.precision))
    if not domain:
        return result
    extent = max(abs(domain.lo), abs(domain.hi))
    for exponent in range(fmt.precision, min(fmt.emax, extent.bit_length() - 1) + 1):
        low, high = (1 << exponent), min(maximum, (1 << (exponent+1)) - 1)
        band = domain & Domain.of((low, high), (-high, -low))
        if band:
            result |= band.with_congruence(0, 1 << (exponent-fmt.precision+1))
    return result


def _fractional_possible(lo, hi, bits):
    if not lo.is_integer() or not hi.is_integer():
        return True
    return (lo < hi and
            (not next_float(lo,bits).is_integer() or not next_float(hi,bits,-1).is_integer()))


def _normalize_pieces(pieces, bits):
    groups = {}
    for piece in pieces:
        lo,hi = quantize(piece.lo,bits),quantize(piece.hi,bits)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi or lo <= 0 <= hi:
            raise FloatDomainError("invalid finite predicate interval")
        integers = Domain.of((math.ceil(lo),math.floor(hi)))
        integer,excluded = piece.integer,piece.excluded
        if integer is not None:
            integer = representable_integers(integer & integers,bits) - excluded
            if not integer:
                continue
            lo,hi = quantize(integer.lo,bits,"up"),quantize(integer.hi,bits,"down")
            excluded = Domain()
        else:
            excluded = representable_integers(excluded & integers,bits)
            if excluded and not _fractional_possible(lo,hi,bits) and not (representable_integers(integers,bits) - excluded):
                continue
        groups.setdefault((integer,excluded),[]).append((lo,hi))
    result = tuple(FloatPiece(lo,hi,integer,excluded)
                   for (integer,excluded),intervals in groups.items()
                   for lo,hi in _merge(intervals,bits))
    if len(result) > 4096:
        raise FloatDomainError("floating predicate domain exceeded the interval budget")
    return tuple(sorted(result,key=lambda p:(p.lo,p.hi,str(p.integer),str(p.excluded))))


def _piece_intersection(a,b,lo,hi):
    if a.integer is None and b.integer is None:
        return FloatPiece(lo,hi,excluded=a.excluded | b.excluded)
    if a.integer is None:
        return FloatPiece(lo,hi,b.integer-a.excluded)
    if b.integer is None:
        return FloatPiece(lo,hi,a.integer-b.excluded)
    return FloatPiece(lo,hi,a.integer & b.integer)


def _piece_difference(a,b,lo,hi):
    if a.integer is None and b.integer is None:
        return FloatPiece(lo,hi,b.excluded-a.excluded)
    if a.integer is None:
        return FloatPiece(lo,hi,excluded=a.excluded | b.integer)
    if b.integer is None:
        return FloatPiece(lo,hi,a.integer & b.excluded)
    return FloatPiece(lo,hi,a.integer-b.integer)


@dataclass(frozen=True)
class FloatDomain:
    bits: int = 64
    intervals: tuple = ()
    zeros: int = 0
    pos_inf: bool = False
    neg_inf: bool = False
    nan: bool = False
    filters: tuple[FloatPiece, ...] = ()

    def __post_init__(self):
        binary_format(self.bits)
        if type(self.zeros) is not int or self.zeros & ~BOTH_ZEROS:
            raise FloatDomainError("invalid signed-zero set")
        if any(type(flag) is not bool for flag in (self.pos_inf, self.neg_inf, self.nan)):
            raise FloatDomainError("nonfinite flags must be boolean")
        intervals = _merge(self.intervals, self.bits)
        if self.filters:
            pieces = _normalize_pieces(self.filters,self.bits)
            intervals = _merge(((p.lo,p.hi) for p in pieces),self.bits)
            if all(p.integer is None and not p.excluded for p in pieces):
                pieces = ()
            object.__setattr__(self,"filters",pieces)
        object.__setattr__(self, "intervals", intervals)

    @property
    def pieces(self):
        return self.filters or tuple(FloatPiece(lo,hi) for lo,hi in self.intervals)

    def raw_bits(self):
        """Enclose the IEEE encodings, including both signs of NaN."""
        fmt = ">f" if self.bits == 32 else ">d"
        sign = 1 << (self.bits - 1)

        def word(value):
            return int.from_bytes(struct.pack(fmt, value), "big")

        ranges = []
        for lo, hi in self.intervals:
            first, last = word(lo), word(hi)
            ranges.append((min(first, last), max(first, last)))
        infinity = word(math.inf)
        if self.zeros & POS_ZERO:
            ranges.append((0, 0))
        if self.zeros & NEG_ZERO:
            ranges.append((sign, sign))
        if self.pos_inf:
            ranges.append((infinity, infinity))
        if self.neg_inf:
            ranges.append((sign + infinity, sign + infinity))
        if self.nan:
            ranges.extend(((infinity + 1, sign - 1), (sign + infinity + 1, 2 * sign - 1)))
        return Domain.of(*ranges)

    @classmethod
    def from_raw_bits(cls, domain, bits):
        """Decode unsigned IEEE encoding ranges without enumerating words."""
        binary_format(bits)
        fmt = ">f" if bits == 32 else ">d"
        sign = 1 << (bits - 1)
        infinity = int.from_bytes(struct.pack(fmt, math.inf), "big")

        def value(word):
            return struct.unpack(fmt, word.to_bytes(bits // 8, "big"))[0]

        intervals = []
        zeros = 0
        pos_inf = neg_inf = nan = False
        for lo, hi in domain.ranges:
            first, last = max(lo, 1), min(hi, infinity - 1)
            if first <= last:
                intervals.append((value(first), value(last)))
            first, last = max(lo, sign + 1), min(hi, sign + infinity - 1)
            if first <= last:
                intervals.append((value(last), value(first)))
            if lo <= 0 <= hi:
                zeros |= POS_ZERO
            if lo <= sign <= hi:
                zeros |= NEG_ZERO
            pos_inf |= lo <= infinity <= hi
            neg_inf |= lo <= sign + infinity <= hi
            nan |= (max(lo, infinity + 1) <= min(hi, sign - 1) or
                    max(lo, sign + infinity + 1) <= min(hi, 2 * sign - 1))
        return cls(bits, tuple(intervals), zeros, pos_inf, neg_inf, nan)

    @classmethod
    def integer_set(cls, domain, bits=64):
        domain = representable_integers(domain,bits)
        pieces = []
        for lo,hi in domain.ranges:
            for a,b in ((lo,min(hi,-1)),(max(lo,1),hi)):
                if a <= b:
                    pieces.append(FloatPiece(quantize(a,bits,"up"),quantize(b,bits,"down"),domain))
        return cls(bits,zeros=BOTH_ZEROS if Domain.point(0).within(domain) else 0,filters=tuple(pieces))

    @property
    def integer_only(self):
        return (not self.pos_inf and not self.neg_inf and not self.nan and
                all(p.integer is not None or not _fractional_possible(p.lo,p.hi,self.bits) for p in self.pieces))

    def exact_integer_values(self):
        result = Domain.point(0) if self.zeros else Domain()
        for p in self.pieces:
            values = Domain.of((math.ceil(p.lo),math.floor(p.hi)))
            if p.integer is not None:
                values &= p.integer
            result |= representable_integers(values,self.bits)-p.excluded
        return result

    @classmethod
    def point(cls, value, bits=64):
        value = quantize(value, bits)
        if math.isnan(value):
            return cls(bits, nan=True)
        if math.isinf(value):
            return cls(bits, pos_inf=value > 0, neg_inf=value < 0)
        if value == 0:
            return cls(bits, zeros=NEG_ZERO if _negative(value) else POS_ZERO)
        return cls(bits, ((value, value),))

    @classmethod
    def interval(cls, lo, hi, bits=64, *, nan=False):
        """Representable values inside a mathematical closed interval.

        Decimal bounds are exact.  Endpoints round inward; intervals that fall
        strictly between two adjacent floats are empty.  A numeric interval
        containing zero includes both signs, unlike point(-0.0).
        """
        fmt = binary_format(bits)
        low_special, high_special = _special(lo), _special(hi)
        if (low_special is not None and math.isnan(low_special) or
                high_special is not None and math.isnan(high_special)):
            raise FloatDomainError("NaN is not an ordered interval bound")
        lower = low_special if low_special is not None else _fraction(lo)
        upper = high_special if high_special is not None else _fraction(hi)
        if lower > upper:
            return cls(bits, nan=nan)
        negative_inf = lower == -math.inf
        positive_inf = upper == math.inf
        if lower == math.inf or upper == -math.inf:
            return cls(bits, pos_inf=lower == math.inf, neg_inf=upper == -math.inf, nan=nan)
        low = -fmt.max_finite if negative_inf else quantize(lower, bits, "up")
        high = fmt.max_finite if positive_inf else quantize(upper, bits, "down")
        intervals = []
        if low <= high:
            if low < 0 and low != math.inf:
                intervals.append((low, min(high, -fmt.min_subnormal)))
            if high > 0 and high != -math.inf:
                intervals.append((max(low, fmt.min_subnormal), high))
        zeros = BOTH_ZEROS if lower <= 0 <= upper else 0
        return cls(bits, tuple(intervals), zeros, positive_inf, negative_inf, nan)

    @classmethod
    def enclose(cls, lo, hi, bits=64):
        """Outward-rounded enclosure of an exact mathematical interval.

        Use this when a proof computes real bounds outside IEEE operations.
        binary() is tighter: it applies the specified RNE operation once to
        exact extrema, rather than accumulating host rounding errors.
        """
        lower = _special(lo) if _special(lo) is not None else _fraction(lo)
        upper = _special(hi) if _special(hi) is not None else _fraction(hi)
        if lower > upper:
            return cls(bits)
        return cls.interval(quantize(lo,bits,"down"),quantize(hi,bits,"up"),bits)

    @classmethod
    def top(cls, bits=64):
        return cls.interval(-math.inf, math.inf, bits, nan=True)

    @classmethod
    def finite(cls, bits=64):
        limit = binary_format(bits).max_finite
        return cls.interval(-limit, limit, bits)

    @classmethod
    def union(cls, domains, bits=None):
        domains = tuple(domains)
        bits = domains[0].bits if domains and bits is None else 64 if bits is None else bits
        result = cls(bits)
        for domain in domains:
            result = result | domain
        return result

    def _same_format(self, other):
        if not isinstance(other, FloatDomain) or self.bits != other.bits:
            raise FloatDomainError("floating domains must have the same format; convert explicitly")

    def __bool__(self):
        return bool(self.intervals or self.zeros or self.pos_inf or self.neg_inf or self.nan)

    @property
    def ranges(self):
        pieces = list(self.intervals)
        if self.zeros:
            pieces.append((0.0, 0.0))
        if self.neg_inf:
            pieces.append((-math.inf, -math.inf))
        if self.pos_inf:
            pieces.append((math.inf, math.inf))
        return tuple(sorted(pieces))

    @property
    def lo(self):
        if not self.ranges:
            raise FloatDomainError("empty or NaN-only domain has no ordered lower bound")
        return self.ranges[0][0]

    @property
    def hi(self):
        if not self.ranges:
            raise FloatDomainError("empty or NaN-only domain has no ordered upper bound")
        return self.ranges[-1][1]

    def contains(self, value):
        special = _special(value)
        if special is not None:
            return self.nan if math.isnan(special) else self.pos_inf if special > 0 else self.neg_inf
        exact = _fraction(value)
        if not exact:
            return bool(self.zeros & (NEG_ZERO if _negative(value) else POS_ZERO))
        if not is_representable(exact, self.bits):
            return False
        for p in self.pieces:
            if p.lo <= exact <= p.hi:
                integral = exact.denominator == 1
                if p.integer is not None:
                    if integral and Domain.point(int(exact)).within(p.integer):
                        return True
                elif not integral or not Domain.point(int(exact)).within(p.excluded):
                    return True
        return False

    def __or__(self, other):
        self._same_format(other)
        filters = self.pieces + other.pieces if self.filters or other.filters else ()
        return FloatDomain(self.bits, self.intervals + other.intervals,
                           self.zeros | other.zeros, self.pos_inf or other.pos_inf,
                           self.neg_inf or other.neg_inf, self.nan or other.nan, filters)

    def __and__(self, other):
        self._same_format(other)
        if self.filters or other.filters:
            pieces = tuple(_piece_intersection(a,b,max(a.lo,b.lo),min(a.hi,b.hi))
                           for a in self.pieces for b in other.pieces if max(a.lo,b.lo)<=min(a.hi,b.hi))
            return FloatDomain(self.bits,zeros=self.zeros & other.zeros,
                               pos_inf=self.pos_inf and other.pos_inf,neg_inf=self.neg_inf and other.neg_inf,
                               nan=self.nan and other.nan,filters=pieces)
        intervals = [(max(a,c), min(b,d)) for a,b in self.intervals for c,d in other.intervals
                     if max(a,c) <= min(b,d)]
        return FloatDomain(self.bits, tuple(intervals), self.zeros & other.zeros,
                           self.pos_inf and other.pos_inf, self.neg_inf and other.neg_inf,
                           self.nan and other.nan)

    def __sub__(self, other):
        self._same_format(other)
        if self.filters or other.filters:
            pieces = self.pieces
            for b in other.pieces:
                remaining = []
                for a in pieces:
                    lo,hi = max(a.lo,b.lo),min(a.hi,b.hi)
                    if lo>hi:
                        remaining.append(a)
                        continue
                    if a.lo<lo:
                        remaining.append(FloatPiece(a.lo,next_float(lo,self.bits,-1),a.integer,a.excluded))
                    remaining.append(_piece_difference(a,b,lo,hi))
                    if hi<a.hi:
                        remaining.append(FloatPiece(next_float(hi,self.bits),a.hi,a.integer,a.excluded))
                pieces = _normalize_pieces(remaining,self.bits)
            return FloatDomain(self.bits,zeros=self.zeros & ~other.zeros,
                               pos_inf=self.pos_inf and not other.pos_inf,neg_inf=self.neg_inf and not other.neg_inf,
                               nan=self.nan and not other.nan,filters=pieces)
        intervals = []
        for lo, hi in self.intervals:
            pieces = [(lo,hi)]
            for start, stop in other.intervals:
                remaining = []
                for a,b in pieces:
                    if b < start or a > stop:
                        remaining.append((a,b))
                    else:
                        if a < start:
                            remaining.append((a,next_float(start,self.bits,-1)))
                        if b > stop:
                            remaining.append((next_float(stop,self.bits,1),b))
                pieces = remaining
            intervals.extend(pieces)
        return FloatDomain(self.bits, tuple(intervals), self.zeros & ~other.zeros,
                           self.pos_inf and not other.pos_inf, self.neg_inf and not other.neg_inf,
                           self.nan and not other.nan)

    def within(self, other):
        return not (self - other)

    def without_nan(self):
        return FloatDomain(self.bits, self.intervals, self.zeros, self.pos_inf, self.neg_inf,filters=self.filters)

    def without_zero(self):
        return FloatDomain(self.bits, self.intervals, 0, self.pos_inf, self.neg_inf, self.nan,self.filters)

    def only_zero(self):
        return FloatDomain(self.bits, zeros=self.zeros)

    def negate(self):
        zeros = (POS_ZERO if self.zeros & NEG_ZERO else 0) | (NEG_ZERO if self.zeros & POS_ZERO else 0)
        filters = tuple(FloatPiece(-p.hi,-p.lo,None if p.integer is None else p.integer.affine(-1,0),
                                   p.excluded.affine(-1,0)) for p in self.filters)
        return FloatDomain(self.bits, tuple((-hi,-lo) for lo,hi in self.intervals), zeros,
                           self.neg_inf,self.pos_inf,self.nan,filters)

    def truth(self):
        # IEEE NaN != 0 is true, matching C numeric condition semantics.
        false = bool(self.zeros)
        true = bool(self.intervals or self.pos_inf or self.neg_inf or self.nan)
        return Domain.of(*([(0,0)] if false else []), *([(1,1)] if true else []))

    def compare(self, op, other, *, same_value=False):
        self._same_format(other)
        if op not in ("==", "!=", "<", "<=", ">", ">="):
            raise FloatDomainError("unknown floating comparison")
        if not self or not other:
            return Domain()
        possible = set()
        if self.nan or other.nan:
            possible.add(int(op == "!="))
        if same_value:
            if self.ranges and other.ranges:
                possible.add(int(op in ("==", "<=", ">=")))
        else:
            for lo,hi in self.ranges:
                for low,high in other.ranges:
                    if op in ("==", "!="):
                        equal = max(lo,low) <= min(hi,high)
                        unequal = lo != hi or low != high or lo != low
                        if equal:
                            possible.add(int(op == "=="))
                        if unequal:
                            possible.add(int(op == "!="))
                    else:
                        yes,no = {"<": (lo < high,hi >= low), "<=": (lo <= high,hi > low),
                                  ">": (hi > low,lo <= high), ">=": (hi >= low,lo < high)}[op]
                        if yes:
                            possible.add(1)
                        if no:
                            possible.add(0)
        return Domain.of(*((n,n) for n in possible))

    def _parts(self):
        yield from (("finite",lo,hi) for lo,hi in self.intervals)
        if self.zeros & POS_ZERO:
            yield "zero",0.0,0.0
        if self.zeros & NEG_ZERO:
            yield "zero",-0.0,-0.0
        if self.pos_inf:
            yield "inf",math.inf,math.inf
        if self.neg_inf:
            yield "inf",-math.inf,-math.inf

    @classmethod
    def _rounded_range(cls, lo, hi, bits, zero_sign=POS_ZERO):
        """Enclose RNE(exact x) for every real x in [lo,hi]."""
        low,high = quantize(lo,bits),quantize(hi,bits)
        result = cls.interval(low,high,bits)
        zeros = 0
        if low <= 0 <= high:
            if lo < 0:
                zeros |= NEG_ZERO
            if hi > 0:
                zeros |= POS_ZERO
            if lo <= 0 <= hi:
                zeros |= zero_sign
        return cls(bits,result.intervals,zeros,result.pos_inf,result.neg_inf)

    def binary(self, op, other):
        """Enclose IEEE arithmetic, including zero division and invalid NaN."""
        self._same_format(other)
        if op not in ("+","-","*","/"):
            raise FloatDomainError("unsupported floating arithmetic operator")
        if op == "-":
            return self.binary("+",other.negate())
        if not self or not other:
            return FloatDomain(self.bits)
        output = FloatDomain(self.bits,nan=self.nan or other.nan)
        for ak,a,b in self._parts():
            for bk,c,d in other._parts():
                negative = _negative(a) != _negative(c)
                if op == "+":
                    if ak == "inf" or bk == "inf":
                        result = (FloatDomain(self.bits,nan=True) if ak == bk == "inf" and a != c
                                  else FloatDomain.point(a if ak == "inf" else c,self.bits))
                    else:
                        sign = NEG_ZERO if ak == bk == "zero" and _negative(a) and _negative(c) else POS_ZERO
                        result = self._rounded_range(_fraction(a)+_fraction(c),_fraction(b)+_fraction(d),self.bits,sign)
                elif op == "*":
                    if ak == "inf" or bk == "inf":
                        result = (FloatDomain(self.bits,nan=True) if ak == "zero" or bk == "zero"
                                  else FloatDomain.point(-math.inf if negative else math.inf,self.bits))
                    elif ak == "zero" or bk == "zero":
                        result = FloatDomain.point(-0.0 if negative else 0.0,self.bits)
                    else:
                        values = [_fraction(x)*_fraction(y) for x in (a,b) for y in (c,d)]
                        result = self._rounded_range(min(values),max(values),self.bits)
                else:
                    if ak == bk == "inf" or ak == bk == "zero":
                        result = FloatDomain(self.bits,nan=True)
                    elif ak == "inf" or bk == "zero":
                        result = FloatDomain.point(-math.inf if negative else math.inf,self.bits)
                    elif bk == "inf" or ak == "zero":
                        result = FloatDomain.point(-0.0 if negative else 0.0,self.bits)
                    else:
                        values = [_fraction(x)/_fraction(y) for x in (a,b) for y in (c,d)]
                        result = self._rounded_range(min(values),max(values),self.bits)
                output = output | result
        if op in ("+","*") and self.integer_only and other.integer_only:
            finite = FloatDomain.integer_set(output.exact_integer_values(),self.bits)
            return FloatDomain(self.bits,finite.intervals,output.zeros,output.pos_inf,output.neg_inf,output.nan,finite.filters)
        return output

    @classmethod
    def from_integers(cls, domain, bits=64, *, exact=True):
        if exact and not integers_exact(domain,bits):
            raise FloatConversionError("integer domain is not proved exactly representable")
        if exact:
            result = cls.integer_set(domain,bits)
            return cls(bits,result.intervals,POS_ZERO if result.zeros else 0,filters=result.filters)
        result = cls(bits)
        for lo,hi in domain.ranges:
            if not isinstance(lo,int) or not isinstance(hi,int):
                raise FloatConversionError("integer-to-float source must have finite bounds")
            result = result | cls._rounded_range(Fraction(lo),Fraction(hi),bits)
        return result

    def convert(self, bits, strategy="exact"):
        """Change physical floating format; 'sat' clamps ±overflow/Infinity.

        NaNs remain NaNs.  Integer saturation is separate and traps NaN instead
        of ever feeding it to a C integer cast.
        """
        target = binary_format(bits)
        if strategy not in ("exact","round","sat"):
            raise FloatConversionError("floating conversion requires exact, round or sat")
        if bits == self.bits and strategy != "sat":
            return self
        if strategy == "exact" and self.integer_only:
            values = self.exact_integer_values()
            if not integers_exact(values,bits):
                raise FloatConversionError("floating narrowing is not proved exact")
            result = FloatDomain.integer_set(values,bits)
            return FloatDomain(bits,result.intervals,self.zeros,filters=result.filters)
        if strategy == "exact" and bits < self.bits:
            if any(lo != hi or not is_representable(lo,bits) for lo,hi in self.intervals):
                raise FloatConversionError("floating narrowing is not proved exact")
        result = FloatDomain(bits,zeros=self.zeros,nan=self.nan,
                             pos_inf=self.pos_inf and strategy != "sat",neg_inf=self.neg_inf and strategy != "sat")
        for lo,hi in self.intervals:
            if strategy == "sat":
                lo,hi = max(-target.max_finite,lo),min(target.max_finite,hi)
                if lo > hi:
                    lo = hi = target.max_finite if lo > 0 else -target.max_finite
            result = result | self._rounded_range(_fraction(lo),_fraction(hi),bits)
        if strategy == "sat":
            if self.pos_inf:
                result = result | FloatDomain.point(target.max_finite,bits)
            if self.neg_inf:
                result = result | FloatDomain.point(-target.max_finite,bits)
        if self.filters and bits >= self.bits and strategy != "sat":
            filters = tuple(FloatPiece(p.lo,p.hi,p.integer,p.excluded) for p in self.pieces)
            return FloatDomain(bits,result.intervals,result.zeros,result.pos_inf,result.neg_inf,result.nan,filters)
        if self.integer_only:
            finite = FloatDomain.integer_set(result.exact_integer_values(),bits)
            return FloatDomain(bits,finite.intervals,result.zeros,result.pos_inf,result.neg_inf,result.nan,finite.filters)
        return result

    def to_integers(self, target, *, rounding="trunc", overflow="reject"):
        """Checked explicit float-to-integer conversion, never an unchecked cast.

        'reject': prove the rounded image lies in target and excludes NaN/Inf.
        'trap': normal results are intersected with target; all other inputs trap.
        'sat': clamp rounded values to one contiguous target interval; Infinity
        clamps and NaN traps.  This documented NaN policy avoids C undefined
        behavior and must be mirrored by the emitter before casting.
        Explicit integer 'round' means ties away from zero, as C round/roundf.
        """
        if rounding not in ("trunc","floor","ceil","round"):
            raise FloatConversionError("explicit float-to-integer rounding strategy required")
        if overflow not in ("reject","trap","sat"):
            raise FloatConversionError("unknown float-to-integer overflow strategy")
        if not target or not isinstance(target.lo,int) or not isinstance(target.hi,int):
            raise FloatConversionError("integer target must be nonempty and finitely bounded")
        image = Domain.of(*((_integer_round(lo,rounding),_integer_round(hi,rounding))
                            for lo,hi in self.intervals), *((0,0),) if self.zeros else ())
        if self.integer_only:
            image = self.exact_integer_values()
        nonfinite = self.nan or self.pos_inf or self.neg_inf
        if overflow == "reject":
            if nonfinite or not image.within(target):
                raise FloatConversionError("float-to-integer conversion may be nonfinite or outside the target; use sat or trap")
            return IntegerConversion(image)
        if overflow == "trap":
            if not target.predicates:
                valid = self & conversion_input_domain(target,self.bits,rounding)
                normal = Domain.of(*((_integer_round(lo,rounding),_integer_round(hi,rounding))
                                     for lo,hi in valid.intervals), *((0,0),) if valid.zeros else ())
                return IntegerConversion(normal & target,bool(self - valid))
            return IntegerConversion(image & target, nonfinite or bool(image - target))
        if target.predicates or len(target.ranges) != 1:
            raise FloatConversionError("saturation requires one contiguous integer target interval")
        lo,hi = target.lo,target.hi
        spans = [(min(max(a,lo),hi),min(max(b,lo),hi)) for a,b in image.ranges]
        if self.pos_inf:
            spans.append((hi,hi))
        if self.neg_inf:
            spans.append((lo,lo))
        return IntegerConversion(Domain.of(*spans),self.nan)

    def to_dict(self):
        raw = {"bits":self.bits,"intervals":[[lo.hex(),hi.hex()] for lo,hi in self.intervals],
               "zeros":self.zeros,"pos_inf":self.pos_inf,"neg_inf":self.neg_inf,"nan":self.nan}
        if self.filters:
            raw["filters"] = [[p.lo.hex(),p.hi.hex(),None if p.integer is None else encode_domain(p.integer),
                               encode_domain(p.excluded)] for p in self.filters]
        return raw

    @classmethod
    def from_dict(cls, raw):
        required = {"bits","intervals","zeros","pos_inf","neg_inf","nan"}
        if not isinstance(raw,dict) or set(raw) not in (required,required | {"filters"}):
            raise FloatDomainError("invalid floating domain record")
        intervals = raw["intervals"]
        if not isinstance(intervals,list) or len(intervals) > 4096:
            raise FloatDomainError("invalid floating interval list")
        try:
            parsed = tuple((float.fromhex(lo),float.fromhex(hi)) for lo,hi in intervals)
        except (ValueError,TypeError,OverflowError) as exc:
            raise FloatDomainError("invalid floating interval encoding") from exc
        filters = raw.get("filters",[])
        if not isinstance(filters,list) or len(filters)>4096:
            raise FloatDomainError("invalid floating predicate list")
        try:
            pieces = tuple(FloatPiece(float.fromhex(lo),float.fromhex(hi),
                                      None if integer is None else decode_domain(integer),decode_domain(excluded))
                           for lo,hi,integer,excluded in filters)
        except (ValueError,TypeError,OverflowError,InterfaceError) as exc:
            raise FloatDomainError("invalid floating predicate encoding") from exc
        result = cls(raw["bits"],parsed,raw["zeros"],raw["pos_inf"],raw["neg_inf"],raw["nan"],pieces)
        if pieces and result.intervals != _merge(parsed,result.bits):
            raise FloatDomainError("floating predicates disagree with interval envelope")
        return result

    def __str__(self):
        parts = [f"{{{lo.hex()}}}" if lo == hi else f"[{lo.hex()}, {hi.hex()}]" for lo,hi in self.intervals]
        if self.filters:
            parts = [(f"[{p.lo.hex()}, {p.hi.hex()}]" +
                     (f" & integer({p.integer})" if p.integer is not None else f" minus integer({p.excluded})" if p.excluded else ""))
                     for p in self.filters]
        if self.zeros & NEG_ZERO:
            parts.append("{-0.0}")
        if self.zeros & POS_ZERO:
            parts.append("{+0.0}")
        if self.neg_inf:
            parts.append("{-Infinity}")
        if self.pos_inf:
            parts.append("{+Infinity}")
        if self.nan:
            parts.append("{NaN}")
        return " | ".join(parts) or "empty"


def integers_exact(domain, bits=64):
    """Sufficient proof that every integer is exactly representable.

    The contiguous exact prefix is handled directly.  Beyond it, each binary
    exponent band requires a multiple of its ULP.  Domain.congruence() permits
    proving sparse aligned integer sets, rather than rejecting all large ints.
    """
    fmt = binary_format(bits)
    if not domain:
        return True
    if not isinstance(domain.lo,int) or not isinstance(domain.hi,int):
        return False
    maximum = int(fmt.max_finite)
    if domain.lo < -maximum or domain.hi > maximum:
        return False
    prefix = 1 << fmt.precision
    outside = domain - Domain.of((-prefix,prefix))
    if not outside:
        return True
    top = max(abs(outside.lo),abs(outside.hi)).bit_length()
    for exponent in range(fmt.precision,top):
        spacing = 1 << (exponent - fmt.precision + 1)
        low,high = 1 << exponent,(1 << (exponent + 1)) - 1
        for start,stop in ((low,high),(-high,-low)):
            part = outside & Domain.of((start,stop))
            if not part:
                continue
            residue,modulus = part.congruence()
            if residue % spacing or modulus % spacing:
                return False
    return True


def conversion_input_domain(target, bits=64, rounding="trunc"):
    """Exact finite float inputs whose rounded integers lie in interval target.

    This is the domain to test BEFORE a C integer cast.  Its endpoints are
    representable floats, and never use an inaccurately rounded integer limit
    such as (double)UINT64_MAX.  Sparse polynomial targets require a separate
    integer membership check after a physical-container safety guard.
    """
    binary_format(bits)
    if target.predicates or rounding not in ("trunc","floor","ceil","round"):
        raise FloatConversionError("conversion input guard requires interval targets and an explicit rounding mode")
    result = FloatDomain(bits)
    for lower,upper in target.ranges:
        if not isinstance(lower,int) or not isinstance(upper,int):
            raise FloatConversionError("integer target must have finite bounds")
        if rounding == "trunc":
            lo,open_lo = (Fraction(lower),False) if lower > 0 else (Fraction(lower-1),True)
            hi,open_hi = (Fraction(upper),False) if upper < 0 else (Fraction(upper+1),True)
        elif rounding == "floor":
            lo,hi,open_lo,open_hi = Fraction(lower),Fraction(upper+1),False,True
        elif rounding == "ceil":
            lo,hi,open_lo,open_hi = Fraction(lower-1),Fraction(upper),True,False
        else:
            lo,hi = Fraction(2*lower-1,2),Fraction(2*upper+1,2)
            open_lo,open_hi = lower <= 0,upper >= 0
        first,last = quantize(lo,bits,"up"),quantize(hi,bits,"down")
        if open_lo and is_representable(lo,bits):
            first = next_float(first,bits,1)
        if open_hi and is_representable(hi,bits):
            last = next_float(last,bits,-1)
        if first <= last:
            result = result | (FloatDomain.interval(first,last,bits) & FloatDomain.finite(bits))
    return result


def c_literal(value, bits=64):
    value = quantize(value,bits)
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "(-INFINITY)" if value < 0 else "INFINITY"
    return value.hex() + ("f" if bits == 32 else "")
