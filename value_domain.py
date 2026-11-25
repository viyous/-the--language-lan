"""Integer sets with interval bounds and symbolic polynomial predicates."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, islice
from math import comb, gcd, inf, isqrt, lcm


class DomainComplexity(Exception):
    pass


ACTIVE_ORACLES = ContextVar("lan_intersection_oracles", default=None)
DEFAULT_SOLVER_STEPS = 2048
ACTIVE_SOLVER_STEPS = ContextVar("lan_solver_steps", default=DEFAULT_SOLVER_STEPS)


def supports_compact_polynomial(poly):
    return (len(poly.coefficients) <= 9 and poly.denominator.bit_length() <= 128
            and all(c.bit_length() <= 128 for c in poly.coefficients))


@dataclass(frozen=True)
class IntersectionRule:
    result: "Domain"
    location: str
    lower: int | float = -inf
    upper: int | float = inf


@dataclass
class IntersectionOracles:
    exact: dict
    polynomial_pairs: dict
    on_apply: object

    def exact_match(self, left, right):
        rule = self.exact.get(frozenset((left, right)))
        if rule is not None:
            self.on_apply(rule)
        return rule

    def rewrite(self, clause, used):
        required = frozenset(clause.require)
        for pair, rules in self.polynomial_pairs.items():
            if not pair <= required:
                continue
            for rule in rules:
                lo, hi = max(clause.lo, rule.lower), min(clause.hi, rule.upper)
                if rule in used or lo > hi:
                    continue
                self.on_apply(rule)
                remaining = tuple(sorted(required - pair))
                pieces = [
                    Clause(max(lo, c.lo), min(hi, c.hi),
                           remaining + c.require, clause.exclude + c.exclude)
                    for c in rule.result.clauses()
                ]
                if clause.lo < lo:
                    pieces.append(Clause(clause.lo, lo - 1, clause.require, clause.exclude))
                if hi < clause.hi:
                    pieces.append(Clause(hi + 1, clause.hi, clause.require, clause.exclude))
                return rule, pieces
        return None


def evaluate(coefficients, n):
    value = 0
    for coefficient in reversed(coefficients):
        value = value * n + coefficient
    return value


def first_at_least(coefficients, lo, hi, direction, target):
    """First index in a monotone segment with direction * f(n) >= target."""
    if target == -inf:
        return lo
    if direction * evaluate(coefficients, lo) >= target:
        return lo
    if hi is None:
        # Beyond the Cauchy bound for direction*f(n)-target, its sign
        # is the sign of the leading coefficient (positive on this tail).
        rest = max(abs(coefficients[0] - direction * target),
                   *(abs(c) for c in coefficients[1:-1]), 0)
        hi = max(lo, 2 + rest // abs(coefficients[-1]))
    if direction * evaluate(coefficients, hi) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if direction * evaluate(coefficients, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def sign_ranges(coefficients):
    """Disjoint ranges of natural-number indices with constant sign."""
    if len(coefficients) == 1:
        n = coefficients[0]
        return ((0, None, (n > 0) - (n < 0)),)
    result = []
    next_index = 0
    for lo, hi, direction in monotone_segments(coefficients):
        lo = max(lo, next_index)
        if hi is not None and lo > hi:
            continue
        if not direction:
            value = evaluate(coefficients, lo)
            pieces = [(lo, hi, (value > 0) - (value < 0))]
        else:
            start_zero = first_at_least(coefficients, lo, hi, direction, 0)
            start_pos = first_at_least(coefficients, lo, hi, direction, 1)
            pieces = []
            if start_zero is None:
                pieces.append((lo, hi, -direction))
            else:
                if start_zero > lo:
                    pieces.append((lo, start_zero - 1, -direction))
                if start_pos is None or start_zero < start_pos:
                    pieces.append((start_zero, hi if start_pos is None else start_pos - 1, 0))
                if start_pos is not None:
                    pieces.append((start_pos, hi, direction))
        for start, end, sign in pieces:
            if result and result[-1][2] == sign:
                result[-1] = result[-1][0], end, sign
            else:
                result.append((start, end, sign))
        if hi is None:
            break
        next_index = hi + 1
    return tuple(result)


@lru_cache(maxsize=256)
def monotone_segments(coefficients):
    nonconstant = coefficients[1:]
    if all(c >= 0 for c in nonconstant):
        return ((0, None, 1),)
    if all(c <= 0 for c in nonconstant):
        return ((0, None, -1),)
    # A positive (negative) forward difference means f(n+1) > (<) f(n).
    # Compute its signs recursively, reducing the degree by one each time.
    delta = tuple(sum(comb(j, i) * coefficients[j] for j in range(i + 1, len(coefficients)))
                  for i in range(len(coefficients) - 1))
    return tuple((lo, None if hi is None else hi + 1, sign)
                 for lo, hi, sign in sign_ranges(delta))


@lru_cache(maxsize=1024)
def polynomial_residues(coefficients, denominator, modulus):
    period = modulus * denominator
    if period > 1024:
        return frozenset(range(modulus))
    reduced = tuple(c % period for c in coefficients)
    return frozenset((evaluate(reduced, n) % period) // denominator for n in range(period))


@dataclass(frozen=True, order=True)
class Polynomial:
    # Numerator coefficients in ascending order. The quotient is integer-valued.
    coefficients: tuple[int, ...]
    denominator: int = 1

    @staticmethod
    def reduced(coefficients, denominator):
        divisor = gcd(*coefficients, denominator)
        return Polynomial(tuple(c // divisor for c in coefficients), denominator // divisor)

    @property
    def base(self):
        return self.coefficients[0] // self.denominator

    @property
    def modulus(self):
        return self.fixed_divisor(self.coefficients, self.denominator)

    @staticmethod
    @lru_cache(maxsize=1024)
    def fixed_divisor(coefficients, denominator):
        base = coefficients[0]
        # The first d+1 values determine all integer finite differences.
        return gcd(*((evaluate(coefficients, n) - base) // denominator
                     for n in range(1, len(coefficients))))

    def eval(self, n):
        return evaluate(self.coefficients, n) // self.denominator

    def windows(self, lower, upper):
        """Clip each monotone index segment to f(n) in [lower, upper]."""
        if len(self.coefficients) == 2:
            slope = self.coefficients[1] // self.denominator
            direction = 1 if slope > 0 else -1
            step, base = abs(slope), direction * self.base
            bottom, top = (lower, upper) if direction > 0 else (-upper, -lower)
            start = 0 if bottom == -inf else max(0, (bottom - base + step - 1) // step)
            end = None if top == inf else (top - base) // step
            if end is None or start <= end:
                yield start, end, direction
            return
        for lo, hi, direction in monotone_segments(self.coefficients):
            if not direction:
                if lower <= self.eval(lo) <= upper:
                    yield lo, lo, direction
                continue
            bottom, top = (lower, upper) if direction > 0 else (-upper, -lower)
            start = (lo if bottom == -inf else
                     first_at_least(self.coefficients, lo, hi, direction, bottom * self.denominator))
            if start is None:
                continue
            beyond = (None if top == inf else
                      first_at_least(self.coefficients, start, hi, direction, (top + 1) * self.denominator))
            end = hi if beyond is None else beyond - 1
            if end is None or start <= end:
                yield start, end, direction

    @lru_cache(maxsize=4096)
    def bounds(self, lower, upper):
        lo, hi = inf, -inf
        for start, end, direction in self.windows(lower, upper):
            first = self.eval(start)
            last = self.eval(end) if end is not None else (inf if direction > 0 else -inf)
            lo, hi = min(lo, first, last), max(hi, first, last)
        return lo, hi

    def contains(self, value):
        if (value - self.base) % self.modulus:
            return False
        if len(self.coefficients) == 2:
            return (value - self.base) // (self.coefficients[1] // self.denominator) >= 0
        return next(self.windows(value, value), None) is not None

    def rebase(self, start):
        return tuple(sum(comb(j, i) * self.coefficients[j] * start ** (j - i)
                         for j in range(i, len(self.coefficients)))
                     for i in range(len(self.coefficients)))

    def restrict_indices(self, start, step):
        return Polynomial.reduced(tuple(c * step ** i for i, c in enumerate(self.rebase(start))),
                                  self.denominator)

    def affine(self, scale, shift):
        terms = (self.coefficients[0] * scale + shift * self.denominator,
                 *(c * scale for c in self.coefficients[1:]))
        return Polynomial.reduced(terms, self.denominator)

    @lru_cache(maxsize=1024)
    def image_within(self, other):
        """Certify p(N) is contained in q(N) by an integer composition."""
        if self == other:
            return True
        degree = len(self.coefficients) - 1
        other_degree = len(other.coefficients) - 1
        if (degree > other_degree >= 2 and degree % other_degree == 0
                and supports_compact_polynomial(self) and supports_compact_polynomial(other)):
            from poly_intersection import natural_composition_inclusion

            return natural_composition_inclusion(other, self)
        if degree < 2 or len(other.coefficients) != len(self.coefficients):
            return False
        p, q = self.coefficients, other.coefficients
        ratio, rem = divmod(p[-1] * other.denominator, q[-1] * self.denominator)
        if rem or ratio < 1:
            return False
        lo, hi = 1, 1 << ((ratio.bit_length() + degree - 1) // degree)
        while lo < hi:
            mid = (lo + hi) // 2
            if mid ** degree < ratio:
                lo = mid + 1
            else:
                hi = mid
        stride = lo
        if stride ** degree != ratio:
            return False
        factor = stride ** (degree - 1) * self.denominator
        offset, rem = divmod(p[-2] * other.denominator - q[-2] * factor,
                             degree * q[-1] * factor)
        if rem or offset < 0:
            return False
        return all(c * stride ** i * self.denominator == p[i] * other.denominator
                   for i, c in enumerate(other.rebase(offset)))

    def __str__(self):
        terms = []
        for degree, coefficient in enumerate(self.coefficients):
            if not coefficient:
                continue
            if not degree:
                terms.append(str(coefficient))
                continue
            monomial = "n" if degree == 1 else "*".join(["n"] * degree)
            terms.append(monomial if coefficient == 1 else f"{coefficient}*{monomial}")
        text = " + ".join(terms).replace(" + -", " - ")
        return f"poly({text})" if self.denominator == 1 else f"poly(({text})/{self.denominator})"


@dataclass(frozen=True)
class Clause:
    lo: int | float
    hi: int | float
    require: tuple[Polynomial, ...] = ()
    exclude: tuple[Polynomial, ...] = ()


def merge_ranges(ranges):
    merged = []
    for lo, hi in sorted(ranges):
        if lo > hi or lo == inf or hi == -inf:
            continue
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return tuple(merged)


@lru_cache(maxsize=512)
def _bounded_polynomial_intersection(left, right, lo, hi, max_steps, max_values):
    from poly_intersection import PolynomialIntersection

    return PolynomialIntersection(left, right, lo, hi).solve_set(
        max_steps=max_steps, max_values=max_values)


def bounded_polynomial_intersection(left, right, lo, hi):
    steps = ACTIVE_SOLVER_STEPS.get()
    max_values = 128 if steps <= DEFAULT_SOLVER_STEPS else min(steps, 10_000)
    return _bounded_polynomial_intersection(left, right, lo, hi, steps, max_values)


bounded_polynomial_intersection.cache_clear = _bounded_polynomial_intersection.cache_clear
bounded_polynomial_intersection.cache_info = _bounded_polynomial_intersection.cache_info


def normalize(clause):
    lo, hi = clause.lo, clause.hi
    required, excluded = set(clause.require), set(clause.exclude)
    if lo > hi or lo == inf or hi == -inf or required & excluded:
        return None
    residue, modulus = 0, 1
    for poly in sorted(required):
        lo, hi = poly.bounds(lo, hi)
        if lo > hi:
            return None
        # Combine all necessary congruences using the generalized CRT.
        step = poly.modulus
        common = gcd(modulus, step)
        diff = poly.base - residue
        if diff % common:
            return None
        reduced_step = step // common
        multiplier = (diff // common) * pow(modulus // common, -1, reduced_step) % reduced_step
        residue += modulus * multiplier
        modulus *= reduced_step
        if len(poly.coefficients) == 2 and step == 1:
            required.remove(poly)  # Already enforced by the lower bound.
    if lo != -inf:
        lo += (residue - lo) % modulus
    if hi != inf:
        hi -= (hi - residue) % modulus
    if lo > hi:
        return None
    for poly in tuple(excluded):
        image_lo, image_hi = poly.bounds(-inf, inf)
        if hi < image_lo or lo > image_hi:
            excluded.remove(poly)
        elif any(p.image_within(poly) for p in required):
            return None
        elif len(poly.coefficients) == 2:
            step = poly.modulus
            common = gcd(modulus, step)
            if (residue - poly.base) % common:
                excluded.remove(poly)
            elif image_lo <= lo <= hi <= image_hi and modulus % step == 0 and (residue - poly.base) % step == 0:
                return None
    if len(required) > 1 and any(len(p.coefficients) > 2 for p in required):
        # Integer polynomials respect congruence: f(n + m) == f(n) (mod m).
        # Disjoint residue images certify emptiness, without enumerating values.
        for m in (8, 9, 5, 7, 11, 13, 16, 25, 27, 32, 64):
            common = gcd(modulus, m)
            possible = {r for r in range(m) if (r - residue) % common == 0}
            for poly in required:
                possible.intersection_update(polynomial_residues(poly.coefficients, poly.denominator, m))
            if not possible:
                return None
    if lo == hi:
        if all(p.contains(lo) for p in required) and not any(p.contains(lo) for p in excluded):
            return Clause(lo, hi)
        return None
    if required and (excluded or len(required) > 1) and any(len(p.coefficients) > 2 for p in required):
        # Exact search is cheap when at least one nonlinear image contributes
        # only a few points in this particular interval.
        for poly in sorted(required):
            if len(poly.coefficients) <= 2:
                continue
            windows = tuple(poly.windows(lo, hi))
            if any(end is None for _, end, _ in windows):
                continue
            if sum(end - start + 1 for start, end, _ in windows) > 32:
                continue
            witnesses = {poly.eval(n) for start, end, _ in windows for n in range(start, end + 1)}
            witnesses = {v for v in witnesses if all(p.contains(v) for p in required)
                         and not any(p.contains(v) for p in excluded)}
            if not witnesses:
                return None
            lo, hi = min(witnesses), max(witnesses)
            if lo == hi:
                return Clause(lo, hi)
            break
    ordered = tuple(sorted(required))
    if len(ordered) > 1 and -(1 << 63) <= lo <= hi < 1 << 64:
        candidates = [p for p in ordered
                      if 2 <= len(p.coefficients) <= 9
                      and p.denominator.bit_length() <= 64
                      and all(c.bit_length() <= 128 for c in p.coefficients)]
        nonlinear = next((p for p in candidates if len(p.coefficients) > 2), None)
        other = next((p for p in candidates if p != nonlinear), None)
        if nonlinear is not None and other is not None:
            scale = lcm(nonlinear.denominator, other.denominator)
            scaled_lo, scaled_hi = lo * scale, hi * scale
            if -(1 << 63) <= scaled_lo <= scaled_hi < 1 << 64:
                left_c = tuple(c * (scale // nonlinear.denominator) for c in nonlinear.coefficients)
                right_c = tuple(c * (scale // other.denominator) for c in other.coefficients)
                if all(c.bit_length() <= 128 for c in (*left_c, *right_c)):
                    left, right = sorted((left_c, right_c))
                    result = bounded_polynomial_intersection(left, right, scaled_lo, scaled_hi)
                    if result.kind == "empty":
                        return None
                    if result.kind == "explicit":
                        # The external solver uses Z. Lan uses natural generators.
                        values = [v // scale for v in result.values if v % scale == 0
                                  and all(p.contains(v // scale) for p in ordered)
                                  and not any(p.contains(v // scale) for p in excluded)]
                        if not values:
                            return None
                        if values[0] == values[-1]:
                            return Clause(values[0], values[0])
                        lo, hi = values[0], values[-1]
    return Clause(lo, hi, ordered, tuple(sorted(excluded)))


@lru_cache(maxsize=512)
def polynomial_residue_images(poly, residue, modulus):
    # An integer-valued polynomial with denominator d is periodic modulo m
    # on input classes modulo m*d, even when d is not invertible modulo m.
    period = modulus * poly.denominator
    if period > 256:
        return None
    residues = [r for r in range(period) if poly.eval(r) % modulus == residue]
    if len(residues) > 32:
        return None
    return tuple(poly.restrict_indices(r, period) for r in residues)


@lru_cache(maxsize=512)
def quadratic_intersection_branches(left, right):
    """Factor equal quadratic images when the completed-square norm is zero."""
    c, b, a = (coefficient * right.denominator for coefficient in left.coefficients)
    h, e, d = (coefficient * left.denominator for coefficient in right.coefficients)
    common = gcd(a, d)
    ra, rd = a // common, d // common
    product = ra * rd
    if product <= 0 or ra * (4 * d * h - e * e) != rd * (4 * a * c - b * b):
        return None
    root = isqrt(product)
    if root * root != product:
        return None
    branches = []
    multiplier = 2 * a * rd
    for sign in (1, -1):
        divisor = sign * root * 2 * d
        bias = sign * root * e - rd * b
        divisor_gcd = gcd(multiplier, divisor)
        if bias % divisor_gcd:
            continue
        period = abs(divisor) // divisor_gcd
        residue = (bias // divisor_gcd * pow(multiplier // divisor_gcd, -1, period)) % period
        y = (multiplier * residue - bias) // divisor
        dy = multiplier * period // divisor
        if dy > 0:
            start = residue + period * max(0, (-y + dy - 1) // dy)
            branches.append(left.restrict_indices(start, period))
        elif y >= 0:
            count = y // -dy + 1
            if count > 128:
                return None
            branches.extend(left.eval(residue + period * n) for n in range(count))
    return tuple(branches)


@lru_cache(maxsize=512)
def compact_polynomial_pair(left, right):
    if len(right.coefficients) == 2 or len(left.coefficients) == 2:
        line, poly = (right, left) if len(right.coefficients) == 2 else (left, right)
        return polynomial_residue_images(poly, line.base % line.modulus, line.modulus)
    from poly_intersection import natural_composition_slice

    for outer, target in ((left, right), (right, left)):
        composition = natural_composition_slice(outer, target, max_prefix=128)
        if composition is not None:
            start, exceptions = composition
            return (target.restrict_indices(start, 1), *exceptions)
    if len(left.coefficients) == len(right.coefficients) == 3:
        return quadratic_intersection_branches(left, right)
    return None


def compact_clause(clause):
    if len(clause.require) < 2:
        return (clause,)
    candidates = filter(supports_compact_polynomial, clause.require)
    for left, right in combinations(islice(candidates, 8), 2):
        if len(left.coefficients) == len(right.coefficients) == 2:
            step = lcm(left.modulus, right.modulus)
            # The enclosing clause has already checked CRT compatibility
            # and clipped to both one-sided affine images.
            start = (clause.lo, step) if clause.lo != -inf else (clause.hi, -step)
            images = (Polynomial(start),)
        else:
            images = compact_polynomial_pair(left, right)
        if images is None:
            continue
        remaining = tuple(p for p in clause.require if p not in (left, right))
        branches = (Clause(clause.lo, clause.hi, (*remaining, image), clause.exclude)
                    if isinstance(image, Polynomial) else Clause(image, image, remaining, clause.exclude)
                    for image in images if isinstance(image, Polynomial) or clause.lo <= image <= clause.hi)
        return tuple(item for branch in branches if (item := normalize(branch)) is not None)
    return (clause,)


def subtract_clause(left, right):
    lo, hi = max(left.lo, right.lo), min(left.hi, right.hi)
    if lo > hi:
        yield left
        return
    if left.lo < lo:
        yield Clause(left.lo, lo - 1, left.require, left.exclude)
    if hi < left.hi:
        yield Clause(hi + 1, left.hi, left.require, left.exclude)
    # Inside the overlap, at least one of the right-hand predicates must fail.
    for poly in right.require:
        yield Clause(lo, hi, left.require, (*left.exclude, poly))
    for poly in right.exclude:
        yield Clause(lo, hi, (*left.require, poly), left.exclude)


@dataclass(frozen=True)
class Domain:
    intervals: tuple = ()
    predicates: tuple[Clause, ...] = ()

    @staticmethod
    def of(*ranges):
        return Domain(merge_ranges(ranges))

    @staticmethod
    def point(n):
        return Domain(((int(n), int(n)),))

    @staticmethod
    def from_clauses(clauses):
        groups = {}
        oracles = ACTIVE_ORACLES.get()
        if oracles is not None and oracles.polynomial_pairs:
            clauses = Domain.rewrite_clauses(clauses, oracles)
        for count, clause in enumerate(clauses, 1):
            if count > 100_000:
                raise DomainComplexity("polynomial value set exceeded the predicate budget")
            item = normalize(clause)
            if item is not None:
                pieces = compact_clause(item)
                if len(groups) + len(pieces) > 512 or any(
                        len(groups.get((piece.require, piece.exclude), ())) + len(pieces) > 2048
                        for piece in pieces):
                    pieces = (item,)
                for piece in pieces:
                    group = groups.setdefault((piece.require, piece.exclude), set())
                    group.add((piece.lo, piece.hi))
                    if len(groups) > 512 or len(group) > 2048:
                        raise DomainComplexity("polynomial value set exceeded the predicate budget")
        intervals = merge_ranges(groups.pop(((), ()), ()))
        predicates = tuple(Clause(lo, hi, req, exc) for (req, exc), spans in sorted(groups.items())
                           for lo, hi in merge_ranges(spans))
        if len(predicates) > 2048:
            raise DomainComplexity("polynomial value set exceeded the predicate budget")
        return Domain(intervals, predicates)

    @staticmethod
    def rewrite_clauses(clauses, oracles):
        count = 0
        for original in clauses:
            pending = [(original, frozenset())]
            while pending:
                count += 1
                if count > 100_000:
                    raise DomainComplexity("oracle rewriting exceeded the predicate budget")
                clause, used = pending.pop()
                item = normalize(clause)
                if item is None:
                    continue
                rewrite = oracles.rewrite(item, used)
                if rewrite is None:
                    yield item
                else:
                    rule, pieces = rewrite
                    pending.extend((piece, used | {rule}) for piece in pieces)

    @staticmethod
    def union(domains):
        domains = tuple(domains)
        if not any(d.predicates for d in domains):
            return Domain.of(*(span for d in domains for span in d.intervals))
        return Domain.from_clauses(c for d in domains for c in d.clauses())

    def clauses(self):
        yield from (Clause(lo, hi) for lo, hi in self.intervals)
        yield from self.predicates

    @property
    def ranges(self):
        """An interval overapproximation for operations without a symbolic rule."""
        if not self.predicates:
            return self.intervals
        return merge_ranges((*self.intervals, *((c.lo, c.hi) for c in self.predicates)))

    @property
    def lo(self):
        return self.ranges[0][0]

    @property
    def hi(self):
        return self.ranges[-1][1]

    def __bool__(self):
        return bool(self.intervals or self.predicates)

    def __str__(self):
        parts = []
        for c in self.clauses():
            bounds = f"{{{c.lo}}}" if c.lo == c.hi else f"[{c.lo}, {c.hi}]"
            predicates = [str(p) for p in c.require] + [f"!{p}" for p in c.exclude]
            parts.append(" & ".join([bounds, *predicates]))
        return " | ".join(parts) or "empty"

    def __or__(self, other):
        return Domain.union((self, other))

    def __and__(self, other):
        oracles = ACTIVE_ORACLES.get()
        if oracles is not None and oracles.exact:
            rule = oracles.exact_match(self, other)
            if rule is not None:
                return rule.result
        if not self.predicates and not other.predicates:
            spans = []
            i = j = 0
            while i < len(self.intervals) and j < len(other.intervals):
                a, b = self.intervals[i]
                c, d = other.intervals[j]
                if max(a, c) <= min(b, d):
                    spans.append((max(a, c), min(b, d)))
                if b <= d:
                    i += 1
                if d <= b:
                    j += 1
            return Domain.of(*spans)
        return Domain.from_clauses(
            Clause(max(x.lo, y.lo), min(x.hi, y.hi), x.require + y.require, x.exclude + y.exclude)
            for x in self.clauses() for y in other.clauses()
        )

    def __sub__(self, other):
        if not self.predicates and not other.predicates:
            spans = []
            j = 0
            for a, b in self.intervals:
                while j < len(other.intervals):
                    c, d = other.intervals[j]
                    if d < a:
                        j += 1
                        continue
                    if c > b:
                        break
                    if a < c:
                        spans.append((a, c - 1))
                    a = max(a, d + 1)
                    if a > b:
                        break
                    j += 1
                if a <= b:
                    spans.append((a, b))
            return Domain.of(*spans)
        remaining = self
        work = 0
        for right in other.clauses():
            work += (len(remaining.intervals) + len(remaining.predicates)) * (
                2 + len(right.require) + len(right.exclude)
            )
            if work > 100_000:
                raise DomainComplexity("polynomial value set exceeded the predicate budget")
            remaining = Domain.from_clauses(
                piece for left in remaining.clauses() for piece in subtract_clause(left, right)
            )
            if not remaining:
                break
        return remaining

    def within(self, other):
        return not (self - other)

    def congruence(self):
        """A necessary residue and modulus; modulus zero means a singleton."""
        residue, modulus = None, 0
        for clause in self.clauses():
            if clause.lo == clause.hi:
                base, step = clause.lo, 0
            elif clause.require:
                step = lcm(*(p.modulus for p in clause.require))
                base = clause.lo if clause.lo != -inf else clause.hi
                if base == inf:
                    return 0, 1
            else:
                return 0, 1
            if residue is None:
                residue, modulus = base, step
            else:
                modulus = gcd(modulus, step, residue - base)
            if modulus == 1:
                return 0, 1
        return (0, 1) if residue is None else (residue, modulus)

    def with_congruence(self, residue, modulus):
        if not self or modulus <= 1 or self.lo == -inf:
            return self
        first = self.lo + (residue - self.lo) % modulus
        poly = Polynomial((first, modulus))
        return Domain.from_clauses(Clause(c.lo, c.hi, (*c.require, poly), c.exclude)
                                   for c in self.clauses())

    def bitset_difference(self, other):
        """Exact Minkowski difference for independent sets in a small span."""
        if self.predicates or other.predicates or not self.intervals or not other.intervals:
            return None
        a, b = self.intervals, other.intervals
        if len(a) * len(b) < 128:
            return None
        if self.lo == -inf or self.hi == inf or other.lo == -inf or other.hi == inf:
            return None
        width = self.hi - self.lo + other.hi - other.lo + 1
        if width > 8192:
            return None
        left = tuple((lo - self.lo, hi - self.lo) for lo, hi in a)
        right = tuple((other.hi - hi, other.hi - lo) for lo, hi in b)
        nleft = sum(hi - lo + 1 for lo, hi in left)
        nright = sum(hi - lo + 1 for lo, hi in right)
        shifts = min(nleft, nright)
        work = len(a) + len(b) + shifts * ((width + 63) // 64)
        if work >= len(a) * len(b):
            return None
        dense, sparse = (left, right) if nright <= nleft else (right, left)
        mask = 0
        for lo, hi in dense:
            mask |= ((1 << (hi - lo + 1)) - 1) << lo
        result = 0
        for lo, hi in sparse:
            for shift in range(lo, hi + 1):
                result |= mask << shift
        offset = self.lo - other.hi
        spans = []
        while result:
            zeros = (result & -result).bit_length() - 1
            result >>= zeros
            offset += zeros
            ones = (result ^ (result + 1)).bit_length() - 1
            spans.append((offset, offset + ones - 1))
            result >>= ones
            offset += ones
        return Domain(tuple(spans)), work + len(spans)

    def witness(self):
        """A verified member, or None if this bounded search found none."""
        for clause in self.clauses():
            candidates = set(range(-2, 3))
            if clause.lo != -inf:
                candidates.update(clause.lo + i for i in range(8))
            if clause.hi != inf:
                candidates.update(clause.hi - i for i in range(8))
            for poly in clause.require:
                for start, end, _ in poly.windows(clause.lo, clause.hi):
                    stop = start + 7 if end is None else min(start + 7, end)
                    candidates.update(poly.eval(n) for n in range(start, stop + 1))
                    if end is not None:
                        candidates.update(poly.eval(n) for n in range(max(start, end - 7), end + 1))
            for n in sorted(candidates):
                if (clause.lo <= n <= clause.hi and all(p.contains(n) for p in clause.require)
                        and not any(p.contains(n) for p in clause.exclude)):
                    return n
        return None

    def affine(self, scale, shift):
        if not scale:
            return Domain.point(shift)
        if scale == 1 and shift == 0:
            return self
        if not self.predicates and abs(scale) == 1:
            return Domain.of(*((lo + shift, hi + shift) if scale == 1 else (shift - hi, shift - lo)
                               for lo, hi in self.intervals))
        clauses = []
        for c in self.clauses():
            a, b = c.lo * scale + shift, c.hi * scale + shift
            lo, hi = min(a, b), max(a, b)
            required = tuple(p.affine(scale, shift) for p in c.require)
            if abs(scale) > 1 and not required:
                required = (Polynomial((lo, abs(scale))),)
            excluded = tuple(p.affine(scale, shift) for p in c.exclude)
            clauses.append(Clause(lo, hi, required, excluded))
        return Domain.from_clauses(clauses)
