"""Exact bounded intersections of integer polynomial images over Z."""

from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from math import gcd, isfinite, isqrt
from time import monotonic

from value_domain import Polynomial, evaluate, sign_ranges

INT64_MIN = -(1 << 63)
UINT64_MAX = (1 << 64) - 1
_MODULI = (16, 9, 5, 7, 11, 13, 17, 19, 25, 32, 31, 64)
_QUADRATIC_LIMIT = 20_000


class SearchLimitExceeded(RuntimeError):
    """The search budget was exhausted without deciding emptiness."""


@dataclass(frozen=True)
class Witness:
    value: int
    x: int
    y: int


@dataclass(frozen=True)
class IntersectionResult:
    status: str  # nonempty, empty, or unknown
    method: str
    steps: int
    witness: Witness | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Progression:
    first: int
    step: int
    count: int

    def contains(self, value):
        index, rem = divmod(value - self.first, self.step)
        return not rem and 0 <= index < self.count


@dataclass(frozen=True)
class PolynomialImage:
    coefficients: tuple[int, ...]
    lower: int
    upper: int

    def contains(self, value):
        return self.lower <= value <= self.upper and _root(self.coefficients, value) is not None


@dataclass(frozen=True)
class IntersectionSet:
    kind: str  # empty, explicit, progression, polynomial, unknown
    method: str
    steps: int
    values: tuple[int, ...] = ()
    progression: Progression | None = None
    images: tuple[PolynomialImage, ...] = ()
    reason: str | None = None

    def contains(self, value):
        if self.kind == "unknown":
            raise SearchLimitExceeded(self.reason)
        if type(value) is not int:
            raise TypeError("value must be an integer")
        if self.progression is not None:
            return self.progression.contains(value)
        index = bisect_left(self.values, value)
        return ((index < len(self.values) and self.values[index] == value)
                or any(image.contains(value) for image in self.images))


class _Budget:
    def __init__(self, steps, timeout, backend):
        if steps is not None and (type(steps) is not int or steps < 0):
            raise ValueError("max_steps must be a nonnegative integer or None")
        if timeout is not None and (not isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative or None")
        if backend not in ("builtin", "sympy"):
            raise ValueError("backend must be 'builtin' or 'sympy'")
        self.limit = steps
        self.end = None if timeout is None else monotonic() + timeout
        self.steps = 0
        self.method = "preprocess"
        # The external symbolic call cannot be preempted in this process.
        self.symbolic = backend == "sympy" and timeout is None

    def check(self):
        if self.end is not None and monotonic() >= self.end:
            raise SearchLimitExceeded("time limit exceeded")

    def step(self):
        self.check()
        if self.limit is not None and self.steps >= self.limit:
            raise SearchLimitExceeded("candidate limit exceeded")
        self.steps += 1


def _coefficients(values):
    result = tuple(values)
    if not result or any(type(v) is not int for v in result):
        raise TypeError("polynomial coefficients must be a nonempty sequence of integers")
    size = len(result)
    while size > 1 and result[size - 1] == 0:
        size -= 1
    return result[:size]


@lru_cache(maxsize=4096)
def _halves(coefficients):
    mirrored = tuple(-a if i % 2 else a for i, a in enumerate(coefficients))
    return Polynomial(coefficients), Polynomial(mirrored)


def _windows(coefficients, lower, upper):
    """Closed, bounded intervals of actual integer inputs, not values."""
    for sign, poly in zip((1, -1), _halves(coefficients)):
        for start, end, _ in poly.windows(lower, upper):
            if sign > 0:
                yield start, end
            elif end > 0:
                yield -end, -max(start, 1)


def _root(coefficients, value):
    if len(coefficients) == 1:
        return 0 if coefficients[0] == value else None
    if len(coefficients) == 2:
        q, r = divmod(value - coefficients[0], coefficients[1])
        return q if not r else None
    if len(coefficients) == 3:
        c, b, a = coefficients
        discriminant = b*b - 4*a*(c-value)
        if discriminant < 0:
            return None
        d = isqrt(discriminant)
        if d*d != discriminant:
            return None
        for numerator in (-b+d, -b-d):
            q, r = divmod(numerator, 2*a)
            if not r:
                return q
        return None
    window = next(_windows(coefficients, value, value), None)
    return None if window is None else window[0]


def _linear_congruence(a, b, modulus):
    """a*x = b modulo modulus -> (residue, period), or no solution."""
    divisor = gcd(a, modulus)
    if b % divisor:
        return None
    period = modulus // divisor
    return (b // divisor * pow(a // divisor, -1, period)) % period, period


def _compose(outer, inner):
    result = (outer[-1],)
    for coefficient in reversed(outer[:-1]):
        terms = [0] * (len(result) + len(inner) - 1)
        for i, a in enumerate(result):
            if a:
                for j, b in enumerate(inner):
                    terms[i+j] += a*b
        terms[0] += coefficient
        result = tuple(terms)
    return result


def _compositions(outer, target, budget):
    """Recover the integer polynomial inners that satisfy target = outer(inner)."""
    m, n = len(outer)-1, len(target)-1
    if n % m or n < m or n > 64:
        return
    ratio, rem = divmod(target[-1], outer[-1])
    if rem or not ratio or (ratio < 0 and m % 2 == 0):
        return
    absolute = abs(ratio)
    low, high = 1, 1 << ((absolute.bit_length()+m-1)//m)
    while low < high:
        budget.check()
        mid = (low+high)//2
        if mid**m < absolute:
            low = mid+1
        else:
            high = mid
    if low**m != absolute:
        return
    leads = (low, -low) if m % 2 == 0 else ((low,) if ratio > 0 else (-low,))
    degree = n//m
    for lead in leads:
        inner = [0]*degree + [lead]
        factor = m*outer[-1]*lead**(m-1)
        for k in range(degree-1, -1, -1):
            budget.check()
            index = (m-1)*degree+k
            missing = target[index]-_compose(outer, inner)[index]
            inner[k], rem = divmod(missing, factor)
            if rem:
                break
        else:
            if _compose(outer, inner) == target:
                yield tuple(inner)


def _composition_inclusion(outer, target, budget):
    return next(_compositions(outer, target, budget), None) is not None


def _integer_compositions(outer, target):
    if len(outer.coefficients) < 2 or len(target.coefficients) < 2:
        return ()
    outer_numerator = tuple(c * target.denominator for c in outer.coefficients)
    target_numerator = tuple(c * outer.denominator for c in target.coefficients)
    budget = _Budget(None, None, "builtin")
    return _compositions(outer_numerator, target_numerator, budget)


def natural_composition_inclusion(outer, target):
    """Certify target(N) is contained in outer(N) by a nonnegative inner."""
    return any(all(sign >= 0 for _, _, sign in sign_ranges(inner))
               for inner in _integer_compositions(outer, target))


def natural_composition_slice(outer, target, max_prefix):
    """Give an exact target-generator tail and surviving exceptional values."""
    best_start = None
    for inner in _integer_compositions(outer, target):
        start = 0
        for _, end, sign in sign_ranges(inner):
            if sign < 0:
                if end is None or end >= max_prefix:
                    break
                start = end + 1
        else:
            if best_start is None or start < best_start:
                best_start = start
    if best_start is None:
        return None
    prefix = {target.eval(n) for n in range(best_start)}
    return best_start, tuple(v for v in sorted(prefix) if outer.contains(v))


@lru_cache(maxsize=4096)
def _mod_values(coefficients, modulus):
    reduced = tuple(c % modulus for c in coefficients)
    return tuple(evaluate(reduced, x) % modulus for x in range(modulus))


def _pell_unit(d, budget):
    a0 = isqrt(d)
    m, divisor, a = 0, 1, a0
    p0, p, q0, q = 1, a, 0, 1
    limit = _QUADRATIC_LIMIT if budget.limit is None else max(_QUADRATIC_LIMIT, budget.limit)
    # Beyond this size the bounded seed route cannot be cheap.
    while p <= limit:
        budget.check()
        if p*p-d*q*q == 1:
            return p, q
        m = divisor*a-m
        divisor = (d-m*m)//divisor
        a = (a0+m)//divisor
        p0, p = p, a*p+p0
        q0, q = q, a*q+q0
    return None


@lru_cache(maxsize=2048)
def _symbolic_pell(d, n):
    from sympy.solvers.diophantine.diophantine import diop_DN

    u, v = map(int, diop_DN(d, 1)[0])
    seeds = set()
    for x, y in diop_DN(d, n):
        X, Y = abs(int(x)), abs(int(y))
        while True:
            left, right = u*X-d*v*Y, u*Y-v*X
            if left < 0 or right < 0:
                break
            X, Y = left, right
        seeds.add((X, Y))
    return u, v, tuple(sorted(seeds))


@dataclass(frozen=True)
class PolynomialIntersection:
    """Images of f and g on all integers, restricted to an inclusive interval.

    Pass coefficients in ascending order. For example, (1, 0, 2) is 1+2*x*x.
    """

    f: tuple[int, ...]
    g: tuple[int, ...]
    lower: int = INT64_MIN
    upper: int = UINT64_MAX

    def __post_init__(self):
        object.__setattr__(self, "f", _coefficients(self.f))
        object.__setattr__(self, "g", _coefficients(self.g))
        if type(self.lower) is not int or type(self.upper) is not int:
            raise TypeError("bounds must be integers")
        if not all(INT64_MIN <= n <= UINT64_MAX for n in (self.lower, self.upper)):
            raise ValueError("bounds must be inside [INT64_MIN, UINT64_MAX]")

    def witness(self, value):
        if type(value) is not int:
            raise TypeError("value must be an integer")
        if not self.lower <= value <= self.upper:
            return None
        x = _root(self.f, value)
        if x is not None:
            y = _root(self.g, value)
            if y is not None:
                return Witness(value, x, y)
        return None

    def contains(self, value):
        return self.witness(value) is not None

    def find(self, *, max_steps=None, timeout=None, backend="builtin"):
        """Decide exactly by default. With a budget, exhaustion is 'unknown'."""
        budget = _Budget(max_steps, timeout, backend)
        try:
            found = self._find(budget)
        except SearchLimitExceeded as exc:
            return IntersectionResult("unknown", budget.method, budget.steps, reason=str(exc))
        return IntersectionResult("empty" if found is None else "nonempty",
                                  budget.method, budget.steps, found)

    def is_empty(self, *, max_steps=None, timeout=None, backend="builtin"):
        result = self.find(max_steps=max_steps, timeout=timeout, backend=backend)
        if result.status == "unknown":
            raise SearchLimitExceeded(f"{result.method}: {result.reason} after {result.steps} steps")
        return result.status == "empty"

    def solve_set(self, *, max_steps=None, timeout=None, max_values=10_000, backend="builtin"):
        """Return an exact compact or explicit set, or unknown without partial values."""
        if max_values is not None and (type(max_values) is not int or max_values < 0):
            raise ValueError("max_values must be a nonnegative integer or None")
        budget = _Budget(max_steps, timeout, backend)
        values, images = set(), set()

        def collect(hit):
            if hit is not None:
                values.add(hit.value)
                if max_values is not None and len(values) > max_values:
                    raise SearchLimitExceeded("explicit output limit exceeded")

        def add_image(coefficients, lower, upper):
            images.add(PolynomialImage(tuple(coefficients), lower, upper))

        try:
            lo, hi, f, g = self.lower, self.upper, self.f, self.g
            if lo > hi:
                return IntersectionSet("empty", "bounds", 0)
            budget.check()
            if len(f) == len(g) == 2:
                budget.method = "crt"
                progression = _linear_congruence(f[1], g[0]-f[0], abs(g[1]))
                if progression is not None:
                    x, period = progression
                    value, step = evaluate(f, x), abs(f[1])*period
                    first = lo + (value-lo) % step
                    if first <= hi:
                        return IntersectionSet("progression", budget.method, 0,
                                               progression=Progression(first, step, (hi-first)//step+1))
            elif lo < hi and len(f) > 1 and len(g) > 1:
                budget.method = "composition"
                selected = None
                if f == g or f == _halves(g)[1].coefficients or _composition_inclusion(f, g, budget):
                    selected = g
                elif _composition_inclusion(g, f, budget):
                    selected = f
                if selected is not None:
                    if next(_windows(selected, lo, hi), None) is not None:
                        add_image(selected, lo, hi)
                else:
                    collect(self._find(budget, collect, add_image))
            else:
                collect(self._find(budget, collect, add_image))
        except SearchLimitExceeded as exc:
            return IntersectionSet("unknown", budget.method, budget.steps, reason=str(exc))
        if images:
            ordered = tuple(sorted(images, key=lambda image: (image.coefficients, image.lower, image.upper)))
            return IntersectionSet("polynomial", budget.method, budget.steps, images=ordered)
        return IntersectionSet("explicit" if values else "empty", budget.method, budget.steps,
                               values=tuple(sorted(values)))

    def _find(self, budget, collect=None, add_image=None):
        lo, hi = self.lower, self.upper
        f, g = self.f, self.g
        if lo > hi:
            budget.method = "bounds"
            return None
        budget.check()
        if len(f) == 1 or len(g) == 1 or lo == hi:
            budget.method = "point"
            return self.witness(lo if lo == hi else f[0] if len(f) == 1 else g[0])
        if len(f) == len(g) == 2:
            budget.method = "crt"
            progression = _linear_congruence(f[1], g[0]-f[0], abs(g[1]))
            if progression is None:
                return None
            x, period = progression
            value, step = evaluate(f, x), abs(f[1])*period
            first = lo + (value-lo) % step
            return self.witness(first)
        fw, gw = tuple(_windows(f, lo, hi)), tuple(_windows(g, lo, hi))
        if not fw or not gw:
            budget.method = "bounds"
            return None
        if f == g or f == _halves(g)[1].coefficients:
            budget.method = "identical-image"
            return self.witness(evaluate(f, fw[0][0]))
        f_values = [evaluate(f, n) for span in fw for n in span]
        g_values = [evaluate(g, n) for span in gw for n in span]
        low, high = max(min(f_values), min(g_values)), min(max(f_values), max(g_values))
        if low > high:
            budget.method = "bounds"
            return None
        fmod, gmod = _halves(f)[0].modulus, _halves(g)[0].modulus
        common = _linear_congruence(fmod, g[0]-f[0], gmod)
        if common is None or low+(f[0]+fmod*common[0]-low) % (fmod*common[1]) > high:
            budget.method = "fixed-divisor"
            return None
        if (low, high) != (lo, hi):
            fw, gw = tuple(_windows(f, low, high)), tuple(_windows(g, low, high))
            if not fw or not gw:
                budget.method = "bounds"
                return None

        budget.method = "modular-obstruction"
        filters = []
        for m in _MODULI:
            budget.check()
            fm, gm = _mod_values(f, m), _mod_values(g, m)
            shared = set(fm).intersection(gm)
            if not shared:
                return None
            filters.append((m, frozenset(i for i, v in enumerate(fm) if v in shared),
                            frozenset(i for i, v in enumerate(gm) if v in shared)))

        if len(f) == 2 or len(g) == 2:
            line, curve, windows, swapped = (f, g, gw, False) if len(f) == 2 else (g, f, fw, True)
            modulus = abs(line[1])
            if modulus <= 4096:
                budget.method = "linear-residues"
                for index, value in enumerate(_mod_values(curve, modulus)):
                    if value != line[0] % modulus:
                        continue
                    for start, end in windows:
                        n = start + (index-start) % modulus
                        if n <= end:
                            if add_image is not None:
                                add_image(_compose(curve, (index, modulus)), low, high)
                                break
                            v = evaluate(curve, n)
                            t = (v-line[0])//line[1]
                            return Witness(v, n, t) if swapped else Witness(v, t, n)
                return None
        if len(f) == len(g) == 3:
            decided, witness = self._quadratics(budget, fw, low, high, collect, add_image)
            if decided:
                return witness
        return self._scan(budget, fw, gw, filters, collect)

    def _quadratics(self, budget, fw, lo, hi, collect=None, add_image=None):
        c, b, a = self.f
        h, e, d = self.g
        # Divide the quadratic equation by gcd(a,d) before multiplying by
        # d/gcd(a,d). This avoids an unnecessary square factor in D and N.
        common = gcd(a, d)
        reduced_a, reduced_d = a//common, d//common
        discriminant = reduced_a*reduced_d
        norm = reduced_d*(reduced_a*(4*d*h-e*e)-reduced_d*(4*a*c-b*b))
        xmax = abs(reduced_d)*isqrt(max(4*a*lo, 4*a*hi)-4*a*c+b*b)
        ymax = isqrt(max(4*d*lo, 4*d*hi)-4*d*h+e*e)

        def recover(X, Y):
            for u in (X, -X) if X else (0,):
                for v in (Y, -Y) if Y else (0,):
                    x, rx = divmod(u-reduced_d*b, 2*a*reduced_d)
                    y, ry = divmod(v-e, 2*d)
                    if not rx and not ry:
                        value = evaluate(self.f, x)
                        if lo <= value <= hi and evaluate(self.g, y) == value:
                            return Witness(value, x, y)
            return None

        quad_limit = _QUADRATIC_LIMIT if budget.limit is None else max(_QUADRATIC_LIMIT, budget.limit)
        if discriminant < 0:
            budget.method = "quadratic-ellipse"
            if norm < 0:
                return True, None
            radius = min(ymax, isqrt(norm//(-discriminant)))
            if radius > quad_limit:
                return False, None
            for Y in range(radius+1):
                budget.step()
                x2 = norm+discriminant*Y*Y
                X = isqrt(x2)
                if X <= xmax and X*X == x2 and (hit := recover(X, Y)) is not None:
                    if collect is None:
                        return True, hit
                    collect(hit)
            return True, None

        square = isqrt(discriminant)
        if square*square == discriminant:
            budget.method = "quadratic-factor"
            if norm == 0:
                for sign in (1, -1):
                    progress = _linear_congruence(2*a*reduced_d, sign*square*e-reduced_d*b, abs(2*d*square))
                    if progress is None:
                        continue
                    residue, period = progress
                    for start, end in fw:
                        x = start+(residue-start) % period
                        if x <= end:
                            if add_image is not None:
                                add_image(_compose(self.f, (residue, period)), lo, hi)
                                break
                            Y = sign*(reduced_d*(2*a*x+b)//square)
                            y = (Y-e)//(2*d)
                            return True, Witness(evaluate(self.f, x), x, y)
                return True, None
            radius = isqrt(abs(norm))
            if radius <= (512 if budget.symbolic else quad_limit):
                candidates = range(1, radius+1)
            elif budget.symbolic and abs(norm) <= 10**12:
                from sympy import divisors

                budget.method = "quadratic-factor-sympy"
                candidates = (int(p) for p in divisors(abs(norm)) if p <= radius)
            else:
                return False, None
            for p in candidates:
                budget.step()
                q, rem = divmod(norm, p)
                if rem or (p+q) % 2 or (q-p) % (2*square):
                    continue
                X, Y = abs((p+q)//2), abs((q-p)//(2*square))
                if X <= xmax and Y <= ymax and (hit := recover(X, Y)) is not None:
                    if collect is None:
                        return True, hit
                    collect(hit)
            return True, None

        budget.method = "quadratic-pell"
        if norm == 0:
            return True, recover(0, 0)
        unit = _pell_unit(discriminant, budget)
        radius = None
        if unit is not None:
            u, v = unit
            # Every nonnegative solution descends under the inverse unit to
            # a seed with X+Y*sqrt(D) < unit*sqrt(abs(norm)).
            radius = (u+v*(square+1)+1)*(isqrt(abs(norm))+1)//(2*square)
            radius = min(ymax, radius)

        def builtin_seeds():
            for Y in range(radius+1):
                budget.step()
                x2 = norm+discriminant*Y*Y
                if x2 < 0:
                    continue
                X = isqrt(x2)
                if X*X == x2 and (u*X-discriminant*v*Y < 0 or u*Y-v*X < 0):
                    yield X, Y

        if radius is not None and radius <= (512 if budget.symbolic else quad_limit):
            seeds = builtin_seeds()
        elif budget.symbolic and discriminant <= 8192 and abs(norm) <= 10**12:
            budget.method = "quadratic-pell-sympy"
            u, v, seeds = _symbolic_pell(discriminant, norm)
        else:
            return False, None
        for X, Y in seeds:
            while X <= xmax and Y <= ymax:
                budget.step()
                if (hit := recover(X, Y)) is not None:
                    if collect is None:
                        return True, hit
                    collect(hit)
                X, Y = u*X+discriminant*v*Y, v*X+u*Y
        return True, None

    def _scan(self, budget, fw, gw, filters, collect=None):
        budget.method = "bounded-sieve"
        reverse = sum(b-a+1 for a, b in gw) < sum(b-a+1 for a, b in fw)
        f, g, windows = (self.g, self.f, gw) if reverse else (self.f, self.g, fw)
        target_windows = fw if reverse else gw
        target_segments = [(lo, hi, evaluate(g, lo), evaluate(g, hi)) for lo, hi in target_windows]

        def target_root(value):
            if len(g) <= 3:
                return _root(g, value)
            for lo, hi, first, last in target_segments:
                if not min(first, last) <= value <= max(first, last):
                    continue
                ascending = first <= last
                while lo <= hi:
                    mid = (lo+hi)//2
                    v = evaluate(g, mid)
                    if v == value:
                        return mid
                    if (v < value) == ascending:
                        lo = mid+1
                    else:
                        hi = mid-1
            return None
        side = 2 if reverse else 1
        wheel, residues = 1, (0,)
        remaining = []
        for rule in filters:
            m, allowed = rule[0], rule[side]
            if len(allowed) == m:
                continue
            period = wheel*m//gcd(wheel, m)
            if period <= 4096:
                residues = tuple(r+i for i in range(0, period, wheel) for r in residues
                                 if (r+i) % m in allowed)
                wheel = period
            else:
                remaining.append((m, allowed))
        for lo, hi in windows:
            for residue in residues:
                budget.check()
                first = lo+(residue-lo) % wheel
                for x in range(first, hi+1, wheel):
                    budget.step()
                    if any(x % m not in allowed for m, allowed in remaining):
                        continue
                    v = evaluate(f, x)
                    y = target_root(v)
                    if y is not None:
                        hit = Witness(v, y, x) if reverse else Witness(v, x, y)
                        if collect is None:
                            return hit
                        collect(hit)
        return None


def _find_job(job, max_steps, timeout, backend):
    return job.find(max_steps=max_steps, timeout=timeout, backend=backend)


def solve_many(jobs, *, max_steps=None, timeout=None, workers=1, chunksize=16, backend="builtin"):
    """Yield ordered results; timeout applies to each job, not the batch."""
    if type(workers) is not int or workers < 1 or type(chunksize) is not int or chunksize < 1:
        raise ValueError("workers and chunksize must be positive integers")
    run = partial(_find_job, max_steps=max_steps, timeout=timeout, backend=backend)
    if workers == 1:
        yield from map(run, jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(run, jobs, chunksize=chunksize)
