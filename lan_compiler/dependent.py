"""Linear relations and dependent upper bounds."""

from __future__ import annotations

from dataclasses import replace
from math import gcd, inf

from difference_domain import Difference
from array_domain import ArrayDomain
from value_domain import Domain, evaluate

from .typesys import Array, Slice, Span, Value, ValueOrigin, ZERO
from .domain_utils import c_int
from .interface_types import combine_offset_polynomial, OFFSET_BINARY_OPS, slice_offset_terms

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class DependentBoundMixin:
    @staticmethod
    def linear_value(value, *, prefer_symbolic=False):
        if value.floating or value.boolean or value.enum or value.character:
            return None
        if not prefer_symbolic and value.domain.lo == value.domain.hi:
            return frozenset(), value.domain.lo
        if value.linear is not None:
            return value.linear
        if value.affine is not None:
            form = value.affine
            factors = frozenset(((form.origin, 1),))
            return frozenset(((factors, form.scale),)), form.offset
        if value.domain.lo == value.domain.hi:
            return frozenset(), value.domain.lo
        return None

    def contraction_difference_domain(self, left, right):
        def bounds(form, numerator):
            source = self.linear_value(numerator)
            if form is None or source is None or numerator.domain.lo < 0 or len(form[0]) != 1:
                return None
            factors, coefficient = next(iter(form[0]))
            if len(factors) != 1:
                return None
            atom, exponent = next(iter(factors))
            if exponent != 1 or not isinstance(atom, tuple) or atom[0] != "integer-op":
                return None
            bias = form[1]
            if coefficient == 1 and atom[1] == "&":
                for masked in atom[2]:
                    if masked[0] != source[0]:
                        continue
                    shift = source[1] - masked[1]
                    if numerator.domain.lo < shift:
                        continue
                    # Only bits that the mask clears can contribute to the loss.
                    maximum = numerator.domain.hi - shift
                    loss = maximum
                    for terms, mask in atom[2] - {masked}:
                        if not terms:
                            possible_bits = (1 << maximum.bit_length()) - 1
                            loss = min(loss, ~mask & possible_bits)
                    return (Domain.of((bias - shift - loss, bias - shift)),
                            Domain.of((bias, bias + maximum)))
            if len(atom) != 4 or atom[2] != source:
                return None
            if coefficient == 1 and atom[1] == ">>":
                # Every valid right shift of a nonnegative operand is bounded by it.
                return (Domain.of((bias - numerator.domain.hi, bias)),
                        Domain.of((bias, bias + numerator.domain.hi)))
            if atom[1] not in ("/", "%"):
                return None
            terms, constant = atom[3]
            if coefficient != 1:
                if atom[1] != "/" or terms or coefficient != constant or constant <= 0:
                    return None
                # Multiplying a nonnegative truncated quotient by its divisor
                # removes exactly the remainder, even across many periods.
                lo, hi = numerator.domain.lo, numerator.domain.hi
                return (Domain.of((bias - min(hi, constant - 1), bias)),
                        Domain.of((bias + lo // constant * constant, bias + hi // constant * constant)))
            divisors = Domain.point(constant)
            if terms:
                if len(terms) != 1:
                    return None
                factors, scale = next(iter(terms))
                if len(factors) != 1:
                    return None
                origin, power = next(iter(factors))
                if power != 1 or not isinstance(origin, ValueOrigin):
                    return None
                divisors = origin.domain.affine(scale, constant)
            if type(divisors.lo) is not int or type(divisors.hi) is not int or divisors.lo < 1:
                return None
            lo, hi = numerator.domain.lo, numerator.domain.hi
            if atom[1] == "/":
                # n - trunc(n/d) increases with either nonnegative n or positive d.
                low = lo - lo // divisors.lo
                high = hi - hi // divisors.hi
            elif divisors.lo == divisors.hi:
                divisor = divisors.lo
                low = lo // divisor * divisor
                high = hi // divisor * divisor
            else:
                # A nonnegative remainder is bounded by both n and d - 1.
                low = max(0, lo - (divisors.hi - 1))
                high = hi
            contracted = (Domain.of((bias + lo // divisors.hi, bias + hi // divisors.lo))
                          if atom[1] == "/" else Domain.of((bias, bias + min(hi, divisors.hi - 1))))
            return Domain.of((bias - high, bias - low)), contracted

        def difference(value, numerator):
            direct = bounds(self.linear_value(value), numerator)
            if direct is not None:
                return direct[0]
            complement = bounds(self.combine_linear("-", numerator, value), numerator)
            return complement[1].affine(-1, 0) if complement is not None else None

        direct = difference(left, right)
        if direct is not None:
            return direct
        reverse = difference(right, left)
        return reverse.affine(-1, 0) if reverse is not None else None

    def combine_linear(self, op, left, right):
        if op not in OFFSET_BINARY_OPS:
            return None
        a, b = self.linear_value(left), self.linear_value(right)
        if a is None or b is None:
            return None
        if op in ("&", "|") and a == b:
            return a
        if op not in ("+", "-", "*"):
            atom = (("integer-op", op, frozenset((a, b))) if op in ("&", "|", "^")
                    else ("integer-op", op, a, b))
            return frozenset(((frozenset(((atom, 1),)), 1),)), 0
        left_terms = dict(a[0]) | {frozenset(): a[1]}
        right_terms = dict(b[0]) | {frozenset(): b[1]}
        polynomial = combine_offset_polynomial(op, left_terms, right_terms)
        if polynomial is None:
            return None
        bias = polynomial.pop(frozenset(), 0)
        return frozenset(polynomial.items()), bias

    def assume_dependent_expression(self, token, state, value, op, limit,
                                    diagnostic="dependent parameter expression has no possible input"):
        if self.binary(token, op, value, limit, state).domain == ZERO:
            token.error(diagnostic)
        return self.remember_integer_comparison(state, value, op, limit)

    def remember_integer_comparison(self, state, left, op, right):
        if op == "==":
            state = self.remember_integer_comparison(state, left, "<=", right)
            return self.remember_integer_comparison(state, right, "<=", left)
        if op in (">", ">="):
            left, right = right, left
        difference = self.combine_linear("-", left, right)
        if difference is None or not difference[0]:
            return state
        terms, bias = difference
        if op == "!=":
            divisor = gcd(*(scale for _, scale in terms))
            if bias % divisor:
                return state
            reduced = frozenset((factors, scale // divisor) for factors, scale in terms)
            inverse = frozenset((factors, -scale) for factors, scale in reduced)
            excluded = -bias // divisor
            facts = {("dependent_excluded", reduced, excluded): True,
                     ("dependent_excluded", inverse, -excluded): True}
            return replace(state, predicates=state.predicates | facts)
        key = ("dependent_upper", terms)
        upper = min(state.predicates.get(key, inf), -bias - int(op in ("<", ">")))
        updates = {key: upper}
        divisor = gcd(*(scale for _, scale in terms))
        if divisor > 1:
            normalized = frozenset((factors, scale // divisor) for factors, scale in terms)
            reduced = ("dependent_upper", normalized)
            updates[reduced] = min(state.predicates.get(reduced, inf), upper // divisor)
        return replace(state, predicates=state.predicates | updates)

    def dependent_difference_domain(self, left, right, state):
        if state is None:
            return None
        difference = self.combine_linear("-", left, right)
        if difference is None:
            return None
        terms, bias = difference
        if not terms:
            return Domain.point(bias)
        upper = self.dependent_upper_bound(terms, state)
        inverse = frozenset((factors, -scale) for factors, scale in terms)
        lower = -self.dependent_upper_bound(inverse, state)
        divisor = gcd(*(scale for _, scale in terms))
        reduced = frozenset((factors, scale // divisor) for factors, scale in terms)
        excluded = [divisor * key[2] + bias for key in state.predicates
                    if isinstance(key, tuple) and len(key) == 3
                    and key[0] == "dependent_excluded" and key[1] == reduced]
        if lower > upper or lower == -inf and upper == inf and not excluded:
            return None
        domain = Domain.of((lower + bias, upper + bias)) - Domain.of(*((n, n) for n in excluded))
        return domain or None

    @staticmethod
    def dependent_upper_bound(terms, state, *, split_mask=True):
        direct = state.predicates.get(("dependent_upper", terms))
        if direct is not None:
            return direct
        divisor = gcd(*(scale for _, scale in terms))
        scaled = inf
        if divisor > 1:
            normalized = frozenset((factors, scale // divisor) for factors, scale in terms)
            scaled = Compiler.dependent_upper_bound(normalized, state, split_mask=split_mask) * divisor
        facts = {key[1]: bound for key, bound in state.predicates.items()
                 if isinstance(key, tuple) and len(key) == 2 and key[0] == "dependent_upper"}

        def subtract(total, part):
            remaining = dict(total)
            for factors, scale in part:
                remaining[factors] = remaining.get(factors, 0) - scale
            return frozenset((factors, scale) for factors, scale in remaining.items() if scale)

        origin_domains, origin_places = {}, {}
        for uid, form in state.forms.items():
            if form.scale == 1 and uid in state.zone.ids:
                origin_places.setdefault(form.origin, Difference(uid, offset=-form.offset))
            original = origin_domains.get(form.origin, form.origin.domain)
            narrowed = original & form.preimage(state.values[uid])
            if narrowed:
                origin_domains[form.origin] = narrowed

        def interval_upper(remaining):
            total, residual = 0, []
            for factors, scale in remaining:
                low = high = 1
                for origin, power in factors:
                    if not isinstance(origin, ValueOrigin):
                        return inf
                    domain = origin_domains.get(origin, origin.domain)
                    if not all(type(bound) is int for bound in (domain.lo, domain.hi)):
                        return inf
                    ends = domain.lo ** power, domain.hi ** power
                    first = 0 if power % 2 == 0 and domain.lo <= 0 <= domain.hi else min(ends)
                    last = max(ends)
                    products = low * first, low * last, high * first, high * last
                    low, high = min(products), max(products)
                total += scale * (high if scale > 0 else low)
                if len(factors) == 1:
                    origin, power = next(iter(factors))
                    if power == 1:
                        residual.append((scale, origin))
            if len(remaining) == len(residual) == 2:
                (negative, low), (positive, high) = sorted(residual, key=lambda item: item[0])
                if positive == -negative and low in origin_places and high in origin_places:
                    difference = origin_places[high].combine(origin_places[low], -1)
                    bounds = state.zone.range(difference)
                    if bounds is not None:
                        total = min(total, positive * bounds[1])
            return total

        baseline = interval_upper(terms)
        # Bound operations that can only decrease the magnitude of their left operand.
        def negate_form(form):
            factors, bias = form
            return frozenset((term, -coefficient) for term, coefficient in factors), -bias

        def lower_bound(form):
            inverse, negated_bias = negate_form(form)
            upper = interval_upper(inverse)
            if facts and inverse and upper != inf:
                upper = min(upper, Compiler.dependent_upper_bound(inverse, state, split_mask=split_mask))
            return -negated_bias - upper

        def paired_shift_source(shift, inner_op):
            shifted, bias = shift[2]
            if bias or len(shifted) != 1:
                return None
            factors, coefficient = next(iter(shifted))
            if coefficient != 1 or len(factors) != 1:
                return None
            inner, exponent = next(iter(factors))
            if (exponent == 1 and isinstance(inner, tuple) and inner[:2] == ("integer-op", inner_op)
                    and inner[3] == shift[3]):
                return inner[2]
            return None

        for factors, scale in terms:
            if scale < 0 and len(factors) == 1:
                atom, power = next(iter(factors))
                if power == 1 and isinstance(atom, tuple) and atom[0] == "integer-op":
                    op = atom[1]
                    if op == "&":
                        operands = atom[2]
                    elif op in ("%", ">>"):
                        operands = (atom[2],)
                    elif op == "<<":
                        operands = (paired_shift_source(atom, ">>") or atom[2],)
                    else:
                        operands = ()
                else:
                    operands = ()
                if any(lower_bound(operand) >= 0 for operand in operands):
                    remaining = subtract(terms, frozenset(((factors, scale),)))
                    upper = (Compiler.dependent_upper_bound(remaining, state, split_mask=split_mask)
                             if remaining else 0)
                    baseline = min(baseline, upper)
                    break
            consumed = frozenset(((factors, scale),))
            if len(factors) == 1:
                if scale <= 0:
                    continue
                atom, power = next(iter(factors))
            else:
                # A quotient multiplied by its positive divisor rounds down.
                quotient = next(((atom, power) for atom, power in factors
                                 if power == 1 and isinstance(atom, tuple) and atom[:2] == ("integer-op", "/")), None)
                if quotient is None:
                    continue
                atom, power = quotient
                divisor_terms, divisor_bias = atom[3]
                aligned = combine_offset_polynomial("*", {frozenset((quotient,)): 1},
                                                    dict(divisor_terms) | {frozenset(): divisor_bias})
                if aligned is None or factors not in aligned or lower_bound(atom[3]) < 1:
                    continue
                multiplier, remainder = divmod(scale, aligned[factors])
                if remainder or not multiplier:
                    continue
                consumed = frozenset((term, coefficient * multiplier) for term, coefficient in aligned.items())
                coefficients = dict(terms)
                if any(coefficients.get(term) != coefficient for term, coefficient in consumed):
                    continue
                scale = multiplier
            if power != 1 or not isinstance(atom, tuple) or atom[0] != "integer-op":
                continue
            op = atom[1]
            source_scale = scale
            if op == "&" and len(atom[2]) == 2:
                sources = tuple(operand for operand in atom[2] if operand[0])
                if len(sources) > 1 and not split_mask:
                    continue
            elif op == "%":
                sources = (atom[2],)
            elif op in ("<<", ">>") and lower_bound(atom[3]) >= 0:
                source = paired_shift_source(atom, ">>" if op == "<<" else "<<")
                if source is None and op == ">>":
                    source = atom[2]
                if source is None:
                    continue
                sources = (source,)
            elif op == "/":
                divisor = lower_bound(atom[3])
                numerator = atom[2]
                if divisor < 1:
                    divisor = lower_bound(negate_form(atom[3]))
                    numerator = negate_form(numerator)
                if divisor < 1:
                    continue
                if scale < 0:
                    if lower_bound(numerator) < 0:
                        continue
                    sources = ((frozenset(), 0),)
                else:
                    sources = (numerator,)
                if len(factors) == 1 and divisor >= scale:
                    source_scale = 1
            else:
                continue
            sources = tuple(source for source in sources if lower_bound(source) >= 0)
            if not sources:
                continue
            remaining = subtract(terms, consumed)
            for source_terms, bias in sources:
                replaced = subtract(remaining, frozenset((term, -source_scale * coefficient)
                                                         for term, coefficient in source_terms))
                upper = (Compiler.dependent_upper_bound(replaced, state, split_mask=split_mask and len(sources) == 1)
                         if replaced else 0)
                baseline = min(baseline, source_scale * bias + upper)
            break
        if not facts:
            return baseline
        complements = [(subtract(terms, part), bound) for part, bound in facts.items()]
        pair = min((bound + facts.get(rest, inf) for rest, bound in complements), default=inf)
        interval = min((bound + interval_upper(rest) for rest, bound in complements), default=inf)
        if pair != inf or len(facts) > 32:
            return min(pair, interval, scaled, baseline)
        def remaining_bound(remaining):
            return min(facts.get(remaining, inf), interval_upper(remaining))

        combined = min((first + second + remaining_bound(subtract(rest, part))
                        for rest, first in complements for part, second in facts.items()), default=inf)
        return min(interval, combined, scaled, baseline)

    def prove_order(self, t, left, op, right, state):
        return self.binary(t, op, left, right, state).domain == Domain.point(1)

    def integer_expression_value(self, token, expression, arguments, state, *, require_total=False,
                                 refine_bounds=False):
        def evaluate(part):
            value = evaluate_offset(part)
            if refine_bounds:
                bound = self.dependent_difference_domain(value, Value(ZERO, "0"), state)
                if bound:
                    value = replace(value, domain=(value.domain & bound) or value.domain)
            return value

        def evaluate_offset(expr):
            if expr[0] == "param":
                return arguments[expr[1]]
            if expr[0] == "const":
                return Value(Domain.point(expr[1]), c_int(expr[1]))
            if expr[0] == "len":
                key = ("len", expr[1])
                if isinstance(arguments, dict) and key in arguments:
                    value = arguments[key]
                else:
                    source = arguments[expr[1]]
                    value = (source.length if isinstance(source, Span) else
                             Value(Domain.point(source.ty.length), c_int(source.ty.length)))
                return replace(value, width=expr[2], signed=False)
            if expr[0] == "cast":
                value = replace(evaluate(expr[5]), width=expr[3], signed=expr[4])
                if require_total and expr[1] == "trap" and not value.domain.within(expr[2].domain):
                    raise ValueError("offset conversion may trap on a returning path")
                return self.cast(token, value, expr[1], expr[2])
            return self.binary(token, expr[1], evaluate(expr[2]), evaluate(expr[3]), state)

        return evaluate(expression)

    def slice_offset_value(self, token, ty, arguments, state, *, require_total=False):
        if ty.offset_expression is not None:
            return self.integer_expression_value(token, ty.offset_expression, arguments, state,
                                                 require_total=require_total)
        result = Value(Domain.point(ty.offset_bias), c_int(ty.offset_bias))
        for factors, scale in slice_offset_terms(ty):
            value = arguments[factors[0]]
            for index in factors[1:]:
                value = self.binary(token, "*", value, arguments[index], state)
            if scale != 1:
                factor = Value(Domain.point(scale), c_int(scale))
                value = self.binary(token, "*", value, factor, state)
            result = self.binary(token, "+", result, value, state)
        return result

    @staticmethod
    def span_start(span):
        if span.start is not None:
            return span.start
        offsets = {offset for _, offset in span.positions()}
        if len(offsets) == 1:
            domain = next(iter(offsets))
            if domain is not None and domain.lo == domain.hi:
                return Value(domain, c_int(domain.lo))
        return None

    def add_span_start(self, span, offset, state, token):
        start = self.span_start(span)
        if start is None:
            return None
        if start.domain == ZERO:
            return self.capture_value(self.fresh_value(offset, state), state)
        return self.capture_value(self.binary(token, "+", start, offset, state), state)

    @staticmethod
    def sum_index_domains(left, right):
        if not left or not right:
            return Domain()
        if left.lo == left.hi:
            return right.affine(1, left.lo)
        if right.lo == right.hi:
            return left.affine(1, right.lo)
        left_ranges, right_ranges = left.ranges, right.ranges
        if len(left_ranges) * len(right_ranges) <= 64:
            result = Domain.of(*((a + c, b + d) for a, b in left_ranges for c, d in right_ranges))
            if not left.predicates and not right.predicates:
                return result
        else:
            result = Domain.of((left.lo + right.lo, left.hi + right.hi))
        left_residue, left_modulus = left.congruence()
        right_residue, right_modulus = right.congruence()
        return result.with_congruence(left_residue + right_residue, gcd(left_modulus, right_modulus))

    @staticmethod
    def shift_positions(span, offset):
        return tuple((owner, None if start is None else Compiler.sum_index_domains(start, offset))
                     for owner, start in span.positions())

    def actual_array(self, state, owner):
        if owner.uid in state.arrays:
            return state.arrays[owner.uid]
        if owner.uid < 0 and not owner.static_storage:
            data = next(data + term or b"\0" for (data, term), sym in self.text_literals.items() if sym.uid == owner.uid)
            return ArrayDomain.from_values(len(data), owner.ty.element.domain, data)
        ty = owner.ty
        length = ty.length if isinstance(ty, Array) else ty.domain.hi
        if owner.static_storage and isinstance(ty, Slice) and ty.content_bound is not None:
            return ArrayDomain.filled(length, ty.element.domain, ty.content_bound)
        return ArrayDomain.unknown(length, ty.element.domain)
