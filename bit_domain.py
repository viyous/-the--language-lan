"""Bound bitwise images with a finite prefix automaton (no value enumeration)."""
from functools import lru_cache
from value_domain import Domain


@lru_cache(maxsize=4096)
def unsigned_bounds(op, lo, hi, low, high, width):
    # Four flags record whether an operand prefix still equals its bounds.
    states = {(True, True, True, True): (0, 0)}
    for bit in range(width - 1, -1, -1):
        next_states = {}
        bounds = ((lo >> bit) & 1, (hi >> bit) & 1,
                  (low >> bit) & 1, (high >> bit) & 1)
        for tight, (minimum, maximum) in states.items():
            for a in (0, 1):
                for b in (0, 1):
                    digits = (a, a, b, b)
                    if any(tight[i] and (digits[i] < bounds[i] if i % 2 == 0
                                        else digits[i] > bounds[i]) for i in range(4)):
                        continue
                    flags = tuple(tight[i] and digits[i] == bounds[i] for i in range(4))
                    value = a & b if op == "&" else a | b if op == "|" else a ^ b
                    lower, upper = (minimum << 1) | value, (maximum << 1) | value
                    if flags in next_states:
                        oldlo, oldhi = next_states[flags]
                        lower, upper = min(lower, oldlo), max(upper, oldhi)
                    next_states[flags] = lower, upper
        states = next_states
    return min(pair[0] for pair in states.values()), max(pair[1] for pair in states.values())


def bitwise_domain(op, left, right):
    signed = left.lo < 0 or right.lo < 0
    if signed:
        width = max((-min(left.lo, right.lo) - 1).bit_length() + 1,
                    max(0, left.hi, right.hi).bit_length() + 1)
    else:
        width = max(1, left.hi.bit_length(), right.hi.bit_length())
    modulus = 1 << width
    if op == "&":
        for source, mask in ((left, right), (right, left)):
            if mask.lo != mask.hi or mask.lo < 0:
                continue
            selected = mask.lo
            period = 1 << selected.bit_length()
            residue, step = source.congruence()
            if step % period == 0:
                return Domain.point(residue & selected), width
            if source.predicates:
                values = []
                for clause in source.clauses():
                    residue, step = Domain.from_clauses((clause,)).congruence()
                    if step % period:
                        break
                    values.append((residue & selected, residue & selected))
                else:
                    return Domain.of(*values), width
                if period <= 64 and len(source.predicates) <= 64:
                    bounds = Domain.of((source.lo, source.hi))
                    possible = {residue & selected for residue in range(period)
                                if source & bounds.with_congruence(residue, period)}
                    return Domain.of(*((value, value) for value in possible)), width

    def spans(domain):
        for lo, hi in domain.ranges:
            if lo < 0:
                yield lo + modulus, min(hi, -1) + modulus
            if hi >= 0:
                yield max(lo, 0), hi

    result = []
    for lo, hi in spans(left):
        for low, high in spans(right):
            a, b = unsigned_bounds(op, lo, hi, low, high, width)
            # Each input span has a fixed sign bit, hence so does the result.
            if signed and a & (modulus >> 1):
                a, b = a - modulus, b - modulus
            result.append((a, b))
    domain = Domain.of(*result)
    left_residue, left_step = left.congruence()
    right_residue, right_step = right.congruence()
    low_mask = 0
    for bit in range(width):
        flag = 1 << bit
        left_known = left_step % (flag * 2) == 0
        right_known = right_step % (flag * 2) == 0
        forced = ((op == "&" and (left_known and not left_residue & flag or right_known and not right_residue & flag))
                  or (op == "|" and (left_known and left_residue & flag or right_known and right_residue & flag)))
        if not (left_known and right_known or forced):
            break
        low_mask |= flag
    if low_mask:
        value = {"&": left_residue & right_residue, "|": left_residue | right_residue,
                 "^": left_residue ^ right_residue}[op]
        domain = domain.with_congruence(value & low_mask, low_mask + 1)
    return domain, width


def mask_preimage(domain, mask, expected):
    """Return the matching input intervals when their count fits the budget."""
    period = 1 << mask.bit_length()
    free = (period - 1) ^ mask
    suffix = (free ^ (free + 1)) >> 1
    free ^= suffix
    if free.bit_count() > 6:
        return None
    residues = [expected]
    while free:
        bit = free & -free
        residues += [residue | bit for residue in residues]
        free ^= bit
    ranges = []
    for lo, hi in domain.ranges:
        first, last = lo // period, hi // period
        if len(ranges) + (last - first + 1) * len(residues) > 64:
            return None
        for block in range(first, last + 1):
            for residue in residues:
                start = block * period + residue
                ranges.append((max(lo, start), min(hi, start + suffix)))
    return domain & Domain.of(*ranges)
