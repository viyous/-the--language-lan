"""Small closed difference-bound matrices for mathematical integer values."""

from dataclasses import dataclass, field


ZONE_LIMIT = 24


@dataclass(frozen=True)
class Difference:
    """The expression positive - negative + offset. ID zero denotes zero."""

    positive: int = 0
    negative: int = 0
    offset: int = 0

    def combine(self, other, sign):
        terms = {}
        for uid, coefficient in ((self.positive, 1), (self.negative, -1),
                                 (other.positive, sign), (other.negative, -sign)):
            if uid:
                terms[uid] = terms.get(uid, 0) + coefficient
        positive = [uid for uid, coefficient in terms.items() if coefficient > 0]
        negative = [uid for uid, coefficient in terms.items() if coefficient < 0]
        if len(positive) > 1 or len(negative) > 1 or any(abs(n) > 1 for n in terms.values()):
            return None
        return Difference(positive[0] if positive else 0, negative[0] if negative else 0,
                          self.offset + sign * other.offset)

    def negated(self):
        return Difference(self.negative, self.positive, -self.offset)


@dataclass(frozen=True)
class Zone:
    ids: tuple = (0,)
    bounds: dict = field(default_factory=dict)

    def bound(self, left, right):
        return 0 if left == right else self.bounds[left, right]

    @classmethod
    def independent(cls, values, tracked=None):
        tracked = sorted(values)[:ZONE_LIMIT] if tracked is None else sorted(tracked)
        ids = (0, *tracked)
        lower = {0: 0} | {uid: values[uid].lo for uid in tracked}
        upper = {0: 0} | {uid: values[uid].hi for uid in tracked}
        return cls(ids, {(a, b): upper[a] - lower[b] for a in ids for b in ids if a != b})

    def range(self, expr):
        if expr.positive == expr.negative:
            return expr.offset, expr.offset
        if expr.positive not in self.ids or expr.negative not in self.ids:
            return None
        return (-self.bound(expr.negative, expr.positive) + expr.offset,
                self.bound(expr.positive, expr.negative) + expr.offset)

    def constrain(self, expr, maximum):
        """Intersect with expr <= maximum, maintaining shortest-path closure."""
        left, right, upper = expr.positive, expr.negative, maximum - expr.offset
        if left == right:
            return self if upper >= 0 else None
        if left not in self.ids or right not in self.ids or upper >= self.bound(left, right):
            return self
        if self.bound(right, left) + upper < 0:
            return None
        bounds = self.bounds.copy()
        for a in self.ids:
            prefix = self.bound(a, left) + upper
            for b in self.ids:
                if a != b:
                    bounds[a, b] = min(bounds[a, b], prefix + self.bound(right, b))
        return Zone(self.ids, bounds)

    def restrict(self, uid, lo, hi):
        narrowed = self.constrain(Difference(uid), hi)
        return None if narrowed is None else narrowed.constrain(Difference(0, uid), -lo)

    def assign(self, uid, lo, hi, expr):
        """Forget a write target, then bind it to an interval or an old slot plus a constant."""
        existing = uid in self.ids
        if not existing and len(self.ids) > ZONE_LIMIT:
            return self
        ids = self.ids if existing else (0, *sorted((*self.ids[1:], uid)))
        bounds = self.bounds.copy()
        source = (expr.positive if expr is not None and not expr.negative
                  and expr.positive in self.ids else None)
        for other in ids:
            if other == uid:
                continue
            if source is None:
                bounds[uid, other] = hi + self.bound(0, other)
                bounds[other, uid] = self.bound(other, 0) - lo
            else:
                bounds[uid, other] = self.bound(source, other) + expr.offset
                bounds[other, uid] = self.bound(other, source) - expr.offset
        return Zone(ids, bounds).restrict(uid, lo, hi)

    def project(self, kept):
        ids = (0, *(uid for uid in self.ids[1:] if uid in kept))
        if ids == self.ids:
            return self
        # A closed matrix has already propagated paths through removed slots.
        return Zone(ids, {(a, b): self.bounds[a, b] for a in ids for b in ids if a != b})

    def join(self, other):
        ids = (0, *(uid for uid in self.ids[1:] if uid in other.ids))
        return Zone(ids, {(a, b): max(self.bounds[a, b], other.bounds[a, b])
                          for a in ids for b in ids if a != b})

    def widen(self, joined, values, forgotten):
        # A growing edge is not reintroduced by a later closure. This gives
        # each edge one opportunity to widen, even if stable edges imply it.
        ids = joined.ids
        baseline = Zone.independent(values, ids[1:])
        bounds = baseline.bounds.copy()
        for key, next_bound in joined.bounds.items():
            if next_bound > self.bounds[key]:
                forgotten.add(key)
            elif key not in forgotten:
                bounds[key] = min(bounds[key], next_bound)
        for k in ids:
            for a in ids:
                if a == k:
                    continue
                for b in ids:
                    if b != a and b != k:
                        bounds[a, b] = min(bounds[a, b], bounds[a, k] + bounds[k, b])
        return Zone(ids, bounds)
