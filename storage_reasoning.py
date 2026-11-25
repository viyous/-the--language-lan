"""Deduce whole-storage facts from checked exit observations."""

from dataclasses import dataclass
from itertools import chain, combinations

from array_domain import ArrayDomain
from value_domain import Domain, DomainComplexity


@dataclass(frozen=True)
class KeyObservation:
    positions: Domain
    values: Domain


@dataclass(frozen=True)
class StorageExit:
    contents: ArrayDomain
    keys: tuple[KeyObservation, ...]
    separated: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class StorageObservation:
    parameter: int
    exits: tuple[StorageExit, ...]
    path: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class StorageCount:
    """A count after normal return, on a window relative to an argument."""

    parameter: int
    path: tuple[str | None, ...]
    start: int
    length: int
    selected: Domain
    bounds: tuple[int, int]


@dataclass(frozen=True)
class CountWindow:
    """Captured offset expression and size, relative to one physical owner."""

    start: tuple
    length: int
    column: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowCount:
    footprint: Domain
    bounds: tuple[int, int]

    @classmethod
    def join(cls, alternatives):
        footprint = Domain.of((min(f.footprint.lo for f in alternatives),
                               max(f.footprint.hi for f in alternatives)))
        return cls(footprint, (min(f.bounds[0] for f in alternatives),
                               max(f.bounds[1] for f in alternatives)))


def _lower_count(exit_state, selected, *, invert=False):
    """Lower bound for matching cells, or for cells in the complement."""
    definite, possible = [], []
    for segment in exit_state.contents.segments:
        matches = bool(segment.domain & selected)
        misses = not segment.domain.within(selected)
        can_count, can_miss = (misses, matches) if invert else (matches, misses)
        indices = segment.start, segment.stop - 1
        if can_count:
            possible.append(indices)
        if not can_miss:
            definite.append(indices)
    known = Domain.of(*definite)
    candidates = Domain.of(*possible)
    lower = sum(stop - start + 1 for start, stop in definite)

    # Each remaining observation requires at least one matching cell. The
    # hull weakens sparse index sets to intervals. The greedy stabbing count
    # for intervals is a lower bound even when the real keys have holes.
    required = []
    for index, key in enumerate(exit_state.keys):
        guaranteed = not (key.values & selected) if invert else key.values.within(selected)
        if not guaranteed:
            continue
        footprint = Domain.of(*key.positions.ranges) & candidates
        if footprint and not (footprint & known):
            required.append((index, footprint.lo, footprint.hi))
    last = -1
    by_interval = 0
    for _, start, stop in sorted(required, key=lambda interval: interval[2]):
        if last < start:
            by_interval += 1
            last = stop

    # Different cells are colors. An odd cycle can require more cells than
    # any pairwise distinct subset. Solve the small graph of proved conflicts.
    if exit_state.separated and len(required) > by_interval:
        separated = set(exit_state.separated)
        conflicts = {i: set() for i, _, _ in required}
        for (a, alo, ahi), (b, blo, bhi) in combinations(required, 2):
            if (a, b) in separated or (b, a) in separated or ahi < blo or bhi < alo:
                conflicts[a].add(b)
                conflicts[b].add(a)
        if len(conflicts) <= 8:
            order = sorted(conflicts, key=lambda key: -len(conflicts[key]))
            for limit in range(by_interval, len(order)):
                if _can_color(order, conflicts, limit, 0, []):
                    return lower + limit
            return lower + len(order)
    return lower + by_interval


def _can_color(order, conflicts, limit, index, groups):
    if index == len(order):
        return True
    key = order[index]
    for group in groups:
        if conflicts[key].isdisjoint(group):
            group.add(key)
            possible = _can_color(order, conflicts, limit, index + 1, groups)
            group.remove(key)
            if possible:
                return True
    if len(groups) < limit:
        return _can_color(order, conflicts, limit, index + 1, [*groups, {key}])
    return False


def count_bounds(observation, selected):
    """Bounds common to all reachable exits, using only the observations."""
    bounds = []
    for exit_state in observation.exits:
        length = exit_state.contents.length
        lower = _lower_count(exit_state, selected)
        upper = length - _lower_count(exit_state, selected, invert=True)
        if lower > upper:
            return None
        bounds.append((lower, upper))
    if not bounds:
        return None
    return min(lo for lo, _ in bounds), max(hi for _, hi in bounds)


def value_counts(observation):
    """Discover a bounded number of useful singleton count queries."""
    element = observation.exits[0].contents.element
    length = observation.exits[0].contents.length
    ranges = element.ranges
    if sum(hi - lo + 1 for lo, hi in ranges) <= 8:
        values = {value for lo, hi in ranges for value in range(lo, hi + 1)}
    else:
        values = set()
        for exit_state in observation.exits:
            values.update(seg.domain.lo for seg in exit_state.contents.segments
                          if seg.domain.lo == seg.domain.hi)
            values.update(key.values.lo for key in exit_state.keys if key.values.lo == key.values.hi)
    for value in sorted(values)[:8]:
        try:
            bounds = count_bounds(observation, Domain.point(value))
        except DomainComplexity:
            continue
        if bounds is not None and bounds != (0, length):
            yield value, bounds


def value_set_counts(observation):
    """Count non-singleton value sets already present in the exit facts."""
    element = observation.exits[0].contents.element
    length = observation.exits[0].contents.length
    queries = {}
    for exit_state in observation.exits:
        for domain in chain((seg.domain for seg in exit_state.contents.segments),
                            (key.values for key in exit_state.keys)):
            if domain != element and domain.lo != domain.hi:
                queries.setdefault(domain, None)
    for selected in tuple(queries)[:8]:
        try:
            bounds = count_bounds(observation, selected)
        except DomainComplexity:
            continue
        if bounds is not None and bounds != (0, length):
            yield selected, bounds
