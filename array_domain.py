"""Persistent, compressed contents for a fixed-length scalar array.

An ArrayDomain always covers [0, length); absence never means zero or an
impossible value.  Adjacent equal segments are coalesced.  Operations run over
segments and index-domain intervals, not over every array element, so a zeroed
3 MB array is represented by one segment.

Indices and segment endpoints are element offsets (not byte offsets).  A write
through a nonsingleton index Domain updates one unknown element, hence weakly
unions the written value into every possibly affected segment.  Polynomial
indices may require interval overapproximation; this loses precision but never
excludes a possible index.  Bounds must be proved for the entire index domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from value_domain import Domain
from float_domain import FloatDomain


@dataclass(frozen=True)
class Segment:
    """Every element at start <= index < stop has this possible value domain."""

    start: int
    stop: int
    domain: Domain


def _integer(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _domain(value, element=None):
    if isinstance(value, (Domain, FloatDomain)):
        return value
    if isinstance(element, FloatDomain) and type(value) in (int, float):
        return FloatDomain.point(value, element.bits)
    if type(value) in (int, bool):
        return Domain.point(int(value))
    raise TypeError("element value must be an integer or Domain")


@dataclass(frozen=True)
class ArrayDomain:
    """A complete array content state with an explicit declaration contract.

    Prefer zero(), unknown(), filled() or from_values() for construction.
    Direct construction is validated too: gaps, overlaps, empty domains and
    values outside the declared element domain are rejected.
    """

    length: int
    element: Domain
    segments: tuple[Segment, ...]

    def __post_init__(self):
        _integer(self.length, "length")
        if self.length < 0:
            raise ValueError("array length cannot be negative")
        if not isinstance(self.element, (Domain, FloatDomain)) or not self.element:
            raise ValueError("array requires a nonempty declared element domain")
        end, merged = 0, []
        for segment in self.segments:
            if not isinstance(segment, Segment):
                raise TypeError("segments must be Segment instances")
            _integer(segment.start, "segment start")
            _integer(segment.stop, "segment stop")
            if segment.start != end or not end < segment.stop <= self.length:
                raise ValueError("segments must cover the array without gaps or overlaps")
            self._check_value(segment.domain)
            if merged and merged[-1].domain == segment.domain:
                previous = merged[-1]
                merged[-1] = Segment(previous.start, segment.stop, segment.domain)
            else:
                merged.append(segment)
            end = segment.stop
        if end != self.length:
            raise ValueError("segments must cover every declared array index")
        object.__setattr__(self, "segments", tuple(merged))

    @classmethod
    def filled(cls, length: int, element: Domain, value: Domain | int):
        """A uniform array; no iteration proportional to length is performed."""
        _integer(length, "length")
        domain = _domain(value, element)
        if length == 0:
            # There is no content to assign, but preserve the declaration.
            return cls(length, element, ())
        return cls(length, element, (Segment(0, length, domain),))

    @classmethod
    def zero(cls, length: int, element: Domain):
        """Default zero initialization, checked against the element contract."""
        return cls.filled(length, element, 0)

    @classmethod
    def unknown(cls, length: int, element: Domain):
        """Unknown contents, bounded by the declared element contract."""
        return cls.filled(length, element, element)

    @classmethod
    def from_values(cls, length: int, element: Domain, values: Iterable[Domain | int]):
        """List initialization; both too few and too many elements are errors.

        The iterable is consumed once.  Its adjacent equal domains are merged
        immediately rather than retaining an additional per-element copy.
        """
        _integer(length, "length")
        if length < 0:
            raise ValueError("array length cannot be negative")
        segments, count = [], 0
        for index, value in enumerate(values):
            if index >= length:
                raise ValueError("initializer contains more elements than the declared length")
            domain = _domain(value, element)
            if segments and segments[-1].domain == domain:
                previous = segments[-1]
                segments[-1] = Segment(previous.start, index + 1, domain)
            else:
                segments.append(Segment(index, index + 1, domain))
            count += 1
        if count != length:
            raise ValueError("initializer does not cover the declared array length")
        return cls(length, element, tuple(segments))

    def _check_value(self, value):
        if not isinstance(value, (Domain, FloatDomain)) or not value:
            raise ValueError("an array element cannot have an empty value domain")
        if not value.within(self.element):
            raise ValueError(f"element value {value} exceeds its declaration {self.element}")
        return value

    def _range(self, start, stop):
        _integer(start, "start")
        _integer(stop, "stop")
        if not 0 <= start <= stop <= self.length:
            raise IndexError(f"range [{start}, {stop}) is outside array length {self.length}")
        return start, stop

    def _indices(self, indices):
        if type(indices) is int:
            indices = Domain.point(indices)
        elif not isinstance(indices, Domain):
            raise TypeError("indices must be an integer or Domain")
        bounds = Domain.of((0, self.length - 1))
        if not indices.within(bounds):
            raise IndexError(f"cannot prove indices {indices} fit array length {self.length}")
        return indices & bounds

    def read(self, indices: Domain | int):
        """Union the possible values at every possible requested index.

        An empty index domain represents an unreachable read and returns the
        empty value domain.  Out-of-bounds indices are rejected, never clipped.
        """
        indices = self._indices(indices)
        empty = self.element - self.element
        values = (segment.domain for segment in self.segments
                  if indices & Domain.of((segment.start, segment.stop - 1)))
        result = empty
        for value in values:
            result |= value
        return result

    def read_range(self, start: int, stop: int):
        """Read all indices in a half-open range, including an empty range."""
        self._range(start, stop)
        return self.read(Domain.of((start, stop - 1)))

    def _replace(self, start, stop, replacement):
        """Replace a validated interval using absolute-coordinate segments."""
        if start == stop:
            return self
        result = []
        for segment in self.segments:
            if segment.start >= start:
                break
            result.append(Segment(segment.start, min(segment.stop, start), segment.domain))
        result.extend(replacement)
        for segment in self.segments:
            if segment.stop <= stop:
                continue
            result.append(Segment(max(segment.start, stop), segment.stop, segment.domain))
        return ArrayDomain(self.length, self.element, tuple(result))

    def fill(self, start: int, stop: int, value: Domain | int):
        """Strongly assign every element of a half-open interval."""
        self._range(start, stop)
        value = self._check_value(_domain(value, self.element))
        return self._replace(start, stop, (Segment(start, stop, value),) if start < stop else ())

    def write(self, indices: Domain | int, value: Domain | int):
        """Write one chosen index; exact singletons are strong, all others weak."""
        indices = self._indices(indices)
        value = self._check_value(_domain(value, self.element))
        if not indices:
            return self
        if indices.lo == indices.hi:
            return self.fill(indices.lo, indices.lo + 1, value)
        return self.weak_write(indices, value)

    def weak_write(self, indices: Domain | int, value: Domain | int):
        """An optional/uncertain write, even when indices is a singleton.

        Interval index domains preserve holes exactly.  Predicate index domains
        are split at their interval envelope boundaries, then intersections are
        checked for each piece.  If some indices of a piece are impossible but
        the predicate cannot be represented as a compact run, the whole piece
        is weakly updated.  No possible write is omitted.
        """
        indices = self._indices(indices)
        value = self._check_value(_domain(value, self.element))
        if not indices:
            return self
        boundaries = sorted({point for lo, hi in indices.ranges for point in (lo, hi + 1)})
        result, cursor = [], 0
        for segment in self.segments:
            while cursor < len(boundaries) and boundaries[cursor] <= segment.start:
                cursor += 1
            cuts = [segment.start]
            while cursor < len(boundaries) and boundaries[cursor] < segment.stop:
                cuts.append(boundaries[cursor])
                cursor += 1
            cuts.append(segment.stop)
            for start, stop in zip(cuts, cuts[1:]):
                overlap = indices & Domain.of((start, stop - 1))
                domain = segment.domain | value if overlap else segment.domain
                result.append(Segment(start, stop, domain))
        return ArrayDomain(self.length, self.element, tuple(result))

    def havoc(self, start: int = 0, stop: int | None = None):
        """Forget content facts in a range while retaining the declaration."""
        return self.fill(start, self.length if stop is None else stop, self.element)

    def havoc_indices(self, indices: Domain | int):
        return self.weak_write(indices, self.element)

    def slice(self, start: int, stop: int):
        """A content projection reindexed to zero; no element expansion occurs.

        This is an abstract snapshot, not a runtime ownership or alias object.
        A compiler retains the host identity and translates writes separately.
        """
        self._range(start, stop)
        if start == stop:
            return ArrayDomain(0, self.element, ())
        pieces = tuple(Segment(max(start, s.start) - start, min(stop, s.stop) - start, s.domain)
                       for s in self.segments if s.start < stop and start < s.stop)
        return ArrayDomain(stop - start, self.element, pieces)

    def copy_from(self, source: ArrayDomain, *, dst_start: int, src_start: int, count: int):
        """Snapshot copy with memmove semantics, including self-overlap.

        Source and destination may have different element declarations, but
        every copied possible value must fit the destination declaration.
        Uncopied destination content is preserved.
        """
        if not isinstance(source, ArrayDomain):
            raise TypeError("copy source must be an ArrayDomain")
        _integer(count, "count")
        if count < 0:
            raise ValueError("copy count cannot be negative")
        _integer(dst_start, "destination start")
        _integer(src_start, "source start")
        self._range(dst_start, dst_start + count)
        source._range(src_start, src_start + count)
        snapshot = source.slice(src_start, src_start + count)
        replacement = tuple(Segment(dst_start + s.start, dst_start + s.stop,
                                    self._check_value(s.domain)) for s in snapshot.segments)
        return self._replace(dst_start, dst_start + count, replacement)

    def join(self, other: ArrayDomain):
        """Pointwise union for two control-flow paths with one declaration."""
        if not isinstance(other, ArrayDomain):
            raise TypeError("join requires another ArrayDomain")
        if self.length != other.length or self.element != other.element:
            raise ValueError("joined arrays must have the same length and element declaration")
        if self == other:
            return self
        result, left, right, start = [], 0, 0, 0
        while start < self.length:
            a, b = self.segments[left], other.segments[right]
            stop = min(a.stop, b.stop)
            result.append(Segment(start, stop, a.domain | b.domain))
            start = stop
            if stop == a.stop:
                left += 1
            if stop == b.stop:
                right += 1
        return ArrayDomain(self.length, self.element, tuple(result))

    @classmethod
    def join_all(cls, states: Iterable[ArrayDomain]):
        states = iter(states)
        try:
            result = next(states)
        except StopIteration:
            raise ValueError("cannot join zero array states") from None
        if not isinstance(result, cls):
            raise TypeError("join requires ArrayDomain states")
        for state in states:
            result = result.join(state)
        return result
