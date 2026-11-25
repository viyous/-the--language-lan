"""Per-path abstract analysis state and loop bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from itertools import count

from difference_domain import Difference, Zone
from storage_reasoning import WindowCount
from array_domain import ArrayDomain
from float_domain import FloatDomain
from value_domain import Domain

from .typesys import AffineForm, ValueOrigin
from .domain_utils import domain_union, integer_states, known_float_bit_domain


_PRIMITIVES = (int, float, str, bytes)
_ATOMIC_DOMAINS = (Domain, FloatDomain, ArrayDomain, Difference, WindowCount)


def common_guards(paths):
    """Return guards shared across every path in `paths`, preserving order."""
    first = paths[0].guards
    if len(paths) < 2 or not first:
        return first
    rest = [set(path.guards) for path in paths[1:]]
    return tuple(guard for guard in first if all(guard in other for other in rest))


def _origins_live(value, live_origins):
    if isinstance(value, ValueOrigin):
        return value in live_origins
    if isinstance(value, AffineForm):
        return value.origin in live_origins
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_origins_live(item, live_origins) for item in value)
    return True


def _memo_key_live(node, epochs):
    if not isinstance(node, tuple):
        return True
    if len(node) == 3 and node[0] == "storage" and isinstance(node[2], tuple):
        return all(ep is epochs.get(uid) for uid, ep in node[2])
    return all(_memo_key_live(item, epochs) for item in node if isinstance(item, tuple))


def _key_mentions_uids(node, uids):
    if not isinstance(node, tuple) or not node:
        return False
    if len(node) == 3 and node[0] == "storage" and isinstance(node[2], tuple):
        return node[1] in uids or any(uid in uids for uid, _ in node[2])
    if node[0] in ("property_value", "slice_row", "array_element", "array_untouched", "array_window_post",
                   "storage_count", "storage_count_owner") and len(node) >= 2:
        return node[1] in uids
    return any(_key_mentions_uids(item, uids) for item in node if isinstance(item, tuple))


def _is_live_predicate(key, val, epochs, live_origins):
    if not isinstance(key, tuple) or not key:
        return True
    head = key[0]
    if head in ("property_value", "slice_row") and len(key) == 3:
        return key[2] is epochs.get(key[1])
    if head in ("array_element", "array_untouched", "array_window_post") and len(key) >= 2:
        return isinstance(val, tuple) and bool(val) and val[0] is epochs.get(key[1])
    if head == "return_cases" and len(key) == 2:
        return not isinstance(key[1], ValueOrigin) or key[1] in live_origins
    if head == "dependent_excluded" and len(key) >= 2:
        return _origins_live(key[1], live_origins)
    if head not in ("storage_count", "storage_count_owner"):
        return _memo_key_live(key, epochs)
    return True


def _canonical_sort_key(value, base_epochs, base_origins):
    if value is None:
        return ("0None",)
    kind = type(value)
    if kind is bool:
        return ("1bool", int(value))
    if kind in _PRIMITIVES:
        return (f"2{kind.__name__}", value)
    if isinstance(value, _ATOMIC_DOMAINS):
        if isinstance(value, Domain):
            return ("3Dom", value.lo, value.hi, value.ranges, repr(value.predicates))
        return ("3Atom", kind.__name__, repr(value))
    if kind is object:
        if base_epochs is not None and value in base_epochs:
            return ("4Obj", 0, id(value))
        return ("4Obj", 1, 0)
    if isinstance(value, ValueOrigin):
        is_base = 0 if (base_origins is not None and value in base_origins) else 1
        return ("5Orig", is_base, id(value) if is_base == 0 else 0,
                _canonical_sort_key(value.domain, base_epochs, base_origins))
    if isinstance(value, AffineForm):
        return ("6Affine", _canonical_sort_key(value.origin, base_epochs, base_origins),
                value.scale, value.offset)
    if isinstance(value, Zone):
        return ("7Zone", tuple(sorted(value.ids)), tuple(sorted(value.bounds.items())))
    if kind in (tuple, list):
        return ("8Seq", kind.__name__, tuple(_canonical_sort_key(item, base_epochs, base_origins) for item in value))
    if kind in (frozenset, set):
        return ("9Set", kind.__name__, tuple(sorted(_canonical_sort_key(item, base_epochs, base_origins) for item in value)))
    if kind is dict:
        items = sorted((_canonical_sort_key(k, base_epochs, base_origins),
                        _canonical_sort_key(v, base_epochs, base_origins))
                       for k, v in value.items())
        return ("ADict", tuple(items))
    if is_dataclass(value):
        if not value.__dataclass_params__.eq:
            return ("BId", kind.__name__, id(value))
        return ("BDC", kind.__name__, tuple(
            (f.name, _canonical_sort_key(getattr(value, f.name), base_epochs, base_origins))
            for f in fields(value) if f.compare))
    return ("ZOther", kind.__name__, repr(value))


def _canonical_value(value, base_epochs, base_origins, epoch_ids, origin_ids):
    if value is None:
        return None
    kind = type(value)
    if kind is bool:
        return (bool, value)
    if kind in _PRIMITIVES or isinstance(value, _ATOMIC_DOMAINS):
        return (kind, value)
    if kind is object:
        if base_epochs is not None and value in base_epochs:
            return ("epoch_base", id(value))
        if base_epochs is None:
            return ("epoch_id", id(value))
        return ("epoch_fresh", epoch_ids.setdefault(value, len(epoch_ids)))
    if isinstance(value, ValueOrigin):
        dom = value.domain
        if base_origins is not None and value in base_origins:
            return ("origin_base", id(value), dom)
        if base_origins is None:
            return ("origin_id", id(value), dom)
        return ("origin_fresh", origin_ids.setdefault(value, len(origin_ids)), dom)
    if isinstance(value, AffineForm):
        return ("AffineForm", _canonical_value(value.origin, base_epochs, base_origins, epoch_ids, origin_ids),
                value.scale, value.offset)
    if isinstance(value, Zone):
        return ("Zone", tuple(sorted(value.ids)), tuple(sorted(value.bounds.items())))
    if kind is tuple:
        return tuple(_canonical_value(item, base_epochs, base_origins, epoch_ids, origin_ids) for item in value)
    if kind is list:
        return ("list", tuple(_canonical_value(item, base_epochs, base_origins, epoch_ids, origin_ids) for item in value))
    if kind in (frozenset, set):
        ordered = sorted(value, key=lambda item: _canonical_sort_key(item, base_epochs, base_origins))
        return (kind, tuple(_canonical_value(item, base_epochs, base_origins, epoch_ids, origin_ids) for item in ordered))
    if kind is dict:
        ordered = sorted(value.items(), key=lambda kv: _canonical_sort_key(kv[0], base_epochs, base_origins))
        return ("dict", tuple(
            (_canonical_value(k, base_epochs, base_origins, epoch_ids, origin_ids),
             _canonical_value(v, base_epochs, base_origins, epoch_ids, origin_ids))
            for k, v in ordered))
    if is_dataclass(value):
        if not value.__dataclass_params__.eq:
            return ("id", kind, id(value))
        return ("dc", kind, tuple(
            _canonical_value(getattr(value, f.name), base_epochs, base_origins, epoch_ids, origin_ids)
            for f in fields(value) if f.compare))
    return value

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


@dataclass
class PathState:
    values: dict
    guards: tuple = ()
    bits: dict = field(default_factory=dict)
    forms: dict = field(default_factory=dict)
    zone: Zone = field(default_factory=Zone)
    borrows: dict[int, frozenset[Symbol]] = field(default_factory=dict)
    heap: dict[int, frozenset[str]] = field(default_factory=dict)
    arrays: dict[int, ArrayDomain] = field(default_factory=dict)
    regions: dict[int, tuple] = field(default_factory=dict)
    predicates: dict = field(default_factory=dict)
    epochs: dict = field(default_factory=dict)
    bit_values: dict = field(default_factory=dict)
    row_writes: dict = field(default_factory=dict)
    starts: dict[int, Value] = field(default_factory=dict)
    linear: dict[int, tuple] = field(default_factory=dict)
    # (link field uid, target uid) pairs that point at a *new* object each
    # iteration.  They let the meta-graph tell a loop's repeated allocation
    # apart from a genuine cycle.
    fresh_links: frozenset = frozenset()

    @classmethod
    def with_independent_values(cls, values, guards=(), zone=None, borrows=None, heap=None, arrays=None, regions=None):
        integers = integer_states(values)
        return cls(values, guards, forms={uid: AffineForm(ValueOrigin(domain)) for uid, domain in integers.items()},
                   zone=zone if zone is not None else Zone.independent(integers),
                   borrows=borrows or {}, heap=heap or {}, arrays=arrays or {}, regions=regions or {})

    def update(self, sym, domain, form=None, difference=None, linear=None):
        forms = {uid: value for uid, value in self.linear.items() if uid != sym.uid}
        if linear is not None:
            forms[sym.uid] = linear
        if isinstance(domain, FloatDomain):
            return replace(self, values=self.values | {sym.uid: domain}, linear=forms,
                           forms={uid: value for uid, value in self.forms.items() if uid != sym.uid})
        # A value copied earlier keeps its original origin after a rebind.
        relation = form if form is not None and form.scale else AffineForm(ValueOrigin(domain))
        zone = self.zone.assign(sym.uid, domain.lo, domain.hi, difference)
        if zone is None:
            zone = self.zone.assign(sym.uid, domain.lo, domain.hi, None)
        return replace(self, values=self.values | {sym.uid: domain},
                       forms=self.forms | {sym.uid: relation}, zone=zone, linear=forms)

    def narrow(self, sym, domain):
        domain = domain & self.values[sym.uid]
        if not domain:
            return None
        values = self.values | {sym.uid: domain}
        if isinstance(domain, FloatDomain):
            return replace(self, values=values)
        form = self.forms.get(sym.uid)
        if form is not None:
            original = None
            for uid, related in self.forms.items():
                if uid == sym.uid or related.origin is not form.origin:
                    continue
                if original is None:
                    original = form.preimage(domain) & form.origin.domain
                    if not original:
                        return None
                values[uid] = values[uid] & original.affine(related.scale, related.offset)
                if not values[uid]:
                    return None
        zone = self.zone
        for uid in zone.ids[1:]:
            value = values[uid]
            if value != self.values[uid]:
                zone = zone.restrict(uid, value.lo, value.hi)
                if zone is None:
                    return None
        return replace(self, values=values, zone=zone).with_zone(zone)

    def with_zone(self, zone):
        if zone is None:
            return None
        values = self.values.copy()
        for uid in zone.ids[1:]:
            lo, hi = zone.range(Difference(uid))
            value = values[uid]
            if value.lo < lo or hi < value.hi:
                values[uid] = value & Domain.of((lo, hi))
                if not values[uid]:
                    return None
        return replace(self, values=values, zone=zone)

    def bind_equal(self, dst, form):
        return replace(self, forms=self.forms | {dst.uid: form})

    def guard(self, label):
        return replace(self, guards=self.guards + (label,))

    def guard_label(self, limit=4):
        if not self.guards:
            return "entry"
        if len(self.guards) <= limit:
            return " and ".join(self.guards)
        return f"... ({len(self.guards) - 3} more) and " + " and ".join(self.guards[-3:])

    def with_bits(self, sym, facts):
        return replace(self, bits=self.bits | {sym.uid: facts})

    def touch(self, uid):
        key = ("array_untouched", uid)
        predicates = self.predicates
        if key in predicates or ("storage_count_owner", uid) in predicates:
            predicates = {name: value for name, value in predicates.items()
                          if name != key and name[:2] not in (
                              ("storage_count", uid), ("storage_count_owner", uid))}
        return replace(self, epochs=self.epochs | {uid: object()}, predicates=predicates)

    def forget_bit_values(self, uid, offset=None, width=None):
        def retained(key):
            owner, start, count, _ = key
            return owner != uid or offset is not None and (start + count <= offset or offset + width <= start)
        return replace(self, bit_values={key: domain for key, domain in self.bit_values.items() if retained(key)})

    def bind_span(self, sym, span):
        state = self.touch(sym.uid).update(sym, span.length.domain, span.length.affine,
                                           span.length.difference, span.length.linear)
        state.borrows = state.borrows | {sym.uid: span.owners}
        state.regions = state.regions | {sym.uid: span.positions()}
        if span.row_identity is not None:
            key = ("slice_row", sym.uid, state.epochs[sym.uid])
            state.predicates = state.predicates | {key: span.row_identity}
        state.starts = {uid: start for uid, start in state.starts.items() if uid != sym.uid}
        if span.start is not None:
            state.starts[sym.uid] = span.start
        return state

    def with_array(self, sym, contents):
        return replace(self, arrays=self.arrays | {sym.uid: contents})

    def absorb(self, other):
        # Expression evaluation owns this cursor. Maps stay persistent so
        # branch snapshots and previously captured scalar values remain valid.
        self.values, self.forms, self.zone = other.values, other.forms, other.zone
        self.bits, self.borrows, self.heap = other.bits, other.borrows, other.heap
        self.arrays, self.regions = other.arrays, other.regions
        self.starts, self.linear = other.starts, other.linear
        self.row_writes = other.row_writes
        self.predicates, self.epochs, self.bit_values = other.predicates, other.epochs, other.bit_values
        self.fresh_links = other.fresh_links

    @staticmethod
    def _collect_origins(value, out):
        if isinstance(value, ValueOrigin):
            out.add(value)
        elif isinstance(value, AffineForm):
            out.add(value.origin)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                PathState._collect_origins(item, out)

    @staticmethod
    def baseline(paths):
        """Snapshot epoch and ValueOrigin identities present on entry paths."""
        epochs = {ep for path in paths for ep in path.epochs.values()}
        origins = {form.origin for path in paths for form in path.forms.values()}
        for path in paths:
            for start in path.starts.values():
                if start.affine is not None:
                    origins.add(start.affine.origin)
            for term_entry in path.linear.values():
                PathState._collect_origins(term_entry, origins)
            for key in path.predicates:
                if isinstance(key, tuple) and key[:1] == ("return_cases",):
                    PathState._collect_origins(key[1], origins)
        return frozenset(epochs), frozenset(origins)

    def canonical_key(self, baseline_epochs=None, baseline_origins=None):
        """Return a hashable key identifying semantic state (excluding trace guards)."""
        epoch_ids, origin_ids = {}, {}

        def canon(val):
            return _canonical_value(val, baseline_epochs, baseline_origins, epoch_ids, origin_ids)

        is_error = bool(self.guards and self.guards[-1].startswith("error@"))
        live_origins = set()
        for form in self.forms.values():
            live_origins.add(form.origin)
        for start in self.starts.values():
            if start.affine is not None:
                live_origins.add(start.affine.origin)
        for entry in self.linear.values():
            PathState._collect_origins(entry, live_origins)
        live_preds = tuple(
            (canon(k), canon(v))
            for k, v in sorted(
                ((k, v) for k, v in self.predicates.items()
                 if _is_live_predicate(k, v, self.epochs, live_origins)),
                key=lambda kv: _canonical_sort_key(kv[0], baseline_epochs, baseline_origins),
            )
        )
        return (
            is_error,
            tuple(sorted(self.values.items())),
            tuple(sorted(self.zone.ids)),
            tuple(sorted(self.zone.bounds.items())),
            tuple(sorted(self.arrays.items())),
            tuple((uid, canon(ep)) for uid, ep in sorted(self.epochs.items())),
            tuple((uid, canon(form)) for uid, form in sorted(self.forms.items())),
            tuple((uid, canon(start)) for uid, start in sorted(self.starts.items())),
            tuple((uid, canon(lin)) for uid, lin in sorted(self.linear.items())),
            live_preds,
            tuple(sorted(self.bits.items())),
            tuple((canon(k), v) for k, v in sorted(
                self.bit_values.items(),
                key=lambda kv: _canonical_sort_key(kv[0], baseline_epochs, baseline_origins),
            )),
            tuple((canon(k), v) for k, v in sorted(
                self.row_writes.items(),
                key=lambda kv: _canonical_sort_key(kv[0], baseline_epochs, baseline_origins),
            )),
            tuple((uid, canon(borrow)) for uid, borrow in sorted(self.borrows.items())),
            tuple(sorted(self.heap.items())),
            tuple((uid, canon(region)) for uid, region in sorted(self.regions.items())),
            canon(self.fresh_links),
        )

    @staticmethod
    def _has_zone_relation(zone, uid):
        if uid not in zone.ids:
            return False
        hi_u, neg_lo_u = zone.bounds[uid, 0], zone.bounds[0, uid]
        for other in zone.ids:
            if other == 0 or other == uid:
                continue
            if (zone.bounds[uid, other] < hi_u + zone.bounds[0, other]
                    or zone.bounds[other, uid] < zone.bounds[other, 0] + neg_lo_u):
                return True
        return False

    def live_canonical_key(self, ignored_uids, baseline_epochs=None, baseline_origins=None):
        """Canonical state key ignoring scalar slots in `ignored_uids`."""
        if not ignored_uids:
            return self.canonical_key(baseline_epochs, baseline_origins)
        epoch_ids, origin_ids = {}, {}

        def canon(val):
            return _canonical_value(val, baseline_epochs, baseline_origins, epoch_ids, origin_ids)

        is_error = bool(self.guards and self.guards[-1].startswith("error@"))
        live_values = {uid: val for uid, val in self.values.items() if uid not in ignored_uids}
        live_forms = {uid: form for uid, form in self.forms.items() if uid not in ignored_uids}
        live_linear = {uid: lin for uid, lin in self.linear.items() if uid not in ignored_uids}
        live_epochs = {uid: ep for uid, ep in self.epochs.items() if uid not in ignored_uids}
        live_zone = self.zone.project(live_values.keys())
        live_origins = set()
        for form in live_forms.values():
            live_origins.add(form.origin)
        for start in self.starts.values():
            if start.affine is not None:
                live_origins.add(start.affine.origin)
        for entry in live_linear.values():
            PathState._collect_origins(entry, live_origins)
        live_preds = tuple(
            (canon(k), canon(v))
            for k, v in sorted(
                ((k, v) for k, v in self.predicates.items()
                 if not _key_mentions_uids(k, ignored_uids)
                 and _is_live_predicate(k, v, self.epochs, live_origins)),
                key=lambda kv: _canonical_sort_key(kv[0], baseline_epochs, baseline_origins),
            )
        )
        return (
            is_error,
            tuple(sorted(live_values.items())),
            tuple(sorted(live_zone.ids)),
            tuple(sorted(live_zone.bounds.items())),
            tuple(sorted(self.arrays.items())),
            tuple((uid, canon(ep)) for uid, ep in sorted(live_epochs.items())),
            tuple((uid, canon(form)) for uid, form in sorted(live_forms.items())),
            tuple((uid, canon(start)) for uid, start in sorted(self.starts.items())),
            tuple((uid, canon(lin)) for uid, lin in sorted(live_linear.items())),
            live_preds,
            tuple(sorted(self.bits.items())),
            tuple((canon(k), v) for k, v in sorted(
                self.bit_values.items(),
                key=lambda kv: _canonical_sort_key(kv[0], baseline_epochs, baseline_origins),
            )),
            tuple((canon(k), v) for k, v in sorted(
                self.row_writes.items(),
                key=lambda kv: _canonical_sort_key(kv[0], baseline_epochs, baseline_origins),
            )),
            tuple((uid, canon(borrow)) for uid, borrow in sorted(self.borrows.items())),
            tuple(sorted(self.heap.items())),
            tuple((uid, canon(region)) for uid, region in sorted(self.regions.items())),
            canon(self.fresh_links),
        )

    @classmethod
    def _merge_on_scalars(cls, group, scalar_uids, update_zone_uids=()):
        """Merge a group of paths that agree on all state outside `scalar_uids`."""
        first = group[0]
        values = first.values.copy()
        forms = first.forms.copy()
        linear = first.linear.copy()
        epochs = first.epochs.copy()
        changed_dead = set()
        zone = first.zone
        for uid in scalar_uids:
            if uid not in values:
                continue
            vals = [p.values[uid] for p in group]
            if any(v != vals[0] for v in vals[1:]):
                union = domain_union(vals)
                values[uid] = union
                if isinstance(union, FloatDomain):
                    forms.pop(uid, None)
                else:
                    forms[uid] = AffineForm(ValueOrigin(union))
                linear.pop(uid, None)
                if uid not in update_zone_uids:
                    changed_dead.add(uid)
            elif any(p.forms.get(uid) != first.forms.get(uid) for p in group[1:]):
                if isinstance(values[uid], FloatDomain):
                    forms.pop(uid, None)
                else:
                    forms[uid] = AffineForm(ValueOrigin(values[uid]))
            if any(p.linear.get(uid) != first.linear.get(uid) for p in group[1:]):
                linear.pop(uid, None)
            ep0 = first.epochs.get(uid)
            if any(p.epochs.get(uid) is not ep0 for p in group[1:]):
                epochs[uid] = object()
        if changed_dead:
            zone = zone.project(values.keys() - changed_dead)
        for uid in update_zone_uids:
            if uid in values and isinstance(values[uid], Domain):
                u = values[uid]
                updated = zone.assign(uid, u.lo, u.hi, None)
                if updated is not None:
                    zone = updated
        predicates = {
            k: v for k, v in first.predicates.items()
            if not _key_mentions_uids(k, scalar_uids)
            or all(p.predicates.get(k) == v for p in group[1:])
        }
        return replace(first, values=values, forms=forms, linear=linear, epochs=epochs,
                       zone=zone, predicates=predicates, guards=common_guards(group))

    @classmethod
    def fold_dead(cls, paths, dead_uids, baseline_epochs=None, baseline_origins=None):
        """Fold paths that are canonically identical on all live variables."""
        if len(paths) < 2:
            return list(paths) if not isinstance(paths, list) else paths
        if not dead_uids:
            return cls.fold(paths, baseline_epochs, baseline_origins)
        groups = {}
        for path in paths:
            key = path.live_canonical_key(dead_uids, baseline_epochs, baseline_origins)
            groups.setdefault(key, []).append(path)
        if len(groups) == len(paths):
            return list(paths) if not isinstance(paths, list) else paths
        result = []
        for group in groups.values():
            if len(group) == 1:
                result.append(group[0])
            else:
                result.append(cls._merge_on_scalars(group, dead_uids))
        return result

    @classmethod
    def fold_independent_scalar(cls, paths, dead_uids, pinned_uids, baseline_epochs=None, baseline_origins=None):
        """Losslessly merge paths differing only in a single independent integer scalar."""
        if len(paths) < 2:
            return list(paths) if not isinstance(paths, list) else paths
        first = paths[0]
        excluded = set(dead_uids) | set(pinned_uids)
        candidates = [
            uid for uid, val in first.values.items()
            if uid not in excluded and isinstance(val, Domain)
            and any(p.values.get(uid) != val for p in paths[1:])
        ]
        if not candidates:
            return list(paths) if not isinstance(paths, list) else paths
        current = list(paths)
        for cand in candidates:
            if len(current) < 2:
                break
            if any(
                not isinstance(p.values.get(cand), Domain)
                or cand in p.borrows or cand in p.heap or cand in p.arrays
                or cand in p.regions or cand in p.bits or cand in p.starts
                or any(k[0] == cand for k in p.bit_values)
                or any(k[1][0].uid == cand for k in p.row_writes)
                or any(isinstance(k, tuple) and len(k) >= 2
                       and k[0] in ("storage_count", "storage_count_owner", "array_element",
                                    "array_untouched", "array_window_post", "slice_row")
                       and k[1] == cand for k in p.predicates)
                or cls._has_zone_relation(p.zone, cand)
                for p in current
            ):
                continue
            # Ensure `cand` does not share an AffineForm origin with any other live variable.
            shared_origin = False
            for p in current:
                form = p.forms.get(cand)
                if form is None:
                    continue
                origin = form.origin
                if any(uid != cand and uid not in dead_uids and other.origin is origin
                       for uid, other in p.forms.items()):
                    shared_origin = True
                    break
                for uid, lin in p.linear.items():
                    if uid != cand and uid not in dead_uids:
                        lin_origins = set()
                        cls._collect_origins(lin, lin_origins)
                        if origin in lin_origins:
                            shared_origin = True
                            break
                if shared_origin:
                    break
            if shared_origin:
                continue
            ignored = dead_uids | {cand}
            groups = {}
            for path in current:
                key = path.live_canonical_key(ignored, baseline_epochs, baseline_origins)
                groups.setdefault(key, []).append(path)
            if len(groups) == len(current):
                continue
            merged = []
            for group in groups.values():
                if len(group) == 1:
                    merged.append(group[0])
                    continue
                union = domain_union(p.values[cand] for p in group)
                if not isinstance(union, Domain) or union.predicates or len(union.ranges) > 4:
                    merged.extend(group)
                else:
                    merged.append(cls._merge_on_scalars(group, ignored, update_zone_uids=(cand,)))
            current = merged
        return current

    @classmethod
    def fold(cls, paths, baseline_epochs=None, baseline_origins=None):
        """Deduplicate canonically equivalent states, preserving shared guards."""
        if len(paths) < 2:
            return list(paths) if not isinstance(paths, list) else paths
        groups = {}
        for path in paths:
            key = path.canonical_key(baseline_epochs, baseline_origins)
            groups.setdefault(key, []).append(path)
        if len(groups) == len(paths):
            return list(paths) if not isinstance(paths, list) else paths
        result = []
        for group in groups.values():
            if len(group) == 1:
                result.append(group[0])
            else:
                guards = common_guards(group)
                result.append(replace(group[0], guards=guards) if guards != group[0].guards else group[0])
        return result

    @classmethod
    def join(cls, paths):
        first = paths[0]
        values = {uid: domain_union(p.values[uid] for p in paths) for uid in first.values}
        zone = first.zone
        for other in paths[1:]:
            zone = zone.join(other.zone)
        guards = common_guards(paths)
        result = cls.with_independent_values(values, guards, zone,
            {uid: frozenset().union(*(p.borrows.get(uid, frozenset()) for p in paths))
             for uid in set().union(*(p.borrows.keys() for p in paths))},
            {uid: frozenset().union(*(p.heap.get(uid, frozenset()) for p in paths))
             for uid in set().union(*(p.heap.keys() for p in paths))})
        ordered_links = set().union(*(p.fresh_links for p in paths))
        for path in paths:
            if not ordered_links:
                break
            ordered_links.difference_update(
                (field, uid) for field, uid in ordered_links - path.fresh_links
                if any(target.uid == uid for target in path.borrows.get(field, ())))
        result.fresh_links = frozenset(ordered_links)
        for path in paths:
            for key, bits in path.row_writes.items():
                result.row_writes[key] = result.row_writes.get(key, Domain.of()) | bits
        result.linear = {uid: form for uid, form in first.linear.items()
                         if all(path.linear.get(uid) == form for path in paths)}
        result.starts = {uid: start for uid, start in first.starts.items()
                         if all(path.starts.get(uid) == start for path in paths)}
        result.predicates = {key: value for key, value in first.predicates.items()
                             if all(p.predicates.get(key) == value for p in paths)}
        for key in first.predicates:
            if key[:1] == ("storage_count",) and all(key in path.predicates for path in paths):
                if key not in result.predicates:
                    result.predicates[key] = WindowCount.join([path.predicates[key] for path in paths])
                result.predicates["storage_count_owner", key[1]] = True
            if (isinstance(key, tuple) and len(key) == 2 and key[0] == "dependent_upper"
                    and all(key in path.predicates for path in paths)):
                result.predicates[key] = max(path.predicates[key] for path in paths)
        for uid in set().union(*(p.epochs.keys() for p in paths)):
            epoch = first.epochs.get(uid)
            result.epochs[uid] = epoch if all(p.epochs.get(uid) is epoch for p in paths) else object()
            fact = first.predicates.get(("property_value", uid, epoch))
            if (fact is not None and uid in result.values
                    and all(p.predicates.get(("property_value", uid, p.epochs.get(uid))) == fact for p in paths)):
                result.predicates["property_value", uid, result.epochs[uid]] = fact
        for key in first.predicates:
            if not (isinstance(key, tuple) and len(key) == 3 and key[0] == "array_element"):
                continue
            uid = key[1]
            facts = [path.predicates.get(key) for path in paths]
            if not all(item is not None and item[0] is path.epochs.get(uid)
                       for path, item in zip(paths, facts)):
                result.predicates.pop(key, None)
                continue
            footprints = [item[2] for item in facts]
            footprint = domain_union(footprints) if all(item is not None for item in footprints) else None
            result.predicates[key] = (result.epochs.get(uid),
                                      domain_union(item[1] for item in facts), footprint)
        bit_keys = set().union(*(path.bit_values.keys() for path in paths))
        for key in bit_keys:
            domains = []
            for path in paths:
                value = path.bit_values.get(key)
                if value is None:
                    uid, offset, width, order = key
                    mask, data = path.bits.get(uid, (0, 0))
                    if order.startswith("float:"):
                        value = known_float_bit_domain(mask, data, offset, width, order.removeprefix("float:"))
                    else:
                        window = (1 << width) - 1
                        if (mask >> offset) & window == window:
                            raw = (data >> offset) & window
                            orders = ("le", "be") if order == "native" else (order,)
                            value = Domain.union(Domain.point(Compiler.bit_order(raw, width, endian))
                                                 for endian in orders)
                if value is None:
                    break
                domains.append(value)
            else:
                result.bit_values[key] = domain_union(domains)
        for uid, form in first.forms.items():
            if all(p.forms.get(uid) == form for p in paths):
                result.forms[uid] = form
        for uid, facts in first.bits.items():
            mask, data = facts
            for other in paths[1:]:
                omask, odata = other.bits.get(uid, (0, 0))
                mask &= omask & ~(data ^ odata)
            result.bits[uid] = (mask, data & mask)
        for uid in first.arrays:
            result.arrays[uid] = ArrayDomain.join_all(p.arrays.get(uid, first.arrays[uid].havoc()) for p in paths)
        for uid in first.regions:
            positions = {}
            for path in paths:
                for owner, offset in path.regions[uid]:
                    if owner in positions:
                        old = positions[owner]
                        offset = old | offset if old is not None and offset is not None else None
                    positions[owner] = offset
            result.regions[uid] = tuple(positions.items())
        return result

    def project(self, ids, storage_ids, bit_scope):
        # A link target that is still alive stays reachable after its own
        # variable goes out of scope: the chain root kept outside the loop
        # still points at body-allocated nodes, and following `next` through
        # them needs their field targets.  Only live heap targets extend the
        # closure, so blocks that dropped a binding do not resurrect it.
        kept = {uid: self.borrows[uid] for uid in ids if uid in self.borrows}
        heap = {uid: states for uid, states in self.heap.items() if uid in ids}
        pending = [target for targets in kept.values() for target in targets if target.uid]
        while pending:
            target = pending.pop()
            if self.heap.get(target.uid) != frozenset(("Alive",)):
                continue
            heap.setdefault(target.uid, frozenset(("Alive",)))
            for field in target.fields:
                if field.uid in kept or field.uid not in self.borrows:
                    continue
                kept[field.uid] = self.borrows[field.uid]
                pending.extend(item for item in kept[field.uid] if item.uid)
        return PathState({uid: self.values[uid] for uid in ids}, self.guards,
                         forms={uid: self.forms[uid] for uid in ids if uid in self.forms},
                         bits={uid: facts for uid, facts in self.bits.items() if uid <= bit_scope},
                         zone=self.zone.project(ids),
                         borrows=kept,
                         heap=heap,
                         arrays={uid: state for uid, state in self.arrays.items() if storage_ids is None or uid in storage_ids},
                         regions={uid: places for uid, places in self.regions.items() if uid in ids},
                         starts={uid: start for uid, start in self.starts.items() if uid in ids},
                         linear={uid: form for uid, form in self.linear.items() if uid in ids},
                         predicates={key: value for key, value in self.predicates.items()
                                     if key[:1] not in (("storage_count",), ("storage_count_owner",))
                                     or storage_ids is None or key[1] in storage_ids},
                         epochs=self.epochs, bit_values=self.bit_values,
                         row_writes=self.row_writes,
                         fresh_links=frozenset((field, target) for field, target in self.fresh_links
                                               if field in kept))


@dataclass
class LoopEdges:
    depth: int
    continues: list[PathState]
    continue_label: str
    entry_alias_marker: object = field(default_factory=object)
    symbol_cutoff: int = 0
    previous_allocations: frozenset = frozenset()
    append_cursors: set = field(default_factory=set)


@dataclass(frozen=True)
class CountedLoop:
    statement: Node
    limit: int
    updates: tuple
    induction: Symbol | None = None
    body_domain: Domain | None = None
    head_domain: Domain | None = None


@dataclass(frozen=True)
class LoopDelta:
    """Bounds on an increment, as (slope, offset) in the iteration number."""

    lower: tuple[int, int] = (0, 0)
    upper: tuple[int, int] = (0, 0)

    @classmethod
    def constant(cls, lo, hi):
        return cls((0, lo), (0, hi))

    def add(self, other):
        return LoopDelta(tuple(a + b for a, b in zip(self.lower, other.lower)),
                         tuple(a + b for a, b in zip(self.upper, other.upper)))

    def hull(self, other):
        return LoopDelta(tuple(map(min, self.lower, other.lower)),
                         tuple(map(max, self.upper, other.upper)))

    def scale(self, factor):
        lo, hi = (self.lower, self.upper) if factor >= 0 else (self.upper, self.lower)
        return LoopDelta(tuple(factor * n for n in lo), tuple(factor * n for n in hi))

    def constant_value(self):
        if self.lower == self.upper and self.lower[0] == 0:
            return self.lower[1]
        return None

    def prefix(self, turns):
        def sums(line):
            slope, offset = line
            candidates = (0, turns)
            if slope:
                crossing = -offset // slope
                candidates += tuple(max(0, min(turns, n)) for n in (crossing, crossing + 1))
            return tuple(n * offset + slope * (n * (n - 1) // 2) for n in candidates)

        return min(sums(self.lower)), max(sums(self.upper))
