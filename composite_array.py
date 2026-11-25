"""Compressed composite-array storage and compiler integration.

The abstract state is a persistent tree. Array nodes use interval segments,
including arrays of records and arrays of arrays; neither zero initialization
nor a dynamic weak update enumerates all elements of a large entity.
"""
from dataclasses import dataclass, field, replace
from functools import reduce
from math import gcd
from array_domain import ArrayDomain, Segment
from value_domain import Domain
from table_interface import c_field


TEXT_SUMMARY_BYTES = 256


@dataclass(frozen=True)
class TextContent:
    length: Domain
    data: bytes


@dataclass(frozen=True)
class RecordValue:
    fields: tuple

    def join(self, other):
        if tuple(k for k, _ in self.fields) != tuple(k for k, _ in other.fields):
            raise ValueError("record schemas differ")
        return RecordValue(tuple((k, join_value(a, b))
                                 for (k, a), (_, b) in zip(self.fields, other.fields)))


def join_value(left, right):
    if isinstance(left, TextContent) or isinstance(right, TextContent):
        if left == right:
            return left
        return text_length(left) | text_length(right)
    return left.join(right) if isinstance(left, (RecordValue, ArrayDomain)) else left | right


def text_length(value):
    return value.length if isinstance(value, TextContent) else value


def join_values(values):
    return reduce(join_value, values)


@dataclass(frozen=True)
class TreeArray(ArrayDomain):
    """ArrayDomain protocol for recursively structured element values."""

    def __post_init__(self):
        if type(self.length) is not int or self.length < 0:
            raise ValueError("invalid array length")
        end, merged = 0, []
        for s in self.segments:
            if s.start != end or not end < s.stop <= self.length:
                raise ValueError("segments must cover every array element exactly once")
            if merged and merged[-1].domain == s.domain:
                merged[-1] = Segment(merged[-1].start, s.stop, s.domain)
            else:
                merged.append(s)
            end = s.stop
        if end != self.length:
            raise ValueError("segments must cover every array element")
        object.__setattr__(self, "segments", tuple(merged))

    @classmethod
    def filled(cls, length, element, value):
        return cls(length, element, (Segment(0, length, value),) if length else ())

    @classmethod
    def unknown(cls, length, element):
        return cls.filled(length, element, element)

    @classmethod
    def from_values(cls, length, element, values):
        values = tuple(values)
        if len(values) != length:
            raise ValueError("initializer length differs")
        return cls(length, element, tuple(Segment(i, i + 1, value) for i, value in enumerate(values)))

    def read(self, indices):
        indices = self._indices(indices)
        if not indices:
            raise ValueError("unreachable composite read")
        return join_values(s.domain for s in self.segments
                           if indices & Domain.of((s.start, s.stop - 1)))

    def _replace(self, start, stop, replacement):
        self._range(start, stop)
        result = []
        for s in self.segments:
            if s.start < start:
                result.append(Segment(s.start, min(start, s.stop), s.domain))
        result.extend(replacement)
        for s in self.segments:
            if s.stop > stop:
                result.append(Segment(max(stop, s.start), s.stop, s.domain))
        return TreeArray(self.length, self.element, tuple(result))

    def fill(self, start, stop, value):
        return self._replace(start, stop, (Segment(start, stop, value),) if start != stop else ())

    def transform(self, indices, function, weak=False):
        indices = self._indices(indices)
        if not indices:
            return self
        # Intersect each segment with the domain's interval overapproximation.
        # Discrete polynomial holes may be updated weakly, never excluded.
        exact = indices.lo == indices.hi and not weak
        result = self
        for lo, hi in indices.ranges:
            for s in self.segments:
                start, stop = max(s.start, lo), min(s.stop, hi + 1)
                if start < stop:
                    value = function(s.domain)
                    if not exact:
                        value = join_value(s.domain, value)
                    result = result.fill(start, stop, value)
        return result

    def write(self, indices, value):
        return self.transform(indices, lambda old: value)

    def weak_write(self, indices, value):
        return self.transform(indices, lambda old: value, True)

    def havoc(self, start=0, stop=None):
        return self.fill(start, self.length if stop is None else stop, self.element)

    def havoc_indices(self, indices):
        return self.weak_write(indices, self.element)

    def slice(self, start, stop):
        self._range(start, stop)
        return TreeArray(stop - start, self.element,
                         tuple(Segment(max(s.start, start) - start, min(s.stop, stop) - start, s.domain)
                               for s in self.segments if s.start < stop and s.stop > start))

    def copy_from(self, source, *, dst_start, src_start, count):
        self._range(dst_start, dst_start + count)
        snapshot = source.slice(src_start, src_start + count)
        return self._replace(dst_start, dst_start + count,
                             tuple(Segment(dst_start + s.start, dst_start + s.stop, s.domain)
                                   for s in snapshot.segments))

    def join(self, other):
        if self.length != other.length or self.element != other.element:
            raise ValueError("array schemas differ")
        left, right, start, result = 0, 0, 0, []
        while start < self.length:
            a, b = self.segments[left], other.segments[right]
            stop = min(a.stop, b.stop)
            result.append(Segment(start, stop, join_value(a.domain, b.domain)))
            start = stop
            left += stop == a.stop
            right += stop == b.stop
        return TreeArray(self.length, self.element, tuple(result))


def read_path(value, path):
    for part in path:
        value = dict(value.fields)[part] if isinstance(part, str) else value.read(part)
    return value


def project_scalar_column(contents, fields):
    """Read a scalar leaf from each record segment, keeping the row boundaries."""
    element = read_path(contents.element, fields)
    if not isinstance(element, Domain):
        return None
    segments = tuple(Segment(s.start, s.stop, read_path(s.domain, fields)) for s in contents.segments)
    return ArrayDomain(contents.length, element, segments)


def write_path(value, path, replacement, weak=False):
    if not path:
        return join_value(value, replacement) if weak else replacement
    head, *tail = path
    if isinstance(head, str):
        return RecordValue(tuple((k, write_path(v, tail, replacement, weak) if k == head else v)
                                 for k, v in value.fields))
    if not isinstance(value, TreeArray):
        return value.weak_write(head, replacement) if weak else value.write(head, replacement)
    return value.transform(head, lambda old: write_path(old, tail, replacement, weak))


@dataclass(frozen=True)
class Projection:
    root: object
    steps: tuple
    resolved: bool = False
    packed_target: object = None
    captured: tuple | None = None
    index_forms: tuple = field(default=(), compare=False, hash=False)
    entry_end: int | None = None
    entry_index: object = None
    index_values: tuple = field(default=(), compare=False, hash=False)
    nested_entry_indices: tuple = ()
    descriptor_epoch: object = field(default=None, compare=False, hash=False)
    absolute_index: object = field(default=None, compare=False, hash=False)
    always_returns: bool = False

    @property
    def first_index(self):
        return self.index_values[0] if self.index_values else None


def install(g):
    C, A, S, R = g["Compiler"], g["Array"], g["Scalar"], g["Record"]
    Symbol, Span, Slice, HeapArray = (g[k] for k in ("Symbol", "Span", "Slice", "HeapArray"))
    Link = g["RecordLink"]
    Interface = g["Interface"]
    Value, Node, ZERO, BOOL = (g[k] for k in ("Value", "Node", "ZERO", "BOOL"))
    uq, size = g["unqualified"], g["storage_size"]
    old = {}

    def wrap(name):
        def decorate(fn):
            old[name] = getattr(C, name)
            setattr(C, name, fn)
            return fn
        return decorate

    def composite(ty):
        return isinstance(ty, A) and not isinstance(ty.element, S)

    def schema(ty, zero=False):
        ty = uq(ty)
        if isinstance(ty, S):
            return g["zero_domain"](ty) if zero else ty.domain
        if isinstance(ty, Link):
            return frozenset((None,))
        if isinstance(ty, Slice):
            return ZERO if zero else ty.domain
        if isinstance(ty, R):
            return RecordValue(tuple((name, schema(field, zero)) for name, field in ty.fields))
        return TreeArray.filled(ty.length, schema(ty.element), schema(ty.element, zero))

    def encode_result_content(value):
        if isinstance(value, TextContent) and len(value.data) <= TEXT_SUMMARY_BYTES:
            return {"text_bytes": value.data.hex()}
        if isinstance(value, ArrayDomain):
            return {"array": [[segment.start, segment.stop, encode_result_content(segment.domain)]
                              for segment in value.segments]}
        if isinstance(value, RecordValue):
            return {"record": [encode_result_content(content) for _, content in value.fields]}
        if isinstance(value, g["FloatDomain"]):
            return {"float": value.to_dict()}
        return g["encode_domain"](text_length(value))

    def result_content_size(value):
        if isinstance(value, TextContent) and len(value.data) <= TEXT_SUMMARY_BYTES:
            return 1 + len(value.data)
        if isinstance(value, ArrayDomain):
            return 1 + sum(result_content_size(segment.domain) for segment in value.segments)
        if isinstance(value, RecordValue):
            return 1 + sum(result_content_size(content) for _, content in value.fields)
        return 1

    def decode_result_content(ty, encoded, budget):
        ty = uq(ty)
        budget[0] -= 1
        error = g["InterfaceError"]
        if budget[0] < 0:
            raise error("record result summary exceeds the content limit")
        if isinstance(ty, A):
            g["interface_fields"](encoded, ("array",), "array result field")
            raw = encoded["array"]
            if not isinstance(raw, list) or not 0 < len(raw) <= budget[0]:
                raise error("invalid array result segments")
            segments = []
            end = 0
            for entry in raw:
                if (not isinstance(entry, list) or len(entry) != 3 or type(entry[0]) is not int
                        or type(entry[1]) is not int or entry[0] != end or not end < entry[1] <= ty.length):
                    raise error("invalid array result segment")
                end = entry[1]
                segments.append(Segment(entry[0], end, decode_result_content(ty.element, entry[2], budget)))
            if end != ty.length:
                raise error("array result segments do not cover the declared length")
            return TreeArray(ty.length, schema(ty.element), tuple(segments))
        if isinstance(ty, R):
            g["interface_fields"](encoded, ("record",), "array result record")
            raw = encoded["record"]
            if not isinstance(raw, list) or len(raw) != len(ty.fields):
                raise error("invalid array result record")
            return RecordValue(tuple((name, decode_result_content(field, item, budget))
                                     for (name, field), item in zip(ty.fields, raw)))
        if not isinstance(ty, (S, Slice)):
            raise error("unsupported record result field")
        if isinstance(ty, Slice) and isinstance(encoded, dict) and "text_bytes" in encoded:
            g["interface_fields"](encoded, ("text_bytes",), "text result field")
            raw = encoded["text_bytes"]
            if not isinstance(raw, str) or len(raw) > 2 * min(budget[0], TEXT_SUMMARY_BYTES):
                raise error("invalid text result bytes")
            try:
                data = bytes.fromhex(raw)
            except ValueError as exc:
                raise error("invalid text result bytes") from exc
            length = Domain.point(len(data))
            if (data.hex() != raw or not length.within(ty.domain)
                    or any(not (ty.element.domain & Domain.point(byte)) for byte in set(data))):
                raise error("text result bytes exceed the declared contract")
            budget[0] -= len(data)
            return TextContent(length, data)
        if isinstance(ty, S) and ty.floating:
            g["interface_fields"](encoded, ("float",), "floating record result field")
            try:
                domain = g["FloatDomain"].from_dict(encoded["float"])
            except (TypeError, ValueError) as exc:
                raise error("invalid floating record result field") from exc
            if domain.bits != ty.bits:
                raise error("invalid floating record result width")
        else:
            domain = g["decode_domain"](encoded)
        if not domain or not domain.within(ty.domain):
            raise error("record result field exceeds its declared contract")
        return domain

    g["encode_result_content"] = encode_result_content
    g["decode_result_content"] = decode_result_content
    g["join_result_contents"] = join_values
    g["result_content_size"] = result_content_size

    def zero_check(self, ty, token):
        ty = uq(ty)
        if isinstance(ty, S):
            self.range_check(token, Value(g["zero_domain"](ty), "0", boolean=ty.boolean, enum=ty.enum,
                                          character=ty.character), ty)
        elif isinstance(ty, Link):
            return
        elif isinstance(ty, Slice):
            if not ZERO.within(ty.domain):
                token.error("default string field length does not fit its contract")
        elif isinstance(ty, A):
            zero_check(self, ty.element, token)
        else:
            for _, field in ty.fields:
                zero_check(self, field, token)

    def has_projection(sym):
        return sym is not None and sym.projection is not None

    def root_value(self, sym, state):
        root = sym.projection.root
        values = []
        if isinstance(root.ty, Slice):
            if root.uid in state.arrays:
                return state.arrays[root.uid]
            span = root.ty.projection
            length = span.length.domain if span is not None else state.values[root.uid]
            positions = span.positions() if span is not None else state.regions.get(root.uid, ())
            for owner, offset in positions:
                contents = self.actual_array(state, owner)
                if offset is not None and offset.lo == offset.hi and length.lo == length.hi:
                    values.append(contents.slice(offset.lo, offset.lo + length.lo))
                else:
                    maximum = root.ty.domain.hi
                    candidates = self.array_indices(contents, offset, Domain.of((0, maximum - 1)))
                    value = contents.read(candidates) if candidates else schema(root.ty.element)
                    values.append(TreeArray.filled(maximum, schema(root.ty.element), value))
            return join_values(values)
        for owner in self.physical_places(root, state):
            values.append(self.actual_array(state, owner))
        return join_values(values)

    def projected(self, sym, state):
        if sym.projection.captured is not None:
            return join_values(projected(self, target, state) for target in sym.projection.captured)
        return read_path(root_value(self, sym, state), sym.projection.steps)

    def set_projected(self, sym, state, value, prune_token=None, effect="@prune", *, record_write=True):
        projection = sym.projection
        if projection.captured is not None:
            for target in projection.captured:
                updated = join_value(projected(self, target, state), value) if len(projection.captured) > 1 else value
                state = set_projected(self, target, state, updated, prune_token, effect, record_write=record_write)
            return state

        def note_write(state, owner, *, steps=projection.steps):
            if not record_write:
                return state
            if prune_token is None:
                if not isinstance(sym.ty, S) or not isinstance(owner.ty, (A, Slice)) or owner.bit_overlay:
                    steps = None
                return self.write_actual_storage(state, owner, mutation_steps=steps)
            if effect == "@attach":
                return self.note_attachment(state, owner, value, prune_token, head=True)
            return self.note_chain_prune(state, owner, prune_token, effect=effect)

        row, indices = sym, None
        if (record_write and prune_token is None and isinstance(sym.ty, S)
                and isinstance(projection.root.ty, (A, Slice)) and len(projection.steps) > 1
                and all(isinstance(step, Domain) for step in projection.steps)):
            row_type = projection.root.ty
            for _ in projection.steps[:-1]:
                row_type = uq(row_type.element)
            row = replace(sym, ty=row_type, projection=replace(projection, steps=projection.steps[:-1]))
            indices = projection.steps[-1]
        elif isinstance(sym.ty, A) and isinstance(sym.ty.element, S):
            indices = Domain.of((0, sym.ty.length - 1))

        if (record_write and prune_token is None and projection.packed_target is not None
                and isinstance(projection.root.ty, (A, Slice)) and projection.root.uid not in self.alias_members
                and projection.root.link_access is None):
            row_type = projection.root.ty
            for step in projection.steps[:-1]:
                row_type = uq(dict(row_type.fields)[step] if isinstance(step, str) else row_type.element)
            row = replace(sym, ty=row_type, projection=replace(projection, steps=projection.steps[:-1]))
            indices = projection.steps[-1]

        if isinstance(projection.root.ty, Slice) and projection.root.uid not in state.arrays:
            span = projection.root.ty.projection
            positions = span.positions() if span is not None else state.regions.get(projection.root.uid, ())
            field = self.fixed_projected_field(sym, state=state) if record_write and prune_token is None else None
            physical_row = (self.physical_array_row(row, state)
                            if record_write and prune_token is None and indices is not None
                            else None)
            if physical_row is not None:
                state = self.write_actual_storage(state, physical_row, indices)
            elif field is not None:
                state = self.write_actual_storage(state, field[0], field_path=field[1])
            else:
                for owner, offset in positions:
                    first, *tail = projection.steps
                    indices = self.array_indices(self.actual_array(state, owner), offset, first)
                    state = note_write(state, owner, steps=(indices, *tail))
            for owner, offset in positions:
                contents = self.actual_array(state, owner)
                first, *tail = projection.steps
                index = self.array_indices(contents, offset, first)
                updated = write_path(contents, (index, *tail), value, len(positions) != 1 or offset is None or offset.lo != offset.hi)
                state = state.with_array(owner, updated)
            return state
        roots = self.physical_places(projection.root, state)
        if not record_write:
            pass
        elif prune_token is None and indices is not None and self.array_row_write_key(row) is not None:
            state = self.write_actual_storage(state, row, indices)
        elif (prune_token is None and isinstance(projection.root.ty, (A, Slice, HeapArray))
                and isinstance(projection.root.ty.element, S) and len(projection.steps) == 1
                and not projection.root.bit_overlay and not getattr(projection.root.ty, "packed_bits", False)):
            for root in roots:
                state = self.write_actual_storage(state, root, projection.steps[0])
        elif prune_token is None:
            field = self.fixed_projected_field(sym)
            if field is not None and len(roots) == 1 and roots[0] == field[0]:
                state = self.write_actual_storage(state, field[0], field_path=field[1])
            elif (isinstance(sym.ty, S) and isinstance(projection.root.ty, (A, Slice))
                  and not projection.root.bit_overlay):
                for root in roots:
                    state = note_write(state, root)
            else:
                state = self.write_storage(state, projection.root)
        else:
            for root in roots:
                state = note_write(state, root)
        for root in roots:
            contents = self.actual_array(state, root)
            updated = write_path(contents, projection.steps, value, len(roots) != 1)
            state = state.with_array(root, updated)
        return state

    def havoc_projected_storage(self, state, sym):
        return set_projected(self, sym, state, schema(sym.ty))

    C.havoc_projected_storage = havoc_projected_storage

    def havoc_array_field(self, state, root, steps, ty):
        def leaf(place):
            projection = place.projection
            if projection is None:
                projection = Projection(place, steps, resolved=True)
            else:
                captured = projection.captured
                projection = replace(projection, steps=(*projection.steps, *steps),
                                     captured=None if captured is None else tuple(leaf(target) for target in captured))
            return replace(place, ty=ty, projection=projection)

        return havoc_projected_storage(self, state, leaf(root))

    C.havoc_array_field = havoc_array_field

    def materialize(self, sym, state, *, allow_effectful_links=False, require_proven=False, propagate_exit=False):
        if sym.link_access is not None:
            parent, name = sym.link_access
            place = materialize(self, parent, state, require_proven=require_proven,
                                allow_effectful_links=allow_effectful_links and propagate_exit,
                                propagate_exit=propagate_exit)
            if has_projection(place) and place.projection.always_returns:
                return replace(sym, projection=place.projection, c_place=place.c)
            op = "->" if isinstance(parent.ty, Link) else "."
            access = (place, name) if propagate_exit else sym.link_access
            return replace(sym, c_place=f"({place.c}){op}{c_field(name)}", link_access=access)
        if not has_projection(sym) or sym.projection.resolved:
            return sym
        projection = sym.projection
        root = projection.root
        stable_slice = allow_effectful_links and isinstance(root.ty, Slice) and root.link_access is None
        captured_alias = allow_effectful_links and isinstance(root.ty, A) and root.uid in self.alias_members
        entry = replace(state) if root.link_access is not None or stable_slice or captured_alias else None
        if root.link_access is not None:
            root = materialize(self, root, state, allow_effectful_links=allow_effectful_links,
                               require_proven=require_proven, propagate_exit=propagate_exit)
            if has_projection(root) and root.projection.always_returns:
                return replace(sym, c_place=root.c,
                               projection=replace(projection, resolved=True, always_returns=True))
        self.require_live(root, state, projection.steps[0].token if isinstance(projection.steps[0], Node)
                          else self.nodes[0].token)
        if root.uid in self.alias_members and self.contains_record_links(root.ty) and not captured_alias:
            projection.steps[0].token.error("linked array projection through a rebindable alias is not supported")
        ty, code, steps, packed_target = root.ty, root.c, [], None
        length_value = None
        if isinstance(ty, Slice):
            span = ty.projection
            owners = span.owners if span is not None else state.borrows.get(root.uid, ())
            for owner in owners:
                self.require_target_live(owner, state, projection.steps[0].token)
            if span is None:
                length_value = Value(state.values[root.uid], f'(lan_int){root.c}.len', difference=g['Difference'](root.uid))
                span = Span(ty.element, length_value, root.depth, code=root.c, readonly=ty.readonly)
            length_value = self.capture_value(span.length, state)
            code = span.c_data()
        ordered = root.link_access is not None
        index_forms = []
        index_values = []
        nested_entry_indices = []
        stable_local = (isinstance(root.ty, A) and root.link_access is None and not root.borrowed
                        and not root.heap_owner and root.uid not in self.param_places)
        captured_array = isinstance(root.ty, A) and (root.link_access is not None or captured_alias)
        for step in projection.steps:
            if (not isinstance(step, str) and self.contains_record_links(root.ty) and self.has_effects(step, allow_debug_checks=True)
                    and not (allow_effectful_links and (stable_local or captured_array) or stable_slice)):
                step.token.error("linked array projection requires an index without side effects")
            if isinstance(step, str):
                ty = uq(dict(ty.fields)[step])
                code = f"({code}).{c_field(step)}"
                steps.append(step)
                index_forms.append(None)
                index_values.append(None)
            else:
                debug = self.debug_bounds
                if require_proven:
                    self.debug_bounds = False
                try:
                    index = self.expr(step, state)
                finally:
                    self.debug_bounds = debug
                if propagate_exit and index.always_returns:
                    exit_code = f"((void)({code}), (void)({index.code}), 0)"
                    return replace(sym, c_place=exit_code,
                                   projection=replace(projection, resolved=True, always_returns=True))
                index_values.append(index)
                integer = not (index.floating or index.boolean or index.enum or index.character)
                maximum = ty.length if isinstance(ty, A) else length_value.domain.hi
                proven = integer and index.domain.lo >= 0 and (index.domain.hi < ty.length if isinstance(ty, A)
                          else self.prove_order(step.token, index, "<", length_value, state))
                if not proven and (require_proven or not integer or not self.debug_bounds):
                    step.token.error(f"cannot prove index {index.domain} fits array length {getattr(ty, 'length', state.values.get(root.uid))}")
                valid = index.domain & Domain.of((0, maximum - 1)) if self.debug_bounds else index.domain
                if not valid:
                    step.token.error("projected index cannot be in bounds")
                if proven:
                    length = ty.length if isinstance(ty, A) else length_value.domain
                    self.trace.append(f"  verify projection index id#{root.uid} at "
                                      f"{step.token.line}:{step.token.col}: {index.domain} fits length {length} => proven")
                else:
                    self.trace.append(f"  debug check projection index at {step.token.line}:{step.token.col}: {index.domain}")
                if steps and isinstance(ty, A) and valid.lo != valid.hi:
                    entry_index = array_entry_index(self, ty.length, ZERO, index)
                    if entry_index is not None:
                        nested_entry_indices.append((len(steps), entry_index))
                steps.append(valid)
                index_forms.append(index.affine)
                ordered |= self.debug_bounds or self.has_effects(step)
                base = f"lan_projection_base_{step.token.line}_{step.token.col}"
                position = f"lan_projection_index_{step.token.line}_{step.token.col}"
                address_index, direct_index = position, index.code
                if self.debug_bounds:
                    check = self.helper("debug_index", None)
                    length_code = f"UINT64_C({ty.length})" if isinstance(ty, A) else length_value.code
                    address_index = f"{check}({position}, {length_code})"
                    direct_index = f"{check}({index.code}, {length_code})"
                elif not self.literal_endpoint(step):
                    if isinstance(ty, A):
                        length_code = f"UINT64_C({ty.length})"
                        address_index = self.assumed_index(position, length_code)
                        direct_index = self.assumed_index(index.code, length_code)
                    elif not ordered:
                        direct_index = self.assumed_index(index.code, length_value.code)
                if isinstance(ty, A) and ty.packed_bits:
                    # Scalar packed indices use the existing bit helpers, and
                    # the parent reference supplies byte storage.
                    packed_target = (code, direct_index)
                    helper = self.helper('bit_read', None)
                    if ordered:
                        self.helper("global_temp", None, ("const uint8_t *", base))
                        self.helper("global_temp", None, ("lan_int", position))
                        code = f"({base} = ({code}), {position} = ({index.code}), {helper}({base}, {address_index}))"
                    else:
                        code = f"{helper}({code}, {direct_index})"
                else:
                    if ordered:
                        self.helper("global_temp", None, (f"{ty.element.c} *", base))
                        self.helper("global_temp", None, ("lan_int", position))
                        code = f"(*({base} = ({code}), {position} = ({index.code}), &{base}[{address_index}]))"
                    else:
                        code = f"({code})[{direct_index}]"
                ty = ty.element
        object.__setattr__(sym, "c_place", code)
        resolved = replace(sym, c_place=code, projection=Projection(
            root, tuple(steps), True, packed_target, index_forms=tuple(index_forms), index_values=tuple(index_values),
            nested_entry_indices=tuple(nested_entry_indices)))
        if entry is not None:
            targets = captured_projection_owners(self, resolved, entry)
            if targets:
                resolved = replace(resolved, projection=replace(resolved.projection, captured=targets))
            elif not (stable_slice and root.uid in entry.arrays):
                self.nodes[0].token.error("cannot prove the backing storage of the indirect projection")
        return resolved

    @wrap("place")
    def place(self, node):
        if node.kind == "index":
            host = self.place(node.args[0])
            if host is not None and isinstance(host.ty, (A, Slice)) and (
                    not isinstance(host.ty.element, S) or isinstance(host.ty, A) and has_projection(host)):
                if has_projection(host):
                    p = host.projection
                    root, steps = p.root, p.steps
                else:
                    root, steps = host, ()
                ty = host.ty.element
                return Symbol(host.name + "_element", root.uid, ty, host.depth,
                              heap_owner=host.heap_owner,
                              readonly=host.readonly or isinstance(host.ty, Slice) and host.ty.readonly,
                              borrowed=host.borrowed,
                              projection=Projection(root, steps + (node.args[1],)))
        if node.kind in ("member", "length") and self.enum_member_type(node) is None:
            host = self.place(node.args[0])
            if has_projection(host) and isinstance(host.ty, R):
                name = "len" if node.kind == "length" else node.args[1].text
                fields = dict(host.ty.fields)
                if name not in fields:
                    node.token.error(f"record has no field {name!r}")
                field = fields[name]
                p = host.projection
                return Symbol(name, p.root.uid, uq(field), host.depth, heap_owner=host.heap_owner,
                              readonly=host.readonly or isinstance(field, g['ReadOnly']), borrowed=host.borrowed,
                              projection=Projection(p.root, p.steps + (name,)))
        return old["place"](self, node)

    C.comparison_place = materialize

    def projected_link_key(self, sym, state):
        projection = sym.projection
        root = projection.root
        owners = (root.ty.projection.owners if isinstance(root.ty, Slice) and root.ty.projection is not None
                  else state.borrows.get(root.uid, ()))
        def topology_epoch(owner):
            return (None if owner.uid in self.chain_fields and isinstance(owner.ty, Slice) and not owner.ty.prune
                    else state.epochs.get(owner.uid))

        epochs = tuple((owner.uid, topology_epoch(owner)) for owner in sorted(owners, key=lambda owner: owner.uid))
        steps = tuple(self.predicate_key(step, state) if isinstance(step, Node) else step for step in projection.steps)
        return "projected_link", root.uid, topology_epoch(root), epochs, steps

    def projected_link_guard_key(self, access, field, state):
        if has_projection(access):
            return projected_link_key(self, access, state)
        # Different aliases can have identical abstract index domains while
        # pointing at different records. Tie the guard to the actual access.
        indirect_epochs = tuple(sorted(state.epochs.items())) if access.link_access is not None else ()
        return ("projected_link", self.chain_access_key("captured_link", access, state),
                projected_link_key(self, field, state), indirect_epochs)

    C.projected_link_guard_key = projected_link_guard_key

    @wrap("link_targets")
    def link_targets(self, sym, state):
        fields = self.physical_places(sym, state, allow_null=True)
        if not any(has_projection(field) for field in fields):
            return old["link_targets"](self, sym, state)
        result = set()
        for field in fields:
            if not has_projection(field):
                result.update(old["link_targets"](self, field, state))
                continue
            key = projected_link_guard_key(self, sym, field, state)
            targets = projected(self, materialize(self, field, state), state)
            targets = frozenset(self.null_symbol if target is None else target for target in targets)
            known = state.predicates.get(key)
            if known is not None:
                targets = targets - {self.null_symbol} if known else targets & {self.null_symbol}
            result.update(targets)
        return frozenset(result)

    def content_link_targets(value):
        if isinstance(value, frozenset):
            yield from (target for target in value if target is not None)
        elif isinstance(value, RecordValue):
            for _, field in value.fields:
                yield from content_link_targets(field)
        elif isinstance(value, TreeArray):
            for segment in value.segments:
                yield from content_link_targets(segment.domain)

    def all_equal(parts):
        result = Domain.point(1)
        for part in parts:
            if part == ZERO:
                return ZERO
            if part == BOOL:
                result = BOOL
        return result

    def aligned_segments(a, b):
        left_segments, right_segments = iter(a.segments), iter(b.segments)
        x, y = next(left_segments, None), next(right_segments, None)
        start = 0
        while x is not None and y is not None:
            end = min(x.stop, y.stop)
            yield start, end, x.domain, y.domain
            start = end
            if x.stop == end:
                x = next(left_segments, None)
            if y.stop == end:
                y = next(right_segments, None)

    def array_equal(element, a, b, compare_links=None):
        return all_equal(value_equal(element, x, y, compare_links) for _, _, x, y in aligned_segments(a, b))

    def refine_equal_array(self, element, a, b):
        segments = []
        for start, stop, x, y in aligned_segments(a, b):
            value = refine_equal_value(self, element, x, y)
            if value is None:
                return None
            segments.append(Segment(start, stop, value))
        return type(a)(a.length, a.element, tuple(segments))

    C.refine_equal_array = refine_equal_array

    def refine_equal_value(self, ty, a, b):
        ty = uq(ty)
        if isinstance(ty, S):
            if ty.floating:
                domain = a.without_nan() & b.without_nan()
                if a.zeros and b.zeros:
                    domain |= a.only_zero()
            else:
                domain = a & b
            return domain or None
        if isinstance(ty, Slice):
            length = text_length(a) & text_length(b)
            if not length or isinstance(a, TextContent) and isinstance(b, TextContent) and a.data != b.data:
                return None
            if isinstance(a, TextContent):
                return a
            return b if isinstance(b, TextContent) else length
        if isinstance(ty, R):
            fields = []
            for (name, field), (_, x), (_, y) in zip(ty.fields, a.fields, b.fields):
                value = refine_equal_value(self, field, x, y)
                if value is None:
                    return None
                fields.append((name, value))
            return RecordValue(tuple(fields))
        if isinstance(ty, A):
            return refine_equal_array(self, ty.element, a, b)
        return a

    def value_equal(ty, a, b, compare_links=None):
        ty = uq(ty)
        if isinstance(ty, S):
            if ty.floating:
                return a.compare("==", b)
            if not (a & b):
                return ZERO
            return Domain.point(1) if a.lo == a.hi == b.lo == b.hi else BOOL
        if isinstance(ty, R):
            return all_equal(value_equal(field, x, y, compare_links)
                             for (_, field), (_, x), (_, y) in zip(ty.fields, a.fields, b.fields))
        if isinstance(ty, A):
            return array_equal(ty.element, a, b, compare_links)
        if isinstance(ty, Link) and compare_links is not None:
            return compare_links(a, b)
        if isinstance(ty, Slice):
            if isinstance(a, TextContent) and isinstance(b, TextContent):
                return Domain.point(int(a.data == b.data))
            a, b = text_length(a), text_length(b)
            if not (a & b):
                return ZERO
            if a == b == ZERO:
                return Domain.point(1)
        return BOOL

    def aggregate_equality_paths(self, left, right, state, *, packed=False, ranked=False, remaining=2):
        repeated = self.repeated_allocation_uids(state, left.token)
        places = []
        for node in (left, right):
            sym = self.place(node) if packed or ranked else self.record_place(node)
            if (sym is None or sym.bit_overlay
                    or (sym.link_access is not None and not ranked)):
                return [state]
            if ranked and isinstance(sym.ty, Link):
                targets = tuple(self.link_targets(sym, state))
            elif has_projection(sym):
                sym = materialize(self, sym, state, allow_effectful_links=ranked)
                if remaining:
                    for step, form in zip(sym.projection.steps, sym.projection.index_forms):
                        if not isinstance(step, Domain):
                            continue
                        candidates = self.split_affine_value(step, form, state)
                        if len(candidates) != 1 or candidates[0] is not state:
                            return [path for candidate in candidates
                                    for path in aggregate_equality_paths(self, left, right, candidate,
                                                                         packed=packed, ranked=ranked,
                                                                         remaining=remaining - 1)]
                targets = captured_projection_owners(self, sym, state)
            else:
                targets = self.physical_places(sym, state, allow_null=True)
            if len(targets) != 1 or not targets[0].uid:
                return [state]
            target = targets[0]
            if target.chain_summary or target.heap_owner in repeated:
                return [state]
            if has_projection(target) and any(
                    not isinstance(step, str) and step.lo != step.hi for step in target.projection.steps):
                return [state]
            places.append(target)
        a, b = places
        if packed and a.ty.length != b.ty.length:
            return []
        x, y = (record_value(self, sym, state, track_text=True) for sym in places)

        def apply(path, sym, content):
            if has_projection(sym):
                root, steps = sym.projection.root, sym.projection.steps
                array = self.actual_array(path, root)
                return path.with_array(root, write_path(array, steps, content))
            if isinstance(sym.ty, S):
                return path.narrow(sym, content)
            if isinstance(sym.ty, Slice):
                if isinstance(content, TextContent):
                    return restore_record_text(self, sym, path, content)
                return path.narrow(sym, content)
            if isinstance(sym.ty, A):
                return self.restore_record_array(sym, path, content)
            if isinstance(sym.ty, Link):
                # Equality at a positive rank does not equate link addresses.
                return path
            for child, (_, value) in zip(sym.fields, content.fields):
                path = apply(path, child, value)
                if path is None:
                    break
            return path

        for sym, original, expected in ((a, x, y), (b, y, x)):
            narrowed = refine_equal_value(self, sym.ty, original, expected)
            if narrowed is None:
                return []
            state = apply(state, sym, narrowed)
            if state is None:
                return []
        return [state]

    C.aggregate_equality_paths = aggregate_equality_paths

    def ranked_content_equal(self, compare, state, rank, token):
        root = (None, None, rank)
        pending, results = [root], {}
        true = Domain.point(1)
        repeated = self.repeated_allocation_uids(state, token)
        while pending:
            key = pending[-1]
            if key in results:
                pending.pop()
                continue
            left, right, depth = key
            dependencies = set()

            def compare_links(xs, ys):
                if depth == 1:
                    return self.link_address_domain(xs, ys, repeated)
                if not xs or not ys:
                    return BOOL
                result = Domain()
                for x in xs:
                    for y in ys:
                        self.spend(token)
                        x_null, y_null = x is None or not x.uid, y is None or not y.uid
                        if x_null or y_null:
                            result |= true if x_null == y_null else ZERO
                        else:
                            pair = (x, y, depth - 1)
                            if pair not in results:
                                dependencies.add(pair)
                            result |= results.get(pair, BOOL)
                return result

            if key == root:
                result = compare(compare_links)
            else:
                x, y = (record_value(self, sym, state, track_text=True) for sym in (left, right))
                result = value_equal(left.ty, x, y, compare_links)
            # Unresolved child pairs contribute both outcomes until evaluated.
            if result == BOOL and dependencies:
                pending.extend(dependencies)
            else:
                results[key] = result
        return results[root]

    def compare_ranked_targets(self, left, right, state, rank, token):
        return ranked_content_equal(self, lambda links: links(left, right), state, rank + 1, token)

    C.compare_ranked_targets = compare_ranked_targets

    def compare_record_contents(self, left, right, state):
        return value_equal(left.ty, record_value(self, left, state, track_text=True),
                           record_value(self, right, state, track_text=True))

    C.compare_record_contents = compare_record_contents

    def window_segments(array, start, first, last, limit):
        for segment in array.segments:
            end = min(segment.stop, limit)
            if segment.start < end and (start is None
                    or start & Domain.of((segment.start - last, end - 1 - first))):
                yield segment

    def compare_span_contents(self, left, right, state, token, *, rank=0):
        def contents(span, count):
            portions = []
            positions = span.positions()
            end = self.span_end_domain(token, span, state, positions)
            for owner, start in positions:
                array = self.actual_array(state, owner)
                if end and end.hi < array.length:
                    array = array.slice(0, max(0, end.hi))
                if start is not None and start.lo == start.hi and 0 <= start.lo <= array.length - count:
                    portions.append(array.slice(start.lo, start.lo + count))
                    continue
                starts = self.bounded_window_values(start, 8) if start is not None else None
                if (starts and len(starts) < start.hi - start.lo + 1
                        and 0 <= starts[0] and starts[-1] <= array.length - count):
                    portions.extend(array.slice(value, value + count) for value in starts)
                    continue
                if start is not None and count <= 8:
                    bounds = Domain.of((0, array.length - 1))
                    positions = [start.affine(1, index) & bounds for index in range(count)]
                    if not all(positions):
                        return None
                    portions.append(type(array).from_values(count, array.element,
                                                            (array.read(selected) for selected in positions)))
                    continue
                if start is None or len(array.segments) > 64:
                    values = [segment.domain for segment in
                              window_segments(array, start, 0, count - 1, array.length)]
                    if not values:
                        return None
                    portions.append(type(array).filled(count, array.element, join_values(values)))
                    continue
                # The set of candidate segments changes only at these relative positions.
                cuts = {0, count}
                for segment in array.segments:
                    cuts.update(point for point in (segment.start - start.hi, segment.stop - start.lo)
                                if 0 < point < count)
                boundaries = sorted(cuts)
                segments = []
                for lo, hi in zip(boundaries, boundaries[1:]):
                    values = [segment.domain for segment in
                              window_segments(array, start, lo, hi - 1, array.length)]
                    if not values:
                        return None
                    segments.append(Segment(lo, hi, join_values(values)))
                portions.append(type(array)(count, array.element, tuple(segments)))
            if span.text:
                octets = self.types["uint8"].domain
                portions = [ArrayDomain(count, octets, portion.segments) for portion in portions]
            return join_values(portions) if portions else None

        def compare_prefix(count):
            if count == 0:
                return Domain.point(1)
            if isinstance(left.element, S) and left.element == right.element:
                indices = Domain.of((0, count - 1))
                element = left.element
                lhs = self.known_span_window(state, left, indices, element.domain)
                rhs = self.known_span_window(state, right, indices, element.domain)
                uniform = value_equal(element, lhs, rhs)
                if uniform != BOOL:
                    return uniform
            a, b = contents(left, count), contents(right, count)
            if a is None or b is None:
                return BOOL
            if rank:
                return ranked_content_equal(self, lambda links: array_equal(left.element, a, b, links),
                                            state, rank, token)
            return array_equal(left.element, a, b)

        common = min(left.length.domain.lo, right.length.domain.lo)
        prefix = compare_prefix(common)
        if prefix == ZERO:
            return ZERO
        if not self.prove_order(token, left.length, "==", right.length, state):
            return BOOL
        longest = max(left.length.domain.hi, right.length.domain.hi)
        if longest == common:
            return prefix
        return Domain.point(1) if compare_prefix(longest) == Domain.point(1) else BOOL

    C.compare_span_contents = compare_span_contents

    def record_link_targets(self, sym, state):
        return frozenset(content_link_targets(record_value(self, sym, state)))

    C.record_link_targets = record_link_targets

    def graph_array_contents(self, ty, nodes):
        def contents(ty):
            ty = uq(ty)
            if isinstance(ty, Link):
                return frozenset((self.null_symbol, nodes[ty.identity]))
            if isinstance(ty, R):
                return RecordValue(tuple((name, contents(field)) for name, field in ty.fields))
            if isinstance(ty, A):
                value = contents(ty.element)
                return TreeArray.filled(ty.length, schema(ty.element), value)
            return schema(ty)

        return contents(ty)

    C.graph_array_contents = graph_array_contents

    def span_link_targets(self, span, state, indices=None, *, absolute_limit=None):
        bounds = Domain.of((0, span.length.domain.hi - 1))
        indices = bounds if indices is None else indices & bounds
        if not indices:
            return frozenset()
        if absolute_limit is not None:
            return frozenset(
                target for owner, start in span.positions()
                for lo, hi in indices.ranges
                for segment in window_segments(self.actual_array(state, owner), start, lo, hi, absolute_limit)
                for target in content_link_targets(segment.domain))
        values = self.read_span_contents(state, span, indices)
        return frozenset(content_link_targets(values))

    C.span_link_targets = span_link_targets

    def initialize_chain_span(self, sym, state, token, payload_index=None):
        ty = sym.ty
        root = Symbol("chain_element", len(self.symbols) + 1, ty.element, sym.depth,
                      readonly=ty.readonly, borrowed=True)
        self.symbols.append(root)
        self.record_symbols(root, token, zero=False)
        state = self.initialize_record_parameter(root, state)
        state = self.initialize_chain_parameter(root, ty.chain_limit, state, token, shared=True,
                                                payload_index=payload_index)
        value = record_value(self, root, state)
        return state, TreeArray.filled(ty.domain.hi, schema(ty.element), value)

    C.initialize_chain_span = initialize_chain_span

    def initialize_graph_span(self, sym, state, token, payload_index=None):
        ty = sym.ty
        root = Symbol("graph_element", len(self.symbols) + 1, ty.element, sym.depth, readonly=ty.readonly, borrowed=True)
        self.symbols.append(root)
        self.record_symbols(root, token, zero=False)
        state = self.initialize_graph_parameter(root, state, token, payload_index=payload_index)
        return state, TreeArray.filled(ty.domain.hi, schema(ty.element), record_value(self, root, state))

    C.initialize_graph_span = initialize_graph_span

    def forget_chain_payload(value, record):
        fields = dict(record.fields)
        return RecordValue(tuple((name, uq(fields[name]).domain if isinstance(uq(fields[name]), S) else content)
                                 for name, content in value.fields))

    def havoc_chain_span_storage(self, state, root):
        self.note_transition_write(root)
        contents = state.arrays[root.uid]
        indices = Domain.of((0, contents.length - 1))
        contents = contents.transform(indices, lambda value: forget_chain_payload(value, root.ty.element))
        return state.touch(root.uid).with_array(root, contents)

    C.havoc_chain_span_storage = havoc_chain_span_storage

    def havoc_chain_span(self, state, span, token, graph=False):
        targets = span_link_targets(self, span, state)
        indices = Domain.of((0, span.length.domain.hi - 1))
        for owner, offset in span.positions():
            state = self.write_actual_storage(state, owner)
            contents = self.actual_array(state, owner)
            selected = self.array_indices(contents, offset, indices)
            contents = contents.transform(selected, lambda value: forget_chain_payload(value, span.element))
            state = state.with_array(owner, contents)
        return self.havoc_chain_payload(state, targets, token, graph=graph)

    C.havoc_chain_span = havoc_chain_span

    def prune_chain_span(self, state, span, token, skip=False):
        roots = tuple(self.chain_nodes(state, span_link_targets(self, span, state), token))
        indices = Domain.of((0, span.length.domain.hi - 1))
        null = frozenset((self.null_symbol,))
        original = state
        effect = "@skip" if skip else "@prune"

        def new_next(value):
            targets = null
            if skip:
                children = content_link_targets(read_path(value, ("next",)))
                targets |= frozenset(self.chain_nodes(original, children, token))
            return write_path(value, ("next",), targets, weak=True)

        for owner, offset in span.positions():
            state = self.note_chain_prune(state, owner, token, effect=effect)
            contents = self.actual_array(state, owner)
            selected = self.array_indices(contents, offset, indices)
            contents = contents.transform(selected, new_next)
            state = state.with_array(owner, contents)
        if skip:
            return self.havoc_chain_skips(state, roots, token)
        for root in roots:
            state = self.prune_next(state, self.child_place(root, ("next",)), token, weak=True)
        return state

    C.prune_chain_span = prune_chain_span

    def prune_graph_span(self, state, span, token):
        roots = span_link_targets(self, span, state)
        indices = Domain.of((0, span.length.domain.hi - 1))
        links = {name for name, field in span.element.fields if isinstance(uq(field), Link)}
        null = frozenset((self.null_symbol,))

        def add_nulls(value):
            return RecordValue(tuple((name, content | null if name in links else content)
                                     for name, content in value.fields))

        for owner, offset in span.positions():
            state = self.note_chain_prune(state, owner, token)
            contents = self.actual_array(state, owner)
            selected = self.array_indices(contents, offset, indices)
            state = state.with_array(owner, contents.transform(selected, add_nulls))
        return self.prune_graph_links(state, roots, token)

    C.prune_graph_span = prune_graph_span

    def attach_chain_span(self, state, entry, span, targets, token, distance):
        roots = ()
        if distance != 1:
            suffix = span_link_targets(self, span, entry)
            roots = (tuple(self.chain_nodes(entry, suffix, token)) if distance is None
                     else self.chain_nodes_at(entry, suffix, distance - 2, token))
        indices = Domain.of((0, span.length.domain.hi - 1))
        for owner, offset in span.positions() if distance in (None, 1) else ():
            state = self.note_attachment(state, owner, targets, token, head=True)
            contents = self.actual_array(state, owner)
            selected = self.array_indices(contents, offset, indices)
            contents = contents.transform(selected, lambda value: write_path(value, ("next",), targets, weak=True))
            state = state.with_array(owner, contents)
        for root in roots:
            state = self.attach_next(state, self.child_place(root, ("next",)), targets, token, weak=True)
        return state

    C.attach_chain_span = attach_chain_span

    def attach_graph_span(self, state, span, targets, token):
        roots = span_link_targets(self, span, state)
        identity = next(iter(targets)).ty.identity
        links = {name for name, field in span.element.fields
                 if isinstance(uq(field), Link) and uq(field).identity == identity}
        indices = Domain.of((0, span.length.domain.hi - 1))
        if links:
            for owner, offset in span.positions():
                state = self.note_attachment(state, owner, targets, token, head=True)
                contents = self.actual_array(state, owner)
                selected = self.array_indices(contents, offset, indices)
                contents = contents.transform(selected, lambda value: RecordValue(tuple(
                    (name, content | targets if name in links else content) for name, content in value.fields)))
                state = state.with_array(owner, contents)
        return self.attach_graph_links(state, roots, targets, token)

    C.attach_graph_span = attach_graph_span

    def chain_projection_owners(sym, state):
        root = sym.projection.root
        span = root.ty.projection if isinstance(root.ty, Slice) else None
        return span.owners if span is not None else state.borrows.get(root.uid, (root,))

    def check_link_lifetimes(self, value, depth, state, token):
        for target in content_link_targets(value):
            if target.uid:
                self.require_target_live(target, state, token)
                if target.depth > depth:
                    token.error("record link would outlive its target storage")

    @wrap("store_link")
    def store_link(self, state, sym, targets, token, places=None, skip=False):
        if places is None:
            places = self.physical_places(sym, state)
        if not any(has_projection(place) for place in places):
            return old["store_link"](self, state, sym, targets, token, places, skip)
        for place in places:
            if has_projection(place):
                check_link_lifetimes(self, targets, place.depth, state, token)
                value = targets
                if len(places) != 1:
                    value |= projected(self, place, state)
                state = set_projected(self, place, state, value)
            else:
                value = targets | state.borrows[place.uid] if len(places) != 1 else targets
                state = old["store_link"](self, state, sym, value, token, (place,), skip)
        return state

    @wrap("assign_link")
    def assign_link(self, sym, op, node, paths, *, propagate_target=False):
        if not has_projection(sym):
            return old["assign_link"](self, sym, op, node, paths, propagate_target=propagate_target)
        if op.text != "=" or sym.readonly:
            op.error("record link assignment requires '=' and mutable storage")
        result = []
        nonnull = True
        exit_code = None

        def remember_exit(*lines):
            nonlocal exit_code
            candidate = "{ " + " ".join(lines) + " }"
            if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                exit_code = candidate

        for state in paths:
            field = materialize(self, sym, state, propagate_exit=True, allow_effectful_links=propagate_target)
            if field.projection.always_returns:
                remember_exit(f"(void)({field.c});")
                continue
            root = field.projection.root
            owners = (self.physical_places(root, state) if root.link_access is not None
                      else chain_projection_owners(field, state))
            attaching = any(self.attachment_parameter(owner) is not None for owner in owners)
            pruning = any(owner.uid in self.chain_fields for owner in owners)
            early = []
            if attaching:
                alias = g["BorrowRecord"](self.link_record(sym.ty, op), readonly=True)
                targets, code = self.alias_source(node, state, alias, op, capture_exit=early)
                code = f"({sym.ty.c})({code})"
            else:
                targets, code = self.link_initializer(sym.ty, node, state, op, capture_exit=early)
            if early:
                remember_exit(f"(void)(&({field.c}));", f"(void)({early[0].code});")
                continue
            effect = "@attach" if attaching else "@prune"
            if pruning and not attaching and targets != frozenset((self.null_symbol,)):
                if not self.skips_successor(field, node, state):
                    op.error("cannot rewrite a chain slice link to a non-null target outside its existing suffix")
                effect = "@skip"
            for owner in owners:
                check_link_lifetimes(self, targets, owner.depth, state, op)
            result.append(set_projected(self, field, state, targets, prune_token=op if pruning else None, effect=effect))
            nonnull = nonnull and bool(targets) and self.null_symbol not in targets
            emitted = field, code
        if not result:
            return [], [exit_code, "__builtin_unreachable();"]
        field, code = emitted
        return result, [self.link_assignment_code(field, op, node, code, nonnull=nonnull)]

    @wrap("copy_span_contents")
    def copy_span_contents(self, state, dst, src, token, *, correlated=False):
        if any(owner.uid in self.chain_fields for owner in dst.owners):
            token.error("cannot overwrite linked records through a chain slice")
        if self.contains_record_links(src.element) and src.length.domain.hi:
            value = self.read_span_contents(state, src, Domain.of((0, src.length.domain.hi - 1)))
            for owner in dst.owners:
                check_link_lifetimes(self, value, owner.depth, state, token)
        return old["copy_span_contents"](self, state, dst, src, token, correlated=correlated)

    @wrap("restrict")
    def restrict(self, state, expr, domain, *, enum_guard=False):
        sym = self.place(expr)
        if sym is None or not isinstance(sym.ty, S) or (sym.ty.result_arms is None and not enum_guard):
            return old["restrict"](self, state, expr, domain)
        targets = self.physical_places(sym, state, allow_null=True)
        if len(targets) != 1 or not has_projection(targets[0]):
            return old["restrict"](self, state, expr, domain)
        target = targets[0]
        projection = target.projection
        if projection.root.link_access is not None or any(
                isinstance(step, Node) and self.has_effects(step) for step in projection.steps):
            return old["restrict"](self, state, expr, domain)
        resolved = materialize(self, target, state)
        root = projection.root
        candidates = (self.split_slice_windows(root, expr.token, state)
                      if isinstance(root.ty, Slice) and projection.captured is None else [state])
        for index in resolved.projection.index_values:
            if index is None or index.affine is None:
                continue
            expanded = [part for candidate in candidates
                        for part in self.split_affine_value(index.domain, index.affine, candidate)]
            if expanded and len(expanded) <= self.PATH_LIMIT:
                candidates = expanded

        result = []
        for candidate in candidates:
            resolved = materialize(self, target, candidate)
            owners = captured_projection_owners(self, resolved, candidate)
            if len(owners) != 1 or any(isinstance(step, Domain) and step.lo != step.hi
                                       for step in owners[0].projection.steps):
                result.extend(old["restrict"](self, candidate, expr, domain))
                continue
            owner = owners[0]
            narrowed = projected(self, owner, candidate) & domain
            if narrowed:
                result.append(set_projected(self, owner, candidate, narrowed, record_write=False))
        return result

    @wrap("storage_value")
    def storage_value(self, sym, state):
        if not has_projection(sym):
            targets = self.physical_places(sym, state)
            if any(has_projection(target) for target in targets):
                domains = (projected(self, target, state) if has_projection(target) else state.values[target.uid]
                           for target in targets)
                ty = sym.ty
                return Value(g["domain_union"](domains), materialize(self, sym, state).c, ty.bits, ty.boolean,
                             enum=ty.enum, character=ty.character, signed=ty.signed,
                             result_arms=ty.result_arms, result_error_enums=ty.result_error_enums)
            return old['storage_value'](self, materialize(self, sym, state), state)
        sym = materialize(self, sym, state)
        ty = sym.ty
        return Value(projected(self, sym, state), sym.c, ty.bits, ty.boolean,
                     enum=ty.enum, character=ty.character, signed=ty.signed,
                     result_arms=ty.result_arms, result_error_enums=ty.result_error_enums)

    @wrap("store_value")
    def store_value(self, state, sym, value):
        if has_projection(sym):
            return set_projected(self, materialize(self, sym, state), state, value.domain)
        targets = self.physical_places(sym, state)
        if any(has_projection(target) for target in targets):
            for target in targets:
                if has_projection(target):
                    domain = value.domain
                    if len(targets) != 1:
                        domain |= projected(self, target, state)
                    state = set_projected(self, target, state, domain)
                else:
                    state = self.write_actual_storage(state, target)
                    state = state.update(target, state.values[target.uid] | value.domain)
            return state
        return old['store_value'](self, state, sym, value)

    @wrap("actual_array")
    def actual_array(self, state, owner):
        if has_projection(owner):
            return projected(self, materialize(self, owner, state), state)
        if composite(owner.ty):
            return state.arrays.get(owner.uid, schema(owner.ty))
        return old['actual_array'](self, state, owner)

    # Store projected ordinary-array contents back into the enclosing tree.
    old_with_array = g['PathState'].with_array
    def with_array(state, sym, contents):
        if has_projection(sym):
            p = sym.projection
            if not p.resolved:
                raise AssertionError('unresolved array projection')
            root = p.root
            base = state.arrays[root.uid] if root.uid in state.arrays else schema(root.ty)
            return old_with_array(state, root, write_path(base, p.steps, contents))
        return old_with_array(state, sym, contents)
    g['PathState'].with_array = with_array

    def slice_entry_end(self, root, offset, index_value, index_shift=0):
        origin = self.param_places.get(root.uid)
        if (index_value is None or offset is None or offset.lo != offset.hi
                or not isinstance(root.ty, Slice) or origin is None):
            return None
        linear = self.linear_value(index_value, prefer_symbolic=True)
        if linear is None:
            return None
        terms, bias = linear
        bias += offset.lo + index_shift
        if len(terms) != 1 or not -root.ty.domain.hi <= bias < 0:
            return None
        ((factors, scale),) = terms
        if scale != 1 or len(factors) != 1:
            return None
        ((atom, power),) = factors
        if power == 1 and self.entry_offset_origins.get(atom) == ("len", origin[0], 32):
            contract = self.functions[self.current][1]
            if g["valid_dependent_alias_index"](("const", -bias - 1), contract.params, contract.bounds,
                                               contract.parameter_expressions, root.ty.domain.lo, origin[0]):
                return -bias
        return None

    def array_entry_index(self, length, offset, index_value, slice_source=None):
        if index_value is None or offset is None or offset.lo != offset.hi:
            return None
        linear = self.linear_value(index_value, prefer_symbolic=True)
        if linear is None:
            return None
        terms, bias = linear
        bias += offset.lo
        contract = self.functions[self.current][1]
        params = contract.params
        if len(terms) == 1:
            ((factors, scale),) = terms
            if scale == 1 and len(factors) == 1:
                ((atom, power),) = factors
                origin = self.entry_offset_origins.get(atom, ())
                if power == 1 and origin and origin[0] == "param":
                    parameter = origin[1]
                    domain = uq(params[parameter]).domain
                    if 0 <= domain.lo + bias and domain.hi + bias < length:
                        return g["AliasParamIndex"](parameter, bias)
        if not terms:
            return None
        try:
            expression = self.inferred_offset_expression((terms, bias), params)
            bounded = g["valid_dependent_alias_index"](expression, params, contract.bounds,
                                                       contract.parameter_expressions, length, slice_source)
        except (ValueError, g["InterfaceError"]):
            return None
        if bounded:
            return g["AliasIndexExpression"](expression)
        return None

    def root_entry_index(self, root, offset, index_value):
        if root.uid not in self.param_places:
            return None
        if isinstance(root.ty, A):
            length = root.ty.length
        elif isinstance(root.ty, Slice):
            length = root.ty.domain.lo
        else:
            return None
        source = self.param_places[root.uid][0] if isinstance(root.ty, Slice) else None
        return array_entry_index(self, length, offset, index_value, source)

    def projected_entry_index(self, projection):
        end, index = projection.entry_end, projection.entry_index
        first = projection.steps[0] if projection.steps else None
        if end is None and index is None and isinstance(first, Domain) and first.lo != first.hi:
            end = slice_entry_end(self, projection.root, ZERO, projection.first_index)
            index = root_entry_index(self, projection.root, ZERO, projection.first_index)
        return end, index

    def finite_alias_indices(self, indices, budget=64):
        if indices.predicates:
            points = self.bounded_window_values(indices, budget)
            if points:
                return Domain.of(*((point, point) for point in points))
        return indices

    def alias_span_indices(self, contents, offset, indices):
        index = (self.array_indices(contents, offset, indices) if offset is None else
                 self.sum_index_domains(offset, indices) & Domain.of((0, contents.length - 1)))
        if (index and offset is not None and offset.lo != offset.hi and indices.lo != indices.hi
                and (offset.predicates or indices.predicates)):
            starts = self.bounded_window_values(offset, 64)
            steps = self.bounded_window_values(indices, 64 // len(starts)) if starts else None
            if steps:
                index &= Domain.of(*((start + step, start + step) for start in starts for step in steps))
            else:
                start_residue, start_modulus = offset.congruence()
                step_residue, step_modulus = indices.congruence()
                index = index.with_congruence(start_residue + step_residue, gcd(start_modulus, step_modulus))
        return finite_alias_indices(self, index)

    def alias_span_entry(self, index, offset, index_value, start, index_shift=0):
        if offset is not None and offset.lo != offset.hi and start is not None and index_value is not None:
            if self.related(start, index_value):
                form = self.combine_forms("+", start, index_value)
                domain = self.correlated_domain(form, start, index_value) if form is not None else None
                if domain:
                    index = finite_alias_indices(self, (index & domain.affine(1, index_shift)) or index)
            linear = self.combine_linear("+", start, index_value)
            if linear is not None:
                terms, bias = linear
                if not terms:
                    index = (Domain.point(bias + index_shift) & index) or index
                return index, ZERO, Value(index, "", linear=linear)
        return index, offset, index_value

    def span_element_places(self, span, indices, state, *, index_value=None, index_shift=0):
        result = []
        positions = span.positions()
        start = span.start if len(positions) == 1 else None
        for owner, offset in positions:
            contents = self.actual_array(state, owner)
            index = alias_span_indices(self, contents, offset, indices)
            if not index:
                continue
            index, entry_offset, entry_value = alias_span_entry(self, index, offset, index_value, start, index_shift)
            projection = owner.projection
            root = projection.root if projection is not None else owner
            steps = (*projection.steps, index) if projection is not None else (index,)
            nested_indices = ()
            if projection is None:
                entry_end = (slice_entry_end(self, root, entry_offset, entry_value, index_shift)
                             if index.lo != index.hi else None)
                entry_index = (root_entry_index(self, root, entry_offset, entry_value)
                               if index.lo != index.hi and not index_shift else None)
            else:
                entry_end, entry_index = projected_entry_index(self, projection)
                nested_indices = projection.nested_entry_indices
                if (root.uid in self.param_places and isinstance(owner.ty, A)
                        and index.lo != index.hi and not index_shift):
                    nested_index = array_entry_index(self, owner.ty.length, entry_offset, entry_value)
                    if nested_index is not None:
                        nested_indices = (*nested_indices, (len(steps) - 1, nested_index))
            result.append(Symbol("element", root.uid, span.element, root.depth, heap_owner=root.heap_owner,
                                 readonly=span.readonly, borrowed=root.borrowed,
                                 projection=Projection(root, steps, resolved=True, entry_end=entry_end,
                                                       entry_index=entry_index, nested_entry_indices=nested_indices)))
        return tuple(result)

    C.span_element_places = span_element_places

    def value_record(ty):
        ty = uq(ty)
        if isinstance(ty, R):
            return all(value_record(field) for _, field in ty.fields)
        if isinstance(ty, A):
            return value_record(ty.element)
        return isinstance(ty, (S, Link)) or isinstance(ty, Slice) and g["record_text_type"](ty)

    @wrap("alias_source")
    def alias_source(self, node, state, ty, token, *, capture_exit=None):
        source = self.place(node) if node is not None and node.kind != "null" else None
        if (has_projection(source) and not isinstance(source.ty, Link)
                and isinstance(ty, (g["BorrowArray"], g["BorrowScalar"], g["BorrowRecord"]))):
            if source.ty != self.alias_storage(ty) or source.readonly and not ty.readonly:
                token.error("projected alias requires the same storage type and compatible mutability")
            if isinstance(source.ty, R) and not value_record(source.ty):
                token.error("projected record aliases currently require value-only fields")
            propagate_exit = capture_exit is not None
            source = materialize(self, source, state, propagate_exit=propagate_exit,
                                 allow_effectful_links=propagate_exit and self.has_propagation(node))
            if source.projection.always_returns:
                capture_exit.append(Value(ZERO, source.c, effectful=True, always_returns=True))
                return frozenset(), source.c
            if source.projection.packed_target is not None:
                token.error("packed projected alias requires bit-address lowering")
            targets = captured_projection_owners(self, source, state)
            if not targets:
                token.error("cannot prove the backing storage of the projected alias")
            if self.contains_record_links(source.ty) and any(owner.uid in self.chain_fields for owner in targets):
                token.error("cannot capture a rebindable alias to an abstract chain or graph element")
            address = source.c if isinstance(source.ty, A) else f"&({source.c})"
            return frozenset(targets), address
        if (node is not None and node.kind == "index" and isinstance(ty, g["BorrowScalar"])
                and self.span_kind(node.args[0]) == "ordinary"):
            captured = []
            access = self.subscript(node, state, captured, propagate_exit=capture_exit is not None)
            if isinstance(access, Value):
                capture_exit.append(access)
                return frozenset(), access.code
            span, index, indices = access
            if span.element != ty.scalar or span.readonly and not ty.readonly:
                token.error("projected alias requires the same storage type and compatible mutability")
            targets = span_element_places(self, span, indices, state, index_value=captured[0])
            return frozenset(targets), f"&({span.c_data()}[{index}])"
        return old["alias_source"](self, node, state, ty, token, capture_exit=capture_exit)

    def alias_result_projection(self, sym, path, index_values):
        for step, index_value in zip(path, index_values):
            if isinstance(step, str):
                sym = self.child_place(sym, (step,))
                continue
            if has_projection(sym):
                projection = sym.projection
                root, steps = projection.root, projection.steps
                entry_end, entry_index = projection.entry_end, projection.entry_index
                nested_indices = projection.nested_entry_indices
            else:
                root, steps, entry_end, entry_index = sym, (), None, None
                nested_indices = ()
            element = sym.ty.element
            indices = Domain.point(step) if isinstance(step, int) else step
            indices = finite_alias_indices(self, indices & Domain.of((0, sym.ty.length - 1)))
            if index_value is not None and indices.lo != indices.hi:
                if not steps:
                    entry_index = root_entry_index(self, root, ZERO, index_value)
                elif root.uid in self.param_places:
                    nested = array_entry_index(self, sym.ty.length, ZERO, index_value)
                    if nested is not None:
                        nested_indices = (*nested_indices, (len(steps), nested))
            # Dynamic locations are analysis-only. The caller uses the returned C pointer.
            code = f"({sym.c})[{step}]" if isinstance(step, int) else ""
            sym = replace(sym, ty=uq(element), c_place=code,
                          readonly=sym.readonly or isinstance(element, g["ReadOnly"]),
                          projection=Projection(root, steps + (indices,), resolved=True,
                                                entry_end=entry_end, entry_index=entry_index,
                                                nested_entry_indices=nested_indices))
        return sym

    C.alias_result_projection = alias_result_projection

    def captured_projection_owners(self, sym, state):
        projection = sym.projection
        if projection.captured is not None:
            return projection.captured
        root = projection.root
        start = None
        if isinstance(root.ty, Slice):
            original = root.ty.projection
            positions = original.positions() if original is not None else state.regions.get(root.uid, ())
            if len(positions) == 1:
                start = original.start if original is not None else state.starts.get(root.uid)
            first, *tail = projection.steps
            locations = ((owner, (alias_span_indices(self, self.actual_array(state, owner), offset, first), *tail), offset)
                         for owner, offset in positions)
        else:
            locations = ((owner, projection.steps, ZERO) for owner in self.physical_places(root, state))
        result = []
        for owner, steps, offset in locations:
            if not steps[0]:
                continue
            first, offset, index_value = alias_span_entry(self, steps[0], offset, projection.first_index, start)
            path_segments = 1
            resolved_steps = []
            for step in (first, *steps[1:]):
                if isinstance(step, Domain) and path_segments <= 64:
                    step = finite_alias_indices(self, step, 64 // path_segments)
                    path_segments *= max(1, len(step.ranges))
                resolved_steps.append(step)
            steps = tuple(resolved_steps)
            first = steps[0]
            nested_indices = projection.nested_entry_indices
            if has_projection(owner):
                base = owner.projection.root
                prefix = owner.projection.steps
                steps = (*prefix, *steps)
                entry_end, entry_index = projected_entry_index(self, owner.projection)
                nested_indices = (*owner.projection.nested_entry_indices,
                                  *((i + len(prefix), index) for i, index in nested_indices))
                if (base.uid in self.param_places and isinstance(owner.ty, A)
                        and isinstance(first, Domain) and first.lo != first.hi):
                    nested = array_entry_index(self, owner.ty.length, offset, index_value)
                    if nested is not None:
                        nested_indices = (*nested_indices, (len(prefix), nested))
            else:
                base = owner
                entry_end = entry_index = None
                if isinstance(steps[0], Domain) and steps[0].lo != steps[0].hi:
                    entry_end = slice_entry_end(self, base, offset, index_value)
                    entry_index = root_entry_index(self, base, offset, index_value)
            result.append(replace(sym, uid=base.uid, depth=base.depth, heap_owner=base.heap_owner,
                                  borrowed=base.borrowed, static_storage=base.static_storage,
                                  projection=Projection(base, steps, resolved=True,
                                                        entry_end=entry_end, entry_index=entry_index,
                                                        nested_entry_indices=nested_indices)))
        return tuple(result)

    C.captured_projection_owners = captured_projection_owners

    def capture_alias_argument(self, sym, targets, code):
        if sym is not None and isinstance(sym.ty, Link):
            if not any(target.uid for target in targets):
                return self.null_symbol
            return replace(sym, ty=self.link_record(sym.ty, self.nodes[0].token),
                           c_place=f"(*({code}))", link_access=None, projection=None,
                           captured_places=tuple(targets))
        if sym is not None and sym.link_access is not None:
            return replace(sym, link_access=None, captured_places=tuple(targets))
        if sym is None:
            if not targets:
                self.nodes[0].token.error("cannot prove the backing storage of the alias argument")
            target = next(iter(targets))
            sym = replace(target, c_place=f"(*({code}))")
        if not has_projection(sym):
            return sym
        return replace(sym, projection=replace(sym.projection, resolved=True, captured=tuple(targets)))

    C.capture_alias_argument = capture_alias_argument

    def capture_projected_argument(self, sym, state, token, *, allow_effectful_links=False, propagate_exit=False):
        root = sym.projection.root
        direct = not sym.projection.resolved
        direct_slice = direct and isinstance(root.ty, Slice)
        epoch = state.epochs.get(root.uid) if direct_slice else None
        start = None
        if (direct and isinstance(root.ty, A) and root.uid not in self.alias_members
                and root.link_access is None and not root.bit_overlay):
            start = Value(ZERO, "")
        elif direct_slice:
            span = root.ty.projection
            positions = span.positions() if span is not None else state.regions.get(root.uid, ())
            if len(positions) == 1:
                owner, offset = positions[0]
                if (isinstance(owner.ty, A) and not has_projection(owner) and offset is not None
                        and owner.uid not in self.alias_members and owner.link_access is None):
                    if offset.lo == offset.hi:
                        start = Value(offset, "")
                    else:
                        start = span.start if span is not None else state.starts.get(root.uid)
        sym = materialize(self, sym, state, allow_effectful_links=allow_effectful_links,
                          propagate_exit=propagate_exit)
        if sym.projection.always_returns:
            return sym
        if sym.projection.packed_target is not None:
            token.error("packed projected alias requires bit-address lowering")
        targets = captured_projection_owners(self, sym, state)
        if not targets:
            # Abstract chain elements are represented by their parameter tree.
            if isinstance(sym.projection.root.ty, Slice) and sym.projection.root.uid in state.arrays:
                return sym
            token.error("cannot prove the backing storage of the record argument")
        if direct_slice and state.epochs.get(root.uid) is not epoch:
            epoch = start = None
        absolute_index = None
        if start is not None and len(targets) == 1 and sym.projection.first_index is not None:
            absolute_index = self.combine_linear("+", start, sym.projection.first_index)
        return replace(sym, projection=replace(sym.projection, captured=targets, descriptor_epoch=epoch,
                                               absolute_index=absolute_index))

    C.capture_projected_argument = capture_projected_argument

    @wrap("physical_places")
    def physical_places(self, sym, state, allow_null=False):
        if has_projection(sym) and sym.projection.captured is not None:
            return tuple(target for target in sym.projection.captured if target.uid or allow_null)
        return old["physical_places"](self, sym, state, allow_null)

    @wrap("chain_anchor")
    def chain_anchor(self, node, state):
        sym = self.place(node)
        if not has_projection(sym) or not isinstance(sym.ty, R):
            return old["chain_anchor"](self, node, state)
        targets = captured_projection_owners(self, materialize(self, sym, state), state)
        if len(targets) != 1:
            node.token.error("cannot prove a unique chain anchor")
        return targets[0]

    @wrap("record_text_domain")
    def record_text_domain(self, sym, state):
        if has_projection(sym):
            return text_length(projected(self, materialize(self, sym, state), state))
        return g["domain_union"](text_length(projected(self, target, state)) if has_projection(target)
                                 else state.values[target.uid] for target in self.physical_places(sym, state))

    @wrap("ordinary_span_value")
    def ordinary_span_value(self, node, state, *, expected_encoding=None):
        sym = self.place(node)
        projected_span = has_projection(sym) and (
            isinstance(sym.ty, Slice) or isinstance(sym.ty, A) and not sym.ty.packed_bits)
        if not projected_span:
            return old['ordinary_span_value'](self, node, state, expected_encoding=expected_encoding)
        direct_projection = not sym.projection.resolved
        sym = materialize(self, sym, state, allow_effectful_links=self.has_propagation(node),
                          propagate_exit=True)
        if sym.projection.always_returns:
            code = f"((void)({sym.c}), (lan_slice){{0}})"
            return Span(sym.ty.element, Value(ZERO, "0", always_returns=True), sym.depth, code=code)
        if isinstance(sym.ty, Slice):
            value = projected(self, sym, state)
            return self.record_text_span(sym, text_length(value), state,
                                         data=value.data if isinstance(value, TextContent) else None)
        projection = sym.projection
        root = projection.root
        identity = None
        if (root.link_access is None and root.uid not in self.alias_members
                and (not isinstance(root.ty, Slice) or direct_projection)
                and not self.has_effects(node, allow_debug_checks=True)
                and len(projection.index_values) == len(projection.steps)):
            steps = tuple(step if isinstance(step, str) else step.lo if step.lo == step.hi
                          else index.linear or index.affine
                          for step, index in zip(projection.steps, projection.index_values))
            if all(step is not None for step in steps):
                epoch = state.epochs.get(root.uid) if isinstance(root.ty, Slice) else None
                identity = root.uid, epoch, steps
                if isinstance(root.ty, Slice):
                    span = root.ty.projection
                    positions = span.positions() if span is not None else state.regions.get(root.uid, ())
                    if len(positions) == 1:
                        owner, offset = positions[0]
                        if (isinstance(owner.ty, A) and not has_projection(owner)
                                and owner.link_access is None and owner.uid not in self.alias_members
                                and offset is not None):
                            if offset.lo == offset.hi:
                                start = Value(offset, "")
                            else:
                                start = span.start if span is not None else state.starts.get(root.uid)
                            if start is not None:
                                absolute = self.combine_linear("+", start, projection.index_values[0])
                                if absolute is not None:
                                    identity = owner.uid, None, (absolute, *steps[1:])

        if isinstance(root.ty, Slice) or root.uid in self.alias_members or root.link_access is not None:
            owners = captured_projection_owners(self, sym, state)
            if not owners:
                node.token.error("cannot prove the backing storage of the row slice")
            origins = tuple((owner, ZERO) for owner in owners)
            return Span(sym.ty.element, Value(Domain.point(sym.ty.length), str(sym.ty.length)),
                        max(owner.depth for owner in owners), host=sym, readonly=sym.readonly,
                        text=sym.ty.encoding, borrowed_from=frozenset(owners), origins=origins, row_identity=identity)
        return Span(sym.ty.element, Value(Domain.point(sym.ty.length), str(sym.ty.length)),
                    sym.depth, host=sym, readonly=sym.readonly, text=sym.ty.encoding, row_identity=identity)

    @wrap("read_span_contents")
    def read_span_contents(self, state, span, indices, *, absolute_limit=None):
        if not isinstance(span.element, S):
            return join_values(self.actual_array(state, owner).read(self.array_indices(
                self.actual_array(state, owner), offset, indices)) for owner, offset in span.positions())
        return old['read_span_contents'](self, state, span, indices, absolute_limit=absolute_limit)

    @wrap("havoc_span_contents")
    def havoc_span_contents(self, state, span):
        if not isinstance(span.element, S):
            count = span.length.domain
            return self.write_span_contents(state, span, Domain.of((0, count.hi - 1)), schema(span.element),
                                            fill=count.lo == count.hi, bulk=True)
        return old['havoc_span_contents'](self, state, span)

    @wrap("expr_value")
    def expr_value(self, node, state):
        sym = self.place(node) if node.kind in ('name', 'member', 'index', 'length') else None
        if (sym is not None and isinstance(sym.ty, S) and sym.link_access is not None
                and self.has_propagation(node)):
            sym = materialize(self, sym, state, allow_effectful_links=True, propagate_exit=True)
            if has_projection(sym) and sym.projection.always_returns:
                return Value(ZERO, sym.c, effectful=True, always_returns=True)
            self.require_live(sym, state, node.token)
            return replace(self.storage_value(sym, state), effectful=True)
        if has_projection(sym) and not isinstance(sym.ty, Link):
            if isinstance(sym.ty, S):
                sym = materialize(self, sym, state, allow_effectful_links=self.has_propagation(node),
                                  propagate_exit=True)
                if sym.projection.always_returns:
                    return Value(ZERO, sym.c, effectful=True, always_returns=True)
                value = self.storage_value(sym, state)
                if not self.debug_bounds:
                    code = self.assume_scalar_result(node.token, sym.ty, value.code)
                    value = replace(value, code=code)
                return value
            node.token.error('use an index or named field to access a composite array element')
        return old['expr_value'](self, node, state)

    def view_assignment(self, target_node, sym, op, node, paths):
        def immutable(ty):
            if isinstance(ty, g["ReadOnly"]):
                return True
            if isinstance(ty, R):
                return any(immutable(field) for _, field in ty.fields)
            return isinstance(ty, A) and immutable(ty.element)

        if op.text != "=":
            op.error("composite bit view assignment requires '='")
        if sym.readonly or immutable(sym.ty):
            op.error("cannot modify an immutable composite value or field")
        def direct_storage(place):
            root = place.projection.root if has_projection(place) else place
            return (not root.borrowed and not root.heap_owner and not root.bit_overlay
                    and root.link_access is None and root.uid not in self.alias_members
                    and not place.bit_overlay and place.link_access is None)

        independent = True
        result = []
        exit_code = None

        def remember_exit(code, prefix=""):
            nonlocal exit_code
            candidate = f"{{ {prefix} (void)({code}); }}"
            if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                exit_code = candidate

        self.propagating_expression_nodes |= self.propagation_nodes(target_node)
        pointer = f"lan_view_dst_{op.line}_{op.col}"
        for state in paths:
            dst = (self.capture_projected_argument(sym, state, op, allow_effectful_links=True,
                                                   propagate_exit=True)
                   if has_projection(sym)
                   else materialize(self, sym, state))
            if has_projection(dst) and dst.projection.always_returns:
                remember_exit(dst.c)
                continue
            self.require_live(dst, state, op)
            targets = self.physical_places(dst, state)
            sources, early = [], []
            value, source_type, address = view_initializer(self, dst.ty, node, state, captured_places=sources,
                                                           capture_exit=early)
            if early:
                remember_exit(early[0][1].code, f"(void)(&({dst.c}));")
                continue
            try:
                separate = bool(targets and sources) and all(
                    direct_storage(target) and direct_storage(source) and not self.records_may_overlap(target, source)
                    for target in targets for source in sources)
                if independent and not separate and has_projection(sym) and not sym.projection.resolved:
                    operand = node.args[0]
                    if (not self.has_effects(operand, allow_debug_checks=True)
                            and not any(isinstance(step, Node) and self.has_effects(step, allow_debug_checks=True)
                                        for step in sym.projection.steps)):
                        original_source = self.place(operand)
                        if has_projection(original_source) and not original_source.projection.resolved:
                            separate = disjoint_projections(
                                self, materialize(self, sym, state), materialize(self, original_source, state), state, op)
                independent = independent and separate
            except g["DomainComplexity"]:
                independent = False
            updated = state
            for target in targets:
                self.require_target_live(target, state, op, access=dst)
                content = value if len(targets) == 1 else join_value(record_value(self, target, state), value)
                if not has_projection(target):
                    for leaf in self.storage_leaves(target):
                        updated = self.write_storage(updated, leaf)
                updated = store_record_value(self, target, updated, content)
            result.append(updated)
            guard = " and ".join(state.guards) or "entry"
            origins = ", ".join(sorted({trace_location(source) for source in sources})) or "temporary value"
            self.trace.append(f"  update entity {trace_location(dst)} [{guard}]: bit view copy from {origins}")
            emitted = dst, source_type, address
        if not result:
            return [], [exit_code, "__builtin_unreachable();"]
        dst, source_type, address = emitted
        strategy = "disjoint, restrict copy" if independent else "may overlap, memmove"
        self.trace.append(f"  lower bit view assignment at {op.line}:{op.col}: {strategy}")
        code = view_copy_code(self, f"(*{pointer})", source_type, address, move=not independent)
        return result, ["{", f"    {dst.ty.c} *{pointer} = &({dst.c});",
                        *("    " + line for line in code), "}"]

    def trace_location(sym):
        if has_projection(sym):
            steps = "".join(f".{step}" if isinstance(step, str) else
                            f"[{step.lo}]" if step.lo == step.hi else f"[in {step}]"
                            for step in sym.projection.steps)
            return f"id#{sym.projection.root.uid}{steps}"
        if sym.link_access is not None:
            parent, name = sym.link_access
            return f"{trace_location(parent)}.{name}"
        return f"id#{sym.uid}"

    @wrap("assign")
    def assign(self, stmt, paths):
        target, op, node = stmt.args
        sym = self.place(target)
        if (sym is not None and (isinstance(sym.ty, R) or isinstance(sym.ty, A) and not sym.ty.packed_bits)
                and node.kind == "cast" and node.args[1] == "view"
                and not (target.kind == "name" and sym.uid in self.alias_types)):
            return view_assignment(self, target, sym, op, node, paths)
        if not has_projection(sym) or not isinstance(sym.ty, S):
            return old['assign'](self, stmt, paths)
        if sym.readonly:
            op.error('cannot modify an immutable composite array element')
        result, code = [], []
        error_code = None
        self.propagating_expression_nodes |= self.propagation_nodes(target)
        for path in paths:
            dst = materialize(self, sym, path, allow_effectful_links=self.has_propagation(target),
                              propagate_exit=True)
            if dst.projection.always_returns:
                candidate = f"(void)({dst.c});"
                if error_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(error_code):
                    error_code = candidate
                continue
            ty, lhs = dst.ty, dst.c
            old_value = self.storage_value(dst, path)
            prefix = []
            packed_target = dst.projection.packed_target
            # Resolve the destination address before evaluating the RHS.
            if packed_target is not None and (self.has_effects(target) or self.has_effects(node)):
                data, position = packed_target
                pointer = f'lan_composite_bits_{op.line}_{op.col}'
                offset = f'lan_composite_bit_{op.line}_{op.col}'
                prefix.extend((f'uint8_t *{pointer} = {data};', f'lan_int {offset} = {position};'))
                packed_target = pointer, offset
                old_value = replace(old_value, code=f"{self.helper('bit_read', None)}({pointer}, {offset})")
            elif packed_target is None and (self.has_effects(target) or self.has_effects(node)):
                pointer = f'lan_composite_target_{op.line}_{op.col}'
                prefix.append(f'{ty.c} *{pointer} = &({lhs});')
                lhs = f'(*{pointer})'
                old_value = replace(old_value, code=lhs)
            root = dst.projection.root
            fixed_target = (isinstance(root.ty, A) and root.uid not in self.alias_members
                            and root.link_access is None and not root.bit_overlay)
            captured = ()
            if (op.text == "=" and ty.boolean and isinstance(root.ty, Slice)
                    and dst.link_access is None and packed_target is None):
                captured = captured_projection_owners(self, dst, path)
                if captured:
                    dst = replace(dst, projection=replace(dst.projection, captured=captured))
            rhs = (self.scalar_rhs_paths(node, path, max(1, self.PATH_LIMIT // len(paths)))
                   if op.text == "=" and ty.boolean and (fixed_target or captured)
                   else [(path, self.expr(node, path))])
            for current, value in rhs:
                if value.always_returns:
                    candidate = "{ " + " ".join([*prefix, f"(void)({value.code});"]) + " }"
                    if error_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(error_code):
                        error_code = candidate
                    continue
                if packed_target is not None:
                    if op.text not in ("=", "&=", "|=", "^="):
                        op.error("packed bit assignment supports =, &=, |= and ^=")
                    self.require_bit(op, value, node)
                if op.text != '=':
                    value = self.binary(op, op.text[:-1], old_value, value, current)
                value = self.coerce_storage(op, value, ty)
                self.range_check(op, value, ty)
                if captured:
                    self.require_live(dst, current, op)
                result.append(set_projected(self, dst, current, value.domain))
                guard = " and ".join(current.guards) or "entry"
                self.trace.append(f"  update entity {trace_location(dst)} [{guard}] := {value.domain}")
                if packed_target is not None:
                    data, position = packed_target
                    code = [f"{self.helper('bit_write', None)}({data}, {position}, {value.code});"]
                else:
                    code = [f'{lhs} = {value.code};']
                if prefix:
                    code = ['{ ' + ' '.join(prefix + code) + ' }']
        self.budget_paths(result, op)
        if not result and error_code is not None:
            return [], [error_code, "__builtin_unreachable();"]
        return result, code

    A.domain = property(lambda ty: schema(ty))
    R.domain = property(lambda ty: schema(ty))

    class ContentFactory(ArrayDomain):
        @classmethod
        def zero(cls, length, element):
            if isinstance(element, (RecordValue, TreeArray)):
                def zeros(value):
                    if isinstance(value, RecordValue):
                        return RecordValue(tuple((k, zeros(v)) for k, v in value.fields))
                    if isinstance(value, TreeArray):
                        return TreeArray.filled(value.length, value.element, zeros(value.element))
                    return ZERO
                return TreeArray.filled(length, element, zeros(element))
            return ArrayDomain.zero(length, element)

        @classmethod
        def unknown(cls, length, element):
            if isinstance(element, (RecordValue, TreeArray)):
                return TreeArray.unknown(length, element)
            return ArrayDomain.unknown(length, element)

        @classmethod
        def join_all(cls, values):
            return reduce(lambda a, b: a.join(b), values)

    g['ArrayDomain'] = ContentFactory

    @wrap('record_symbols')
    def record_symbols(self, sym, token, zero=True):
        if composite(sym.ty):
            if zero:
                zero_check(self, sym.ty, token)
            return
        return old['record_symbols'](self, sym, token, zero)

    def known_record_text(self, sym, state, length):
        targets = self.physical_places(sym, state)
        if len(targets) != 1 or has_projection(targets[0]):
            return None
        target = targets[0]
        data = state.predicates.get(("text_bytes", target.uid, state.epochs.get(target.uid)))
        return data if data is not None and length == Domain.point(len(data)) else None

    C.known_record_text = known_record_text

    def record_value(self, sym, state, *, track_text=False):
        if has_projection(sym):
            return projected(self, materialize(self, sym, state), state)
        if sym.link_access is not None and isinstance(sym.ty, R):
            return join_values(record_value(self, target, state) for target in self.physical_places(sym, state))
        if isinstance(sym.ty, S):
            return self.storage_value(sym, state).domain
        if isinstance(sym.ty, Link):
            return self.link_targets(sym, state)
        if isinstance(sym.ty, Interface):
            return frozenset(self.physical_places(sym, state))
        if isinstance(sym.ty, Slice):
            length = self.record_text_domain(sym, state)
            if track_text:
                data = known_record_text(self, sym, state, length)
                if data is not None:
                    return TextContent(length, data)
            return length
        if isinstance(sym.ty, R):
            return RecordValue(tuple((child.name, record_value(self, child, state, track_text=track_text))
                                     for child in sym.fields))
        if sym.ty.packed_bits:
            mask, data = self.storage_bits(state, sym)
            if not mask:
                return schema(sym.ty)
            return TreeArray.from_values(sym.ty.length, BOOL,
                (Domain.point((data >> i) & 1) if (mask >> i) & 1 else BOOL for i in range(sym.ty.length)))
        return self.storage_array(state, sym)

    C.record_value = record_value

    def record_field_domains(self, sym, state):
        def fields(ty, content, prefix):
            ty = uq(ty)
            if isinstance(ty, (S, Slice, A)):
                yield prefix, content
            elif isinstance(ty, R):
                domains = dict(content.fields)
                for name, field in ty.fields:
                    yield from fields(field, domains[name], prefix + (name,))

        return dict(fields(sym.ty, record_value(self, sym, state, track_text=True), ()))

    C.record_field_domains = record_field_domains

    def restore_record_text(self, sym, state, value):
        state = state.touch(sym.uid).update(sym, text_length(value))
        if isinstance(value, TextContent):
            key = ("text_bytes", sym.uid, state.epochs[sym.uid])
            return replace(state, predicates=state.predicates | {key: value.data})
        return state

    C.restore_record_text = restore_record_text

    def store_record_value(self, sym, state, value):
        if has_projection(sym):
            return set_projected(self, materialize(self, sym, state), state, value)
        if isinstance(sym.ty, S):
            return self.store_value(state, sym, Value(value, '', sym.ty.bits, sym.ty.boolean))
        if isinstance(sym.ty, Link):
            return self.store_link(state, sym, frozenset(self.null_symbol if target is None else target
                                                        for target in value), self.nodes[0].token)
        if isinstance(sym.ty, Interface):
            guard = " and ".join(state.guards) or "entry"
            self.trace.append(f"  define entity id#{sym.uid} {sym.name} [{guard}]: "
                              f"interface targets {self.trace_alias_targets(value)}")
            state = state.update(sym, ZERO)
            return replace(state, borrows=state.borrows | {sym.uid: value})
        if isinstance(sym.ty, Slice):
            return restore_record_text(self, sym, state, value)
        if isinstance(sym.ty, R):
            fields = dict(value.fields)
            for child in sym.fields:
                state = store_record_value(self, child, state, fields[child.name])
            return state
        return self.restore_record_array(sym, self.write_storage(state, sym), value)

    def restore_record_array(self, sym, state, value):
        if sym.ty.packed_bits:
            mask = data = 0
            if sym.ty.length <= 4096:
                for s in value.segments:
                    if s.domain.lo == s.domain.hi:
                        part = ((1 << (s.stop - s.start)) - 1) << s.start
                        mask |= part
                        if s.domain.lo:
                            data |= part
            return self.store_bits(state, sym, (mask, data))
        if isinstance(sym.ty.element, S) and isinstance(value, TreeArray):
            value = ArrayDomain(value.length, value.element, value.segments)
        return self.store_array(state, sym, value)

    C.restore_record_array = restore_record_array

    @wrap('copy_record_states')
    def copy_record_states(self, dst, src, paths):
        if self.contains_record_links(dst.ty):
            for state in paths:
                value = record_value(self, src, state)
                root = dst.projection.root if has_projection(dst) else dst
                span = root.ty.projection if isinstance(root.ty, Slice) else None
                owners = span.owners if span is not None else state.borrows.get(root.uid, (root,))
                for owner in owners:
                    check_link_lifetimes(self, value, owner.depth, state, self.nodes[0].token)
        if not has_projection(dst) and not has_projection(src):
            return old['copy_record_states'](self, dst, src, paths)
        out = []
        for state in paths:
            source = materialize(self, src, state)
            value = record_value(self, source, state)
            out.append(store_record_value(self, materialize(self, dst, state), state, value))
        return out

    def disjoint_projections(self, left, right, state, token):
        a, b = left.projection, right.projection
        if left.link_access is not None or right.link_access is not None:
            return False
        for projection in (a, b):
            if projection is not None and (projection.root.link_access is not None or projection.captured is not None):
                return False
        if a is None or b is None:
            projected, direct = (b, left) if a is None else (a, right)
            return (projected is not None and isinstance(projected.root.ty, A)
                    and not self.records_may_overlap(left, right)
                    and self.independent_bit_storage(direct, projected.root, state))

        def disjoint_steps(start):
            for index, (first, second) in enumerate(zip(a.steps[start:], b.steps[start:]), start):
                if isinstance(first, str) != isinstance(second, str):
                    return False
                if isinstance(first, str):
                    if first != second:
                        return True
                elif index < min(len(a.index_values), len(b.index_values)):
                    x, y = a.index_values[index], b.index_values[index]
                    if x is not None and y is not None and self.offset_equality_domain(token, x, y, state) == ZERO:
                        return True
            return False

        if a.root.uid != b.root.uid:
            def backing_indices(projection):
                root = projection.root
                if root.bit_overlay:
                    return ()
                start = None
                if isinstance(root.ty, A):
                    positions = ((root, ZERO),)
                elif isinstance(root.ty, Slice):
                    span = root.ty.projection
                    positions = span.positions() if span is not None else state.regions.get(root.uid, ())
                    if len(positions) == 1:
                        start = self.span_start(span) if span is not None else state.starts.get(root.uid)
                else:
                    return ()
                index = projection.steps[0]
                result = []
                for owner, offset in positions:
                    absolute = None
                    if offset is not None and isinstance(index, Domain):
                        position = projection.first_index
                        base = Value(offset, "0") if offset.lo == offset.hi else start
                        if position is not None and base is not None:
                            position = replace(position, domain=index)
                            absolute = self.binary(token, "+", position, replace(base, domain=offset), state)
                        else:
                            absolute = Value(self.sum_index_domains(offset, index), "0")
                    result.append((owner, absolute))
                return result

            def disjoint_backing(x, position, y, other_position):
                if not isinstance(x.ty, A) or not isinstance(y.ty, A):
                    return False
                if (x == y and x.projection is None and not x.bit_overlay
                        and x.link_access is None and x.uid not in self.alias_members):
                    return ((position is not None and other_position is not None
                             and self.offset_equality_domain(token, position, other_position, state) == ZERO)
                            or disjoint_steps(1))
                return self.independent_bit_storage(x, y, state)

            first, second = backing_indices(a), backing_indices(b)
            return bool(first and second) and all(
                disjoint_backing(x, indices, y, other_indices)
                for x, indices in first for y, other_indices in second)
        return disjoint_steps(0)

    C.disjoint_projections = disjoint_projections

    @wrap('assign_record_copy')
    def assign_record_copy(self, target, op, node, paths):
        dst, src = self.record_place(target), self.record_place(node.args[0])
        if not any(sym is not None and (has_projection(sym) or sym.link_access is not None)
                   for sym in (dst, src)):
            return old['assign_record_copy'](self, target, op, node, paths)
        if op.text != '=' or src is None or dst.ty != src.ty:
            op.error('record copy requires the same record type and =')
        if dst.readonly or any(isinstance(ty, g['ReadOnly']) for _, ty in dst.ty.fields):
            op.error('cannot modify an immutable record or field')
        self.propagating_expression_nodes |= self.propagation_nodes(target) | self.propagation_nodes(node.args[0])
        out = []
        exit_code = None

        def remember_exit(code, prefix=""):
            nonlocal exit_code
            candidate = f"{{ {prefix} (void)({code}); }}"
            if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                exit_code = candidate

        disjoint = not (self.has_effects(target, allow_debug_checks=True)
                        or self.has_effects(node.args[0], allow_debug_checks=True))
        for state in paths:
            left = materialize(self, dst, state, allow_effectful_links=self.has_propagation(target),
                               propagate_exit=True)
            if has_projection(left) and left.projection.always_returns:
                remember_exit(left.c)
                continue
            self.require_live(left, state, op)
            right = materialize(self, src, state, allow_effectful_links=self.has_propagation(node.args[0]),
                                propagate_exit=True)
            if has_projection(right) and right.projection.always_returns:
                remember_exit(right.c, f"(void)(&({left.c}));")
                continue
            self.require_live(right, state, node.token)
            if has_projection(left) and any(owner.uid in self.chain_fields for owner in chain_projection_owners(left, state)):
                op.error("cannot overwrite linked records through a chain slice")
            disjoint = disjoint and disjoint_projections(self, left, right, state, op)
            value = record_value(self, right, state)
            guard = " and ".join(state.guards) or "entry"
            if left.link_access is not None:
                destinations = self.physical_places(left, state)
                updated = state
                for destination in destinations:
                    if destination.readonly or any(leaf.readonly for leaf in self.storage_leaves(destination)):
                        op.error("cannot modify an immutable record or field")
                    if self.contains_record_links(left.ty):
                        check_link_lifetimes(self, value, destination.depth, state, op)
                    stored = value if len(destinations) == 1 else join_value(record_value(self, destination, state), value)
                    updated = store_record_value(self, destination, updated, stored)
                out.append(updated)
                self.trace.append(f"  update entity {trace_location(left)} [{guard}]: copy record from "
                                  f"{trace_location(right)}")
                emitted = left, right
                continue
            if self.contains_record_links(left.ty):
                root = left.projection.root if has_projection(left) else left
                span = root.ty.projection if isinstance(root.ty, Slice) else None
                owners = span.owners if span is not None else state.borrows.get(root.uid, (root,))
                for owner in owners:
                    check_link_lifetimes(self, value, owner.depth, state, op)
            out.append(store_record_value(self, left, state, value))
            self.trace.append(f"  update entity {trace_location(left)} [{guard}]: copy record from "
                              f"{trace_location(right)}")
            emitted = left, right
        if not out:
            return [], [exit_code, "__builtin_unreachable();"]
        left, right = emitted
        pointer = f'lan_record_dst_{op.line}_{op.col}'
        if disjoint:
            copy = self.helper("disjoint_object_copy", None)
            write = f'{copy}({pointer}, &({right.c}), sizeof(*{pointer}));'
        else:
            write = f'*{pointer} = {right.c};'
        strategy = "disjoint, restrict copy" if disjoint else "may overlap, value assignment"
        self.trace.append(f"  lower record copy at {op.line}:{op.col}: {strategy}")
        return out, [f'{{ {left.ty.c} *{pointer} = &({left.c}); {write} }}']

    def view_initializer(self, ty, node, state, *, captured_places=None, capture_exit=None):
        operand, _, annotation = node.args
        if self.resolve(annotation) != ty:
            node.token.error("composite bit view initializer requires the declared destination type")
        source, address, domains = self.composite_view_source(ty, operand, [state], node.token,
                                                              captured_places=captured_places,
                                                              capture_exit=capture_exit)
        if capture_exit:
            return None, None, None
        domain, = domains

        def contents(target, path=()):
            if isinstance(target, R):
                return RecordValue(tuple((name, contents(uq(field), path + (name,)))
                                         for name, field in target.fields))
            actual = domain.get(path) if isinstance(domain, dict) else domain
            if actual is None:
                return schema(target)
            if isinstance(target, A) and not isinstance(actual, ArrayDomain):
                return ArrayDomain.filled(target.length, target.element.domain, actual)
            return actual

        return contents(ty), source, address

    def view_copy_code(self, code, source, address, *, move=False):
        copy = "memmove" if move else self.helper("disjoint_object_copy", None)
        return [
            f'_Static_assert(sizeof({code}) == sizeof({source.c}), "Lan composite bit view width");',
            f'{copy}(&({code}), {address}, sizeof({code}));',
        ]

    def checked_copy_source(self, ty, node, state, *, propagate_exit=False):
        if self.contains_interface_storage(ty):
            node.token.error("records with interface fields cannot be copied")
        source = self.place(node.args[0])
        if source is None or source.ty != ty:
            node.token.error("composite copy initializer requires the same type")
        source = materialize(self, source, state, propagate_exit=propagate_exit)
        if has_projection(source) and source.projection.always_returns:
            return source
        self.require_live(source, state, node.token)
        return source

    def initialize(self, ty, node, code, state, token, key, *, track_text=False, fresh=True,
                   propagate_exit=False, heap_interface_offsets=None):
        """Return content state and ordered C statements for one initializer."""
        if isinstance(ty, Interface):
            if node is None or node.kind != "name":
                (token if node is None else node.token).error(
                    "interface field requires an initializer naming an existing record or interface")
            source, descriptor = self.interface_argument(node, state, ty)
            targets = self.interface_field_targets(source, state, len(self.scope), node.token)
            offsets = f"lan_field_offsets_{key}"
            helper = self.helper("interface_rebind", None)
            buffer = (f"size_t *{offsets} = {heap_interface_offsets[key]};" if heap_interface_offsets is not None
                      else f"size_t {offsets}[{len(ty.fields)}];")
            return targets, [buffer,
                             f"{code} = {helper}({descriptor}, {offsets}, {len(ty.fields)});"]
        if isinstance(ty, Link):
            early = [] if propagate_exit else None
            targets, pointer = self.link_initializer(ty, node, state, token, capture_exit=early)
            if early:
                return None, [f"(void)({early[0].code});"]
            return targets, [f"{code} = {pointer};"]
        if isinstance(ty, Slice):
            if node is None:
                node = Node(token, "text_literal", (b"",))
            expected_enc = ty.text if isinstance(ty.text, g["TextEncoding"]) else None
            span = self.ordinary_span(node, state, expected_encoding=expected_enc)
            if propagate_exit and span.length.always_returns:
                return None, [f"(void)({span.c_value()});"]
            if self.debug_bounds and ty.length_bound is not None:
                span = self.checked_slice_length(node.token, span, ty, "text field")
            span = self.check_span_contract(node.token, span, ty, state)
            if not span.owners or any(owner.uid >= 0 and not owner.static_storage for owner in span.owners):
                node.token.error("record string field requires static backing storage")
            data = self.literal_bytes(span) if track_text else None
            content = span.length.domain if data is None else TextContent(span.length.domain, data)
            return content, [f"{code} = {span.c_value()};"]
        if node is None:
            if self.contains_interface_storage(ty):
                token.error("a record with interface fields requires a local field initializer")
            zero_check(self, ty, token)
            return schema(ty, True), [f'memset(&({code}), 0, sizeof({code}));']
        if isinstance(ty, S):
            value = self.expr(node, state)
            if propagate_exit and value.always_returns:
                return None, [f"(void)({value.code});"]
            value = self.coerce_storage(node.token, value, ty)
            self.range_check(node.token, value, ty)
            return value.domain, [f'{code} = {value.code};']
        if node.kind == "cast" and node.args[1] == "view" and (isinstance(ty, R) or not ty.packed_bits):
            early = [] if propagate_exit else None
            value, source, address = view_initializer(self, ty, node, state, capture_exit=early)
            if early:
                return None, [f"(void)({early[0][1].code});"]
            return value, view_copy_code(self, code, source, address, move=not fresh)
        if node.kind == 'copy':
            src = checked_copy_source(self, ty, node, state, propagate_exit=propagate_exit)
            if has_projection(src) and src.projection.always_returns:
                return None, [f"(void)({src.c});"]
            copy = self.helper('disjoint_object_copy', None) if fresh else 'memmove'
            return record_value(self, src, state, track_text=track_text), [f'{copy}(&({code}), &({src.c}), sizeof({code}));']
        if isinstance(ty, R):
            values, lines = {}, []
            for name, field, value in self.record_initializer_fields(ty, node):
                content, code_lines = initialize(self, field, value, f"({code}).{c_field(name)}",
                                                 state, token, f"{key}_{name}", track_text=track_text, fresh=fresh,
                                                 propagate_exit=propagate_exit,
                                                 heap_interface_offsets=heap_interface_offsets)
                lines.extend(code_lines)
                if content is None:
                    return None, lines
                values[name] = content
            return RecordValue(tuple((name, values[name]) for name, _ in ty.fields)), lines
        if ty.packed_bits and node.kind not in ('list', 'repeat'):
            literal = g['bit_literal_value'](node)
            if literal is not None:
                data, width = literal
                g['require_bit_literal_width'](node.token, ty.length, width)
                bytes_ = g['bit_literal_bytes'](data, width)
                return self.bit_literal_contents(data, width), [
                    f'memcpy({code}, (const uint8_t[]){bytes_}, {ty.c_length});']
            width = g['bit_literal_width'](node)
            if width is not None:
                g['require_bit_literal_width'](node.token, ty.length, width)
            value = self.expr(node, state)
            if propagate_exit and value.always_returns:
                return None, [f"(void)({value.code});"]
            if value.boolean or value.domain.lo < 0 or value.domain.hi >= 1 << ty.length or ty.length > 64:
                node.token.error('packed row initializer requires an integer fitting at most 64 bits')
            if value.domain.lo == value.domain.hi:
                tree = TreeArray.from_values(ty.length, BOOL,
                    (Domain.point((value.domain.lo >> i) & 1) for i in range(ty.length)))
            else:
                tree = schema(ty)
            helper = self.helper('bit_store', None)
            return tree, [f'{helper}({code}, 0, {ty.length}, (uint64_t)({value.code}), 0);']
        if getattr(ty, "encoding", None) is not None and node.kind == "text_literal":
            raw = ty.encoding.encode_fixed(node.args[0].decode("utf-8"), ty.length, node.token)
            lit_sym = self.literal_storage(raw, b"")
            content = ArrayDomain.from_values(ty.length, ty.element.domain, [Domain.point(b) for b in raw])
            return content, [f"memcpy({code}, {lit_sym.c}, {ty.c_length});"]
        if node.kind not in ('list', 'repeat'):
            node.token.error('array initializer requires a list, repetition or explicit copy')
        elements = node.args[0] if node.kind == 'list' else (node.args[0],)
        length = (len(elements) if node.kind == 'list'
                  else self.const_int(node.args[1], ()) if isinstance(node.args[1], Node) else node.args[1])
        if length != ty.length:
            node.token.error(f'array initializer has {length} elements, expected {ty.length}')
        lines, values = [], []
        if ty.packed_bits and node.kind == "list":
            lines.append(f"memset({code}, 0, sizeof({code}));")
        for i, item in enumerate(elements):
            if node.kind == 'repeat':
                tmp = f'lan_composite_init_{key}_{i}'
                lines.append(f'{ty.element.c} {tmp};')
                dst = tmp
            else:
                dst = f'({code})[{i}]'
            if ty.packed_bits:
                value = self.expr(item, state)
                if propagate_exit and value.always_returns:
                    return None, [*lines, f"(void)({value.code});"]
                self.require_bit(item.token, value, item)
                values.append(value.domain)
                bit_write = self.helper('bit_write', None)
                if node.kind == 'repeat':
                    lines = [f'memset({code}, 255 * !!({value.code}), sizeof({code}));']
                else:
                    lines.append(f'{bit_write}({code}, {i}, {value.code});')
            else:
                content, statements = initialize(self, ty.element, item, dst, state, token, f'{key}_{i}',
                                                 track_text=track_text, fresh=fresh, propagate_exit=propagate_exit)
                lines.extend(statements)
                if content is None:
                    return None, lines
                values.append(content)
            if node.kind == 'repeat' and not ty.packed_bits:
                it = f'lan_composite_i_{key}_{i}'
                lines.extend(g['c_counted_loop'](it, 0, ty.length,
                                                  [f'memcpy(&(({code})[{it}]), &{tmp}, sizeof({tmp}));']))
        declaration = schema(ty.element)
        tree = (TreeArray.filled(ty.length, declaration, values[0]) if node.kind == 'repeat'
                else TreeArray.from_values(ty.length, declaration, values))
        return tree, lines

    @wrap('array_init')
    def array_init(self, sym, node, paths, t, bit_facts=None, allocation=None, *, continuing_paths=None):
        if not composite(sym.ty):
            return old['array_init'](self, sym, node, paths, t, bit_facts, allocation,
                                     continuing_paths=continuing_paths)
        for state in paths:
            contents, code = initialize(self, sym.ty, node, sym.c, state, t, str(sym.uid), track_text=True)
            if self.contains_record_links(sym.ty):
                check_link_lifetimes(self, contents, sym.depth or len(self.scope), state, t)
            state.absorb(state.with_array(sym, contents))
        return [f'{sym.ty.c} {sym.c};', *code]

    def propagating_array_initializer(self, sym, node, paths, token):
        previous = self.propagating_expression_nodes
        self.propagating_expression_nodes = self.propagation_nodes(node, composite_literal=True)
        continuing = []
        success_lines = error_lines = None
        error_rank = (-1, -1)
        try:
            for path in paths:
                contents, lines = initialize(self, sym.ty, node, sym.c, path, token, str(sym.uid),
                                             track_text=True, propagate_exit=True)
                if contents is None:
                    rank = self.propagation_code_rank(" ".join(lines))
                    if error_lines is None or rank > error_rank:
                        error_lines, error_rank = lines, rank
                    continue
                if self.contains_record_links(sym.ty):
                    check_link_lifetimes(self, contents, sym.depth or len(self.scope), path, token)
                continuing.append(self.restore_record_array(sym, path, contents))
                success_lines = lines
        finally:
            self.propagating_expression_nodes = previous
        declaration = f"{sym.ty.c} {sym.c};"
        if success_lines is not None:
            return continuing, [declaration, *success_lines]
        return [], [declaration, *error_lines, "__builtin_unreachable();"]

    C.propagating_array_initializer = propagating_array_initializer

    def define_initialized_record(self, name, ty, node, paths, heap=False, readonly=False, static=False):
        sym = self.declare(name, ty, heap=heap, readonly=readonly, static=static, allocated_record=heap,
                           ownership_origin="allocate" if heap else None)
        self.record_symbols(sym, name, zero=False)
        self.scope[-1].pop(name.text)
        previous_propagating_nodes = self.propagating_expression_nodes
        previous_heap_records = self.initializing_heap_records
        heap_interface_offsets = None
        try:
            if static:
                content, initializer = self.static_array(ty, node, name)
                code = [f"static const {ty.c} {sym.c} = {initializer};"]
                paths = [store_record_value(self, sym, state, content) for state in paths]
                return paths, code
            if heap:
                allocator = self.helper("heap_array", None)
                self.initializing_heap_records = (*previous_heap_records, sym)
                pointer = sym.c_var
                fields = [field for field in sym.fields if isinstance(field.ty, Interface)]
                if fields:
                    storage = f"lan_interface_storage_{sym.uid}"
                    allocation = f"lan_interface_heap_{sym.uid}"
                    heap_interface_offsets = {}
                    count = 0
                    for field in fields:
                        heap_interface_offsets[f"{sym.uid}_{field.name}"] = f"&{allocation}->offsets[{count}]"
                        count += len(field.ty.fields)
                    code = [f"typedef struct {{ {ty.c} record; size_t offsets[{count}]; }} {storage};",
                            f"{storage} *{allocation} = ({storage} *){allocator}(1, sizeof({storage})).data;",
                            f"{ty.c} *{pointer} = &{allocation}->record;"]
                else:
                    code = [f"{ty.c} *{pointer} = ({ty.c} *){allocator}(1, sizeof({ty.c})).data;"]
            else:
                code = [f"{ty.c} {sym.c};"]
            self.propagating_expression_nodes = self.propagation_nodes(node, composite_literal=True)
            result = []
            success_lines = error_lines = None
            error_rank = (-1, -1)
            for path in paths:
                if heap:
                    path = replace(path.update(sym, ZERO), heap=path.heap | {sym.uid: frozenset(("Alive",))})
                content, lines = initialize(self, ty, node, sym.c, path, name, str(sym.uid), track_text=True,
                                            propagate_exit=True, heap_interface_offsets=heap_interface_offsets)
                if content is None:
                    rank = self.propagation_code_rank(" ".join(lines))
                    if error_lines is None or rank > error_rank:
                        error_lines, error_rank = lines, rank
                    continue
                result.append(store_record_value(self, sym, path, content))
                success_lines = lines
            if success_lines is not None:
                return result, code + success_lines
            return [], code + error_lines + ["__builtin_unreachable();"]
        finally:
            self.propagating_expression_nodes = previous_propagating_nodes
            self.initializing_heap_records = previous_heap_records
            self.scope[-1][name.text] = sym

    C.define_initialized_record = define_initialized_record

    @wrap('record_code')
    def record_code(self):
        emitted, lines = set(), []
        def add(ty):
            ty = uq(ty)
            if isinstance(ty, g['Contract']):
                for param in ty.params:
                    add(param)
                add(ty.result)
                return
            if isinstance(ty, g['TaggedResult']):
                name = ty.c
                if name not in emitted:
                    add(ty.success)
                    emitted.add(name)
                    guard = name.upper() + '_DEFINED'
                    lines.extend([f'#ifndef {guard}', f'#define {guard}',
                                  'typedef struct {',
                                  '    uint8_t ok;',
                                  f'    union {{ {ty.success.c} value; {ty.error.c} error; }} payload;',
                                  f'}} {name};', '#endif'])
                return
            if isinstance(ty, g['ExternCallable']):
                if ty.c not in emitted:
                    add(ty.contract)
                    emitted.add(ty.c)
                    guard = ty.c.upper() + '_DEFINED'
                    lines.extend([f'#ifndef {guard}', f'#define {guard}',
                                  C.c_fnptr_typedef(ty.c, ty.contract, foreign=True), '#endif'])
                return
            if isinstance(ty, g['InterfaceResult']):
                add(ty.interface)
                return
            if isinstance(ty, g['Interface']):
                for _, field in ty.fields:
                    add(field)
                return
            if isinstance(ty, g['AliasResult']):
                add(ty.alias)
                return
            if isinstance(ty, g['HeapObject']):
                add(ty.base)
                return
            if isinstance(ty, (g['CopyRecord'], g['BorrowRecord'], g['HeapRecord'])):
                add(ty.record)
                return
            if isinstance(ty, (g['BorrowArray'], g['CopyArray'])):
                add(ty.array)
                if isinstance(ty, g['BorrowArray']):
                    return
                if ty.c in emitted:
                    return
                emitted.add(ty.c)
                guard = ty.c.upper() + '_DEFINED'
                elem = 'uint8_t' if ty.array.packed_bits else ty.array.element.c
                lines.extend([f'#ifndef {guard}', f'#define {guard}',
                              f'typedef struct {{ {elem} data[{ty.array.c_length}]; }} {ty.c};', '#endif'])
                return
            if isinstance(ty, (Slice, HeapArray)):
                add(ty.element)
                return
            if not isinstance(ty, (A, R)) or ty.c in emitted:
                return
            if isinstance(ty, A):
                add(ty.element)
                emitted.add(ty.c)
                guard = ty.c.upper() + '_DEFINED'
                elem = 'uint8_t' if ty.packed_bits else ty.element.c
                lines.extend([f'#ifndef {guard}', f'#define {guard}',
                              f'typedef {elem} {ty.c}[{ty.c_length}];', '#endif'])
                return
            for _, field in ty.fields:
                add(field)
            emitted.add(ty.c)
            guard = ty.c.upper() + '_DEFINED'
            lines.extend([f'#ifndef {guard}', f'#define {guard}', f'typedef struct lan_record_tag_{ty.identity} {{'])
            for name, field in ty.layout:
                lines.append(f'    {uq(field).c} {c_field(name)};')
            lines.extend([f'}} {ty.c};', '#endif'])
        for ty in self.types.values():
            add(ty)
        for sym in self.symbols:
            add(sym.ty)
        for _, contract in self.functions.values():
            add(contract)
        for binding in (*self.imported_functions.values(), *self.externs.values()):
            add(binding.contract)
        return lines

    @wrap('storage_bits')
    def storage_bits(self, state, sym):
        if sym.bit_overlay:
            return old['storage_bits'](self, state, sym)
        if not has_projection(sym):
            targets = self.physical_places(sym, state)
            if not any(has_projection(target) for target in targets):
                return old['storage_bits'](self, state, sym)
            mask = (1 << sym.ty.length) - 1 if sym.ty.length <= 4096 else 0
            data = None
            for target in targets:
                known, value = self.storage_bits(state, target)
                mask &= known
                if data is None:
                    data = value
                else:
                    mask &= ~(data ^ value)
            return mask, (data or 0) & mask
        sym = materialize(self, sym, state)
        tree = projected(self, sym, state)
        mask = data = 0
        if sym.ty.length <= 4096:
            for s in tree.segments:
                if s.domain.lo == s.domain.hi:
                    part = ((1 << (s.stop - s.start)) - 1) << s.start
                    mask |= part
                    if s.domain.lo:
                        data |= part
        return mask, data

    @wrap('store_bits')
    def store_bits(self, state, sym, facts, *, record_write=True):
        if sym.bit_overlay:
            return old['store_bits'](self, state, sym, facts)
        if not has_projection(sym):
            targets = self.physical_places(sym, state)
            if not any(has_projection(target) for target in targets):
                return old['store_bits'](self, state, sym, facts)
            for target in targets:
                mask, data = facts
                if len(targets) > 1:
                    known, value = self.storage_bits(state, target)
                    mask &= known & ~(value ^ data)
                state = self.store_bits(state, target, (mask, data & mask), record_write=record_write)
            return state
        sym = materialize(self, sym, state)
        mask, data = facts
        if sym.ty.length > 4096:
            tree = schema(sym.ty)
        else:
            tree = TreeArray.from_values(sym.ty.length, BOOL,
                (Domain.point((data >> i) & 1) if (mask >> i) & 1 else BOOL for i in range(sym.ty.length)))
        return set_projected(self, sym, state, tree, record_write=record_write)

    @wrap('read_bits')
    def read_bits(self, t, state, sym, *args, **kwargs):
        return old['read_bits'](self, t, state, materialize(self, sym, state), *args, **kwargs)

    @wrap('update_bits')
    def update_bits(self, state, sym, *args, **kwargs):
        sym = materialize(self, sym, state)
        if sym.ty.length > 4096 and any(has_projection(target) for target in self.physical_places(sym, state)):
            return self.store_bits(self.write_storage(state, sym), sym, (0, 0))
        return old['update_bits'](self, state, sym, *args, **kwargs)

    previous_assign = C.assign
    def composite_assign(self, stmt, paths):
        target, op, node = stmt.args
        sym = self.place(target)
        if has_projection(sym) and isinstance(sym.ty, A):
            if sym.ty.packed_bits and op.text in ("&=", "|=", "^=") and g['bit_literal_width'](node) is not None:
                window = g['Node'](target.token, "slice", (target, None, None, False))
                return self.assign_bit_span(window, op, node, paths)
            if not sym.ty.packed_bits and node.kind == "cast" and node.args[1] == "view":
                return previous_assign(self, stmt, paths)
            if op.text != '=' or sym.readonly:
                op.error('composite element assignment requires = and mutable storage')
            result = []
            copying = node.kind == "copy"
            exit_code = None

            def remember_exit(*lines):
                nonlocal exit_code
                candidate = "{ " + " ".join(lines) + " }"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate

            source = node.args[0] if copying else node
            self.propagating_expression_nodes |= (self.propagation_nodes(target)
                                                  | self.propagation_nodes(source, composite_literal=True))
            capture = self.has_effects(target) or self.has_effects(node)
            address = f"lan_row_dst_{op.line}_{op.col}"
            disjoint = copying and not (self.has_effects(target, allow_debug_checks=True)
                                        or self.has_effects(node.args[0], allow_debug_checks=True))
            for state in paths:
                dst = materialize(self, sym, state, propagate_exit=True)
                if dst.projection.always_returns:
                    remember_exit(f"(void)({dst.c});")
                    continue
                if copying:
                    src = checked_copy_source(self, dst.ty, node, state, propagate_exit=True)
                    if has_projection(src) and src.projection.always_returns:
                        remember_exit(f"(void)(&({dst.c}));", f"(void)({src.c});")
                        continue
                    disjoint = disjoint and disjoint_projections(self, dst, src, state, op)
                    content = record_value(self, src, state)
                else:
                    prefix = [f"{dst.ty.c} *{address} = &({dst.c});"] if capture else []
                    destination = f"(*{address})" if capture else dst.c
                    content, statements = initialize(self, dst.ty, node, destination, state, op,
                                                     f'{op.line}_{op.col}', fresh=False, propagate_exit=True)
                    if content is None:
                        remember_exit(*prefix, *statements)
                        continue
                    code = ["{", *prefix, *statements, "}"] if capture else statements
                if self.contains_record_links(dst.ty):
                    root = dst.projection.root
                    span = root.ty.projection if isinstance(root.ty, Slice) else None
                    owners = span.owners if span is not None else state.borrows.get(root.uid, (root,))
                    for owner in owners:
                        check_link_lifetimes(self, content, owner.depth, state, op)
                result.append(set_projected(self, dst, state, content))
                guard = " and ".join(state.guards) or "entry"
                if copying:
                    self.trace.append(f"  update entity {trace_location(dst)} [{guard}]: copy array from "
                                      f"{trace_location(src)}, len {{{dst.ty.length}}}")
                    emitted = dst, src
                else:
                    self.trace.append(f"  update entity {trace_location(dst)} [{guard}]: initialize array, "
                                      f"len {{{dst.ty.length}}}")
            if not result:
                return [], [exit_code, "__builtin_unreachable();"]
            if copying:
                dst, src = emitted
                copy = self.helper("disjoint_object_copy", None) if disjoint else "memmove"
                strategy = "disjoint, restrict copy" if disjoint else "may overlap, memmove"
                self.trace.append(f"  lower array row copy at {op.line}:{op.col}: {strategy}")
                code = [f"{{ void *{address} = &({dst.c}); {copy}({address}, &({src.c}), sizeof({dst.c})); }}"]
            return result, code
        return previous_assign(self, stmt, paths)
    C.assign = composite_assign
