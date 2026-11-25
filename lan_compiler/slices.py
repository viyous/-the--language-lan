"""Slice values, definitions, copies and assignment."""

from __future__ import annotations

from dataclasses import field, replace
from itertools import count

from difference_domain import Difference
from value_domain import Domain

from .encoding import TextEncoding
from .typesys import Array, BOOL, HeapArray, Scalar, Slice, Span, storage_size, Symbol, Value, ZERO
from .domain_utils import c_int, domain_union
from .interface_types import slice_offset_terms


class SliceMixin:
    def slice_endpoint(self, node, state, role):
        previous = self.trace_endpoint
        self.trace_endpoint = True
        try:
            value = self.expr(node, state)
        finally:
            self.trace_endpoint = previous
        self.trace.append(f"    eval {role} endpoint [{node.token.line}:{node.token.col}] => {value.domain}")
        return value

    @staticmethod
    def slice_endpoint_exit(host, *endpoints):
        parts = [f"(void)({part})" for part in (host.c_value(), *(value.code for value in endpoints))]
        return Span(host.element, Value(ZERO, "0", always_returns=True), host.depth,
                    code=f"({', '.join(parts)}, (lan_slice){{0}})")

    def comparison_spans(self, left, right, state):
        a = self.ordinary_span(left, state)
        b = None if a.length.always_returns else self.ordinary_span(right, state)
        if b is not None and not b.length.always_returns:
            return a, b, None
        parts = [f"(void)({span.c_value()})" for span in (a, b) if span is not None]
        exit_value = Value(BOOL, f"({', '.join(parts)}, 0)", 8, True, effectful=True, always_returns=True)
        return a, b, exit_value

    def ordinary_span(self, node, state, *, expected_encoding=None):
        span = self.ordinary_span_value(node, state, expected_encoding=expected_encoding)
        return replace(span, length=self.capture_value(span.length, state))

    def record_text_domain(self, sym, state):
        return domain_union(state.values[p.uid] for p in self.physical_places(sym, state))

    def record_text_span(self, sym, domain, state, *, data=None):
        ty = sym.ty
        owner = Symbol("record_text", -(1 << 61) - sym.uid, ty, readonly=True, static_storage=True)
        places = self.physical_places(sym, state) if sym.projection is None else ()
        length = Value(domain, f"(lan_int)({sym.c}).len")
        if len(places) == 1 and places[0].projection is None and not places[0].chain_summary:
            uid = places[0].uid
            length = replace(length, affine=state.forms.get(uid), difference=Difference(uid),
                             linear=state.linear.get(uid))
        return Span(ty.element, length, 0, code=sym.c,
                    borrowed_from=frozenset((owner,)), readonly=True, text=ty.text,
                    origins=((owner, ZERO),), known_text=data)

    def ordinary_span_value(self, node, state, *, expected_encoding=None):
        if isinstance(self.composite_call_type(node), Slice):
            arguments, early = [], []
            ty, code = self.call(node, state, arguments, capture_early_return=early)
            if early:
                return Span(ty.element, replace(early[0], domain=ZERO), 0,
                            code=f"((void)({code}), (lan_slice){{0}})")
            code = self.assume_slice_result(node.token, ty.domain, code)
            length = Value(ty.domain, f"(lan_int)({code}).len", effectful=True)
            if ty.source_param is not None:
                base = arguments[ty.source_param]
                if ty.source_field:
                    field = self.child_place(base, ty.source_field)
                    owners = frozenset(self.physical_places(field, state))
                    base = Span(field.ty.element, Value(Domain.point(field.ty.length), c_int(field.ty.length)),
                                max(owner.depth for owner in owners), readonly=base.readonly or field.readonly,
                                borrowed_from=owners, origins=tuple((owner, ZERO) for owner in owners))
                if slice_offset_terms(ty) or ty.offset_expression is not None:
                    offset = self.slice_offset_value(node.token, ty, arguments, state)
                elif ty.source_offset is not None:
                    offset = Value(Domain.point(ty.source_offset), c_int(ty.source_offset))
                else:
                    offset = None
                positions = (tuple((owner, None) for owner in base.owners) if offset is None
                             else self.shift_positions(base, offset.domain))
                return Span(ty.element, length, base.depth, code=code, borrowed_from=base.owners,
                            readonly=ty.readonly or base.readonly, text=ty.text,
                            origins=positions,
                            start=self.add_span_start(base, offset, state, node.token) if offset is not None else None)
            # Result calls may select different static objects even at the same site.
            # This owner is provenance only; no per-object facts are stored on it.
            owner = Symbol("static_result", -(1 << 62), ty, readonly=True, static_storage=True)
            return Span(ty.element, length, 0, code=code, borrowed_from=frozenset((owner,)),
                        readonly=True, text=ty.text, origins=((owner, ZERO),))
        if node.kind == "text_literal":
            if isinstance(expected_encoding, TextEncoding):
                return self.encode_literal_span(node, expected_encoding)
            data, = node.args
            encoding = "ascii" if data.isascii() or not data else "utf8"
            return Span(self.types[encoding], Value(Domain.point(len(data)), c_int(len(data))),
                        0, host=self.literal_storage(data), readonly=True, text=encoding)
        if node.kind == "cast" and self.span_kind(node) == "ordinary":
            return self.cast_span(node, state)
        sym = self.place(node)
        if sym is not None:
            ty = sym.ty
            if sym.link_access is not None and isinstance(ty, (Array, Slice)):
                propagate = self.has_propagation(node)
                sym = self.comparison_place(sym, state, allow_effectful_links=propagate,
                                            propagate_exit=propagate)
                if sym.projection is not None and sym.projection.always_returns:
                    code = f"((void)({sym.c}), (lan_slice){{0}})"
                    return Span(ty.element, Value(ZERO, "0", always_returns=True), sym.depth, code=code)
            if isinstance(ty, Array) and not ty.packed_bits:
                self.require_live(sym, state, node.token)
                owners = frozenset(self.physical_places(sym, state))
                depth = max(owner.depth for owner in owners)
                if sym.uid in self.alias_members or sym.link_access is not None:
                    if sym.link_access is None:
                        sym = self.comparison_place(sym, state)
                    return Span(ty.element, Value(Domain.point(ty.length), c_int(ty.length)), depth,
                                code=f"(lan_slice){{{sym.c}, {ty.length}}}", borrowed_from=owners, readonly=sym.readonly,
                                text=ty.encoding, origins=tuple((owner, ZERO) for owner in owners))
                return Span(ty.element, Value(Domain.point(ty.length), c_int(ty.length)),
                            depth, host=sym, offset=0, readonly=sym.readonly, text=ty.encoding)
            if isinstance(ty, HeapArray):
                self.require_live(sym, state, node.token)
                return Span(ty.element, Value(state.values[sym.uid], f"(lan_int){sym.c}.len",
                                              affine=state.forms.get(sym.uid), difference=Difference(sym.uid),
                                              linear=state.linear.get(sym.uid)),
                            sym.depth, code=sym.c, borrowed_from=frozenset((sym,)), readonly=sym.readonly,
                            origins=((sym, ZERO),))
            if isinstance(ty, Slice):
                if sym.c_place and sym.readonly and sym.uid not in state.borrows:
                    self.require_live(sym, state, node.token)
                    length = self.record_text_domain(sym, state)
                    place = sym if sym.link_access is not None else self.comparison_place(sym, state)
                    return self.record_text_span(place, length, state,
                                                 data=self.known_record_text(sym, state, length))
                if ty.projection is not None:
                    for owner in ty.projection.owners:
                        self.require_target_live(owner, state, node.token)
                    return ty.projection
                owners = state.borrows[sym.uid]
                for owner in owners:
                    self.require_target_live(owner, state, node.token)
                return Span(ty.element, Value(state.values[sym.uid], f"(lan_int){sym.c}.len",
                                              affine=state.forms.get(sym.uid),
                                              difference=Difference(sym.uid), linear=state.linear.get(sym.uid)),
                            max(owner.depth for owner in owners), code=sym.c, borrowed_from=owners,
                            readonly=ty.readonly, text=ty.text, terminated=ty.terminated,
                            origins=state.regions.get(sym.uid, ()), start=state.starts.get(sym.uid),
                            row_identity=state.predicates.get(("slice_row", sym.uid, state.epochs.get(sym.uid))))
        if node.kind != "slice":
            node.token.error("expected an ordinary array or slice")
        base, lower, upper, closed = node.args
        host = self.ordinary_span(base, state)
        if host.length.always_returns:
            return host
        lo = self.slice_endpoint(lower, state, "lower") if lower else Value(ZERO, "0")
        if lo.always_returns:
            return self.slice_endpoint_exit(host, lo)
        hi = self.slice_endpoint(upper, state, "upper") if upper else host.length
        if hi.always_returns:
            return self.slice_endpoint_exit(host, lo, hi)
        t = node.token
        if lo.floating or hi.floating or lo.boolean or hi.boolean or lo.enum or hi.enum or lo.character or hi.character:
            t.error("slice endpoints require integers")
        end_cmp = "<" if upper and closed else "<="
        nonnegative = lo.domain.lo >= 0
        ordered = self.prove_order(t, lo, "<=", hi, state)
        bounded = self.prove_order(t, hi, end_cmp, host.length, state)
        if not (nonnegative and ordered and bounded) and self.ssa_loop_slice(node):
            nonnegative = ordered = bounded = True
        proven = nonnegative and ordered and bounded
        if proven or self.debug_bounds:
            guard = state.guard_label()
            origin = self.span_origins(host)
            self.trace.append(f"  verify slice_bounds {origin} [{guard}] at {t.line}:{t.col}")
            for condition, established in (
                    (f"lower {lo.domain} >= 0", nonnegative),
                    (f"lower {lo.domain} <= upper {hi.domain}", ordered),
                    (f"upper {hi.domain} {end_cmp} host len {host.length.domain}", bounded)):
                status = "proven" if established else "requires Debug check"
                self.trace.append(f"    check {condition} => {status}")
        if self.debug_bounds:
            return self.debug_slice(node, state, host, lo, hi, proven)
        if not proven:
            t.error(f"cannot prove slice bounds: lower {lo.domain}, upper {hi.domain}, "
                    f"host length {host.length.domain}")
        length = self.binary(t, "-", hi, lo, state)
        if upper and closed:
            length = self.binary(t, "+", length, Value(Domain.point(1), "1"), state)
        length = Value(length.domain & Domain.of((0, host.length.domain.hi)),
                       length.code, affine=length.affine, difference=length.difference, linear=length.linear)
        text = "utf8" if host.text == "c" else host.text
        terminated = self.slice_preserves_terminator(t, host, hi if upper else None, closed, state)
        origin = self.span_origins(host)
        self.trace.append(f"  verify slice {origin}: lower {lo.domain}, upper {hi.domain}, "
                          f"host len {host.length.domain}, result len {length.domain} => proven")
        if (host.host is not None
                and lo.domain.lo == lo.domain.hi and hi.domain.lo == hi.domain.hi
                and (lower is None or self.fixed_slice_endpoint(lower))
                and (upper is None or self.fixed_slice_endpoint(upper))):
            return Span(host.element, Value(length.domain, c_int(length.domain.lo)),
                        host.depth, host=host.host, offset=host.offset + lo.domain.lo,
                        readonly=host.readonly, text=text, terminated=terminated,
                        borrowed_from=host.borrowed_from,
                        origins=self.shift_positions(host, lo.domain) if host.origins else (),
                        relative_bounds=(lo, hi, bool(upper and closed)), row_identity=host.row_identity)
        assume = self.helper("contract_assume", None)
        helper = self.helper("slice_sub", host.element.c, assume)
        # An omitted upper bound is conveyed separately so a nested base is evaluated once.
        end = f"(size_t)({hi.code}) + {int(closed)}" if upper else "0"
        codes = [host.c_value(), f"(size_t)({lo.code})", end, str(int(upper is None))]
        code = (self.ordered_call(t, helper, ["lan_slice", "size_t", "size_t", "int"], codes)
                if self.has_effects(node) else f"{helper}({', '.join(codes)})")
        return Span(host.element, length, host.depth, code=code, borrowed_from=host.owners,
                    readonly=host.readonly, text=text, terminated=terminated,
                    origins=self.shift_positions(host, lo.domain), start=self.add_span_start(host, lo, state, t),
                    relative_bounds=(lo, hi, bool(upper and closed)), row_identity=host.row_identity)

    def fixed_slice_endpoint(self, node):
        if self.literal_endpoint(node):
            return True
        if node.kind == "name":
            ty = self.lookup(node).ty
            return isinstance(ty, Scalar) and not ty.floating and ty.domain.lo == ty.domain.hi
        if node.kind == "length":
            base = node.args[0]
            if self.has_effects(base):
                return False
            while base.kind == "slice":
                base, lower, upper, _ = base.args
                if any(end is not None and not self.fixed_slice_endpoint(end) for end in (lower, upper)):
                    return False
            if base.kind == "text_literal":
                return True
            symbol = self.place(base)
            if symbol is None:
                return False
            ty = symbol.ty
            if isinstance(ty, Array):
                return True
            if isinstance(ty, Slice):
                length = ty.projection.length.domain if ty.projection is not None else ty.domain
                return length.lo == length.hi
            return False
        if node.kind == "cast":
            operand, strategy, _ = node.args
            return strategy == "safe" and self.fixed_slice_endpoint(operand)
        if node.kind == "unary" and node.args[0] in ("+", "-"):
            return self.fixed_slice_endpoint(node.args[1])
        if node.kind == "binary" and node.args[0] in ("+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"):
            return all(self.fixed_slice_endpoint(part) for part in node.args[1:])
        return False

    def slice_preserves_terminator(self, token, host, upper, inclusive, state):
        if not self.terminated(host, state, token):
            return False
        if upper is None:
            return True
        if inclusive:
            upper = self.binary(token, "+", upper, Value(Domain.point(1), "1"), state)
        return self.prove_order(token, upper, "==", host.length, state)

    def debug_slice(self, node, state, host, lo, hi, proven):
        _, _, upper, closed = node.args
        token = node.token
        inclusive = int(bool(upper and closed))
        maximum = host.length.domain.hi
        if not proven:
            valid_hi = hi.domain & Domain.of((0, maximum - inclusive))
            valid_lo = lo.domain & Domain.of((0, valid_hi.hi)) if valid_hi else Domain.of()
            if not valid_lo:
                token.error("slice endpoints cannot form an in-bounds range")
            lo = replace(lo, domain=valid_lo)
            hi = replace(hi, domain=valid_hi)
            self.trace.append(f"  debug check slice {self.span_origins(host)} at {token.line}:{token.col}")
        length = self.binary(token, "-", hi, lo, state)
        if inclusive:
            length = self.binary(token, "+", length, Value(Domain.point(1), "1"), state)
        valid_length = length.domain & Domain.of((inclusive, maximum))
        if not valid_length:
            token.error("slice endpoints cannot form an in-bounds range")
        length = replace(length, domain=valid_length)
        if not proven:
            lo = replace(lo, domain=lo.domain & Domain.of((0, maximum - valid_length.lo)))
            if not lo.domain:
                token.error("slice endpoints cannot form an in-bounds range")
            hi = replace(hi, domain=hi.domain & Domain.of((lo.domain.lo + valid_length.lo - inclusive,
                                                          maximum - inclusive)))
            if not hi.domain:
                token.error("slice endpoints cannot form an in-bounds range")
        self.trace.append(f"  verify slice {self.span_origins(host)}: lower {lo.domain}, upper {hi.domain}, "
                          f"host len {host.length.domain}, result len {valid_length} => "
                          f"{'proven' if proven else 'checked at runtime'}")
        assume = self.helper("contract_assume", None)
        helper = self.helper("debug_slice_sub", host.element.c, assume)
        codes = [host.c_value(), lo.code, hi.code if upper else "0", str(int(upper is None)), str(inclusive)]
        code = self.ordered_call(token, helper, ["lan_slice", "lan_int", "lan_int", "int", "int"], codes)
        length = replace(length, code=f"(lan_int)({code}).len", effectful=True)
        return Span(host.element, length, host.depth, code=code, borrowed_from=host.owners,
                    readonly=host.readonly, text="utf8" if host.text == "c" else host.text,
                    terminated=(upper is None or proven) and self.slice_preserves_terminator(
                        token, host, hi if upper else None, inclusive, state),
                    origins=self.shift_positions(host, lo.domain), start=self.add_span_start(host, lo, state, token),
                    relative_bounds=(lo, hi, bool(inclusive)), row_identity=host.row_identity,
                    bounds_proven=proven)

    def slice_length(self, node, state):
        span = self.ordinary_span(node, state)
        old = span.length
        if old.always_returns:
            return replace(old, code=f"((void)({span.c_value()}), 0)", effectful=True)
        code = old.code if span.host is not None or node.kind == "name" else f"(lan_int)({span.code}).len"
        effectful = self.has_effects(node)
        if span.host is not None and effectful:
            code = f"((void)({span.c_data()}), {code})"
        return Value(old.domain, code, 0 if span.host else 32,
                     affine=old.affine, difference=old.difference, linear=old.linear, effectful=effectful)

    @staticmethod
    def is_copy(node):
        return node.kind == "copy"

    def check_span_contract(self, token, span, ty, state=None, *, descriptor_only=False,
                            runtime_elements=False):
        original = span
        compatible_text = ty.text == "utf8" and span.text in ("ascii", "c")
        if ty.text == "c":
            if not descriptor_only and not self.terminated(span, state, token):
                token.error('cannot prove a NUL terminator after the slice and no interior NUL')
            compatible_text = span.text in ("ascii", "utf8", "c")
            if compatible_text:
                span = replace(span, element=ty.element, text="c")
        if ty.text is not None and ty.text != span.text and not compatible_text:
            token.error(f"slice does not satisfy the {ty.text} text contract")
        if span.element != ty.element and not compatible_text:
            token.error("slice requires the same element type")
        if span.readonly and not ty.readonly:
            token.error("cannot use an immutable slice as writable storage")
        if self.proven_presence(span.length.domain - ty.domain, token):
            token.error(f"cannot prove slice length {span.length.domain} fits {ty.domain}")
        if not (descriptor_only or runtime_elements) and ty.content_bound is not None and span.length.domain.hi:
            actual = span.element.domain
            if state is not None:
                indices = Domain.of((0, span.length.domain.hi - 1))
                actual = self.read_span_contract_contents(token, state, span, indices, ty.content_bound)
            if not actual.within(ty.content_bound):
                token.error(f'cannot prove slice element contents {actual} fit {ty.content_bound}')
        if descriptor_only and ty.text == "c":
            return replace(original, readonly=ty.readonly)
        return replace(span, element=ty.element, text=ty.text, readonly=ty.readonly)

    def static_slice_projection(self, node, spans, paths):
        first = spans[0]
        if first.host is not None:
            return first if first.host.projection is None else None
        if node.kind != "slice" or self.has_effects(node):
            return None
        projection = None
        for span, state in zip(spans, paths):
            lower, _, _ = span.relative_bounds
            count = span.length.domain
            if lower.domain.lo != lower.domain.hi or count.lo != count.hi:
                return None
            base = self.ordinary_span(node.args[0], state)
            if base.host is None or base.host.projection is not None:
                return None
            candidate = replace(span, host=base.host, offset=base.offset + lower.domain.lo,
                                code="", length=Value(count, c_int(count.lo)))
            if projection is not None and (candidate.host != projection.host
                    or candidate.offset != projection.offset or count != projection.length.domain
                    or candidate.positions() != projection.positions()):
                return None
            projection = candidate
        return projection

    def slice_binding_assumptions(self, token, sym, spans):
        domain = Domain.union(span.length.domain for span in spans)
        predicates = self.slice_descriptor_predicates(token, sym.c, domain)
        if not predicates:
            return []
        assume = self.helper("contract_assume", None)
        self.trace.extend(f"  assume slice binding {predicate}" for predicate in predicates)
        return [f"{assume}({predicate});" for predicate in predicates]

    def define_slice(self, name, node, paths, ty=None):
        self.propagating_expression_nodes |= self.propagation_nodes(node, composite_call=True, condition=True)
        continuing, spans = [], []
        exit_code = None
        expected_enc = ty.text if ty is not None and isinstance(ty.text, TextEncoding) else None
        for path in paths:
            span = self.ordinary_span(node, path, expected_encoding=expected_enc)
            if span.length.always_returns:
                candidate = f"(void)({span.c_value()});"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            continuing.append(path)
            spans.append(span)
        if not continuing:
            return [], [exit_code, "__builtin_unreachable();"]
        paths = continuing
        first = spans[0]
        ty = ty if ty is not None else Slice(first.element, readonly=first.readonly, text=first.text)
        checked_length = (self.debug_bounds and ty.length_bound is not None
                          and any(self.proven_presence(span.length.domain - ty.domain, node.token) for span in spans))
        if checked_length:
            spans = [self.checked_slice_length(node.token, span, ty, "binding") for span in spans]
        spans = [self.check_span_contract(node.token, span, ty, path) for span, path in zip(spans, paths)]
        first = spans[0]
        stable = name.text not in self.rebound_slices
        projection = self.static_slice_projection(node, spans, paths) if stable and not checked_length else None
        ty = replace(ty, projection=projection, terminated=stable and all(self.terminated(s, p, name) for s, p in zip(spans, paths)))
        sym = self.declare(name, ty)
        for path, span in zip(paths, spans):
            self.trace_slice_binding("define entity", sym, span, path)
        if projection is not None:
            return paths, []
        code = [f"lan_slice {sym.c} = {first.c_value()};", *self.slice_binding_assumptions(name, sym, spans)]
        return [p.bind_span(sym, span) for p, span in zip(paths, spans)], code

    def span_origins(self, span):
        if any(owner.uid in self.param_places and isinstance(owner.ty, Slice) for owner in span.owners):
            return "caller storage"
        return " | ".join("read-only literal" if uid < 0 else f"id#{uid}" if uid else "caller storage"
                          for uid in sorted(owner.uid for owner in span.owners))

    @staticmethod
    def trace_span_positions(span):
        positions = sorted((owner.uid, str(offset) if offset is not None else "unknown")
                           for owner, offset in span.positions())
        return " | ".join(f"id#{uid} at {offset}" for uid, offset in positions) or "unknown"

    def trace_slice_binding(self, action, sym, span, path):
        guard = path.guard_label()
        offsets = self.trace_span_positions(span)
        content = "" if sym.ty.content_bound is None else f", elem contract {sym.ty.content_bound}"
        self.trace.append(f"  {action} id#{sym.uid} {sym.name} [{guard}]: view of {self.span_origins(span)}, "
                          f"len {span.length.domain}, elem {span.element.domain}, positions {offsets}{content}")

    def slice_copy_content_paths(self, token, target, source, paths, pairs):
        extra_paths = self.PATH_LIMIT - len(paths)
        result = []

        def exact_pair(dst, src):
            if dst.length.domain != src.length.domain or dst.length.domain.lo != dst.length.domain.hi:
                return False
            return all(len(positions) == 1 and positions[0][1] is not None
                       and positions[0][1].lo == positions[0][1].hi
                       for positions in (dst.positions(), src.positions()))

        def exact_windows(state, dst, src, capacity, remaining=2):
            if not (self.span_has_effects(target, dst) or self.span_has_effects(source, src)):
                dst, src = self.refine_span_position(state, dst), self.refine_span_position(state, src)
            if exact_pair(dst, src):
                return [(state, dst, src)]
            if not remaining or self.span_has_effects(target, dst) or self.span_has_effects(source, src):
                return None
            for value in (self.span_start(dst), self.span_start(src), dst.length, src.length):
                if value is None:
                    continue
                candidates = self.split_affine_value(value.domain, value.affine, state)
                if not candidates or len(candidates) > capacity or candidates[0] is state:
                    continue
                windows = []
                for index, candidate in enumerate(candidates):
                    # Reserve one path for each sibling that still needs analysis.
                    available = capacity - len(windows) - (len(candidates) - index - 1)
                    narrow_dst = self.ordinary_span(target, candidate)
                    narrow_src = self.ordinary_span(source, candidate)
                    branch = exact_windows(candidate, narrow_dst, narrow_src, available, remaining - 1)
                    if branch is None:
                        break
                    windows.extend(branch)
                else:
                    return windows
            return None

        for state, (dst, src) in zip(paths, pairs):
            windows = exact_windows(state, dst, src, extra_paths + 1) if extra_paths > 0 else None
            if windows is None:
                windows = [(state, dst, src)]
            extra_paths -= len(windows) - 1
            for path, dest, src in windows:
                pure = not (self.span_has_effects(target, dest) or self.span_has_effects(source, src))
                stable = pure or (self.debug_bounds and not self.has_effects(target, allow_debug_checks=True)
                                  and not self.has_effects(source, allow_debug_checks=True))
                values = []
                uniform = None
                if stable and isinstance(dest.element, Scalar):
                    indices = Domain.of((0, src.length.domain.hi - 1))
                    if indices:
                        actual = self.read_span_contents(path, src, indices)
                        uniform = self.known_span_window(path, src, indices, actual)
                    for index in range(min(8, dest.length.domain.lo, src.length.domain.lo)):
                        relative = Domain.point(index)
                        key = self.array_element_key(src, Value(relative, ""))
                        actual = self.read_span_contents(path, src, relative)
                        values.append(self.known_array_element(path, key, actual))
                updated = self.copy_span_contents(path, dest, src, token, correlated=pure)
                if values:
                    updated = self.remember_array_window(updated, dest, values)
                if uniform and uniform != src.element.domain:
                    updated = self.remember_uniform_array_window(updated, dest, uniform)
                result.append(updated)
        return result

    def slice_copy_disjoint(self, token, target, source, state, dst, src, *, splits_left=2):
        pure = not (self.span_has_effects(target, dst) or self.span_has_effects(source, src))
        if pure:
            dst, src = self.refine_span_position(state, dst), self.refine_span_position(state, src)
        if self.disjoint_spans(dst, src):
            return True
        if not pure:
            if self.has_effects(target, allow_debug_checks=True) or self.has_effects(source, allow_debug_checks=True):
                return False
            starts = self.slice_address_starts(target, source, dst, src, state)
            return self.disjoint_captured_windows(token, dst, src, state, starts)
        if (dst.element == src.element and storage_size(dst.element) == 1
                and max(dst.length.domain.hi, src.length.domain.hi) <= 1
                and self.slice_address_domain(token, target, source, dst, src, state) == ZERO):
            return True
        if target.kind == source.kind == "slice":
            base = self.predicate_key(target.args[0], state)
            if (base is not None and base == self.predicate_key(source.args[0], state)
                    and (self.slice_ends_before(token, dst, src, state)
                         or self.slice_ends_before(token, src, dst, state))):
                return True
        starts = self.slice_address_starts(target, source, dst, src, state)
        if self.disjoint_captured_windows(token, dst, src, state, starts):
            return True
        if splits_left:
            for start in starts or (self.span_start(dst), self.span_start(src)):
                if start is None:
                    continue
                candidates = self.split_affine_value(start.domain, start.affine, state)
                if candidates and (len(candidates) != 1 or candidates[0] is not state):
                    return all(self.slice_copy_disjoint(
                        token, target, source, candidate,
                        self.ordinary_span(target, candidate), self.ordinary_span(source, candidate),
                        splits_left=splits_left - 1) for candidate in candidates)
        return False

    def equal_slice_lengths(self, token, left, right, state):
        if self.prove_order(token, left, "==", right, state):
            return True
        a, b = left.affine, right.affine
        if a is None or b is None or not a.scale or a.scale != b.scale:
            return False
        origins = {}
        for uid, form in state.forms.items():
            if form.scale == 1 and uid in state.zone.ids:
                origins.setdefault(form.origin, Difference(uid, offset=-form.offset))
        if a.origin not in origins or b.origin not in origins:
            return False
        delta = origins[a.origin].combine(origins[b.origin], -1)
        bounds = state.zone.range(delta)
        return bounds is not None and bounds[0] == bounds[1] and a.scale * bounds[0] + a.offset == b.offset

    def assign_slice(self, target, op, node, paths, rebind=False):
        if op.text != "=":
            op.error("slice assignment uses '='")
        exit_code = None

        def remember_exit(*codes):
            nonlocal exit_code
            candidate = "{ " + " ".join(f"(void)({code});" for code in codes) + " }"
            if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                exit_code = candidate

        self.propagating_expression_nodes |= self.propagation_nodes(target, condition=True)
        if self.is_copy(node):
            source = node.args[0]
            self.propagating_expression_nodes |= self.propagation_nodes(source, condition=True)
            continuing, pairs = [], []
            for path in paths:
                dst, src, exited = self.comparison_spans(target, source, path)
                if exited is not None:
                    remember_exit(exited.code)
                    continue
                continuing.append(path)
                pairs.append((dst, src))
            if not continuing:
                return [], [exit_code, "__builtin_unreachable();"]
            paths = continuing
            independent = True
            checked = False
            for index, (p, (dst, src)) in enumerate(zip(paths, pairs)):
                if not (self.span_has_effects(target, dst) or self.span_has_effects(source, src)):
                    dst, src = self.refine_span_position(p, dst), self.refine_span_position(p, src)
                    pairs[index] = dst, src
                if dst.readonly:
                    op.error("cannot write to an immutable slice")
                text_bytes = (src.text is not None and self.byte_span(dst)
                              and dst.element.character is None and src.element.domain.within(dst.element.domain))
                if dst.element != src.element and not text_bytes:
                    op.error("slice copy requires the same element type")
                if not self.equal_slice_lengths(op, dst.length, src.length, p):
                    common = dst.length.domain & src.length.domain
                    if not self.debug_bounds or not common:
                        op.error(f"cannot prove slice copy lengths are equal: {dst.length.domain} vs {src.length.domain}")
                    checked = True
                    dst = replace(dst, length=replace(dst.length, domain=common))
                    src = replace(src, length=replace(src.length, domain=common))
                    pairs[index] = dst, src
                independent &= self.slice_copy_disjoint(op, target, source, p, dst, src)
            paths = self.slice_copy_content_paths(op, target, source, paths, pairs)
            dst, src = pairs[0]
            strategy = "memcpy" if independent else "memmove"
            assume = self.helper("contract_assume", None)
            helper = self.helper("slice_copy", strategy, (assume, self.debug_bounds))
            status = "debug checked" if checked else "proven"
            self.trace.append(f"  verify slice copy lengths equal => {status}; {strategy}")
            codes = [dst.c_value(), src.c_value(), f"sizeof({dst.element.c})"]
            call = (self.ordered_call(op, helper, ["lan_slice", "lan_slice", "size_t"], codes)
                    if self.has_effects(target) or self.has_effects(node) else f"{helper}({', '.join(codes)})")
            return paths, [call + ";"]
        if rebind:
            sym = self.lookup(target)
            self.propagating_expression_nodes |= self.propagation_nodes(node, composite_call=True, condition=True)
            continuing, spans = [], []
            expected_enc = sym.ty.text if isinstance(sym.ty.text, TextEncoding) else None
            for p in paths:
                span = self.ordinary_span(node, p, expected_encoding=expected_enc)
                if span.length.always_returns:
                    remember_exit(span.c_value())
                    continue
                continuing.append(p)
                if self.debug_bounds and sym.ty.length_bound is not None:
                    span = self.checked_slice_length(op, span, sym.ty, "rebind")
                span = self.check_span_contract(op, span, sym.ty, p)
                if sym.ty.graph:
                    self.verify_live_graph(self.span_link_targets(span, p), p, op)
                elif sym.ty.chain_limit is not None:
                    for root in self.span_link_targets(span, p):
                        if not root.uid:
                            continue
                        self.verify_chain(root, self.null_symbol, p, op)
                if span.depth > sym.depth:
                    op.error("slice rebind would outlive its array storage")
                spans.append(span)
                self.trace_slice_binding("update entity", sym, span, p)
            if not continuing:
                return [], [exit_code, "__builtin_unreachable();"]
            self.note_transition_write(sym, descriptor=True)
            code = [f"{sym.c} = {spans[0].c_value()};", *self.slice_binding_assumptions(op, sym, spans)]
            return [p.bind_span(sym, span) for p, span in zip(continuing, spans)], code
        if self.span_kind(node):
            op.error("copying into a slice range requires 'as: copy'")
        continuing = []
        for p in paths:
            dst = self.ordinary_span(target, p)
            if dst.length.always_returns:
                remember_exit(dst.c_value())
                continue
            pure = not self.span_has_effects(target, dst)
            write_dst = self.refine_span_position(p, dst) if pure else dst
            end = self.span_end_domain(op, write_dst, p, write_dst.positions()) if pure else None
            val = self.expr(node, p)
            if val.always_returns:
                remember_exit(dst.c_value(), val.code)
                continue
            if dst.readonly:
                op.error("cannot write to an immutable slice")
            val = self.coerce_storage(op, val, dst.element)
            self.range_check(op, val, dst.element)
            updated = self.fill_span_contents(p, write_dst, val.domain, end)
            stable = pure or (self.debug_bounds and not self.has_effects(target, allow_debug_checks=True))
            if stable and not self.has_effects(node):
                updated = self.remember_uniform_array_window(updated, write_dst, val.domain)
            p.absorb(updated)
            continuing.append(p)
            emitted_dst, emitted_val = dst, val
        if not continuing:
            return [], [exit_code, "__builtin_unreachable();"]
        dst, val = emitted_dst, emitted_val
        assume = self.helper("contract_assume", None)
        helper = self.helper("slice_fill", dst.element.c, assume)
        codes = [dst.c_value(), val.code]
        call = (self.ordered_call(op, helper, ["lan_slice", dst.element.c], codes)
                if self.has_effects(target) or self.has_effects(node) else f"{helper}({', '.join(codes)})")
        return continuing, [call + ";"]
