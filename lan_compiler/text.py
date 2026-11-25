"""Literals, text spans and encoding proofs."""

from __future__ import annotations

from dataclasses import replace
from itertools import count

from value_domain import Domain

from .encoding import TextEncoding
from .domain_utils import c_int
from .typesys import Array, bit_literal_value, HeapArray, Scalar, Slice, Span, Symbol, Value, ZERO

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class TextMixin:
    def literal_storage(self, data, terminator=b"\0"):
        key = (data, terminator)
        if key not in self.text_literals:
            uid = -1 - len(self.text_literals)
            raw = data + terminator
            ty = Array(self.types["uint8"], max(1, len(raw)))
            symbol = Symbol("literal", uid, ty, 0, f"{self.helper_prefix()}_text_{-uid}")
            self.text_literals[key] = symbol
        return self.text_literals[key]

    def literal_data(self, span):
        if span.host is None or span.host.uid >= 0:
            return None
        return next(data + term for (data, term), sym in self.text_literals.items() if sym is span.host)

    def encode_literal_span(self, node, encoding):
        encoded = encoding.encode(node.args[0].decode("utf-8"), node.token)
        sym = self.literal_storage(encoded, encoding.terminator)
        return Span(self.types["uint8"], Value(Domain.point(len(encoded)), c_int(len(encoded))),
                    0, host=sym, readonly=True, text=encoding)

    def literal_bytes(self, span):
        if span.known_text is not None:
            return span.known_text
        data = self.literal_data(span)
        if data is not None:
            return data[span.offset:span.offset + span.length.domain.lo]
        return None

    def terminated(self, span, state=None, token=None):
        data = self.literal_data(span)
        if data is not None:
            end = span.offset + span.length.domain.lo
            return data[end:end + 1] == b"\0" and b"\0" not in data[span.offset:end]
        if state is None:
            return False
        if ((span.text == 'c' or span.terminated) and span.owners
                and all(isinstance(owner.ty, Slice) and owner.ty.text == 'c'
                        and state.epochs.get(owner.uid) is None for owner in span.owners)):
            return True
        if not self.byte_span(span):
            return False
        count = span.length.domain
        positions = span.positions()
        exact_end = self.span_end_domain(token, span, state, positions)
        for owner, offset in positions:
            if offset is None:
                return False
            contents = self.actual_array(state, owner)
            end = self.sum_index_domains(offset, count)
            if exact_end is not None:
                end &= exact_end
            indices = Domain.of((0, count.hi - 1))
            interior = self.sum_index_domains(offset, indices)
            if end:
                interior &= Domain.of((0, end.hi - 1))
            key = self.array_element_key(span, span.length)
            if (not end or not end.within(Domain.of((0, contents.length - 1)))
                    or self.known_array_element(state, key, self.read_array_contents(state, owner, end)) != ZERO
                    or interior and self.known_span_window(
                        state, span, indices, self.read_array_contents(state, owner, interior)) & ZERO):
                windows = self.span_candidate_windows(token, span, state)
                return bool(windows) and all(
                    end < self.actual_array(state, owner).length
                    and self.read_array_contents(state, owner, Domain.point(end)) == ZERO
                    and (start == end or not self.read_array_contents(state, owner, Domain.of((start, end - 1))) & ZERO)
                    for owner, start, end in windows
                )
        return bool(positions)

    def span_end_domain(self, token, span, state, positions):
        if token is not None and len(positions) == 1:
            start = self.span_start(span)
            if start is not None:
                return self.binary(token, "+", start, span.length, state).domain
        return None

    def span_proves_ascii(self, token, span, state):
        ascii_domain = self.types["ascii"].domain
        if span.element.domain.within(ascii_domain) or span.length.domain.hi == 0:
            return True
        limit = None
        if len(span.positions()) == 1:
            start = self.span_start(span)
            if start is not None:
                limit = self.binary(token, "+", start, span.length, state).domain.hi
        indices = Domain.of((0, span.length.domain.hi - 1))
        actual = self.read_span_contents(state, span, indices, absolute_limit=limit)
        return self.known_span_window(state, span, indices, actual).within(ascii_domain)

    def span_proves_utf8(self, token, span, state):
        if self.span_proves_ascii(token, span, state):
            return True
        windows = self.span_candidate_windows(token, span, state)
        return bool(windows) and all(
            self.utf8_byte_domains_valid(self.read_array_contents(state, owner, Domain.point(index))
                                         for index in range(start, end))
            for owner, start, end in windows
        )

    @staticmethod
    def bounded_window_values(domain, budget):
        residue, modulus = domain.congruence()
        step = max(modulus, 1)
        values = []
        for lo, hi in domain.ranges:
            first = lo + (residue - lo) % step
            budget -= (hi - first) // step + 1
            if budget < 0:
                return None
            values.extend(value for value in range(first, hi + 1, step) if domain & Domain.point(value))
        return values

    def span_candidate_windows(self, token, span, state):
        positions = span.positions()
        count = span.length.domain
        if not positions or count.hi > 256:
            return []
        if span.length.affine is not None:
            form = span.length.affine
            count = count.with_congruence(form.offset, abs(form.scale))
        lengths = self.bounded_window_values(count, 64)
        if not lengths:
            return []
        end_domain = self.span_end_domain(token, span, state, positions)
        absolute_start = self.span_start(span) if len(positions) == 1 else None
        window_budget = 64
        byte_budget = 2048
        windows = []
        for owner, offsets in positions:
            if offsets is None:
                return []
            if absolute_start is not None and absolute_start.affine is not None:
                form = absolute_start.affine
                offsets = offsets.with_congruence(form.offset, abs(form.scale))
            pairs = []
            if absolute_start is not None and self.related(absolute_start, span.length):
                start_value = replace(absolute_start, domain=offsets & absolute_start.domain)
                pairs = self.bit_window_candidates(start_value, replace(span.length, domain=count))
            if not pairs:
                starts = self.bounded_window_values(offsets, window_budget // len(lengths))
                if starts is None:
                    return []
                pairs = [(start, length) for start in starts for length in lengths]
            window_budget -= len(pairs)
            contents = self.actual_array(state, owner)
            for start, length in pairs:
                end = start + length
                if (end > contents.length
                        or end_domain is not None and not end_domain & Domain.point(end)):
                    continue
                byte_budget -= length
                if byte_budget < 0:
                    return []
                windows.append((owner, start, end))
        return windows

    @staticmethod
    def utf8_byte_domains_valid(domains):
        # State records the remaining continuation bytes and the next byte's range.
        boundary = (0, 0, 0)
        leads = (
            (Domain.of((0x00, 0x7F)), boundary),
            (Domain.of((0xC2, 0xDF)), (1, 0x80, 0xBF)),
            (Domain.point(0xE0), (2, 0xA0, 0xBF)),
            (Domain.of((0xE1, 0xEC), (0xEE, 0xEF)), (2, 0x80, 0xBF)),
            (Domain.point(0xED), (2, 0x80, 0x9F)),
            (Domain.point(0xF0), (3, 0x90, 0xBF)),
            (Domain.of((0xF1, 0xF3)), (3, 0x80, 0xBF)),
            (Domain.point(0xF4), (3, 0x80, 0x8F)),
        )
        valid_lead = Domain.of((0x00, 0x7F), (0xC2, 0xF4))
        states = {boundary}
        for byte in domains:
            if not byte:
                return False
            next_states = set()
            for remaining, lo, hi in states:
                if remaining:
                    if not byte.within(Domain.of((lo, hi))):
                        return False
                    next_states.add((remaining - 1, 0x80, 0xBF)
                                    if remaining > 1 else boundary)
                else:
                    if not byte.within(valid_lead):
                        return False
                    next_states.update(target for lead, target in leads if byte & lead)
            states = next_states
        return states == {boundary}

    def checked_slice_length(self, token, span, target, context):
        uncertain = self.proven_presence(span.length.domain - target.domain, token)
        if not self.debug_bounds or target.length_bound is None:
            if uncertain:
                token.error(f"cannot prove slice length {span.length.domain} fits {target.domain}")
            return span
        valid = span.length.domain & target.domain
        if not valid:
            token.error(f"cannot prove slice length {span.length.domain} fits {target.domain}")
        # A single emitted expression can represent several incoming analysis paths.
        descriptor = f"lan_checked_length_{token.line}_{token.col}"
        test = self.membership(token, target.domain)
        code = self.assumed_result("lan_slice", descriptor, span.c_value(),
                                   [f"{test}((lan_int){descriptor}.len)"], checked=True)
        length = replace(span.length, domain=valid, code=f"(lan_int)({code}).len", effectful=True)
        self.trace.append(f"  debug check slice {context} length at {token.line}:{token.col}: "
                          f"{valid} subset {target.domain}")
        return replace(span, host=None, offset=0, code=code, length=length, bounds_proven=False,
                       borrowed_from=span.owners, origins=span.positions())

    def cast_span(self, node, state):
        operand, strategy, annotation = node.args
        target = self.resolve(annotation)
        token = node.token
        if (operand.kind == "text_literal" and isinstance(target, Slice)
                and isinstance(target.text, TextEncoding) and strategy in ("safe", "view")):
            span = self.encode_literal_span(operand, target.text)
            span = self.checked_slice_length(token, span, target, "conversion")
            return replace(span, element=target.element, text=target.text, readonly=True)
        span = self.ordinary_span(operand, state)
        if span.length.always_returns:
            return replace(span, element=target.element, text=target.text, readonly=True)
        if target.text == "c":
            if strategy != "view" or not self.terminated(span, state, token):
                token.error("cannot prove a NUL terminator after the slice and no interior NUL")
        elif strategy == "trap":
            if target.text != "utf8" or not self.byte_span(span):
                token.error("slice trap supports byte slices converted to string")
            content = self.literal_bytes(span)
            if content is not None:
                try:
                    content.decode("utf-8")
                except UnicodeDecodeError:
                    token.error("invalid UTF-8 in string conversion")
            elif not self.span_proves_utf8(token, span, state):
                helper = self.helper("utf8_trap", None)
                span = replace(span, host=None, borrowed_from=span.owners, origins=span.positions(),
                               code=f"{helper}({span.c_value()})")
        elif strategy != "view":
            token.error("slice conversion requires 'as: view' or 'as: trap string'")
        if (not self.byte_span(span) or not isinstance(target.element, Scalar) or target.element.bits != 8
                or target.element.signed or target.element.boolean or target.element.enum
                or not span.element.domain.within(target.element.domain)):
            token.error("slice view must preserve the storage width and element value domain")
        span = self.checked_slice_length(token, span, target, "conversion")
        # Different element contracts cannot give a second writable interpretation.
        readonly = span.readonly or target.readonly or span.element != target.element
        return replace(span, element=target.element, text=target.text, readonly=readonly,
                       terminated=self.terminated(span, state, token))

    @staticmethod
    def byte_span(span):
        ty = span.element
        return isinstance(ty, Scalar) and ty.bits == 8 and not ty.signed and not ty.boolean and ty.enum is None

    def literal_code(self):
        return [f"static const uint8_t {sym.c}[] __attribute__((unused)) = "
                f"{{{', '.join(f'0x{byte:02x}' for byte in (data + term or bytes(1)))}}};"
                for (data, term), sym in self.text_literals.items()]

    def span_kind(self, node):
        if isinstance(self.composite_call_type(node), Slice):
            return "ordinary"
        if node.kind == "text_literal":
            return "ordinary"
        if node.kind == "cast" and node.args[1] in ("view", "trap", "safe"):
            operand, strategy, annotation = node.args
            target = self.resolve(annotation)
            if isinstance(target, Slice):
                if strategy in ("view", "trap") and self.span_kind(operand) == "ordinary":
                    return "ordinary"
                if strategy == "safe" and operand.kind == "text_literal" and isinstance(target.text, TextEncoding):
                    return "ordinary"
        if node.kind == "slice":
            return self.span_kind(node.args[0])
        place = self.place(node)
        if place is not None:
            ty = place.ty
            if isinstance(ty, (Slice, HeapArray)) or isinstance(ty, Array) and not ty.packed_bits:
                return "ordinary"
            if isinstance(ty, Array):
                return "packed"
        return None

    @staticmethod
    def literal_endpoint(node):
        # A singleton domain alone does not mean the expression can be erased:
        # it could contain a function call or a trapping conversion.
        if node.kind == "number":
            return True
        if node.kind == "unary" and node.args[0] in ("+", "-"):
            return Compiler.literal_endpoint(node.args[1])
        if node.kind == "binary" and node.args[0] in ("+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"):
            return all(Compiler.literal_endpoint(part) for part in node.args[1:])
        return False

    def const_int_endpoint(self, node, env=()):
        if self.literal_endpoint(node):
            return True
        if node.kind == "name":
            name, = node.args
            scope = self.compile_time_binding(name, env)
            if scope is not None:
                return type(scope[name][0]) is int
            if (runtime_const := self._active_runtime_constant(node)) is not None:
                return type(runtime_const[0]) is int
            if name in self.module_consts or name in self.const_nodes:
                entry = self.resolve_module_const(name)
                return entry is not None and type(entry.scalar_value) is int
            if (type_const := self._singleton_type_constant(name)) is not None:
                return type(type_const[0]) is int
            return False
        if node.kind == "member":
            if self.enum_member_type(node, env=env) is not None:
                return True
            if (runtime_const := self._active_runtime_constant(node)) is not None:
                return type(runtime_const[0]) is int
            return False
        if node.kind == "length":
            target = node.args[0]
            if target.kind == "text_literal" or bit_literal_value(target) is not None:
                return True
            if not self.has_effects(target):
                if (runtime_const := self._active_runtime_constant(node)) is not None:
                    return type(runtime_const[0]) is int
                place = self.place(target) if self.scope else None
                if place is not None and isinstance(place.ty, Array):
                    return True
            return False
        if node.kind == "unary" and node.args[0] in ("+", "-", "~"):
            return self.const_int_endpoint(node.args[1], env)
        if node.kind == "binary" and node.args[0] in ("+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"):
            return all(self.const_int_endpoint(part, env) for part in node.args[1:])
        return False
