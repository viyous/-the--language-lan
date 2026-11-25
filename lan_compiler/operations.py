"""Range checks, SSA hooks, subscripts, casts and binary operations."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import count
from math import gcd

from difference_domain import Difference
from optimization_bridge import LoopReplay
from bit_domain import bitwise_domain
from float_domain import FloatDomain, FloatDomainError, c_literal, conversion_input_domain, binary_format
from value_domain import ACTIVE_ORACLES, Domain, DomainComplexity
from table_interface import c_field

from .typesys import (AffineForm, Array, bit_literal_width, BOOL, CopyArray, is_callback_enum, Record,
                      require_bit_literal_width, Scalar, Slice, Symbol, unqualified, Value, WIDE, ZERO)
from .domain_utils import c_int, c_uint, container, plain_integer_parameter, trunc_div
from .state import PathState
from .interface_types import product_congruence, square_domain
from .c_emit import c_predicate_sequence


class OperationMixin:
    @staticmethod
    def check_integer_view(t, val, width):
        if is_callback_enum(val.enum):
            t.error("function values cannot be reinterpreted as bits")
        if val.boolean or val.width != width:
            t.error("integer view requires source and target of the same fixed width")
        if not any(val.domain.within(container(width, signed)) for signed in (False, True)):
            t.error("integer view requires a value representable at its source width")

    def range_check(self, t, value, ty, explicit=False):
        if is_callback_enum(value.enum) != is_callback_enum(ty.enum):
            t.error("a callable value must be assigned to a compatible callable type")
        if ty.floating:
            value = self.float_coerce(t, value, ty.bits)
        elif value.floating:
            t.error("float-to-integer conversion requires an explicit rounding strategy")
        if value.boolean != ty.boolean:
            t.error("implicit conversion between boolean and integer is not allowed")
        if not explicit and value.enum != ty.enum:
            enum_error = (value.enum is not None and ty.result_arms is not None
                          and value.enum.identity in ty.result_error_enums
                          and value.domain.within(ty.result_arms[1]))
            enum_to_integer = value.enum is not None and ty.enum is None and ty.result_arms is None
            if not (enum_error or enum_to_integer):
                t.error("enum identity differs; use an explicit conversion")
        if not explicit and value.character != ty.character:
            if not (value.character == "ascii" and ty.character is not None):
                t.error("character encoding differs; use an explicit conversion")
        outside = value.domain - ty.domain
        if self.proven_presence(outside, t):
            detail = ""
            form = value.affine
            if form is not None and isinstance(outside, Domain):
                source = getattr(self, "entry_offset_origins", {}).get(form.origin)
                if source is not None and source[0] == "param":
                    try:
                        example = (form.origin.domain & form.preimage(outside)).witness()
                    except DomainComplexity:
                        example = None
                    if example is not None:
                        parameter = self.functions[self.current][0].args[2][source[1]]
                        detail = f"; candidate entry {parameter.text}={example} gives {form.scale * example + form.offset}"
            t.error(f"cannot prove {value.domain} fits {ty.domain}{detail}")

    def ssa_narrow(self, node, value, domain, kind, label):
        replay = getattr(self, "optimization_replay", None)
        if (replay is None or value.floating or value.boolean or value.enum or value.character
                or domain.predicates or value.domain.within(domain)):
            return value
        if isinstance(replay, LoopReplay):
            result = replay.range_result(node, domain.intervals, kind)
            if result is None or result.status != "proved":
                return value
            if note := replay.report(node, label, result):
                self.trace.append(note)
        else:
            if not replay.proves_range(node, domain.intervals, kind):
                return value
            self.trace.append(f"  ssa {label} proved at {node.token.line}:{node.token.col}")
        return replace(value, domain=(value.domain & domain) or domain)

    def ssa_loop_index(self, node, value, length):
        replay = getattr(self, "optimization_replay", None)
        result = replay.access_result(node) if isinstance(replay, LoopReplay) else None
        if result is None or result.status != "proved":
            return value, False
        bound = Domain.of((0, length.domain.hi - 1))
        if not bound:
            return value, False
        if note := replay.report(node, "index", result):
            self.trace.append(note)
        return replace(value, domain=(value.domain & bound) or bound), True

    def ssa_loop_slice(self, node):
        replay = getattr(self, "optimization_replay", None)
        result = replay.slice_result(node) if isinstance(replay, LoopReplay) else None
        if result is None or result.status != "proved":
            return False
        if note := replay.report(node, "slice bounds", result):
            self.trace.append(note)
        return True

    def ssa_call_order(self, node, target, op, term):
        replay = getattr(self, "optimization_replay", None)
        if replay is None:
            return False
        result = replay.argument_result(node, target, op, term)
        if result is None or result.status != "proved":
            return False
        if isinstance(replay, LoopReplay):
            if note := replay.report(node, "dependent argument", result):
                self.trace.append(note)
        else:
            self.trace.append(f"  ssa dependent argument proved at {node.token.line}:{node.token.col}")
        return True

    def emit_value(self, t, domain, code, width=0, boolean=False, affine=None, difference=None, signed=False):
        if not domain.within(WIDE):
            t.error("integer intermediate exceeds the supported signed 128-bit range")
        return Value(domain, code, width, boolean, affine, difference, signed=signed)

    def helper(self, kind, domain, extra=None):
        key = (kind, domain, extra)
        if key not in self.helpers:
            self.helpers[key] = f"{self.helper_prefix()}_{kind}_{len(self.helpers)}"
        return self.helpers[key]

    def allocating_helper(self, kind):
        self.memory_effects.add(self.current)
        return self.helper(kind, None)

    def integer_range_predicates(self, token, value, domain, physical_domain):
        if domain == physical_domain or not domain or not physical_domain:
            return []
        unsigned = physical_domain.lo >= 0 and physical_domain.hi >= (1 << 63)
        cast = "(lan_uint)" if unsigned else "(lan_int)"
        if domain.lo == domain.hi:
            return [f"{cast}{value} == {cast}{c_int(domain.lo)}"]
        predicates = []
        if domain.lo > physical_domain.lo:
            predicates.append(f"{cast}{value} >= {cast}{c_int(domain.lo)}")
        if domain.hi < physical_domain.hi:
            predicates.append(f"{cast}{value} <= {cast}{c_int(domain.hi)}")
        for (_, high), (low, _) in zip(domain.ranges, domain.ranges[1:]):
            predicates.append(c_predicate_sequence(
                [f"{cast}{value} <= {cast}{c_int(high)}", f"{cast}{value} >= {cast}{c_int(low)}"],
                conjunction=False))
        if domain.predicates:
            clause = domain.predicates[0]
            if (not domain.intervals and len(domain.predicates) == 1 and not clause.exclude
                    and len(clause.require) == 1 and len(clause.require[0].coefficients) == 2):
                step = c_uint(clause.require[0].modulus)
                base = c_int(clause.lo)
                predicates.append(f"((lan_uint){cast}{value} - (lan_uint){base}) % {step} == 0")
            else:
                d_cast = "(lan_uint)" if domain.lo >= 0 and domain.hi >= (1 << 63) else "(lan_int)"
                predicates.append(f"{self.membership(token, domain)}({d_cast}{value})")
        return predicates

    def float_range_predicates(self, token, value, domain):
        for piece in domain.filters:
            integers = piece.integer if piece.integer is not None else piece.excluded
            if integers.predicates and not integers.within(WIDE):
                # The integer membership helper takes a signed 128-bit value.
                domain = replace(domain, filters=())
                break
        if domain == FloatDomain.top(domain.bits):
            return []
        return [f"{self.membership(token, domain)}({value})"]

    def membership(self, token, domain):
        if isinstance(domain, FloatDomain):
            predicates = []
            for piece in domain.filters:
                integer = piece.integer if piece.integer is not None else piece.excluded
                integer_helper = None
                if integer.predicates:
                    if not integer.within(WIDE):
                        token.error("floating polynomial membership exceeds the signed 128-bit integer proof backend")
                    integer_helper = self.membership(token,integer)
                bounds = FloatDomain.union(FloatDomain.interval(lo,hi,domain.bits) for lo,hi in integer.ranges)
                bound_helper = self.membership(token,bounds)
                predicates.append(self.helper("float_integer_in",None,(domain.bits,bound_helper,integer_helper)))
            return self.helper("float_in", domain,tuple(predicates))
        for clause in domain.predicates:
            for poly in clause.require + clause.exclude:
                self.helper("poly", poly)
        return self.helper("in", domain)

    def ordered_call(self, token, callee, types, codes):
        return f"{callee}({', '.join(codes)})"

    def assumed_index(self, code, length):
        assume = self.helper("contract_assume", None)
        helper = self.helper("assume_index", None, assume)
        return f"{helper}({code}, {length})"

    def subscript(self, expr, state, capture_index=None, *, propagate_exit=False):
        host, idx = expr.args
        if self.span_kind(host) == "ordinary":
            span = self.ordinary_span(host, state)
            if propagate_exit and span.length.always_returns:
                return replace(span.length, code=f"((void)({span.c_value()}), 0)", effectful=True)
            value = self.expr(idx, state)
            if propagate_exit and value.always_returns:
                return replace(value, code=f"((void)({span.c_value()}), ({value.code}))")
            bound = Domain.of((0, span.length.domain.lo - 1))
            if not isinstance(getattr(self, "optimization_replay", None), LoopReplay):
                value = self.ssa_narrow(idx, value, bound, "operands", "index")
            integer = not (value.floating or value.boolean or value.enum or value.character)
            proven = integer and value.domain.lo >= 0 and self.prove_order(idx.token, value, "<", span.length, state)
            if not proven and integer and idx.kind == "binary" and idx.args[0] == "-":
                _, length, distance = idx.args
                if length.kind == "length":
                    key = self.predicate_key(host, state)
                    if key is not None and key == self.predicate_key(length.args[0], state):
                        if distance.kind == "number" and type(distance.args[0]) is int:
                            domain = Domain.point(distance.args[0])
                        elif distance.kind == "name":
                            symbol = self.place(distance)
                            domain = self.storage_value(symbol, state).domain if (
                                symbol is not None and plain_integer_parameter(symbol.ty)) else None
                        else:
                            domain = None
                        proven = bool(domain and domain.within(Domain.of((1, span.length.domain.lo))))
            if not proven and integer:
                value, proven = self.ssa_loop_index(idx, value, span.length)
            if capture_index is not None:
                capture_index.append(value)
            valid = value.domain
            origin = self.span_origins(span)
            if proven:
                self.trace.append(f"  verify index {origin} at {idx.token.line}:{idx.token.col}: "
                                  f"{value.domain} subset {bound if value.domain.within(bound) else '[0, len)'} => proven")
            else:
                if not integer or not self.debug_bounds:
                    detail = self.index_diagnostic(idx, value.domain if integer else None, span.positions())
                    idx.token.error(f"cannot prove index {value.domain} is within {bound} for length {span.length.domain}{detail}")
                valid &= Domain.of((0, span.length.domain.hi - 1))
                if not valid:
                    detail = self.index_diagnostic(idx, value.domain, span.positions())
                    idx.token.error(f"ordinary index cannot be in bounds for any possible length{detail}")
                self.trace.append(f"  debug check index {origin} at {idx.token.line}:{idx.token.col}: {value.domain}")
            if span.host is None or self.debug_bounds or self.has_effects(host) or self.has_effects(idx):
                if self.debug_bounds:
                    check = self.helper("debug_index", None)
                else:
                    assume = self.helper("contract_assume", None)
                    check = self.helper("assume_index", None, assume)
                assume = self.helper("contract_assume", None)
                helper = self.helper("slice_index", span.element.c, (check, assume))
                code = f"{helper}({span.c_value()}, {value.code})"
                span = replace(span, host=None, offset=0, code=code, borrowed_from=span.owners, origins=span.positions())
                return span, "0", valid
            index = (f"(size_t)({value.code})" if self.literal_endpoint(idx) else
                     self.assumed_index(value.code, span.length.code))
            return span, index, valid
        arr = self.array(host)
        if arr.link_access is not None:
            propagate = propagate_exit and self.has_propagation(host)
            arr = self.comparison_place(arr, state, allow_effectful_links=propagate,
                                        propagate_exit=propagate)
            if arr.projection is not None and arr.projection.always_returns:
                return Value(ZERO, arr.c, effectful=True, always_returns=True)
        self.require_live(arr, state, expr.token)
        checked = self.fixed_array_index(idx, state, arr, capture_index, propagate_exit=propagate_exit)
        if isinstance(checked, Value):
            return replace(checked, code=f"((void)({arr.c}), ({checked.code}))")
        index, domain = checked
        return arr, index, domain

    def index_diagnostic(self, idx, domain, origins):
        detail = f"; index expression: {idx.source_text}" if idx.source_text is not None else ""
        if not domain:
            return detail
        offsets = []
        for owner, start in origins:
            if not start or (owner.uid in self.param_places and isinstance(owner.ty, Slice)):
                offsets.append(f"{owner.name}: unknown")
                continue
            bounds = Domain.of((start.lo + domain.lo, start.hi + domain.hi))
            offsets.append(f"{owner.name}: {bounds}")
        backing = " | ".join(sorted(set(offsets))) or "unknown"
        return f"{detail}; backing element offset: {backing}"

    def fixed_array_index(self, idx, state, arr, capture_index=None, *, propagate_exit=False):
        value = self.expr(idx, state)
        if propagate_exit and value.always_returns:
            return value
        bound = Domain.of((0, arr.ty.length - 1))
        value = self.ssa_narrow(idx, value, bound, "operands", "index")
        if capture_index is not None:
            capture_index.append(value)
        integer = not (value.floating or value.boolean or value.enum or value.character)
        proven = integer and value.domain.within(bound)
        if proven:
            self.trace.append(f"  verify index id#{arr.uid} at {idx.token.line}:{idx.token.col}: "
                              f"{value.domain} subset {bound} => proven")
        elif not integer or not self.debug_bounds:
            detail = self.index_diagnostic(idx, value.domain if integer else None, ((arr, ZERO),))
            idx.token.error(f"cannot prove index {value.domain} is within {bound} for {arr.name!r}{detail}")
        else:
            self.trace.append(f"  debug check index id#{arr.uid} at {idx.token.line}:{idx.token.col}: {value.domain}")
        if self.debug_bounds:
            valid = value.domain & bound
            if not valid:
                detail = self.index_diagnostic(idx, value.domain, ((arr, ZERO),))
                idx.token.error(f"index cannot be in bounds for this array{detail}")
            check = self.helper("debug_index", None)
            return f"{check}({value.code}, UINT64_C({arr.ty.length}))", valid
        if not self.literal_endpoint(idx):
            return self.assumed_index(value.code, f"UINT64_C({arr.ty.length})"), value.domain
        index_type = "uint64_t" if arr.ty.packed_bits else "size_t"
        return f"({index_type})({value.code})", value.domain

    def composite_call_projection(self, node, state, host, projections):
        early = []
        result, call = self.call(host, state, capture_early_return=early)
        if early:
            return early[0]
        array_result = isinstance(result, CopyArray)
        ty = result.array if array_result else result
        temp = f"lan_composite_read_{node.token.line}_{node.token.col}"
        sym = Symbol("composite result", len(self.symbols) + 1, ty, len(self.scope), readonly=True)
        self.symbols.append(sym)
        contents = self.array_call_contents(host) if array_result else None
        record_fields = None if array_result else self.record_call_fields(host)
        field_path = ()
        summary = contents
        self.helper("global_temp", None, (result.c, temp))
        setup = [f"{temp} = {call}"]
        access = f"{temp}.data" if array_result else temp
        def index_exit(value):
            return replace(value, code=f"({', '.join(setup)}, {value.code})", effectful=True)

        for depth, projection in enumerate(projections):
            if projection.kind == "length":
                if isinstance(ty, Array):
                    domain = Domain.point(ty.length)
                    setup.append(f"(void)({access})")
                    length = c_int(ty.length)
                elif isinstance(ty, Slice):
                    from composite_array import text_length

                    domain = ty.domain if contents is None else text_length(contents)
                    length = f"(lan_int)({access}).len"
                    if not self.debug_bounds:
                        guaranteed = ty.domain if summary is None else text_length(summary)
                        size = f"{temp}_length"
                        predicates = self.integer_range_predicates(
                            projection.token, size, guaranteed, container(64, False))
                        length = self.assumed_result("lan_int", size, length, predicates)
                else:
                    projection.token.error("length requires an array or text field in the returned value")
                return Value(domain, f"({', '.join(setup)}, {length})", effectful=True)
            if projection.kind == "member":
                name = projection.args[1]
                members = dict(ty.fields) if isinstance(ty, Record) else {}
                if name.text not in members:
                    name.error(f"unknown record field {name.text!r} in returned value")
                ty = unqualified(members[name.text])
                if record_fields is not None:
                    field_path += (name.text,)
                    if not isinstance(ty, Record):
                        contents = summary = record_fields.get(field_path)
                        record_fields = None
                else:
                    if contents is not None:
                        contents = dict(contents.fields)[name.text]
                    if summary is not None:
                        summary = dict(summary.fields)[name.text]
                access += f".{c_field(name.text)}"
                continue
            subscript = projection.args[1]
            literal_index = self.literal_endpoint(subscript)
            position = f"{temp}_index_{depth}"
            if isinstance(ty, Slice):
                from composite_array import TextContent, text_length

                length = ty.domain if contents is None else text_length(contents)
                value = self.expr(subscript, state)
                if value.always_returns:
                    return index_exit(value)
                bound = Domain.of((0, length.lo - 1))
                integer = not (value.floating or value.boolean or value.enum or value.character)
                proven = integer and value.domain.within(bound)
                if not proven and (not integer or not self.debug_bounds):
                    detail = self.index_diagnostic(subscript, value.domain if integer else None, ())
                    subscript.token.error(f"cannot prove index {value.domain} is within {bound} for returned text{detail}")
                indices = value.domain & Domain.of((0, length.hi - 1))
                if not indices:
                    detail = self.index_diagnostic(subscript, value.domain, ())
                    subscript.token.error(f"index cannot be in bounds for returned text{detail}")
                site = f"{subscript.token.line}:{subscript.token.col}"
                if proven:
                    self.trace.append(f"  verify index returned text at {site}: {indices} subset {bound} => proven")
                else:
                    self.trace.append(f"  debug check index returned text at {site}: {value.domain}")
                text = f"{temp}_text"
                self.helper("global_temp", None, ("lan_slice", text))
                setup.append(f"{text} = {access}")
                if self.debug_bounds:
                    check = self.helper("debug_index", None)
                    index = f"{check}({value.code}, {text}.len)"
                else:
                    index = self.assumed_index(value.code, f"{text}.len")
                self.helper("global_temp", None, ("size_t", position))
                setup.append(f"{position} = {index}")
                assume = self.helper("contract_assume", None)
                setup.append(f"{assume}({text}.data != NULL)")
                ty = ty.element
                access = f"((const {ty.c} *){text}.data)[{position}]"

                def byte_domain(text, selection):
                    if not isinstance(text, TextContent):
                        return None
                    return Domain.union(Domain.point(byte) for offset, byte in enumerate(text.data)
                                        if selection is None or selection & Domain.point(offset))

                contents = byte_domain(contents, indices)
                summary = byte_domain(summary, indices if literal_index else None)
                continue
            if not isinstance(ty, Array):
                subscript.token.error("index requires an array or text field in the returned value")
            checked = self.fixed_array_index(subscript, state, replace(sym, ty=ty), propagate_exit=True)
            if isinstance(checked, Value):
                return index_exit(checked)
            index, indices = checked
            self.helper("global_temp", None, ("size_t", position))
            setup.append(f"{position} = {index}")
            if contents is not None:
                contents = contents.read(indices)
            if summary is not None:
                possible = indices if literal_index else Domain.of((0, ty.length - 1))
                summary = summary.read(possible)
            if ty.packed_bits:
                helper = self.helper("bit_read", None)
                access = f"{helper}({access}, {position})"
            else:
                access = f"{access}[{position}]"
            ty = ty.element
        element = ty
        if not isinstance(element, Scalar):
            node.token.error("bind the result before accessing a non-scalar value")
        domain = element.domain if contents is None else contents
        code = f"({', '.join(setup)}, {access})"
        if not self.debug_bounds:
            # The same emitted expression can serve several incoming paths.
            # Join all positions for a dynamic index before emitting an assumption.
            contract = element if summary is None else replace(element, domain=summary)
            code = self.assume_scalar_result(node.token, contract, code)
        return Value(domain, code, element.bits, element.boolean, enum=element.enum,
                     character=element.character, signed=element.signed, effectful=True,
                     result_arms=element.result_arms, result_error_enums=element.result_error_enums)

    def require_bit(self, token, value, node=None):
        if node is not None and (width := bit_literal_width(node)) is not None:
            require_bit_literal_width(token, 1, width)
        if value.boolean or value.floating or value.enum or value.character:
            token.error("bit value must be the integer 0 or 1")
        if self.proven_presence(value.domain - BOOL, token):
            token.error(f"bit value must be 0 or 1; got {value.domain}")

    def float_coerce(self, token, value, bits, *, rounding=False, explicit=False):
        if value.boolean or value.enum or value.character:
            token.error("floating conversion requires a numeric operand")
        try:
            if value.literal is not None:
                domain = FloatDomain.point(value.literal, bits)
                return Value(domain, c_literal(value.literal, bits), bits, literal=value.literal)
            if value.width > bits and not rounding and not explicit:
                token.error(f"implicit narrowing from {value.width} to {bits} bits; use an explicit round conversion")
            if value.floating:
                domain = value.domain.convert(bits, "round" if rounding else "exact")
            else:
                domain = FloatDomain.from_integers(value.domain, bits, exact=not rounding)
        except FloatDomainError as exc:
            token.error(str(exc))
        target = "float" if bits == 32 else "double"
        helper = self.helper("float_convert", None, (value.intermediate_c, target))
        return Value(domain, f"{helper}({value.code})", bits, effectful=value.effectful)

    def coerce_storage(self, token, value, ty):
        if ty.floating:
            return self.float_coerce(token, value, ty.bits)
        if value.floating:
            token.error("float-to-integer conversion requires an explicit trunc, floor, ceil or round strategy")
        return self.coerce_result_enum(value, ty)

    def coerce_result_enum(self, value, ty):
        if is_callback_enum(ty.enum):
            return value
        if ty.enum is not None and value.enum is None and value.domain.within(ty.domain):
            identity = ty.enum.identity
            narrowed_result = (value.result_arms is not None and identity in value.result_error_enums
                               and value.domain.within(value.result_arms[1]))
            others = set(value.result_error_enums) - {identity}
            if narrowed_result and others:
                known = {item.enum.identity: item.enum for item in self.types.values()
                         if isinstance(item, Scalar) and item.enum is not None}
                narrowed_result = all(other in known and not any(
                    value.domain & Domain.point(code) for _, code in known[other].members)
                    for other in others)
            if narrowed_result or identity in value.guarded_enums:
                return replace(value, enum=ty.enum, width=ty.bits, signed=ty.signed,
                               code=f"(({ty.c})({value.code}))", result_arms=None, result_error_enums=())
        return value

    def float_binary(self, token, op, left, right):
        if is_callback_enum(left.enum) or is_callback_enum(right.enum):
            token.error("function values only support equality comparison with other function values")
        if op in ("^^", "&&", "||"):
            a = left.domain.truth() if left.floating else BOOL
            b = right.domain.truth() if right.floating else BOOL
            if op == "^^":
                result = Domain.of(*((int(x != y), int(x != y)) for x in range(a.lo,a.hi+1) for y in range(b.lo,b.hi+1)))
                helper = self.helper("binary_eval", None, ("uint8_t", left.intermediate_c, right.intermediate_c, "((a != 0) != (b != 0))"))
                code = f"{helper}({left.code}, {right.code})"
                return Value(result, code, 8, True, effectful=left.effectful or right.effectful)
        widths = [value.domain.bits for value in (left, right) if value.floating and value.width]
        bits = max(widths) if widths else 64
        x, y = self.float_coerce(token, left, bits), self.float_coerce(token, right, bits)
        kind = "float" if bits == 32 else "double"
        comparisons = {"==", "!=", "===", "!==", "<", "<=", ">", ">="}
        op = {"===": "==", "!==": "!="}.get(op, op)
        if op in comparisons:
            domain = x.domain.compare(op, y.domain)
            helper = self.helper("binary_eval", None, ("uint8_t", kind, kind, f"(a {op} b)"))
            code = f"{helper}({x.code}, {y.code})"
        elif op in ("+", "-", "*", "/"):
            domain = x.domain.binary(op, y.domain)
            helper = self.helper("float_binary", None, (op, bits))
            code = f"{helper}({x.code}, {y.code})"
        else:
            token.error(f"operator {op!r} requires integer operands")
        return Value(domain, code, 8 if op in comparisons else bits, op in comparisons,
                     effectful=left.effectful or right.effectful)

    def float_cast(self, token, value, strategy, ty):
        if is_callback_enum(value.enum) or is_callback_enum(ty.enum):
            token.error("function values cannot be converted or reinterpreted")
        if ty.boolean:
            if strategy != "safe":
                token.error("boolean conversion uses plain 'as boolean'")
            return Value(value.domain.truth(), f"(({value.code}) != 0)", 8, True)
        if value.boolean or value.enum or value.character or ty.enum or ty.character:
            token.error("floating conversions require numeric source and target types")
        if strategy == "view":
            if value.width != ty.bits or not value.width:
                token.error("floating bit view requires equal physical widths")
            if not value.floating:
                self.check_integer_view(token, value, ty.bits)
            source_c = value.intermediate_c if value.floating else ("int" if value.signed else "uint") + str(value.width) + "_t"
            helper = self.helper("float_view", None, (source_c, ty.c))
            if value.floating:
                domain = value.domain if ty.floating else value.domain.raw_bits()
                if ty.signed:
                    sign = 1 << (ty.bits - 1)
                    domain = (domain & Domain.of((0, sign - 1))) | (
                        domain & Domain.of((sign, sign * 2 - 1))).affine(1, -sign * 2)
            else:
                domain = FloatDomain.from_raw_bits(self.bit_raw_domain(value.domain, ty.bits), ty.bits)
            result = Value(domain, f"{helper}({value.code})", ty.bits, signed=ty.signed)
            if self.proven_presence(domain - ty.domain, token):
                token.error(f"cannot prove bit-view result {domain} fits {ty.domain}")
            return result
        if ty.floating:
            rounding = strategy in ("round", "sat", "round_sat", "round_trap")
            if strategy not in ("safe", "round", "sat", "trap", "round_sat", "round_trap"):
                token.error("numeric-to-float conversion uses round, sat or trap")
            if value.floating and value.width > ty.bits and not rounding:
                token.error("floating narrowing requires an explicit round or sat strategy")
            converted = replace(self.float_coerce(token, value, ty.bits, rounding=rounding, explicit=True), literal=None)
            if strategy in ("sat", "round_sat"):
                if (not ty.domain.ranges or ty.domain != FloatDomain.interval(ty.domain.lo, ty.domain.hi, ty.bits)):
                    # A full float type is a natural finite saturation container.
                    full = ty.domain == FloatDomain.top(ty.bits)
                    if not full:
                        token.error("floating saturation requires one numeric interval or a full floating type")
                    lo, hi = -binary_format(ty.bits).max_finite, binary_format(ty.bits).max_finite
                else:
                    lo, hi = ty.domain.lo, ty.domain.hi
                    if not math.isfinite(lo) or not math.isfinite(hi):
                        token.error("floating saturation needs finite bounds")
                domain = FloatDomain(ty.bits, nan=converted.domain.nan and ty.domain.nan)
                for a,b in converted.domain.ranges:
                    domain |= FloatDomain.interval(min(max(a,lo),hi), min(max(b,lo),hi), ty.bits)
                helper = self.helper("float_sat", None, (ty.bits, lo, hi, ty.domain.nan))
                return Value(domain, f"{helper}({converted.code})", ty.bits)
            if strategy in ("trap", "round_trap"):
                domain = converted.domain & ty.domain
                if not domain:
                    token.error("trap conversion cannot produce a value in the floating target")
                helper = self.helper("float_trap", ty.domain, self.membership(token, ty.domain))
                return Value(domain, f"{helper}({converted.code})", ty.bits)
            self.range_check(token, converted, ty, explicit=True)
            return converted
        if not value.floating:
            token.error("internal floating conversion expected a floating operand")
        pieces = strategy.split("_")
        rounding = pieces[0]
        overflow = pieces[1] if len(pieces) == 2 else "reject"
        try:
            conversion = value.domain.to_integers(ty.domain, rounding=rounding, overflow=overflow)
        except FloatDomainError as exc:
            token.error(str(exc))
        if conversion.definitely_traps:
            token.error("trap conversion cannot produce a value in the integer target")
        physical_target = container(ty.bits, ty.signed) if ty.domain.predicates else ty.domain
        allowed = conversion_input_domain(physical_target, value.domain.bits, rounding)
        guard = self.membership(token, allowed)
        predicate = self.membership(token, ty.domain) if ty.domain.predicates else None
        helper = self.helper("float_integer", ty.domain, (value.domain.bits, ty.bits, ty.signed, rounding, overflow, guard, predicate))
        return Value(conversion.domain, f"{helper}({value.code})", ty.bits, signed=ty.signed)

    def cast(self, t, val, strategy, ty):
        if is_callback_enum(val.enum) or is_callback_enum(ty.enum):
            if strategy == "safe" and val.enum == ty.enum:
                self.range_check(t, val, ty)
                return val
            t.error("function values cannot be converted or reinterpreted; use callback(function)")
        if val.floating or ty.floating:
            return self.float_cast(t, val, strategy, ty)
        if strategy == "trunc":
            strategy = "safe"
        if strategy not in ("safe", "sat", "wrap", "view", "trap"):
            t.error("integer conversions support safe, trunc, sat, wrap, view or trap")
        if ty.boolean:
            if strategy != "safe":
                t.error("boolean conversion uses plain 'as boolean'")
            return Value(Domain.point(0) if val.domain == ZERO else (BOOL if val.domain & ZERO else Domain.point(1)),
                         f"(({val.code}) != 0)", 8, True)
        if val.boolean:
            # A boolean already holds exactly {0, 1}, which fits every integer
            # type, so the conversion is a plain cast of the stored value.
            val = Value(Domain.of((0, 1)), val.code, 8)
        linear = self.linear_value(val)
        if linear is not None and strategy != "safe":
            atom = ("integer-cast", strategy, ty, linear)
            linear = frozenset(((frozenset(((atom, 1),)), 1),)), 0
        if strategy == "safe":
            self.range_check(t, val, ty, explicit=True)
            return Value(val.domain, f"(({ty.c})({val.code}))", ty.bits,
                         affine=val.affine, difference=val.difference, enum=ty.enum, character=ty.character,
                         signed=ty.signed, linear=linear)
        if strategy == "trap":
            domain = val.domain & ty.domain
            if not self.proven_presence(domain, t):
                t.error("trap conversion cannot produce a value in the target type")
            guard = self.membership(t, ty.domain)
            unsigned = bool(ty.domain) and ty.domain.lo >= 0 and ty.domain.hi >= (1 << 63)
            needs_nonneg = bool(val.domain) and val.domain.lo < 0 and unsigned
            needs_upper = bool(val.domain) and val.domain.lo >= 0 and val.domain.hi >= (1 << 63) and not unsigned
            name = self.helper("trap", ty.domain, (guard, needs_nonneg, needs_upper))
            return Value(domain, f"{name}({val.code})", ty.bits,
                         affine=val.affine, difference=val.difference, enum=ty.enum, character=ty.character,
                         signed=ty.signed, linear=linear)
        if strategy == "view":
            self.check_integer_view(t, val, ty.bits)
        if strategy == "sat":
            if ty.domain.predicates or len(ty.domain.ranges) != 1:
                t.error("saturation requires a nonempty contiguous target range")
            lo, hi = ty.domain.lo, ty.domain.hi
            domain = Domain.of(*((min(max(a, lo), hi), min(max(b, lo), hi)) for a, b in val.domain.ranges))
            unsigned = ty.domain.lo >= 0 and ty.domain.hi >= (1 << 63)
            needs_nonneg = val.domain.lo < 0 and unsigned
            needs_upper = val.domain.lo >= 0 and val.domain.hi >= (1 << 63) and not unsigned
            name = self.helper("sat", ty.domain, (needs_nonneg, needs_upper))
        else:
            full = container(ty.bits, ty.signed)
            lo, modulus = full.lo, 1 << ty.bits
            spans = []
            for a, b in val.domain.ranges:
                q0, r0 = divmod(a - lo, modulus)
                q1, r1 = divmod(b - lo, modulus)
                if q0 == q1:
                    spans.append((lo + r0, lo + r1))
                elif q1 == q0 + 1:
                    spans.extend(((lo + r0, full.hi), (lo, lo + r1)))
                else:
                    spans.extend(full.ranges)
            domain = Domain.of(*spans)
            if val.domain.within(full):
                domain = val.domain
            if self.proven_presence(domain - ty.domain, t):
                t.error(f"{strategy} can produce {domain}, outside {ty.domain}")
            name = self.helper("wrap", full, ty.bits)
        unchanged = val.domain.within(ty.domain)
        return Value(domain, f"{name}({val.code})", ty.bits, affine=val.affine if unchanged else None,
                     difference=val.difference if unchanged else None, enum=ty.enum, character=ty.character,
                     signed=ty.signed, linear=linear)

    @staticmethod
    def linear_difference(op, x, y):
        if x.floating or y.floating:
            return None
        left = Difference(offset=x.domain.lo) if x.domain.lo == x.domain.hi else x.difference
        right = Difference(offset=y.domain.lo) if y.domain.lo == y.domain.hi else y.difference
        if left is not None and right is not None:
            return left.combine(right, 1 if op == "+" else -1)
        return None

    @staticmethod
    def related(x, y):
        return x.affine is not None and y.affine is not None and x.affine.origin is y.affine.origin

    def combine_forms(self, op, x, y):
        left, right = x.affine, y.affine
        if op in ("+", "-") and self.related(x, y):
            sign = 1 if op == "+" else -1
            form = AffineForm(left.origin, left.scale + sign * right.scale, left.offset + sign * right.offset)
        elif left is not None and y.domain.lo == y.domain.hi:
            n = y.domain.lo
            if op in ("+", "-"):
                form = left.transform(1, n if op == "+" else -n)
            elif op == "*":
                form = left.transform(n, 0)
            elif op == "<<":
                form = left.transform(1 << n, 0)
            else:
                return None
        elif right is not None and x.domain.lo == x.domain.hi:
            n = x.domain.lo
            if op in ("+", "-"):
                form = right.transform(1 if op == "+" else -1, n)
            elif op == "*":
                form = right.transform(n, 0)
            else:
                return None
        else:
            return None
        return form if max(form.scale.bit_length(), form.offset.bit_length()) <= 256 else None

    @staticmethod
    def correlated_domain(form, x, y):
        if not form.scale:
            return Domain.point(form.offset)
        original = form.origin.domain & x.affine.preimage(x.domain) & y.affine.preimage(y.domain)
        return original.affine(form.scale, form.offset) if original else None

    @staticmethod
    def capture_value(value, state):
        if value.floating:
            return replace(value, affine=None, difference=None, versions=())
        diff = value.difference
        if diff is None:
            return value
        ids = {diff.positive, diff.negative} - {0}
        return replace(value, versions=tuple((uid, state.forms.get(uid)) for uid in sorted(ids)))

    @staticmethod
    def fresh_value(value, state):
        if state is not None and any(state.forms.get(uid) != form for uid, form in value.versions):
            return replace(value, difference=None, versions=())
        return value

    def binary(self, t, op, x, y, state=None, *, wrap32=False):
        if op == "·":
            t.error("'·' is only supported in compile-time expressions and type objects")
        x, y = self.fresh_value(x, state), self.fresh_value(y, state)
        if x.floating or y.floating:
            return self.float_binary(t, op, x, y)
        linear = self.combine_linear(op, x, y)
        if not x.effectful and not y.effectful:
            return replace(self.binary_value(t, op, x, y, state, wrap32=wrap32), linear=linear)
        comparisons = {"==", "!=", "===", "!==", "<", "<=", ">", ">="}
        cross_sign_cmp = (op in comparisons and ((x.domain.lo < 0 and y.domain.lo >= 0 and y.domain.hi >= (1 << 63))
                                                 or (y.domain.lo < 0 and x.domain.lo >= 0 and x.domain.hi >= (1 << 63))))
        if not x.effectful and not cross_sign_cmp:
            return replace(self.binary_value(t, op, x, y, state, wrap32=wrap32), effectful=True, linear=linear)
        if not y.effectful and op not in ("<<", ">>") and not cross_sign_cmp:
            return replace(self.binary_value(t, op, x, y, state, wrap32=wrap32), effectful=True, linear=linear)
        left_c = x.intermediate_c
        right_c = y.intermediate_c
        result = self.binary_value(t, op, replace(x, code="a", effectful=False),
                                   replace(y, code="b", effectful=False), state, wrap32=wrap32)
        ret_c = "uint8_t" if result.boolean else ("uint32_t" if wrap32 else result.intermediate_c)
        helper = self.helper("binary_eval", None, (ret_c, left_c, right_c, result.code))
        return replace(result, code=f"{helper}({x.code}, {y.code})", effectful=True, linear=linear)

    def assumed_divisor(self, token, code, *, c_type="lan_int"):
        divisor = f"lan_divisor_{token.line}_{token.col}"
        return self.assumed_result(c_type, divisor, code, [f"{divisor} != 0"])

    def assumed_shift_count(self, token, code):
        count = f"lan_shift_count_{token.line}_{token.col}"
        return self.assumed_result("int32_t", count, code, [f"{count} >= 0", f"{count} < 128"])

    @staticmethod
    def _integer_compare_code(c_op, x, y):
        a, b = x.domain, y.domain
        u32_max = (1 << 32) - 1
        i32_min, i32_max = -(1 << 31), (1 << 31) - 1
        if a.lo >= 0 and b.lo >= 0 and a.hi <= u32_max and b.hi <= u32_max:
            return f"(((uint32_t)({x.code})) {c_op} ((uint32_t)({y.code})))"
        if i32_min <= a.lo and a.hi <= i32_max and i32_min <= b.lo and b.hi <= i32_max:
            return f"(((int32_t)({x.code})) {c_op} ((int32_t)({y.code})))"
        if a.lo >= 0 and b.lo >= 0 and (a.hi >= (1 << 63) or b.hi >= (1 << 63)):
            return f"(((lan_uint)({x.code})) {c_op} ((lan_uint)({y.code})))"
        if a.lo < 0 and b.lo >= 0 and b.hi >= (1 << 63):
            neg = {"==": "0", "!=": "1", "<": "1", "<=": "1", ">": "0", ">=": "0"}[c_op]
            return f"(((lan_int)({x.code})) < 0 ? {neg} : (((lan_uint)({x.code})) {c_op} ((lan_uint)({y.code}))))"
        if b.lo < 0 and a.lo >= 0 and a.hi >= (1 << 63):
            neg = {"==": "0", "!=": "1", "<": "0", "<=": "0", ">": "1", ">=": "1"}[c_op]
            return f"(((lan_int)({y.code})) < 0 ? {neg} : (((lan_uint)({x.code})) {c_op} ((lan_uint)({y.code}))))"
        return f"(((lan_int)({x.code})) {c_op} ((lan_int)({y.code})))"

    @staticmethod
    def _narrow_arith_c(op, a, b, *, wrap32=False):
        if wrap32 and op in ("+", "-", "*", "&", "|", "^"):
            return "uint32_t"
        if wrap32 and op == "<<" and b.lo >= 0 and b.hi < 32:
            return "uint32_t"
        u32_max = (1 << 32) - 1
        i32_min, i32_max = -(1 << 31), (1 << 31) - 1
        a_u32 = a.lo >= 0 and a.hi <= u32_max
        b_u32 = b.lo >= 0 and b.hi <= u32_max
        a_i32 = i32_min <= a.lo and a.hi <= i32_max
        b_i32 = i32_min <= b.lo and b.hi <= i32_max
        if op == "+":
            lo, hi = a.lo + b.lo, a.hi + b.hi
            if a_u32 and b_u32 and hi <= u32_max:
                return "uint32_t"
            if a_i32 and b_i32 and i32_min <= lo and hi <= i32_max:
                return "int32_t"
        elif op == "-":
            lo, hi = a.lo - b.hi, a.hi - b.lo
            if a_u32 and b_u32 and lo >= 0:
                return "uint32_t"
            if a_i32 and b_i32 and i32_min <= lo and hi <= i32_max:
                return "int32_t"
        elif op == "*":
            products = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
            lo, hi = min(products), max(products)
            if a_u32 and b_u32 and hi <= u32_max:
                return "uint32_t"
            if a_i32 and b_i32 and i32_min <= lo and hi <= i32_max:
                return "int32_t"
        elif op in ("/", "%"):
            if a_u32 and b_u32:
                return "uint32_t"
            if a_i32 and b_i32 and not (a.lo == i32_min and b.lo <= -1 <= b.hi):
                return "int32_t"
        elif op in ("&", "|", "^"):
            if a_u32 and b_u32:
                return "uint32_t"
            if a_i32 and b_i32:
                return "int32_t"
        elif op in ("<<", ">>"):
            if a_u32 and b.lo >= 0 and b.hi < 32:
                if op == ">>" or (a.hi << b.hi) <= u32_max:
                    return "uint32_t"
        unsigned_op = (a.lo >= 0 and b.lo >= 0 and (a.hi >= (1 << 63) or b.hi >= (1 << 63))
                       and not (op == "-" and a.lo < b.hi))
        if not unsigned_op and op in ("+", "*", "<<") and a.lo >= 0 and b.lo >= 0:
            max_result = (a.hi + b.hi if op == "+"
                          else a.hi * b.hi if op == "*"
                          else (a.hi << min(b.hi, 64)))
            if max_result >= (1 << 63):
                unsigned_op = True
        return "lan_uint" if unsigned_op else "lan_int"

    def binary_value(self, t, op, x, y, state=None, *, wrap32=False):
        comparisons = {"==", "!=", "===", "!==", "<", "<=", ">", ">="}
        a, b = x.domain, y.domain
        width = max(x.width, y.width)
        c_op = {"===": "==", "!==": "!=", "^^": "!="}.get(op, op)
        arith_c = self._narrow_arith_c(op, a, b, wrap32=wrap32)
        if op in comparisons:
            code = self._integer_compare_code(c_op, x, y)
        else:
            code = f"((({arith_c})({x.code})) {c_op} (({arith_c})({y.code})))"
        if (is_callback_enum(x.enum) or is_callback_enum(y.enum)) and op not in ("==", "!=", "===", "!=="):
            t.error("function values only support equality comparison")
        if op == "^^":
            return Value(BOOL, f"((({x.code}) != 0) != (({y.code}) != 0))", 8, True)
        polynomial = self.linear_value(x) if op in comparisons or op in ("-", "*", "&", "|", "^", "/", "%") else None
        same = polynomial is not None and polynomial == self.linear_value(y)
        contraction_difference = self.contraction_difference_domain(x, y) if op in comparisons or op == "-" else None
        if op in comparisons:
            if x.boolean != y.boolean:
                t.error("comparison mixes boolean and integer")
            if x.enum != y.enum:
                t.error("comparison requires the same enum identity; use an explicit conversion")
            if x.character != y.character and not (x.character and y.character
                                                     and "ascii" in (x.character, y.character)):
                t.error("comparison requires compatible character encodings")
            if same:
                return Value(Domain.point(c_op in ("==", "<=", ">=")), code, 8, True)
            known = self.dependent_difference_domain(x, y, state)
            if known is not None:
                yes, no = self.compare_domains(c_op, known, ZERO, t)
                if not (yes and no):
                    return Value(Domain.point(yes), code, 8, True)
            if contraction_difference is not None:
                yes, no = self.compare_domains(c_op, contraction_difference, ZERO, t)
                if not (yes and no):
                    return Value(Domain.point(yes), code, 8, True)
            difference = self.linear_difference("-", x, y)
            bounds = state.zone.range(difference) if state is not None and difference is not None else None
            if bounds is not None:
                yes, no = self.compare_domains(c_op, Domain.of(bounds), ZERO, t)
                if not (yes and no):
                    return Value(Domain.point(yes), code, 8, True)
            delta = self.combine_forms("-", x, y) if self.related(x, y) else None
            difference = self.correlated_domain(delta, x, y) if delta else None
            if difference is None:
                yes, no = self.compare_domains(c_op, a, b, t)
            else:
                yes, no = self.compare_domains(c_op, difference, ZERO, t)
            return Value(BOOL if yes and no else Domain.point(yes), code, 8, True)
        if x.boolean or y.boolean or x.enum or y.enum or x.character or y.character:
            t.error("arithmetic requires integers; convert enums and characters explicitly")
        if op in ("-", "^") and same:
            return self.emit_value(t, ZERO, code, width, signed=x.signed or y.signed)
        if op in ("/", "%"):
            if b & ZERO:
                t.error(f"cannot prove divisor {b} is nonzero")
            divisor = self.assumed_divisor(t, y.code, c_type=arith_c)
            code = f"((({arith_c})({x.code})) {op} ({divisor}))"
        if op in ("/", "%") and same:
            return self.emit_value(t, Domain.point(int(op == "/")), code, width, signed=x.signed or y.signed)
        if op == "%" and a & Domain.point(WIDE.lo) and b & Domain.point(-1):
            t.error("remainder would overflow the C intermediate division")
        if op in ("<<", ">>"):
            if b.lo < 0 or b.hi >= 128:
                t.error("cannot prove shift count is in [0, 128)")
            count = self.assumed_shift_count(t, y.code)
            if arith_c == "uint32_t":
                code = f"(((uint32_t)({x.code})) {op} (unsigned)({count}))"
            elif op == "<<" or a.lo < 0 or b.hi >= 64:
                helper = self.helper("shift_left" if op == "<<" else "shift_right", None)
                code = f"{helper}({x.code}, (unsigned)({count}))"
            else:
                code = f"((({arith_c})({x.code})) >> (unsigned)({count}))"
        form = self.combine_forms(op, x, y)
        difference = self.linear_difference(op, x, y) if op in ("+", "-") else None
        bounds = state.zone.range(difference) if state is not None and difference is not None else None

        contract_difference = self.dependent_difference_domain(x, y, state) if op == "-" else None

        def finish(domain):
            if contract_difference is not None:
                constrained = domain & contract_difference
                if constrained:
                    domain = constrained
            if bounds is not None:
                constrained = domain & Domain.of(bounds)
                if constrained:
                    if constrained != domain:
                        self.trace.append(f"  difference bound at {t.line}:{t.col}: {constrained}")
                    domain = constrained
            if op in ("+", "-") and not domain.predicates and (a.predicates or b.predicates):
                r1, m1 = a.congruence()
                r2, m2 = b.congruence()
                refined = domain.with_congruence(r1 + r2 if op == "+" else r1 - r2, gcd(m1, m2))
                if refined:
                    domain = refined
            if op == '%' and b.lo == b.hi:
                residue, modulus = a.congruence()
                domain = domain.with_congruence(residue, gcd(modulus, abs(b.lo)))
            if op == "<<":
                scale = 1 << b.lo
                if b.lo == b.hi:
                    residue, modulus = a.congruence()
                    domain = domain.with_congruence(residue * scale, modulus * scale)
                else:
                    domain = domain.with_congruence(0, scale)
            if op == ">>" and b.lo == b.hi:
                scale = 1 << b.lo
                residue, modulus = a.congruence()
                if modulus % scale == 0:
                    domain = domain.with_congruence(residue // scale, modulus // scale)
            return self.emit_value(t, domain, code, width, affine=form, difference=difference,
                                   signed=x.signed or y.signed)

        if op in ("*", "&", "|") and same:
            common = a & b
            if common:
                if op != "*":
                    return finish(common)
                domain = square_domain(common)
                shared = product_congruence(common, common)
                if shared is not None:
                    domain = domain.with_congruence(*shared)
                return finish(domain)
        if op == "-" and contraction_difference is not None:
            return finish(contraction_difference)
        if op == "<<" and a.lo >= 0 and a.hi << b.hi > WIDE.hi:
            linear = self.combine_linear(op, x, y)
            if linear is not None and linear[0]:
                upper = self.dependent_upper_bound(linear[0], state if state is not None else PathState({}))
                if 0 <= upper + linear[1] <= WIDE.hi:
                    return finish(Domain.of((0, upper + linear[1])))
        if op in ("&", "|", "^"):
            domain, bits = bitwise_domain(op, a, b)
            self.spend(t, len(a.ranges) * len(b.ranges) * bits)
            return finish(domain)

        if form and op in ("+", "-") and self.related(x, y):
            correlated = self.correlated_domain(form, x, y)
            if correlated is not None:
                if not form.scale:
                    self.trace.append(f"  affine cancellation at {t.line}:{t.col}: {correlated}")
                return finish(correlated)
        if bounds is not None and bounds[0] == bounds[1]:
            return finish(Domain.point(bounds[0]))
        affine = None
        if b.lo == b.hi:
            n = b.lo
            if op in ("+", "-") and a.predicates:
                affine = a.affine(1, n if op == "+" else -n)
            elif op == "*":
                affine = a.affine(n, 0)
            elif op == "<<" and a.predicates:
                affine = a.affine(1 << n, 0)
        if affine is None and a.lo == a.hi and op == "*":
            affine = b.affine(a.lo, 0)
        if affine is None and a.lo == a.hi and b.predicates:
            if op == "+":
                affine = b.affine(1, a.lo)
            elif op == "-":
                affine = b.affine(-1, a.lo)
        if affine is not None:
            return finish(affine)
        if op == "-":
            bitset = a.bitset_difference(b)
            if bitset is not None:
                domain, work = bitset
                self.spend(t, work)
                return finish(domain)
        self.spend(t, len(a.ranges) * len(b.ranges))
        spans = []
        for lo, hi in a.ranges:
            for low, high in b.ranges:
                if op == "+":
                    span = lo + low, hi + high
                elif op == "-":
                    span = lo - high, hi - low
                elif op in ("*", "/"):
                    fn = (lambda i, j: i * j) if op == "*" else trunc_div
                    endpoints = [fn(i, j) for i in (lo, hi) for j in (low, high)]
                    span = min(endpoints), max(endpoints)
                elif op == "%":
                    limit = max(abs(low), abs(high)) - 1
                    span = max(lo, -limit) if lo < 0 else 0, min(hi, limit) if hi > 0 else 0
                    if lo == hi and low == high:
                        r = lo - trunc_div(lo, low) * low
                        span = r, r
                elif op in ("<<", ">>"):
                    endpoints = [i << j if op == "<<" else i >> j
                                 for i in (lo, hi) for j in (low, high)]
                    span = min(endpoints), max(endpoints)
                elif lo == hi and low == high:
                    result = {"&": lo & low, "|": lo | low, "^": lo ^ low}[op]
                    span = result, result
                elif op == "&":
                    span = 0, min(hi, high)
                else:
                    span = 0, (1 << max(hi.bit_length(), high.bit_length())) - 1
                spans.append(span)
        domain = Domain.of(*spans)
        if op == "*":
            # Interval product alone loses divisibility: 3*n times [0, 255]
            # is still a multiple of three on every path.
            shared = product_congruence(a, b)
            if shared is not None:
                domain = domain.with_congruence(*shared)
        return finish(domain)

    @staticmethod
    def proven_presence(domain, token):
        if isinstance(domain, FloatDomain):
            return bool(domain)
        if not domain:
            return False
        if domain.intervals or domain.witness() is not None:
            return True
        token.error(f"cannot decide whether {domain} is empty; supply an intersection fact with theorem or axiom")

    def check_intersection_type(self, ty, token):
        if (ACTIVE_ORACLES.get() is not None and isinstance(ty, Scalar) and not ty.floating
                and any(len(c.require) > 1 for c in ty.domain.predicates)):
            self.proven_presence(ty.domain, token)

    def compare_domains(self, op, a, b, token):
        if op in ("==", "!="):
            equal = self.proven_presence(a & b, token)
            unequal = a.lo != a.hi or b.lo != b.hi or a.lo != b.lo
            return (equal, unequal) if op == "==" else (unequal, equal)
        return {
            "<": (a.lo < b.hi, a.hi >= b.lo),
            "<=": (a.lo <= b.hi, a.hi > b.lo),
            ">": (a.hi > b.lo, a.lo <= b.hi),
            ">=": (a.hi >= b.lo, a.lo < b.hi),
        }[op]
