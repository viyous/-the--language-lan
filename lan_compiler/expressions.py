"""Expression values, path restriction and conditions."""

from __future__ import annotations

from dataclasses import field, replace
from itertools import count
from math import inf

from optimization_bridge import LoopReplay
from float_domain import FloatDomain, next_float, c_literal
from value_domain import Domain, DomainComplexity
from table_interface import c_field

from .lexer import Token
from .syntax import Node
from .typesys import (AffineForm, AliasResult, Array, bit_literal_shift, bit_literal_value, BitView, BOOL,
                      CopyArray, HeapArray, HeapObject, HeapRecord, InterfaceResult, is_callback_enum,
                      ReadOnly, Record, Scalar, Slice, Span, Symbol, TaggedResult, unqualified, Value,
                      ValueOrigin, WIDE, ZERO)
from .domain_utils import c_int, c_uint, container, domain_union, replay_scalar
from .state import PathState
from .interface_types import OFFSET_BINARY_OPS
from .c_emit import c_predicate_choice

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class ExpressionMixin:
    def expr(self, node, state):
        previous_active = getattr(self, "_active_paths", ())
        self._active_paths = (state,)
        try:
            value = self.capture_value(self.expr_value(node, state), state)
            effectful = value.effectful or self.has_effects(node)
            if value.enum is None and state.predicates:
                key = self.predicate_key(node, state, allow_debug_checks=True)
                guarded = state.predicates.get(("enum_membership", key))
                if guarded:
                    value = replace(value, guarded_enums=guarded)
            return replace(value, effectful=effectful)
        finally:
            self._active_paths = previous_active

    def assumed_result(self, c_type, result, code, predicates, *, checked=False):
        if not predicates:
            return code
        if checked:
            checks = " ".join(f"if (!({predicate})) __builtin_trap();" for predicate in predicates)
            helper = self.helper("checked_result", None, (c_type, result, checks))
            return f"{helper}({code})"
        self.helper("global_temp", None, (c_type, result))
        assume = self.helper("contract_assume", None)
        checks = ", ".join(f"{assume}({predicate})" for predicate in predicates)
        return f"({result} = {code}, {checks}, {result})"

    def assume_scalar_result(self, token, ty, code):
        result = f"lan_call_result_{token.line}_{token.col}"
        predicates = (self.float_range_predicates(token, result, ty.domain) if ty.floating else
                      self.integer_range_predicates(token, result, ty.domain, container(ty.bits, ty.signed)))
        return self.assumed_result(ty.c, result, code, predicates)

    @staticmethod
    def slice_data_predicates(value, domain):
        if domain.hi == 0:
            return []
        if domain.lo > 0:
            return [f"{value}.data != NULL"]
        return [c_predicate_choice(f"{value}.len == 0", "1", f"{value}.data != NULL")]

    def slice_descriptor_predicates(self, token, value, domain):
        predicates = self.integer_range_predicates(token, f"{value}.len", domain, container(64, False))
        return predicates + self.slice_data_predicates(value, domain)

    def assume_slice_result(self, token, domain, code):
        result = f"lan_slice_result_{token.line}_{token.col}"
        predicates = self.slice_descriptor_predicates(token, result, domain)
        return self.assumed_result("lan_slice", result, code, predicates)

    def keyed_host(self, node):
        if node.kind != "index":
            return None
        host = self.place(node.args[0])
        return host if host is not None and isinstance(host.ty, Record) and host.ty.key_type is not None else None

    def keyed_field(self, node, host, state, *, writable=False):
        token = node.token
        key_expr = node.args[1]
        if host.projection is not None:
            host = self.capture_projected_argument(host, state, token,
                                                   allow_effectful_links=True, propagate_exit=True)
            if host.projection.always_returns:
                return Value(ZERO, host.c, effectful=True, always_returns=True), ()
        elif host.link_access is not None:
            host = self.comparison_place(host, state, allow_effectful_links=True, propagate_exit=True)
            if host.projection is not None and host.projection.always_returns:
                return Value(ZERO, host.c, effectful=True, always_returns=True), ()
        self.require_live(host, state, token)
        owners = self.physical_places(host, state)
        key = self.expr(key_expr, state)
        if key.always_returns:
            return replace(key, code=f"((void)&({host.c}), ({key.code}))"), ()
        enum = host.ty.key_type
        if key.enum != enum.enum or not key.domain.within(enum.domain):
            key_expr.token.error(f"keyed record requires a key of enum {enum.enum.name}")
        candidates = [(index, name, field) for index, (name, field) in enumerate(host.ty.keyed_fields)
                      if key.domain & Domain.point(index)]
        if not candidates:
            key_expr.token.error("keyed record access has no possible field")
        self.spend(token, len(enum.enum.members))
        first = unqualified(candidates[0][2])
        if any(not self.keyed_scalar_compatible(first, unqualified(field)) for _, _, field in candidates):
            key_expr.token.error("key may select fields of different types; narrow the key first")
        # The same source expression can be checked under several incoming
        # states, but has one C representation.
        previous = self.keyed_access_types.setdefault(node, first)
        if not self.keyed_scalar_compatible(previous, first):
            key_expr.token.error("key selects different field types across paths; separate the accesses into branches")
        if writable and (host.readonly or any(isinstance(field, ReadOnly) for _, _, field in candidates)):
            token.error("key may select an immutable field")
        choices = tuple((index, self.child_place(owner, (name,)))
                        for owner in owners for index, name, _ in candidates)
        targets = tuple(dict.fromkeys(field for _, field in choices))
        for target in targets:
            self.require_target_live(target, state, token)
            if writable and target.readonly:
                token.error("key may select an immutable field")
        if self.enum_member_type(key_expr) is not None and len(candidates) == 1:
            code = f"({host.c}).{c_field(candidates[0][1])}"
        else:
            offsets = self.helper("keyed_offsets", host.ty)
            index = key.code
            if self.debug_bounds:
                check = self.helper("debug_index", None)
                index = f"{check}({index}, {len(enum.enum.members)})"
            qualifier = "" if writable else "const "
            helper = self.helper("keyed_access", first.c, (host.ty.c, qualifier, offsets))
            code = f"(*{helper}(&({host.c}), {index}))"
        return Symbol(host.name, host.uid, first, host.depth, c_place=code, captured_places=targets), choices

    @staticmethod
    def keyed_scalar_compatible(left, right):
        if is_callback_enum(left.enum) or is_callback_enum(right.enum):
            return left == right
        return (Compiler.same_scalar_storage(left, right)
                and left.result_arms == right.result_arms
                and left.result_error_enums == right.result_error_enums)

    def expr_value(self, node, state):
        t, k, a = node.token, node.kind, node.args
        self.spend(t)
        if k == "callback_ref":
            self.callable_target(a[0])
            selector = self.callable_selectors[a[0].text]
            return Value(Domain.point(selector), str(selector), 16, enum=self.callable_enum)
        if k == "storage_count":
            return self.count_in_span(node, state)
        if k in ("bit_literal", "binary", "unary") and (literal := bit_literal_value(node)) is not None:
            value, width = literal
            if width > 64:
                t.error("bit literal wider than 64 bits requires a packed array or slice destination")
            return Value(Domain.point(value), c_int(value), width)
        if k == "binary" and a[0] in ("<<", ">>") and (literal := bit_literal_value(a[1])) is not None:
            op, _, amount = a
            value, width = literal
            if width > 64:
                t.error("dynamic shifts of bit literals wider than 64 bits are not supported")
            shift = self.expr(amount, state)
            if shift.always_returns:
                return shift
            if shift.floating or shift.boolean or shift.enum or shift.character or shift.domain.lo < 0:
                amount.token.error("bit literal shift requires a nonnegative integer")
            results = [bit_literal_shift(op, value, width, count) for count in range(width)
                       if shift.domain & Domain.point(count)]
            if shift.domain.hi >= width:
                results.append(0)
            mask = c_int((1 << width) - 1)
            body = f"(a >= {width} ? 0 : (lan_int)(((lan_uint){c_int(value)} {op} a) & {mask}))"
            helper = self.helper("unary_eval", None, ("lan_int", "lan_int", body))
            code = f"{helper}({shift.code})"
            return Value(Domain.union(Domain.point(n) for n in results), code, width, effectful=shift.effectful)
        if k == "guard_snapshot":
            return self.fresh_value(a[0], state)
        if (host := self.keyed_host(node)) is not None:
            field, _ = self.keyed_field(node, host, state)
            if isinstance(field, Value):
                return field
            value = self.storage_value(field, state)
            key = self.predicate_key(node, state)
            fact = state.predicates.get(("keyed_field_domain", key)) if key is not None else None
            if fact is not None:
                narrowed = value.domain & fact
                if narrowed:
                    value = replace(value, domain=narrowed, affine=None, difference=None, linear=None)
            return value
        if k in ("index", "member", "length"):
            host, projections = node, []
            if k == "length":
                projections.append(node)
                host = a[0]
            while host.kind in ("index", "member"):
                projections.append(host)
                host = host.args[0]
            if isinstance(self.composite_call_type(host), (CopyArray, Record)):
                return self.composite_call_projection(node, state, host, reversed(projections))
        alias = self.alias_condition(node, state)
        if alias is not None and k != "name":
            yes, no, code = alias
            if not yes and not no and self.has_propagation(node):
                return Value(BOOL, code, 8, True, effectful=True, always_returns=True)
            return Value(BOOL if yes and no else Domain.point(bool(yes)), code, 8, True)
        if k == "float_number":
            value, = a
            return Value(FloatDomain.point(value, 64), c_literal(value, 64), literal=value)
        if k == "rank_compare":
            return self.compare_ranked(node, state)
        if k == "number":
            n, = a
            if n > WIDE.hi:
                t.error("integer literal exceeds the supported signed 128-bit range")
            return Value(Domain.point(n), c_int(n))
        if k == "character":
            value, = a
            ty = self.types["ascii" if value <= 127 else "unicode"]
            return Value(Domain.point(value), c_int(value), ty.bits, character=ty.character)
        if k == "boolean":
            return Value(Domain.point(a[0]), str(int(a[0])), 8, True)
        if (enum := self.enum_member_type(node)) is not None:
            value = self.enum_member_value(node, enum)
            return Value(Domain.point(value), c_int(value), enum.bits, enum=enum.enum)
        if k == "name" and not any(a[0] in scope for scope in reversed(self.scope)):
            if (type_const := self._singleton_type_constant(a[0])) is not None:
                return self.constant_value(type_const[0])
        sym = self.place(node)
        if sym is not None:
            self.require_live(sym, state, t)
            if isinstance(sym.ty, Record):
                t.error("use a named field to access a record")
            if isinstance(sym.ty, (Array, HeapArray, Slice)):
                t.error("use an index or .len to access an array or slice")
            if isinstance(sym.ty, BitView):
                t.error("use a named field to access a bit view")
            return self.storage_value(sym, state)
        if k == "length":
            if self.span_kind(a[0]) == "ordinary":
                return self.slice_length(a[0], state)
            array = self.array(a[0])
            if array.projection is not None or array.link_access is not None:
                array = self.comparison_place(array, state, propagate_exit=True,
                                              allow_effectful_links=self.has_propagation(a[0]))
                if array.projection is not None and array.projection.always_returns:
                    return Value(ZERO, array.c, effectful=True, always_returns=True)
            self.require_live(array, state, t)
            n = array.ty.length
            code = c_int(n)
            effectful = self.has_effects(a[0])
            if effectful:
                code = f"((void)({array.c}), {code})"
            return Value(Domain.point(n), code, effectful=effectful)
        if k == "index":
            index_values = []
            access = self.subscript(node, state, capture_index=index_values, propagate_exit=True)
            if isinstance(access, Value):
                return access
            sym, index, indices = access
            if isinstance(sym, Span):
                ty = sym.element
                data = self.literal_bytes(sym)
                domain = (self.read_span_contents(state, sym, indices) if data is None or indices.lo != indices.hi
                          else Domain.point(data[indices.lo]))
                if not (self.span_has_effects(a[0], sym) or self.has_effects(a[1])):
                    span, selected = self.correlated_span_element(state, sym, indices, index_values[0])
                    if span is not sym:
                        domain = self.read_span_contents(state, span, selected)
                    domain = self.known_span_window(state, sym, indices, domain)
                    domain = self.known_array_element(state, self.array_element_key(sym, index_values[0]), domain)
                code = f"{sym.c_data()}[{index}]"
                if not self.debug_bounds:
                    # A single C expression can serve several analysis paths.
                    # Only the declared element domain holds on every path.
                    code = self.assume_scalar_result(t, ty, code)
                return Value(domain, code, ty.bits, ty.boolean,
                             enum=ty.enum, character=ty.character, signed=ty.signed,
                             result_arms=ty.result_arms, result_error_enums=ty.result_error_enums)
            ty = sym.ty.element
            if sym.ty.packed_bits:
                helper = self.helper("bit_read", None)
                code = f"{helper}({sym.c}, {index})"
                offset = indices.lo if indices.lo == indices.hi else None
                value = self.known_bits(state, sym, offset, 1, "le")
                domain = ty.domain if value is None else Domain.point(value)
                if offset is None and not any(self.has_effects(part) for part in node.args):
                    position = replace(index_values[0], domain=indices)
                    domain = self.bit_window_read_domain(state, sym, position, Value(Domain.point(1), "1"))
                return Value(domain, code, ty.bits)
            else:
                code = f"{sym.c}[{index}]"
            return Value(ty.domain, code, ty.bits, ty.boolean, enum=ty.enum, character=ty.character, signed=ty.signed,
                         result_arms=ty.result_arms, result_error_enums=ty.result_error_enums)
        if k == "slice":
            if self.span_kind(node) == "ordinary":
                t.error("use an index or .len to access an ordinary slice")
            if self.dynamic_bit_slice(node):
                window = self.dynamic_bit_window(node, state)
                if isinstance(window, Value):
                    return window
                sym, start, width, setup = window
                if setup:
                    value = self.read_bits(t, state, sym, start, width)
                    if all(not self.has_effects(part) for part in node.args if isinstance(part, Node)):
                        value = replace(value, domain=self.bit_window_read_domain(state, sym, start, Value(Domain.point(width), "")))
                    steps = ", ".join(s.rstrip(";") for s in setup)
                    return replace(value, code=f"({steps}, {value.code})")
            else:
                sym, start, width = self.bit_span(node)
            return self.read_bits(t, state, sym, start, width)
        if k == "member":
            sym, bit = self.bit_field_target(node)
            value = self.read_bits(t, state, sym, bit.offset, bit.width, bit.ty, bit.endian)
            if not self.debug_bounds:
                ty = bit.ty
                if not ty.floating:
                    ty = replace(ty, domain=ty.domain & container(bit.width, ty.signed))
                code = self.assume_scalar_result(t, ty, value.code)
                value = replace(value, code=code)
            return replace(value, result_arms=bit.ty.result_arms, result_error_enums=bit.ty.result_error_enums)
        if k == "call":
            relation, early_return, cases = [], [], []
            ty, code = self.call(node, state, capture_result_relation=relation, capture_early_return=early_return,
                                 capture_return_cases=cases)
            if early_return:
                return early_return[0]
            if ty is None:
                t.error("void function has no value")
            if isinstance(ty, TaggedResult):
                t.error("use '?' to unwrap the composite result")
            if isinstance(ty, (Record, CopyArray, Slice, HeapRecord, HeapObject, HeapArray, AliasResult, InterfaceResult)):
                t.error("bind a composite result to a declaration before accessing its contents")
            # Call analysis can narrow the result on one path. Shared C
            # uses the declared domain and the runtime dependent predicates.
            declared = self.composite_call_signature(node).result
            code = self.assume_scalar_result(t, declared, code)
            value = Value(ty.domain, code, ty.bits, ty.boolean, enum=ty.enum, character=ty.character,
                          signed=ty.signed, result_arms=ty.result_arms, result_error_enums=ty.result_error_enums)
            if relation:
                exact, bounds = relation[0]
                if exact is not None:
                    value = replace(value, affine=exact.affine, difference=exact.difference, linear=exact.linear)
                else:
                    value = replace(value, affine=AffineForm(ValueOrigin(value.domain)))
                for op, bound in bounds:
                    state.absorb(self.assume_dependent_expression(t, state, value, op, bound,
                                                                 "dependent result bounds have no possible value"))
            if cases:
                if value.affine is None:
                    value = replace(value, affine=AffineForm(ValueOrigin(value.domain)))
                origin = value.affine.origin
                key = ("return_cases", origin)
                rows, places = cases[0]
                previous = state.predicates.get(key, ())
                if len(previous) < 4 and sum(item[:1] == ("return_cases",) for item in state.predicates) < 32:
                    state.predicates = state.predicates | {key: (*previous, (value.affine, rows, places))}
            return value
        if k == "propagate":
            operand, = a
            if self.composite_call_type(node) is not None:
                t.error("bind the unwrapped composite result or access a supported field or element")
            value = self.expr(operand, state)
            if value.always_returns:
                return value
            if value.result_arms is None:
                t.error("'?' requires a result union written as Success | NamedError")
            success, errors = value.result_arms
            good, error = value.domain & success, value.domain & errors
            inferring = self.inferred_returns is not None
            target = self.functions[self.current][1].result
            if not inferring and target is None:
                t.error("cannot propagate an error from a void function")
            if not inferring and not isinstance(target, (Scalar, TaggedResult)):
                t.error("numeric error propagation requires a result contract with an error arm")
            tmp = f"lan_error_{t.line}_{t.col}"
            exit_code = target.wrap(tmp, success=False) if isinstance(target, TaggedResult) else tmp
            error_paths = []
            if self.proven_presence(error, t):
                error_paths = self.propagated_error_paths(t, self.restrict(state, operand, error))
                propagated = replace(value, domain=error, code=tmp, effectful=False, literal=None,
                                     result_arms=None, result_error_enums=(), guarded_enums=frozenset())
                if inferring:
                    if error_paths:
                        self.inferred_returns.append((t, replace(propagated, result_arms=(success - success, error),
                                                                 result_error_enums=value.result_error_enums)))
                elif isinstance(target, TaggedResult):
                    if (value.result_error_enums != (target.error.enum.identity,)
                            or self.proven_presence(error - target.error.domain, t)):
                        t.error("propagated error must fit the caller's named error arm")
                else:
                    self.range_check(t, propagated, target)
                    if target.result_arms is not None:
                        foreign_enums = set(value.result_error_enums) - set(target.result_error_enums)
                        if foreign_enums and target.result_error_enums:
                            known = {ty.enum.identity: ty.enum for ty in self.types.values()
                                     if isinstance(ty, Scalar) and ty.enum is not None}
                            if any(identity not in known or self.proven_presence(
                                    error & Domain.of(*((code, code) for _, code in known[identity].members)), t)
                                   for identity in foreign_enums):
                                t.error("propagated error enum identity differs from the caller's error arm")
                        converted = self.float_coerce(t, propagated, target.bits) if target.floating else propagated
                        if self.proven_presence(converted.domain - target.result_arms[1], t):
                            t.error("propagated error must fit the caller's error arm")
                    for error_path in error_paths:
                        self.check_result_bounds(t, propagated, error_path)
                    if target.result_arms is not None and not target.floating:
                        self.result_exit_domains.extend(error for _ in error_paths)
                leaving = [path.guard(f"error@{t.line}:{t.col}") for path in error_paths]
                self.trace_stack_exit(leaving, 0, "error return")
                self.return_states.extend(leaving)
            test = self.membership(t, errors)
            self.trace.append(f"  propagate error {error}; continue with {good}")
            normal = self.restrict(state, operand, good)
            location = f"@{t.line}:{t.col}"
            self.states(f"propagate {location} return error {error}", error_paths)
            self.states(f"propagate {location} continue success {good}", normal)
            if normal:
                state.absorb(PathState.join(normal))
            representation = Scalar(value.domain, value.width, value.signed, floating=value.floating).c
            code = (f"({{ __label__ lan_error_continue; {representation} {tmp} = {value.code}; "
                    f"if (!{test}({tmp})) goto lan_error_continue; {self.propagation_cleanup()}return {exit_code}; "
                    f"lan_error_continue: {tmp}; }})")
            if not good:
                return Value(success, code, value.width, signed=value.signed, effectful=True, always_returns=True)
            return replace(value, domain=good, code=code, effectful=True, literal=None,
                           result_arms=None, result_error_enums=(), guarded_enums=frozenset())
        if k == "cast":
            operand, strategy, annotation = a
            ty = self.scalar_type(annotation)
            if (strategy == "safe" and not ty.floating and not ty.boolean and operand.kind == "slice"
                    and self.span_kind(operand) == "packed" and self.dynamic_bit_slice(operand)):
                value = self.read_variable_bits(operand, state)
                return value if value.always_returns else self.cast(t, value, strategy, ty)
            if ty.boolean and strategy == "safe" and self.alias_node(operand) is not None:
                yes, no, code = self.alias_condition(operand, state)
                return Value(BOOL if yes and no else Domain.point(bool(yes)), code, 8, True)
            if ty.boolean and strategy == "safe":
                if self.span_kind(operand) == "ordinary":
                    length = self.slice_length(operand, state)
                    return length if length.always_returns else self.cast(t, length, strategy, ty)
                packed = self.packed_truth(operand, state)
                if packed is not None:
                    return packed
            if strategy == "view":
                sym = self.place(operand)
                if sym is not None and isinstance(sym.ty, Array) and sym.ty.packed_bits:
                    return self.packed_scalar_view(t, state, sym, ty)
                if sym is not None and isinstance(sym.ty, (Array, Record)):
                    return self.view_composite_scalar(t, state, sym, ty)
                if operand.kind == "cast" and operand.args[1] == "view":
                    inner, _, annotation = operand.args
                    intermediate = self.resolve(annotation)
                    if isinstance(intermediate, Array) and intermediate.packed_bits:
                        return self.inline_packed_scalar_view(t, state, inner, intermediate, ty)
                    if isinstance(intermediate, Record) or isinstance(intermediate, Array) and not intermediate.packed_bits:
                        sym, (view_state,) = self.composite_view_temporary(intermediate, inner, [state], operand.token)
                        return self.view_composite_scalar(t, view_state, sym, ty)
                sym, (view_state,) = self.composite_call_view_source(operand, [state])
                if sym is not None:
                    if isinstance(sym.ty, Array) and sym.ty.packed_bits:
                        return self.packed_scalar_view(t, view_state, sym, ty)
                    return self.view_composite_scalar(t, view_state, sym, ty)
            wrap32_ids = set()
            if strategy == "wrap" and not ty.floating and not ty.boolean and ty.bits <= 32:
                def collect_wrap32(n):
                    if n.kind == "binary" and n.args[0] in ("+", "-", "*", "&", "|", "^"):
                        wrap32_ids.add(id(n))
                        collect_wrap32(n.args[1])
                        collect_wrap32(n.args[2])
                    elif n.kind == "binary" and n.args[0] == "<<" and n.args[2].kind == "number" and 0 <= n.args[2].args[0] < 32:
                        wrap32_ids.add(id(n))
                        collect_wrap32(n.args[1])
                    elif n.kind == "unary" and n.args[0] in ("-", "+", "~"):
                        wrap32_ids.add(id(n))
                        collect_wrap32(n.args[1])
                collect_wrap32(operand)
            saved_wrap32 = self.wrap32_nodes
            if wrap32_ids:
                self.wrap32_nodes = saved_wrap32 | wrap32_ids
            try:
                value = self.expr(operand, state)
            finally:
                self.wrap32_nodes = saved_wrap32
            if value.always_returns and id(node) in self.propagating_expression_nodes:
                return value
            if (strategy in ("safe", "trunc") and replay_scalar(ty) and not ty.boolean
                    and isinstance(getattr(self, "optimization_replay", None), LoopReplay)):
                value = self.ssa_narrow(node, value, ty.domain, "conversions", "conversion")
            result = self.cast(t, value, strategy, ty)
            if strategy == "trap":
                normal = self.restrict(state, operand, result.domain)
                if normal:
                    state.absorb(PathState.join(normal))
            return result
        if k == "in":
            val = self.expr(a[0], state)
            if is_callback_enum(val.enum):
                t.error("function values do not support numeric membership tests")
            if val.boolean:
                t.error("membership tests currently require an integer")
            domain = self.float_set(a[1], val.domain.bits) if val.floating else self.numeric_type(a[1]).domain & WIDE
            helper = self.membership(t, domain)
            d_unsigned = bool(domain) and domain.lo >= 0 and domain.hi >= (1 << 63)
            v_unsigned = bool(val.domain) and val.domain.lo >= 0 and val.domain.hi >= (1 << 63)
            if not val.floating and bool(val.domain) and val.domain.lo < 0 and d_unsigned:
                code = f"(((lan_int)({val.code})) >= 0 && {helper}((lan_uint)({val.code})))"
            elif not val.floating and v_unsigned and not d_unsigned:
                code = f"(((lan_uint)({val.code})) <= (lan_uint){c_int((1 << 63) - 1)} && {helper}((lan_int)({val.code})))"
            else:
                code = f"{helper}({val.code})"
            if val.always_returns:
                return Value(BOOL, code, 8, True, effectful=True, always_returns=True)
            yes = self.proven_presence(val.domain & domain, t)
            no = self.proven_presence(val.domain - domain, t)
            return Value(BOOL if yes and no else Domain.point(yes), code, 8, True)
        if k == "binary":
            op, left, right = a
            if op in ("&&", "||"):
                yes, no, code = self.condition(node, state)
                if not yes and not no:
                    return Value(BOOL, code, 8, True, effectful=True, always_returns=True)
                state.absorb(PathState.join(yes + no))
                return Value(BOOL if yes and no else Domain.point(bool(yes)), code, 8, True)
            rewritten = self.rewrite_property_binary(node, state)
            if rewritten is not None:
                return rewritten
            span_compare = self.compare_spans(node, state)
            if span_compare is not None:
                return span_compare
            lhs = self.expr(left, state)
            track_exit = id(node) in self.propagating_expression_nodes
            if track_exit and lhs.always_returns:
                return lhs
            rhs = self.expr(right, state)
            if track_exit and rhs.always_returns:
                return replace(rhs, code=f"((void)({lhs.code}), ({rhs.code}))")
            result = self.binary_expression(t, op, left, right, lhs, rhs, state,
                                            wrap32=id(node) in self.wrap32_nodes)
            if self.trace_endpoint and op in OFFSET_BINARY_OPS:
                self.trace.append(f"    eval expr [{t.line}:{t.col}]: {lhs.domain} {op} {rhs.domain} => {result.domain}")
            return result
        if k == "unary":
            op, operand = a
            if op == "!":
                yes, no, code = self.condition(node, state)
                if not yes and not no:
                    return Value(BOOL, code, 8, True, effectful=True, always_returns=True)
                state.absorb(PathState.join(yes + no))
                return Value(BOOL if yes and no else Domain.point(bool(yes)), code, 8, True)
            if op == "-" and operand.kind == "number" and operand.args[0] == -WIDE.lo:
                return Value(Domain.point(WIDE.lo), c_int(WIDE.lo))
            val = self.expr(operand, state)
            if val.always_returns and id(node) in self.propagating_expression_nodes:
                return val
            if val.floating:
                if op not in ("+", "-"):
                    t.error("floating unary arithmetic supports + and -")
                literal = val.literal
                if op == "-" and literal is not None:
                    literal = literal[1:] if literal.startswith("-") else "-" + literal
                return replace(val, domain=val.domain if op == "+" else val.domain.negate(),
                               code=f"({op}({val.code}))", literal=literal)
            if val.boolean or val.enum or val.character:
                t.error(f"unary {op!r} is not supported for this operand")
            if op == "~":
                signed = val.signed or not val.width
                if signed:
                    mask, width = -1, val.width
                else:
                    width = val.width
                    if not val.domain.within(container(width, False)):
                        t.error("complement operand exceeds its physical width; cast to a wider integer first")
                    mask = (1 << width) - 1
                domain = val.domain.affine(-1, mask)
                if not signed and width <= 32:
                    code = f"((uint32_t)(UINT32_C({mask}) ^ (uint32_t)({val.code})))"
                elif signed and (id(node) in self.wrap32_nodes or (-(1 << 31) <= val.domain.lo and val.domain.hi <= (1 << 31) - 1)):
                    code = f"(~(uint32_t)({val.code}))" if id(node) in self.wrap32_nodes else f"(~(int32_t)({val.code}))"
                else:
                    code = f"((lan_int)({c_uint(mask)} ^ (lan_uint)({val.code})))" if not signed else f"(~(lan_int)({val.code}))"
                form = val.affine.transform(-1, mask) if val.affine else None
                difference = val.difference.negated() if val.difference else None
                if difference is not None:
                    difference = replace(difference, offset=difference.offset + mask)
                value = self.emit_value(t, domain, code, width, affine=form, difference=difference, signed=signed)
                return replace(value, linear=self.combine_linear("-", Value(Domain.point(mask), c_int(mask)), val))
            domain = val.domain if op == "+" else val.domain.affine(-1, 0)
            form = val.affine.transform(-1 if op == "-" else 1, 0) if val.affine else None
            difference = val.difference
            if difference is not None and op == "-":
                difference = difference.negated()
            if id(node) in self.wrap32_nodes:
                unary_code = f"({op}((uint32_t)({val.code})))"
            elif op == "+" and 0 <= val.domain.lo and val.domain.hi <= (1 << 32) - 1:
                unary_code = f"(((uint32_t)({val.code})))"
            elif -(1 << 31) <= val.domain.lo and val.domain.hi <= (1 << 31) - 1 and -(1 << 31) <= domain.lo and domain.hi <= (1 << 31) - 1:
                unary_code = f"({op}((int32_t)({val.code})))"
            else:
                unary_code = f"({op}((lan_int)({val.code})))"
            value = self.emit_value(t, domain, unary_code, val.width,
                                    affine=form, difference=difference, signed=val.signed or op == "-")
            linear = (self.combine_linear("-", Value(ZERO, "0"), val) if op == "-"
                      else self.linear_value(val))
            return replace(value, linear=linear)
        if k == "text_literal":
            t.error("use an index, length or slice operation on a string")
        if k == "copy":
            t.error("as: copy requires a slice assignment destination")
        if k == "bit_view":
            t.error("declare a bit field view with 'define name = host[...]'")
        t.error("array initializers are only valid in a fixed array declaration")

    @staticmethod
    def _uniform_array_slice(contents, selected):
        if len(contents.segments) <= 1:
            return True
        first = None
        for seg in contents.segments:
            if seg.stop <= selected.lo or seg.start > selected.hi:
                continue
            if selected & Domain.of((seg.start, seg.stop - 1)):
                if first is None:
                    first = seg.domain
                elif seg.domain != first:
                    return False
        return True

    def restrict(self, state, expr, domain, *, enum_guard=False):
        if not self.proven_presence(domain, expr.token):
            return []
        previous_active = getattr(self, "_active_paths", ())
        self._active_paths = (state,)
        try:
            return self._restrict_inner(state, expr, domain, enum_guard=enum_guard)
        finally:
            self._active_paths = previous_active

    def _restrict_inner(self, state, expr, domain, *, enum_guard=False):
        if expr.kind == "propagate" and not self.has_effects(expr.args[0]):
            return self.restrict(state, expr.args[0], domain, enum_guard=enum_guard)
        if (host := self.keyed_host(expr)) is not None:
            # Inline record fields have individual storage epochs. Projected
            # records and link targets require a different dependency key.
            key = (self.predicate_key(expr, state)
                   if host.projection is None and host.link_access is None and expr.args[0].kind == "name"
                   else None)
            if key is None:
                return [state]
            tag = "keyed_field_domain", key
            existing = state.predicates.get(tag)
            if existing is not None:
                domain &= existing
            return [replace(state, predicates=state.predicates | {tag: domain})] if domain else []
        subject = expr.args[0] if expr.kind == "slice" else expr
        place = self.place(subject)
        if place is not None and place.projection is not None:
            return [state]
        if (expr.kind == "slice" and self.span_kind(expr) != "ordinary" and self.dynamic_bit_slice(expr)
                and not any(self.has_effects(part) for part in expr.args if isinstance(part, Node))):
            _, offset, _, _ = self.bit_region(expr, state)
            candidates = self.split_affine_value(offset.domain, offset.affine, state)
            paths = []
            for candidate in candidates:
                sym, first, size, _ = self.bit_region(expr, candidate)
                if first.domain.lo == first.domain.hi and size.domain.lo == size.domain.hi:
                    index, count = first.domain.lo, size.domain.lo
                    compatible = domain & self.bit_read_domain(candidate, sym, index, count)
                    if compatible:
                        paths.extend(self.restrict_bit_read(candidate, sym, index, count, compatible))
                else:
                    paths.append(candidate)
            self.budget_paths(paths, expr.token)
            return paths
        if not self.has_effects(expr):
            if expr.kind == 'member' and self.field_place(expr) is None and self.enum_member_type(expr) is None:
                sym, bit = self.bit_field_target(expr)
                return self.restrict_bit_read(state, sym, bit.offset, bit.width, domain, bit.endian)
            if expr.kind == 'slice' and self.span_kind(expr) != 'ordinary':
                if self.dynamic_bit_slice(expr):
                    return [state]
                sym, offset, width = self.bit_span(expr)
                return self.restrict_bit_read(state, sym, offset, width, domain)
        if (expr.kind == "index" and self.span_kind(expr.args[0]) != "ordinary"
                and not any(self.has_effects(part) for part in expr.args)):
            captured = []
            sym, _, indices = self.subscript(expr, state, capture_index=captured)
            if indices.lo == indices.hi:
                return self.restrict_bit_read(state, sym, indices.lo, 1, domain)
            candidates = self.split_affine_value(indices, captured[0].affine, state)
            if len(candidates) > 1 or candidates and candidates[0] is not state:
                paths = []
                for candidate in candidates:
                    resolved, _, positions = self.subscript(expr, candidate)
                    if positions.lo == positions.hi:
                        paths.extend(self.restrict_bit_read(candidate, resolved, positions.lo, 1, domain))
                    else:
                        paths.append(candidate)
                self.budget_paths(paths, expr.token)
                return paths
        if (expr.kind == "index" and self.span_kind(expr.args[0]) == "ordinary"
                and not any(self.has_effects(part) for part in expr.args)):
            captured = []
            span, _, indices = self.subscript(expr, state, capture_index=captured)
            positions = span.positions()
            skip_split = False
            if len(positions) == 1:
                owner, offset = positions[0]
                if not owner.chain_summary and not (owner.static_storage and isinstance(owner.ty, Slice)):
                    refined = self.refined_span_offset(state, span, offset)
                    contents = self.actual_array(state, owner)
                    selected = self.array_indices(contents, refined, indices)
                    if selected:
                        current_val = contents.read(selected)
                        if hasattr(current_val, "within") and current_val.within(domain):
                            skip_split = True
                        elif (self._uniform_array_slice(contents, selected)
                              and self.array_element_key(span, captured[0]) is not None):
                            skip_split = True
            candidates = [state] if skip_split else self.split_affine_value(indices, captured[0].affine, state)
            if not skip_split:
                joint = [path for candidate in candidates for path in self.split_slice_start(span, candidate)]
                if len(joint) <= self.PATH_LIMIT:
                    candidates = joint
            result = []
            for candidate in candidates:
                if candidate is state:
                    current, selected_indices, index_value = span, indices, captured[0]
                else:
                    narrowed_index = []
                    current, _, selected_indices = self.subscript(expr, candidate, capture_index=narrowed_index)
                    index_value = narrowed_index[0]
                positions = current.positions()
                if len(positions) != 1:
                    result.append(candidate)
                    continue
                owner, offset = positions[0]
                if owner.chain_summary or owner.static_storage and isinstance(owner.ty, Slice):
                    result.append(candidate)
                    continue
                offset = self.refined_span_offset(candidate, current, offset)
                contents = self.actual_array(candidate, owner)
                selected = self.array_indices(contents, offset, selected_indices)
                if not selected:
                    result.append(candidate)
                    continue
                narrowed = contents.read(selected) & domain
                if not narrowed:
                    continue
                if selected.lo == selected.hi:
                    result.append(candidate.with_array(owner, contents.fill(selected.lo, selected.lo + 1, narrowed)))
                    continue
                key = self.array_element_key(current, index_value)
                if key is not None:
                    facts = candidate.predicates
                    if key in facts or sum(1 for _ in self.current_array_element_facts(candidate, owner.uid)) < 8:
                        narrowed = self.known_array_element(candidate, key, narrowed)
                        if not narrowed:
                            continue
                        candidate = replace(candidate, predicates=facts | {
                            key: (candidate.epochs.get(owner.uid), narrowed, selected)})
                result.append(candidate)
            self.budget_paths(result, expr.token)
            return result
        if expr.kind == "binary" and expr.args[0] == "&" and domain.lo == domain.hi:
            _, left, right = expr.args
            if left.kind == "number":
                left, right = right, left
            place = self.place(left)
            if place is not None and isinstance(place.ty, Scalar) and right.kind == "number":
                modulus = right.args[0] + 1
                if modulus & (modulus - 1) == 0:
                    current = self.storage_value(place, state).domain
                    return self.restrict(state, left, current.with_congruence(domain.lo, modulus))
        if (field := self.field_place(expr)) is not None:
            sym = field
        elif expr.kind == "length" and expr.args[0].kind == "name":
            sym = self.lookup(expr.args[0])
            if not isinstance(sym.ty, (Slice, HeapArray)) or (isinstance(sym.ty, Slice) and sym.ty.projection):
                return [state]
        elif expr.kind == "name":
            sym = self.lookup(expr)
        else:
            return [state]
        targets = self.physical_places(sym, state)
        if any(target.projection is not None for target in targets):
            return [state]
        repeated = self.repeated_allocation_uids(state, expr.token)
        if any(target.heap_owner in repeated for target in targets):
            return [state]
        if any(target.chain_summary for target in targets):
            if not isinstance(sym.ty, Scalar):
                return [state]
            key = self.chain_access_key("chain_value", sym, state)
            narrowed = self.storage_value(sym, state).domain & domain
            return [replace(state, predicates=state.predicates | {key: narrowed})] if narrowed else []
        if sym.link_access is not None:
            if len(targets) != 1:
                return [state]
            refined = state.narrow(targets[0], domain)
            return [refined] if refined is not None else []
        if sym.uid in self.alias_members:
            root, names = self.alias_members[sym.uid]
            result = []
            for owner in state.borrows[root]:
                if not owner.uid:
                    continue
                target = self.child_place(owner, names)
                refined = replace(state, borrows=state.borrows | {root: frozenset((owner,))}).narrow(target, domain)
                if refined is not None:
                    result.append(refined)
            return result
        refined = state.narrow(sym, domain)
        return [refined] if refined is not None else []

    def assume_difference(self, state, difference, op, token):
        if difference is None:
            return state
        zone = state.zone
        if op in ("<", "<=", "=="):
            zone = zone.constrain(difference, -1 if op == "<" else 0)
        if zone is not None and op in (">", ">=", "=="):
            zone = zone.constrain(difference.negated(), -1 if op == ">" else 0)
        if op == "!=":
            bounds = zone.range(difference)
            if bounds is not None and bounds[0] >= 0:
                zone = zone.constrain(difference.negated(), -1)
            elif bounds is not None and bounds[1] <= 0:
                zone = zone.constrain(difference, -1)
        if zone != state.zone:
            self.spend(token, len(state.zone.ids) ** 2)
        return state.with_zone(zone)

    def compare_path(self, state, left, right, a, b, op, possible, difference, samples=None):
        if not possible:
            return []
        state = self.assume_difference(state, difference, op, left.token)
        if state is None:
            return []
        if left.kind == right.kind == "name" and self.lookup(left) == self.lookup(right):
            return [state] if op in ("==", "<=", ">=") else []
        if op == "==":
            a = b = a & b
        elif op == "!=":
            old_a, old_b = a, b
            if b.lo == b.hi:
                a = a - b
            if old_a.lo == old_a.hi:
                b = old_b - old_a
        else:
            if op in (">", ">="):
                left, right, a, b = right, left, b, a
                if samples is not None:
                    samples = samples[::-1]
            delta = int(op in ("<", ">"))
            a, b = a & Domain.of((WIDE.lo, b.hi - delta)), b & Domain.of((a.lo + delta, WIDE.hi))
        paths = [n for s in self.restrict(state, left, a) for n in self.restrict(s, right, b)]
        if samples is not None:
            x, y = samples
            paths = [path for before in paths
                     if (after := self.refine_return_arguments(before, x, a)) is not None
                     if (path := self.refine_return_arguments(after, y, b)) is not None]
        if op == "==" and left.kind == right.kind == "name":
            src, dst = self.lookup(left), self.lookup(right)
            old_left, old_right = state.forms.get(src.uid), state.forms.get(dst.uid)
            # The zone already records equality. Keep an origin that a slice captured
            # so later proofs can still relate the descriptor to this variable.
            captured = old_right is not None and any(
                start.affine is not None and start.affine.origin is old_right.origin
                for start in state.starts.values())
            if captured and difference is not None and state.zone.range(difference) == (0, 0):
                return paths
            if old_left is not None and (old_right is None or old_left.origin is not old_right.origin):
                paths = [path.bind_equal(dst, old_left) for path in paths]
        return paths

    @staticmethod
    def saved_return_cases(state, value):
        if value.affine is None:
            return ()
        return state.predicates.get(("return_cases", value.affine.origin), ())

    def constrain_call_arguments(self, state, requirements, places):
        for requirement, place in zip(requirements, places):
            if place is None:
                continue
            uid, epoch = place
            if uid in state.values and state.epochs.get(uid) is epoch and not state.values[uid].within(requirement):
                state = state.narrow(self.symbols[uid - 1], requirement)
                if state is None:
                    return None
        return state

    def refine_return_arguments(self, state, value, domain):
        if not isinstance(domain, Domain):
            return state
        summaries = self.saved_return_cases(state, value)
        if not summaries:
            return state
        original = state
        try:
            origin = value.affine.preimage(domain)
            for form, rows, places in summaries:
                selected = origin.affine(form.scale, form.offset)
                active = [(index, uid) for index, place in enumerate(places) if place is not None
                          for uid, epoch in (place,)
                          if uid in state.values and state.epochs.get(uid) is epoch]
                matching = [row for row in rows if row.result & selected and all(
                    row.arguments[index] & state.values[uid] for index, uid in active)]
                if not matching:
                    return None
                arguments = tuple(domain_union(row.arguments[index] for row in matching)
                                  for index in range(len(places)))
                state = self.constrain_call_arguments(state, arguments, places)
                if state is None:
                    return None
        except DomainComplexity:
            return original
        return state

    def refine_return_paths(self, paths, value, domain):
        return [refined for path in paths
                if (refined := self.refine_return_arguments(path, value, domain)) is not None]

    def guarded_value_equality(self, left, right, state):
        if self.has_effects(left) or self.has_effects(right):
            return False
        unresolved = []
        equality_edges = None

        def match_index(i, j):
            if i is None or j is None:
                return i is j
            index = self.predicate_key(i, state)
            if index is None or index != self.predicate_key(j, state):
                unresolved.append((i, j))
            return True

        def exclusive_end(host, high, closed):
            if high is None:
                return Node(host.token, "length", (host, host.token))
            if closed:
                one = Node(high.token, "number", (1,))
                return Node(high.token, "binary", ("+", high, one))
            return high

        def connected(a, b):
            nonlocal equality_edges
            if a is None or b is None:
                return False
            for x, y in ((a, b), (b, a)):
                if (state.predicates.get(("binary", ("===", x, y))) is True
                        or state.predicates.get(("binary", ("!==", x, y))) is False):
                    return True
            if equality_edges is None:
                equality_edges = {}
                for key, outcome in state.predicates.items():
                    if not isinstance(key, tuple) or len(key) != 2 or key[0] != "binary" or len(key[1]) != 3:
                        continue
                    op, x, y = key[1]
                    if (op == "===" and outcome is True) or (op == "!==" and outcome is False):
                        equality_edges.setdefault(x, set()).add(y)
                        equality_edges.setdefault(y, set()).add(x)
            pending = [a]
            seen = {a}
            while pending:
                neighbors = equality_edges.get(pending.pop(), ())
                if b in neighbors:
                    return True
                for neighbor in neighbors:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(neighbor)
            return False

        while True:
            if connected(self.predicate_key(left, state), self.predicate_key(right, state)):
                return all(self.prove_order(i.token, self.expr(i, replace(state)), "==",
                                            self.expr(j, replace(state)), state)
                           for i, j in unresolved)
            if left.kind != right.kind:
                return False
            if left.kind == "slice":
                a, low, high, closed = left.args
                b, other_low, other_high, other_closed = right.args
                if self.span_kind(a) != "ordinary" or self.span_kind(b) != "ordinary":
                    return False
                if low is not None or other_low is not None:
                    zero = Node(left.token, "number", (0,))
                    if not match_index(low if low is not None else zero, other_low if other_low is not None else zero):
                        return False
                if high is not None or other_high is not None:
                    if high is None or other_high is None or closed != other_closed:
                        high = exclusive_end(a, high, closed)
                        other_high = exclusive_end(b, other_high, other_closed)
                    if not match_index(high, other_high):
                        return False
                left, right = a, b
                continue
            if left.kind not in ("index", "member"):
                return False
            a, i = left.args
            b, j = right.args
            if left.kind == "index":
                if self.span_kind(a) != "ordinary" or self.span_kind(b) != "ordinary":
                    return False
                if not match_index(i, j):
                    return False
            elif i.text != j.text or self.record_place(a) is None or self.record_place(b) is None:
                return False
            left, right = a, b

    def binary_expression(self, token, op, left, right, x, y, state, *, wrap32=False):
        comparison = op in ("==", "!=", "===", "!==", "<", "<=", ">", ">=")
        if comparison and x.enum != y.enum:
            if x.enum is not None and x.enum.identity in y.result_error_enums:
                domain = Domain.of(*((code, code) for _, code in x.enum.members))
                y = self.coerce_result_enum(y, Scalar(domain, x.width, x.signed, enum=x.enum))
            elif y.enum is not None and y.enum.identity in x.result_error_enums:
                domain = Domain.of(*((code, code) for _, code in y.enum.members))
                x = self.coerce_result_enum(x, Scalar(domain, y.width, y.signed, enum=y.enum))
        result = self.binary(token, op, x, y, state, wrap32=wrap32)
        integer_cancellation = op in ("-", "^") and not (x.floating or y.floating)
        if (comparison or integer_cancellation) and self.guarded_value_equality(left, right, state):
            if comparison:
                return replace(result, domain=Domain.point(op in ("==", "===", "<=", ">=")))
            return replace(result, domain=ZERO, affine=None, difference=None, linear=None)
        return result

    def predicate_key(self, node, state, *, allow_debug_checks=False):
        if node.kind == "guard_outcomes" or self.has_effects(node, allow_debug_checks=allow_debug_checks):
            return None

        def dependencies(sym, seen=None):
            seen = set() if seen is None else seen
            if sym.uid in seen:
                return ()
            seen.add(sym.uid)
            result = [(sym.uid, state.epochs.get(sym.uid))]
            if isinstance(sym.ty, BitView):
                result.extend(dependencies(sym.ty.host, seen))
            if isinstance(sym.ty, Slice) and sym.ty.projection is not None:
                for owner in sym.ty.projection.owners:
                    result.extend(dependencies(owner, seen))
            for target in self.physical_places(sym, state, allow_null=True):
                result.append((target.uid, state.epochs.get(target.uid)))
                for child in target.fields:
                    result.extend(dependencies(child, seen))
            for owner in state.borrows.get(sym.uid, ()):
                result.extend(dependencies(owner, seen))
            return tuple(result)

        def canonical(value):
            if isinstance(value, Node):
                if value.kind == 'name':
                    name, = value.args
                    sym = next((scope[name] for scope in reversed(self.scope) if name in scope), None)
                    if sym is not None:
                        return ('storage', sym.uid, dependencies(sym))
                return (value.kind, tuple(canonical(arg) for arg in value.args))
            if isinstance(value, Token):
                return value.text
            if isinstance(value, (tuple, list)):
                return tuple(canonical(arg) for arg in value)
            return value
        return canonical(node)

    def condition(self, node, state, *, require_boolean=False):
        previous_active = getattr(self, "_active_paths", ())
        self._active_paths = (state,)
        try:
            key = self.predicate_key(node, state)
            if (node.kind == "binary" and node.args[0] in ("==", "!=")
                    and (key is None and self.debug_bounds or any(operand.kind == "slice" for operand in node.args[1:]))
                    and all(self.span_kind(operand) == "ordinary" for operand in node.args[1:])):
                key = self.slice_address_fact_key(node, state)
            known = state.predicates.get(key) if key is not None else None
            # Still produce the real expression on every path: a cached proof must
            # never replace a shared C condition by a path-specific constant.
            yes, no, code = self.condition_value(node, replace(state), require_boolean=require_boolean)
            if known is True:
                no = []
            elif known is False:
                yes = []
            if key is not None:
                yes = [replace(path, predicates=path.predicates | {key: True}) for path in yes]
                no = [replace(path, predicates=path.predicates | {key: False}) for path in no]
            return yes, no, code
        finally:
            self._active_paths = previous_active

    def float_compare_path(self, state, left, right, x, y, op, truth, possible):
        """IEEE comparisons: unordered values belong to false relational branches."""
        if not possible:
            return []
        bits = max((v.domain.bits for v in (x,y) if v.floating and v.literal is None), default=64)
        a = self.float_coerce(left.token, x, bits).domain
        b = self.float_coerce(right.token, y, bits).domain
        numeric_a, numeric_b = a.without_nan(), b.without_nan()
        if op in ("==", "!="):
            equality = truth == (op == "==")
            if equality:
                common = numeric_a & numeric_b
                if a.zeros and b.zeros:
                    common |= FloatDomain(bits, zeros=3)
                ra, rb = common, common
            else:
                ra, rb = a, b
                # Any NaN on the other side can make every value unequal.
                if not b.nan and b.ranges and b.lo == b.hi:
                    remove = b | FloatDomain(bits, zeros=3) if b.zeros else b
                    ra = a - remove
                if not a.nan and a.ranges and a.lo == a.hi:
                    remove = a | FloatDomain(bits, zeros=3) if a.zeros else a
                    rb = b - remove
        else:
            relation = op if truth else {"<":">=", "<=":">", ">":"<=", ">=":"<"}[op]
            if not numeric_a or not numeric_b:
                return [] if truth else [state]
            def upper(value, strict):
                if strict and value == -inf:
                    return FloatDomain(bits)
                return FloatDomain.interval(-inf, next_float(value,bits,-1) if strict else value, bits)
            def lower(value, strict):
                if strict and value == inf:
                    return FloatDomain(bits)
                return FloatDomain.interval(next_float(value,bits,1) if strict else value, inf, bits)
            if relation in ("<", "<="):
                ra = numeric_a & upper(numeric_b.hi, relation == "<")
                rb = numeric_b & lower(numeric_a.lo, relation == "<")
            else:
                ra = numeric_a & lower(numeric_b.lo, relation == ">")
                rb = numeric_b & upper(numeric_a.hi, relation == ">")
            if not truth:
                ra = a if b.nan else ra | FloatDomain(bits,nan=a.nan)
                rb = b if a.nan else rb | FloatDomain(bits,nan=b.nan)
        paths = [state]
        for node, value, narrowed in ((left,x,ra),(right,y,rb)):
            if not narrowed:
                return []
            # Integer-to-float promotions do not change the integer storage domain.
            if not value.floating:
                continue
            if value.domain.bits != bits:
                narrowed = narrowed.convert(value.domain.bits, strategy="round")
            paths = [q for p in paths for q in self.restrict(p,node,narrowed)]
        return paths

    def packed_truth(self, node, state):
        symbol = self.place(node)
        if symbol is None or not isinstance(symbol.ty, Array) or not symbol.ty.packed_bits:
            return None
        symbol = self.comparison_place(symbol, state, propagate_exit=True)
        if symbol.projection is not None and symbol.projection.always_returns:
            return Value(BOOL, symbol.c, 8, True, effectful=True, always_returns=True)
        self.require_live(symbol, state, node.token)
        # A packed array has a positive fixed length. Its contents do not affect
        # truth, but an index in the address can still have side effects.
        return Value(Domain.point(1), f"((void)({symbol.c}), 1)", 8, True, effectful=self.has_effects(node))

    def algebraic_contract(self, callee):
        if callee.kind != "name":
            return None
        name = callee.args[0]
        if name in self.functions:
            contract = self.functions[name][1]
        else:
            binding = self.imported_functions.get(name)
            contract = binding.contract if binding else None
        if contract is not None and (contract.identity_property or contract.idempotent_property
                                     or contract.commutative_property or contract.diagonal_property):
            return contract
        return None

    def algebraic_function_key(self, callee):
        contract = self.algebraic_contract(callee)
        if contract is None:
            return None
        name = callee.args[0]
        c_name = (self.function_c_name(name) if name in self.functions
                  else self.imported_functions[name].c_name)
        return c_name, contract
