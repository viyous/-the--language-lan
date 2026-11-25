"""Array and packed initializers, bit views and composite overlays."""

from __future__ import annotations

import copy
from bisect import bisect_right
from dataclasses import field, fields, replace
from itertools import count
from math import gcd

from array_domain import ArrayDomain
from float_domain import FloatDomain
from value_domain import Domain
from table_interface import c_field

from .lexer import Token
from .syntax import Node
from .typesys import (Array, bit_literal_bytes, bit_literal_value, bit_literal_width, BitField, BitView,
                      BOOL, contains_callable, CopyArray, is_callback_enum, ReadOnly, Record, RecordLink,
                      require_bit_literal_width, Scalar, Slice, storage_alignment, storage_size, Symbol,
                      unqualified, Value, WIDTHS, ZERO)
from .domain_utils import container, zero_domain
from .c_emit import c_counted_loop, indent

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class ViewMixin:
    def _state_fingerprint(self, paths):
        return tuple(
            (path.guard_label(), tuple(sorted(path.values.items())), tuple(sorted(path.bit_values.items())))
            for path in paths
        )

    def states(self, label, paths):
        stack = getattr(self, "_stmt_trace_stack", None)
        if stack is None:
            stack = self._stmt_trace_stack = []
        if label not in ("before", "after"):
            for frame in stack:
                frame[2] = True
        if not paths and label in ("true", "false"):
            self.trace.append(f"  {label} => unreachable")
            return
        fp = None
        if label in ("before", "after"):
            fp = self._state_fingerprint(paths)
            last_fp = getattr(self, "_last_state_fp", None)
            last_end = getattr(self, "_last_state_end", -1)
            if label == "before":
                for frame in stack:
                    frame[2] = True
                stack.append([fp, len(self.trace), False])
                if fp == last_fp and 0 <= len(self.trace) - last_end <= 2:
                    return
            elif label == "after":
                entry_fp, entry_start, nested = stack.pop() if stack else (None, -1, True)
                if paths and fp == entry_fp and not nested and 0 <= len(self.trace) - entry_start <= 12:
                    self._last_state_fp = fp
                    self._last_state_end = len(self.trace)
                    return
        for path in paths:
            guard = path.guard_label()
            entries = sorted(path.values.items())
            omitted = 0
            replay = getattr(self, "optimization_replay", None)
            if replay is not None and replay.branches >= 128 and len(entries) > 32:
                omitted = len(entries) - 32
                entries = entries[:16] + entries[-16:]
            values = ", ".join(f"id#{uid} = {domain}" for uid, domain in entries)
            if omitted:
                values += f", ... {omitted} entities omitted"
            self.trace.append(f"  {label} [{guard}] {{ {values} }}")
            self.trace_bit_field_states(path)
            if label in ("true", "false", "loop head") or label.startswith("loop exit"):
                bounds = [(a, b, c) for (a, b), c in path.zone.bounds.items()
                          if a and b and c < path.values[a].hi - path.values[b].lo]
                for a, b, c in sorted(bounds)[:12]:
                    self.trace.append(f"    difference id#{a} - id#{b} <= {c}")
        if paths:
            self._last_state_fp = self._state_fingerprint(paths) if fp is None else fp
            self._last_state_end = len(self.trace)

    def array_init(self, sym, node, paths, t, bit_facts=None, allocation=None, *, continuing_paths=None):
        ty = sym.ty
        if node is None:
            self.range_check(t, Value(zero_domain(ty.element), "0", boolean=ty.element.boolean, enum=ty.element.enum, character=ty.element.character), ty.element)
            if not ty.packed_bits:
                for path in paths:
                    path.absorb(path.with_array(sym, ArrayDomain.zero(ty.length, ty.element.domain)))
            return [allocation or f"{ty.element.c} {sym.c}[{ty.c_length}] = {{0}};"]
        if ty.encoding is not None and node.kind == "text_literal":
            raw = ty.encoding.encode_fixed(node.args[0].decode("utf-8"), ty.length, node.token)
            domain = ArrayDomain.from_values(ty.length, ty.element.domain, [Domain.point(b) for b in raw])
            for path in paths:
                path.absorb(path.with_array(sym, domain))
            bytes_init = "{" + ", ".join(f"0x{b:02x}" for b in raw) + "}"
            return [allocation or f"{ty.element.c} {sym.c}[{ty.c_length}] = {bytes_init};"]
        if node.kind not in ("list", "repeat"):
            node.token.error("expected a list or repeat initializer for a fixed array")
        if node.kind == "list":
            elements, = node.args
            length = len(elements)
        else:
            value, length_node = node.args
            length = self.const_int(length_node, ()) if isinstance(length_node, Node) else length_node
            elements = (value,)
        if length != ty.length:
            node.token.error(f"array initializer has {length} elements, expected {ty.length}")
        exit_code = None
        continuing_facts = []
        for p, path in enumerate(paths):
            values, evaluated = [], []
            known, data = bit_facts[p] if bit_facts is not None else (0, 0)
            for index, element in enumerate(elements):
                value = self.expr(element, path)
                if continuing_paths is not None and value.always_returns:
                    candidate = "{ " + " ".join(f"(void)({code});" for code in (*evaluated, value.code)) + " }"
                    if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                        exit_code = candidate
                    break
                if ty.packed_bits:
                    self.require_bit(element.token, value, element)
                    if bit_facts is not None and value.domain.lo == value.domain.hi:
                        mask = (1 << ty.length) - 1 if node.kind == "repeat" else 1 << index
                        known, data = known | mask, data | (mask if value.domain.lo else 0)
                else:
                    value = self.coerce_storage(element.token, value, ty.element)
                    self.range_check(element.token, value, ty.element)
                    values.append(value.domain)
                evaluated.append(value.code)
            else:
                codes = evaluated
                if continuing_paths is not None:
                    continuing_paths.append(path)
                if bit_facts is not None:
                    continuing_facts.append((known, data))
                if not ty.packed_bits:
                    content = (ArrayDomain.filled(ty.length, ty.element.domain, values[0])
                               if node.kind == "repeat" else ArrayDomain.from_values(ty.length, ty.element.domain, values))
                    path.absorb(path.with_array(sym, content))
        if continuing_paths is not None and not continuing_paths:
            return [exit_code, "__builtin_unreachable();"]
        if bit_facts is not None:
            bit_facts[:] = continuing_facts
        if node.kind == "repeat":
            initializer = f"(UINT8_MAX * !!({codes[0]}))" if ty.packed_bits else codes[0]
            return [
                f"const {ty.element.c} lr_{sym.uid} = {initializer};",
                allocation or f"{ty.element.c} {sym.c}[{ty.c_length}];",
                *c_counted_loop(f"li_{sym.uid}", 0, ty.c_length,
                                [f"{sym.c}[li_{sym.uid}] = lr_{sym.uid};"]),
            ]
        declarations = []
        if any(self.has_effects(element) for element in elements):
            for i, code in enumerate(codes):
                temp = f"lan_init_{sym.uid}_{i}"
                declarations.append(f"{ty.element.c} {temp} = {code};")
                codes[i] = temp
        if ty.packed_bits:
            codes = [" | ".join(f"((unsigned)({code}) << {bit})"
                                for bit, code in enumerate(codes[i:i+8]))
                     for i in range(0, len(codes), 8)]
        initializer = f"{{{', '.join(codes)}}}"
        if allocation is not None:
            return [*declarations, allocation,
                    f"memcpy({sym.c}, (const {ty.element.c}[]) {initializer}, {ty.c_length});"]
        return [*declarations, f"{ty.element.c} {sym.c}[{ty.c_length}] = {initializer};"]

    def packed_init(self, sym, node, paths, t, declare_storage=True, *, allocation=None):
        if sym.readonly:
            t.error("cannot modify an immutable bit array")
        for state in paths:
            self.require_live(sym, state, t)
        zero = f"uint8_t {sym.c}[{sym.ty.c_length}] = {{0}};"
        prefix = [zero] if declare_storage else [allocation] if allocation else []
        literal = bit_literal_value(node) if node is not None else None
        if literal is not None:
            value, width = literal
            require_bit_literal_width(node.token, sym.ty.length, width)
            if sym.checked_overlay:
                node.token.error("assign a bit literal to a checked overlay through a bit slice")
            if declare_storage or allocation is not None:
                if width <= 4096:
                    paths = [path.with_bits(sym, ((1 << width) - 1, value)) for path in paths]
            else:
                paths = [self.update_bits(path, sym, 0, width, Domain.point(value)) for path in paths]
            if declare_storage:
                return paths, [f"uint8_t {sym.c}[{sym.ty.c_length}] = {bit_literal_bytes(value, width)};"]
            return paths, [*([allocation] if allocation else []), self.write_bit_literal(sym, 0, value, width)]
        if node is not None and (width := bit_literal_width(node)) is not None:
            require_bit_literal_width(node.token, sym.ty.length, width)
            changed, code, early = [], None, None
            for path in paths:
                value = self.expr(node, path)
                if value.always_returns:
                    candidate = f"(void)({value.code});"
                    if early is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(early):
                        early = candidate
                    continue
                if declare_storage or allocation is not None:
                    known = self.domain_bit_facts(value.domain, width)
                    changed.append(path.with_bits(sym, known))
                else:
                    changed.append(self.update_bits(path, sym, 0, width, value.domain))
                code = self.write_bits(sym, 0, width, value, "le")
            if not changed:
                return [], [early, "__builtin_unreachable();"]
            return changed, [*prefix, code]
        if node is None or node.kind in ("list", "repeat"):
            facts = None
            if sym.ty.length <= 4096:
                initial_mask = (1 << sym.ty.length) - 1 if node is None else 0
                facts = [(initial_mask, 0)] * len(paths)
            code = self.array_init(sym, node, paths, t, facts)
            if facts is not None:
                paths = [p.with_bits(sym, known) for p, known in zip(paths, facts)]
            return paths, code
        if declare_storage and sym.ty.length <= 4096:
            paths = [p.with_bits(sym, ((1 << sym.ty.length) - 1, 0)) for p in paths]
        if node.kind == "slice":
            if self.dynamic_bit_slice(node):
                result = []
                exit_code = None
                independent = not self.has_effects(node, allow_debug_checks=True)
                for state in paths:
                    region = self.bit_region(node, state, capture=True, propagate_exit=True)
                    if isinstance(region, Value):
                        candidate = f"(void)({region.code});"
                        if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                            exit_code = candidate
                        continue
                    src, start, width, setup = region
                    independent = independent and self.independent_bit_storage(sym, src, state)
                    if width.domain.hi > sym.ty.length:
                        node.token.error("bit slice does not fit the declared bit array")
                    if start.domain.lo == start.domain.hi and width.domain.lo == width.domain.hi:
                        updated = self.copied_bit_state(state, sym, 0, src, start.domain.lo, width.domain.lo)
                    else:
                        updated = self.update_bits(state, sym, 0, width.domain.hi, BOOL)
                    result.append(updated)
                if not result:
                    return [], [exit_code, "__builtin_unreachable();"]
                load, store = self.helper("bit_load", None), self.helper("bit_store", None)
                copy = self.helper("bit_copy", independent or None, (load, store))
                if independent:
                    self.trace.append("  initialize from dynamic bit slice with disjoint byte storage: restrict")
                args = [sym.c, "0", src.c, f"(uint64_t)({start.code})", f"(uint64_t)({width.code})"]
                code = self.ordered_call(node.token, copy,
                                         ["uint8_t *", "uint64_t", "const uint8_t *", "uint64_t", "uint64_t"], args)
                if allocation:
                    return result, [*setup, allocation, code + ";"]
                return result, [*prefix, "{", *indent([*setup, code + ";"]), "}"]
            src, start, width = self.bit_span(node)
            for state in paths:
                self.require_live(src, state, node.token)
            if width > sym.ty.length:
                node.token.error("bit slice does not fit the declared bit array")
            paths = [self.copied_bit_state(p, sym, 0, src, start, width) for p in paths]
            independent = all(self.independent_bit_storage(sym, src, state) for state in paths)
            return paths, [*prefix, self.copy_bits(sym, 0, src, start, width, disjoint=independent)]
        if node.kind != "cast" or node.args[1] != "view":
            node.token.error("packed array initialization requires a list, slice, or explicit view")
        operand, _, annotation = node.args
        ty = self.resolve(annotation)
        if ty != sym.ty:
            annotation.token.error("view target must match the declared packed bit array")
        src = self.place(operand)
        if src is not None:
            self.require_live(src, paths[0], operand.token)
            if isinstance(src.ty, Array):
                for state in paths:
                    self.require_live(src, state, operand.token)
                if src.ty != ty:
                    operand.token.error("packed array view requires equal bit widths")
                paths = [self.copied_bit_state(p, sym, 0, src, 0, ty.length) for p in paths]
                independent = all(self.independent_bit_storage(sym, src, state) for state in paths)
                return paths, [*prefix, self.copy_bits(sym, 0, src, 0, ty.length, disjoint=independent)]
        if ty.length not in WIDTHS:
            node.token.error("integer view requires a packed bit array of 8, 16, 32, or 64 bits")
        result = []
        exit_code = None
        for path in paths:
            value = self.expr(operand, path)
            if value.always_returns:
                candidate = f"(void)({value.code});"
                if not declare_storage and not allocation:
                    candidate = f"{{ (void)({sym.c}); {candidate} }}"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            if value.floating:
                value = self.float_cast(node.token, value, "view", Scalar(container(ty.length, False), ty.length))
            self.check_integer_view(node.token, value, ty.length)
            result.append(self.update_bits(path, sym, 0, ty.length, value.domain, "native", retain_domain=True))
            written_value = value
        if not result:
            return [], [exit_code, "__builtin_unreachable();"]
        if allocation:
            saved = f"lan_init_bits_{sym.uid}"
            prefix = [f"uint64_t {saved} = (uint64_t)({written_value.code});", allocation]
            written_value = replace(written_value, code=saved)
        return result, [*prefix, self.write_bits(sym, 0, ty.length, written_value, "native")]

    @staticmethod
    def bit_patterns_writable(ty):
        if isinstance(ty, ReadOnly):
            return False
        if isinstance(ty, Scalar):
            return (not (ty.floating or ty.boolean or ty.enum or ty.character)
                    and ty.domain == container(ty.bits, ty.signed))
        if isinstance(ty, Array):
            return ty.packed_bits or Compiler.bit_patterns_writable(ty.element)
        if isinstance(ty, Record):
            return all(Compiler.bit_patterns_writable(field) for _, field in ty.fields)
        return False

    @staticmethod
    def scalar_bit_component(ty):
        if isinstance(ty, Record) and len(ty.fields) == 1:
            ty = unqualified(ty.fields[0][1])
        elif isinstance(ty, Array) and not ty.packed_bits and ty.length == 1:
            ty = ty.element
        if isinstance(ty, Scalar) and not (ty.boolean or ty.enum or ty.character):
            return ty
        return None

    def scalar_component_value(self, source, state):
        component = self.scalar_bit_component(source.ty)
        if component is None:
            return None
        if isinstance(source.ty, Record):
            name = source.ty.fields[0][0]
            return self.storage_value(self.child_place(source, (name,)), state)
        if isinstance(source.ty, Array):
            domain = self.storage_array(state, source).read(0)
            return Value(domain, f"({source.c})[0]", component.bits, signed=component.signed)
        return self.storage_value(source, state)

    def packed_scalar_view(self, token, state, sym, target):
        if is_callback_enum(target.enum):
            token.error("function values cannot be reinterpreted as bits")
        if target.boolean or target.bits != sym.ty.length:
            token.error("view requires a packed array and integer of the same fixed width")
        effectful = False
        if sym.projection is not None:
            effectful = self.debug_bounds or any(
                isinstance(step, Node) and self.has_effects(step) for step in sym.projection.steps)
            sym = self.comparison_place(sym, state, propagate_exit=True)
            if sym.projection.always_returns:
                return Value(ZERO, sym.c, effectful=True, always_returns=True)
        value = self.read_bits(token, state, sym, 0, target.bits, target, "native")
        return replace(value, effectful=True) if effectful else value

    def inline_packed_view(self, token, paths, operand, packed, *, capture_exit=None):
        source = self.place(operand)
        if source is None and operand.kind == "cast" and operand.args[1] == "view":
            inner, _, annotation = operand.args
            intermediate = self.resolve(annotation)
            if isinstance(intermediate, Record) or isinstance(intermediate, Array) and not intermediate.packed_bits:
                source, paths = self.composite_view_temporary(intermediate, inner, paths, operand.token)
        if source is None:
            source, paths = self.composite_call_view_source(operand, paths)
        if source is not None and isinstance(source.ty, Array) and source.ty.packed_bits:
            if source.ty != packed:
                token.error("packed array view requires equal bit widths")
            return source, paths
        name = Token(f"lan_inline_bits_{len(self.symbols) + 1}", token.line, token.col)
        self.scope.append({})
        try:
            if source is not None and isinstance(source.ty, (Scalar, Array, Record)):
                paths, code = self.define_bit_overlay(name, packed, source, paths, token, inline=True,
                                                      propagate_exit=capture_exit is not None)
            elif operand.kind == "index" and self.span_kind(operand.args[0]) == "ordinary":
                paths, code = self.define_element_bit_overlay(name, packed, operand, paths, inline=True,
                                                              propagate_exit=capture_exit is not None)
            elif source is None:
                source, paths = self.scalar_expression_view_source(operand, paths, capture_exit=capture_exit)
                if source is None:
                    return None, paths
                paths, code = self.define_bit_overlay(name, packed, source, paths, token, inline=True)
            else:
                token.error("inline packed bit view requires scalar or fixed composite storage")
            if not paths and capture_exit is not None:
                exit_expr = code[0].rstrip(";")
                capture_exit.append(Value(ZERO, f"({exit_expr}, 0)",
                                          effectful=True, always_returns=True))
                return None, paths
            return self.scope[-1][name.text], paths
        finally:
            self.scope.pop()

    def inline_packed_scalar_view(self, token, state, operand, packed, target):
        early = []
        source, paths = self.inline_packed_view(token, [state], operand, packed, capture_exit=early)
        if early:
            return early[0]
        view_state, = paths
        if source.projection is not None:
            source = self.capture_projected_argument(source, view_state, token, allow_effectful_links=True)
        return self.packed_scalar_view(token, view_state, source, target)

    def view_composite_scalar(self, token, state, source, target):
        if contains_callable(source.ty) or is_callback_enum(target.enum):
            token.error("function values cannot be reinterpreted as bits")
        if self.contains_interface_storage(source.ty):
            token.error("records with interface fields cannot be reinterpreted as bits")
        if target.boolean or target.enum or target.character:
            token.error("composite bit view requires a plain integer or floating target")
        if storage_size(source.ty) * 8 != target.bits:
            token.error("composite bit view requires equal physical widths")
        source = (self.capture_projected_argument(source, state, token, allow_effectful_links=True,
                                                  propagate_exit=True)
                  if source.projection is not None
                  else self.comparison_place(source, state))
        if source.projection is not None and source.projection.always_returns:
            return Value(ZERO, source.c, effectful=True, always_returns=True)
        self.require_live(source, state, token)
        scalar = self.scalar_component_value(source, state)
        if scalar is not None:
            value = self.cast(token, scalar, "view", target)
        else:
            domain = FloatDomain.top(target.bits) if target.floating else container(target.bits, target.signed)
            helper = self.helper("object_scalar_view", target.c)
            address = source.c if isinstance(source.ty, Array) else f"&({source.c})"
            value = Value(domain, f"{helper}({address})", target.bits, signed=target.signed)
            self.range_check(token, value, target)
        self.helper("static_assert", None,
                    f'_Static_assert(sizeof({source.ty.c}) == sizeof({target.c}), "Lan composite bit view width");')
        return value

    def array_view_compatible(self, source, target):
        return (isinstance(source, Array) and isinstance(target, Array)
                and not source.packed_bits and not target.packed_bits and source.length == target.length
                and self.view_leaf_compatible(source.element, target.element))

    def view_leaf_compatible(self, source, target):
        if isinstance(source, Scalar) and isinstance(target, Scalar):
            return (self.scalar_bit_component(source) is not None
                    and self.scalar_bit_component(target) is not None and source.bits == target.bits)
        if isinstance(source, Record) and isinstance(target, Record):
            return self.record_view_fields(source, target) is not None
        if isinstance(source, Array) and isinstance(target, Array) and target.packed_bits:
            return source.packed_bits and source.length == target.length
        return self.array_view_compatible(source, target)

    def record_view_fields(self, source, target, old_path=(), new_path=()):
        if not isinstance(source, Record) or not isinstance(target, Record) or len(source.layout) != len(target.layout):
            return None
        fields = []
        for (old_name, old_type), (new_name, new_type) in zip(source.layout, target.layout):
            old_type, new_type = unqualified(old_type), unqualified(new_type)
            if storage_size(old_type) != storage_size(new_type) or storage_alignment(old_type) != storage_alignment(new_type):
                return None
            old = old_path + (old_name,)
            new = new_path + (new_name,)
            if isinstance(old_type, Record) and isinstance(new_type, Record):
                nested = self.record_view_fields(old_type, new_type, old, new)
                if nested is None:
                    return None
                fields.extend(nested)
            elif self.view_leaf_compatible(old_type, new_type):
                fields.append((old, new, new_type))
            else:
                return None
        return fields

    def view_contents(self, token, source, target, contents):
        from composite_array import RecordValue, TreeArray

        self.spend(token)
        if isinstance(target, Scalar):
            value = Value(contents, "0", source.bits, signed=source.signed)
            return self.cast(token, value, "view", target).domain
        if isinstance(target, Record):
            old_fields = dict(contents.fields)
            new_fields = {}
            for (old_name, old_type), (new_name, new_type) in zip(source.layout, target.layout):
                new_fields[new_name] = self.view_contents(
                    token, unqualified(old_type), unqualified(new_type), old_fields[old_name])
            return RecordValue(tuple((name, new_fields[name]) for name, _ in target.fields))
        if target.packed_bits:
            return contents
        segments = tuple(replace(segment, domain=self.view_contents(
            token, source.element, target.element, segment.domain)) for segment in contents.segments)
        if isinstance(target.element, Scalar):
            return ArrayDomain(target.length, target.element.domain, segments)
        return TreeArray(target.length, self.graph_array_contents(target.element, {}), segments)

    def view_layout_assertions(self, source, target, fields):
        if isinstance(target, Array):
            source, target = source.element, target.element
            if isinstance(target, Array):
                yield from self.view_layout_assertions(source, target, None)
                return
            fields = self.record_view_fields(source, target)
        if not fields:
            return

        def member(names):
            return ".".join(c_field(name) for name in names)

        def field_type(record, names):
            for name in names:
                record = unqualified(dict(record.fields)[name])
            return record

        for old, new, ty in fields:
            yield (f'_Static_assert(offsetof({source.c}, {member(old)}) == '
                   f'offsetof({target.c}, {member(new)}), "Lan composite bit view field offset");')
            if isinstance(ty, Array):
                yield from self.view_layout_assertions(field_type(source, old), ty, None)

    def composite_view_temporary(self, target, operand, paths, token):
        source, address, domains = self.composite_view_source(target, operand, paths, token)
        self.helper("static_assert", None,
                    f'_Static_assert(sizeof({target.c}) == sizeof({source.c}), "Lan composite bit view width");')
        copy = self.helper("disjoint_object_copy", None)
        pointer = f"{copy}(&({target.c}){{0}}, {address}, sizeof({target.c}))"
        sym = Symbol("view_temporary", len(self.symbols) + 1, target, len(self.scope),
                     c_place=f"(*(({target.c} *){pointer}))", readonly=True)
        self.symbols.append(sym)
        self.record_symbols(sym, token, zero=False)
        return sym, self.composite_view_states(sym, domains, paths)

    def composite_view_leaf_domains(self, target, contents):
        def leaves(ty, value, path):
            if isinstance(ty, Record):
                fields = dict(value.fields)
                for name, field in ty.fields:
                    yield from leaves(unqualified(field), fields[name], path + (name,))
            else:
                yield path, value

        return dict(leaves(target, contents, ())) if isinstance(target, Record) else contents

    def composite_source_bits(self, token, state, source, assertions):
        def segment_at(value, index):
            return value.segments[bisect_right(value.segments, index, key=lambda item: item.start) - 1]

        def record_parts(ty, value):
            position = 0
            fields = dict(value.fields)
            for name, field in ty.layout:
                alignment = storage_alignment(field)
                position = (position + alignment - 1) // alignment * alignment
                size = storage_size(field)
                yield name, position, size, field, fields[name]
                position += size

        def read(ty, value, first, width):
            self.spend(token)
            ty = unqualified(ty)
            last = first + width
            if isinstance(ty, Scalar):
                if ty.floating:
                    scalar = Value(value, "0", ty.bits)
                    raw = self.float_cast(token, scalar, "view", self.types[f"uint{ty.bits}"]).domain
                else:
                    raw = self.bit_raw_domain(value, ty.bits)
                mask, word = self.domain_bit_facts(raw, ty.bits)
                window = (1 << width) - 1
                return tuple(((self.bit_order(mask, ty.bits, order) >> first) & window,
                              (self.bit_order(word, ty.bits, order) >> first) & window)
                             for order in ("le", "be"))
            if isinstance(ty, Array) and ty.packed_bits:
                mask = word = 0
                for segment in value.slice(first, last).segments:
                    if segment.domain.lo == segment.domain.hi:
                        part = ((1 << (segment.stop - segment.start)) - 1) << segment.start
                        mask |= part
                        if segment.domain.lo:
                            word |= part
                return ((mask, word), (mask, word))
            if isinstance(ty, Record):
                parts = []
                for name, position, size, field, content in record_parts(ty, value):
                    if position * 8 < last and (position + size) * 8 > first:
                        assertions.add(f'_Static_assert(offsetof({ty.c}, {c_field(name)}) == {position}, '
                                       '"Lan source view field offset");')
                        parts.append((position * 8, size * 8, field, content))
            elif isinstance(ty, Array):
                size = storage_size(ty.element)
                stride = size * 8
                assertions.add(f'_Static_assert(sizeof({ty.element.c}) == {size}, "Lan source view array stride");')
                parts = ((i * stride, stride, ty.element, segment_at(value, i).domain)
                         for i in range(first // stride, (last - 1) // stride + 1))
            else:
                return ((0, 0), (0, 0))
            masks, words = [0, 0], [0, 0]
            for start, size, field, contents in parts:
                lo, hi = max(first, start), min(last, start + size)
                for order, (mask, word) in enumerate(read(field, contents, lo - start, hi - lo)):
                    masks[order] |= mask << (lo - first)
                    words[order] |= word << (lo - first)
            return tuple(zip(masks, words))

        def repeated_windows(ty, value, first, width):
            self.spend(token)
            ty = unqualified(ty)
            if isinstance(ty, Record):
                for _, position, size, field, content in record_parts(ty, value):
                    if position * 8 <= first and first + width <= (position + size) * 8:
                        return repeated_windows(field, content, first - position * 8, width)
            if isinstance(ty, Array):
                stride = 1 if ty.packed_bits else storage_size(ty.element) * 8
                index, phase = divmod(first, stride)
                segment = segment_at(value, index)
                count = (segment.stop * stride - first) // width
                period = stride // gcd(stride, width)
                if count > period:
                    return count, period
                if not ty.packed_bits and phase + width <= stride:
                    return repeated_windows(ty.element, segment.domain, phase, width)
            return 1, 1

        contents = self.record_value(source, state)
        return (lambda offset, width: read(source.ty, contents, offset, width),
                lambda offset, width: repeated_windows(source.ty, contents, offset, width))

    def packed_composite_contents(self, token, state, source, target, assertions, offset=0, *, source_bits=None):
        from array_domain import Segment
        from composite_array import RecordValue, TreeArray, join_values

        self.spend(token)
        target = unqualified(target)
        read_source, repeat_source = source_bits if source_bits is not None else (None, None)
        if isinstance(target, Scalar):
            signed = target.signed and not target.floating
            if read_source is None:
                domain = self.bit_read_domain(state, source, offset, target.bits, "native", signed)
            else:
                width = target.bits
                window = (1 << width) - 1
                domains = []
                for order, (mask, data) in zip(("le", "be"), read_source(offset, width)):
                    known = self.bit_order(mask, width, order)
                    word = self.bit_order(data, width, order)
                    domains.append(self.masked_bit_domain(known, word, width))
                domain = Domain.union(domains)
                if signed:
                    positive = domain & container(width - 1, False)
                    negative = (domain & Domain.of((1 << (width - 1), window))).affine(1, -(1 << width))
                    domain = positive | negative

            if target.floating:
                return self.float_cast(token, Value(domain, "0", target.bits), "view", target).domain
            value = Value(domain, "0", target.bits, target.boolean, enum=target.enum,
                          character=target.character, signed=target.signed)
            self.range_check(token, value, target)
            return domain
        if isinstance(target, Record):
            fields = {}
            position = 0
            for name, field in target.layout:
                alignment = storage_alignment(field)
                position = (position + alignment - 1) // alignment * alignment
                assertions.add(f'_Static_assert(offsetof({target.c}, {c_field(name)}) == {position}, '
                               '"Lan packed view field offset");')
                fields[name] = self.packed_composite_contents(
                    token, state, source, field, assertions, offset + position * 8, source_bits=source_bits)
                position += storage_size(field)
            return RecordValue(tuple((name, fields[name]) for name, _ in target.fields))
        if not isinstance(target, Array):
            token.error("packed bit view cannot initialize references or borrowed storage")
        if target.packed_bits:
            if target.length > 4096:
                return TreeArray.unknown(target.length, BOOL)
            self.spend(token, target.length)
            if read_source is None:
                mask, data = self.storage_bits(state, source)
                mask, data = mask >> offset, data >> offset
            else:
                (mask, data), (other_mask, other_data) = read_source(offset, target.length)
                mask &= other_mask & ~(data ^ other_data)
            return TreeArray.from_values(target.length, BOOL, (
                Domain.point((data >> bit) & 1) if (mask >> bit) & 1 else BOOL
                for bit in range(target.length)))
        if storage_size(target) > 4096 and self.bit_patterns_writable(target):
            return self.graph_array_contents(target, {})
        stride = storage_size(target.element)
        assertions.add(f'_Static_assert(sizeof({target.element.c}) == {stride}, "Lan packed view array stride");')
        segments = []
        index = 0
        while index < target.length:
            start = offset + index * stride * 8
            count, period = (1, 1) if repeat_source is None else repeat_source(start, stride * 8)
            end = min(target.length, index + count)
            phases = tuple(self.packed_composite_contents(
                token, state, source, target.element, assertions, start + phase * stride * 8, source_bits=source_bits)
                for phase in range(min(period, end - index)))
            if all(value == phases[0] for value in phases):
                segments.append(Segment(index, end, phases[0]))
            elif end - index > self.gas:
                segments.append(Segment(index, end, join_values(phases)))
            else:
                self.spend(token, end - index)
                segments.extend(Segment(i, i + 1, phases[(i - index) % period]) for i in range(index, end))
            index = end
        if isinstance(target.element, Scalar):
            return ArrayDomain(target.length, target.element.domain, tuple(segments))
        return TreeArray(target.length, self.graph_array_contents(target.element, {}), tuple(segments))

    def scalar_expression_view_source(self, operand, paths, *, capture_exit=None):
        values = [self.expr(operand, path) for path in paths]
        if capture_exit is not None:
            continuing = [(path, value) for path, value in zip(paths, values) if not value.always_returns]
            if not continuing:
                capture_exit.append(max(values, key=lambda value: self.propagation_code_rank(value.code)))
                return None, []
            paths, values = zip(*continuing)
        first = values[0]
        width = first.width
        if width not in WIDTHS or any(value.width != width or value.floating != first.floating for value in values):
            operand.token.error("bit view expression requires a consistent fixed physical width")
        if first.floating:
            ty = Scalar(FloatDomain.top(width), width, floating=True)
            domains = [value.domain for value in values]
        else:
            for value in values:
                self.check_integer_view(operand.token, value, width)
            ty = Scalar(container(width, False), width)
            domains = [self.bit_raw_domain(value.domain, width) for value in values]
        place = f"(({ty.c}){{({ty.c})({first.code})}})"
        source = Symbol("view_scalar", len(self.symbols) + 1, ty, len(self.scope), c_place=place, readonly=True)
        self.symbols.append(source)
        return source, [path.update(source, domain) for path, domain in zip(paths, domains)]

    def composite_call_view_source(self, operand, paths):
        result = self.composite_call_type(operand)
        if not isinstance(result, (Record, CopyArray)):
            return None, paths
        calls = [self.call(operand, path)[1] for path in paths]
        ty = result.array if isinstance(result, CopyArray) else result
        place = f"(({result.c}[]){{{calls[0]}}})[0]"
        if isinstance(result, CopyArray):
            place += ".data"
        source = Symbol("view_call", len(self.symbols) + 1, ty, len(self.scope), c_place=place, readonly=True)
        self.symbols.append(source)
        if isinstance(result, Record):
            self.record_symbols(source, operand.token, zero=False)
            fields = self.record_call_fields(operand)
            return source, [self.restore_record_call_fields(source, path, fields) for path in paths]
        contents = self.array_call_contents(operand)
        if contents is not None:
            return source, [self.restore_record_array(source, path, contents) for path in paths]
        if ty.packed_bits:
            return source, paths
        return source, [path.with_array(source, ArrayDomain.unknown(ty.length, ty.element.domain)) for path in paths]

    def composite_view_source(self, target, operand, paths, token, *, captured_places=None, capture_exit=None):
        if contains_callable(target):
            token.error("function values cannot be reinterpreted as bits")
        unrestricted = self.bit_patterns_writable(target)
        component = self.scalar_bit_component(target)
        source = self.place(operand)
        if source is None and operand.kind == "cast" and operand.args[1] == "view":
            inner, _, annotation = operand.args
            intermediate = self.resolve(annotation)
            if isinstance(intermediate, Array) and intermediate.packed_bits:
                source, paths = self.inline_packed_view(operand.token, paths, inner, intermediate)
            elif isinstance(intermediate, (Record, Array)):
                source, paths = self.composite_view_temporary(intermediate, inner, paths, operand.token)
        if source is None:
            source, paths = self.composite_call_view_source(operand, paths)
        if source is None:
            source, paths = self.scalar_expression_view_source(operand, paths)
        if not isinstance(source.ty, (Array, Record, Scalar)):
            token.error("composite bit view requires addressable storage")
        if contains_callable(source.ty):
            token.error("function values cannot be reinterpreted as bits")
        if self.contains_interface_storage(source.ty) or self.contains_interface_storage(target):
            token.error("records with interface fields cannot be reinterpreted as bits")
        if storage_size(source.ty) != storage_size(target):
            token.error("composite bit view requires equal physical widths")
        arrays = self.array_view_compatible(source.ty, target)
        fields = self.record_view_fields(source.ty, target)
        packed = isinstance(source.ty, Array) and source.ty.packed_bits
        can_repack = not self.contains_record_links(source.ty)
        if not unrestricted and component is None and not (arrays or fields or packed or can_repack):
            token.error("composite view target must accept every physical bit pattern or have a provable layout")
        addresses, domains, packed_assertions = [], [], set()
        for path in paths:
            place = (self.capture_projected_argument(source, path, token, allow_effectful_links=True,
                                                     propagate_exit=capture_exit is not None)
                     if source.projection is not None
                     else self.comparison_place(source, path))
            if place.projection is not None and place.projection.always_returns:
                capture_exit.append((path, Value(ZERO, place.c, effectful=True, always_returns=True)))
                continue
            self.require_live(place, path, token)
            if captured_places is not None:
                captured_places.extend(self.physical_places(place, path))
            addresses.append(place.c if isinstance(place.ty, Array) else f"&({place.c})")
            scalar = self.scalar_component_value(place, path) if component is not None else None
            if component is not None and isinstance(place.ty, Array) and place.ty.packed_bits:
                domains.append(self.packed_scalar_view(token, path, place, component).domain)
            elif scalar is not None:
                domains.append(self.cast(token, scalar, "view", component).domain)
            elif arrays:
                domains.append(self.view_contents(token, place.ty, target, self.storage_array(path, place)))
            elif fields:
                field_domains = {}
                for old, new, new_type in fields:
                    self.spend(token)
                    field = self.child_place(place, old)
                    if isinstance(new_type, Array):
                        field_domains[new] = self.view_contents(token, field.ty, new_type, self.record_value(field, path))
                    else:
                        value = self.storage_value(field, path)
                        field_domains[new] = self.cast(token, value, "view", new_type).domain
                domains.append(field_domains)
            elif not unrestricted and (packed or can_repack):
                bits = None if packed else self.composite_source_bits(token, path, place, packed_assertions)
                contents = self.packed_composite_contents(token, path, place, target, packed_assertions, source_bits=bits)
                domains.append(self.composite_view_leaf_domains(target, contents))
            elif unrestricted:
                domains.append(None)
            else:
                token.error("cannot prove the physical bit pattern fits the composite target")
        if not addresses:
            return None, None, []
        address = addresses[0]
        if arrays or fields:
            for item in dict.fromkeys(self.view_layout_assertions(source.ty, target, fields)):
                self.helper("static_assert", None, item)
        if packed_assertions:
            for item in sorted(packed_assertions):
                self.helper("static_assert", None, item)
        return source.ty, address, domains

    def composite_view_states(self, sym, domains, paths):
        from array_domain import ArrayDomain as ArrayContents

        leaves = tuple(self.named_record_leaves(sym))
        result = []
        for path, domain in zip(paths, domains):
            for names, leaf in leaves:
                actual = domain.get(names) if isinstance(domain, dict) else domain
                if isinstance(leaf.ty, Scalar):
                    path = path.update(leaf, actual if actual is not None else leaf.ty.domain)
                elif leaf.ty.packed_bits:
                    if actual is not None:
                        path = self.restore_record_array(leaf, path, actual)
                else:
                    array = leaf.ty
                    if isinstance(actual, ArrayContents):
                        contents = actual
                    elif actual is not None:
                        contents = ArrayDomain.filled(array.length, array.element.domain, actual)
                    else:
                        contents = (ArrayDomain.unknown(array.length, array.element.domain)
                                    if isinstance(array.element, Scalar) else self.graph_array_contents(array, {}))
                    path = path.with_array(leaf, contents)
            result.append(path)
        return result

    def define_composite_view(self, name, target, operand, paths, readonly, token):
        early = []
        source, address, domains = self.composite_view_source(target, operand, paths, token, capture_exit=early)
        if early:
            if source is None:
                value = max((value for _, value in early), key=lambda value: self.propagation_code_rank(value.code))
                return [], [f"(void)({value.code});", "__builtin_unreachable();"]
            skipped = {id(path) for path, _ in early}
            paths = [path for path in paths if id(path) not in skipped]
        sym = self.declare(name, target, readonly=readonly)
        if isinstance(target, Record):
            self.record_symbols(sym, name, zero=False)
            declaration = f"{target.c} {sym.c};"
            destination = f"&{sym.c}"
        else:
            declaration = f"{target.element.c} {sym.c}[{target.c_length}];"
            destination = sym.c
        result = self.composite_view_states(sym, domains, paths)
        self.trace.append(f"  define entity id#{sym.uid} {name.text}: composite bit view value")
        copy = self.helper("disjoint_object_copy", None)
        return result, [declaration,
                        f'_Static_assert(sizeof({sym.c}) == sizeof({source.c}), "Lan composite bit view width");',
                        f"{copy}({destination}, {address}, sizeof({sym.c}));"]

    def checked_overlay_fields(self, ty, path=(), offset=0):
        ty = unqualified(ty)
        if isinstance(ty, (RecordLink, Slice)):
            return [(path, offset)]
        if isinstance(ty, Scalar):
            return [(path, offset)]
        if isinstance(ty, Array):
            if ty.packed_bits or self.checked_overlay_fields(ty.element) is not None:
                return [(path, offset)]
            return None
        if not isinstance(ty, Record):
            return None
        fields = []
        position = 0
        for name, field in ty.layout:
            alignment = storage_alignment(field)
            position = (position + alignment - 1) // alignment * alignment
            nested = self.checked_overlay_fields(field, path + (name,), offset + position)
            if nested is None:
                return None
            fields.extend(nested)
            position += storage_size(field)
        return fields

    def checked_overlay_layout_assertions(self, ty):
        ty = unqualified(ty)
        if isinstance(ty, Array) and not ty.packed_bits:
            yield (f'_Static_assert(sizeof({ty.element.c}) == {storage_size(ty.element)}, '
                   '"Lan checked bit view array stride");')
            yield from self.checked_overlay_layout_assertions(ty.element)
        elif isinstance(ty, Record):
            for names, offset in self.checked_overlay_fields(ty):
                member = ".".join(c_field(field) for field in names)
                yield (f'_Static_assert(offsetof({ty.c}, {member}) == {offset}, '
                       '"Lan checked bit view field offset");')
                field_type = ty
                for name in names:
                    field_type = unqualified(dict(field_type.fields)[name])
                if isinstance(field_type, (RecordLink, Slice)):
                    yield (f'_Static_assert(sizeof((({ty.c} *)0)->{member}) == {storage_size(field_type)}, '
                           '"Lan checked bit view protected field size");')
                if isinstance(field_type, Array):
                    yield from self.checked_overlay_layout_assertions(field_type)

    def define_bit_overlay(self, name, ty, source, paths, token, *, inline=False, propagate_exit=False):
        if contains_callable(source.ty):
            token.error("function values cannot be reinterpreted as bits")
        if storage_size(source.ty) * 8 != ty.length:
            token.error("whole-object bit view requires equal physical widths (the same fixed width)")
        readonly = source.readonly or not self.bit_patterns_writable(source.ty)
        checked = readonly and not source.readonly and self.checked_overlay_fields(source.ty) is not None
        continuing, bindings = [], []
        exit_code = None
        for path in paths:
            place = (self.capture_projected_argument(source, path, token, allow_effectful_links=True,
                                                     propagate_exit=not inline or propagate_exit)
                     if source.projection is not None
                     else self.comparison_place(source, path))
            if place.projection is not None and place.projection.always_returns:
                candidate = f"(void)({place.c});"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            self.require_live(place, path, token)
            targets = frozenset(self.physical_places(place, path))
            address = place.c if isinstance(place.ty, Array) else f"&({place.c})"
            continuing.append(path)
            bindings.append((targets, address, None))
        if not continuing:
            return [], [exit_code, "__builtin_unreachable();"]
        return self.bind_bit_overlay(name, ty, source.ty, continuing, readonly, bindings, inline=inline, checked=checked)

    def define_element_bit_overlay(self, name, ty, operand, paths, *, inline=False, propagate_exit=False):
        continuing, bindings = [], []
        readonly, mutable = False, True
        exit_code = None
        for path in paths:
            access = self.subscript(operand, path, propagate_exit=not inline or propagate_exit)
            if isinstance(access, Value):
                candidate = f"(void)({access.code});"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            span, index, indices = access
            element = span.element
            if contains_callable(element):
                operand.token.error("function values cannot be reinterpreted as bits")
            if not isinstance(element, Scalar) or storage_size(element) * 8 != ty.length:
                operand.token.error("array element bit view requires equal physical widths")
            mutable &= not span.readonly
            readonly |= span.readonly or not self.bit_patterns_writable(element)
            continuing.append(path)
            bindings.append((span.owners, f"&({span.c_data()}[{index}])", self.shift_positions(span, indices)))
        if not continuing:
            return [], [exit_code, "__builtin_unreachable();"]
        checked = readonly and mutable
        return self.bind_bit_overlay(name, ty, element, continuing, readonly, bindings, inline=inline, checked=checked)

    def bind_bit_overlay(self, name, ty, source_type, paths, readonly, bindings, *, inline=False, checked=False):
        sym = self.declare(name, ty, readonly=readonly, borrowed=True)
        qualifier = "const " if readonly and not checked else ""
        width = f'_Static_assert(sizeof({source_type.c}) * 8 == {ty.length}, "Lan bit view width");'
        if checked:
            width += " " + " ".join(dict.fromkeys(self.checked_overlay_layout_assertions(source_type)))
        address = f"({qualifier}uint8_t *)({bindings[0][1]})"
        if inline:
            for item in (f'_Static_assert(sizeof({source_type.c}) * 8 == {ty.length}, "Lan bit view width");',
                         *(dict.fromkeys(self.checked_overlay_layout_assertions(source_type)) if checked else ())):
                self.helper("static_assert", None, item)
            sym = replace(sym, c_place=address)
        sym = replace(sym, bit_overlay=True, checked_overlay=checked)
        self.symbols[sym.uid - 1] = sym
        self.scope[-1][name.text] = sym
        self.alias_members[sym.uid] = (sym.uid, ())
        result = []
        for path, (targets, _, positions) in zip(paths, bindings):
            state = replace(path.update(sym, ZERO), borrows=path.borrows | {sym.uid: targets})
            if positions is not None:
                state = replace(state, regions=state.regions | {sym.uid: positions})
            result.append(state)
        return result, [] if inline else [width, f"{qualifier}uint8_t *{sym.c} = {address};"]

    def define_bit_view(self, name, node, paths):
        base, entries = node.args
        host = self.packed(base)
        sources = [(self.capture_projected_argument(host, state, node.token)
                    if host.projection is not None else host) for state in paths]
        for state, source in zip(paths, sources):
            self.require_live(source, state, node.token)
        code = []
        captured = None
        if host.uid in self.alias_members or host.projection is not None:
            old = host
            uid = len(self.symbols) + 1
            host = Symbol(name.text + "$storage", uid, old.ty, len(self.scope),
                          f"lv_{uid}_view_storage", readonly=old.readonly, borrowed=True,
                          bit_overlay=old.bit_overlay, checked_overlay=old.checked_overlay)
            self.symbols.append(host)
            for state, source in zip(paths, sources):
                targets = frozenset(self.physical_places(source, state))
                binding = replace(state.update(host, ZERO), borrows=state.borrows | {uid: targets})
                if old.bit_overlay and old.uid in state.regions:
                    binding = replace(binding, regions=binding.regions | {uid: state.regions[old.uid]})
                state.absorb(binding)
            self.alias_members[uid] = (uid, ())
            code.append(f"{'const ' if host.readonly and not host.checked_overlay else ''}uint8_t *{host.c} = {sources[0].c};")
            captured = host
        fields = {}
        for field_name, annotation, size, start, endian in entries:
            if field_name.text == "len" or field_name.text in fields:
                field_name.error(f"invalid or repeated bit field name {field_name.text!r}")
            ty = self.scalar_type(annotation)
            if is_callback_enum(ty.enum):
                annotation.token.error("bit fields cannot contain function values")
            width, offset = self.const_int(size, ()), self.const_int(start, ())
            if ty.floating and width != ty.bits:
                size.token.error("floating bit fields require their complete physical width")
            if width <= 0 or width > ty.bits or (ty.boolean and width != 1):
                size.token.error("bit field width must fit its declared container; boolean uses one bit")
            if offset < 0 or offset + width > host.ty.length:
                start.token.error("bit field is outside its host array")
            if endian not in ("le", "be", "native"):
                field_name.error("bit field endian must be le, be or native")
            if endian != "le" and (width % 8 or offset % 8):
                field_name.error("be and native bit fields require byte-aligned width and offset")
            fields[field_name.text] = BitField(offset, width, ty, endian)
        sym = self.declare(name, BitView(host, fields))
        if captured is not None:
            sym.fields.append(captured)
        for field_name, bit in fields.items():
            self.trace.append(f"  view id#{sym.uid}.{field_name} on id#{host.uid}: "
                              f"bit({bit.width}, {bit.offset}, {bit.endian}), "
                              f"type {bit.ty.c} & {bit.ty.domain}")
        return code
