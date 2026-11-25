"""Definitions and assignments."""

from __future__ import annotations

import re
from dataclasses import field, replace
from itertools import count

from value_domain import Domain

from .syntax import Node
from .typesys import (AliasResult, Array, bit_literal_width, BOOL, BorrowArray, BorrowRecord, BorrowScalar,
                      CopyArray, DynamicBitWritePlan, HeapArray, HeapObject, HeapRecord, Interface,
                      InterfaceResult, is_callback_enum, ReadOnly, Record, RecordLink,
                      require_bit_slice_literal_width, Scalar, Slice, Span, StaticReadOnly, Symbol,
                      unqualified, Value, WIDTHS, ZERO)
from .domain_utils import container, domain_union, physical, plain_integer_parameter
from .state import PathState
from .interface_types import graph_or_chain_parameter
from .summaries import GUARD_COMPARISONS
from .c_emit import indent


class StatementMixin:
    def define(self, stmt, paths):
        name, annotation, node = stmt.args
        if name.text in self.scope[-1]:
            name.error(f"duplicate definition of {name.text!r}")
        ty = self.resolve(annotation) if annotation else None
        readonly = isinstance(ty, ReadOnly)
        ty = unqualified(ty)
        if self.contains_interface_storage(ty) and not isinstance(ty, (Record, HeapRecord)):
            name.error("a record with interface fields requires a local field initializer")
        if self.property_local_types is not None:
            ty = self.property_local_types.get(stmt, ty)
            return self.define_scalar(stmt, paths, ty, readonly, property_local=True)
        if annotation is not None and isinstance(ty, (HeapObject, HeapRecord, HeapArray)) and node is not None:
            source = self.place(node) if node.kind == "name" else None
            if source is not None and self.is_whole_heap_object(source) and source.uid not in self.alias_members:
                return self.define_owned_move(name, ty, readonly, node, paths)
        if node is not None and node.kind == "cast" and node.args[1] == "view":
            operand, _, annotation_type = node.args
            target = self.resolve(annotation_type)
            if isinstance(target, Record) or isinstance(target, Array) and not target.packed_bits:
                if ty is not None and ty != target:
                    node.token.error("composite bit view requires the declared target type")
                return self.define_composite_view(name, target, operand, paths, readonly, node.token)
        if isinstance(ty, Slice) and graph_or_chain_parameter(ty):
            name.error("chain qualifier is only valid on function parameters")
        if isinstance(ty, Slice) and ty.copy:
            annotation.token.error("slice pass: copy is only valid on function parameters")
        if isinstance(ty, Interface):
            return self.define_interface(name, ty, node, paths)
        if isinstance(ty, BorrowRecord) and graph_or_chain_parameter(ty):
            name.error("chain and graph qualifiers are only valid on function parameters")
        if isinstance(ty, (BorrowRecord, BorrowScalar, BorrowArray)):
            return self.define_alias(name, ty, node, paths)
        if annotation is None and node is not None:
            if node.kind == "copy":
                source = self.record_place(node.args[0])
                if source is not None and self.contains_interface_storage(source.ty):
                    return self.define_record(name, source.ty, node, paths)
            result_type = self.composite_call_type(node)
            if isinstance(result_type, Record):
                return self.define_record(name, result_type, node, paths)
            if isinstance(result_type, HeapRecord):
                return self.define_record(name, result_type.record, node, paths, heap=True)
            if isinstance(result_type, HeapObject):
                return self.define_heap_object(name, result_type.base, node, paths, readonly)
            if isinstance(result_type, HeapArray):
                return self.define_heap_array(name, result_type, node, paths, readonly)
            if isinstance(result_type, InterfaceResult):
                return self.define_interface(name, result_type.interface, node, paths)
            if isinstance(result_type, AliasResult):
                return self.define_alias(name, result_type.alias, node, paths)
            if isinstance(result_type, CopyArray):
                return self.define_array_call(name, result_type.array, node, paths, readonly)
            if node.kind in ("name", "member"):
                place = self.place(node)
                if place is not None and isinstance(place.ty, Interface):
                    return self.define_interface(name, place.ty, node, paths)
            record = self.record_place(node)
            if record is not None:
                alias = self.alias_types.get(record.uid, BorrowRecord(record.ty, record.readonly))
                return self.define_alias(name, alias, node, paths)
        if node and node.kind == "bit_view":
            if annotation:
                annotation.token.error("a bit field view is inferred from its declaration")
            return paths, self.define_bit_view(name, node, paths)
        if not annotation and node and node.kind == "cast" and node.args[1] == "view":
            target = self.resolve(node.args[2])
            if isinstance(target, Array) and target.packed_bits:
                ty = target
        if isinstance(ty, Slice):
            if node is None:
                if ty.text is None:
                    name.error("a slice declaration needs an array or slice initializer")
                node = Node(name, "text_literal", (b"",))
            return self.define_slice(name, node, paths, ty)
        encoded_buffer_literal = (
            node is not None and node.kind == "text_literal" and (
                isinstance(ty, Array) and ty.encoding is not None
                or isinstance(ty, StaticReadOnly) and isinstance(ty.base, Array) and ty.base.encoding is not None
            )
        )
        if node is not None and self.span_kind(node) == "ordinary" and not encoded_buffer_literal:
            if annotation is not None:
                annotation.token.error("a slice view requires a slice annotation or an inferred type")
            return self.define_slice(name, node, paths)
        if not annotation and node and node.kind == "slice":
            node.token.error("a bit slice needs an explicit integer or bit array container")
        if isinstance(ty, StaticReadOnly):
            return self.define_rodata(name, ty.base, node, paths)
        if isinstance(ty, HeapObject):
            return self.define_heap_object(name, ty.base, node, paths, readonly)
        if isinstance(ty, HeapRecord):
            return self.define_record(name, ty.record, node, paths, heap=True, readonly=readonly)
        if isinstance(ty, HeapArray):
            return self.define_heap_array(name, ty, node, paths, readonly=readonly)
        if isinstance(ty, Record):
            return self.define_record(name, ty, node, paths, readonly=readonly)
        if isinstance(ty, Array):
            if ty.packed_bits and node is not None and node.kind == "cast" and node.args[1] == "view":
                operand, _, target = node.args
                source = self.place(operand)
                if source is None and operand.kind == "index" and self.span_kind(operand.args[0]) == "ordinary":
                    if self.resolve(target) != ty:
                        target.token.error("view target must match the declared packed bit array")
                    return self.define_element_bit_overlay(name, ty, operand, paths)
                if source is not None and (isinstance(source.ty, (Scalar, Record)) or
                                           isinstance(source.ty, Array) and not source.ty.packed_bits):
                    if self.resolve(target) != ty:
                        target.token.error("view target must match the declared packed bit array")
                    return self.define_bit_overlay(name, ty, source, paths, operand.token)
            if isinstance(self.composite_call_type(node), CopyArray):
                return self.define_array_call(name, ty, node, paths, readonly)
            # Initializers see any outer binding, never the variable being initialized.
            sym = Symbol(name.text, len(self.symbols) + 1, ty)
            if node is not None and node.kind in ("list", "repeat") and self.has_propagation(node):
                paths, code = self.propagating_array_initializer(sym, node, paths, stmt.token)
            elif ty.packed_bits:
                paths, code = self.packed_init(sym, node, paths, stmt.token)
            else:
                code = self.array_init(sym, node, paths, stmt.token)
            self.declare(name, ty, readonly=readonly)
            self.trace_array_definition(sym, paths, zeroed=node is None)
            return paths, code
        return self.define_scalar(stmt, paths, ty, readonly)

    def trace_array_definition(self, sym, paths, *, zeroed=False):
        ty = sym.ty
        label = f"id#{sym.uid}"
        if ty.packed_bits:
            layout = f"bit[{ty.length}], storage={ty.c_length} bytes"
        else:
            layout = f"array len={ty.length}, elem={ty.element.domain}"
        for path in paths:
            guard = path.guard_label()
            self.trace.append(f"  define entity {label} {sym.name}: {layout} [{guard}]")
            if not ty.packed_bits:
                self.trace_content(label, ty, self.actual_array(path, sym))
            elif zeroed:
                self.trace.append(f"    {label}[0, {ty.length}) := {{0}}")
            elif (facts := path.bits.get(sym.uid)) is not None:
                known, data = facts
                self.trace.append(f"    {label} bits known={known:#x}, value={data:#x}")

    @staticmethod
    def has_propagation(node):
        pending = [node]
        while pending:
            part = pending.pop()
            if isinstance(part, Node):
                if part.kind == "propagate":
                    return True
                pending.extend(part.args)
            elif isinstance(part, (tuple, list)):
                pending.extend(part)
        return False

    @staticmethod
    def propagation_code_rank(code):
        # Shortened alternatives belong to the same expression. The one
        # reaching the later postfix operator includes the earlier prefix.
        return max(((int(line), int(column))
                    for line, column in re.findall(r"\breturn lan_error_(\d+)_(\d+);", code)),
                   default=(-1, -1))

    def boolean_cast_operand(self, node):
        if node.kind == "cast" and node.args[1] == "safe":
            ty = unqualified(self.resolve(node.args[2]))
            if isinstance(ty, Scalar) and ty.boolean:
                return node.args[0]
        return None

    def scalar_rhs_paths(self, node, state, limit):
        aggregate_comparison = False
        if (node.kind in ("binary", "rank_compare") and node.args[0] in ("==", "!=", "===", "!==")
                and not self.has_effects(node, allow_debug_checks=True)):
            operands = node.args[-2:]
            record_types = (Record, RecordLink) if node.kind == "rank_compare" else (Record,)
            left_kind, right_kind = map(self.span_kind, operands)
            aggregate_comparison = (
                left_kind == right_kind and left_kind in ("ordinary", "packed")
                or all(isinstance(getattr(self.place(operand), "ty", None), record_types)
                       for operand in operands))
        if not aggregate_comparison:
            operands = [node]
            while operands:
                operand = operands.pop()
                truth_operand = (operand is not node or operand.kind in ("in", "rank_compare")
                                 or operand.kind == "binary" and operand.args[0] in GUARD_COMPARISONS)
                if self.call_guard(operand) is not None or truth_operand and self.has_propagation(operand):
                    break
                if (operand.kind == "binary" and operand.args[0] in ("&&", "||", "^^")
                        or operand.kind == "unary" and operand.args[0] == "!"):
                    operands.extend(operand.args[1:])
                else:
                    cast_operand = self.boolean_cast_operand(operand)
                    if cast_operand is not None:
                        operands.append(cast_operand)
            else:
                return [(state, self.expr(node, state))]
        yes, no, code = self.condition(node, state)
        if not yes and not no:
            return [(state, Value(BOOL, code, 8, True, effectful=True, always_returns=True))]
        if len(yes) + len(no) > limit:
            domain = BOOL if yes and no else Domain.point(bool(yes))
            return [(PathState.join(yes + no), Value(domain, code, 8, True))]
        return [(path, Value(Domain.point(outcome), code, 8, True))
                for paths, outcome in ((yes, 1), (no, 0)) for path in paths]

    def saved_property_value(self, node, ty, state):
        if (node is None or node.kind not in ("call", "name", "cast", "unary", "binary")
                or not plain_integer_parameter(ty)
                or self.has_effects(node, allow_property_calls=True)):
            return None
        used = []
        term = self.property_term(node, state, used, use_saved=True)
        return (used[0], term) if term is not None and used else None

    def define_scalar(self, stmt, paths, ty, readonly, *, property_local=False, value_check=None):
        name, annotation, node = stmt.args
        if annotation is not None and not isinstance(ty, Scalar):
            annotation.token.error("expected a scalar or array variable type")
        if ty is None and node is None:
            name.error("a definition needs a type or an initializer")
        if node is not None:
            limit = max(1, self.PATH_LIMIT // len(paths))
            evaluated = [pair for path in paths for pair in self.scalar_rhs_paths(node, path, limit)]
            continuing = [(path, value) for path, value in evaluated if not value.always_returns]
            if not continuing:
                code = max((value.code for _, value in evaluated), key=self.propagation_code_rank)
                return [], [f"(void)({code});", "__builtin_unreachable();"]
            paths, values = map(list, zip(*continuing))
            self.budget_paths(paths, node.token)
        else:
            values = [Value(ZERO, "0", boolean=ty.boolean, enum=ty.enum,
                            character=ty.character, signed=ty.signed) for path in paths]
        if value_check is not None:
            for value in values:
                value_check(value)
        if ty is None:
            domain = domain_union(val.domain for val in values)
            if values[0].result_arms is not None:
                success, errors = values[0].result_arms
                ty = Scalar(success | errors, values[0].width, values[0].signed,
                            floating=values[0].floating, result_arms=values[0].result_arms,
                            result_error_enums=values[0].result_error_enums)
            elif values[0].floating:
                ty = self.types[f"float{values[0].domain.bits}"]
            elif values[0].boolean:
                ty = self.types["boolean"]
            elif values[0].character:
                ty = self.types[values[0].character]
            elif is_callback_enum(values[0].enum):
                name.error("a callback binding requires an explicit callable type")
            elif values[0].enum:
                info = values[0].enum
                ty = Scalar(Domain.of(*((n, n) for _, n in info.members)), values[0].width,
                            info.signed, enum=info)
            else:
                minimum = max(val.width for val in values)
                for bits in WIDTHS:
                    full = container(bits, domain.lo < 0)
                    if bits >= minimum and domain.within(full):
                        ty = Scalar(full, bits, domain.lo < 0)
                        break
                if ty is None:
                    name.error(f"no fixed-width integer container fits inferred domain {domain}")
        self.check_intersection_type(ty, name)
        ty = physical(ty, name)
        if property_local:
            if not plain_integer_parameter(ty):
                name.error("algebraic property requires local integer variables")
            self.property_local_types[stmt] = ty
        values = [self.coerce_storage(node.token if node else name, value, ty) for value in values]
        if plain_integer_parameter(ty):
            values = [self.ssa_narrow(stmt, value, ty.domain, "writes", "initializer") for value in values]
        for value in values:
            self.range_check(node.token if node else name, value, ty)
        saved = [self.saved_property_value(node, ty, path) for path in paths]
        sym = self.declare(name, ty, readonly=readonly)
        self.trace.append(f"  define entity id#{sym.uid} {name.text}: {ty.c} & {ty.domain}")
        updated = []
        for path, value, fact in zip(paths, values, saved):
            state = path.update(sym, value.domain, value.affine, value.difference, value.linear)
            if fact is not None:
                key = ("property_value", sym.uid, state.epochs.get(sym.uid))
                state = replace(state, predicates=state.predicates | {key: fact})
            updated.append(state)
        return updated, [f"{ty.c} {sym.c} = {values[0].code};"]

    def assign(self, stmt, paths):
        target, op, node = stmt.args
        if (host := self.keyed_host(target)) is not None:
            return self.assign_keyed(host, target, op, node, paths)
        if target.kind == "name" and op.text == "=" and node.kind == "name":
            owner = self.lookup(target)
            if (self.is_whole_heap_object(owner) and owner.uid not in self.alias_members
                    and any("Tombstone" in path.heap.get(owner.uid, ()) for path in paths)):
                return self.refill_owned_slot(owner, target.token, node, paths)
        if target.kind == "name" and self.lookup(target).uid in self.alias_types and node.kind != "copy":
            return self.rebind_alias(self.lookup(target), op, node, paths)
        if target.kind == "name" and isinstance(self.lookup(target).ty, Interface):
            return self.rebind_interface(self.lookup(target), op, node, paths)
        if node.kind == "copy" and self.record_place(target) is not None:
            return self.assign_record_copy(target, op, node, paths)
        if target.kind == "slice" and self.span_kind(target) == "ordinary":
            return self.assign_slice(target, op, node, paths)
        if target.kind == "name" and isinstance(self.lookup(target).ty, Slice):
            return self.assign_slice(target, op, node, paths, rebind=True)
        field = self.field_place(target)
        if field is not None and isinstance(field.ty, Interface):
            return self.rebind_interface(field, op, node, paths)
        if field is not None and isinstance(field.ty, RecordLink):
            self.propagating_expression_nodes |= (self.propagation_nodes(target)
                                                  | self.propagation_nodes(node, composite_call=True))
            return self.assign_link(field, op, node, paths, propagate_target=self.has_propagation(target))
        if target.kind == "slice" or target.kind == "member" and field is None:
            return self.assign_bit_span(target, op, node, paths)
        if target.kind == "name" or field is not None:
            sym = field if field is not None else self.lookup(target)
            if isinstance(sym.ty, Array) and sym.ty.packed_bits:
                if bit_literal_width(node) is not None and op.text in ("=", "&=", "|=", "^="):
                    window = Node(target.token, "slice", (target, None, None, False))
                    return self.assign_bit_span(window, op, node, paths)
                if op.text == "=" and node.kind == "cast" and node.args[1] == "view":
                    return self.packed_init(sym, node, paths, op, declare_storage=False)
        result = []
        exit_code = None
        variable = field if field is not None else self.lookup(target) if target.kind == "name" else None
        linked = variable is not None and variable.link_access is not None
        linked_exit = linked and self.has_propagation(target)
        if target.kind == "index" or linked_exit:
            self.propagating_expression_nodes |= self.propagation_nodes(target)
        for path in paths:
            element_key = None
            if variable is not None:
                sym = self.comparison_place(variable, path, allow_effectful_links=linked_exit,
                                            propagate_exit=linked_exit)
                if sym.projection is not None and sym.projection.always_returns:
                    candidate = f"(void)({sym.c});"
                    if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                        exit_code = candidate
                    continue
                self.require_live(sym, path, op)
                if sym.readonly:
                    op.error("cannot modify an immutable field")
                if not isinstance(sym.ty, Scalar):
                    target.token.error("whole-array, view or record assignment is not supported")
                ty, lhs = sym.ty, sym.c
                old = self.storage_value(sym, path)
            elif target.kind == "index":
                index_values = []
                access = self.subscript(target, path, capture_index=index_values, propagate_exit=True)
                if isinstance(access, Value):
                    candidate = f"(void)({access.code});"
                    if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                        exit_code = candidate
                    continue
                sym, index, indices = access
                if isinstance(sym, Span):
                    if not isinstance(sym.element, Scalar):
                        op.error("composite array element assignment requires an explicit copy or initializer")
                    if sym.readonly:
                        op.error("cannot write to an immutable slice")
                    lhs = f"{sym.c_data()}[{index}]"
                    original_span, original_indices = sym, indices
                    host, subscript = target.args
                    pure_index = not (self.span_has_effects(host, sym) or self.has_effects(subscript))
                    if pure_index:
                        if not self.has_effects(node):
                            element_key = self.array_element_key(sym, index_values[0])
                        sym, indices = self.correlated_span_element(path, sym, indices, index_values[0])
                    ty = sym.element
                    domain = self.known_array_element(path, element_key, self.read_span_contents(path, sym, indices))
                    if pure_index:
                        domain = self.known_span_window(path, original_span, original_indices, domain)
                    old = Value(domain, lhs, ty.bits, ty.boolean,
                                enum=ty.enum, character=ty.character, signed=ty.signed)
                elif sym.ty.packed_bits:
                    if sym.readonly and not sym.checked_overlay:
                        op.error("cannot modify an immutable bit array")
                    if op.text not in ("=", "&=", "|=", "^="):
                        op.error("packed bit assignment supports =, &=, |= and ^=")
                    base, setup = sym.c, ""
                    if sym.link_access is not None and self.has_effects(target.args[0]):
                        base = f"lan_bit_base_{op.line}_{op.col}"
                        setup = f"uint8_t *{base} = {sym.c}; "
                    value = self.expr(node, path)
                    if value.always_returns:
                        candidate = f"{{ (void)({sym.c}); (void)({index}); (void)({value.code}); }}"
                        if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                            exit_code = candidate
                        continue
                    self.require_bit(op, value, node)
                    helper = self.helper("bit_write", None)
                    position = f"lan_bit_index_{op.line}_{op.col}"
                    if op.text != "=":
                        load = self.helper("bit_read", None)
                        old_domain = (self.bit_window_read_domain(path, sym, replace(index_values[0], domain=indices),
                                                                Value(Domain.point(1), "1"))
                                      if not self.has_effects(node) else BOOL)
                        old_bit = Value(old_domain, f"{load}({base}, {position})", 8, effectful=True)
                        value = self.binary(op, op.text[0], old_bit, value, path)
                    code = (f"{{ {setup}uint64_t {position} = {index}; "
                            f"{helper}({base}, {position}, (uint8_t)({value.code})); }}")
                    offset = indices.lo if indices.lo == indices.hi else None
                    pure = not self.has_effects(node) and not any(self.has_effects(part) for part in target.args)
                    position_value = replace(index_values[0], domain=indices)
                    width = Value(Domain.point(1), "1")
                    mapped = self.bit_window_affine_value(position_value, width, value) if pure else None

                    def value_at(start, size):
                        if mapped is None:
                            return value.domain
                        point = mapped(start, size)
                        return value.domain & Domain.point(point) if point is not None else Domain()

                    if sym.checked_overlay:
                        candidates = self.bit_window_candidates(position_value, width) if pure else ()
                        windows = self.checked_bit_windows(op, sym, indices, width.domain, candidates)
                        updates = [(start, size, domain) for start, size in windows if (domain := value_at(start, size))]
                        result.append(self.checked_overlay_write(op, path, sym, updates, "le"))
                    elif offset is None and (pure or sym.projection is None and sym.link_access is None):
                        result.append(self.update_dynamic_bit_window(
                            path, sym, position_value, width, value_at if pure else None, value_domain=value.domain))
                    else:
                        result.append(self.update_bits(path, sym, offset, 1, value.domain))
                    if self.literal_endpoint(target.args[1]):
                        code = self.bit_write_plan(code, sym, offset, 1, value, "le", target, op, node)
                    continue
                else:
                    raise AssertionError("expected a packed array or ordinary span")
            else:
                target.token.error("expected a variable or array element as the assignment target")
            prefix = []
            if (target.kind == "index" or linked) and (self.has_effects(target) or self.has_effects(node)):
                pointer = f"lan_target_{op.line}_{op.col}"
                prefix.append(f"{ty.c} *{pointer} = &({lhs});")
                lhs = f"(*{pointer})"
                old = replace(old, code=lhs)
                if linked:
                    sym = replace(sym, captured_places=self.physical_places(sym, path))
            rhs = (self.scalar_rhs_paths(node, path, max(1, self.PATH_LIMIT // len(paths)))
                   if op.text == "=" and ty.boolean
                   else [(path, self.expr(node, path))])
            for current, value in rhs:
                if value.always_returns:
                    candidate = f"(void)({value.code});"
                    if prefix:
                        candidate = "{ " + " ".join(prefix) + " " + candidate + " }"
                    if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                        exit_code = candidate
                    continue
                if op.text != "=":
                    value = self.binary(op, op.text[:-1], old, value, current)
                value = self.coerce_storage(op, value, ty)
                if plain_integer_parameter(ty):
                    value = self.ssa_narrow(stmt, value, ty.domain, "writes", "assignment")
                self.range_check(op, value, ty)
                updated = (self.store_value(current, sym, value) if variable is not None
                           else self.write_span_contents(current, sym, indices, value.domain, element_key=element_key))
                if element_key is not None:
                    updated = self.remember_array_element(current, updated, element_key, sym, indices, value.domain)
                if (op.text == "=" and variable is not None and target.kind == "name"
                        and not (sym.borrowed or sym.heap_owner or sym.bit_overlay or sym.uid in self.alias_types)):
                    fact = self.saved_property_value(node, ty, current)
                    if fact is not None:
                        key = ("property_value", sym.uid, updated.epochs.get(sym.uid))
                        updated = replace(updated, predicates=updated.predicates | {key: fact})
                result.append(updated)
                if variable is not None:
                    guard = current.guard_label()
                    self.trace.append(f"  update entity id#{sym.uid} [{guard}] := {value.domain}")
                code = f"{lhs} = {value.code};"
                if prefix:
                    code = "{ " + " ".join(prefix) + " " + code + " }"
        self.budget_paths(result, op)
        if not result and exit_code is not None:
            return [], [exit_code, "__builtin_unreachable();"]
        return result, [code]

    def assign_keyed(self, host, target, op, expression, paths):
        result, code, exit_code = [], None, None
        for path in paths:
            field, choices = self.keyed_field(target, host, path, writable=True)
            if isinstance(field, Value):
                candidate = f"(void)({field.code});"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            pointer = f"lan_keyed_target_{op.line}_{op.col}"
            key_expr = target.args[1]
            key_symbol = self.place(key_expr) if key_expr.kind == "name" else None
            split_key = (key_symbol is not None and key_symbol.uid in path.values
                         and not (key_symbol.borrowed or key_symbol.bit_overlay or key_symbol.heap_owner
                                  or key_symbol.uid in self.alias_members)
                         and not self.has_effects(expression) and len(choices) <= 8
                         and len(choices) * len(paths) <= self.PATH_LIMIT)
            setup = f"{field.ty.c} *{pointer} = &({field.c});"
            targets = self.physical_places(field, path)
            old = None
            old_fields = {}
            if op.text != "=":
                value = self.storage_value(field, path)
                saved = f"lan_keyed_old_{op.line}_{op.col}"
                setup += f" {field.ty.c} {saved} = *{pointer};"
                old = replace(value, code=saved)
                if len(targets) > 1:
                    old_fields = {storage: replace(self.storage_value(storage, path), code=saved)
                                  for storage in targets}
            value = self.expr(expression, path)
            if value.always_returns:
                candidate = f"{{ {setup} (void)({value.code}); }}"
                if exit_code is None or self.propagation_code_rank(candidate) > self.propagation_code_rank(exit_code):
                    exit_code = candidate
                continue
            operand = value
            if old is not None:
                value = self.binary(op, op.text[:-1], old, value, path)
            value = self.coerce_storage(op, value, field.ty)
            new_fields = {}
            for storage in targets:
                self.require_target_live(storage, path, op)
                post = value
                if storage in old_fields:
                    post = self.binary(op, op.text[:-1], old_fields[storage], operand, path)
                    post = self.coerce_storage(op, post, storage.ty)
                self.range_check(op, post, storage.ty)
                new_fields[storage] = post
            if len(targets) == 1:
                result.append(self.store_value(path, field, value))
            elif not old_fields and not split_key and not any(storage.projection is not None for storage in targets):
                result.append(self.store_value(path, field, value))
            else:
                # Each candidate can remain unchanged on this execution.
                # In particular, this must remain true when two fields share
                # one resolved array projection and its backing storage.
                outcomes = []
                self.spend(op, len(choices) * max(1, len(path.values)))
                for index, storage in choices:
                    current = path.narrow(key_symbol, Domain.point(index)) if split_key else path
                    if current is None:
                        continue
                    selected = replace(field, captured_places=(storage,))
                    outcomes.append(self.store_value(current, selected, new_fields[storage]))
                if split_key:
                    result.extend(outcomes)
                else:
                    result.append(PathState.join(outcomes))
            code = f"{{ {setup} *{pointer} = {value.code}; }}"
        self.budget_paths(result, op)
        return (result, [code]) if result else ([], [exit_code, "__builtin_unreachable();"])

    def checked_bit_windows(self, token, symbol, offset, width, candidates=()):
        limit = symbol.ty.length
        if candidates:
            return [(start, size) for start, size in candidates if 0 <= start and 0 < size <= limit - start]
        positions = offset & Domain.of((0, limit - 1))
        sizes = width & Domain.of((1, limit))
        windows = []
        for lo, hi in positions.ranges:
            self.spend(token, hi - lo + 1)
            for start in range(lo, hi + 1):
                for low, high in sizes.ranges:
                    end = min(high, limit - start)
                    if low <= end:
                        self.spend(token, end - low + 1)
                        windows.extend((start, size) for size in range(low, end + 1))
        return windows

    def bit_copy_source_position(self, start, origin, count, shift):
        if shift.lo == shift.hi:
            return lambda position, size: position - shift.lo
        return self.bit_window_affine_value(start, count, origin)

    def bit_window_affine_value(self, start, width, value):
        key = start if self.related(start, value) and start.affine.scale else width
        if not self.related(key, value) or not key.affine.scale:
            return None

        def value_at(position, size):
            selected = position if key is start else size
            base, remainder = divmod(selected - key.affine.offset, key.affine.scale)
            if remainder:
                return None
            return value.affine.scale * base + value.affine.offset

        return value_at

    def checked_bit_copy(self, token, state, dst, start, width, src, origin, candidates=(), *, source_at=None):
        self.require_live(src, state, token)
        readers = []
        if src.bit_overlay and src.uid not in state.regions:
            targets = self.physical_places(src, state)
            if all(isinstance(target.ty, Scalar) for target in targets):
                readers = [self.composite_source_bits(token, state, target, set())[0] for target in targets]
        def read_source(index, size):
            self.spend(token)
            domain = self.bit_read_domain(state, src, index, size)
            if readers:
                domain &= Domain.union(self.masked_bit_domain(mask, data, size)
                                       for reader in readers for mask, data in reader(index, size))
            return domain

        by_width = {}
        updates = []
        for position, size in self.checked_bit_windows(token, dst, start, width, candidates):
            if source_at is not None:
                index = source_at(position, size)
                if index is not None and 0 <= index <= src.ty.length - size:
                    updates.append((position, size, read_source(index, size)))
                continue
            if size not in by_width:
                offsets = origin & Domain.of((0, src.ty.length - size))
                if not offsets:
                    continue
                if sum(hi - lo + 1 for lo, hi in offsets.ranges) > 256:
                    by_width[size] = container(size, False)
                else:
                    domains = []
                    for lo, hi in offsets.ranges:
                        for index in range(lo, hi + 1):
                            domains.append(read_source(index, size))
                    by_width[size] = Domain.union(domains)
            updates.append((position, size, by_width[size]))
        return self.checked_overlay_write(token, state, dst, updates, "le")

    def assign_dynamic_bits(self, target, op, node, paths):
        states = []
        error_code = None
        store = self.helper("bit_store", None)
        host, lower, upper, _ = target.args
        literal_width = bit_literal_width(node)
        fusion_host = None
        if (not self.debug_bounds and op.text == "=" and host.kind == "name"
                and all(part is None or self.independent_bit_rhs(part) for part in (lower, upper, node))):
            candidate = self.packed(host)
            if self.direct_bit_fusion_host(candidate):
                fusion_host = candidate
        windows = []
        for path in paths:
            region = self.bit_region(target, path, capture=True, propagate_exit=True)
            if isinstance(region, Value):
                candidate = f"(void)({region.code});"
                rank = self.propagation_code_rank(candidate)
                if error_code is None or rank > error_rank:
                    error_code, error_rank = [candidate], rank
                continue
            symbol, offset, width, setup = region
            if symbol.readonly and not symbol.checked_overlay:
                op.error("cannot modify an immutable bit array")
            if literal_width is not None:
                require_bit_slice_literal_width(node.token, width.domain, literal_width)
            if width.domain.hi > 64:
                op.error("scalar bit slice assignment requires at most 64 bits")
            suffix = f"{op.line}_{op.col}"
            data, position, count = (f"lan_bit_{part}_{suffix}" for part in ("data", "offset", "width"))
            fixed = offset.domain.lo == offset.domain.hi and width.domain.lo == width.domain.hi
            old = None
            if op.text != "=":
                load = self.helper("bit_load", None)
                domain = (self.bit_read_domain(path, symbol, offset.domain.lo, width.domain.lo)
                          if fixed else container(width.domain.hi, False))
                if all(not self.has_effects(part) for part in target.args if isinstance(part, Node)):
                    domain = self.bit_window_read_domain(path, symbol, offset, width)
                old = Value(domain, f"{load}({data}, {position}, {count}, 0)", 64)
            assumed_width = self.assumed_scalar_bit_width(op, width.code)
            endpoints = [f"uint64_t {position} = (uint64_t)({offset.code});",
                         f"unsigned {count} = (unsigned)({assumed_width});"]
            address = [*setup, f"uint8_t *{data} = {symbol.c};", *endpoints]
            value = self.expr(node, path)
            if value.always_returns:
                prefix = address + ([f"(void)({old.code});"] if old is not None else [])
                candidate = f"(void)({value.code});"
                rank = self.propagation_code_rank(candidate)
                if error_code is None or rank > error_rank:
                    error_code, error_rank = ["{", *indent([*prefix, candidate]), "}"], rank
                continue
            if old is not None:
                value = self.binary(op, op.text[:-1], old, value, path)
            if value.floating or value.enum or value.character or value.boolean:
                op.error("bit slice assignment requires a fitting integer")
            pure = not self.has_effects(node) and all(
                not self.has_effects(part) for part in target.args if isinstance(part, Node))
            mapped = self.bit_window_affine_value(offset, width, value) if pure else None

            def value_at(start, size):
                if mapped is None:
                    return value.domain
                result = mapped(start, size)
                return value.domain & Domain.point(result) if result is not None else Domain()

            bounds = container(width.domain.lo, False)
            candidates = self.bit_window_candidates(offset, width) if pure and (mapped or symbol.checked_overlay) else ()
            if self.proven_presence(value.domain - bounds, op):
                writes = [(size, domain) for start, size in candidates
                          if start + size <= symbol.ty.length and (domain := value_at(start, size))]
                if not writes or any(self.proven_presence(domain - container(size, False), op)
                                     for size, domain in writes):
                    op.error(f"cannot prove {value.domain} fits the minimum {width.domain.lo}-bit field range {bounds}")

            if symbol.checked_overlay:
                windows = self.checked_bit_windows(op, symbol, offset.domain, width.domain, candidates)
                updates = [(start, size, domain) for start, size in windows if (domain := value_at(start, size))]
                changed = self.checked_overlay_write(op, path, symbol, updates, "le")
            elif fixed:
                changed = self.update_bits(path, symbol, offset.domain.lo, width.domain.lo, value.domain)
            else:
                changed = self.update_dynamic_bit_window(
                    path, symbol, offset, width, value_at if pure else None, value_domain=value.domain)
            states.append(changed)
            code = ["{", *indent([*address,
                                  f"{store}({data}, {position}, {count}, (uint64_t)({value.code}), 0);"]), "}"]
            if fusion_host is not None and not value.effectful:
                windows.append((offset.domain.lo, offset.domain.hi, width.domain.hi, offset.affine))
        if not states and error_code is not None:
            return [], [*error_code, "__builtin_unreachable();"]
        if windows and len(windows) == len(paths):
            affine = windows[0][3]
            if any(window[3] != affine for window in windows):
                affine = None
            # Fused emission captures the direct host once for the whole group.
            plan = DynamicBitWritePlan(
                fusion_host.c, (min(window[0] for window in windows), max(window[1] for window in windows)),
                max(window[2] for window in windows), affine, (*setup[1:], *endpoints),
                position, count, value.code, tuple(code))
            return states, [plan]
        return states, code
