"""Callbacks and returning aliases, owned objects, slices, arrays and records."""

from __future__ import annotations

import copy
from dataclasses import field, fields, replace

from shape_domain import CHAIN, REPEAT
from array_domain import ArrayDomain
from float_domain import FloatDomain
from value_domain import Domain
from table_interface import c_field, InterfaceError

from .encoding import TextEncoding
from .lexer import Token
from .syntax import Node
from .typesys import (AliasResult, Array, CallableType, CopyArray, ExternCallable, HeapArray, HeapObject,
                      HeapRecord, Interface, InterfaceResult, Record, RecordLink, Scalar, Slice, SliceEndIndex,
                      Symbol, tagged_error_type, TaggedResult, unqualified, ZERO)
from .domain_utils import container, domain_union, physical
from .interface_types import (CHAIN_REACHABLE, decode_offset_expression, encode_offset_expression,
                              linked_owned_record, offset_expression_width, OWNED_CHAIN_LONG,
                              slice_offset_terms, valid_slice_provenance)


class ReturnMixin:
    def callback_type(self, callee):
        if callee.kind == "callback_ref":
            return self.resolve(Node(callee.token, "callable_type", (callee.args,)))
        if callee.kind == "call":
            signature = self.composite_call_signature(callee)
            return signature.result if signature and isinstance(signature.result, (CallableType, ExternCallable)) else None
        if callee.kind == "name" and not any(callee.args[0] in scope for scope in self.scope):
            return None
        if callee.kind == "index":
            host = callee.args[0]
            while host.kind == "slice":
                host = host.args[0]
            if host.kind == "name" and not any(host.args[0] in scope for scope in self.scope):
                return None
            place = self.place(host)
            if place is not None and isinstance(place.ty, (Array, Slice, HeapArray)):
                element = place.ty.element
                return element if isinstance(element, (CallableType, ExternCallable)) else None
        if callee.kind in ("name", "member", "index"):
            place = self.place(callee)
            if place is not None and isinstance(place.ty, (CallableType, ExternCallable)):
                return place.ty
        return None

    def composite_call_signature(self, node):
        if node is None:
            return None
        if node.kind == "propagate" and node.args[0].kind == "call":
            contract = self.composite_call_signature(node.args[0])
            if contract is not None and isinstance(contract.result, TaggedResult):
                return replace(contract, result=contract.result.success)
            return None
        if node.kind != "call":
            return None
        callee = node.args[0]
        if callee.kind != "name" or not self.named_function(callee.args[0]):
            ty = self.callback_type(callee)
            return ty.contract if ty is not None else None
        name, = callee.args
        if name in self.functions:
            return self.functions[name][1]
        binding = self.externs.get(name) or self.imported_functions.get(name)
        return binding.contract if binding is not None else None

    def composite_call_type(self, node):
        signature = self.composite_call_signature(node)
        result = signature.result if signature is not None else None
        return result if isinstance(result, (Record, CopyArray, Slice, HeapRecord, HeapObject, HeapArray, AliasResult, InterfaceResult)) else None

    def alias_return_place(self, target):
        projection = target.projection
        if projection is None:
            return self.param_places.get(target.uid)
        origin = self.param_places.get(projection.root.uid)
        if origin is None:
            return None
        index, path = origin
        nested_indices = dict(projection.nested_entry_indices)
        for position, step in enumerate(projection.steps):
            if position == 0 and projection.entry_end is not None:
                path += (SliceEndIndex(projection.entry_end),)
            elif position == 0 and projection.entry_index is not None:
                path += (projection.entry_index,)
            elif position in nested_indices:
                path += (nested_indices[position],)
            elif isinstance(step, str):
                path += (step,)
            elif isinstance(step, Domain) and step:
                path += (step.lo if step.lo == step.hi else step,)
            else:
                return None
        return index, path

    @staticmethod
    def merge_alias_places(places):
        if not places or None in places:
            return None
        source, path = next(iter(places))
        if len(places) == 1:
            return source, path
        if not path or any(index != source or not other or other[:-1] != path[:-1]
                           or not isinstance(other[-1], (int, Domain)) for index, other in places):
            return None
        indices = domain_union(Domain.point(other[-1]) if isinstance(other[-1], int) else other[-1]
                               for _, other in places)
        return source, (*path[:-1], indices)

    def inferred_alias_origins(self, places, token):
        merged = self.merge_alias_places(places)
        if merged is not None:
            return (merged,)
        if None in places:
            token.error("inferred alias must refer to one borrowed parameter or field")
        reachable_sources = {source for source, path in places if path == (CHAIN_REACHABLE,)}
        places = {(source, path) for source, path in places
                  if source not in reachable_sources or not all(step == "next" for step in path)}
        return tuple(sorted(places, key=lambda place: (place[0], repr(place[1]))))

    @staticmethod
    def trace_alias_targets(targets):
        return "{" + ", ".join(f"id#{uid}" for uid in sorted({target.uid for target in targets})) + "}"

    @staticmethod
    def composite_return_code(code, tagged, *, success=True):
        if tagged is not None:
            code = tagged.wrap(code, success=success)
        return f"return {code};"

    @staticmethod
    def early_composite_return(code):
        return [], [f"(void)({code});", "__builtin_unreachable();"]

    def return_tagged(self, node, tagged, paths):
        success, error = tagged.success, tagged.error
        signature = self.composite_call_signature(node)
        if signature is not None and isinstance(signature.result, TaggedResult):
            if signature.result != tagged:
                node.token.error("forwarding a composite result requires the same result contract")
            if isinstance(success, (AliasResult, Slice)):
                node.token.error("use '?' when returning a borrowed result so its source can be checked")
            paths, _, code = self.initializer_calls(node, paths)
            if not paths:
                return self.early_composite_return(code)
            self.check_heap_exit(paths, 0, node.token)
            self.return_states.extend(replace(path) for path in paths)
            if isinstance(success, (Record, HeapRecord)):
                self.record_return_fields.append(self.record_call_fields(node))
            if linked_owned_record(success):
                self.record_owned_chain_return(self.owned_chain_call_lengths(node), node.token)
            if isinstance(success, CopyArray):
                self.array_return_contents.append(self.array_call_contents(node))
            return [], [f"return {code};"]
        error_type = self.enum_member_type(node)
        if error_type is None and signature is not None:
            error_type = signature.result
        if error_type is None and node.kind in ("name", "member", "index"):
            place = self.place(node)
            if place is not None:
                error_type = place.ty
        if error_type is None and node.kind == "cast":
            error_type = self.resolve(node.args[2])
        if tagged_error_type(error_type):
            code = early_code = None
            for path in paths:
                value = self.expr(node, path)
                if value.always_returns:
                    if early_code is None or self.propagation_code_rank(value.code) > self.propagation_code_rank(early_code):
                        early_code = value.code
                    continue
                self.range_check(node.token, value, error)
                self.check_heap_exit([path], 0, node.token)
                self.return_states.append(replace(path))
                self.trace.append(f"  return tagged error {value.domain}")
                code = value.code
            if code is None:
                return self.early_composite_return(early_code)
            return [], [self.composite_return_code(code, tagged, success=False)]
        if isinstance(success, Record):
            return self.return_record(node, success, paths, tagged)
        if isinstance(success, CopyArray):
            return self.return_array(node, success, paths, tagged)
        if isinstance(success, Slice):
            return self.return_slice(node, success, paths, tagged)
        if isinstance(success, AliasResult):
            return self.return_alias(node, success, paths, tagged=tagged)
        return self.return_owned(node, success, paths, tagged)

    def return_alias(self, node, result, paths, inferred_alias=None, *, tagged=None):
        code = early_code = None
        for path in paths:
            alias = result.alias if result is not None else inferred_alias
            early = []
            targets, value = self.alias_source(node, path, alias, node.token, capture_exit=early)
            if early:
                if early_code is None or self.propagation_code_rank(value) > self.propagation_code_rank(early_code):
                    early_code = value
                continue
            code = value
            live = targets - {self.null_symbol}
            if result is None:
                places = {self.alias_return_place(target) for target in live}
                if places:
                    origins = self.inferred_alias_origins(places, node.token)
                    params = self.functions[self.current][1].params
                    index, names = origins[0]
                    nullable = self.null_symbol in targets or any(getattr(params[i], "nullable", False) for i, _ in origins)
                    result_type = AliasResult(replace(alias, nullable=nullable), index, names, origins[1:])
                else:
                    result_type = self.null_symbol
                self.inferred_returns.append((node.token, result_type))
            for target in live:
                if result is not None and self.alias_return_place(target) not in result.origins():
                    node.token.error("returned alias must refer to the declared borrowed parameter")
                self.require_target_live(target, path, node.token)
            self.check_heap_exit([path], 0, node.token)
            guard = path.guard_label()
            self.trace.append(f"  return alias {self.trace_alias_targets(targets)} [{guard}]")
            self.return_states.append(replace(path))
        if code is None:
            return self.early_composite_return(early_code)
        return [], [self.composite_return_code(code, tagged)]

    @staticmethod
    def owned_result_fits(actual, expected):
        if isinstance(actual, HeapArray) and isinstance(expected, HeapArray):
            return actual.element == expected.element and actual.domain.within(expected.domain)
        return actual == expected

    @staticmethod
    def is_whole_heap_object(target):
        return (target.uid != 0 and target.projection is None
                and (isinstance(target.ty, HeapArray) or target.heap_owner == target.uid))

    def require_terminal_owner(self, ty, owner, path, token):
        if not isinstance(ty, HeapRecord) or not self.contains_record_links(ty.record):
            return
        if not linked_owned_record(ty):
            token.error("an owning linked record requires one self-referential next link")
        if self.link_targets(self.child_place(owner, ("next",)), path) != frozenset((self.null_symbol,)):
            token.error("cannot transfer ownership of a linked record unless next is provably null")

    def initialize_terminal_owner(self, owner, path):
        field = self.child_place(owner, ("next",))
        path = path.update(field, ZERO)
        return replace(path, borrows=path.borrows | {field.uid: frozenset((self.null_symbol,))})

    def owned_chain_transfer(self, ty, root, path, token):
        """Check that every live node transfers with the returned pointer."""
        if not linked_owned_record(ty):
            return [root], ()
        if root.uid in self.repeated_allocation_uids(path, token):
            token.error("cannot transfer an individual instance inside a symbolic repeated allocation group")
        if self.link_targets(self.child_place(root, ("next",)), path) == frozenset((self.null_symbol,)):
            return [root], (1,)
        owned, summary, stops, _, shared, edges, nodes = self.free_closure(root, path, token)
        if stops:
            token.error("cannot transfer an owning chain that reaches non-owning storage")
        if shared or summary.shape.kind not in (CHAIN, REPEAT):
            token.error("cannot transfer ownership of a chain unless next links form one finite chain")
        for owner in owned:
            if (owner.ty != ty.record or owner.readonly
                    or path.heap.get(owner.uid) != frozenset(("Alive",))):
                token.error("every node of a transferred chain must be a live mutable heap allocation")
        walk = self.free_walk(root, nodes, edges)
        if walk is None:
            token.error("cannot prove a unique next link for every node of the transferred chain")
        order, steps, wrap, _ = walk
        for index, location in enumerate(order):
            targets = self.link_targets(self.child_place(nodes[location], ("next",)), path)
            if index == len(order) - 1:
                expected = {self.null_symbol} if wrap is None else {self.null_symbol, nodes[wrap]}
            else:
                expected = {nodes[steps[index][1]]}
            if targets != expected:
                token.error("cannot prove every allocation is reached by the transferred chain")
        length = min(len(order), OWNED_CHAIN_LONG)
        lengths = ((length,) if wrap is None else tuple(range(length, OWNED_CHAIN_LONG + 1)))
        return [nodes[location] for location in order], lengths

    def record_owned_chain_return(self, lengths, token):
        if lengths != (1,) and lengths and self.current in self.recursive_groups:
            token.error("recursive functions cannot yet return a nonterminal owning chain")
        self.owned_chain_return_lengths.update(lengths)

    def owned_chain_call_lengths(self, node):
        if node.kind == "propagate":
            node = node.args[0]
        callee = node.args[0]
        if callee.kind == "name":
            name = callee.args[0]
            if name in self.functions:
                return self.owned_chain_results.get(name, (1,))
            if binding := self.imported_functions.get(name):
                return binding.owned_chain_lengths or (1,)
        callback = self.callback_type(callee)
        if callback is not None:
            lengths = {length for name in callback.candidates
                       for length in self.owned_chain_results.get(name, (1,))}
            return tuple(sorted(lengths))
        return (1,)

    def owned_chain_nodes(self, root, limit, token):
        nodes = [root]
        for index in range(1, limit):
            place = f"(*(({root.ty.c} *)({nodes[-1].c}).f_next))"
            uid = len(self.symbols) + 1
            node = Symbol("owned_chain_node", uid, root.ty, root.depth, place,
                          heap_owner=uid, readonly=root.readonly, allocated_record=True,
                          chain_summary=index == OWNED_CHAIN_LONG - 1)
            self.symbols.append(node)
            self.record_symbols(node, token, zero=False)
            nodes.append(node)
        return nodes

    def restore_owned_chain(self, nodes, length, path):
        alive, tombstone = frozenset(("Alive",)), frozenset(("Tombstone",))
        path = replace(path, heap=path.heap | {
            node.uid: alive if index < length else tombstone for index, node in enumerate(nodes)})
        for index, node in enumerate(nodes):
            if index:
                path = self.initialize_record_parameter(node, path.update(node, ZERO))
            link = self.child_place(node, ("next",))
            if index + 1 < length:
                targets = frozenset((nodes[index + 1],))
            elif index + 1 == length == OWNED_CHAIN_LONG:
                targets = frozenset((self.null_symbol, node))
                path = replace(path, fresh_links=path.fresh_links | {(link.uid, node.uid)})
            else:
                targets = frozenset((self.null_symbol,))
            path = path.update(link, ZERO)
            path = replace(path, borrows=path.borrows | {link.uid: targets})
        return path

    def ownership_source(self, operand, token, paths):
        """Resolve the whole heap owner whose lifetime this site ends."""
        if operand.kind != "name":
            token.error("ownership transfer requires a named whole heap owner")
        source = self.lookup(operand)
        if (not self.is_whole_heap_object(source) or source.uid in self.alias_members
                or source.readonly):
            token.error("ownership transfer requires a named mutable whole heap owner")
        for path in paths:
            self.require_live(source, path, token)
            if source.uid not in path.heap:
                token.error("ownership transfer requires a live heap allocation")
        self.consume_parameter(source, "move")
        return source

    def define_owned_move(self, name, expected, readonly, node, paths):
        source = self.ownership_source(node, node.token, paths)
        ty = source.ty
        if isinstance(ty, HeapArray):
            actual = replace(ty, count=None)
        elif isinstance(ty, Record):
            actual = HeapRecord(ty)
        else:
            actual = HeapObject(ty)
        if expected is not None:
            if isinstance(actual, HeapArray) and isinstance(expected, HeapArray):
                if (expected.count is not None or expected.element != actual.element
                        or any(not path.values[source.uid].within(expected.domain) for path in paths)):
                    name.error("ownership transfer requires a matching owned array length contract")
                ty = expected
            elif expected != actual:
                name.error("ownership transfer requires a matching owning heap type")
        array = isinstance(ty, HeapArray)
        target = self.declare(name, ty, heap=not array, readonly=readonly, ownership_origin="move")
        if isinstance(ty, Record):
            self.record_symbols(target, name, zero=False)
            paths = self.copy_record_states(target, source, paths)
            for field in target.fields:
                if isinstance(field.ty, Interface):
                    previous = self.child_place(source, (field.name,))
                    paths = self.bind_interface("transfer entity", field, name,
                                                [(previous, "")] * len(paths), paths)
        moved = []
        for path in paths:
            state = path.update(target, path.values.get(source.uid, ZERO),
                                path.forms.get(source.uid), linear=path.linear.get(source.uid))
            if source.uid in path.arrays:
                state = state.with_array(target, path.arrays[source.uid])
            if source.uid in path.bits:
                state = state.with_bits(target, path.bits[source.uid])
            guard = path.guard_label()
            self.trace.append(f"  transfer ownership id#{source.uid} {source.name} [{guard}] "
                              f"to id#{target.uid} {target.name}")
            self.trace.append(f"  mark tombstone id#{source.uid} {source.name} [{guard}]: moved binding")
            moved.append(replace(state, heap=state.heap | {
                source.uid: frozenset(("Tombstone",)),
                target.uid: frozenset(("Alive",)),
            }))
        self.note_transition_write(source, lifetime=True)
        if array:
            return moved, [f"lan_slice {target.c} = {source.c};"]
        pointer = target.c_var
        if isinstance(ty, Array):
            return moved, [f"uint8_t (*{pointer})[{ty.c_length}] = &({source.c});"]
        return moved, [f"{ty.c} *{pointer} = &({source.c});"]

    def refill_owned_slot(self, symbol, token, operand, paths):
        if (len(self.scope) != 2 or self.scope[-1].get(symbol.name) != symbol or symbol.readonly
                or any(path.heap.get(symbol.uid) != frozenset(("Tombstone",)) for path in paths)):
            token.error("ownership refill requires a released mutable owner in the function's outer block")
        if operand.args[0] == symbol.name:
            token.error("ownership refill requires a different live owner")
        if isinstance(symbol.ty, HeapArray):
            expected = replace(symbol.ty, count=None)
        elif isinstance(symbol.ty, Record):
            expected = HeapRecord(symbol.ty)
        else:
            expected = HeapObject(symbol.ty)
        name = replace(token, text=symbol.name)
        # Each allocation needs its own identity. Borrows of the old allocation
        # must still resolve to its tombstone after the name acquires a new owner.
        del self.scope[-1][symbol.name]
        try:
            return self.define_owned_move(name, expected, False, operand, paths)
        except Exception:
            self.scope[-1][symbol.name] = symbol
            raise

    def return_owned(self, node, ty, paths, tagged=None):
        if self.contains_interface_storage(ty):
            node.token.error("a record with interface fields is local-only")
        storage = ty.record if isinstance(ty, HeapRecord) else ty.base if isinstance(ty, HeapObject) else None
        if node.kind == "call" or self.composite_call_type(node) is not None:
            paths, actual, code = self.initializer_calls(node, paths)
            if not paths:
                return self.early_composite_return(code)
            if not self.owned_result_fits(actual, ty):
                node.token.error("owned result requires a matching heap type")
            if isinstance(ty, HeapRecord):
                self.record_return_fields.append(self.record_call_fields(node))
                if linked_owned_record(ty):
                    self.record_owned_chain_return(self.owned_chain_call_lengths(node), node.token)
            self.check_heap_exit(paths, 0, node.token)
            self.return_states.extend(replace(path) for path in paths)
        else:
            for path in paths:
                sym = self.place(node)
                if isinstance(ty, HeapArray):
                    if sym is None or not isinstance(sym.ty, HeapArray) or sym.ty.element != ty.element or sym.readonly:
                        node.token.error("owned result requires a matching mutable owning heap array")
                    self.require_live(sym, path, node.token)
                    if not path.values[sym.uid].within(ty.domain):
                        node.token.error("cannot prove owned array result length fits its contract")
                else:
                    if sym is None or sym.ty != storage or sym.readonly:
                        node.token.error("owned result requires a matching mutable owning heap record or object")
                    self.require_live(sym, path, node.token)
                targets = ((sym,) if isinstance(ty, HeapArray)
                           else self.physical_places(sym, path, allow_null=True))
                for target in targets:
                    if target.readonly or not self.is_whole_heap_object(target):
                        node.token.error("owned result requires a matching mutable owning heap record or object")
                    owned, lengths = self.owned_chain_transfer(ty, target, path, node.token)
                    self.record_owned_chain_return(lengths, node.token)
                    if isinstance(ty, HeapRecord):
                        self.record_return_fields.append(self.record_field_domains(target, path))
                    for owner in owned:
                        self.consume_parameter(owner, "move")
                        self.note_transition_write(owner, lifetime=True)
                    transferred = replace(path, heap=path.heap | {
                        owner.uid: frozenset(("Tombstone",)) for owner in owned})
                    self.check_heap_exit([transferred], 0, node.token)
                    guard = path.guard_label()
                    self.trace.append(f"  transfer ownership id#{target.uid} {target.name} [{guard}] to result")
                    self.trace.append(f"  mark tombstone id#{target.uid} {target.name} [{guard}]: moved binding")
                    if len(owned) > 1:
                        self.trace.append(f"  transfer whole chain [{guard}]: " +
                                          ", ".join(f"id#{owner.uid}" for owner in owned))
                    self.return_states.append(transferred)
                code = sym.c if isinstance(ty, HeapArray) else f"&({sym.c})"
        return [], [self.composite_return_code(code, tagged)]

    def inferred_offset_expression(self, form, params):
        budget = 129

        def expression_key(expr):
            return repr(encode_offset_expression(expr))

        def convert(poly, depth):
            nonlocal budget
            budget -= 1 + len(poly[0])
            if budget < 0 or depth > 16:
                raise ValueError("offset expression exceeds interface limits")
            terms, bias = poly
            products = []
            for factors, scale in terms:
                atoms = []
                for atom, exponent in factors:
                    if atom in self.entry_offset_origins:
                        expr = self.entry_offset_origins[atom]
                    elif isinstance(atom, tuple) and atom[0] == "integer-op":
                        op = atom[1]
                        if op in ("&", "|", "^"):
                            operands = sorted((convert(part, depth + 1) for part in atom[2]), key=expression_key)
                            left, right = operands[0], operands[-1]
                        else:
                            left, right = (convert(part, depth + 1) for part in atom[2:])
                        expr = ("binary", op, left, right)
                    elif isinstance(atom, tuple) and atom[0] == "integer-cast":
                        _, strategy, target, operand = atom
                        source = convert(operand, depth + 1)
                        width = offset_expression_width(source, params)
                        expr = ("cast", strategy, target, width, False, source)
                    else:
                        raise ValueError("offset does not depend only on entry values")
                    atoms.extend([expr] * exponent)
                atoms.sort(key=expression_key)
                product = atoms[0]
                for atom in atoms[1:]:
                    product = ("binary", "*", product, atom)
                if scale != 1:
                    product = ("binary", "*", ("const", scale), product)
                products.append(product)
            if bias or not products:
                products.append(("const", bias))
            products.sort(key=expression_key)
            expression = products[0]
            for product in products[1:]:
                expression = ("binary", "+", expression, product)
            return expression

        return decode_offset_expression(convert(form, 0))

    def infer_slice_offset(self, span, result, params):
        start = self.span_start(span)
        linear = self.linear_value(start, prefer_symbolic=True) if start is not None else None
        if linear is None:
            return result
        polynomial, bias = linear
        if not polynomial or len(polynomial) > 64 or abs(bias) > 1 << 64:
            return result
        compact = all(
            0 < abs(coefficient) <= 64
            and sum(exponent for _, exponent in factors) <= 8
            and all(self.entry_offset_origins.get(origin, (None,))[0] == "param"
                    for origin, _ in factors)
            for factors, coefficient in polynomial
        )
        if compact:
            terms, products = [], []
            for factors, coefficient in polynomial:
                indices = tuple(sorted(self.entry_offset_origins[origin][1]
                                       for origin, exponent in factors for _ in range(exponent)))
                if len(indices) == 1:
                    terms.append((indices[0], coefficient))
                else:
                    products.append((indices, coefficient))
            candidate = replace(result, offset_terms=tuple(sorted(terms)),
                                offset_products=tuple(sorted(products)), offset_bias=bias)
        else:
            try:
                expression = self.inferred_offset_expression(linear, params)
            except (ValueError, InterfaceError):
                return result
            candidate = replace(result, offset_expression=expression)
        return candidate if valid_slice_provenance(params, candidate) else result

    def return_slice(self, node, ty, paths, tagged=None):
        code = early_code = None
        expected_enc = ty.text if ty is not None and isinstance(ty.text, TextEncoding) else None
        for path in paths:
            span = self.ordinary_span(node, path, expected_encoding=expected_enc)
            if span.length.always_returns:
                value = span.c_value()
                if early_code is None or self.propagation_code_rank(value) > self.propagation_code_rank(early_code):
                    early_code = value
                continue
            result = ty
            if result is None:
                places = {self.param_places.get(owner.uid) for owner in span.owners}
                source = next(iter(places)) if len(places) == 1 else None
                offsets = {offset for _, offset in span.positions()}
                offset = next(iter(offsets)) if len(offsets) == 1 else None
                fixed = offset.lo if offset is not None and offset.lo == offset.hi else None
                result = Slice(span.element, length_bound=span.length.domain,
                               readonly=span.readonly or source is None, text=span.text,
                               source_param=source[0] if source is not None else None,
                               source_field=source[1] if source is not None else (),
                               source_offset=fixed if source is not None else None)
                alternative = None
                if source is not None:
                    params = self.functions[self.current][1].params
                    without_offset = replace(result, source_offset=None)
                    inferred = self.infer_slice_offset(span, without_offset, params)
                    if fixed is None:
                        result = inferred
                    elif inferred != without_offset:
                        alternative = inferred
                self.inferred_returns.append((node.token, result))
                self.inferred_slice_paths.append((replace(path), alternative, self.span_start(span)))
            if ty is not None and self.debug_bounds:
                span = self.checked_slice_length(node.token, span, result, "return")
            self.check_span_contract(node.token, span, result, path)
            if result.source_param is not None:
                if not span.owners or any(self.param_places.get(owner.uid) != (result.source_param, result.source_field)
                                          for owner in span.owners):
                    node.token.error("returned slice must come from the declared borrowed parameter")
                if result.source_offset is not None and any(
                        offset != Domain.point(result.source_offset) for _, offset in span.positions()):
                    node.token.error(f"cannot prove returned slice starts at offset {result.source_offset}")
                if result.source_offset is not None:
                    self.trace.append(f"    check return view offset {result.source_offset} => proven")
                if ty is not None and (slice_offset_terms(result) or result.offset_expression is not None):
                    start = self.span_start(span)
                    inputs = {index: self.storage_value(snapshot, path)
                              for index, snapshot in self.result_offset_snapshots.items()}
                    expected = self.slice_offset_value(node.token, result, inputs, path)
                    linear = self.linear_value(start) if start is not None else None
                    same = linear is not None and linear == self.linear_value(expected)
                    if start is None or not same and not self.prove_order(node.token, start, "==", expected, path):
                        node.token.error("cannot prove returned slice starts at its declared entry offset")
                    self.trace.append(f"    check return view start {start.domain} == entry offset {expected.domain} => proven")
            elif not span.owners or not all(owner.uid < 0 or owner.static_storage or
                       (owner.projection is not None and owner.projection.root.static_storage)
                       for owner in span.owners):
                node.token.error("returned slice may outlive its backing storage; static storage is required")
            self.check_heap_exit([path], 0, node.token)
            guard = path.guard_label()
            positions = self.trace_span_positions(span)
            self.trace.append(f"  return view {self.span_origins(span)}, len {span.length.domain} [{guard}]: "
                              f"positions {positions}")
            self.return_states.append(replace(path))
            code = span.c_value()
        if code is None:
            return self.early_composite_return(early_code)
        return [], [self.composite_return_code(code, tagged)]

    def initializer_calls(self, node, paths):
        continuing = []
        continuing_code = error_code = None
        for path in paths:
            early_return = []
            actual, code = self.call(node, path, capture_early_return=early_return)
            if early_return:
                if error_code is None or self.propagation_code_rank(code) > self.propagation_code_rank(error_code):
                    error_code = code
            else:
                continuing.append(path)
                continuing_code = code
        return continuing, actual, continuing_code or error_code

    def define_array_call(self, name, ty, node, paths, readonly):
        paths, result, code = self.initializer_calls(node, paths)
        if not isinstance(result, CopyArray) or result.array != ty:
            node.token.error("array result requires the same fixed array type")
        if not paths:
            return [], [f"(void)({code});", "__builtin_unreachable();"]
        sym = self.declare(name, ty, readonly=readonly)
        carrier = f"lan_result_{sym.uid}"
        sym = replace(sym, c_place=f"{carrier}.data")
        self.scope[-1][name.text] = sym
        self.symbols[sym.uid - 1] = sym
        contents = self.array_call_contents(node)
        if contents is not None:
            paths = [self.restore_record_array(sym, path, contents) for path in paths]
        elif not ty.packed_bits:
            paths = [p.with_array(sym, ArrayDomain.unknown(ty.length, ty.element.domain)) for p in paths]
        self.trace_array_definition(sym, paths)
        return paths, [f"{result.c} {carrier} = {code};"]

    def check_array_call_argument(self, node, actual, target):
        if actual == target:
            return
        contents = self.array_call_contents(node)
        values = None if contents is None else (contents,)
        while isinstance(actual, Array) and isinstance(target, Array):
            if actual == target:
                return
            if actual.length != target.length or actual.packed_bits or target.packed_bits:
                node.token.error("array argument requires compatible fixed array types")
            if values is not None:
                values = tuple(segment.domain for value in values for segment in value.segments)
            actual, target = actual.element, target.element
        if (not isinstance(actual, Scalar) or not isinstance(target, Scalar)
                or replace(actual, domain=target.domain) != target):
            node.token.error("array argument requires compatible fixed array types")
        domain = actual.domain if values is None else domain_union(values)
        if not domain.within(target.domain):
            node.token.error(f"cannot prove array argument contents {domain} fit {target.domain}")

    def array_call_contents(self, node):
        call = node.args[0] if node.kind == "propagate" else node
        callee = call.args[0]
        if callee.kind != "name":
            return None
        name = callee.args[0]
        if name in self.functions:
            return self.array_results.get(name)
        binding = self.imported_functions.get(name)
        return binding.result_array if binding is not None else None

    def trace_array_return(self, ty, path, contents):
        from array_domain import ArrayDomain as ArrayContents

        guard = path.guard_label()
        self.trace.append(f"  return array [{guard}]")
        if isinstance(contents, ArrayContents):
            self.array_return_contents.append(contents)
            self.trace_content("result", ty, contents)
        else:
            self.array_return_contents.append(None)
            self.trace.append(f"    result.len := {{{ty.length}}}")

    def return_array(self, node, result, paths, tagged=None):
        ty = result.array
        temp = f"lan_result_{node.token.line}_{node.token.col}"
        if node.kind == "call" or self.composite_call_type(node) is not None:
            paths, actual, code = self.initializer_calls(node, paths)
            if not paths:
                return self.early_composite_return(code)
            if actual != result:
                node.token.error("array result requires the same fixed array type")
            contents = self.array_call_contents(node)
            self.check_heap_exit(paths, 0, node.token)
            for path in paths:
                self.trace_array_return(ty, path, contents)
            self.return_states.extend(replace(path) for path in paths)
            return [], [self.composite_return_code(code, tagged)]
        if node.kind == "cast" and node.args[1] == "view":
            operand, _, annotation = node.args
            if self.resolve(annotation) != ty:
                node.token.error("array bit view result requires the declared target type")
            source, address, contents = self.composite_view_source(ty, operand, paths, node.token)
            self.check_heap_exit(paths, 0, node.token)
            for path, value in zip(paths, contents):
                self.trace_array_return(ty, path, value)
            self.return_states.extend(replace(path) for path in paths)
            copy = self.helper("disjoint_object_copy", None)
            return [], [f"{result.c} {temp};",
                        f"_Static_assert(sizeof({temp}.data) == sizeof({source.c}), \"Lan composite bit view width\");",
                        f"{copy}({temp}.data, {address}, sizeof({temp}.data));",
                        self.composite_return_code(temp, tagged)]
        if node.kind in ("list", "repeat"):
            uid = len(self.symbols) + 1
            initializer = Symbol(f"result_{uid}", uid, ty)
            if tagged is not None and self.has_propagation(node):
                paths, setup = self.propagating_array_initializer(initializer, node, paths, node.token)
            elif ty.packed_bits:
                paths, setup = self.packed_init(initializer, node, paths, node.token)
            else:
                setup = self.array_init(initializer, node, paths, node.token)
            if not paths:
                return [], setup
            source = initializer.c
            for path in paths:
                self.trace_array_return(ty, path, self.record_value(initializer, path))
        else:
            setup = []
            continuing = []
            early_code = None
            for path in paths:
                contents = None
                if not self.is_copy(node):
                    node.token.error("array return requires as: copy, a list or an array call")
                operand = node.args[0]
                if ty.packed_bits:
                    sym = self.place(operand)
                    if sym is None or sym.ty != ty:
                        node.token.error("packed result requires the same fixed bit array type")
                    sym = self.comparison_place(sym, path)
                    self.require_live(sym, path, node.token)
                    source = sym.c
                    contents = self.record_value(sym, path)
                else:
                    span = self.ordinary_span(operand, path)
                    if span.length.always_returns:
                        value = span.c_value()
                        if early_code is None or self.propagation_code_rank(value) > self.propagation_code_rank(early_code):
                            early_code = value
                        continue
                    contract = Slice(ty.element, length_bound=Domain.point(ty.length), readonly=True)
                    if self.debug_bounds:
                        span = self.checked_slice_length(operand.token, span, contract, "array return")
                    self.check_span_contract(operand.token, span, contract, path)
                    source = span.c_data()
                    windows = []
                    for owner, start in span.positions():
                        array = self.actual_array(path, owner)
                        if (start is None or start.lo != start.hi or start.lo < 0
                                or start.lo + ty.length > array.length):
                            break
                        windows.append(array.slice(start.lo, start.lo + ty.length))
                    else:
                        if windows:
                            contents = ArrayDomain.join_all(windows)
                self.trace_array_return(ty, path, contents)
                continuing.append(path)
            paths = continuing
            if not paths:
                return self.early_composite_return(early_code)
        self.check_heap_exit(paths, 0, node.token)
        self.return_states.extend(replace(path) for path in paths)
        copy = self.helper("disjoint_object_copy", None)
        return [], [*setup, f"{result.c} {temp};",
                    f"{copy}({temp}.data, {source}, sizeof({temp}.data));",
                    self.composite_return_code(temp, tagged)]

    def record_call_fields(self, node):
        call = node.args[0] if node.kind == "propagate" else node
        callee = call.args[0]
        if callee.kind != "name":
            return {}
        name = callee.args[0]
        if name in self.functions:
            return dict(self.record_results.get(name, ()))
        binding = self.externs.get(name) or self.imported_functions.get(name)
        return dict(binding.result_fields) if binding else {}

    def native_record_checks(self, token, record, access):
        checks = []
        for name, field in record.fields:
            member = f"{access}{c_field(name)}"
            field = unqualified(field)
            if isinstance(field, Record):
                checks.extend(self.native_record_checks(token, field, f"{member}."))
            elif isinstance(field, Scalar):
                checks.append(f"{self.membership(token, field.domain)}({member})")
            elif isinstance(field, Slice):
                checks.append(f"{self.membership(token, field.domain)}({member}.len)")
                checks.extend(self.slice_data_predicates(member, field.domain))
                content = self.checked_native_span_content(token, field, member)
                if content != member:
                    checks.append(f"({content}, 1)")
            elif isinstance(field, Array):
                predicate = self.native_array_validator(token, field)
                if predicate:
                    checks.append(f"{predicate}(&({member}))")
        return checks

    def native_record_validator(self, token, record):
        checks = tuple(self.native_record_checks(token, record, "data->"))
        return self.helper("debug_native_record", record, checks) if checks else None

    def native_array_validator(self, token, array):
        if array.packed_bits:
            return None
        element = array.element
        if isinstance(element, Array):
            predicate = self.native_array_validator(token, element)
        elif isinstance(element, Record):
            predicate = self.native_record_validator(token, element)
        elif isinstance(element, Scalar):
            full = FloatDomain.top(element.bits) if element.floating else container(element.bits, element.signed)
            predicate = self.membership(token, element.domain) if element.domain != full else None
        else:
            predicate = None
        return self.helper("debug_native_array", array, predicate) if predicate else None

    def checked_native_span_content(self, token, ty, code):
        element = ty.element
        if isinstance(element, Scalar):
            domain = ty.content_bound if isinstance(ty, Slice) and ty.content_bound is not None else element.domain
            physical = FloatDomain.top(element.bits) if element.floating else container(element.bits, element.signed)
            predicate = self.membership(token, domain) if domain != physical else None
        elif isinstance(element, Record):
            predicate = self.native_record_validator(token, element)
        else:
            predicate = self.native_array_validator(token, element)
        if predicate:
            check = self.helper("debug_native_elements", element.c, (predicate, not isinstance(element, Scalar)))
            code = f"{check}({code})"
        if isinstance(ty, Slice) and ty.text in ("utf8", "c"):
            code = f"{self.helper('utf8_trap', None)}({code})"
        return code

    def checked_native_c_string(self, token, ty, code, available):
        predicate = self.membership(token, ty.domain)
        check = self.helper("debug_native_text", ty.domain.hi, predicate)
        return self.checked_native_span_content(token, ty, f"{check}({code}, {available})")

    def assume_composite_result(self, node, ty, code, *, foreign=False):
        if foreign and self.debug_bounds and isinstance(ty, Scalar):
            result = f"lan_native_result_{node.token.line}_{node.token.col}"
            predicate = self.membership(node.token, ty.domain)
            return self.assumed_result(ty.c, result, code, [f"{predicate}({result})"], checked=True)
        if foreign and isinstance(ty, Slice) and ty.text == "c":
            if self.debug_bounds:
                return self.checked_native_c_string(node.token, ty, code, str(ty.domain.hi))
            assume = self.helper("contract_assume", None)
            length = str(ty.domain.lo) if ty.domain.lo == ty.domain.hi else "strlen(a)"
            body = f"{assume}(a != NULL); return (lan_slice){{a, {length}}}"
            helper = self.helper("unary_eval", None, ("lan_slice", "const char *", body))
            return f"{helper}({code})"
        if foreign and self.debug_bounds and isinstance(ty, (Slice, HeapArray)):
            result = f"lan_native_span_{node.token.line}_{node.token.col}"
            length = self.membership(node.token, ty.domain)
            predicates = [f"{length}({result}.len)", *self.slice_data_predicates(result, ty.domain)]
            code = self.assumed_result(ty.c, result, code, predicates, checked=True)
            return self.checked_native_span_content(node.token, ty, code)
        if isinstance(ty, InterfaceResult) or isinstance(ty, AliasResult) and not ty.alias.nullable:
            kind = "interface" if isinstance(ty, InterfaceResult) else "alias"
            result = f"lan_{kind}_result_{node.token.line}_{node.token.col}"
            return self.assumed_result(ty.c, result, code, [f"{result} != NULL"],
                                       checked=foreign and self.debug_bounds)

        checked = foreign and self.debug_bounds
        if checked and isinstance(ty, CopyArray):
            predicate = self.native_array_validator(node.token, ty.array)
            result = f"lan_native_array_{node.token.line}_{node.token.col}"
            return self.assumed_result(ty.c, result, code,
                                       [f"{predicate}(&{result}.data)"] if predicate else [], checked=True)
        if checked and isinstance(ty, HeapObject):
            result = f"lan_native_object_{node.token.line}_{node.token.col}"
            predicates = [f"{result} != NULL"]
            if isinstance(ty.base, Scalar):
                predicate = self.membership(node.token, ty.base.domain)
                predicates.append(f"{predicate}(*{result})")
            elif (predicate := self.native_array_validator(node.token, ty.base)) is not None:
                predicates.append(f"{predicate}({result})")
            return self.assumed_result(ty.c, result, code, predicates, checked=True)

        from composite_array import text_length

        if not isinstance(ty, (Record, HeapRecord)):
            return code
        result = f"lan_record_result_{node.token.line}_{node.token.col}"
        record = ty.record if isinstance(ty, HeapRecord) else ty
        access = f"{result}->" if isinstance(ty, HeapRecord) else f"{result}."
        predicates = [f"{result} != NULL"] if isinstance(ty, HeapRecord) else []
        if linked_owned_record(ty) and self.owned_chain_call_lengths(node) == (1,):
            predicates.append(f"{access}f_next == NULL")
        if checked:
            predicates.extend(self.native_record_checks(node.token, record, access))
            return self.assumed_result(ty.c, result, code, predicates, checked=True)
        fields = self.record_call_fields(node)

        def visit(record_type, access, path=()):
            for name, field in record_type.fields:
                member = f"{access}{c_field(name)}"
                names = path + (name,)
                field = unqualified(field)
                if isinstance(field, Record):
                    visit(field, f"{member}.", names)
                elif isinstance(field, Scalar):
                    domain = fields.get(names, field.domain)
                    if field.floating:
                        predicates.extend(self.float_range_predicates(node.token, member, domain))
                    else:
                        predicates.extend(self.integer_range_predicates(node.token, member, domain, container(field.bits, field.signed)))
                elif isinstance(field, Slice):
                    domain = text_length(fields.get(names, field.domain))
                    predicates.extend(self.integer_range_predicates(node.token, f"{member}.len", domain, container(64, False)))

        visit(record, access)
        return self.assumed_result(ty.c, result, code, predicates)

    def restore_record_call_fields(self, sym, path, fields):
        for names, field in self.named_record_leaves(sym):
            if isinstance(field.ty, Slice):
                path = self.restore_record_text(field, path, fields.get(names, field.ty.domain))
            elif isinstance(field.ty, Scalar):
                path = path.update(field, fields.get(names, field.ty.domain))
            elif isinstance(field.ty, RecordLink):
                continue  # The ownership contract initializes this separately.
            elif names in fields:
                path = self.restore_record_array(field, path, fields[names])
            elif not field.ty.packed_bits:
                path = path.with_array(field, ArrayDomain.unknown(field.ty.length, field.ty.element.domain))
        return path

    def define_record_call(self, name, ty, node, paths, heap, readonly):
        paths, actual, code = self.initializer_calls(node, paths)
        expected = HeapRecord(ty) if isinstance(actual, HeapRecord) and heap else ty
        if actual != expected:
            node.token.error("record result requires the same nominal type and ownership")
        if not paths:
            return [], [f"(void)({code});", "__builtin_unreachable();"]
        ownership = ("owned_result" if isinstance(actual, HeapRecord) else "allocate") if heap else None
        sym = self.declare(name, ty, heap=heap, readonly=readonly, ownership_origin=ownership)
        self.record_symbols(sym, name, zero=False)
        result = []
        fields = self.record_call_fields(node)
        lengths = self.owned_chain_call_lengths(node) if linked_owned_record(actual) else ()
        nodes = self.owned_chain_nodes(sym, max(lengths), name) if lengths else ()
        for path in paths:
            path = self.restore_record_call_fields(sym, path, fields)
            if heap:
                path = replace(path.update(sym, ZERO), heap=path.heap | {sym.uid: frozenset(("Alive",))})
            if lengths:
                result.extend(self.restore_owned_chain(nodes, length, path) for length in lengths)
            else:
                result.append(path)
        if not heap:
            return result, [f"{ty.c} {sym.c} = {code};"]
        origin = "owned result" if isinstance(actual, HeapRecord) else "allocated from value result"
        self.trace.append(f"  define entity id#{sym.uid} {sym.name}: {origin}, Alive")
        pointer = sym.c_var
        if isinstance(actual, HeapRecord):
            return result, [f"{ty.c} *{pointer} = {code};"]
        alloc = self.allocating_helper("heap_array")
        temp = f"lan_result_{sym.uid}"
        return result, [f"{ty.c} {temp} = {code};",
                        f"{ty.c} *{pointer} = ({ty.c} *){alloc}(1, sizeof({ty.c})).data;",
                        f"{sym.c} = {temp};"]

    def trace_content(self, label, field, value):
        from array_domain import ArrayDomain as ArrayContents
        from composite_array import RecordValue, text_length

        field = unqualified(field)
        if isinstance(field, Slice):
            self.trace.append(f"    {label}.len := {text_length(value)}")
        elif isinstance(value, ArrayContents):
            self.trace.append(f"    {label}.len := {{{value.length}}}")
            for segment in value.segments:
                self.trace_content(f"{label}[{segment.start}, {segment.stop})", field.element, segment.domain)
        elif isinstance(value, RecordValue):
            members = dict(field.fields)
            for name, member in value.fields:
                self.trace_content(f"{label}.{name}", members[name], member)
        elif isinstance(value, (Domain, FloatDomain)):
            self.trace.append(f"    {label} := {value}")

    def record_return_state(self, ty, fields, path):
        self.record_return_fields.append(fields)
        guard = path.guard_label()
        self.trace.append(f"  return record {ty.c} [{guard}]")
        for names, value in sorted(fields.items()):
            field = ty
            for name in names:
                field = unqualified(dict(field.fields)[name])
            self.trace_content(f"result.{'.'.join(names)}", field, value)

    def return_record(self, node, ty, paths, tagged=None):
        if self.contains_interface_storage(ty):
            node.token.error("a record with interface fields is local-only")
        view = node.kind == "cast" and node.args[1] == "view"
        if view or node.kind == "record_initializer":
            name = Token(f"lan_result_{len(self.symbols) + 1}", node.token.line, node.token.col)
            self.scope.append({})
            try:
                if view:
                    operand, _, annotation = node.args
                    if self.resolve(annotation) != ty:
                        node.token.error("record bit view result requires the declared target type")
                    result, code = self.define_composite_view(name, ty, operand, paths, False, node.token)
                else:
                    result, code = self.define_initialized_record(name, ty, node, paths)
                if not result:
                    return [], code
                sym = self.scope[-1][name.text]
            finally:
                self.scope.pop()
            self.check_heap_exit(result, 0, node.token)
            for path in result:
                self.record_return_state(ty, self.record_field_domains(sym, path), path)
            self.return_states.extend(replace(path) for path in result)
            return [], [*code, self.composite_return_code(sym.c, tagged)]
        if node.kind == "call" or self.composite_call_type(node) is not None:
            paths, actual, code = self.initializer_calls(node, paths)
            if not paths:
                return self.early_composite_return(code)
            if actual != ty:
                node.token.error("record result requires the same nominal type")
            fields = self.record_call_fields(node)
            self.check_heap_exit(paths, 0, node.token)
            for path in paths:
                self.record_return_state(ty, fields, path)
            self.return_states.extend(replace(path) for path in paths)
            return [], [self.composite_return_code(code, tagged)]
        for path in paths:
            if node.kind != "copy":
                node.token.error("record return requires as: copy, a field initializer or a record call")
            source = self.record_place(node.args[0])
            if source is None or source.ty != ty:
                node.token.error("record result requires the same nominal type")
            source = self.comparison_place(source, path)
            self.require_live(source, path, node.token)
            fields = self.record_field_domains(source, path)
            code = source.c
            self.check_heap_exit([path], 0, node.token)
            self.record_return_state(ty, fields, path)
            self.return_states.append(replace(path))
        return [], [self.composite_return_code(code, tagged)]
