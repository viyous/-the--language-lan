"""Symbols, places, record leaves, heap objects and static data."""

from __future__ import annotations

from dataclasses import field, fields, replace
from fractions import Fraction

from array_domain import ArrayDomain
from float_domain import c_literal, FloatDomain
from value_domain import Domain
from table_interface import c_field, c_ident

from .encoding import ConstList, EncodingTag, JsonMap, TextEncoding
from .errors import FloatConstant, LanError
from .syntax import Node, TypeArgument
from .typesys import (Array, bit_literal_bytes, bit_literal_value, BOOL, BorrowScalar, CallableType,
                      HeapObject, Interface, ModuleConst, ReadOnly, Record, RecordLink, require_bit_literal_width,
                      Scalar, Slice, Span, StaticReadOnly, Symbol, unqualified, Value, ZERO)
from .domain_utils import c_int, physical, zero_domain
from .state import PathState

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class DeclarationMixin:
    def declare(self, name, ty, heap=False, readonly=False, borrowed=False, static=False, *,
                interface_layout=None, allocated_record=False, ownership_origin=None):
        if name.text in self.scope[-1]:
            name.error(f"duplicate definition of {name.text!r}")
        if isinstance(ty, CallableType) and self.named_function(name.text):
            name.error("a callback binding cannot shadow a declared function")
        uid = len(self.symbols) + 1
        place = f"(*lv_{uid}_{c_ident(name.text)})" if heap or borrowed and not isinstance(ty, Array) else ""
        sym = Symbol(name.text, uid, ty, len(self.scope), place, uid if heap else 0, readonly, borrowed,
                     static_storage=static, interface_layout=interface_layout, allocated_record=allocated_record,
                     semantic_binding=self.semantic_module.declarations.get(name), ownership_origin=ownership_origin)
        self.scope[-1][name.text] = sym
        self.symbols.append(sym)
        return sym

    def lookup(self, expr):
        name, = expr.args
        for scope in reversed(self.scope):
            if name in scope:
                return scope[name]
        if name in self.module_consts:
            entry = self.module_consts[name]
            if entry.sym is None:
                expr.token.error(f"compile-time constant {name!r} cannot be used at runtime")
            return entry.sym
        expr.token.error(f"unknown variable {name!r}")

    def array(self, expr):
        sym = self.place(expr)
        if sym is None or not isinstance(sym.ty, Array):
            expr.token.error("expected a fixed array")
        return sym

    def packed(self, expr):
        sym = self.array(expr)
        if not sym.ty.packed_bits:
            expr.token.error("this operation requires a packed bit array")
        return sym

    def enum_member_type(self, node, *, env=None):
        if node.kind not in ("member", "length"):
            return None
        namespace = node.args[0]
        keyed = namespace.kind == "member" and namespace.args[1].text == "key"
        owner = namespace.args[0] if keyed else namespace
        if owner.kind != "name":
            return None
        name = owner.args[0]
        scopes = getattr(self, "scope", ()) if env is None else env
        local = self.compile_time_binding(name, scopes)
        if local is not None:
            value = local[name][0] if env is not None else None
            if not isinstance(value, TypeArgument):
                return None
            ty = value.value
        else:
            ty = self.types.get(name)
            if env is not None and ty is None and name in self.type_nodes:
                ty = self.resolve(Node(owner.token, "type_name", (name,)), env)
        if keyed:
            return ty.key_type if isinstance(ty, Record) else None
        return ty if isinstance(ty, Scalar) and ty.enum is not None else None

    @staticmethod
    def enum_member_value(node, ty):
        name = "len" if node.kind == "length" else node.args[1].text
        for member, value in ty.enum.members:
            if member == name and ty.domain & Domain.point(value):
                return value
        node.token.error(f"unknown member {name!r} of enum {ty.enum.name}")

    def place(self, node):
        if node.kind == "name":
            name, = node.args
            if (not any(name in scope for scope in reversed(self.scope))
                    and name not in self.module_consts
                    and self._singleton_type_constant(name) is not None):
                return None
            return self.lookup(node)
        if node.kind in ("member", "length") and self.enum_member_type(node) is None:
            parent = self.place(node.args[0])
            if (parent is not None and isinstance(self.alias_types.get(parent.uid), BorrowScalar)
                    and node.kind == "member" and node.args[1].text == "value"):
                return parent
            if parent is not None and (isinstance(parent.ty, RecordLink) or parent.link_access is not None
                                       and isinstance(parent.ty, Record)):
                name = "len" if node.kind == "length" else node.args[1].text
                linked = isinstance(parent.ty, RecordLink)
                record = self.link_record(parent.ty, node.token) if linked else parent.ty
                field = dict(record.fields).get(name)
                if field is None:
                    node.token.error(f"unknown record field {name!r}")
                op = "->" if linked else "."
                frozen = parent.readonly or linked and parent.ty.readonly or isinstance(field, ReadOnly)
                return Symbol(name, parent.uid, unqualified(field), parent.depth, f"({parent.c}){op}{c_field(name)}",
                              readonly=frozen, borrowed=parent.borrowed, link_access=(parent, name))
            if parent is not None and isinstance(parent.ty, (Record, Interface)):
                name = "len" if node.kind == "length" else node.args[1].text
                for field in parent.fields:
                    if field.name == name:
                        return field
                node.token.error(f"unknown record field {name!r}")
        return None

    def record_place(self, node):
        sym = self.place(node)
        return sym if sym is not None and isinstance(sym.ty, Record) else None

    def field_place(self, node):
        if node.kind in ("member", "length"):
            return self.place(node)
        return None

    def record_symbols(self, sym, token, zero=True):
        self.spend(token)
        if isinstance(sym.ty, (Record, Interface)):
            children = []
            for index, (name, ty) in enumerate(sym.ty.fields):
                readonly = sym.readonly or isinstance(ty, ReadOnly)
                if isinstance(unqualified(ty), Interface):
                    readonly = readonly or unqualified(ty).readonly
                code = f"{sym.c}.{c_field(name)}"
                if isinstance(sym.ty, Interface):
                    base = unqualified(ty)
                    const = "const " if readonly else ""
                    code = f"(*(({const}{base.c} *)((const char *){sym.c}.data + {sym.c}.offsets[{index}])))"
                field = Symbol(name, len(self.symbols) + 1, unqualified(ty), sym.depth,
                               code, sym.heap_owner, readonly, sym.borrowed,
                               static_storage=sym.static_storage, chain_summary=sym.chain_summary,
                               interface_offsets=(f"lan_field_offsets_{sym.uid}_{name}"
                                                  if isinstance(unqualified(ty), Interface) else None))
                self.symbols.append(field)
                if sym.uid in self.alias_members:
                    root, names = self.alias_members[sym.uid]
                    self.alias_members[field.uid] = (root, names + (name,))
                elif isinstance(field.ty, Interface):
                    self.alias_members[field.uid] = (field.uid, ())
                children.append(field)
                self.trace.append(f"  field id#{field.uid} {name} of id#{sym.uid}")
                self.record_symbols(field, token, zero)
            sym.fields.extend(children)
        elif isinstance(sym.ty, RecordLink):
            return
        elif zero and isinstance(sym.ty, Slice):
            if not ZERO.within(sym.ty.domain):
                token.error("default string field length does not fit its contract")
        elif zero:
            scalar = sym.ty.element if isinstance(sym.ty, Array) else sym.ty
            self.range_check(token, Value(ZERO, "0", boolean=scalar.boolean, enum=scalar.enum, character=scalar.character), scalar)

    @staticmethod
    def named_record_leaves(sym, prefix=()):
        if isinstance(sym.ty, Record):
            for child in sym.fields:
                yield from Compiler.named_record_leaves(child, prefix + (child.name,))
        else:
            yield prefix, sym

    @staticmethod
    def storage_leaves(sym):
        if isinstance(sym.ty, (Record, Interface)):
            for child in sym.fields:
                if isinstance(child.ty, Interface):
                    continue
                yield from Compiler.storage_leaves(child)
        else:
            yield sym

    def initialize_interface_parameters(self, root, path, token, owning=False):
        """Model a borrowed interface field as an opaque target.

        The callee cannot know which object the field selects, so reads fall
        back to the declared schema domains and writes through the field are
        rejected.  Normal fields of the same record keep their usual facts.
        """
        for field in root.fields:
            if not isinstance(field.ty, Interface) or field.uid in path.borrows:
                continue
            # An owning parameter holds the record but not the object its
            # interface selects, so the callee may read through the field and
            # nothing more: a write would change a caller object with no place
            # to record it.
            target = Symbol(f"{field.name}$target", len(self.symbols) + 1, field.ty.schema,
                            root.depth, readonly=field.ty.readonly or owning)
            self.symbols.append(target)
            self.opaque_interface_targets.add(target.uid)
            self.record_symbols(target, token, zero=False)
            self.opaque_interface_targets.update(child.uid for child in target.fields)
            if owning:
                self.owned_interface_targets.add(target.uid)
                self.owned_interface_targets.update(child.uid for child in target.fields)
            # A borrowed record chains its fields to the parameter root; the
            # interface selects its own target instead, so it becomes a root.
            self.alias_members[field.uid] = (field.uid, ())

            def declared_alias(node, names):
                self.alias_members[node.uid] = (field.uid, names)
                for child in node.fields:
                    declared_alias(child, names + (child.name,))

            for child in field.fields:
                declared_alias(child, (child.name,))
            # Writes through the field record a step through the reference, so a
            # caller resolves the same path against the objects it knows about.
            # An owning heap parameter has no borrowed place of its own: the
            # interface still selects an object outside the allocation.
            index, prefix = self.param_places.get(root.uid, (None, ()))
            if index is not None:
                self.register_borrow(target, index, prefix + (field.name, "*"))
            path = path.update(target, ZERO)
            path = self.initialize_record_parameter(target, path)
            path = replace(path, borrows=path.borrows | {field.uid: frozenset((target,))})
            self.trace.append(f"  interface id#{field.uid} {field.name}: opaque target id#{target.uid}")
        return path

    def define_heap_object(self, name, ty, node, paths, readonly):
        if isinstance(self.composite_call_type(node), HeapObject):
            paths, actual, code = self.initializer_calls(node, paths)
            if actual != HeapObject(ty):
                node.token.error("owned result requires a matching heap type")
            if not paths:
                return [], [f"(void)({code});", "__builtin_unreachable();"]
            sym = self.declare(name, ty, heap=True, readonly=readonly, ownership_origin="owned_result")
            if isinstance(ty, Scalar):
                paths = [path.update(sym, ty.domain) for path in paths]
            paths = [replace(path, heap=path.heap | {sym.uid: frozenset(("Alive",))}) for path in paths]
            self.trace.append(f"  define entity id#{sym.uid} {sym.name}: owned result, Alive")
            return paths, [f"{ty.c} *{sym.c_var} = {code};"]
        if isinstance(ty, Scalar):
            ty = physical(ty, name)
            values, continuing = [], []
            error_code = None
            for path in paths:
                value = (self.expr(node, path) if node else
                         Value(zero_domain(ty), "0", boolean=ty.boolean, enum=ty.enum,
                               character=ty.character, signed=ty.signed))
                if value.always_returns:
                    if error_code is None or self.propagation_code_rank(value.code) > self.propagation_code_rank(error_code):
                        error_code = value.code
                    continue
                value = self.coerce_storage(name, value, ty)
                self.range_check(name, value, ty)
                continuing.append(path)
                values.append(value)
            if not continuing:
                return [], [f"(void)({error_code});", "__builtin_unreachable();"]
            paths = continuing
            sym = self.declare(name, ty, heap=True, readonly=readonly, ownership_origin="allocate")
            paths = [p.update(sym, v.domain, v.affine, v.difference, v.linear) for p, v in zip(paths, values)]
            pointer = sym.c_var
            alloc = self.allocating_helper("heap_array")
            code = [f"{ty.c} lan_init_{sym.uid} = {values[0].code};",
                    f"{ty.c} *{pointer} = ({ty.c} *){alloc}(1, sizeof({ty.c})).data;",
                    f"{sym.c} = lan_init_{sym.uid};"]
        else:
            sym = self.declare(name, ty, heap=True, readonly=readonly, ownership_origin="allocate")
            pointer = sym.c_var
            alloc = self.allocating_helper("heap_array")
            allocation = (f"uint8_t (*{pointer})[{ty.c_length}] = "
                          f"(uint8_t (*)[{ty.c_length}]){alloc}({ty.c_length}, 1).data;")
            # Initializer expressions must resolve names before the new binding.
            self.scope[-1].pop(name.text)
            init = replace(sym, readonly=False)
            try:
                if node is None or node.kind in ("list", "repeat"):
                    facts = ([(0, 0)] * len(paths)) if node is not None and ty.length <= 4096 else None
                    continuing = [] if self.has_propagation(node) else None
                    if continuing is not None:
                        self.propagating_expression_nodes |= self.propagation_nodes(node, composite_literal=True)
                    code = self.array_init(init, node, paths, name, facts, allocation,
                                           continuing_paths=continuing)
                    if continuing is not None:
                        paths = continuing
                    if facts is not None:
                        paths = [p.with_bits(sym, value) for p, value in zip(paths, facts)]
                    elif node is None and ty.length <= 4096:
                        paths = [p.with_bits(sym, ((1 << ty.length) - 1, 0)) for p in paths]
                else:
                    if ty.length <= 4096:
                        paths = [p.with_bits(sym, ((1 << ty.length) - 1, 0)) for p in paths]
                    paths, code = self.packed_init(init, node, paths, name, declare_storage=False,
                                                   allocation=allocation)
            finally:
                self.scope[-1][name.text] = sym
        if not paths:
            return [], code
        self.trace.append(f"  heap id#{sym.uid} {sym.name}: Alive")
        return [replace(p, heap=p.heap | {sym.uid: frozenset(("Alive",))}) for p in paths], code

    def static_scalar(self, ty, node, token):
        if node is None:
            value = Value(zero_domain(ty), "0", boolean=ty.boolean, enum=ty.enum,
                          character=ty.character, signed=ty.signed)
        else:
            def validate(expr):
                if self.enum_member_type(expr) is not None:
                    return
                if expr.kind in ("name", "member", "length", "index"):
                    if self._active_runtime_constant(expr) is not None:
                        return
                    if expr.kind == "name" and (expr.args[0] in self.module_consts or expr.args[0] in self.const_nodes):
                        self.resolve_module_const(expr.args[0])
                        return
                    if expr.kind == "name" and self._singleton_type_constant(expr.args[0]) is not None:
                        return
                    if expr.kind == "length" and (expr.args[0].kind == "text_literal"
                                                  or bit_literal_value(expr.args[0]) is not None):
                        return
                    if expr.kind in ("member", "length", "index"):
                        root = expr
                        while root.kind in ("member", "length", "index"):
                            if root.kind == "index":
                                validate(root.args[1])
                            root = root.args[0]
                        if root.kind == "name" and (root.args[0] in self.module_consts or root.args[0] in self.const_nodes):
                            self.resolve_module_const(root.args[0])
                            return
                    expr.token.error("rodata initializer requires a compile-time constant")
                if expr.kind not in ("number", "bit_literal", "float_number", "character", "boolean", "unary", "binary", "cast", "in"):
                    expr.token.error("rodata initializer requires a compile-time constant")
                operands = expr.args[:1] if expr.kind in ("cast", "in") else expr.args
                for part in operands:
                    if isinstance(part, Node):
                        validate(part)

            validate(node)
            if self.has_effects(node):
                node.token.error("rodata initializer cannot have runtime effects")
            active = getattr(self, "_active_paths", ())
            if active:
                evaluated = []
                agreed = None
                for path in active:
                    candidate = self.expr(node, path)
                    c = self.constant_from_value(token, candidate)
                    if agreed is None:
                        agreed = c
                    elif type(agreed) is not type(c) or agreed != c:
                        node.token.error("rodata initializer requires a compile-time constant")
                    evaluated.append(candidate)
                value = evaluated[0]
            else:
                value = self.expr(node, self.module_const_entry_state())
        if ty is None:
            if value.floating:
                token.error("packed bit initializer requires 0 or 1")
            self.require_bit(token, value, node)
        else:
            value = self.coerce_storage(token, value, ty)
            self.range_check(token, value, ty)
        constant = self.constant_from_value(token, value)
        return value.domain, c_literal(constant, ty.bits) if value.floating else c_int(int(constant))

    @staticmethod
    def record_initializer_fields(ty, node):
        if node is None:
            return [(name, unqualified(field), None) for name, field in ty.fields]
        if node.kind != "record_initializer":
            node.token.error("expected a record field initializer")
        fields = dict(ty.fields)
        seen = set()
        result = []
        for name, value in node.args[0]:
            if name.text not in fields:
                name.error(f"unknown initializer field {name.text!r}")
            if name.text in seen:
                name.error(f"duplicate initializer field {name.text!r}")
            seen.add(name.text)
            result.append((name.text, unqualified(fields[name.text]), value))
        result.extend((name, unqualified(field), None) for name, field in ty.fields if name not in seen)
        return result

    def static_array(self, ty, node, token):
        from composite_array import RecordValue, TreeArray
        if isinstance(ty, Scalar):
            return self.static_scalar(ty, node, token)
        if isinstance(ty, Slice):
            if node is None:
                node = Node(token, "text_literal", (b"",))
            if node.kind != "text_literal":
                node.token.error("rodata string field requires a string literal")
            expected_enc = ty.text if isinstance(ty.text, TextEncoding) else None
            span = self.check_span_contract(
                node.token, self.ordinary_span(node, PathState({}), expected_encoding=expected_enc), ty)
            return span.length.domain, f"{{{span.c_data()}, {span.length.domain.lo}}}"
        if isinstance(ty, RecordLink):
            if node is not None and node.kind != "null":
                node.token.error("rodata record link requires null initialization")
            return frozenset((self.null_symbol,)), "NULL"
        if isinstance(ty, Record):
            contents = {}
            initializers = []
            for name, field, value in self.record_initializer_fields(ty, node):
                content, code = self.static_array(field, value, token)
                contents[name] = content
                initializers.append(f".{c_field(name)} = {code}")
            fields = tuple((name, contents[name]) for name, _ in ty.fields)
            return RecordValue(fields), "{" + ", ".join(initializers) + "}"
        if isinstance(ty, Array) and ty.encoding is not None and node is not None and node.kind == "text_literal":
            raw = ty.encoding.encode_fixed(node.args[0].decode("utf-8"), ty.length, node.token)
            content = ArrayDomain.from_values(ty.length, ty.element.domain, [Domain.point(b) for b in raw])
            initializer = "{" + ", ".join(f"0x{b:02x}" for b in raw) + "}"
            return content, initializer
        literal = bit_literal_value(node) if node is not None and ty.packed_bits else None
        if literal is not None:
            data, width = literal
            require_bit_literal_width(node.token, ty.length, width)
            content = self.bit_literal_contents(data, width)
            return content, bit_literal_bytes(data, width)
        if node is None:
            elements, repeat = (None,), True
        elif node.kind == "list":
            elements, = node.args
            repeat = False
            if len(elements) != ty.length:
                node.token.error(f"array initializer requires {ty.length} elements")
        elif node.kind == "repeat":
            element, length_node = node.args
            length = self.const_int(length_node, ()) if isinstance(length_node, Node) else length_node
            if length != ty.length:
                node.token.error(f"array initializer requires {ty.length} elements")
            elements, repeat = (element,), True
        else:
            node.token.error("rodata array requires a constant list or repeat initializer")

        if ty.packed_bits:
            values = []
            for element in elements:
                if element is None:
                    values.append(0)
                    continue
                domain, _ = self.static_scalar(None, element, element.token)
                values.append(domain.lo)
            if repeat:
                byte = "255" if values[0] else "0"
                initializer = "{0}" if not values[0] else f"{{[0 ... {ty.c_length - 1}] = {byte}}}"
            else:
                data = [sum(bit << offset for offset, bit in enumerate(values[start:start + 8]))
                        for start in range(0, len(values), 8)]
                initializer = "{" + ", ".join(str(byte) for byte in data) + "}"
            packed = (TreeArray.filled(ty.length, BOOL, Domain.point(values[0])) if repeat else
                      TreeArray.from_values(ty.length, BOOL, (Domain.point(bit) for bit in values)))
            return packed, initializer

        values, codes = zip(*(self.static_array(ty.element, element, token) for element in elements))
        if isinstance(ty.element, Scalar):
            content = (ArrayDomain.filled(ty.length, ty.element.domain, values[0]) if repeat else
                       ArrayDomain.from_values(ty.length, ty.element.domain, values))
        else:
            content = (TreeArray.filled(ty.length, ty.element.domain, values[0]) if repeat else
                       TreeArray.from_values(ty.length, ty.element.domain, values))
        if node is None:
            initializer = "{0}"
        elif repeat:
            if codes[0] in ("0", "((lan_int)UINT64_C(0))", "((lan_uint)UINT64_C(0))"):
                initializer = "{0}"
            elif ty.length <= 256:
                initializer = "{" + ", ".join([codes[0]] * ty.length) + "}"
            else:
                initializer = f"{{[0 ... {ty.length - 1}] = {codes[0]}}}"
        else:
            initializer = "{" + ", ".join(codes) + "}"
        return content, initializer

    def _apply_record_const_content(self, sym, content, path):
        by_name = dict(content.fields)
        for child in sym.fields:
            val = by_name[child.name]
            if isinstance(child.ty, Scalar):
                path = path.update(child, val)
            elif isinstance(child.ty, Record):
                path = self._apply_record_const_content(child, val, path)
            elif isinstance(child.ty, Array):
                if child.ty.packed_bits:
                    mask, data = 0, 0
                    if child.ty.length <= 4096:
                        mask = (1 << child.ty.length) - 1
                        for part in val.segments:
                            if part.domain.lo:
                                data |= ((1 << (part.stop - part.start)) - 1) << part.start
                    path = path.with_bits(child, (mask, data))
                else:
                    path = path.with_array(child, val)
            elif isinstance(child.ty, RecordLink):
                path = replace(path, links=path.links | {child.uid: val})
            elif isinstance(child.ty, Slice):
                path = path.update(child, val)
        return path

    def apply_module_const(self, entry, path):
        sym, ty, content = entry.sym, entry.ty, entry.content
        if isinstance(ty, Scalar):
            return path.update(sym, content)
        if isinstance(ty, Slice):
            return path.bind_span(sym, entry.span).with_array(sym, entry.array_content)
        if isinstance(ty, Array):
            if ty.packed_bits:
                return path.with_bits(sym, entry.bits)
            return path.with_array(sym, content)
        if isinstance(ty, Record):
            return self._apply_record_const_content(sym, content, path)
        return path

    def module_const_entry_state(self):
        path = PathState({})
        for entry in self.module_consts.values():
            path = self.apply_module_const(entry, path)
        return path

    def eval_top_const(self, node):
        """Evaluate a module constant's initial compile-time value in top-down source order."""
        name_tok, annotation, init_node = node.args
        name = name_tok.text
        if name in self.const_compile_values or name in self.module_consts:
            return
        declared_ty = None
        if annotation is not None:
            ty = self.resolve(annotation)
            while isinstance(ty, (StaticReadOnly, ReadOnly)):
                ty = ty.base
            if isinstance(ty, Scalar):
                ty = physical(ty, name_tok)
            elif isinstance(ty, Slice):
                ty = replace(ty, readonly=True)
            elif not isinstance(ty, (Array, Record)):
                name_tok.error("module constants require a scalar, fixed array, record or string type")
            declared_ty = ty
        if declared_ty is None:
            if init_node is None:
                name_tok.error("module constant without a type requires an initializer")
            if init_node.kind in ("text_literal", "character"):
                return
            if init_node.kind == "member" and (enum := self.enum_member_type(init_node)) is not None:
                self.const_compile_values[name] = (self.enum_member_value(init_node, enum), enum)
                return
            try:
                val = self.const_expr(init_node, ())
            except LanError:
                return
            if isinstance(val, (int, bool, FloatConstant, Fraction, JsonMap, ConstList)):
                if isinstance(val, Fraction) and not isinstance(val, FloatConstant):
                    val = self.constant_from_value(name_tok, self.float_coerce(name_tok, self.constant_value(val), 64))
                self.const_compile_values[name] = (val, None)
            return
        if isinstance(declared_ty, Scalar):
            if init_node is None:
                default = (FloatConstant(0.0, declared_ty.bits) if declared_ty.floating
                           else (False if declared_ty.boolean else 0))
                self.const_compile_values[name] = (default, declared_ty)
                return
            try:
                val = self.const_expr(init_node, ())
                val = self.check_constant(name_tok, val, declared_ty)
                self.const_compile_values[name] = (val, declared_ty)
            except LanError:
                return
        elif isinstance(declared_ty, Array) and isinstance(declared_ty.element, Scalar) and declared_ty.encoding is None:
            if init_node is None:
                default = (FloatConstant(0.0, declared_ty.element.bits) if declared_ty.element.floating
                           else (False if declared_ty.element.boolean else 0))
                self.const_compile_values[name] = (ConstList((default,) * declared_ty.length), declared_ty)
                return
            if init_node.kind in ("list", "repeat", "name", "binary", "call"):
                try:
                    val = self.const_expr(init_node, ())
                    val = self.check_constant(name_tok, val, declared_ty)
                    self.const_compile_values[name] = (val, declared_ty)
                except LanError:
                    return
        elif isinstance(declared_ty, Slice) and declared_ty.text is None and isinstance(declared_ty.element, Scalar):
            if init_node is not None and init_node.kind in ("list", "repeat", "name", "binary", "call"):
                try:
                    val = self.const_expr(init_node, ())
                    val = self.check_constant(name_tok, val, declared_ty)
                    self.const_compile_values[name] = (val, declared_ty)
                except LanError:
                    return

    def _materialize_compile_const(self, name, name_tok, val, declared_ty):
        """Materialize a module constant from its final compile-time value."""
        from composite_array import TreeArray
        if isinstance(val, JsonMap):
            entry = ModuleConst(None, val, val, name=name)
            self.module_consts[name] = entry
            self.trace.append(f"const {name}: compile-time json map ({len(val.entries)} entries)")
            return entry
        prefix = "lan_const" if self.module_id is None else f"lan_m{self.module_id}_const"
        c_name = f"{prefix}_{c_ident(name)}"
        uid = len(self.symbols) + 1
        if isinstance(val, ConstList):
            if declared_ty is None:
                if not val.items:
                    name_tok.error("untyped empty list constant requires an explicit type")
                if all(type(item) is bool for item in val.items):
                    elem_ty = Scalar(BOOL, 8, boolean=True)
                elif any(isinstance(item, (FloatConstant, Fraction)) for item in val.items):
                    elem_ty = Scalar(FloatDomain.full(64), 64, floating=True)
                elif all(type(item) is int for item in val.items):
                    elem_ty = physical(Scalar(Domain.of(*((v, v) for v in val.items))), name_tok)
                else:
                    name_tok.error("list constant requires homogeneous scalar elements")
                ty = Array(elem_ty, len(val.items))
            else:
                ty = declared_ty
            checked = self.check_constant(name_tok, val, ty)
            if isinstance(ty, Array):
                if ty.packed_bits:
                    values = [int(v) for v in checked.items]
                    data = [sum(bit << offset for offset, bit in enumerate(values[start:start + 8]))
                            for start in range(0, len(values), 8)]
                    c_init = "{" + ", ".join(str(byte) for byte in data) + "}"
                    content = TreeArray.from_values(ty.length, BOOL, (Domain.point(bit) for bit in values))
                    mask = (1 << ty.length) - 1 if ty.length <= 4096 else 0
                    bits_data = sum(bit << i for i, bit in enumerate(values)) if ty.length <= 4096 else 0
                    sym = Symbol(name, uid, ty, 0, c_name, 0, readonly=True, borrowed=False, static_storage=True,
                                 semantic_binding=self.semantic_module.declarations.get(name_tok))
                    self.symbols.append(sym)
                    self.module_const_code.append(f"static const {ty.c} {c_name} = {c_init};")
                    entry = ModuleConst(sym, ty, content, bits=(mask, bits_data), c_init=c_init, name=name)
                    self.module_consts[name] = entry
                    self.trace.append(f"const id#{sym.uid} {name}: rodata")
                    self.trace_content(f"id#{sym.uid}", ty, content)
                    return entry
                domains = [FloatDomain.point(v, ty.element.bits) if ty.element.floating else Domain.point(int(v))
                           for v in checked.items]
                c_elems = [c_literal(v, ty.element.bits) if ty.element.floating else c_int(int(v))
                           for v in checked.items]
                content = ArrayDomain.from_values(ty.length, ty.element.domain, domains)
                c_init = "{" + ", ".join(c_elems) + "}"
                sym = Symbol(name, uid, ty, 0, c_name, 0, readonly=True, borrowed=False, static_storage=True,
                             semantic_binding=self.semantic_module.declarations.get(name_tok))
                self.symbols.append(sym)
                self.module_const_code.append(f"static const {ty.c} {c_name} = {c_init};")
                entry = ModuleConst(sym, ty, content, c_init=c_init, name=name)
                self.module_consts[name] = entry
                self.trace.append(f"const id#{sym.uid} {name}: rodata")
                self.trace_content(f"id#{sym.uid}", ty, content)
                return entry
            if isinstance(ty, Slice):
                n = len(checked.items)
                ty = replace(ty, length_bound=Domain.point(n), readonly=True)
                domains = [FloatDomain.point(v, ty.element.bits) if ty.element.floating else Domain.point(int(v))
                           for v in checked.items]
                c_elems = [c_literal(v, ty.element.bits) if ty.element.floating else c_int(int(v))
                           for v in checked.items]
                data_name = f"{c_name}_data"
                if n == 0:
                    self.module_const_code.append(f"static const {ty.element.c} {data_name}[1] = {{0}};")
                else:
                    self.module_const_code.append(f"static const {ty.element.c} {data_name}[{n}] = {{{', '.join(c_elems)}}};")
                c_init = f"{{{data_name}, {n}}}"
                sym = Symbol(name, uid, ty, 0, c_name, 0, readonly=True, borrowed=False, static_storage=True,
                             semantic_binding=self.semantic_module.declarations.get(name_tok))
                self.symbols.append(sym)
                self.module_const_code.append(f"static const {ty.c} {c_name} = {c_init};")
                span = Span(ty.element, Value(Domain.point(n), f"(lan_int){c_name}.len"), 0, code=c_name,
                            borrowed_from=frozenset((sym,)), readonly=True, text=ty.text, origins=((sym, ZERO),))
                array_content = (ArrayDomain.from_values(n, ty.element.domain, domains)
                                 if n else ArrayDomain.filled(0, ty.element.domain, ty.element.domain))
                entry = ModuleConst(sym, ty, Domain.point(n), span=span, array_content=array_content,
                                    c_init=c_init, name=name)
                self.module_consts[name] = entry
                self.trace.append(f"const id#{sym.uid} {name}: rodata")
                return entry
        # Scalar constant
        if declared_ty is None:
            if type(val) is bool:
                ty = Scalar(BOOL, 8, boolean=True)
            elif isinstance(val, FloatConstant):
                ty = Scalar(FloatDomain.full(val.bits), val.bits, floating=True)
            else:
                ty = physical(Scalar(Domain.point(val)), name_tok)
        else:
            ty = declared_ty
        val = self.check_constant(name_tok, val, ty)
        if ty.floating:
            content = FloatDomain.point(val, ty.bits)
            c_init = c_literal(val, ty.bits)
        else:
            content = Domain.point(int(val))
            c_init = c_int(int(val))
        sym = Symbol(name, uid, ty, 0, c_name, 0, readonly=True, borrowed=False, static_storage=True,
                     semantic_binding=self.semantic_module.declarations.get(name_tok))
        self.symbols.append(sym)
        self.module_const_code.append(f"static const {ty.c} {c_name} = {c_init};")
        entry = ModuleConst(sym, ty, content, scalar_value=val, c_init=c_init, name=name)
        self.module_consts[name] = entry
        self.trace.append(f"const id#{sym.uid} {name}: rodata")
        self.trace_content(f"id#{sym.uid}", ty, content)
        return entry

    def resolve_module_const(self, name):
        if name in self.module_consts:
            return self.module_consts[name]
        if name not in self.const_nodes:
            return None
        node = self.const_nodes[name]
        name_tok, annotation, init_node = node.args
        if name in self.const_compile_values:
            val, declared_ty = self.const_compile_values[name]
            return self._materialize_compile_const(name, name_tok, val, declared_ty)
        if name in self.resolving_consts:
            name_tok.error(f"cyclic constant definition involving {name!r}")
        self.resolving_consts.add(name)
        try:
            if annotation is not None:
                ty = self.resolve(annotation)
                while isinstance(ty, (StaticReadOnly, ReadOnly)):
                    ty = ty.base
                if isinstance(ty, Scalar):
                    ty = physical(ty, name_tok)
                elif isinstance(ty, Slice):
                    ty = replace(ty, readonly=True)
                elif not isinstance(ty, (Array, Record)):
                    name_tok.error("module constants require a scalar, fixed array, record or string type")
            else:
                if init_node is None:
                    name_tok.error("module constant without a type requires an initializer")
                if init_node.kind == "text_literal":
                    raw = init_node.args[0]
                    encoding = "ascii" if all(b < 128 for b in raw) else "utf8"
                    elem = Scalar(Domain.of((0, 127 if encoding == "ascii" else 255)), 8, character=encoding)
                    ty = Slice(elem, length_bound=Domain.point(len(raw)), readonly=True, text=encoding)
                else:
                    saved_scope = self.scope
                    self.scope = [{}]
                    try:
                        val = self.expr(init_node, self.module_const_entry_state())
                    finally:
                        self.scope = saved_scope
                    if val.boolean:
                        ty = Scalar(BOOL, 8, boolean=True)
                    elif val.floating:
                        ty = Scalar(FloatDomain.full(val.domain.bits), val.domain.bits, floating=True)
                    elif val.enum is not None:
                        ty = Scalar(val.domain, val.width, val.signed, enum=val.enum)
                    elif val.character is not None:
                        ty = Scalar(val.domain, val.width, val.signed, character=val.character)
                    else:
                        ty = physical(Scalar(val.domain), name_tok)
            saved_scope = self.scope
            self.scope = [{}]
            try:
                content, c_init = self.static_array(ty, init_node, name_tok)
            finally:
                self.scope = saved_scope
            prefix = "lan_const" if self.module_id is None else f"lan_m{self.module_id}_const"
            c_name = f"{prefix}_{c_ident(name)}"
            uid = len(self.symbols) + 1
            sym = Symbol(name, uid, ty, 0, c_name, 0, readonly=True, borrowed=False, static_storage=True,
                         semantic_binding=self.semantic_module.declarations.get(name_tok))
            self.symbols.append(sym)
            if isinstance(ty, Record):
                self.record_symbols(sym, name_tok, zero=False)
            scalar_value = None
            bits = None
            span = None
            array_content = None
            if isinstance(ty, Scalar):
                scalar_value = self.constant_from_value(
                    name_tok, Value(content, "", ty.bits, ty.boolean, enum=ty.enum, character=ty.character, signed=ty.signed))
            elif isinstance(ty, Array) and ty.packed_bits:
                mask, data = 0, 0
                if ty.length <= 4096:
                    mask = (1 << ty.length) - 1
                    for part in content.segments:
                        if part.domain.lo:
                            data |= ((1 << (part.stop - part.start)) - 1) << part.start
                bits = (mask, data)
            elif isinstance(ty, Slice):
                if init_node is None:
                    raw = b""
                elif isinstance(ty.text, TextEncoding):
                    raw = ty.text.encode(init_node.args[0].decode("utf-8"), init_node.token)
                else:
                    raw = init_node.args[0]
                span = Span(ty.element, Value(Domain.point(len(raw)), f"(lan_int){c_name}.len"), 0, code=c_name,
                            borrowed_from=frozenset((sym,)), readonly=True, text=ty.text, origins=((sym, ZERO),))
                array_content = (ArrayDomain.from_values(len(raw), ty.element.domain, [Domain.point(b) for b in raw])
                                 if raw else ArrayDomain.filled(0, ty.element.domain, ty.element.domain))
            self.module_const_code.append(f"static const {ty.c} {c_name} = {c_init};")
            entry = ModuleConst(sym, ty, content, scalar_value=scalar_value, bits=bits, span=span,
                                array_content=array_content, c_init=c_init, name=name)
            self.module_consts[name] = entry
            self.trace.append(f"const id#{sym.uid} {name}: rodata")
            self.trace_content(f"id#{sym.uid}", ty, content)
            return entry
        finally:
            self.resolving_consts.remove(name)

    def define_rodata(self, name, ty, node, paths):
        if isinstance(ty, Record):
            if node is not None:
                return self.define_initialized_record(name, ty, node, paths, readonly=True, static=True)
            return self.define_record(name, ty, None, paths, readonly=True, static=True)
        if isinstance(ty, Scalar):
            ty = physical(ty, name)
        content, code = self.static_array(ty, node, name)
        sym = self.declare(name, ty, readonly=True, static=True)
        if isinstance(ty, Scalar):
            paths = [p.update(sym, content) for p in paths]
        elif ty.packed_bits:
            mask, data = 0, 0
            if ty.length <= 4096:
                mask = (1 << ty.length) - 1
                for part in content.segments:
                    if part.domain.lo:
                        data |= ((1 << (part.stop - part.start)) - 1) << part.start
            paths = [p.with_bits(sym, (mask, data)) for p in paths]
        else:
            paths = [p.with_array(sym, content) for p in paths]
        self.trace.append(f"  define entity id#{sym.uid} {sym.name}: rodata")
        self.trace_content(f"id#{sym.uid}", ty, content)
        return paths, [f"static const {ty.c} {sym.c} = {code};"]

    def named_function(self, name):
        return name in self.functions or name in self.imported_functions or name in self.externs
