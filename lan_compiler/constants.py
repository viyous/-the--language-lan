"""Compile-time constants and type-function evaluation."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from itertools import count

from float_domain import FloatDomain, c_literal
from value_domain import ACTIVE_ORACLES, Domain

from .encoding import ConstList, EncodingTag, JsonMap, TextEncoding
from .errors import FloatConstant, FloatLiteral, LanError
from .syntax import Node, TypeArgument, TypeObject
from .parser import Parser
from .typesys import Array, bit_literal_value, contains_callable, HeapArray, Scalar, Slice, Value
from .domain_utils import c_int, trunc_div

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class ConstantEvaluationMixin:
    @staticmethod
    def compile_time_binding(name, env):
        for scope in reversed(env):
            if name in scope:
                return scope

    def type_parameter_binding(self, name, env):
        scope = self.compile_time_binding(name, env)
        if scope is not None and isinstance(value := scope[name][0], TypeArgument):
            return value

    def const_binding(self, name, env, token):
        scope = self.compile_time_binding(name, env)
        if scope is None:
            token.error(f"{name!r} is not a compile-time variable")
        if isinstance(scope[name][0], TypeArgument):
            token.error(f"type parameter {name!r} cannot be used as a compile-time value")
        return scope

    def _module_const_expr(self, node):
        """Evaluate a pure member/index expression rooted at a module constant."""
        root = node
        while root.kind in ("member", "length", "index"):
            root = root.args[0]
        if root.kind != "name" or (root.args[0] not in self.module_consts and root.args[0] not in self.const_nodes):
            return None
        self.resolve_module_const(root.args[0])
        if self.has_effects(node):
            return None
        saved_scope = self.scope
        self.scope = [{}]
        try:
            val = self.expr(node, self.module_const_entry_state())
            if val.always_returns or val.effectful:
                return None
            return (self.constant_from_value(node.token, val),)
        except LanError:
            return None
        finally:
            self.scope = saved_scope

    def _active_runtime_constant(self, node):
        """Try to evaluate a pure expression to a single compile-time constant across all active paths."""
        active = getattr(self, "_active_paths", ())
        if not active or not self.scope or self.has_effects(node):
            return None
        sentinel = object()
        agreed = sentinel
        for path in active:
            try:
                val = self.expr(node, path)
                if val.always_returns or val.effectful:
                    return None
                c = self.constant_from_value(node.token, val)
            except LanError:
                return None
            if agreed is sentinel:
                agreed = c
            elif type(agreed) is not type(c) or agreed != c:
                return None
        return None if agreed is sentinel else (agreed,)

    def _singleton_type_constant(self, name):
        """Check if `name` is a module-level singleton value-set type alias (e.g. `type N = {4};`)."""
        if name in self.types and (name not in self.type_object_nodes or name in self.type_nodes):
            ty = self.types[name]
        elif name in self.type_nodes and name not in self.resolving:
            try:
                ty = self.resolve(self.type_nodes[name])
            except LanError:
                return None
        else:
            return None
        if not isinstance(ty, Scalar) or ty.enum is not None:
            return None
        # Do not treat named physical full-width types (`uint8`, `boolean`, etc.) as constants unless singleton
        if ty.floating:
            domain = ty.domain
            if not domain.nan and domain.ranges and domain.lo == domain.hi:
                if domain.zeros == 3:
                    return None
                if domain.zeros in (1, 2) and len(domain.ranges) == 1:
                    return (FloatConstant(-0.0 if domain.zeros == 2 else 0.0, domain.bits),)
                return (FloatConstant(domain.lo, domain.bits),)
            return None
        if ty.domain.lo == ty.domain.hi and math.isfinite(ty.domain.lo):
            return (bool(ty.domain.lo) if ty.boolean else int(ty.domain.lo),)
        return None

    def const_int(self, node, env):
        value = self.const_expr(node, env)
        if type(value) is not int:
            node.token.error("expected a compile-time integer")
        return value

    def _const_map_key(self, node, env):
        val = self.const_expr(node, env)
        if isinstance(val, str):
            return val
        if type(val) is int:
            if not (0 <= val <= 0x10FFFF) or 0xD800 <= val <= 0xDFFF:
                node.token.error(f"Unicode code point {val:#x} is not a valid scalar value")
            return chr(val)
        node.token.error("mapping key must be a string or Unicode code point integer")

    @staticmethod
    def _const_map_bytes(token, value):
        if value is None:
            return None
        if type(value) is int:
            if not (0 <= value <= 255):
                token.error(f"mapping byte value {value} must be in [0, 255]")
            return bytes([value])
        if isinstance(value, ConstList):
            out = bytearray()
            for item in value.items:
                if type(item) is not int or not (0 <= item <= 255):
                    token.error(f"mapping byte list element {item!r} must be an integer in [0, 255]")
                out.append(item)
            return bytes(out)
        if isinstance(value, bytes):
            return value
        token.error("mapping value must be a byte (0..255), a byte list, or null")

    def call_const_type_function(self, name, args, caller_env, token):
        imported = self.imported_type_functions.get(name)
        owner, target = (imported.owner, imported.name) if imported else (self, name)
        node = owner.type_functions[target]
        params = node.args[1]
        if len(args) != len(params):
            token.error(f"type function {name!r} expects {len(params)} arguments")
        values = []
        for arg, (_, annotation) in zip(args, params):
            val = self.const_expr(arg, caller_env)
            if annotation.kind == "type_parameter":
                values.append(TypeArgument(val))
            else:
                values.append(val)
        values = tuple(values)
        if imported is None:
            result = self.evaluate_type(target, node, values, token)
        else:
            self.trace.append(f"  evaluate imported type {name} from {ascii(owner.filename)}")
            context = ACTIVE_ORACLES.set(owner.oracles)
            try:
                result = owner.evaluate_type(target, node, values, node.token)
            except LanError as exc:
                if exc.module is None:
                    exc.module = owner.source_path
                raise
            finally:
                ACTIVE_ORACLES.reset(context)
        return result.value if isinstance(result, TypeArgument) else result

    def const_expr(self, node, env):
        t, k, a = node.token, node.kind, node.args
        self.spend(t)
        if k in ("bit_literal", "unary", "binary") and (literal := bit_literal_value(node)) is not None:
            result = literal[0]
        elif k == "null":
            result = None
        elif k == "text_literal":
            result = a[0].decode("utf-8")
        elif k == "json_map":
            entries = {}
            for key_kind, key_node, val_node in a[0]:
                key = key_node.args[0] if key_kind == "literal" else self._const_map_key(key_node, env)
                if val_node.kind == "null":
                    bval = None
                else:
                    bval = self._const_map_bytes(val_node.token, self.const_expr(val_node, env))
                entries[key] = bval
            result = JsonMap(tuple(entries.items()))
        elif k == "record_initializer" and not a[0]:
            result = JsonMap(())
        elif k == "list":
            result = ConstList(tuple(self.const_expr(item, env) for item in a[0]))
        elif k == "repeat":
            elem = self.const_expr(a[0], env)
            n = self.const_int(a[1], env) if isinstance(a[1], Node) else a[1]
            if not 0 <= n <= 65536:
                t.error("compile-time repeat length must be between 0 and 65536")
            result = ConstList((elem,) * n)
        elif k == "call":
            callee, call_args = a
            if callee.kind != "name":
                t.error("expression is not supported at compile time")
            cname = callee.args[0]
            if cname == "chr":
                if len(call_args) != 1:
                    t.error("chr() expects 1 argument")
                code = self.const_int(call_args[0], env)
                if not (0 <= code <= 0x10FFFF) or 0xD800 <= code <= 0xDFFF:
                    t.error(f"Unicode code point {code:#x} is not a valid scalar value")
                result = chr(code)
            elif cname == "ord":
                if len(call_args) != 1:
                    t.error("ord() expects 1 argument")
                s = self.const_expr(call_args[0], env)
                if not isinstance(s, str) or len(s) != 1:
                    t.error("ord() requires a single-character string")
                result = ord(s)
            elif cname in self.type_functions or cname in self.imported_type_functions:
                result = self.call_const_type_function(cname, call_args, env, t)
            else:
                t.error(f"function {cname!r} is not supported at compile time")
        elif k == "float_number":
            result = FloatLiteral(a[0])
        elif k in ("number", "boolean", "character"):
            result = a[0]
        elif k == "name":
            scope = self.compile_time_binding(a[0], env)
            if scope is not None:
                bound = scope[a[0]][0]
                if isinstance(bound, TypeArgument):
                    if isinstance(bound.value, (JsonMap, ConstList, str, int, bool, FloatConstant, Fraction)):
                        result = bound.value
                    else:
                        t.error(f"type parameter {a[0]!r} cannot be used as a compile-time value")
                else:
                    result = bound
            elif (runtime_const := self._active_runtime_constant(node)) is not None:
                result = runtime_const[0]
            elif a[0] in self.const_compile_values:
                result = self.const_compile_values[a[0]][0]
            elif a[0] in self.module_consts or a[0] in self.const_nodes:
                entry = self.resolve_module_const(a[0])
                if isinstance(entry.ty, JsonMap):
                    result = entry.ty
                elif entry.scalar_value is None:
                    t.error(f"module constant {a[0]!r} is not a scalar compile-time value")
                else:
                    result = entry.scalar_value
            elif (type_const := self._singleton_type_constant(a[0])) is not None:
                result = type_const[0]
            else:
                t.error(f"{a[0]!r} is not a compile-time variable")
        elif k == "length":
            target = a[0]
            ct_val = None
            if target.kind == "name":
                scope = self.compile_time_binding(target.args[0], env)
                if scope is not None:
                    raw_bound = scope[target.args[0]][0]
                    ct_val = raw_bound.value if isinstance(raw_bound, TypeArgument) else raw_bound
                elif target.args[0] in self.const_compile_values:
                    ct_val = self.const_compile_values[target.args[0]][0]
            if isinstance(ct_val, JsonMap):
                result = len(ct_val.entries)
            elif isinstance(ct_val, ConstList):
                result = len(ct_val.items)
            elif isinstance(ct_val, str):
                result = len(ct_val)
            elif target.kind == "text_literal":
                result = len(target.args[0])
            elif (literal := bit_literal_value(target)) is not None:
                result = literal[1]
            elif (runtime_const := self._active_runtime_constant(node)) is not None:
                result = runtime_const[0]
            elif target.kind == "name" and (target.args[0] in self.module_consts or target.args[0] in self.const_nodes):
                entry = self.resolve_module_const(target.args[0])
                if isinstance(entry.ty, Array):
                    result = entry.ty.length
                elif isinstance(entry.ty, Slice) and entry.ty.domain.lo == entry.ty.domain.hi and math.isfinite(entry.ty.domain.lo):
                    result = int(entry.ty.domain.lo)
                else:
                    t.error("expected a compile-time length")
            else:
                place = self.place(target) if self.scope else None
                if place is not None and isinstance(place.ty, Array):
                    result = place.ty.length
                elif (place is not None and isinstance(place.ty, (Slice, HeapArray))
                      and place.ty.domain.lo == place.ty.domain.hi and math.isfinite(place.ty.domain.lo)):
                    result = int(place.ty.domain.lo)
                else:
                    t.error("expected a compile-time length")
        elif k == "member":
            enum = self.enum_member_type(node, env=env)
            if enum is not None:
                result = self.enum_member_value(node, enum)
            elif (runtime_const := self._active_runtime_constant(node)) is not None:
                result = runtime_const[0]
            elif (const_val := self._module_const_expr(node)) is not None:
                result = const_val[0]
            else:
                t.error("expected a compile-time enum member")
        elif k == "index":
            host_ct = None
            if a[0].kind == "name":
                scope = self.compile_time_binding(a[0].args[0], env)
                if scope is not None:
                    raw_bound = scope[a[0].args[0]][0]
                    host_ct = raw_bound.value if isinstance(raw_bound, TypeArgument) else raw_bound
                elif a[0].args[0] in self.const_compile_values:
                    host_ct = self.const_compile_values[a[0].args[0]][0]
            elif a[0].kind in ("list", "repeat", "json_map"):
                host_ct = self.const_expr(a[0], env)
            if isinstance(host_ct, ConstList):
                idx = self.const_int(a[1], env)
                if not 0 <= idx < len(host_ct.items):
                    t.error(f"compile-time index {idx} is out of bounds for length {len(host_ct.items)}")
                result = host_ct.items[idx]
            elif isinstance(host_ct, JsonMap):
                key = self._const_map_key(a[1], env)
                bval = host_ct.get(key)
                if bval is None:
                    t.error(f"key {key!r} is not present in mapping")
                result = bval[0] if len(bval) == 1 else ConstList(tuple(bval))
            elif a[0].kind == "text_literal":
                idx = self.const_int(a[1], env)
                data = a[0].args[0]
                if not 0 <= idx < len(data):
                    t.error(f"compile-time string index {idx} is out of bounds for length {len(data)}")
                result = data[idx]
            elif (runtime_const := self._active_runtime_constant(node)) is not None:
                result = runtime_const[0]
            elif (const_val := self._module_const_expr(node)) is not None:
                result = const_val[0]
            else:
                t.error("expression is not supported at compile time")
        elif k == "in":
            value = self.const_expr(a[0], env)
            target = self.numeric_type(a[1],env)
            if isinstance(value,(Fraction,FloatConstant)) or target.floating:
                bits = value.bits if isinstance(value,FloatConstant) else target.bits if target.floating else 64
                literal = value.text if isinstance(value,FloatLiteral) else value
                return FloatDomain.point(literal,bits).within(self.float_set(a[1],bits,env))
            return Domain.point(value).within(target.domain)
        elif k == "unary":
            op, operand = a
            if op == "!":
                return not self.const_expr(operand, env)
            value = self.const_expr(operand, env)
            if op == "~" and type(value) is not int:
                t.error("bitwise complement requires an integer")
            result = {"+": lambda: value, "-": lambda: -value, "~": lambda: ~value}[op]()
            if isinstance(value,FloatConstant):
                result = FloatConstant(result,value.bits)
            elif isinstance(value,FloatLiteral) and op == "-":
                result = FloatLiteral(value.text[1:] if value.text.startswith("-") else "-"+value.text)
        elif k == "binary":
            op, left, right = a
            if op == "&&":
                return bool(self.const_expr(left, env)) and bool(self.const_expr(right, env))
            if op == "||":
                return bool(self.const_expr(left, env)) or bool(self.const_expr(right, env))
            if op == "^^":
                return bool(self.const_expr(left, env)) != bool(self.const_expr(right, env))
            x, y = self.const_expr(left, env), self.const_expr(right, env)
            if op == "·":
                if isinstance(x, JsonMap) and isinstance(y, JsonMap):
                    result = x.merge(y)
                elif isinstance(x, ConstList) and isinstance(y, ConstList):
                    result = x.concat(y)
                elif isinstance(x, str) and isinstance(y, str):
                    result = x + y
                else:
                    t.error("compile-time '·' requires two mapping objects, two lists, or two strings")
            elif op in ("==", "!=", "===", "!==", "<", "<=", ">", ">="):
                if (type(x) is bool) != (type(y) is bool):
                    t.error("compile-time comparison mixes boolean and numeric values")
                if isinstance(x,(Fraction,FloatConstant)) or isinstance(y,(Fraction,FloatConstant)):
                    return self.const_float_binary(t,op,x,y)
                return {"==": x == y, "===": x == y, "!=": x != y, "!==": x != y,
                        "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
            else:
                result = self.const_binary(t, op, x, y)
        elif k == "cast":
            value = self.const_expr(a[0],env)
            result = self.constant_from_value(t,self.cast(t,self.constant_value(value),a[1],self.scalar_type(a[2],env)))
        else:
            t.error("expression is not supported at compile time")
        return self.limit_constant(t, result)

    @staticmethod
    def limit_constant(token, value):
        if type(value) is int and value.bit_length() > 4096:
            token.error("compile-time integer exceeds the 4096-bit computation limit")
        return value

    @staticmethod
    def constant_value(value):
        if isinstance(value,FloatConstant):
            return Value(FloatDomain.point(value,value.bits),c_literal(value,value.bits),value.bits)
        if isinstance(value,Fraction):
            text = value.text if isinstance(value,FloatLiteral) else str(value)
            return Value(FloatDomain.point(text,64),c_literal(text,64),literal=text)
        return Value(Domain.point(value),c_int(int(value)),8 if type(value) is bool else 0,type(value) is bool)

    @staticmethod
    def constant_from_value(token,value):
        if value.floating:
            domain = value.domain
            if domain.nan and not domain.ranges:
                return FloatConstant(math.nan,domain.bits)
            if domain.zeros in (1,2) and len(domain.ranges)==1:
                return FloatConstant(-0.0 if domain.zeros==2 else 0.0,domain.bits)
            if not domain.nan and domain.ranges and domain.lo==domain.hi:
                return FloatConstant(domain.lo,domain.bits)
        elif value.domain.lo==value.domain.hi:
            return bool(value.domain.lo) if value.boolean else value.domain.lo
        token.error("compile-time scalar expression does not determine one value")

    def const_float_binary(self,token,op,x,y):
        return self.constant_from_value(token,self.float_binary(token,op,self.constant_value(x),self.constant_value(y)))

    def const_binary(self, t, op, x, y):
        if isinstance(x,(Fraction,FloatConstant)) or isinstance(y,(Fraction,FloatConstant)):
            return self.const_float_binary(t,op,x,y)
        if type(x) is not int or type(y) is not int:
            t.error("compile-time arithmetic requires integers")
        if op in ("/", "%") and not y:
            t.error("division by zero at compile time")
        if op in ("<<", ">>") and not 0 <= y <= 4096:
            t.error("compile-time shift count must be between 0 and 4096")
        operators = {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y,
                     "/": lambda: trunc_div(x, y), "%": lambda: x - trunc_div(x, y) * y,
                     "&": lambda: x & y, "|": lambda: x | y, "^": lambda: x ^ y,
                     "<<": lambda: x << y, ">>": lambda: x >> y}
        return self.limit_constant(t, operators[op]())

    def check_constant(self, token, value, ty):
        if ty is None:
            if isinstance(value, (JsonMap, ConstList, str)):
                return value
            if isinstance(value, (FloatConstant, Fraction)):
                return self.constant_from_value(token, self.float_coerce(token, self.constant_value(value), 64))
            return value
        if isinstance(ty, Array):
            if not isinstance(value, ConstList):
                token.error("compile-time array value requires a list")
            if len(value.items) != ty.length:
                token.error(f"compile-time array length {len(value.items)} does not match expected {ty.length}")
            if isinstance(ty.element, Scalar):
                return ConstList(tuple(self.check_constant(token, item, ty.element) for item in value.items))
            return value
        if isinstance(ty, Slice):
            if not isinstance(value, ConstList):
                token.error("compile-time slice value requires a list")
            if not Domain.point(len(value.items)).within(ty.domain):
                token.error(f"compile-time slice length {len(value.items)} does not fit {ty.domain}")
            if isinstance(ty.element, Scalar):
                return ConstList(tuple(self.check_constant(token, item, ty.element) for item in value.items))
            return value
        if value is None or isinstance(value, (JsonMap, ConstList, str)):
            token.error("compile-time scalar value expected")
        if ty.floating:
            converted = self.float_coerce(token, self.constant_value(value), ty.bits)
            self.range_check(token, converted, ty)
            return self.constant_from_value(token, converted)
        if isinstance(value, (FloatConstant, Fraction)):
            token.error("float-to-integer conversion requires explicit rounding at compile time")
        if ty.boolean != (type(value) is bool) or not Domain.point(value).within(ty.domain):
            token.error(f"compile-time value {value} does not fit {ty.domain}")
        return value

    def const_block(self, stmts, parents, scope=None):
        with self.code_block_budget(stmts, is_type=True):
            return self.evaluate_const_block(stmts, parents, scope)

    def type_function_result(self, annotation, env):
        if annotation.kind in ("json_map", "list", "repeat", "number", "float_number", "boolean",
                               "character", "text_literal", "binary", "unary", "call", "index",
                               "length", "bit_literal", "cast"):
            return self.const_expr(annotation, env)
        if annotation.kind == "record_initializer" and not annotation.args[0]:
            return JsonMap(())
        if annotation.kind == "type_name":
            scope = self.compile_time_binding(annotation.args[0], env)
            if scope is not None:
                bound = scope[annotation.args[0]][0]
                if not isinstance(bound, TypeArgument):
                    return bound
                if isinstance(bound.value, (JsonMap, ConstList, str, int, bool, FloatConstant, Fraction)):
                    return bound.value
        if annotation.kind == "array" and annotation.args[0].kind == "type_name":
            scope = self.compile_time_binding(annotation.args[0].args[0], env)
            if scope is not None:
                bound = scope[annotation.args[0].args[0]][0]
                inner = bound.value if isinstance(bound, TypeArgument) else bound
                if isinstance(inner, (JsonMap, ConstList)):
                    host = Node(annotation.args[0].token, "name", (annotation.args[0].args[0],))
                    return self.const_expr(Node(annotation.token, "index", (host, annotation.args[1])), env)
        result = self.resolve(annotation, env)
        if result is None:
            annotation.token.error("type function return requires a type object, type or value set")
        if isinstance(result, (JsonMap, ConstList)):
            return result
        if isinstance(result, TypeObject):
            return self.materialize_type_object(result, env)
        if isinstance(result, Scalar) and not result.boolean:
            while annotation.kind == "type_group":
                annotation = annotation.args[0]
            if annotation.kind == "type_name" and self.type_parameter_binding(annotation.args[0], env) is not None:
                return result
            physical_result = result.enum is not None or result.character is not None or (
                result.bits and annotation.kind in ("type_object", "type_merge", "type_apply", "seal"))
            if not physical_result:
                return replace(result, computed=True)
        return result

    def evaluate_const_block(self, stmts, parents, scope):
        if scope is None:
            scope = {}
        env = (*parents, scope)
        for stmt in stmts:
            t, k, a = stmt.token, stmt.kind, stmt.args
            self.spend(t)
            if k == "define":
                name, annotation, initializer = a
                if name.text in scope:
                    name.error(f"duplicate definition of {name.text!r}")
                if annotation is not None and annotation.kind == "type_name" and annotation.args == ("type",):
                    ty = None
                else:
                    ty = self.resolve(annotation, env) if annotation else None
                    if annotation and not isinstance(ty, (Scalar, Array)):
                        annotation.token.error("compile-time local requires an integer, boolean or fixed array type")
                if initializer is None and ty is None:
                    name.error("a definition needs a type or an initializer")
                if initializer is not None:
                    value = self.const_expr(initializer, env)
                elif isinstance(ty, Array):
                    value = ConstList((False if isinstance(ty.element, Scalar) and ty.element.boolean else 0,) * ty.length)
                else:
                    value = False if ty.boolean else 0
                value = self.check_constant(name, value, ty)
                scope[name.text] = value, ty
            elif k == "assign":
                target, op, expr = a
                if target.kind == "index":
                    host, idx_node = target.args
                    if host.kind != "name":
                        target.token.error("compile-time index assignment requires a local variable")
                    hname, = host.args
                    binding = self.compile_time_binding(hname, env)
                    if binding is None:
                        host.token.error(f"{hname!r} is not a compile-time variable")
                    old_raw, ty = binding[hname]
                    was_type_arg = isinstance(old_raw, TypeArgument)
                    old = old_raw.value if was_type_arg else old_raw
                    if isinstance(old, JsonMap):
                        if op.text != "=":
                            op.error("mapping entry assignment uses '='")
                        key = self._const_map_key(idx_node, env)
                        bval = None if expr.kind == "null" else self._const_map_bytes(expr.token, self.const_expr(expr, env))
                        updated = old.with_entry(key, bval)
                    elif isinstance(old, ConstList):
                        idx = self.const_int(idx_node, env)
                        if not 0 <= idx < len(old.items):
                            idx_node.token.error(f"compile-time index {idx} is out of bounds for length {len(old.items)}")
                        val = self.const_expr(expr, env)
                        if op.text != "=":
                            val = self.const_binary(op, op.text[:-1], old.items[idx], val)
                        if isinstance(ty, Array):
                            val = self.check_constant(op, val, ty.element)
                        updated = old.with_index(idx, val)
                    else:
                        target.token.error("compile-time index assignment requires a mapping object or list variable")
                    binding[hname] = (TypeArgument(updated), ty) if was_type_arg else (updated, ty)
                    continue
                if target.kind != "name":
                    target.token.error("compile-time assignment requires a local variable")
                name, = target.args
                binding = self.compile_time_binding(name, env)
                if binding is None:
                    target.token.error(f"{name!r} is not a compile-time variable")
                old_raw, ty = binding[name]
                was_type_arg = isinstance(old_raw, TypeArgument)
                if was_type_arg and not isinstance(old_raw.value, (JsonMap, ConstList, str, int, bool, FloatConstant, Fraction)):
                    target.token.error(f"type parameter {name!r} cannot be used as a compile-time value")
                old = old_raw.value if was_type_arg else old_raw
                value = self.const_expr(expr, env)
                if op.text != "=":
                    value = self.const_binary(op, op.text[:-1], old, value)
                value = self.check_constant(op, value, ty)
                binding[name] = (TypeArgument(value), ty) if was_type_arg else (value, ty)
            elif k == "if":
                condition, yes, no = a
                flow = self.const_block(yes if self.const_expr(condition, env) else no, env)
                if flow:
                    return flow
            elif k == "while":
                condition, body = a
                while self.const_expr(condition, env):
                    flow = self.const_block(body, env)
                    if flow and flow[0] == "return":
                        return flow
            elif k == "interation":
                name, initial, annotation, count, step, body = a
                ty = self.iteration_type(annotation, env)
                start = self.check_constant(initial.token, self.const_expr(initial, env), ty)
                times = self.const_expr(count, env)
                if type(times) is not int or not 0 <= times <= self.types["uint64"].domain.hi:
                    count.token.error("interation count must fit uint64")
                stride = self.const_expr(step, env)
                if isinstance(stride, bool):
                    step.token.error("interation step must support +")
                if not ty.floating and isinstance(stride, (FloatConstant, Fraction)):
                    step.token.error("an integer interation variable requires an integer step")
                if ty.floating:
                    stride = self.check_constant(step.token, stride, self.types[f"float{ty.bits}"])
                loop_scope = {name.text: (start, ty)}
                for _ in range(times):
                    with self.code_block_budget(body, is_type=True):
                        self.spend(t)
                    flow = self.const_block(body, (*env, loop_scope))
                    if flow and flow[0] == "return":
                        return flow
                    value = self.const_binary(step.token, "+", loop_scope[name.text][0], stride)
                    loop_scope[name.text] = self.check_constant(step.token, value, ty), ty
            elif k == "return":
                if a[0] is None:
                    t.error("type function must return a type object, type or value set")
                return k, self.type_function_result(a[0], env)
            elif k == "continue":
                return k, None
            else:
                t.error("statement is not supported in a type function")
        return None

    @staticmethod
    def value_argument(node):
        if node.kind != "type_application_argument":
            node.token.error("expected a compile-time value expression")
        return node.args[0]

    def type_argument(self, node, env):
        annotation = node
        if node.kind == "type_application_argument":
            expr_node, tokens = node.args
            if expr_node.kind in ("json_map", "list", "repeat"):
                return TypeArgument(self.const_expr(expr_node, env))
            if expr_node.kind == "name":
                name = expr_node.args[0]
                scope = self.compile_time_binding(name, env)
                if scope is not None and isinstance(scope[name][0], (JsonMap, ConstList)):
                    return TypeArgument(scope[name][0])
                if name in self.const_compile_values and isinstance(self.const_compile_values[name][0], (JsonMap, ConstList)):
                    return TypeArgument(self.const_compile_values[name][0])
            parser = Parser((*tokens, replace(tokens[-1], text="<eof>")))
            try:
                annotation = parser.type_expr()
                parser.need("<eof>")
            except LanError:
                try:
                    val = self.const_expr(expr_node, env)
                    if isinstance(val, (JsonMap, ConstList)):
                        return TypeArgument(val)
                except LanError:
                    pass
                node.token.error("expected a type expression for a 'type' parameter")
            value = self.resolve(annotation, env)
            if isinstance(value, TypeObject):
                value = self.materialize_type_object(value, env)
            return TypeArgument(value)
        value = self.resolve(annotation, env)
        if isinstance(value, TypeObject):
            value = self.materialize_type_object(value, env)
        return TypeArgument(value)

    @staticmethod
    def type_cache_key(value):
        if isinstance(value, TypeArgument):
            if isinstance(value.value, (JsonMap, ConstList)):
                return TypeArgument, Compiler.type_cache_key(value.value)
            # Scalar equality omits result arms and exact float literals.
            return TypeArgument, Compiler.type_object_value(value.value, resolved=True)
        if isinstance(value, JsonMap):
            return JsonMap, value.entries
        if isinstance(value, ConstList):
            return ConstList, tuple(Compiler.type_cache_key(x) for x in value.items)
        if isinstance(value, FloatLiteral):
            return FloatLiteral, value.text
        if isinstance(value, FloatConstant):
            return FloatConstant, value.bits, value.hex()
        # Python treats True and 1 as equal, but Lan parameters distinguish them.
        return type(value), value

    def compute_type(self, name, args, caller_env, token):
        imported = self.imported_type_functions.get(name)
        owner, target = (imported.owner, imported.name) if imported else (self, name)
        node = owner.type_functions[target]
        params = node.args[1]
        if len(args) != len(params):
            token.error(f"type function {name!r} expects {len(params)} arguments")
        values = tuple(self.type_argument(arg, caller_env) if annotation.kind == "type_parameter"
                       else self.const_expr(self.value_argument(arg), caller_env)
                       for arg, (_, annotation) in zip(args, params))
        if imported is None:
            return self.evaluate_type(target, node, values, token)
        self.trace.append(f"  evaluate imported type {name} from {ascii(owner.filename)}")
        context = ACTIVE_ORACLES.set(owner.oracles)
        try:
            result = owner.evaluate_type(target, node, values, node.token)
        except LanError as exc:
            if exc.module is None:
                exc.module = owner.source_path
            raise
        finally:
            ACTIVE_ORACLES.reset(context)
        if contains_callable(result, owner.record_types):
            token.error("callable values are local to their Lan module")
        return result

    def evaluate_type(self, name, node, values, token):
        _, params, body = node.args
        key = name, tuple(map(self.type_cache_key, values))
        if key in self.type_cache:
            return self.type_cache[key]
        if key in self.active_types or len(self.active_types) >= 32:
            token.error(f"type function {name!r} exceeded the compile-time recursion limit")
        if not self.active_types:
            self.type_block_gas.clear()
        self.active_types.add(key)
        try:
            scope = {}
            for (param, annotation), value in zip(params, values):
                if annotation.kind == "type_parameter":
                    scope[param.text] = value, None
                    continue
                ty = self.resolve(annotation, (scope,))
                if not isinstance(ty, (Scalar, Array)):
                    annotation.token.error("type function parameter requires an integer or boolean type (or fixed array)")
                value = self.check_constant(param, value, ty)
                scope[param.text] = value, ty
            flow = self.const_block(body, (), scope)
            if flow is None or flow[0] != "return":
                node.args[0].error(f"type function {name!r} did not return a type or value set")
            result = flow[1]
            self.type_cache[key] = result
            return result
        finally:
            self.active_types.remove(key)

    def eval_expand(self, node):
        fn_tok, target_tok, extra_args = node.args
        fn_name, target_name = fn_tok.text, target_tok.text
        if target_name not in self.const_nodes:
            target_tok.error(f"expand target {target_name!r} must be a module-level constant declared in this module")
        if target_name not in self.const_compile_values:
            target_tok.error(f"module constant {target_name!r} is not initialized before expand or does not support expand")
        imported = self.imported_type_functions.get(fn_name)
        if imported is None and fn_name not in self.type_functions:
            fn_tok.error(f"unknown type function {fn_name!r} in expand")
        owner, fn_target = (imported.owner, imported.name) if imported else (self, fn_name)
        fn_node = owner.type_functions[fn_target]
        params = fn_node.args[1]
        if len(params) != 1 + len(extra_args):
            fn_tok.error(f"type function {fn_name!r} expects {len(params)} arguments, got {1 + len(extra_args)}")
        cur_val, declared_ty = self.const_compile_values[target_name]
        first_param_ann = params[0][1]
        first_arg = TypeArgument(cur_val) if first_param_ann.kind == "type_parameter" else cur_val
        rest_vals = tuple(
            self.type_argument(arg, ()) if annotation.kind == "type_parameter"
            else self.const_expr(self.value_argument(arg), ())
            for arg, (_, annotation) in zip(extra_args, params[1:])
        )
        values = (first_arg, *rest_vals)
        if imported is None:
            new_val = self.evaluate_type(fn_target, fn_node, values, fn_tok)
        else:
            self.trace.append(f"  expand {target_name} via imported type function {fn_name} from {ascii(owner.filename)}")
            context = ACTIVE_ORACLES.set(owner.oracles)
            try:
                new_val = owner.evaluate_type(fn_target, fn_node, values, fn_node.token)
            except LanError as exc:
                if exc.module is None:
                    exc.module = owner.source_path
                raise
            finally:
                ACTIVE_ORACLES.reset(context)
        if isinstance(new_val, TypeArgument):
            new_val = new_val.value
        if declared_ty is not None:
            new_val = self.check_constant(node.token, new_val, declared_ty)
        elif not isinstance(new_val, (int, bool, FloatConstant, Fraction, JsonMap, ConstList)):
            node.token.error(f"expand of {target_name!r} must return a compile-time constant value, list, or mapping object")
        self.const_compile_values[target_name] = (new_val, declared_ty)
