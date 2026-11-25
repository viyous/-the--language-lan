"""Algebraic properties and interface accessor inlining."""

from __future__ import annotations

from dataclasses import field

from value_domain import Domain

from .syntax import Node
from .typesys import Scalar, unqualified, Value, ZERO
from .domain_utils import plain_integer_parameter, trunc_div
from .state import PathState
from .interface_types import OFFSET_BINARY_OPS, square_domain
from .summaries import (guard_boolean_parameter, interface_accessor_member, INTERFACE_FIELD_OPERATORS,
                        interface_index_domain, interface_index_step, valid_interface_accessor)


class PropertyMixin:
    def property_term(self, expr, state, used, *, use_saved=False):
        if use_saved and expr.kind == "name":
            name, = expr.args
            if any(name in scope for scope in reversed(self.scope)):
                sym = self.lookup(expr)
                saved = state.predicates.get(("property_value", sym.uid, state.epochs.get(sym.uid)))
                if saved is not None:
                    name, term = saved
                    used.append(name)
                    return term
        if expr.kind != "call":
            if expr.kind == "cast" and expr.args[1] == "safe":
                source, _, annotation = expr.args
                target = self.resolve(annotation)
                if plain_integer_parameter(target):
                    operand = self.property_term(source, state, used, use_saved=use_saved)
                    if operand is not None:
                        return "cast", target, operand
            if expr.kind == "unary" and expr.args[0] == "~":
                source = expr.args[1]
                ty = None
                if source.kind == "name":
                    ty = self.lookup(source).ty
                elif source.kind == "cast" and source.args[1] == "safe":
                    ty = self.resolve(source.args[2])
                elif source.kind == "call":
                    contract = self.algebraic_contract(source.args[0])
                    ty = contract.result if contract is not None else None
                if plain_integer_parameter(ty):
                    ty = unqualified(ty)
                    mask = -1 if ty.signed or not ty.bits else (1 << ty.bits) - 1
                    operand = self.property_term(source, state, used, use_saved=use_saved)
                    if operand is not None:
                        return "complement", mask, operand
            if (expr.kind == "unary" and expr.args[0] in ("+", "-")
                    or expr.kind == "binary" and expr.args[0] in OFFSET_BINARY_OPS):
                operands = tuple(self.property_term(arg, state, used, use_saved=use_saved) for arg in expr.args[1:])
                if None not in operands:
                    operator = expr.args[0]
                    if expr.kind == "binary" and operator in ("+", "*", "&", "|", "^"):
                        operands = frozenset(operands)
                    return expr.kind, operator, operands
            key = self.predicate_key(expr, state)
            if key is not None:
                return "value", key
            return None
        callee, arguments = expr.args
        key = self.algebraic_function_key(callee)
        if key is None:
            return None
        _, contract = key
        if len(arguments) != len(contract.params):
            return None
        name = callee.args[0]
        if not used:
            used.append(name)
        keys = tuple(self.property_term(arg, state, used, use_saved=use_saved) for arg in arguments)
        if None in keys:
            return None
        if contract.identity_property:
            return keys[0]
        if contract.idempotent_property:
            inner = keys[0]
            return inner if inner[:2] == ("property", key) else ("property", key, inner)
        if contract.diagonal_property:
            return keys[0] if keys[0] == keys[1] else ("property", key, keys)
        return "property", key, frozenset(keys)

    def rewrite_property_binary(self, node, state):
        op, left, right = node.args
        if (op not in ("==", "!=", "===", "!==", "<", "<=", ">", ">=", "-", "^", "+", "*", "&", "|", "/", "%")
                or self.has_effects(node, allow_property_calls=True)):
            return None

        used = []

        shared = self.shared_property_state(node)
        terms_state = shared if shared is not None else state
        key = self.property_term(left, terms_state, used, use_saved=shared is not None)
        if key is None or key != self.property_term(right, terms_state, used, use_saved=shared is not None) or not used:
            return None
        left_type, right_type = self.expr(left, state), self.expr(right, state)
        if any(value.floating or value.boolean or value.enum or value.character for value in (left_type, right_type)):
            return None
        name = used[0]
        if op in ("/", "%"):
            common = left_type.domain & right_type.domain
            if not common or common & ZERO:
                return None
            value = int(op == "/")
            self.trace.append(f"  apply property {name}: cancel equivalent nonzero calls with {op}")
            return self.emit_value(node.token, Domain.point(value), str(value),
                                   max(left_type.width, right_type.width),
                                   signed=left_type.signed or right_type.signed)
        if op in ("+", "*", "&", "|"):
            common = left_type.domain & right_type.domain
            if not common:
                return None
            width = max(left_type.width, right_type.width)
            signed = left_type.signed or right_type.signed
            code = left_type.code
            domain = common
            if op in ("+", "*"):
                domain = common.affine(2, 0) if op == "+" else square_domain(common)
                helper = self.helper("unary_eval", None, ("lan_int", "lan_int", f"(a {op} a)"))
                code = f"{helper}({code})"
            self.trace.append(f"  apply property {name}: evaluate equivalent calls once for {op}")
            return self.emit_value(node.token, domain, code, width, signed=signed)
        if op in ("-", "^"):
            self.trace.append(f"  apply property {name}: cancel equivalent calls with {op}")
            x = Value(ZERO, "0", left_type.width, signed=left_type.signed)
            y = Value(ZERO, "0", right_type.width, signed=right_type.signed)
            return self.binary(node.token, op, x, y, state)
        self.trace.append(f"  apply property {name}: eliminate equivalent calls in comparison")
        value = int(op in ("==", "===", "<=", ">="))
        return Value(Domain.point(value), str(value), 8, True)

    def shared_property_state(self, node):
        context = self.property_codegen_context
        return context[1] if context is not None and id(node) in context[0] else None

    def shared_property_context(self, root, paths):
        if (root is None or self.has_effects(root, allow_property_calls=True)
                or not any(isinstance(key, tuple) and key[0] == "property_value"
                           for path in paths for key in path.predicates)):
            return None
        expressions, eligible = [root], set()
        while expressions:
            expr = expressions.pop()
            if isinstance(expr, Node):
                if expr.kind in ("binary", "call"):
                    eligible.add(id(expr))
                expressions.extend(expr.args)
            elif isinstance(expr, (tuple, list)):
                expressions.extend(expr)
        return (eligible, PathState.join(paths)) if eligible else None

    def interface_index_affine(self, expression, parameters, contract, depth=0, *, resolve_local=None):
        if depth >= 32:
            return None
        if resolve_local is not None:
            expression = resolve_local(expression)
        if expression.kind == "number" and type(expression.args[0]) is int:
            return {}, expression.args[0]
        if expression.kind == "name" and expression.args[0] in parameters:
            index = parameters[expression.args[0]]
            ty = contract.params[index]
            if plain_integer_parameter(ty):
                domain = unqualified(ty).domain
                if domain.lo == domain.hi:
                    return {}, domain.lo
            return {index: 1}, 0
        if expression.kind == "cast" and expression.args[1] == "safe":
            operand, _, annotation = expression.args
            if plain_integer_parameter(self.resolve(annotation)):
                return self.interface_index_affine(operand, parameters, contract, depth + 1,
                                                   resolve_local=resolve_local)
            return None
        if expression.kind == "unary" and expression.args[0] in ("+", "-", "~"):
            op, operand = expression.args
            inner = self.interface_index_affine(operand, parameters, contract, depth + 1,
                                                    resolve_local=resolve_local)
            if inner is None or op == "+":
                return inner
            terms, bias = inner
            mask = 0
            if op == "~":
                operand = resolve_local(operand) if resolve_local is not None else operand
                if operand.kind == "name" and operand.args[0] in parameters:
                    ty = contract.params[parameters[operand.args[0]]]
                elif operand.kind == "cast" and operand.args[1] == "safe":
                    ty = self.resolve(operand.args[2])
                else:
                    return None
                if not plain_integer_parameter(ty):
                    return None
                ty = unqualified(ty)
                mask = -1 if ty.signed or not ty.bits else (1 << ty.bits) - 1
            return {index: -scale for index, scale in terms.items()}, mask - bias
        if expression.kind == "length":
            access = self.interface_access_path(expression, parameters, contract, depth + 1,
                                                    resolve_local=resolve_local)
            if access is not None:
                length = interface_accessor_member(contract, *access)
                if isinstance(length, Scalar) and length.domain.lo == length.domain.hi:
                    return {}, length.domain.lo
            return None
        if expression.kind != "binary" or expression.args[0] not in (
                "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"):
            return None
        op, left, right = expression.args
        a = self.interface_index_affine(left, parameters, contract, depth + 1, resolve_local=resolve_local)
        b = self.interface_index_affine(right, parameters, contract, depth + 1, resolve_local=resolve_local)
        if a is None or b is None:
            return None
        terms, bias = a
        other, other_bias = b
        if op in ("&", "|", "^"):
            operation = {"&": int.__and__, "|": int.__or__, "^": int.__xor__}[op]
            if not terms and not other:
                return {}, operation(bias, other_bias)
            if not terms:
                terms, bias, other, other_bias = other, other_bias, terms, bias
            if other:
                return None
            if op != "&":
                if other_bias < 0 or any(scale % (1 << other_bias.bit_length()) for scale in terms.values()):
                    return None
                return terms, operation(bias, other_bias)
            # A positive mask selects fixed low bits, while a negative mask can clear them.
            low_mask = other_bias if other_bias >= 0 else ~other_bias
            period = 1 << low_mask.bit_length()
            if any(scale % period for scale in terms.values()):
                return None
            return (terms if other_bias < 0 else {}), bias & other_bias
        if op in ("/", "%"):
            if other or not other_bias:
                return None
            if not terms:
                quotient = trunc_div(bias, other_bias)
                return {}, quotient if op == "/" else bias - quotient * other_bias
            if any(scale % other_bias for scale in terms.values()):
                return None
            remainder = 0
            if bias % other_bias:
                # With divisible coefficients, only the numerator's sign can change the remainder.
                step = interface_index_step(dict(sorted(terms.items())), bias)
                domain = interface_index_domain(contract, step)
                if domain is None or domain.lo < 0 < domain.hi:
                    return None
                remainder = bias % abs(other_bias) if domain.lo >= 0 else -(-bias % abs(other_bias))
            if op == "%":
                return {}, remainder
            return {index: scale // other_bias for index, scale in terms.items()}, (bias - remainder) // other_bias
        if op in ("<<", ">>"):
            if other or not 0 <= other_bias < 128:
                return None
            if op == "<<":
                return {index: scale << other_bias for index, scale in terms.items()}, bias << other_bias
            divisor = 1 << other_bias
            if any(scale % divisor for scale in terms.values()):
                return None
            return {index: scale // divisor for index, scale in terms.items()}, bias >> other_bias
        if op == "*":
            if not terms:
                return {i: bias * scale for i, scale in other.items()}, bias * other_bias
            if not other:
                return {i: other_bias * scale for i, scale in terms.items()}, bias * other_bias
            return None
        sign = 1 if op == "+" else -1
        for i, scale in other.items():
            terms[i] = terms.get(i, 0) + sign * scale
        return {i: scale for i, scale in terms.items() if scale}, bias + sign * other_bias

    def interface_access_path(self, expression, parameters, contract, depth=0, *, resolve_local=None):
        if depth >= 32:
            return None
        resolve = resolve_local or (lambda value: value)
        expression = resolve(expression)
        path = []
        while expression is not None and expression.kind in ("member", "index", "length"):
            if len(path) == 16:
                return None
            if expression.kind == "length":
                path.append("@len")
                expression = resolve(expression.args[0])
                continue
            base, step = expression.args
            if expression.kind == "member":
                path.append(step.text)
            else:
                index = self.interface_index_affine(step, parameters, contract, depth + 1,
                                                        resolve_local=resolve_local)
                if index is None:
                    step = resolve(step)
                    if step.kind != "binary" or step.args[0] != "-":
                        return None
                    _, length, distance = step.args
                    length, distance = resolve(length), resolve(distance)
                    if length.kind != "length":
                        return None
                    distance_index = self.interface_index_affine(distance, parameters, contract, depth + 1,
                                                                     resolve_local=resolve_local)
                    if distance_index is None:
                        return None
                    indexed = self.interface_access_path(base, parameters, contract, depth + 1,
                                                             resolve_local=resolve_local)
                    measured = self.interface_access_path(length.args[0], parameters, contract, depth + 1,
                                                              resolve_local=resolve_local)
                    if indexed is None or indexed != measured:
                        return None
                    path.append(("from_end", interface_index_step(*distance_index)))
                    expression = resolve(base)
                    continue
                terms, bias = index
                path.append(interface_index_step(dict(sorted(terms.items())), bias))
            expression = resolve(base)
        path = tuple(reversed(path))
        if expression is not None and expression.kind == "name" and expression.args[0] in parameters:
            return parameters[expression.args[0]], path
        return None

    def function_interface_accessor(self, declaration, contract, mutations, *, depth=0):
        if mutations != () or depth >= 32:
            return None
        body = declaration.args[3]
        if not 1 <= len(body) <= 17:
            return None
        parameters = {parameter.text: index for index, parameter in enumerate(declaration.args[2])}

        def local_prefix(statements, bindings):
            if not 1 <= len(statements) <= 17:
                return None
            bindings = bindings.copy()
            for index, statement in enumerate(statements):
                if statement.kind != "define":
                    return bindings, statements[index:]
                name, annotation, initializer = statement.args
                if annotation is not None or initializer is None or name.text in parameters or name.text in bindings:
                    return None
                bindings[name.text] = initializer
            return None

        prefix = local_prefix(body, {})
        if prefix is None:
            return None
        locals, body = prefix

        used = set()

        def resolve_local(expression, bindings=locals, seen=used):
            for _ in range(len(bindings)):
                if expression is None or expression.kind != "name" or expression.args[0] not in bindings:
                    break
                name = expression.args[0]
                seen.add(name)
                expression = bindings[name]
            return expression

        def access(expression):
            return self.interface_access_path(expression, parameters, contract, resolve_local=resolve_local)

        def strip_return_casts(expression, resolve):
            expression = resolve(expression)
            while expression is not None and expression.kind == "cast" and expression.args[1] == "safe":
                operand, _, annotation = expression.args
                if not plain_integer_parameter(self.resolve(annotation)):
                    return None
                expression = resolve(operand)
            return expression

        def branch_access(statements):
            prefix = local_prefix(statements, locals)
            if prefix is None:
                return None
            bindings, tail = prefix
            if len(tail) != 1 or tail[0].kind != "return":
                return None
            seen = set()
            def resolve_branch(expression):
                return resolve_local(expression, bindings, seen)

            expression = strip_return_casts(tail[0].args[0], resolve_branch)
            selected = self.interface_access_path(expression, parameters, contract, resolve_local=resolve_branch)
            if bindings.keys() - locals.keys() - seen:
                return None
            used.update(seen & locals.keys())
            return selected

        if body[0].kind == "if":
            condition, yes, no = body[0].args
            if not no:
                no = body[1:]
            elif len(body) != 1:
                return None
            condition = resolve_local(condition)
            while condition is not None and condition.kind == "unary" and condition.args[0] == "!":
                condition = resolve_local(condition.args[1])
                yes, no = no, yes
            if condition is None:
                return None
            if condition.kind == "name":
                selector = parameters.get(condition.args[0])
            else:
                selector = access(condition)
            if selector is None:
                return None
            first = branch_access(yes)
            second = branch_access(no)
            if first is None or second is None or len(used) != len(locals):
                return None
            accessor = (*first, "choose", second[1], second[0], selector)
            return accessor if valid_interface_accessor(contract, accessor, mutations) else None
        if len(body) != 1 or body[0].kind != "return":
            return None
        expression = strip_return_casts(body[-1].args[0], resolve_local)
        if expression is not None and expression.kind == "binary" and expression.args[0] in INTERFACE_FIELD_OPERATORS:
            op, left, right = expression.args
            first = access(left)
            second = access(right)
            accessor = None
            if first is not None and second is not None:
                accessor = (*first, op, second[1])
                if first[0] != second[0]:
                    accessor += (second[0],)
        elif expression is not None and expression.kind == "call":
            accessor = self.forwarded_interface_accessor(expression, parameters, contract, resolve_local, depth)
        else:
            accessor = access(expression)
        if len(used) != len(locals):
            return None
        return accessor if accessor is not None and valid_interface_accessor(contract, accessor, mutations) else None

    def forwarded_interface_accessor(self, expression, parameters, contract, resolve_local, depth):
        callee, arguments = expression.args
        if callee.kind != "name":
            return None
        positions = []
        indices = []
        selectors = {}
        selector_fields = {}
        inverted_selectors = set()
        for argument_index, argument in enumerate(arguments):
            argument = resolve_local(argument)
            inverted = False
            while argument.kind == "unary" and argument.args[0] == "!":
                inverted = not inverted
                argument = resolve_local(argument.args[1])
            position = parameters.get(argument.args[0]) if argument.kind == "name" else None
            index = None if inverted else self.interface_index_affine(
                argument, parameters, contract, resolve_local=resolve_local)
            if argument.kind == "boolean":
                selectors[argument_index] = argument.args[0] != inverted
            else:
                if position is None and index is None:
                    field = self.interface_access_path(argument, parameters, contract, resolve_local=resolve_local)
                    if field is None or not guard_boolean_parameter(interface_accessor_member(contract, *field)):
                        return None
                    selector_fields[argument_index] = field
                if inverted:
                    inverted_selectors.add(argument_index)
            positions.append(position)
            indices.append(index)
        name = callee.args[0]
        if name in self.functions:
            summary = self.function_interface_accessor(*self.functions[name], self.mutations.get(name), depth=depth + 1)
        else:
            binding = self.imported_functions.get(name)
            summary = binding.interface_accessor if binding is not None else None
        if summary is None:
            return None
        index, path, *arithmetic = summary

        def remap(steps):
            mapped = []
            for step in steps:
                if isinstance(step, tuple) and step[0] == "affine":
                    terms, bias = step[1:]
                elif isinstance(step, tuple) and step[0] == "param":
                    terms = ((step[1], step[3] if len(step) == 4 else 1),)
                    bias = step[2] if len(step) >= 3 else 0
                elif isinstance(step, tuple) and step[0] == "from_end" and isinstance(step[1], tuple):
                    distance = remap((step[1],))
                    if distance is None:
                        return None
                    mapped.append(("from_end", distance[0]))
                    continue
                else:
                    mapped.append(step)
                    continue
                combined = {}
                for i, scale in terms:
                    if indices[i] is None:
                        return None
                    factors, offset = indices[i]
                    bias += scale * offset
                    for parameter, coefficient in factors.items():
                        combined[parameter] = combined.get(parameter, 0) + scale * coefficient
                combined = {parameter: coefficient for parameter, coefficient in combined.items() if coefficient}
                mapped.append(interface_index_step(combined, bias))
            return tuple(mapped)

        if arithmetic and arithmetic[0] == "choose":
            _, other, other_index, selector = arithmetic
            if type(selector) is int and selector in selectors:
                if not selectors[selector]:
                    index, path = other_index, other
                arithmetic = []
            elif type(selector) is int and selector in inverted_selectors:
                arithmetic = ["choose", path, index, selector]
                index, path = other_index, other

        path = remap(path)
        if positions[index] is None or path is None:
            return None
        if arithmetic:
            op, other, *source = arithmetic
            other = remap(other)
            mapped = []
            for offset, item in enumerate(source):
                if op == "choose" and offset == 1 and type(item) is int and item in selector_fields:
                    mapped.append(selector_fields[item])
                elif isinstance(item, tuple):
                    mapped.append((positions[item[0]], remap(item[1])))
                else:
                    mapped.append(positions[item])
            if other is None or any(item is None or isinstance(item, tuple) and None in item for item in mapped):
                return None
            return positions[index], path, op, other, *mapped
        return positions[index], path
