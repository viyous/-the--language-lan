"""Guard function summaries and condition values."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from itertools import count

from bit_domain import bitwise_domain, mask_preimage
from float_domain import FloatDomain
from value_domain import Domain

from .errors import LanError
from .state import PathState
from .syntax import Node
from .typesys import (AffineForm, BitView, is_callback_enum, Scalar, Span, unqualified, Value, ValueOrigin,
                      WIDE, ZERO)
from .domain_utils import container, plain_integer_parameter, trunc_div
from .summaries import (GUARD_ARITHMETIC, GUARD_BITWISE, GUARD_COMPARISONS, GUARD_DIVISION,
                        guard_parameter_kind, GUARD_SHIFTS, valid_guard_divisor, valid_guard_shift_count)


class GuardMixin:
    def function_guard(self, declaration, contract, *, active=frozenset(), budget=None):
        if declaration.args[0].text in self.recursive_functions or id(declaration) in active or len(active) >= 32:
            return None
        active = active | {id(declaration)}
        if not isinstance(contract.result, Scalar) or not contract.result.boolean:
            return None
        if not all(guard_parameter_kind(ty) for ty in contract.params):
            return None
        statements = declaration.args[3]
        parameters = {param.text: (guard_parameter_kind(ty), i)
                      for i, (param, ty) in enumerate(zip(declaration.args[2], contract.params))}
        if budget is None:
            budget = [128]

        def operand(expr, bindings, depth=0):
            budget[0] -= 1
            if expr is None or depth > 32 or budget[0] < 0:
                return None
            if expr.kind == "name":
                value = bindings.get(expr.args[0])
                return value if value is not None and value[0] in ("param", "const", "arith", "len") else None
            if expr.kind == "length" and expr.args[0].kind == "name":
                value = bindings.get(expr.args[0].args[0])
                if value is not None and value[0] == "slice_param":
                    return "len", value[1]
                if value is not None and value[0] == "array_param":
                    return "const", contract.params[value[1]].array.length
            if expr.kind == "number":
                return "const", expr.args[0]
            if expr.kind == "cast" and expr.args[1] == "safe":
                source, _, annotation = expr.args
                if plain_integer_parameter(self.resolve(annotation)):
                    return operand(source, bindings, depth + 1)
                return None
            if expr.kind == "unary" and expr.args[0] in ("+", "-"):
                op, inner = expr.args
                value = operand(inner, bindings, depth + 1)
                if value is None or op == "+":
                    return value
                if value[0] == "const":
                    result = -value[1]
                    return ("const", result) if WIDE.lo <= result <= WIDE.hi else None
                return "arith", "-", ("const", 0), value
            if expr.kind == "binary" and expr.args[0] in GUARD_ARITHMETIC:
                op, left, right = expr.args
                left, right = operand(left, bindings, depth + 1), operand(right, bindings, depth + 1)
                if left is not None and right is not None:
                    if op in GUARD_SHIFTS and not valid_guard_shift_count(right, contract):
                        return None
                    if left[0] == right[0] == "const":
                        x, y = left[1], right[1]
                        if op in GUARD_SHIFTS:
                            result = x << y if op == "<<" else x >> y
                        elif op in GUARD_BITWISE:
                            result = bitwise_domain(op, Domain.point(x), Domain.point(y))[0].lo
                        elif op in ("/", "%"):
                            if y == 0:
                                return None
                            quotient = trunc_div(x, y)
                            result = quotient if op == "/" else x - quotient * y
                        else:
                            result = x + y if op == "+" else x - y if op == "-" else x * y
                        return ("const", result) if WIDE.lo <= result <= WIDE.hi else None
                    if op in ("/", "%"):
                        if right == ("const", 1):
                            return left if op == "/" else ("const", 0)
                        if not valid_guard_divisor(right, contract):
                            return None
                    return "arith", op, left, right
            return None

        def compose(expr, bindings, depth):
            callee, args = expr.args
            if callee.kind != "name":
                return None
            name = callee.args[0]
            if name in self.functions:
                target, signature = self.functions[name]
                summary = self.function_guard(target, signature, active=active, budget=budget)
            else:
                binding = self.imported_functions.get(name)
                if binding is None:
                    return None
                signature, summary = binding.contract, binding.guard
            if summary is None or summary[0] == "outcomes" or len(args) != len(signature.params):
                return None
            actuals = []
            for arg, ty in zip(args, signature.params):
                kind = guard_parameter_kind(ty)
                if kind == "param":
                    actual = operand(arg, bindings, depth + 1)
                elif kind == "bool_param":
                    actual = predicate(arg, bindings, depth + 1)
                else:
                    actual = bindings.get(arg.args[0]) if arg.kind == "name" else None
                    if actual is not None and actual[0] != kind:
                        actual = None
                if actual is None:
                    return None
                actuals.append(actual)

            def substitute(value):
                budget[0] -= 1
                if budget[0] < 0:
                    return None
                kind, *parts = value
                if kind in ("param", "bool_param"):
                    return actuals[parts[0]]
                if kind == "len":
                    return "len", actuals[parts[0]][1]
                if kind in ("const", "bool"):
                    return value
                if kind == "in":
                    subject = substitute(parts[0])
                    return None if subject is None else (kind, subject, parts[1])
                prefix, children = (parts[:1], parts[1:]) if kind == "arith" else ([], parts)
                mapped = tuple(substitute(child) for child in children)
                if None in mapped:
                    return None
                if kind == "arith" and prefix[0] in GUARD_SHIFTS and not valid_guard_shift_count(mapped[1], contract):
                    return None
                if kind == "arith" and prefix[0] in GUARD_DIVISION and not valid_guard_divisor(mapped[1], contract):
                    return None
                return kind, *prefix, *mapped

            return substitute(summary)

        def negate(value):
            if value[0] == "bool":
                return "bool", not value[1]
            if value[0] == "!":
                return value[1]
            return "!", value

        def predicate(expr, bindings, depth=0):
            budget[0] -= 1
            if expr is None or depth > 32 or budget[0] < 0:
                return None
            if expr.kind == "name":
                value = bindings.get(expr.args[0])
                return value if value is not None and value[0] not in ("param", "const", "arith", "len", "slice_param", "array_param") else None
            if expr.kind == "boolean":
                return "bool", bool(expr.args[0])
            if expr.kind == "call":
                return compose(expr, bindings, depth)
            if expr.kind == "binary":
                op, left, right = expr.args
                if op in ("&&", "||", "^^"):
                    left, right = predicate(left, bindings, depth + 1), predicate(right, bindings, depth + 1)
                    if left is not None and right is not None:
                        if op == "^^":
                            return choose(left, negate(right), right)
                        if left == right:
                            return left
                        if right[0] == "bool":
                            return right if right[1] == (op == "||") else left
                        if left[0] == "bool":
                            return left if left[1] == (op == "||") else right
                elif op in GUARD_COMPARISONS:
                    numeric = operand(left, bindings), operand(right, bindings)
                    if None in numeric and op in ("==", "!=", "===", "!=="):
                        left = predicate(left, bindings, depth + 1)
                        right = predicate(right, bindings, depth + 1)
                        if left is not None and right is not None:
                            same = ("bool", True) if left == right else choose(left, right, negate(right))
                            return negate(same) if op in ("!=", "!==") else same
                    left, right = numeric
                    if left is not None and right is not None and left[0] == right[0] == "const":
                        x, y = left[1], right[1]
                        return "bool", {"==": x == y, "===": x == y, "!=": x != y, "!==": x != y,
                                        "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
                else:
                    return None
                if left is not None and right is not None:
                    return op, left, right
            elif expr.kind == "in":
                value, ty = expr.args
                subject = operand(value, bindings)
                if subject is not None:
                    return "in", subject, self.numeric_type(ty).domain & WIDE
            elif expr.kind == "unary" and expr.args[0] == "!":
                inner = predicate(expr.args[1], bindings, depth + 1)
                if inner is not None:
                    return negate(inner)
            return None

        def choose(test, yes, no):
            if test[0] == "bool":
                return yes if test[1] else no
            if yes is None or no is None:
                return None
            if yes == no:
                return yes
            if yes == ("bool", True) and no == ("bool", False):
                return test
            if yes == ("bool", False) and no == ("bool", True):
                return negate(test)
            return "choose", test, yes, no

        comparison_outcomes = {
            "<": 0b001, "==": 0b010, "===": 0b010, ">": 0b100,
            "<=": 0b011, ">=": 0b110, "!=": 0b101, "!==": 0b101,
        }
        reversed_comparison = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}

        def known(test, facts):
            op, *parts = test
            outcomes = 0b111
            for condition, truth in reversed(facts):
                if test == condition:
                    return "bool", truth
                if test == negate(condition):
                    return "bool", not truth
                other = condition[0]
                if op not in comparison_outcomes or other not in comparison_outcomes:
                    continue
                if test[1:] == condition[1:]:
                    allowed = comparison_outcomes[other]
                elif test[1:] == condition[1:][::-1]:
                    allowed = comparison_outcomes[reversed_comparison.get(other, other)]
                else:
                    continue
                outcomes &= allowed if truth else 0b111 ^ allowed
            if op in comparison_outcomes and outcomes:
                matching = outcomes & comparison_outcomes[op]
                if matching == outcomes or not matching:
                    return "bool", bool(matching)
            if op == "!":
                return negate(known(parts[0], facts))
            if op in ("&&", "||"):
                left, right = (known(part, facts) for part in parts)
                absorbing = "bool", op == "||"
                if left == absorbing or right == absorbing:
                    return absorbing
                if left[0] == "bool":
                    return right
                if right[0] == "bool" or left == right:
                    return left
                return op, left, right
            return test

        def assume(test, truth, facts):
            pending = [(test, truth)]
            while pending:
                condition, truth = pending.pop()
                if (condition, truth) in facts:
                    continue
                facts += ((condition, truth),)
                op, *parts = condition
                if op == "!":
                    pending.append((parts[0], not truth))
                elif (op == "&&" and truth) or (op == "||" and not truth):
                    pending.extend((part, truth) for part in parts)
            return facts

        def counted_loop_exit(test, body, bindings):
            if test.kind != "binary" or not body:
                return None
            op, counter, limit = test.args
            reverse = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "!=": "!=", "!==": "!=="}
            if op not in reverse:
                return None
            updates = {}
            for statement in body:
                if statement.kind != "assign":
                    return None
                target, assignment, step = statement.args
                if target.kind != "name":
                    return None
                key = target.args[0]
                if key in updates or bindings.get(key, (None,))[0] not in ("const", "param", "arith", "len"):
                    return None
                operator = assignment.text
                if operator == "=" and step.kind == "binary":
                    operation, left, right = step.args
                    if operation == "+" and right.kind == "name" and right.args == target.args:
                        left, right = right, left
                    if operation in ("+", "-") and left.kind == "name" and left.args == target.args:
                        operator, step = operation + "=", right
                if operator not in ("+=", "-="):
                    return None
                updates[key] = (operator, step)
            if counter.kind != "name" or counter.args[0] not in updates:
                counter, limit = limit, counter
                op = reverse[op]
            if counter.kind != "name" or counter.args[0] not in updates:
                return None
            name = counter.args[0]
            if bindings[name][0] != "const":
                return None

            stable = {key: value for key, value in bindings.items() if key not in updates}
            bound = operand(limit, stable)
            if bound is None or bound[0] != "const":
                return None
            strides = {}
            for key, (assignment, expr) in updates.items():
                step = operand(expr, stable)
                if step is None or step[0] != "const":
                    return None
                strides[key] = step[1] if assignment == "+=" else -step[1]
            start, bound, stride = bindings[name][1], bound[1], strides[name]
            if op in ("!=", "!=="):
                if not stride or (bound - start) % stride:
                    return None
                iterations = (bound - start) // stride
                if iterations <= 0:
                    return None
            else:
                if op in (">", ">="):
                    distance, progress = start - bound, -stride
                else:
                    distance, progress = bound - start, stride
                distance += op in ("<=", ">=")
                if distance <= 0 or progress <= 0:
                    return None
                iterations = (distance + progress - 1) // progress
            result = {}
            for key, stride in strides.items():
                initial = bindings[key]
                delta = iterations * stride
                if initial[0] == "const":
                    end = initial[1] + delta
                    if not WIDE.lo <= end <= WIDE.hi:
                        return None
                    result[key] = "const", end
                elif WIDE.lo <= delta <= WIDE.hi:
                    result[key] = ("arith", "+", initial, ("const", delta)) if delta else initial
                else:
                    return None
            return bindings | result

        def block(statements, bindings, continuation, depth=0, edges=None, facts=()):
            if depth > 32:
                return None
            bindings = dict(bindings)
            shadowed = {}

            def exit_scope(resume):
                return lambda current, facts: resume(current | shadowed, facts)

            continuation = exit_scope(continuation)
            if edges is not None:
                edges = exit_scope(edges)
            for index, stmt in enumerate(statements):
                budget[0] -= 1
                if budget[0] < 0:
                    return None
                if stmt.kind == "return":
                    result = predicate(stmt.args[0], bindings, depth)
                    return known(result, facts) if result is not None else None
                if stmt.kind == "guard_assert":
                    subject, rows = stmt.args
                    if (subject.kind != "name" or subject.args[0] not in bindings
                            or any(guard is not None and not self.pure_guard(guard) for _, guard, _ in rows)):
                        return None
                    # Normal body checking proves the rows; they do not modify bindings.
                    continue
                if stmt.kind == "define":
                    name, annotation, initializer = stmt.args
                    if name.text in bindings:
                        shadowed.setdefault(name.text, bindings[name.text])
                    if annotation is not None:
                        ty = unqualified(self.resolve(annotation))
                        if not isinstance(ty, Scalar) or not (plain_integer_parameter(ty) or ty.boolean):
                            return None
                    if initializer is None and annotation is not None:
                        value = ("bool", False) if ty.boolean else ("const", 0)
                    else:
                        value = operand(initializer, bindings) or predicate(initializer, bindings, depth)
                    if value is None:
                        return None
                    bindings[name.text] = value
                    continue
                if stmt.kind == "assign":
                    target, op, initializer = stmt.args
                    if target.kind != "name" or target.args[0] not in bindings:
                        return None
                    if op.text == "=":
                        value = operand(initializer, bindings) or predicate(initializer, bindings, depth)
                    elif op.text[:-1] in GUARD_ARITHMETIC and op.text.endswith("="):
                        value = operand(Node(op, "binary", (op.text[:-1], target, initializer)), bindings)
                    else:
                        return None
                    if value is None:
                        return None
                    bindings[target.args[0]] = value
                    continue
                if stmt.kind == "continue":
                    if edges is None:
                        return None
                    return edges(bindings, facts)

                def resume(branch_bindings, facts):
                    outer = {name: branch_bindings[name] for name in bindings}
                    return block(statements[index + 1:], outer, continuation, depth, edges, facts)

                if stmt.kind == "while":
                    test_node, loop_body = stmt.args[:2]
                    exit_bindings = counted_loop_exit(test_node, loop_body, bindings)
                    if exit_bindings is not None:
                        return resume(exit_bindings, facts)

                    def loop(current, seen, iterations, facts):
                        current = {name: current[name] for name in bindings}
                        key = tuple(current.items())
                        test = predicate(test_node, current, depth + 1)
                        if test is None:
                            return None
                        test = known(test, facts)
                        if test == ("bool", False):
                            return resume(current, facts)
                        if iterations >= 16 or key in seen:
                            return None

                        def again(updated, facts):
                            return loop(updated, seen | {key}, iterations + 1, facts)

                        if test == ("bool", True):
                            return block(loop_body, current, again, depth + 1, again, facts)
                        hit = block(loop_body, current, again, depth + 1, again, assume(test, True, facts))
                        miss = resume(current, assume(test, False, facts))
                        return choose(test, hit, miss)

                    return loop(bindings, frozenset(), 0, facts)
                if stmt.kind != "if":
                    return None
                test, yes, no = stmt.args
                test = predicate(test, bindings, depth + 1)
                if test is None:
                    return None
                test = known(test, facts)
                if test[0] == "bool":
                    return block(yes if test[1] else no, bindings, resume, depth + 1, edges, facts)
                yes = block(yes, bindings, resume, depth + 1, edges, assume(test, True, facts))
                no = block(no, bindings, resume, depth + 1, edges, assume(test, False, facts))
                return choose(test, yes, no)
            return continuation(bindings, facts)

        try:
            summary = block(statements, parameters, lambda _bindings, _facts: None)
        except LanError:
            return None
        if summary is None:
            return None
        pending = [(summary, 0)]
        size = 0
        while pending:
            item, depth = pending.pop()
            size += 1
            if size > 128 or depth > 32:
                return None
            if item[0] in ("&&", "||", "!", "choose"):
                pending.extend((part, depth + 1) for part in item[1:])
            elif item[0] == "arith":
                pending.extend((part, depth + 1) for part in item[2:])
            elif item[0] == "in":
                pending.append((item[1], depth + 1))
            elif item[0] in GUARD_COMPARISONS:
                pending.extend((part, depth + 1) for part in item[1:])
        return summary

    def call_guard(self, node):
        if node.kind != "call" or node.args[0].kind != "name":
            return None
        name = node.args[0].args[0]
        arguments = node.args[1]
        if name in self.functions:
            declaration, contract = self.functions[name]
            summary = self.function_guard(declaration, contract) or self.verified_guards.get(name)
        else:
            binding = self.imported_functions.get(name)
            if binding is None:
                return None
            contract, summary = binding.contract, binding.guard
        return summary if len(arguments) == len(contract.params) else None

    def instantiate_call_guard(self, node, summary, arguments, state):

        def operand(value):
            if value[0] == "arith":
                _, op, left, right = value
                return Node(node.token, "binary", (op, operand(left), operand(right)))
            kind, item = value
            if kind == "len":
                argument = arguments[item]
                return argument if argument.kind == "guard_snapshot" else Node(node.token, "length", (argument,))
            return arguments[item] if kind == "param" else Node(node.token, "number", (item,))

        def shift_count(value):
            if value[0] == "const":
                return value[1]
            if value[0] == "param":
                domain = self.expr(arguments[value[1]], state).domain
                if domain.lo == domain.hi:
                    return domain.lo
            return None

        def affine(value):
            kind, *parts = value
            if kind == "const":
                return None, 0, parts[0]
            if kind == "param":
                return parts[0], 1, 0
            if kind != "arith":
                return None
            op, left, right = parts
            if op == "<<" and (count := shift_count(right)) is not None:
                right = "const", count
            left, right = affine(left), affine(right)
            if left is None or right is None:
                return None
            a, scale_a, offset_a = left
            b, scale_b, offset_b = right
            if a is not None and b is not None and a != b:
                return None
            index = a if a is not None else b
            if op == "+":
                return index, scale_a + scale_b, offset_a + offset_b
            if op == "-":
                return index, scale_a - scale_b, offset_a - offset_b
            if op == "<<" and b is None and 0 <= offset_b < 128:
                return index, scale_a << offset_b, offset_a << offset_b
            if op == "*" and (scale_a == 0 or scale_b == 0):
                return index, scale_a * offset_b + scale_b * offset_a, offset_a * offset_b
            return None

        def affine_guard(op, left, right):
            difference = affine(("arith", "-", left, right))
            if difference is None:
                return None
            index, scale, offset = difference
            if index is None or scale == 0:
                return None
            op = {"===": "==", "!==": "!="}.get(op, op)
            if scale < 0:
                scale, offset = -scale, -offset
                op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(op, op)
            bound = Fraction(-offset, scale)
            if op == "<":
                domain = Domain.of((WIDE.lo, math.ceil(bound) - 1))
            elif op == "<=":
                domain = Domain.of((WIDE.lo, math.floor(bound)))
            elif op == ">":
                domain = Domain.of((math.floor(bound) + 1, WIDE.hi))
            elif op == ">=":
                domain = Domain.of((math.ceil(bound), WIDE.hi))
            else:
                domain = Domain.point(bound.numerator) if bound.denominator == 1 else Domain()
                if op == "!=":
                    domain = WIDE - domain
            return Node(node.token, "guard_domain", (arguments[index], domain & WIDE))

        def shift_guard(op, left, right):
            if right[0] == "arith" and right[1] in GUARD_SHIFTS:
                left, right = right, left
                op = reverse_comparison.get(op, op)
            if left[0] != "arith" or left[1] not in GUARD_SHIFTS or affine(right) is None:
                return None
            _, operation, source, amount = left
            if affine(source) is None:
                return None
            return Node(node.token, "guard_shift", (op, operation, operand(source), operand(amount), operand(right)))

        def congruence_preimage(bounds, residue, modulus, form):
            if form is None:
                return bounds.with_congruence(residue, modulus)
            bounds = form.preimage(bounds) & WIDE
            common = math.gcd(form.scale, modulus)
            delta = residue - form.offset
            if delta % common:
                return Domain()
            period = modulus // common
            if period == 1:
                return bounds
            residue = (delta // common) * pow(form.scale // common, -1, period) % period
            return bounds.with_congruence(residue, period)

        def select_guard_cases(symbolic, cases):
            expr = operand(symbolic)
            _, result = cases[-1]
            for value, branch in reversed(cases[:-1]):
                test = affine_guard("==", symbolic, ("const", value))
                if test is None:
                    test = Node(node.token, "guard_domain", (expr, Domain.point(value)))
                result = Node(node.token, "guard_choose", (test, branch, result))
            return result

        def bitwise_guard(op, left, right):
            if right[0] == "arith" and right[1] in GUARD_BITWISE:
                left, right = right, left
                op = reverse_comparison.get(op, op)
            if left[0] != "arith" or left[1] not in GUARD_BITWISE or affine(right) is None:
                return None
            _, operation, source, mask = left
            if source[0] == "const":
                source, mask = mask, source
            source_form = affine(source)
            if source_form is None or affine(mask) is None:
                return None
            masks = (mask[1],) if mask[0] == "const" else self.bounded_window_values(
                self.expr(operand(mask), state).domain, 8)
            if not masks:
                return None
            values = (right[1],) if right[0] == "const" else self.bounded_window_values(
                self.expr(operand(right), state).domain, 8 // len(masks))
            if not values:
                return None

            cases = []
            for candidate in masks:
                comparisons = [(value, bitwise_preimage(op, operation, source, source_form, candidate, value))
                               for value in values]
                if any(branch is None for _, branch in comparisons):
                    return None
                cases.append((candidate, select_guard_cases(right, comparisons)))
            return select_guard_cases(mask, cases)

        def bitwise_preimage(op, operation, source, source_form, mask, expected):
            subject = operand(source)
            form = None
            index, scale, offset = source_form
            if operation == "&" and index is not None and scale:
                form = AffineForm(ValueOrigin(WIDE), scale, offset)
                subject = arguments[index]
            if op in ("<", "<=", ">", ">="):
                if operation != "&" or mask < 0:
                    return None
                period = 1 << mask.bit_length()
                threshold = min(period, max(0, expected + (op in ("<=", ">"))))
                if mask == period - 1:
                    lower = threshold <= period - threshold
                    residues = range(threshold) if lower else range(threshold, period)
                    if min(threshold, period - threshold) > 64:
                        return None
                elif period <= 64:
                    below, above = [], []
                    for residue in range(period):
                        (below if residue & mask < threshold else above).append(residue)
                    lower = len(below) <= len(above)
                    residues = below if lower else above
                else:
                    return None
                domain = Domain.union(congruence_preimage(WIDE, residue, period, form) for residue in residues)
                if lower != (op in ("<", "<=")):
                    domain = WIDE - domain
                return Node(node.token, "guard_domain", (subject, domain))
            if operation == "^":
                domain = Domain.point(mask ^ expected) & WIDE
            elif operation == "|":
                if mask < 0:
                    return None
                if expected & mask != mask:
                    domain = Domain()
                else:
                    suffix = (mask ^ (mask + 1)) >> 1
                    free = mask ^ suffix
                    if free.bit_count() > 6:
                        return None
                    starts = [expected & ~mask]
                    while free:
                        bit = free & -free
                        starts += [start | bit for start in starts]
                        free ^= bit
                    domain = Domain.of(*((start, start | suffix) for start in starts)) & WIDE
            elif mask < 0:
                return None
            elif expected < 0 or expected & ~mask:
                domain = Domain()
            else:
                period = 1 << mask.bit_length()
                free = (period - 1) ^ mask
                if free.bit_count() > 6:
                    return Node(node.token, "guard_mask", (operand(source), mask, expected,
                                                         op in ("!=", "!==")))
                residues = [expected]
                while free:
                    bit = free & -free
                    residues += [residue | bit for residue in residues]
                    free ^= bit
                domain = Domain.union(congruence_preimage(WIDE, residue, period, form) for residue in residues)
            if op in ("!=", "!=="):
                domain = WIDE - domain
            return Node(node.token, "guard_domain", (subject, domain))

        def remainder_preimage(op, expected, modulus, form):
            limit = modulus - 1

            def residue_domain(remainder):
                lo = 0 if remainder > 0 else WIDE.lo
                hi = 0 if remainder < 0 else WIDE.hi
                return congruence_preimage(Domain.of((lo, hi)), remainder, modulus, form)

            if op in ("==", "===", "!=", "!=="):
                domain = residue_domain(expected) if -limit <= expected <= limit else Domain()
                return WIDE - domain if op in ("!=", "!==") else domain

            threshold = min(limit + 1, max(-limit, expected + (op in ("<=", ">"))))
            use_lower = threshold + limit <= limit + 1 - threshold
            residues = range(-limit, threshold) if use_lower else range(threshold, limit + 1)
            if residues.stop - residues.start > 64:
                return None
            domain = Domain.union(residue_domain(remainder) for remainder in residues)
            return domain if use_lower == (op in ("<", "<=")) else WIDE - domain

        def quotient_preimage(op, expected, divisor):
            if divisor < 0:
                divisor, expected = -divisor, -expected
                op = reverse_comparison.get(op, op)
            first = expected * divisor - (divisor - 1 if expected <= 0 else 0)
            last = expected * divisor + (divisor - 1 if expected >= 0 else 0)
            if op in ("<", "<="):
                return Domain.of((WIDE.lo, first - 1 if op == "<" else last)) & WIDE
            if op in (">", ">="):
                return Domain.of((last + 1 if op == ">" else first, WIDE.hi)) & WIDE
            domain = Domain.of((first, last)) & WIDE
            return WIDE - domain if op in ("!=", "!==") else domain

        reverse_comparison = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}

        def division_guard(op, left, right):
            if right[0] == "arith" and right[1] in GUARD_DIVISION:
                left, right = right, left
                op = reverse_comparison.get(op, op)
            if left[0] != "arith" or left[1] not in GUARD_DIVISION or affine(right) is None:
                return None
            numerator, denominator = map(operand, left[2:])
            numerator_form = affine(left[2])
            form = None
            if numerator_form is not None:
                index, scale, offset = numerator_form
                if index is not None and scale:
                    numerator = arguments[index]
                    form = AffineForm(ValueOrigin(WIDE), scale, offset)
            expected = operand(right)
            divisors = self.bounded_window_values(self.expr(denominator, state).domain, 8)
            if not divisors or 0 in divisors:
                return None
            values = self.bounded_window_values(self.expr(expected, state).domain, 8 // len(divisors))
            if not values:
                return None

            def member(expr, domain):
                return Node(node.token, "guard_domain", (expr, domain))

            cases = []
            for divisor in divisors:
                comparisons = []
                for value in values:
                    if left[1] == "%":
                        domain = remainder_preimage(op, value, abs(divisor), form)
                        if domain is None:
                            return None
                    else:
                        domain = quotient_preimage(op, value, divisor)
                        if form is not None:
                            domain = form.preimage(domain) & WIDE
                    comparisons.append((value, member(numerator, domain)))
                cases.append((divisor, select_guard_cases(right, comparisons)))
            return select_guard_cases(left[3], cases)

        def predicate(value):
            kind, *parts = value
            if kind == "bool_param":
                return arguments[parts[0]]
            if kind == "bool":
                return Node(node.token, "boolean", (parts[0],))
            if kind in ("choose", "outcomes"):
                return Node(node.token, "guard_" + kind, tuple(predicate(part) for part in parts))
            if kind == "!":
                return Node(node.token, "unary", (kind, predicate(parts[0])))
            if kind == "in":
                return Node(node.token, "guard_domain", (operand(parts[0]), parts[1]))
            if kind in GUARD_COMPARISONS and any(part[0] == "arith" for part in parts):
                narrowed = (affine_guard(kind, *parts) or shift_guard(kind, *parts)
                            or bitwise_guard(kind, *parts) or division_guard(kind, *parts))
                if narrowed is not None:
                    return narrowed
            convert = predicate if kind in ("&&", "||") else operand
            return Node(node.token, "binary", (kind, *(convert(part) for part in parts)))

        return predicate(summary)

    @staticmethod
    def shift_preimage(op, operation, expected, count):
        scale = 1 << count
        if operation == "<<":
            first, last = -(-expected // scale), expected // scale
        else:
            first = expected * scale
            last = first + scale - 1
        if op in ("<", "<="):
            return Domain.of((WIDE.lo, first - 1 if op == "<" else last)) & WIDE
        if op in (">", ">="):
            return Domain.of((last + 1 if op == ">" else first, WIDE.hi)) & WIDE
        domain = Domain.of((first, last)) & WIDE
        return WIDE - domain if op in ("!=", "!==") else domain

    def restrict_guard_operand(self, state, expr, value, domain):
        paths = self.refine_return_paths(self.restrict(state, expr, domain), value, domain)
        form = value.affine
        if expr.kind != "binary" or form is None or not form.scale:
            return paths
        original = form.preimage(domain)
        for uid, variable in state.forms.items():
            if variable.origin is form.origin and variable.scale:
                narrowed = original.affine(variable.scale, variable.offset)
                return [part for path in paths
                        if (part := path.narrow(self.symbols[uid - 1], narrowed)) is not None]
        return paths

    def modulo_guard(self, left, right):
        """`x % k == r` with literal k and r picks one residue class.

        Only a plain integer variable is narrowed: a class on a projection
        would have to be pushed through the place it came from.
        """
        for expression, constant in ((left, right), (right, left)):
            if expression.kind != "binary" or expression.args[0] != "%":
                continue
            dividend, modulus = expression.args[1], expression.args[2]
            if dividend.kind != "name" or modulus.kind != "number" or constant.kind != "number":
                continue
            step, residue = int(modulus.args[0]), int(constant.args[0])
            if step <= 1:
                continue
            sym = self.lookup(dividend)
            ty = unqualified(sym.ty)
            if (not isinstance(ty, Scalar) or ty.floating or ty.boolean or ty.enum
                    or sym.projection is not None or sym.uid in self.alias_members):
                continue
            return dividend, sym, residue % step, step
        return None

    def condition_value(self, node, state, *, require_boolean=False):
        self.spend(node.token)
        cast_operand = self.boolean_cast_operand(node)
        if cast_operand is not None:
            yes, no, code = self.condition(cast_operand, state)
            return yes, no, f"!!({code})"
        alias = self.alias_condition(node, state)
        if alias is not None:
            if require_boolean and node.kind not in ("binary", "in"):
                node.token.error("while condition must be boolean")
            return alias
        if node.kind == "unary" and node.args[0] == "!":
            yes, no, code = self.condition(node.args[1], state)
            return no, yes, f"(!({code}))"
        if self.span_kind(node) == "ordinary":
            if require_boolean:
                node.token.error("while condition must be boolean")
            return self.condition(Node(node.token, "length", (node,)), state)
        packed = self.packed_truth(node, state)
        if packed is not None:
            if require_boolean:
                node.token.error("while condition must be boolean")
            if packed.always_returns:
                return [], [], packed.code
            return [state], [], packed.code
        if node.kind == "rank_compare":
            op, _, left, right = node.args
            value = self.compare_ranked(node, state)
            if value.always_returns:
                return [], [], value.code
            yes = [state] if value.domain.hi else []
            no = [state] if not value.domain.lo else []
            matching = yes if op == "===" else no
            if matching:
                comparison = Node(node.token, "binary", (op, left, right))
                if self.span_kind(left) == self.span_kind(right) == "ordinary":
                    matching[:] = self.span_equality_paths(comparison, state)
                elif not self.has_effects(comparison, allow_debug_checks=True):
                    matching[:] = self.aggregate_equality_paths(left, right, state, ranked=True)
            return yes, no, value.code
        if node.kind == "binary":
            op, left, right = node.args
            if op in ("&&", "||", "^^"):
                yes, no, left_code = self.condition(left, state)
                first, second = (yes, no) if op == "&&" else (no, yes)
                right_yes, right_no = [], []
                if node not in self.rhs_codes:
                    self.rhs_codes[node] = (len(self.rhs_codes), "0", False)
                tag = self.rhs_codes[node][0]
                groups = ((yes, True), (no, False)) if op == "^^" else ((first, False),)
                for paths, invert in groups:
                    for path in paths:
                        ry, rn, right_code = self.condition(right, path)
                        continues = bool(ry or rn)
                        # A shortened error path must not replace code also used
                        # by inputs that reach the end of the right operand.
                        _, saved_code, saved_continues = self.rhs_codes[node]
                        if continues or (not saved_continues and
                                         self.propagation_code_rank(right_code) >= self.propagation_code_rank(saved_code)):
                            self.rhs_codes[node] = (tag, right_code, continues)
                        right_yes.extend(rn if invert else ry)
                        right_no.extend(ry if invert else rn)
                base_epochs, base_origins = self.current_pinned_baseline()
                if op == "^^":
                    right_yes = PathState.fold(right_yes, base_epochs, base_origins)
                    right_no = PathState.fold(right_no, base_epochs, base_origins)
                    self.budget_paths(right_yes + right_no, node.token)
                    code = f"((!!({left_code})) != (!!(@{tag}@)))"
                    return right_yes, right_no, code
                code = f"(({left_code}) ? 1 : !!(@{tag}@))" if op == "||" else f"(({left_code}) ? !!(@{tag}@) : 0)"
                if op == "&&":
                    return (PathState.fold(right_yes, base_epochs, base_origins),
                            PathState.fold(second + right_no, base_epochs, base_origins), code)
                return (PathState.fold(second + right_yes, base_epochs, base_origins),
                        PathState.fold(right_no, base_epochs, base_origins), code)
            rewritten = self.rewrite_property_binary(node, state)
            if rewritten is not None:
                if require_boolean and not rewritten.boolean:
                    node.token.error("while condition must be boolean")
                yes = [state] if rewritten.domain - ZERO else []
                no = [state] if rewritten.domain & ZERO else []
                return yes, no, rewritten.code
            span_compare = self.compare_spans(node, state)
            if span_compare is not None:
                if span_compare.always_returns:
                    return [], [], span_compare.code
                yes = [state] if span_compare.domain.hi else []
                no = [state] if not span_compare.domain.lo else []
                if op == '===' and yes:
                    yes = self.span_equality_paths(node, state)
                elif op == '!==' and no:
                    no = self.span_equality_paths(node, state)
                elif op in ("==", "!=") and yes and no:
                    yes, no = self.shallow_address_paths(node, state, yes, no)
                return yes, no, span_compare.code
            op = {"===": "==", "!==": "!="}.get(op, op)
            opposites = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}
            if op in opposites:
                x = self.expr(left, state)
                track_exit = id(node) in self.propagating_expression_nodes
                if track_exit and x.always_returns:
                    return [], [], x.code
                before = state.forms, state.arrays, state.bits
                y = self.expr(right, state)
                if track_exit and y.always_returns:
                    return [], [], f"((void)({x.code}), ({y.code}))"
                x, y = self.fresh_value(x, state), self.fresh_value(y, state)
                test = self.binary_expression(node.token, op, left, right, x, y, state)
                if before != (state.forms, state.arrays, state.bits) or self.has_effects(right):
                    left = Node(left.token, "snapshot", ())
                if x.floating or y.floating:
                    yes = self.float_compare_path(state,left,right,x,y,op,True,test.domain.hi == 1)
                    no = self.float_compare_path(state,left,right,x,y,op,False,test.domain.lo == 0)
                    return yes,no,test.code
                difference = self.linear_difference("-", x, y)
                samples = x, y
                yes = self.compare_path(state, left, right, x.domain, y.domain, op, test.domain.hi == 1,
                                        difference, samples)
                # A loop head has to rejoin every class its body can produce;
                # until the widening step tracks residue classes the guard
                # only refines straight-line code.
                if op == "==" and yes and not self.loops:
                    guard = self.modulo_guard(left, right)
                    if guard is not None:
                        subject, sym, residue, modulus = guard
                        refined = []
                        for path in yes:
                            current = path.values.get(sym.uid, sym.ty.domain)
                            narrowed = current.with_congruence(residue, modulus)
                            refined.extend(self.restrict(path, subject, narrowed))
                        yes = refined
                no = self.compare_path(state, left, right, x.domain, y.domain,
                                       opposites[op], test.domain.lo == 0, difference, samples)
                if not self.has_effects(node):
                    if difference is None or op == "!=":
                        yes = [self.remember_integer_comparison(path, x, op, y) for path in yes]
                    if difference is None or opposites[op] == "!=":
                        no = [self.remember_integer_comparison(path, x, opposites[op], y) for path in no]
                return yes, no, test.code
        if node.kind == "guard_outcomes":
            yes, _, _ = self.condition(node.args[0], state)
            no, _, _ = self.condition(node.args[1], state)
            return yes, no, "0"
        if node.kind == "guard_choose":
            test, yes_guard, no_guard = node.args
            hit, miss, _ = self.condition(test, state)
            yes, no = [], []
            for states, guard in ((hit, yes_guard), (miss, no_guard)):
                for path in states:
                    true_paths, false_paths, _ = self.condition(guard, path)
                    yes.extend(true_paths)
                    no.extend(false_paths)
            self.budget_paths(yes + no, node.token)
            return yes, no, "0"
        if node.kind == "guard_shift":
            op, operation, source, amount, expected = node.args
            count_value = self.expr(amount, state)
            counts = self.bounded_window_values(count_value.domain, 8)
            expected_value = self.expr(expected, state) if counts else None
            values = (self.bounded_window_values(expected_value.domain, 8 // len(counts))
                      if expected_value is not None else None)
            if values is None:
                shifted = Node(node.token, "binary", (operation, source, amount))
                return self.condition(Node(node.token, "binary", (op, shifted, expected)), state)
            yes, no = [], []
            for count in counts:
                selected = Domain.point(count)
                paths = self.restrict_guard_operand(state, amount, count_value, selected)
                for path in paths:
                    for value in values:
                        domain = self.shift_preimage(op, operation, value, count)
                        guard = Node(node.token, "guard_domain", (source, domain))
                        for candidate in self.restrict_guard_operand(path, expected, expected_value, Domain.point(value)):
                            hit, miss, _ = self.condition(guard, candidate)
                            yes.extend(hit)
                            no.extend(miss)
            self.budget_paths(yes + no, node.token)
            return yes, no, "0"
        if node.kind == "guard_mask":
            expr, mask, expected, negate = node.args
            value = self.expr(expr, state)
            selected = mask_preimage(value.domain, mask, expected)
            if selected is None:
                return [state], [state], "0"
            other = value.domain - selected
            yes, no = (other, selected) if negate else (selected, other)
            hit = self.restrict_guard_operand(state, expr, value, yes)
            miss = self.restrict_guard_operand(state, expr, value, no)
            return hit, miss, "0"
        if node.kind == "guard_domain":
            expr, domain = node.args
            value = self.expr(expr, state)
            yes = self.restrict_guard_operand(state, expr, value, value.domain & domain)
            no = self.restrict_guard_operand(state, expr, value, value.domain - domain)
            return yes, no, "0"
        if node.kind == "in":
            expr, ty = node.args
            val = self.expr(expr, state)
            if is_callback_enum(val.enum):
                node.token.error("function values do not support numeric membership tests")
            member_type = None if val.floating else self.numeric_type(ty)
            domain = self.float_set(ty, val.domain.bits) if val.floating else member_type.domain
            code = f"{self.membership(node.token, domain if val.floating else domain & WIDE)}({val.code})"
            if val.always_returns:
                return [], [], code
            enum_guard = member_type is not None and member_type.enum is not None and val.enum is None
            yes_domain, no_domain = val.domain & domain, val.domain - domain
            yes = self.refine_return_paths(self.restrict(state, expr, yes_domain, enum_guard=enum_guard),
                                           val, yes_domain)
            if enum_guard:
                key = self.predicate_key(expr, state, allow_debug_checks=True)
                if key is not None:
                    key = ("enum_membership", key)
                    yes = [replace(path, predicates=path.predicates | {
                        key: path.predicates.get(key, frozenset()) | {member_type.enum.identity}})
                           for path in yes]
            no = self.refine_return_paths(self.restrict(state, expr, no_domain, enum_guard=enum_guard),
                                          val, no_domain)
            return yes, no, code
        summary = self.call_guard(node)
        if summary is not None:
            arguments = node.args[1]
            if any(self.has_effects(arg) for arg in arguments):
                values, places, early_return = [], {}, []
                ty, code = self.call(node, state, values, guard_places=places, capture_early_return=early_return)
                if early_return:
                    return [], [], code
                code = self.assume_scalar_result(node.token, self.composite_call_signature(node).result, code)
                captured = []
                for index, (arg, value) in enumerate(zip(arguments, values)):
                    if index in places and state.epochs.get(places[index][0]) is places[index][1]:
                        captured.append(arg)
                    else:
                        if isinstance(value, Span):
                            value = value.length
                        if not isinstance(value, Value):
                            # Fixed-array lengths are constants in the summary.
                            captured.append(arg)
                            continue
                        snapshot = replace(self.fresh_value(value, state), code="0", effectful=False)
                        captured.append(Node(arg.token, "guard_snapshot", (snapshot,)))
                arguments = captured
            else:
                code = self.expr(node, state).code
            guard = self.instantiate_call_guard(node, summary, arguments, state)
            yes, no, _ = self.condition(guard, state)
            self.trace.append(f"  apply call guard result at {node.token.line}:{node.token.col}")
            return yes, no, code
        val = self.expr(node, state)
        if val.always_returns:
            return [], [], val.code
        if is_callback_enum(val.enum):
            node.token.error("a function value cannot be used as a condition")
        if require_boolean and not val.boolean:
            node.token.error("while condition must be boolean")
        if val.floating:
            return self.restrict(state,node,val.domain.without_zero()), self.restrict(state,node,val.domain.only_zero()), val.code
        yes, no = val.domain - ZERO, val.domain & ZERO
        return (self.refine_return_paths(self.restrict(state, node, yes), val, yes),
                self.refine_return_paths(self.restrict(state, node, no), val, no), val.code)

    def trace_bit_field_states(self, path):
        if not path.bit_values and not path.bits:
            return
        for scope in self.scope:
            for view in scope.values():
                if not isinstance(view.ty, BitView):
                    continue
                host = view.ty.host
                targets = self.physical_places(host, path, allow_null=True)
                alive = frozenset(("Alive",))
                if not targets or any(not target.uid or path.heap.get(target.heap_owner or target.uid, alive) != alive
                                      for target in targets):
                    continue
                for name, bit in view.ty.fields.items():
                    locations = self.bit_fact_locations(path, host, bit.offset)
                    if not locations:
                        continue
                    if host.bit_overlay:
                        domain = Domain.union(self.bit_window_fact(path, target, start, bit.width, bit.endian)
                                              for target, start in locations)
                    else:
                        domain = self.bit_read_domain(path, host, bit.offset, bit.width, bit.endian)
                    raw_top = container(bit.width, False)
                    if bit.ty.floating:
                        top = FloatDomain.top(bit.width)
                        exact = FloatDomain.union(
                            path.bit_values.get((target.uid, start, bit.width, "float:" + bit.endian), top)
                            for target, start in locations)
                        domain = FloatDomain.from_raw_bits(domain, bit.width) & exact
                        if domain == top:
                            continue
                    elif domain == raw_top:
                        continue
                    elif bit.ty.signed:
                        sign = 1 << (bit.width - 1)
                        negative = (domain & Domain.of((sign, 2 * sign - 1))).affine(1, -2 * sign)
                        domain = (domain & Domain.of((0, sign - 1))) | negative
                    self.trace.append(f"    state(id#{view.uid}.{name}) := {domain}")
