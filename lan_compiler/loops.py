"""Loop thresholds, widening, induction bounds and transitions."""

from __future__ import annotations

from dataclasses import field, fields, replace
from itertools import count

from semantic_ir import LoopRecurrence
from float_domain import FloatDomain
from value_domain import Domain

from .errors import OracleRequest, ProofBudgetError
from .syntax import Node
from .typesys import Array, BitField, HeapArray, Scalar, Slice
from .domain_utils import domain_union, physical, replay_scalar
from .state import CountedLoop, LoopDelta
from .c_emit import indent


class LoopMixin:
    def loop_thresholds(self, condition):
        # Include neighbors for strict comparisons and for the first exit value.
        thresholds = set()

        def add(*values):
            for value in values:
                thresholds.update((value - 1, value, value + 1))

        pending = [condition]
        while pending:
            node = pending.pop()
            if not isinstance(node, Node):
                if isinstance(node, tuple):
                    pending.extend(node)
                continue
            if node.kind == "number":
                value = node.args[0]
                add(value, -value)
            elif node.kind == "interval":
                pending.extend(node.args[:2])
            elif node.kind == "set":
                pending.extend(node.args[0])
            elif node.kind == "type_name":
                ty = self.types.get(node.args[0])
                if isinstance(ty, Scalar):
                    for lo, hi in ty.domain.ranges:
                        add(lo, hi)
            elif node.kind == "length" and node.args[0].kind == "name":
                name = node.args[0].args[0]
                for scope in reversed(self.scope):
                    if name in scope:
                        ty = scope[name].ty
                        if isinstance(ty, Array):
                            add(ty.length)
                        break
            else:
                pending.extend(node.args)
        return thresholds

    def widen_loop_head(self, previous, joined, thresholds):
        result = {}
        for uid, new in joined.items():
            # A block-local binding can survive in the joined state when a
            # linked heap node keeps it reachable.  It is not a loop-carried
            # variable, and after the body's symbol table is rolled back its
            # uid may name no live symbol at all.
            if uid not in previous:
                continue
            old = previous[uid]
            if not 0 < uid <= len(self.symbols):
                # The binding outlived its symbol table entry; keep the value
                # it already had instead of asking for a type that is gone.
                result[uid] = old
                continue
            if isinstance(new, FloatDomain) and new != old:
                result[uid] = self.symbols[uid - 1].ty.domain
                continue
            if new == old:
                result[uid] = old
                continue
            allowed = self.symbols[uid - 1].ty.domain
            lo, hi = new.lo, new.hi
            if lo < old.lo:
                lo = max((n for n in thresholds if allowed.lo <= n <= lo), default=allowed.lo)
            if hi > old.hi:
                hi = min((n for n in thresholds if hi <= n <= allowed.hi), default=allowed.hi)
            result[uid] = Domain.of((lo, hi)) & allowed
        return result

    def loop_induction_bounds(self, stmt, paths, counted):
        """Bound scalar writes using a stable limit and progress on every back edge."""
        condition, body = stmt.args[:2]
        visible = {name: symbol for scope in self.scope for name, symbol in scope.items()}
        operations = tuple(self.body_operations((condition, body)))
        if len(operations) > 512:
            return {}
        # An overlay, rebind or nested loop can write through a different ID.
        if any(item.kind in ("propagate", "while", "interation", "bit_view")
               or item.kind == "cast" and item.args[1] == "view" for item in operations):
            return {}

        def by_value_call(node):
            callee = node.args[0]
            if callee.kind != "name":
                return False
            name, = callee.args
            local = self.functions.get(name)
            if local is not None:
                contract = local[1]
                clean = (self.mutations.get(name) == () and name not in self.memory_effects
                         and name not in self.external_effects)
            else:
                binding = self.externs.get(name) or self.imported_functions.get(name)
                if binding is None:
                    return False
                contract = binding.contract
                clean = binding.pure
            return clean and all(isinstance(param, Scalar) for param in contract.params)

        if any(item.kind == "call" and not by_value_call(item) for item in operations):
            return {}

        borrowed = {target for path in paths for targets in path.borrows.values() for target in targets}

        def integer(symbol):
            return (symbol is not None and replay_scalar(symbol.ty) and not symbol.ty.boolean
                    and all(symbol.uid in path.values for path in paths))

        def direct(symbol):
            return (not (symbol.borrowed or symbol.heap_owner or symbol.c_place or symbol.bit_overlay
                         or symbol.link_access is not None or symbol.projection is not None)
                    and symbol.uid not in self.alias_members
                    and symbol not in borrowed)

        writes = set()
        for item in operations:
            if item.kind == "define":
                name, annotation, _ = item.args
                if annotation is not None and annotation.kind == "refine":
                    annotation = annotation.args[0]
                if name.text in visible or (annotation is not None and (
                        annotation.kind != "type_name" or not isinstance(self.types.get(annotation.args[0]), Scalar))):
                    return {}
            if item.kind == "assign":
                target = item.args[0]
                symbol = visible.get(target.args[0]) if target.kind == "name" else None
                if symbol is None or not direct(symbol) or not replay_scalar(symbol.ty):
                    return {}
                writes.add(symbol.name)
        if not writes or len(writes) * len(operations) > 8192:
            return {}
        unstable = writes if counted is None else writes | {counted.statement.args[0].text}

        entry_ranges = {}

        def entry(name):
            if name not in entry_ranges:
                uid = visible[name].uid
                entry_ranges[name] = (min(path.values[uid].lo for path in paths),
                                      max(path.values[uid].hi for path in paths))
            return entry_ranges[name]

        def term(node, counter=None, induction=None, progress=LoopDelta()):
            if node.kind == "number":
                return LoopDelta.constant(node.args[0], node.args[0])
            if node.kind == "name":
                name, = node.args
                if name == counter:
                    return induction.add(progress) if induction is not None else None
                if name not in unstable and integer(visible.get(name)) and direct(visible[name]):
                    return LoopDelta.constant(*entry(name))
            if node.kind in ("length", "index") and node.args[0].kind == "name":
                name, = node.args[0].args
                symbol = visible.get(name)
                if symbol is None or not isinstance(symbol.ty, (Array, Slice, HeapArray)):
                    return None
                if node.kind == "index":
                    element = symbol.ty.element
                    if replay_scalar(element) and not element.boolean:
                        return LoopDelta.constant(element.domain.lo, element.domain.hi)
                elif isinstance(symbol.ty, Array):
                    return LoopDelta.constant(symbol.ty.length, symbol.ty.length)
                elif (name not in unstable and symbol.projection is None and symbol.link_access is None
                      and symbol.uid not in self.alias_members
                      and all(symbol.uid in path.values for path in paths)):
                    return LoopDelta.constant(*entry(name))
            if node.kind == "unary" and node.args[0] in ("+", "-"):
                value = term(node.args[1], counter, induction, progress)
                return value.scale(-1 if node.args[0] == "-" else 1) if value is not None else None
            if node.kind == "cast" and node.args[1] == "safe":
                annotation = node.args[2]
                ty = self.types.get(annotation.args[0]) if annotation.kind == "type_name" else None
                if replay_scalar(ty) and not ty.boolean:
                    return term(node.args[0], counter, induction, progress)
            if node.kind == "binary" and node.args[0] in ("+", "-", "*"):
                op, left, right = node.args
                left = term(left, counter, induction, progress)
                right = term(right, counter, induction, progress)
                if left is None or right is None:
                    return None
                if op == "*":
                    factor = left.constant_value()
                    if factor is not None:
                        return right.scale(factor)
                    factor = right.constant_value()
                    return left.scale(factor) if factor is not None else None
                return left.add(right.scale(-1) if op == "-" else right)
            return None

        def increment(item, counter=None, induction=None, progress=LoopDelta()):
            target, op, value = item.args
            op = op.text
            if op == "=" and value.kind == "binary":
                operator, left, right = value.args
                if operator == "+" and right.kind == "name" and right.args == target.args:
                    left, right = right, left
                if operator in ("+", "-") and left.kind == "name" and left.args == target.args:
                    op, value = operator + "=", right
            if op not in ("+=", "-="):
                return None
            change = term(value, counter, induction, progress)
            return change.scale(-1) if change is not None and op == "-=" else change

        def summarize(statements, name, counter=None, induction=None, *,
                      change=LoopDelta(), progress=LoopDelta()):
            """Collect deltas on fallthrough and continue edges."""
            back = None
            for item in statements:
                if change is None:
                    break
                if item.kind == "if":
                    _, yes, no = item.args
                    left, left_progress, left_back = summarize(
                        yes, name, counter, induction, change=change, progress=progress)
                    right, right_progress, right_back = summarize(
                        no, name, counter, induction, change=change, progress=progress)
                    progress = merge(left_progress, right_progress)
                    change = merge(left, right)
                    back = merge(back, merge(left_back, right_back))
                elif item.kind == "assign":
                    target, = item.args[0].args
                    if target == name:
                        delta = increment(item, counter, induction, progress)
                        if delta is None:
                            raise ValueError("unsupported loop update")
                        change = change.add(delta)
                    if target == counter:
                        progress = progress.add(increment(item))
                elif item.kind in ("return", "continue"):
                    if item.kind == "continue":
                        back = merge(back, change)
                    change = progress = None
            return change, progress, back

        def merge(left, right):
            return right if left is None else left if right is None else left.hull(right)

        def comparison(node):
            op, left, right = node.args
            reverse = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
            if left.kind != "name" or left.args[0] not in writes:
                left, right, op = right, left, reverse[op]
            if left.kind != "name" or left.args[0] not in writes:
                return None
            name, = left.args
            if not integer(visible[name]):
                return None
            limit = term(right)
            if limit is None:
                return None
            try:
                normal, _, continued = summarize(body, name)
            except ValueError:
                return None
            step = merge(normal, continued)
            step = step.constant_value() if step is not None else None
            increasing = op in ("<", "<=")
            if not step or (step > 0) != increasing:
                return None
            lo, hi = entry(name)
            distance = limit.upper[1] - lo if increasing else hi - limit.lower[1]
            distance += op in ("<=", ">=")
            turns = max(0, (distance + abs(step) - 1) // abs(step))
            return turns, name, LoopDelta((step, lo), (step, hi))

        candidates = []
        if counted is not None:
            # The generated latch writes the user-visible counter separately.
            # Summarize the body only when it does not write that counter.
            if counted.limit <= 8:
                return {}
            candidates.append((counted.limit, None, None))
        else:
            pending = [condition]
            while pending:
                test = pending.pop()
                if test.kind == "binary":
                    if test.args[0] == "&&":
                        pending.extend(test.args[1:])
                    elif test.args[0] in ("<", "<=", ">", ">="):
                        candidate = comparison(test)
                        if candidate is not None:
                            candidates.append(candidate)
        if not candidates:
            return {}
        turns, counter, induction = min(candidates, key=lambda item: item[0])
        if not turns:
            return {}
        bounds = {}
        for name in sorted(writes):
            if not integer(visible[name]):
                continue
            try:
                normal, _, continued = summarize(body, name, counter, induction)
            except ValueError:
                continue
            back = merge(normal, continued)
            if back is None:
                continue
            lo, hi = entry(name)
            body_lo, body_hi = back.prefix(turns - 1)
            head_lo, head_hi = back.prefix(turns)
            bounds[visible[name].uid] = (Domain.of((lo + body_lo, hi + body_hi)),
                                         Domain.of((lo + head_lo, hi + head_hi)))
        return bounds

    def possible_loop_writes(self, stmt, bindings, paths):
        writes = set()
        unresolved = False

        def locate(expr, scope):
            if expr.kind == "name":
                return scope.get(expr.args[0])
            if expr.kind in ("index", "slice"):
                return locate(expr.args[0], scope)
            if expr.kind in ("member", "length"):
                parent = locate(expr.args[0], scope)
                if parent is not None:
                    name = "len" if expr.kind == "length" else expr.args[1].text
                    return next((child for child in parent.fields if child.name == name), parent)
            return None

        def record(symbol):
            nonlocal unresolved
            if symbol is None:
                unresolved = True
                return
            writes.add(symbol)
            if symbol.uid in self.alias_members:
                root, fields = self.alias_members[symbol.uid]
                for path in paths:
                    for target in path.borrows.get(root, ()):
                        if target.uid:
                            writes.add(self.child_place(target, fields))

        def expression(value, scope):
            nonlocal unresolved
            if isinstance(value, Node):
                if value.kind == "call" and value.args[0].kind == "name":
                    name, = value.args[0].args
                    args = value.args[1]
                    if name == "free" and len(args) == 1:
                        record(locate(args[0], scope))
                    else:
                        summary = self.mutations.get(name)
                        if summary is None:
                            binding = self.externs.get(name) or self.imported_functions.get(name)
                            if binding is not None:
                                summary = binding.mutations
                                if summary is None:
                                    summary = self.unknown_mutations(binding.contract)
                        if summary is None:
                            unresolved = True
                        else:
                            for index, fields, _ in summary:
                                if index < len(args) and args[index].kind != "null":
                                    symbol = locate(args[index], scope)
                                    for field in fields:
                                        if symbol is not None:
                                            symbol = next((child for child in symbol.fields if child.name == field), symbol)
                                    record(symbol)
                for child in value.args:
                    expression(child, scope)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    expression(child, scope)

        def block(statements, outer):
            scope = outer.copy()
            for node in statements:
                if node.kind == "define":
                    name, _, value = node.args
                    expression(value, scope)
                    scope[name.text] = None
                elif node.kind == "assign":
                    target, _, value = node.args
                    record(locate(target, scope))
                    expression(target, scope)
                    expression(value, scope)
                elif node.kind == "if":
                    guard, yes, no = node.args
                    expression(guard, scope)
                    block(yes, scope)
                    block(no, scope)
                elif node.kind == "while":
                    guard, body = node.args[:2]
                    expression(guard, scope)
                    block(body, scope)
                elif node.kind == "interation":
                    name, initial, _, count, step, body = node.args
                    for operand in (initial, count, step):
                        expression(operand, scope)
                    block(body, scope | {name.text: None})
                else:
                    expression(node, scope)

        condition, body = stmt.args[:2]
        expression(condition, bindings)
        block(body, bindings)
        return writes, unresolved

    def loop(self, stmt, paths, following, *, counted=None):
        token = stmt.token
        description = self.conditions[token] if counted is None else "interation count"
        self.trace.append(f"  loop cond [{token.line}:{token.col}] ({description})")
        bindings = {name: symbol for scope in self.scope for name, symbol in scope.items()}
        frames = getattr(self, "transition_frames", None)
        if frames is None:
            frames = self.transition_frames = []
        writes = {}
        cutoff = len(self.symbols)
        trace_start = len(self.trace)
        frames.append(writes)
        try:
            if counted is None:
                result = self.verify_loop_transition(stmt, paths, following)
                self.record_loop_exit_theorem(stmt, result[0])
            else:
                result = self.analyze_loop(stmt, paths, counted=counted)
            parameters = []
            carried = []
            for target, indices in writes.items():
                symbol, aspect = target[:2] if isinstance(target, tuple) else (target, None)
                if symbol.uid > cutoff:
                    continue
                carried.append((target, indices))
                name = f"id#{symbol.uid} {symbol.name}"
                if aspect is not None:
                    name += f".{aspect}"
                elif indices is not None:
                    name += f" indices {Domain.of(*indices)}"
                parameters.append((symbol.uid, name))
            signature = ", ".join(name for _, name in sorted(parameters))
            kernel = f"  loop kernel @{stmt.token.line}:{stmt.token.col} ({signature})"
            self.trace.insert(trace_start, kernel)
            self.trace.append(f"  end loop kernel @{stmt.token.line}:{stmt.token.col}")
            anchor = getattr(counted.statement if counted else stmt, "semantic_anchor", None)
            condition_anchor = getattr(stmt.args[0], "semantic_anchor", None)
            if anchor is not None and (condition_anchor is not None or counted is not None):
                previous = self.loop_recurrences.get(anchor)
                if previous is not None:
                    carried = list(previous.writes) + [entry for entry in carried if entry not in previous.writes]
                count_anchor = getattr(counted.statement.args[3], "semantic_anchor", None) if counted else None
                self.loop_recurrences[anchor] = LoopRecurrence(condition_anchor, tuple(carried), count_anchor)
            return result
        except ProofBudgetError as exc:
            if exc.request is None and counted is None:
                entries, unknowns = [], []
                possible, unresolved = self.possible_loop_writes(stmt, bindings, paths)
                observed = {key[0] if isinstance(key, tuple) else key for key in writes}
                for sym in sorted(observed | possible, key=lambda symbol: symbol.uid):
                    if sym.uid > cutoff:
                        continue
                    values = [path.values[sym.uid] for path in paths if sym.uid in path.values]
                    known = domain_union(values) if values else "tracked storage"
                    status = "observed" if sym in observed else "possible"
                    entries.append(f"id#{sym.uid} {sym.name} entry: {known} ({status} write)")
                    unknowns.append(f"{sym.name}: ?")
                if unknowns:
                    entries.insert(0, "@transition => { " + ", ".join(unknowns) + " }")
                if unresolved:
                    entries.append("Some write targets need a lifetime or alias proof.")
                if not entries:
                    entries.append("No outer storage writes were identified.")
                exc.request = OracleRequest(self.source_path, self.source_digest,
                                            stmt.token.line, stmt.token.col, tuple(entries))
            raise
        finally:
            frames.pop()

    def iteration_type(self, annotation, env=()):
        ty = self.resolve(annotation, env)
        if (not isinstance(ty, Scalar) or ty.boolean or ty.enum or ty.character
                or ty.result_arms is not None):
            annotation.token.error("interation variable type must support +")
        self.check_intersection_type(ty, annotation.token)
        return physical(ty, annotation.token)

    @staticmethod
    def iteration_operand(token, value, *, floating):
        if (value.boolean or value.enum or value.character or value.result_arms is not None
                or value.floating and not floating):
            token.error("interation requires a nonnegative integer count and a numeric step compatible with +")

    def iteration_read_only(self, body, name):
        # The counter can be mutated through a binding, an argument or a view.
        # Leave those bodies to the ordinary loop analysis.
        def mentions(expr):
            return any(item.kind == "name" and item.args[0] == name for item in self.body_operations(expr))

        for item in self.body_operations(body):
            if item.kind == "assign":
                target = item.args[0]
                while target.kind in ("member", "index", "slice", "bit_view"):
                    target = target.args[0]
                if target.kind == "name" and target.args[0] == name:
                    return False
            if item.kind == "call" and mentions(item.args[1]):
                return False
            if item.kind == "cast" and mentions(item):
                return False
            if (item.kind == "define" and item.args[1] is not None
                    and item.args[1].kind == "slice_qualified" and mentions(item.args[2])):
                return False
            if item.kind == "assign" and item.args[1].text == "=" and mentions(item.args[2]):
                # Rebinding an existing alias can expose the loop variable to
                # a later write through a different name.
                if item.args[2].kind == "name":
                    return False
        return True

    def interation(self, stmt, paths):
        name, initial, annotation, count, step, body = stmt.args
        token = stmt.token
        ty = self.iteration_type(annotation)
        self.scope.append({})
        lines = []
        previous = self.propagating_expression_nodes

        def capture(binding, expression, target=None):
            nonlocal paths
            self.propagating_expression_nodes = self.propagation_nodes(expression)
            definition = Node(binding, "define", (binding, None, expression))
            validator = None
            if binding == limit:
                validator = lambda value: self.iteration_operand(expression.token, value, floating=False)
            elif binding == stride:
                validator = lambda value: self.iteration_operand(expression.token, value, floating=ty.floating)
            paths, code = self.define_scalar(definition, paths, target, False, value_check=validator)
            lines.extend(code)

        def temporary(label):
            return replace(token, text=f"$interation_{label}_{token.line}_{token.col}")

        def ref(binding):
            return Node(binding, "name", (binding.text,))

        limit, stride, turn = map(temporary, ("count", "step", "turn"))
        variable = None
        try:
            capture(name, initial, ty)
            if paths:
                variable = self.scope[-1].pop(name.text)
                capture(limit, count, self.types["uint64"])
            if paths:
                capture(stride, step, self.types[f"float{ty.bits}"] if ty.floating else None)
            if paths:
                self.scope[-1][name.text] = variable
                # The count and step are captured once. The hidden turn is
                # independent of assignments to the user-visible variable.
                capture(turn, Node(token, "number", (0,)), self.types["uint64"])
                self.propagating_expression_nodes = frozenset()
                limit_symbol = self.scope[-1][limit.text]
                upper = max(path.values[limit_symbol.uid].hi for path in paths)
                if upper:
                    condition = Node(token, "binary", ("<", ref(turn), ref(limit)))
                    plus = replace(step.token, text="+=")
                    updates = (
                        Node(step.token, "assign", (ref(name), plus, ref(stride))),
                        Node(token, "assign", (ref(turn), replace(token, text="+="), Node(token, "number", (1,)))),
                    )
                    counted = CountedLoop(stmt, upper, updates)
                    if upper > 8 and not ty.floating and self.iteration_read_only(body, name.text):
                        strides = domain_union(path.values[self.scope[-1][stride.text].uid] for path in paths)
                        starts = domain_union(path.values[variable.uid] for path in paths)
                        lower_step, upper_step = min(0, strides.lo), max(0, strides.hi)
                        counted = replace(
                            counted, induction=variable,
                            body_domain=Domain.of((starts.lo + (upper - 1) * lower_step,
                                                   starts.hi + (upper - 1) * upper_step)),
                            head_domain=Domain.of((starts.lo + upper * lower_step,
                                                   starts.hi + upper * upper_step)))
                    paths, code = self.loop(Node(token, "while", (condition, body)), paths, (),
                                            counted=counted)
                    lines.extend(code)
            self.trace_stack_exit(paths, len(self.scope) - 1, "interation exit")
            ids = {sym.uid for sym in self.scope[-1].values()}
            if variable is not None:
                ids.add(variable.uid)
            paths = [path.project(path.values.keys() - ids, None, len(self.symbols)) for path in paths]
            return paths, ["{", *indent(lines), "}"]
        finally:
            self.propagating_expression_nodes = previous
            self.scope.pop()

    def slice_transition_region(self, sym, index, token, path, *, allow_row=False):
        dynamic = isinstance(index, Node)
        if dynamic:
            self.prove_loop_endpoint_reads(index, path)
        base = index if dynamic else Node(token, "name", (sym.name,))
        span = self.refine_span_position(path, self.ordinary_span(base, path))
        if dynamic and self.debug_bounds and not span.bounds_proven:
            token.error("loop transition fails at exit: cannot prove slice bounds")
        start, stop = (0, None) if dynamic else index
        tail = stop is None
        stop = span.length.domain.lo if tail else stop
        if stop > span.length.domain.lo or start > stop:
            token.error("loop transition fails at exit: cannot prove slice bounds")
        positions = span.positions()
        if not positions:
            return None
        owner = positions[0][0]
        if (owner.projection is not None and not (allow_row and self.array_row_write_key(owner) is not None)
                or any(candidate != owner or offset is None for candidate, offset in positions)):
            return None
        first = max(offset.hi for _, offset in positions) + start
        last = min(offset.lo for _, offset in positions) + stop
        if tail:
            end = self.span_end_domain(token, span, path, positions)
            if end is not None:
                last = max(last, end.lo)
        return owner, first, max(first, last)

    def fixed_projected_field(self, symbol, *, symbolic=False, state=None):
        projection = symbol.projection
        if (projection is None or projection.captured is not None or not isinstance(symbol.ty, Scalar)
                or not isinstance(projection.root.ty, (Array, Slice))
                or not any(isinstance(step, str) for step in projection.steps)):
            return None
        root = projection.root
        steps = tuple(self.loop_row_step(step) for step in projection.steps)
        if not all(step is not None and (symbolic or isinstance(step, (str, int))) for step in steps):
            return None
        if isinstance(root.ty, Slice) and not symbolic:
            if state is None or root.uid in state.arrays or not isinstance(steps[0], int):
                return None
            span = root.ty.projection
            positions = span.positions() if span is not None else state.regions.get(root.uid, ())
            if len(positions) != 1:
                return None
            root, offset = positions[0]
            if offset is None or offset.lo != offset.hi:
                return None
            steps = (offset.lo + steps[0], *steps[1:])
        if (not isinstance(root.ty, (Array, Slice)) or root.borrowed or root.uid in self.alias_members
                or root.link_access is not None or root.projection is not None):
            return None
        return root, steps

    @staticmethod
    def projected_scalar_region(symbol):
        projection = symbol.projection
        if projection is None or projection.captured is not None or len(projection.steps) != 1:
            return None
        root = projection.root
        index = projection.steps[0]
        if (not isinstance(root.ty, (Array, Slice, HeapArray)) or not isinstance(root.ty.element, Scalar)
                or not isinstance(index, Domain) or index.lo != index.hi):
            return None
        return root, index.lo, index.hi + 1

    def prove_loop_endpoint_reads(self, index, state):
        if not self.debug_bounds:
            return
        endpoints = tuple(endpoint for endpoint in index.args[1:3]
                          if endpoint is not None and self.has_effects(endpoint))
        if not endpoints:
            return
        self.debug_bounds = False
        try:
            for endpoint in endpoints:
                self.expr(endpoint, state)
        finally:
            self.debug_bounds = True

    def packed_transition_window(self, index, path):
        self.prove_loop_endpoint_reads(index, path)
        bits, start, width, _ = self.bit_region(index, path, require_proven=True)
        windows = [(offset, size) for offset, size in self.bit_window_candidates(start, width)
                   if offset + size <= bits.ty.length]
        first = max((offset for offset, _ in windows), default=start.domain.hi)
        stop = min((offset + size for offset, size in windows), default=start.domain.lo + width.domain.lo)
        return bits, first, max(first, stop)

    def loop_transition_coverage(self, claims, exits):
        fixed, dynamic, projected, array_aliases = {}, [], [], []
        for sym, _, token, index in claims.values():
            if sym.projection is not None:
                continue
            if index is None and sym.uid in self.alias_members:
                projected.append(sym)
            if isinstance(index, BitField):
                bits = Domain.of((index.offset, index.offset + index.width - 1))
                if sym.uid in self.alias_members or sym.link_access is not None:
                    array_aliases.append((sym, bits))
                else:
                    fixed[sym.uid] = fixed.get(sym.uid, Domain.of()) | bits
                continue
            if isinstance(index, Node):
                dynamic.append((sym, token, index))
                continue
            if not isinstance(index, tuple):
                continue
            if isinstance(sym.ty, Array):
                start, stop = index
                region = Domain.of((start, stop - 1))
                if sym.uid in self.alias_members or sym.link_access is not None:
                    array_aliases.append((sym, region))
                else:
                    fixed[sym.uid] = fixed.get(sym.uid, Domain.of()) | region
            else:
                dynamic.append((sym, token, index))
        if not (dynamic or projected or array_aliases) or not exits:
            return fixed, [fixed] * len(exits)
        coverage = None
        by_path = []
        for path in exits:
            current = fixed.copy()
            for sym, indices in array_aliases:
                targets = self.physical_places(sym, path, allow_null=True)
                if len(targets) == 1 and targets[0].uid and targets[0].projection is None:
                    uid = targets[0].uid
                    current[uid] = current.get(uid, Domain.of()) | indices
            for sym, token, index in dynamic:
                if isinstance(index, Node) and isinstance(sym.ty, Array) and sym.ty.packed_bits:
                    bits, start, stop = self.packed_transition_window(index, path)
                    targets = self.physical_places(bits, path, allow_null=True)
                    if len(targets) != 1 or not targets[0].uid or targets[0].projection is not None:
                        continue
                    owner = targets[0]
                else:
                    region = self.slice_transition_region(sym, index, token, path)
                    if region is None:
                        continue
                    owner, start, stop = region
                current[owner.uid] = current.get(owner.uid, Domain.of()) | Domain.of((start, stop - 1))
            for sym in projected:
                targets = self.physical_places(sym, path, allow_null=True)
                if len(targets) == 1:
                    region = self.projected_scalar_region(targets[0])
                    if region is not None:
                        owner, start, stop = region
                        current[owner.uid] = current.get(owner.uid, Domain.of()) | Domain.of((start, stop - 1))
            by_path.append(current)
            coverage = current if coverage is None else {
                uid: coverage[uid] & current[uid] for uid in coverage.keys() & current.keys()
            }
        return coverage, by_path
