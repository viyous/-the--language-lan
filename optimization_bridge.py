"""Replay Lan's pure scalar subset for bounded SSA and SCT proofs."""

from operator import add, sub, mul, lt, le, gt, ge

from ssa_verify import Obligation, Program
from size_change import Change, Transition, check_termination, infer_integer_transition
from value_domain import ACTIVE_SOLVER_STEPS, DEFAULT_SOLVER_STEPS


LIMITS = dict(timeout_ms=30, rlimit=20_000, max_nodes=3000)


def active_limits():
    steps = ACTIVE_SOLVER_STEPS.get()
    if steps <= DEFAULT_SOLVER_STEPS:
        return LIMITS
    scale = steps / DEFAULT_SOLVER_STEPS
    return dict(
        timeout_ms=min(60_000, int(LIMITS["timeout_ms"] * scale)),
        rlimit=min(100_000_000, int(LIMITS["rlimit"] * scale)),
        max_nodes=min(200_000, int(LIMITS["max_nodes"] * min(scale, 10))),
    )
ARITHMETIC = {"+": add, "-": sub, "*": mul}
ORDER = {"<": lt, "<=": le, ">": gt, ">=": ge}
SCALAR_NAMES = {"bool", "boolean", *(f"{sign}int{bits}"
                for sign in ("", "u") for bits in (8, 16, 32, 64))}


def source_key(node):
    anchor = getattr(node, "semantic_anchor", None)
    return node if anchor is None else anchor


def truth(value):
    return value if value.sort == "bool" else ~value.eq(0)


def expression(node, flow, on_call=None, guard=None, on_index=None, on_slice=None, on_cast=None):
    k, a = node.kind, node.args
    arena = flow.program.arena
    if guard is None:
        guard = flow.guard

    def read(child, path=guard):
        return expression(child, flow, on_call, path, on_index, on_slice, on_cast)

    def storage(child):
        if child.kind == "name":
            return flow.get(child.args[0])
        if child.kind == "slice" and on_slice is not None:
            return read(child)
        raise ValueError("storage outside scalar replay")

    if k == "name":
        value = flow.get(a[0])
        if value.sort == "storage":
            raise ValueError("composite value outside scalar replay")
        return value
    if k in ("number", "boolean") and type(a[0]) in (int, bool):
        return arena.expr(a[0])
    if k == "call" and on_call is not None and a[0].kind == "name":
        args = [read(arg) for arg in a[1]]
        return on_call(node, a[0].args[0], args, guard)
    if k == "index" and on_index is not None:
        return on_index(storage(a[0]), a[1], read(a[1]), guard)
    if k == "length" and on_index is not None:
        return on_index(storage(a[0]), None, None, guard)
    if k == "slice" and on_slice is not None:
        host, lo, hi, closed = a
        return on_slice(node, storage(host), read(lo) if lo is not None else None,
                        read(hi) if hi is not None else None, bool(hi is not None and closed), guard)
    if k == "cast" and on_cast is not None:
        return on_cast(node, read(a[0]), guard)
    if k == "unary":
        op, operand = a
        value = read(operand)
        if op == "!":
            return ~truth(value)
        if op in ("+", "-"):
            arena.expr(value, "int")
            return value if op == "+" else 0 - value
    if k == "binary":
        op, lhs, rhs = a
        left = read(lhs)
        right_guard = guard
        if op in ("&&", "||"):
            test = truth(left)
            right_guard = guard & (test if op == "&&" else ~test)
        right = read(rhs, right_guard)
        if op in ARITHMETIC:
            return ARITHMETIC[op](left, right)
        if op in ORDER:
            return ORDER[op](left, right)
        if op in ("==", "===", "!=", "!=="):
            equal = left.eq(right)
            return equal if op in ("==", "===") else ~equal
        if op == "&&":
            return truth(left) & truth(right)
        if op == "||":
            return truth(left) | truth(right)
    raise ValueError("expression outside scalar replay")


class ScalarReplay:
    def __init__(self, inputs, scalar_types=None, signatures=None, exchange=None):
        self.program = Program()
        self.returns = {}
        self.branches = 0
        self.statements = 0
        self.scalar_types = scalar_types
        self.signatures = signatures
        self.exchange = exchange
        self.arrays = {}
        self.operands = {}
        self.writes = {}
        self.calls = {}
        self.cache = {}
        for name, (boolean, lo, hi) in inputs.items():
            if boolean:
                value = (self.program.arena.expr(bool(lo)) if lo == hi and lo is not None
                         else self.program.boolean(name))
            else:
                value = self.program.integer(name, lo, hi)
            self.program.flow.declare(name, value)

    def fresh(self, shape):
        boolean, lo, hi = shape
        name = f"#replay:{len(self.program.arena.nodes)}"
        return self.program.boolean(name) if boolean else self.program.integer(name, lo, hi)

    def array_value(self, shape, length):
        arena = self.program.arena
        array = arena.intern("storage", "storage", data=len(self.arrays))
        self.arrays[array.id] = shape, arena.expr(length, "int")
        return array

    def declare_array(self, flow, name, shape, length):
        flow.declare(name, self.array_value(shape, length))

    def index(self, array, node, value, guard):
        shape, length = self.arrays[array.id]
        if node is None:
            return length
        self.operands[source_key(node)] = (guard, value)
        return self.fresh(shape)

    def indexed_write(self, target, op, source, flow, read):
        raise ValueError("array writes outside scalar replay")

    def read(self, node, flow, on_call=None):
        return expression(node, flow, on_call or (self.call if self.signatures is not None else None),
                          on_index=self.index if self.scalar_types is not None else None)

    def call(self, node, target, args, guard):
        parameters, result = self.signatures(target)
        if len(args) != len(parameters):
            raise ValueError("call arity differs from its contract")
        for argument, value, boolean in zip(node.args[1], args, parameters):
            self.program.arena.expr(value, "bool" if boolean else "int")
            self.operands[source_key(argument)] = (guard, value)
        self.calls[source_key(node)] = (guard, args)
        return self.fresh(result) if result is not None else self.program.arena.expr(0)

    def block(self, statements, flow, *, on_continue=None, on_call=None):
        def read(node):
            return self.read(node, flow, on_call)

        for node in statements:
            self.statements += 1
            if self.statements > 4096 or len(self.program.arena.nodes) > 20000:
                raise ValueError("scalar replay budget")
            k, a = node.kind, node.args
            if k == "define":
                name, annotation, value = a
                try:
                    flow.get(name.text)
                except KeyError:
                    pass
                else:
                    raise ValueError("shadowing outside scalar replay")
                if (self.scalar_types is not None and annotation is not None and annotation.kind == "array"
                        and annotation.args[0].kind == "type_name" and annotation.args[1].kind == "number"
                        and (value is None or value.kind in ("list", "repeat")
                             and all(item.kind in ("number", "boolean") for item in
                                     (value.args[0] if value.kind == "list" else value.args[:1])))):
                    shape = self.scalar_types[annotation.args[0].args[0]]
                    self.declare_array(flow, name.text, shape, annotation.args[1].args[0])
                    continue
                base = annotation
                if self.scalar_types is not None and base is not None and base.kind == "refine":
                    base = base.args[0]
                types = SCALAR_NAMES if self.scalar_types is None else self.scalar_types
                if base is not None and (base.kind != "type_name" or base.args[0] not in types):
                    raise ValueError("definition outside scalar replay")
                if value is None:
                    if self.scalar_types is None or base is None:
                        raise ValueError("definition outside scalar replay")
                    initial = self.program.arena.expr(False if self.scalar_types[base.args[0]][0] else 0)
                else:
                    initial = read(value)
                if initial.sort == "storage" and not (annotation is None and value.kind == "slice"):
                    raise ValueError("composite binding outside scalar replay")
                self.writes[source_key(node)] = (flow.guard, initial)
                flow.declare(name.text, initial)
            elif k == "assign":
                target, op, source = a
                if target.kind == "index":
                    self.writes[source_key(node)] = (flow.guard, self.indexed_write(target, op, source, flow, read))
                    continue
                if target.kind != "name":
                    raise ValueError("assignment outside scalar replay")
                name, = target.args
                if flow.get(name).id in self.arrays:
                    raise ValueError("array writes outside scalar replay")
                value = read(source)
                if op.text != "=":
                    value = ARITHMETIC[op.text[:-1]](flow.get(name), value)
                self.writes[source_key(node)] = (flow.guard, value)
                flow.assign(name, value)
            elif k == "if":
                condition, yes, no = a
                self.branches += 1
                flow.branch(truth(read(condition)),
                            lambda branch: self.block(yes, branch, on_continue=on_continue, on_call=on_call),
                            lambda branch: self.block(no, branch, on_continue=on_continue, on_call=on_call))
            elif (k == "return" and (on_continue is None or self.scalar_types is not None)
                  and (a[0] is not None or self.scalar_types is not None
                       or on_call is not None or self.signatures is not None)):
                if a[0] is not None:
                    self.returns[source_key(node)] = (flow.guard, read(a[0]))
                flow.assume(False)
            elif k == "eval" and (on_call is not None or self.signatures is not None):
                read(a[0])
            elif k == "continue" and on_continue is not None:
                on_continue(flow)
                flow.assume(False)
            else:
                raise ValueError("statement outside scalar replay")

    def proves_range(self, node, intervals, kind):
        site = source_key(node)
        entry = getattr(self, kind).get(site)
        if entry is None:
            return False
        key = kind, site, intervals
        if key in self.cache:
            return self.cache[key]
        guard, value = entry
        if value.sort != "int":
            return False
        goal = self.program.arena.expr(False)
        for lo, hi in intervals:
            goal = goal | ((value >= lo) & (value <= hi))
        obligation = Obligation("Lan range", guard, goal)
        limits = active_limits()
        if self.exchange is None:
            proved = self.program.verify(obligation, **limits).status == "proved"
        else:
            proved = self.exchange.prove_range(node, kind, intervals, self.program, obligation, limits)
        self.cache[key] = proved
        return proved

    def proves_argument_order(self, node, target, op, term):
        result = self.argument_result(node, target, op, term)
        return result is not None and result.status == "proved"

    def argument_result(self, node, target, op, term):
        site = source_key(node)
        if site not in self.calls:
            return None
        guard, args = self.calls[site]

        def evaluate(expr):
            if expr[0] == "param":
                return args[expr[1]]
            if expr[0] == "const":
                return self.program.arena.expr(expr[1])
            if expr[0] == "binary":
                return ARITHMETIC[expr[1]](evaluate(expr[2]), evaluate(expr[3]))
            raise ValueError("term outside scalar replay")

        try:
            goal = ORDER[op](args[target], evaluate(term))
        except (TypeError, ValueError, KeyError, IndexError):
            return None
        return self.verify_argument(site, target, op, term, guard, goal)

    def verify_argument(self, site, target, op, term, guard, goal):
        return self.program.verify(Obligation("Lan call", guard, goal), **active_limits())


class LoopReplay(ScalarReplay):
    """One iteration, with independent runtime reads and symbolic lengths."""

    def __init__(self, inputs, arrays, scalar_types, signatures):
        super().__init__(inputs, scalar_types, signatures)
        self.lengths = {}
        self.windows = {}
        self.conversions = {}
        self.reported = set()
        for name, (shape, lo, hi) in arrays.items():
            length = lo if lo == hi else self.program.integer(f"#len:{name}", lo, hi)
            self.declare_array(self.program.flow, name, shape, length)

    def index(self, array, node, value, guard):
        if node is not None:
            self.lengths[source_key(node)] = self.arrays[array.id][1]
        return super().index(array, node, value, guard)

    def read(self, node, flow, on_call=None):
        return expression(node, flow, on_call or self.call, on_index=self.index,
                          on_slice=self.slice, on_cast=self.cast)

    def slice(self, node, host, lower, upper, closed, guard):
        shape, length = self.arrays[host.id]
        arena = self.program.arena
        lower = arena.expr(0 if lower is None else lower, "int")
        upper = arena.expr(length if upper is None else upper, "int")
        end = upper + 1 if closed else upper
        self.windows[source_key(node)] = guard, lower, upper, end, length
        return self.array_value(shape, end - lower)

    def cast(self, node, operand, guard):
        _, strategy, annotation = node.args
        base = annotation.args[0] if annotation.kind == "refine" else annotation
        if strategy not in ("safe", "trunc") or base.kind != "type_name" or operand.sort == "storage":
            raise ValueError("conversion outside scalar replay")
        boolean, _, _ = self.scalar_types[base.args[0]]
        if boolean:
            return truth(operand)
        value = (self.program.arena.op("ite", operand, 1, 0) if operand.sort == "bool" else operand)
        self.conversions[source_key(node)] = guard, value
        # A checked conversion to integer preserves the mathematical value.
        # Its own proof must succeed before subsequent obligations are used.
        return value

    def indexed_write(self, target, op, source, flow, read):
        # Lengths are stable and no content fact survives a read, so the
        # replay needs no array update. The compiler still checks the store.
        old = read(target)
        value = read(source)
        return value if op.text == "=" else ARITHMETIC[op.text[:-1]](old, value)

    def verify(self, key, guard, goal):
        if key not in self.cache:
            guard = self.program.arena.slice_guard(guard, goal)
            self.cache[key] = self.program.verify(Obligation("Lan loop safety", guard, goal), **active_limits())
        return self.cache[key]

    def report(self, node, label, result):
        key = source_key(node), label
        if key in self.reported:
            return None
        self.reported.add(key)
        return (f"  ssa {label} proved at {node.token.line}:{node.token.col}"
                f" (loop slice: {result.slice_nodes} SSA nodes)")

    def range_result(self, node, intervals, kind):
        site = source_key(node)
        entry = getattr(self, kind).get(site)
        if entry is None or entry[1].sort != "int":
            return None
        guard, value = entry
        goal = self.program.arena.expr(False)
        for lo, hi in intervals:
            goal |= (value >= lo) & (value <= hi)
        return self.verify((kind, site, intervals), guard, goal)

    def proves_range(self, node, intervals, kind):
        result = self.range_result(node, intervals, kind)
        return result is not None and result.status == "proved"

    def access_result(self, node):
        site = source_key(node)
        entry = self.operands.get(site)
        if entry is None or entry[1].sort != "int":
            return None
        guard, value = entry
        return self.verify(("index", site), guard, (value >= 0) & (value < self.lengths[site]))

    def slice_result(self, node):
        site = source_key(node)
        if site not in self.windows:
            return None
        guard, lower, upper, end, length = self.windows[site]
        return self.verify(("slice", site), guard, (lower >= 0) & (lower <= upper) & (end <= length))

    def verify_argument(self, site, target, op, term, guard, goal):
        return self.verify(("argument", site, goal.id), guard, goal)


def replay_loop(inputs, arrays, condition, body, scalar_types, signatures):
    try:
        replay = LoopReplay(inputs, arrays, scalar_types, signatures)
        flow = replay.program.flow
        if condition is not None:
            flow.assume(replay.program.arena.expr(replay.read(condition, flow), "bool"))
    except (KeyError, TypeError, ValueError, RecursionError):
        return None
    try:
        replay.block(body, flow, on_continue=lambda _: None)
    except (KeyError, TypeError, ValueError, RecursionError):
        # No proof after an opaque effect: block stops at the first unsupported
        # operation. Already recorded source sites keep their preceding guards.
        pass
    return replay


def replay_function(inputs, body, scalar_types, signatures, *, exchange=None):
    try:
        replay = ScalarReplay(inputs, scalar_types, signatures, exchange)
        replay.block(body, replay.program.flow)
        return replay
    except (KeyError, TypeError, ValueError, RecursionError):
        return None


def prove_recursive_group(functions, helpers):
    """Check same-group calls using declared domains and entry SSA values."""
    measures = {name: tuple(i for i, (boolean, _, _) in enumerate(inputs.values()) if not boolean)
                for name, (inputs, _, _) in functions.items()}
    if len(functions) > 32 or any(len(slots) > 4 for slots in measures.values()):
        return False
    transitions = []
    sites = 0
    try:
        for name, (inputs, _, body) in functions.items():
            replay = ScalarReplay(inputs)
            flow = replay.program.flow
            parameters = tuple(inputs)
            before = {i: flow.get(parameters[i]) for i in measures[name]}

            def on_call(node, target, args, guard):
                nonlocal sites
                if target in functions:
                    target_inputs, result, _ = functions[target]
                    parameters = target_inputs.values()
                else:
                    parameters, result = helpers[target]
                parameters = tuple(parameters)
                if len(args) != len(parameters):
                    raise ValueError("recursive call argument count differs from its contract")
                for arg, (boolean, _, _) in zip(args, parameters):
                    replay.program.arena.expr(arg, "bool" if boolean else "int")
                sites += 1
                if sites > 64:
                    raise ValueError("recursive call-site budget exceeded")
                label = f"{node.token.line}:{node.token.col}"

                def proves(goal):
                    return replay.program.verify(Obligation(label, guard, goal), **active_limits()).status == "proved"

                # All measured parameters have finite storage domains. A
                # common offset from the smallest lower bound embeds them in N.
                # Ordinary checking independently validates the call contracts.
                edges = []
                if target in functions:
                    for i, old in before.items():
                        for j in measures[target]:
                            if proves(args[j] < old):
                                edges.append((i, j, Change.STRICT))
                            elif proves(args[j] <= old):
                                edges.append((i, j, Change.WEAK))
                    if edges or not proves(replay.program.arena.expr(False)):
                        transitions.append(Transition(name, target, tuple(edges), label))
                fresh = f"#call:{sites}"
                if result is None:
                    return replay.program.arena.expr(0)
                boolean, lo, hi = result
                return replay.program.boolean(fresh) if boolean else replay.program.integer(fresh, lo, hi)

            replay.block(body, flow, on_call=on_call)
    except (KeyError, TypeError, ValueError, RecursionError):
        return False
    return check_termination(measures, transitions, max_graphs=1024,
                             max_compositions=10_000).status == "proved"


def prove_loop(inputs, condition, body):
    measures = [name for name, (boolean, _, _) in inputs.items() if not boolean]
    if not 1 <= len(measures) <= 4:
        return False
    try:
        replay = ScalarReplay(inputs)
        flow = replay.program.flow
        before = {name: flow.get(name) for name in measures}
        edges = []

        def capture(edge):
            if len(edges) >= 4:
                raise ValueError("loop edge budget")
            edges.append((edge.guard, {name: edge.get(name) for name in measures}))

        flow.assume(truth(expression(condition, flow)))
        flow.branch(replay.program.arena.expr(True),
                    lambda branch: replay.block(body, branch, on_continue=capture))
        capture(flow)
        transitions = [infer_integer_transition(
            replay.program, "head", "head", before, after, guard=guard, **active_limits())
            for guard, after in edges]
        return check_termination({"head": measures}, transitions).status == "proved"
    except (KeyError, TypeError, ValueError, RecursionError):
        return False
