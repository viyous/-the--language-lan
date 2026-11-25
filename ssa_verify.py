"""Standalone SSA verification prototype for acyclic integer/Boolean code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    arena: "Arena"
    id: int
    sort: str

    def __bool__(self):
        raise TypeError("symbolic expressions cannot be used as Python booleans")

    def __add__(self, other):
        return self.arena.op("add", self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.arena.op("sub", self, other)

    def __rsub__(self, other):
        return self.arena.op("sub", other, self)

    def __mul__(self, other):
        return self.arena.op("mul", self, other)

    __rmul__ = __mul__

    def __le__(self, other):
        return self.arena.op("le", self, other)

    def __lt__(self, other):
        return self.arena.op("lt", self, other)

    def __ge__(self, other):
        return self.arena.op("le", other, self)

    def __gt__(self, other):
        return self.arena.op("lt", other, self)

    def __and__(self, other):
        return self.arena.op("and", self, other)

    def __or__(self, other):
        return self.arena.op("or", self, other)

    def __invert__(self):
        return self.arena.op("not", self)

    def eq(self, other):
        return self.arena.op("eq", self, other)


@dataclass(frozen=True)
class Node:
    op: str
    sort: str
    args: tuple[int, ...]
    data: object = None


class Arena:
    def __init__(self):
        self.nodes = []
        self._ids = {}
        self.inputs = {}

    def intern(self, op, sort, args=(), data=None):
        node = Node(op, sort, tuple(a.id for a in args), data)
        index = self._ids.get(node)
        if index is None:
            index = len(self.nodes)
            self.nodes.append(node)
            self._ids[node] = index
        return Expr(self, index, sort)

    def expr(self, value, sort=None):
        if type(value) is bool:
            value = self.intern("const", "bool", data=value)
        elif type(value) is int:
            value = self.intern("const", "int", data=value)
        if not isinstance(value, Expr) or value.arena is not self:
            raise TypeError("expected a literal or expression from this program")
        if sort is not None and value.sort != sort:
            raise TypeError(f"expected {sort}, got {value.sort}")
        return value

    def input(self, name, sort, lower=None, upper=None):
        if not isinstance(name, str) or not name or name in self.inputs:
            raise ValueError("input names must be nonempty and unique")
        if any(x is not None and type(x) is not int for x in (lower, upper)):
            raise TypeError("integer bounds must be integers")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("input domain cannot be empty")
        value = self.intern("input", sort, data=(name, lower, upper))
        self.inputs[name] = value
        return value

    def op(self, op, *values):
        args = tuple(self.expr(value) for value in values)
        nodes = tuple(self.nodes[a.id] for a in args)
        if op == "ite":
            self.expr(args[0], "bool")
            self.expr(args[2], args[1].sort)
            if nodes[0].op == "const":
                return args[1] if nodes[0].data else args[2]
            if args[1] == args[2]:
                return args[1]
            return self.intern(op, args[1].sort, args)
        if op in ("and", "or", "not"):
            for arg in args:
                self.expr(arg, "bool")
            if op != "not":
                identity = op == "and"
                for i, node in enumerate(nodes):
                    if node.op == "const":
                        return args[1-i] if node.data == identity else args[i]
                if args[0] == args[1]:
                    return args[0]
        elif op == "eq":
            self.expr(args[1], args[0].sort)
            if args[0] == args[1]:
                return self.expr(True)
        else:
            for arg in args:
                self.expr(arg, "int")
            if op == "mul" and not any(n.op == "const" for n in nodes):
                raise ValueError("only multiplication by an integer constant is supported")
            if op == "sub" and args[0] == args[1]:
                return self.expr(0)
            if op in ("lt", "le"):
                left, lo = self.constant_offset(args[0].id)
                right, hi = self.constant_offset(args[1].id)
                if left == right:
                    return self.expr(lo < hi if op == "lt" else lo <= hi)
        if all(n.op == "const" for n in nodes):
            return self.expr(evaluate(op, tuple(n.data for n in nodes)))
        return self.intern(op, "int" if op in ("add", "sub", "mul") else "bool", args)

    def constant_offset(self, index):
        offset = 0
        while True:
            node = self.nodes[index]
            if node.op not in ("add", "sub"):
                return index, offset
            left, right = node.args
            if self.nodes[right].op == "const":
                amount = self.nodes[right].data
                offset += amount if node.op == "add" else -amount
                index = left
            elif node.op == "add" and self.nodes[left].op == "const":
                offset += self.nodes[left].data
                index = right
            else:
                return index, offset

    def slice(self, *roots):
        seen, pending = set(), [r.id for r in roots]
        while pending:
            i = pending.pop()
            if i not in seen:
                seen.add(i)
                pending.extend(self.nodes[i].args)
        return sorted(seen)

    def slice_guard(self, guard, goal):
        """Keep conjuncts connected to the goal through shared SSA inputs.

        Dropping conjuncts weakens the premise, so only a proof (not a
        counterexample) is conclusive for the original obligation.
        """
        parts, pending, seen = [], [guard.id], set()
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            node = self.nodes[index]
            if node.op == "and":
                pending.extend(node.args)
            else:
                parts.append(index)
        if len(parts) < 2:
            return guard

        def inputs(root):
            return {i for i in self.slice(Expr(self, root, "bool")) if self.nodes[i].op == "input"}

        # Index conjuncts by input, then follow transitive dependencies.
        dependencies = [inputs(part) for part in parts]
        by_input = {}
        for position, variables in enumerate(dependencies):
            for variable in variables:
                by_input.setdefault(variable, []).append(position)
        kept = {i for i, variables in enumerate(dependencies) if not variables}
        pending = list(inputs(goal.id))
        while pending:
            variable = pending.pop()
            for position in by_input.pop(variable, ()):
                if position not in kept:
                    kept.add(position)
                    pending.extend(dependencies[position])
        if len(kept) == len(parts):
            return guard
        result = self.expr(True)
        for position in sorted(kept):
            result &= Expr(self, parts[position], "bool")
        return result


def evaluate(op, args):
    a = args[0]
    b = args[1] if len(args) > 1 else None
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "eq":
        return a == b
    if op == "and":
        return a and b
    if op == "or":
        return a or b
    if op == "not":
        return not a
    if op == "ite":
        return b if a else args[2]
    raise ValueError(op)


@dataclass(frozen=True)
class Obligation:
    label: str
    guard: Expr
    goal: Expr


class Flow:
    def __init__(self, program, guard, parent=None):
        self.program = program
        self.guard = guard
        self.parent = parent
        self._locals = {}
        self._writes = set()
        self._declared = set()
        self._blocked = False

    def _open(self):
        if self._blocked:
            raise RuntimeError("use the active branch scope")

    def get(self, name):
        scope = self
        while scope is not None:
            if name in scope._locals:
                return scope._locals[name]
            scope = scope.parent
        raise KeyError(name)

    def declare(self, name, value):
        self._open()
        if name in self._locals:
            raise ValueError(f"duplicate variable {name!r}")
        self._locals[name] = self.program.arena.expr(value)
        self._declared.add(name)

    def assign(self, name, value):
        self._open()
        old = self.get(name)
        self._locals[name] = self.program.arena.expr(value, old.sort)
        if name not in self._declared:
            self._writes.add(name)

    def assume(self, predicate):
        self._open()
        self.guard = self.guard & predicate

    def check(self, goal, label="check"):
        self._open()
        obligation = Obligation(label, self.guard, self.program.arena.expr(goal, "bool"))
        self.program.obligations.append(obligation)
        return obligation

    def branch(self, condition, yes, no=None):
        self._open()
        arena = self.program.arena
        condition = arena.expr(condition, "bool")
        before_yes, before_no = self.guard & condition, self.guard & ~condition
        left, right = Flow(self.program, before_yes, self), Flow(self.program, before_no, self)
        self._blocked = True
        try:
            yes(left)
            if no is not None:
                no(right)
        finally:
            self._blocked = False
            left._blocked = right._blocked = True
        for name in sorted(left._writes | right._writes):
            old = self.get(name)
            # A declaration in a branch is local even if it shadows this name.
            a = left.get(name) if name in left._writes else old
            b = right.get(name) if name in right._writes else old
            self.assign(name, arena.op("ite", condition, a, b))
        if left.guard != before_yes or right.guard != before_no:
            self.guard = left.guard | right.guard


@dataclass(frozen=True)
class Verification:
    status: str
    method: str
    slice_nodes: int
    model: dict | None = None
    reason: str | None = None


class Program:
    def __init__(self):
        self.arena = Arena()
        self.obligations = []
        self.flow = Flow(self, self.arena.expr(True))

    def integer(self, name, lower=None, upper=None):
        return self.arena.input(name, "int", lower, upper)

    def boolean(self, name):
        return self.arena.input(name, "bool")

    def verify(self, obligation, *, timeout_ms=1000, rlimit=100_000, max_nodes=20_000, use_smt=True):
        if any(type(n) is not int or n <= 0 for n in (timeout_ms, rlimit, max_nodes)):
            raise ValueError("verification limits must be positive integers")
        self.arena.expr(obligation.guard, "bool")
        self.arena.expr(obligation.goal, "bool")
        nodes = self.arena.slice(obligation.guard, obligation.goal)
        if len(nodes) > max_nodes:
            return Verification("unknown", "budget", len(nodes), reason="slice node limit")
        local = _abstract(self.arena, nodes)
        if local[obligation.guard.id] is False or local[obligation.goal.id] is True:
            return Verification("proved", "interval", len(nodes))
        if not use_smt:
            return Verification("unknown", "interval", len(nodes), reason="SMT disabled")
        try:
            import z3
        except ImportError:
            return Verification("unknown", "smt", len(nodes), reason="install z3-solver")
        return _solve(z3, self.arena, nodes, obligation, timeout_ms, rlimit)


def _abstract(arena, ids):
    result = {}
    for i in ids:
        node = arena.nodes[i]
        args = [result[a] for a in node.args]
        if node.op == "const":
            out = node.data if node.sort == "bool" else (node.data, node.data)
        elif node.op == "input":
            out = None if node.sort == "bool" or None in node.data[1:] else node.data[1:]
        elif node.op == "ite":
            cond, a, b = args
            if cond is not None:
                out = a if cond else b
            elif a == b:
                out = a
            elif node.sort == "int" and a is not None and b is not None:
                out = min(a[0], b[0]), max(a[1], b[1])
            else:
                out = None
        elif node.op in ("add", "sub", "mul", "lt", "le", "eq") and all(a is not None for a in args):
            a, b = args
            if isinstance(a, tuple):
                lo, hi = a
                low, high = b
                if node.op == "add":
                    out = lo + low, hi + high
                elif node.op == "sub":
                    out = lo - high, hi - low
                elif node.op == "mul":
                    products = (lo*low, lo*high, hi*low, hi*high)
                    out = min(products), max(products)
                elif node.op == "eq":
                    out = True if lo == hi == low == high else False if hi < low or high < lo else None
                else:
                    delta = int(node.op == "lt")
                    out = True if hi + delta <= low else False if lo + delta > high else None
            else:
                out = a == b
        elif node.op in ("and", "or"):
            decisive = node.op == "or"
            out = decisive if any(a is decisive for a in args) else (not decisive if None not in args else None)
        elif node.op == "not":
            out = None if args[0] is None else not args[0]
        else:
            out = None
        result[i] = out
    return result


def _solve(z3, arena, ids, obligation, timeout_ms, rlimit):
    solver = z3.SolverFor("QF_LIA")
    solver.set(timeout=timeout_ms, rlimit=rlimit)
    values, inputs = {}, {}
    for i in ids:
        node = arena.nodes[i]
        args = [values[a] for a in node.args]
        if node.op == "const":
            value = z3.BoolVal(node.data) if node.sort == "bool" else z3.IntVal(node.data)
        elif node.op == "input":
            name, lower, upper = node.data
            value = z3.Bool(f"n{i}") if node.sort == "bool" else z3.Int(f"n{i}")
            inputs[name] = value
            if lower is not None:
                solver.add(value >= lower)
            if upper is not None:
                solver.add(value <= upper)
        else:
            logic = {"ite": z3.If, "not": z3.Not, "and": z3.And, "or": z3.Or}
            value = logic[node.op](*args) if node.op in logic else evaluate(node.op, args)
        values[i] = value
    solver.add(values[obligation.guard.id], z3.Not(values[obligation.goal.id]))
    status = solver.check()
    if status == z3.unsat:
        return Verification("proved", "smt", len(ids))
    if status == z3.sat:
        model = solver.model()
        sample = {}
        for name, term in inputs.items():
            val = model.eval(term, model_completion=True)
            sample[name] = z3.is_true(val) if z3.is_bool(val) else val.as_long()
        return Verification("refuted", "smt", len(ids), model=sample)
    return Verification("unknown", "smt", len(ids), reason=solver.reason_unknown())
