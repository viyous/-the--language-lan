"""Standalone size-change termination checker for caller-proved measures."""

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum


class Change(IntEnum):
    WEAK = 1    # new <= old
    STRICT = 2  # new < old


@dataclass(frozen=True)
class Transition:
    source: object
    target: object
    edges: tuple  # (old measure ID, new measure ID, Change)
    label: str = ""


@dataclass(frozen=True)
class Result:
    status: str  # proved, not_proved, unknown
    graphs: int
    compositions: int
    cycle: tuple[int, ...] = ()  # indexes in the supplied transitions
    graph: Transition | None = None
    reason: str | None = None


def strongly_connected_components(vertices, arcs):
    """Iterative Kosaraju, so deep acyclic call graphs need no Python recursion."""
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for a, b in arcs:
        outgoing[a].append(b)
        incoming[b].append(a)
    order, seen = [], set()
    exhausted = object()
    for root in vertices:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(outgoing[root]))]
        while stack:
            v, children = stack[-1]
            child = next(children, exhausted)
            if child is exhausted:
                order.append(v)
                stack.pop()
            elif child not in seen:
                seen.add(child)
                stack.append((child, iter(outgoing[child])))
    component = {}
    for root in reversed(order):
        if root in component:
            continue
        pending = [root]
        component[root] = root
        while pending:
            for parent in incoming[pending.pop()]:
                if parent not in component:
                    component[parent] = root
                    pending.append(parent)
    return component


def _compose(left, right):
    outgoing = defaultdict(list)
    for b, c, change in right:
        outgoing[b].append((c, change))
    result = {}
    for a, b, first in left:
        for c, second in outgoing[b]:
            pair = (a, c)
            result[pair] = max(result.get(pair, 0), first, second)
    return tuple((a, c, result[a, c]) for a, c in sorted(result))


def check_termination(locations, transitions, *, max_graphs=10_000, max_compositions=100_000):
    """Check SCT, given a mapping of control locations to ordered measure IDs.

    Every possible transition must be supplied separately, including ones with
    no proven edges. All stated edges must hold in a common well-founded order.
    """
    if any(type(n) is not int or n < 0 for n in (max_graphs, max_compositions)):
        raise ValueError("budgets must be nonnegative integers")
    slots = {location: tuple(ids) for location, ids in locations.items()}
    positions = {loc: {name: i for i, name in enumerate(ids)} for loc, ids in slots.items()}
    if any(len(positions[loc]) != len(ids) for loc, ids in slots.items()):
        raise ValueError("duplicate measure ID at a location")
    transitions = tuple(transitions)
    normalized = []
    for transition in transitions:
        src, dst = transition.source, transition.target
        if src not in slots or dst not in slots:
            raise ValueError("transition refers to an undeclared location")
        pairs = {}
        for old, new, change in transition.edges:
            if old not in positions[src] or new not in positions[dst]:
                raise ValueError("edge refers to an undeclared measure")
            if not isinstance(change, Change):
                raise TypeError("edge relation must be Change.WEAK or Change.STRICT")
            key = positions[src][old], positions[dst][new]
            pairs[key] = max(pairs.get(key, 0), change)
        normalized.append((src, dst, tuple((a, b, pairs[a, b]) for a, b in sorted(pairs))))
    components = strongly_connected_components(slots, ((a, b) for a, b, _ in normalized))
    base = defaultdict(list)
    pending, parents = deque(), {}
    work = 0

    def add(graph, previous, index):
        if graph not in parents:
            if len(parents) >= max_graphs:
                return False
            parents[graph] = previous, index
            pending.append(graph)
        return True

    def unknown(reason):
        return Result("unknown", len(parents), work, reason=reason)

    for index, graph in enumerate(normalized):
        a, b, edges = graph
        if components[a] == components[b] and graph not in parents:
            base[a].append((b, edges, index))
            if not add(graph, None, index):
                return unknown("graph budget")

    while pending:
        graph = pending.popleft()
        a, b, edges = graph
        if a == b and not any(i == j and change == Change.STRICT for i, j, change in edges):
            if work >= max_compositions:
                return unknown("composition budget")
            work += 1
            if _compose(edges, edges) == edges:
                route = []
                cursor = graph
                while cursor is not None:
                    cursor, index = parents[cursor]
                    route.append(index)
                obstruction = Transition(a, a, tuple((slots[a][i], slots[a][j], change)
                                                       for i, j, change in edges))
                return Result("not_proved", len(parents), work, tuple(reversed(route)), obstruction)
        # Extending by original edges alone reaches every nonempty path word.
        for target, right, index in base[b]:
            if work >= max_compositions:
                return unknown("composition budget")
            work += 1
            if not add((a, target, _compose(edges, right)), graph, index):
                return unknown("graph budget")
    return Result("proved", len(parents), work)


def infer_integer_transition(program, source, target, before, after, *, guard=True, label="", **limits):
    """Prove edges over nonnegative integer measures with the SSA prototype.

    The caller supplies the complete transition guard and simultaneous before
    and after expressions. Unknown obligations produce no corresponding edge.
    """
    from ssa_verify import Obligation

    arena = program.arena
    guard = arena.expr(guard, "bool")
    before = {key: arena.expr(value, "int") for key, value in before.items()}
    after = {key: arena.expr(value, "int") for key, value in after.items()}

    def proves(goal):
        return program.verify(Obligation(label, guard, goal), **limits).status == "proved"

    old_measures = {key: value for key, value in before.items() if proves(value >= 0)}
    new_measures = {key: value for key, value in after.items() if proves(value >= 0)}
    edges = []
    for a, old in old_measures.items():
        for b, new in new_measures.items():
            if proves(new < old):
                edges.append((a, b, Change.STRICT))
            elif proves(new <= old):
                edges.append((a, b, Change.WEAK))
    return Transition(source, target, tuple(edges), label)
