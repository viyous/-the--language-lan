"""Meta-graph of record links, with per-root summaries kept up to date.

A `next` link is a directed edge.  Reachability R* is a preorder and its SCC
quotient is a partial order: a node whose component is acyclic and whose
out-degree never exceeds one is a *chain*, and the same shape closed on itself
is a *cycle*.  Edge weights record how many elements sit between two nodes, so
the graph also answers "how far apart are these two nodes at least".

One traversal of a root yields everything a client needs -- the chain length,
the distance from every node to the end, the lasso prefix/period -- and the
result is cached.  Later queries are O(1); a structural change only invalidates
the summaries whose closure contains the changed node, so analysis never
re-walks a chain just to ask its length again.
"""

from dataclasses import dataclass
from math import inf

EMPTY = "empty"
CHAIN = "chain"
TREE = "tree"
DAG = "dag"
LASSO = "lasso"
GRAPH = "graph"
REPEAT = "repeat"


class ShapeOverflow(Exception):
    """The reachable graph exceeded the analysis budget."""


@dataclass(frozen=True)
class Shape:
    kind: str
    nodes: int
    length: int | None  # node count of the longest walk; None when a cycle closes it
    prefix: int = 0     # lasso/repeat: nodes before the repeated section
    period: int = 0     # lasso/repeat: nodes per turn
    cyclic: bool = False
    repeat: int = 0     # repeat: fresh nodes allocated per iteration

    @property
    def bounded(self):
        return self.length is not None

    def length_for(self, rounds):
        """Length of a repeated chain after `rounds` iterations (0 = prefix)."""
        if self.kind == REPEAT:
            return self.prefix + rounds * self.period
        return self.length

    def describe(self):
        if self.kind == CHAIN:
            return f"chain of {self.length} nodes"
        if self.kind == LASSO:
            return f"lasso {self.prefix}+{self.period}"
        if self.kind == REPEAT:
            return f"repeat {self.prefix}+{self.period}*rounds"
        if self.kind == TREE:
            return f"tree of {self.nodes} nodes, depth {self.length}"
        if self.kind == DAG:
            return f"dag of {self.nodes} nodes, longest path {self.length}"
        if self.kind == GRAPH:
            return f"graph of {self.nodes} nodes with a cycle"
        return "empty"


@dataclass(frozen=True)
class Summary:
    """Result of one traversal: everything later queries read in O(1)."""

    root: object
    order: tuple
    reach: frozenset
    distances: dict   # node -> nodes remaining on the longest walk (None = infinite)
    shape: Shape
    cycle: tuple = ()  # one closing cycle, for diagnostics


def _key(node):
    """Deterministic order across integer ids and (uid, path) locations."""
    return (0, node, "") if isinstance(node, int) else (1, 0, repr(node))


class LinkGraph:
    """Directed meta-graph with cached per-root summaries."""

    def __init__(self, limit=4096):
        self.limit = limit
        self._succ = {}
        self._weight = {}
        self._pred = {}
        self._fresh = set()
        self._summaries = {}
        self._version = 0

    # -- structure ---------------------------------------------------------

    def add(self, node):
        self._succ.setdefault(node, set())

    def link(self, node, targets, weight=1, fresh=False):
        """Add edges node -> each target.  `weight` is the least element gap.

        `fresh` marks a turn of a loop that allocates a new node each round:
        the abstract edge points back at the allocation site, but every turn
        yields a *different* object, so the walk does not really cycle.
        """
        self.add(node)
        changed = False
        for target in targets:
            if target is None:
                continue
            self.add(target)
            if target not in self._succ[node]:
                self._succ[node].add(target)
                self._pred.setdefault(target, set()).add(node)
                changed = True
            if weight > self._weight.get((node, target), 0):
                self._weight[(node, target)] = weight
                changed = True
            if fresh and (node, target) not in self._fresh:
                self._fresh.add((node, target))
                changed = True
        if changed:
            self._invalidate(node)
        return changed

    def unlink(self, node, target=None):
        self.add(node)
        removed = [target] if target is not None else list(self._succ[node])
        for item in removed:
            self._succ[node].discard(item)
            self._pred.get(item, set()).discard(node)
            self._weight.pop((node, item), None)
        if removed:
            self._invalidate(node)

    def _invalidate(self, node):
        self._version += 1
        stale = [root for root, summary in self._summaries.items() if node in summary.reach]
        for root in stale:
            del self._summaries[root]

    def successors(self, node):
        return frozenset(self._succ.get(node, ()))

    def weight(self, node, target):
        return self._weight.get((node, target), 1)

    def roots(self):
        return tuple(sorted(self._summaries, key=_key))

    # -- one traversal -----------------------------------------------------

    def summary(self, root):
        cached = self._summaries.get(root)
        if cached is not None:
            return cached
        order = self._closure(root)
        cycle = self._cycle(root, order)
        reach = frozenset(order)
        if cycle is None:
            distance = {}
            for node in reversed(order):
                distance[node] = 1 + max((distance[child] for child in self._succ.get(node, ())), default=0)
            shape = self._acyclic_shape(order)
        else:
            distance = {node: None for node in order}
            if self._is_repeat(cycle):
                prefix, period = self._lasso(root)
                shape = Shape(REPEAT, len(order), None, prefix, period, False, period)
            else:
                shape = self._cyclic_shape(root, order, cycle)
        summary = Summary(root, order, reach, distance, shape, cycle or ())
        self._summaries[root] = summary
        return summary

    def _closure(self, root):
        """Deterministic breadth-first reachable set."""
        if root is None:
            return ()
        seen, order, pending = {root}, [root], [root]
        while pending:
            node = pending.pop(0)
            for child in sorted(self._succ.get(node, ()), key=_key):
                if child in seen:
                    continue
                if len(seen) >= self.limit:
                    raise ShapeOverflow(f"link graph exceeds {self.limit} nodes")
                seen.add(child)
                order.append(child)
                pending.append(child)
        return tuple(order)

    def _cycle(self, root, order):
        """One closing cycle inside the reachable set, or None."""
        grey, black = set(), set()
        for start in order:
            if start in black:
                continue
            stack = [start]
            grey.add(start)
            work = [(start, iter(sorted(self._succ.get(start, ()), key=_key)))]
            while work:
                node, children = work[-1]
                descended = False
                for child in children:
                    if child in grey:
                        return tuple(stack[stack.index(child):])
                    if child in black:
                        continue
                    grey.add(child)
                    stack.append(child)
                    work.append((child, iter(sorted(self._succ.get(child, ()), key=_key))))
                    descended = True
                    break
                if descended:
                    continue
                work.pop()
                stack.pop()
                grey.discard(node)
                black.add(node)
        return None

    def _acyclic_shape(self, order):
        outdeg = max((len(self._succ.get(node, ())) for node in order), default=0)
        length = self._longest(order)
        if outdeg <= 1:
            return Shape(CHAIN, len(order), length)
        indeg = {node: 0 for node in order}
        for node in order:
            for child in self._succ.get(node, ()):
                indeg[child] = indeg.get(child, 0) + 1
        kind = TREE if max(indeg.values(), default=0) <= 1 else DAG
        return Shape(kind, len(order), length)

    def _longest(self, order):
        depth = {}
        for node in reversed(order):
            depth[node] = 1 + max((depth[child] for child in self._succ.get(node, ())), default=0)
        return depth.get(order[0], 0) if order else 0

    def _cyclic_shape(self, root, order, cycle):
        outdeg = max((len(self._succ.get(node, ())) for node in order), default=0)
        prefix, period = self._lasso(root)
        if outdeg <= 1 and period:
            return Shape(LASSO, len(order), None, prefix, period, True)
        return Shape(GRAPH, len(order), None, 0, 0, True)

    def _is_repeat(self, cycle):
        """True when every closing edge is a fresh allocation turn."""
        return bool(cycle) and all(
            (cycle[index], cycle[(index + 1) % len(cycle)]) in self._fresh
            for index in range(len(cycle)))

    def _lasso(self, root):
        seen, order, node = {}, [], root
        while node is not None and node not in seen:
            seen[node] = len(order)
            order.append(node)
            targets = self._succ.get(node, ())
            if len(targets) != 1:
                return 0, 0
            node = next(iter(targets))
        if node is None:
            return 0, 0
        return seen[node], len(order) - seen[node]

    # -- cached queries ----------------------------------------------------

    def length(self, root):
        """Node count of the longest walk; None when a cycle closes it."""
        return self.summary(root).shape.length

    def distance(self, node, root=None):
        """Nodes remaining from `node` to the end of its own chain (O(1))."""
        summary = self.summary(node if root is None else root)
        return summary.distances.get(node)

    def shape(self, root):
        return self.summary(root).shape

    def closure(self, root):
        return self.summary(root).order

    def reaches(self, start, goal):
        return goal is not None and goal in self.summary(start).reach

    def materialized(self, root, rounds):
        """Expand a repeated section into `rounds` concrete vertices.

        A repeated section is an id set: every turn allocates a node that
        differs from its siblings only by identity, so the abstract edge points
        back at the allocation site.  Once the caller knows the range -- the
        number of rounds -- the set degrades into concrete vertices, and ordinary
        chain queries (exact length, per-node distance, a free walk) apply.
        """
        shape = self.summary(root).shape
        if shape.kind != REPEAT:
            return self
        rounds = max(0, rounds)
        prefix, period = shape.prefix, shape.period
        path, node = [], root
        for _ in range(prefix):
            path.append(node)
            targets = self._succ.get(node, ())
            node = next(iter(targets)) if len(targets) == 1 else None
        entry = node
        cycle, node = [], entry
        for _ in range(period):
            cycle.append(node)
            targets = self._succ.get(node, ())
            node = next(iter(targets)) if len(targets) == 1 else None
        clone = LinkGraph(self.limit)
        for index, item in enumerate(path):
            clone.add(item)
            if index + 1 < len(path):
                clone.link(item, (path[index + 1],))
        if not cycle or any(item is None for item in cycle):
            return clone
        previous = path[-1] if path else None
        for round_index in range(rounds):
            keys = [(item, round_index) for item in cycle]
            for index, key in enumerate(keys):
                clone.add(key)
                if index + 1 < len(keys):
                    clone.link(key, (keys[index + 1],))
            if previous is not None:
                clone.link(previous, (keys[0],))
            previous = keys[-1]
        return clone

    def gap(self, start, goal):
        """Least element distance from `start` to `goal`, or None.

        Path weights are lower bounds, so the maximum path sum is the tightest
        provable separation.  Cycles make the answer unbounded, which is
        reported as `inf`.
        """
        if start == goal:
            return 0
        summary = self.summary(start)
        if goal not in summary.reach:
            return None
        if summary.shape.cyclic:
            return inf
        best = {}
        for node in reversed(summary.order):
            if node == goal:
                best[node] = 0
                continue
            candidates = [(self.weight(node, child), best.get(child)) for child in self._succ.get(node, ())]
            scores = [weight + rest for weight, rest in candidates if rest is not None]
            best[node] = max(scores) if scores else None
        return best.get(start)

    def attach_creates_cycle(self, attach_point, donor):
        """True when linking `donor` under `attach_point` closes a cycle.

        The new edge closes a cycle exactly when the attach point is already
        reachable from the donor, because that path plus the new edge returns
        to the attach point.
        """
        return self.reaches(donor, attach_point)
