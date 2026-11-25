from importlib.util import find_spec
import unittest

from size_change import Change, Transition, check_termination, infer_integer_transition
from ssa_verify import Program


class SizeChangeTests(unittest.TestCase):
    def test_mutual_recursion_and_permuted_measures(self):
        edges = (
            Transition("f", "g", (("x", "v", Change.WEAK), ("y", "u", Change.STRICT))),
            Transition("g", "f", (("u", "x", Change.WEAK), ("v", "y", Change.WEAK))),
        )
        self.assertEqual(check_termination({"f": ("x", "y"), "g": ("u", "v")}, edges).status, "proved")

    def test_two_individually_decreasing_calls_can_alternate_without_descent(self):
        edges = (
            Transition("f", "f", (("x", "x", Change.STRICT),)),
            Transition("f", "f", (("y", "y", Change.STRICT),)),
        )
        result = check_termination({"f": ("x", "y")}, edges)
        self.assertEqual(result.status, "not_proved")
        self.assertEqual(result.cycle, (0, 1))
        self.assertEqual(result.graph.edges, ())

    def test_empty_edge_is_a_possible_transition_and_budget_is_unknown(self):
        edges = (Transition("f", "f", ()),)
        self.assertEqual(check_termination({"f": ()}, edges).status, "not_proved")
        self.assertEqual(check_termination({"f": ()}, edges, max_compositions=0).status, "unknown")

    def test_acyclic_calls_do_not_need_size_edges(self):
        n = 3000
        edges = [Transition(i, i+1, ()) for i in range(n-1)]
        result = check_termination({i: () for i in range(n)}, edges, max_graphs=0)
        self.assertEqual((result.status, result.compositions), ("proved", 0))

    @unittest.skipUnless(find_spec("z3"), "optional Z3 backend unavailable")
    def test_ssa_adapter_requires_well_founded_bounds(self):
        p = Program()
        x = p.integer("x")
        transition = infer_integer_transition(p, "loop", "loop", {"i": x}, {"i": x-1}, guard=x > 0)
        self.assertEqual(transition.edges, (("i", "i", Change.STRICT),))
        unbounded = infer_integer_transition(p, "loop", "loop", {"i": x}, {"i": x-1})
        self.assertEqual(unbounded.edges, ())


if __name__ == "__main__":
    unittest.main()
