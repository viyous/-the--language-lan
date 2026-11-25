from importlib.util import find_spec
import unittest

from ssa_verify import Program


class SSAVerificationTests(unittest.TestCase):
    def test_thousand_independent_branches_keep_small_goal_slice(self):
        p = Program()
        for i in range(1000):
            flag = p.boolean(f"flag{i}")
            name = f"x{i}"
            p.flow.declare(name, 0)
            p.flow.branch(flag, lambda scope: scope.assign(name, 1))
        result = p.verify(p.flow.check(p.flow.get("x999") <= 1), use_smt=False)
        self.assertEqual((result.status, result.method), ("proved", "interval"))
        self.assertLessEqual(result.slice_nodes, 8)
        self.assertLess(len(p.arena.nodes), 5000)

    def test_nested_writes_and_local_shadow(self):
        p = Program()
        flag = p.boolean("flag")
        p.flow.declare("x", 3)

        def then(scope):
            scope.declare("x", 1)
            scope.branch(flag, lambda inside: inside.assign("x", 8))

        p.flow.branch(flag, then)
        self.assertEqual(p.verify(p.flow.check(p.flow.get("x").eq(3))).status, "proved")

    def test_prototype_refuses_nonlinearity_and_cross_program_terms(self):
        p, q = Program(), Program()
        x, y = p.integer("x"), p.integer("y")
        with self.assertRaisesRegex(ValueError, "integer constant"):
            x*y
        with self.assertRaises(TypeError):
            x+q.integer("x")

    def test_no_unknown_is_reported_as_proved(self):
        p = Program()
        x = p.integer("x")
        obligation = p.flow.check(x <= 0)
        self.assertEqual(p.verify(obligation, use_smt=False).status, "unknown")
        self.assertEqual(p.verify(obligation, max_nodes=1).status, "unknown")

    @unittest.skipUnless(find_spec("z3"), "optional Z3 backend unavailable")
    def test_shared_ite_proves_correlation_and_refutes_false_goal(self):
        p = Program()
        flag = p.boolean("flag")
        p.flow.declare("x", 0)
        p.flow.declare("y", 0)
        p.flow.branch(flag, lambda s: s.assign("y", 1), lambda s: s.assign("x", 1))
        total = p.flow.get("x") + p.flow.get("y")
        proof = p.verify(p.flow.check(total.eq(1)))
        self.assertEqual((proof.status, proof.method), ("proved", "smt"))
        bad = p.verify(p.flow.check(p.flow.get("x").eq(0)))
        self.assertEqual(bad.status, "refuted")
        self.assertEqual(bad.model, {"flag": False})

    @unittest.skipUnless(find_spec("z3"), "optional Z3 backend unavailable")
    def test_assumptions_keep_exact_activation_and_do_not_apply_retroactively(self):
        p = Program()
        flag = p.boolean("flag")
        x = p.integer("x", 0, 8)
        before = p.flow.check(x <= 2)

        def then(scope):
            scope.assume(x <= 2)
            scope.check(x <= 2)

        p.flow.branch(flag, then, lambda scope: scope.assume(False))
        after = p.flow.check(x <= 2)
        self.assertEqual(p.verify(before).status, "refuted")
        self.assertEqual(p.verify(p.obligations[1]).status, "proved")
        self.assertEqual(p.verify(after).status, "proved")


if __name__ == "__main__":
    unittest.main()
