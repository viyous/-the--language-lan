from importlib.util import find_spec
from random import Random
import unittest

from poly_intersection import (
    INT64_MIN, UINT64_MAX, PolynomialIntersection, SearchLimitExceeded, solve_many,
)
from value_domain import evaluate


class PolynomialIntersectionTests(unittest.TestCase):
    def assert_witness(self, problem, result):
        self.assertEqual(result.status, "nonempty")
        hit = result.witness
        self.assertEqual(evaluate(problem.f, hit.x), hit.value)
        self.assertEqual(evaluate(problem.g, hit.y), hit.value)
        self.assertTrue(problem.lower <= hit.value <= problem.upper)

    def test_membership_over_all_integers_and_extreme_values(self):
        p = PolynomialIntersection((0, 0, 0, 1), (0, 1))
        self.assertEqual(p.witness(INT64_MIN).x, -(1 << 21))
        self.assertFalse(p.contains(UINT64_MAX + 1))
        large = (1 << 32)-1
        p = PolynomialIntersection((0, 0, 1), (0, 1))
        self.assertTrue(p.contains(large*large))
        self.assertFalse(p.contains(large*large+1))

    def test_large_shift_does_not_bound_the_generator_by_degree(self):
        shift = 10**35
        p = PolynomialIntersection((-shift**3, 3*shift**2, -3*shift, 1), (0, 0, 1), 64, 64)
        self.assertEqual(p.find().witness.x, shift+4)

    def test_crt_and_fixed_divisor(self):
        p = PolynomialIntersection((1, -4), (3, 6), 8, 10)
        self.assertEqual(p.find().witness.value, 9)
        p = PolynomialIntersection((0, 0, 101), (1, 0, 101))
        self.assertEqual(p.find(max_steps=0).status, "empty")

    def test_quadratic_paths(self):
        pell = PolynomialIntersection((0, 0, 1), (1, 0, 2), 10**16)
        result = pell.find(max_steps=1000)
        self.assertEqual(result.method, "quadratic-pell")
        self.assert_witness(pell, result)
        factored = PolynomialIntersection((0, 0, 1), (1, 0, 1), 2)
        self.assertTrue(factored.is_empty(max_steps=1000))
        ellipse = PolynomialIntersection((0, 0, 1), (25, 0, -1), 2, 24)
        self.assert_witness(ellipse, ellipse.find())
        degenerate = PolynomialIntersection((0, 0, 1), (0, 0, 4), 5, 16)
        self.assertEqual(degenerate.find().witness.value, 16)

    def test_unknown_is_not_empty_and_batch_order_is_preserved(self):
        undecided = PolynomialIntersection((0, 0, 0, 1), (1, 0, 1))
        impossible = PolynomialIntersection((0, 2), (1, 2))
        self.assertEqual([r.status for r in solve_many([undecided, impossible], max_steps=0)],
                         ["unknown", "empty"])
        with self.assertRaises(SearchLimitExceeded):
            undecided.is_empty(max_steps=0)
        self.assert_witness(undecided, undecided.find())

    @unittest.skipUnless(find_spec("sympy"), "optional SymPy backend unavailable")
    def test_optional_pell_backend_checks_large_fundamental_solution(self):
        p = PolynomialIntersection((0, 0, 1), (1, 0, 61), 2)
        result = p.find(max_steps=100, backend="sympy")
        self.assertEqual(result.method, "quadratic-pell-sympy")
        self.assert_witness(p, result)
        self.assertEqual(abs(result.witness.x), 1766319049)

    def test_differential_signed_and_nonmonotone_cases(self):
        rng = Random(98211)
        for _ in range(600):
            f = tuple(rng.randint(-8, 8) for _ in range(rng.randrange(2, 6)))
            g = tuple(rng.randint(-8, 8) for _ in range(rng.randrange(2, 6)))
            lo, hi = sorted((rng.randrange(-80, 81), rng.randrange(-80, 81)))
            p = PolynomialIntersection(f, g, lo, hi)
            # A root of f(t)-value or g(t)-value is bounded by 1+max coefficient
            # ratio here, so [-100,100] covers all inputs for this test corpus.
            left = {evaluate(f, n) for n in range(-100, 101)}
            right = {evaluate(g, n) for n in range(-100, 101)}
            expected = any(lo <= v <= hi for v in left & right)
            result = p.find()
            self.assertEqual(result.status, "nonempty" if expected else "empty", (p, result))
            if expected:
                self.assert_witness(p, result)

    def test_complete_set_uses_compact_representations(self):
        result = PolynomialIntersection((0, 1), (0, -1)).solve_set(max_values=0)
        self.assertEqual(result.kind, "progression")
        self.assertEqual(result.progression.count, 3 * (1 << 63))
        result = PolynomialIntersection((0, 0, 1), (0, 0, 0, 0, 1)).solve_set(max_values=0)
        self.assertEqual(result.kind, "polynomial")
        self.assertEqual(result.images[0].coefficients, (0, 0, 0, 0, 1))
        self.assertTrue(result.contains(81))
        self.assertFalse(result.contains(9))
        # One quadratic can contain many values of an independent line.
        result = PolynomialIntersection((1, 3), (0, 0, 1), 0, 100).solve_set(max_values=0)
        self.assertEqual(result.kind, "polynomial")
        self.assertEqual({i for i in range(101) if result.contains(i)}, {1, 4, 16, 25, 49, 64, 100})

    def test_complete_pell_values_and_budget(self):
        p = PolynomialIntersection((0, 0, 1), (1, 0, 2), 0, 100_000)
        result = p.solve_set()
        self.assertEqual(result.values, (1, 9, 289, 9801))
        truncated = p.solve_set(max_values=2)
        self.assertEqual((truncated.kind, truncated.values), ("unknown", ()))
        with self.assertRaises(SearchLimitExceeded):
            truncated.contains(1)

    def test_complete_sets_match_brute_force(self):
        rng = Random(147)
        for _ in range(400):
            f = tuple(rng.randint(-6, 6) for _ in range(rng.randrange(1, 6)))
            g = tuple(rng.randint(-6, 6) for _ in range(rng.randrange(1, 6)))
            lo, hi = sorted((rng.randrange(-45, 46), rng.randrange(-45, 46)))
            p = PolynomialIntersection(f, g, lo, hi)
            result = p.solve_set()
            lhs = {evaluate(f, x) for x in range(-60, 61)}
            rhs = {evaluate(g, y) for y in range(-60, 61)}
            expected = {v for v in lhs & rhs if lo <= v <= hi}
            actual = {v for v in range(lo-1, hi+2) if result.contains(v)}
            self.assertEqual(actual, expected, (p, result))


if __name__ == "__main__":
    unittest.main()
