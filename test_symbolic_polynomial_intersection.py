import unittest

from lanc import compile_source
from poly_intersection import PolynomialIntersection, natural_composition_inclusion
from table_interface import decode_domain, encode_domain
from value_domain import Clause, Domain, Polynomial


def bounded(left, right, lo=0, hi=(1 << 64) - 1, exclude=()):
    return Domain.from_clauses([Clause(lo, hi, (left, right), exclude)])


class SymbolicPolynomialIntersectionTests(unittest.TestCase):
    def test_positive_composition_and_interface_round_trip(self):
        square = Polynomial((0, 0, 1))
        fourth_power = Polynomial((0, 0, 0, 0, 1))
        result = bounded(square, fourth_power)
        self.assertEqual(len(result.predicates), 1)
        self.assertEqual(result.predicates[0].require, (fourth_power,))
        self.assertTrue(result & Domain.point(65535 ** 4))
        self.assertFalse(result & Domain.point(9))
        self.assertTrue(result.within(Domain.from_clauses([Clause(0, (1 << 64) - 1, (square,))])))
        self.assertEqual(decode_domain(encode_domain(result)), result)

    def test_integer_composition_may_not_preserve_natural_domain(self):
        outer = Polynomial((0, 1, 2))
        target = Polynomial((0, 0, -1, 0, 2))
        self.assertEqual(PolynomialIntersection(outer.coefficients, target.coefficients).solve_set().kind,
                         "polynomial")
        self.assertFalse(natural_composition_inclusion(outer, target))
        result = bounded(outer, target)
        self.assertTrue(result & Domain.point(0))
        self.assertFalse(result & Domain.point(1))

    def test_composition_can_prove_a_natural_tail(self):
        outer = Polynomial((0, 1, 2))
        target = Polynomial((6, 0, -7, 0, 2))
        self.assertFalse(natural_composition_inclusion(outer, target))
        result = bounded(outer, target)
        self.assertEqual(result.predicates[0].require, (Polynomial((10, 36, 41, 16, 2)),))
        self.assertTrue(result & Domain.point(10))
        self.assertFalse(result & Domain.point(6))

    def test_compiler_uses_compact_result_as_an_original_operand_subtype(self):
        _, trace = compile_source("""
            define f(x: uint32 & poly(n*n) & poly(n*n*n*n)) -> uint32 & poly(n*n) {
                return x;
            }
        """)
        row, = [line for line in trace.splitlines() if line.strip().startswith("return ")]
        self.assertEqual(row.count("poly("), 1)
        self.assertIn("poly(n*n*n*n)", row)

    def test_crt_respects_one_sided_affine_generators(self):
        a = Polynomial((0, 2))
        b = Polynomial((7, -3))
        result = bounded(a, b, -100, 100)
        self.assertEqual(result, Domain.of((4, 4)))
        a = Polynomial((-1, -2))
        b = Polynomial((0, -3))
        result = bounded(a, b, -100, 100)
        self.assertEqual(len(result.predicates[0].require), 1)
        self.assertTrue(result & Domain.point(-99))
        self.assertFalse(result & Domain.point(-96))

    def test_residue_images_keep_exclusions_and_integer_valued_denominator(self):
        triangle = Polynomial((0, 1, 1), 2)
        residue_one = Polynomial((1, 3))
        odd = Polynomial((1, 2))
        result = bounded(triangle, residue_one, exclude=(odd,))
        self.assertTrue(result.predicates)
        self.assertTrue(all(len(clause.require) == 1 for clause in result.predicates))
        for value in range(300):
            expected = triangle.contains(value) and residue_one.contains(value) and not odd.contains(value)
            self.assertEqual(bool(result & Domain.point(value)), expected, value)

    def test_degenerate_quadratic_factorization_keeps_natural_branches(self):
        even_square = Polynomial((0, 0, 4))
        shifted_square = Polynomial((1, 2, 1))
        result = bounded(even_square, shifted_square)
        self.assertEqual(result.predicates[0].require, (Polynomial((4, 8, 4)),))
        self.assertTrue(result & Domain.point(4))
        self.assertFalse(result & Domain.point(0))

        # Triangular numbers whose indices are both even and 1 modulo 3.
        even_triangle = Polynomial((0, 1, 2))
        shifted_triangle = Polynomial((2, 9, 9), 2)
        result = bounded(even_triangle, shifted_triangle)
        self.assertEqual(result.predicates[0].require, (Polynomial((10, 27, 18)),))
        self.assertTrue(result & Domain.point(55))
        self.assertFalse(result & Domain.point(3))


if __name__ == "__main__":
    unittest.main()
