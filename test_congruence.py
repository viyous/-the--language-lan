import unittest

from lanc import LanError, compile_source


class CongruenceTests(unittest.TestCase):
    """Residue classes survive arithmetic and guard refinement.

    A value domain is an interval plus polynomial constraints, so "a multiple
    of six between 0 and 600" is written `[0, 600] & poly(6*n)`.
    """

    def test_a_constant_factor_keeps_the_class(self):
        _, table = compile_source("""
            define f(x: uint8[0, 100]) -> uint32 {
                define y = x * 6;
                return y;
            }
        """)
        self.assertIn("id#2 = [0, 600] & poly(6*n)", table)

    def test_a_product_of_two_values_keeps_the_class(self):
        _, table = compile_source("""
            define f(k: uint8[0, 10], y: uint8[0, 300]) -> uint32 {
                define x = k * 3;
                define p = x * y;
                return p;
            }
        """)
        self.assertIn("id#3 = [0, 30] & poly(3*n)", table)
        # The interval product is [0, 7650]; every value is still a multiple
        # of three, because one factor always is.
        self.assertIn("id#4 = [0, 7650] & poly(3*n)", table)

    def test_the_guard_selects_one_class(self):
        _, table = compile_source("""
            define f(x: uint8[0, 30], y: uint8[0, 30]) -> uint32 {
                if (x % 3 == 0) {
                    define p = x * y;
                    return p;
                }
                return 0;
            }
        """)
        self.assertIn("id#1 = [0, 30] & poly(3*n)", table)
        self.assertIn("id#3 = [0, 900] & poly(3*n)", table)

    def test_a_nonzero_residue_is_kept(self):
        _, table = compile_source("""
            define f(x: uint8[0, 30], y: uint8[0, 30]) -> uint32 {
                if (x % 4 == 1) {
                    define p = x * y;
                    return p;
                }
                return 0;
            }
        """)
        self.assertIn("id#1 = [1, 29] & poly(1 + 4*n)", table)
        # 1 (mod 4) times a range covers every residue, so the product keeps
        # no class at all.
        self.assertIn("id#3 = [0, 870]", table)

    def test_a_square_refines_the_step(self):
        _, table = compile_source("""
            define f(x: uint8[0, 30]) -> uint32 {
                if (x % 3 == 0) {
                    define q = x * x;
                    return q;
                }
                return 0;
            }
        """)
        self.assertIn("id#2 = [0, 900] & poly(9*n)", table)

    def test_two_classes_multiply_to_their_gcd(self):
        _, table = compile_source("""
            define f(x: uint8[0, 30], y: uint8[0, 30]) -> uint32 {
                if (x % 3 == 0) {
                    if (y % 2 == 0) {
                        define p = x * y;
                        return p;
                    }
                }
                return 0;
            }
        """)
        self.assertIn("id#3 = [0, 900] & poly(6*n)", table)

    def test_the_class_makes_a_remainder_provable(self):
        _, table = compile_source("""
            define f(k: uint8[0, 10], y: uint8[0, 30]) -> uint32 {
                define x = k * 3;
                define p = x * y;
                return p % 3;
            }
        """)
        self.assertIn("return {0}", table)

    def test_a_product_is_not_collapsed_to_zero(self):
        # 3 * 1 = 3: intersecting the per-factor classes would be unsound, so
        # the product of 3*n and a range stays 3*n and `p != 0` stays open.
        _, table = compile_source("""
            define f(k: uint8[0, 10], y: uint8[0, 3]) -> uint32 {
                define x = k * 3;
                define p = x * y;
                if (p != 0) { return 1; }
                return 0;
            }
        """)
        self.assertIn("id#4 = [0, 90] & poly(3*n)", table)
        self.assertIn("return {1} [", table)

    def test_a_loop_guard_still_compiles(self):
        # Loop heads rejoin every class their body can produce, so the guard
        # narrows straight-line code only for now.
        _, table = compile_source("""
            define f(limit: uint8[0, 30]) -> uint32 {
                define total: uint32 = 0;
                define i: uint8 = 0;
                while (i < limit && total < 1000) {
                    if (i % 2 == 1) {
                        i += 1;
                        continue;
                    }
                    total += i as uint32;
                    i += 1;
                }
                return total;
            }
        """)
        self.assertIn("loop", table)

    def test_a_factor_of_one_keeps_the_plain_interval(self):
        _, table = compile_source("""
            define f(x: uint8[0, 100], y: uint8[0, 100]) -> uint32 {
                define p = x * y;
                return p;
            }
        """)
        self.assertIn("id#3 = [0, 10000]", table)
        self.assertNotIn("id#3 = [0, 10000] & poly", table)


if __name__ == "__main__":
    unittest.main()
