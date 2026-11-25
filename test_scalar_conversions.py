import unittest

from lanc import LanError, compile_source


class BooleanConversionTests(unittest.TestCase):
    def test_a_boolean_converts_to_any_integer_type(self):
        for annotation, domain in (("uint8", "[0, 1]"), ("int8", "[0, 1]"), ("uint64", "[0, 1]")):
            with self.subTest(annotation=annotation):
                c, table = compile_source(
                    f"define f(flag: boolean) -> {annotation} {{ return flag as {annotation}; }}")
                self.assertIn(f"return {domain} [entry]", table)
                self.assertIn(f"return (({annotation.replace('uint64', 'uint64_t').replace('uint8', 'uint8_t')}", c)

    def test_boolean_arithmetic_stays_exact(self):
        _, table = compile_source("""
            define f(a: boolean, b: boolean) -> uint8 {
                return (a as uint8) + (b as uint8);
            }
        """)
        self.assertIn("return [0, 2] [entry]", table)

    def test_boolean_conversion_keeps_the_value_zero_or_one(self):
        c, _ = compile_source("define f(flag: boolean) -> uint8 { return flag as uint8; }")
        self.assertIn("((uint8_t)(lv_1_flag))", c)


if __name__ == "__main__":
    unittest.main()
