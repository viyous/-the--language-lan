import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


class CInteropAnd32BitArithTests(unittest.TestCase):
    def test_as_colon_widening_and_invalid_strategy(self):
        c, _ = compile_source("""
            type Small = uint16[0, 1000];
            define widen(x: uint16) -> uint32 {
                define a: uint32 = x as: uint32;
                define b: uint32 = x as uint32;
                define d: uint16 = a as: wrap uint16;
                return (b + d) as: uint32[0, 131070];
            }
        """)
        self.assertIn("((uint32_t)(lv_1_x))", c)
        with self.assertRaisesRegex(LanError, "unsupported conversion strategy 'wrpa'"):
            compile_source("define f(x: uint32) -> uint16 { return x as: wrpa uint16; }")

    def test_multiline_array_literal_trailing_comma_and_diagnostic(self):
        c, _ = compile_source("""
            define table_lookup(i: uint8[0, 3]) -> uint8 {
                define arr: uint8[4] = [
                    1,
                    2,
                    3,
                    4,
                ];
                return arr[i];
            }
        """)
        self.assertIn("lan_fn_table_lookup", c)
        with self.assertRaisesRegex(LanError, r"expected ',' or '\]'"):
            compile_source("""
                define bad() -> uint8 {
                    define arr: uint8[4] = [
                        1, 2
                        3, 4
                    ];
                    return arr[0];
                }
            """)

    def test_extern_callable_nonnull_and_nullable(self):
        source = """
            type RejectFn = extern callable(x: uint32) -> boolean writes {};

            define pick_first(
                lo: uint32[0, 100],
                hi: uint32[0, 100],
                reject: RejectFn | null
            ) -> uint32[0, 100] {
                define cur: uint32[0, 100] = lo;
                if (reject != null) {
                    if (reject(cur)) {
                        return hi;
                    }
                }
                return cur;
            }

            define is_zero(x: uint32) -> boolean {
                return x == 0;
            }

            define test_local(lo: uint32[0, 100], hi: uint32[0, 100]) -> uint32[0, 100] {
                define a: uint32[0, 100] = pick_first(lo, hi, callback(is_zero));
                define b: uint32[0, 100] = pick_first(a, hi, null);
                return b;
            }

            export { pick_first };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
            path = Path(scratch) / "rng.lang"
            compiler = Compiler(source, path.name)
            c, table = ModuleGraph(path, compiler, library=True, export_prefix="game_").compile()
            self.assertRegex(c, r"typedef uint8_t \(\*lan_fnptr_[0-9a-f]{24}\)\(uint32_t\);")
            self.assertRegex(c, r"uint32_t game_pick_first\(uint32_t arg_0, uint32_t arg_1, lan_fnptr_[0-9a-f]{24} arg_2\)")
            self.assertIn("&lan_fn_is_zero", c)
            self.assertIn("NULL", c)
            self.assertIn('"kind":"extern_callable"', table)
            path.with_suffix(".table").write_text(table, encoding="ascii")
            consumer_c, _ = compile_source(
                'import { pick_first } from "./rng.table";\n'
                'define call_it(cb: extern callable(x: uint32) -> boolean) -> uint32[0, 100] {\n'
                '    return pick_first(1, 2, cb);\n'
                '}\n',
                path.with_name("consumer.lang"),
            )
            self.assertRegex(consumer_c, r"typedef uint8_t \(\*lan_fnptr_[0-9a-f]{24}\)\(uint32_t\);")

    def test_extern_callable_rejects_unchecked_null_and_effect_violation(self):
        with self.assertRaisesRegex(LanError, "cannot prove nullable extern callable"):
            compile_source("""
                define bad(cb: extern callable(x: uint32) -> boolean | null) -> boolean {
                    return cb(1);
                }
            """)
        with self.assertRaisesRegex(LanError, "write set exceeds"):
            compile_source("""
                define mutate(x: uint32 · {pass: alias}) -> void { x = 1; }
                define call_ro(cb: extern callable(x: uint32 · {pass: alias}) -> void writes {}, v: uint32 · {pass: alias}) -> void {
                    cb(v);
                }
                define main_test() -> void {
                    define v: uint32 = 0;
                    call_ro(callback(mutate), v);
                }
            """)

    def test_closed_callable_avoids_typeof(self):
        c, _ = compile_source("""
            define f1(x: uint8) -> uint8 { return x; }
            define f2(x: uint8) -> uint8 { return x; }
            define run(cb: callable(f1, f2), x: uint8) -> uint8 { return cb(x); }
        """)
        self.assertNotIn("__typeof__", c)
        self.assertRegex(c, r"typedef uint8_t \(\*lh_callback_fnptr_\d+\)\(uint8_t\);")

    def test_32bit_arithmetic_and_wrap_avoid_64bit_ops(self):
        c, _ = compile_source("""
            define iso_step(seed: uint32) -> uint32 {
                return (seed * 1103515245 + 12345) as: wrap uint32;
            }

            define uniform_mod(val: uint32, bound: uint32[1, 1000]) -> uint32[0, 999] {
                return val % bound;
            }

            define wide_mul(a: uint32, b: uint32) -> uint64 {
                return a * b;
            }
        """)
        iso_start = c.index("lan_fn_iso_step(uint32_t lv_1_seed) {")
        mod_start = c.index("lan_fn_uniform_mod(uint32_t lv_2_val, uint32_t lv_3_bound) {")
        wide_start = c.index("lan_fn_wide_mul(uint32_t lv_4_a, uint32_t lv_5_b) {")
        iso_body = c[iso_start:mod_start]
        self.assertIn("((uint32_t)(lv_1_seed)) *", iso_body)
        self.assertNotIn("(lan_int)(lv_1_seed)", iso_body)
        self.assertNotIn("(lan_uint)(lv_1_seed)", iso_body)

        mod_body = c[mod_start:wide_start]
        self.assertIn("((uint32_t)(lv_2_val)) %", mod_body)
        self.assertIn("uint32_t lan_divisor_", c)
        self.assertNotIn("(lan_int)(lv_2_val)", mod_body)

        wide_body = c[wide_start:]
        self.assertIn("(lan_uint)", wide_body)


if __name__ == "__main__":
    unittest.main()
