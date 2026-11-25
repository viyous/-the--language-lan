import unittest

from lanc import LanError, compile_source


class CompileTimeConstantsTests(unittest.TestCase):
    def test_local_compile_time_constants_in_array_lengths_and_repeat_initializers(self):
        c, table = compile_source("""
            define f() -> uint8 {
                define w: uint8 = 4;
                define h: uint8 = 2;
                define a: uint8[w * h] = [7; w * h];
                define b: uint8[a.len] = [a[0] + 1; a.len];
                return b[a.len - 1];
            }
        """)
        self.assertIn("array len=8", table)
        self.assertIn("return {8}", table)
        self.assertIn("uint8_t lv_3_a[8]", c)
        self.assertIn("uint8_t lv_4_b[8]", c)

    def test_local_compile_time_constants_in_refined_types_bounds_sets_and_poly(self):
        _, table = compile_source("""
            define f(x: uint8) -> boolean {
                define hi: uint8 = 10;
                define step: uint8 = 4;
                define y: uint8[0, hi] = 3;
                define z: uint8 & <= hi = y;
                define m: uint8 & poly(step * n) = 8;
                return (x in {y, m}) || (z in uint8[0, hi]);
            }
        """)
        self.assertIn("define entity id#4 y: uint8_t & [0, 10]", table)
        self.assertIn("define entity id#5 z: uint8_t & [0, 10]", table)
        self.assertIn("return {1}", table)

    def test_local_compile_time_constants_in_type_functions_rodata_and_bitfields(self):
        c, table = compile_source("""
            type Row(n: uint8) {
                return array · {len: n, elem: uint8};
            }
            type Config = { width: uint8, offset: uint8 };

            define f(v: uint32, idx: uint8[0, 3]) -> uint16 {
                define base: uint8 = 10;
                define table_ro: uint8[4] · {storage: rodata} = [base, base + 1, base + 2, base + 3];
                define r: Row(table_ro.len) = [table_ro[0]; table_ro.len];
                define cfg: Config = { width: 8, offset: 16 };
                define reg: bit[32] = v as: view bit[32];
                define fs = reg[b: uint8@bit(cfg.width, cfg.offset)];
                return (table_ro[idx] as uint16) + (r[3] as uint16) + (fs.b as uint16);
            }
        """)
        self.assertIn("static const ", c)
        self.assertIn("lv_4_table_ro = {", c)
        self.assertIn("bit(8, 16, le)", table)

    def test_top_level_type_constant_alias_and_string_compile_time_length(self):
        _, table = compile_source("""
            type BASE = 4;
            type STRIDE = BASE * 2;

            define f(s: uint8[..] · {len: >= BASE}, i: uint8[0, s.len - BASE]) -> uint16 {
                define a: uint8[STRIDE] = [STRIDE; STRIDE];
                define tag: uint8["HELLO".len] = ["A"[0] as uint8; "HELLO".len];
                return (s[i] as uint16) + (a[BASE - 1] as uint16) + (tag[0] as uint16);
            }
        """)
        self.assertIn("array len=8", table)
        self.assertIn("array len=5", table)

    def test_branch_narrowed_compile_time_constant_and_non_constant_rejection(self):
        _, table = compile_source("""
            define f(n: uint8) -> uint8 {
                if (n == 4) {
                    define a: uint8[n] = [9; n];
                    return a[3];
                }
                return 0;
            }
        """)
        self.assertIn("array len=4", table)

        with self.assertRaisesRegex(LanError, "'n' is not a compile-time variable"):
            compile_source("""
                define bad(n: uint8) -> uint8 {
                    define a: uint8[n];
                    return a[0];
                }
            """)

        with self.assertRaisesRegex(LanError, "rodata initializer requires a compile-time constant"):
            compile_source("""
                define bad_rodata(n: uint8) -> uint8 {
                    define t: uint8[2] · {storage: rodata} = [n, 1];
                    return t[0];
                }
            """)

    def test_generated_c_is_32bit_portable_without_int128(self):
        c, _ = compile_source("""
            type Square = uint64 & poly(n * n);
            type ShiftedCube = int64 & poly((n - 4) * (n - 4) * (n - 4));

            define check(u: uint64, s: int64, count: uint8[0, 127]) -> boolean {
                define wrapped: uint64 = (u + 1) as: wrap uint64;
                define shifted: int64 = (s >> count) as: wrap int64;
                return (wrapped in Square) || (shifted in ShiftedCube) || (u > s);
            }
        """)
        self.assertNotIn("__int128", c)
        self.assertIn("typedef int64_t lan_int;", c)
        self.assertIn("typedef uint64_t lan_uint;", c)
        self.assertIn("uint32_t positive", c)

    def test_generated_c_avoids_gnu_statement_expressions_and_guards_hosted_headers(self):
        c, _ = compile_source("""
            type Pair = {a: uint8[0, 10], b: uint8[0, 10]};

            define helper(x: uint8[0, 10]) -> uint8[0, 10] {
                return x;
            }

            define check(
                arr: uint8[0, 10][4],
                p: Pair,
                i: uint8[0, 3]
            ) -> uint8[0, 20] {
                define v: uint8[0, 10] = helper(arr[i]);
                if ((v > 0 && p.a <= 10) || ((p.b > 0) ^^ true)) {
                    return (v + p.a) as uint8[0, 20];
                }
                return v;
            }
        """)
        self.assertNotIn("({", c)
        self.assertNotIn("__label__", c)
        self.assertNotIn("__auto_type", c)
        self.assertNotIn("__typeof__", c)
        self.assertIn("#if !defined(__STDC_HOSTED__) || __STDC_HOSTED__", c)
        self.assertIn("#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L", c)



if __name__ == "__main__":
    unittest.main()
