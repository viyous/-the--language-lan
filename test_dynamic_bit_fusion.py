import unittest

from lanc import LanError, compile_source


AFFINE = """
    define move_bits(i: uint8[0,120]) -> uint8 {
        define bits: bit[128] = [1;128];
        bits[i+2,i+6) = 2;
        bits[i,i+4) = 5;
        return bits[i,i+8) as: view uint8;
    }
"""

VARIABLE_WIDTH = """
    define varying_bits(i: uint8[0,7], n: uint8[1,6]) -> uint8 {
        define bits: bit[16] = [1;16];
        bits[i,i+n) = 1;
        bits[i+3,i+5) = 3;
        return bits[i,i+8) as: view uint8;
    }
"""

FULL_WIDTH = """
    define wide_bits(i: uint8[0,8]) -> uint64 {
        define bits: bit[128];
        bits[8*i,8*i+64) = 18446744073709551615;
        bits[8*i+63,8*i+64) = 0;
        return bits[8*i,8*i+64) as: view uint64;
    }
"""

DEPENDENT = """
    define read_before_write(i: uint8[0,7]) -> uint8 {
        define bits: bit[16];
        bits[i,i+1) = 1;
        bits[i+1,i+2) = bits[i];
        return bits[i,i+8) as: view uint8;
    }
"""

MIXED_ENDIAN = """
    define mixed_bits(i: uint8[0,7], value: uint16) -> uint16 {
        define bits: bit[16];
        define fields = bits[be: uint16@bit(16,0, endian: be), le: uint16@bit(16,0)];
        fields.be = value;
        bits[i,i+1) = 1;
        return fields.le;
    }
"""

PROJECTED_COPY = """
    define copy_rows(i: uint8[0,1], n: uint8[0,7]) -> uint8[0,1] {
        define rows: bit[16][3];
        rows[i][n,n+4) = 15;
        rows[i+1][n,n+4) = rows[i][n,n+4);
        return rows[i+1][n];
    }
"""


class DynamicBitFusionTests(unittest.TestCase):
    def test_binary_literals_preserve_width_and_bit_order(self):
        c, trace = compile_source("""
            define f() -> uint16[33026,33026] {
                define bits: bit[16] = @1000000100000011;
                bits[0] ^= @1;
                return bits[0,16) as: view uint16;
            }
            define bit_value() -> uint8[2,2] {
                define bits: bit[8] = @10000000;
                bits[0] = 1;
                return bits[7] + bits[0];
            }
        """)
        self.assertIn("{0x03, 0x81}", c)
        self.assertIn("return {33026}", trace)
        self.assertIn("return {2}", trace)

    def test_literal_operators_work_in_fixed_and_dynamic_slices(self):
        _, trace = compile_source("""
            define fixed() -> uint8[41,41] {
                define bits: bit[8] = @00000000;
                bits[0,8) = (~@01010101 >> 2) ^ (@00000001 << 1) | @00000001;
                return bits[0,8) as: view uint8;
            }
            define moving(i: uint8[0,8]) -> uint8 {
                define bits: bit[16];
                bits[i,i+8) = @10100101 & @11111111;
                return bits[i,i+8) as: view uint8;
            }
            define shifted(n: uint8[0,8]) -> uint8[0,128] {
                define bits: bit[8] = @10000000 << n;
                return bits[0,8) as: view uint8;
            }
        """)
        self.assertIn("return {41}", trace)

    def test_wide_literals_lower_to_packed_bytes_and_copy(self):
        literal = "@1" + "0" * 70 + "1"
        c, trace = compile_source(f"""
            define fixed() -> uint8[2,2] {{
                define bits: bit[80];
                bits[2,74) = {literal};
                bits[2,74) ^= @{('0' * 72)};
                return bits[2] + bits[73];
            }}
            define moving(i: uint8[0,8]) -> uint8 {{
                define bits: bit[80];
                bits[i,i+72) = {literal};
                return bits[i+71];
            }}
        """)
        self.assertIn("{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80}", c)
        self.assertIn("return {2}", trace)

    def test_literal_initializes_heap_and_nested_packed_storage(self):
        _, trace = compile_source("""
            type Packet = { flags: bit[8] };
            define f() -> uint8[3,3] {
                define packet: Packet = { flags: @10000000 };
                define rows: bit[8][2] = [@00000001; 2];
                define bits: bit[8] · {storage: heap} = @00000001;
                rows[0] |= @00000010;
                define answer = packet.flags[7] + rows[0][1] + bits[0];
                free(bits);
                return answer;
            }
        """)
        self.assertIn("return {3}", trace)

    def test_invalid_bit_literals_and_unsafe_stores_are_rejected(self):
        for body, error in (
            ("define bits: bit[8] = @10200000;", "binary digits"),
            ("define bits: bit[8] = @1;", "does not match destination width"),
            ("define bits: bit[8]; bits[0] = @01;", "does not match destination width"),
            ("define bits: bit[8]; bits[0] = true;", "integer 0 or 1"),
            ("define bits: bit[8]; bits[0] = 2;", "bit value must be 0 or 1"),
            ("define bits: bit[8]; bits[0,4) = @0000 ^ @00;", "require equal widths"),
            ("define bits: bit[8]; bits[0,n) = @0000;", "cannot prove bit slice width"),
        ):
            with self.subTest(body=body), self.assertRaisesRegex(LanError, error):
                compile_source("define f(n: uint8[1,8]) -> void { " + body + " }")

    def test_moving_span_and_reverse_offset_order(self):
        c, trace = compile_source(AFFINE)
        self.assertIn("fuse 2 dynamic bit writes", trace)
        self.assertIn(" - UINT64_C(2)) & ~UINT64_C(7)", c)
        self.assertIn("lan_fused_data & ~lan_fused_part", c)

    def test_absolute_span_supports_variable_width(self):
        c, trace = compile_source(VARIABLE_WIDTH)
        self.assertIn("fuse 2 dynamic bit writes", trace)
        self.assertIn("lan_fused_base = UINT64_C(0)", c)
        self.assertIn("UINT64_MAX >> (64 - lan_bit_width_", c)

    def test_full_width_moving_span_with_aligned_anchor(self):
        c, trace = compile_source(FULL_WIDTH)
        self.assertIn("fuse 2 dynamic bit writes", trace)
        self.assertIn("lan_fused_byte_mask_7", c)
        self.assertNotIn("lan_fused_byte_mask_8", c)

    def test_static_big_endian_write_can_precede_dynamic_write(self):
        c, trace = compile_source(MIXED_ENDIAN)
        self.assertIn("fuse 2 dynamic bit writes", trace)
        self.assertIn("__builtin_bswap64(lan_fused_ordered) >> 48", c)
        self.assertIn("lan_fused_shift <= 64u - lan_bit_width_", c)

    def test_dynamic_copy_between_provably_distinct_projected_rows_uses_restrict(self):
        c, trace = compile_source(PROJECTED_COPY)
        self.assertIn("copy dynamic bit slices with disjoint byte storage: restrict", trace)
        self.assertIn("uint8_t *restrict dst", c)
        uncertain = PROJECTED_COPY.replace("n: uint8[0,7]", "j: uint8[0,1], n: uint8[0,7]").replace("i+1", "j")
        c, trace = compile_source(uncertain)
        self.assertNotIn("copy dynamic bit slices with disjoint byte storage: restrict", trace)
        self.assertNotIn("uint8_t *restrict dst", c)

    def test_dependent_writes_debug_and_large_unrelated_spans_fall_back(self):
        _, trace = compile_source(DEPENDENT)
        self.assertNotIn("fuse 2 dynamic bit writes", trace)
        c, trace = compile_source(AFFINE, debug_bounds=True)
        self.assertNotIn("fuse 2 dynamic bit writes", trace)
        self.assertIn("debug_bit_range", c)
        _, trace = compile_source("""
            define distant(i: uint8[0,7]) -> void {
                define bits: bit[128];
                bits[i,i+1) = @1;
                bits[i+96,i+97) = @1;
            }
        """)
        self.assertNotIn("fuse 2 dynamic bit writes", trace)
