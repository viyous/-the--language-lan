"""Tests for custom string encoding types (`type string ... = Map as slice|tag`) and `expand`."""

import tempfile
import unittest
from pathlib import Path

from lan_compiler import LanError, compile_source


class CustomEncodingAndExpandTests(unittest.TestCase):
    def test_inline_json_map_slice_and_tokens(self):
        c, _ = compile_source("""
            type string GbaText = {
                " ": 0x00,
                "A": 0xBB,
                "B": 0xBC,
                "妙": [0x1E, 0x66],
                "蛙": [0x1E, 0x67],
                "{PLAYER}": [0xFD, 0x01],
                "\\n": 0xFE,
                "@terminator": 0xFF,
                "@pad": 0xFF,
            } as slice;

            define greet() -> uint8 {
                define msg: GbaText = "{PLAYER}妙蛙\\nA";
                // FD 01 (2) + 1E 66 (2) + 1E 67 (2) + FE (1) + BB (1) = 8 bytes (excluding 0xFF terminator)
                if (msg.len == 8) {
                    return msg[0];
                }
                return 0;
            }
        """)
        self.assertIn("0xfd, 0x01, 0x1e, 0x66, 0x1e, 0x67, 0xfe, 0xbb, 0xff", c)

    def test_default_mode_is_slice_and_tag_mode(self):
        c, _ = compile_source("""
            define MAP = {
                "A": 0xBB,
                "B": 0xBC,
                "@terminator": 0xFF,
            };
            type string DefaultSlice = MAP;
            type string TagEnc = MAP as tag;

            define check_both() -> uint8 {
                define s1: DefaultSlice = "AB";
                define s2: TagEnc[..] = "AB";
                define buf: TagEnc[4] = "AB";
                if (s1.len == 2 and s2.len == 2 and buf.len == 4) {
                    return buf[3];
                }
                return 0;
            }
        """)
        # Fixed buffer of size 4: 0xbb, 0xbc, 0xff (terminator), 0xff (default pad = terminator[0])
        self.assertIn("0xbb, 0xbc, 0xff, 0xff", c)

    def test_fixed_buffer_stack_rodata_module_const_and_record(self):
        c, _ = compile_source("""
            type string GbaText = {
                "妙": [0x1E, 0x66],
                "蛙": [0x1E, 0x67],
                "种": [0x1E, 0x68],
                "子": [0x1E, 0x69],
                "@terminator": 0xFF,
                "@pad": 0x00,
            } as slice;

            define SPECIES_NAME: GbaText[10] = "妙蛙种子";

            type Species = {
                id: uint16,
                name: GbaText[10],
            } · {repr: c};

            define STATIC_SPECIES: Species = {
                id: 1,
                name: "妙蛙",
            };

            define build_species() -> uint16 {
                define local_buf: GbaText[10] = "妙蛙种子";
                define ro_buf: GbaText[10] · {storage: rodata} = "妙蛙";
                define sp: Species = { id: 1, name: "妙蛙" };
                return local_buf[8] + ro_buf[4] + sp.name[5] + SPECIES_NAME[9] + STATIC_SPECIES.name[4];
            }
        """)
        # "妙蛙种子" = 8 bytes + 0xff terminator + 1 byte pad 0x00 = 10 bytes
        self.assertIn("0x1e, 0x66, 0x1e, 0x67, 0x1e, 0x68, 0x1e, 0x69, 0xff, 0x00", c)
        # "妙蛙" = 4 bytes + 0xff terminator + 5 bytes pad 0x00 = 10 bytes
        self.assertIn("0x1e, 0x66, 0x1e, 0x67, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00", c)

    def test_fixed_buffer_overflow_and_unmapped_char_errors(self):
        with self.assertRaisesRegex(LanError, "exceeds fixed capacity"):
            compile_source("""
                type string GbaText = {
                    "妙": [0x1E, 0x66],
                    "蛙": [0x1E, 0x67],
                    "@terminator": 0xFF,
                };
                define f() -> uint8 {
                    // 2 + 2 + 1 = 5 bytes > 4
                    define buf: GbaText[4] = "妙蛙";
                    return buf[0];
                }
            """)
        with self.assertRaisesRegex(LanError, "is not mapped in encoding"):
            compile_source("""
                type string GbaText = { "A": 0xBB, "@terminator": 0xFF };
                define f() -> uint8 {
                    define s: GbaText = "AZ";
                    return s[0];
                }
            """)

    def test_map_null_deletes_entry_and_no_terminator(self):
        c, _ = compile_source("""
            define RAW_MAP = { "A": 0x41, "B": 0x42, "@terminator": null };
            type string RawEnc = RAW_MAP as slice;
            define f() -> uint8 {
                define s: RawEnc = "AB";
                define buf: RawEnc[4] = "AB";
                return s[0] + buf[2];
            }
        """)
        # RawEnc[4] with empty terminator defaults pad to 0x00: 0x41, 0x42, 0x00, 0x00
        self.assertIn("0x41, 0x42, 0x00, 0x00", c)
        # Deleting an entry via null
        with self.assertRaisesRegex(LanError, "is not mapped in encoding"):
            compile_source("""
                define BASE = { "A": 0xBB, "B": 0xBC };
                type string Narrow = BASE · { "B": null };
                define f() -> uint8 {
                    define s: Narrow = "AB";
                    return s[0];
                }
            """)

    def test_strict_encoding_isolation_views_and_extern_ffi(self):
        # string cannot pass to GbaText
        with self.assertRaisesRegex(LanError, "does not satisfy"):
            compile_source("""
                type string GbaText = { "A": 0xBB, "@terminator": 0xFF };
                define take_gba(s: GbaText) -> uint8 { return s.len as: sat uint8; }
                define caller(s: string) -> uint8 { return take_gba(s); }
            """)
        # GbaText cannot pass to c_string
        with self.assertRaisesRegex(LanError, "cannot prove a NUL terminator|does not satisfy"):
            compile_source("""
                type string GbaText = { "A": 0xBB, "@terminator": 0xFF };
                define take_c(s: c_string) -> uint8 { return s.len as: sat uint8; }
                define caller(s: GbaText) -> uint8 { return take_c(s); }
            """)
        # Even a GbaText with 0x00 terminator cannot pass to c_string
        with self.assertRaisesRegex(LanError, "does not satisfy"):
            compile_source("""
                type string CustomZero = { "A": 0xBB, "@terminator": 0x00 };
                define take_c(s: c_string) -> uint8 { return s.len as: sat uint8; }
                define caller() -> uint8 {
                    define s: CustomZero = "A";
                    return take_c(s);
                }
            """)
        # Different encodings cannot mix
        with self.assertRaisesRegex(LanError, "does not satisfy"):
            compile_source("""
                type string EncA = { "A": 0xBB, "@terminator": 0xFF };
                type string EncB = { "A": 0x41, "@terminator": 0x00 };
                define take_a(s: EncA) -> uint8 { return s.len as: sat uint8; }
                define caller(s: EncB) -> uint8 { return take_a(s); }
            """)
        # Zero-copy view, explicit cast, and extern C const uint8_t * FFI
        c, _ = compile_source("""
            type string GbaText = { "A": 0xBB, "B": 0xBC, "@terminator": 0xFF } as slice;
            extern contract gba_print(text: GbaText) -> void writes {external};

            define run(raw: uint8[..] · {immutable: true}) -> uint8 {
                define literal = "AB" as GbaText;
                gba_print("AB");
                gba_print(literal);
                define bytes = literal as: view uint8[..] · {immutable: true};
                define as_gba = raw as: view GbaText;
                gba_print(as_gba);
                return bytes[0];
            }
        """)
        self.assertIn("extern void gba_print(const uint8_t *);", c)
        self.assertIn("0xbb, 0xbc, 0xff", c)

    def test_expand_json_map_loops_chr_and_index_assign(self):
        c, _ = compile_source("""
            define GBA_MAP = {
                " ": 0x00,
                "@terminator": 0xFF,
            };

            type AddAsciiUpper(m: type) {
                interation(i; 0; uint8; 26; 1) {
                    m[0x41 + i] = 0xBB + i;
                }
                return m;
            }

            type AddAsciiLower(m: type) {
                interation(i; 0; uint8; 26; 1) {
                    m[chr(0x61 + i)] = 0xD5 + i;
                }
                return m;
            }

            type AddDigits(m: type) {
                interation(i; 0; uint8; 10; 1) {
                    m = m · { [0x30 + i]: 0xA1 + i };
                }
                return m;
            }

            expand AddAsciiUpper(GBA_MAP);
            expand AddAsciiLower(GBA_MAP);
            expand AddDigits(GBA_MAP);

            type string GbaText = GBA_MAP as slice;

            define test_msg() -> uint8 {
                define s: GbaText = "Az09 ";
                // A=0xBB, z=0xD5+25=0xEE, 0=0xA1, 9=0xAA, ' '=0x00, term=0xFF
                return s[0];
            }
        """)
        self.assertIn("0xbb, 0xee, 0xa1, 0xaa, 0x00, 0xff", c)
        # Pure compile-time JsonMap emits no C global
        self.assertNotIn("lan_const_GBA_MAP", c)

    def test_expand_top_down_order_and_final_materialization_scalars(self):
        c, _ = compile_source("""
            define MAX_HP: uint16 = 100;
            define BEFORE_HP = MAX_HP * 2;

            type Scale(v: uint16, k: uint16) {
                return v * k;
            }

            expand Scale(MAX_HP, 3);
            define AFTER_HP = MAX_HP + 5;

            define check() -> uint16 {
                // MAX_HP final value is 300, BEFORE_HP is 200, AFTER_HP is 305
                if (MAX_HP == 300 and BEFORE_HP == 200 and AFTER_HP == 305) {
                    return MAX_HP + BEFORE_HP + AFTER_HP;
                }
                return 0;
            }
        """)
        # Each constant emitted ONCE with its final value
        self.assertEqual(c.count("lan_const_MAX_HP ="), 1)
        self.assertIn("lan_const_MAX_HP = ((lan_int)UINT64_C(300));", c)
        self.assertIn("lan_const_BEFORE_HP = ((lan_int)UINT64_C(200));", c)
        self.assertIn("lan_const_AFTER_HP = ((lan_int)UINT64_C(305));", c)

    def test_expand_fixed_array_untyped_list_and_slice_const(self):
        c, _ = compile_source("""
            define FIXED: uint16[4] = [1, 2, 3, 4];
            define LUT = [10, 20];
            define SLICE_LUT: uint16[..] = [100, 200];

            type DoubleFixed(a: uint16[4]) {
                interation(i; 0; uint8; 4; 1) {
                    a[i] = a[i] * 2;
                }
                return a;
            }

            type Append(a: type, v: uint16) {
                return a · [v];
            }

            expand DoubleFixed(FIXED);
            expand Append(LUT, 30);
            expand Append(SLICE_LUT, 300);

            define read_all() -> uint16 {
                if (FIXED[3] == 8 and LUT.len == 3 and LUT[2] == 30
                        and SLICE_LUT.len == 3 and SLICE_LUT[2] == 300) {
                    return FIXED[3] + LUT[2] + SLICE_LUT[2];
                }
                return 0;
            }
        """)
        self.assertIn("lan_const_FIXED = {((lan_int)UINT64_C(2)), ((lan_int)UINT64_C(4)), ((lan_int)UINT64_C(6)), ((lan_int)UINT64_C(8))};", c)
        self.assertIn("lan_const_LUT = {((lan_int)UINT64_C(10)), ((lan_int)UINT64_C(20)), ((lan_int)UINT64_C(30))};", c)
        self.assertIn("lan_const_SLICE_LUT_data[3] = {((lan_int)UINT64_C(100)), ((lan_int)UINT64_C(200)), ((lan_int)UINT64_C(300))};", c)

    def test_expand_type_violation_and_runtime_json_map_rejected(self):
        with self.assertRaisesRegex(LanError, "does not fit"):
            compile_source("""
                define SMALL: uint8 = 100;
                type Add200(v: uint16) { return v + 200; }
                expand Add200(SMALL);
            """)
        with self.assertRaisesRegex(LanError, "cannot be used at runtime"):
            compile_source("""
                define M = { "A": 0xBB };
                define f() -> uint8 { return M; }
            """)

    def test_cross_module_expand_and_encoding_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "part1.lang").write_text("""
                define BASE_MAP = {
                    " ": 0x00,
                    "@terminator": 0xFF,
                };
                type AddPokemon(m: type) {
                    return m · {
                        "妙": [0x1E, 0x66],
                        "蛙": [0x1E, 0x67],
                    };
                }
                export { BASE_MAP, AddPokemon };
            """, encoding="utf-8")
            (root / "enc.lang").write_text("""
                import { BASE_MAP, AddPokemon } from "./part1.lang";
                define GBA_MAP = BASE_MAP;
                expand AddPokemon(GBA_MAP);
                type string GbaText = GBA_MAP as slice;
                type string GbaChar = GBA_MAP as tag;

                define measure(s: GbaText) -> uint8 {
                    return s.len as: sat uint8;
                }
                export { GbaText, GbaChar, measure };
            """, encoding="utf-8")
            (root / "main.lang").write_text("""
                import { GbaText, GbaChar, measure } from "./enc.lang";
                define run() -> uint16 {
                    define name: GbaText = "妙蛙";
                    define buf: GbaChar[6] = "妙蛙";
                    return measure(name) + buf[4];
                }
            """, encoding="utf-8")
            main_path = root / "main.lang"
            c, _ = compile_source(main_path.read_text(encoding="utf-8"), main_path)
            self.assertIn("0x1e, 0x66, 0x1e, 0x67, 0xff", c)
            self.assertIn("0x1e, 0x66, 0x1e, 0x67, 0xff, 0xff", c)

    def test_chinese_identifiers_with_custom_encoding_and_expand(self):
        c, _ = compile_source("""
            define 码表 = { " ": 0x00, "@terminator": 0xFF };
            type 追加宝可梦(表: type) {
                return 表 · { "皮": [0x1E, 0x10], "卡": [0x1E, 0x11], "丘": [0x1E, 0x12] };
            }
            expand 追加宝可梦(码表);
            type string 宝可梦文本 = 码表 as slice;

            define 获取名字() -> 宝可梦文本 {
                return "皮卡丘";
            }
        """)
        self.assertTrue(c.isascii())
        self.assertIn("0x1e, 0x10, 0x1e, 0x11, 0x1e, 0x12, 0xff", c)


if __name__ == "__main__":
    unittest.main()
