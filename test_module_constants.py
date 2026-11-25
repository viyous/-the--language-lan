import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


class ModuleConstantTests(unittest.TestCase):
    def test_scalar_typed_and_inferred_constants(self):
        c, _ = compile_source("""
            define MAX_HP: uint16 = 999;
            define LCG_MULT: uint32 = 1103515245;
            define LCG_INC = 12345;
            define FLAG = true;
            define SCALE: float32 = 1.5;
            define DOUBLE_HP: uint16 = MAX_HP * 2;

            type Hp = uint16[0, MAX_HP];

            define clamp_hp(v: uint16) -> Hp {
                if (FLAG && v > MAX_HP) {
                    return MAX_HP;
                }
                return v;
            }

            define step(seed: uint32) -> uint32 {
                return (seed * LCG_MULT + LCG_INC) as: wrap uint32;
            }

            define check_double() -> uint16[1998, 1998] {
                return DOUBLE_HP;
            }

            define scaled(x: float32[0.0, 10.0]) -> float32[0.0, 15.0] {
                return x * SCALE;
            }
        """)
        self.assertIn("static const uint16_t lan_const_MAX_HP = ((lan_int)UINT64_C(999));", c)
        self.assertIn("static const uint32_t lan_const_LCG_MULT = ((lan_int)UINT64_C(1103515245));", c)
        self.assertIn("static const uint16_t lan_const_LCG_INC = ((lan_int)UINT64_C(12345));", c)
        self.assertIn("static const uint8_t lan_const_FLAG = ((lan_int)UINT64_C(1));", c)
        self.assertIn("static const float lan_const_SCALE =", c)

    def test_array_record_and_string_module_constants(self):
        c, _ = compile_source("""
            define N = 4;
            type Move = {power: uint8[0, 200], acc: uint8[0, 100]};

            define LUT: uint8[N] = [10, 20, 30, 40];
            define ZEROS: uint8[8] = [0; 8];
            define ORIGIN: Move = {power: 40, acc: 100};
            define MOVES: Move[2] = [
                {power: 40, acc: 100},
                {power: 90, acc: 85},
            ];
            define LABEL = "GBA";

            define get_lut(i: uint8[0, LUT.len - 1]) -> uint8[10, 40] {
                return LUT[i];
            }

            define get_move_power(i: uint8[0, 1]) -> uint8[40, 90] {
                return MOVES[i].power;
            }

            define get_origin_acc() -> uint8[100, 100] {
                return ORIGIN.acc;
            }

            define get_label_len() -> uint8[3, 3] {
                return LABEL.len as uint8[3, 3];
            }

            define view_lut() -> uint8[..] · {immutable: true, elem: [10, 40]} {
                return LUT[, ];
            }
        """)
        self.assertIn("static const lan_array_", c)
        self.assertIn("lan_const_LUT", c)
        self.assertIn("lan_const_MOVES", c)
        self.assertIn("lan_const_ORIGIN", c)
        self.assertIn("lan_const_LABEL", c)
        self.assertNotIn("[0 ...", c)

    def test_module_const_used_in_types_and_repeat_counts(self):
        c, _ = compile_source("""
            define SIZE: uint8 = 6;
            define INIT_VAL: uint8 = 7;

            define make_sum() -> uint8[42, 42] {
                define buf: uint8[SIZE] = [INIT_VAL; SIZE];
                return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5];
            }
        """)
        self.assertIn("lan_const_SIZE", c)

    def test_module_constants_are_immutable_and_reject_cycles(self):
        with self.assertRaisesRegex(LanError, "immutable|read-only|cannot assign|readonly"):
            compile_source("""
                define X: uint8 = 5;
                define bad() -> void { X = 6; }
            """)
        with self.assertRaisesRegex(LanError, "immutable|read-only|cannot assign|readonly"):
            compile_source("""
                define ARR: uint8[2] = [1, 2];
                define bad() -> void { ARR[0] = 9; }
            """)
        with self.assertRaisesRegex(LanError, "cyclic constant definition"):
            compile_source("""
                define A: uint8 = B;
                define B: uint8 = A;
            """)

    def test_export_and_import_module_constants(self):
        provider = """
            define MAX_LEVEL: uint8 = 100;
            define TABLE: uint8[3] = [5, 10, 15];
            export { MAX_LEVEL, TABLE };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
            prov_path = Path(scratch) / "defs.lang"
            prov_path.write_text(provider, encoding="utf-8")
            # Import both scalar and array constants from .lang source
            user_src = """
                import { MAX_LEVEL, TABLE } from "./defs.lang";
                define check(i: uint8[0, 2]) -> uint8[105, 115] {
                    return MAX_LEVEL + TABLE[i];
                }
            """
            user_path = Path(scratch) / "user.lang"
            c, _ = ModuleGraph(user_path, Compiler(user_src, user_path.name)).compile()
            self.assertIn("lan_m1_const_MAX_LEVEL", c)
            self.assertIn("lan_m1_const_TABLE", c)

            # Export to .table and import scalar constant from .table
            _, table = ModuleGraph(prov_path, Compiler(provider, prov_path.name), library=True).compile()
            prov_path.with_suffix(".table").write_text(table, encoding="ascii")
            table_user = """
                import { MAX_LEVEL } from "./defs.table";
                define cap() -> uint8[100, 100] { return MAX_LEVEL; }
            """
            tc, _ = compile_source(table_user, Path(scratch) / "table_user.lang")
            self.assertIn("static const uint8_t lan_const_imp_", tc)


if __name__ == "__main__":
    unittest.main()
