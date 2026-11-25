import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


def normalized(text):
    """Drop the source-derived parts: positions, record digests and proof notes."""
    text = re.sub(r"lan_record_[0-9a-f]{24}", "lan_record_X", text)
    text = re.sub(r"lan_record_tag_[0-9a-f]{64}", "tag_X", text)
    text = re.sub(r"LAN_RECORD_[0-9A-F]{24}", "REC_X", text)
    text = re.sub(r"lan_api_[0-9a-f]+", "api_X", text)
    text = re.sub(r"_(?:\d+)_(?:\d+)", "_L", text)
    text = re.sub(r"\b\d+:\d+\b", "P", text)
    return "\n".join(line for line in text.splitlines()
                     if "implicit contract" not in line and "recursive result" not in line)


DIRECT = """
    define countdown(n: uint8) {
        if (n == 0) { return 0; }
        n -= 1;
        return countdown(n);
    }
"""

MUTUAL = """
    define even(n: uint8) { if (n == 0) { return true; } return odd(n); }
    define odd(n: uint8) { if (n == 0) { return false; } return even(n - 1); }
"""


class RecursiveResultInferenceTests(unittest.TestCase):
    def test_direct_recursion_infers_the_parameter_container(self):
        c, table = compile_source(DIRECT)
        self.assertIn("recursive result countdown: uint8_t [0, 255]", table)
        self.assertIn("uint8_t lan_fn_countdown(uint8_t ", c)
        self.assertIn("implicit contract countdown", table)

    def test_mutually_recursive_group_shares_one_assumption(self):
        c, table = compile_source(MUTUAL)
        self.assertIn("recursive result even: uint8_t [0, 1]", table)
        self.assertIn("recursive result odd: uint8_t [0, 1]", table)
        self.assertIn("sct recursive termination proved: even, odd", table)
        self.assertIn("uint8_t lan_fn_even(uint8_t ", c)

    def test_declared_and_inferred_members_can_share_a_group(self):
        _, table = compile_source(MUTUAL.replace("define even(n: uint8)", "define even(n: uint8) -> boolean", 1))
        self.assertIn("recursive result odd: uint8_t [0, 1]", table)
        self.assertNotIn("recursive result even", table)

    def test_signed_results_keep_their_sign(self):
        c, table = compile_source("""
            define down(n: int8) { if (n <= -5) { return -5; } return down(n - 1); }
        """)
        self.assertIn("recursive result down: int8_t [-128, 127]", table)
        self.assertIn("int8_t lan_fn_down(int8_t ", c)

    def test_void_recursion_stays_void(self):
        c, table = compile_source("define walk(n: uint8) { if (n == 0) { return; } walk(n - 1); }")
        self.assertIn("recursive result walk: void", table)
        self.assertIn("void lan_fn_walk(uint8_t ", c)

    def test_inferred_function_compiles_like_the_declared_one(self):
        pairs = (
            DIRECT,
            MUTUAL,
            """
            type Ctx = { flag: uint8, note: uint8 };
            define walk(ctx: Ctx, n: uint8) {
                if (n == 0) { return ctx.note; }
                ctx.flag = 1;
                return walk(ctx, n - 1);
            }
            """,
        )
        for source in pairs:
            with self.subTest(source=source):
                explicit_c, explicit_table = compile_source(self._declared(source))
                inferred_c, inferred_table = compile_source(source)
                self.assertEqual(normalized(explicit_c), normalized(inferred_c))
                self.assertEqual(normalized(explicit_table), normalized(inferred_table))

    @staticmethod
    def _declared(source):
        """Give every definition in a sample its explicit result contract."""
        declared = {"define countdown(n: uint8) {": "define countdown(n: uint8) -> uint8 {",
                    "define even(n: uint8) {": "define even(n: uint8) -> boolean {",
                    "define odd(n: uint8) {": "define odd(n: uint8) -> boolean {",
                    "define walk(ctx: Ctx, n: uint8) {": "define walk(ctx: Ctx, n: uint8) -> uint8 {"}
        for bare, typed in declared.items():
            source = source.replace(bare, typed)
        return source

    def test_a_result_that_never_stabilises_is_rejected(self):
        with self.assertRaisesRegex(LanError, "cannot infer the result of recursive function 'grow'"):
            compile_source("define grow(n: uint8) { if (n == 0) { return 0; } return grow(n - 1) + n; }")

    def test_body_errors_survive_speculation(self):
        with self.assertRaisesRegex(LanError, "cannot prove index"):
            compile_source("""
                define f(n: uint8) {
                    define a: uint8[2];
                    a[0] = 1;
                    if (n == 0) { return 0; }
                    return f(n - 1) + a[5];
                }
            """)

    def test_speculation_leaves_no_trace_of_the_failed_rounds(self):
        _, table = compile_source(DIRECT)
        self.assertEqual(table.count("@function countdown"), 1)
        self.assertEqual(table.count("param id#"), 1)
        self.assertEqual(table.count("recursive result countdown"), 1)

    def test_inferred_result_crosses_the_table_interface(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = DIRECT + "export { countdown };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source(
                'import { countdown } from "./provider.table";'
                ' define use_it(n: uint8) -> uint8 { return countdown(n); }',
                root / "consumer.lang",
            )
            self.assertIn("_countdown(", c)


if __name__ == "__main__":
    unittest.main()
