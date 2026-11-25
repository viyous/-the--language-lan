import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


FORWARD_CALL = """
    define wrapper(x) { return take(x); }
    define take(value: uint16[2,9]) -> uint16[2,9] { return value; }
"""


class ParameterInferenceTests(unittest.TestCase):
    def test_annotated_parameter_retains_its_type_at_return(self):
        c, table = compile_source("define f(x: uint8) { return x; }")
        self.assertIn("uint8_t lan_fn_f(uint8_t ", c)
        self.assertIn("implicit contract f", table)

    def test_infers_from_a_direct_annotated_use(self):
        for source in ("define f(x) -> uint8 { return x; }",
                       "define f(x) { define y: uint8 = x; return y; }"):
            with self.subTest(source=source):
                c, table = compile_source(source)
                self.assertIn("uint8_t lan_fn_f(uint8_t ", c)
                self.assertIn("inferred parameter f.x: uint8_t", table)

    def test_rejects_missing_conflicting_or_shadowed_evidence(self):
        cases = (
            ("define f(x) { return x; }", "cannot infer parameter"),
            ("define f(x) -> uint8 { define y: uint16 = x; return x; }", "conflicting inferred types"),
            ("define f(x) -> uint8 { if (true) { define x: uint8 = 1; return x; } return 0; }",
             "cannot infer parameter"),
        )
        for source, error in cases:
            with self.subTest(source=source), self.assertRaisesRegex(LanError, error):
                compile_source(source)

    def test_infers_the_exact_scalar_contract_of_a_forwarded_argument(self):
        c, table = compile_source(FORWARD_CALL)
        self.assertIn("uint16_t lan_fn_wrapper(uint16_t ", c)
        self.assertIn("inferred parameter wrapper.x: uint16_t & [2, 9]", table)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(FORWARD_CALL + "define bad() -> uint16 { return wrapper(10); }")

    def test_infers_from_native_calls_inside_other_expressions(self):
        source = """
            define wrapper(flag) -> boolean { if (choose(flag)) { return true; } return false; }
            extern contract choose(value: boolean) -> boolean writes {};
        """
        _, table = compile_source(source)
        self.assertIn("inferred parameter wrapper.flag: uint8_t & [0, 1]", table)

    def test_uses_the_explicit_contract_loaded_from_table(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
            directory = Path(scratch)
            provider = directory / "provider.lang"
            source = "define take(value: uint16[2,9]) -> uint16 { return value; } export { take };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (directory / "provider.table").write_text(table, encoding="ascii")
            _, trace = compile_source('import { take } from "./provider.table"; '
                                      'define wrapper(x) { return take(x); }',
                                      directory / "consumer.lang")
            self.assertIn("inferred parameter wrapper.x: uint16_t & [2, 9]", trace)

    def test_call_evidence_checks_conflicts_shadowing_and_dependent_contracts(self):
        cases = (
            ("define take(a: uint8, b: uint8 & <= a) -> uint8 { return b; } "
             "define f(x) { return take(x, 3); }", "cannot prove dependent parameter bound"),
            ("define take(a: uint8) -> uint8 { return a; } "
             "define f(x) { if (true) { define x: uint8 = 1; return take(x); } return 0; }",
             "cannot infer parameter"),
            ("define take(a: uint8) -> uint8 { return a; } "
             "define f(x) { define wide: uint16 = x; return take(x); }", "conflicting inferred types"),
        )
        for source, error in cases:
            with self.subTest(source=source), self.assertRaisesRegex(LanError, error):
                compile_source(source)
