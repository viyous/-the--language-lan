import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


SOURCE = """
enum Err: uint8 { Bad = 250 }
type R = uint8[0,4] | Err;
type Fail = R & {250};
type Good = R & {3};
define fail() -> Fail { return Err.Bad; }
define good() -> Good { return 3; }
define forward() { return fail()?; }
"""


class RefinedResultArmsTests(unittest.TestCase):
    def test_only_error_is_still_classified_in_an_inferred_contract(self):
        _, trace = compile_source(SOURCE + "define relay() -> R { return forward()?; }")
        self.assertIn("propagate error {250}; continue with empty", trace)
        self.assertRegex(trace, r'implicit contract forward .*"kind": "result_union", "success": \[\]')

    def test_only_success_and_only_error_roundtrip_through_table(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
            directory = Path(scratch)
            path = directory / "provider.lang"
            compiler = Compiler(SOURCE + "export { Err, R, forward, good };", path.name)
            _, table = ModuleGraph(path, compiler, library=True).compile()
            (directory / "provider.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
import { Err, R, forward, good } from "./provider.table";
define relay() -> R { return forward()?; }
define success() -> uint8[3,3] { return good()?; }
""", directory / "consumer.lang")
            self.assertIn("propagate error {250}; continue with empty", trace)
            self.assertIn("propagate error empty; continue with {3}", trace)

    def test_intersection_cannot_erase_foreign_error_enum_identity(self):
        source = SOURCE + """
enum Other: uint8 { Bad = 250 }
type OtherResult = uint8[0,4] | Other;
type Combined = (R & OtherResult) & {250};
define mixed() -> Combined { return 250; }
define rejected() -> R { return mixed()?; }
"""
        with self.assertRaisesRegex(LanError, "propagated error enum identity differs"):
            compile_source(source)
        source = source.replace("define rejected() -> R { return mixed()?; }",
                                "define rejected() -> Err { return mixed(); }")
        with self.assertRaisesRegex(LanError, "enum identity differs"):
            compile_source(source)
