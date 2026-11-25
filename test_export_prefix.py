import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph

SOURCE = """
define iso_step(v: uint32) -> uint32 { return v; }
define hidden(v: uint8) -> uint8 { return v; }
export { iso_step };
"""


def build(source, prefix):
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
        path = Path(scratch) / "game.lang"
        compiler = Compiler(source, path.name)
        return ModuleGraph(path, compiler, library=True, export_prefix=prefix).compile()[0]


class ExportPrefixTests(unittest.TestCase):
    def test_default_has_only_fingerprinted_symbol(self):
        c = build(SOURCE, None)
        self.assertRegex(c, r"uint32_t lan_api_\w+_iso_step\(uint32_t arg_0\)")
        self.assertNotRegex(c, r"\bgame_iso_step\b")

    def test_prefix_adds_stable_alias_next_to_fingerprinted_symbol(self):
        c = build(SOURCE, "game_")
        self.assertRegex(c, r"uint32_t lan_api_\w+_iso_step\(uint32_t arg_0\) \{")
        self.assertRegex(c, r"(?m)^uint32_t game_iso_step\(uint32_t arg_0\) \{\n    return lan_fn_iso_step\(arg_0\);")
        self.assertNotIn("game_hidden", c)
        self.assertRegex(c, r"static __attribute__\(\(unused\)\) uint32_t lan_fn_iso_step")

    def test_empty_prefix_uses_bare_export_name(self):
        c = build(SOURCE, "")
        self.assertRegex(c, r"(?m)^uint32_t iso_step\(uint32_t arg_0\) \{")

    def test_reserved_or_invalid_alias_is_rejected(self):
        for prefix in ("lan_fn_", "lh_", "1x-"):
            with self.subTest(prefix=prefix), self.assertRaises(LanError):
                build(SOURCE, prefix)
        with self.assertRaises(LanError):
            build(SOURCE.replace("iso_step };", "iso_step as main };"), "")


if __name__ == "__main__":
    unittest.main()
