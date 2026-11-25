import re
import unittest

from lanc import LanError, compile_source


LAN_SOURCE = """
define take(s: uint8[..] · {elem: {7}, len: {2}, immutable: true}) -> uint8 {
    return s[0];
}
define receive(x: uint8) -> uint8 {
    define data: uint8[2] = [x; 2];
    return take(data[0,2));
}
"""

NATIVE_SOURCE = """
extern contract inspect(s: uint8[..] · {elem: {7}, len: {2}, immutable: true}, y: uint8) -> uint8 writes {};
define change(a: uint8[2] · {pass: alias}, x: uint8) -> uint8 {
    a[0] = x;
    a[1] = x;
    return x;
}
define receive(x: uint8) -> uint8 {
    define data: uint8[2] = [6; 2];
    return inspect(data[0,2), change(data, x));
}
"""

DEPENDENT_SOURCE = """
extern contract bounded(a: uint8[..] · {immutable: true}, n: uint8 & <= a.len * 2) -> uint8 writes {};
define receive_bound(x: uint8) -> uint8 {
    define data: uint8[2];
    return bounded(data[0,2), x);
}
"""


class DebugArgumentContentsTests(unittest.TestCase):
    def test_debug_falls_back_to_a_runtime_scan_for_a_lan_argument(self):
        with self.assertRaisesRegex(LanError, "cannot prove slice element contents"):
            compile_source(LAN_SOURCE)
        c, trace = compile_source(LAN_SOURCE, debug_bounds=True)
        self.assertIn("debug check slice argument elements", trace)
        self.assertRegex(c, r"\(void\)lh_debug_native_elements_\d+\(lan_arg_\d+_\d+_0\)[,;]\s*lan_fn_take\(")

    def test_debug_checks_native_argument_after_later_writes(self):
        with self.assertRaisesRegex(LanError, "cannot prove slice element contents"):
            compile_source(NATIVE_SOURCE)
        c, _ = compile_source(NATIVE_SOURCE, debug_bounds=True)
        call_line = next(line for line in c.splitlines() if "return " in line and " inspect(" in line)
        self.assertEqual(call_line.count("lan_fn_change("), 1)
        self.assertEqual(len(re.findall(r"lan_slice(?:\*/)? lan_arg_\d+_\d+_0 =", call_line)), 1)
        self.assertLess(call_line.index("lan_fn_change("), call_line.index("debug_native_elements"))
        self.assertLess(call_line.index("debug_native_elements"), call_line.index(" inspect("))

    def test_debug_validation_is_emitted_for_shared_multibranch_call(self):
        source = """
define consume(s: uint8[..] · {elem: {7}, immutable: true}) -> void {}
define receive(flag: boolean, x: uint8) -> void {
    define data: uint8[2] = [x; 2];
    if (flag) { data[0,2) = 7; }
    consume(data[0,2));
}
"""
        c, _ = compile_source(source, debug_bounds=True)
        self.assertEqual(len(re.findall(r"\(void\)lh_debug_native_elements_\d+\(", c)), 1)

    def test_debug_checks_dependent_bounds_from_the_captured_slice_descriptor(self):
        for bound, error in (("<= a.len * 2", "expression"), ("< a.len", "bound")):
            source = DEPENDENT_SOURCE.replace("<= a.len * 2", bound)
            with self.subTest(bound=bound):
                with self.assertRaisesRegex(LanError, f"cannot prove dependent parameter {error}"):
                    compile_source(source)
                c, trace = compile_source(source, debug_bounds=True)
                self.assertIn("debug check dependent arguments", trace)
                call = next(line for line in c.splitlines() if "return " in line and " bounded(" in line)
                self.assertEqual(len(re.findall(r"lan_slice(?:\*/)? lan_arg_\d+_\d+_0 =", call)), 1)
                check = call.index("oracle_check")
                self.assertLess(check, call.index(" bounded("))
                self.assertRegex(call[check:call.index(" bounded(")], r"lan_arg_\d+_\d+_0\.len")
