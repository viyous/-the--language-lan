import unittest

from lanc import LanError, compile_source


ALIASES = """
    define choose(flag: boolean, a: uint8 · {pass: alias}, b: uint8 · {pass: alias}) {
        define p: uint8 · {pass: alias} = a;
        if (flag) { p = b; }
        return p;
    }
    define f(flag: boolean) -> uint8[7,7] {
        define a: uint8 = 1;
        define b: uint8 = 2;
        define p: uint8 · {pass: alias} = choose(flag, a, b);
        define keep_running = true;
        /** @oracle @transition => { state(keep_running) := {0}; state(p.value) := {7}; } */
        while (keep_running) { p.value = 7; keep_running = false; }
        return p.value;
    }
"""

GUARDED = """
    define f(flag: boolean) -> uint8 {
        define data: uint8[4];
        define s = data[0,4);
        define run = true;
        /** @oracle @ensures {
            when (flag) state(s[0,2)) := {7};
            when (!flag) state(s[0,2)) := {8};
        } */
        while (run) {
            if (flag) { s[0,2) = 7; }
            else { s[0,2) = 8; }
            run = false;
        }
        return s[0];
    }
"""


class GuardedStateOracleTests(unittest.TestCase):
    def test_distinct_exit_contents_can_be_claimed_by_guard(self):
        _, trace = compile_source(GUARDED)
        self.assertIn("verified postcondition", trace)
        rows = [line.strip() for line in trace.splitlines() if line.strip().startswith("when (")]
        self.assertEqual(len(rows), 2)
        start = GUARDED.index("when (flag)")
        end = GUARDED.index("} */", start)
        compile_source(GUARDED[:start] + "\n".join(rows) + GUARDED[end:])
        with self.assertRaisesRegex(LanError, "guarded exit"):
            compile_source(GUARDED.replace(":= {8}", ":= {7}"))
        with self.assertRaisesRegex(LanError, "guarded exit"):
            compile_source(GUARDED.replace("run = false;", "flag = !flag; run = false;"))

    def test_guard_rejects_calls_and_is_not_an_assumption(self):
        source = "define check(flag: boolean) -> boolean { return flag; }\n" + GUARDED
        with self.assertRaisesRegex(LanError, "guard must be side-effect free"):
            compile_source(source.replace("when (flag)", "when (check(flag))"))
        source = GUARDED.replace("-> uint8", "-> uint8[7,7]")
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source)

    def test_transition_covers_each_concrete_target_of_an_alias(self):
        _, trace = compile_source(ALIASES)
        self.assertIn("verified transition", trace)
        self.assertIn("return {7}", trace)

    def test_rebind_and_old_target_write_need_separate_coverage(self):
        old_write = ALIASES.replace("p.value = 7; keep_running = false;", "p.value = 7; p = b; p.value = 7; keep_running = false;")
        target_only = old_write.replace("state(p.value) := {7};",
                                        "state(p.value) := {7}; state(p.target) := {b};")
        with self.assertRaisesRegex(LanError, "omits modified storage 'a'"):
            compile_source(target_only)
        compile_source(target_only.replace("state(p.target) := {b};",
                                           "state(p.target) := {b}; state(a) := [1,7];"))
        simple_rebind = """
            define f() -> uint8 {
                define a: uint8 = 7;
                define b: uint8 = 7;
                define p: uint8 · {pass: alias} = a;
                define keep_running = true;
                /** @oracle @transition => { state(keep_running) := {0}; state(p.value) := {7}; } */
                while (keep_running) { p = b; keep_running = false; }
                return p.value;
            }
        """
        with self.assertRaisesRegex(LanError, "omits modified storage 'p'"):
            compile_source(simple_rebind)
        compile_source(simple_rebind.replace("state(p.value) := {7};", "state(p.target) := {b};"))

    def test_slice_rebind_covers_owner_start_and_length(self):
        source = """
            define f() -> uint8 {
                define a: uint8[4]; define b: uint8[4]; define s = a[0,2);
                define keep_running = true;
                /** @oracle @transition => { state(keep_running) := {0};
                    state(s.len) := {2};
                    state(s.target) := {b};
                    state(s.start) := {1};
                } */
                while (keep_running) { s = b[1,3); keep_running = false; }
                return s[0];
            }
        """
        _, trace = compile_source(source)
        self.assertIn("verified transition", trace)
        with self.assertRaisesRegex(LanError, "omits modified storage"):
            compile_source(source.replace("state(s.target) := {b};", ""))
        with self.assertRaisesRegex(LanError, "unexpected start offset"):
            compile_source(source.replace("state(s.start) := {1};", "state(s.start) := {0};"))
