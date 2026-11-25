import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


SLICE = """
    define f(flag: boolean) -> uint8 {
        define data: uint8[4];
        define s = data[0,4);
        define run = true;
        /** @oracle @ensures { state(s[0,2)) := {7}; } */
        while (run) {
            if (flag) { s[0] = 7; s[1] = 7; }
            else { s[0,2) = 7; }
            run = false;
        }
        return s[0];
    }
"""

GRAPH = """
    type R = { x: uint8, next: R | null };
    define cut(a: R) -> void {
        a.next = null;
    }
"""

GRAPH_CALL = """
    define f() -> uint8 {
        define a: R;
        define b: R;
        a.next = b;
        define keep_running = true;
        /** @oracle @ensures { state(a.next) := {null}; } */
        while (keep_running) { cut(a); keep_running = false; }
        return a.x;
    }
"""

DEEP_GRAPH = GRAPH.replace("a.next = null;", "if (a.next != null) { a.next.next = null; }")
DEEP_CALL = """
    define deep_case() -> uint8[7,7] {
        define a: R;
        define b: R = { x: 7 };
        define c: R;
        a.next = b;
        b.next = c;
        define keep_running = true;
        /** @oracle @ensures { state(a.next) := {b}; state(b.next) := {null}; } */
        while (keep_running) { cut(a); keep_running = false; }
        return a.next.x;
    }
"""


class CompositeStateOracleTests(unittest.TestCase):
    def test_record_transition_flattens_to_reusable_table_rows(self):
        source = """
            type Pair = { x: uint8, y: uint8 };
            define f() -> uint8 {
                define p: Pair = { x: 2, y: 4 };
                /** @oracle @transition => { state(p) := { x: {0}, y: {4} }; } */
                while (p.x > 0) { p.x -= 1; }
                return p.y;
            }
        """
        _, table = compile_source(source)
        rows = [line.strip() for line in table.splitlines() if line.strip().startswith("state(")]
        self.assertEqual(rows, ["state(p.x) := { 0 };", "state(p.y) := { 4 };"])
        compile_source(source.replace("state(p) := { x: {0}, y: {4} };", " ".join(rows)))
        with self.assertRaisesRegex(LanError, "omits modified storage"):
            compile_source(source.replace("x: {0}, y: {4}", "y: {4}"))

    def test_slice_postcondition_checks_all_write_paths(self):
        _, table = compile_source(SLICE)
        self.assertIn("verified postcondition", table)
        self.assertIn("return {7}", table)
        with self.assertRaisesRegex(LanError, "postcondition fails at exit"):
            compile_source(SLICE.replace("s[1] = 7", "s[1] = 8"))

    def test_heap_postcondition_checks_actual_liveness(self):
        source = """
            define f() -> uint8 {
                define p: uint8 · {storage: heap} = 3;
                define keep_running = true;
                /** @oracle @ensures { state(p) := Tombstone; } */
                while (keep_running) { free(p); keep_running = false; }
                return 0;
            }
        """
        _, table = compile_source(source)
        self.assertIn("state(p) := Tombstone;", table)
        with self.assertRaisesRegex(LanError, "cannot prove p is Alive"):
            compile_source(source.replace(":= Tombstone", ":= Alive"))

    def test_transition_tracks_liveness_separately_from_payload(self):
        source = """
            define f() -> uint8 {
                define p: uint8 · {storage: heap} = 3;
                define keep_running = true;
                /** @oracle @transition => { state(keep_running) := {0}; state(p) := Tombstone; } */
                while (keep_running) { free(p); keep_running = false; }
                return 0;
            }
        """
        _, table = compile_source(source)
        self.assertIn("state(p) := Tombstone;", table)
        with self.assertRaisesRegex(LanError, "omits modified storage"):
            compile_source(source.replace("free(p);", "p = 4; free(p);"))
        with self.assertRaisesRegex(LanError, "require a verified transition"):
            compile_source(source.replace("@transition", "@trusted @transition"))

    def test_graph_root_link_postcondition_survives_table_import(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            consumer = root / "consumer.lang"
            imports = 'import { R, cut } from "./provider.table";'
            for body, verified in ((GRAPH, True),
                                   (GRAPH.replace("a.next = null;", "if (a.x == 1) { a.next = null; }"), False)):
                exported = body + "export { R, cut };"
                _, table = ModuleGraph(provider, Compiler(exported, provider.name), library=True).compile()
                (root / "provider.table").write_text(table, encoding="ascii")
                if verified:
                    self.assertIn("mutation param#0.next => null", table)
                    _, result = compile_source(imports + GRAPH_CALL, consumer)
                    self.assertIn("verified postcondition", result)
                else:
                    with self.assertRaisesRegex(LanError, "unexpected targets for next"):
                        compile_source(imports + GRAPH_CALL, consumer)

    def test_guarded_graph_path_preserves_other_links_across_table_import(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = DEEP_GRAPH + "export { R, cut };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            self.assertIn("next.next => null when reachable", table)
            _, trace = compile_source('import { R, cut } from "./provider.table";' + DEEP_CALL,
                                      root / "consumer.lang")
            self.assertIn("verified postcondition", trace)
            self.assertIn("return {7}", trace)

    def test_extra_prune_condition_cannot_be_summarized_as_an_unconditional_cut(self):
        source = DEEP_GRAPH.replace("a.next != null", "a.next != null && a.x == 1")
        with self.assertRaisesRegex(LanError, "unexpected targets for next"):
            compile_source(source + DEEP_CALL)

    def test_known_null_prefix_keeps_unrelated_links(self):
        source = """
            define f() -> uint8[7,7] {
                define a: R;
                define b: R = { x: 7 };
                define c: R;
                b.next = c;
                define keep_running = true;
                /** @oracle @ensures { state(a.next) := {null}; state(b.next) := {c}; } */
                while (keep_running) { cut(a); keep_running = false; }
                return b.x;
            }
        """
        _, trace = compile_source(DEEP_GRAPH + source)
        self.assertIn("verified postcondition", trace)
