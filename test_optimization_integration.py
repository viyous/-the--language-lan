import json
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from array_domain import ArrayDomain
from lanc import Compiler, LanError, LoweringNode, ModuleGraph, Node, Parser, compile_source, main
from poly_intersection import IntersectionSet, PolynomialIntersection
from semantic_ir import lower_module
from storage_reasoning import KeyObservation, StorageExit, StorageObservation, count_bounds
from table_interface import TableRecord, encode_domain
from value_domain import Clause, Domain, Polynomial, bounded_polynomial_intersection


def branch_program(result="x + y", count=8):
    params = ", ".join(f"b{i}: uint8[0,1]" for i in range(count))
    independent = "\n".join(f"if (b{i} == 1) {{ }} else {{ }}" for i in range(1, count))
    return f"""
        define example({params}) -> uint8[1,1] {{
            define x: uint8 = 0;
            define y: uint8 = 0;
            if (b0 == 1) {{ x = 1; y = 0; }} else {{ x = 0; y = 1; }}
            {independent}
            return {result};
        }}
    """


class OptimizationIntegrationTests(unittest.TestCase):
    def test_module_outputs_are_independent_of_checkout_path(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            products = []
            for name in ("checkout-one", "checkout-two"):
                project = Path(directory) / name
                library = project / "lib"
                library.mkdir(parents=True)
                provider = library / "model.lang"
                provider.write_text("type Counter = { value: uint8 }; export { Counter };", encoding="utf-8")
                entry = project / "app" / "entry.lang"
                entry.parent.mkdir()
                source = '''
                    import { Counter } from "../lib/model.lang";
                    define read(c: Counter) -> uint8 { return c.value; }
                    export { read };
                '''
                compiler = Compiler(source, entry)
                graph = ModuleGraph(entry, compiler, library=True, module_root=project)
                code, table = graph.compile()
                self.assertIn("lib/model.lang", table)
                self.assertNotIn(name, table)
                products.append((code, table))
            self.assertEqual(*products)

    def test_implicit_integer_narrowing_requires_a_range_proof(self):
        compile_source("""
            define byte_value(x: uint8) -> uint8 { return x; }
            define example(x: uint16[0,255]) -> uint8 {
                define y: uint8 = x;
                y = x;
                return byte_value(x);
            }
            define direct(x: uint16[0,255]) -> uint8 { return x; }
        """)
        with self.assertRaisesRegex(LanError, r"candidate entry x=255 gives 256"):
            compile_source("define increment(x: uint8) -> uint8 { x += 1; return x; }")
        with self.assertRaisesRegex(LanError, r"cannot prove"):
            compile_source("define narrow(x: uint16) -> uint8 { return x; }")

    def test_ordinary_tautological_while_rejects_before_body_checks(self):
        with self.assertRaisesRegex(LanError, "while condition is always true"):
            compile_source("""
                define f(x: uint8) -> void {
                    while (x == x) {
                        define a: uint8[2];
                        a[9] = 0;
                    }
                }
            """)
        compile_source("define f() -> void { while (true) { return; } }")

    def test_while_bounds_an_affine_sum_without_an_exit_guard(self):
        source = """
            define sum(n: uint16[0, 1000]) -> uint32[0, 499500] {
                define i: uint16 = 0;
                define total: uint32 = 0;
                while (i < n) {
                    total += i;
                    i += 1;
                }
                return total;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("loop recurrence total: body entry [0, 498501]; header [0, 499500]", trace)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source.replace("total: uint32", "total: uint16"))

    def test_while_accounts_for_continue_and_statement_order(self):
        source = """
            define sum(n: uint16[0, 1000], skip: boolean) -> uint32[0, 500500] {
                define i: uint16 = 0;
                define total: uint32 = 0;
                while (i < n) {
                    i += 1;
                    if (skip) { continue; }
                    total += i;
                }
                return total;
            }
        """
        compile_source(source)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source.replace("i += 1;\n                    if (skip) { continue; }",
                                          "if (skip) { continue; }\n                    i += 1;")
                           .replace("total: uint32", "total: uint16"))

    def test_while_uses_stable_length_and_array_element_bounds(self):
        compile_source("""
            define sum(data: uint8[..] · {len: [0, 256]}) -> uint16[0, 65280] {
                define i: uint16 = 0;
                define total: uint16 = 0;
                while (i < data.len) {
                    total += data[i];
                    i += 1;
                }
                return total;
            }
        """)

    def test_break_is_rejected_in_both_runtime_and_type_code(self):
        for source in (
            "define f() -> void { while (true) { break; } }",
            "type f(n: uint8) { while (true) { break; } return {0}; }",
        ):
            with self.subTest(source=source), self.assertRaisesRegex(LanError, "break was removed"):
                compile_source(source)

    def test_counted_loop_updates_on_continue_and_keeps_bindings_local(self):
        source = """
            define example() -> uint8[46,46] {
                define i: uint8 = 42;
                define turns: uint8 = 3;
                define sum: uint8 = 0;
                interation(i; 1; uint8; turns; 1) {
                    turns = 0;
                    if (i == 2) { continue; }
                    sum += i;
                }
                return sum + i;
            }
            define nested() -> uint8[8,8] {
                define sum: uint8 = 0;
                interation(i; 1; uint8; 2; 1) {
                    interation(i; i; uint8; 2; 1) { sum += i; }
                }
                return sum;
            }
        """
        compiler = Compiler(source, "<input>")
        compiler.compile()
        ir = compiler.semantic_module
        loops = [op for op in ir.operations if op.opcode == "counted_loop"]
        continuation = next(op for op in ir.operations if op.opcode == "continue")
        self.assertEqual(continuation.control_target, loops[0].anchor)
        self.assertEqual(ir.loops[loops[0].anchor].count, loops[0].args[3].anchor)
        with self.assertRaisesRegex(LanError, "unknown variable 'i'"):
            compile_source("define f() -> uint8 { interation(i; 0; uint8; 1; 1) {} return i; }")

    def test_counted_loop_checks_latch_and_large_strided_access(self):
        compile_source("""
            define fill(a: uint8[1000], n: uint16[0,500]) -> void {
                interation(i; 0; uint16; n; 2) { a[i] = 7; }
                interation(i; 1000; uint16; 500; -2) { a[i - 1] = 7; }
            }
            define skip() -> void { interation(i; 255; uint8; 0; 1) {} }
            define leave() -> uint8[255,255] {
                interation(i; 255; uint8; 1; 1) { return i; }
            }
            type Row = { state: uint8[0,2] };
            type Cabinet = { rows: Row[1000] };
            define fill_rows(c: Cabinet, step: uint8[1,2]) -> void {
                interation(i; 0; uint16; 500; step) { c.rows[i].state = 2; }
            }
        """)
        with self.assertRaisesRegex(LanError, r"cannot prove \{256\} fits"):
            compile_source("define f() -> void { interation(i; 255; uint8; 1; 1) { continue; } }")
        with self.assertRaisesRegex(LanError, r"must support \+"):
            compile_source("define f() -> void { interation(i; false; boolean; 0; 1) {} }")

    def test_counted_loop_runs_in_a_type_function(self):
        source = """
            type sum(n: uint8) {
                define total: uint8 = 0;
                interation(i; 1; uint8; n; 1) {
                    if (i == 2) { continue; }
                    total += i;
                }
                return {total};
            }
            define f() -> uint8[4,4] {
                define value: sum(3) in {uint8} 8 = 4;
                return value;
            }
        """
        compile_source(source)
        with self.assertRaisesRegex(LanError, "compile-time value 256 does not fit"):
            compile_source(source.replace("interation(i; 1;", "interation(i; 255;"))

    def test_keyed_record_layout_and_path_sensitive_updates(self):
        source = """
            type enum Gear { Boots: uint16, Chest: uint32, Helmet: uint16 }
            type KeyCount = uint8[0, Gear.key.Helmet];
            define last_key() -> KeyCount { return Gear.key.Helmet; }
            define read(g: Gear, k: Gear.key) -> uint32 {
                if (k == Gear.key.Chest) { return g[k]; }
                return g[k];
            }
            define update(k: Gear.key) -> uint16[42,42] {
                define g: Gear;
                if (k == Gear.key.Chest) { return 42; }
                g[k] = 42;
                return g[k];
            }
        """
        c, _ = compile_source(source)
        self.assertRegex(c, r"keyed_offsets_\d+\[\] = \{offsetof\([^,]+, f_Boots\), "
                            r"offsetof\([^,]+, f_Chest\), offsetof\([^,]+, f_Helmet\)\}")
        self.assertRegex(c, r"keyed_offsets_\d+\[lan_keyed_index_")
        with self.assertRaisesRegex(LanError, "different types"):
            compile_source(source + "define unsafe(g: Gear, k: Gear.key) -> uint32 { return g[k]; }")
        with self.assertRaisesRegex(LanError, "requires a key of enum Gear.key"):
            compile_source(source + """
                type enum Other { Boots: uint16, Chest: uint32, Helmet: uint16 };
                define wrong(g: Gear) -> uint16 { return g[Other.key.Boots]; }
            """)
        with self.assertRaisesRegex(LanError, "immutable"):
            compile_source(source.replace("Boots: uint16,", "Boots: uint16 · {immutable: true},"))
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source + """
                define unsafe_row(k: Gear.key) -> uint16[42,42] {
                    define rows: Gear[2];
                    if (k == Gear.key.Chest) { return 42; }
                    rows[0][k] = 42;
                    return rows[0].Helmet;
                }
            """)
        refined = """
            type enum Limits { Low: uint8[0,3], High: uint8[5,9] };
            define bump(k: Limits.key) -> uint8 & {3, 9} {
                define limits: Limits = { Low: 2, High: 8 };
                limits[k] += 1;
                return limits[k];
            }
        """
        compile_source(refined)
        with self.assertRaisesRegex(LanError, r"cannot prove \{4\} fits"):
            compile_source(refined.replace("Low: 2", "Low: 3"))
        guarded = """
            define nonzero(g: Gear, k: Gear.key) -> uint16[1,65535] {
                if (k == Gear.key.Chest || g[k] == 0) { return 1; }
                return g[k];
            }
        """
        compile_source(source + guarded)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source + guarded.replace("return g[k];", "g.Helmet = 0; return g[k];"))

    def test_keyed_record_interface_tracks_only_candidate_field_writes(self):
        source = """
            type enum Gear { Helmet: uint16, Chest: uint32, Boots: uint16 };
            define set_armor(g: Gear, k: Gear.key) -> void {
                if (k != Gear.key.Chest) { g[k] = 42; }
            }
            export { Gear, set_armor };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            filename = root / "provider.lang"
            compiler = Compiler(source, filename)
            _, table = ModuleGraph(filename, compiler, library=True, module_root=root).compile()
            self.assertEqual({path for _, path, _ in compiler.exports["set_armor"].mutations},
                             {("Helmet",), ("Boots",)})
            (root / "provider.table").write_text(table, encoding="ascii")
            compile_source("""
                import { Gear as Armor, set_armor } from "./provider.table";
                define read(g: Armor, k: Armor.key) -> uint32 {
                    if (k == Armor.key.Chest) { return g[k]; }
                    return g[k];
                }
                define check(k: Armor.key) -> uint32[70000,70000] {
                    define g: Armor = { Chest: 70000 };
                    set_armor(g, k);
                    return g[Armor.key.Chest];
                }
            """, root / "consumer.lang", module_root=root)

    def test_interface_union_summary_keeps_success_and_forwarded_error(self):
        source = """
            enum ReadError: uint8 { Missing = 250, Other = 251 };
            type ReadResult = uint8[0, 100] | ReadError;
            define read(ok: boolean) -> ReadResult {
                if (!ok) { return ReadError.Missing; }
                return 42;
            }
            define relay(ok: boolean) -> ReadResult { return read(ok)?; }
            export { ReadResult, relay };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            compiler = Compiler(source, provider)
            _, table = ModuleGraph(provider, compiler, library=True).compile()
            (root / "provider.table").write_text(table, encoding="utf-8")
            _, trace = compile_source("""
                import { ReadResult, relay } from "./provider.table";
                define plus_one(ok: boolean) -> ReadResult {
                    define value = relay(ok)?;
                    return value + 1;
                }
            """, root / "consumer.lang")
        self.assertIn("propagate error {250}; continue with {42}", trace)
        self.assertIn("return {43}", trace)

    def test_exports_keep_private_and_transitive_side_effects(self):
        from table_interface import TableRecord

        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)

            def compile_library(name, source):
                path = root / f"{name}.lang"
                _, table = ModuleGraph(path, Compiler(source, path), library=True).compile()
                interface = path.with_suffix(".table")
                interface.write_text(table, encoding="utf-8")
                return TableRecord.read(interface).exports

            provider = compile_library("provider", """
                extern contract log_value(n: uint8) -> void writes { external };
                define release(x: uint8) -> void { free(x); }
                define make() {
                    define x: uint8 · {storage: heap} = 7;
                    return x;
                }
                define work(out: uint8 · {pass: alias}) -> void {
                    release(make());
                    out = 9;
                    log_value(out);
                }
                define pure(n: uint8) -> uint8 { return n; }
                export { make, release, work, pure };
            """)
            self.assertEqual(set(provider), {"make", "release", "work", "pure"})
            self.assertTrue(provider["make"]["memory"])
            self.assertTrue(provider["release"]["memory"])
            self.assertEqual(provider["release"]["owning"], [[0, "free"]])
            self.assertFalse(provider["pure"]["memory"])
            self.assertFalse(provider["pure"]["external_effect"])
            relay = compile_library("relay", """
                import { work } from "./provider.table";
                define forward(out: uint8 · {pass: alias}) -> void { work(out); }
                export { forward };
            """)
            self.assertTrue(relay["forward"]["memory"])
            self.assertTrue(relay["forward"]["external_effect"])
            self.assertEqual(relay["forward"]["mutations"], provider["work"]["mutations"])
            compile_source("""
                import { forward } from "./relay.table";
                define check() -> uint8[9, 9] {
                    define out: uint8 = 0;
                    forward(out);
                    return out;
                }
            """, root / "caller.lang")

    def test_second_pass_deduces_scalar_return_cases(self):
        compiler = Compiler("""
            define code(x: uint8[0,9], y: uint8[0,9]) -> uint8[0,1] {
                if (x < 4 && y < 3) { return 1; }
                return 0;
            }
            define changed(x: uint8[0,9], flag: boolean) -> uint8[0,1] {
                if (flag) { x = 9; return 1; }
                x = 1;
                return 0;
            }
        """, "<input>")
        compiler.compile()
        results = {}
        for fact in compiler.source_theorems:
            if fact.kind == "return_case":
                results.setdefault(fact.function, []).append(fact.conclusion.operands)
        x = ("parameter", 0), encode_domain(Domain.of((0, 3)))
        y = ("parameter", 1), encode_domain(Domain.of((0, 2)))
        yes = ("result",), encode_domain(Domain.point(1))
        self.assertIn(((x, y), yes), results["code"])
        self.assertIn(((yes,), x), results["code"])
        self.assertNotIn(((x,), yes), results["code"])
        for premises, target in results["changed"]:
            self.assertTrue(all(operand != ("parameter", 0) for operand, _ in (*premises, target)))

    def test_storage_count_combines_an_odd_cycle_of_proved_inequalities(self):
        contents = ArrayDomain.unknown(4, Domain.of((0, 2)))
        keys = (KeyObservation(Domain.of((0, 3)), Domain.point(2)),) * 5
        edges = ((0, 1), (1, 2), (2, 3), (3, 4))
        def bounds(separated):
            return count_bounds(StorageObservation(0, (StorageExit(contents, keys, separated),)), Domain.point(2))
        self.assertEqual(bounds((*edges, (0, 4))), (3, 4))
        self.assertEqual(bounds(edges), (2, 4))

    def test_storage_count_deduction_uses_exit_facts_without_changing_codegen(self):
        sample = Path(__file__).resolve().parent / "design_review" / "equipment_slot_graph.lang"
        source = sample.read_text(encoding="utf-8")
        compiler = Compiler(source, "<input>")
        code, _ = compiler.compile()
        counts = {fact.function: fact.conclusion.operands[2:]
                  for fact in compiler.source_theorems
                  if fact.kind == "storage_count" and fact.conclusion.operands[:2] == (0, 2)}
        self.assertEqual(counts["replace_with_one"], (1, 1))
        self.assertEqual(counts["enable_two"], (2, 64))
        self.assertNotIn("apply_slot_reply", counts)
        self.assertIn((0, encode_domain(Domain.of((1, 2))), 1, 64),
                      [fact.conclusion.operands for fact in compiler.source_theorems
                       if fact.kind == "storage_set_count" and fact.function == "register_item"])

        unchecked_distinctness = source.replace("if (first != second) {\n        slot[first] = 2;",
                                                "if (true) {\n        slot[first] = 2;")
        compiler = Compiler(unchecked_distinctness, "<input>")
        compiler.compile()
        self.assertIn((0, 2, 1, 64), [fact.conclusion.operands for fact in compiler.source_theorems
                                     if fact.kind == "storage_count" and fact.function == "enable_two"])
        with patch.object(Compiler, "derive_storage_theorems"):
            self.assertEqual(compile_source(source)[0], code)

    def test_keyed_slice_facts_follow_backing_storage_and_loop_edges(self):
        sample = Path(__file__).resolve().parent / "design_review" / "equipment_slot_graph.lang"
        source = sample.read_text(encoding="utf-8")
        _, trace = compile_source(source)
        self.assertIn("at a proved disjoint offset", trace)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source.replace("if (first != second) { return 2; }", ""))
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source.replace("if (key != watched) { slot[key] = 1; }", "slot[key] = 1;"))

    def test_composite_storage_counts_follow_aliases_and_all_exits(self):
        from io import StringIO

        sample = Path(__file__).resolve().parent / "design_review" / "equipment_slot_graph.lang"
        source = sample.read_text(encoding="utf-8")
        route = ("rows", None, "badge", "state")
        for program, lower in (
            (source, 0),
            (source.replace("if (clear_selected) { selected.badge.state = 0; }", ""), 1),
        ):
            compiler = Compiler(program, sample)
            project = ModuleGraph(sample, compiler)
            project.compile()
            counts = {(fact.function, fact.conclusion.operands[0]): fact
                      for fact in compiler.source_theorems
                      if fact.kind == "storage_count" and fact.conclusion.operands[1] == 2}
            page = counts["replace_page_selection", (0, ("slots",))]
            self.assertEqual(page.conclusion.operands[2:], (1, 1))
            badge = counts["select_badge", (0, route)]
            self.assertEqual(badge.conclusion.operands[2:], (lower, 63))
            theorem, proof = StringIO(), StringIO()
            project.write_analysis_artifacts(theorem, proof)
            evidence = next(item for item in map(json.loads, proof.getvalue().splitlines())
                            if item.get("kind") == "storage_observation" and item["id"] == badge.evidence)
            self.assertEqual(evidence["parameter"], 0)
            self.assertEqual(evidence["path"], list(route))

    def test_storage_counts_across_table_and_disjoint_slot_writes(self):
        sample = Path(__file__).resolve().parent / "design_review" / "equipment_slot_graph.lang"
        source = sample.read_text(encoding="utf-8") + """
            define write_slot(slot: SlotView) -> void { slot[2] = 2; }
            export { SlotView, SlotKey, Equipment, EquipmentPage, Cabinet, replace_with_one,
                     replace_page_selection, select_badge, write_slot };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "equipment.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider), library=True).compile()
            interface = provider.with_suffix(".table")
            interface.write_text(table, encoding="utf-8")
            self.assertIn([0, [], 0, 64, encode_domain(Domain.point(2)), 1, 1],
                          TableRecord.read(interface).exports["replace_with_one"]["storage_counts"])
            imports = """
                import { SlotKey, Equipment, EquipmentPage, Cabinet, replace_page_selection,
                         select_badge, write_slot } from "./equipment.table";
                import { relay } from "./relay.table";
            """
            relay = root / "relay.lang"
            forward = """
                import { Equipment, replace_with_one } from "./equipment.table";
                define relay(eq: Equipment, flag: boolean) -> void {
                    replace_with_one(eq[64,128), flag);
                    eq[1] = 2;
                }
                export { relay };
            """
            _, relayed = ModuleGraph(relay, Compiler(forward, relay), library=True).compile()
            relay.with_suffix(".table").write_text(relayed, encoding="utf-8")
            caller = root / "caller.lang"
            direct = """
                import { replace_with_one } from "./equipment.table";
                define enabled(eq: Equipment, first: SlotKey, second: SlotKey, flag: boolean) -> uint8[1,1] {
                    if (first == second) { return 1; }
                    define selected = eq[first * 64, first * 64 + 64);
                    define other = eq[second * 64, second * 64 + 64);
                    replace_with_one(selected, flag);
                    write_slot(other);
                    return selected.count({2});
                }
            """
            code, _ = compile_source(imports + direct + """
                define from_relay(eq: Equipment, flag: boolean) -> uint8[1,1] {
                    relay(eq, flag);
                    return eq[64,128).count({2});
                }
                define from_page(page: EquipmentPage, flag: boolean) -> uint8[1,1] {
                    replace_page_selection(page, flag);
                    page.version = 5;
                    return page.slots.count({2});
                }
                define from_cabinet(cabinet: Cabinet, flag: boolean, clear: boolean) -> uint8[0,63] {
                    select_badge(cabinet, flag, clear);
                    cabinet.rows[0].badge.serial = 5;
                    return cabinet.rows.count({2}, badge.state);
                }
            """, caller)
            self.assertIn(".f_badge.f_state", code)
            for disjoint_write in ("other[0,) = 2;", "other[0,) = selected as: copy;"):
                compile_source(imports + direct.replace("write_slot(other);", disjoint_write), caller)
            for overlap in (direct.replace("write_slot(other);", "write_slot(selected);"),
                            direct.replace("write_slot(other);", "selected[0,) = 2;"),
                            direct.replace("if (first == second) { return 1; }", "")):
                with self.assertRaisesRegex(LanError, "cannot prove"):
                    compile_source(imports + overlap, caller)

            # Invalid windows must be rejected before a call can use them.
            record = TableRecord.read(interface)
            record.exports["replace_with_one"]["storage_counts"][0][2] = 1
            interface.write_text(record.encode(), encoding="utf-8")
            with self.assertRaisesRegex(LanError, "invalid storage count region or bounds"):
                compile_source(imports + direct, caller)

    def test_storage_count_at_loop_join_and_alias_write(self):
        setup = """
            type Cell = uint8[0,2];
            type Slot = Cell[..] · {len: {8}};
            type Storage = Cell[..] · {len: {80}};
            define init(slot: Slot, flag: boolean) -> void {
                slot[0,) = 0;
                if (flag) { slot[0] = 2; }
                else { slot[1] = 2; }
            }
        """
        loop = """
            define check(slots: Storage, first: uint8[0,9], second: uint8[0,9],
                         flag: boolean, n: uint8[0,7]) -> uint8[1,1] {
                if (first == second) { return 1; }
                define selected = slots[first * 8, first * 8 + 8);
                define other = slots[second * 8, second * 8 + 8);
                init(selected, flag);
                define i: uint8 = 0;
                while (i < n) { other[i] = 2; i += 1; }
                return selected.count({2});
            }
        """
        compile_source(setup + loop)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(setup + loop.replace("other[i] = 2", "selected[i] = 2"))
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(setup + """
                define check(first: Slot, other: Slot, flag: boolean) -> uint8[1,1] {
                    init(first, flag);
                    other[3] = 2;
                    return first.count({2});
                }
            """)

    def test_return_cases_cross_table_and_guard_only_unchanged_entry_values(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)

            def library(name, source):
                path = root / f"{name}.lang"
                _, table = ModuleGraph(path, Compiler(source, path), library=True).compile()
                path.with_suffix(".table").write_text(table, encoding="utf-8")

            library("rules", """
                define classify(x: uint8[0,9], y: uint8[0,9]) -> uint8[0,1] {
                    if (x < 4 && y < 3) { return 1; }
                    return 0;
                }
                export { classify };
            """)
            library("forward", """
                import { classify } from "./rules.table";
                define forwarded(x: uint8[0,9], y: uint8[0,9]) -> uint8[0,1] { return classify(x, y); }
                export { forwarded };
            """)
            caller = root / "caller.lang"
            program = """
                import { forwarded as classify } from "./forward.table";
                define known() -> uint8[1,1] { return classify(3, 2); }
                define inverse(x: uint8[0,9], y: uint8[0,9]) -> uint8[0,5] {
                    define result = classify(x, y);
                    if (result in {1}) { return x + y; }
                    return 0;
                }
                define inverse_direct(x: uint8[0,9], y: uint8[0,9]) -> uint8[0,5] {
                    if (classify(x, y) > 0) { return x + y; }
                    return 0;
                }
            """
            compile_source(program, caller)
            for reassigned in ("x = 9;", "set_nine(x);"):
                with self.assertRaisesRegex(LanError, "cannot prove"):
                    compile_source(program.replace("if (result in {1})", reassigned + " if (result in {1})") + """
                        define set_nine(n: uint8[0,9] · {pass: alias}) -> void { n = 9; }
                    """, caller)

    def test_keyed_slice_fact_is_invalidated_by_a_potential_alias(self):
        source = """
            define f(a: uint8[..] · {len: {16}}, b: uint8[..] · {len: {16}},
                     key: uint8[0, 15]) -> uint8[1, 1] {
                a[key] = 1;
                b[key] = 2;
                return a[key];
            }
        """
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(source)

    def test_module_lowering_preserves_source_programs_and_oracle_operands(self):
        def shape(value):
            if isinstance(value, Node):
                return value.kind, value.token, shape(value.args), value.source_text, shape(value.state_guard)
            if isinstance(value, (list, tuple)):
                return type(value), tuple(map(shape, value))
            return value

        suite = Path(__file__).resolve().parent / "test_suite"
        cases = ("08_bitwindows", "10_slices", "12_interfaces", "16_recursion", "18_ownership",
                 "19_chains", "22_modules", "23_oracles", "24_graphs", "27_comprehensive")
        sources = [suite / case / f"{case}.lang" for case in cases]
        for path in sources:
            with self.subTest(source=path.name):
                parsed = Parser(path.read_text(encoding="utf-8-sig")).module()
                ir = lower_module(parsed, Node)
                lowered = ir.lowering_body(LoweringNode.from_operation)
                self.assertEqual(shape(lowered), shape(parsed))
                self.assertEqual([operation.anchor for operation in ir.operations], list(range(len(ir.operations))))

    def test_complete_lowering_records_checked_storage_and_control_targets(self):
        source = """
            type Item = { value: uint8 };
            define read(n: uint8) -> uint8 {
                define owned: Item · {storage: heap} = { value: 7 };
                define moved: Item · {storage: heap} = owned;
                define count: uint8 = n;
                while (count > 0) {
                    count -= 1;
                    continue;
                }
                define value = moved.value;
                free(moved);
                return value;
            }
        """
        compiler = Compiler(source, "<test>")
        _, trace = compiler.compile()
        self.assertIn("return {7}", trace)
        ir = compiler.semantic_module
        operations = ir.operations
        fn = next(operation for operation in operations if operation.kind == "function")
        loop = next(operation for operation in operations if operation.opcode == "loop")
        self.assertEqual({op.control_target for op in operations if op.opcode == "continue"}, {loop.anchor})
        self.assertEqual(next(op for op in operations if op.opcode == "return").control_target, fn.anchor)
        release = next(op for op in operations if op.opcode == "release")
        owned, moved = [op.binding for op in operations if op.opcode == "bind_owner"]
        self.assertEqual(release.args[1][0].binding, moved)
        self.assertEqual(release.args[1][0].opcode, "address_binding")
        self.assertEqual(ir.storage[owned][0].ownership, "allocate")
        self.assertEqual(ir.storage[moved][0].ownership, "move")
        self.assertEqual(ir.contracts["read"].result.bits, 8)

    def test_polynomial_bound_reaches_return_checker(self):
        program = ("define bounded(x: uint32[0,100000] & poly(n*n) & poly(2*n*n+1)) "
                   "-> uint32[1,9801] { return x; }")
        _, table = compile_source(program)
        self.assertIn("return [1, 9801] & poly", table)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(program.replace("9801", "9800"))

    def test_polynomial_integer_only_witnesses_are_removed(self):
        p = Polynomial((0, 3, 2))
        q = Polynomial((0, 0, 1))
        integer = PolynomialIntersection(p.coefficients, q.coefficients, 0, 100000).solve_set()
        self.assertIn(11025, integer.values)
        domain = Domain.from_clauses([Clause(0, 100000, (p, q))])
        self.assertEqual(domain.hi, 324)
        self.assertTrue(domain & Domain.point(324))
        self.assertFalse(domain & Domain.point(9))

    def test_polynomial_unknown_does_not_prove_an_intersection(self):
        program = ("define unknown(x: uint32[0,100000] & poly(n*n) & poly(2*n*n+1)) "
                   "-> uint32[1,9801] { return x; }")
        bounded_polynomial_intersection.cache_clear()
        try:
            with patch.object(PolynomialIntersection, "solve_set",
                              return_value=IntersectionSet("unknown", "budget", 0)):
                with self.assertRaisesRegex(LanError, "cannot prove|cannot decide"):
                    compile_source(program)
        finally:
            bounded_polynomial_intersection.cache_clear()

    @unittest.skipUnless(find_spec("z3"), "optional Z3 backend unavailable")
    def test_ssa_compacts_at_path_budget_and_checks_return(self):
        _, table = compile_source(branch_program())
        self.assertIn("path budget", table)
        self.assertIn("ssa return proved", table)
        _, small_table = compile_source(branch_program(count=7))
        self.assertNotIn("path budget", small_table)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(branch_program("x + y + 1"))

    @unittest.skipUnless(find_spec("z3"), "optional Z3 backend unavailable")
    def test_ssa_many_branches_stay_within_analysis_budget(self):
        _, table = compile_source(branch_program(count=24))
        self.assertIn("ssa return proved", table)

    @unittest.skipUnless(find_spec("z3"), "optional Z3 backend unavailable")
    def test_cli_keeps_interfaces_theorems_and_proofs_separate(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source = root / "provider.lang"
            source.write_text(branch_program() + "\nexport { example };\n", encoding="utf-8")
            self.assertEqual(main([str(source), "--library"]), 0)
            table = source.with_suffix(".table").read_text(encoding="ascii")
            self.assertEqual(len(table.splitlines()), 1)
            self.assertNotIn("ssa return proved", table)

            def records(suffix):
                return [json.loads(line) for line in source.with_suffix(suffix).read_text(encoding="ascii").splitlines()]

            theorem_header, *facts = records(".theorem")
            proof_header, *evidence = records(".proof")
            self.assertEqual(theorem_header["format"], "lan.theorem")
            self.assertEqual(proof_header["format"], "lan.proof")
            fact = next(fact for fact in facts if fact["kind"] == "returns")
            self.assertEqual(fact["intervals"], [[1, 1]])
            self.assertIsInstance(fact["anchor"], int)
            proof = next(item for item in evidence if item["kind"] == "ssa_obligation"
                         and item["id"] == fact["evidence"] and item["function"] == fact["function"])
            self.assertIn(proof["goal"], {node[0] for node in proof["nodes"]})
            self.assertIn(proof["guard"], {node[0] for node in proof["nodes"]})
            self.assertEqual(proof["source"], fact["source"])
            self.assertEqual(main([str(source), "--library"]), 0)
            self.assertEqual(records(".theorem"), [theorem_header, *facts])

            for suffix in (".theorem", ".proof", ".lang"):
                source.with_suffix(suffix).unlink()
            compile_source('import { example } from "./provider.table";'
                           'define use() -> uint8[1,1] { return example(0,0,0,0,0,0,0,0); }',
                           root / "consumer.lang")

    @unittest.skipUnless(find_spec("z3"), "optional Z3 backend unavailable")
    def test_ssa_discharges_storage_index_and_call_obligations(self):
        source = branch_program().replace("-> uint8[1,1]", "-> uint8")
        cases = (
            ("initializer", "", "define a: uint8[1,1] = VALUE; return a;"),
            ("assignment", "", "define a: uint8[1,1] = 1; a = VALUE; return a;"),
            ("index", "", "define data: uint8[2] = [0, 1]; return data[VALUE];"),
            ("argument", "define take(v: uint8[1,1]) -> uint8 { return v; }",
             "return take(VALUE);"),
            ("dependent argument", "define take(a: uint8, b: uint8 & <= a) -> uint8 { return b; }",
             "return take(1, VALUE);"),
        )
        for label, helper, result in cases:
            with self.subTest(check=label):
                template = helper + source.replace("return x + y;", result)
                _, trace = compile_source(template.replace("VALUE", "x + y"))
                self.assertIn(f"ssa {label} proved", trace)
                with self.assertRaisesRegex(LanError, "cannot prove"):
                    compile_source(template.replace("VALUE", "x + y + 1"))

    def test_loop_exit_fact_does_not_require_a_termination_proof(self):
        program = ("define countdown(n: uint8) -> uint8 { "
                   "while (n > 0) { if (n > 3) { n -= 1; continue; } n -= 1; } return n; }")
        for source in (program, program.replace("n -= 1; continue;", "continue;")):
            with self.subTest(source=source):
                compiler = Compiler(source, "<loop>")
                _, trace = compiler.compile()
                self.assertIn("return {0}", trace)
                self.assertTrue(any(fact.conclusion.relation == "condition_false_at_exit"
                                    for fact in compiler.source_theorems))
