import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, InterfaceError, LanError, ModuleGraph, compile_source, decode_mutations


GRAPH = """
    type R = { payload: uint8, next: R | null, alt: R | null };
    define cut(a: R) -> void {
        if (a.next != null) { a.next.alt = null; }
        a.next = null;
        if (a.next != null) { a.next.next = null; }
    }
    define wrapper(a: R) -> void { cut(a); }
"""

CALL = """
    define use() -> uint8[9,9] {
        define a: R;
        define b: R = { payload: 9 };
        define c: R;
        a.next = b;
        a.alt = c;
        b.next = c;
        b.alt = c;
        define keep_running = true;
        /** @oracle @ensures {
            state(a.next) := {null};
            state(a.alt) := {c};
            state(b.next) := {c};
            state(b.alt) := {null};
        } */
        while (keep_running) { wrapper(a); keep_running = false; }
        return b.payload;
    }
"""

ATTACH = """
    type R = { payload: uint8, next: R | null, alt: R | null };
    define connect(a: R,
                   donor: R) -> void {
        if (a.next != null) { a.next.alt = donor; }
    }
"""

ATTACH_CALL = """
    define use() -> uint8[9,9] {
        define a: R;
        define b: R;
        define d: R = { payload: 9 };
        a.next = b;
        define keep_running = true;
        /** @oracle @ensures {
            state(a.next) := {b};
            state(a.alt) := {null};
            state(b.alt) := {d};
            state(b.next) := {null};
        } */
        while (keep_running) { connect(a, d); keep_running = false; }
        return b.alt.payload;
    }
"""


EDIT = """
    type R = { payload: uint8, next: R | null, alt: R | null };
    define edit(a: R,
                donor: R) -> void {
        if (a.next != null) { a.next.alt = donor; }
        a.next = null;
        if (a.next != null) { a.next.next = donor; }
        a.alt = donor;
    }
    define relay(donor: R,
                 a: R) -> void {
        edit(a, donor);
    }
"""

EDIT_CALL = """
    define use() -> uint8[9,9] {
        define a: R;
        define b: R;
        define c: R;
        define d: R = { payload: 9 };
        a.next = b;
        b.next = c;
        define keep_running = true;
        /** @oracle @ensures {
            state(a.next) := {null};
            state(a.alt) := {d};
            state(b.next) := {c};
            state(b.alt) := {d};
        } */
        while (keep_running) { relay(d, a); keep_running = false; }
        return b.alt.payload;
    }
"""


GRAPH_READER = """
    type R = { payload: uint8, next: R | null, alt: R | null };
    define peek(node: R) -> uint8 {
        if (node.next != null) { return node.next.payload; }
        return node.payload;
    }
"""


class GraphForwardingInferenceTests(unittest.TestCase):
    def test_forwards_to_a_readonly_graph_declared_later(self):
        c, table = compile_source("""
            define wrapper(root: R) -> uint8 { return peek(root); }
        """ + GRAPH_READER)
        self.assertIn("inferred graph parameter wrapper.root", table)
        self.assertRegex(c, r"uint8_t lan_fn_wrapper\(const lan_record_\w+ \*")
        self.assertIn("verify live graph nodes", table)

    def test_inferred_wrapper_checks_live_graph_at_the_call_site(self):
        source = GRAPH_READER + """
            define wrapper(root: R) -> uint8 { return peek(root); }
            define caller() -> uint8 {
                define root: R;
                define target: R · {storage: heap};
                root.alt = target;
                free(target);
                return wrapper(root);
            }
        """
        with self.assertRaisesRegex(LanError, "may be a Tombstone"):
            compile_source(source)

    def test_mutable_callee_infers_a_writable_graph_and_cut_summary(self):
        source = GRAPH
        source += "define implicit(root: R) -> void { wrapper(root); }"
        _, trace = compile_source(source + CALL.replace("wrapper(a)", "implicit(a)"))
        self.assertIn("inferred graph parameter implicit.root", trace)

    def test_inferred_attachment_with_separate_scalar_borrow(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define attach(root: R, donor: R, out: uint8 · {pass:alias}) -> void {
                root.next = null;
                root.alt = donor;
                out = 1;
            }
            define use() -> void {
                define root: R;
                define donor: R;
                define count: uint8 = 0;
                attach(root, donor, count);
            }
        """
        _, trace = compile_source(source)
        self.assertIn("inferred graph parameter attach.root", trace)
        with self.assertRaisesRegex(LanError, "scalar borrow is disjoint"):
            compile_source(source.replace("attach(root, donor, count)", "attach(root, donor, donor.payload)"))

    def test_one_actual_parameter_cannot_mix_graph_and_chain_requirements(self):
        with self.assertRaisesRegex(LanError, "cannot infer a lifetime contract"):
            compile_source("""
                type R = { payload: uint8, next: R | null };
                define peek(a: R, b: R) -> uint8 {
                    assert chain(head: b -> tail: null);
                    return a.payload;
                }
                define wrapper(root: R) -> uint8 { return peek(root, root); }
            """)

    def test_a_qualified_linked_parameter_stays_out_of_inference(self):
        # Borrowing and shape are derived from the body, and only a bare
        # declaration takes part in that pass, so the hint names the fix.
        with self.assertRaisesRegex(LanError, "leave the type bare"):
            compile_source("""
                type R = { payload: uint8, next: R | null };
                define peek(a: R · {pass: alias}) -> uint8 { return a.payload; }
            """)

    def test_a_native_cannot_take_a_linked_record(self):
        # A native has no body, so there is nothing to infer a shape from.
        with self.assertRaisesRegex(LanError, "a native function cannot take a linked record"):
            compile_source("""
                type R = { payload: uint8, next: R | null };
                extern contract c_peek(a: R) -> uint8 writes {};
            """)

    def test_an_immutable_linked_parameter_keeps_taking_part_in_inference(self):
        # `pass` decides borrow versus copy, which is exactly what the pass
        # derives, so declaring it conflicts; `immutable` only forces a read.
        _, trace = compile_source("""
            type R = { payload: uint8, next: R | null };
            define peek(a: R · {immutable: true}) -> uint8 { return a.payload; }
        """)
        self.assertIn("inferred graph parameter peek.a", trace)

    def test_graph_contract_is_inferred_from_and_preserved_in_a_table(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = GRAPH_READER + "export { R, peek };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")

            relay = root / "relay.lang"
            source = """
                import { R, peek } from "./provider.table";
                define wrapper(root: R) -> uint8 { return peek(root); }
                export { R, wrapper };
            """
            _, table = ModuleGraph(relay, Compiler(source, relay.name), library=True).compile()
            self.assertIn("inferred graph parameter wrapper.root", table)
            (root / "relay.table").write_text(table, encoding="ascii")
            c, trace = compile_source("""
                import { R, wrapper } from "./relay.table";
                define run() -> uint8 {
                    define root: R = { payload: 9 };
                    return wrapper(root);
                }
            """, root / "consumer.lang")
            self.assertRegex(c, r"extern uint8_t lan_api_\w+_wrapper\(const lan_record_\w+ \*\)")
            self.assertIn("verify live graph nodes", trace)


class GraphMutationProgramTests(unittest.TestCase):
    def test_ordered_cuts_and_forwarding_across_table(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = GRAPH + "export { R, wrapper };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            _, trace = compile_source('import { R, wrapper } from "./provider.table";' + CALL,
                                      root / "consumer.lang")
            self.assertIn("verified postcondition", trace)

    def test_unknown_extra_guard_changes_only_the_named_edge(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define cut(a: R, flag: boolean) -> void {
                if (a.next != null && flag) { a.next.alt = null; }
            }
            define use(flag: boolean) -> void {
                define a: R;
                define b: R;
                define c: R;
                a.next = b;
                b.alt = c;
                define keep_running = true;
                /** @oracle @ensures { state(a.next) := {b}; state(b.alt) := {c, null}; } */
                while (keep_running) { cut(a, flag); keep_running = false; }
            }
        """
        _, trace = compile_source(source)
        self.assertIn("next.alt => possibly null", trace)
        with self.assertRaisesRegex(LanError, "unexpected targets"):
            compile_source(source.replace(":= {c, null}", ":= {null}"))

    def test_repeated_null_only_effect_keeps_unrelated_edges(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define cut(a: R, flag: boolean) -> void {
                define searching = true;
                while (searching && flag && a.next != null) { a.next.alt = null; searching = false; }
            }
            define use(flag: boolean) -> void {
                define a: R;
                define b: R;
                define c: R;
                a.next = b;
                b.alt = c;
                define keep_running = true;
                /** @oracle @ensures { state(a.next) := {b}; state(b.alt) := {c, null}; } */
                while (keep_running) { cut(a, flag); keep_running = false; }
            }
        """
        _, trace = compile_source(source)
        self.assertIn("next.alt => possibly null", trace)

    def test_a_payload_write_keeps_the_ordered_program_precise(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define cut(a: R) -> void {
                a.payload = 3;
                a.next = null;
            }
            define use() -> void {
                define a: R;
                define b: R;
                a.next = b;
                a.alt = b;
                define keep_running = true;
                /** @oracle @ensures { state(a.next) := {null}; state(a.alt) := {b}; } */
                while (keep_running) { cut(a); keep_running = false; }
            }
        """
        _, trace = compile_source(source)
        # Only `next` is cut: `alt` stays put, so the scalar write must not
        # degrade the whole summary into "any edge may be null".
        self.assertIn("param#0.next => null", trace)

    def test_a_local_binding_before_the_cut_keeps_the_program_precise(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define cut(a: R) -> void {
                define seen = a.payload;
                a.next = null;
            }
            define use() -> void {
                define a: R;
                define b: R;
                a.next = b;
                a.alt = b;
                define keep_running = true;
                /** @oracle @ensures { state(a.next) := {null}; state(a.alt) := {b}; } */
                while (keep_running) { cut(a); keep_running = false; }
            }
        """
        _, trace = compile_source(source)
        self.assertIn("param#0.next => null", trace)

    def test_payload_havoc_may_accompany_an_ordered_program(self):
        compiler = Compiler(GRAPH, "provider.lang")
        compiler.compile()
        contract = compiler.functions["cut"][1]
        decode_mutations([[0, ["@graph"], None],
                          [0, ["@graph_path_prune", "0", "must", "next"], None]], contract)
        with self.assertRaisesRegex(InterfaceError, "cannot be combined"):
            decode_mutations([[0, ["@prune"], None],
                              [0, ["@graph_path_prune", "0", "must", "next"], None]], contract)

    def test_interface_checks_order_and_route_field_types(self):
        compiler = Compiler(GRAPH, "provider.lang")
        compiler.compile()
        contract = compiler.functions["cut"][1]
        for names in (
            ["@graph_path_prune", "1", "must", "next"],
            ["@graph_path_prune", "0", "must", "next", "payload"],
        ):
            with self.subTest(names=names), self.assertRaises(InterfaceError):
                decode_mutations([[0, names, None]], contract)

    def test_guarded_donor_attachment_survives_table_import(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = ATTACH + "export { R, connect };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            self.assertIn("next.alt => attach param#1", table)
            _, trace = compile_source('import { R, connect } from "./provider.table";' + ATTACH_CALL,
                                      root / "consumer.lang")
            self.assertIn("verified postcondition", trace)
        with self.assertRaisesRegex(LanError, "attachment requires disjoint graphs"):
            compile_source(ATTACH + ATTACH_CALL.replace("a.next = b;", "a.next = b; a.alt = d;"))

    def test_multiple_attaches_keep_their_routes_across_table_import(self):
        source = ATTACH.replace("a.next.alt = donor;", "a.next.alt = donor; a.next.next = donor;")
        caller = ATTACH_CALL.replace("state(b.next) := {null}", "state(b.next) := {d}")
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source += "export { R, connect };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn('["@graph_path_edit","0","attach","must","next","alt"]', table)
            self.assertIn('["@graph_path_edit","1","attach","must","next","next"]', table)
            (root / "provider.table").write_text(table, encoding="ascii")
            _, trace = compile_source('import { R, connect } from "./provider.table";' + caller,
                                      root / "consumer.lang")
            self.assertIn("verified postcondition", trace)

    def test_mixed_graph_edits_resolve_routes_in_order_through_a_forwarder(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = EDIT + "export { R, relay };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn('[1,["@graph_path_edit","0","attach","must","next","alt"],null]', table)
            self.assertIn('[1,["@graph_path_edit","1","cut","must","next"],null]', table)
            self.assertIn('[1,["@graph_path_edit","2","attach","must","next","next"],null]', table)
            (root / "provider.table").write_text(table, encoding="ascii")
            _, trace = compile_source('import { R, relay } from "./provider.table";' + EDIT_CALL,
                                      root / "consumer.lang")
            self.assertIn("verified postcondition", trace)

    def test_possible_attachment_does_not_change_the_donor_on_a_later_cut(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define edit(a: R,
                        donor: R, flag: boolean) {
                if (flag) { a.next = donor; }
                if (!flag && a.next != null) { a.next.alt = null; }
            }
            define use(flag: boolean) {
                define a: R;
                define b: R;
                define c: R;
                define d: R;
                define e: R;
                a.next = b;
                b.alt = c;
                d.alt = e;
                define keep_running = true;
                /** @oracle @ensures {
                    state(a.next) := {b, d};
                    state(b.alt) := {c, null};
                    state(d.alt) := {e};
                } */
                while (keep_running) { edit(a, d, flag); keep_running = false; }
            }
        """
        _, trace = compile_source(source)
        self.assertIn("verified postcondition", trace)
        with self.assertRaisesRegex(LanError, "read-only attachment source"):
            compile_source(source.replace("if (flag) { a.next = donor; }", "a.next = donor;")
                           .replace("if (!flag && a.next != null) { a.next.alt = null; }", "a.next.alt = null;"))

    def test_ordered_edit_interface_rejects_invalid_effects(self):
        compiler = Compiler(EDIT + ATTACH[ATTACH.index("    define connect"):], "provider.lang")
        compiler.compile()
        for name, effects in (
            ("edit", [["@graph_path_edit", "1", "attach", "must", "next"]]),
            ("edit", [["@graph_path_edit", "0", "attach", "must", "payload"]]),
            ("edit", [["@graph_path_edit", "0", "replace", "must", "next"]]),
            ("edit", [["@graph_path_edit", "0", "cut", "may", "next"], ["@attach_graph"]]),
            ("connect", [["@graph_path_edit", "0", "cut", "must", "next"]]),
        ):
            with self.subTest(name=name, effects=effects), self.assertRaises(InterfaceError):
                decode_mutations([[0, effect, None] for effect in effects], compiler.functions[name][1])

    def test_payload_write_keeps_an_attachment_contract(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define paint(a: R,
                         donor: R) -> void {
                a.payload = 3;
                if (a.next != null) { a.next.alt = donor; }
                a.next = null;
            }
            export { R, paint };
        """
        caller = """
            define use() -> uint8 {
                define a: R;
                define b: R;
                define c: R;
                define d: R = { payload: 9 };
                a.next = b;
                b.next = c;
                define keep_running = true;
                /** @oracle @ensures {
                    state(a.next) := {null};
                    state(b.alt) := {d};
                    state(b.next) := {c};
                } */
                while (keep_running) { paint(a, d); keep_running = false; }
                return b.alt.payload;
            }
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            # A payload write widens the scalars of the reachable set and says
            # nothing about the topology, so the ordered edits stay exact.
            self.assertIn('[0,["@graph"],null]', table)
            self.assertIn('["@graph_path_edit","0","attach","must","next","alt"]', table)
            self.assertIn('["@graph_path_edit","1","cut","must","next"]', table)
            self.assertNotIn("@attach_graph", table)
            (root / "provider.table").write_text(table, encoding="ascii")
            _, trace = compile_source('import { R, paint } from "./provider.table";' + caller,
                                      root / "consumer.lang")
            self.assertIn("verified postcondition", trace)

    def test_payload_write_on_a_reachable_node_keeps_the_attachment(self):
        source = """
            type R = { payload: uint8, next: R | null, alt: R | null };
            define paint(a: R,
                         donor: R) -> void {
                if (a.next != null) { a.next.payload = 7; }
                if (a.next != null) { a.next.alt = donor; }
            }
            export { R, paint };
        """
        caller = """
            define use() -> uint8 {
                define a: R;
                define b: R;
                define d: R = { payload: 9 };
                a.next = b;
                define keep_running = true;
                /** @oracle @ensures {
                    state(b.alt) := {d};
                    state(a.next) := {b};
                } */
                while (keep_running) { paint(a, d); keep_running = false; }
                return b.alt.payload;
            }
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn('[0,["@graph"],null]', table)
            self.assertIn('["@graph_path_attach","must","next","alt"]', table)
            (root / "provider.table").write_text(table, encoding="ascii")
            _, trace = compile_source('import { R, paint } from "./provider.table";' + caller,
                                      root / "consumer.lang")
            self.assertIn("verified postcondition", trace)

    def test_chain_payload_write_keeps_the_attachment(self):
        source = """
            type Node = { payload: uint8, next: Node | null };
            define paint(h: Node,
                         donor: Node) -> void {
                h.payload = 3;
                h.next = donor;
            }
            export { Node, paint };
        """
        caller = """
            define use() -> uint8 {
                define h: Node · {storage: heap} = { payload: 1 };
                define d: Node = { payload: 9 };
                paint(h, d);
                define v = h.payload;
                free(h);
                return v;
            }
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn('[0,["@graph_path_attach","must","next"],null]', table)
            self.assertIn('[0,["@graph"],null]', table)
            (root / "provider.table").write_text(table, encoding="ascii")
            compile_source('import { Node, paint } from "./provider.table";' + caller,
                           root / "consumer.lang")

    def test_attachment_still_rejects_a_link_write_to_the_chain_head(self):
        source = """
            type Node = { payload: uint8, next: Node | null };
            define paint(h: Node,
                         donor: Node) -> void {
                if (h.next != null) { h.next.payload = 3; }
                h.next = donor;
            }
        """
        compiler = Compiler(source, "provider.lang")
        compiler.compile()
        # The link itself is the attachment, so a summary that claims both a
        # coarse prune and the attachment stays contradictory.
        contract = compiler.functions["paint"][1]
        with self.assertRaises(InterfaceError):
            decode_mutations([[0, ["@prune"], None], [0, ["@attach"], None]], contract)
