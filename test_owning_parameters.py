import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


SCALAR = """
    define drop: contract(x: uint8 · {storage: heap}) -> uint8 (x) {
        define value = x;
        free(x);
        return value;
    }
    define caller: contract() -> uint8 () {
        define owner: uint8 · {storage: heap} = 7;
        return drop(owner);
    }
"""

RECORD = """
    type Pair = { left: uint8, right: uint16 };
    define drop: contract(pair: Pair · {storage: heap}) -> uint8 (pair) {
        define value = pair.left;
        free(pair);
        return value;
    }
    define caller: contract() -> uint8 () {
        define owner: Pair · {storage: heap} = { left: 3, right: 4 };
        return drop(owner);
    }
"""

ARRAY = """
    define drop: contract(items: uint8[..] · {len: {4}} · {storage: heap}) -> uint8 (items) {
        define value = items[1];
        free(items);
        return value;
    }
    define caller: contract() -> uint8 () {
        define owner: uint8[4] · {storage: heap};
        owner[1] = 9;
        return drop(owner);
    }
"""

FREED_IN_BODY = "free(x);"

LINKED_NODE = """
    type Node = { value: uint8, next: Node · {pass: alias} | null };
    define make(n: uint8) -> Node · {storage: heap} {
        if (n != 0) { return make(n - 1); }
        define owner: Node · {storage: heap} = { value: 7 };
        return owner;
    }
    define relay(owner: Node · {storage: heap}, n: uint8) -> Node · {storage: heap} {
        if (n != 0) { return relay(owner, n - 1); }
        return owner;
    }
    define drop(owner: Node · {storage: heap}) -> uint8 {
        define value = owner.value;
        free(owner);
        return value;
    }
"""


class OwningLinkedRecordTests(unittest.TestCase):
    def test_recursion_transfers_a_node_with_a_null_next(self):
        c, table = compile_source(LINKED_NODE + """
            define run(n: uint8) -> uint8 {
                define owner = relay(make(n), n);
                if (owner.next != null) { return 0; }
                return drop(owner);
            }
        """)
        self.assertRegex(c, r"lan_record_\w+ \* lan_fn_make\(")
        self.assertIn("f_next == NULL", c)
        self.assertIn("owning parameters relay: param#0 owner (move)", table)
        self.assertIn("owning parameters drop: param#0 owner (free)", table)

    def test_handoff_takes_the_tail_unless_it_is_detached(self):
        source = LINKED_NODE + """
            define run() -> void {
                define tail: Node · {storage: heap};
                define head: Node · {storage: heap} = { next: tail };
                head.next = null;
                drop(head);
                free(tail);
            }
        """
        compile_source(source)
        with self.assertRaisesRegex(LanError, "already be freed or moved"):
            compile_source(source.replace("head.next = null;", ""))
        _, table = compile_source(LINKED_NODE + """
            define make_pair() -> Node · {storage: heap} {
                define tail: Node · {storage: heap};
                define head: Node · {storage: heap} = { next: tail };
                return head;
            }
        """)
        self.assertIn("owning result make_pair: chain lengths (2,)", table)

    def test_owned_result_can_infer_the_single_node_contract(self):
        _, table = compile_source(LINKED_NODE + """
            define inferred() {
                define owner: Node · {storage: heap} = { value: 2 };
                return owner;
            }
            define run() -> uint8 { return drop(inferred()); }
        """)
        self.assertIn("implicit contract inferred", table)
        self.assertIn("transfer ownership of an unnamed result", table)

    def test_null_next_contract_survives_table_export(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = LINKED_NODE + "export { Node, make, relay, drop };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source("""
                import { Node, make, relay, drop } from "./provider.table";
                define run() -> uint8 {
                    free(make(0));
                    define owner: Node · {storage: heap} = relay(make(3), 3);
                    if (owner.next != null) { return 0; }
                    return drop(owner);
                }
            """, root / "consumer.lang")
            self.assertRegex(c, r"extern lan_record_\w+ \* lan_api_\w+_make\(uint8_t\)")
            self.assertIn("f_next == NULL", c)

    def test_an_owning_boundary_rejects_other_link_shapes(self):
        with self.assertRaisesRegex(LanError, "one self-referential next link"):
            compile_source("""
                type Tree = { child: Tree · {pass: alias} | null };
                define drop(owner: Tree · {storage: heap}) -> void {}
            """)


class OwningParameterTests(unittest.TestCase):
    def test_release_requirements_follow_binding_ids_and_forwarded_calls(self):
        source = """
            define relay(x: uint8) -> uint8 { return drop(x); }
            define drop(x: uint8) -> uint8 {
                define value = x;
                define ref: uint8 · {pass: alias} = x;
                free(ref);
                return value;
            }
            define caller() -> uint8 {
                define x: uint8 · {storage: heap} = 7;
                return relay(x);
            }
            define shadow(x: uint8) -> uint8 {
                if (true) {
                    define x: uint8 · {storage: heap} = 1;
                    free(x);
                }
                if (false) { free(x); }
                return x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("inferred consuming parameter relay.x", trace)
        self.assertIn("owning parameters drop: param#0 x (free)", trace)
        self.assertNotIn("inferred consuming parameter shadow.x", trace)
        with self.assertRaisesRegex(LanError, "requires a live mutable heap owner"):
            compile_source(source.replace("uint8 · {storage: heap} = 7", "uint8 = 7"))

    def test_consumed_parameter_is_a_pointer_and_caller_hands_it_over(self):
        c, table = compile_source(SCALAR)
        self.assertRegex(c, r"uint8_t lan_fn_drop\(uint8_t \* lv_\d+_x\)")
        self.assertRegex(c, r"lan_fn_drop\(&\(\(\*lv_\d+_owner\)\)\)")
        self.assertIn("heap id#1 x: Alive, owning parameter", table)
        self.assertIn("consume param#0 x: freed in body", table)
        self.assertIn("owning parameters drop: param#0 x (free)", table)
        self.assertRegex(table, r"transfer ownership id#\d+ owner \[entry\] to param#0 drop")

    def test_records_and_arrays_are_consumed_with_their_own_descriptors(self):
        c, table = compile_source(RECORD)
        self.assertRegex(c, r"lan_record_\w+ \* lv_\d+_pair")
        self.assertIn("owning parameters drop: param#0 pair (free)", table)
        c, table = compile_source(ARRAY)
        self.assertRegex(c, r"lan_slice lv_\d+_items")
        self.assertIn("owning parameters drop: param#0 items (free)", table)

    def test_ownership_may_be_forwarded_instead_of_freed(self):
        _, table = compile_source("""
            define take: contract(x: uint8 · {storage: heap}) -> uint8 · {storage: heap} (x) {
                return x;
            }
            define caller: contract() -> uint8 () {
                define owner: uint8 · {storage: heap} = 7;
                define next = take(owner);
                define value = next;
                free(next);
                return value;
            }
        """)
        self.assertIn("consume param#0 x: transferred by move", table)
        self.assertIn("owning parameters take: param#0 x (move)", table)

    def test_unreleased_parameter_is_a_leak_on_every_exit(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'x' leaves"):
            compile_source(SCALAR.replace(FREED_IN_BODY, ""))
        with self.assertRaisesRegex(LanError, "heap leak: 'x' leaves"):
            compile_source(SCALAR.replace(FREED_IN_BODY, "return value;"))

    def test_a_local_alias_does_not_take_ownership(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'owner' leaves"):
            compile_source("""
                define drop(owner: uint8 · {storage: heap}) -> uint8 {
                    define alias: uint8 · {pass: alias} = owner;
                    return alias;
                }
            """)

    def test_table_records_consumption_of_the_owner_behind_an_alias(self):
        source = """
            define drop(owner: uint8 · {storage: heap}) -> void {
                define alias: uint8 · {pass: alias} = owner;
                free(alias);
            }
            define forward(owner: uint8 · {storage: heap}) -> uint8 · {storage: heap} {
                define alias: uint8 · {pass: alias} = owner;
                return alias;
            }
            export { drop, forward };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn("owning parameters drop: param#0 owner (free)", table)
            self.assertIn("owning parameters forward: param#0 owner (move)", table)
            (root / "provider.table").write_text(table, encoding="ascii")
            compile_source("""
                import { drop, forward } from "./provider.table";
                define run() -> void {
                    define owner: uint8 · {storage: heap} = 3;
                    drop(forward(owner));
                }
            """, root / "consumer.lang")

    def test_only_borrowed_writes_appear_in_the_mutation_summary(self):
        _, table = compile_source("""
            define drop(owner: uint8 · {storage: heap}, out: uint8 · {pass: alias}) -> void {
                owner = 3;
                out = owner;
                free(owner);
            }
        """)
        self.assertNotIn("mutation param#0", table)
        self.assertIn("mutation param#1", table)

    def test_a_propagating_argument_can_exit_before_the_handoff(self):
        source = """
            enum Err: uint8 { Bad = 250 }
            type Result = uint8[0,4] | Err;
            define fail() -> Result { return Err.Bad; }
            define make() -> uint8 · {storage: heap} {
                define value: uint8 · {storage: heap} = 1;
                return value;
            }
            define drop(owner: uint8 · {storage: heap}, value: uint8) -> uint8[0,4] {
                free(owner);
                return 0;
            }
            define caller() -> Result {
                define owner: uint8 · {storage: heap} = 1;
                return drop(owner, fail()?);
            }
        """
        with self.assertRaisesRegex(LanError, "heap leak: 'owner' leaves"):
            compile_source(source)
        with self.assertRaisesRegex(LanError, "before an unnamed heap result is handed to the callee"):
            compile_source(source.replace("define owner: uint8 · {storage: heap} = 1;\n", "").replace(
                "return drop(owner, fail()?);", "return drop(make(), fail()?);"))

    def test_consumed_argument_cannot_be_used_or_freed_again(self):
        with self.assertRaisesRegex(LanError, "may be a Tombstone"):
            compile_source(SCALAR.replace("return drop(owner);", "drop(owner); return owner;"))
        with self.assertRaisesRegex(LanError, "may already be freed or moved"):
            compile_source(SCALAR.replace("return drop(owner);", "drop(owner); free(owner); return 0;"))

    def test_only_a_live_mutable_owner_can_be_consumed(self):
        with self.assertRaisesRegex(LanError, "requires a live mutable heap owner"):
            compile_source("""
                define drop: contract(x: uint8 · {storage: heap}) -> void (x) { free(x); }
                define caller: contract() -> uint8 () {
                    define plain: uint8 = 7;
                    drop(plain);
                    return 0;
                }
            """)
        with self.assertRaisesRegex(LanError, "requires the same heap type"):
            compile_source("""
                type Pair = { left: uint8 };
                define drop: contract(x: uint8 · {storage: heap}) -> void (x) { free(x); }
                define caller: contract() -> uint8 () {
                    define owner: Pair · {storage: heap} = { left: 7 };
                    drop(owner);
                    return 0;
                }
            """)

    def test_consuming_a_parameter_only_on_one_path_leaks_on_the_other(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'owner' leaves scope before free"):
            compile_source("""
                define drop: contract(x: uint8 · {storage: heap}) -> void (x) { free(x); }
                define caller: contract(flag: boolean) -> uint8 (flag) {
                    define owner: uint8 · {storage: heap} = 1;
                    if (flag) { drop(owner); }
                    return 0;
                }
            """)

    def test_forwarding_to_another_consumer_counts_as_consumption(self):
        _, table = compile_source("""
            define inner: contract(x: uint8 · {storage: heap}) -> void (x) { free(x); }
            define outer: contract(x: uint8 · {storage: heap}) -> void (x) { inner(x); }
        """)
        self.assertIn("owning parameters inner: param#0 x (free)", table)
        self.assertIn("owning parameters outer: param#0 x (move)", table)

    def test_an_owning_result_can_be_consumed_without_a_binding(self):
        c, table = compile_source("""
            define make: contract() -> uint8 · {storage: heap} () {
                define x: uint8 · {storage: heap} = 7;
                return x;
            }
            define drop: contract(x: uint8 · {storage: heap}) -> void (x) { free(x); }
            define caller: contract() -> uint8 () { drop(make()); return 0; }
        """)
        self.assertIn("transfer ownership of an unnamed result", table)
        self.assertIn("(void)(lan_fn_drop(lan_fn_make()));", c)

    def test_an_owning_result_can_be_freed_directly(self):
        c, table = compile_source("""
            define make: contract() -> uint8 · {storage: heap} () {
                define x: uint8 · {storage: heap} = 7;
                return x;
            }
            define caller: contract() -> uint8 () { free(make()); return 0; }
        """)
        self.assertIn("free unnamed result", table)
        self.assertIn("free((void *)(lan_fn_make()));", c)
        with self.assertRaisesRegex(LanError, "free requires an owning heap entity"):
            compile_source("define plain: contract() -> uint8 () { return 1; }"
                           " define caller: contract() -> uint8 () { free(plain()); return 0; }")

    def test_an_unnamed_result_must_match_the_consuming_contract(self):
        with self.assertRaisesRegex(LanError, "requires the same heap type"):
            compile_source("""
                define make: contract() -> uint8 · {storage: heap} () {
                    define x: uint8 · {storage: heap} = 7;
                    return x;
                }
                define drop: contract(x: uint16 · {storage: heap}) -> void (x) { free(x); }
                define caller: contract() -> uint8 () { drop(make()); return 0; }
            """)
        with self.assertRaisesRegex(LanError, "named heap owner or an owning result"):
            compile_source("""
                define box: contract() -> uint8 () { return 1; }
                define drop: contract(x: uint8 · {storage: heap}) -> void (x) { free(x); }
                define caller: contract() -> uint8 () { drop(box()); return 0; }
            """)

    def test_native_functions_cannot_take_ownership(self):
        with self.assertRaisesRegex(LanError, "native function cannot take ownership"):
            compile_source("""
                extern contract release(x: uint8 · {storage: heap}) -> void writes { memory };
            """)

    def test_owning_summary_crosses_the_table_interface(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = SCALAR.replace("define caller: contract() -> uint8 () {\n        define owner:"
                                    " uint8 · {storage: heap} = 7;\n        return drop(owner);\n    }", "")
            source += "export { drop };"
            table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()[1]
            self.assertIn('"owning":[[0,"free"]]', table)
            (root / "provider.table").write_text(table, encoding="ascii")
            c, consumer = compile_source(
                'import { drop } from "./provider.table"; '
                'define caller: contract() -> uint8 () {'
                ' define owner: uint8 · {storage: heap} = 7; return drop(owner); }',
                root / "consumer.lang",
            )
            self.assertIn("lan_fn_drop" if False else "_drop(&((*lv_", c)
            self.assertIn("handed to callee", consumer)


if __name__ == "__main__":
    unittest.main()
