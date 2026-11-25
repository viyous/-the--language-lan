import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source
from table_interface import api_symbol


PROVIDER = """
type Node = {value: uint8, next: Node | null};
type Owner = Node · {storage: heap};

define make_list(n: uint8) -> Owner {
    define head: Owner = {value: 7};
    define tail: Node · {pass: alias} = head;
    define i = n;
    while (i > 0) {
        define item: Node · {storage: heap} = {value: 1};
        tail.next = item;
        tail = item;
        i -= 1;
    }
    return head;
}

define move_list(head: Owner) -> Owner { return head; }
define dispose(head: Owner) -> void { free(head); }
"""

CONSUMER = """
define observe(n: uint8) -> uint8 {
    define head = move_list(make_list(n));
    define result = head.value;
    if (head.next != null) {
        head.next.value = 9;
        result = head.next.value;
    }
    free(head);
    return result;
}

define release(n: uint8) -> void {
    define head = make_list(n);
    dispose(head);
    free(make_list(n));
}
"""


class OwnedChainHandoffTests(unittest.TestCase):
    def test_dynamic_chain_is_rebuilt_at_a_function_boundary(self):
        c, trace = compile_source(PROVIDER + CONSUMER)
        self.assertIn("transfer whole chain", trace)
        self.assertIn("owning result make_list: chain lengths (1, 2, 3, 4, 5)", trace)
        self.assertIn("owning result move_list: chain lengths (1, 2, 3, 4, 5)", trace)
        self.assertIn("free chain from id#temp_", trace)
        self.assertNotIn("f_next == NULL", c)

    def test_fixed_chain_rebuilds_a_nonnull_successor(self):
        _, trace = compile_source(PROVIDER + """
define make_pair() {
    define tail: Owner = {value: 1};
    define head: Owner = {value: 2, next: tail};
    return head;
}
define read() -> uint8[4,4] {
    define pair = make_pair();
    pair.next.value = 4;
    define v = pair.next.value;
    free(pair);
    return v;
}
""")
        self.assertIn("implicit contract make_pair", trace)
        self.assertIn("owning result make_pair: chain lengths (2,)", trace)

    def test_detaching_a_proven_prefix_keeps_both_owned_parts_live(self):
        compile_source(PROVIDER + """
define release_parts(n: uint8) -> void {
    define head = make_list(n);
    if (head.next != null) {
        define tail: Node · {pass: alias} = head.next;
        head.next = null;
        free(tail);
    }
    free(head);
}
""")

    def test_handoff_invalidates_a_borrow_of_a_descendant(self):
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(PROVIDER + """
define bad(n: uint8) -> uint8 {
    define head = make_list(n);
    if (head.next != null) {
        define tail: Node · {pass: alias} = head.next;
        dispose(head);
        return tail.value;
    }
    free(head);
    return 0;
}
""")

    def test_callee_may_require_a_terminal_node(self):
        source = PROVIDER + """
define terminal_only(head: Owner) -> void {
    if (head.next != null) { return; }
    free(head);
}
define good() -> void {
    define head: Owner;
    terminal_only(head);
}
"""
        compile_source(source)
        for consume in ("terminal_only(make_list(n));", "define head = make_list(n); terminal_only(head);"):
            with self.subTest(consume=consume), self.assertRaisesRegex(LanError, "requires next|unless next"):
                compile_source(source + f"define bad(n: uint8) -> void {{ {consume} }}")

    def test_nonowning_boundary_and_cycle_cannot_escape(self):
        with self.assertRaisesRegex(LanError, "non-owning storage"):
            compile_source(PROVIDER + """
define bad() -> Owner {
    define tail: Node;
    define head: Owner = {next: tail};
    return head;
}
""")
        with self.assertRaisesRegex(LanError, "one finite chain"):
            compile_source(PROVIDER + """
define bad() -> Owner {
    define a: Owner;
    define b: Owner = {next: a};
    a.next = b;
    return b;
}
""")

    def test_full_success_chain_survives_tagged_error_propagation(self):
        compile_source(PROVIDER + """
enum Err: uint8 { Missing = 0 }
type Result = Owner | Err;
define try_make(n: uint8) -> Result {
    if (n == 0) { return Err.Missing; }
    return make_list(n);
}
define forward(n: uint8) -> Result { return try_make(n); }
define release(n: uint8) -> uint8[7,7] | Err {
    free(forward(n)?);
    return 7;
}
""")

    def test_shape_and_owning_parameter_summaries_cross_table_only_import(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
            path = Path(scratch) / "provider.lang"
            source = PROVIDER + "export { Node, Owner, make_list, move_list, dispose };"
            provider_c, table = ModuleGraph(path, Compiler(source, path.name), library=True).compile()
            path.with_suffix(".table").write_text(table, encoding="ascii")
            self.assertIn('"owned_chain_lengths":[1,2,3,4,5]', table)
            self.assertIn('"owned_chain_parameters":[0]', table)
            consumer_c, _ = compile_source(
                'import { make_list, move_list, dispose } from "./provider.table";\n' + CONSUMER,
                path.with_name("consumer.lang"))
            header = json.loads(table.splitlines()[0].removeprefix("@interface "))
            old_symbol = api_symbol(header["abi"], "dispose", header["exports"]["dispose"]["type"])
            self.assertNotIn(old_symbol, provider_c)
            self.assertNotIn(old_symbol, consumer_c)
