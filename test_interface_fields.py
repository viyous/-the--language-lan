import re
import unittest

from lanc import LanError, compile_source


TYPES = """
    type View = interface { x: uint8 };
    type Item = { x: uint8, y: uint8 };
    type Holder = { view: View };
"""

MUTATE = """
    define update() -> uint8[9,9] {
        define item: Item = { x: 7 };
        define holder: Holder = { view: item };
        holder.view.x = 9;
        return item.x;
    }
"""

REORDER = """
    type Both = interface { y: uint8, x: uint8 };
    define read_x(view: View) -> uint8 { return view.x; }
    define read_reordered() -> uint8 {
        define item: Item = { x: 7, y: 42 };
        define both: Both = item;
        define holder: Holder = { view: both };
        return read_x(holder.view);
    }
"""

REBIND = """
    type Other = { padding: uint16, x: uint8 };
    define rebound() -> uint8[79,79] {
        define a: Item = { x: 7 };
        define b: Other = { x: 8 };
        define holder: Holder = { view: a };
        define saved: View = holder.view;
        define keep_running = true;
        /** @oracle @transition => { state(keep_running) := {0}; state(holder.view.target) := {b}; } */
        while (keep_running) { holder.view = b; keep_running = false; }
        holder.view.x = 9;
        return saved.x * 10 + b.x;
    }
"""


class InterfaceFieldTests(unittest.TestCase):
    def test_mutation_updates_the_actual_record(self):
        c, table = compile_source(TYPES + MUTATE)
        self.assertIn("lan_interface f_view;", c)
        self.assertIn(".f_view.offsets[0]", c)
        self.assertIn("interface targets", table)
        self.assertIn("return {9}", table)

    def test_copies_reordered_offsets_and_can_pass_the_field_as_an_interface(self):
        c, _ = compile_source(TYPES + REORDER)
        self.assertIn("const size_t[]", c)
        self.assertIn("lan_field_offsets_", c)
        self.assertIn("lan_fn_read_x(", c)

    def test_rebinding_keeps_independent_snapshot_and_can_be_proved(self):
        c, trace = compile_source(TYPES + REBIND)
        self.assertRegex(c, r"lan_interface lv_\d+_saved = lh_interface_rebind_\d+")
        self.assertIn("state(holder . view . target) := { b };", trace)
        self.assertIn("return {79}", trace)

    def test_record_copy_snapshots_offsets_and_preserves_field_aliases(self):
        c, trace = compile_source(TYPES + """
            type Other = { padding: uint16, x: uint8 };
            define f() -> uint8[95,95] {
                define a: Item = { x: 7 };
                define b: Other = { x: 4 };
                define holder: Holder = { view: a };
                define saved = holder as: copy;
                define assigned: Holder = saved as: copy;
                holder.view = b;
                assigned = holder as: copy;
                holder.view = a;
                saved.view.x = 9;
                assigned.view.x = 5;
                return holder.view.x * 10 + b.x;
            }
        """)
        self.assertEqual(len(set(re.findall(r"size_t (lan_field_offsets_\d+_view)\[1\]", c))), 3)
        self.assertIn("return {95}", trace)

    def test_record_copy_checks_referent_lifetime_and_readonly_access(self):
        with self.assertRaisesRegex(LanError, "outlive its target storage"):
            compile_source(TYPES + """
                define f() {
                    define a: Item;
                    define dst: Holder = { view: a };
                    if (true) {
                        define b: Item;
                        define src: Holder = { view: b };
                        dst = src as: copy;
                    }
                }
            """)
        with self.assertRaisesRegex(LanError, "immutable interface into a writable field"):
            compile_source(TYPES + """
                define f(h: Holder · {immutable: true}) {
                    define writable: Holder = h as: copy;
                }
            """)

    def test_record_copy_preserves_link_fields_alongside_interfaces(self):
        _, trace = compile_source(TYPES + """
            type Cell = { n: uint8, next: Cell | null };
            type Mixed = { view: View, next: Cell | null };
            define f() -> uint8[10,10] {
                define item: Item = { x: 3 };
                define cell: Cell = { n: 7 };
                define mixed: Mixed = { view: item, next: cell };
                define saved = mixed as: copy;
                mixed.next = null;
                return saved.view.x + saved.next.n;
            }
        """)
        self.assertIn("return {10}", trace)

    def test_rebinding_rejects_targets_with_shorter_lifetimes_and_wrong_claims(self):
        with self.assertRaisesRegex(LanError, "outlive its target storage"):
            compile_source(TYPES + """
                define f() {
                    define i: Item;
                    define h: Holder = { view: i };
                    if (true) { define inner: Item; h.view = inner; }
                }
            """)
        with self.assertRaisesRegex(LanError, "unexpected targets"):
            compile_source(TYPES + REBIND.replace("target) := {b}", "target) := {a}"))
        with self.assertRaisesRegex(LanError, "direct non-heap record target"):
            compile_source(TYPES + """
                define f() {
                    define i: Item;
                    define h: Holder = { view: i };
                    define owner: Item · {storage: heap};
                    h.view = owner;
                    free(owner);
                }
            """)

    def test_missing_initializer_and_incompatible_source_fail(self):
        cases = (
            ("define f() { define h: Holder = {}; }", "interface field requires an initializer"),
            ("type Other = { y: uint8 }; define f() { define o: Other; define h: Holder = {view:o}; }",
             "missing field 'x'"),
            ("type Other = { x: uint16 }; define f() { define o: Other; define h: Holder = {view:o}; }",
             "incompatible type"),
        )
        for source, error in cases:
            with self.subTest(source=source), self.assertRaisesRegex(LanError, error):
                compile_source(TYPES + source)

    def test_readonly_view_and_local_storage_boundary(self):
        cases = (
            ("type Frozen = { view: View · {immutable: true} }; "
             "define f() { define i: Item; define h: Frozen = {view:i}; h.view.x = 9; }", "immutable field"),
            ("define f() { define i: Item; define h: Holder = {view:i}; return h as: copy; }", "local-only"),
            ("type Frozen = { view: View · {immutable: true} }; "
             "define f() { define i: Item; define h: Frozen = {view:i}; h.view = i; }",
             "cannot rebind an immutable interface field"),
            ("type Batch = Holder[2];", "arrays of records with interface fields"),
            ("define f() { define i: Item; define h: Holder = {view:i}; "
             "define owner: Holder · {storage: heap} = h; }", "local field initializer"),
        )
        for source, error in cases:
            with self.subTest(source=source), self.assertRaisesRegex(LanError, error):
                compile_source(TYPES + source)


PASSING = TYPES + """
    define read_view(h: Holder) -> uint8 { return h.view.x; }
    define call_view(h: Holder) -> uint8 { return read_x(h.view); }
    define read_x(view: View) -> uint8 { return view.x; }
"""


class InterfaceFieldParameterTests(unittest.TestCase):
    def test_borrowed_record_descriptor_can_be_copied_into_local_storage(self):
        c, _ = compile_source(TYPES + """
            define f(h: Holder) -> uint8 {
                define local = h as: copy;
                return local.view.x;
            }
        """)
        self.assertRegex(c, r"lan_record_\w+ lv_\d+_local = \(\*lv_\d+_h\);")
        self.assertRegex(c, r"size_t lan_field_offsets_\d+_view\[1\];")

    def test_borrowed_record_keeps_its_interface_field_readable(self):
        c, table = compile_source(PASSING + """
            define caller() -> uint8 {
                define item: Item = { x: 7, y: 42 };
                define holder: Holder = { view: item };
                define first = read_view(holder);
                return call_view(holder);
            }
        """)
        self.assertIn("opaque target", table)
        self.assertRegex(c, r"lan_fn_read_view\(lan_record_\w+ \* lv_\d+_h\)")
        self.assertIn("interface targets {id#13}", table)

    def test_normal_fields_are_summarised_and_interface_reads_stay_declared(self):
        _, table = compile_source(TYPES + """
            type Sized = { count: uint8, view: View };
            define bump(holder: Sized) -> uint8 {
                holder.count = 1;
                return holder.view.x;
            }
            define caller() -> uint8 {
                define item: Item = { x: 7, y: 42 };
                define holder: Sized = { count: 0, view: item };
                define seen = bump(holder);
                return holder.count;
            }
        """)
        self.assertIn("mutation param#0.count => {1}", table)
        self.assertIn("return [0, 255] [entry]", table)

    def test_writes_through_a_borrowed_interface_field_reach_the_caller_target(self):
        _, table = compile_source(TYPES + """
            type Sized = { count: uint8, view: View };
            define bump(holder: Sized) -> uint8 {
                holder.view.x = 9;
                return holder.view.x;
            }
            define caller() -> uint8 {
                define item: Item = { x: 7, y: 42 };
                define holder: Sized = { count: 0, view: item };
                define seen = bump(holder);
                return item.y;
            }
        """)
        self.assertIn("mutation param#0.view.*.x => {9}", table)
        self.assertIn("apply interface write", table)
        self.assertIn("return {42} [entry]", table)

    def test_a_compounded_write_through_an_interface_field_is_summarised(self):
        _, table = compile_source(TYPES + """
            type Sized = { view: View };
            define bump(holder: Sized) -> uint8 {
                holder.view.x += 0;
                return holder.view.x;
            }
        """)
        self.assertIn("mutation param#0.view.*.x", table)

    def test_only_the_selected_target_is_invalidated(self):
        _, table = compile_source(TYPES + """
            define bump(holder: Holder) -> uint8 {
                holder.view.x = 9;
                return 0;
            }
            define caller() -> uint8 {
                define a: Item = { x: 1, y: 2 };
                define b: Item = { x: 3, y: 4 };
                define ha: Holder = { view: a };
                define hb: Holder = { view: b };
                bump(ha);
                return a.y + b.x;
            }
        """)
        self.assertIn("apply interface write", table)
        self.assertIn("return {5} [entry]", table)

    def test_a_wrapper_relays_the_interface_write(self):
        _, table = compile_source(TYPES + """
            define bump(holder: Holder) -> uint8 {
                holder.view.x = 9;
                return 0;
            }
            define outer(holder: Holder) -> uint8 {
                return bump(holder);
            }
            define caller() -> uint8 {
                define item: Item = { x: 7, y: 42 };
                define holder: Holder = { view: item };
                define seen = outer(holder);
                return item.x + item.y;
            }
        """)
        self.assertEqual(table.count("mutation param#0.view.*.x => {9}"), 2)
        self.assertIn("return {51} [entry]", table)

    def test_writes_reach_fields_of_a_nested_target(self):
        _, table = compile_source("""
            type Inner = { flag: uint8, spare: uint8 };
            type Nested = interface { inner: Inner };
            type NestedItem = { inner: Inner, other: uint8 };
            type NestedHolder = { view: Nested };
            define bump(holder: NestedHolder) -> uint8 {
                holder.view.inner.flag = 1;
                return 0;
            }
            define caller() -> uint8 {
                define item: NestedItem = { inner: { flag: 0, spare: 5 }, other: 3 };
                define holder: NestedHolder = { view: item };
                define seen = bump(holder);
                return item.inner.spare;
            }
        """)
        self.assertIn("mutation param#0.view.*.inner.flag => {1}", table)
        self.assertIn("return {5} [entry]", table)

    def test_borrowed_field_rebind_and_forwarder_preserve_snapshots(self):
        c, trace = compile_source(TYPES + """
            type Other = { padding: uint16, x: uint8 };
            define rebind(h: Holder, i: Other) -> uint8 {
                h.view = i;
                return h.view.x;
            }
            define forward(i: Other, h: Holder) -> uint8 { return rebind(h, i); }
            define use() -> uint8[95,95] {
                define a: Item = { x: 9 };
                define b: Other = { x: 4 };
                define h: Holder = { view: a };
                define saved = h as: copy;
                forward(b, h);
                h.view.x = 5;
                return saved.view.x * 10 + b.x;
            }
        """)
        self.assertRegex(c, r"\(size_t \*\)\(\*lv_\d+_h\)\.f_view\.offsets")
        self.assertIn("mutation param#0.view.target => param#1", trace)
        self.assertIn("mutation param#1.view.target => param#0", trace)
        self.assertIn("return {95} [entry]", trace)

    def test_borrowed_field_rebind_rejects_escaping_or_immutable_donors(self):
        setter = TYPES + "define set(h: Holder, i: Item) { h.view = i; }"
        with self.assertRaisesRegex(LanError, "outlive its target storage"):
            compile_source(setter + """
                define f() {
                    define old: Item;
                    define h: Holder = { view: old };
                    if (true) { define inner: Item; set(h, inner); }
                }
            """)
        with self.assertRaisesRegex(LanError, "requires mutable storage"):
            compile_source(setter + """
                define f() {
                    define old: Item;
                    define h: Holder = { view: old };
                    define donor: Item · {immutable: true} = { x: 1 };
                    set(h, donor);
                }
            """)
        with self.assertRaisesRegex(LanError, "direct non-heap record target"):
            compile_source(setter + """
                define f() {
                    define old: Item;
                    define h: Holder = { view: old };
                    define donor: Item · {storage: heap};
                    set(h, donor);
                    free(donor);
                }
            """)

    def test_borrowed_field_rebind_rejects_conflicting_writes(self):
        cases = (
            ("define f(h: Holder) { define i: Item; h.view = i; }",
             "compatible borrowed record parameter"),
        )
        for source, error in cases:
            with self.subTest(source=source), self.assertRaisesRegex(LanError, error):
                compile_source(TYPES + source)

    def test_repeated_rebind_of_one_field_keeps_the_last_assignment(self):
        source = TYPES + """
            type Other = { padding: uint16, x: uint8 };
            define set(h: Holder, i: Item, j: Other) { h.view = i; h.view = j; }
            define choose(h: Holder, i: Item, j: Other, swap: boolean) {
                if (swap) { h.view = i; h.view = j; }
            }
            define use() -> uint8[5,5] {
                define old: Item = { x: 9 };
                define i: Item = { x: 4 };
                define j: Other = { x: 5 };
                define h: Holder = { view: old };
                set(h, i, j);
                return h.view.x;
            }
            define use_guarded(swap: boolean) -> uint8[5,9] {
                define old: Item = { x: 9 };
                define i: Item = { x: 4 };
                define j: Other = { x: 5 };
                define h: Holder = { view: old };
                choose(h, i, j, swap);
                return h.view.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("mutation param#0.view.target => param#2", trace)
        self.assertIn("mutation param#0.view.target => param#2 when param#3 == 1", trace)
        with self.assertRaisesRegex(LanError, "cannot prove.*fits"):
            compile_source(source.replace("uint8[5,9]", "uint8[5,5]"))

    def test_a_later_unconditional_rebind_shadows_a_guarded_one(self):
        source = TYPES + """
            type Other = { padding: uint16, x: uint8 };
            define choose(h: Holder, i: Item, j: Other, swap: boolean) {
                if (swap) { h.view = i; }
                h.view = j;
            }
            define use(swap: boolean) -> uint8[5,5] {
                define old: Item = { x: 9 };
                define i: Item = { x: 4 };
                define j: Other = { x: 5 };
                define h: Holder = { view: old };
                choose(h, i, j, swap);
                return h.view.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("mutation param#0.view.target => param#2", trace)

    def test_multiple_interface_fields_can_rebind_with_distinct_guards(self):
        source = TYPES.replace("view: View };", "view: View, backup: View, mark: uint8 };") + """
            type Other = { padding: uint16, x: uint8 };
            define choose(h: Holder, a: Item, b: Other, swap: boolean, change_backup: boolean) {
                if (swap) { h.view = b; h.mark = 7; } else { h.view = a; h.mark = 7; }
                if (change_backup) { h.backup = b; }
            }
            define forward(swap: boolean, h: Holder, a: Item, change_backup: boolean, b: Other) {
                choose(h, a, b, swap, change_backup);
            }
            define use(swap: boolean, change_backup: boolean) -> uint32[98457,98587] {
                define old: Item = { x: 9 };
                define old_backup: Item = { x: 8 };
                define a: Item = { x: 4 };
                define b: Other = { x: 5 };
                define h: Holder = { view: old, backup: old_backup };
                define saved = h as: copy;
                forward(swap, h, a, change_backup, b);
                return saved.view.x * 10000 + saved.backup.x * 1000 + h.view.x * 100 + h.backup.x * 10 + h.mark;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("mutation param#1.view.target => param#2 when param#0 == 0 else param#4", trace)
        self.assertIn("mutation param#1.backup.target => param#4 when param#3 == 1", trace)
        _, trace = compile_source(source.replace("forward(swap, h, a, change_backup, b);",
                                                 "forward(false, h, a, true, b);")
                                        .replace("[98457,98587]", "[98457,98457]"))
        self.assertIn("return {98457}", trace)

    def test_two_guards_union_the_possible_targets(self):
        _, trace = compile_source(TYPES + """
            define set(h: Holder, a: Item, b: Item, p: boolean, q: boolean) {
                if (p) { h.view = a; }
                if (q) { h.view = b; }
            }
            define use(p: boolean, q: boolean) -> uint8 {
                define old: Item = { x: 9 };
                define a: Item = { x: 4 };
                define b: Item = { x: 5 };
                define h: Holder = { view: old };
                set(h, a, b, p, q);
                return h.view.x;
            }
        """)
        self.assertIn("param#0.view.target => param#1 when param#3 == 1", trace)
        self.assertIn("param#0.view.target => param#2 when param#4 == 1", trace)
        # Neither guard is taken on some path, so the original target survives.
        self.assertIn("return [4, 5] | {9}", trace)

    def test_nested_guards_need_every_literal(self):
        source = TYPES + """
            define set(h: Holder, i: Item, p: boolean, q: boolean) {
                if (p) { if (q) { h.view = i; } }
            }
            define use(p: boolean, q: boolean) -> uint8 {
                define old: Item = { x: 9 };
                define i: Item = { x: 4 };
                define h: Holder = { view: old };
                set(h, i, p, q);
                return h.view.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("param#0.view.target => param#1 when param#2 == 1 and param#3 == 1", trace)
        self.assertIn("return {4} | {9}", trace)
        # Both literals hold at the call site, so the rebind becomes exact.
        _, folded = compile_source(source.replace("set(h, i, p, q);", "set(h, i, true, true);"))
        self.assertIn("return {4} [entry]", folded)

    def test_guarded_arms_apply_as_one_may_set(self):
        # Two arms must be applied together: applying the second alone would
        # clear the first one's target instead of adding to it.
        _, trace = compile_source(TYPES + """
            define set(h: Holder, a: Item, b: Item, p: boolean, q: boolean) {
                if (p) { h.view = a; }
                if (q) { h.view = b; }
            }
            define use() -> uint8 {
                define old: Item = { x: 9 };
                define a: Item = { x: 4 };
                define b: Item = { x: 5 };
                define h: Holder = { view: old };
                set(h, a, b, true, true);
                return h.view.x;
            }
        """)
        self.assertIn("return [4, 5] [entry]", trace)

    def test_branch_tree_crosses_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, ModuleGraph

        source = TYPES + """
            define set(h: Holder, i: Item, p: boolean, q: boolean) {
                if (p) { if (q) { h.view = i; } }
            }
            export { Holder, Item, set };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "tree.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn('"@interface_rebind","view","1","2","1","3","1"', table)
            (root / "tree.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
                import { Holder, Item, set } from "./tree.table";
                define use(p: boolean, q: boolean) -> uint8 {
                    define old: Item = { x: 9 };
                    define i: Item = { x: 4 };
                    define h: Holder = { view: old };
                    set(h, i, p, q);
                    return h.view.x;
                }
            """, root / "consumer.lang")
            self.assertIn("return {4} | {9}", trace)

    def test_interface_write_between_guarded_arms_is_rejected(self):
        source = TYPES + """
            define set(h: Holder, a: Item, b: Item, p: boolean, q: boolean) {
                if (p) { h.view = a; }
                h.view.x = 3;
                if (q) { h.view = b; }
            }
        """
        with self.assertRaisesRegex(LanError, "between two guarded rebinds"):
            compile_source(source)

    def test_multiple_interface_rebinds_cross_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, InterfaceError, ModuleGraph, decode_mutations

        source = TYPES.replace("view: View };", "view: View, backup: View };") + """
            define choose(h: Holder, a: Item, b: Item, swap: boolean) {
                if (swap) { h.view = b; h.backup = a; } else { h.view = a; h.backup = b; }
            }
            export { Holder, Item, choose };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "choice.lang"
            compiler = Compiler(source, provider.name)
            _, table = ModuleGraph(provider, compiler, library=True).compile()
            self.assertIn('["@interface_rebind","view","1","3","0","2"]', table)
            self.assertIn('["@interface_rebind","backup","1","3","1","2"]', table)
            (root / "choice.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
                import { Holder, Item, choose } from "./choice.table";
                define use() -> uint8[54,54] {
                    define old: Item = { x: 9 };
                    define a: Item = { x: 4 };
                    define b: Item = { x: 5 };
                    define h: Holder = { view: old, backup: old };
                    choose(h, a, b, true);
                    return h.view.x * 10 + h.backup.x;
                }
            """, root / "consumer.lang")
            self.assertIn("return {54}", trace)
            with self.assertRaisesRegex(InterfaceError, "invalid interface rebind"):
                decode_mutations([[0, ["@interface_rebind", "view", "1"], None],
                                  [0, ["@interface_rebind", "view", "2"], None]],
                                 compiler.functions["choose"][1])

    def test_ordered_rebind_crosses_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, ModuleGraph

        source = TYPES + """
            type Other = { padding: uint16, x: uint8 };
            define choose(h: Holder, a: Item, b: Other, swap: boolean) {
                if (swap) { h.view = a; h.view = b; }
            }
            export { Holder, Item, Other, choose };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "ordered.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn('["@interface_rebind","view","2","3","1"]', table)
            (root / "ordered.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
                import { Holder, Item, Other, choose } from "./ordered.table";
                define use(swap: boolean) -> uint8[5,9] {
                    define old: Item = { x: 9 };
                    define a: Item = { x: 4 };
                    define b: Other = { x: 5 };
                    define h: Holder = { view: old };
                    choose(h, a, b, swap);
                    return h.view.x;
                }
            """, root / "consumer.lang")
            self.assertIn("return {5} | {9}", trace)

    def test_multiple_borrowed_holders_rebind_only_when_disjoint(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, ModuleGraph

        provider_source = TYPES + """
            define update(left: Holder, right: Holder, a: Item, b: Item, change: boolean) {
                left.view = a;
                if (change) { right.view = b; }
            }
            export { Holder, Item, update };
        """
        consumer = """
            import { Holder, Item, update } from "./holders.table";
            define forward(left: Holder, a: Item, right: Holder, b: Item, change: boolean) {
                update(left, right, a, b, change);
            }
            define use(change: boolean) -> uint8[45,49] {
                define old: Item = { x: 9 };
                define a: Item = { x: 4 };
                define b: Item = { x: 5 };
                define left: Holder = { view: old };
                define right: Holder = { view: old };
                forward(left, a, right, b, change);
                return left.view.x * 10 + right.view.x;
            }
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "holders.lang"
            _, table = ModuleGraph(provider, Compiler(provider_source, provider.name), library=True).compile()
            (root / "holders.table").write_text(table, encoding="ascii")
            _, trace = compile_source(consumer, root / "consumer.lang")
            self.assertIn("mutation param#0.view.target => param#1", trace)
            self.assertIn("mutation param#2.view.target => param#3 when param#4 == 1", trace)
            with self.assertRaisesRegex(LanError, "receivers must be disjoint"):
                compile_source(consumer.replace("forward(left, a, right, b, change);",
                                                "forward(left, a, left, b, change);"), root / "consumer.lang")

    def test_same_holder_can_rebind_distinct_fields_through_two_parameters(self):
        types = TYPES.replace("view: View };", "view: View, backup: View };")
        source = types + """
            type Other = { padding: uint16, x: uint8 };
            define set_both(h: Holder, h2: Holder, a: Item, b: Other) {
                h.view = a;
                h2.backup = b;
            }
            define use() -> uint8[45,45] {
                define old: Item = { x: 9 };
                define a: Item = { x: 4 };
                define b: Other = { x: 5 };
                define h: Holder = { view: old, backup: old };
                set_both(h, h, a, b);
                return h.view.x * 10 + h.backup.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("mutation param#0.view.target => param#2", trace)
        self.assertIn("mutation param#1.backup.target => param#3", trace)
        with self.assertRaisesRegex(LanError, "distinct fields"):
            compile_source(types + """
                define set_same(h: Holder, h2: Holder, a: Item) {
                    h.view = a;
                    h2.view = a;
                }
                define bad() {
                    define old: Item;
                    define a: Item;
                    define h: Holder = { view: old, backup: old };
                    set_same(h, h, a);
                }
            """)

    def test_dynamic_array_index_after_rebind_updates_only_the_donor_footprint(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, InterfaceError, ModuleGraph, decode_mutations

        provider_source = """
            type View = interface { data: uint8[4] };
            type Item = { data: uint8[4] };
            type Holder = { view: View };
            define update(h: Holder, d: Item, i: uint8[0,2]) { h.view = d; h.view.data[i] = 7; }
            export { Holder, Item, update };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "arrays.lang"
            compiler = Compiler(provider_source, provider.name)
            _, table = ModuleGraph(provider, compiler, library=True).compile()
            self.assertIn('["data","@range:0:3"],null', table)
            (root / "arrays.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
                import { Holder, Item, update } from "./arrays.table";
                define use(i: uint8[0,2]) -> uint8[94,94] {
                    define old: Item = { data: [9,9,9,9] };
                    define d: Item = { data: [4,4,4,4] };
                    define h: Holder = { view: old };
                    update(h, d, i);
                    return old.data[0] * 10 + d.data[3];
                }
            """, root / "consumer.lang")
            self.assertIn("return {94}", trace)
            with self.assertRaises(InterfaceError):
                decode_mutations([[0, ["@interface_rebind", "view", "1"], None],
                                  [1, ["data", "@range:0:3"], [[7, 7]]]], compiler.functions["update"][1])
        with self.assertRaises(LanError):
            compile_source(provider_source.replace("h.view = d; h.view.data[i] = 7;",
                                                   "h.view.data[i] = 7; h.view = d;"))

    def test_one_setter_can_rebind_a_stack_and_a_live_heap_holder(self):
        source = TYPES + """
            type Other = { pad: uint16, x: uint8 };
            define set(a: Holder, b: Holder, donor: Other) { a.view = donor; b.view = donor; }
            define use() -> uint8[95,95] {
                define old: Item = { x: 9 };
                define donor: Other = { x: 4 };
                define a: Holder = { view: old };
                define b: Holder · {storage: heap} = { view: old };
                define moved: Holder · {storage: heap} = b;
                define saved = a as: copy;
                set(a, moved, donor);
                moved.view.x = 5;
                free(moved);
                return saved.view.x * 10 + a.view.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("return {95}", trace)
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(source.replace("set(a, moved, donor);", "set(a, b, donor);"))

    def test_conditional_borrowed_field_rebind_keeps_both_possible_targets(self):
        source = TYPES + """
            type Other = { padding: uint16, x: uint8 };
            define set(h: Holder, donor: Other, keep: boolean) {
                if (!keep) { h.view = donor; }
            }
            define forward(keep: boolean, donor: Other, h: Holder) { set(h, donor, keep); }
            define use(keep: boolean) -> uint8[4,9] {
                define old: Item = { x: 9 };
                define donor: Other = { x: 4 };
                define h: Holder = { view: old };
                forward(keep, donor, h);
                return h.view.x;
            }
        """
        for argument, result in (("keep", "[4,9]"), ("false", "[4,4]"), ("true", "[9,9]")):
            with self.subTest(argument=argument):
                _, trace = compile_source(source.replace("forward(keep, donor, h);", f"forward({argument}, donor, h);").replace(
                    "uint8[4,9]", f"uint8{result}"))
                self.assertIn("mutation param#0.view.target => param#1 when param#2 == 0", trace)
                self.assertIn("mutation param#2.view.target => param#1 when param#0 == 0", trace)
        with self.assertRaisesRegex(LanError, "cannot prove.*fits"):
            compile_source(source.replace("uint8[4,9]", "uint8[4,4]"))

    def test_exhaustive_rebind_discards_the_old_target_and_preserves_snapshots(self):
        source = TYPES + """
            type Other = { padding: uint16, x: uint8 };
            define choose(h: Holder, a: Item, b: Other, second: boolean) {
                if (!second) { h.view = a; } else { h.view = b; }
            }
            define forward(second: boolean, a: Item, h: Holder, b: Other) {
                choose(h, a, b, second);
            }
            define use(second: boolean) -> uint8[94,95] {
                define old: Item = { x: 9 };
                define a: Item = { x: 4 };
                define b: Other = { x: 5 };
                define h: Holder = { view: old };
                define saved = h as: copy;
                forward(second, a, h, b);
                return saved.view.x * 10 + h.view.x;
            }
        """
        for argument, result in (("second", "[94,95]"), ("false", "[94,94]"), ("true", "[95,95]")):
            with self.subTest(argument=argument):
                _, trace = compile_source(source.replace("forward(second, a, h, b);", f"forward({argument}, a, h, b);")
                                          .replace("uint8[94,95]", f"uint8{result}"))
                self.assertIn("mutation param#2.view.target => param#1 when param#0 == 0 else param#3", trace)

    def test_exhaustive_rebind_crosses_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, InterfaceError, ModuleGraph, decode_mutations

        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "choice.lang"
            source = TYPES + """
                type Other = { padding: uint16, x: uint8 };
                define choose(h: Holder, a: Item, b: Other, second: boolean) {
                    if (second) { h.view = b; } else { h.view = a; }
                }
                export { Holder, Item, Other, choose };
            """
            compiler = Compiler(source, provider.name)
            _, table = ModuleGraph(provider, compiler, library=True).compile()
            self.assertIn('["@interface_rebind","view","1","3","0","2"]', table)
            (root / "choice.table").write_text(table, encoding="ascii")
            compile_source("""
                import { Holder, Item, Other, choose } from "./choice.table";
                define use(second: boolean) -> uint8[4,5] {
                    define old: Item = { x: 9 };
                    define a: Item = { x: 4 };
                    define b: Other = { x: 5 };
                    define h: Holder = { view: old };
                    choose(h, a, b, second);
                    return h.view.x;
                }
            """, root / "consumer.lang")
            for donor in ("0", "3", "4", "02"):
                with self.subTest(donor=donor), self.assertRaisesRegex(InterfaceError, "invalid interface rebind"):
                    decode_mutations([[0, ["@interface_rebind", "view", "1", "3", "0", donor], None]],
                                     compiler.functions["choose"][1])

    def test_exhaustive_rebind_checks_lifetime_of_either_selected_donor(self):
        source = TYPES + """
            define choose(h: Holder, a: Item, b: Item, second: boolean) {
                if (second) { h.view = b; } else { h.view = a; }
            }
            define use(second: boolean) -> uint8[4,4] {
                define old: Item = { x: 9 };
                define a: Item = { x: 4 };
                define h: Holder = { view: old };
                if (true) { define inner: Item; choose(h, a, inner, second); }
                return h.view.x;
            }
        """
        compile_source(source.replace("choose(h, a, inner, second)", "choose(h, a, inner, false)"))
        with self.assertRaisesRegex(LanError, "outlive its target storage"):
            compile_source(source)

    def test_conditional_rebind_checks_donor_lifetime_on_possible_write(self):
        source = TYPES + """
            define set(h: Holder, donor: Item, b: boolean) { if (b) { h.view = donor; } }
            define use(b: boolean) -> uint8[9,9] {
                define old: Item = { x: 9 };
                define h: Holder = { view: old };
                if (true) { define inner: Item; set(h, inner, b); }
                return h.view.x;
            }
        """
        compile_source(source.replace("set(h, inner, b)", "set(h, inner, false)"))
        with self.assertRaisesRegex(LanError, "outlive its target storage"):
            compile_source(source)

    def test_borrowed_field_rebind_is_visible_to_a_loop_oracle(self):
        _, trace = compile_source(TYPES + """
            define set(h: Holder, i: Item) { h.view = i; }
            define f() -> uint8[4,4] {
                define old: Item = { x: 9 };
                define donor: Item = { x: 4 };
                define h: Holder = { view: old };
                define keep_running = true;
                /** @oracle @transition => { state(keep_running) := {0}; state(h.view.target) := {donor}; } */
                while (keep_running) { set(h, donor); keep_running = false; }
                return h.view.x;
            }
        """)
        self.assertIn("verified transition", trace)
        self.assertIn("return {4}", trace)

    def test_borrowed_field_rebind_crosses_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, InterfaceError, ModuleGraph, decode_mutations
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "setter.lang"
            source = TYPES + "define set(h: Holder, i: Item) { h.view = i; } export { Holder, Item, set };"
            compiler = Compiler(source, provider.name)
            _, table = ModuleGraph(provider, compiler, library=True).compile()
            self.assertIn('["@interface_rebind","view","1"]', table)
            (root / "setter.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
                import { Holder, Item, set } from "./setter.table";
                define use() -> uint8[4,4] {
                    define old: Item = { x: 9 };
                    define donor: Item = { x: 4 };
                    define h: Holder = { view: old };
                    set(h, donor);
                    return h.view.x;
                }
            """, root / "consumer.lang")
            self.assertIn("return {4} [entry]", trace)
            contract = compiler.functions["set"][1]
            malformed = (
                [[0, ["@interface_rebind", "view", "0"], None]],
                [[0, ["@interface_rebind", "missing", "1"], None]],
                [[0, ["@interface_rebind", "view", "01"], None]],
                [[0, ["@interface_rebind", "view", "1"], None], [0, ["view", "*", "x"], None]],
            )
            for summary in malformed:
                with self.subTest(summary=summary), self.assertRaisesRegex(InterfaceError, "invalid interface rebind"):
                    decode_mutations(summary, contract)

            guarded = TYPES + """
                define set(h: Holder, i: Item, b: boolean) { if (b) { h.view = i; } }
                export { Holder, Item, set };
            """
            compiler = Compiler(guarded, provider.name)
            _, table = ModuleGraph(provider, compiler, library=True).compile()
            self.assertIn('["@interface_rebind","view","1","2","1"]', table)
            (root / "setter.table").write_text(table, encoding="ascii")
            compile_source("""
                import { Holder, Item, set } from "./setter.table";
                define forward(b: boolean, h: Holder, i: Item) { set(h, i, b); }
                define use() -> uint8[4,4] {
                    define old: Item = { x: 9 };
                    define donor: Item = { x: 4 };
                    define h: Holder = { view: old };
                    forward(true, h, donor);
                    return h.view.x;
                }
            """, root / "consumer.lang")
            contract = compiler.functions["set"][1]
            for guard, expected in (("0", "1"), ("02", "1"), ("2", "2")):
                with self.subTest(guard=guard, expected=expected), self.assertRaises(InterfaceError):
                    decode_mutations([[0, ["@interface_rebind", "view", "1", guard, expected], None]], contract)

    def test_borrowed_record_with_interface_field_crosses_the_table_interface(self):
        from pathlib import Path as _Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, ModuleGraph
        with TemporaryDirectory(dir=_Path(__file__).resolve().parent) as directory:
            root = _Path(directory)
            provider = root / "provider.lang"
            source = (TYPES + "define read_view: contract(h: Holder) -> uint8 (h) {"
                              " return h.view.x; } export { Holder, read_view };")
            table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()[1]
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source(
                'import { Holder, read_view } from "./provider.table";'
                ' type View = interface { x: uint8 };'
                ' type Item = { x: uint8, y: uint8 };'
                ' define caller: contract() -> uint8 () {'
                ' define item: Item = { x: 5, y: 6 }; define holder: Holder = { view: item };'
                ' return read_view(holder); }',
                root / "consumer.lang",
            )
            self.assertIn("_read_view(lan_record_", c)

    def test_an_interface_write_crosses_the_table_interface(self):
        from pathlib import Path as _Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, ModuleGraph
        with TemporaryDirectory(dir=_Path(__file__).resolve().parent) as directory:
            root = _Path(directory)
            provider = root / "writer.lang"
            source = (TYPES + "define bump: contract(h: Holder) -> uint8 (h) {"
                              " h.view.x = 9; return 0; } export { Holder, bump };")
            table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()[1]
            self.assertIn('["view","*","x"]', table)
            (root / "writer.table").write_text(table, encoding="ascii")
            _, consumer_table = compile_source(
                'import { Holder, bump } from "./writer.table";'
                ' type View = interface { x: uint8 };'
                ' type Item = { x: uint8, y: uint8 };'
                ' define caller: contract() -> uint8 () {'
                ' define item: Item = { x: 7, y: 42 }; define holder: Holder = { view: item };'
                ' bump(holder); return item.x + item.y; }',
                root / "consumer.lang",
            )
            self.assertIn("apply interface write", consumer_table)
            self.assertIn("return {51} [entry]", consumer_table)

    def test_interface_fields_still_cannot_be_returned_or_owned(self):
        with self.assertRaisesRegex(LanError, "cannot be nested"):
            compile_source(TYPES + "type Deep = { inner: Holder }; define f(d: Deep) -> uint8 { return 0; }")
        with self.assertRaisesRegex(LanError, "local-only"):
            compile_source(TYPES + "define f() -> Holder { define i: Item; "
                                   "define h: Holder = { view: i }; return h; }")

    def test_interface_rebind_with_direct_scalar_writes_before_and_after(self):
        source = TYPES.replace("type Holder = { view: View };", """
            type Stats = { first: uint8, last: uint8 };
            type Holder = { view: View, stats: Stats };
        """) + """
            define set(h: Holder, i: Item) {
                h.stats.first = 1;
                h.view = i;
                h.stats.last = 2;
            }
            define use() -> uint8[124,124] {
                define old: Item = { x: 9 };
                define donor: Item = { x: 4 };
                define h: Holder = { view: old };
                set(h, donor);
                return h.stats.first * 100 + h.stats.last * 10 + h.view.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("mutation param#0.view.target => param#1", trace)
        self.assertIn("mutation param#0.stats.first => {1}", trace)
        self.assertIn("mutation param#0.stats.last => {2}", trace)
        self.assertIn("return {124} [entry]", trace)

    def test_interface_rebind_with_scalar_writes_on_aliased_arguments(self):
        _, trace = compile_source(TYPES.replace("view: View", "view: View, mark: uint8") + """
            define set(h: Holder, i: Item, mirror: Holder) {
                h.mark = 2;
                h.view = i;
                mirror.mark = 4;
            }
            define use() -> uint8[44,44] {
                define old: Item = { x: 9 };
                define donor: Item = { x: 4 };
                define h: Holder = { view: old };
                set(h, donor, h);
                return h.mark * 10 + h.view.x;
            }
        """)
        self.assertIn("return {44} [entry]", trace)

    def test_mixed_rebind_preserves_the_original_address_of_an_interface_argument(self):
        source = """
            type Payload = { x: uint8 };
            type View = interface { node: Payload };
            type Item = { node: Payload };
            type Holder = { view: View };
            define set(h: Holder, donor: Item, p: Payload) { h.view = donor; p.x = 3; }
            define use() -> uint8[34,34] {
                define old: Item = { node: { x: 9 } };
                define donor: Item = { node: { x: 4 } };
                define h: Holder = { view: old };
                set(h, donor, h.view.node);
                return old.node.x * 10 + donor.node.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("return {34} [entry]", trace)
        with self.assertRaisesRegex(LanError, "cannot prove.*fits"):
            compile_source(source.replace("uint8[34,34]", "uint8[3,3]").replace(
                "old.node.x * 10 + donor.node.x", "donor.node.x"))

    def test_interface_rebind_and_scalar_writes_cross_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, InterfaceError, ModuleGraph, decode_mutations

        source = TYPES.replace("view: View", "view: View, mark: uint8") + """
            define set(h: Holder, i: Item) { h.view = i; h.view.x = 4; h.mark = 7; }
            export { Holder, Item, set };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "mixed.lang"
            compiler = Compiler(source, provider.name)
            _, table = ModuleGraph(provider, compiler, library=True).compile()
            self.assertIn('["@interface_rebind","view","1"]', table)
            self.assertIn('["mark"]', table)
            self.assertIn("mutation param#1.x => {4}", table)
            (root / "mixed.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
                import { Holder, Item, set } from "./mixed.table";
                define use() -> uint8[74,74] {
                    define old: Item = { x: 9 };
                    define donor: Item = { x: 5 };
                    define h: Holder = { view: old };
                    set(h, donor);
                    return h.mark * 10 + h.view.x;
                }
            """, root / "consumer.lang")
            self.assertIn("return {74} [entry]", trace)
            with self.assertRaises(InterfaceError):
                decode_mutations([[0, ["@interface_rebind", "view", "1"], None],
                                  [0, ["view", "*", "x"], None]], compiler.functions["set"][1])

    def test_unknown_interface_write_before_rebind_targets_the_old_value(self):
        _, trace = compile_source(TYPES + """
            define set(h: Holder, i: Item) { h.view.x = 3; h.view = i; }
            define use() -> uint8[34,34] {
                define old: Item = { x: 9 };
                define donor: Item = { x: 4 };
                define h: Holder = { view: old };
                set(h, donor);
                return old.x * 10 + h.view.x;
            }
        """)
        self.assertIn("mutation param#0.view.*.x => {3} (before rebind)", trace)
        self.assertIn("return {34} [entry]", trace)

    def test_before_rebind_write_crosses_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, ModuleGraph

        source = TYPES + """
            define set(h: Holder, i: Item) { h.view.x = 3; h.view = i; }
            export { Holder, Item, set };
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "before.lang"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            self.assertIn('["@interface_before","view","*","x"]', table)
            self.assertIn('["@interface_rebind","view","1"]', table)
            (root / "before.table").write_text(table, encoding="ascii")
            _, trace = compile_source("""
                import { Holder, Item, set } from "./before.table";
                define use() -> uint8[34,34] {
                    define old: Item = { x: 9 };
                    define donor: Item = { x: 4 };
                    define h: Holder = { view: old };
                    set(h, donor);
                    return old.x * 10 + h.view.x;
                }
            """, root / "consumer.lang")
            self.assertIn("apply interface write", trace)
            self.assertIn("return {34} [entry]", trace)

    def test_unknown_interface_write_after_rebind_applies_to_the_selected_value(self):
        _, trace = compile_source(TYPES + """
            define set(h: Holder, i: Item, b: boolean) {
                if (b) { h.view = i; }
                h.view.x = 3;
            }
            define use(b: boolean) -> uint8 {
                define old: Item = { x: 9 };
                define donor: Item = { x: 4 };
                define h: Holder = { view: old };
                set(h, donor, b);
                return old.x;
            }
        """)
        self.assertIn("mutation param#0.view.*.x => [0, 255] (after rebind)", trace)
        self.assertIn("apply interface write", trace)

    def test_mixed_rebind_rejects_writes_between_rebinds(self):
        statements = "if (b) { h.view = i; h.view.x = 3; } else { h.view.x = 4; h.view = i; }"
        with self.assertRaises(LanError):
            compile_source(TYPES + f"define set(h: Holder, i: Item, b: boolean) {{ {statements} }}")

    def test_mixed_rebind_resolves_a_scalar_write_through_the_new_binding(self):
        _, trace = compile_source(TYPES + """
            define set(h: Holder, i: Item) { h.view = i; h.view.x = 3; }
            define use() -> uint8[93,93] {
                define old: Item = { x: 9 };
                define donor: Item = { x: 4 };
                define h: Holder = { view: old };
                set(h, donor);
                return old.x * 10 + h.view.x;
            }
        """)
        self.assertIn("mutation param#1.x => {3}", trace)
        self.assertIn("return {93} [entry]", trace)

    def test_mixed_rebind_resolves_branch_writes_to_their_donors(self):
        source = """
            type Payload = { x: uint8 };
            type View = interface { node: Payload };
            type Item = { node: Payload };
            type Holder = { view: View };
            define set(h: Holder, a: Item, d: Item, b: boolean) {
                if (b) { h.view = a; h.view.node.x = 3; }
                else { h.view = d; h.view.node.x = 5; }
            }
            define use(b: boolean) -> uint8[9,9] {
                define old: Item = { node: { x: 9 } };
                define first: Item = { node: { x: 4 } };
                define second: Item = { node: { x: 8 } };
                define h: Holder = { view: old };
                set(h, first, second, b);
                return old.node.x;
            }
        """
        _, trace = compile_source(source)
        self.assertIn("mutation param#1.node.x", trace)
        self.assertIn("mutation param#2.node.x", trace)
        self.assertNotIn("mutation param#0.view.*", trace)
        self.assertIn("return {9} [entry]", trace)

    def test_call_writes_only_to_the_interface_target_selected_at_runtime(self):
        source = TYPES + """
            define set(h: Holder, i: Item, b: boolean) { if (b) { h.view = i; } }
            define touch(h: Holder) { h.view.x = 3; }
            define use(b: boolean) -> uint8[3,9] {
                define old: Item = { x: 9 };
                define donor: Item = { x: 4 };
                define h: Holder = { view: old };
                set(h, donor, b);
                touch(h);
                return old.x;
            }
        """
        compile_source(source)
        with self.assertRaisesRegex(LanError, "cannot prove.*fits"):
            compile_source(source.replace("uint8[3,9]", "uint8[3,3]"))

    def test_overlapping_possible_targets_keep_both_scalar_write_effects(self):
        source = """
            type Payload = { x: uint8[1,9] };
            type View = interface { node: Payload };
            type Item = { node: Payload };
            type Holder = { view: View };
            define choose(h: Holder, i: Item, b: boolean) { if (b) { h.view = i; } }
            define mutate(p: Payload, q: Payload) { p.x = 1; q.x = 2; }
            define use(a_selected: boolean, b_selected: boolean) -> uint8[1,9] {
                define a: Item = { node: { x: 7 } };
                define shared: Item = { node: { x: 9 } };
                define c: Item = { node: { x: 4 } };
                define left: Holder = { view: a };
                define right: Holder = { view: shared };
                choose(left, shared, a_selected);
                choose(right, c, b_selected);
                mutate(left.view.node, right.view.node);
                return shared.node.x;
            }
        """
        compile_source(source)
        with self.assertRaisesRegex(LanError, "cannot prove.*fits"):
            compile_source(source.replace("-> uint8[1,9]", "-> uint8[2,9]"))

    def test_local_heap_holder_keeps_offsets_and_targets_across_a_move(self):
        source = TYPES.replace("view: View", "view: View, second: View") + """
            type Other = { pad: uint16, x: uint8 };
            define use() -> uint16[959,959] {
                define old: Item = { x: 9 };
                define donor: Other = { x: 4 };
                define holder: Holder · {storage: heap} = { view: old, second: old };
                define saved: View = holder.view;
                define moved: Holder · {storage: heap} = holder;
                moved.view = donor;
                moved.view.x = 5;
                define backup = moved as: copy;
                free(moved);
                return saved.x * 100 + backup.view.x * 10 + backup.second.x;
            }
        """
        c, trace = compile_source(source)
        self.assertRegex(c, r"typedef struct \{ \w+ record; size_t offsets\[2\]; \} lan_interface_storage_\d+;")
        self.assertRegex(c, r"&lan_interface_heap_\d+->offsets\[1\]")
        self.assertIn("transfer entity", trace)
        self.assertIn("return {959} [entry]", trace)
        for expired in ("holder.view.x", "moved.view.x"):
            with self.subTest(expired=expired), self.assertRaisesRegex(LanError, "Tombstone"):
                compile_source(source.replace("saved.x * 100 + backup.view.x * 10 + backup.second.x", expired))
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(source.replace("free(moved);", "free(moved); moved.view = donor;"))

    def test_heap_holder_does_not_allow_a_donor_to_escape_its_block(self):
        source = TYPES + """
            define use() {
                define old: Item;
                define holder: Holder · {storage: heap} = { view: old };
                if (true) { define inner: Item; holder.view = inner; }
                free(holder);
            }
        """
        with self.assertRaisesRegex(LanError, "outlive its target storage"):
            compile_source(source)

    def test_heap_holder_owning_result_stays_local(self):
        source = TYPES + """
            define use() {
                define old: Item;
                define holder: Holder · {storage: heap} = { view: old };
                return holder;
            }
        """
        with self.assertRaisesRegex(LanError, "local-only"):
            compile_source(source)

    def test_shallow_comparison_requires_a_live_heap_interface_descriptor(self):
        source = """
            type Payload = { x: uint8 };
            type View = interface { node: Payload };
            type Item = { node: Payload };
            type Holder = { view: View };
            define use() -> boolean {
                define item: Item = { node: { x: 9 } };
                define holder: Holder · {storage: heap} = { view: item };
                free(holder);
                return holder.view.node == item.node;
            }
        """
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(source)


class OwnedInterfaceFieldParameterTests(unittest.TestCase):
    """A record with interface fields may cross a call as an owning parameter.

    The interface selects an object outside the allocation, so the callee can
    read through the field but cannot write through it or rebind it: neither
    change has a caller-visible summary.  The record itself is owned, so the
    callee releases it like any other heap parameter.
    """

    SOURCE = TYPES + """
        define peek(h: Holder · {storage: heap}) -> uint8 {
            define v = h.view.x;
            free(h);
            return v;
        }
        define use() -> uint8 {
            define item: Item = { x: 7, y: 1 };
            define holder: Holder · {storage: heap} = { view: item };
            return peek(holder);
        }
    """

    def test_owning_parameter_reads_through_the_interface_field(self):
        c, table = compile_source(self.SOURCE)
        self.assertRegex(table, r"heap id#\d+ h: Alive, owning parameter")
        self.assertRegex(table, r"interface id#\d+ view: opaque target")
        self.assertIn("owning parameters peek: param#0 h (free)", table)
        body = c[c.index("uint8_t lan_fn_peek"):]
        body = body[:body.index("\n}\n") + 3]
        self.assertIn(".f_view.data", body)
        self.assertIn("free((void *)", body)

    def test_owning_parameter_must_release_the_record(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'h' leaves scope before free"):
            compile_source(TYPES + """
                define peek(h: Holder · {storage: heap}) -> uint8 {
                    return h.view.x;
                }
            """)

    def test_owning_parameter_cannot_write_through_the_field(self):
        with self.assertRaisesRegex(
                LanError, "cannot write through the interface field of an owning parameter"):
            compile_source(TYPES + """
                define bump(h: Holder · {storage: heap}) -> void {
                    h.view.x = 9;
                    free(h);
                }
            """)

    def test_owning_parameter_cannot_rebind_the_field(self):
        with self.assertRaisesRegex(
                LanError, "cannot rebind the interface field of an owning parameter"):
            compile_source(TYPES + """
                define swap(h: Holder · {storage: heap}, other: Item) -> void {
                    h.view = other;
                    free(h);
                }
            """)

    def test_a_heap_field_can_be_copied_before_the_release(self):
        # The copy carries its own offset table, so it stays valid after the
        # callee frees the record it was read from.
        for annotation in ("", ": View"):
            with self.subTest(annotation=annotation):
                c, _ = compile_source(TYPES + f"""
                    define peek(h: Holder · {{storage: heap}}) -> uint8 {{
                        define v{annotation} = h.view;
                        free(h);
                        return v.x;
                    }}
                """)
                body = c[c.index("uint8_t lan_fn_peek"):]
                body = body[:body.index("\n}\n") + 3]
                self.assertIn("size_t lan_offsets_", body)
                self.assertIn("lh_interface_rebind_", body)

    def test_a_stack_record_still_allows_the_copy(self):
        _, table = compile_source(TYPES + """
            define use(a: Item) -> uint8 {
                define holder: Holder = { view: a };
                define saved: View = holder.view;
                return saved.x;
            }
        """)
        self.assertIn("interface targets", table)

    def test_a_result_record_still_stays_local(self):
        with self.assertRaisesRegex(LanError, "local-only"):
            compile_source(TYPES + """
                define make(item: Item) -> Holder · {storage: heap} {
                    define holder: Holder · {storage: heap} = { view: item };
                    return holder;
                }
            """)

    def test_a_local_heap_record_keeps_rebinding(self):
        _, table = compile_source(TYPES + """
            define use(a: Item, b: Item) -> uint8 {
                define holder: Holder · {storage: heap} = { view: a };
                holder.view = b;
                define v = holder.view.x;
                free(holder);
                return v;
            }
        """)
        self.assertIn("interface targets", table)
