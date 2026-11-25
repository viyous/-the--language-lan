import re
import unittest

from lanc import LanError, compile_source
from shape_domain import CHAIN, DAG, GRAPH, LASSO, REPEAT, TREE, LinkGraph, ShapeOverflow


NODE = "type Node = { value: uint8, next: Node | null };"

ALTERNATING = """
    type Odd = { value: uint8, next: Even | null };
    type Even = { value: uint8, next: Odd | null };
"""

BUILT_CHAIN = NODE + """
    define build() -> uint8 {
        define a: Node = { value: 1 };
        define b: Node = { value: 2, next: a };
        define c: Node = { value: 3, next: b };
        assert chain(head: c -> tail: null);
        return c.value;
    }
"""

LOOP = NODE + """
    define loop() -> uint8 {
        define a: Node = { value: 1 };
        define b: Node = { value: 2, next: a };
        a.next = b;
        assert chain(head: a -> tail: null);
        return 0;
    }
"""

PARAMETER = NODE + """
    define walk(h: Node) -> uint8 {
        assert chain(head: h -> tail: null);
        return 0;
    }
"""

LOOP_CHAIN = NODE + """
    define build(n: uint8) -> uint8 {
        define head: Node · {storage: heap} = { value: 0 };
        define tail: Node · {pass: alias} = head;
        define i = n;
        while (i > 0) {
            define node: Node · {storage: heap} = { value: 1 };
            tail.next = node;
            tail = node;
            i -= 1;
        }
        assert chain(head: head -> tail: null);
        define v = head.value;
        free(head);
        return v;
    }
"""

FOUR_PER_TURN = NODE + """
    define build(n: uint8) -> uint8 {
        define head: Node · {storage: heap} = { value: 0 };
        define tail: Node · {pass: alias} = head;
        define i = n;
        while (i > 0) {
            define a: Node · {storage: heap} = { value: 1 };
            define b: Node · {storage: heap} = { value: 2 };
            define c: Node · {storage: heap} = { value: 3 };
            define d: Node · {storage: heap} = { value: 4 };
            tail.next = a;
            a.next = b;
            b.next = c;
            c.next = d;
            tail = d;
            i -= 1;
        }
        assert chain(head: head -> tail: null);
        define v = head.value;
        free(head);
        return v;
    }
"""

FREE_BOUNDARIES = NODE + """
    type OtherLink = { value: uint8, other: OtherLink | null };
    define free_stack_tail(flag: boolean) -> uint8[7,7] {
        define tail: Node = { value: 7 };
        define head: Node · {storage: heap};
        define next: Node · {storage: heap};
        head.next = next;
        if (flag) { next.next = tail; }
        free(head);
        return tail.value;
    }
    define free_other_link() -> void {
        define node: OtherLink · {storage: heap};
        free(node);
    }
    define make_array() -> uint8[..] · {storage: heap} {
        define array: uint8[2] · {storage: heap};
        return array;
    }
    define free_array_result() -> void { free(make_array()); }
"""

RECURSIVE_WALK = """
    define last(h: Node) -> uint8 {
        if (h.next == null) { return h.value; }
        return last(h.next);
    }
"""

RELEASE_ONE_INSTANCE = LOOP_CHAIN.replace("define build(n: uint8)", "define release_one()").replace(
    "define i = n;", "define i: uint8 = 3;").replace("define v = head.value;", """
        define middle: Node · {pass: alias} = head.next.next;
        head.next.next = middle.next;
        middle.next = null;
        free(middle);
        define v = head.next.next.value;
    """)

DYNAMIC_RELEASE = RELEASE_ONE_INSTANCE.replace("release_one()", "release_one(n: uint8[3,255])").replace(
    "i: uint8 = 3;", "i: uint8 = n;")
GUARDED_RELEASE = DYNAMIC_RELEASE.replace("n: uint8[3,255]", "n: uint8").replace(
    "define middle: Node", "define v: uint8 = 0; if (head.next != null && head.next.next != null) {"
    " define middle: Node").replace("free(middle);", "v = middle.value; free(middle); }").replace(
    "define v = head.next.next.value;", "")
REPEAT_ADDRESS_COMPARISON = LOOP_CHAIN.replace("n: uint8)", "n: uint8[2,255])").replace(
    "define v = head.value;", """
        define first: Node · {pass: alias} = head.next;
        define v: uint8 = 0;
        if (first != tail) { v = 1; }
    """)
INDEXED_RELEASE = LOOP_CHAIN.replace("n: uint8)", "n: uint8, index: uint8)").replace(
    "assert chain", """
        tail = head;
        define remaining = index;
        define searching = true;
        while (searching && (tail.next != null && tail.next.next != null)) {
            if (remaining == 0) {
                define victim: Node · {pass: alias} = tail.next;
                tail.next = victim.next;
                victim.next = null;
                free(victim);
                searching = false;
            } else {
                tail = tail.next;
                remaining -= 1;
            }
        }
        assert chain""").replace("define v = head.value;", "define v = tail.value;")


class MetaGraphTests(unittest.TestCase):
    def chain(self, *nodes):
        graph = LinkGraph()
        for index, node in enumerate(nodes):
            graph.link(node, nodes[index + 1:index + 2])
        return graph

    def test_one_traversal_yields_length_and_per_node_distance(self):
        graph = self.chain(0, 1, 2, 3)
        summary = graph.summary(0)
        self.assertEqual(summary.shape.kind, CHAIN)
        self.assertEqual(summary.shape.length, 4)
        self.assertEqual(summary.distances, {3: 1, 2: 2, 1: 3, 0: 4})
        self.assertEqual(graph.distance(2), 2)
        self.assertIs(graph.summary(0), summary)

    def test_cache_is_dropped_by_a_structural_change(self):
        graph = self.chain(0, 1, 2)
        first = graph.summary(0)
        graph.link(0, (7,))
        second = graph.summary(0)
        self.assertIsNot(first, second)
        self.assertEqual(second.shape.kind, TREE)
        self.assertEqual(second.shape.nodes, 4)
        self.assertEqual(graph.closure(7), (7,))

    def test_cycle_is_reported_as_a_lasso(self):
        graph = LinkGraph()
        for index in range(4):
            graph.link(index, ((index + 1) % 4,))
        summary = graph.summary(0)
        self.assertEqual(summary.shape.kind, LASSO)
        self.assertEqual(summary.shape.period, 4)
        self.assertIsNone(summary.shape.length)
        self.assertEqual(summary.cycle, (0, 1, 2, 3))

    def test_branching_without_a_cycle_is_a_tree_or_dag(self):
        tree = LinkGraph()
        tree.link(0, (1, 2))
        tree.link(1, (3, 4))
        tree.link(2, (5,))
        self.assertEqual(tree.summary(0).shape.kind, TREE)
        dag = LinkGraph()
        dag.link(0, (1, 2))
        dag.link(1, (3,))
        dag.link(2, (3,))
        self.assertEqual(dag.summary(0).shape.kind, DAG)

    def test_a_cycle_with_branching_is_a_graph(self):
        graph = LinkGraph()
        graph.link(0, (1, 2))
        graph.link(1, (0,))
        self.assertEqual(graph.summary(0).shape.kind, GRAPH)

    def test_gap_is_the_lower_bound_between_two_nodes(self):
        graph = LinkGraph()
        graph.link(0, (1,), weight=1)
        graph.link(1, (2,), weight=4)
        graph.link(2, (3,), weight=1)
        self.assertEqual(graph.gap(0, 3), 6)
        self.assertEqual(graph.gap(0, 0), 0)
        self.assertIsNone(graph.gap(3, 0))

    def test_attachment_cycle_is_detected_by_reachability(self):
        graph = self.chain(0, 1, 2)
        self.assertFalse(graph.attach_creates_cycle(0, 2))
        self.assertTrue(graph.attach_creates_cycle(2, 0))
        self.assertFalse(graph.attach_creates_cycle(2, 9))

    def test_overflow_is_explicit(self):
        graph = LinkGraph(limit=3)
        graph.link(0, (1,))
        graph.link(1, (2,))
        graph.link(2, (3,))
        with self.assertRaises(ShapeOverflow):
            graph.summary(0)

    def test_a_fresh_turn_is_a_repeated_chain_not_a_cycle(self):
        graph = LinkGraph()
        graph.link(0, (1,))
        graph.link(1, (2,), fresh=True)
        graph.link(2, (3,), fresh=True)
        graph.link(3, (4,), fresh=True)
        graph.link(4, (1,), fresh=True)
        shape = graph.summary(0).shape
        self.assertEqual(shape.kind, REPEAT)
        self.assertEqual((shape.prefix, shape.period), (1, 4))
        self.assertEqual(shape.length_for(0), 1)
        self.assertEqual(shape.length_for(10), 41)

    def test_materializing_the_repeated_section_yields_the_exact_chain(self):
        graph = LinkGraph()
        graph.link(0, (1,))
        graph.link(1, (2,), fresh=True)
        graph.link(2, (3,), fresh=True)
        graph.link(3, (4,), fresh=True)
        graph.link(4, (1,), fresh=True)
        self.assertEqual(graph.summary(0).shape.length_for(10), 41)
        expanded = graph.materialized(0, 10).summary(0).shape
        self.assertEqual(expanded.kind, CHAIN)
        self.assertEqual(expanded.length, 41)
        prefix_only = graph.materialized(0, 0).summary(0).shape
        self.assertEqual((prefix_only.kind, prefix_only.length), (CHAIN, 1))

    def test_materializing_a_plain_chain_is_identity(self):
        graph = self.chain(0, 1, 2)
        self.assertIs(graph.materialized(0, 5), graph)

    def test_a_partly_fresh_cycle_is_still_a_lasso(self):
        graph = LinkGraph()
        graph.link(0, (1,))
        graph.link(1, (2,))
        graph.link(2, (0,), fresh=True)
        self.assertEqual(graph.summary(0).shape.kind, LASSO)


class ChainDerivationTests(unittest.TestCase):
    def test_a_locally_built_chain_reports_its_derived_length(self):
        _, table = compile_source(BUILT_CHAIN)
        self.assertIn("chain length 3 derived from the link meta-graph", table)

    def test_a_cycle_is_rejected_with_the_loop(self):
        with self.assertRaises(LanError) as caught:
            compile_source(LOOP)
        self.assertIn("next links form a cycle", str(caught.exception))
        self.assertIn("->", str(caught.exception))

    def test_source_shape_annotations_are_rejected(self):
        for annotation in ("Node · {chain:true}", "Node[..] · {chain:2}", "Node · {graph:true}",
                           "Node · {prune:true}", "Node · {attach:1}"):
            source = NODE + f"""
                define f(h: {annotation}) -> uint8 {{ return 0; }}
            """
            with self.assertRaises(LanError) as caught:
                compile_source(source)
            self.assertIn("removed shape annotation", str(caught.exception))

    def test_an_unbounded_chain_parameter_keeps_no_static_bound(self):
        _, table = compile_source(PARAMETER)
        self.assertIn("chain length has no static upper bound", table)

    def test_closing_a_real_cycle_after_the_loop_is_still_rejected(self):
        source = LOOP_CHAIN.replace("assert chain(head: head -> tail: null);",
                                    "tail.next = head; assert chain(head: head -> tail: null);")
        with self.assertRaisesRegex(LanError, "rewrite a link into or inside a symbolic repeated allocation group"):
            compile_source(source)
        with self.assertRaisesRegex(LanError, "next links form a cycle"):
            compile_source(source.replace("define i = n;", "define i: uint8 = 2;"))

    def test_a_loop_built_chain_reports_one_fresh_node_per_round(self):
        _, table = compile_source(LOOP_CHAIN)
        self.assertIn("chain length 1 + 1*rounds (fresh allocation each iteration)", table)

    def test_a_loop_that_allocates_four_nodes_per_round_reports_the_period(self):
        _, table = compile_source(FOUR_PER_TURN)
        self.assertIn("chain length 1 + 4*rounds (fresh allocation each iteration)", table)

    def test_free_labels_special_and_repeated_vertices(self):
        _, table = compile_source(LOOP_CHAIN)
        self.assertIn("Tombstone (special vertex)", table)
        self.assertIn("Tombstone (id set, one node per round)", table)

    def test_free_releases_the_whole_chain_in_one_pass(self):
        c, _ = compile_source(LOOP_CHAIN)
        self.assertIn("while (lan_free_", c)
        self.assertIn("->f_next", c)

    def test_known_loop_iterations_have_distinct_allocations(self):
        _, table = compile_source(RELEASE_ONE_INSTANCE)
        self.assertIn("materialize chain allocations across 3 loop iterations", table)
        releases = [line for line in table.splitlines() if "node: Tombstone (special vertex)" in line]
        self.assertEqual(len(releases), 3)
        self.assertEqual(len(set(releases)), 3)
        self.assertIn("return {1}", table)
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(RELEASE_ONE_INSTANCE.replace("define v = head.next.next.value;", "define v = middle.value;"))

    def test_loop_body_may_initialize_a_heap_field_from_the_counter(self):
        source = NODE + """
            define build(n: uint8[0,8]) -> uint8 {
                define head: Node · {storage: heap} = { value: 0 };
                define tail: Node · {pass: alias} = head;
                define i: uint8 = 0;
                while (i < n) {
                    define node: Node · {storage: heap} = { value: i };
                    tail.next = node;
                    tail = node;
                    i += 1;
                }
                free(head);
                return 0;
            }
        """
        _, table = compile_source(source)
        self.assertIn("loop fixed point", table)

    def test_small_ascending_allocation_loop_is_materialized(self):
        source = RELEASE_ONE_INSTANCE.replace("define i: uint8 = 3;", "define i: uint8 = 0;")
        source = source.replace("while (i > 0)", "while (i < 3)").replace("i -= 1;", "i += 1;")
        _, table = compile_source(source)
        self.assertIn("materialize chain allocations across 3 loop iterations", table)

    def test_materialized_loop_checks_counter_overflow(self):
        source = RELEASE_ONE_INSTANCE.replace("define i: uint8 = 3;", "define i: uint8 = 254;")
        source = source.replace("while (i > 0)", "while (i < 255)").replace("i -= 1;", "i += 2;")
        with self.assertRaisesRegex(LanError, r"cannot prove \{256\} fits"):
            compile_source(source)

    def test_dynamic_chain_can_release_an_individually_tracked_middle(self):
        for source in (DYNAMIC_RELEASE, RELEASE_ONE_INSTANCE.replace("uint8 = 3;", "uint8 = 100;")):
            with self.subTest(source=source):
                c, table = compile_source(source)
                self.assertIn("materialize up to 3 chain prefix iterations", table)
                self.assertIn("id set, one node per round", table)
                self.assertIn("return {1}", table)
                self.assertIn("lan_loop_head_", c)
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(DYNAMIC_RELEASE.replace("define v = head.next.next.value;", "define v = middle.value;"))

    def test_dynamic_chain_with_zero_or_one_iteration_keeps_short_paths(self):
        _, table = compile_source(GUARDED_RELEASE)
        self.assertIn("return {0}", table)
        self.assertIn("return {1}", table)

    def test_release_by_runtime_index_preserves_live_chain_and_cursor(self):
        c, table = compile_source(INDEXED_RELEASE)
        self.assertIn("release indexed interior chain node; repeated suffix stays Alive", table)
        self.assertIn("chain length 1 + 1*rounds", table)
        self.assertRegex(c, r"free\(\(void \*\)lan_removed_\d+_\d+\);")

    def test_runtime_index_release_rejects_a_live_alias_into_repeated_chain(self):
        source = INDEXED_RELEASE.replace("n: uint8,", "n: uint8[1,255],").replace(
            "tail = head;", "define saved: Node · {pass: alias} = tail; tail = head;")
        with self.assertRaisesRegex(LanError, "another reference into the repeated chain"):
            compile_source(source)

    def test_runtime_index_release_can_remove_the_tail_or_only_member(self):
        source = INDEXED_RELEASE.replace(" && tail.next.next != null", "").replace(
            "define v = tail.value;", """
                define v = tail.value;
                if (head.next != null) { v = head.next.value; }
            """)
        c, table = compile_source(source)
        self.assertIn("release indexed chain node; remaining family is live or empty", table)
        self.assertIn("chain length 1 derived from the link meta-graph", table)
        self.assertRegex(c, r"while \(lv_\d+_tail->f_next\) \{")

    def test_runtime_index_release_requires_traversal_guards_and_unit_step(self):
        for pattern, replacement in (("tail.next != null && tail.next.next != null", "true"),
                                     ("remaining -= 1;", "remaining -= 2;")):
            with self.subTest(pattern=pattern), self.assertRaises(LanError):
                compile_source(INDEXED_RELEASE.replace(pattern, replacement))

    def test_free_rejects_partial_release_through_a_symbolic_tail(self):
        for expression in ("free(tail);", "if (tail.next == null) { free(tail); }"):
            with self.subTest(expression=expression), self.assertRaisesRegex(LanError, "symbolic repeated allocation group"):
                compile_source(LOOP_CHAIN.replace("free(head);", f"{expression} free(head);"))

    def test_addresses_of_two_instances_from_one_site_can_differ(self):
        _, table = compile_source(REPEAT_ADDRESS_COMPARISON)
        self.assertIn("return {1}", table)

    def test_null_tail_does_not_make_other_repeated_links_null(self):
        source = LOOP_CHAIN.replace("n: uint8)", "n: uint8[2,255])").replace(
            "define v = head.value;", """
                define v: uint8 = 0;
                if (tail.next == null && head.next.next != null) { v = 1; }
            """)
        _, table = compile_source(source)
        self.assertIn("return {1}", table)

    def test_write_through_tail_does_not_narrow_all_repeated_nodes(self):
        source = LOOP_CHAIN.replace("n: uint8)", "n: uint8[2,255])").replace(
            "define v = head.value;", "tail.value = 7; define v = head.next.value;")
        compile_source(source)
        with self.assertRaisesRegex(LanError, "cannot prove.*fits"):
            compile_source(source.replace("-> uint8 {", "-> uint8[7,7] {"))

    def test_scalar_guard_on_tail_does_not_narrow_an_earlier_node(self):
        source = LOOP_CHAIN.replace("n: uint8)", "n: uint8[2,255])").replace("{ value: 1 }", "{ value: i }").replace(
            "define v = head.value;", """
                define v = head.value;
                if (tail.value == 1) { define known: uint8[1,1] = head.next.value; v = known; }
            """)
        with self.assertRaisesRegex(LanError, "cannot prove.*fits"):
            compile_source(source)

    def test_free_tombstones_every_node_of_the_chain(self):
        source = NODE + """
            define f() -> uint8 {
                define a: Node · {storage: heap} = { value: 1 };
                define b: Node · {storage: heap} = { value: 2 };
                a.next = b;
                free(a);
                return b.value;
            }
        """
        with self.assertRaises(LanError) as caught:
            compile_source(source)
        self.assertIn("may be a Tombstone", str(caught.exception))

    def test_free_rejects_a_cyclic_chain(self):
        source = NODE + """
            define f() -> uint8 {
                define a: Node · {storage: heap} = { value: 1 };
                define b: Node · {storage: heap} = { value: 2 };
                a.next = b;
                b.next = a;
                free(a);
                return 0;
            }
        """
        with self.assertRaises(LanError) as caught:
            compile_source(source)
        self.assertIn("break it into a chain first", str(caught.exception))

    def test_free_walks_a_doubly_linked_list_along_its_spine(self):
        source = """
            type Node = { value: uint8, next: Node | null, prev: Node | null };
            define build() -> uint8 {
                define a: Node · {storage: heap} = { value: 1 };
                define b: Node · {storage: heap} = { value: 2 };
                define c: Node · {storage: heap} = { value: 3 };
                a.next = b;
                b.prev = a;
                b.next = c;
                c.prev = b;
                define v = a.value + b.value + c.value;
                free(a);
                return v;
            }
        """
        c, table = compile_source(source)
        # `prev` is a back pointer: only `next` owns nodes, so one walk covers
        # the list and no visited set or recursive helper is needed.
        self.assertIn("free spine over f_next: 3 nodes", table)
        self.assertIn("->f_next", c)
        self.assertNotIn("lh_free_tree", c)

    def test_free_walks_a_doubly_linked_list_built_by_a_loop(self):
        source = """
            type Node = { value: uint8, next: Node | null, prev: Node | null };
            define build(n: uint8[0, 8]) -> void {
                define head: Node · {storage: heap};
                define tail: Node · {pass: alias} = head;
                define i = n;
                while (i > 0) {
                    define node: Node · {storage: heap};
                    node.prev = tail;
                    tail.next = node;
                    tail = node;
                    i -= 1;
                }
                free(head);
            }
        """
        _, table = compile_source(source)
        self.assertIn("free spine over f_next", table)

    def test_free_rejects_a_back_pointer_that_skips_a_node(self):
        source = """
            type Node = { value: uint8, next: Node | null, prev: Node | null };
            define build() -> void {
                define a: Node · {storage: heap};
                define b: Node · {storage: heap};
                define c: Node · {storage: heap};
                a.next = b;
                b.prev = a;
                b.next = c;
                c.prev = a;
                free(a);
            }
        """
        with self.assertRaisesRegex(LanError, "break it into a chain first"):
            compile_source(source)

    def test_free_releases_a_tree_in_post_order(self):
        source = """
            type Tree = { value: uint8, left: Tree | null, right: Tree | null };
            define build() -> uint8 {
                define root: Tree · {storage: heap} = { value: 1 };
                define a: Tree · {storage: heap} = { value: 2 };
                define b: Tree · {storage: heap} = { value: 3 };
                root.left = a;
                root.right = b;
                define total = root.value + a.value + b.value;
                free(root);
                return total;
            }
        """
        c, table = compile_source(source)
        self.assertIn("post-order walk over f_left, f_right", table)
        self.assertRegex(c, r"static void lh_free_tree_\d+\(lan_record_\w+ \*p\)")
        self.assertIn("p->f_left", c)
        self.assertIn("p->f_right", c)

    def test_free_tombstones_every_branch_of_a_tree(self):
        source = """
            type Tree = { value: uint8, left: Tree | null, right: Tree | null };
            define build() -> uint8 {
                define root: Tree · {storage: heap} = { value: 1 };
                define a: Tree · {storage: heap} = { value: 2 };
                define b: Tree · {storage: heap} = { value: 3 };
                root.left = a;
                root.right = b;
                free(root);
                return b.value;
            }
        """
        with self.assertRaisesRegex(LanError, "may be a Tombstone"):
            compile_source(source)

    def test_free_rejects_a_tree_with_a_shared_node(self):
        source = """
            type Tree = { value: uint8, left: Tree | null, right: Tree | null };
            define build() -> void {
                define root: Tree · {storage: heap};
                define a: Tree · {storage: heap};
                root.left = a;
                root.right = a;
                free(root);
            }
        """
        with self.assertRaisesRegex(LanError, "freed twice"):
            compile_source(source)

    def test_free_releases_a_chain_that_alternates_between_record_types(self):
        source = ALTERNATING + """
            define f() -> uint8 {
                define a: Odd · {storage: heap} = { value: 1 };
                define b: Even · {storage: heap} = { value: 2 };
                define c: Odd · {storage: heap} = { value: 3 };
                a.next = b;
                b.next = c;
                define total = a.value + b.value + c.value;
                free(a);
                return total;
            }
        """
        c, table = compile_source(source)
        # One pass still releases every node: each step follows the link its
        # own record declares, so the walk crosses the two C record types.
        self.assertIn("free spine over f_next: 3 nodes", table)
        self.assertIn("2 typed steps, no repeat", table)
        self.assertRegex(c, r"lan_record_\w+ \*lan_free_1_step0 = lan_free_1->f_next;")
        self.assertRegex(c, r"free\(\(void \*\)lan_free_1_step1\);")

    def test_free_walks_a_repeated_section_whose_types_take_turns(self):
        source = ALTERNATING + """
            define f(n: uint8[0, 8]) -> uint8 {
                define head: Odd · {storage: heap} = { value: 0 };
                define tail: Odd · {pass: alias} = head;
                define i: uint8 = n;
                while (i > 0) {
                    define even: Even · {storage: heap} = { value: 2 };
                    define odd: Odd · {storage: heap} = { value: 1 };
                    even.next = odd;
                    tail.next = even;
                    tail = odd;
                    i -= 1;
                }
                define total = tail.value;
                free(head);
                return total;
            }
        """
        c, table = compile_source(source)
        # The repeated section alternates, so the loop body unrolls one type
        # cycle instead of following a single typed cursor.
        self.assertIn("1 fixed nodes then a type cycle of 2", table)
        self.assertRegex(c, r"lan_record_\w+ \*lan_free_\d+_turn = lan_free_\d+_fixed0;")
        self.assertRegex(c, r"free\(\(void \*\)lan_free_\d+_round1\);")
        self.assertNotIn("while (lan_free_1 &&", c)

    def test_free_follows_a_different_field_on_each_record_type(self):
        source = """
            type Odd = { value: uint8, next: Even | null };
            type Even = { value: uint8, after: Odd | null };
            define f() -> uint8 {
                define a: Odd · {storage: heap} = { value: 1 };
                define b: Even · {storage: heap} = { value: 2 };
                define c: Odd · {storage: heap} = { value: 3 };
                a.next = b;
                b.after = c;
                define total = a.value + b.value + c.value;
                free(a);
                return total;
            }
        """
        c, table = compile_source(source)
        self.assertIn("free spine over f_next, f_after", table)
        self.assertIn("->f_next;", c)
        self.assertIn("->f_after;", c)
        self.assertRegex(c, r"free\(\(void \*\)lan_free_1_step1\);")

    def test_free_releases_a_tree_whose_children_have_another_record_type(self):
        source = """
            type Leaf = { value: uint8, next: Leaf | null };
            type Branch = { value: uint8, back: Leaf | null };
            type Root = { value: uint8, left: Leaf | null, right: Branch | null };
            define f() -> uint8 {
                define root: Root · {storage: heap} = { value: 1 };
                define left: Leaf · {storage: heap} = { value: 2 };
                define right: Branch · {storage: heap} = { value: 3 };
                define back: Leaf · {storage: heap} = { value: 4 };
                right.back = back;
                root.left = left;
                root.right = right;
                define total = root.value + left.value + right.value + back.value;
                free(root);
                return total;
            }
        """
        c, table = compile_source(source)
        self.assertIn("free tree from id#1: post-order walk over f_left, f_right", table)
        helpers = re.findall(r"static void (lh_free_tree_\d+)\(lan_record_\w+ \*p\);", c)
        # One helper per record type, all declared up front because they call
        # each other in both directions.
        self.assertEqual(len(helpers), 3, helpers)
        self.assertEqual(len(set(helpers)), 3, helpers)
        self.assertIn("p->f_left", c)
        self.assertIn("p->f_back", c)

    def test_free_tombstones_nodes_of_another_record_type(self):
        source = ALTERNATING + """
            define f() -> uint8 {
                define a: Odd · {storage: heap} = { value: 1 };
                define b: Even · {storage: heap} = { value: 2 };
                a.next = b;
                free(a);
                return b.value;
            }
        """
        with self.assertRaisesRegex(LanError, "may be a Tombstone"):
            compile_source(source)

    def test_free_rejects_a_cycle_that_closes_through_the_walk_field(self):
        source = ALTERNATING + """
            define f() -> uint8 {
                define a: Odd · {storage: heap} = { value: 1 };
                define b: Even · {storage: heap} = { value: 2 };
                a.next = b;
                b.next = a;
                free(a);
                return 0;
            }
        """
        with self.assertRaisesRegex(LanError, "break it into a chain first"):
            compile_source(source)

    def test_free_rejects_a_mixed_chain_that_ends_at_stack_storage(self):
        source = ALTERNATING + """
            define f() -> uint8 {
                define tail: Odd = { value: 9 };
                define a: Odd · {storage: heap} = { value: 1 };
                define b: Even · {storage: heap} = { value: 2 };
                a.next = b;
                b.next = tail;
                define total = a.value;
                free(a);
                return total;
            }
        """
        with self.assertRaisesRegex(LanError, "free cannot stop a chain that changes record type"):
            compile_source(source)

    def test_free_rejects_a_mixed_tree_with_a_shared_child(self):
        source = """
            type Leaf = { value: uint8 };
            type Root = { value: uint8, left: Leaf | null, right: Leaf | null };
            define f() -> uint8 {
                define root: Root · {storage: heap} = { value: 1 };
                define leaf: Leaf · {storage: heap} = { value: 2 };
                root.left = leaf;
                root.right = leaf;
                free(root);
                return 0;
            }
        """
        with self.assertRaisesRegex(LanError, "freed twice"):
            compile_source(source)

    def test_free_rejects_a_cycle_between_allocations_in_the_same_iteration(self):
        source = NODE + """
            define f(n: uint8) -> void {
                define head: Node · {storage: heap};
                define running = n > 0;
                while (running) {
                    define a: Node · {storage: heap};
                    define b: Node · {storage: heap};
                    a.next = b;
                    b.next = a;
                    head.next = a;
                    running = false;
                }
                free(head);
            }
        """
        with self.assertRaisesRegex(LanError, "break it into a chain first"):
            compile_source(source)
        with self.assertRaisesRegex(LanError, "cannot prove the loop cursor advances to a fresh tail"):
            compile_source(LOOP_CHAIN.replace("tail.next = node;\n            tail = node;",
                                              "tail.next = node;\n            tail = node;\n"
                                              "            tail.next = node; running = false;")
                           .replace("while (i > 0)", "define running = i > 0;\n        while (running)")
                           .replace("assert chain(head: head -> tail: null);", "")
                           .replace("i -= 1;", ""))

    def test_generated_free_stops_before_stack_and_uses_the_actual_array_pointer(self):
        c, _ = compile_source(FREE_BOUNDARIES)
        self.assertRegex(c, r"while \(lan_free_\d+ && \(const void \*\)lan_free_\d+ != \(const void \*\)&\(lv_\d+_tail\)\)")
        other = c[c.index("void lan_fn_free_other_link(void) {"):c.index("lan_slice lan_fn_make_array(void) {")]
        self.assertIn("free((void *)", other)
        self.assertNotIn("f_next", other)
        self.assertIn("free((void *)((lan_fn_make_array()).data));", c)

    def test_chain_parameter_crosses_the_table_interface(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from lanc import Compiler, ModuleGraph
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = (NODE + "define walk(h: Node) -> uint8 {"
                             " assert chain(head: h -> tail: null); return 0; }"
                             " export { Node, walk };")
            table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()[1]
            self.assertIn('"chain":true', table)
            self.assertIn("inferred chain parameter walk.h", table)
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source(
                'import { Node, walk } from "./provider.table";'
                ' define caller: contract(h: Node) -> uint8 (h) { return walk(h); }',
                root / "consumer.lang",
            )
            self.assertIn("walk", c)

    def test_chain_assert_spans_two_record_types(self):
        source = ALTERNATING + """
            define f() -> uint8 {
                define a: Odd · {storage: heap} = { value: 1 };
                define b: Even · {storage: heap} = { value: 2 };
                define c: Odd · {storage: heap} = { value: 3 };
                a.next = b;
                b.next = c;
                assert chain(head: a -> tail: null);
                define total = a.value + b.value + c.value;
                free(a);
                return total;
            }
        """
        _, table = compile_source(source)
        # The chain is proved over the meta-graph, so the nodes it crosses may
        # have different record types as long as every one of them has `next`.
        self.assertIn("verify chain(head: id#1 -> tail: id#0)", table)
        self.assertIn("alive id#4; next in {id#7}", table)
        self.assertIn("chain length 3 derived from the link meta-graph", table)

    def test_chain_assert_counts_an_alternating_repeat(self):
        source = ALTERNATING + """
            define f(n: uint8[0, 8]) -> uint8 {
                define head: Odd · {storage: heap} = { value: 0 };
                define tail: Odd · {pass: alias} = head;
                define i: uint8 = n;
                while (i > 0) {
                    define even: Even · {storage: heap} = { value: 2 };
                    define odd: Odd · {storage: heap} = { value: 1 };
                    even.next = odd;
                    tail.next = even;
                    tail = odd;
                    i -= 1;
                }
                assert chain(head: head -> tail: null);
                define total = tail.value;
                free(head);
                return total;
            }
        """
        _, table = compile_source(source)
        self.assertIn("chain length 1 + 2*rounds", table)

    def test_chain_assert_rejects_a_tail_of_another_type_that_is_not_reached(self):
        source = """
            type Node = { value: uint8, next: Node | null };
            type Other = { value: uint8, next: Other | null };
            define f() -> uint8 {
                define a: Node · {storage: heap} = { value: 1 };
                define b: Other · {storage: heap} = { value: 2 };
                assert chain(head: a -> tail: b);
                define total = a.value;
                free(a);
                free(b);
                return total;
            }
        """
        with self.assertRaisesRegex(LanError, "does not reach the declared tail"):
            compile_source(source)

    def test_chain_assert_rejects_a_cycle_across_two_record_types(self):
        source = ALTERNATING + """
            define f() -> uint8 {
                define a: Odd · {storage: heap} = { value: 1 };
                define b: Even · {storage: heap} = { value: 2 };
                a.next = b;
                b.next = a;
                assert chain(head: a -> tail: null);
                define total = a.value;
                free(a);
                return total;
            }
        """
        with self.assertRaisesRegex(LanError, "next links form a cycle"):
            compile_source(source)

    def test_recursive_walk_uses_a_chain_built_by_while(self):
        source = LOOP_CHAIN.replace(NODE, NODE + RECURSIVE_WALK).replace(
            "define v = head.value;", "define v = last(head);")
        c, table = compile_source(source)
        self.assertIn("sct recursive chain descent proved: last", table)
        self.assertIn("chain length 1 + 1*rounds", table)
        self.assertIn("lan_fn_last(", c)

    def test_recursive_next_call_infers_a_read_only_chain(self):
        source = LOOP_CHAIN.replace(NODE, NODE + RECURSIVE_WALK).replace(
            "define v = head.value;", "define v = last(head);")
        _, table = compile_source(source)
        self.assertIn("inferred chain parameter last.h", table)
        self.assertIn("sct recursive chain descent proved: last", table)
        with self.assertRaisesRegex(LanError, "next links form a cycle"):
            compile_source(source + """
                define invalid() -> uint8 {
                    define a: Node;
                    a.next = a;
                    return last(a);
                }
            """)

    def test_recursive_chain_calls_still_require_nonnull_acyclic_inputs(self):
        with self.assertRaisesRegex(LanError, "requires an initializer proved non-null"):
            compile_source(NODE + RECURSIVE_WALK.replace(
                "if (h.next == null) { return h.value; }", ""))
        with self.assertRaisesRegex(LanError, "next links form a cycle"):
            compile_source(NODE + RECURSIVE_WALK + """
                define f() -> uint8 {
                    define a: Node;
                    a.next = a;
                    return last(a);
                }
            """)

    def test_chain_descent_can_span_a_mutually_recursive_group(self):
        source = NODE + """
            define a(h: Node) -> uint8 { return b(h); }
            define b(h: Node) -> uint8 {
                if (h.next == null) { return h.value; }
                return a(h.next);
            }
        """
        _, table = compile_source(source)
        self.assertIn("sct recursive chain descent proved: a, b", table)
        _, table = compile_source(source.replace("return a(h.next);", "return a(h);"))
        self.assertNotIn("sct recursive chain descent proved", table)

    def test_forwarding_a_node_unchanged_in_recursion_accepts_a_cycle(self):
        _, trace = compile_source(NODE + """
            define read_later(h: Node, n: uint8) -> uint8 {
                if (n == 0) { return h.value; }
                return read_later(h, n - 1);
            }
            define demo() -> uint8 {
                define head: Node;
                head.next = head;
                return read_later(head, 2);
            }
        """)
        self.assertNotIn("inferred chain parameter read_later.h", trace)

    def test_heap_cycles_without_a_surviving_root_leak(self):
        cycle = """
            define a: Node · {storage: heap};
            define b: Node · {storage: heap};
            a.next = b;
            b.next = a;
        """
        with self.assertRaisesRegex(LanError, "heap leak: 'a' leaves scope before free"):
            compile_source(NODE + "define leak() -> void {" + cycle + "}")
        for initial in ("2", "n"):
            with self.subTest(initial=initial):
                source = NODE + f"""
                    define leak(n: uint8) -> void {{
                        define i: uint8 = {initial};
                        while (i > 0) {{ {cycle} i -= 1; }}
                    }}
                """
                with self.assertRaisesRegex(LanError, "heap leak: 'a' leaves scope before free"):
                    compile_source(source)

    def test_symbolic_loop_cannot_overwrite_an_older_allocation(self):
        body = "tail.next = node;\n            tail = node;"
        direct = LOOP_CHAIN.replace(body, "head.next = node;")
        with self.assertRaisesRegex(LanError, "overwrite a link to live allocations from previous loop iterations"):
            compile_source(direct)
        for rebinding in ("", "if (first) { tail = node; first = false; }"):
            with self.subTest(rebinding=rebinding):
                source = LOOP_CHAIN.replace("define i = n;", "define i = n; define first = true;")
                source = source.replace("tail = node;", rebinding)
                with self.assertRaisesRegex(LanError, "loop cursor advances to a fresh tail"):
                    compile_source(source)
        with self.assertRaisesRegex(LanError, "loop cursor advances to a fresh tail"):
            compile_source(FOUR_PER_TURN.replace("tail = d;", "tail = a;"))

    def test_symbolic_links_keep_their_repeat_summary_on_noop_writes(self):
        source = LOOP_CHAIN.replace("define v = head.value;", "head.next = head.next; define v = head.value;")
        _, table = compile_source(source)
        self.assertIn("chain length 1 + 1*rounds", table)
        self.assertIn("id set, one node per round", table)
        for rewrite in ("tail.next = null;", "define other: Node; other.next = tail;",
                        "define other: Node = { next: tail };"):
            with self.subTest(rewrite=rewrite):
                with self.assertRaisesRegex(LanError, "rewrite a link into or inside a symbolic repeated allocation group"):
                    compile_source(source.replace("head.next = head.next;", rewrite))

    def test_live_outer_record_can_retain_several_heap_descendants(self):
        for initial in ("2", "n"):
            for stack in (False, True):
                with self.subTest(initial=initial, stack=stack):
                    source = FOUR_PER_TURN.replace("define i = n;", f"define i: uint8 = {initial};")
                    if stack:
                        source = source.replace("head: Node · {storage: heap}", "head: Node").replace(
                            "free(head);", """
                                if (head.next != null) {
                                    define first: Node · {pass: alias} = head.next;
                                    free(first);
                                }
                            """)
                    _, table = compile_source(source)
                    self.assertEqual("materialize chain allocations across 2 loop iterations" in table,
                                     initial == "2")

    def test_links_in_expired_stack_or_freed_heap_storage_cannot_retain_an_owner(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'child' leaves scope before free"):
            compile_source(NODE + """
                define leak() -> void {
                    define head: Node;
                    define child: Node · {storage: heap};
                    head.next = child;
                }
            """)
        source = """
            type Node = { value: uint8, other: Node | null };
            define leak() -> void {
                define head: Node · {storage: heap};
                define child: Node · {storage: heap};
                head.other = child;
                free(head);
                CHILD_RELEASE
            }
        """
        # A link owns its target whatever its name, so `free(head)` releases
        # `child` as well and a second release is caught as a double free.
        with self.assertRaisesRegex(LanError, "may already be freed or moved"):
            compile_source(source.replace("CHILD_RELEASE", "free(child);"))
        compile_source(source.replace("CHILD_RELEASE", ""))

if __name__ == "__main__":
    unittest.main()
