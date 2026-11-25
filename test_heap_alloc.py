import unittest

from lanc import LanError, compile_source


BUILD = """
    define build(n: uint8[1, 8]) -> uint8[..] · {len: [1, 8]} · {storage: heap} {
        define data: uint8[n] · {storage: heap};
        define i: uint8 = 0;
        while (i < n) {
            data[i] = i + 1;
            i += 1;
        }
        return data;
    }
"""


class DynamicLengthArrayTest(unittest.TestCase):
    def test_length_comes_from_the_parameter_at_run_time(self):
        c, table = compile_source(BUILD + """
            define use(n: uint8[1, 8]) -> uint8 {
                define items: uint8[..] · {len: [1, 8]} · {storage: heap} = build(n);
                define v = items[0];
                free(items);
                return v;
            }
        """)
        # The element count is the parameter itself, not a compile-time constant.
        self.assertRegex(c, r"lh_heap_array_\d+\(\(size_t\)\(lv_1_n\), sizeof\(uint8_t\)\)")
        self.assertIn("heap id#2 data: Alive, len [1, 8]", table)

    def test_allocation_helper_rejects_a_failed_allocation(self):
        c, _ = compile_source(BUILD + """
            define f() -> uint8 {
                define data: uint8[2] · {storage: heap};
                define v = data[0];
                free(data);
                return v;
            }
        """)
        start = c.index("static inline lan_slice lh_heap_array_")
        helper = c[start:c.index("\n}\n", start) + 3]
        self.assertIn("void *data = calloc(len, size);", helper)
        # calloc reports overflow and exhaustion the same way, so there is one
        # trap and no partially initialized object to release.
        self.assertIn("__builtin_trap();", helper)
        self.assertNotIn("return NULL", helper)

    def test_constant_length_stays_a_constant(self):
        c, _ = compile_source("""
            define f() -> uint8 {
                define data: uint8[3] · {storage: heap};
                define v = data[0];
                free(data);
                return v;
            }
        """)
        self.assertRegex(
            c, r"lh_heap_array_\d+\(\(size_t\)\(\(\(lan_int\)UINT64_C\(3\)\)\), sizeof\(uint8_t\)\)")

    def test_a_zero_length_allocation_is_rejected(self):
        # calloc(0, n) may return NULL, which the helper would treat as failure,
        # so the length range has to start at one.
        with self.assertRaisesRegex(LanError, r"cannot prove heap length \[0, 8\] fits"):
            compile_source("""
                define build(n: uint8[0, 8]) -> uint8[..] · {len: [0, 8]} · {storage: heap} {
                    define data: uint8[n] · {storage: heap};
                    return data;
                }
            """)

    def test_only_scalar_elements_can_be_allocated_in_bulk(self):
        # A record array would need a per-element constructor, so the language
        # allocates records one at a time instead.
        with self.assertRaisesRegex(LanError, "expected a scalar type"):
            compile_source("""
                type Node = { value: uint8, next: Node | null };
                define f(n: uint8[1, 4]) -> uint8 {
                    define nodes: Node[n] · {storage: heap};
                    define v = nodes[0].value;
                    free(nodes);
                    return v;
                }
            """)

    def test_a_fixed_capacity_can_be_returned_as_a_bounded_slice(self):
        _, table = compile_source("""
            define make() -> uint8[..] · {len: [1, 8]} · {storage: heap} {
                define array: uint8[4] · {storage: heap};
                array[0] = 1;
                return array;
            }
            define use() -> uint8 {
                define items: uint8[..] · {len: [1, 8]} · {storage: heap} = make();
                define v = items[0];
                free(items);
                return v;
            }
        """)
        self.assertIn("heap id#1 array: Alive, len {4}", table)
        self.assertIn("id#2 items: owned result, Alive, len [1, 8]", table)

    def test_an_unreleased_allocation_is_a_leak(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'data' leaves scope before free"):
            compile_source("""
                define f(n: uint8[1, 8]) -> uint8 {
                    define data: uint8[n] · {storage: heap};
                    return 0;
                }
            """)

    def test_an_exit_path_without_free_is_a_leak(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'items' leaves scope before free"):
            compile_source(BUILD + """
                define use(n: uint8[1, 8]) -> uint8 {
                    define items: uint8[..] · {len: [1, 8]} · {storage: heap} = build(n);
                    if (n > 4) { return 0; }
                    free(items);
                    return 0;
                }
            """)

    def test_an_owning_parameter_must_be_freed(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'items' leaves scope before free"):
            compile_source("""
                define drop: contract(items: uint8[..] · {len: [1, 8]} · {storage: heap})
                            -> uint8 (items) {
                    return 0;
                }
            """)

    def test_owning_parameter_accepts_a_variable_length_argument(self):
        _, table = compile_source(BUILD + """
            define drop: contract(items: uint8[..] · {len: [1, 8]} · {storage: heap})
                        -> uint8 (items) {
                define first = items[0];
                free(items);
                return first;
            }
            define use(n: uint8[1, 8]) -> uint8 {
                define items: uint8[..] · {len: [1, 8]} · {storage: heap} = build(n);
                return drop(items);
            }
        """)

    def test_an_allocation_inside_a_loop_is_freed_each_round(self):
        c, table = compile_source("""
            define churn(rounds: uint8[1, 4]) -> uint8 {
                define acc: uint8 = 0;
                define r: uint8 = 0;
                while (r < rounds && acc < 200) {
                    define data: uint8[3] · {storage: heap};
                    data[0] = 1;
                    acc += data[0];
                    free(data);
                    r += 1;
                }
                return acc;
            }
        """)
        body = c[c.index("uint8_t lan_fn_churn"):]
        body = body[:body.index("\n}\n") + 3]
        # The allocation and its release both sit inside the loop body.
        self.assertIn("lh_heap_array_1", body)
        self.assertIn("free((void *)lv_", body)

    def test_a_chain_built_from_a_run_time_count(self):
        _, table = compile_source("""
            type Node = { value: uint8, next: Node | null };
            define list_total(n: uint8[0, 6]) -> uint16 {
                define head: Node · {storage: heap} = { value: 0 };
                define tail: Node · {pass: alias} = head;
                define i: uint8 = 0;
                while (i < n) {
                    define node: Node · {storage: heap} = { value: i + 1 };
                    tail.next = node;
                    tail = node;
                    i += 1;
                }
                assert chain(head: head -> tail: null);
                define total: uint16 = 0;
                define cursor: Node · {pass: alias} | null = head;
                define steps: uint8 = 0;
                while (cursor != null && steps < 8 && total < 2000) {
                    total += cursor.value as uint16;
                    cursor = cursor.next;
                    steps += 1;
                }
                free(head);
                return total;
            }
        """)
        self.assertIn("chain length 1 + 1*rounds (fresh allocation each iteration)", table)


if __name__ == "__main__":
    unittest.main()
