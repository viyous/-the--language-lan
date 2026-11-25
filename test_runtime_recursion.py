import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


DIRECT = """
    define countdown(n: uint8) -> uint8 {
        if (n == 0) { return 0; }
        n -= 1;
        return countdown(n);
    }
"""

MUTUAL = """
    define even(n: uint8) -> boolean {
        if (n == 0) { return true; }
        return odd(n);
    }
    define odd(n: uint8) -> boolean {
        if (n == 0) { return false; }
        return even(n - 1);
    }
"""

DEPENDENT = """
    define count_up(n: uint8) -> uint8 & >= n & <= n {
        if (n == 0) { return 0; }
        n -= 1;
        return count_up(n) + 1;
    }
"""

SCALAR_RESULTS = """
    enum Err: uint8 { Bad = 250 }
    type Result = uint8[0,4] | Err;
    define lookup(n: uint8, valid: boolean) -> Result {
        if (n == 0) {
            if (!valid) { return Err.Bad; }
            return 3;
        }
        return lookup(n - 1, valid)?;
    }
    define keep(n: uint8, x: float32) -> float32 {
        if (n == 0) { return x; }
        return keep(n - 1, x);
    }
"""


class RuntimeRecursionTests(unittest.TestCase):
    def test_direct_recursion_uses_entry_parameter_as_measure(self):
        c, table = compile_source(DIRECT)
        self.assertIn("sct recursive termination proved: countdown", table)
        self.assertIn("lan_fn_countdown(", c.split("lan_fn_countdown(", 2)[2])

    def test_mutual_recursion_accepts_weak_edge_followed_by_strict_edge(self):
        c, table = compile_source(MUTUAL)
        self.assertIn("sct recursive termination proved: even, odd", table)
        self.assertLess(c.index("lan_fn_odd("), c.index("lan_fn_even(", c.index("lan_fn_even(") + 1))

    def test_signed_recursion_and_separately_checked_helper(self):
        source = """
            define reset() -> int8 { return -5; }
            define down(n: int8) -> int8 {
                if (n <= -5) { return reset(); }
                return down(n - 1);
            }
        """
        _, table = compile_source(source)
        self.assertIn("sct recursive termination proved: down", table)

    def test_unproved_termination_is_not_a_contract_error(self):
        for source in (MUTUAL.replace("even(n - 1)", "even(n)"), DIRECT.replace("n -= 1", "n = n")):
            _, table = compile_source(source)
            self.assertNotIn("sct recursive termination proved", table)

    def test_invalid_contracts_fail(self):
        samples = (
            (DIRECT.replace("n == 0", "n == 1"), "cannot prove"),
            (DIRECT.replace("return 0;", "return -1;"), "cannot prove"),
        )
        for source, error in samples:
            with self.subTest(error=error), self.assertRaisesRegex(LanError, error):
                compile_source(source)

    def test_exported_recursive_contract_can_be_imported_from_table(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = DIRECT + "export { countdown };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source(
                'import { countdown } from "./provider.table"; '
                'define use_it(n: uint8) -> uint8 { return countdown(n); }',
                root / "consumer.lang",
            )
            self.assertIn("extern uint8_t", c)

    def test_dependent_recursive_result_refers_to_entry_parameter(self):
        c, table = compile_source(DEPENDENT)
        self.assertIn("lan_dependent_result", c)
        self.assertIn("sct recursive termination proved: count_up", table)
        with self.assertRaisesRegex(LanError, "cannot prove dependent result bound"):
            compile_source(DEPENDENT.replace("return count_up(n) + 1", "return count_up(n)"))

    def test_recursive_compound_parameter_and_result_bounds_are_checked(self):
        source = """
            define down(n: uint8[0,60], k: uint8[0,120] & <= n * 2) -> uint8 & <= k {
                if (k == 0 || k == 120) { return 0; }
                return down(n, k - 1);
            }
            define first(n: uint8[0,120]) -> uint8 & <= n * 2 {
                if (n == 0) { return 0; }
                return second(n);
            }
            define second(n: uint8[0,120]) -> uint8 & <= n * 2 {
                if (n == 0) { return 0; }
                return first(n - 1);
            }
        """
        _, table = compile_source(source)
        self.assertIn("sct recursive termination proved: down", table)
        self.assertIn("sct recursive termination proved: first, second", table)
        with self.assertRaisesRegex(LanError, "cannot prove dependent parameter expression"):
            compile_source(source.replace("down(n, k - 1)", "down(n, k + 1)"))
        with self.assertRaisesRegex(LanError, "cannot prove dependent result bound"):
            compile_source(source.replace("if (n == 0) { return 0; }", "if (n == 0) { return 1; }"))

    def test_dependent_recursive_result_survives_table_export(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = DEPENDENT + "export { count_up };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source(
                'import { count_up } from "./provider.table"; '
                'define check() -> uint8[7,7] { return count_up(7); }',
                root / "consumer.lang",
            )
            self.assertIn("extern uint8_t", c)
            self.assertIn("lan_dependent_result", c)

    def test_by_value_float_and_classified_result_can_recur(self):
        c, table = compile_source(SCALAR_RESULTS)
        self.assertIn("float lan_fn_keep(", c)
        self.assertIn("propagate error {250}", table)
        self.assertNotIn("sct recursive termination proved", table)
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(SCALAR_RESULTS.replace("return 3;", "return 5;"))
        with self.assertRaisesRegex(LanError, "propagated error enum identity differs"):
            compile_source("""
                enum First: uint8 { Bad = 250 }
                enum Second: uint8 { Bad = 250 }
                type A = uint8[0,4] | First;
                type B = uint8[0,4] | Second;
                define first(n: uint8) -> A {
                    if (n == 0) { return First.Bad; }
                    return second(n - 1)?;
                }
                define second(n: uint8) -> B {
                    if (n == 0) { return Second.Bad; }
                    return first(n - 1)?;
                }
            """)
        with self.assertRaisesRegex(LanError, "recursive functions require by-value scalars"):
            compile_source("""
                define nullable(n: uint8, x: uint8 · {pass: alias} | null) -> uint8 {
                    if (n == 0) { return 0; }
                    return nullable(n - 1, x);
                }
            """)


class TypeExecutionTests(unittest.TestCase):
    def test_evaluates_only_the_selected_branch_and_executes_recursion(self):
        source = """
            type Count(n: uint8) {
                if (n == 0 || false && 1 / 0 == 0) { return {7}; }
                return Count(n - 1);
            }
            type Result = Count(5) in {uint8} 8;
            define result() -> Result { return 7; }
        """
        _, table = compile_source(source)
        self.assertIn("return {7}", table)
        self.assertNotIn("sct recursive termination proved", table)

    def test_errors_when_execution_reaches_division_by_zero(self):
        with self.assertRaisesRegex(LanError, "division by zero at compile time"):
            compile_source("type Fail() { return {1 / 0}; } type X = Fail();")

    def test_nonterminating_type_execution_reports_compile_error(self):
        with self.assertRaisesRegex(LanError, "type evaluation budget exhausted"):
            compile_source("type Loop() { while (true) {} return {0}; } type X = Loop();")
        with self.assertRaisesRegex(LanError, "compile-time recursion limit"):
            compile_source("type Loop() { return Loop(); } type X = Loop();")


OWNED_RECURSION = """
    define make(n: uint8) -> uint8 · {storage: heap} {
        if (n != 0) { return make(n - 1); }
        define value: uint8 · {storage: heap} = 7;
        return value;
    }
"""


class OwningResultRecursionTests(unittest.TestCase):
    def test_scalar_result_transfers_to_the_caller(self):
        c, table = compile_source(OWNED_RECURSION + """
            define read(n: uint8) -> uint8 {
                define owner = make(n);
                define value = owner;
                free(owner);
                return value;
            }
        """)
        self.assertIn("uint8_t * lan_fn_make(", c)
        self.assertIn("return lan_fn_make(", c)
        self.assertRegex(table, r"define entity id#\d+ owner: owned result, Alive")
        self.assertRegex(table, r"mark tombstone id#\d+ owner: Tombstone")

    def test_mutual_recursion_returns_an_owning_record(self):
        c, table = compile_source("""
            type Pair = { first: uint8, second: uint16 };
            define make(n: uint8) -> Pair · {storage: heap} { return build(n); }
            define build(n: uint8) -> Pair · {storage: heap} {
                if (n != 0) { return make(n - 1); }
                define pair: Pair · {storage: heap} = { first: 7, second: 9 };
                return pair;
            }
            define read(n: uint8) -> uint8 {
                define owner = make(n);
                define value = owner.first;
                free(owner);
                return value;
            }
        """)
        self.assertRegex(c, r"lan_record_\w+ \* lan_fn_make\(")
        self.assertIn(" = lan_fn_build(", c)
        self.assertIn(" = lan_fn_make(", c)
        self.assertRegex(table, r"transfer ownership id#\d+ pair .* to result")

    def test_recursive_array_result_preserves_its_length(self):
        source = """
            define make(n: uint8) -> uint8[..] · {len: {2}} · {storage: heap} {
                if (n != 0) { return make(n - 1); }
                define items: uint8[2] · {storage: heap};
                items[1] = 7;
                return items;
            }
            define read(n: uint8) -> uint8 {
                define owner = make(n);
                define value = owner[1];
                free(owner);
                return value;
            }
        """
        c, _ = compile_source(source)
        self.assertIn("lan_slice lan_fn_make(", c)
        with self.assertRaisesRegex(LanError, "cannot prove owned array result length"):
            compile_source(source.replace("items: uint8[2]", "items: uint8[3]"))

    def test_recursive_call_and_its_result_obey_lifetime_checks(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'extra' leaves scope"):
            compile_source(OWNED_RECURSION.replace("return make(n - 1);",
                "define extra: uint8 · {storage: heap} = 1; return make(n - 1);"))
        with self.assertRaisesRegex(LanError, "heap leak: 'owner' leaves scope"):
            compile_source(OWNED_RECURSION + """
                define read() -> uint8 { define owner = make(3); return 0; }
            """)
        with self.assertRaisesRegex(LanError, "may be a Tombstone"):
            compile_source(OWNED_RECURSION + """
                define read() -> uint8 { define owner = make(3); free(owner); return owner; }
            """)

    def test_recursive_owned_result_survives_table_export(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = OWNED_RECURSION + "export { make };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source(
                'import { make } from "./provider.table"; '
                'define drop() -> void { free(make(3)); }',
                root / "consumer.lang",
            )
            self.assertIn("extern uint8_t *", c)
            self.assertIn("free((void *)(lan_api_", c)


OWNING_PARAMETER_RECURSION = """
    define drop(owner: uint8 · {storage: heap}, n: uint8) -> uint8 {
        if (n != 0) {
            owner = 7;
            return drop(owner, n - 1);
        }
        define value = owner;
        free(owner);
        return value;
    }
"""


class OwningParameterRecursionTests(unittest.TestCase):
    def test_recursive_call_can_forward_an_owner_to_the_base_case(self):
        c, table = compile_source(OWNING_PARAMETER_RECURSION + """
            define run() -> uint8 {
                define owner: uint8 · {storage: heap} = 3;
                return drop(owner, 2);
            }
        """)
        self.assertRegex(c, r"uint8_t lan_fn_drop\(uint8_t \* lv_\d+_owner")
        self.assertIn("owning parameters drop: param#0 owner (move)", table)
        self.assertNotIn("mutation param#0", table)

    def test_mutual_recursion_can_transfer_an_owned_array_to_its_result(self):
        _, table = compile_source("""
            type Items = uint8[..] · {len: {2}} · {storage: heap};
            define pass(items: Items, n: uint8) -> Items { return relay(items, n); }
            define relay(items: Items, n: uint8) -> Items {
                if (n == 0) { return items; }
                items[1] = 7;
                return pass(items, n - 1);
            }
            define read() -> uint8 {
                define items: uint8[2] · {storage: heap};
                define owner = pass(items, 2);
                define value = owner[1];
                free(owner);
                return value;
            }
        """)
        self.assertIn("owning parameters pass: param#0 items (move)", table)
        self.assertIn("owning parameters relay: param#0 items (move)", table)

    def test_an_owner_must_be_consumed_on_each_returning_path(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'owner' leaves scope"):
            compile_source(OWNING_PARAMETER_RECURSION.replace("free(owner);", ""))
        with self.assertRaisesRegex(LanError, "may be a Tombstone"):
            compile_source(OWNING_PARAMETER_RECURSION.replace(
                "return drop(owner, n - 1);",
                "define result = drop(owner, n - 1); return owner;"))

    def test_recursive_consuming_parameter_survives_table_export(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            provider = root / "provider.lang"
            source = OWNING_PARAMETER_RECURSION + "export { drop };"
            _, table = ModuleGraph(provider, Compiler(source, provider.name), library=True).compile()
            (root / "provider.table").write_text(table, encoding="ascii")
            c, _ = compile_source("""
                import { drop } from "./provider.table";
                define run() -> uint8 {
                    define owner: uint8 · {storage: heap} = 3;
                    return drop(owner, 2);
                }
            """, root / "consumer.lang")
            self.assertRegex(c, r"extern uint8_t lan_api_\w+_drop\(uint8_t \*, uint8_t\)")


BORROWED_RECORD = """
    type Ctx = { flag: uint8, note: uint8 };
    define set_flag(ctx: Ctx, n: uint8) -> uint8 {
        if (n == 0) { return 0; }
        ctx.flag = 1;
        return set_flag(ctx, n - 1);
    }
"""


class BorrowedRecursionTests(unittest.TestCase):
    def test_borrowed_record_recursion_summarises_only_the_touched_field(self):
        _, table = compile_source(BORROWED_RECORD + """
            define caller() -> uint8 {
                define ctx: Ctx = { flag: 0, note: 9 };
                define before = ctx.flag;
                define seen = set_flag(ctx, 2);
                return ctx.note;
            }
        """)
        self.assertIn("recursive writes set_flag: param#0.flag", table)
        self.assertIn("mutation param#0.flag => [0, 255]", table)
        self.assertNotIn("mutation param#0.note", table)
        self.assertIn("return {9}", table)
        self.assertNotIn("sct recursive termination proved", table)

    def test_mutual_recursion_shares_the_borrowed_summary(self):
        c, table = compile_source("""
            type Ctx = { depth: uint8, spare: uint8 };
            define even(ctx: Ctx, n: uint8) -> uint8 {
                if (n == 0) { return 1; }
                ctx.depth = 1;
                return odd(ctx, n - 1);
            }
            define odd(ctx: Ctx, n: uint8) -> uint8 {
                if (n == 0) { return 0; }
                return even(ctx, n - 1);
            }
            define caller() -> uint8 {
                define ctx: Ctx = { depth: 0, spare: 4 };
                define seen = even(ctx, 2);
                return ctx.spare;
            }
        """)
        self.assertIn("recursive writes even: param#0.depth", table)
        self.assertIn("recursive writes odd: param#0.depth", table)
        self.assertIn("mutation param#0.depth => [0, 255]", table)
        self.assertNotIn("mutation param#0.spare", table)
        self.assertIn("return {4}", table)
        self.assertIn("lan_fn_odd(", c)

    def test_the_summary_reaches_a_fixed_point_across_a_relay_chain(self):
        _, table = compile_source("""
            type Ctx = { tail: uint8, spare: uint8 };
            define outer(ctx: Ctx, n: uint8) -> uint8 {
                if (n == 0) { return 0; }
                return middle(ctx, n - 1);
            }
            define middle(ctx: Ctx, n: uint8) -> uint8 {
                if (n == 0) { return 0; }
                return leaf(ctx, n - 1);
            }
            define leaf(ctx: Ctx, n: uint8) -> uint8 {
                if (n == 0) { return 0; }
                ctx.tail = 7;
                return outer(ctx, n - 1);
            }
        """)
        for name in ("outer", "middle", "leaf"):
            with self.subTest(name=name):
                self.assertIn(f"recursive writes {name}: param#0.tail", table)
        self.assertNotIn("mutation param#0.spare", table)

    def test_a_helper_summary_flows_into_the_group_summary(self):
        _, table = compile_source("""
            type Ctx = { touched: uint8, spare: uint8 };
            define mark(ctx: Ctx) -> void { ctx.touched = 1; }
            define walk(ctx: Ctx, n: uint8) -> uint8 {
                if (n == 0) { return 0; }
                mark(ctx);
                return walk(ctx, n - 1);
            }
            define caller() -> uint8 {
                define ctx: Ctx = { touched: 0, spare: 3 };
                define seen = walk(ctx, 2);
                return ctx.spare;
            }
        """)
        self.assertIn("recursive writes walk: param#0.touched", table)
        self.assertIn("return {3}", table)

    def test_a_group_that_writes_nothing_reports_no_summary(self):
        _, table = compile_source("""
            type Ctx = { a: uint8 };
            define peek(ctx: Ctx, n: uint8) -> uint8 {
                if (n == 0) { return ctx.a; }
                return peek(ctx, n - 1);
            }
        """)
        self.assertIn("recursive writes peek: none", table)
        self.assertNotIn("mutation param", table)

    def test_writes_through_an_alias_and_through_nested_fields_are_traced(self):
        _, table = compile_source("""
            type Inner = { flag: uint8, spare: uint8 };
            type Outer = { inner: Inner, count: uint8 };
            define walk(ctx: Outer, n: uint8) -> uint8 {
                if (n == 0) { return 0; }
                define view = ctx;
                view.inner.flag = 1;
                return walk(ctx, n - 1);
            }
        """)
        self.assertIn("recursive writes walk: param#0.inner.flag", table)
        self.assertNotIn("mutation param#0.inner.spare", table)
        self.assertNotIn("mutation param#0.count", table)

    def test_borrowed_slice_recursion_invalidates_element_facts(self):
        _, table = compile_source("""
            define fill(items: uint8[..] · {len: [1, 8]}, i: uint8) -> uint8 {
                if (i == 0) { return 0; }
                items[0] = 7;
                return fill(items, i - 1);
            }
            define caller() -> uint8 {
                define buffer: uint8[4];
                define seen = fill(buffer, 3);
                return buffer[0];
            }
        """)
        self.assertIn("mutation param#0. => content", table)

    def test_immutable_and_copied_parameters_need_no_summary(self):
        for annotation in ("Ctx · {immutable: true}", "Ctx · {pass: copy}"):
            with self.subTest(annotation=annotation):
                _, table = compile_source(f"""
                    type Ctx = {{ flag: uint8 }};
                    define peek(ctx: {annotation}, n: uint8) -> uint8 {{
                        if (n == 0) {{ return ctx.flag; }}
                        return peek(ctx, n - 1);
                    }}
                """)
                self.assertNotIn("mutation param", table)

    def test_interface_and_nullable_borrows_stay_out_of_recursion(self):
        for annotation in ("View", "uint8 · {pass: alias} | null"):
            with self.subTest(annotation=annotation), self.assertRaisesRegex(
                    LanError, "recursive functions require by-value scalars"):
                compile_source(f"""
                    type View = interface {{ x: uint8 }};
                    define walk(ctx: {annotation}, n: uint8) -> uint8 {{
                        if (n == 0) {{ return 0; }}
                        return walk(ctx, n - 1);
                    }}
                """)
