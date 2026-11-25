import re
import unittest

from lanc import AffineForm, PathState, ValueOrigin, compile_source
from value_domain import Domain


class IfAndProofOptimizationTests(unittest.TestCase):
    def test_identical_post_if_states_fold_across_branches(self):
        branches = "\n".join(
            f"""
                if (x == {i}) {{
                    define tmp{i}: uint8 = 7;
                    x = 0;
                    code = tmp{i};
                }} else {{
                    x = 0;
                    code = 7;
                }}
            """
            for i in range(10)
        )
        _, table = compile_source(f"""
            define f(x: uint8) -> uint8 {{
                define code: uint8 = 0;
                {branches}
                return code;
            }}
        """)
        self.assertNotIn("path budget", table)
        self.assertEqual(table.count("  return {7} ["), 1)
        self.assertIn("  return {7} [entry]", table)

    def test_resolved_early_return_guards_do_not_accumulate(self):
        checks = "\n".join(
            f"    if (x == {i}) {{ return {i + 1}; }}"
            for i in range(12)
        )
        _, table = compile_source(f"""
            define check_many(x: uint8) -> uint8 {{
            {checks}
                return 0;
            }}
        """)
        self.assertIn("  return {0} [entry]", table)
        self.assertEqual(table.count("=false and"), 0)

    def test_long_continuing_guards_are_abbreviated(self):
        guards = tuple(f"{10 + i}:5=false" for i in range(8))
        state = PathState({}, guards=guards)
        self.assertEqual(
            state.guard_label(),
            "... (5 more) and 15:5=false and 16:5=false and 17:5=false",
        )
        short_state = PathState({}, guards=guards[:4])
        self.assertEqual(short_state.guard_label(), "10:5=false and 11:5=false and 12:5=false and 13:5=false")

    def test_uniform_array_index_comparison_does_not_split_paths(self):
        _, table = compile_source("""
            define find_match(a: uint8[8] · {pass: alias}, i: uint8[0, 7], target: uint8) -> uint8 {
                if (a[i] == target) {
                    return a[i];
                }
                return 0;
            }
        """)
        self.assertEqual(table.count("  true ["), 1)
        self.assertEqual(table.count("  false ["), 1)

    def test_uniform_array_index_narrowing_keeps_element_knowledge_without_splitting(self):
        _, table = compile_source("""
            define capped_read(a: uint8[8] · {pass: alias}, i: uint8[0, 7]) -> uint8 {
                if (a[i] <= 40) {
                    return a[i];
                }
                return 40;
            }
        """)
        self.assertEqual(table.count("  true ["), 1)
        self.assertEqual(table.count("  false ["), 1)
        self.assertIn("  return [0, 40] [", table)
        self.assertEqual(len(re.findall(r"verify index", table)), 4)

    def test_fold_deduplicates_more_than_64_identical_states_and_join_keeps_common_guards(self):
        ep0 = object()
        paths = []
        for idx in range(70):
            origin = ValueOrigin(Domain.point(3))
            paths.append(
                PathState(
                    {1: Domain.point(3)},
                    guards=("1:1=true", f"{10 + idx}:5={'true' if idx % 2 else 'false'}"),
                    forms={1: AffineForm(origin)},
                    epochs={1: object() if idx % 2 else object(), 2: ep0},
                )
            )
        folded = PathState.fold(paths, frozenset({ep0}), frozenset())
        self.assertEqual(len(folded), 1)
        self.assertEqual(folded[0].guards, ("1:1=true",))

        p_true = PathState({1: Domain.point(1)}, guards=("1:1=true", "2:5=true"))
        p_false = PathState({1: Domain.point(2)}, guards=("1:1=true", "2:5=false"))
        joined = PathState.join([p_true, p_false])
        self.assertEqual(joined.guards, ("1:1=true",))

    def test_consecutive_identical_before_trace_is_not_repeated(self):
        _, table = compile_source("""
            define seq() -> uint8 {
                define a: uint8 = 1;
                define b: uint8 = 2;
                define c: uint8 = 3;
                return a + b + c;
            }
        """)
        # Only the initial statement needs `before [entry] {  }`; subsequent
        # statements directly follow `after` without duplicating the same line.
        self.assertEqual(table.count("  before ["), 1)

    def test_many_one_shot_dead_checks_do_not_explode_or_break_correlations(self):
        params = ", ".join(f"f{i}: boolean" for i in range(16))
        checks = "\n".join(f"    if (f{i}) {{ log_event({i}); }}" for i in range(16))
        _, table = compile_source(f"""
            extern contract log_event(code: uint8) -> void writes {{external}};
            define battle_step(b0: boolean, {params}) -> uint8[1, 1] {{
                define x: uint8 = 0;
                define y: uint8 = 0;
                if (b0) {{ x = 1; y = 0; }} else {{ x = 0; y = 1; }}
                {checks}
                return x + y;
            }}
        """)
        self.assertNotIn("path budget", table)
        self.assertIn("  return {1} [", table)

    def test_many_one_shot_pure_checks_on_live_variable_stay_single_path(self):
        checks = "\n".join(f"    if (x > {i * 10}) {{ log_event({i}); }}" for i in range(1, 21))
        _, table = compile_source(f"""
            extern contract log_event(code: uint8) -> void writes {{external}};
            define inspect(x: uint8) -> uint8 {{
                {checks}
                return x;
            }}
        """)
        self.assertNotIn("path budget", table)
        self.assertEqual(table.count("  return [0, 255] ["), 1)

    def test_many_one_shot_dead_flag_writes_fold_to_distinct_live_states(self):
        params = ", ".join(f"b{i}: boolean" for i in range(14))
        writes = "\n".join(f"    if (b{i}) {{ ctx.a = 1; }}" for i in range(14))
        _, table = compile_source(f"""
            type Ctx = {{ a: uint8, b: uint8 }};
            define mark_any(ctx: Ctx, {params}) -> uint8 {{
                {writes}
                return ctx.b;
            }}
        """)
        self.assertNotIn("path budget", table)

    def test_many_independent_clamps_collapse_without_path_budget(self):
        params = ", ".join(f"s{i}: uint16" for i in range(10))
        clamps = "\n".join(f"    if (s{i} > 999) {{ s{i} = 999; }}" for i in range(10))
        total = " + ".join(f"s{i}" for i in range(10))
        _, table = compile_source(f"""
            define clamp_all({params}) -> uint16[0, 9990] {{
                {clamps}
                return {total};
            }}
        """)
        self.assertNotIn("path budget", table)
        self.assertEqual(table.count("  return [0, 9990] ["), 1)

    def test_live_correlations_and_zone_relations_are_never_lost(self):
        # 1) Relational difference bound: y >= x after clamping x
        compile_source("""
            define diff_after_clamp(x: uint8) -> uint8 {
                define y: uint8 = x;
                if (x > 100) { x = 100; }
                return y - x;
            }
        """)
        # 2) Zone relation: x >= y after max-assign
        compile_source("""
            define diff_after_max(x: uint8, y: uint8) -> uint8 {
                if (x < y) { x = y; }
                return x - y;
            }
        """)
        # 3) Live boolean flag correlation preserved when flag is reused later
        compile_source("""
            define reuse_flag(b: boolean) -> uint8[1, 1] {
                define x: uint8 = 0;
                if (b) { x = 1; }
                if (b) { return x; }
                return 1;
            }
        """)
        # 4) Guard function correlation preserved when boolean return depends on parameter
        compile_source("""
            define is_nonzero(x: uint8) -> boolean {
                define ok: boolean = false;
                if (x > 0) { ok = true; }
                return ok;
            }
            define safe_div(a: uint8, x: uint8) -> uint8 {
                if (is_nonzero(x)) {
                    return a / x;
                }
                return 0;
            }
        """)


if __name__ == "__main__":
    unittest.main()
