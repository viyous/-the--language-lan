"""Tests for compile-time configurable reasoning step budgets (`--max-steps` / `max_steps`)."""

import tempfile
import unittest
from pathlib import Path

from lan_compiler import LanError, compile_source, main
from optimization_bridge import LIMITS, active_limits
from value_domain import ACTIVE_SOLVER_STEPS, DEFAULT_SOLVER_STEPS


class SolverStepsTests(unittest.TestCase):
    def test_equation_unsolved_at_default_steps_succeeds_with_higher_max_steps(self):
        # Mordell equation n^3 = m^2 + 2 on [0, 10^17]: unique solution is 27 (3^3 = 5^2 + 2).
        # Takes ~8510 steps in bounded-sieve: fails at default 2048 steps, succeeds with max_steps=20_000.
        program = """
            type Cube = uint64[0, 100000000000000000] & poly(n * n * n);
            type ShiftSq = uint64[0, 100000000000000000] & poly(n * n + 2);
            theorem equal(Cube & ShiftSq, {27});
            define f(x: Cube & ShiftSq) -> uint8[27, 27] {
                return x as uint8[27, 27];
            }
        """
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(program)
        # Same process, higher step budget must not reuse stale unknown cache
        c, _ = compile_source(program, max_steps=20_000)
        self.assertIn("lan_fn_f", c)

    def test_disjoint_equation_solved_with_higher_max_steps(self):
        # Mordell equation n^3 + 7 = m^2 has no integer solutions; takes ~8510 steps on [0, 10^17].
        program = """
            type CubePlus7 = uint64[0, 100000000000000000] & poly(n * n * n + 7);
            type Square = uint64[0, 100000000000000000] & poly(n * n);
            theorem disjoint(CubePlus7, Square);
        """
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(program)
        compile_source(program, max_steps=20_000)

    def test_low_max_steps_blocks_and_higher_max_steps_solves(self):
        program = """
            define bounded(x: uint32[0, 100000] & poly(n * n) & poly(2 * n * n + 1)) -> uint32[1, 9801] {
                return x;
            }
        """
        with self.assertRaisesRegex(LanError, "cannot prove|cannot decide"):
            compile_source(program, max_steps=2)
        _, table = compile_source(program, max_steps=2048)
        self.assertIn("return [1, 9801] & poly", table)

    def test_quadratic_factor_beyond_default_quadratic_limit(self):
        # n^2 + 125000001 = m^2: norm = 4*125000001, isqrt(norm) = 22360 > _QUADRATIC_LIMIT (20_000).
        # Fails at default 2048 steps, succeeds when max_steps is raised to 30_000.
        program = """
            type A = uint64[0, 1000000000000] & poly(n * n + 125000001);
            type B = uint64[0, 1000000000000] & poly(n * n);
            define f(x: A & B) -> uint64[125000001, 1000000000000] {
                return x;
            }
        """
        with self.assertRaisesRegex(LanError, "cannot prove|cannot decide"):
            compile_source(program)
        compile_source(program, max_steps=30_000)

    def test_rational_polynomial_square_triangular_numbers(self):
        # n*(n+1)/2 = m^2 on uint32: largest solution <= 2^32-1 is 1631432881 (40391^2).
        program = """
            type Tri = uint32 & poly((n * (n + 1)) / 2);
            type Sq  = uint32 & poly(n * n);
            define f(x: Tri & Sq) -> uint32[0, 1631432881] {
                return x;
            }
        """
        _, table = compile_source(program, max_steps=10_000)
        self.assertIn("return [0, 1631432881]", table)

    def test_cli_max_steps_and_solver_steps_and_multimodule_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dep = root / "math_dep.lang"
            dep.write_text("""
                type Cube = uint64[0, 100000000000000000] & poly(n * n * n);
                type ShiftSq = uint64[0, 100000000000000000] & poly(n * n + 2);
                theorem equal(Cube & ShiftSq, {27});
                define only_27(x: Cube & ShiftSq) -> uint8[27, 27] {
                    return x as uint8[27, 27];
                }
                export { only_27 };
            """, encoding="utf-8")
            app = root / "app.lang"
            app.write_text("""
                import { only_27 } from "./math_dep.lang";
                define main_check() -> uint8 {
                    return 1;
                }
            """, encoding="utf-8")
            # Default steps fail in dependency
            self.assertEqual(main([str(app), "--check"]), 1)
            # --max-steps propagates to dependency and succeeds
            self.assertEqual(main([str(app), "--check", "--max-steps", "20000"]), 0)
            # --solver-steps alias also succeeds
            self.assertEqual(main([str(app), "--check", "--solver-steps", "20000"]), 0)

    def test_invalid_max_steps_rejected_and_active_limits_scale(self):
        with self.assertRaises(ValueError):
            compile_source("define f() -> uint8 { return 1; }", max_steps=-1)
        self.assertEqual(active_limits(), LIMITS)
        token = ACTIVE_SOLVER_STEPS.set(DEFAULT_SOLVER_STEPS * 10)
        try:
            scaled = active_limits()
            self.assertEqual(scaled["rlimit"], LIMITS["rlimit"] * 10)
            self.assertEqual(scaled["timeout_ms"], LIMITS["timeout_ms"] * 10)
        finally:
            ACTIVE_SOLVER_STEPS.reset(token)


if __name__ == "__main__":
    unittest.main()
