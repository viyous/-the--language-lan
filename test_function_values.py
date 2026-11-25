import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


HANDLERS = """
    type Gear = {energy: uint8, damage: uint8, slot: uint8};

    define tick(gear: Gear, amount: uint8[0, 9]) -> void {
        gear.energy = amount;
    }

    define attacked(gear: Gear, amount: uint8[0, 9]) -> void {
        gear.damage = amount;
    }

    type Handler = callable(tick, attacked);
"""

CALLBACKS = HANDLERS + """
    type Hooks = {active: Handler};

    define dispatch(cb: Handler, gear: Gear, amount: uint8[0, 9]) -> void {
        cb(gear, amount);
    }

    define route(index: uint8[0, 1], gear: Gear) -> void {
        define choices: Handler[2] = [callback(tick), callback(attacked)];
        choices[index](gear, 7);
    }

    define keep_slot(cb: Handler) -> uint8[3, 3] {
        define gear: Gear = {slot: 3};
        cb(gear, 5);
        return gear.slot;
    }

    define next_index(count: uint8[0, 1] · {pass: alias}) -> uint8[0, 1] {
        count = (count + 1) as: sat uint8[0, 1];
        return 0;
    }

    define set_and_switch(hooks: Hooks, count: uint8[0, 3] · {pass: alias}) -> uint8[0, 9] {
        hooks.active = callback(attacked);
        count = (count + 1) as: sat uint8[0, 3];
        return 9;
    }

    define capture_once() -> int32 {
        define gear: Gear = {energy: 1, damage: 2, slot: 3};
        define hooks: Hooks = {active: callback(tick)};
        define argument_count: uint8[0, 3] = 0;
        hooks.active(gear, set_and_switch(hooks, argument_count));
        if (gear.energy != 9 || gear.damage != 2 || argument_count != 1) { return 4; }
        define choices: Handler[2] = [callback(tick), callback(attacked)];
        define selector_count: uint8[0, 1] = 0;
        choices[next_index(selector_count)](gear, 6);
        if (gear.energy != 6 || gear.damage != 2 || selector_count != 1) { return 5; }
        return 0;
    }

    define main() -> int32 {
        define gear: Gear = {energy: 1, damage: 2, slot: 3};
        define hooks: Hooks = {active: callback(tick)};
        dispatch(hooks.active, gear, 8);
        if (gear.energy != 8 || gear.damage != 2 || gear.slot != 3) { return 1; }
        route(1, gear);
        if (gear.energy != 8 || gear.damage != 7 || gear.slot != 3) { return 2; }
        if (keep_slot(callback(attacked)) != 3) { return 3; }
        return capture_once();
    }
"""


FALLIBLE = """
    enum Fail: uint8 { Missing = 255 };
    type Checked = uint8[0, 9] | Fail;
    type Advanced = uint8[1, 10] | Fail;

    define filter(value: uint8[0, 9]) -> Checked {
        if (value == 0) { return Fail.Missing; }
        return value;
    }

    define accept(value: uint8[0, 9]) -> Checked {
        return value;
    }

    type Handler = callable(filter, accept);
    define dispatch(cb: Handler, value: uint8[0, 9]) -> Advanced {
        return cb(value)? + 1;
    }

    define main() -> int32 {
        if (!(dispatch(callback(filter), 0) in Fail)) { return 1; }
        if (dispatch(callback(accept), 0) != 1) { return 2; }
        if (dispatch(callback(filter), 9) != 10) { return 3; }
        return 0;
    }
"""


OWNERSHIP = """
    type Cell = uint8 · {storage: heap};

    define drop(cell: Cell) -> uint8 {
        define n: uint8 = cell;
        free(cell);
        return n;
    }

    define parity_and_drop(cell: Cell) -> uint8 {
        define n: uint8 = cell;
        free(cell);
        return n & 1;
    }

    type Drop = callable(drop, parity_and_drop);

    define discard(cb: Drop, cell: Cell) -> uint8 {
        return cb(cell);
    }

    define main() -> int32 {
        define first: Cell = 7;
        if (discard(callback(parity_and_drop), first) != 1) { return 1; }
        define second: Cell = 6;
        if (discard(callback(drop), second) != 6) { return 2; }
        return 0;
    }
"""


class FunctionValueTests(unittest.TestCase):
    def test_dispatch_passes_and_stores_handlers(self):
        c, _ = compile_source(CALLBACKS)
        self.assertIn("lan_fn_tick(", c)
        self.assertIn("lan_fn_attacked(", c)
        self.assertIn("lan_fn_route(", c)

    def test_unknown_handler_can_write_either_field(self):
        for field in ("energy", "damage"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(LanError, "cannot prove"):
                    compile_source(HANDLERS + f"""
                        define stale(cb: Handler) -> uint8[0, 0] {{
                            define gear: Gear;
                            cb(gear, 7);
                            return gear.{field};
                        }}
                    """)

    def test_candidate_can_be_defined_after_a_dispatching_function(self):
        compile_source("""
            type Gear = {energy: uint8, slot: uint8};
            type Handler = callable(tick);

            define dispatch(cb: Handler) -> uint8[3, 3] {
                define gear: Gear = {slot: 3};
                cb(gear);
                return gear.slot;
            }

            define tick(gear: Gear) -> void {
                gear.energy = 9;
            }
        """)

    def test_call_checks_common_parameter_contract(self):
        with self.assertRaisesRegex(LanError, "cannot prove"):
            compile_source(HANDLERS + """
                define invalid(cb: Handler, gear: Gear, amount: uint8) -> void {
                    cb(gear, amount);
                }
            """)

        with self.assertRaisesRegex(LanError, "cannot prove index|out of bounds"):
            compile_source(HANDLERS + """
                define invalid(index: uint8[0, 2], gear: Gear) -> void {
                    define choices: Handler[2] = [callback(tick), callback(attacked)];
                    choices[index](gear, 7);
                }
            """)

    def test_indirect_call_preserves_common_error_union(self):
        compile_source(FALLIBLE)

    def test_indirect_call_consumes_owned_arguments_once(self):
        compile_source(OWNERSHIP)
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(OWNERSHIP + """
                define invalid(cb: Drop) -> uint8 {
                    define cell: Cell = 6;
                    cb(cell);
                    return cell;
                }
            """)
        with self.assertRaisesRegex(LanError, "live mutable heap owner"):
            compile_source(OWNERSHIP + """
                define invalid(cb: Drop) -> uint8 {
                    define cell: uint8 = 6;
                    return cb(cell);
                }
            """)

    def test_candidate_array_writes_invalidate_only_possible_positions(self):
        source = """
            define first(values: uint8[4]) -> void { values[0] = 7; }
            define second(values: uint8[4]) -> void { values[2] = 9; }
            type Handler = callable(first, second);

            define check(cb: Handler) -> uint8[4, 4] {
                define values: uint8[4] = [4; 4];
                cb(values);
                return values[1];
            }
        """
        compile_source(source)
        for position in (0, 2):
            with self.subTest(position=position):
                with self.assertRaisesRegex(LanError, "cannot prove"):
                    compile_source(source.replace("return values[1]", f"return values[{position}]"))

    def test_indirect_cycles_are_diagnosed(self):
        with self.assertRaisesRegex(LanError, "indirect recursion"):
            compile_source("""
                define target(x: uint8) -> uint8 { return dispatch(callback(target), x); }
                type Handler = callable(target);
                define dispatch(cb: Handler, x: uint8) -> uint8 { return cb(x); }
            """)

    def test_exported_wrapper_preserves_callback_writes(self):
        provider = HANDLERS + """
            define apply(gear: Gear, is_tick: boolean) -> void {
                define handler: Handler = callback(attacked);
                if (is_tick) { handler = callback(tick); }
                handler(gear, 7);
            }
            export { Gear, apply };
        """
        consumer = """
            import { Gear, apply } from "./provider.table";
            define wrap(is_tick: boolean) -> uint8[3, 3] {
                define gear: Gear = {slot: 3};
                apply(gear, is_tick);
                return gear.slot;
            }
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            library = root / "provider.lang"
            _, table = ModuleGraph(library, Compiler(provider, library), library=True).compile()
            library.with_suffix(".table").write_text(table, encoding="utf-8")
            compile_source(consumer, root / "consumer.lang")
            for field in ("energy", "damage"):
                with self.subTest(field=field):
                    stale = consumer.replace("return gear.slot;", f"return gear.{field};")
                    with self.assertRaisesRegex(LanError, "cannot prove"):
                        compile_source(stale, root / "consumer.lang")

    def test_callbacks_require_a_known_candidate_and_compatible_contracts(self):
        invalid_attempts = (
            """
                define foreign_target(gear: Gear, amount: uint8[0, 9]) -> void {}
                define invalid() -> void {
                    define cb: Handler = callback(foreign_target);
                }
            """,
            """
                define wide(gear: Gear, amount: uint16) -> void {}
                type Incompatible = callable(tick, wide);
            """,
            """
                define invalid() -> void {
                    define cb: Handler = 1;
                }
            """,
            """
                define invalid() -> void {
                    define cb: Handler;
                }
            """,
            """
                type Hooks = {active: Handler, priority: uint8};
                define invalid() -> void {
                    define hook: Hooks = {priority: 0};
                }
            """,
            """
                define invalid(code: uint16) -> void {
                    define cb: Handler = code as: trap Handler;
                }
            """,
            """
                type Hooks = {active: Handler};
                define invalid() -> void {
                    define hook: Hooks = {active: callback(tick)};
                    define bits: bit[16] = hook as: view bit[16];
                    bits[0] = 0;
                }
            """,
        )
        for attempt in invalid_attempts:
            with self.subTest(attempt=attempt):
                with self.assertRaises(LanError):
                    compile_source(HANDLERS + attempt)

    def test_callback_selectors_are_not_numeric_or_bit_values(self):
        for declaration, error in (
            ("define expose(cb: Handler) -> uint16 { return cb; }", "compatible callable type"),
            ("define order(cb: Handler) -> boolean { return cb < cb; }", "only support equality"),
            ("define truth(cb: Handler) -> uint8 { if (cb) { return 1; } return 0; }", "used as a condition"),
            ("define member(cb: Handler) -> boolean { return cb in {1}; }", "numeric membership"),
            ("type Forged = uint16[0,0] | Handler;", "numeric set operations"),
            ("define view(bits: bit[16]) -> Handler { return bits as: view Handler; }", "reinterpreted as bits"),
            ("define view(bits: bit[16]) -> void { define v = bits[f: Handler@bit(16, 0)]; }",
             "bit fields cannot contain function values"),
            ("type Hooks = {cb: Handler}; extern contract accept(h: Hooks) -> void writes {};", "local to"),
            ("type Hooks = {cb: Handler}; export {Hooks};", "cannot be exported"),
            ("type Parent = {link: Hook · {pass: alias} | null}; type Hook = {cb: Handler}; export {Parent};",
             "cannot be exported"),
            ("type Hooks = {cb: Handler}; type View = interface {cb: uint16}; "
             "extern contract accept(h: View · {immutable: true}) -> void writes {}; "
             "define expose(h: Hooks) -> void { accept(h); }", "incompatible type"),
        ):
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(LanError, error):
                    compile_source(HANDLERS + declaration)


if __name__ == "__main__":
    unittest.main()
