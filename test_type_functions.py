import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


class TypeObjectTests(unittest.TestCase):
    def test_integer_and_property_aliases_merge(self):
        compiler = Compiler("""
            type Word = int;
            type Width = {bits: 16 * 2};
            type I = Word · Width · {bits: 0x20};
            type U = {kind: uint} · Width;
            define read(x: I) -> int32 { return x; }
        """, "<test>")
        compiler.compile()
        self.assertEqual(compiler.types["I"], compiler.types["int32"])
        self.assertEqual(compiler.types["U"], compiler.types["uint32"])

    def test_duplicate_properties_are_equal_or_rejected(self):
        compiler = Compiler("""
            type A = {a: 1};
            type B = A · {b: 2, a: 1};
        """, "<test>")
        compiler.compile()
        self.assertEqual([(key.text, value.args[0].args[0]) for key, value in compiler.types["B"].entries],
                         [("a", 1), ("b", 2)])
        for annotation, property_name in (("int32 · {bits: 64}", "bits"),
                                          ("int · uint", "kind"),
                                          ("{a: true} · {a: 1}", "a")):
            with self.subTest(annotation=annotation):
                with self.assertRaisesRegex(LanError, f"conflicting type object property '{property_name}'"):
                    compile_source(f"type Conflict = {annotation};")

    def test_arrays_embed_types_and_keep_bounds(self):
        source = """
            type Byte = uint8;
            type Row = array · {len: 4, elem: Byte} · {elem: uint8};
            type Matrix = array · {len: 3, elem: Row};
            define pick() -> uint8 {
                define matrix: Matrix;
                matrix[1][2] = 7;
                return matrix[1][2];
            }
        """
        c, _ = compile_source(source)
        self.assertRegex(c, r"typedef uint8_t (lan_array_\w+)\[4\];")
        self.assertRegex(c, r"typedef lan_array_\w+ lan_array_\w+\[3\];")
        with self.assertRaisesRegex(LanError, "index out of bounds|cannot prove"):
            compile_source(source.replace("matrix[1][2]", "matrix[1][4]", 1))

    def test_records_compose_and_can_still_refer_to_their_own_type(self):
        compile_source("""
            type Parts = {value: uint8};
            type Link = {pass: alias};
            type Node = Parts · {next: Node · Link | null};
            define read(x: Node) -> uint8 { return x.value; }
        """)

    def test_width_can_be_computed_in_a_type_function(self):
        source = """
            type width(n: uint8) {
                define bits = 8;
                while (bits < n) { bits *= 2; }
                return {bits: bits};
            }
            type word(n: uint8) { return int · width(n); }
            type Props = width(32);
            type I = int · Props · width(32);
            define read(x: word(32)) -> I { return x; }
        """
        compile_source(source)
        with self.assertRaisesRegex(LanError, "conflicting type object property 'bits'"):
            compile_source(source.replace("int · Props · width(32)", "int · Props · width(64)"))

    def test_type_function_returns_distinct_record_layouts(self):
        compiler = Compiler("""
            type row(n: uint8) { return array · {len: n, elem: uint8}; }
            type packet(n: uint8) {
                return {data: row(n)} · {tag: uint8};
            }
            type A = packet(4);
            type B = packet(8);
            type AAgain = packet(4);
            define read(a: A, b: B) -> uint16 {
                return a.data[3] + b.data[7];
            }
        """, "<test>")
        compiler.compile()
        self.assertNotEqual(compiler.types["A"].identity, compiler.types["B"].identity)
        self.assertEqual(compiler.types["A"].identity, compiler.types["AAgain"].identity)

    def test_type_function_captures_local_properties(self):
        compile_source("""
            type properties(n: uint8) {
                define v = n + 1;
                return {a: n} · {b: v + 1};
            }
            type P = properties(7) · {a: 7, b: 9};
        """)

    def test_adjacent_braces_require_a_merge_operator(self):
        with self.assertRaisesRegex(LanError, "use '·' to merge type objects"):
            compile_source("type I = int{bits: 32};")

    def test_property_objects_can_be_read_from_a_table(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            folder = Path(directory)
            provider = folder / "provider.lang"
            provider.write_text("""
                type Width = {bits: 32};
                type Row = array · {len: 4, elem: uint8};
                type Shape = {len: 3, elem: Row};
                export { Width, Shape };
            """, encoding="utf-8")
            _, table = ModuleGraph(provider, Compiler(provider.read_text(encoding="utf-8"), provider),
                                   library=True).compile()
            (folder / "provider.table").write_text(table, encoding="utf-8")
            provider.unlink()
            consumer = folder / "consumer.lang"
            consumer.write_text("""
                import { Width, Shape } from "./provider.table";
                type I = int · Width;
                type Matrix = array · Shape;
                define read(x: I) -> I {
                    define matrix: Matrix;
                    matrix[2][3] = 7;
                    return x;
                }
            """, encoding="utf-8")
            c, _ = compile_source(consumer.read_text(encoding="utf-8"), consumer)
            self.assertRegex(c, r"typedef lan_array_\w+ lan_array_\w+\[3\];")


BELOW = """
    type below(n: uint8) {
        return uint8[0, n - 1];
    }
"""

MASK = """
    type mask_of(bits: uint8) {
        define value = 1;
        define i: uint8 = 0;
        while (i < bits) {
            value *= 2;
            i += 1;
        }
        return uint16[0, value - 1];
    }
"""

UPTO = """
    type upto(n: uint8) {
        if (n == 0) { return uint8[0, 0]; }
        return upto(n - 1) | uint8[0, n];
    }
"""

CHOOSE = """
    type choose(flag: boolean) {
        if (flag) { return uint8[0, 3]; }
        return uint8[4, 7];
    }
"""


class TypeFunctionTests(unittest.TestCase):
    def test_type_parameters_build_nested_layouts(self):
        declarations = """
            type Row(T: type, length: uint8) { return array · {len: length, elem: T}; }
            type Matrix(T: type, rows: uint8, cols: uint8) { return Row(Row(T, cols), rows); }
            type Box(T: type) { return {value: T}; }
            type Identity(T: type) { return T; }
            type UpTo(T: type, limit: T) { return [0, limit] in {T} 16; }
            type Cell = { value: uint16[0, 1000] };
            type Grid = Matrix(Cell, 2, 3);
            type BoxedGrid = Box(Grid);
            type U16 = Identity(uint16);
            type Index = UpTo(U16, 2);
            type Cells = Row({value: uint16[0, 1000]} · {tag: uint8}, 3);
            define read(grid: BoxedGrid, cells: Cells) -> Identity(uint16) {
                return grid.value[1][2].value + cells[1].value;
            }
        """
        compiler = Compiler(declarations, "<test>")
        c, _ = compiler.compile()
        grid = compiler.types["Grid"]
        self.assertEqual((grid.length, grid.element.length), (2, 3))
        self.assertEqual(grid.element.element, compiler.types["Cell"])
        self.assertEqual(compiler.types["U16"], compiler.types["uint16"])
        self.assertEqual((compiler.types["Index"].bits, compiler.types["Index"].domain.hi), (16, 2))
        self.assertIn(f"typedef {grid.element.c} {grid.c}[2];", c)
        with self.assertRaisesRegex(LanError, "index out of bounds|cannot prove"):
            compile_source(declarations.replace("grid.value[1][2]", "grid.value[1][3]"))
        with self.assertRaisesRegex(LanError, "array must have positive length"):
            compile_source(declarations.replace("Matrix(Cell, 2, 3)", "Matrix(Cell, 2, 0)"))

    def test_type_parameters_compose_property_objects_and_preserve_contracts(self):
        declarations = """
            type Width = {bits: 16};
            type Word(P: (type)) { return uint · P · Width; }
            type MakeWidth(n: uint8) { return {bits: n}; }
            type FromLocal(n: uint8) { return Word({bits: n}); }
            type FirstError = {1};
            type SecondError = {0};
            type R1 = uint8[0, 0] | FirstError;
            type R2 = uint8[1, 1] | SecondError;
            type Identity(T: type) { return T; }
            type A = Identity(R1);
            type B = Identity(R2);
            type C = FromLocal(16);
            type D = Word((Width));
            type E = Word(MakeWidth(16));
        """
        compiler = Compiler(declarations, "<test>")
        compiler.compile()
        self.assertNotEqual(compiler.types["A"].result_arms, compiler.types["B"].result_arms)
        self.assertEqual(compiler.types["A"].result_arms, compiler.types["R1"].result_arms)
        self.assertEqual(compiler.types["B"].result_arms, compiler.types["R2"].result_arms)
        for name in ("C", "D", "E"):
            self.assertEqual(compiler.types[name], compiler.types["uint16"])
        with self.assertRaisesRegex(LanError, "conflicting type object property 'bits'"):
            compile_source(declarations.replace("FromLocal(16)", "FromLocal(32)"))
        with self.assertRaisesRegex(LanError, "exceeded the compile-time recursion limit"):
            compile_source("type loop(P: type) { return loop(P · {bits: 16}); } type A = loop({bits: 16});")

    def test_type_and_value_parameters_cannot_be_interchanged(self):
        declarations = """
            type Identity(T: type) { return T; }
            type Row(T: type, length: uint8) { return T[length]; }
            type Value(n: uint8) { return uint8[0, n]; }
            type Invalid(T: type) { if (T) { return uint8; } return T; }
        """
        for invocation, error in (
                ("Identity(16)", "expected a type expression"),
                ("Value({bits: 16})", "expression is not supported at compile time"),
                ("Value(uint16)", "is not a compile-time variable"),
                ("Row(uint16, 256)", "compile-time value 256 does not fit"),
                ("Invalid(uint8)", "type parameter 'T' cannot be used as a compile-time value")):
            with self.subTest(invocation=invocation):
                with self.assertRaisesRegex(LanError, error):
                    compile_source(declarations + f"type Test = {invocation};")

    def test_type_arguments_cross_source_and_table_imports(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            types = root / "types.lang"
            source = """
                type Cell = {value: uint16[0, 1000]};
                type Shape = {len: 3, elem: Cell};
                export { Cell, Shape };
            """
            _, table = ModuleGraph(types, Compiler(source, types), library=True).compile()
            types.with_suffix(".table").write_text(table, encoding="utf-8")
            (root / "factory.lang").write_text("""
                type Cell = {value: uint8};
                type Row(T: type, length: uint8) { return array · {len: length, elem: T}; }
                type WithShape(P: type) { return array · P; }
                export { Row, WithShape };
            """, encoding="utf-8")
            provider = root / "provider.lang"
            source = """
                import { Cell, Shape } from "./types.table";
                import { Row as MakeRow, WithShape } from "./factory.lang";
                type Grid = MakeRow(WithShape(Shape), 2);
                define read(grid: Grid) -> uint16[0, 1000] { return grid[1][2].value; }
                export { Grid, read };
            """
            _, table = ModuleGraph(provider, Compiler(source, provider), library=True).compile()
            provider.with_suffix(".table").write_text(table, encoding="utf-8")
            consumer = """
                import { Grid, read } from "./provider.table";
                define use(grid: Grid) -> uint16[0, 1000] {
                    grid[1][2].value = 700;
                    return read(grid);
                }
            """
            c, _ = compile_source(consumer, root / "consumer.lang")
            self.assertIn("uint16_t f_value;", c)
            with self.assertRaisesRegex(LanError, "cannot prove"):
                compile_source(consumer.replace("= 700", "= 1001"), root / "consumer.lang")

    def test_a_type_function_evaluates_to_a_value_set(self):
        _, table = compile_source(BELOW + """
            define f() -> uint8 {
                define v: below(10) in {uint8, uint16} 16 = 5;
                return v;
            }
        """)
        # The argument is a compile-time constant, so the annotation is the set
        # the body returned with `n` replaced.
        self.assertIn("define entity id#1 v: uint8_t & [0, 9]", table)

    def test_a_computed_index_bound_needs_no_runtime_check(self):
        c, table = compile_source(BELOW + """
            define pick(table: uint8[4], i: below(4) in {uint8} 8) -> uint8 {
                return table[i];
            }
            define f() -> uint8 {
                define row: uint8[4] = [1, 2, 3, 4];
                return pick(row, 3);
            }
        """)
        self.assertIn("param id#2 i: uint8_t & [0, 3]", table)
        self.assertIn("verify index id#1 at", table)
        self.assertIn("[0, 3] subset [0, 3] => proven", table)
        # The subscript is proven in range, so the body carries no trap.
        body = c[c.index("uint8_t lan_fn_pick("):]
        body = body[:body.index("\n}") + 2]
        self.assertNotIn("__builtin_trap", body)

    def test_a_type_function_runs_compile_time_loops(self):
        _, table = compile_source(MASK + """
            define f() -> uint16 {
                define small: mask_of(4) in {uint8, uint16, uint32} 32 = 15;
                define wide: mask_of(9) in {uint8, uint16, uint32} 32 = 511;
                return small + wide;
            }
        """)
        # 0..15 fits uint8, 0..511 does not, so the seal picks the smallest
        # candidate container that holds the computed set.
        self.assertIn("define entity id#1 small: uint8_t & [0, 15]", table)
        self.assertIn("define entity id#2 wide: uint16_t & [0, 511]", table)

    def test_a_type_function_may_recurse(self):
        _, table = compile_source(UPTO + """
            define f() -> uint8 {
                define v: upto(3) in {uint8} 8 = 3;
                return v;
            }
        """)
        self.assertIn("define entity id#1 v: uint8_t & [0, 3]", table)
        with self.assertRaisesRegex(LanError, "exceeded the compile-time recursion limit"):
            compile_source("""
                type grow(n: uint8) {
                    return grow(n + 1);
                }
                define f() -> uint8 {
                    define v: grow(1) in {uint8} 8 = 1;
                    return v;
                }
            """)

    def test_a_boolean_argument_selects_a_branch(self):
        _, table = compile_source(CHOOSE + """
            define f(flag: boolean) -> uint8 {
                define low: choose(true) in {uint8} 8 = 3;
                define high: choose(false) in {uint8} 8 = 4;
                if (flag) { return low; }
                return high;
            }
        """)
        self.assertIn("define entity id#2 low: uint8_t & [0, 3]", table)
        self.assertIn("define entity id#3 high: uint8_t & [4, 7]", table)

    def test_a_type_function_may_return_a_signed_set(self):
        _, table = compile_source("""
            type around(radius: uint8) {
                return int16[0, radius] | int16[0 - radius, 0 - 1];
            }
            define f() -> int16 {
                define v: around(3) in {int8, int16} 16 = 0 - 3;
                return v;
            }
        """)
        self.assertIn("define entity id#1 v: int8_t & [-3, 3]", table)

    def test_a_computed_type_requires_a_physical_seal(self):
        with self.assertRaisesRegex(LanError, "a computed type requires a physical seal"):
            compile_source(BELOW + """
                define f() -> uint8 {
                    define v: below(4) = 1;
                    return v;
                }
            """)
        # The seal also has to contain the computed set.
        with self.assertRaisesRegex(LanError, "no physical seal candidate up to 8 bits"):
            compile_source(MASK + """
                define f() -> uint8 {
                    define v: mask_of(9) in {uint8} 8 = 1;
                    return v;
                }
            """)

    def test_argument_count_and_parameter_types_are_checked(self):
        with self.assertRaisesRegex(LanError, "expects 1 arguments"):
            compile_source(BELOW + """
                define f() -> uint8 {
                    define v: below(1, 2) in {uint8} 8 = 1;
                    return v;
                }
            """)
        with self.assertRaisesRegex(LanError, "parameter requires an integer or boolean type"):
            compile_source("""
                type bad(n: Node) {
                    return uint8[0, 1];
                }
                type Node = { value: uint8 };
                define f() -> uint8 {
                    define v: bad(1) in {uint8} 8 = 1;
                    return v;
                }
            """)

    def test_a_missing_or_bare_return_is_rejected(self):
        with self.assertRaisesRegex(LanError, "did not return a type or value set"):
            compile_source("""
                type silent(n: uint8) {
                    define value = n;
                }
                define f() -> uint8 {
                    define v: silent(1) in {uint8} 8 = 1;
                    return v;
                }
            """)
        with self.assertRaisesRegex(LanError, "type function must return a type object, type or value set"):
            compile_source("""
                type silent(n: uint8) {
                    if (n > 0) { return; }
                    return uint8[0, 1];
                }
                define f() -> uint8 {
                    define v: silent(1) in {uint8} 8 = 1;
                    return v;
                }
            """)

    def test_a_type_function_crosses_the_module_interface(self):
        provider = BELOW + """
            define pick(row: uint8[4], i: below(4) in {uint8} 8) -> uint8 {
                return row[i];
            }
            export { below, pick };
        """
        consumer = """
            import { below, pick } from "./provider.lang";
            define use() -> uint8 {
                define row: uint8[4] = [1, 2, 3, 4];
                define i: below(3) in {uint8} 8 = 2;
                return pick(row, i);
            }
        """
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / "provider.lang").write_text(provider, encoding="utf-8")
            source = root / "consumer.lang"
            source.write_text(consumer, encoding="utf-8")
            c, table = ModuleGraph(source, Compiler(consumer, source.name)).compile()
            # A type function is evaluated from its own source, so the
            # consumer's set comes from the provider's body, not from the table.
            self.assertRegex(table, r"evaluate imported type below from '(?:[^'\n]+/)?provider\.lang'")
            self.assertIn("define entity id#2 i: uint8_t & [0, 2]", table)
            self.assertIn("fn_pick", c)


if __name__ == "__main__":
    unittest.main()
