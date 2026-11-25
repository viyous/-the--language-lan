import unittest

from lanc import compile_source


def parameters(count):
    return "ctx: Ctx, " + ", ".join(f"b{index}: boolean" for index in range(count))


def branches(count, template="    if (b{index}) {{ ctx.a = 1; }}"):
    return "\n".join(template.format(index=index) for index in range(count))


def live_tail(count, expr="ctx.b"):
    uses = " + ".join(f"(b{i} as uint8)" for i in range(count))
    return f"    define live_sum: uint16 = {uses};\n    if (live_sum > 1000) {{ return 0; }}\n    return {expr};"


def program(count, template="    if (b{index}) {{ ctx.a = 1; }}", tail=None):
    if tail is None:
        tail = live_tail(count, "ctx.b")
    return (f"type Ctx = {{ a: uint8, b: uint8 }};\n"
            f"define f({parameters(count)}) -> uint8 {{\n{branches(count, template)}\n{tail}\n}}")


class PathBudgetTests(unittest.TestCase):
    def test_independent_branches_are_folded_instead_of_rejected(self):
        _, table = compile_source(program(12))
        self.assertIn("path budget", table)
        self.assertNotIn("too many live paths", table)

    def test_a_program_that_would_explode_still_compiles(self):
        _, table = compile_source(program(14))
        self.assertIn("path budget", table)

    def test_folding_keeps_the_facts_every_path_agrees_on(self):
        template = "    if (b{index}) {{ ctx.a = 1; acc += 1; }} else {{ ctx.a = 1; }}"
        body = "    define acc: uint16 = 0;\n" + branches(9, template) + "\n" + live_tail(9, "ctx.a")
        src = f"type Ctx = {{ a: uint8, b: uint8 }};\ndefine f({parameters(9)}) -> uint8 {{\n{body}\n}}"
        _, table = compile_source(src)
        self.assertIn("path budget", table)
        self.assertIn("return {1}", table)

    def test_a_small_branch_count_stays_fully_path_sensitive(self):
        _, table = compile_source(program(3))
        self.assertNotIn("path budget", table)


if __name__ == "__main__":
    unittest.main()
