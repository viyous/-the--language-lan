import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lanc import Compiler, LanError, ModuleGraph, compile_source


RECORD = """
type Packet = { value: uint8 };
enum LookupError: uint8 { Missing = 0, Invalid = 2 }
type Found = Packet | LookupError;
type Code = uint8[7,7] | LookupError;
define find(ok: boolean) -> Found {
    if (!ok) { return LookupError.Missing; }
    return {value: 7};
}
define forward(ok: boolean) -> Found { return find(ok); }
define read(ok: boolean) -> Code { return forward(ok)?.value; }
"""

ARRAY = """
enum Err: uint8 { Missing = 0 }
type Pair = uint8[2] | Err;
type Code = uint8[17,17] | Err;
define pair(ok: boolean) -> Pair {
    if (!ok) { return Err.Missing; }
    return [8, 9];
}
define read(ok: boolean) -> Code {
    define p = pair(ok)?;
    return p[0] + p[1];
}
"""

BORROW = """
type Packet = { key: uint8 };
enum Err: uint8 { Missing = 0 }
type Key = uint8 · {pass: alias, from: 0, field: key} | Err;
type Window = uint8[..] · {from: 0, offset: 0, len: {2}} | Err;
type Code = uint8[7,7] | Err;
define key(p: Packet, ok: boolean) -> Key {
    if (!ok) { return Err.Missing; }
    return p.key;
}
define window(a: uint8[3], ok: boolean) -> Window {
    if (!ok) { return Err.Missing; }
    return a[0,2);
}
define relay_key(p: Packet, ok: boolean) -> Key { return key(p, ok)?; }
define relay_window(a: uint8[3], ok: boolean) -> Window { return window(a, ok)?; }
define read(ok: boolean) -> Code {
    define a: uint8[3] = [1, 2, 3];
    define view = relay_window(a, ok)?;
    view[1] = 7;
    define p: Packet = {key: a[1]};
    define ref = relay_key(p, ok)?;
    return ref;
}
"""

OWNED = """
type Packet = { value: uint8 };
enum Err: uint8 { Missing = 0 }
type Owner = Packet · {storage: heap};
type OwnedResult = Owner | Err;
type Code = uint8[7,7] | Err;
define create(ok: boolean) -> OwnedResult {
    if (!ok) { return Err.Missing; }
    define p: Owner = {value: 7};
    return p;
}
define relay(ok: boolean) -> OwnedResult { return create(ok)?; }
define read(ok: boolean) -> Code {
    define p = relay(ok)?;
    define value = p.value;
    free(p);
    return value;
}
"""


class CompositeResultTests(unittest.TestCase):
    def test_zero_error_code_and_success_field_facts_survive_forwarding(self):
        c, trace = compile_source(RECORD)
        self.assertIn("uint8_t ok;", c)
        self.assertRegex(c, r"\.ok = 0, \.payload\.error = .*UINT64_C\(0\)")
        self.assertIn(".ok = 1, .payload.value =", c)
        self.assertRegex(c, r"if \(!lan_result_\d+_\d+\.ok\)")
        self.assertEqual(c.count("= lan_fn_forward("), 1)
        self.assertIn("propagate tagged error", trace)

    def test_fixed_array_contents_are_known_only_on_success(self):
        _, trace = compile_source(ARRAY)
        self.assertIn("return {17}", trace)

    def test_scalar_error_can_propagate_inside_a_composite_initializer(self):
        source = RECORD + """
define from_scalar(ok: boolean) -> Found { return {value: read(ok)?}; }
type Pair = uint8[2] | LookupError;
define from_scalar_pair(ok: boolean) -> Pair { return [read(ok)?, 9]; }
define score(ok: boolean) -> uint8[16,16] | LookupError {
    define p = from_scalar_pair(ok)?;
    return from_scalar(ok)?.value + p[1];
}
"""
        compile_source(source)

    def test_error_enum_identity_and_domain_are_checked(self):
        failures = (
            ("enum Other: uint8 { Missing = 0, Invalid = 2 }",
             "uint8[7,7] | Other", "explicit result contract"),
            ("type MissingOnly = LookupError & {0};",
             "Packet | MissingOnly", "caller's named error arm"),
            ("", "void", "explicit result contract"),
        )
        for declaration, result, message in failures:
            with self.subTest(result=result), self.assertRaisesRegex(LanError, message):
                compile_source(RECORD + declaration + f"""
define invalid(ok: boolean) -> {result} {{
    define p = find(ok)?;
}}
""")
        with self.assertRaisesRegex(LanError, "enum identity differs"):
            compile_source(RECORD + """
enum Other: uint8 { SameCode = 0 }
define invalid() -> Found { return Other.SameCode; }
""")

    def test_borrowed_payloads_track_original_storage(self):
        compile_source(BORROW)
        for expression, expected in [
            ("window(b, ok)", "so its source can be checked"),
            ("window(b, ok)?", "declared borrowed parameter"),
        ]:
            with self.subTest(expression=expression), self.assertRaisesRegex(LanError, expected):
                compile_source(BORROW + f"""
define invalid(a: uint8[3], b: uint8[3], ok: boolean) -> Window {{
    return {expression};
}}
""")
        with self.assertRaisesRegex(LanError, "declared borrowed parameter"):
            compile_source(BORROW + """
define invalid(p: Packet, ok: boolean) -> Key {
    define local: Packet = {key: 7};
    return key(local, ok)?;
}
""")

    def test_owning_success_must_be_released(self):
        c, trace = compile_source(OWNED)
        self.assertIn("transfer ownership", trace)
        self.assertIn("free(", c)
        with self.assertRaisesRegex(LanError, "heap leak: 'p'"):
            compile_source(OWNED.replace("free(p);", ""))
        with self.assertRaisesRegex(LanError, "Tombstone"):
            compile_source(OWNED.replace("return value;", "return p.value;"))
        with self.assertRaisesRegex(LanError, "must be unwrapped or returned"):
            compile_source(OWNED + "define discard() -> void { create(true); }")

    def test_error_exit_checks_preexisting_owners(self):
        with self.assertRaisesRegex(LanError, "heap leak: 'prior'"):
            compile_source(OWNED.replace("define p = relay(ok)?;", """
    define prior: uint8 · {storage: heap} = 1;
    define p = relay(ok)?;
    free(prior);
"""))

    def test_error_during_heap_record_initialization_releases_partial_object(self):
        c, _ = compile_source(RECORD + """
type Owner = Packet · {storage: heap};
type Result = Owner | LookupError;
define create(ok: boolean) -> Result {
    define p: Owner = {value: find(ok)?.value};
    return p;
}
""")
        self.assertRegex(c, r"if \(!lan_result_\d+_\d+\.ok\) \{ free\(lv_\d+_p\); return")

    def test_heap_array_retains_its_length_after_unwrapping(self):
        compile_source("""
enum Err: uint8 { Missing = 0 }
type Buffer = uint8[..] · {len: {2}} · {storage: heap};
type Result = Buffer | Err;
define create(ok: boolean) -> Result {
    if (!ok) { return Err.Missing; }
    define buf: uint8[2] · {storage: heap};
    return buf;
}
define read(ok: boolean) -> uint8[1,255] | Err {
    define buf = create(ok)?;
    buf[1] = 7;
    define value = buf[1];
    free(buf);
    return value;
}
""")

    def test_propagation_in_call_arguments_can_skip_the_composite_call(self):
        c, _ = compile_source(RECORD + """
type Failure = Code & {0};
define fail() -> Failure { return LookupError.Missing; }
define skip() -> Found { return find((fail()? == 7))?; }
""")
        body = c.split("lan_fn_skip(void) {")[1]
        self.assertNotIn("lan_fn_find(", body)

    def test_both_branches_observe_a_consumed_owner(self):
        compile_source("""
enum Err: uint8 { Missing = 0 }
type Owner = uint8[7,7] · {storage: heap};
type Result = Owner | Err;
define recycle(p: Owner, ok: boolean) -> Result {
    if (!ok) { free(p); return Err.Missing; }
    return p;
}
define read(ok: boolean) -> uint8[7,7] | Err {
    define old: Owner = 7;
    define value = recycle(old, ok)?;
    define copy = value;
    free(value);
    return copy;
}
""")

    def test_both_branches_observe_mutations_even_when_result_is_forwarded(self):
        source = RECORD + """
define touch(mark: uint8 · {pass: alias}, ok: boolean) -> Found {
    if (!ok) { mark = 1; return LookupError.Missing; }
    mark = 2;
    return {value: 7};
}
define relay(mark: uint8 · {pass: alias}, ok: boolean) -> Found {
    return touch(mark, ok)?;
}
define unwrap(mark: uint8 · {pass: alias}, ok: boolean) -> Code {
    return relay(mark, ok)?.value;
}
define read_mark(ok: boolean) -> uint8[1,2] {
    define mark: uint8 = 0;
    unwrap(mark, ok);
    return mark;
}
"""
        compile_source(source)
        with self.assertRaisesRegex(LanError, r"fits \{2\}"):
            compile_source(source.replace("-> uint8[1,2]", "-> uint8[2,2]"))

    def test_table_only_import_preserves_types_summaries_and_borrows(self):
        modules = (
            (RECORD, "Packet, LookupError, Found, forward",
             "define use(ok: boolean) -> uint8[7,7] | LookupError { return forward(ok)?.value; }"),
            (ARRAY, "Err, Pair, pair",
             "define use(ok: boolean) -> uint8[9,9] | Err { return pair(ok)?[1]; }"),
            (BORROW, "Err, Window, window, Key, relay_key, Packet", """
define use(ok: boolean) -> uint8[7,7] | Err {
    define a: uint8[3] = [1, 2, 3];
    define view = window(a, ok)?;
    view[1] = 7;
    define p: Packet = {key: a[1]};
    define ref = relay_key(p, ok)?;
    return ref;
}
"""),
            (OWNED, "Err, OwnedResult, relay", """
define use(ok: boolean) -> uint8[7,7] | Err {
    define p = relay(ok)?;
    define value = p.value;
    free(p);
    return value;
}
"""),
        )
        for source, names, caller in modules:
            with (self.subTest(exports=names),
                  TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch):
                path = Path(scratch) / "provider.lang"
                compiler = Compiler(source + f"export {{ {names} }};", path.name)
                provider_c, table = ModuleGraph(path, compiler, library=True).compile()
                path.with_suffix(".table").write_text(table, encoding="ascii")
                consumer_c, _ = compile_source(
                    f'import {{ {names} }} from "./provider.table";\n{caller}', path.with_name("consumer.lang"))
                signatures = re.findall(r"\b(lan_result_[a-f0-9]{24} lan_api_\w+\()", provider_c)
                self.assertTrue(any(f"extern {signature}" in consumer_c for signature in signatures))
