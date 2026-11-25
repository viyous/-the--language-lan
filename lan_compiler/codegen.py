"""C helper and polynomial code generation and compilation entry points."""

from __future__ import annotations

import re
from dataclasses import field

from semantic_ir import Storage
from float_domain import c_literal, conversion_input_domain
from value_domain import ACTIVE_ORACLES, ACTIVE_SOLVER_STEPS, DomainComplexity
from table_interface import api_symbol, c_field, c_ident
from .errors import LanError

from .typesys import (Array, C_HEADER, FunctionBinding, HeapArray, Interface, is_callback_enum, Record,
                      Scalar, Slice, WIDE)
from .domain_utils import ascii_table_trace, c_int, c_uint, container
from .interface_types import chain_api_signature, encode_interface_type
from .c_emit import c_counted_loop, c_predicate_clauses, indent

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class CodegenMixin:
    def helper_code(self):
        lines = []
        if any(kind.startswith("float_") or
               (kind in ("record_value_equal", "slice_value_equal") and self.contains_float(domain))
               for kind, domain, _ in self.helpers):
            lines.extend([
                "/* Lan floating operations require IEEE binary32/binary64 and round-to-nearest-even. */",
                "#ifdef __FAST_MATH__",
                '#error "Lan floating semantics do not support fast-math"',
                "#endif",
                '_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && DBL_MANT_DIG == 53, "unsupported floating formats");',
                '_Static_assert(sizeof(float) == 4 && sizeof(double) == 8, "unsupported floating storage");',
                '_Static_assert(FLT_EVAL_METHOD == 0, "Lan requires arithmetic at the declared floating precision");',
                "#pragma STDC FP_CONTRACT OFF",
            ])
        if any(kind == "poly" and self.poly_uses_limbs(poly) for kind, poly, _ in self.helpers):
            lines.extend(self.poly_limb_support())
        # Release helpers may call each other in both directions, so every one
        # of them is declared before the first definition.
        lines.extend(f"static void {name}({domain.c} *p);"
                     for (kind, domain, _), name in list(self.helpers.items())
                     if kind == "free_tree")
        for (kind, domain, extra), name in list(self.helpers.items()):
            if kind == "static_assert":
                lines.append(extra)
                continue
            if kind == "global_temp":
                c_type, var_name = extra
                lines.append(f"static {c_type} {var_name};")
                continue
            if kind == "callback_fnptr":
                lines.append(self.c_fnptr_typedef(name, extra, foreign=False))
                continue
            if kind == "interface_rebind":
                lines.extend([
                    f"static inline lan_interface {name}(lan_interface source, size_t *offsets, size_t count) {{",
                    "    memmove(offsets, source.offsets, count * sizeof(*offsets));",
                    "    return (lan_interface){source.data, offsets};",
                    "}",
                ])
                continue
            if kind == "interface_offsets":
                offsets = ", ".join(f"offsetof({domain.c}, {c_field(field)})" for field, _ in extra.fields)
                lines.append(f"static const size_t {name}[] = {{{offsets}}};")
                continue
            if kind == "keyed_offsets":
                offsets = ", ".join(f"offsetof({domain.c}, {c_field(field)})" for field, _ in domain.keyed_fields)
                lines.append(f"static const size_t {name}[] = {{{offsets}}};")
                continue
            if kind == "keyed_access":
                host_c, qualifier, offsets = extra
                lines.append(f"static inline {qualifier}{domain} *{name}({qualifier}{host_c} *base, size_t lan_keyed_index_0) {{ "
                             f"return ({qualifier}{domain} *)(({qualifier}unsigned char *)base + {offsets}[lan_keyed_index_0]); }}")
                continue
            if kind == "chain_reachable":
                lines.extend([
                    f"static inline int {name}({domain} cursor, const void *result) {{",
                    "    while (cursor) {",
                    "        if ((const void *)cursor == result) return 1;",
                    "        cursor = cursor->f_next;",
                    "    }",
                    "    return 0;",
                    "}",
                ])
                continue
            if kind == "slice_from_end":
                lines.append(f"static inline {domain} {name}(lan_slice text, size_t offset) {{ "
                             f"return ((const {domain} *)text.data)[text.len - offset]; }}")
                continue
            if kind == "copy_view_arg":
                source_c, destination, pointer, copy = extra
                lines.extend([
                    f"static inline {domain} {name}(const void *address) {{",
                    f"    {domain} temp;",
                    f'    _Static_assert(sizeof({destination}) == sizeof({source_c}), "Lan composite bit view width");',
                    f"    {copy}({pointer}, address, sizeof({destination}));",
                    "    return temp;",
                    "}",
                ])
                continue
            if kind == "copy_array_arg":
                lines.extend([
                    f"static inline {domain} {name}(const void *address) {{",
                    f"    {domain} temp;",
                    f"    {extra}(temp.data, address, sizeof(temp.data));",
                    "    return temp;",
                    "}",
                ])
                continue
            if kind == "debug_index":
                lines.extend([
                    "#include <assert.h>",
                    f"static inline uint64_t {name}(lan_int index, uint64_t length) {{",
                    "    if (index < 0) goto lan_invalid_index;",
                    "    if (index >= (lan_int)length) goto lan_invalid_index;",
                    "    return (uint64_t)index;",
                    "lan_invalid_index:",
                    '    assert(0 && "invalid array index");',
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "slice_index":
                check, assume = extra
                lines.extend([
                    f"static inline lan_slice {name}(lan_slice base, lan_int offset) {{",
                    f"    size_t position = {check}(offset, base.len);",
                    f"    {assume}(base.data != NULL);",
                    f"    return (lan_slice){{(({domain} *)base.data) + position, 1}};",
                    "}",
                ])
                continue
            if kind == "assume_index":
                lines.extend([
                    f"static inline uint64_t {name}(lan_int index, lan_int length) {{",
                    f"    {extra}(index >= 0);",
                    f"    {extra}(index < length);",
                    "    return (uint64_t)index;",
                    "}",
                ])
                continue
            if kind in ("assumed_result", "checked_result"):
                c_type, param, checks = extra
                lines.append(f"static inline {c_type} {name}({c_type} {param}) {{ {checks} return {param}; }}")
                continue
            if kind == "binary_eval":
                ret_c, left_c, right_c, body = extra
                lines.append(f"static inline {ret_c} {name}({left_c} a, {right_c} b) {{ return {body}; }}")
                continue
            if kind == "unary_eval":
                ret_c, arg_c, body = extra
                stmt = body if "return " in body else f"return {body}"
                lines.append(f"static inline {ret_c} {name}({arg_c} a) {{ {stmt}; }}")
                continue
            if kind == "count_in_span":
                membership, access = extra
                lines.extend([
                    f"static inline uint32_t {name}(lan_slice descriptor) {{",
                    f"    const {domain} *data = descriptor.data;",
                    "    uint32_t lan_count_total = 0;",
                    *indent(c_counted_loop("lan_i", "0", "descriptor.len",
                                           [f"lan_count_total += {membership}({access});"])),
                    "    return lan_count_total;",
                    "}",
                ])
                continue
            if kind == "contract_assume":
                lines.extend([
                    f"static inline void {name}(int lan_assume) {{",
                    "#if defined(__clang__)",
                    "    __builtin_assume(lan_assume != 0);",
                    "#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))",
                    "    if (!lan_assume) __builtin_unreachable();",
                    "#else",
                    "    (void)lan_assume;",
                    "#endif",
                    "}",
                ])
                continue
            if kind == "oracle_check":
                checked = [
                    "#include <assert.h>",
                    f"#define {name}(test) ((void)((test) ? 0 : (assert(test), __builtin_trap(), 0)))",
                ]
                if self.debug_bounds:
                    lines.extend(checked)
                else:
                    lines.extend([
                        "#if defined(LAN_ORACLE_DEBUG)",
                        *checked,
                        "#else",
                        f"static inline void {name}(int lan_assume) {{",
                        "#if defined(__clang__)",
                        "    __builtin_assume(lan_assume != 0);",
                        "#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))",
                        "    if (!lan_assume) __builtin_unreachable();",
                        "#else",
                        "    (void)lan_assume;",
                        "#endif",
                        "}",
                        "#endif",
                    ])
                continue
            if kind == "float_convert":
                source_c, target_c = extra
                lines.extend([f"static inline {target_c} {name}({source_c} value) {{",
                              f"    volatile {target_c} result = ({target_c})value;",
                              "    return result;", "}"])
                continue
            if kind == "float_binary":
                op, bits = extra
                ctype = "float" if bits == 32 else "double"
                lines.extend([f"static inline {ctype} {name}({ctype} x, {ctype} y) {{",
                              f"    volatile {ctype} result = x {op} y;", "    return result;", "}"])
                continue
            if kind == "disjoint_object_copy":
                lines.extend([
                    f"static inline void *{name}(void *restrict dst, const void *restrict src, size_t size) {{",
                    "    return memcpy(dst, src, size);",
                    "}",
                ])
                continue
            if kind == "object_scalar_view":
                lines.extend([f"static inline {domain} {name}(const void *address) {{",
                              f"    {domain} value; memcpy(&value, address, sizeof(value));",
                              "    return value;", "}"])
                continue
            if kind == "float_view":
                source_c, target_c = extra
                lines.extend([f"static inline {target_c} {name}({source_c} value) {{",
                              f"    {target_c} result; memcpy(&result, &value, sizeof(result));",
                              "    return result;", "}"])
                continue
            if kind == "float_integer_in":
                bits,bounds,predicate = extra
                ctype = "float" if bits == 32 else "double"
                truncation = "truncf" if bits == 32 else "trunc"
                lines.extend([f"static inline uint8_t {name}({ctype} value) {{",
                              f"    if (!isfinite(value)) goto lan_invalid;",
                              f"    if (value != {truncation}(value)) goto lan_invalid;",
                              f"    if (!{bounds}(value)) goto lan_invalid;",
                              "    return " + (f"{predicate}(value < 0 ? (lan_int)value : (lan_int)(lan_uint)value)" if predicate else "1") + ";",
                              "lan_invalid:", "    return 0;", "}"])
                continue
            if kind == "float_in":
                ctype = "float" if domain.bits == 32 else "double"
                clauses = [[f"value >= {c_literal(lo, domain.bits)}", f"value <= {c_literal(hi, domain.bits)}"]
                           for lo, hi in domain.intervals]
                if domain.filters:
                    clauses = [[f"value >= {c_literal(p.lo, domain.bits)}", f"value <= {c_literal(p.hi, domain.bits)}",
                                f"{'' if p.integer is not None else '!'}{predicate}(value)"]
                               for p, predicate in zip(domain.filters, extra)]
                if domain.zeros:
                    zeros = ["value == 0"]
                    if domain.zeros != 3:
                        zeros.append("!signbit(value)" if domain.zeros == 1 else "signbit(value)")
                    clauses.append(zeros)
                if domain.pos_inf:
                    clauses.append(["value == INFINITY"])
                if domain.neg_inf:
                    clauses.append(["value == -INFINITY"])
                if domain.nan:
                    clauses.append(["isnan(value)"])
                lines.extend([f"static inline uint8_t {name}({ctype} value) {{",
                              *c_predicate_clauses(clauses), "}"])
                continue
            if kind == "float_trap":
                ctype = "float" if domain.bits == 32 else "double"
                lines.extend([f"static inline {ctype} {name}({ctype} value) {{",
                              f"    if (!{extra}(value)) goto lan_invalid;", "    return value;",
                              "lan_invalid:", "    __builtin_trap();", "}"])
                continue
            if kind == "float_sat":
                bits, lower, upper, allow_nan = extra
                ctype = "float" if bits == 32 else "double"
                lines.extend([f"static inline {ctype} {name}({ctype} value) {{",
                              "    if (isnan(value)) goto lan_nan;",
                              f"    if (value < {c_literal(lower,bits)}) goto lan_low;",
                              f"    if (value > {c_literal(upper,bits)}) goto lan_high;",
                              "    return value;",
                              "lan_low:", f"    return {c_literal(lower,bits)};",
                              "lan_high:", f"    return {c_literal(upper,bits)};",
                              "lan_nan:", "    " + ("return value;" if allow_nan else "__builtin_trap();"), "}"])
                continue
            if kind == "float_integer":
                source_bits, target_bits, signed, rounding, overflow, guard, predicate = extra
                ctype = "float" if source_bits == 32 else "double"
                target = f"{'int' if signed else 'uint'}{target_bits}_t"
                rounding_function = rounding + ("f" if source_bits == 32 else "")
                lines.append(f"static inline {target} {name}({ctype} value) {{")
                if overflow == "sat":
                    allowed = conversion_input_domain(domain, source_bits, rounding)
                    lines.extend(["    if (isnan(value)) goto lan_invalid;",
                                  f"    if (value < {c_literal(allowed.lo,source_bits)}) goto lan_low;",
                                  f"    if (value > {c_literal(allowed.hi,source_bits)}) goto lan_high;"])
                elif overflow == "trap":
                    lines.append(f"    if (!{guard}(value)) goto lan_invalid;")
                lines.append(f"    {target} result = ({target}){rounding_function}(value);")
                if overflow == "trap" and predicate is not None:
                    lines.append(f"    if (!{predicate}((lan_int)result)) goto lan_invalid;")
                lines.append("    return result;")
                if overflow == "sat":
                    lines.extend(["lan_low:", f"    return ({target}){c_int(domain.lo)};",
                                  "lan_high:", f"    return ({target}){c_int(domain.hi)};"])
                if overflow in ("sat", "trap"):
                    lines.extend(["lan_invalid:", "    __builtin_trap();"])
                lines.append("}")
                continue
            if kind == "shift_left":
                lines.extend([
                    f"static inline lan_int {name}(lan_int x, unsigned count) {{",
                    "    if (count >= 64) return 0;",
                    "    lan_uint bits = (lan_uint)x << count;",
                    "    if (x < 0) goto lan_negative;",
                    "    return (lan_int)bits;",
                    "lan_negative:",
                    "    return -1 - (lan_int)(~bits);",
                    "}",
                ])
                continue
            if kind == "shift_right":
                lines.extend([
                    f"static inline lan_int {name}(lan_int x, unsigned count) {{",
                    "    if (count >= 64) return x < 0 ? -1 : 0;",
                    "    if (x < 0) goto lan_negative;",
                    "    return (lan_int)((lan_uint)x >> count);",
                    "lan_negative:",
                    "    return -1 - (lan_int)((lan_uint)(~x) >> count);",
                    "}",
                ])
                continue
            if kind == "heap_array":
                lines.extend([
                    f"static inline lan_slice {name}(size_t len, size_t size) {{",
                    "    void *data = calloc(len, size);",
                    "    if (!data) goto lan_allocation_failed;",
                    "    return (lan_slice){data, len};",
                    "lan_allocation_failed:",
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "utf8_trap":
                # Decode one scalar at a time, rejecting non-shortest forms.
                lines.extend([
                    f"static inline lan_slice {name}(lan_slice s) {{",
                    "    const uint8_t *data = s.data;",
                    "    size_t i = 0;",
                    "lan_utf8_head: ;",
                    "    if (i >= s.len) goto lan_utf8_done;",
                    "    uint32_t scalar = data[i++];",
                    "    if (scalar < 0x80) goto lan_utf8_head;",
                    "    unsigned tails;",
                    "    uint32_t minimum;",
                    "    if (scalar < 0xc2) goto lan_utf8_invalid;",
                    "    if (scalar <= 0xdf) goto lan_utf8_two;",
                    "    if (scalar <= 0xef) goto lan_utf8_three;",
                    "    if (scalar > 0xf4) goto lan_utf8_invalid;",
                    "    tails = 3; minimum = 0x10000; scalar &= 0x07;",
                    "    goto lan_utf8_tail;",
                    "lan_utf8_three:",
                    "    tails = 2; minimum = 0x800; scalar &= 0x0f;",
                    "    goto lan_utf8_tail;",
                    "lan_utf8_two:",
                    "    tails = 1; minimum = 0x80; scalar &= 0x1f;",
                    "lan_utf8_tail: ;",
                    "    if (s.len - i < tails) goto lan_utf8_invalid;",
                    "    unsigned n = 0;",
                    "lan_utf8_tail_head: ;",
                    "    if (n >= tails) goto lan_utf8_scalar;",
                    "    uint8_t tail = data[i++];",
                    "    if ((tail & 0xc0) != 0x80) goto lan_utf8_invalid;",
                    "    scalar = (scalar << 6) | (tail & 0x3f);",
                    "    ++n;",
                    "    goto lan_utf8_tail_head;",
                    "lan_utf8_scalar:",
                    "    if (scalar < minimum) goto lan_utf8_invalid;",
                    "    if (scalar > 0x10ffff) goto lan_utf8_invalid;",
                    "    if (scalar < 0xd800) goto lan_utf8_head;",
                    "    if (scalar > 0xdfff) goto lan_utf8_head;",
                    "lan_utf8_invalid:",
                    "    __builtin_trap();",
                    "lan_utf8_done:",
                    "    return s;",
                    "}",
                ])
                continue
            if kind == "debug_native_slice_window":
                lines.extend([
                    f"static inline int {name}(const void *base, size_t count, lan_slice span, size_t stride) {{",
                    "    if (!base) goto lan_null_base;",
                    "    if (!span.data) goto lan_invalid_window;",
                    "    uintptr_t start = (uintptr_t)base, address = (uintptr_t)span.data;",
                    "    if (address < start) goto lan_invalid_window;",
                    "    uintptr_t distance = address - start;",
                    "    if (distance % stride) goto lan_invalid_window;",
                    "    uintptr_t index = distance / stride;",
                    "    if (index > count) goto lan_invalid_window;",
                    "    return span.len <= count - index;",
                    "lan_null_base:",
                    "    if (count) goto lan_invalid_window;",
                    "    if (span.data) goto lan_invalid_window;",
                    "    return !span.len;",
                    "lan_invalid_window:",
                    "    return 0;",
                    "}",
                ])
                continue
            if kind == "debug_native_record":
                lines.extend([
                    f"static inline int {name}(const {domain.c} *data) {{",
                    *(f"    if (!({check})) goto lan_native_record_invalid;" for check in extra),
                    "    return 1;",
                    "lan_native_record_invalid:",
                    "    return 0;",
                    "}",
                ])
                continue
            if kind == "debug_native_array":
                element = "(*data)[index]"
                if isinstance(domain.element, (Array, Record)):
                    element = f"&({element})"
                lines.extend([
                    f"static inline int {name}(const {domain.c} *data) {{",
                    "    size_t index = 0;",
                    "lan_native_array_scan:",
                    f"    if (index >= {domain.length}) goto lan_native_array_valid;",
                    f"    if (!{extra}({element})) goto lan_native_array_invalid;",
                    "    ++index;",
                    "    goto lan_native_array_scan;",
                    "lan_native_array_valid:",
                    "    return 1;",
                    "lan_native_array_invalid:",
                    "    return 0;",
                    "}",
                ])
                continue
            if kind == "debug_native_elements":
                predicate, by_reference = extra
                element = "&data[index]" if by_reference else "data[index]"
                lines.extend([
                    f"static inline lan_slice {name}(lan_slice span) {{",
                    f"    const {domain} *data = span.data;",
                    "    size_t index = 0;",
                    "lan_native_element_scan:",
                    "    if (index >= span.len) goto lan_native_element_done;",
                    f"    if (!{predicate}({element})) goto lan_invalid_native_element;",
                    "    ++index;",
                    "    goto lan_native_element_scan;",
                    "lan_native_element_done:",
                    "    return span;",
                    "lan_invalid_native_element:",
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "debug_native_text":
                lines.extend([
                    "#include <assert.h>",
                    f"static inline lan_slice {name}(const char *data, size_t available) {{",
                    "    size_t length = 0;",
                    "    if (!data) goto lan_invalid_native_text;",
                    "lan_native_text_scan:",
                    "    if (length > available) goto lan_invalid_native_text;",
                    f"    if (length > UINT64_C({domain})) goto lan_invalid_native_text;",
                    "    if (!data[length]) goto lan_native_text_end;",
                    "    ++length;",
                    "    goto lan_native_text_scan;",
                    "lan_native_text_end:",
                    f"    if (!{extra}(length)) goto lan_invalid_native_text;",
                    "    return (lan_slice){data, length};",
                    "lan_invalid_native_text:",
                    '    assert(0 && "native c_string violates its length or termination contract");',
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "debug_heap_length":
                lines.extend([
                    "#include <assert.h>",
                    f"static inline size_t {name}(lan_int count) {{",
                    f"    int valid = {extra}(count);",
                    "    if (!valid) goto lan_invalid_length;",
                    "    return (size_t)count;",
                    "lan_invalid_length:",
                    "    assert(valid);",
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "debug_bit_copy_width":
                lines.extend([
                    "#include <assert.h>",
                    f"static inline void {name}(uint64_t destination, uint64_t source) {{",
                    "    if (destination != source) goto lan_invalid_width;",
                    "    return;",
                    "lan_invalid_width:",
                    "    assert(destination == source);",
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "debug_bit_range":
                lines.extend([
                    "#include <assert.h>",
                    f"static inline void {name}(lan_int lo, lan_int hi, uint64_t length, int closed) {{",
                    "    if (lo < 0) goto lan_invalid_range;",
                    "    if (closed) goto lan_closed_range;",
                    "    if (lo >= hi) goto lan_invalid_range;",
                    "    if (hi > (lan_int)length) goto lan_invalid_range;",
                    "    return;",
                    "lan_closed_range:",
                    "    if (lo > hi) goto lan_invalid_range;",
                    "    if (hi >= (lan_int)length) goto lan_invalid_range;",
                    "    return;",
                    "lan_invalid_range:",
                    "    assert(0 && \"invalid bit range\");",
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "debug_slice_sub":
                lines.extend([
                    "#include <assert.h>",
                    f"static inline lan_slice {name}(lan_slice s, lan_int lo, lan_int hi, int rest, int closed) {{",
                    "    if (!rest) goto lan_slice_bounds;",
                    "    hi = (lan_int)s.len;",
                    "lan_slice_bounds:;",
                    "    if (lo < 0) goto lan_invalid_slice;",
                    "    if (hi < lo) goto lan_invalid_slice;",
                    "    if (hi > (lan_int)s.len) goto lan_invalid_slice;",
                    "    if (!closed) goto lan_slice_data;",
                    "    if (hi == (lan_int)s.len) goto lan_invalid_slice;",
                    "lan_slice_data: ;",
                    "    if (!s.len) goto lan_slice_pointer;",
                    f"    {extra}(s.data != NULL);",
                    "lan_slice_pointer:;",
                    "    const void *data = s.data;",
                    "    if (!lo) goto lan_slice_result;",
                    f"    data = (const {domain} *)data + (size_t)lo;",
                    "lan_slice_result:",
                    "    return (lan_slice){data, (size_t)hi - (size_t)lo + (size_t)closed};",
                    "lan_invalid_slice:",
                    "    assert(0 && \"invalid slice range\");",
                    "    __builtin_trap();",
                    "}",
                ])
                continue
            if kind == "slice_sub":
                lines.extend([
                    f"static inline lan_slice {name}(lan_slice s, size_t lo, size_t hi, int rest) {{",
                    "    if (!rest) goto lan_slice_bounds;",
                    "    hi = s.len;",
                    "lan_slice_bounds:",
                    f"    {extra}(lo <= hi);",
                    f"    {extra}(hi <= s.len);",
                    "    if (!s.len) goto lan_slice_pointer;",
                    f"    {extra}(s.data != NULL);",
                    "lan_slice_pointer:;",
                    "    const void *data = s.data;",
                    "    if (!lo) goto lan_slice_result;",
                    f"    data = (const {domain} *)data + lo;",
                    "lan_slice_result:",
                    "    return (lan_slice){data, hi - lo};",
                    "}",
                ])
                continue
            if kind == "slice_clone":
                lines.extend([
                    f"static inline void *{name}(const void *source, size_t bytes) {{",
                    "    size_t capacity = bytes;",
                    "    if (capacity) goto lan_clone_allocate;",
                    "    capacity = 1;",
                    "lan_clone_allocate:;",
                    "    void *copy = malloc(capacity);",
                    "    if (!copy) goto lan_allocation_failed;",
                    "    if (!bytes) goto lan_clone_done;",
                    "    memcpy(copy, source, bytes);",
                    "lan_clone_done:",
                    "    return copy;",
                    "lan_allocation_failed:",
                    "    abort();",
                    "}",
                    f"static inline void {name}_free(void **owner) {{ free(*owner); }}",
                ])
                continue
            if kind == "slice_copy":
                assume, checked = extra
                if checked:
                    lines.append("#include <assert.h>")
                lines.append(f"static inline void {name}(lan_slice dst, lan_slice src, size_t size) {{")
                if checked:
                    lines.append("    if (dst.len != src.len) goto lan_length_mismatch;")
                lines.extend([
                    f"    {assume}(dst.len == src.len);",
                    "    if (!dst.len) goto lan_copy_done;",
                    f"    {assume}(dst.data != NULL);",
                    f"    {assume}(src.data != NULL);",
                ])
                if domain == "memcpy":
                    lines.extend([
                        "    void *restrict destination = (void *)dst.data;",
                        "    const void *restrict source = src.data;",
                        "    memcpy(destination, source, dst.len * size);",
                    ])
                else:
                    lines.append("    memmove((void *)dst.data, src.data, dst.len * size);")
                lines.extend(["lan_copy_done:", "    return;"])
                if checked:
                    lines.extend(["lan_length_mismatch:", "    assert(dst.len == src.len);",
                                  "    __builtin_trap();"])
                lines.append("}")
                continue
            if kind == "slice_fill":
                lines.extend([
                    f"static inline void {name}(lan_slice dst, {domain} value) {{",
                    "    if (!dst.len) goto lan_fill_done;",
                    f"    {extra}(dst.data != NULL);",
                    f"    {domain} *data = ({domain} *)dst.data;",
                    *indent(c_counted_loop("i", 0, "dst.len", ["data[i] = value;"])),
                    "lan_fill_done:",
                    "    return;",
                    "}",
                ])
                continue
            if kind == "free_tree":
                # Post-order: every branch is released before its parent, so a
                # tree needs no visited set and no explicit stack.  One helper
                # per record type lets a tree hold children of another type.
                entries = extra or ()
                calls = [f"    {self.helper('free_tree', child, self.free_tree_plans[child.identity][1])}"
                         f"(p->{c_field(field)});" for field, child in entries]
                lines.extend([
                    f"static void {name}({domain.c} *p) {{",
                    "    if (!p) return;",
                    *calls,
                    "    free((void *)p);",
                    "}",
                ])
                continue
            if kind in ("record_rank_equal", "record_value_equal"):
                nullable = kind == "record_rank_equal" or extra == "nullable"
                rank = extra if kind == "record_rank_equal" else 0
                null_branch = ["    if (!a) goto lan_compare_null;", "    if (!b) goto lan_compare_null;"] if nullable else []
                null_result = ["lan_compare_null:", "    return a == b;"] if nullable else []
                body = self.value_equality_statements(domain, "*a", "*b", rank=rank)
                failure = ["lan_unequal:", "    return 0;"] if body else []
                lines.extend([f"static inline int {name}(const {domain.c} *a, const {domain.c} *b) {{",
                              *null_branch, *indent(body), "    return 1;", *failure, *null_result, "}"])
                continue
            if kind in ("slice_rank_equal", "slice_value_equal"):
                ranked = kind == "slice_rank_equal"
                args = "lan_slice a, lan_slice b" if ranked else "lan_slice a, lan_slice b, size_t size"
                assume, element_helper = extra if ranked else (extra, None)
                body = ([f"if (!{element_helper}(&x[i], &y[i])) goto lan_unequal;"] if ranked else
                        ["(void)size;", *self.value_equality_statements(domain, "x[i]", "y[i]")])
                lines.extend([
                    f"static inline int {name}({args}) {{",
                    "    if (a.len != b.len) goto lan_unequal;",
                    "    if (!a.len) goto lan_slice_compare_equal;",
                    f"    {assume}(a.data != NULL);",
                    f"    {assume}(b.data != NULL);",
                    f"    const {domain.c} *x = (const {domain.c} *)a.data, *y = (const {domain.c} *)b.data;",
                    "    size_t i = 0;",
                    "lan_slice_compare_head:",
                    "    if (i >= a.len) goto lan_slice_compare_equal;",
                    *indent(body),
                    "    ++i;",
                    "    goto lan_slice_compare_head;",
                    "lan_slice_compare_equal:",
                    "    return 1;",
                    "lan_unequal:",
                    "    return 0;",
                    "}",
                ])
                continue
            if kind == "slice_equal":
                lines.extend([
                    f"static inline int {name}(lan_slice a, lan_slice b, size_t size) {{",
                    "    if (a.len != b.len) goto lan_unequal;",
                    "    if (!a.len) goto lan_equal;",
                    f"    {extra}(a.data != NULL);",
                    f"    {extra}(b.data != NULL);",
                    "    if (memcmp(a.data, b.data, a.len * size)) goto lan_unequal;",
                    "lan_equal:",
                    "    return 1;",
                    "lan_unequal:",
                    "    return 0;",
                    "}",
                ])
                continue
            if kind == "in":
                unsigned = bool(domain) and domain.lo >= 0 and domain.hi >= (1 << 63)
                param_c = "lan_uint" if unsigned else "lan_int"
                bound_lo = 0 if unsigned else -(1 << 63)
                bound_hi = (1 << 64) - 1 if unsigned else (1 << 63) - 1
                clauses = []
                for clause in domain.clauses():
                    lo = max(clause.lo, bound_lo)
                    hi = min(clause.hi, bound_hi)
                    if lo > hi:
                        continue
                    if lo == hi:
                        terms = [f"x == ({param_c}){c_int(lo)}"]
                    else:
                        terms = []
                        if lo != bound_lo:
                            terms.append(f"x >= ({param_c}){c_int(lo)}")
                        if hi != bound_hi:
                            terms.append(f"x <= ({param_c}){c_int(hi)}")
                    if not unsigned and lo < 0 and any(self._poly_search_domain(p)[2] for p in clause.require):
                        terms.append("x >= 0")
                    for p in clause.require:
                        p_unsigned = self._poly_search_domain(p)[2]
                        arg = "(lan_uint)x" if p_unsigned and not unsigned else "x"
                        terms.append(f"{self.helpers[('poly', p, None)]}({arg})")
                    for p in clause.exclude:
                        p_unsigned = self._poly_search_domain(p)[2]
                        if p_unsigned and not unsigned and lo < 0:
                            terms.append(f"(x < 0 || !{self.helpers[('poly', p, None)]}((lan_uint)x))")
                        else:
                            arg = "(lan_uint)x" if p_unsigned and not unsigned else "x"
                            terms.append(f"!{self.helpers[('poly', p, None)]}({arg})")
                    clauses.append(terms)
                lines.extend([f"static inline lan_int {name}({param_c} x) {{", "    (void)x;",
                              *c_predicate_clauses(clauses), "}"])
                continue
            elif kind == "poly":
                lines.extend(self.poly_helper(name, domain))
                continue
            elif kind == "bit_read":
                lines.append(f"static inline uint8_t {name}(const uint8_t *bytes, uint64_t i) {{ "
                             "return (uint8_t)((bytes[i >> 3] >> (i & 7u)) & 1u); }")
                continue
            elif kind == "bit_write":
                lines.extend([
                    f"static inline void {name}(uint8_t *bytes, uint64_t i, uint8_t value) {{",
                    "    uint8_t *byte = &bytes[i >> 3];",
                    "    uint8_t mask = (uint8_t)(1u << (i & 7u));",
                    "    if (value) goto lan_bit_set;",
                    "    *byte &= (uint8_t)~mask;",
                    "    return;",
                    "lan_bit_set:",
                    "    *byte |= mask;",
                    "}",
                ])
                continue
            elif kind == "native_big":
                lines.append(f"static inline int {name}(void) {{ "
                             "const uint16_t one = 1; return ((const unsigned char *)&one)[0] == 0; }")
                continue
            elif kind in ("bit_load", "bit_store"):
                loading = kind == "bit_load"
                if loading:
                    signature = f"static inline uint64_t {name}(const uint8_t *bytes, uint64_t start, unsigned width, int reverse)"
                    setup = ["    uint64_t value = 0;"]
                    access = ["    value |= (uint64_t)((bytes[i >> 3] >> off) & mask) << shift;"]
                else:
                    signature = f"static inline void {name}(uint8_t *bytes, uint64_t start, unsigned width, uint64_t value, int reverse)"
                    setup = []
                    access = ["    uint8_t *byte = &bytes[i >> 3];",
                              "    *byte = (uint8_t)((*byte & ~(mask << off)) | (((value >> shift) & mask) << off));"]
                lines.extend([
                    signature + " {",
                    *setup,
                    "    unsigned done = 0;",
                    "lan_bit_chunk_head: ;",
                    "    if (done >= width) goto lan_bit_chunk_end;",
                    "    uint64_t i = start + done;",
                    "    unsigned off = (unsigned)(i & 7u), take = 8u - off;",
                    "    if (take <= width - done) goto lan_bit_chunk_size;",
                    "    take = width - done;",
                    "lan_bit_chunk_size: ;",
                    "    unsigned mask = (1u << take) - 1u, shift = done;",
                    "    if (!reverse) goto lan_bit_chunk_access;",
                    "    shift = width - done - take;",
                    "lan_bit_chunk_access: ;",
                    *access,
                    "    done += take;",
                    "    goto lan_bit_chunk_head;",
                    "lan_bit_chunk_end:",
                    "    return value;" if loading else "    return;",
                    "}",
                ])
                continue
            elif kind == "bit_copy":
                load, store = extra
                qualifier = "restrict " if domain else ""
                direction = [] if domain else [
                    "    uintptr_t d = (uintptr_t)dst + offset / 8;",
                    "    uintptr_t s = (uintptr_t)src + start / 8;",
                    "    if (d > s) goto lan_copy_back;",
                    "    if (d < s) goto lan_copy_forward;",
                    "    if (offset % 8 > start % 8) goto lan_copy_back;",
                ]
                backward = [] if domain else [
                    "lan_copy_back:",
                    "    if (!count) goto lan_copy_done;",
                    "    unsigned back = 64;",
                    "    if (count >= 64) goto lan_copy_back_chunk;",
                    "    back = (unsigned)count;",
                    "lan_copy_back_chunk:",
                    "    count -= back;",
                    f"    {store}(dst, offset + count, back, {load}(src, start + count, back, 0), 0);",
                    "    goto lan_copy_back;",
                ]
                lines.extend([
                    f"static inline void {name}(uint8_t *{qualifier}dst, uint64_t offset,",
                    f"                             const uint8_t *{qualifier}src, uint64_t start, uint64_t count) {{",
                    *direction,
                    "lan_copy_forward:",
                    "    if (!count) goto lan_copy_done;",
                    "    unsigned forward = 64;",
                    "    if (count >= 64) goto lan_copy_forward_chunk;",
                    "    forward = (unsigned)count;",
                    "lan_copy_forward_chunk:",
                    f"    {store}(dst, offset, forward, {load}(src, start, forward, 0), 0);",
                    "    offset += forward;",
                    "    start += forward;",
                    "    count -= forward;",
                    "    goto lan_copy_forward;",
                    *backward,
                    "lan_copy_done:",
                    "    return;",
                    "}",
                ])
                continue
            elif kind == "bit_sign":
                lines.append(f"static inline lan_int {name}(uint64_t value, unsigned width) {{ "
                             "if (value & (UINT64_C(1) << (width - 1))) goto lan_negative; "
                             "return (lan_int)value; "
                             "lan_negative: ; lan_uint mask = width < 64 ? (((lan_uint)1 << width) - 1) : ~(lan_uint)0; "
                             "return -1 - (lan_int)(mask ^ value); }")
                continue
            elif kind == "trap":
                guard, needs_nonneg, needs_upper = extra if isinstance(extra, tuple) else (extra, False, False)
                unsigned = bool(domain) and domain.lo >= 0 and domain.hi >= (1 << 63)
                ret_c = "lan_uint" if unsigned else "lan_int"
                param_c = "lan_int" if needs_nonneg else ("lan_uint" if unsigned or needs_upper else "lan_int")
                checks = []
                if needs_nonneg:
                    checks.append("if (x < 0) goto lan_invalid_integer;")
                if needs_upper:
                    checks.append(f"if (x > (lan_uint){c_int((1 << 63) - 1)}) goto lan_invalid_integer;")
                checks.append(f"if (!{guard}(({ret_c})x)) goto lan_invalid_integer;")
                body = f"{' '.join(checks)} return ({ret_c})x; lan_invalid_integer: __builtin_trap();"
                lines.append(f"static inline {ret_c} {name}({param_c} x) {{ {body} }}")
                continue
            elif kind == "sat":
                needs_nonneg, needs_upper = extra if isinstance(extra, tuple) else (False, False)
                unsigned = domain.lo >= 0 and domain.hi >= (1 << 63)
                ret_c = "lan_uint" if unsigned else "lan_int"
                param_c = "lan_int" if needs_nonneg else ("lan_uint" if unsigned or needs_upper else "lan_int")
                lo, hi = f"({ret_c}){c_int(domain.lo)}", f"({ret_c}){c_int(domain.hi)}"
                if needs_nonneg:
                    body = (f"if (x < 0 || (lan_uint)x < {lo}) goto lan_low; "
                            f"if ((lan_uint)x > {hi}) goto lan_high; return (lan_uint)x; "
                            f"lan_low: return {lo}; lan_high: return {hi};")
                elif needs_upper:
                    if domain.hi < 0:
                        body = f"(void)x; return {hi};"
                    else:
                        body = (f"{f'if (x < (lan_uint){lo}) goto lan_low; ' if domain.lo > 0 else ''}"
                                f"if (x > (lan_uint){hi}) goto lan_high; return (lan_int)x; "
                                f"{f'lan_low: return {lo}; ' if domain.lo > 0 else ''}lan_high: return {hi};")
                else:
                    body = (f"if (x < {lo}) goto lan_low; if (x > {hi}) goto lan_high; return x; "
                            f"lan_low: return {lo}; lan_high: return {hi};")
                lines.append(f"static inline {ret_c} {name}({param_c} x) {{ {body} }}")
                continue
            else:
                if extra <= 32:
                    mask32 = (1 << extra) - 1
                    if domain.lo < 0:
                        if extra < 32:
                            body = (f"uint32_t r = x & UINT32_C({mask32}); "
                                    f"if (r <= UINT32_C({domain.hi})) return (int32_t)r; "
                                    f"return (int32_t)(r - UINT32_C({1 << extra}));")
                        else:
                            body = (f"if (x <= UINT32_C({domain.hi})) return (int32_t)x; "
                                    "return -1 - (int32_t)(~x);")
                        lines.append(f"static inline int32_t {name}(uint32_t x) {{ {body} }}")
                    else:
                        lines.append(f"static inline uint32_t {name}(uint32_t x) {{ return x & UINT32_C({mask32}); }}")
                    continue
                if extra >= 64:
                    body = "return x;"
                else:
                    mask = c_uint((1 << extra) - 1)
                    body = f"lan_uint r = (lan_uint)x & {mask};"
                    if domain.lo < 0:
                        body += (f" if (r <= (lan_uint){c_int(domain.hi)}) goto lan_wrap_done;"
                                 f" return (lan_int)r - {c_int(1 << extra)}; lan_wrap_done:")
                    body += " return (lan_int)r;"
            lines.append(f"static inline lan_int {name}(lan_int x) {{ {body} }}")
        return lines

    @staticmethod
    def _poly_search_domain(poly):
        lo, _ = poly.bounds(WIDE.lo, WIDE.hi)
        if lo >= 0:
            return 0, (1 << 64) - 1, True
        return -(1 << 63), (1 << 63) - 1, False

    @staticmethod
    def poly_uses_limbs(poly):
        if len(poly.coefficients) <= 2:
            return False
        if Compiler.poly_uses_fast_search(poly):
            return False
        bound_lo, bound_hi, _ = Compiler._poly_search_domain(poly)
        return any(start < end for start, end, _ in poly.windows(bound_lo, bound_hi))

    @staticmethod
    def poly_uses_fast_search(poly):
        bound_lo, bound_hi, _ = Compiler._poly_search_domain(poly)
        return (poly.denominator == 1 and bound_lo <= poly.base <= bound_hi
                and all(0 <= c <= bound_hi for c in poly.coefficients[1:]))

    def poly_limb_support(self):
        return [
            f"static inline void {self.helper_prefix()}_limb_madd(uint32_t *v, size_t count, const uint32_t *factor,",
            "                                lan_uint n, const uint32_t *add) {",
            "    uint64_t n_lo = (uint32_t)n, n_hi = (uint64_t)n >> 32;",
            "    uint64_t carry = 0;",
            *indent(c_counted_loop("i", 0, "count", [
                "uint64_t p0 = (uint64_t)factor[i] * n_lo + (uint32_t)carry + add[i];",
                "uint64_t p1 = (uint64_t)factor[i] * n_hi + (carry >> 32) + (p0 >> 32);",
                "v[i] = (uint32_t)p0;",
                "carry = p1;",
            ])),
            "}",
        ]

    def poly_signed_compare(self, name, coefficients, denominator, offset_max, *, unsigned=False):
        degree = len(coefficients)
        positive = tuple(max(0, c) for c in coefficients)
        negative = tuple(max(0, -c) for c in coefficients)
        # Bound both Horner accumulators independently, including cancellation
        # and the 64-bit target. This makes the generated 32-bit limb width exact.
        bound = max(sum(c * offset_max ** i for i, c in enumerate(side))
                    for side in (positive, negative)) + (denominator << 64)
        limbs = (bound.bit_length() + 31) // 32
        madd = self.helper_prefix() + "_limb_madd"
        if limbs * degree > 65_536:
            raise DomainComplexity("poly runtime evaluation exceeds the fixed-limb code generation budget")

        def words(value):
            result = []
            index = 0
            while value:
                word = value & ((1 << 32) - 1)
                if word:
                    result.append(f"[{index}] = UINT32_C({word})")
                value >>= 32
                index += 1
            return ", ".join(result)

        def rows(values):
            return [f"        [{i}] = {{ {words(v)} }}," for i, v in enumerate(values) if v] or ["        {0},"]

        param_c = "lan_uint" if unsigned else "lan_int"
        target_lines = ([
            f"    {madd}(neg, {limbs}, denominator, (lan_uint)x, neg);",
        ] if unsigned else [
            "    if (x < 0) goto lan_poly_negative;",
            f"    {madd}(neg, {limbs}, denominator, (lan_uint)x, neg);",
            "    goto lan_poly_compare_init;",
            "lan_poly_negative:",
            f"    {madd}(pos, {limbs}, denominator, (lan_uint)0 - (lan_uint)x, pos);",
            "lan_poly_compare_init:",
        ])
        return [
            f"static inline int {name}(lan_uint n, {param_c} x) {{",
            f"    static const uint32_t positive[{degree}][{limbs}] = {{",
            *rows(positive), "    };",
            f"    static const uint32_t negative[{degree}][{limbs}] = {{",
            *rows(negative), "    };",
            f"    static const uint32_t denominator[{limbs}] = {{ {words(denominator)} }};",
            f"    uint32_t pos[{limbs}] = {{0}}, neg[{limbs}] = {{0}};",
            f"    size_t i = {degree};",
            "lan_poly_eval:",
            "    if (!i) goto lan_poly_target;",
            "    --i;",
            f"    {madd}(pos, {limbs}, pos, n, positive[i]);",
            f"    {madd}(neg, {limbs}, neg, n, negative[i]);",
            "    goto lan_poly_eval;",
            "lan_poly_target:",
            *target_lines,
            f"    i = {limbs};",
            "lan_poly_compare:",
            "    if (!i) goto lan_poly_same;",
            "    --i;",
            "    if (pos[i] < neg[i]) goto lan_poly_less;",
            "    if (pos[i] > neg[i]) goto lan_poly_greater;",
            "    goto lan_poly_compare;",
            "lan_poly_same:",
            "    return 0;",
            "lan_poly_less:",
            "    return -1;",
            "lan_poly_greater:",
            "    return 1;",
            "}",
        ]

    @staticmethod
    def poly_search_code(compare, lower, upper, direction, guards=()):
        condition = " && ".join(f"!({guard})" for guard in guards)
        body = [
            f"lan_uint lo = {lower}, hi = {upper};",
            "while (lo <= hi) {",
            "    lan_uint mid = lo + (hi - lo) / 2;",
            f"    int cmp = {compare};",
            "    if (!cmp) goto lan_poly_found;",
            f"    if (cmp {'>' if direction > 0 else '<'} 0) {{",
            "        if (!mid) break;",
            "        hi = mid - 1;",
            "    } else {",
            "        if (mid == hi) break;",
            "        lo = mid + 1;",
            "    }",
            "}",
        ]
        if condition:
            return [f"if ({condition}) {{", *indent(body), "}"]
        return ["{", *indent(body), "}"]

    def poly_general_helper(self, name, poly):
        bound_lo, bound_hi, unsigned = self._poly_search_domain(poly)
        param_c = "lan_uint" if unsigned else "lan_int"
        comparisons = []
        body = [f"static inline int {name}({param_c} x) {{"]
        for index, (start, end, direction) in enumerate(poly.windows(bound_lo, bound_hi)):
            a, b = poly.eval(start), poly.eval(end)
            lo, hi = min(a, b), max(a, b)
            if start == end:
                body.append(f"    if (x == ({param_c}){c_int(a)}) goto lan_poly_found;")
                continue
            span = end - start
            if span >= 1 << 64:
                raise DomainComplexity("poly runtime search offset exceeds 64 bits")
            compare = f"{name}_cmp_{index}"
            comparisons.extend(self.poly_signed_compare(compare, poly.rebase(start), poly.denominator, span,
                                                        unsigned=unsigned))
            guards = (f"x < ({param_c}){c_int(lo)}", f"x > ({param_c}){c_int(hi)}")
            body.extend(indent(self.poly_search_code(f"{compare}(mid, x)", 0, c_uint(span), direction, guards)))
        body.extend(["    (void)x;", "    return 0;"])
        if len(body) > 3:
            body.extend(["lan_poly_found:", "    return 1;"])
        body.append("}")
        return comparisons + body

    def poly_helper(self, name, poly):
        bound_lo, bound_hi, unsigned = self._poly_search_domain(poly)
        param_c = "lan_uint" if unsigned else "lan_int"
        if len(poly.coefficients) == 2:
            lo, hi = poly.bounds(bound_lo, bound_hi)
            if lo > hi:
                condition = "(void)x; return 0;"
            elif lo == hi:
                condition = f"return x == ({param_c}){c_int(lo)};"
            else:
                step = c_uint(poly.modulus)
                condition = (f"if (x < ({param_c}){c_int(lo)}) goto lan_poly_outside; "
                             f"if (x > ({param_c}){c_int(hi)}) goto lan_poly_outside; "
                             f"return ((lan_uint)x - (lan_uint){c_int(lo)}) % {step} == 0; "
                             "lan_poly_outside: return 0;")
            return [f"static inline int {name}({param_c} x) {{ {condition} }}"]
        if not self.poly_uses_fast_search(poly):
            return self.poly_general_helper(name, poly)
        base = f"({param_c}){c_int(poly.base)}"
        prefix = [f"static inline int {name}({param_c} x) {{",
                  f"    if (x < {base}) goto lan_poly_outside;",
                  f"    lan_uint target = (lan_uint)x - (lan_uint){base};"]
        # Compare f(n)-f(0) to target using Horner evaluation. The guard keeps
        # every multiply and addition in range, even at UINT64_MAX.
        compare = [f"static inline int {name}_cmp(lan_uint n, lan_uint target) {{",
                   "    lan_uint value = 0;"]
        for coefficient in reversed((0, *poly.coefficients[1:])):
            if coefficient:
                constant = f"(lan_uint){c_int(coefficient)}"
                compare.append(f"    if (target < {constant}) goto lan_poly_above;")
                compare.append(f"    if (value > (target - {constant}) / n) goto lan_poly_above;")
                compare.append(f"    value = value * n + {constant};")
            else:
                compare.extend(["    if (value > target / n) goto lan_poly_above;", "    value *= n;"])
        compare.extend(["    if (value != target) goto lan_poly_below;", "    return 0;",
                        "lan_poly_below: return -1;", "lan_poly_above: return 1;", "}"])
        return [
            *compare, *prefix,
            "    if (!target) goto lan_poly_found;",
            *indent(self.poly_search_code(f"{name}_cmp(mid, target)", 1, "target", 1)),
            "lan_poly_outside:",
            "    return 0;",
            "lan_poly_found:",
            "    return 1;",
            "}",
        ]

    def compile(self, include_header=True, *, include_trace=True):
        self.register_names()
        self.type_object_nodes = self.type_nodes.copy()
        # Rules are checked against the ordinary mathematics, never themselves.
        context = ACTIVE_ORACLES.set(None)
        steps_context = (ACTIVE_SOLVER_STEPS.set(self.max_steps)
                         if self.max_steps is not None else None)
        try:
            self.oracles = self.register_oracles()
            ACTIVE_ORACLES.set(self.oracles)
            self.prepare_type_objects()
            return self.compile_with_oracles(include_header, include_trace=include_trace)
        finally:
            ACTIVE_ORACLES.reset(context)
            if steps_context is not None:
                ACTIVE_SOLVER_STEPS.reset(steps_context)

    RESERVED_EXPORT_PREFIXES = ("lan_fn_", "lan_api_", "lan_m", "lan_record_", "lan_array_", "lan_result_", "lh_")

    def stable_export_name(self, name):
        """C symbol for `--export-prefix`: prefix + exported name, with no contract fingerprint."""
        alias = f"{self.export_prefix}{c_ident(name)}"
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", alias) or alias == "main"
                or alias.startswith(self.RESERVED_EXPORT_PREFIXES)):
            raise LanError(f"stable export name {alias!r} collides with a reserved or invalid C identifier; "
                           "choose another --export-prefix")
        return alias

    def compile_with_oracles(self, include_header, *, include_trace=True):
        self.register()
        order = self.function_order()
        compiled = [self.function(name, *self.functions[name]) for name in order]
        self.check_callback_cycles()
        self.derive_storage_theorems()
        self.derive_return_theorems()
        self.semantic_module.loops = self.loop_recurrences.copy()
        self.source_theorems.extend(fact for fact in self.loop_exit_facts.values() if fact is not None)
        self.semantic_module.contracts = {name: contract for name, (_, contract) in self.functions.items()}
        for bindings in (self.externs, self.imported_functions):
            self.semantic_module.contracts.update((name, binding.contract) for name, binding in bindings.items())
        self.semantic_module.types = self.types.copy()
        for symbol in self.symbols:
            if symbol.semantic_binding is None:
                continue
            mode = ("static" if symbol.static_storage else "heap" if symbol.heap_owner or isinstance(symbol.ty, HeapArray)
                    else "borrow" if symbol.borrowed or isinstance(symbol.ty, (Interface, Slice)) else "local")
            storage = Storage(symbol.ty, mode, symbol.readonly, symbol.ownership_origin)
            entries = self.semantic_module.storage.setdefault(symbol.semantic_binding, [])
            if storage not in entries:
                entries.append(storage)
        self.publish_exports()
        wrappers = []
        if self.interface_abi is not None:
            for name, binding in self.exports.items():
                if not isinstance(binding, FunctionBinding) or binding.foreign:
                    continue
                contract = binding.contract
                signature = chain_api_signature(encode_interface_type(contract), binding.owned_chain_lengths,
                                                binding.owned_chain_parameters)
                symbol = api_symbol(self.interface_abi, name, signature)
                args = ", ".join(f"arg_{i}" for i in range(len(contract.params)))
                statement = f"{binding.c_name}({args});"
                if contract.result is not None:
                    statement = "return " + statement
                wrappers.extend(["", f"{self.c_signature(symbol, contract, named=True)} {{",
                                 *indent([statement]), "}"])
                if self.export_prefix is not None:
                    alias = self.stable_export_name(name)
                    wrappers.extend(["", f"{self.c_signature(alias, contract, named=True)} {{",
                                     *indent([statement]), "}"])
        lines = [C_HEADER.rstrip("\n")] if include_header else []
        lines.extend(["", *self.record_code(), *self.literal_code(), *self.module_const_code, *self.helper_code(), ""])
        external = {binding.c_name: binding
                    for binding in (*self.imported_functions.values(), *self.externs.values())
                    if binding.external}
        lines.extend(f"extern {self.c_signature(name, binding.contract, foreign=binding.foreign)};"
                     for name, binding in external.items())
        lines.extend(head + ";" for head, _ in compiled)
        for _, body in compiled:
            lines.extend(["", *body])
        lines.extend(wrappers)
        if not self.library and self.module_id is None and "main" in self.functions:
            node, contract = self.functions["main"]
            if contract.params or not isinstance(contract.result, Scalar) or contract.result.floating or not contract.result.domain.within(container(32, True)):
                node.args[0].error("main requires no arguments and an integer result fitting int32")
            if contract.result.boolean or is_callback_enum(contract.result.enum):
                node.args[0].error("main requires an integer result")
            lines.extend(["", "int main(void) { return (int)lan_fn_main(); }"])
        replacements = {str(tag): code for tag, code, _ in self.rhs_codes.values()}
        placeholder = re.compile(r"@(\d+)@")
        def expand(match):
            return placeholder.sub(expand, replacements[match[1]])

        trace = ascii_table_trace(self.trace) if include_trace else ""
        return placeholder.sub(expand, "\n".join(lines) + "\n"), trace

    @staticmethod
    def c_param_type(ty, foreign=False):
        if foreign and isinstance(ty, Slice):
            if ty.text == "c":
                return "const char *"
            return f"{'const ' if ty.readonly else ''}{ty.element.c} *"
        return ty.c

    @staticmethod
    def c_signature(name, contract, named=False, foreign=False):
        ty = contract.result
        result = "const char *" if foreign and isinstance(ty, Slice) and ty.text == "c" else ty.c if ty else "void"
        params = [Compiler.c_param_type(ty, foreign) + (f" arg_{i}" if named else "")
                  for i, ty in enumerate(contract.params)]
        return f"{result} {name}({', '.join(params) or 'void'})"

    @staticmethod
    def c_fnptr_typedef(name, contract, foreign=False):
        ty = contract.result
        result = "const char *" if foreign and isinstance(ty, Slice) and ty.text == "c" else ty.c if ty else "void"
        params = [Compiler.c_param_type(param, foreign) for param in contract.params]
        return f"typedef {result} (*{name})({', '.join(params) or 'void'});"
