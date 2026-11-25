"""Small C code emission helpers shared by the backend."""

from __future__ import annotations


def c_predicate_sequence(terms, *, conjunction):
    if not terms:
        return str(int(conjunction))
    if len(terms) == 1:
        return f"({terms[0]})"
    if conjunction:
        expr = f"!!({terms[-1]})"
        for term in reversed(terms[:-1]):
            expr = f"(({term}) ? {expr} : 0)"
        return expr
    expr = f"!!({terms[-1]})"
    for term in reversed(terms[:-1]):
        expr = f"(({term}) ? 1 : {expr})"
    return expr


def c_predicate_choice(condition, when_true, when_false):
    return f"(({condition}) ? !!({when_true}) : !!({when_false}))"


def c_predicate_clauses(clauses):
    lines = []
    for index, terms in enumerate(clauses):
        if not terms:
            return [*lines, "    return 1;"]
        failure = f"lan_predicate_next_{index}"
        lines.extend(f"    if (!({term})) goto {failure};" for term in terms)
        lines.extend(["    return 1;", f"{failure}: ;"])
    return [*lines, "    return 0;"]


def c_counted_loop(index, start, stop, body):
    return [
        "{",
        f"    size_t {index} = {start};",
        f"    while ({index} < {stop}) {{",
        *indent(indent(body)),
        f"        ++{index};",
        "    }",
        "}",
    ]


def indent(lines):
    return ["    " + line for line in lines]
