"""Syntax tree nodes and type-object vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(eq=False)
class Node:
    token: Token
    kind: str
    args: tuple
    source_text: str | None = None
    state_guard: Node | None = None


@dataclass(eq=False)
class LoweringNode(Node):
    semantic_anchor: int | None = None

    @classmethod
    def from_operation(cls, operation, args, state_guard):
        return cls(operation.token, operation.kind, args, operation.source_text, state_guard, operation.anchor)


@dataclass(frozen=True)
class TypeObject:
    entries: tuple


@dataclass(frozen=True)
class TypeArgument:
    value: object


TYPE_QUALIFIERS = ("len", "elem", "immutable", "pass", "storage", "from", "offset", "field", "repr")
REMOVED_TYPE_QUALIFIERS = ("chain", "graph", "prune", "attach")
TYPE_OBJECT_WORDS = {
    "kind": ("int", "uint", "array"), "storage": ("heap", "stack", "rodata"),
    "pass": ("alias", "copy"), "repr": ("c",),
}
TYPE_OBJECT_SCALARS = ("bits", "len", "offset", "from", "immutable")
TYPE_OBJECT_MARKERS = ("object_value", "object_word", "object_path")


def type_object_kind(token, kind):
    return replace(token, text="kind"), Node(replace(token, text=kind), "object_word", (kind,))
