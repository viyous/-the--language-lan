"""Custom string encoding types, compile-time JSON mapping tables and list values."""

from __future__ import annotations

from dataclasses import dataclass

from table_interface import digest


@dataclass(frozen=True)
class TextEncoding:
    """A custom compile-time string encoding defined from a JSON mapping object."""

    identity: str
    name: str
    chars: tuple[tuple[str, bytes], ...] = ()
    tokens: tuple[tuple[str, bytes], ...] = ()
    terminator: bytes = b""
    pad: int = 0

    def __eq__(self, other):
        return isinstance(other, TextEncoding) and self.identity == other.identity

    def __hash__(self):
        return hash(("TextEncoding", self.identity))

    def __str__(self):
        return f"encoding({self.name})"

    def encode(self, text: str, token) -> bytes:
        """Encode Unicode text using greedy longest-match over tokens and single characters."""
        chars = dict(self.chars)
        ordered_tokens = sorted(self.tokens, key=lambda item: -len(item[0]))
        out = bytearray()
        i, n = 0, len(text)
        while i < n:
            matched = False
            for tok_str, seq in ordered_tokens:
                if text.startswith(tok_str, i):
                    out.extend(seq)
                    i += len(tok_str)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            seq = chars.get(ch)
            if seq is None:
                token.error(
                    f"character {ch!r} (U+{ord(ch):04X}) is not mapped in encoding {self.name!r}"
                )
            out.extend(seq)
            i += 1
        return bytes(out)

    def encode_fixed(self, text: str, capacity: int, token) -> bytes:
        """Encode into a fixed-length byte buffer, appending terminator and padding with `self.pad`."""
        raw = self.encode(text, token) + self.terminator
        if len(raw) > capacity:
            token.error(
                f"encoded string length {len(raw)} exceeds fixed capacity {capacity} for encoding {self.name!r}"
            )
        if len(raw) < capacity:
            raw += bytes([self.pad]) * (capacity - len(raw))
        return raw


@dataclass(frozen=True)
class EncodingTag:
    """Encoding tag type (`type string Name = Map as tag;`): use `Name[..]` for slices or `Name[N]` for buffers."""

    encoding: TextEncoding

    @property
    def c(self):
        return "uint8_t"


@dataclass(frozen=True)
class JsonMap:
    """Compile-time JSON-like key->bytes mapping object, composable via `·` and `expand`."""

    entries: tuple[tuple[str, bytes | None], ...] = ()

    def merge(self, other: JsonMap) -> JsonMap:
        merged = dict(self.entries)
        for key, value in other.entries:
            merged[key] = value
        return JsonMap(tuple(merged.items()))

    def with_entry(self, key: str, value: bytes | None) -> JsonMap:
        merged = dict(self.entries)
        merged[key] = value
        return JsonMap(tuple(merged.items()))

    def get(self, key: str) -> bytes | None:
        return dict(self.entries).get(key)

    def to_encoding(self, name: str, token) -> TextEncoding:
        chars: dict[str, bytes] = {}
        tokens: dict[str, bytes] = {}
        terminator = b""
        pad = None
        for key, value in self.entries:
            if key == "@terminator":
                terminator = b"" if value is None else value
            elif key == "@pad":
                if value is None or len(value) != 1:
                    token.error("'@pad' in encoding map must be a single byte in [0, 255]")
                pad = value[0]
            else:
                if value is None:
                    chars.pop(key, None)
                    tokens.pop(key, None)
                    continue
                if len(value) == 0:
                    token.error(f"encoding map entry {key!r} must map to a non-empty byte sequence")
                if len(key) == 1:
                    chars[key] = value
                else:
                    tokens[key] = value
        return build_encoding(chars, tokens, terminator=terminator, pad=pad, name=name)


@dataclass(frozen=True)
class ConstList:
    """Compile-time homogeneous/untyped list value, composable via `·` and `expand`."""

    items: tuple = ()

    def concat(self, other: ConstList) -> ConstList:
        return ConstList(self.items + other.items)

    def with_index(self, index: int, value) -> ConstList:
        items = list(self.items)
        items[index] = value
        return ConstList(tuple(items))


def build_encoding(
    chars: dict[str, bytes],
    tokens: dict[str, bytes],
    *,
    terminator: bytes = b"",
    pad: int | None = None,
    name: str = "custom",
) -> TextEncoding:
    pad_byte = (terminator[0] if len(terminator) == 1 else 0) if pad is None else pad
    sorted_chars = tuple(sorted(chars.items()))
    sorted_tokens = tuple(sorted(tokens.items()))
    ident_parts = [
        f"encoding:{terminator.hex()}:{pad_byte:02x}",
        *(f"c:{k}={v.hex()}" for k, v in sorted_chars),
        *(f"t:{k}={v.hex()}" for k, v in sorted_tokens),
    ]
    return TextEncoding(
        identity=digest("\n".join(ident_parts)),
        name=name,
        chars=sorted_chars,
        tokens=sorted_tokens,
        terminator=terminator,
        pad=pad_byte,
    )
