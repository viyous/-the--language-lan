"""Tokens, lexical tables and the tokenizer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from value_domain import Domain

from .errors import LanError


@dataclass(frozen=True)
class Token:
    text: str
    line: int
    col: int

    def error(self, message):
        raise LanError(f"{self.line}:{self.col}: {message}")


LEX = re.compile(
    r'\s+|//[^\n]*|/\*[\s\S]*?\*/|"(?:[^"\\\r\n]|\\[^\r\n])*"|'
    r"'(?:[^'\\\r\n]|\\[^\r\n])*'|"
    r"0[xX][0-9a-fA-F]+|0[bB][01]+|(?:[0-9]+\.[0-9]+|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+|[0-9]+|"
    r"[^\W\d]\w*|<<=|>>=|===|!==|->|=>|==|!=|<=|>=|&&|\|\||\^\^|"
    r"<<|>>|[+*/%&|^\-]=|[{}\[\](),:;=+*/%&|^<>!~.?@·\\-]"
)
NAME = re.compile(r"[^\W\d]\w*\Z")
RESERVED = set(
    "type define contract if else while interation break continue return true false void boolean "
    "in as not and or xor assert enum extern import export null interface poly bit callable callback".split()
)
C_KEYWORDS = set(("auto break case char const continue default do double else enum extern float for goto "
                  "if inline int long register restrict return short signed sizeof static struct switch "
                  "typedef union unsigned void volatile while _Alignas _Alignof _Atomic _Bool _Complex "
                  "_Generic _Imaginary _Noreturn _Static_assert _Thread_local").split())
LOGIC = {"and": "&&", "or": "||", "not": "!", "xor": "^^"}
PREC = {
    "||": 1, "^^": 2, "&&": 3, "|": 4, "^": 5, "&": 6,
    "==": 7, "!=": 7, "===": 7, "!==": 7, "in": 7,
    "<": 8, ">": 8, "<=": 8, ">=": 8, "<<": 9, ">>": 9,
    "+": 10, "-": 10, "·": 10, "*": 11, "/": 11, "%": 11, "as": 12,
}


CHARACTERS = {
    "ascii": (8, Domain.of((0, 127))),
    "utf8": (8, Domain.of((0, 255))),
    "utf16": (16, Domain.of((0, 65535))),
    "unicode": (32, Domain.of((0, 0xd7ff), (0xe000, 0x10ffff))),
}


def decode_literal(token):
    escapes = {"\\": "\\", '"': '"', "'": "'", "/": "/", "n": "\n",
               "r": "\r", "t": "\t", "b": "\b", "f": "\f", "0": "\0"}
    source = token.text[1:-1]
    chars, i = [], 0
    while i < len(source):
        value = source[i]
        i += 1
        if value == "\\":
            escape = source[i]
            i += 1
            if escape in escapes:
                value = escapes[escape]
            elif escape in ("x", "u", "U"):
                width = {"x": 2, "u": 4, "U": 8}[escape]
                digits = source[i:i + width]
                if len(digits) != width or not re.fullmatch("[0-9a-fA-F]+", digits):
                    token.error("invalid hexadecimal literal escape")
                code = int(digits, 16)
                if code > 0x10ffff or 0xd800 <= code <= 0xdfff:
                    token.error("literal escape must denote a Unicode scalar value")
                value = chr(code)
                i += width
            else:
                token.error(f"unsupported literal escape \\{escape}")
        if 0xd800 <= ord(value) <= 0xdfff:
            token.error("literal must contain Unicode scalar values")
        chars.append(value)
    return "".join(chars)


def lex(source):
    line, col, pos = 1, 1, 0
    while pos < len(source):
        m = LEX.match(source, pos)
        if not m:
            Token("", line, col).error(f"unexpected character {source[pos]!r}")
        s = m[0]
        if source.startswith("/*", pos) and not s.startswith("/*"):
            Token("", line, col).error("unterminated block comment")
        is_oracle = s.startswith("/*") and re.match(r"/\*+\s*(?:\*\s*)?@oracle\b", s)
        if is_oracle or (not s[0].isspace() and not s.startswith(("//", "/*"))):
            yield Token(s, line, col)
        if "\n" in s:
            line += s.count("\n")
            col = len(s.rsplit("\n", 1)[1]) + 1
        else:
            col += len(s)
        pos = m.end()
    yield Token("<eof>", line, col)
