"""Recursive-descent parser producing syntax nodes."""

from __future__ import annotations

import json
import re
from dataclasses import field, fields, replace
from itertools import count

from .errors import LanError
from .lexer import decode_literal, lex, LOGIC, NAME, PREC, RESERVED
from .syntax import Node, TYPE_OBJECT_MARKERS, TYPE_OBJECT_SCALARS, TYPE_OBJECT_WORDS, TYPE_QUALIFIERS


class Parser:
    def __init__(self, source: str | tuple[Token, ...]):
        self.tokens = list(lex(source)) if isinstance(source, str) else source
        self.conditions = {}
        self.pos = 0
        self.loop_depth = 0
        self.in_type_function = False
        self.in_implementation_contract = False

    @property
    def cur(self):
        return self.tokens[self.pos]

    def take(self):
        t = self.cur
        if t.text != "<eof>":
            self.pos += 1
        return t

    def accept(self, s):
        if self.cur.text == s:
            return self.take()
        return None

    def need(self, s):
        return self.accept(s) or self.cur.error(f"expected {s!r}, got {self.cur.text!r}")

    def name(self):
        t = self.take()
        if not NAME.fullmatch(t.text) or t.text in RESERVED:
            t.error("expected an identifier")
        return t

    def node(self, t, kind, *args):
        return Node(t, kind, args)

    def comma_list(self, end, parse):
        result = []
        while self.cur.text != end:
            result.append(parse())
            if not self.accept(","):
                if self.cur.text != end:
                    self.cur.error(f"expected ',' or {end!r}, got {self.cur.text!r}")
                break
        self.need(end)
        return tuple(result)

    def integer(self):
        sign = -1 if self.accept("-") else 1
        t = self.take()
        return sign * self.number(t)

    @staticmethod
    def number(t):
        if not t.text[0].isdigit():
            t.error("expected an integer literal")
        try:
            return int(t.text, 0 if t.text.lower().startswith(("0x", "0b")) else 10)
        except ValueError:
            t.error("invalid or excessively long integer literal")

    def interval(self, t, lo):
        hi = self.expr()
        end = self.take()
        if end.text not in ("]", ")"):
            end.error("expected an interval closing bracket")
        return self.node(t, "interval", lo, hi, t.text == "(", end.text == ")")

    def implementation_params(self):
        """Recognize the bare parameter list that precedes a function body."""
        pos = self.pos + 1
        while NAME.fullmatch(self.tokens[pos].text):
            pos += 1
            if self.tokens[pos].text != ",":
                break
            pos += 1
        return self.tokens[pos].text == ")" and self.tokens[pos + 1].text == "{"

    def parameter(self):
        name = self.name()
        self.need(":")
        return name, self.type_expr()

    def type_function_parameter(self):
        name, annotation = self.parameter()
        while annotation.kind == "type_group":
            annotation = annotation.args[0]
        if annotation.kind == "type_name" and annotation.args == ("type",):
            annotation = replace(annotation, kind="type_parameter", args=())
        return name, annotation

    def type_declaration_rhs(self):
        start = self.pos
        try:
            annotation = self.type_expr()
        except LanError as type_err:
            self.pos = start
            try:
                expr = self.expr()
            except LanError:
                raise type_err
            if self.cur.text == ";":
                return self.node(expr.token, "set", (expr,))
            raise type_err
        if self.cur.text == ";":
            return annotation
        self.pos = start
        try:
            expr = self.expr()
        except LanError:
            self.pos = start
            return self.type_expr()
        if self.cur.text == ";":
            return self.node(expr.token, "set", (expr,))
        self.pos = start
        return self.type_expr()

    def type_application_argument(self):
        # The callee decides whether an ambiguous operand is a value or a type.
        start = self.pos
        try:
            expression = self.expr()
        except LanError:
            pass
        else:
            if self.cur.text in (",", ")", "]"):
                return self.node(expression.token, "type_application_argument", expression,
                                 tuple(self.tokens[start:self.pos]))
        self.pos = start
        return self.type_expr()

    def implementation_parameter(self):
        name = self.name()
        return name, self.type_expr() if self.accept(":") else None

    def parenthesized_type(self, token):
        # A comma after the first constant expression marks an open interval.
        # Otherwise the parentheses group type operations.
        start = self.pos
        try:
            lower = self.expr()
        except LanError:
            lower = None
        if lower is not None and self.accept(","):
            return self.interval(token, lower)
        self.pos = start
        result = self.type_expr()
        self.need(")")
        return self.node(token, "type_group", result)

    def type_expr(self, min_prec=0, sets=True):
        t = self.take()
        if t.text == "[":
            lo = self.expr()
            self.need(",")
            left = self.interval(t, lo)
        elif t.text == "(":
            left = self.parenthesized_type(t)
        elif t.text in ("<", "<=", ">", ">="):
            # Arithmetic/shift operators may appear in the endpoint. Keep
            # set operators and the surrounding boolean guard outside it.
            left = self.node(t, "bound", self.expr(PREC["<<"]))
        elif t.text == "{":
            if self.is_json_map_start():
                left = self.node(t, "json_map", self.comma_list("}", self.json_map_entry))
            elif NAME.fullmatch(self.cur.text) and self.tokens[self.pos + 1].text == ":":
                entries = self.comma_list("}", self.type_object_entry)
                if all(value.kind not in TYPE_OBJECT_MARKERS
                       and not (name.text in ("len", "elem") and value.kind in ("set", "interval", "bound"))
                       for name, value in entries):
                    left = self.node(t, "record", entries, False)
                else:
                    left = self.node(t, "type_object", entries)
            else:
                left = self.node(t, "set", self.comma_list("}", self.expr))
        elif t.text == "interface":
            self.need("{")
            left = self.node(t, "interface", self.comma_list("}", self.parameter), False)
        elif t.text == "contract":
            self.need("(")
            params = self.comma_list(")", self.parameter)
            self.need("->")
            left = self.node(t, "contract", params, self.type_expr())
        elif t.text == "callable" and self.accept("("):
            left = self.node(t, "callable_type", self.comma_list(")", self.name))
        elif t.text == "extern" and self.accept("callable"):
            self.need("(")
            params = self.comma_list(")", self.parameter)
            self.need("->")
            result_ty = self.type_expr(sets=False)
            writes = self.write_set() if self.cur.text == "writes" else self.node(t, "writes", ())
            left = self.node(t, "extern_callable", params, result_ty, writes)
        elif t.text == "poly":
            self.need("(")
            left = self.node(t, "poly", self.expr())
            self.need(")")
        elif NAME.fullmatch(t.text):
            left = self.node(t, "type_name", t.text)
        else:
            t.error("expected a type or value set")
        if left.kind == "type_name":
            if self.cur.text == "." and self.tokens[self.pos + 1].text == "key":
                self.take()
                left = self.node(self.take(), "record_key_type", left)
            while self.cur.text in ("[", "("):
                if self.cur.text == "(" and self.in_implementation_contract and self.implementation_params():
                    break
                op = self.take()
                if op.text == "(":
                    args = []
                    while self.cur.text not in (")", "]"):
                        args.append(self.type_application_argument())
                        if not self.accept(","):
                            break
                    end = self.take()
                    if end.text not in (")", "]"):
                        end.error("expected a type argument or interval closing bracket")
                    left = self.node(op, "type_apply", left, tuple(args), end.text)
                    continue
                if self.accept("."):
                    self.need(".")
                    self.need("]")
                    left = self.node(op, "slice_type", left, None, False)
                    break
                lo = self.expr()
                if self.accept(","):
                    left = self.node(op, "refine", left, self.interval(op, lo))
                else:
                    self.need("]")
                    left = self.node(op, "array", left, lo)
        while min_prec <= 3 and (operator := self.accept("·")):
            left = self.node(operator, "type_merge", left, self.type_expr(4, sets))
        priorities = {"|": 1, "&": 2, "\\": 2}
        while sets and priorities.get(self.cur.text, -1) >= min_prec:
            op = self.take()
            right = self.type_expr(priorities[op.text] + 1)
            left = self.node(op, "type_op", left, right)
        if sets and min_prec == 0 and (seal := self.accept("in")):
            self.need("{")
            bases = self.comma_list("}", self.name)
            left = self.node(seal, "seal", left, bases, self.integer())
            while operator := self.accept("·"):
                left = self.node(operator, "type_merge", left, self.type_expr(4))
        if (self.cur.text == "{" and self.tokens[self.pos + 2].text == ":"
                and self.tokens[self.pos + 1].text in (*TYPE_QUALIFIERS, "bits", "kind")):
            self.cur.error("use '·' to merge type objects")
        return left

    def type_object_entry(self):
        key = self.name()
        self.need(":")
        if self.cur.text == "<eof>":
            self.cur.error("expected a type object property value")
        if key.text in TYPE_OBJECT_WORDS and self.tokens[self.pos + 1].text in (",", "}"):
            if self.cur.text in TYPE_OBJECT_WORDS[key.text]:
                value = self.take()
                return key, self.node(value, "object_word", value.text)
        indexed_field = (self.cur.text == "[" and self.tokens[self.pos + 1].text != "<eof>"
                         and self.tokens[self.pos + 2].text == "]")
        if key.text == "field" and (indexed_field or self.tokens[self.pos + 1].text in (".", "[")):
            path = [] if self.cur.text == "[" else [self.name().text]
            while True:
                if self.accept("["):
                    path.append(self.integer())
                    self.need("]")
                elif self.accept("."):
                    path.append(self.name().text)
                else:
                    break
            return key, self.node(key, "object_path", tuple(path))
        following = self.tokens[self.pos + 1].text
        scalar_expression = (NAME.fullmatch(self.cur.text) and following in ("+", "-", "*", "/", "%", "<<", ">>"))
        offset_expression = key.text == "offset" and following in (".", "[", "(")
        if (scalar_expression or offset_expression or self.cur.text in ("true", "false", "+", "-")
                or self.cur.text[0].isdigit()):
            value = self.expr()
            return key, self.node(value.token, "object_value", value)
        if key.text in TYPE_OBJECT_SCALARS and self.cur.text == "(":
            start = self.pos
            try:
                value = self.expr()
                if self.cur.text in (",", "}") and value.kind != "name":
                    return key, self.node(value.token, "object_value", value)
            except LanError:
                pass
            self.pos = start
        return key, self.type_expr()

    def is_json_map_start(self):
        if self.cur.text.startswith(('"', "'")) and self.tokens[self.pos + 1].text == ":":
            return True
        if self.cur.text == "[":
            depth = 0
            pos = self.pos
            while pos < len(self.tokens) and self.tokens[pos].text != "<eof>":
                tok = self.tokens[pos].text
                if tok == "[":
                    depth += 1
                elif tok == "]":
                    depth -= 1
                    if depth == 0:
                        return pos + 1 < len(self.tokens) and self.tokens[pos + 1].text == ":"
                pos += 1
        return False

    def json_map_entry(self):
        if self.accept("["):
            key = self.expr()
            self.need("]")
            key_kind = "computed"
        elif self.cur.text.startswith(('"', "'")):
            tok = self.take()
            key = self.node(tok, "map_key", decode_literal(tok))
            key_kind = "literal"
        else:
            self.cur.error("expected a string literal, character literal, or '[expr]' key in mapping object")
        self.need(":")
        value = self.expr()
        return key_kind, key, value

    def initializer_field(self):
        name = self.name()
        self.need(":")
        return name, self.expr()

    def expr(self, min_prec=0, *, allow_as=True):
        t = self.take()
        op = LOGIC.get(t.text, t.text)
        if t.text == "(":
            left = self.expr()
            self.need(")")
        elif op in ("-", "+", "!", "~"):
            left = self.node(t, "unary", op, self.expr(13))
        elif t.text in ("true", "false"):
            left = self.node(t, "boolean", t.text == "true")
        elif t.text == "null":
            left = self.node(t, "null")
        elif t.text == "callback" and self.accept("("):
            target = self.name()
            self.need(")")
            left = self.node(t, "callback_ref", target)
        elif t.text == "@":
            bits = self.take()
            if (bits.line != t.line or bits.col != t.col + 1
                    or not bits.text or any(bit not in "01" for bit in bits.text)):
                t.error("bit literal requires @ followed by binary digits")
            left = self.node(t, "bit_literal", int(bits.text, 2), len(bits.text))
        elif t.text.startswith(("\"", "'")):
            literal = decode_literal(t)
            if t.text[0] == "'":
                if len(literal) != 1:
                    t.error("character literal requires one Unicode scalar value")
                left = self.node(t, "character", ord(literal))
            else:
                left = self.node(t, "text_literal", literal.encode("utf-8"))
        elif re.fullmatch(r"(?:[0-9]+\.[0-9]+|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+", t.text):
            left = self.node(t, "float_number", t.text)
        elif t.text[0].isdigit():
            left = self.node(t, "number", self.number(t))
        elif t.text == "{":
            if self.is_json_map_start():
                left = self.node(t, "json_map", self.comma_list("}", self.json_map_entry))
            else:
                left = self.node(t, "record_initializer", self.comma_list("}", self.initializer_field))
        elif t.text == "[":
            if self.accept("]"):
                left = self.node(t, "list", ())
            else:
                first = self.expr()
                if self.accept(";"):
                    count = self.expr()
                    if count.kind == "number" or (count.kind == "unary" and count.args[0] == "-" and count.args[1].kind == "number"):
                        count = count.args[0] if count.kind == "number" else -count.args[1].args[0]
                    self.need("]")
                    left = self.node(t, "repeat", first, count)
                else:
                    items = (first,)
                    if self.accept(","):
                        items += self.comma_list("]", self.expr)
                    elif self.cur.text != "]":
                        self.cur.error(f"expected ',', ';' or ']', got {self.cur.text!r}")
                    else:
                        self.need("]")
                    left = self.node(t, "list", items)
        elif NAME.fullmatch(t.text) and t.text not in RESERVED:
            left = self.node(t, "name", t.text)
        else:
            t.error("expected an expression")
        while True:
            if call := self.accept("("):
                if left.kind == "member" and left.args[1].text == "count":
                    host = left.args[0]
                    selected = self.type_expr()
                    column = []
                    if self.accept(","):
                        column.append(self.name().text)
                        while self.accept("."):
                            column.append(self.name().text)
                    self.need(")")
                    left = self.node(call, "storage_count", host, selected, tuple(column))
                else:
                    left = self.node(call, "call", left, self.comma_list(")", self.expr))
                continue
            if index := self.accept("["):
                if NAME.fullmatch(self.cur.text) and self.tokens[self.pos + 1].text == ":":
                    left = self.node(index, "bit_view", left, self.comma_list("]", self.bit_field))
                    continue
                start_pos = self.pos
                start = None if self.cur.text == "," else self.expr()
                if self.accept(","):
                    stop = None if self.cur.text in (")", "]") else self.expr()
                    end = self.take()
                    if end.text not in (")", "]"):
                        end.error("expected a slice closing bracket")
                    left = self.node(index, "slice", left, start, stop, end.text == "]")
                else:
                    if start is not None:
                        start = replace(start, source_text=self.state_source(start_pos))
                    self.need("]")
                    left = self.node(index, "index", left, start)
                continue
            if dot := self.accept("."):
                name = self.name()
                left = self.node(dot, "length" if name.text == "len" else "member", left, name)
                continue
            if question := self.accept("?"):
                left = self.node(question, "propagate", left)
                continue
            op = LOGIC.get(self.cur.text, self.cur.text)
            if op == "as" and not allow_as:
                return left
            prec = PREC.get(op, -1)
            if prec < min_prec:
                return left
            tok = self.take()
            if op == "as":
                strategies = ("safe", "sat", "wrap", "view", "trap", "trunc", "floor", "ceil", "round",
                              "trunc_sat", "trunc_trap", "floor_sat", "floor_trap", "ceil_sat", "ceil_trap",
                              "round_sat", "round_trap", "c_string", "move", "copy")
                if self.accept(":"):
                    builtins = ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64",
                                "float32", "float64", "boolean", "ascii", "utf8", "unicode")
                    if self.cur.text in strategies:
                        strategy = self.take().text
                    elif self.cur.text in builtins or self.cur.text in ("(", "[", "{"):
                        strategy = "safe"
                    else:
                        saved = self.pos
                        candidate = self.take().text
                        if NAME.fullmatch(self.cur.text) and self.cur.text not in RESERVED:
                            tok.error(f"unsupported conversion strategy {candidate!r}")
                        self.pos = saved
                        strategy = "safe"
                else:
                    strategy = "safe"
                if strategy == "c_string":
                    left = self.node(tok, "cast", left, "view", self.node(tok, "type_name", "c_string"))
                    continue
                if strategy == "move":
                    tok.error("ownership moves are inferred; declare the target with storage: heap or return it as an owned result")
                if strategy == "copy":
                    left = self.node(tok, strategy, left)
                    continue
                if strategy in ("trunc", "floor", "ceil", "round") and self.cur.text in ("sat", "trap"):
                    strategy += "_" + self.take().text
                elif strategy in ("sat", "trap") and self.cur.text in ("trunc", "floor", "ceil", "round"):
                    strategy = self.take().text + "_" + strategy
                if strategy not in ("safe", "sat", "wrap", "view", "trap", "trunc", "floor", "ceil", "round",
                                    "trunc_sat", "trunc_trap", "floor_sat", "floor_trap", "ceil_sat", "ceil_trap", "round_sat", "round_trap"):
                    tok.error(f"unsupported conversion strategy {strategy!r}")
                left = self.node(tok, "cast", left, strategy, self.type_expr(sets=False))
            elif op == "in":
                left = self.node(tok, "in", left, self.type_expr(sets=False))
            elif op in ("===", "!==") and self.accept("@"):
                rank = self.integer()
                if rank < 0:
                    tok.error("comparison rank must be nonnegative")
                right = self.expr(prec + 1)
                if rank == 0:
                    left = self.node(tok, "binary", "==" if op == "===" else "!=", left, right)
                else:
                    left = self.node(tok, "rank_compare", op, rank, left, right)
            else:
                left = self.node(tok, "binary", op, left, self.expr(prec + 1, allow_as=allow_as))

    def bit_field(self):
        name = self.name()
        self.need(":")
        ty = self.type_expr()
        self.need("@")
        self.need("bit")
        self.need("(")
        width = self.expr()
        self.need(",")
        offset = self.expr()
        endian = "le"
        if self.accept(','):
            self.need('endian')
            self.need(':')
            endian = self.name().text
        self.need(")")
        if self.accept("{"):
            self.need("endian")
            self.need(":")
            endian = self.name().text
            self.need("}")
        return name, ty, width, offset, endian

    def block(self):
        self.need("{")
        result = []
        while self.cur.text not in ("}", "<eof>"):
            result.append(self.stmt())
        self.need("}")
        return result

    def branch_condition(self, token):
        self.need("(")
        start = self.pos
        condition = self.expr()
        self.conditions[token] = " ".join(part.text for part in self.tokens[start:self.pos])
        self.need(")")
        return condition

    def state_subject(self):
        start = self.pos
        name = self.name()
        subject = self.node(name, "name", name.text)
        while self.cur.text in (".", "["):
            if dot := self.accept("."):
                field = self.name()
                subject = self.node(dot, "member", subject, field)
                continue
            bracket = self.need("[")
            first = None if self.cur.text == "," else self.expr()
            if self.accept(","):
                last = None if self.cur.text in (")", "]") else self.expr()
                end = self.take()
                if end.text not in (")", "]"):
                    end.error("expected an array interval closing bracket")
                subject = self.node(bracket, "slice", subject, first, last, end.text == "]")
            else:
                self.need("]")
                subject = self.node(bracket, "index", subject, first)
        return replace(subject, source_text=self.state_source(start))

    def state_source(self, start):
        return " ".join(token.text for token in self.tokens[start:self.pos])

    def state_value(self, subject):
        if (self.cur.text == "{" and NAME.fullmatch(self.tokens[self.pos + 1].text)
                and self.tokens[self.pos + 2].text == ":"):
            self.need("{")

            def field():
                name = self.name()
                self.need(":")
                child = replace(self.node(name, "member", subject, name),
                                source_text=f"{subject.source_text}.{name.text}")
                return self.state_value(child)

            return tuple(claim for fields in self.comma_list("}", field) for claim in fields)
        start = self.pos
        annotation = self.type_expr()
        return ((subject, replace(annotation, source_text=self.state_source(start))),)

    def loop_claims(self, *, guards=False):
        if self.cur.text not in ("state", "when") or self.tokens[self.pos + 1].text != "(":
            def legacy_claim():
                subject = self.state_subject()
                self.need(":")
                return self.state_value(subject)

            return tuple(claim for fields in self.comma_list("}", legacy_claim) for claim in fields)
        claims = []
        while not self.accept("}"):
            guard = None
            if self.accept("when"):
                if not guards:
                    self.tokens[self.pos - 1].error("guarded state clauses require @ensures")
                self.need("(")
                start = self.pos
                guard = replace(self.expr(), source_text=self.state_source(start))
                self.need(")")
            self.need("state")
            self.need("(")
            subject = self.state_subject()
            self.need(")")
            self.need(":")
            self.need("=")
            claims.extend((replace(place, state_guard=guard), value)
                          for place, value in self.state_value(subject))
            self.need(";")
        return tuple(claims)

    def loop_oracle(self, token):
        body = re.sub(r"(?m)^[ \t]*\* ?", "", token.text[2:-2])
        parser = Parser(body)
        parser.tokens = [replace(t, line=t.line + token.line - 1) for t in parser.tokens]
        parser.need("@")
        parser.need("oracle")
        invariant, measure, transition, ensures, trusted = None, None, None, None, False
        while parser.cur.text != "<eof>":
            parser.need("@")
            directive = parser.take()
            if directive.text == "trusted":
                if trusted:
                    directive.error("duplicate @trusted directive")
                trusted = True
                continue
            if directive.text == "transition":
                parser.need("=>")
            parser.need("{")
            if directive.text == "transition" and transition is None:
                transition = parser.loop_claims()
            elif directive.text == "invariant" and invariant is None:
                invariant = parser.loop_claims()
            elif directive.text == "ensures" and ensures is None:
                ensures = parser.loop_claims(guards=True)
            elif directive.text == "proof" and measure is None:
                parser.need("decreases")
                parser.need(":")
                measure = parser.expr()
                parser.need("}")
            else:
                directive.error("expected a unique @transition, @invariant, @ensures or @proof loop directive")
        if invariant is None and measure is None and transition is None and ensures is None:
            token.error("loop oracle requires a transition, invariant, postcondition or proof")
        if trusted and transition is None:
            token.error("@trusted requires a transition")
        return invariant or (), measure, transition, trusted, ensures or ()

    def stmt(self):
        t = self.cur
        if t.text.startswith("/*"):
            self.take()
            if self.cur.text != "while" or self.in_type_function:
                t.error("local oracle must immediately precede a runtime while loop")
            annotations = self.loop_oracle(t)
            loop = self.stmt()
            return self.node(loop.token, "while", *loop.args, *annotations)
        if self.accept("define"):
            name = self.name()
            ty = self.type_expr() if self.accept(":") else None
            value = self.expr() if self.accept("=") else None
            self.need(";")
            return self.node(t, "define", name, ty, value)
        if self.accept("assert"):
            if self.cur.text == "chain" and self.tokens[self.pos + 1].text == "(":
                self.take()
                self.need("(")
                self.need("head")
                self.need(":")
                head = self.expr()
                self.need("->")
                self.need("tail")
                self.need(":")
                tail = self.expr()
                self.need(")")
                self.need(";")
                return self.node(t, "chain_assert", head, tail)
            subject = self.expr()
            self.need(":")
            self.need("{")
            rows = []
            while self.cur.text != "}":
                marker = self.need("[")
                guard = None if self.accept("_") else self.expr()
                self.need("]")
                self.need("=>")
                claim = None if self.accept("unreachable") else self.type_expr()
                rows.append((marker, guard, claim))
                if not self.accept(","):
                    break
                if guard is None and self.cur.text != "}":
                    marker.error("the wildcard guard must be last")
            self.need("}")
            self.need(";")
            return self.node(t, "guard_assert", subject, tuple(rows))
        if self.accept("if"):
            cond = self.branch_condition(t)
            yes = self.block()
            no = ([self.stmt()] if self.cur.text == "if" else self.block()) if self.accept("else") else []
            return self.node(t, "if", cond, yes, no)
        if self.accept("while"):
            cond = self.branch_condition(t)
            self.loop_depth += 1
            body = self.block()
            self.loop_depth -= 1
            return self.node(t, "while", cond, body)
        if self.accept("interation"):
            self.need("(")
            name = self.name()
            self.need(";")
            initial = self.expr()
            self.need(";")
            annotation = self.type_expr()
            self.need(";")
            count = self.expr()
            self.need(";")
            step = self.expr()
            self.need(")")
            self.loop_depth += 1
            body = self.block()
            self.loop_depth -= 1
            return self.node(t, "interation", name, initial, annotation, count, step, body)
        if t.text == "break":
            t.error("break was removed; express the exit in the loop condition")
        if t.text == "continue":
            self.take()
            if not self.loop_depth:
                t.error(f"{t.text} requires an enclosing loop")
            self.need(";")
            return self.node(t, t.text)
        if self.accept("return"):
            if self.cur.text == ";":
                value = None
            elif self.in_type_function:
                start = self.pos
                try:
                    value = self.type_expr()
                except LanError:
                    value = None
                if value is None or self.cur.text != ";":
                    self.pos = start
                    value = self.expr()
            else:
                value = self.expr()
            self.need(";")
            return self.node(t, "return", value)
        if t.text in RESERVED:
            t.error(f"{t.text!r} statements are not supported in this version")
        left = self.expr()
        if self.cur.text in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="):
            op = self.take()
            right = self.expr()
            self.need(";")
            return self.node(t, "assign", left, op, right)
        self.need(";")
        return self.node(t, "eval", left)

    def module(self):
        result = []
        while self.cur.text != "<eof>":
            t = self.take()
            if t.text.startswith("/*"):
                result.append(self.oracle(t))
            elif t.text in ("theorem", "axiom"):
                relation = self.name()
                self.need("(")
                operands = self.comma_list(")", self.type_expr)
                self.need(";")
                result.append(self.node(t, "theorem", relation, operands, t.text == "axiom"))
            elif t.text in ("import", "export"):
                self.need("{")
                items = self.comma_list("}", self.module_name)
                if t.text == "import":
                    self.need("from")
                    path = self.take()
                    if not path.text.startswith('"'):
                        path.error("expected a double-quoted relative module path")
                    try:
                        target = json.loads(path.text)
                    except ValueError:
                        path.error("invalid module path string")
                    result.append(self.node(t, "import", items, path, target))
                else:
                    result.append(self.node(t, "export", items))
                self.need(";")
            elif t.text == "enum":
                name = self.name()
                base = self.type_expr() if self.accept(":") else self.node(t, "type_name", "uint8")
                self.need("{")
                members = self.comma_list("}", self.enum_member)
                self.accept(";")
                result.append(self.node(t, "type", name, self.node(t, "enum", name, base, members)))
            elif t.text == "type":
                if self.cur.text == "string" and self.tokens[self.pos + 1].text not in ("=", "("):
                    self.take()
                    name = self.name()
                    self.need("=")
                    map_expr = self.expr(allow_as=False)
                    mode = "slice"
                    if self.accept("as"):
                        mode_tok = self.name()
                        if mode_tok.text not in ("slice", "tag"):
                            mode_tok.error("expected 'slice' or 'tag' after 'as' in type string declaration")
                        mode = mode_tok.text
                    self.need(";")
                    result.append(self.node(t, "string_encoding", name, map_expr, mode))
                    continue
                if marker := self.accept("enum"):
                    name = self.name()
                    self.need("{")
                    fields = self.comma_list("}", self.parameter)
                    self.accept(";")
                    result.append(self.node(t, "type", name, self.node(marker, "keyed_record", name, fields)))
                    continue
                name = self.name()
                if self.accept("("):
                    params = self.comma_list(")", self.type_function_parameter)
                    self.in_type_function = True
                    body = self.block()
                    self.in_type_function = False
                    result.append(self.node(t, "type_function", name, params, body))
                else:
                    self.need("=")
                    result.append(self.node(t, "type", name, self.type_declaration_rhs()))
                    self.need(";")
            elif t.text == "expand":
                fn_name = self.name()
                self.need("(")
                target = self.name()
                extra = []
                while self.accept(","):
                    extra.append(self.type_application_argument())
                self.need(")")
                self.need(";")
                result.append(self.node(t, "expand", fn_name, target, tuple(extra)))
            elif t.text == "extern":
                keyword = self.need("contract")
                name = self.name()
                self.need("(")
                params = self.comma_list(")", self.parameter)
                self.need("->")
                ty = self.node(keyword, "contract", params, self.type_expr())
                writes = self.write_set()
                self.need(";")
                result.append(self.node(t, "extern", name, ty, writes))
            elif t.text == "define":
                name = self.name()
                if self.accept("="):
                    value = self.expr()
                    self.need(";")
                    result.append(self.node(t, "const", name, None, value))
                    continue
                if self.accept(":"):
                    self.in_implementation_contract = True
                    ty = self.type_expr()
                    self.in_implementation_contract = False
                    if self.accept("="):
                        value = self.expr()
                        self.need(";")
                        result.append(self.node(t, "const", name, ty, value))
                        continue
                    if self.accept(";"):
                        result.append(self.node(t, "const", name, ty, None))
                        continue
                    self.need("(")
                    args = self.comma_list(")", self.name)
                else:
                    self.need("(")
                    params = self.comma_list(")", self.implementation_parameter)
                    args = tuple(param for param, _ in params)
                    returns = self.type_expr() if self.accept("->") else None
                    ty = self.node(t, "contract", params, returns)
                if property_token := self.accept("property"):
                    if ty.kind != "contract":
                        property_token.error("property requires an inline contract")
                    self.need(":")
                    self.need("forall")
                    self.need("(")
                    quantified = self.name()
                    other = self.name() if self.accept(",") else None
                    self.need(")")
                    self.need("=>")
                    names = (quantified.text,) if other is None else (quantified.text, other.text)
                    diagonal = other is None and len(args) == 2 and quantified.text == args[0].text
                    if not diagonal and names != tuple(arg.text for arg in args):
                        property_token.error("property must quantify the function parameters in order")
                    if diagonal:
                        x = quantified.text
                        for part in (name.text, "(", x, ",", x, ")", "==", x):
                            self.need(part)
                        property_kind = "diagonal"
                    elif other is not None:
                        x, y = names
                        for part in (name.text, "(", x, ",", y, ")", "==", name.text, "(", y, ",", x, ")"):
                            self.need(part)
                        property_kind = "commutative"
                    elif self.accept("result"):
                        self.need("==")
                        if self.name().text != quantified.text:
                            property_token.error("identity property must return the quantified parameter")
                        property_kind = "identity"
                    else:
                        for part in (name.text, "(", name.text, "(", quantified.text, ")", ")",
                                     "==", name.text, "(", quantified.text, ")"):
                            self.need(part)
                        property_kind = "idempotent"
                    ty = self.node(ty.token, "contract", *ty.args, property_token, property_kind)
                result.append(self.node(t, "function", name, ty, args, self.block()))
            else:
                t.error("expected an import, export, type declaration, external contract or function definition")
        return result

    def write_set(self):
        """The write set of a native function, taken from its documentation.

        A native has no body to analyse, so the effects it may have on its
        caller are declared instead: `writes {}` for a reader, `writes { a,
        b.field }` for the parameters it writes, `writes { memory }` when it
        allocates or releases memory, `writes { external }` for effects on
        state outside Lan, and `writes { all }` for everything at once.
        """
        keyword = self.need("writes")
        self.need("{")
        entries = []
        if not self.accept("}"):
            while True:
                entries.append(self.write_entry())
                if not self.accept(","):
                    self.need("}")
                    break
        return self.node(keyword, "writes", tuple(entries))

    def write_entry(self):
        first = self.name()
        path = [first.text]
        while self.accept("."):
            path.append(self.name().text)
        return self.node(first, "write_path", tuple(path))

    def enum_member(self):
        name = self.name()
        return name, self.expr() if self.accept("=") else None

    def module_name(self):
        source = self.name()
        return source, self.name() if self.accept("as") else source

    @staticmethod
    def oracle(token):
        contents = " ".join(re.sub(r"^\s*\*?\s*", "", line)
                            for line in token.text[2:-2].splitlines()).strip()
        match = re.fullmatch(r"@oracle\s+@theorem\s+(.+?)\s*=>\s*(.+)", contents)
        if not match:
            token.error("expected '@oracle @theorem A & B => Result'")
        try:
            lhs, rhs = (Parser(part) for part in match.groups())
            intersection = lhs.type_expr()
            result = rhs.type_expr()
            lhs.need("<eof>")
            rhs.need("<eof>")
        except LanError as exc:
            token.error(f"invalid intersection theorem: {exc}")
        while intersection.kind == "type_group":
            intersection = intersection.args[0]
        if intersection.kind != "type_op" or intersection.token.text != "&":
            token.error("the left side of an intersection theorem must be A & B")
        return Node(token, "oracle", (*intersection.args, result))
