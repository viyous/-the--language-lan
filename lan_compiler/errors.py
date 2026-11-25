"""Compiler exceptions, oracle requests and literal number wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


class FloatConstant(float):
    def __new__(cls, value, bits=64):
        result = float.__new__(cls, value)
        result.bits = bits
        return result


class FloatLiteral(Fraction):
    def __new__(cls, text):
        result = Fraction.__new__(cls, text)
        result.text = text
        return result


class LanError(Exception):
    def __init__(self, message, module=None):
        super().__init__(message)
        self.module = module


class OracleRequestError(LanError):
    request = None


class ProofBudgetError(OracleRequestError):
    pass


DANGLING_RELEASE_HEADLINE = "Aliases potentially invalidated by this release."
ALIVE = frozenset(("Alive",))


@dataclass(frozen=True)
class OracleRequest:
    path: str
    source_digest: str
    line: int
    column: int
    entries: tuple[str, ...]
    headline: str = "Loop proof budget exhausted. Fill in the unknown exit values."

    def comment(self, newline="\n"):
        return newline.join(["/** @oracle_request",
                              f" * {self.headline}",
                              *(f" * {entry}" for entry in self.entries),
                              " */"])
