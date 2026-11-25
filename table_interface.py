"""Versioned declaration records at the start of a Lan state table."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import inf
import re

from value_domain import Clause, Domain, Polynomial, evaluate


VERSION = 1
HEADER = "@interface "
HEADER_LIMIT = 4 * 1024 * 1024
IDENTIFIER = re.compile(r"[^\W\d]\w*\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def c_ident(name):
    """Map any Lan Unicode identifier injectively to a valid pure-ASCII C89 identifier.

    Pure ASCII identifiers not starting with 'lanu_' are returned unchanged.
    """
    if name.isascii():
        return f"lanu__{name[5:]}" if name.startswith("lanu_") else name
    return "lanu_" + name.encode("utf-8").hex()


def c_field(name):
    return f"f_{c_ident(name)}"


class InterfaceError(Exception):
    pass


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def source_identity(module_path):
    return digest("lan module:" + module_path)


def api_symbol(abi, name, signature):
    contract = digest(json.dumps(signature, sort_keys=True, separators=(",", ":")))
    return f"lan_api_{abi}_{contract}_{c_ident(name)}"


def fields(value, expected, label):
    if not isinstance(value, dict) or value.keys() != set(expected):
        raise InterfaceError(f"invalid {label} fields")
    return value


def integer(value, label):
    if type(value) is not int or value.bit_length() > 16_384:
        raise InterfaceError(f"invalid integer in {label}")
    return value


def identifier(value, label):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise InterfaceError(f"invalid identifier in {label}")
    return value


def encode_domain(domain):
    def polynomial(poly):
        return [poly.denominator, list(poly.coefficients)]

    entries = [
        [None if c.lo == -inf else c.lo, None if c.hi == inf else c.hi,
         [polynomial(p) for p in c.require], [polynomial(p) for p in c.exclude]]
        for c in domain.clauses()
    ]
    if len(entries) > 8192 or any(len(entry[2]) + len(entry[3]) > 256 for entry in entries):
        raise InterfaceError("generated value domain exceeds the interface predicate limit")
    return entries


def decode_domain(raw):
    def polynomial(item):
        if not isinstance(item, list) or len(item) != 2:
            raise InterfaceError("invalid polynomial in value domain")
        denominator = integer(item[0], "polynomial denominator")
        values = item[1]
        if denominator <= 0 or not isinstance(values, list) or not 2 <= len(values) <= 65:
            raise InterfaceError("invalid polynomial denominator or degree")
        coefficients = tuple(integer(n, "polynomial coefficient") for n in values)
        if not coefficients[-1] or any(evaluate(coefficients, n) % denominator
                                        for n in range(len(coefficients))):
            raise InterfaceError("invalid integer-valued polynomial")
        return Polynomial.reduced(coefficients, denominator)

    if not isinstance(raw, list) or len(raw) > 8192:
        raise InterfaceError("invalid value domain")
    intervals, clauses = [], []
    for item in raw:
        if not isinstance(item, list) or len(item) != 4:
            raise InterfaceError("invalid interval in value domain")
        lo, hi, require, exclude = item
        lo = -inf if lo is None else integer(lo, "interval lower bound")
        hi = inf if hi is None else integer(hi, "interval upper bound")
        if (lo > hi or not isinstance(require, list) or not isinstance(exclude, list)
                or len(require) + len(exclude) > 256):
            raise InterfaceError("invalid bounds or predicates in value domain")
        if require or exclude:
            clauses.append(Clause(lo, hi, tuple(map(polynomial, require)), tuple(map(polynomial, exclude))))
        else:
            intervals.append((lo, hi))
    return Domain.of(*intervals) | Domain.from_clauses(clauses)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InterfaceError(f"duplicate interface field {key!r}")
        result[key] = value
    return result


def read_digest(value, label):
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise InterfaceError(f"invalid {label}")
    return value


@dataclass(frozen=True)
class TableRecord:
    identity: str
    abi: str
    modules: dict
    exports: dict

    def encode(self):
        if len(self.modules) > 4096 or len(self.exports) > 4096:
            raise InterfaceError("generated interface exceeds the module or export limit")
        payload = {"version": VERSION, "module": self.identity, "abi": self.abi,
                   "modules": self.modules, "exports": self.exports}
        line = HEADER + json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if len(line) + 1 > HEADER_LIMIT:
            raise InterfaceError("generated interface exceeds the size limit")
        return line + "\n"

    @staticmethod
    def read(path):
        with path.open("rb") as stream:
            header = stream.readline(HEADER_LIMIT + 1)
        if len(header) > HEADER_LIMIT:
            raise InterfaceError("interface declaration exceeds the size limit")
        line = header.decode("utf-8-sig")
        if not line.startswith(HEADER):
            raise InterfaceError("state table has no declaration interface; regenerate it with lanc")
        try:
            raw = json.loads(line[len(HEADER):], object_pairs_hook=unique_object)
        except (ValueError, RecursionError) as exc:
            raise InterfaceError(f"invalid interface declaration: {exc}") from exc
        record = fields(raw, ("version", "module", "abi", "modules", "exports"), "interface")
        if type(record["version"]) is not int or record["version"] != VERSION:
            raise InterfaceError(f"unsupported .table interface version {record['version']!r}")
        identity = read_digest(record["module"], "module identity")
        abi = read_digest(record["abi"], "ABI identity")
        modules, exports = record["modules"], record["exports"]
        if not isinstance(modules, dict) or identity not in modules or len(modules) > 4096:
            raise InterfaceError("invalid module graph in interface")
        for key, raw_module in modules.items():
            read_digest(key, "dependency identity")
            module = fields(raw_module, ("name", "imports"), "module")
            name, imports = module["name"], module["imports"]
            if not isinstance(name, str) or not name or len(name) > 4096 or not name.isprintable():
                raise InterfaceError("invalid module display name")
            if (not isinstance(imports, list) or len(imports) > len(modules)
                    or any(not isinstance(dep, str) or dep not in modules for dep in imports)
                    or len(set(imports)) != len(imports)):
                raise InterfaceError("invalid module dependency list")
        reachable, pending = set(), [identity]
        while pending:
            key = pending.pop()
            if key in reachable:
                continue
            reachable.add(key)
            pending.extend(modules[key]["imports"])
        if reachable != modules.keys():
            raise InterfaceError("interface module graph contains unrelated modules")
        if not isinstance(exports, dict) or len(exports) > 4096:
            raise InterfaceError("invalid export list")
        for name in exports:
            identifier(name, "export")
        return TableRecord(identity, abi, modules, exports)
