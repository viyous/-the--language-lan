"""Imported module interfaces (.table) and export collection."""

from __future__ import annotations

from dataclasses import dataclass, fields

from float_domain import c_literal, FloatDomain
from table_interface import InterfaceError, TableRecord, api_symbol, decode_domain, encode_domain, fields as interface_fields

from .errors import FloatConstant
from .typesys import (Contract, CopyArray, FunctionBinding, HeapArray, HeapObject, HeapRecord,
                      InterfaceTypeFunction, ModuleConst, Scalar, success_type, TaggedResult, TypeFunctionBinding)
from .domain_utils import c_int
from .interface_types import (chain_api_signature, decode_interface_type, encode_interface_type,
                              linked_owned_record, OWNED_CHAIN_LONG)
from .summaries import (decode_guard, decode_mutations, decode_owning, decode_result_fields,
                        decode_return_observation, decode_storage_counts, encode_guard, valid_foreign_symbol,
                        valid_interface_accessor)

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler", "decode_result_content", "encode_result_content")
Compiler = None
decode_result_content = None
encode_result_content = None


def interface_exports(exports):
    encoded = {}
    for name, binding in exports.items():
        if isinstance(binding, FunctionBinding):
            encoded[name] = {"kind": "function", "type": encode_interface_type(binding.contract)}
            if binding.foreign:
                encoded[name]["foreign_symbol"] = binding.c_name
            if binding.result_fields:
                encoded[name]["result_fields"] = [[list(names), encode_result_content(value)]
                                                  for names, value in binding.result_fields]
            if binding.result_array is not None:
                encoded[name]["result_array"] = encode_result_content(binding.result_array)
            if binding.result_domain is not None:
                encoded[name]["result_domain"] = encode_domain(binding.result_domain)
            if binding.owned_chain_lengths:
                encoded[name]["owned_chain_lengths"] = list(binding.owned_chain_lengths)
            if binding.owned_chain_parameters:
                encoded[name]["owned_chain_parameters"] = list(binding.owned_chain_parameters)
            if binding.storage_counts:
                encoded[name]["storage_counts"] = [
                    [fact.parameter, list(fact.path), fact.start, fact.length,
                     encode_domain(fact.selected), *fact.bounds] for fact in binding.storage_counts]
            if binding.return_observation is not None:
                observation = binding.return_observation
                encoded[name]["return_observation"] = {
                    "parameters": list(observation.parameters),
                    "cases": [[encode_domain(domain) for domain in (case.result, *case.arguments)]
                              for case in observation.cases],
                }
            if binding.guard is not None:
                encoded[name]["guard"] = encode_guard(binding.guard)
            if binding.interface_accessor is not None:
                index, path, *arithmetic = binding.interface_accessor
                encoded[name]["interface_accessor"] = (
                    [index, list(path), *arithmetic] if arithmetic
                    else path[0] if len(path) == len(binding.contract.params) == 1
                    else [index, path[0] if len(path) == 1 else list(path)])
            if binding.mutations is not None:
                encoded[name]["mutations"] = [[index, list(names), ({"float": post.to_dict()} if isinstance(post, FloatDomain) else encode_domain(post)) if post is not None else None]
                                                for index, names, post in binding.mutations]
            if binding.memory or not binding.foreign:
                encoded[name]["memory"] = binding.memory
            encoded[name]["external_effect"] = binding.external_effect
            if binding.owning:
                encoded[name]["owning"] = [[index, method] for index, method in binding.owning]
        elif isinstance(binding, (TypeFunctionBinding, InterfaceTypeFunction)):
            encoded[name] = {"kind": "type_function"}
        elif isinstance(binding, ModuleConst):
            if isinstance(binding.ty, Scalar) and binding.scalar_value is not None:
                val = binding.scalar_value
                if isinstance(val, FloatConstant):
                    encoded[name] = {"kind": "const", "type": encode_interface_type(binding.ty),
                                     "float": binding.content.to_dict(), "c_init": binding.c_init}
                else:
                    encoded[name] = {"kind": "const", "type": encode_interface_type(binding.ty),
                                     "value": val, "c_init": binding.c_init}
            else:
                encoded[name] = {"kind": "const_source"}
        else:
            encoded[name] = {"kind": "type", "type": encode_interface_type(binding)}
    return encoded


@dataclass
class InterfaceModule:
    filename: str
    record: TableRecord
    exports: dict

    @staticmethod
    def load(path, display_name):
        record = TableRecord.read(path)
        exports = {}
        for name, raw in record.exports.items():
            if not isinstance(raw, dict):
                raise InterfaceError(f"invalid export {name!r}")
            kind = raw.get("kind")
            if kind == "type_function":
                interface_fields(raw, ("kind",), "type function export")
                exports[name] = InterfaceTypeFunction(name)
                continue
            if kind == "const_source":
                interface_fields(raw, ("kind",), "source constant export")
                exports[name] = ModuleConst(None, None, None, source_only=True, name=name)
                continue
            if kind == "const":
                if "float" in raw:
                    interface_fields(raw, ("kind", "type", "float", "c_init"), "float constant export")
                    ty = decode_interface_type(raw["type"])
                    if not isinstance(ty, Scalar) or not ty.floating:
                        raise InterfaceError("invalid float constant export")
                    domain = FloatDomain.from_dict(raw["float"])
                    val = FloatConstant(-0.0 if domain.zeros == 2 else domain.lo, ty.bits)
                    exports[name] = ModuleConst(None, ty, domain, scalar_value=val, c_init=c_literal(val, ty.bits), name=name)
                else:
                    interface_fields(raw, ("kind", "type", "value", "c_init"), "constant export")
                    ty = decode_interface_type(raw["type"])
                    val = raw["value"]
                    if not isinstance(ty, Scalar) or ty.floating or type(val) not in (int, bool):
                        raise InterfaceError("invalid scalar constant export")
                    domain = decode_domain([[int(val), int(val), [], []]])
                    exports[name] = ModuleConst(None, ty, domain, scalar_value=val, c_init=c_int(int(val)), name=name)
                continue
            summaries = tuple(key for key in ("mutations", "result_fields", "result_array", "result_domain", "guard",
                                              "interface_accessor", "owning", "memory", "external_effect",
                                              "storage_counts", "return_observation", "owned_chain_lengths",
                                              "owned_chain_parameters") if key in raw)
            keys = ("kind", "type") + summaries
            if "foreign_symbol" in raw:
                keys += ("foreign_symbol",)
            interface_fields(raw, keys, "export")
            ty = decode_interface_type(raw["type"])
            # Older Lan tables omitted this field even for direct allocation.
            # Native tables get their effects from an explicit writes clause.
            memory = raw.get("memory", kind == "function" and "foreign_symbol" not in raw)
            if type(memory) is not bool:
                raise InterfaceError("invalid memory write summary")
            external_effect = raw.get("external_effect", kind == "function")
            if type(external_effect) is not bool:
                raise InterfaceError("invalid external effect summary")
            if kind == "function" and isinstance(ty, Contract):
                if "foreign_symbol" in raw:
                    symbol = raw["foreign_symbol"]
                    # A native has no body, so its write set is the only
                    # description of what the call does to its caller: it must
                    # be present, and it may not be replaced by any other
                    # summary the compiler derives from a Lan implementation.
                    native = tuple(key for key in summaries if key in ("mutations", "memory", "external_effect"))
                    if (not valid_foreign_symbol(symbol) or native != summaries
                            or "mutations" not in raw
                            or isinstance(ty.result, TaggedResult)
                            or linked_owned_record(ty.result)
                            or any(linked_owned_record(param) for param in ty.params)):
                        raise InterfaceError("invalid native function export")
                    exports[name] = FunctionBinding(ty, symbol, external=True, foreign=True,
                                                    mutations=decode_mutations(raw["mutations"], ty),
                                                    memory=memory, external_effect=external_effect)
                    continue
                mutations = decode_mutations(raw["mutations"], ty) if "mutations" in raw else None
                if (ty.identity_property or ty.idempotent_property or ty.commutative_property
                        or ty.diagonal_property) and mutations != ():
                    raise InterfaceError("algebraic property requires an explicit empty mutation summary")
                payload = success_type(ty.result)
                fields = decode_result_fields(raw["result_fields"], payload) if "result_fields" in raw else ()
                result_array = None
                if "result_array" in raw:
                    if not isinstance(payload, CopyArray) or Compiler.contains_record_links(payload.array):
                        raise InterfaceError("array result summary requires a fixed array result without links")
                    result_array = decode_result_content(payload.array, raw["result_array"], [4096])
                guard = decode_guard(raw["guard"], ty, mutations) if "guard" in raw else None
                accessor = raw.get("interface_accessor")
                if "interface_accessor" in raw:
                    if isinstance(accessor, str) and len(ty.params) == 1:
                        accessor = (0, (accessor,))
                    elif isinstance(accessor, list) and len(accessor) in (2, 4, 5, 6):
                        def decode_index(step):
                            if len(step) == 3 and step[0] == "affine" and isinstance(step[1], list):
                                return step[0], tuple(tuple(term) if isinstance(term, list) else term
                                                      for term in step[1]), step[2]
                            return tuple(step)

                        def decode_path(path):
                            if not isinstance(path, list):
                                return (path,)
                            steps = []
                            for step in path:
                                if isinstance(step, list):
                                    if len(step) == 2 and step[0] == "from_end" and isinstance(step[1], list):
                                        step = (step[0], decode_index(step[1]))
                                    else:
                                        step = decode_index(step)
                                steps.append(step)
                            return tuple(steps)

                        index, path, *arithmetic = accessor
                        accessor = (index, decode_path(path))
                        if arithmetic:
                            op, other, *source = arithmetic
                            if op == "choose" and len(source) == 2:
                                selector = source[1]
                                if isinstance(selector, list) and len(selector) == 2:
                                    source[1] = selector[0], decode_path(selector[1])
                            accessor += (op, decode_path(other), *source)
                    else:
                        raise InterfaceError("invalid interface accessor summary")
                    if not valid_interface_accessor(ty, accessor, mutations):
                        raise InterfaceError("invalid interface accessor summary")
                owning = decode_owning(raw["owning"], ty) if "owning" in raw else ()
                if {index for index, _ in owning} != {index for index, param in enumerate(ty.params)
                                                      if isinstance(param, (HeapRecord, HeapObject, HeapArray))}:
                    raise InterfaceError("owning parameter summary does not cover every owning parameter")
                result_domain = decode_domain(raw["result_domain"]) if "result_domain" in raw else None
                if result_domain is not None and (
                        not isinstance(ty.result, Scalar) or ty.result.floating or ty.result.result_arms is None
                        or not result_domain or not result_domain.within(ty.result.domain)):
                    raise InterfaceError("invalid result union summary")
                storage_counts = decode_storage_counts(raw.get("storage_counts", []), ty)
                return_observation = (decode_return_observation(raw["return_observation"], ty)
                                      if "return_observation" in raw else None)
                lengths = raw.get("owned_chain_lengths", [])
                if "owned_chain_lengths" in raw and (
                        not linked_owned_record(payload) or not isinstance(lengths, list) or not lengths
                        or len(lengths) > OWNED_CHAIN_LONG
                        or any(type(length) is not int or not 1 <= length <= OWNED_CHAIN_LONG for length in lengths)
                        or lengths != sorted(set(lengths)) or lengths == [1]):
                    raise InterfaceError("invalid owned chain result lengths")
                chain_parameters = raw.get("owned_chain_parameters", [])
                if "owned_chain_parameters" in raw and (
                        not isinstance(chain_parameters, list) or not chain_parameters
                        or len(chain_parameters) > len(ty.params)
                        or any(type(index) is not int or not 0 <= index < len(ty.params)
                               or not linked_owned_record(ty.params[index]) for index in chain_parameters)
                        or chain_parameters != sorted(set(chain_parameters))):
                    raise InterfaceError("invalid owning chain parameter summary")
                signature = chain_api_signature(raw["type"], lengths, chain_parameters)
                exports[name] = FunctionBinding(ty, api_symbol(record.abi, name, signature),
                                                external=True, mutations=mutations, result_fields=fields, guard=guard,
                                                interface_accessor=accessor, result_array=result_array,
                                                owning=owning, memory=memory, result_domain=result_domain,
                                                external_effect=external_effect, storage_counts=storage_counts,
                                                return_observation=return_observation, owned_chain_lengths=tuple(lengths),
                                                owned_chain_parameters=tuple(chain_parameters))
            elif kind == "type":
                interface_fields(raw, ("kind", "type"), "type export")
                exports[name] = ty
            else:
                raise InterfaceError(f"invalid export kind for {name!r}")
        return InterfaceModule(display_name, record, exports)
