"""The Lan compiler package (formerly the single file lanc.py).

Layout
------
Front end      errors, lexer, syntax, parser
Types/state    typesys, domain_utils, state
Interfaces     interface_types, summaries, interface_module   (.table encoding/decoding)
Compiler       compiler.Compiler, assembled from one mixin per concern
               (type_resolution, constants, declarations, returns, records, ...,
               loops, functions, recursion, codegen)
Back end       c_emit
Driver         module_graph, driver (compile_source, main)

Linking
-------
A few names cannot be imported normally without an import cycle:
* low-level helpers call static methods through `Compiler`, which is assembled last;
* `typesys` uses `encode_interface_type` from `interface_types`, which depends on `typesys`;
* `composite_array.install` adds the result-content codecs and replaces `ArrayDomain`.
Modules declare such names as `None` placeholders in a "Late-bound names" block;
`_link()` below fills them in once everything is loaded, so every module sees the
same objects the old single-module namespace held.
"""
from __future__ import annotations

import importlib

from composite_array import install as _install_composite_arrays

_MODULES = (
    "errors", "lexer", "syntax", "parser", "encoding", "typesys", "domain_utils", "state",
    "interface_types", "summaries", "interface_module", "c_emit",
    "type_resolution", "constants", "declarations", "returns", "records", "storage_links",
    "aliases_heap", "text", "dependent", "array_contents", "slices", "slice_compare", "bits",
    "operations", "writes", "interfaces", "calls", "expressions", "properties", "guards",
    "views", "statements", "bit_assign", "loops", "loop_chains", "loop_contracts",
    "loop_analysis", "blocks", "registration", "inference", "functions", "theorems",
    "recursion", "codegen", "compiler", "module_graph", "driver",
)

_modules = [importlib.import_module(f"{__name__}.{name}") for name in _MODULES]


class _PackageNamespace(dict):
    """All package-level names merged into one mapping.

    `composite_array.install` reads names from this mapping and writes its additions
    back; each write is propagated to every package module that holds the name."""

    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        for module in _modules:
            if name in vars(module):
                setattr(module, name, value)


def _link():
    namespace = _PackageNamespace()
    for module in _modules:
        late = set(getattr(module, "_LATE_BOUND", ()))
        for name, value in vars(module).items():
            if not name.startswith("__") and name not in late and name != "_LATE_BOUND":
                dict.setdefault(namespace, name, value)
    # Fill the late-bound placeholders of every module.  The result-content codecs are
    # not defined yet; composite_array installs them through the namespace below.
    for module in _modules:
        for name in getattr(module, "_LATE_BOUND", ()):
            if name in namespace:
                setattr(module, name, namespace[name])
    _install_composite_arrays(namespace)
    unresolved = [f"{module.__name__}.{name}" for module in _modules
                  for name in getattr(module, "_LATE_BOUND", ()) if getattr(module, name) is None]
    if unresolved:
        raise ImportError("unresolved late-bound names: " + ", ".join(unresolved))
    return namespace


_namespace = _link()

# Re-export the whole former lanc namespace (helpers included) for compatibility.
__all__ = sorted(name for name in _namespace
                 if not name.startswith("__") and not name.endswith("Mixin") and name not in _MODULES)
globals().update({name: _namespace[name] for name in __all__})
