"""Multi-module project graph: imports, compilation order and artifacts."""

from __future__ import annotations

import json
import os.path
from pathlib import Path, PureWindowsPath

from value_domain import DomainComplexity
from table_interface import InterfaceError, TableRecord, VERSION as INTERFACE_VERSION, digest, encode_domain, source_identity

from .errors import LanError
from .typesys import C_HEADER
from .domain_utils import ascii_table_trace
from .interface_module import interface_exports, InterfaceModule
from .compiler import Compiler


class ModuleGraph:
    def __init__(self, root, compiler, library=False, *, module_root=None, export_prefix=None):
        self.root = root.resolve()
        self.modules = {self.root: compiler}
        compiler.source_path = str(self.root)
        compiler.library = library
        compiler.export_prefix = export_prefix
        self.library = library
        self.edges = {self.root: []}
        self.module_root = Path(module_root).resolve() if module_root is not None else None

    def display_path(self, path):
        try:
            return Path(os.path.relpath(path, self.module_root or self.root.parent)).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def imports(compiler):
        if isinstance(compiler, Compiler):
            return (node for node in compiler.nodes if node.kind == "import")
        return iter(())

    def resolve_import(self, parent, node):
        _, token, target = node.args
        windows = PureWindowsPath(target)
        relative = Path(target)
        if (not target or "\0" in target or relative.is_absolute() or windows.drive or windows.root
                or relative.suffix.lower() not in (".lang", ".table")):
            token.error("module import requires a relative .lang or .table path")
        try:
            return (parent.parent / relative).resolve()
        except (OSError, ValueError, RuntimeError) as exc:
            token.error(f"cannot resolve module path {target!r}: {exc}")

    def load(self, dependency, node):
        token = node.args[1]
        if dependency.suffix.lower() == ".table":
            try:
                interface = InterfaceModule.load(dependency, self.display_path(dependency))
            except (InterfaceError, DomainComplexity, OSError, UnicodeError, RecursionError) as exc:
                token.error(f"cannot read interface {self.display_path(dependency)!r}: {exc}")
            self.modules[dependency] = interface
            self.edges[dependency] = []
            return interface
        try:
            source = dependency.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            token.error(f"cannot read module {self.display_path(dependency)!r}: {exc}")
        try:
            root_compiler = self.modules[self.root]
            compiler = Compiler(source, self.display_path(dependency), module_id=len(self.modules),
                                debug_bounds=root_compiler.debug_bounds,
                                max_steps=root_compiler.max_steps)
        except LanError as exc:
            exc.module = str(dependency)
            raise
        compiler.source_path = str(dependency)
        self.modules[dependency] = compiler
        self.edges[dependency] = []
        return compiler

    def dependency_order(self):
        stack = [(self.root, self.imports(self.modules[self.root]))]
        active = {self.root: 0}
        order = []
        while stack:
            parent, imports = stack[-1]
            node = next(imports, None)
            if node is None:
                stack.pop()
                del active[parent]
                order.append(parent)
                continue
            try:
                dependency = self.resolve_import(parent, node)
                if dependency in active:
                    cycle = [p for p, _ in stack[active[dependency]:]] + [dependency]
                    names = " -> ".join(self.display_path(path) for path in cycle)
                    node.args[1].error(f"cyclic module import: {names}")
                self.edges[parent].append((node, dependency))
                if dependency not in self.modules:
                    compiler = self.load(dependency, node)
                    active[dependency] = len(stack)
                    stack.append((dependency, self.imports(compiler)))
            except LanError as exc:
                if exc.module is None:
                    exc.module = str(parent)
                raise
        return order

    def compile(self, *, include_trace=True):
        order = self.dependency_order()
        sources = [path for path in order if isinstance(self.modules[path], Compiler)]
        if self.module_root is None:
            cwd = Path.cwd().resolve()
            if all(path.is_relative_to(cwd) for path in sources):
                self.module_root = cwd
            else:
                try:
                    self.module_root = Path(os.path.commonpath([path.parent for path in sources]))
                except ValueError as exc:
                    raise LanError("module sources need a common project directory") from exc
        for path in sources:
            if not path.is_relative_to(self.module_root):
                raise LanError("module source is outside the selected module root", self.display_path(path))
            module = self.modules[path]
            module.module_path = self.display_path(path)
            module.filename = module.module_path
            module.trace[0] = f"@module {ascii(module.module_path)}"
        for path in order:
            if isinstance(self.modules[path], InterfaceModule):
                self.modules[path].filename = self.display_path(path)
        identity, abi, manifest = self.interface_graph(order)
        root = self.modules[self.root]
        root.interface_abi = abi
        hide_implementation = self.library or any(node.kind == "export" for node in root.nodes)
        chunks = [C_HEADER]
        for path in order:
            compiler = self.modules[path]
            if isinstance(compiler, InterfaceModule):
                continue
            compiler.internal_linkage = hide_implementation
            try:
                for node, dependency in self.edges[path]:
                    compiler.bind_import(node, self.modules[dependency])
                c, _ = compiler.compile(include_header=False, include_trace=False)
            except (DomainComplexity, InterfaceError) as exc:
                raise LanError(str(exc), str(path)) from exc
            except LanError as exc:
                if exc.module is None:
                    exc.module = str(path)
                raise
            chunks.append(c)
        try:
            header = TableRecord(identity, abi, manifest, interface_exports(root.exports)).encode()
        except (InterfaceError, ValueError) as exc:
            raise LanError(str(exc), str(self.root)) from exc
        if include_trace:
            traces = ["\n".join(self.modules[path].trace) for path in order
                      if isinstance(self.modules[path], Compiler)]
            header += ascii_table_trace(traces)
        return "\n".join(chunks), header

    def write_analysis_artifacts(self, theorem, proof):
        def record(stream, data):
            stream.write(json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n")

        record(theorem, {"format": "lan.theorem", "version": 1})
        record(proof, {"format": "lan.proof", "version": 1})
        for path, compiler in self.modules.items():
            if not isinstance(compiler, Compiler):
                continue
            scope = {"module": self.display_path(path), "source": compiler.source_digest}
            for fact in compiler.source_theorems:
                record(theorem, scope | fact.record())
            for evidence_id, (name, anchor, observation) in enumerate(compiler.storage_evidence):
                record(proof, scope | {"kind": "storage_observation", "id": evidence_id,
                       "function": name, "anchor": anchor, "parameter": observation.parameter,
                       "path": observation.path,
                       "exits": [{
                           "length": exit_state.contents.length,
                           "segments": [(seg.start, seg.stop, encode_domain(seg.domain))
                                        for seg in exit_state.contents.segments],
                           "keys": [(encode_domain(key.positions), encode_domain(key.values))
                                    for key in exit_state.keys],
                           "separated": exit_state.separated,
                       } for exit_state in observation.exits]})
            for evidence_id, (name, anchor, observation) in enumerate(compiler.return_evidence):
                record(proof, scope | {"kind": "return_observation", "id": evidence_id,
                       "function": name, "anchor": anchor, "parameters": observation.parameters,
                       "cases": [{"result": encode_domain(case.result),
                                  "arguments": [encode_domain(domain) for domain in case.arguments]}
                                 for case in observation.cases]})
            for exchange in compiler.proof_exchanges.values():
                for fact in exchange.accepted.values():
                    record(theorem, scope | fact.record())
                for evidence in exchange.evidence:
                    record(proof, scope | {"kind": "ssa_obligation"} | evidence)
            # The old diagnostic trace is still available for consumers that
            # display it. It is not input to the proof verifier or importer.
            record(proof, scope | {"kind": "diagnostics", "trace": compiler.trace})

    def interface_graph(self, order):
        # Interfaces retain source identities and transitive dependencies.
        # An opaque file-level leaf must not hide a logical import cycle.
        identities = {}
        dependencies = {}
        names = {}
        inputs = []
        versions = {}
        for path in order:
            module = self.modules[path]
            if isinstance(module, InterfaceModule):
                record = module.record
                identity = record.identity
                if identity in versions and versions[identity] != record.abi:
                    raise LanError(f"different .table versions of module {module.filename!r}", str(path))
                versions[identity] = record.abi
                for key, item in record.modules.items():
                    dependencies.setdefault(key, set()).update(item["imports"])
                    # Existing tables can carry absolute names. Keep their
                    # identities for linking but avoid republishing the path.
                    label = item["name"]
                    windows = PureWindowsPath(label)
                    if windows.drive or windows.root:
                        label = windows.name or "<module>"
                    names.setdefault(key, label)
                inputs.append(("interface", identity, record.abi))
            else:
                identity = source_identity(module.module_path)
                dependencies.setdefault(identity, set())
                names[identity] = self.display_path(path)
                inputs.append(("source", identity, module.source_digest))
            identities[path] = identity
        for path, edges in self.edges.items():
            dependencies[identities[path]].update(identities[dependency] for _, dependency in edges)
        self.reject_logical_cycles(dependencies, names)
        manifest = {key: {"name": names[key], "imports": sorted(edges)}
                    for key, edges in dependencies.items()}
        identity = identities[self.root]
        abi = digest(json.dumps([INTERFACE_VERSION, identity, sorted(set(inputs)), manifest], sort_keys=True))
        return identity, abi, manifest

    def reject_logical_cycles(self, dependencies, names):
        done = set()
        for root in dependencies:
            if root in done:
                continue
            active = {root: 0}
            stack = [(root, iter(sorted(dependencies[root])))]
            while stack:
                key, edges = stack[-1]
                child = next(edges, None)
                if child is None:
                    stack.pop()
                    del active[key]
                    done.add(key)
                    continue
                if child in active:
                    cycle = [item for item, _ in stack[active[child]:]] + [child]
                    route = " -> ".join(names[item] for item in cycle)
                    raise LanError(f"cyclic module import through .table: {route}", str(self.root))
                if child not in done:
                    active[child] = len(stack)
                    stack.append((child, iter(sorted(dependencies[child]))))
