"""Convenience compile entry point, oracle insertion and the command line."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from float_domain import FloatDomainError
from value_domain import DomainComplexity
from table_interface import digest

from .errors import DANGLING_RELEASE_HEADLINE, LanError, OracleRequestError
from .compiler import Compiler
from .module_graph import ModuleGraph


def compile_source(source, filename="<input>", *, debug_bounds=False, module_root=None, max_steps=None):
    try:
        compiler = Compiler(source, str(filename), debug_bounds=debug_bounds, max_steps=max_steps)
        needs_files = any(ModuleGraph.imports(compiler))
        if filename != "<input>" and (needs_files or any(node.kind == "export" for node in compiler.nodes)):
            path = Path(filename).resolve()
            return ModuleGraph(path, compiler, module_root=module_root).compile()
        if needs_files:
            raise LanError("module imports require a filename to resolve relative paths")
        return compiler.compile()
    except (DomainComplexity, FloatDomainError) as exc:
        raise LanError(str(exc)) from exc


def insert_oracle_request(request, source):
    return insert_oracle_requests((request,), source)


def insert_oracle_requests(requests, source):
    requests = [request for request in requests if Path(request.path).resolve() == source]
    if not requests:
        return False
    original = source.read_bytes()
    bom = b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    content = original[len(bom):].decode("utf-8")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    current_digest = digest(normalized)
    line_starts = [0]
    for line in content.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    newline = "\r\n" if "\r\n" in content else "\n"
    edits = {}
    for request in requests:
        if current_digest != request.source_digest or not 1 <= request.line < len(line_starts):
            continue
        index = line_starts[request.line - 1] + request.column - 1
        prefix = content[:index]
        existing = re.search(r"(?P<note>/\*\* @oracle_request\b(?:(?!\*/)[\s\S])*\*/)\s*\Z", prefix)
        line_start = content[line_starts[request.line - 1]:index]
        indent_text = line_start if not line_start.strip() else ""
        note = request.comment(newline).replace(newline, newline + indent_text)
        resolved = request.headline == DANGLING_RELEASE_HEADLINE and not request.entries
        if existing is not None:
            old_note = existing.group("note")
            lines = [line.strip() for line in old_note.splitlines()]
            generated_release = (len(lines) >= 4 and lines[1] == f"* {DANGLING_RELEASE_HEADLINE}"
                                 and all(re.fullmatch(r"\* .+ may dangle when .+ becomes Tombstone\.", line)
                                         for line in lines[2:-1]))
            if request.headline == DANGLING_RELEASE_HEADLINE and generated_release:
                start, end = existing.span("note")
                if resolved:
                    if content[end:index] == newline + indent_text:
                        end = index
                    edits[start, end] = [""]
                elif old_note != note:
                    edits[start, end] = [note]
            continue
        if not resolved:
            edits.setdefault((index, index), []).append(note + newline + indent_text)
    if not edits:
        return False
    updated = content
    for (start, end), notes in sorted(edits.items(), reverse=True):
        updated = updated[:start] + "".join(dict.fromkeys(notes)) + updated[end:]
    if source.read_bytes() != original:
        return False
    source.write_bytes(bom + updated.encode("utf-8"))
    return True


def insert_project_requests(project, proof_request=None):
    changed = False
    for path, module in project.modules.items():
        if not isinstance(module, Compiler):
            continue
        requests = module.dangling_release_requests(include_resolved=True)
        if proof_request is not None and Path(proof_request.path).resolve() == path:
            requests.append(proof_request)
        changed |= insert_oracle_requests(requests, path)
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Statically check Lan and generate C, a module interface and analysis artifacts.")
    parser.add_argument("source", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check without generating files")
    mode.add_argument("-o", "--output", type=Path, help="path to the generated .c file")
    parser.add_argument("--library", action="store_true", help="emit linkable exports without a C main entry point")
    parser.add_argument("--export-prefix", metavar="PREFIX",
                        help="also emit unfingerprinted C aliases PREFIX<name> for exported functions "
                             "so hand-written C can call Lan; the lan_api_* symbols are still emitted")
    parser.add_argument("--module-root", type=Path, help="base directory for portable module identities")
    parser.add_argument("--debug-bounds", action="store_true", help="emit runtime checks for supported unresolved bounds")
    parser.add_argument("--max-steps", "--solver-steps", dest="max_steps", type=int, default=None, metavar="N",
                        help="maximum reasoning steps for polynomial equation solving and SSA verification (default: 2048)")
    parser.add_argument("--insert-oracle", action="store_true",
                        help="insert proof requests and release notes in the affected source modules")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps must be a nonnegative integer")
    project = None
    try:
        source = args.source.resolve()
        compiler = Compiler(source.read_text(encoding="utf-8-sig"), source.name,
                            debug_bounds=args.debug_bounds, max_steps=args.max_steps)
        project = ModuleGraph(source, compiler, library=args.library, module_root=args.module_root,
                             export_prefix=args.export_prefix)
        c, table = project.compile(include_trace=False)
        if args.insert_oracle and insert_project_requests(project):
            compiler = Compiler(source.read_text(encoding="utf-8-sig"), source.name,
                                debug_bounds=args.debug_bounds, max_steps=args.max_steps)
            project = ModuleGraph(source, compiler, library=args.library, module_root=args.module_root,
                             export_prefix=args.export_prefix)
            c, table = project.compile(include_trace=False)
        if not args.check:
            out = args.output or args.source.with_suffix(".c")
            if out.suffix.lower() != ".c":
                parser.error("output must have a .c suffix")
            companion = out.with_suffix(".table")
            theorem, proof = out.with_suffix(".theorem"), out.with_suffix(".proof")
            outputs = {path.resolve() for path in (out, companion, theorem, proof)}
            if len(outputs) != 4 or outputs.intersection(project.modules):
                parser.error("module sources and compiler outputs must be different files")
            out.write_text(c, encoding="utf-8", newline="\n")
            companion.write_text(table, encoding="ascii", newline="\n")
            with theorem.open("w", encoding="ascii", newline="\n") as theorem_stream, \
                    proof.open("w", encoding="ascii", newline="\n") as proof_stream:
                project.write_analysis_artifacts(theorem_stream, proof_stream)
    except (LanError, FloatDomainError, OSError, UnicodeError, RecursionError) as exc:
        message = "input nesting exceeds compiler limits" if isinstance(exc, RecursionError) else str(exc)
        location = exc.module if isinstance(exc, LanError) and exc.module else args.source
        print(f"{location}:{message}", file=sys.stderr)
        request = exc.request if isinstance(exc, OracleRequestError) else None
        if request is not None:
            print(request.comment(), file=sys.stderr)
        if args.insert_oracle and project is not None:
            try:
                insert_project_requests(project, request)
            except (OSError, UnicodeError) as insertion_error:
                print(f"{location}: unable to insert oracle request: {insertion_error}", file=sys.stderr)
        return 1
    return 0
