"""Loop transition verification, oracle checks and loop claims."""

from __future__ import annotations

from dataclasses import field, fields, replace
from itertools import count

from optimization_bridge import replay_loop
from semantic_ir import Proposition, Theorem
from ssa_verify import Obligation
from float_domain import FloatDomain
from value_domain import Domain

from .errors import LanError, ProofBudgetError
from .syntax import Node
from .typesys import (Array, BitField, BitView, BOOL, BorrowArray, BorrowRecord, BorrowScalar, HeapArray,
                      Interface, is_callback_enum, Record, RecordLink, Scalar, Slice, WIDE, ZERO)
from .domain_utils import container, domain_union, physical, plain_integer_parameter, replay_scalar
from .state import PathState
from .interface_types import interface_mismatch, OFFSET_BINARY_OPS
from .c_emit import c_counted_loop, indent


class LoopContractMixin:
    def verify_loop_transition(self, stmt, paths, following):
        self.reject_tautological_loop_condition(stmt.args[0])
        if (indexed_release := self.release_indexed_chain_node(stmt, paths)) is not None:
            return indexed_release
        transition = stmt.args[4] if len(stmt.args) > 4 else None
        postconditions = stmt.args[6] if len(stmt.args) > 6 else ()
        post_groups = {}
        for claim in postconditions:
            guard = claim[0].state_guard
            if guard is not None and self.has_effects(guard):
                guard.token.error("loop postcondition guard must be side-effect free")
            post_groups.setdefault(guard, []).append(claim)
        post_claims = {guard: self.loop_value_claims(annotations, "postcondition")
                       for guard, annotations in post_groups.items()}
        post_values = tuple(claim for group in post_claims.values() for claim in group.values())
        if transition is None:
            count, prefix = self.chain_loop_materialization(stmt, paths, following)
            if prefix:
                exits, code = self.materialize_chain_prefix(stmt, paths, prefix)
            elif not count:
                exits, code = self.analyze_loop(stmt, paths, transition_claims=post_values,
                                                post_guards=post_groups)
            else:
                exits, code = paths, []
                for _ in range(count):
                    exits, body_code = self.block(stmt.args[1], exits)
                    code.extend(["{", *indent(body_code), "}"])
                condition = stmt.args[0]
                stopped = []
                for path in exits:
                    hit, miss, _ = self.condition(condition, path, require_boolean=True)
                    if hit or not miss:
                        raise ProofBudgetError(f"{stmt.token.line}:{stmt.token.col}: cannot prove materialized loop exit")
                    stopped.extend(path.guard("condition=false") for path in miss)
                exits = stopped
                self.trace.append(f"  materialize chain allocations across {count} loop iterations")
            self.check_loop_postconditions(post_claims, exits)
            if exits:
                self.trace_state_claims(postconditions, "verified postcondition")
            return exits, code
        claims = self.loop_value_claims(transition, "transition")
        trusted = len(stmt.args) > 5 and stmt.args[5]
        if trusted:
            for _, _, token, index in claims.values():
                if index in ("link_targets", "heap_status", "alias_binding", "slice_target", "slice_start"):
                    token.error("link, binding and heap state require a verified transition")
        frame = {}
        frames = getattr(self, "transition_frames", None)
        if frames is None:
            frames = self.transition_frames = []
        cutoff = len(self.symbols)
        row_token = object()
        row_tokens = getattr(self, "row_transition_tokens", None)
        if row_tokens is None:
            row_tokens = self.row_transition_tokens = []
        return_count = len(self.return_states)
        frames.append(frame)
        row_tokens.append(row_token)
        try:
            exits, code = self.analyze_loop(
                stmt, paths, transition_claims=(*claims.values(), *post_values),
                post_guards=post_groups)
        finally:
            row_tokens.pop()
            frames.pop()
        coverage, by_path = self.loop_transition_coverage(claims, exits)
        alias_claims = [sym for sym, _, _, index in claims.values()
                        if index in (None, "link_targets")
                        and (sym.uid in self.alias_members or sym.link_access is not None)]
        alias_coverage = None
        alias_by_path = []
        for path in exits:
            current = set()
            for sym in alias_claims:
                targets = self.physical_places(sym, path, allow_null=True)
                if len(targets) == 1 and targets[0].uid and targets[0].projection is None:
                    current.add(targets[0].uid)
            alias_by_path.append(current)
            alias_coverage = current if alias_coverage is None else alias_coverage & current
        alias_coverage = alias_coverage or set()
        def captured_hosts(sym):
            hosts = []
            for path in paths:
                targets = self.physical_places(sym, path, allow_null=True)
                if len(targets) != 1 or not targets[0].uid or targets[0].projection is not None:
                    return None
                hosts.append(targets[0].uid)
            return tuple(hosts)

        field_claims = [sym for sym, _, _, index in claims.values() if index is None
                        and (sym.uid in self.alias_members or self.fixed_projected_field(sym, symbolic=True) is not None)]
        seen_fields, uncovered_fields = set(), set()
        common_fields = None

        def row_element(region):
            owner, _, steps = region
            return ((owner, "array_row", steps[:-1]), Domain.point(steps[-1])) if isinstance(steps[-1], int) else None

        def covers_row_element(region, rows):
            element = row_element(region)
            return element is not None and element[1].within(rows.get(element[0], Domain.of()))

        row_claims = None
        seen_row_writes, uncovered_row_writes = {}, {}
        projected_claims = [claim for claim in claims.values()
                            if claim[0].projection is not None
                            or claim[0].uid in self.alias_members and isinstance(claim[3], (tuple, Node, BitField))]
        for path in exits if projected_claims or field_claims else ():
            current = {}
            for sym, _, _, index in projected_claims:
                physical = self.physical_array_row(self.comparison_place(sym, path), path)
                row = self.array_row_write_key(physical) if physical is not None else None
                if row is None:
                    continue
                if isinstance(index, Node) and sym.ty.packed_bits:
                    _, first, stop = self.packed_transition_window(index, path)
                elif isinstance(index, Node):
                    region = self.slice_transition_region(sym, index, index.token, path, allow_row=True)
                    if region is None:
                        continue
                    _, first, stop = region
                else:
                    first, stop = ((index.offset, index.offset + index.width)
                                   if isinstance(index, BitField) else index)
                current[row] = current.get(row, Domain.of()) | Domain.of((first, stop - 1))
            declared_fields = set()
            for sym in field_claims:
                targets = self.physical_places(sym, path, allow_null=True) if sym.uid in self.alias_members else (sym,)
                field = (self.fixed_projected_field(self.comparison_place(targets[0], path), state=path)
                         if len(targets) == 1 and targets[0].uid else None)
                if field is not None:
                    declared_fields.add((field[0], "field_path", field[1]))
            for region in declared_fields:
                element = row_element(region)
                if element is not None:
                    row, selected = element
                    current[row] = current.get(row, Domain.of()) | selected
            for (token, row), bits in path.row_writes.items():
                if token is not row_token:
                    continue
                if row[1] == "field_path":
                    seen_fields.add(row)
                    if row not in declared_fields and not covers_row_element(row, current):
                        uncovered_fields.add(row)
                elif row[1] in ("bit_row", "array_row"):
                    seen_row_writes[row] = seen_row_writes.get(row, Domain.of()) | bits
                    uncovered = bits - current.get(row, Domain.of())
                    uncovered_row_writes[row] = uncovered_row_writes.get(row, Domain.of()) | uncovered
            common_fields = declared_fields if common_fields is None else common_fields & declared_fields
            row_claims = current if row_claims is None else {
                row: row_claims[row] & current[row] for row in row_claims.keys() & current.keys()
            }
        row_claims = row_claims or {}
        common_fields = common_fields or set()
        early_row_writes = {}
        for path in self.return_states[return_count:]:
            for (token, row), bits in path.row_writes.items():
                if token is row_token:
                    early_row_writes[row] = early_row_writes.get(row, Domain.of()) | bits
        bit_claims, equivalent_claims = {}, {}
        for sym, _, _, index in claims.values():
            if sym.projection is not None:
                continue
            if isinstance(index, BitField):
                bits = Domain.of((index.offset, index.offset + index.width - 1))
                bit_claims[sym.uid] = bit_claims.get(sym.uid, Domain.of()) | bits
                if (sym.uid in self.alias_members and sym.uid not in self.alias_types
                        and (hosts := captured_hosts(sym)) is not None):
                    equivalent_claims[hosts] = equivalent_claims.get(hosts, Domain.of()) | bits
        seen_indices, uncovered_indices = {}, {}
        seen_scalars, uncovered_scalars = set(), set()
        for path, declared, aliases in zip(exits, by_path, alias_by_path):
            for (token, region), written in path.row_writes.items():
                if token is row_token and region[1] in ("scalar_write", "link_write"):
                    seen_scalars.add(region)
                    if region[0].uid not in aliases:
                        uncovered_scalars.add(region)
                if token is row_token and region[1] == "indexed_write":
                    symbol = region[0]
                    seen_indices[region] = seen_indices.get(region, Domain.of()) | written
                    missing = written - declared.get(symbol.uid, Domain.of())
                    uncovered_indices[region] = uncovered_indices.get(region, Domain.of()) | missing
        omitted = []
        for target, indices in frame.items():
            captured = None
            if isinstance(target, tuple):
                symbol, aspect = target[:2]
                if aspect in ("bit_row", "array_row"):
                    written = Domain.of(*indices)
                    common = row_claims.get(target, Domain.of())
                    remaining = (written - common - seen_row_writes.get(target, Domain.of())
                                 | uncovered_row_writes.get(target, Domain.of())
                                 | (early_row_writes.get(target, Domain.of()) - common))
                    if symbol.uid <= cutoff and remaining:
                        omitted.append(symbol)
                    continue
                if aspect == "field_path":
                    covered = ((symbol.uid, aspect, target[2]) in claims or target in common_fields
                               or covers_row_element(target, row_claims)
                               or target in seen_fields and target not in uncovered_fields
                               and target not in early_row_writes)
                    if symbol.uid <= cutoff and not covered:
                        omitted.append(symbol)
                    continue
                if aspect in ("scalar_write", "link_write"):
                    covered = (symbol.uid in claims or symbol.uid in alias_coverage
                               or target in seen_scalars and target not in uncovered_scalars
                               and target not in early_row_writes)
                    if symbol.uid <= cutoff and not covered:
                        omitted.append(symbol)
                    continue
                if aspect == "bits":
                    captured = target[2]
                else:
                    if symbol.uid <= cutoff and (symbol.uid, aspect) not in claims:
                        omitted.append(symbol)
                    continue
            else:
                symbol = target
            if symbol.uid > cutoff or symbol.uid in claims or symbol.uid in alias_coverage:
                continue
            declared = coverage.get(symbol.uid, Domain.of())
            if captured is not None and captured.uid <= cutoff:
                declared |= bit_claims.get(captured.uid, Domain.of())
                if equivalent_claims:
                    declared |= equivalent_claims.get(captured_hosts(captured), Domain.of())
            region = (symbol, "indexed_write", captured)
            if indices is not None and region in seen_indices:
                written = Domain.of(*indices)
                remaining = ((written - declared - seen_indices[region])
                             | (uncovered_indices[region] - declared)
                             | (early_row_writes.get(region, Domain.of()) - declared))
                if not remaining:
                    continue
            if indices is None and isinstance(symbol.ty, Array):
                indices = [(0, symbol.ty.length - 1)]
            if indices is None or not Domain.of(*indices).within(declared):
                omitted.append(symbol)
        omitted.sort(key=lambda symbol: symbol.uid)
        if omitted:
            stmt.token.error(f"loop transition omits modified storage {omitted[0].name!r}")
        self.check_loop_postconditions(post_claims, exits)
        if exits:
            self.trace_state_claims(postconditions, "verified postcondition")
        if trusted:
            narrowed = []
            for path in exits:
                for sym, domain, token, index in claims.values():
                    field = None
                    if sym.projection is not None or sym.link_access is not None:
                        sym = self.comparison_place(sym, path, require_proven=True)
                    if sym.projection is not None:
                        field = self.fixed_projected_field(sym, state=path) if index is None else None
                        if field is None:
                            row = self.physical_array_row(sym, path)
                            if row is None:
                                token.error("trusted transition requires a definite array row")
                            if isinstance(index, tuple):
                                sym = row
                        if isinstance(index, BitField):
                            index = index.offset, index.offset + index.width
                    self.require_live(sym, path, token)
                    if index == "length":
                        self.slice_length(Node(token, "name", (sym.name,)), path)
                    if isinstance(index, BitField):
                        if len(self.physical_places(sym, path)) != 1:
                            token.error("trusted bit field transition requires a definite storage target")
                        if self.proven_presence(domain - index.ty.domain, token):
                            token.error("trusted transition exceeds the storage contract")
                        before = self.read_bits(token, path, sym, index.offset, index.width, index.ty, index.endian).domain
                        compatible = before & domain
                        refined = (self.restrict_bit_read(path, sym, index.offset, index.width, compatible, index.endian)
                                   if compatible else [])
                        if not refined:
                            token.error(f"trusted transition contradicts loop exit: {before} has no consistent value in {domain}")
                        path = refined[0]
                        continue
                    storage = sym.ty.element if isinstance(index, (tuple, Node)) else sym.ty
                    if self.proven_presence(domain - storage.domain, token):
                        token.error("trusted transition exceeds the storage contract")
                    if isinstance(index, (tuple, Node)) and isinstance(sym.ty, Array) and sym.ty.packed_bits:
                        if isinstance(index, Node):
                            sym, first, stop = self.packed_transition_window(index, path)
                            if sym.projection is not None:
                                sym = self.physical_array_row(sym, path)
                                if sym is None:
                                    token.error("trusted transition requires a definite array row")
                            index = first, stop
                        if index[0] < index[1]:
                            path = self.apply_trusted_packed_claim(token, path, sym, index, domain)
                        continue
                    target, region = sym, None
                    if isinstance(index, Node):
                        region = self.slice_transition_region(sym, index, token, path, allow_row=True)
                        if region is None:
                            token.error("trusted slice content transition requires a known backing owner and bounded positions")
                    elif sym.uid in self.alias_members or sym.link_access is not None:
                        targets = self.physical_places(sym, path)
                        if len(targets) != 1:
                            token.error("trusted alias transition requires a definite storage target")
                        target = targets[0]
                        if isinstance(index, tuple):
                            if target.projection is not None:
                                target = self.physical_array_row(target, path)
                                if target is None:
                                    token.error("trusted array alias transition requires a definite array target")
                            region = target, *index
                        elif target.projection is not None:
                            field = self.fixed_projected_field(target, state=path)
                            if field is None:
                                region = self.projected_scalar_region(target)
                                if region is None or region[0].uid not in path.arrays:
                                    token.error("trusted alias transition requires a definite array element")
                    elif isinstance(index, tuple):
                        if isinstance(sym.ty, Array):
                            region = sym, *index
                        else:
                            region = self.slice_transition_region(sym, index, token, path)
                            if region is None:
                                token.error("trusted slice content transition requires a known backing owner and bounded positions")
                    if field is not None:
                        from composite_array import write_path

                        owner, steps = field
                        before = self.storage_value(target, path).domain
                        compatible = before & domain
                        if not compatible:
                            token.error(f"trusted transition contradicts loop exit: {before} has no consistent value in {domain}")
                        contents = self.actual_array(path, owner)
                        projected = tuple(Domain.point(step) if isinstance(step, int) else step for step in steps)
                        updated = write_path(contents, projected, compatible, False)
                        path = path.with_array(owner, updated)
                        continue
                    if region is None:
                        before = path.values[target.uid]
                        path = path.narrow(target, domain)
                    else:
                        owner, start, stop = region
                        contents = self.actual_array(path, owner)
                        for segment in contents.slice(start, stop).segments:
                            before = segment.domain
                            compatible = before & domain
                            if not compatible:
                                path = None
                                break
                            contents = contents.fill(start + segment.start, start + segment.stop, compatible)
                        if path is not None:
                            path = path.with_array(owner, contents)
                    if path is None:
                        token.error(f"trusted transition contradicts loop exit: {before} has no consistent value in {domain}")
                narrowed.append(path)
            exits = narrowed
            self.trace.append("  loop transition trusted; unlisted storage unchanged")
        else:
            self.check_loop_claims(claims, exits, "transition", "exit")
            self.trace.append("  loop transition verified; unlisted storage unchanged")
        if exits:
            self.trace_state_claims(transition, "trusted transition" if trusted else "verified transition")
        exits = [replace(path, row_writes={key: bits for key, bits in path.row_writes.items()
                                           if key[0] is not row_token}) for path in exits]
        return exits, [*code, *self.oracle_checks(claims, exits[0])] if exits else code

    def trace_state_claims(self, annotations, label):
        if annotations:
            self.trace.append(f"  {label}")
        for subject, value in annotations:
            guard = subject.state_guard
            prefix = f"when ({guard.source_text}) " if guard is not None else ""
            self.trace.append(f"    {prefix}state({subject.source_text}) := {value.source_text};")

    def check_loop_postconditions(self, groups, exits):
        for guard, claims in groups.items():
            if guard is None:
                self.check_loop_claims(claims, exits, "postcondition", "exit")
                continue
            debug = self.debug_bounds
            self.debug_bounds = False
            try:
                for state in exits:
                    yes, _, _ = self.condition(guard, replace(state))
                    self.check_loop_claims(claims, yes, "postcondition", "guarded exit")
            finally:
                self.debug_bounds = debug

    def oracle_checks(self, claims, state):
        checks = []
        for sym, domain, token, index in claims.values():
            if index in ("link_targets", "heap_status", "alias_binding", "slice_target", "slice_start"):
                continue
            if sym.projection is not None or sym.link_access is not None:
                sym = self.comparison_place(sym, state)
            guard = self.membership(token, domain if isinstance(domain, FloatDomain) else domain & WIDE)
            check = self.helper("oracle_check", None)
            projection = sym.ty.projection if isinstance(sym.ty, Slice) else None
            if index is None:
                checks.append(f"{check}({guard}({sym.c}));")
            elif index == "length":
                length = projection.length.code if projection is not None else f"{sym.c}.len"
                checks.append(f"{check}({guard}({length}));")
            elif isinstance(index, BitField):
                value = self.bit_read_code(sym, index.offset, index.width, index.ty, index.endian)
                if index.ty.floating:
                    view = self.helper("float_view", None, (f"uint{index.width}_t", index.ty.c))
                    value = f"{view}({value})"
                checks.append(f"{check}({guard}({value}));")
            elif isinstance(index, Node) and isinstance(sym.ty, Array) and sym.ty.packed_bits:
                bits, start, width, setup = self.bit_region(index, state, capture=True, require_proven=True)
                read = self.helper("bit_read", None)
                first = f"(size_t)({start.code})"
                end = f"({first} + (size_t)({width.code}))"
                body = [f"{check}({guard}({read}({bits.c}, lan_claim_i)));"]
                checks.extend(["{", *indent(setup + c_counted_loop("lan_claim_i", first, end, body)), "}"])
            elif isinstance(index, Node):
                span = self.ordinary_span(index, state)
                checks.extend(["{", f"    lan_slice lan_claim_span = {span.c_value()};",
                               *indent(c_counted_loop("lan_claim_i", 0, "lan_claim_span.len",
                                   [f"{check}({guard}(((const {span.element.c} *)lan_claim_span.data)[lan_claim_i]));"])),
                               "}"])
            else:
                start, stop = index
                if isinstance(sym.ty, Array) and sym.ty.packed_bits:
                    read = self.helper("bit_read", None)
                    checks.extend(c_counted_loop("lan_claim_i", start, stop,
                                                 [f"{check}({guard}({read}({sym.c}, lan_claim_i)));"]))
                    continue
                if projection is not None:
                    data = f"({projection.c_data()})"
                elif isinstance(sym.ty, (Slice, HeapArray)):
                    data = f"((const {sym.ty.element.c} *){sym.c}.data)"
                else:
                    data = f"({sym.c})"
                if stop == start + 1:
                    checks.append(f"{check}({guard}({data}[{start}]));")
                else:
                    length = projection.length.code if projection is not None else f"{sym.c}.len"
                    end = length if stop is None else str(stop)
                    checks.extend(c_counted_loop("lan_claim_i", start, end,
                                                 [f"{check}({guard}({data}[lan_claim_i]));"]))
        return checks

    def loop_row_step(self, step):
        if isinstance(step, str):
            return step
        if isinstance(step, Domain):
            return step.lo if step.lo == step.hi else None
        if self.literal_endpoint(step):
            return self.const_int(step, ())
        if step.kind == "name":
            return "name", step.args[0]
        if (step.kind == "unary" and step.args[0] in ("+", "-", "~")
                or step.kind == "binary" and step.args[0] in OFFSET_BINARY_OPS):
            operands = tuple(self.loop_row_step(arg) for arg in step.args[1:])
            if all(operand is not None for operand in operands):
                return step.kind, step.args[0], operands
        return None

    def loop_value_claims(self, annotations, label):
        claims = {}
        for subject, annotation in annotations:
            token = subject.token
            index = None
            base = self.place(subject.args[0]) if subject.kind == "member" else None
            member = subject.args[1].text if base is not None else None
            alias = self.alias_types.get(base.uid) if base is not None else None
            if (label in ("transition", "postcondition") and member == "target"
                    and isinstance(base.ty, Interface) and base.uid in self.alias_members
                    and member not in dict(base.ty.fields)):
                key = base.uid, "alias_binding"
                if key in claims:
                    token.error(f"duplicate loop {label} variable")
                targets = self.loop_target_set(
                    annotation, lambda owner: isinstance(owner.ty, Record)
                    and interface_mismatch(owner.ty.fields, owner.readonly, base.ty) is None)
                if self.null_symbol in targets:
                    annotation.token.error("interface target state requires backing storage")
                claims[key] = (base, targets, token, "alias_binding")
                continue
            if (label in ("transition", "postcondition") and member == "target"
                    and isinstance(alias, (BorrowScalar, BorrowArray, BorrowRecord))
                    and (not isinstance(alias, BorrowRecord) or member not in dict(alias.record.fields))):
                key = base.uid, "alias_binding"
                if key in claims:
                    token.error(f"duplicate loop {label} variable")
                targets = self.loop_target_set(annotation, lambda owner: owner.ty == base.ty)
                claims[key] = (base, targets, token, "alias_binding")
                continue
            if (label in ("transition", "postcondition") and base is not None and isinstance(base.ty, Slice)
                    and subject.args[0].kind == "name" and member in ("target", "start")):
                sym = base
                index = "slice_target" if member == "target" else "slice_start"
                if index == "slice_target":
                    key = sym.uid, index
                    if key in claims:
                        token.error(f"duplicate loop {label} variable")
                    owners = self.loop_target_set(annotation, lambda owner: isinstance(owner.ty, (Array, HeapArray))
                                                  and owner.ty.element == sym.ty.element)
                    if self.null_symbol in owners:
                        annotation.token.error("slice target state requires backing storage")
                    claims[key] = (sym, owners, token, index)
                    continue
                storage = self.types["uint64"]
            elif (base is not None and isinstance(base.ty, (Slice, HeapArray))
                    and subject.args[1].text == "len" and subject.args[0].kind == "name"):
                if label != "invariant" and not isinstance(base.ty, Slice):
                    token.error("length transitions require a borrowed slice descriptor")
                if label != "invariant" and base.ty.projection is not None:
                    token.error("length transitions require a rebindable slice descriptor")
                sym, storage, index = base, self.types["uint64"], "length"
            elif subject.kind == "member" and base is not None and isinstance(base.ty, BitView):
                sym, index = self.bit_field_target(subject)
                storage = index.ty
            elif subject.kind in ("index", "slice"):
                base = subject.args[0]
                sym = self.place(base)
                fixed = sym is not None and isinstance(sym.ty, Array)
                packed = fixed and sym.ty.packed_bits
                if packed and sym.bit_overlay:
                    token.error(f"packed bit loop {label} requires ordinary packed storage")
                dynamic = sym is not None and base.kind == "name" and isinstance(sym.ty, (Slice, HeapArray))
                if not (fixed or dynamic) or not isinstance(sym.ty.element, Scalar):
                    token.error(f"indexed loop {label} requires a direct scalar array or runtime slice")
                first = subject.args[1]
                try:
                    start = self.const_int(first, ()) if first is not None else 0
                    if subject.kind == "index":
                        stop = start + 1
                    else:
                        last, inclusive = subject.args[2:]
                        stop = self.const_int(last, ()) + inclusive if last is not None else (sym.ty.length if fixed else None)
                except LanError:
                    last, inclusive = subject.args[2:] if subject.kind == "slice" else (first, True)
                    if any(endpoint is not None and self.has_effects(endpoint, allow_debug_checks=True)
                           for endpoint in (first, last)):
                        token.error(f"loop {label} endpoints must be side-effect free")
                    index = Node(token, "slice", (base, first, last, inclusive))
                else:
                    if start < 0 or stop is not None and (start >= stop or fixed and stop > sym.ty.length):
                        token.error(f"loop {label} array index or interval is out of bounds or empty")
                    index = BitField(start, 1, sym.ty.element, "le") if packed and subject.kind == "index" else (start, stop)
                storage = sym.ty.element
            else:
                sym = self.place(subject)
                storage = sym.ty if sym is not None else None
            if label in ("transition", "postcondition") and index is None and sym is not None:
                if isinstance(storage, RecordLink):
                    targets = self.loop_target_set(
                        annotation, lambda owner: isinstance(owner.ty, Record)
                        and owner.ty.identity == storage.identity)
                    key = sym if sym.link_access is not None else sym.uid
                    if key in claims:
                        token.error(f"duplicate loop {label} variable")
                    claims[key] = (sym, targets, token, "link_targets")
                    continue
                if (subject.kind == "name" and annotation.kind == "type_name"
                        and annotation.args[0] in ("Alive", "Tombstone")):
                    key = sym.uid, "heap_status"
                    if key in claims:
                        token.error(f"duplicate loop {label} variable")
                    claims[key] = (sym, annotation.args[0], token, "heap_status")
                    continue
            projection = sym.projection if sym is not None else None
            row_steps = None
            if (projection is not None and isinstance(sym.ty, Array)
                    and projection.root.uid not in self.alias_members):
                packed_row = sym.ty.packed_bits and isinstance(projection.root.ty, (Array, Slice))
                ordinary_row = (not sym.ty.packed_bits and isinstance(index, (tuple, Node))
                                and isinstance(projection.root.ty, (Array, Slice)))
                if packed_row or ordinary_row:
                    steps = tuple(self.loop_row_step(step) for step in projection.steps)
                    if all(step is not None for step in steps):
                        row_steps = steps
            field = self.fixed_projected_field(sym, symbolic=True) if projection is not None and index is None else None
            if (not isinstance(storage, Scalar) or (projection is not None and row_steps is None and field is None)
                    or sym.uid in self.alias_members and index is not None and not isinstance(index, BitField)
                    and not isinstance(sym.ty, Array)):
                token.error(f"loop {label} requires a direct scalar variable or record field")
            if is_callback_enum(storage.enum):
                token.error(f"loop {label} cannot constrain function values numerically")
            key = sym.uid if index is None else (sym.uid, index)
            if row_steps is not None:
                key = sym.uid, row_steps, index
            elif field is not None:
                key = field[0].uid, "field_path", field[1]
            elif sym.link_access is not None:
                key = sym, index
            if key in claims:
                token.error(f"duplicate loop {label} variable")
            if storage.floating:
                domain = self.float_set(annotation, storage.bits)
            else:
                ty = self.resolve(annotation)
                if not isinstance(ty, Scalar) or ty.floating:
                    annotation.token.error(f"loop {label} requires a matching scalar value set")
                domain = ty.domain
            claims[key] = (sym, domain, token, index)
        return claims

    def loop_target_set(self, annotation, matches):
        if annotation.kind != "set" or not annotation.args[0]:
            annotation.token.error("target state requires a nonempty set of storage names or null")
        targets = set()
        for target in annotation.args[0]:
            if target.kind == "null":
                targets.add(self.null_symbol)
                continue
            if target.kind != "name":
                target.token.error("target state requires a storage name or null")
            owner = self.lookup(target)
            if not matches(owner) or owner.uid in self.alias_members or owner.projection is not None:
                target.token.error("target state requires a concrete name of matching storage type")
            targets.add(owner)
        return frozenset(targets)

    @staticmethod
    def packed_claim_mask(index):
        start, stop = index
        return ((1 << (stop - start)) - 1) << start

    def packed_claim_contents(self, state, sym, index):
        if sym.ty.length > 4096:
            return BOOL
        window = self.packed_claim_mask(index)
        mask, data = self.storage_bits(state, sym)
        if mask & window != window:
            return BOOL
        selected = data & window
        return ZERO if not selected else Domain.point(1) if selected == window else BOOL

    def apply_trusted_packed_claim(self, token, state, sym, index, domain):
        allow_zero = bool(domain & ZERO)
        allow_one = bool(domain & Domain.point(1))
        if allow_zero and allow_one:
            return state
        targets = self.physical_places(sym, state)
        if len(targets) != 1 or (targets[0].projection is not None and self.array_row_write_key(targets[0]) is None):
            token.error("trusted packed bit transition requires a definite storage target")
        if sym.ty.length > 4096:
            token.error("trusted packed bit transition exceeds the bit tracking budget")
        window = self.packed_claim_mask(index)
        mask, data = self.storage_bits(state, sym)
        if not (allow_zero or allow_one):
            token.error("trusted transition contradicts loop exit: no bit value is allowed")
        expected = window if allow_one else 0
        if (data ^ expected) & mask & window:
            token.error("trusted transition contradicts loop exit: a known bit is outside the claimed set")
        return self.store_bits(state, sym, (mask | window, (data & ~window) | expected), record_write=False)

    def split_slice_windows(self, sym, token, path):
        uid = sym.uid
        positions = path.regions.get(uid, ())
        if len(positions) == 1:
            span = self.ordinary_span(Node(token, "name", (sym.name,)), path)
            candidates = self.split_slice_start(span, path)
            if candidates and (len(candidates) != 1 or candidates[0] is not path):
                result = []
                for candidate in candidates:
                    resolved = self.refine_span_position(candidate, span)
                    result.append(replace(candidate, regions=candidate.regions | {uid: resolved.positions()}))
                return result
        owners = frozenset(owner for owner, _ in positions)
        windows = []
        if owners and owners == path.borrows.get(uid):
            for owner, offset in positions:
                if offset is None or not offset or not isinstance(offset.lo, int) or not isinstance(offset.hi, int):
                    break
                starts = self.bounded_window_values(offset, 8 - len(windows))
                if not starts:
                    break
                windows.extend((owner, Domain.point(start)) for start in starts)
            else:
                if len(windows) > 1:
                    starts = {key: value for key, value in path.starts.items() if key != uid}
                    return [replace(path, borrows=path.borrows | {uid: frozenset((owner,))},
                                regions=path.regions | {uid: ((owner, offset),)}, starts=starts)
                            for owner, offset in windows]
        return [path]

    @staticmethod
    def loop_link_projections(sym):
        while sym is not None:
            if sym.projection is not None:
                yield sym
                sym = sym.projection.root
            else:
                sym = sym.link_access[0] if sym.link_access is not None else None

    def loop_field_indices(self, fields, endpoints=()):
        field_indices = {}
        pending = [step for sym in fields for place in self.loop_link_projections(sym)
                   for step in place.projection.steps if isinstance(step, Node)]
        pending.extend(endpoint for endpoint in endpoints if endpoint is not None)
        while pending:
            step = pending.pop()
            if step.kind == "name":
                variable = self.lookup(step)
                if plain_integer_parameter(variable.ty) and variable.uid not in self.alias_members:
                    field_indices[variable.uid] = variable
            else:
                pending.extend(part for part in step.args if isinstance(part, Node))
        return field_indices

    def split_loop_field_indices(self, field_indices, paths):
        for uid, variable in sorted(field_indices.items()):
            expanded = []
            for path in paths:
                values = self.bounded_window_values(path.values[uid], 8)
                if values and len(values) > 1:
                    expanded.extend(candidate for value in values
                                    if (candidate := path.narrow(variable, Domain.point(value))) is not None)
                else:
                    expanded.append(path)
            if expanded and len(expanded) <= self.PATH_LIMIT:
                paths = expanded
        return paths

    def check_loop_claims(self, claims, states, label, edge):
        for state in states:
            for sym, domain, token, index in claims.values():
                candidates = [state]
                if isinstance(sym.ty, Slice) and isinstance(index, (tuple, Node)):
                    expanded = self.split_slice_windows(sym, token, state)
                    if len(expanded) <= self.PATH_LIMIT:
                        candidates = expanded
                if sym.link_access is not None:
                    for place in self.loop_link_projections(sym):
                        root = place.projection.root
                        if isinstance(root.ty, Slice):
                            expanded = [part for candidate in candidates
                                        for part in self.split_slice_windows(root, token, candidate)]
                            if len(expanded) <= self.PATH_LIMIT:
                                candidates = expanded
                    endpoints = index.args[1:3] if isinstance(index, Node) else ()
                    candidates = self.split_loop_field_indices(self.loop_field_indices((sym,), endpoints), candidates)
                elif index is None and sym.projection is not None:
                    field = self.fixed_projected_field(sym, symbolic=True)
                    if field is not None:
                        if isinstance(field[0].ty, Slice):
                            candidates = self.split_slice_windows(field[0], token, state)
                        candidates = self.split_loop_field_indices(self.loop_field_indices((sym,)), candidates)
                elif isinstance(index, (tuple, Node)) and sym.projection is not None:
                    root = sym.projection.root
                    if isinstance(root.ty, Slice):
                        candidates = self.split_slice_windows(root, token, state)
                    endpoints = index.args[1:3] if isinstance(index, Node) else ()
                    indices = self.loop_field_indices((sym,), endpoints)
                    candidates = self.split_loop_field_indices(indices, candidates)
                for candidate in candidates:
                    self.check_loop_claim(sym, domain, token, index, candidate, label, edge)

    def check_loop_claim(self, sym, domain, token, index, state, label, edge):
        if index == "heap_status":
            if state.heap.get(sym.uid) != frozenset((domain,)):
                token.error(f"loop {label} fails at {edge}: cannot prove {sym.name} is {domain}")
            return
        if index == "alias_binding":
            actual = state.borrows.get(sym.uid)
            if not actual or not actual.issubset(domain):
                token.error(f"loop {label} fails at {edge}: unexpected targets for {sym.name}")
            return
        if index in ("slice_target", "slice_start"):
            node = Node(token, "name", (sym.name,))
            span = self.refine_span_position(state, self.ordinary_span(node, state))
            positions = span.positions()
            if index == "slice_target":
                if not positions or not {owner for owner, _ in positions}.issubset(domain):
                    token.error(f"loop {label} fails at {edge}: unexpected backing storage for {sym.name}")
            elif (not positions or any(offset is None or self.proven_presence(offset - domain, token)
                                       for _, offset in positions)):
                token.error(f"loop {label} fails at {edge}: unexpected start offset for {sym.name}")
            return
        if isinstance(index, Node):
            self.prove_loop_endpoint_reads(index, state)
        if sym.projection is not None or sym.link_access is not None:
            sym = self.comparison_place(sym, state, require_proven=True)
        self.require_live(sym, state, token)
        if index == "link_targets":
            actual = self.link_targets(sym, state)
            if not actual or not actual.issubset(domain):
                token.error(f"loop {label} fails at {edge}: unexpected targets for {sym.name}")
            return
        if index == "length":
            base = Node(token, "name", (sym.name,))
            actual = self.slice_length(base, state).domain
        elif index is None:
            actual = self.storage_value(sym, state).domain
        elif isinstance(index, Node) and isinstance(sym.ty, Array) and sym.ty.packed_bits:
            bits, start, width, _ = self.bit_region(index, state, require_proven=True)
            windows = [(offset, size) for offset, size in self.bit_window_candidates(start, width)
                       if offset + size <= bits.ty.length]
            if windows:
                actual = Domain.union(self.packed_claim_contents(state, bits, (offset, offset + size))
                                      for offset, size in windows)
            else:
                end = min(bits.ty.length, start.domain.hi + width.domain.hi)
                actual = self.packed_claim_contents(state, bits, (start.domain.lo, end))
        elif isinstance(index, Node):
            span = self.refine_span_position(state, self.ordinary_span(index, state))
            if self.debug_bounds and not span.bounds_proven:
                token.error(f"loop {label} fails at {edge}: cannot prove slice bounds for {sym.name}")
            if span.length.domain.hi == 0:
                return
            indices = Domain.of((0, span.length.domain.hi - 1))
            actual = self.read_span_contract_contents(token, state, span, indices, domain)
        elif isinstance(index, BitField):
            actual = self.read_bits(token, state, sym, index.offset, index.width, index.ty, index.endian).domain
        elif isinstance(sym.ty, Array) and sym.ty.packed_bits:
            actual = self.packed_claim_contents(state, sym, index)
        elif isinstance(sym.ty, (Slice, HeapArray)):
            base = Node(token, "name", (sym.name,))
            span = self.refine_span_position(state, self.ordinary_span(base, state))
            start, stop = index
            if (start if stop is None else stop) > span.length.domain.lo:
                token.error(f"loop {label} fails at {edge}: cannot prove slice bounds for {sym.name}")
            end = span.length.domain.hi if stop is None else stop
            if start == end:
                return
            indices = Domain.of((start, end - 1))
            actual = self.read_span_contract_contents(token, state, span, indices, domain)
        else:
            actual = domain_union(self.actual_array(state, target).read_range(*index)
                                  for target in self.physical_places(sym, state))
        if self.proven_presence(actual - domain, token):
            token.error(f"loop {label} fails at {edge}: {actual} is outside {domain}")

    def record_loop_exit_theorem(self, stmt, exits):
        condition = stmt.args[0]
        anchor = getattr(stmt, "semantic_anchor", None)
        condition_anchor = getattr(condition, "semantic_anchor", None)
        if anchor is None or condition_anchor is None or not exits:
            return
        if not all(path.predicates.get(self.predicate_key(condition, path)) is False for path in exits):
            self.loop_exit_facts[anchor] = None
            return
        self.loop_exit_facts.setdefault(anchor, Theorem(Proposition("condition_false_at_exit", (condition_anchor,)),
                                                       anchor, "loop_exit", (stmt.token.line, stmt.token.col),
                                                       function=self.current))

    def loop_back_cannot_exit(self, condition, paths, false_paths):
        if not paths or self.has_effects(condition):
            return False
        # False paths already describe the values at the condition test.
        # Compare them with the back edges without evaluating the guard again.
        for path in paths:
            for stopped in false_paths:
                self.spend(condition.token)
                if not any(uid in path.values and not (domain & path.values[uid])
                           for uid, domain in stopped.values.items()):
                    return False
        return True

    def scalar_replay_signature(self, name):
        if name in self.functions:
            contract = self.functions[name][1]
        else:
            binding = self.externs.get(name) or self.imported_functions.get(name)
            if binding is None:
                raise ValueError("indirect call outside scalar replay")
            contract = binding.contract
        if (not all(replay_scalar(param) for param in contract.params)
                or contract.result is not None and not replay_scalar(contract.result)):
            raise ValueError("call outside scalar replay")
        result = contract.result
        return (tuple(param.boolean for param in contract.params),
                (result.boolean, result.domain.lo, result.domain.hi) if result is not None else None)

    def loop_replay(self, condition, body, heads):
        def hull(uid):
            domains = [state.values.get(uid) for state in heads]
            if all(domains):
                return min(domain.lo for domain in domains), max(domain.hi for domain in domains)
            return None

        inputs, arrays = {}, {}
        visible = {name: sym for scope in self.scope for name, sym in scope.items()}
        for name, sym in visible.items():
            ty = sym.ty
            if (sym.uid in self.alias_types or sym.uid in self.alias_members or sym.borrowed
                    or sym.link_access is not None or sym.projection is not None
                    or sym.bit_overlay or sym.c_place):
                continue
            if replay_scalar(ty) and not sym.heap_owner:
                bounds = hull(sym.uid)
                if bounds is not None:
                    inputs[name] = (ty.boolean, *bounds)
            elif (isinstance(ty, (Array, Slice, HeapArray)) and replay_scalar(ty.element)
                  and not (isinstance(ty, Array) and ty.packed_bits)
                  and not (isinstance(ty, Slice) and ty.projection is not None)):
                bounds = (ty.length, ty.length) if isinstance(ty, Array) else hull(sym.uid)
                if bounds is not None:
                    element = ty.element
                    # Runtime content is unknown, including changes through
                    # other views. Use the physical range for each load.
                    domain = BOOL if element.boolean else container(element.bits, element.signed)
                    arrays[name] = ((element.boolean, domain.lo, domain.hi), *bounds)
        scalar_types = {name: (ty.boolean, ty.domain.lo, ty.domain.hi)
                        for name, ty in self.types.items() if replay_scalar(ty)}
        return replay_loop(inputs, arrays, condition, body, scalar_types, self.scalar_replay_signature)

    def reject_tautological_loop_condition(self, condition):
        if (condition.kind == "boolean" and condition.args[0] is True) or self.has_effects(condition):
            return
        # Use declared domains rather than just the first iteration's state.
        declared = {sym.uid: sym.ty.domain for scope in self.scope for sym in scope.values()
                    if replay_scalar(sym.ty)}
        replay = self.loop_replay(condition, (), (PathState(declared),))
        if replay is not None:
            program = replay.program
            goal = Obligation("while condition", program.arena.expr(True), program.flow.guard)
            if program.verify(goal, use_smt=False).status == "proved":
                condition.token.error("while condition is always true")
