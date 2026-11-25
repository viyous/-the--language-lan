"""Function ordering, recursive summaries, termination and result inference."""

from __future__ import annotations

import copy
from dataclasses import replace
from itertools import count

from optimization_bridge import prove_recursive_group
from size_change import Change, Transition, check_termination, strongly_connected_components
from float_domain import FloatDomain

from .errors import LanError
from .syntax import Node
from .typesys import (BOOL, BorrowArray, BorrowRecord, BorrowScalar, CopyArray, CopyRecord, HeapArray,
                      HeapObject, HeapRecord, Scalar, Slice, unqualified)
from .domain_utils import container, domain_union, replay_scalar

# Late-bound names, filled in by lan_compiler/__init__.py once every module is loaded
# (importing them here directly would create an import cycle).
_LATE_BOUND = ("Compiler",)
Compiler = None


class RecursionMixin:
    def function_order(self):
        callback_targets = set(self.callable_options)
        for name, (node, _) in self.functions.items():
            pending = list(node.args[3])
            while pending:
                item = pending.pop()
                if isinstance(item, Node):
                    if item.kind == "call" and item.args[0].kind == "name":
                        target = item.args[0].args[0]
                        if target in self.functions:
                            self.calls[name].add(target)
                    elif item.kind in ("callback_ref", "callable_type"):
                        targets = item.args if item.kind == "callback_ref" else item.args[0]
                        callback_targets.update(target.text for target in targets if target.text in self.functions)
                    pending.extend(item.args)
                elif isinstance(item, (list, tuple)):
                    pending.extend(item)
        components = strongly_connected_components(
            self.functions, ((name, target) for name, targets in self.calls.items() for target in sorted(targets)))
        groups = {}
        for name in self.functions:
            groups.setdefault(components[name], []).append(name)
        done, order = set(), []

        def visit(group):
            if group in done:
                return
            names = groups[group]
            for name in names:
                for child in sorted(self.calls[name]):
                    if components[child] != group:
                        visit(components[child])
            if len(names) > 1 or names[0] in self.calls[names[0]]:
                self.prepare_recursive_group(names)
            done.add(group)
            order.extend(names)

        for target in sorted(callback_targets):
            visit(components[target])
        for group in groups:
            visit(group)
        return order

    def check_callback_cycles(self):
        if not self.callback_edges:
            return
        edges = ((name, target) for name in self.functions
                 for target in self.calls[name] | self.callback_edges.get(name, {}).keys())
        components = strongly_connected_components(self.functions, edges)
        for name, targets in self.callback_edges.items():
            for target, token in targets.items():
                if components[name] == components[target]:
                    token.error("indirect recursion through a callback is not supported")

    def seed_recursive_summaries(self, name):
        """Fill in the write summaries of a whole recursive group."""
        for member, summary in self.recursive_summary_fixed_point(self.recursive_groups[name]).items():
            self.recursive_summaries[member] = summary
            self.mutations[member] = summary
            labels = ", ".join(f"param#{index}." + ".".join(path) if path else f"param#{index}"
                               for index, path, _ in summary)
            self.trace.append(f"  recursive writes {member}: {labels or 'none'}")

    def recursive_summary_fixed_point(self, names):
        """Least fixed point of "which parameter paths each member writes".

        A member's summary is the union of the writes in its own body and the
        writes its callees perform through the forwarded arguments. Members of
        one group call each other, so the summaries are mutually dependent and
        cannot be computed bottom-up. The domain is finite because every path
        follows a declared record shape, so growing the summaries from the
        empty set terminates at the least fixed point.
        """
        declared = {name: set() for name in names}
        for _ in range(64):
            grown = False
            for name in names:
                fresh = self.scan_group_writes(name, declared)
                if not fresh <= declared[name]:
                    declared[name] |= fresh
                    grown = True
            if not grown:
                return {name: tuple(sorted((index, path, None) for index, path in declared[name]))
                        for name in names}
        # The iteration did not settle: fall back to the declared shape of every
        # shared parameter so the exported summaries stay an over-approximation.
        return {name: self.whole_parameter_summary(self.functions[name][1]) for name in names}

    @staticmethod
    def whole_parameter_summary(contract):
        entries = []
        for index, ty in enumerate(contract.params):
            plain = unqualified(ty)
            if isinstance(plain, Slice):
                if not plain.readonly and not plain.copy:
                    entries.append((index, (), None))
            elif isinstance(plain, (BorrowScalar, BorrowArray)) and not plain.readonly:
                entries.append((index, (), None))
            elif isinstance(plain, BorrowRecord) and not plain.readonly:
                entries.extend((index, path, None) for path, _ in Compiler.type_leaves(plain.record))
        return tuple(entries)

    def scan_group_writes(self, name, summaries):
        """Collect the (parameter index, path) pairs a recursive body writes.

        This runs before the body is analysed, so it walks the syntax tree and
        consults the summaries of already known callees. Anything it cannot
        resolve widens to the whole parameter, keeping the result a sound over
        approximation to feed back into the fixed point.
        """
        node, contract = self.functions[name]
        _, _, params, body = node.args
        scope, writes = {}, set()
        for index, (param, ty) in enumerate(zip(params, contract.params)):
            plain = unqualified(ty)
            if isinstance(plain, Slice):
                shared = not plain.readonly and not plain.copy
            elif isinstance(plain, (BorrowRecord, BorrowScalar, BorrowArray)):
                shared = not plain.readonly
            else:
                shared = False
            if shared:
                scope[param.text] = (index, ())

        def place(expr, scope):
            if expr.kind == "name":
                return scope.get(expr.args[0])
            if expr.kind in ("index", "slice"):
                return place(expr.args[0], scope)
            if expr.kind == "member":
                parent = place(expr.args[0], scope)
                return None if parent is None else (parent[0], parent[1] + (expr.args[1].text,))
            return None

        def summary_of(callee):
            if callee in summaries:
                return summaries[callee]
            if callee in self.mutations:
                return self.mutations[callee]
            binding = self.externs.get(callee) or self.imported_functions.get(callee)
            if binding is None:
                return None
            return (binding.mutations if binding.mutations is not None
                    else self.unknown_mutations(binding.contract))

        def forward(target, path):
            names = []
            for part in path:
                if not isinstance(part, str):
                    return target[0], target[1] + tuple(names)
                if part.startswith("@"):
                    return target[0], ()
                names.append(part)
            return target[0], target[1] + tuple(names)

        def expression(value, scope):
            if isinstance(value, (tuple, list)):
                for child in value:
                    expression(child, scope)
                return
            if not isinstance(value, Node):
                return
            if value.kind == "call" and value.args[0].kind == "name":
                callee, = value.args[0].args
                arguments = value.args[1]
                summary = summary_of(callee)
                for entry in summary or ():
                    index, path = entry[0], entry[1]
                    if index >= len(arguments) or arguments[index].kind == "null":
                        continue
                    target = place(arguments[index], scope)
                    if target is not None:
                        writes.add(forward(target, path))
                if summary is None:
                    for argument in arguments:
                        target = place(argument, scope)
                        if target is not None:
                            writes.add((target[0], ()))
            for child in value.args:
                expression(child, scope)

        def block(statements, outer):
            scope = dict(outer)
            for statement in statements:
                k, a = statement.kind, statement.args
                if k == "define":
                    target, _, value = a
                    expression(value, scope)
                    scope[target.text] = place(value, scope) if value is not None else None
                elif k == "assign":
                    target, _, value = a
                    expression(target, scope)
                    expression(value, scope)
                    located = place(target, scope)
                    if located is not None:
                        writes.add(located)
                elif k == "if":
                    expression(a[0], scope)
                    block(a[1], scope)
                    block(a[2], scope)
                elif k == "while":
                    expression(a[0], scope)
                    block(a[1], scope)
                elif k == "interation":
                    name, initial, _, count, step, body = a
                    for operand in (initial, count, step):
                        expression(operand, scope)
                    block(body, scope | {name.text: None})
                elif k != "continue":
                    expression(statement, scope)

        block(body, scope)
        return writes

    def prepare_recursive_group(self, names):
        def recursive_parameter_ok(ty):
            plain = unqualified(ty)
            if getattr(plain, "nullable", False):
                return False
            if isinstance(plain, (Scalar, BorrowScalar, Slice, BorrowArray, CopyArray,
                                  HeapRecord, HeapObject, HeapArray)):
                return True
            if isinstance(plain, (BorrowRecord, CopyRecord)):
                record = plain.record
                if Compiler.contains_interface_storage(record):
                    return False
                if not Compiler.contains_record_links(record):
                    return True
                return isinstance(plain, BorrowRecord) and plain.readonly and (plain.graph or plain.chain_limit is True)
            return False

        for name in names:
            node, contract = self.functions[name]
            token, annotation, params, _ = node.args
            if (not all(recursive_parameter_ok(ty) for ty in contract.params)
                    or contract.result is not None
                    and not isinstance(contract.result, (Scalar, HeapRecord, HeapObject, HeapArray))
                    or contract.identity_property or contract.idempotent_property
                    or contract.commutative_property or contract.diagonal_property):
                token.error("recursive functions require by-value scalars, owning heap or supported borrowed parameters "
                            "and a scalar, owning heap or void result contract; "
                            "linked record parameters require a read-only graph or chain")
            if len({param.text for param in params}) != len(params):
                token.error("duplicate recursive function parameter")

        # Members may call each other, so their write summaries are mutually
        # dependent. They are seeded with the least fixed point when the first
        # member is analysed, once every callee outside the group has a summary.
        for name in names:
            self.recursive_groups[name] = tuple(names)
        self.recursive_functions.update(names)
        if any(self.functions[name][1].result is None for name in names):
            # A result contract that is not written down is inferred from the
            # bodies once every callee outside the group has been analysed; the
            # termination proof needs the inferred result, so it waits as well.
            self.pending_recursive_results.update(names)
            return
        self.prove_recursive_termination(names)

    def prove_recursive_termination(self, names):
        def shape(ty):
            return ty.boolean, ty.domain.lo, ty.domain.hi

        def replay_signature(contract):
            return (all(replay_scalar(ty) for ty in contract.params)
                    and (contract.result is None or replay_scalar(contract.result)))

        if not all(replay_signature(self.functions[name][1]) for name in names):
            if self.prove_recursive_chain_descent(names):
                self.trace.append(f"sct recursive chain descent proved: {', '.join(names)}")
            return
        group = {}
        for name in names:
            node, contract = self.functions[name]
            _, _, params, body = node.args
            inputs = {param.text: shape(ty) for param, ty in zip(params, contract.params)}
            group[name] = inputs, shape(contract.result) if contract.result is not None else None, body
        helpers = {}
        for name in sorted(set().union(*(self.calls[name] for name in names)) - group.keys()):
            node, contract = self.functions[name]
            annotation = node.args[1]
            if replay_signature(contract) and (annotation.kind != "contract" or annotation.args[1] is not None):
                helpers[name] = tuple(shape(ty) for ty in contract.params), (
                    shape(contract.result) if contract.result is not None else None)
        if prove_recursive_group(group, helpers):
            self.trace.append(f"sct recursive termination proved: {', '.join(names)}")

    def prove_recursive_chain_descent(self, names):
        # On a verified finite, acyclic chain, following a non-null next
        # edge strictly decreases the number of nodes remaining.
        measures = {
            name: tuple(index for index, ty in enumerate(self.functions[name][1].params)
                        if isinstance(ty, BorrowRecord) and ty.readonly and ty.chain_limit is True)
            for name in names
        }
        if not any(measures.values()):
            return False
        transitions = []
        for name in names:
            _, _, parameters, body = self.functions[name][0].args
            roots = {parameters[index].text: index for index in measures[name]}
            pending = list(body)
            while pending:
                item = pending.pop()
                if isinstance(item, (tuple, list)):
                    pending.extend(item)
                elif isinstance(item, Node):
                    if item.kind == "define" and item.args[0].text in roots:
                        return False
                    if item.kind == "assign":
                        target = item.args[0]
                        if target.kind == "name" and target.args[0] in roots:
                            return False
                    if item.kind == "call":
                        callee, args = item.args
                        if callee.kind == "name" and callee.args[0] in measures:
                            destination = callee.args[0]
                            edges = []
                            for index in measures[destination]:
                                if index >= len(args):
                                    return False
                                base = args[index]
                                change = Change.WEAK
                                while base.kind == "member" and base.args[1].text == "next":
                                    base = base.args[0]
                                    change = Change.STRICT
                                if base.kind == "name" and base.args[0] in roots:
                                    edges.append((roots[base.args[0]], index, change))
                            transitions.append(Transition(name, destination, tuple(edges)))
                    pending.extend(item.args)
        return bool(transitions) and check_termination(measures, transitions).status == "proved"

    # A recursive result contract is a fixed point: the body is checked under an
    # assumed result type, and the assumption is only kept when the returned
    # values fit inside it.  Iteration starts from the widest integer parameter
    # and widens to the enclosing container whenever the assumption is refuted.
    RESULT_ROUNDS = 6

    def infer_recursive_results(self, names):
        missing = [name for name in names if self.functions[name][1].result is None]
        if not missing:
            return
        candidates = {name: self.result_candidates(self.functions[name][1]) for name in missing}
        position = {name: 0 for name in missing}
        seeds = {name: candidates[name][0] for name in missing}
        verified, tried = {}, set()
        self.last_trial_error = None
        self.inferring_results = True
        try:
            for _ in range(self.RESULT_ROUNDS + sum(len(sets) for sets in candidates.values())):
                key = tuple((name, seeds[name]) for name in missing)
                if key in tried:
                    break
                tried.add(key)
                inferred = self.trial_recursive_results(names, seeds)
                if inferred is None:
                    verified = {}
                    for name in missing:
                        if position[name] + 1 < len(candidates[name]):
                            position[name] += 1
                            seeds[name] = candidates[name][position[name]]
                        else:
                            seeds[name] = self.escalated_result(seeds[name])
                    continue
                progress = False
                verified = {}
                for name in missing:
                    candidate = inferred[name]
                    if candidate is None:
                        verified[name] = None
                        continue
                    if not isinstance(candidate, Scalar):
                        self.functions[name][0].args[0].error(
                            "a recursive function infers a result that is not a plain scalar; "
                            "declare the result contract explicitly")
                    if self.result_within(candidate, seeds[name]):
                        if candidate != seeds[name]:
                            seeds[name] = candidate
                            progress = True
                        verified[name] = candidate
                    else:
                        seeds[name] = self.widened_result(seeds[name], candidate)
                        progress = True
                if not progress and len(verified) == len(missing):
                    break
        finally:
            self.inferring_results = False
        unresolved = [name for name in missing if name not in verified]
        if unresolved and self.last_trial_error is not None:
            where, _, detail = str(self.last_trial_error).partition(": ")
            if detail:
                raise LanError(f"{where}: cannot infer the result of recursive function "
                               f"{unresolved[0]!r}; declare the result contract explicitly ({detail})")
            raise self.last_trial_error
        for name in missing:
            if name not in verified:
                self.functions[name][0].args[0].error(
                    f"the result of recursive function {name!r} does not stabilise "
                    f"(widened to {seeds[name].c if seeds[name] is not None else 'void'}); "
                    "declare the result contract explicitly")
            node, contract = self.functions[name]
            self.functions[name] = (node, replace(contract, result=verified[name]))
            result = verified[name]
            detail = "" if result is None or result.floating else f" {result.domain}"
            self.trace.append(f"  recursive result {name}: {result.c if result is not None else 'void'}{detail}")

    def trial_recursive_results(self, names, seeds):
        """Analyse a whole group under an assumed result type, then undo it."""
        saved = self.save_analysis_state()
        original = {name: self.functions[name] for name in names}
        failure = None
        try:
            inferred = {}
            for name in names:
                # Every call site of the group must see the assumption, not the
                # half-inferred contract of an earlier member.
                for member in names:
                    node, contract = original[member]
                    self.functions[member] = (node, replace(contract, result=seeds.get(member, contract.result)))
                self.function(name, *self.functions[name])
                inferred[name] = self.functions[name][1].result
            return inferred
        except LanError as error:
            failure = error
            return None
        finally:
            self.restore_analysis_state(saved)
            self.functions.update(original)
            if failure is not None and self.last_trial_error is None:
                # A failure under the first assumption is the likeliest to be
                # the real diagnostic; later rounds only widen the guess.
                self.last_trial_error = failure

    def save_analysis_state(self):
        copied = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                copied[key] = {item: (copy.copy(entry) if isinstance(entry, (list, set, dict)) else entry)
                               for item, entry in value.items()}
            elif isinstance(value, (list, set)):
                copied[key] = copy.copy(value)
        plain = {key: value for key, value in self.__dict__.items() if key not in copied}
        return copied, plain

    def restore_analysis_state(self, saved):
        copied, plain = saved
        self.__dict__.update(plain)
        self.__dict__.update(copied)

    @staticmethod
    def result_seed(contract):
        integers = [ty for ty in contract.params
                    if isinstance(ty, Scalar) and not ty.boolean and not ty.floating]
        if integers:
            bits = max(ty.bits for ty in integers)
            signed = all(ty.signed for ty in integers)
            return Scalar(container(bits, signed), bits, signed)
        if any(isinstance(ty, Scalar) and ty.boolean for ty in contract.params):
            return Scalar(BOOL, 8, boolean=True)
        return Scalar(container(8, False), 8, False)

    @classmethod
    def result_candidates(cls, contract):
        """Assumptions to try, most likely first, before widening outwards.

        Floating results are never guessed: they are only adopted when a trial
        already reports one, so integer arithmetic is never read as floating.
        """
        seeds = [cls.result_seed(contract), Scalar(BOOL, 8, boolean=True), None]
        for seed in (Scalar(container(bits, signed), bits, signed)
                     for bits in (8, 16, 32, 64) for signed in (False, True)):
            if seed not in seeds:
                seeds.append(seed)
        return tuple(seeds)

    @staticmethod
    def result_within(candidate, seed):
        if seed is None:
            return False
        if candidate.floating or seed.floating:
            if not (candidate.floating and seed.floating):
                return False
            if candidate.bits != seed.bits:
                return False
            return candidate.domain.within(seed.domain)
        if seed.boolean != candidate.boolean:
            # A boolean result fits any integer assumption whose container holds {0, 1}.
            return candidate.boolean and not seed.boolean and candidate.domain.within(seed.domain)
        if candidate.bits > seed.bits:
            return False
        if candidate.signed and not seed.signed:
            return False
        return candidate.domain.within(seed.domain)

    @classmethod
    def widened_result(cls, seed, candidate):
        if seed is None or candidate is None:
            return candidate if seed is None else seed
        if seed.floating or candidate.floating:
            bits = max(seed.bits if seed.floating else 32, candidate.bits if candidate.floating else 32)
            if seed.floating and candidate.floating and seed.bits == candidate.bits:
                return Scalar(FloatDomain.union((seed.domain, candidate.domain), bits),
                              bits, floating=True)
            return Scalar(FloatDomain.top(bits), bits, floating=True)
        signed = seed.signed or candidate.signed or candidate.domain.lo < 0
        boolean = seed.boolean and candidate.boolean
        domain = domain_union((seed.domain, candidate.domain))
        bits = max(seed.bits, candidate.bits, 8)
        while bits < 64 and not domain.within(container(bits, signed)):
            bits *= 2
        if boolean:
            return Scalar(BOOL, 8, boolean=True)
        return Scalar(container(bits, signed), bits, signed)

    @classmethod
    def escalated_result(cls, seed):
        """Widen a refuted assumption to its container, then to a wider container."""
        if seed is None:
            return Scalar(container(8, False), 8, False)
        if seed.floating:
            return Scalar(FloatDomain.top(64), 64, floating=True)
        if seed.boolean:
            return Scalar(BOOL, 8, boolean=True)
        bits = seed.bits if seed.domain != container(seed.bits, seed.signed) else min(seed.bits * 2, 64)
        return Scalar(container(bits, seed.signed), bits, seed.signed)
