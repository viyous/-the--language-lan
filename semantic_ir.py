"""Lossless, ordered Lan operations and the boundary for verified facts."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sequence:
    """A source list, distinct from a tuple of fixed operation operands."""

    items: tuple


@dataclass(frozen=True)
class Operation:
    anchor: int
    opcode: str
    kind: str
    token: object
    args: tuple
    source_text: str | None
    state_guard: "Operation | None"
    binding: int | None = None
    control_target: int | None = None


@dataclass(frozen=True)
class Storage:
    type: object
    mode: str
    readonly: bool
    ownership: str | None = None


@dataclass(frozen=True)
class LoopRecurrence:
    condition: int | None
    writes: tuple
    count: int | None = None


@dataclass
class ModuleIR:
    body: Sequence
    operations: tuple[Operation, ...]
    declarations: dict
    contracts: dict = field(default_factory=dict)
    types: dict = field(default_factory=dict)
    storage: dict = field(default_factory=dict)
    loops: dict = field(default_factory=dict)
    storage_observations: dict = field(default_factory=dict)
    return_observations: dict = field(default_factory=dict)

    def lowering_body(self, make_node):
        """Adapt the operation graph to the current checker and C emitter."""
        nodes = {}

        def materialize(value):
            if isinstance(value, Operation):
                if value.anchor not in nodes:
                    nodes[value.anchor] = make_node(value, materialize(value.args), materialize(value.state_guard))
                return nodes[value.anchor]
            if isinstance(value, Sequence):
                return list(map(materialize, value.items))
            if isinstance(value, tuple):
                return tuple(map(materialize, value))
            return value

        return materialize(self.body)


PLACE_OPERATIONS = {"name": "binding", "member": "field", "index": "element", "slice": "slice",
                    "length": "length", "bit_view": "bit_window"}
OPERATIONS = {"if": "branch", "while": "loop", "propagate": "propagate_error",
              "define": "declare", "record_initializer": "construct_record",
              "list": "construct_array", "repeat": "fill_array"}


def lower_module(nodes, node_type):
    """Lower declarations, type functions and runtime code with the same rules.

    Nested regions retain lexical scope, evaluation order and control targets.
    Syntax whose operation depends on a type (for example owner binding or
    projection) retains its type operands and receives checked storage later.
    """
    declarations = {}
    operations = []
    shared = {}

    def declare(token):
        return declarations.setdefault(token, len(declarations))

    def convert(value, bindings, loop, function):
        if isinstance(value, node_type):
            return lower(value, bindings, loop, function)
        if isinstance(value, list):
            return Sequence(tuple(convert(item, bindings, loop, function) for item in value))
        if isinstance(value, tuple):
            return tuple(convert(item, bindings, loop, function) for item in value)
        return value

    def block(body, bindings, loop, function):
        scope = bindings.copy()
        return Sequence(tuple(lower(stmt, scope, loop, function) for stmt in body))

    def lower(node, bindings, loop, function, role="value"):
        key = id(node), role
        if key in shared:
            return shared[key]
        anchor = len(operations)
        operations.append(None)
        kind, args = node.kind, node.args
        opcode = OPERATIONS.get(kind, kind)
        binding = control_target = None
        if kind == "function":
            name, contract, parameters, body = args
            scope = {param.text: declare(param) for param in parameters}
            args = (name, convert(contract, scope, None, anchor), convert(parameters, scope, None, anchor),
                    block(body, scope, None, anchor))
        elif kind == "type_function":
            name, parameters, body = args
            scope = {param.text: declare(param) for param, _ in parameters}
            args = name, convert(parameters, scope, None, anchor), block(body, scope, None, anchor)
        elif kind == "define":
            name, annotation, initial = args
            args = name, convert(annotation, bindings, loop, function), convert(initial, bindings, loop, function)
            binding = declare(name)
            bindings[name.text] = binding
        elif kind == "const":
            name, annotation, initial = args
            args = name, convert(annotation, bindings, loop, function), convert(initial, bindings, loop, function)
            binding = declare(name)
            bindings[name.text] = binding
            if annotation is not None and annotation.kind == "storage" and annotation.args[1].text == "heap":
                opcode = "bind_owner"
        elif kind == "assign":
            target, operator, value = args
            target = lower(target, bindings, loop, function, role="address")
            binding = target.binding
            args = target, operator, convert(value, bindings, loop, function)
            opcode = "store" if operator.text == "=" else "read_modify_write"
        elif kind == "if":
            condition, yes, no = args
            args = (convert(condition, bindings, loop, function), block(yes, bindings, loop, function),
                    block(no, bindings, loop, function))
        elif kind == "while":
            condition, body, *annotations = args
            args = (convert(condition, bindings, loop, function), block(body, bindings, anchor, function),
                    *convert(tuple(annotations), bindings, loop, function))
        elif kind == "interation":
            name, initial, annotation, count, step, body = args
            binding = declare(name)
            scope = bindings | {name.text: binding}
            args = (name, *convert((initial, annotation, count, step), bindings, loop, function),
                    block(body, scope, anchor, function))
            opcode = "counted_loop"
        elif kind in PLACE_OPERATIONS:
            opcode = ("address_" if role == "address" else "read_") + PLACE_OPERATIONS[kind]
            if kind == "name":
                binding = bindings.get(args[0])
                if role == "callee":
                    opcode = "callee"
            else:
                host = lower(args[0], bindings, loop, function, role)
                binding = host.binding
                args = (host, *convert(args[1:], bindings, loop, function))
        elif kind == "call":
            callee, arguments = args
            release = callee.kind == "name" and callee.args[0] == "free"
            opcode = "release" if release else "call"
            operand_role = "address" if release else "value"
            callee = lower(callee, bindings, loop, function, role="callee")
            operands = tuple(lower(argument, bindings, loop, function, operand_role) for argument in arguments)
            args = callee, Sequence(operands) if isinstance(arguments, list) else operands
        else:
            if kind == "binary" and args[0] in ("&&", "||"):
                opcode = "short_circuit"
            elif kind == "continue":
                control_target = loop
            elif kind in ("return", "propagate"):
                control_target = function
            args = convert(args, bindings, loop, function)
        state_guard = convert(node.state_guard, bindings, loop, function)
        operation = Operation(anchor, opcode, kind, node.token, args, node.source_text,
                              state_guard, binding, control_target)
        operations[anchor] = shared[key] = operation
        return operation

    body = block(nodes, {}, None, None)
    return ModuleIR(body, tuple(operations), declarations)


@dataclass(frozen=True)
class Proposition:
    relation: str
    operands: tuple


@dataclass(frozen=True)
class Theorem:
    conclusion: Proposition
    anchor: int
    kind: str
    location: tuple[int, int]
    status: str = "verified"
    function: str | None = None
    evidence: int | None = None

    def record(self):
        data = {"anchor": self.anchor, "kind": self.kind, "location": self.location,
                "status": self.status, "conclusion": {"relation": self.conclusion.relation,
                                                       "operands": self.conclusion.operands}}
        if self.function is not None:
            data["function"] = self.function
        if self.evidence is not None:
            data["evidence"] = self.evidence
        if self.conclusion.relation == "within":
            data["intervals"] = self.conclusion.operands[0]
        return data


class TheoremExchange:
    """Accept facts only from the verifier, keyed by function and operation."""

    def __init__(self, function):
        self.function = function
        self.accepted = {}
        self.evidence = []

    def prove_range(self, origin, kind, intervals, program, obligation, limits):
        anchor = getattr(origin, "semantic_anchor", None)
        key = anchor, kind, intervals
        if anchor is not None and key in self.accepted:
            return True
        result = program.verify(obligation, **limits)
        if result.status != "proved" or anchor is None:
            return result.status == "proved"
        evidence_id = len(self.evidence)
        guard, goal = obligation.guard, obligation.goal
        arena = program.arena
        nodes = [(i, arena.nodes[i].op, arena.nodes[i].sort, arena.nodes[i].args, arena.nodes[i].data)
                 for i in arena.slice(guard, goal)]
        self.evidence.append({"id": evidence_id, "function": self.function, "guard": guard.id,
                              "goal": goal.id, "method": result.method, "nodes": nodes})
        self.accepted[key] = Theorem(Proposition("within", (intervals,)), anchor, kind,
                                     (origin.token.line, origin.token.col), function=self.function,
                                     evidence=evidence_id)
        return True
