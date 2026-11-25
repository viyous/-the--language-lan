"""The Lan compiler: one class assembled from per-concern mixins.

Each mixin module holds one group of Compiler methods; they share state through
`self` and call each other freely, so the mixins are not usable on their own.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from semantic_ir import lower_module
from float_domain import FloatDomain
from table_interface import c_ident, digest

from .errors import ProofBudgetError
from .lexer import CHARACTERS, Token
from .syntax import LoweringNode, Node, type_object_kind, TYPE_OBJECT_WORDS, TypeObject
from .parser import Parser
from .typesys import BOOL, NullType, Scalar, Slice, Symbol, WIDTHS
from .domain_utils import container
from .type_resolution import TypeResolutionMixin
from .constants import ConstantEvaluationMixin
from .declarations import DeclarationMixin
from .returns import ReturnMixin
from .records import RecordMixin
from .storage_links import StorageLinkMixin
from .aliases_heap import AliasHeapMixin
from .text import TextMixin
from .dependent import DependentBoundMixin
from .array_contents import ArrayContentMixin
from .slices import SliceMixin
from .slice_compare import SliceComparisonMixin
from .bits import BitMixin
from .operations import OperationMixin
from .writes import WriteMixin
from .interfaces import InterfaceMixin
from .calls import CallMixin
from .expressions import ExpressionMixin
from .properties import PropertyMixin
from .guards import GuardMixin
from .views import ViewMixin
from .statements import StatementMixin
from .bit_assign import BitAssignmentMixin
from .loops import LoopMixin
from .loop_chains import LoopChainMixin
from .loop_contracts import LoopContractMixin
from .loop_analysis import LoopAnalysisMixin
from .blocks import BlockMixin
from .registration import RegistrationMixin
from .inference import InferenceMixin
from .functions import FunctionMixin
from .theorems import TheoremMixin
from .recursion import RecursionMixin
from .codegen import CodegenMixin


class Compiler(TypeResolutionMixin,
               ConstantEvaluationMixin,
               DeclarationMixin,
               ReturnMixin,
               RecordMixin,
               StorageLinkMixin,
               AliasHeapMixin,
               TextMixin,
               DependentBoundMixin,
               ArrayContentMixin,
               SliceMixin,
               SliceComparisonMixin,
               BitMixin,
               OperationMixin,
               WriteMixin,
               InterfaceMixin,
               CallMixin,
               ExpressionMixin,
               PropertyMixin,
               GuardMixin,
               ViewMixin,
               StatementMixin,
               BitAssignmentMixin,
               LoopMixin,
               LoopChainMixin,
               LoopContractMixin,
               LoopAnalysisMixin,
               BlockMixin,
               RegistrationMixin,
               InferenceMixin,
               FunctionMixin,
               TheoremMixin,
               RecursionMixin,
               CodegenMixin):
    PATH_LIMIT = 128
    PATH_HARD_LIMIT = 1024
    GAS_LIMIT = 100_000
    LOOP_WIDEN_AFTER = 8
    LOOP_LIMIT = 64

    def __init__(self, source, filename, module_id=None, *, debug_bounds=False, max_steps=None):
        if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
            raise ValueError("max_steps must be a nonnegative integer or None")
        self.debug_bounds = debug_bounds
        self.max_steps = max_steps
        parser = Parser(source)
        self.semantic_module = lower_module(parser.module(), Node)
        self.nodes = self.semantic_module.lowering_body(LoweringNode.from_operation)
        self.conditions = parser.conditions
        self.trace_endpoint = False
        self.filename = filename
        self.source_path = filename
        self.module_path = Path(filename).name
        self.module_id = module_id
        self.source_digest = digest(source)
        self.property_local_types = None
        self.release_notes = {}
        self.interface_abi = None
        self.export_prefix = None
        self.internal_linkage = False
        self.library = False
        self.type_nodes = {}
        self.type_object_nodes = {}
        self.const_nodes = {}
        self.const_compile_values = {}
        self.string_encoding_nodes = {}
        self.module_consts = {}
        self.module_const_code = []
        self.resolving_consts = set()
        self.record_types = {}
        self.type_functions = {}
        self.type_cache = {}
        self.active_types = set()
        self.types = {
            f"{'' if sign else 'u'}int{bits}": Scalar(container(bits, sign), bits, sign)
            for bits in WIDTHS for sign in (False, True)
        } | {"boolean": Scalar(BOOL, 8, boolean=True), "void": None}
        self.types.update((kind, TypeObject((type_object_kind(Token(kind, 0, 0), kind),)))
                          for kind in TYPE_OBJECT_WORDS["kind"])
        self.types.update({name: Scalar(domain, bits, character=name)
                           for name, (bits, domain) in CHARACTERS.items()})
        self.types["rune"] = self.types["unicode"]
        self.types.update({f"float{bits}": Scalar(FloatDomain.top(bits), bits, floating=True) for bits in (32, 64)})
        self.types.update({
            "ascii_string": Slice(self.types["ascii"], readonly=True, text="ascii"),
            "string": Slice(self.types["utf8"], readonly=True, text="utf8"),
            "c_string": Slice(self.types["utf8"], readonly=True, text="c"),
        })
        self.text_literals = {}
        self.types["null"] = NullType()
        self.null_symbol = Symbol("null", 0, self.types["null"], readonly=True)
        self.alias_types, self.alias_members = {}, {}
        self.resolving = set()
        self.functions = {}
        self.callable_signatures = {}
        self.callable_selectors = {}
        self.callable_options = {}
        self.resolving_callables = set()
        self.callback_edges = {}
        self.externs = {}
        # True while a native declaration is being resolved, so the parameter
        # rules can tell "no body to infer from" apart from "the body says
        # nothing about this parameter".
        self.native_signature = False
        self.imported_functions = {}
        self.imported_type_functions = {}
        self.import_names = set()
        self.exports = {}
        self.oracles = None
        self.scope = []
        self.symbols = []
        self.trace = [f"@module {ascii(self.module_path)}"]
        self.proof_exchanges = {}
        self.source_theorems = []
        self.loop_recurrences = {}
        self.storage_evidence = []
        self.storage_counts = {}
        self.call_return_observations = {}
        self.return_evidence = []
        self.loop_exit_facts = {}
        self.helpers = {}
        self.ambient_gas = self.GAS_LIMIT * max(1, len(self.nodes))
        self.budget_block = None
        self.block_gas = {}
        self.type_block_gas = {}
        self.calls = {}
        self.recursive_functions = set()
        self.recursive_groups = {}
        self.recursive_summaries = {}
        self.pending_recursive_results = set()
        self.inferring_results = False
        self.last_trial_error = None
        self.mutations = {}
        self.record_results = {}
        self.owned_chain_results = {}
        self.owned_chain_parameters = {}
        self.array_results = {}
        self.result_domains = {}
        self.verified_guards = {}
        self.param_places = {}
        self.borrow_ancestry = {}
        self.owning = {}
        self.writes = set()
        # Direct allocator effects and effects from imported or native calls.
        self.memory_effects = set()
        self.external_effects = set()
        # Source order of each recorded interface rebind.  Rebinding the same
        # field more than once is meaningful only when the order is known:
        # a later assignment can shadow an earlier one, and a guarded write
        # can select between the old and the new donor.
        self.rebind_order = {}
        # Interface writes that occur before the first rebind of the same
        # field.  Their target is still the old descriptor, so the call site
        # can apply them before applying the rebind summary.
        self.interface_before_writes = set()
        self.interface_after_writes = set()
        self.return_states = []
        self.property_codegen_context = None
        self.propagating_expression_nodes = frozenset()
        self.wrap32_nodes = frozenset()
        self.initializing_heap_records = ()
        self.property_relation_codegen = None
        self.inferred_returns = None
        self.rhs_codes = {}
        self.keyed_access_types = {}
        self.loops: list[LoopEdges] = []
        self._active_paths: tuple = ()
        self._future_stmts: list = []
        self._contract_live_names: frozenset = frozenset()

    def function_c_name(self, name):
        prefix = "lan_fn" if self.module_id is None else f"lan_m{self.module_id}_fn"
        return f"{prefix}_{c_ident(name)}"

    def helper_prefix(self):
        return "lh" if self.module_id is None else f"lh_m{self.module_id}"

    @property
    def gas(self):
        if self.budget_block is None:
            return self.ambient_gas
        is_type, block = self.budget_block
        return (self.type_block_gas if is_type else self.block_gas)[block]

    @gas.setter
    def gas(self, amount):
        if self.budget_block is None:
            self.ambient_gas = amount
        else:
            is_type, block = self.budget_block
            (self.type_block_gas if is_type else self.block_gas)[block] = amount

    @contextmanager
    def code_block_budget(self, statements, *, is_type=False):
        previous = self.budget_block
        # One allowance per lexical block. Revisiting the same body during a
        # loop or type-function recursion continues spending its allowance.
        block = id(statements)
        (self.type_block_gas if is_type else self.block_gas).setdefault(block, self.GAS_LIMIT)
        self.budget_block = is_type, block
        try:
            yield
        finally:
            self.budget_block = previous

    def spend(self, token, amount=1):
        self.gas -= amount
        if self.gas < 0:
            if self.active_types:
                token.error("compile-time type evaluation budget exhausted")
            raise ProofBudgetError(f"{token.line}:{token.col}: analysis budget exhausted in this code block; "
                                   "safety could not be proved")
