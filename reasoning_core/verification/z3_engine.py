"""
Layer 3: Z3 Multi-Logic Verification Engine.

Three verification modes:
1. Modal (S5/S4/KB Kripke semantics) — conceivability, possible worlds
2. Paraconsistent (Belnap 4-valued) — contradictory positions
3. Defeasible (weighted soft constraints) — presumptive reasoning

Plus:
- MicrotheoryManager: separate Z3 contexts per philosophical theory
- UnsatCoreExtractor: error localization for failed proofs
- Maximal Consistent Subsets: largest non-contradictory subsets

Reference: PHILOSOPHER_ENGINE_ARCHITECTURE.md, Layer 3
"""

from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

try:
    from z3 import (
        Solver, DeclareSort, Function, BoolSort, IntSort, RealSort,
        Const, Consts, ForAll, Exists, Implies, And, Or, Not,
        sat, unsat, unknown, BoolVal, IntVal,
        Optimize, If,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class ModalFrame(Enum):
    """Modal logic frame types."""
    S5 = "s5"    # Reflexive + Symmetric + Transitive (= universal)
    S4 = "s4"    # Reflexive + Transitive (no symmetry)
    KB = "kb"    # Reflexive + Symmetric (no transitivity)
    T = "t"      # Reflexive only
    K = "k"      # No constraints (basic modal logic)


class ModalLogicEngine:
    """S5/S4/KB Kripke semantics for conceivability arguments.

    Encodes possible worlds, accessibility relations, and modal
    operators (necessarily, possibly) for Cartesian arguments.

    Key use cases:
    - Real Distinction (conceivability → possibility in S5)
    - Zombie argument (same structure, different domain)
    - Ontological argument (modal version)
    """

    def __init__(self, frame: ModalFrame = ModalFrame.S5):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required")

        self.frame = frame
        self.World = DeclareSort('World')
        self.R = Function('Accessible', self.World, self.World, BoolSort())
        self.solver = Solver()
        self._add_frame_axioms()

    def _add_frame_axioms(self):
        """Add accessibility axioms for the chosen modal frame."""
        w, v, u = Consts('w v u', self.World)

        if self.frame in (ModalFrame.S5, ModalFrame.S4,
                          ModalFrame.KB, ModalFrame.T):
            # Reflexive
            self.solver.add(ForAll([w], self.R(w, w)))

        if self.frame in (ModalFrame.S5, ModalFrame.KB):
            # Symmetric
            self.solver.add(
                ForAll([w, v], Implies(self.R(w, v), self.R(v, w))))

        if self.frame in (ModalFrame.S5, ModalFrame.S4):
            # Transitive
            self.solver.add(
                ForAll([w, v, u],
                       Implies(And(self.R(w, v), self.R(v, u)),
                               self.R(w, u))))

    def necessarily(self, prop_fn, world):
        """□P: P holds in all accessible worlds."""
        w = Const('_nec_w', self.World)
        return ForAll([w], Implies(self.R(world, w), prop_fn(w)))

    def possibly(self, prop_fn, world):
        """◇P: P holds in some accessible world."""
        w = Const('_pos_w', self.World)
        return Exists([w], And(self.R(world, w), prop_fn(w)))

    def check_entailment(self, premises: list, conclusion) -> str:
        """Check if conclusion follows from premises.

        Returns: 'valid' (UNSAT), 'invalid' (SAT), or 'timeout'
        """
        self.solver.push()
        for p in premises:
            self.solver.add(p)
        self.solver.add(Not(conclusion))

        result = self.solver.check()
        self.solver.pop()

        if result == unsat:
            return "valid"
        elif result == sat:
            return "invalid"
        return "timeout"

    def check_consistency(self, assertions: list) -> str:
        """Check if a set of assertions is consistent.

        Returns: 'consistent' (SAT), 'inconsistent' (UNSAT), or 'timeout'
        """
        self.solver.push()
        for a in assertions:
            self.solver.add(a)

        result = self.solver.check()
        self.solver.pop()

        if result == sat:
            return "consistent"
        elif result == unsat:
            return "inconsistent"
        return "timeout"

    def get_model(self):
        """Get a satisfying model (if last check was SAT)."""
        return self.solver.model()


class ParaconsistentEngine:
    """Belnap 4-valued logic for analyzing contradictory positions.

    Four truth values: True, False, Both, Neither
    Allows reasoning about contradictory philosophical positions
    (e.g., "the mind is physical AND the mind is not physical")
    without logical explosion.

    Key use case: when competing theories make contradictory claims,
    paraconsistent logic lets us track both without deriving
    arbitrary conclusions.
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required")

        # Belnap 4-valued: 0=Neither, 1=False, 2=True, 3=Both
        self.solver = Solver()

    def belnap_value(self, name: str):
        """Create a Belnap 4-valued variable (0-3)."""
        v = Const(name, IntSort())
        self.solver.add(And(v >= 0, v <= 3))
        return v

    def belnap_and(self, a, b):
        """Belnap AND: min of truth values."""
        return If(a < b, a, b)

    def belnap_or(self, a, b):
        """Belnap OR: max of truth values."""
        return If(a > b, a, b)

    def belnap_not(self, a):
        """Belnap NOT: 0↔0, 1↔2, 2↔1, 3↔3."""
        return If(a == IntVal(0), IntVal(0),
                  If(a == IntVal(1), IntVal(2),
                     If(a == IntVal(2), IntVal(1),
                        IntVal(3))))

    def is_true(self, a):
        """Check if value is at least True (2 or 3)."""
        return Or(a == IntVal(2), a == IntVal(3))

    def is_false(self, a):
        """Check if value is at least False (1 or 3)."""
        return Or(a == IntVal(1), a == IntVal(3))

    def is_both(self, a):
        """Value is Both (true AND false)."""
        return a == IntVal(3)

    def is_neither(self, a):
        """Value is Neither (not true, not false)."""
        return a == IntVal(0)

    def analyze_contradiction(self, prop_name: str,
                              theory_a: str, theory_b: str) -> Dict:
        """Analyze a proposition under two contradicting theories.

        Returns status of the proposition in Belnap 4-valued logic.
        """
        self.solver.push()

        val = self.belnap_value(prop_name)

        # Theory A says it's true, theory B says it's false
        self.solver.add(self.is_true(val))
        self.solver.add(self.is_false(val))
        # → value must be Both (3)

        result = self.solver.check()
        self.solver.pop()

        return {
            "proposition": prop_name,
            "theory_a": theory_a,
            "theory_b": theory_b,
            "result": "both" if result == sat else "inconsistent",
            "interpretation": (
                f"Under {theory_a} (true) and {theory_b} (false), "
                f"'{prop_name}' has Belnap value BOTH — "
                f"genuinely contradictory without explosion."
            ),
        }


class DefeasibleEngine:
    """Weighted soft constraints for presumptive reasoning.

    Uses Z3 Optimize to handle defeasible rules where:
    - Some rules have priority over others
    - Defaults can be overridden by stronger evidence
    - Ceteris paribus conditions can be violated

    Key use case: competing philosophical interpretations where
    each has some evidence but none is conclusive.
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required")

        self.optimizer = Optimize()
        self.constraints: Dict[str, Tuple] = {}  # name -> (constraint, weight)

    def add_hard(self, name: str, constraint):
        """Add a hard (indefeasible) constraint."""
        self.optimizer.add(constraint)
        self.constraints[name] = (constraint, float('inf'))

    def add_soft(self, name: str, constraint, weight: float = 1.0):
        """Add a soft (defeasible) constraint with weight."""
        self.optimizer.add_soft(constraint, weight)
        self.constraints[name] = (constraint, weight)

    def maximize(self, objective):
        """Set optimization objective."""
        self.optimizer.maximize(objective)

    def check(self) -> str:
        """Find optimal satisfying assignment."""
        result = self.optimizer.check()
        if result == sat:
            return "satisfiable"
        elif result == unsat:
            return "unsatisfiable"
        return "timeout"

    def get_model(self):
        """Get the optimal model."""
        return self.optimizer.model()

    def get_satisfied_constraints(self) -> List[str]:
        """Return which soft constraints are satisfied in optimal model."""
        if self.optimizer.check() != sat:
            return []
        model = self.optimizer.model()
        satisfied = []
        for name, (constraint, weight) in self.constraints.items():
            # Check if constraint holds in model
            try:
                if model.evaluate(constraint):
                    satisfied.append(name)
            except Exception:
                pass
        return satisfied


class MicrotheoryManager:
    """Separate Z3 contexts for competing philosophical positions.

    Each theory (physicalism, dualism, functionalism, etc.) gets
    its own solver with its own axioms. This prevents logical
    explosion from asserting contradictory axioms in the same context.

    Cross-theory consistency checking tests claims against each
    theory independently.
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required")

        self.theories: Dict[str, Solver] = {}
        self.axiom_counts: Dict[str, int] = {}

    def add_theory(self, name: str, axioms: Optional[list] = None):
        """Create a new microtheory context with optional axioms."""
        solver = Solver()
        if axioms:
            for axiom in axioms:
                solver.add(axiom)
        self.theories[name] = solver
        self.axiom_counts[name] = len(axioms) if axioms else 0

    def add_axiom(self, theory_name: str, axiom):
        """Add an axiom to an existing theory."""
        if theory_name not in self.theories:
            raise ValueError(f"Theory '{theory_name}' not found")
        self.theories[theory_name].add(axiom)
        self.axiom_counts[theory_name] = (
            self.axiom_counts.get(theory_name, 0) + 1)

    def check_claim(self, theory_name: str, claim) -> str:
        """Check a claim against a specific theory.

        Returns: 'entailed', 'consistent', 'inconsistent', or 'timeout'
        """
        if theory_name not in self.theories:
            raise ValueError(f"Theory '{theory_name}' not found")

        solver = self.theories[theory_name]

        # Check entailment (negation is UNSAT)
        solver.push()
        solver.add(Not(claim))
        entailment = solver.check()
        solver.pop()

        if entailment == unsat:
            return "entailed"

        # Check consistency (claim is SAT)
        solver.push()
        solver.add(claim)
        consistency = solver.check()
        solver.pop()

        if consistency == sat:
            return "consistent"
        elif consistency == unsat:
            return "inconsistent"
        return "timeout"

    def check_cross_theory(self, claim) -> Dict[str, str]:
        """Test a claim against all theories independently.

        Returns dict mapping theory name to result.
        Useful for seeing which theories accept/reject a claim.
        """
        results = {}
        for name in self.theories:
            results[name] = self.check_claim(name, claim)
        return results

    def find_agreement(self, claim) -> List[str]:
        """Find all theories that entail or are consistent with claim."""
        results = self.check_cross_theory(claim)
        return [
            name for name, result in results.items()
            if result in ("entailed", "consistent")
        ]

    def find_disagreement(self, claim) -> List[str]:
        """Find all theories that are inconsistent with claim."""
        results = self.check_cross_theory(claim)
        return [
            name for name, result in results.items()
            if result == "inconsistent"
        ]

    def get_stats(self) -> Dict:
        return {
            "theories": list(self.theories.keys()),
            "axiom_counts": dict(self.axiom_counts),
            "total_theories": len(self.theories),
        }


class UnsatCoreExtractor:
    """Extract minimal unsatisfiable cores from failed proofs.

    When Z3 reports UNSAT, the unsat core tells us exactly
    which assertions are responsible for the contradiction.
    This is essential for error localization in the self-repair loop.
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver required")

    def extract_core(self, assertions: List,
                     labels: Optional[List[str]] = None) -> List[str]:
        """Find the minimal set of assertions causing UNSAT.

        Args:
            assertions: Z3 assertions (some subset is inconsistent)
            labels: Optional human-readable labels for each assertion

        Returns:
            List of labels (or indices) of core assertions
        """
        solver = Solver()
        solver.set("unsat_core", True)

        if labels is None:
            labels = [f"a{i}" for i in range(len(assertions))]

        # Add tracked assertions
        props = []
        for i, (assertion, label) in enumerate(zip(assertions, labels)):
            p = BoolVal(True)  # tracking variable
            p = Const(f"track_{label}", BoolSort())
            solver.assert_and_track(assertion, p)
            props.append((p, label))

        result = solver.check()
        if result != unsat:
            return []  # Not UNSAT, no core to extract

        core = solver.unsat_core()
        core_labels = []
        for p, label in props:
            if p in core:
                core_labels.append(label)

        return core_labels

    def maximal_consistent_subsets(
            self, assertions: List,
            labels: Optional[List[str]] = None) -> List[List[str]]:
        """Find maximal consistent subsets (MCS).

        When a set of assertions is inconsistent, MCS tells us
        the largest subsets that ARE consistent. Useful for
        identifying which assertions to keep and which to revise.
        """
        if labels is None:
            labels = [f"a{i}" for i in range(len(assertions))]

        # Simple greedy approach: try removing one at a time
        # and check if the rest is consistent
        subsets = []

        for i in range(len(assertions)):
            subset = [a for j, a in enumerate(assertions) if j != i]
            subset_labels = [l for j, l in enumerate(labels) if j != i]

            solver = Solver()
            for a in subset:
                solver.add(a)

            if solver.check() == sat:
                subsets.append(subset_labels)

        # Remove non-maximal subsets
        maximal = []
        for s in subsets:
            is_maximal = True
            for other in subsets:
                if s != other and set(s).issubset(set(other)):
                    is_maximal = False
                    break
            if is_maximal:
                maximal.append(s)

        return maximal if maximal else [labels]
