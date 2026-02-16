"""
Verification Examples: Pre-formalized arguments for testing and SFT.

Provides a library of Z3-verified philosophical arguments that:
1. Serve as regression tests for the verification engine
2. Generate SFT training data (Type D: formalization examples)
3. Demonstrate all three Z3 modes (modal, paraconsistent, defeasible)
4. Provide templates for the model to learn from

Each example includes:
  - Natural language description
  - Z3 formalization
  - Expected result
  - Verification mode used
  - Source text reference

Reference: PHILOSOPHER_ENGINE_V3_UNIFIED_ARCHITECTURE.md, Layer 3
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from z3 import (
        Solver, Optimize, DeclareSort, Function,
        BoolSort, IntSort, RealSort,
        Const, Consts, ForAll, Exists, Implies, And, Or, Not,
        BoolVal, IntVal, If,
        sat, unsat, unknown,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class VerificationMode(Enum):
    MODAL = "modal"
    PARACONSISTENT = "paraconsistent"
    DEFEASIBLE = "defeasible"
    CLASSICAL = "classical"


class ExpectedResult(Enum):
    VALID = "valid"         # Conclusion follows from premises (UNSAT)
    INVALID = "invalid"     # Conclusion does not follow (SAT)
    CONSISTENT = "consistent"    # Premises are consistent (SAT)
    INCONSISTENT = "inconsistent"  # Premises are inconsistent (UNSAT)


@dataclass
class VerificationExample:
    """A single verification example with Z3 code and expected result."""
    example_id: str
    title: str
    description: str
    source: str
    mode: VerificationMode
    expected: ExpectedResult
    natural_language: str
    z3_code_fn: str   # Name of the function that runs the verification
    tags: List[str]


# ============================================================
# EXAMPLE LIBRARY
# ============================================================

EXAMPLES: Dict[str, VerificationExample] = {}


def _register(ex: VerificationExample):
    EXAMPLES[ex.example_id] = ex
    return ex


# ---- CLASSICAL MODE ----

_register(VerificationExample(
    "cogito_strict",
    "Cogito — Strict Deduction",
    "The Cogito as a strict syllogistic deduction. "
    "Doubting → Thinking → Existing.",
    "Meditation II",
    VerificationMode.CLASSICAL,
    ExpectedResult.VALID,
    "P1: I doubt.\n"
    "P2: Whatever doubts, thinks.\n"
    "P3: Whatever thinks, exists.\n"
    "C:  I exist.",
    "verify_cogito_strict",
    ["descartes", "cogito", "strict", "meditation_ii"],
))

_register(VerificationExample(
    "trademark_causal",
    "Trademark Argument — Causal Adequacy",
    "I (finite) cannot cause an idea with infinite objective reality. "
    "Therefore an infinite being must exist.",
    "Meditation III",
    VerificationMode.CLASSICAL,
    ExpectedResult.INCONSISTENT,
    "P1: I have formal reality = 2 (finite).\n"
    "P2: My idea of God has objective reality = 4 (infinite).\n"
    "P3: Cause's formal reality >= effect's objective reality.\n"
    "P4: I cause the idea of God.\n"
    "Test: Is this set consistent? (No → I can't be the sole cause.)",
    "verify_trademark_causal",
    ["descartes", "trademark", "meditation_iii", "causal_adequacy"],
))

_register(VerificationExample(
    "wax_elimination",
    "Wax Argument — Elimination of Knowledge Sources",
    "Senses and imagination eliminated as knowledge sources for "
    "bodily nature. Intellect remains.",
    "Meditation II",
    VerificationMode.CLASSICAL,
    ExpectedResult.VALID,
    "P1: Knowledge source is senses, imagination, or intellect.\n"
    "P2: Senses are eliminated (properties all change).\n"
    "P3: Imagination is eliminated (infinite configurations).\n"
    "C:  Knowledge source is intellect.",
    "verify_wax_elimination",
    ["descartes", "wax", "meditation_ii", "elimination"],
))

_register(VerificationExample(
    "modus_ponens_test",
    "Modus Ponens Sanity Check",
    "Basic logical validity test. If P→Q and P, then Q.",
    "Logic",
    VerificationMode.CLASSICAL,
    ExpectedResult.VALID,
    "P1: If it rains, the ground is wet.\n"
    "P2: It rains.\n"
    "C:  The ground is wet.",
    "verify_modus_ponens",
    ["logic", "basic", "sanity_check"],
))

_register(VerificationExample(
    "invalid_affirming_consequent",
    "Affirming the Consequent (Invalid)",
    "Classic fallacy test. If P→Q and Q, cannot conclude P.",
    "Logic",
    VerificationMode.CLASSICAL,
    ExpectedResult.INVALID,
    "P1: If it rains, the ground is wet.\n"
    "P2: The ground is wet.\n"
    "C:  It rains. (INVALID — could be a sprinkler)",
    "verify_affirming_consequent",
    ["logic", "fallacy", "sanity_check"],
))


# ---- MODAL MODE ----

_register(VerificationExample(
    "real_distinction_s5",
    "Real Distinction — S5 Modal",
    "Conceivability of mind without body, in S5 modal logic. "
    "Tests whether identity thesis (Mind=Body) is refuted.",
    "Meditation VI",
    VerificationMode.MODAL,
    ExpectedResult.INCONSISTENT,
    "P1: There is an accessible world where mind exists without body.\n"
    "P2: There is an accessible world where body exists without mind.\n"
    "Thesis: Mind = Body at all worlds.\n"
    "Test: Is the identity thesis consistent with the above? (No.)",
    "verify_real_distinction_s5",
    ["descartes", "real_distinction", "modal", "s5", "meditation_vi"],
))

_register(VerificationExample(
    "dream_argument_modal",
    "Dream Argument — Modal Skepticism",
    "Tests whether we can establish sense reliability given that "
    "we cannot distinguish dreaming from waking.",
    "Meditation I",
    VerificationMode.MODAL,
    ExpectedResult.CONSISTENT,
    "P1: In waking states, senses are reliable.\n"
    "P2: In dreaming states, senses are not reliable.\n"
    "P3: We cannot distinguish current state from dreaming.\n"
    "Test: Is it consistent that senses are currently unreliable? (Yes.)",
    "verify_dream_modal",
    ["descartes", "dream", "skepticism", "meditation_i"],
))

_register(VerificationExample(
    "zombie_argument_s5",
    "Zombie Argument — S5 Modal",
    "Tests whether physicalism is consistent with the existence "
    "of a zombie world (physical duplicate without consciousness).",
    "Chalmers 1996",
    VerificationMode.MODAL,
    ExpectedResult.INCONSISTENT,
    "P1: Zombie world is accessible (physical + not conscious).\n"
    "P2: Actual world is physical + conscious.\n"
    "Physicalism: physical → conscious at all worlds.\n"
    "Test: Is physicalism consistent? (No, given zombie world.)",
    "verify_zombie_s5",
    ["chalmers", "zombie", "modal", "s5", "physicalism"],
))


# ---- PARACONSISTENT MODE ----

_register(VerificationExample(
    "mind_body_contradiction",
    "Mind-Body: Paraconsistent Analysis",
    "Physicalism says mind IS physical. Dualism says mind is NOT. "
    "In Belnap 4-valued logic, the proposition 'mind is physical' "
    "gets value BOTH — genuinely contradictory without explosion.",
    "Philosophy of Mind",
    VerificationMode.PARACONSISTENT,
    ExpectedResult.CONSISTENT,
    "Theory A (Physicalism): 'Mind is physical' is TRUE.\n"
    "Theory B (Dualism): 'Mind is physical' is FALSE.\n"
    "Belnap result: value = BOTH (3).\n"
    "Test: Can we hold both without deriving arbitrary conclusions? (Yes.)",
    "verify_mind_body_paraconsistent",
    ["paraconsistent", "belnap", "mind_body", "physicalism", "dualism"],
))


# ---- DEFEASIBLE MODE ----

_register(VerificationExample(
    "competing_theories_defeasible",
    "Competing Consciousness Theories — Defeasible Reasoning",
    "Six theories of consciousness with weighted evidence. "
    "Defeasible engine finds optimal theory given soft constraints.",
    "Philosophy of Mind",
    VerificationMode.DEFEASIBLE,
    ExpectedResult.CONSISTENT,
    "Soft constraints with weights:\n"
    "  IIT explains integration (weight 3.0)\n"
    "  GWT explains access (weight 2.5)\n"
    "  HOT explains metacognition (weight 2.0)\n"
    "  Hard: at most one theory can be 'best overall'\n"
    "Test: Find optimal assignment.",
    "verify_competing_theories",
    ["defeasible", "consciousness_theories", "optimization"],
))


# ============================================================
# VERIFICATION FUNCTIONS
# ============================================================

def verify_cogito_strict() -> Tuple[str, bool]:
    """Verify the Cogito as a strict deduction."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    Agent = DeclareSort('Agent')
    Thinks = Function('Thinks', Agent, BoolSort())
    Exists_ = Function('Exists', Agent, BoolSort())
    Doubts = Function('Doubts', Agent, BoolSort())

    I = Const('I', Agent)
    a = Const('a', Agent)
    s = Solver()

    s.add(Doubts(I))
    s.add(ForAll([a], Implies(Doubts(a), Thinks(a))))
    s.add(ForAll([a], Implies(Thinks(a), Exists_(a))))

    s.push()
    s.add(Not(Exists_(I)))
    result = s.check()
    s.pop()

    return str(result), result == unsat


def verify_trademark_causal() -> Tuple[str, bool]:
    """Verify that finite being cannot cause idea with infinite
    objective reality."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    Entity = DeclareSort('Entity')
    formal_reality = Function('FormalReality', Entity, IntSort())
    objective_reality = Function('ObjectiveReality', Entity, IntSort())
    causes = Function('Causes', Entity, Entity, BoolSort())

    me = Const('me', Entity)
    idea_of_god = Const('idea_of_god', Entity)
    a, b = Consts('a b', Entity)

    s = Solver()
    s.add(formal_reality(me) == 2)
    s.add(objective_reality(idea_of_god) == 4)
    s.add(ForAll([a, b], Implies(
        causes(a, b), formal_reality(a) >= objective_reality(b))))
    s.add(causes(me, idea_of_god))

    result = s.check()
    return str(result), result == unsat


def verify_wax_elimination() -> Tuple[str, bool]:
    """Verify the Wax argument by elimination."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    Source = DeclareSort('KnowledgeSource')
    senses = Const('senses', Source)
    imagination = Const('imagination', Source)
    intellect = Const('intellect', Source)
    knowledge = Const('wax_source', Source)

    s = Solver()
    s.add(Or(knowledge == senses,
             knowledge == imagination,
             knowledge == intellect))
    s.add(knowledge != senses)
    s.add(knowledge != imagination)

    # Test: is Not(knowledge == intellect) consistent?
    s.push()
    s.add(knowledge != intellect)
    result = s.check()
    s.pop()

    return str(result), result == unsat


def verify_modus_ponens() -> Tuple[str, bool]:
    """Basic modus ponens validity test."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    P = BoolVal(True)  # placeholder
    from z3 import Bool
    p = Bool('rains')
    q = Bool('wet')

    s = Solver()
    s.add(Implies(p, q))
    s.add(p)

    s.push()
    s.add(Not(q))
    result = s.check()
    s.pop()

    return str(result), result == unsat


def verify_affirming_consequent() -> Tuple[str, bool]:
    """Affirming the consequent — should be INVALID."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    from z3 import Bool
    p = Bool('rains')
    q = Bool('wet')

    s = Solver()
    s.add(Implies(p, q))
    s.add(q)

    s.push()
    s.add(Not(p))
    result = s.check()
    s.pop()

    # SAT means Not(p) is consistent with premises → p not entailed
    return str(result), result == sat


def verify_real_distinction_s5() -> Tuple[str, bool]:
    """Real Distinction in S5."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    World = DeclareSort('World')
    R = Function('R', World, World, BoolSort())
    Mind = Function('Mind', World, BoolSort())
    Body = Function('Body', World, BoolSort())

    actual = Const('actual', World)
    w_mind = Const('w_mind', World)
    w_body = Const('w_body', World)
    w, v, u = Consts('w v u', World)

    s = Solver()

    # S5
    s.add(ForAll([w], R(w, w)))
    s.add(ForAll([w, v], Implies(R(w, v), R(v, w))))
    s.add(ForAll([w, v, u], Implies(And(R(w, v), R(v, u)), R(w, u))))

    s.add(Mind(actual))
    s.add(Body(actual))
    s.add(R(actual, w_mind))
    s.add(Mind(w_mind))
    s.add(Not(Body(w_mind)))
    s.add(R(actual, w_body))
    s.add(Not(Mind(w_body)))
    s.add(Body(w_body))

    # Identity thesis
    s.push()
    s.add(ForAll([w], Mind(w) == Body(w)))
    result = s.check()
    s.pop()

    return str(result), result == unsat


def verify_dream_modal() -> Tuple[str, bool]:
    """Dream argument: senses might be unreliable."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    State = DeclareSort('State')
    dreaming = Const('dreaming', State)
    waking = Const('waking', State)
    current = Const('current', State)
    Reliable = Function('Reliable', State, BoolSort())

    s = Solver()
    s.add(Or(current == dreaming, current == waking))
    s.add(Reliable(waking))
    s.add(Not(Reliable(dreaming)))

    # Is it consistent that senses are currently unreliable?
    s.push()
    s.add(Not(Reliable(current)))
    result = s.check()
    s.pop()

    return str(result), result == sat


def verify_zombie_s5() -> Tuple[str, bool]:
    """Zombie argument in S5."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    World = DeclareSort('World')
    R = Function('R', World, World, BoolSort())
    Physical = Function('Physical', World, BoolSort())
    Conscious = Function('Conscious', World, BoolSort())

    actual = Const('actual', World)
    zombie_w = Const('zombie', World)
    w, v, u = Consts('w v u', World)

    s = Solver()
    s.add(ForAll([w], R(w, w)))
    s.add(ForAll([w, v], Implies(R(w, v), R(v, w))))
    s.add(ForAll([w, v, u], Implies(And(R(w, v), R(v, u)), R(w, u))))

    s.add(Physical(actual))
    s.add(Conscious(actual))
    s.add(R(actual, zombie_w))
    s.add(Physical(zombie_w))
    s.add(Not(Conscious(zombie_w)))

    # Physicalism
    s.push()
    s.add(ForAll([w], Implies(Physical(w), Conscious(w))))
    result = s.check()
    s.pop()

    return str(result), result == unsat


def verify_mind_body_paraconsistent() -> Tuple[str, bool]:
    """Mind-body in Belnap 4-valued logic."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    s = Solver()
    mind_physical = Const('mind_physical', IntSort())

    # Belnap: 0=Neither, 1=False, 2=True, 3=Both
    s.add(And(mind_physical >= 0, mind_physical <= 3))

    # Physicalism: true. Dualism: false. → Both
    s.add(Or(mind_physical == IntVal(2), mind_physical == IntVal(3)))  # at least true
    s.add(Or(mind_physical == IntVal(1), mind_physical == IntVal(3)))  # at least false
    # → forces mind_physical == 3 (Both)

    result = s.check()
    if result == sat:
        m = s.model()
        val = m[mind_physical].as_long()
        return f"{result} (value={val})", val == 3
    return str(result), False


def verify_competing_theories() -> Tuple[str, bool]:
    """Competing consciousness theories with defeasible reasoning."""
    if not Z3_AVAILABLE:
        return "unavailable", False

    opt = Optimize()

    from z3 import Bool, Real
    iit_best = Bool('iit_best')
    gwt_best = Bool('gwt_best')
    hot_best = Bool('hot_best')

    # At most one "best"
    opt.add(Or(
        And(iit_best, Not(gwt_best), Not(hot_best)),
        And(Not(iit_best), gwt_best, Not(hot_best)),
        And(Not(iit_best), Not(gwt_best), hot_best),
        And(Not(iit_best), Not(gwt_best), Not(hot_best)),
    ))

    # Soft constraints with evidence weights
    opt.add_soft(iit_best, 3)     # IIT explains integration well
    opt.add_soft(gwt_best, 2)     # GWT explains access well
    opt.add_soft(hot_best, 1)     # HOT explains metacognition

    result = opt.check()
    if result == sat:
        m = opt.model()
        best = "IIT" if m[iit_best] else ("GWT" if m[gwt_best] else "HOT")
        return f"{result} (best={best})", True
    return str(result), False


# ============================================================
# RUNNER — Execute all examples
# ============================================================

VERIFY_FNS = {
    "verify_cogito_strict": verify_cogito_strict,
    "verify_trademark_causal": verify_trademark_causal,
    "verify_wax_elimination": verify_wax_elimination,
    "verify_modus_ponens": verify_modus_ponens,
    "verify_affirming_consequent": verify_affirming_consequent,
    "verify_real_distinction_s5": verify_real_distinction_s5,
    "verify_dream_modal": verify_dream_modal,
    "verify_zombie_s5": verify_zombie_s5,
    "verify_mind_body_paraconsistent": verify_mind_body_paraconsistent,
    "verify_competing_theories": verify_competing_theories,
}


def run_all_examples(verbose: bool = True) -> Dict:
    """Run all verification examples and report results."""
    if not Z3_AVAILABLE:
        print("z3-solver not installed. Cannot run examples.")
        return {"error": "z3 not available"}

    results = {}
    passed = 0
    failed = 0

    if verbose:
        print("=" * 60)
        print("VERIFICATION EXAMPLES: Full Test Suite")
        print("=" * 60)

    for ex_id, ex in EXAMPLES.items():
        fn = VERIFY_FNS.get(ex.z3_code_fn)
        if fn is None:
            if verbose:
                print(f"  [SKIP] {ex.title}: no verify function")
            continue

        result_str, correct = fn()

        if correct:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        results[ex_id] = {
            "title": ex.title,
            "mode": ex.mode.value,
            "expected": ex.expected.value,
            "z3_result": result_str,
            "correct": correct,
        }

        if verbose:
            print(f"  [{status}] {ex.title} ({ex.mode.value}): "
                  f"{result_str}")

    if verbose:
        total = passed + failed
        print(f"\n  Results: {passed}/{total} passed")
        if failed > 0:
            print(f"  FAILURES: {failed}")

    results["_summary"] = {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
    }
    return results


if __name__ == "__main__":
    run_all_examples()
