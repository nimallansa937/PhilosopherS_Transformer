"""
Zombie Argument Decomposition — Full ASPIC+ + Z3 analysis.

The zombie argument (Chalmers 1996) is structurally parallel to
Descartes' Real Distinction argument. Both use the inference:

    Conceivability → Possibility → Metaphysical conclusion

Descartes:
    Conceive(mind without body) → Possible(mind without body)
                                → Mind ≠ Body (Real Distinction)

Chalmers:
    Conceive(physical duplicate without consciousness)
        → Possible(zombie world) → ¬Physicalism

This module:
1. Decomposes both arguments into ASPIC+ knowledge bases
2. Maps all historical objections as attacks
3. Provides Z3 formalizations for parallel verification
4. Reveals the structural isomorphism between them

Reference: PHILOSOPHER_ENGINE_V3_UNIFIED_ARCHITECTURE.md, Layer 2
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from reasoning_core.argumentation.aspic_engine import (
    ASPICKnowledgeBase, Rule, RuleType, Attack, AttackType, Argument,
)

try:
    from z3 import (
        DeclareSort, Function, BoolSort, Const, Consts,
        ForAll, Exists, Implies, And, Or, Not,
        Solver, sat, unsat,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ============================================================
# PART 1: ASPIC+ Decomposition
# ============================================================

def build_zombie_argument_kb() -> ASPICKnowledgeBase:
    """Full zombie argument + all major objections as ASPIC+ KB.

    Returns an ASPICKnowledgeBase with:
      - The core zombie argument (3 rules)
      - 6 historical objections as attacking arguments
      - Preference orderings reflecting philosophical consensus
    """
    kb = ASPICKnowledgeBase()

    # ---- Core Zombie Argument ----

    # P1 → P2 (conceivability → possibility)
    kb.add_rule(Rule(
        "zombie_r1",
        ["Conceivable(physical_dup_no_consciousness)"],
        "MetaphysicallyPossible(zombie_world)",
        RuleType.DEFEASIBLE,
        "Zombie conceivability implies possibility (CP thesis)",
        "Chalmers 1996, ch. 4",
    ))

    # P2 → P3 (possibility → anti-supervenience)
    kb.add_rule(Rule(
        "zombie_r2",
        ["MetaphysicallyPossible(zombie_world)"],
        "Not(Supervenes(consciousness, physical))",
        RuleType.STRICT,
        "If zombie world possible, consciousness doesn't supervene",
        "Chalmers 1996, ch. 4",
    ))

    # P3 → C (anti-supervenience → anti-physicalism)
    kb.add_rule(Rule(
        "zombie_r3",
        ["Not(Supervenes(consciousness, physical))"],
        "Not(Physicalism)",
        RuleType.STRICT,
        "If consciousness doesn't supervene on physical, physicalism false",
        "Chalmers 1996, ch. 4",
    ))

    kb.build_argument(
        "zombie_main", "Not(Physicalism)",
        ["zombie_r1", "zombie_r2", "zombie_r3"],
        premises=["Conceivable(physical_dup_no_consciousness)"],
    )

    # ---- Objection 1: Type-A Physicalism (Dennett) ----
    # Zombies aren't conceivable — consciousness is functional

    kb.add_rule(Rule(
        "obj1_r1",
        ["Consciousness_is_functional"],
        "Not(Conceivable(physical_dup_no_consciousness))",
        RuleType.DEFEASIBLE,
        "If consciousness = function, functional duplicate IS conscious",
        "Dennett 1991, 1995",
    ))
    kb.build_argument(
        "dennett_objection",
        "Not(Conceivable(physical_dup_no_consciousness))",
        ["obj1_r1"],
        premises=["Consciousness_is_functional"],
    )
    kb.add_attack(Attack(
        "dennett_objection", "zombie_main",
        AttackType.UNDERMINE,
        "Attacks P1: denies conceivability of zombies",
    ))

    # ---- Objection 2: Type-B Physicalism (Block/Stalnaker) ----
    # Conceivability ≠ Possibility (epistemic gap ≠ metaphysical gap)

    kb.add_rule(Rule(
        "obj2_r1",
        ["Conceivability_not_metaphysical_possibility"],
        "CP_thesis_invalid",
        RuleType.DEFEASIBLE,
        "Epistemic possibility does not entail metaphysical possibility",
        "Block & Stalnaker 1999",
    ))
    kb.build_argument(
        "type_b_objection",
        "CP_thesis_invalid",
        ["obj2_r1"],
        premises=["Conceivability_not_metaphysical_possibility"],
    )
    kb.add_attack(Attack(
        "type_b_objection", "zombie_main",
        AttackType.UNDERCUT,
        "Attacks rule zombie_r1: denies CP inference",
    ))

    # ---- Objection 3: Russellian Monism (Stoljar/Chalmers' later view) ----
    # Physics describes structure, not intrinsic nature.
    # Zombies show gap between structure and consciousness,
    # but intrinsic nature (proto-phenomenal) could ground both.

    kb.add_rule(Rule(
        "obj3_r1",
        ["Physics_describes_structure_only",
         "Intrinsic_nature_grounds_consciousness"],
        "Zombie_world_not_genuinely_physical_duplicate",
        RuleType.DEFEASIBLE,
        "Zombie lacks intrinsic nature, so not a true physical duplicate",
        "Stoljar 2001; Chalmers 2010",
    ))
    kb.build_argument(
        "russellian_objection",
        "Zombie_world_not_genuinely_physical_duplicate",
        ["obj3_r1"],
        premises=[
            "Physics_describes_structure_only",
            "Intrinsic_nature_grounds_consciousness",
        ],
    )
    kb.add_attack(Attack(
        "russellian_objection", "zombie_main",
        AttackType.UNDERMINE,
        "Attacks P1: zombie isn't really a physical duplicate",
    ))

    # ---- Objection 4: A Posteriori Necessity (Kripke/Jackson) ----
    # Water ≈ H2O is necessary but a posteriori.
    # Consciousness ≈ physical could be the same:
    # conceivable that water ≠ H2O, yet impossible.

    kb.add_rule(Rule(
        "obj4_r1",
        ["Identity_can_be_necessary_but_aposteriori"],
        "Conceivable_but_impossible(zombie_world)",
        RuleType.DEFEASIBLE,
        "Zombie world may be conceivable but metaphysically impossible",
        "Kripke 1980; Jackson response",
    ))
    kb.build_argument(
        "aposteriori_necessity_objection",
        "Conceivable_but_impossible(zombie_world)",
        ["obj4_r1"],
        premises=["Identity_can_be_necessary_but_aposteriori"],
    )
    kb.add_attack(Attack(
        "aposteriori_necessity_objection", "zombie_main",
        AttackType.UNDERCUT,
        "Attacks zombie_r1: conceivability doesn't entail possibility",
    ))

    # ---- Objection 5: Phenomenal Concept Strategy (Loar/Papineau) ----
    # Our phenomenal concepts make it SEEM like zombies are
    # conceivable, but this is an illusion of our conceptual
    # apparatus, not a real metaphysical possibility.

    kb.add_rule(Rule(
        "obj5_r1",
        ["Phenomenal_concepts_create_illusion_of_gap"],
        "Conceivability_illusory",
        RuleType.DEFEASIBLE,
        "Phenomenal concepts explain away the conceivability intuition",
        "Loar 1997; Papineau 2002",
    ))
    kb.build_argument(
        "phenomenal_concept_objection",
        "Conceivability_illusory",
        ["obj5_r1"],
        premises=["Phenomenal_concepts_create_illusion_of_gap"],
    )
    kb.add_attack(Attack(
        "phenomenal_concept_objection", "zombie_main",
        AttackType.UNDERMINE,
        "Attacks P1: conceivability is an illusion",
    ))

    # ---- Objection 6: Modal Rationalism Failure (Yablo) ----
    # Ideal conceivability ≠ primary possibility.
    # Even ideal rational reflection can't access all
    # metaphysically relevant facts.

    kb.add_rule(Rule(
        "obj6_r1",
        ["Ideal_conceivability_not_perfect_modal_guide"],
        "CP_restricted",
        RuleType.DEFEASIBLE,
        "Even ideal conceivability doesn't track all modal facts",
        "Yablo 1993, 1999",
    ))
    kb.build_argument(
        "modal_rationalism_objection",
        "CP_restricted",
        ["obj6_r1"],
        premises=["Ideal_conceivability_not_perfect_modal_guide"],
    )
    kb.add_attack(Attack(
        "modal_rationalism_objection", "zombie_main",
        AttackType.UNDERCUT,
        "Attacks zombie_r1: limits CP thesis scope",
    ))

    # ---- Preference Orderings ----
    # Chalmers' defense: conceivability thesis is stronger than
    # the individual objections (for argument's internal purposes)
    kb.add_preference("zombie_main", "dennett_objection")
    # Type-B objection is considered the strongest by consensus
    kb.add_preference("type_b_objection", "zombie_main")

    return kb


# ============================================================
# PART 2: Structural Isomorphism with Real Distinction
# ============================================================

@dataclass
class StructuralMapping:
    """Formal mapping between two arguments showing isomorphism."""
    source_arg: str
    target_arg: str
    premise_map: Dict[str, str]    # source premise -> target premise
    rule_map: Dict[str, str]       # source rule -> target rule
    conclusion_map: Dict[str, str]
    shared_structure: str          # Description of common schema
    disanalogies: List[str]        # Where the mapping breaks down


def map_zombie_to_real_distinction() -> StructuralMapping:
    """Show the structural parallel between zombie argument
    and Descartes' Real Distinction.

    Both instantiate the schema:
        Conceive(A without B)
            → Possible(A without B)  [via CP thesis]
            → A ≠ B                  [via separability]

    Descartes: A = mind, B = body
    Chalmers: A = physical structure, B = consciousness
    """
    return StructuralMapping(
        source_arg="real_distinction",
        target_arg="zombie_argument",
        premise_map={
            "CDP(mind_without_body)":
                "Conceivable(physical_dup_no_consciousness)",
            "God_guarantees_CDP_to_possibility":
                "Ideal_conceivability_entails_possibility",
        },
        rule_map={
            "rd_r1 (conceivability → possibility)":
                "zombie_r1 (conceivability → possibility)",
            "rd_r2 (separability → distinction)":
                "zombie_r2 (possibility → anti-supervenience)",
        },
        conclusion_map={
            "Distinct(mind, body)":
                "Not(Physicalism)",
        },
        shared_structure=(
            "SCHEMA: Conceive(A-without-B) → Possible(A-without-B) "
            "→ MetaphysicalConclusion(A, B)\n"
            "Both rely on conceivability-possibility (CP) thesis.\n"
            "Both are vulnerable to attacks on the CP link.\n"
            "Arnauld's objection to Descartes parallels "
            "Type-B physicalist objection to Chalmers."
        ),
        disanalogies=[
            "Descartes requires divine guarantee for CP; "
            "Chalmers uses modal rationalism instead.",
            "Descartes' conclusion is about substance identity; "
            "Chalmers' is about supervenience failure.",
            "Descartes assumes real conceiving (C&D perception); "
            "Chalmers distinguishes positive/negative conceivability.",
            "The zombie argument targets physicalism generally; "
            "the Real Distinction targets only mind-body identity.",
            "Descartes' argument is embedded in a theistic framework; "
            "Chalmers' is naturalistically motivated.",
        ],
    )


# ============================================================
# PART 3: Z3 Formalizations
# ============================================================

def z3_zombie_argument() -> Tuple:
    """Formalize the zombie argument in S5 modal logic.

    Returns (solver, result, countermodel_result):
      solver: Z3 Solver with zombie argument premises
      result: 'unsat' if physicalism is refuted
      countermodel_result: 'sat' if Type-B objection is viable
    """
    if not Z3_AVAILABLE:
        raise ImportError("z3-solver required: pip install z3-solver")

    World = DeclareSort('World')
    R = Function('Accessible', World, World, BoolSort())

    # Properties that hold at worlds
    Physical = Function('PhysicalComplete', World, BoolSort())
    Conscious = Function('Conscious', World, BoolSort())

    actual = Const('actual', World)
    zombie_w = Const('zombie_world', World)
    w, v, u = Consts('w v u', World)

    s = Solver()

    # S5 frame axioms
    s.add(ForAll([w], R(w, w)))                          # Reflexive
    s.add(ForAll([w, v], Implies(R(w, v), R(v, w))))     # Symmetric
    s.add(ForAll([w, v, u],
                 Implies(And(R(w, v), R(v, u)), R(w, u))))  # Transitive

    # Actual world: physical + conscious
    s.add(Physical(actual))
    s.add(Conscious(actual))

    # P1: Zombie world is conceivable → accessible
    s.add(R(actual, zombie_w))
    s.add(Physical(zombie_w))
    s.add(Not(Conscious(zombie_w)))

    # Physicalism := necessarily, physical → conscious
    # ∀w: Physical(w) → Conscious(w)
    physicalism = ForAll([w], Implies(Physical(w), Conscious(w)))

    # Test 1: Is physicalism consistent with zombie premises?
    s.push()
    s.add(physicalism)
    result = s.check()  # UNSAT → physicalism refuted
    s.pop()

    # Test 2: Type-B countermodel — maybe conceivability ≠ possibility
    # Remove zombie world accessibility, check if physicalism survives
    s2 = Solver()
    s2.add(ForAll([w], R(w, w)))
    s2.add(Physical(actual))
    s2.add(Conscious(actual))
    # No zombie world in the model
    s2.add(physicalism)
    countermodel_result = s2.check()  # SAT → physicalism consistent
                                       # (without zombie world)

    return s, result, countermodel_result


def z3_real_distinction() -> Tuple:
    """Formalize Descartes' Real Distinction for parallel comparison.

    Returns (solver, result):
      solver: Z3 Solver with Real Distinction premises
      result: 'unsat' if identity thesis (Mind=Body) is refuted
    """
    if not Z3_AVAILABLE:
        raise ImportError("z3-solver required: pip install z3-solver")

    World = DeclareSort('World')
    R = Function('Accessible', World, World, BoolSort())

    Mind = Function('MindExists', World, BoolSort())
    Body = Function('BodyExists', World, BoolSort())

    actual = Const('actual', World)
    w_mind = Const('w_mind_only', World)
    w_body = Const('w_body_only', World)
    w, v, u = Consts('w v u', World)

    s = Solver()

    # S5 frame
    s.add(ForAll([w], R(w, w)))
    s.add(ForAll([w, v], Implies(R(w, v), R(v, w))))
    s.add(ForAll([w, v, u],
                 Implies(And(R(w, v), R(v, u)), R(w, u))))

    # Actual world: both
    s.add(Mind(actual))
    s.add(Body(actual))

    # Conceivability: worlds where they come apart
    s.add(R(actual, w_mind))
    s.add(Mind(w_mind))
    s.add(Not(Body(w_mind)))

    s.add(R(actual, w_body))
    s.add(Not(Mind(w_body)))
    s.add(Body(w_body))

    # Identity thesis: Mind ≡ Body at all worlds
    identity = ForAll([w], Mind(w) == Body(w))

    s.push()
    s.add(identity)
    result = s.check()  # UNSAT → identity refuted → distinction holds
    s.pop()

    return s, result


def verify_structural_parallel() -> Dict:
    """Run both formalizations and compare results.

    Demonstrates that both arguments have identical logical
    structure despite different domains.
    """
    if not Z3_AVAILABLE:
        return {"error": "z3-solver not installed"}

    _, zombie_result, zombie_counter = z3_zombie_argument()
    _, rd_result = z3_real_distinction()

    mapping = map_zombie_to_real_distinction()

    return {
        "zombie_argument": {
            "physicalism_refuted": str(zombie_result) == "unsat",
            "z3_result": str(zombie_result),
            "type_b_viable": str(zombie_counter) == "sat",
        },
        "real_distinction": {
            "identity_refuted": str(rd_result) == "unsat",
            "z3_result": str(rd_result),
        },
        "structural_parallel": {
            "both_use_CP_thesis": True,
            "both_vulnerable_to_CP_attack": True,
            "shared_schema": mapping.shared_structure,
            "disanalogies": mapping.disanalogies,
            "premise_mapping": mapping.premise_map,
        },
    }


# ============================================================
# PART 4: Extended Arguments (Chinese Room, Knowledge Argument)
# ============================================================

def build_chinese_room_kb() -> ASPICKnowledgeBase:
    """Searle's Chinese Room argument decomposed in ASPIC+.

    Core argument:
        Syntax ≠ Semantics
        → Computation = Syntax manipulation
        → Computation alone doesn't produce understanding
        → Strong AI is false
    """
    kb = ASPICKnowledgeBase()

    kb.add_rule(Rule(
        "cr_r1",
        ["Syntax_not_sufficient_for_semantics"],
        "Computation_is_syntax",
        RuleType.STRICT,
        "Computational operations are purely syntactic",
        "Searle 1980",
    ))
    kb.add_rule(Rule(
        "cr_r2",
        ["Computation_is_syntax", "Mind_requires_semantics"],
        "Computation_insufficient_for_mind",
        RuleType.STRICT,
        "Syntax alone doesn't produce understanding",
        "Searle 1980",
    ))
    kb.build_argument(
        "chinese_room", "Computation_insufficient_for_mind",
        ["cr_r1", "cr_r2"],
        premises=[
            "Syntax_not_sufficient_for_semantics",
            "Mind_requires_semantics",
        ],
    )

    # Systems Reply (Berkeley/Dennett)
    kb.add_rule(Rule(
        "sr_r1",
        ["System_as_whole_may_understand"],
        "Searle_in_room_not_the_right_level",
        RuleType.DEFEASIBLE,
        "The system (room + rules + person) might understand",
        "Searle 1980 responses",
    ))
    kb.build_argument(
        "systems_reply", "Searle_in_room_not_the_right_level",
        ["sr_r1"],
        premises=["System_as_whole_may_understand"],
    )
    kb.add_attack(Attack(
        "systems_reply", "chinese_room",
        AttackType.UNDERMINE,
        "Wrong level of analysis — system vs. person",
    ))

    # Robot Reply
    kb.add_rule(Rule(
        "rr_r1",
        ["Embodied_interaction_grounds_semantics"],
        "Add_body_to_room_gets_understanding",
        RuleType.DEFEASIBLE,
        "Grounded symbol manipulation could produce semantics",
        "Searle 1980 responses",
    ))
    kb.build_argument(
        "robot_reply", "Add_body_to_room_gets_understanding",
        ["rr_r1"],
        premises=["Embodied_interaction_grounds_semantics"],
    )
    kb.add_attack(Attack(
        "robot_reply", "chinese_room",
        AttackType.UNDERCUT,
        "Attacks the inference rule: adds embodiment condition",
    ))

    return kb


def build_knowledge_argument_kb() -> ASPICKnowledgeBase:
    """Jackson's Knowledge Argument (Mary's Room) in ASPIC+.

    Core argument:
        Mary knows all physical facts about color
        → Mary doesn't know what red looks like
        → There are non-physical facts
        → Physicalism is false
    """
    kb = ASPICKnowledgeBase()

    kb.add_rule(Rule(
        "ka_r1",
        ["Mary_knows_all_physical_facts",
         "Mary_learns_something_new_on_release"],
        "Exists_non_physical_fact",
        RuleType.DEFEASIBLE,
        "New knowledge implies non-physical facts",
        "Jackson 1982, 1986",
    ))
    kb.add_rule(Rule(
        "ka_r2",
        ["Exists_non_physical_fact"],
        "Not(Physicalism)",
        RuleType.STRICT,
        "Non-physical facts refute physicalism",
        "Jackson 1982",
    ))
    kb.build_argument(
        "knowledge_argument", "Not(Physicalism)",
        ["ka_r1", "ka_r2"],
        premises=[
            "Mary_knows_all_physical_facts",
            "Mary_learns_something_new_on_release",
        ],
    )

    # Ability Hypothesis (Lewis/Nemirow)
    kb.add_rule(Rule(
        "ah_r1",
        ["Knowing_what_is_ability_not_knowledge"],
        "Mary_gains_ability_not_new_fact",
        RuleType.DEFEASIBLE,
        "Mary gains know-how, not propositional knowledge",
        "Lewis 1988; Nemirow 1990",
    ))
    kb.build_argument(
        "ability_hypothesis", "Mary_gains_ability_not_new_fact",
        ["ah_r1"],
        premises=["Knowing_what_is_ability_not_knowledge"],
    )
    kb.add_attack(Attack(
        "ability_hypothesis", "knowledge_argument",
        AttackType.UNDERMINE,
        "Mary doesn't learn a new fact, just a new ability",
    ))

    # Acquaintance Hypothesis (Conee)
    kb.add_rule(Rule(
        "aq_r1",
        ["Acquaintance_is_third_type_of_knowledge"],
        "Mary_gains_acquaintance_not_propositional_knowledge",
        RuleType.DEFEASIBLE,
        "Knowledge by acquaintance ≠ propositional knowledge",
        "Conee 1994",
    ))
    kb.build_argument(
        "acquaintance_hypothesis",
        "Mary_gains_acquaintance_not_propositional_knowledge",
        ["aq_r1"],
        premises=["Acquaintance_is_third_type_of_knowledge"],
    )
    kb.add_attack(Attack(
        "acquaintance_hypothesis", "knowledge_argument",
        AttackType.UNDERMINE,
        "Distinguishes types of knowledge",
    ))

    return kb


# ============================================================
# RUNNER
# ============================================================

def run_analysis():
    """Run full zombie argument analysis with Z3 verification."""
    print("=" * 60)
    print("ZOMBIE ARGUMENT: Full Decomposition & Verification")
    print("=" * 60)

    # 1. ASPIC+ structure
    kb = build_zombie_argument_kb()
    print(f"\n[1] ASPIC+ Knowledge Base: {kb}")
    print(f"    Arguments: {list(kb.arguments.keys())}")
    print(f"    Attacks:   {len(kb.attacks)}")

    grounded = kb.grounded_extensions()
    print(f"    Grounded extension: {grounded}")

    # 2. Structural mapping
    mapping = map_zombie_to_real_distinction()
    print(f"\n[2] Structural Isomorphism")
    print(f"    Schema: {mapping.shared_structure.split(chr(10))[0]}")
    print(f"    Disanalogies: {len(mapping.disanalogies)}")

    # 3. Z3 verification
    if Z3_AVAILABLE:
        results = verify_structural_parallel()
        print(f"\n[3] Z3 Verification")
        print(f"    Zombie: physicalism refuted = "
              f"{results['zombie_argument']['physicalism_refuted']}")
        print(f"    Real Distinction: identity refuted = "
              f"{results['real_distinction']['identity_refuted']}")
        print(f"    Type-B viable: "
              f"{results['zombie_argument']['type_b_viable']}")
    else:
        print("\n[3] Z3 not available — skipping verification")

    # 4. Related arguments
    cr_kb = build_chinese_room_kb()
    ka_kb = build_knowledge_argument_kb()
    print(f"\n[4] Related Arguments")
    print(f"    Chinese Room: {cr_kb}")
    print(f"    Knowledge Argument: {ka_kb}")

    print(f"\n{'=' * 60}")
    print("Analysis complete.")


if __name__ == "__main__":
    run_analysis()
