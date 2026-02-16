"""
Z3 Formalization Templates for Descartes' Core Arguments.

These templates are referenced by the small model during
formalization tasks and used in SFT training data generation.

Each template returns a Z3 Solver and the verification result,
demonstrating how Cartesian arguments map to formal logic.

Templates:
  - template_cogito(): The Cogito (strict inference)
  - template_real_distinction(): Real Distinction (S5 modal)
  - template_cartesian_circle(): Cartesian Circle (circular dependency)
  - template_trademark(): Trademark Argument (causal adequacy)
  - template_wax(): Wax Argument (elimination)
  - template_dream(): Dream Argument (skeptical)

Usage:
    from descartes_z3 import template_cogito
    solver, result = template_cogito()
    print(f"Cogito verification: {result}")  # UNSAT = valid
"""

from z3 import *


def template_cogito():
    """The Cogito: I think, therefore I am.

    Formalized as a strict deductive inference:
    Doubts(I) -> Thinks(I) -> Exists(I)

    Tests whether Not(Exists(I)) is consistent with the premises.
    UNSAT = Exists(I) is logically entailed = argument is valid.

    Note: This captures the SYLLOGISTIC reading. Descartes himself
    preferred the INTUITION reading (immediate grasp, not inference
    from a general premise).
    """
    Agent = DeclareSort('Agent')
    Thinks = Function('Thinks', Agent, BoolSort())
    Exists = Function('Exists', Agent, BoolSort())
    Doubts = Function('Doubts', Agent, BoolSort())

    I = Const('I', Agent)
    a = Const('a', Agent)
    s = Solver()

    # Premise: I am doubting
    s.add(Doubts(I))
    # Strict rule: doubting entails thinking
    s.add(ForAll([a], Implies(Doubts(a), Thinks(a))))
    # Strict rule: thinking entails existing (the Cogito)
    s.add(ForAll([a], Implies(Thinks(a), Exists(a))))

    # Test: is Not(Exists(I)) consistent with above?
    s.push()
    s.add(Not(Exists(I)))
    result = s.check()  # UNSAT -> Exists(I) is entailed
    s.pop()

    return s, result


def template_real_distinction():
    """The Real Distinction: mind and body are distinct substances.

    Uses S5 modal logic:
    Conceivability -> Possibility -> Distinctness

    The argument proceeds:
    1. I can clearly and distinctly conceive mind without body
    2. What is C&D conceivable, God can create (divine guarantee)
    3. If God can create A without B, they are really distinct
    C. Mind and body are distinct substances

    Tests whether Mind = Body (identity thesis) is consistent
    with the conceivability premises. UNSAT = they are distinct.
    """
    World = DeclareSort('World')
    R = Function('R', World, World, BoolSort())

    Mind = Function('Mind', World, BoolSort())
    Body = Function('Body', World, BoolSort())

    actual = Const('actual', World)
    w_mind_only = Const('w_mind_only', World)
    w_body_only = Const('w_body_only', World)

    w, v, u = Consts('w v u', World)
    s = Solver()

    # S5 frame (reflexive, symmetric, transitive)
    s.add(ForAll([w], R(w, w)))
    s.add(ForAll([w, v], Implies(R(w, v), R(v, w))))
    s.add(ForAll([w, v, u], Implies(And(R(w, v), R(v, u)), R(w, u))))

    # Actual world: both mind and body
    s.add(Mind(actual))
    s.add(Body(actual))

    # Conceivability -> accessible worlds where they come apart
    # World where mind exists without body
    s.add(R(actual, w_mind_only))
    s.add(Mind(w_mind_only))
    s.add(Not(Body(w_mind_only)))

    # World where body exists without mind
    s.add(R(actual, w_body_only))
    s.add(Not(Mind(w_body_only)))
    s.add(Body(w_body_only))

    # Test: is Mind = Body (identity thesis) consistent?
    s.push()
    s.add(ForAll([w], Mind(w) == Body(w)))  # Identity thesis
    result = s.check()  # UNSAT -> they are distinct
    s.pop()

    return s, result


def template_cartesian_circle():
    """The Cartesian Circle: is Descartes' reasoning circular?

    Formalizes: C&D perception -> God exists -> C&D perception reliable

    Arnauld's objection (Fourth Objections):
    Descartes uses C&D perception to prove God, then uses God
    to validate C&D perception. This is circular.

    Descartes' defense: present C&D perception is self-guaranteeing;
    God is needed only for MEMORY of past demonstrations.

    Returns:
      solver, result_without (Arnauld: SAT = circle doesn't force
        reliability), result_with_present (Descartes: UNSAT =
        present-perception axiom makes reliability hold)
    """
    CDP_Reliable = Bool('CDP_Reliable')
    God_Exists = Bool('God_Exists')
    God_Not_Deceiver = Bool('God_Not_Deceiver')
    CDP_Used = Bool('CDP_Used_For_God')

    s = Solver()

    # Descartes' argument structure:
    # 1. C&D perception used to prove God exists
    s.add(Implies(CDP_Reliable, God_Exists))
    s.add(CDP_Used)

    # 2. God's existence validates C&D perception
    s.add(Implies(God_Exists, God_Not_Deceiver))
    s.add(Implies(God_Not_Deceiver, CDP_Reliable))

    # This creates: CDP_Reliable -> God_Exists -> CDP_Reliable
    # Arnauld's objection: without independent support,
    # CDP_Reliable is ungrounded
    s.push()
    s.add(Not(CDP_Reliable))
    result_without = s.check()  # SAT -> circle doesn't force reliability
    s.pop()

    # Descartes' defense: present perception is self-guaranteeing
    Present_CDP = Bool('Present_CDP')
    s.add(Present_CDP)
    s.add(Implies(Present_CDP, CDP_Reliable))

    s.push()
    s.add(Not(CDP_Reliable))
    result_with_present = s.check()  # UNSAT -> with present-perception,
                                     # reliability holds
    s.pop()

    return s, result_without, result_with_present


def template_trademark():
    """The Trademark Argument (Meditation III): God's existence
    from the idea of God.

    Key premises:
    1. I have an idea of an infinite, perfect being
    2. Causal adequacy: cause must have >= reality as effect
    3. Objective reality of idea of God = infinite
    4. I am finite
    C. Only an infinite being could cause this idea -> God exists

    Tests whether a finite cause could produce an idea with
    infinite objective reality. UNSAT = it cannot -> God must exist.
    """
    Entity = DeclareSort('Entity')
    Reality = IntSort()

    formal_reality = Function('FormalReality', Entity, Reality)
    objective_reality = Function('ObjectiveReality', Entity, Reality)
    causes = Function('Causes', Entity, Entity, BoolSort())

    me = Const('me', Entity)
    god = Const('god', Entity)
    idea_of_god = Const('idea_of_god', Entity)
    a, b = Consts('a b', Entity)

    s = Solver()

    # I am finite (formal reality = finite level)
    s.add(formal_reality(me) == 2)  # Finite substance

    # Idea of God has infinite objective reality
    s.add(objective_reality(idea_of_god) == 4)  # Infinite

    # Causal adequacy: cause's formal reality >= effect's objective reality
    s.add(ForAll([a, b], Implies(
        causes(a, b),
        formal_reality(a) >= objective_reality(b)
    )))

    # I cause (have) the idea of God
    s.add(causes(me, idea_of_god))

    # Test: is this consistent? (Can I cause an idea with
    # objective reality > my formal reality?)
    result_direct = s.check()  # UNSAT -> I cannot cause it alone

    # Now add God as alternative cause
    s2 = Solver()
    s2.add(formal_reality(god) == 4)  # God is infinite
    s2.add(objective_reality(idea_of_god) == 4)
    s2.add(ForAll([a, b], Implies(
        causes(a, b),
        formal_reality(a) >= objective_reality(b)
    )))
    s2.add(causes(god, idea_of_god))

    result_with_god = s2.check()  # SAT -> God can cause it

    return s, result_direct, result_with_god


def template_wax():
    """The Wax Argument (Meditation II): bodies are known
    through intellect, not senses.

    Argument from elimination:
    1. Wax changes all sensory properties (color, smell, shape...)
    2. Yet I judge it to be the same wax
    3. This judgment is NOT from senses (properties all changed)
    4. This judgment is NOT from imagination (infinite configurations)
    C. Bodies are known through intellect (understanding) alone

    Formalized as elimination of knowledge sources.
    """
    Source = DeclareSort('KnowledgeSource')
    senses = Const('senses', Source)
    imagination = Const('imagination', Source)
    intellect = Const('intellect', Source)

    # Knowledge source for "same wax" judgment
    knowledge_source = Const('wax_knowledge_source', Source)

    s = Solver()

    # There are exactly three possible sources
    s.add(Or(
        knowledge_source == senses,
        knowledge_source == imagination,
        knowledge_source == intellect
    ))

    # Senses eliminated: all sensory properties changed
    s.add(knowledge_source != senses)

    # Imagination eliminated: infinite possible configurations
    s.add(knowledge_source != imagination)

    # What remains?
    result = s.check()  # SAT
    if result == sat:
        m = s.model()
        remaining = m[knowledge_source]  # Should be intellect
        return s, result, remaining
    return s, result, None


def template_dream():
    """The Dream Argument (Meditation I): skepticism about
    sense experience.

    P1: I have had experiences in dreams indistinguishable
        from waking experience
    P2: I have no certain criterion to distinguish dreaming
        from waking right now
    C: Any current sense experience could be a dream
    -> Sense experience is not a reliable basis for certainty

    Note: This is WEAKER than the Evil Genius argument.
    The Dream Argument only undermines sense certainty.
    The Evil Genius undermines even mathematical truths.
    """
    State = DeclareSort('State')
    dreaming = Const('dreaming', State)
    waking = Const('waking', State)
    current = Const('current_state', State)

    Reliable = Function('SenseReliable', State, BoolSort())
    Distinguishable = Function('Distinguishable', State, State, BoolSort())

    s = Solver()

    # States: either dreaming or waking
    s.add(Or(current == dreaming, current == waking))

    # P1: Senses are reliable only in waking state
    s.add(Reliable(waking))
    s.add(Not(Reliable(dreaming)))

    # P2: Cannot distinguish current state
    s.add(Not(Distinguishable(current, dreaming)))
    s.add(Not(Distinguishable(current, waking)))

    # Can we prove senses are reliable RIGHT NOW?
    s.push()
    s.add(Not(Reliable(current)))
    result = s.check()  # SAT -> consistent that senses are unreliable
    s.pop()             # (we could be dreaming)

    return s, result


# ============================================================
# VERIFICATION RUNNER
# ============================================================

def verify_all():
    """Run all Descartes argument formalizations and report results."""

    print("=" * 60)
    print("Z3 VERIFICATION: Descartes' Core Arguments")
    print("=" * 60)

    # 1. Cogito
    _, result = template_cogito()
    status = "VALID" if result == unsat else "INVALID"
    print(f"\n[1] Cogito (I think, therefore I am)")
    print(f"    Result: {result} -> Argument is {status}")
    print(f"    (UNSAT means the negation of the conclusion is "
          f"inconsistent with premises)")

    # 2. Real Distinction
    _, result = template_real_distinction()
    status = "VALID" if result == unsat else "INVALID"
    print(f"\n[2] Real Distinction (mind ≠ body)")
    print(f"    Result: {result} -> Identity thesis is "
          f"{'inconsistent' if result == unsat else 'consistent'}")

    # 3. Cartesian Circle
    _, result_without, result_with = template_cartesian_circle()
    print(f"\n[3] Cartesian Circle")
    print(f"    Arnauld's objection (without present-perception): "
          f"{result_without}")
    print(f"      -> Circle {'IS' if result_without == sat else 'is NOT'} "
          f"vicious (reliability not forced)")
    print(f"    Descartes' defense (with present-perception): "
          f"{result_with}")
    print(f"      -> Present perception "
          f"{'DOES' if result_with == unsat else 'does NOT'} "
          f"break the circle")

    # 4. Trademark Argument
    _, result_direct, result_with_god = template_trademark()
    print(f"\n[4] Trademark Argument (God from idea of God)")
    print(f"    Can I (finite) cause idea of infinite? {result_direct}")
    print(f"      -> {'NO' if result_direct == unsat else 'YES'} "
          f"(needs infinite cause)")
    print(f"    Can God (infinite) cause idea of infinite? "
          f"{result_with_god}")
    print(f"      -> {'YES' if result_with_god == sat else 'NO'}")

    # 5. Wax Argument
    _, result, source = template_wax()
    print(f"\n[5] Wax Argument (elimination of knowledge sources)")
    print(f"    Result: {result}")
    print(f"    Remaining source: {source}")
    print(f"    -> Bodies known through intellect alone")

    # 6. Dream Argument
    _, result = template_dream()
    print(f"\n[6] Dream Argument (sense skepticism)")
    print(f"    Can senses be unreliable? {result}")
    print(f"    -> {'YES' if result == sat else 'NO'} "
          f"(consistent that we're dreaming)")

    print(f"\n{'=' * 60}")
    print(f"All 6 arguments verified.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    verify_all()
