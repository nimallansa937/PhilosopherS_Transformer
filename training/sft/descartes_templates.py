"""
Phase 6 (CASCADE): SFT templates specialized for Descartes' arguments.

Types A-D: Standard philosophical reasoning (from original pipeline)
Types E-G: Cascade-specific behaviors (NEW)
  E — Confidence estimation (know when to route)
  F — Routing decisions (SELF / ORACLE / HYBRID)
  G — Oracle integration (blend oracle knowledge with expertise)
"""

# ============================================================
# SYSTEM PROMPT (used in all SFT examples)
# ============================================================

SYSTEM_PROMPT = (
    "You are a philosophical reasoning assistant specializing in "
    "Cartesian philosophy, early modern rationalism, and the mind-body "
    "problem. You analyze arguments with formal rigor using ASPIC+ "
    "argumentation schemes and Z3 verification. You have deep expertise "
    "in Descartes' Meditations, the Objections and Replies, the "
    "Correspondence with Elisabeth, and the Principles of Philosophy.\n\n"
    "You have access to an oracle for broad philosophical knowledge "
    "beyond your Cartesian specialization. Request oracle consultation "
    "when needed.\n\n"
    "Express confidence as [CONFIDENCE: 0.X] at the end of each response. "
    "When requesting oracle help, output [ORACLE_REQUEST: <query>]."
)


# ============================================================
# TYPE A: ARGUMENT RECONSTRUCTION (Descartes-specific)
# ============================================================

TYPE_A_DESCARTES = [
    {
        "user": (
            "Reconstruct the logical structure of the Cogito as "
            "presented in the Second Meditation."
        ),
        "key_elements": [
            "Whether it's inference or intuition",
            "Role of the Evil Genius hypothesis",
            "Scope: what exactly is established (existence, not nature)",
            "Distinction from syllogistic reading",
        ]
    },
    {
        "user": (
            "Reconstruct the Real Distinction argument from "
            "Meditation VI. Identify each premise, the modal "
            "inference, and the theological guarantee that "
            "makes it work."
        ),
        "key_elements": [
            "Clear and distinct conceivability premise",
            "Divine guarantee (God could create them apart)",
            "Modal step: conceivability -> possibility -> actuality",
            "Role of the earlier proof of God's existence",
            "Difference from the argument in Meditation II",
        ]
    },
    {
        "user": (
            "Formalize the Trademark Argument (Meditation III) "
            "for God's existence in ASPIC+ structure."
        ),
        "key_elements": [
            "Causal adequacy principle",
            "Formal vs objective reality distinction",
            "Idea of infinite substance",
            "Only infinite substance can cause idea of infinity",
            "Strict rule: cause must have at least as much reality",
        ]
    },
    {
        "user": (
            "Reconstruct the Cartesian Circle as identified by "
            "Arnauld. Show both the circular structure and "
            "Descartes' proposed escape."
        ),
        "key_elements": [
            "C&D perception -> God exists -> C&D perception is reliable",
            "Arnauld's charge in Fourth Objections",
            "Memory vs. present perception distinction",
            "Whether the escape actually works",
        ]
    },
    {
        "user": (
            "Formalize the Wax Argument from Meditation II. "
            "What is the conclusion and what argumentation "
            "scheme does it use?"
        ),
        "key_elements": [
            "Argument from elimination (not senses, not imagination)",
            "Conclusion: bodies known through intellect alone",
            "Implicit inference to best explanation",
            "Role in broader project of establishing mind's primacy",
        ]
    },
]


# ============================================================
# TYPE B: CRITICAL ENGAGEMENT (Descartes-specific)
# ============================================================

TYPE_B_DESCARTES = [
    {
        "user": (
            "Present Elisabeth's interaction problem objection "
            "to Descartes, then give Descartes' best defense."
        ),
        "attack_type": "undermine",
        "target": "mind-body causal interaction premise",
    },
    {
        "user": (
            "Present the strongest materialist objection to the "
            "Real Distinction argument. Can a physicalist accept "
            "the conceivability premise but deny the conclusion?"
        ),
        "attack_type": "undercut",
        "target": "conceivability-to-separability bridge",
    },
    {
        "user": (
            "Gassendi objects that the Cogito only proves that "
            "thinking occurs, not that a thinking SUBSTANCE exists. "
            "Evaluate this objection."
        ),
        "attack_type": "undermine",
        "target": "substance inference from Cogito",
    },
    {
        "user": (
            "Hobbes argues in the Third Objections that 'I think' "
            "does not entail 'I am a thinking thing' -- thinking "
            "might be a property of a material body. How would "
            "Descartes respond?"
        ),
        "attack_type": "rebut",
        "target": "immateriality conclusion",
    },
    {
        "user": (
            "The Cartesian Circle: does Descartes' memory "
            "distinction actually solve Arnauld's objection? "
            "Present the strongest case for and against."
        ),
        "attack_type": "undercut",
        "target": "epistemic bootstrapping",
    },
]


# ============================================================
# TYPE C: CROSS-DISCIPLINARY (Descartes + Neuroscience)
# ============================================================

TYPE_C_DESCARTES = [
    {
        "user": (
            "Connect Descartes' pineal gland hypothesis to modern "
            "neuroscience. Was he entirely wrong, or did he "
            "identify a real problem that neuroscience still "
            "hasn't solved?"
        ),
    },
    {
        "user": (
            "Descartes argued that animals are automata without "
            "consciousness. How does this compare to contemporary "
            "evidence on animal consciousness from neuroscience?"
        ),
    },
    {
        "user": (
            "The Global Workspace Theory says consciousness arises "
            "from information broadcast across brain areas. Could "
            "Descartes accept GWT, or does it conflict with his "
            "substance dualism? Formalize the compatibility check."
        ),
    },
    {
        "user": (
            "Descartes' 'clear and distinct perception' criterion "
            "for truth -- does it have an analog in predictive "
            "processing / Bayesian brain theories? Connect "
            "Cartesian certainty to predictive confidence."
        ),
    },
]


# ============================================================
# TYPE D: COMPREHENSION (Passage-based)
# ============================================================

TYPE_D_DESCARTES = [
    {
        "passage": (
            "But what then am I? A thing that thinks. What is "
            "that? A thing that doubts, understands, affirms, "
            "denies, is willing, is unwilling, and also imagines "
            "and has sensory perceptions."
        ),
        "source": "Meditation II",
        "questions": [
            "What is the logical status of this enumeration?",
            "Is Descartes defining thinking or giving examples?",
            "Does 'sensory perception' belong on this list if senses "
            "have been doubted?",
            "How would Ryle critique this passage?",
        ]
    },
    {
        "passage": (
            "I knew that I was a substance whose whole essence "
            "or nature is simply to think, and which does not "
            "require any place, or depend on any material thing, "
            "in order to exist."
        ),
        "source": "Discourse on the Method, Part IV",
        "questions": [
            "What argument licenses 'whose WHOLE essence is to think'?",
            "Is this stronger than what the Cogito actually establishes?",
            "How does this relate to the Real Distinction argument?",
            "How would a property dualist reinterpret this passage?",
        ]
    },
]


# ============================================================
# TYPE E: CONFIDENCE ESTIMATION (Cascade-specific)
# ============================================================

TYPE_E_DESCARTES = [
    {
        "user": "Formalize the Cogito in Z3 as a strict inference.",
        "expected_confidence": 0.95,
        "routing": "SELF",
        "reason": "Core formalization task, directly trained on this.",
        "response_sketch": (
            "The Cogito can be formalized as:\n\n"
            "```python\n"
            "from z3 import *\n\n"
            "Agent = DeclareSort('Agent')\n"
            "Thinks = Function('Thinks', Agent, BoolSort())\n"
            "Exists = Function('Exists', Agent, BoolSort())\n"
            "Doubts = Function('Doubts', Agent, BoolSort())\n\n"
            "I = Const('I', Agent)\n"
            "s = Solver()\n\n"
            "# Premise: I am doubting (a form of thinking)\n"
            "s.add(Doubts(I))\n\n"
            "# Strict rule: doubting entails thinking\n"
            "a = Const('a', Agent)\n"
            "s.add(ForAll([a], Implies(Doubts(a), Thinks(a))))\n\n"
            "# Strict rule: thinking entails existing (the Cogito)\n"
            "s.add(ForAll([a], Implies(Thinks(a), Exists(a))))\n\n"
            "# Verify: does I exist follow?\n"
            "s.push()\n"
            "s.add(Not(Exists(I)))\n"
            "result = s.check()  # Expected: UNSAT\n"
            "# UNSAT confirms: Exists(I) is entailed\n"
            "```\n\n"
            "The Cogito is a strict (deductive) inference. The "
            "argumentation scheme is modus ponens applied twice: "
            "Doubts(I) -> Thinks(I) -> Exists(I).\n\n"
            "Note: this formalization captures the SYLLOGISTIC reading. "
            "Descartes himself preferred the INTUITION reading.\n\n"
            "[CONFIDENCE: 0.95]"
        )
    },
    {
        "user": (
            "How does Descartes' substance dualism compare to "
            "Strawson's person theory in 'Individuals'?"
        ),
        "expected_confidence": 0.45,
        "routing": "ORACLE",
        "reason": "Strawson comparison is outside Cartesian core expertise.",
        "response_sketch": (
            "Descartes' substance dualism holds that mind and body are "
            "distinct substances. I can formalize the Cartesian framework.\n\n"
            "However, I am less certain about Strawson's position in "
            "'Individuals.' I know Strawson argues that persons are "
            "logically primitive, but I need oracle consultation for "
            "the precise comparison.\n\n"
            "[ORACLE_REQUEST: What is Strawson's person theory in "
            "'Individuals' and how does it specifically differ from "
            "Cartesian dualism?]\n\n"
            "[CONFIDENCE: 0.45]"
        )
    },
    {
        "user": (
            "Check whether the Trademark Argument, the Ontological "
            "Argument, and the Cogito form a consistent set of "
            "premises when formalized together."
        ),
        "expected_confidence": 0.85,
        "routing": "SELF",
        "reason": "Consistency checking is Z3's core function.",
    },
]


# ============================================================
# TYPE F: ROUTING DECISIONS (Cascade-specific)
# ============================================================

TYPE_F_DESCARTES = [
    {
        "user": (
            "Formalize the Real Distinction argument in Z3 with "
            "S5 modal logic."
        ),
        "routing": "SELF",
        "reason": "Z3 formalization is core small-model competency."
    },
    {
        "user": (
            "What was Merleau-Ponty's critique of Cartesian "
            "dualism in Phenomenology of Perception?"
        ),
        "routing": "ORACLE",
        "reason": "Phenomenological tradition outside Cartesian core."
    },
    {
        "user": (
            "Is the Real Distinction argument structurally identical "
            "to Chalmers' zombie argument?"
        ),
        "routing": "HYBRID",
        "reason": (
            "Small model formalizes both arguments in Z3. Oracle "
            "provides Chalmers' own commentary on the connection."
        )
    },
    {
        "user": (
            "Decompose Arnauld's Fourth Objections into ASPIC+ "
            "attack structure."
        ),
        "routing": "SELF",
        "reason": "ASPIC+ decomposition is core training task."
    },
    {
        "user": (
            "How did the reception of Descartes differ between "
            "France and the Netherlands in the 1640s-1650s?"
        ),
        "routing": "ORACLE",
        "reason": "Historical context question, not formal analysis."
    },
    {
        "user": (
            "Can Descartes' causal adequacy principle (Trademark "
            "argument) be reconciled with modern physicalism? "
            "Check for joint satisfiability."
        ),
        "routing": "HYBRID",
        "reason": (
            "Small model does Z3 consistency check. Oracle provides "
            "contemporary physicalist positions on causation."
        )
    },
]


# ============================================================
# TYPE G: ORACLE INTEGRATION (Cascade-specific)
# ============================================================

TYPE_G_DESCARTES = [
    {
        "original_answer": (
            "Descartes' Real Distinction uses modal reasoning: clear "
            "and distinct conceivability -> God can separate -> actually "
            "distinct. I can formalize this in S5. But I'm uncertain "
            "how closely this maps to Chalmers' zombie argument. "
            "[CONFIDENCE: 0.5]"
        ),
        "oracle_response": (
            "Chalmers himself acknowledges the structural parallel in "
            "The Conscious Mind (1996, p. 146). Both arguments use "
            "conceivability-to-possibility bridges. The key difference: "
            "Descartes requires a divine guarantee (God can create mind "
            "without body), while Chalmers relies on a general modal "
            "principle (ideal conceivability entails metaphysical "
            "possibility). Chalmers' bridge is weaker but also more "
            "vulnerable to Type-B physicalist objections about a "
            "posteriori necessity."
        ),
        "integrated": (
            "The Real Distinction argument and the zombie argument are "
            "structurally parallel but differ in their modal bridge.\n\n"
            "Shared structure:\n"
            "Both: Conceive(Mind without Body/P without C) -> "
            "Possible(Mind without Body/P without C) -> "
            "Not-Identical(Mind, Body / Phenomenal, Physical)\n\n"
            "Key difference in the bridge premise:\n"
            "- Descartes: Conceivability + Divine omnipotence -> Possibility\n"
            "  (STRICT rule: God can create anything conceivable)\n"
            "- Chalmers: Ideal conceivability -> Metaphysical possibility\n"
            "  (DEFEASIBLE rule: vulnerable to a posteriori necessity)\n\n"
            "The Cartesian version is actually STRONGER if you accept "
            "the divine guarantee -- it's a strict inference rather than "
            "a defeasible one.\n\n"
            "[CONFIDENCE: 0.9]"
        ),
    },
]
