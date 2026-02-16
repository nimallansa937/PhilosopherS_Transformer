"""
Generate bootstrap questions for meta-learner pre-training.

Distribution:
  40% SELF-answerable (core Cartesian formalization/analysis)
  30% ORACLE-needed (broad philosophy, historical context)
  30% HYBRID (Cartesian core + external knowledge)

This distribution teaches the meta-learner all three routing paths.

Usage:
    python training/eval/generate_bootstrap_questions.py
"""

import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================
# SELF-ANSWERABLE — Core Cartesian formalization/analysis
# The trained 8B model should handle these alone.
# ============================================================

SELF_QUESTIONS = [
    "Formalize the Cogito as a strict inference in Z3.",
    "What is the ASPIC+ attack structure of the Wax Argument?",
    "Is the Real Distinction argument deductively valid?",
    "Decompose Meditation III into its component sub-arguments.",
    "What role does the Evil Genius play in the method of doubt?",
    "Formalize substance dualism: mind and body as distinct sorts.",
    "What is the modal structure of the conceivability argument?",
    "Check consistency of the Trademark Argument premises in Z3.",
    "Identify the argumentation scheme in the Ontological Argument.",
    "What is the logical relationship between Meditations II and VI?",
    "Formalize the clear and distinct perception criterion.",
    "Is the Cogito an inference or an intuition? Formalize both readings.",
    "What type of ASPIC+ attack does Arnauld's Circle represent?",
    "Reconstruct the dreaming argument as a formal inference.",
    "Check whether Descartes' proofs of God are jointly consistent.",
    "Formalize the causal adequacy principle as a Z3 axiom.",
    "What is the argument structure of Meditation V?",
    "Analyze the Wax Argument using ASPIC+ strict rules.",
    "Formalize the distinction between imagination and intellection.",
    "Is the Cogito immune to the Evil Genius? Prove formally.",
    "What are the defeasible premises in Descartes' physics?",
    "Reconstruct Elisabeth's interaction objection in ASPIC+.",
    "Formalize the transparency thesis about mental states.",
    "What is the logical form of Descartes' certainty criterion?",
    "Check if the ontological argument commits petitio principii.",
    "Formalize the relationship between doubt and certainty.",
    "What is the deductive structure of Meditation I?",
    "Identify all enthymematic premises in the Real Distinction.",
    "Formalize Descartes' notion of objective reality as a Z3 type.",
    "Is the move from Meditation II to III deductively valid?",
    "Reconstruct the piece of wax argument step by step.",
    "What logical connectives does Descartes use in the Cogito?",
    "Formalize the divine guarantee of clear perceptions.",
    "Analyze the structure of Gassendi's fifth objections.",
    "Check the consistency of Descartes' substance ontology.",
    "Formalize how extension defines corporeal substance.",
    "What is the ASPIC+ representation of the Cogito?",
    "Reconstruct the argument from divisibility in Meditation VI.",
    "Formalize the principle that nothing comes from nothing.",
    "What are the strict rules in Descartes' method of doubt?",
]

# ============================================================
# ORACLE-NEEDED — Broad philosophy, historical context
# The trained model lacks knowledge here.
# ============================================================

ORACLE_QUESTIONS = [
    "What was Hume's response to Cartesian rationalism?",
    "How did the Port-Royal Logic incorporate Descartes' method?",
    "What is Merleau-Ponty's critique of the Cogito?",
    "How was Descartes received by the Utrecht theologians?",
    "What did Kant say about the ontological argument?",
    "Compare Descartes' doubt with Pyrrhonian skepticism.",
    "What was the relationship between Descartes and Beeckman?",
    "How does Husserl's phenomenological reduction differ from Cartesian doubt?",
    "What neuroscience evidence bears on the interaction problem?",
    "How did occasionalism develop after Descartes' death?",
    "What was La Forge's contribution to Cartesian physics?",
    "How does IIT relate to Cartesian substance dualism?",
    "What was the Condemnation of 1663 about?",
    "Compare Descartes' animal automata with modern animal consciousness research.",
    "What did Strawson argue about persons in 'Individuals'?",
    "How did Locke respond to innate ideas?",
    "What was Regius' disagreement with Descartes?",
    "How does Dennett's heterophenomenology relate to Cartesian introspection?",
    "What was the historical reception of the Passions of the Soul?",
    "How did Cordemoy extend Cartesian physics?",
    "What is the Chinese Room argument and how does it relate to dualism?",
    "How did the Jesuits receive Descartes' natural philosophy?",
    "What was Princess Elisabeth's broader philosophical influence?",
    "Compare Cartesian skepticism with contemporary radical skepticism.",
    "What was Desgabets' response to the Cartesian Circle?",
    "How does Nagel's 'What Is It Like to Be a Bat' relate to Descartes?",
    "What was Clauberg's contribution to Cartesian metaphysics?",
    "How did Spinoza transform Cartesian substance?",
    "What was the role of the Oratory in spreading Cartesianism?",
    "How does functionalism challenge substance dualism?",
]

# ============================================================
# HYBRID — Cartesian core + external knowledge needed
# Requires both specialist analysis and broader context.
# ============================================================

HYBRID_QUESTIONS = [
    "Is the Real Distinction structurally identical to the zombie argument?",
    "Can Descartes' causal adequacy principle survive modern physicalism?",
    "Formalize both Descartes' and Spinoza's substance ontology and check compatibility.",
    "Does Ryle's critique of the 'ghost in the machine' actually refute Descartes?",
    "Compare the modal logic of the Real Distinction with Kripke's identity argument.",
    "Can Global Workspace Theory be reconciled with substance dualism? Formalize.",
    "How does Chalmers' conceivability principle differ from Descartes' divine guarantee?",
    "Formalize Elisabeth's interaction objection alongside Kim's exclusion argument.",
    "Is Descartes' foundationalism compatible with Bayesian epistemology?",
    "Compare Descartes' pineal gland hypothesis with modern binding problem solutions.",
    "Does predictive processing vindicate or refute Cartesian representationalism?",
    "Formalize both Descartes' and Leibniz's arguments for mind-body distinctness.",
    "Can property dualism capture Descartes' insights without substance dualism?",
    "Compare the Cartesian Circle with the problem of the criterion in Chisholm.",
    "Is Descartes' conceivability-possibility bridge valid in S5 vs. S4?",
    "Does quantum indeterminism help with the interaction problem? Formalize.",
    "Compare the structure of the Cogito with Sartre's pre-reflective cogito.",
    "Can Descartes' proofs of God survive Plantinga's modal reformulation?",
    "Formalize the relationship between Cartesian doubt and Moore's paradox.",
    "How does contemporary philosophy of perception affect the wax argument?",
    "Compare ASPIC+ analysis of the Cogito with Brandom's inferentialist reading.",
    "Can IIT's phi metric be formalized as a Cartesian consciousness criterion?",
    "Does Descartes' indivisibility argument work against panpsychism?",
    "Formalize both the Cogito and Wittgenstein's private language argument.",
    "Is there a formal parallel between Cartesian doubt and Bayesian updating?",
    "Compare Descartes' truth rule with contemporary reliabilism.",
    "Can Descartes' substance dualism accommodate split-brain cases?",
    "Formalize the relationship between Cartesian clarity and epistemic justification.",
    "Does Descartes' argument from error parallel contemporary error theory?",
    "Compare the logical structure of the Cogito with Nozick's tracking theory.",
]


def generate(output_path: str = None):
    if output_path is None:
        output_path = str(
            PROJECT_ROOT / "training" / "eval" / "bootstrap_questions.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_questions = []

    for q in SELF_QUESTIONS:
        all_questions.append({
            "question": q,
            "expected_routing": "SELF",
            "category": "cartesian_core"
        })

    for q in ORACLE_QUESTIONS:
        all_questions.append({
            "question": q,
            "expected_routing": "ORACLE",
            "category": "broad_philosophy"
        })

    for q in HYBRID_QUESTIONS:
        all_questions.append({
            "question": q,
            "expected_routing": "HYBRID",
            "category": "cross_domain"
        })

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(all_questions)

    with open(output_path, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    print(f"Generated {len(all_questions)} bootstrap questions")
    print(f"  SELF:   {len(SELF_QUESTIONS)}")
    print(f"  ORACLE: {len(ORACLE_QUESTIONS)}")
    print(f"  HYBRID: {len(HYBRID_QUESTIONS)}")
    print(f"  Saved:  {output_path}")


if __name__ == "__main__":
    generate()
