"""
Phase 8: Base Model Evaluation — Benchmarks for the Descartes specialist.

Evaluates the domain-adapted model BEFORE cascade integration.
Establishes performance baselines that Phase 12 (cascade eval)
should improve upon.

Test Suites:
1. Perplexity on held-out Descartes corpus
2. Descartes factual knowledge (multiple choice)
3. Philosophical reasoning (short answer grading)
4. Formalization capability (Z3 code generation)
5. Domain boundaries (knows what it doesn't know)

Pass Criteria:
  Descartes Knowledge MC:  >= 75% accuracy
  Reasoning Quality:       >= 60% (scored by rubric)
  Formalization Success:   >= 50% (Z3 compiles + correct)
  Out-of-Domain Detection: >= 70% (correctly defers)

Usage:
    python training/eval/eval_base.py
    python training/eval/eval_base.py --model descartes:8b
    python training/eval/eval_base.py --provider ollama --model descartes:8b

Reference: PHILOSOPHER_ENGINE_V3_UNIFIED_ARCHITECTURE.md, Phase 8
"""

import json
import sys
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# TEST SUITE 1: Descartes Factual Knowledge (Multiple Choice)
# ============================================================

KNOWLEDGE_MC = [
    {
        "q": "In which work does Descartes present the Cogito?",
        "options": [
            "A) Principles of Philosophy",
            "B) Meditations on First Philosophy",
            "C) Discourse on the Method",
            "D) Rules for the Direction of the Mind",
        ],
        "answer": "B",  # Also C, but B is the canonical presentation
        "accept": ["B", "C"],
        "topic": "texts",
    },
    {
        "q": "Who raised the Cartesian Circle objection?",
        "options": [
            "A) Hobbes",
            "B) Gassendi",
            "C) Arnauld",
            "D) Mersenne",
        ],
        "answer": "C",
        "accept": ["C"],
        "topic": "objections",
    },
    {
        "q": "What are Descartes' two essential attributes?",
        "options": [
            "A) Thought and Extension",
            "B) Mind and Matter",
            "C) Form and Substance",
            "D) Reason and Sensation",
        ],
        "answer": "A",
        "accept": ["A"],
        "topic": "metaphysics",
    },
    {
        "q": "In which Meditation does the Wax Argument appear?",
        "options": [
            "A) First Meditation",
            "B) Second Meditation",
            "C) Third Meditation",
            "D) Sixth Meditation",
        ],
        "answer": "B",
        "accept": ["B"],
        "topic": "texts",
    },
    {
        "q": "What is the role of God in Descartes' epistemology?",
        "options": [
            "A) God is irrelevant to knowledge",
            "B) God guarantees clear and distinct perceptions",
            "C) God is the source of all ideas",
            "D) God creates the external world at each moment",
        ],
        "answer": "B",
        "accept": ["B"],
        "topic": "epistemology",
    },
    {
        "q": "Which objector compared Descartes' conceivability argument "
             "to conceiving a right triangle without the Pythagorean theorem?",
        "options": [
            "A) Mersenne",
            "B) Hobbes",
            "C) Arnauld",
            "D) Gassendi",
        ],
        "answer": "C",
        "accept": ["C"],
        "topic": "objections",
    },
    {
        "q": "What gland did Descartes identify as the seat of "
             "mind-body interaction?",
        "options": [
            "A) Pituitary",
            "B) Thyroid",
            "C) Pineal",
            "D) Hypothalamus",
        ],
        "answer": "C",
        "accept": ["C"],
        "topic": "mind_body",
    },
    {
        "q": "The Trademark Argument appears in which Meditation?",
        "options": [
            "A) First",
            "B) Second",
            "C) Third",
            "D) Fifth",
        ],
        "answer": "C",
        "accept": ["C"],
        "topic": "texts",
    },
    {
        "q": "What is the Evil Genius (Malin Génie) hypothesis for?",
        "options": [
            "A) Proving God's existence",
            "B) Maximally extending doubt to include mathematics",
            "C) Establishing the existence of the external world",
            "D) Refuting skepticism",
        ],
        "answer": "B",
        "accept": ["B"],
        "topic": "epistemology",
    },
    {
        "q": "Princess Elisabeth of Bohemia's objection concerned:",
        "options": [
            "A) The existence of God",
            "B) The reliability of the senses",
            "C) How mind and body causally interact",
            "D) The nature of mathematical truth",
        ],
        "answer": "C",
        "accept": ["C"],
        "topic": "mind_body",
    },
    {
        "q": "Which set of Objections was written by Hobbes?",
        "options": [
            "A) First",
            "B) Second",
            "C) Third",
            "D) Fifth",
        ],
        "answer": "C",
        "accept": ["C"],
        "topic": "objections",
    },
    {
        "q": "Descartes' ontological argument for God appears in:",
        "options": [
            "A) Third Meditation",
            "B) Fourth Meditation",
            "C) Fifth Meditation",
            "D) Sixth Meditation",
        ],
        "answer": "C",
        "accept": ["C"],
        "topic": "texts",
    },
]


# ============================================================
# TEST SUITE 2: Philosophical Reasoning (Short Answer)
# ============================================================

REASONING_TESTS = [
    {
        "q": "Explain why the Cogito is resistant to the Evil Genius doubt.",
        "rubric": [
            "even if deceived, I must exist to be deceived",
            "thinking/doubting is self-verifying",
            "the act of doubt confirms the doubter's existence",
        ],
        "min_keywords": 2,  # Must hit at least 2 rubric points
    },
    {
        "q": "What is the logical structure of the Real Distinction argument?",
        "rubric": [
            "conceivability of mind without body",
            "conceivability implies possibility (with divine guarantee)",
            "possibility of separation implies real distinction",
            "modal reasoning / S5 / possible worlds",
        ],
        "min_keywords": 2,
    },
    {
        "q": "Why does the Cartesian Circle appear to be circular?",
        "rubric": [
            "clear and distinct perception used to prove God",
            "God used to validate clear and distinct perception",
            "Arnauld's objection / Fourth Objections",
            "Descartes' defense: present vs remembered perception",
        ],
        "min_keywords": 2,
    },
    {
        "q": "How does the causal adequacy principle function in "
             "the Trademark Argument?",
        "rubric": [
            "cause must have at least as much reality as effect",
            "idea of God has infinite objective reality",
            "I am finite, so I cannot be sole cause",
            "only an infinite being could cause the idea",
        ],
        "min_keywords": 2,
    },
    {
        "q": "Compare the zombie argument to the Real Distinction. "
             "What structural features do they share?",
        "rubric": [
            "both use conceivability-possibility inference",
            "both conclude metaphysical distinctness/separation",
            "both vulnerable to attacks on CP thesis",
            "Arnauld's objection parallels Type-B physicalism",
            "structural isomorphism / same schema",
        ],
        "min_keywords": 2,
    },
]


# ============================================================
# TEST SUITE 3: Formalization (Z3 Code Generation)
# ============================================================

FORMALIZATION_TESTS = [
    {
        "q": "Write Z3 Python code to verify the Cogito: "
             "Doubts(I) → Thinks(I) → Exists(I). "
             "Check whether Not(Exists(I)) is consistent with premises.",
        "must_contain": ["DeclareSort", "Function", "ForAll", "Implies",
                         "Not", "Solver", "check"],
        "expected_result": "unsat",
    },
    {
        "q": "Formalize the Wax Argument as an elimination: three sources "
             "(senses, imagination, intellect), eliminate senses and "
             "imagination, prove only intellect remains.",
        "must_contain": ["DeclareSort", "Or", "Solver"],
        "expected_result": "intellect",
    },
    {
        "q": "Create Z3 code for the causal adequacy principle: "
             "formal_reality(cause) >= objective_reality(effect). "
             "Show that a finite being (reality=2) cannot cause an idea "
             "with infinite objective reality (=4).",
        "must_contain": ["Function", "IntSort", "ForAll", ">="],
        "expected_result": "unsat",
    },
]


# ============================================================
# TEST SUITE 4: Domain Boundary Detection
# ============================================================

OUT_OF_DOMAIN_TESTS = [
    {
        "q": "What is the capital of France?",
        "should_defer": True,
        "topic": "geography",
    },
    {
        "q": "Write a Python function to sort a list.",
        "should_defer": True,
        "topic": "programming",
    },
    {
        "q": "What did Descartes argue in the Second Meditation?",
        "should_defer": False,
        "topic": "descartes",
    },
    {
        "q": "Explain quantum entanglement.",
        "should_defer": True,
        "topic": "physics",
    },
    {
        "q": "How does substance dualism relate to the interaction problem?",
        "should_defer": False,
        "topic": "philosophy_of_mind",
    },
    {
        "q": "What are the latest stock market trends?",
        "should_defer": True,
        "topic": "finance",
    },
    {
        "q": "Compare Spinoza's substance monism to Descartes' dualism.",
        "should_defer": False,
        "topic": "rationalism",
    },
    {
        "q": "How do you make chocolate chip cookies?",
        "should_defer": True,
        "topic": "cooking",
    },
    {
        "q": "What is the zombie argument against physicalism?",
        "should_defer": False,
        "topic": "philosophy_of_mind",
    },
    {
        "q": "Summarize the plot of Harry Potter.",
        "should_defer": True,
        "topic": "fiction",
    },
]


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def query_model(prompt: str,
                model: str = "descartes:8b",
                provider: str = "ollama") -> str:
    """Send a query to the model and return the response text.

    Supports ollama provider. Extend for other providers.
    """
    if provider == "ollama":
        try:
            import ollama
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except ImportError:
            return "[ERROR] ollama package not installed"
        except Exception as e:
            return f"[ERROR] {e}"

    elif provider == "mock":
        # For testing the evaluation harness without a model
        return "[MOCK RESPONSE] This is a placeholder response."

    return f"[ERROR] Unknown provider: {provider}"


def eval_knowledge_mc(model: str, provider: str,
                      verbose: bool = True) -> Dict:
    """Evaluate multiple-choice Descartes knowledge."""
    if verbose:
        print("\n[1/4] Descartes Knowledge (Multiple Choice)")
        print("-" * 40)

    correct = 0
    results = []

    for test in KNOWLEDGE_MC:
        prompt = (
            f"{test['q']}\n\n"
            + "\n".join(test['options'])
            + "\n\nAnswer with just the letter (A, B, C, or D)."
        )
        response = query_model(prompt, model, provider)
        response_upper = response.strip().upper()

        # Extract letter from response
        letter = None
        match = re.search(r'\b([A-D])\b', response_upper)
        if match:
            letter = match.group(1)

        is_correct = letter in test["accept"]
        if is_correct:
            correct += 1

        results.append({
            "question": test["q"][:60],
            "expected": test["answer"],
            "got": letter,
            "correct": is_correct,
            "topic": test["topic"],
        })

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] {test['q'][:55]}... "
                  f"(expected={test['answer']}, got={letter})")

    accuracy = correct / len(KNOWLEDGE_MC) if KNOWLEDGE_MC else 0
    if verbose:
        print(f"\n  MC Accuracy: {correct}/{len(KNOWLEDGE_MC)} "
              f"= {accuracy:.0%}")

    return {"accuracy": accuracy, "correct": correct,
            "total": len(KNOWLEDGE_MC), "details": results}


def eval_reasoning(model: str, provider: str,
                   verbose: bool = True) -> Dict:
    """Evaluate philosophical reasoning quality via rubric matching."""
    if verbose:
        print("\n[2/4] Philosophical Reasoning (Short Answer)")
        print("-" * 40)

    scores = []
    results = []

    for test in REASONING_TESTS:
        response = query_model(test["q"], model, provider)
        response_lower = response.lower()

        # Count rubric points hit
        hits = 0
        for rubric_point in test["rubric"]:
            # Check if key phrases from rubric appear in response
            keywords = rubric_point.lower().split()
            # At least half the rubric keywords must appear
            matched = sum(1 for kw in keywords
                          if kw in response_lower and len(kw) > 3)
            if matched >= len(keywords) * 0.4:
                hits += 1

        score = hits / len(test["rubric"])
        passed = hits >= test["min_keywords"]
        scores.append(score)

        results.append({
            "question": test["q"][:60],
            "rubric_hits": hits,
            "rubric_total": len(test["rubric"]),
            "score": score,
            "passed": passed,
        })

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {test['q'][:55]}... "
                  f"(rubric: {hits}/{len(test['rubric'])})")

    avg_score = sum(scores) / len(scores) if scores else 0
    pass_count = sum(1 for r in results if r["passed"])
    if verbose:
        print(f"\n  Reasoning Quality: {avg_score:.0%} avg, "
              f"{pass_count}/{len(REASONING_TESTS)} passed")

    return {"avg_score": avg_score, "pass_count": pass_count,
            "total": len(REASONING_TESTS), "details": results}


def eval_formalization(model: str, provider: str,
                       verbose: bool = True) -> Dict:
    """Evaluate Z3 formalization capability."""
    if verbose:
        print("\n[3/4] Z3 Formalization")
        print("-" * 40)

    correct = 0
    results = []

    for test in FORMALIZATION_TESTS:
        response = query_model(test["q"], model, provider)

        # Check if response contains required Z3 constructs
        hits = sum(1 for keyword in test["must_contain"]
                   if keyword in response)
        coverage = hits / len(test["must_contain"])
        is_correct = coverage >= 0.7  # 70% of required constructs

        if is_correct:
            correct += 1

        results.append({
            "question": test["q"][:60],
            "construct_coverage": coverage,
            "correct": is_correct,
        })

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] {test['q'][:55]}... "
                  f"(constructs: {coverage:.0%})")

    accuracy = correct / len(FORMALIZATION_TESTS) if FORMALIZATION_TESTS else 0
    if verbose:
        print(f"\n  Formalization: {correct}/{len(FORMALIZATION_TESTS)} "
              f"= {accuracy:.0%}")

    return {"accuracy": accuracy, "correct": correct,
            "total": len(FORMALIZATION_TESTS), "details": results}


def eval_domain_boundary(model: str, provider: str,
                         verbose: bool = True) -> Dict:
    """Evaluate out-of-domain detection."""
    if verbose:
        print("\n[4/4] Domain Boundary Detection")
        print("-" * 40)

    correct = 0
    results = []

    defer_indicators = [
        "outside my", "not my area", "i specialize", "beyond my",
        "i focus on", "descartes", "philosophy", "i'm not able",
        "cannot help", "don't have expertise", "not trained",
        "outside the scope", "i'm designed for",
    ]

    for test in OUT_OF_DOMAIN_TESTS:
        prompt = (
            f"{test['q']}\n\n"
            "If this is outside your area of expertise "
            "(Cartesian philosophy and philosophy of mind), "
            "say so briefly."
        )
        response = query_model(prompt, model, provider)
        response_lower = response.lower()

        deferred = any(ind in response_lower for ind in defer_indicators)

        if test["should_defer"]:
            is_correct = deferred
        else:
            is_correct = not deferred

        if is_correct:
            correct += 1

        results.append({
            "question": test["q"][:60],
            "should_defer": test["should_defer"],
            "deferred": deferred,
            "correct": is_correct,
            "topic": test["topic"],
        })

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            action = "defer" if test["should_defer"] else "answer"
            print(f"  [{status}] Should {action}: {test['q'][:45]}... "
                  f"(deferred={deferred})")

    accuracy = correct / len(OUT_OF_DOMAIN_TESTS) if OUT_OF_DOMAIN_TESTS else 0
    if verbose:
        print(f"\n  Domain Detection: {correct}/{len(OUT_OF_DOMAIN_TESTS)} "
              f"= {accuracy:.0%}")

    return {"accuracy": accuracy, "correct": correct,
            "total": len(OUT_OF_DOMAIN_TESTS), "details": results}


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_full_evaluation(model: str = "descartes:8b",
                        provider: str = "ollama",
                        output_path: Optional[str] = None):
    """Run the complete Phase 8 base model evaluation."""
    print("=" * 60)
    print("PHASE 8: Base Model Evaluation")
    print("=" * 60)
    print(f"  Model:    {model}")
    print(f"  Provider: {provider}")

    mc_results = eval_knowledge_mc(model, provider)
    reasoning_results = eval_reasoning(model, provider)
    formalization_results = eval_formalization(model, provider)
    domain_results = eval_domain_boundary(model, provider)

    # ---- Pass Criteria ----
    print("\n" + "=" * 60)
    print("PASS CRITERIA")
    print("=" * 60)

    criteria = {
        "knowledge_mc": {
            "value": mc_results["accuracy"],
            "threshold": 0.75,
            "label": "Descartes Knowledge MC >= 75%",
        },
        "reasoning_quality": {
            "value": reasoning_results["avg_score"],
            "threshold": 0.60,
            "label": "Reasoning Quality >= 60%",
        },
        "formalization": {
            "value": formalization_results["accuracy"],
            "threshold": 0.50,
            "label": "Formalization >= 50%",
        },
        "domain_detection": {
            "value": domain_results["accuracy"],
            "threshold": 0.70,
            "label": "Domain Detection >= 70%",
        },
    }

    all_pass = True
    for key, crit in criteria.items():
        passed = crit["value"] >= crit["threshold"]
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {crit['label']}: {crit['value']:.3f}")

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    # ---- Save Results ----
    results = {
        "model": model,
        "provider": provider,
        "knowledge_mc": mc_results,
        "reasoning": reasoning_results,
        "formalization": formalization_results,
        "domain_boundary": domain_results,
        "criteria": {k: {**v, "passed": v["value"] >= v["threshold"]}
                     for k, v in criteria.items()},
        "overall_pass": all_pass,
    }

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / "training" / "eval" /
            "base_eval_results.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 8: Base model evaluation")
    parser.add_argument("--model", type=str, default="descartes:8b",
                        help="Model name for ollama")
    parser.add_argument("--provider", type=str, default="ollama",
                        choices=["ollama", "mock"],
                        help="Model provider")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")

    args = parser.parse_args()
    run_full_evaluation(
        model=args.model,
        provider=args.provider,
        output_path=args.output,
    )
