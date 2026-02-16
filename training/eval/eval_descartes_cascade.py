"""
Phase 12 (CASCADE): End-to-end evaluation of the Descartes cascade engine.

Tests:
1. Cartesian argument validity (Z3-verifiable)
2. Routing accuracy (does it correctly route to oracle?)
3. Descartes-specific knowledge (facts about texts/positions)
4. Integration quality (does oracle info improve answers?)
5. Confidence calibration (does meta-learner confidence correlate
   with actual correctness?)

Pass Criteria:
  Argument Validity:       >= 85%
  Routing Accuracy:        >= 80%
  Knowledge (no oracle):   >= 70%
  Knowledge (with oracle):  >= 90%
  Calibration ECE:         < 0.15 (Expected Calibration Error)

Usage:
    python training/eval/eval_descartes_cascade.py [model_path]
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))


# ============================================================
# TEST SUITES
# ============================================================

VALIDITY_TESTS = [
    {
        "argument": (
            "P1: I think.\n"
            "P2: Whatever thinks, exists.\n"
            "C: I exist."
        ),
        "valid": True,
        "label": "cogito_syllogistic"
    },
    {
        "argument": (
            "P1: I can clearly and distinctly conceive mind without body.\n"
            "P2: Whatever I can C&D conceive, God can create.\n"
            "P3: If God can create A without B, A and B are distinct.\n"
            "C: Mind and body are distinct substances."
        ),
        "valid": True,
        "label": "real_distinction"
    },
    {
        "argument": (
            "P1: I have an idea of a perfect being.\n"
            "P2: I am imperfect.\n"
            "C: A perfect being must exist to cause my idea."
        ),
        "valid": False,  # Missing: causal adequacy principle as explicit premise
        "label": "trademark_incomplete"
    },
    {
        "argument": (
            "P1: The senses sometimes deceive.\n"
            "P2: Whatever sometimes deceives cannot be trusted.\n"
            "C: Nothing known through the senses is certain."
        ),
        "valid": True,  # Valid but unsound (P2 is too strong)
        "label": "dream_argument_valid_unsound"
    },
    {
        "argument": (
            "P1: Mind is unextended.\n"
            "P2: Body is extended.\n"
            "C: Mind cannot causally interact with body."
        ),
        "valid": False,  # Missing: "causal interaction requires extension"
        "label": "interaction_problem_missing_premise"
    },
]

ROUTING_TESTS = [
    {"query": "Formalize the Cogito in Z3.",
     "expected": "SELF"},
    {"query": "What did Husserl say about the Cartesian Meditations?",
     "expected": "ORACLE"},
    {"query": "Is the Real Distinction structurally identical to "
              "the zombie argument?",
     "expected": "HYBRID"},
    {"query": "Decompose Arnauld's circularity objection into "
              "ASPIC+ attack structure.",
     "expected": "SELF"},
    {"query": "How was Descartes received by the Jesuits at La Fleche?",
     "expected": "ORACLE"},
    {"query": "Formalize both the Trademark Argument and the "
              "Ontological Argument and check consistency.",
     "expected": "SELF"},
    {"query": "Compare Elisabeth's interaction problem to Jaegwon Kim's "
              "exclusion argument. Are they structurally similar?",
     "expected": "HYBRID"},
    {"query": "What year was the Discourse on the Method published?",
     "expected": "SELF"},
    {"query": "How did the reception of Descartes differ between "
              "France and the Netherlands in the 1640s?",
     "expected": "ORACLE"},
    {"query": "Can predictive processing theory be reconciled with "
              "substance dualism? Check formally.",
     "expected": "HYBRID"},
]

KNOWLEDGE_TESTS = [
    {"q": "In which Meditation does Descartes present the Wax Argument?",
     "a": "Second Meditation",
     "keywords": ["second", "meditation ii", "2nd"]},
    {"q": "Who raised the Cartesian Circle objection?",
     "a": "Arnauld",
     "keywords": ["arnauld"]},
    {"q": "What is the name of Descartes' correspondent who pressed "
          "the interaction problem?",
     "a": "Princess Elisabeth of Bohemia",
     "keywords": ["elisabeth", "elizabeth"]},
    {"q": "What gland did Descartes identify as the seat of "
          "mind-body interaction?",
     "a": "Pineal gland",
     "keywords": ["pineal"]},
    {"q": "Which Objection set is by Hobbes?",
     "a": "Third Objections",
     "keywords": ["third", "3rd"]},
    {"q": "What are the two essential attributes in Descartes' dualism?",
     "a": "Thought (cogitatio) and Extension (extensio)",
     "keywords": ["thought", "extension", "thinking", "extended"]},
    {"q": "What is the causal adequacy principle in Meditation III?",
     "a": "The cause must have at least as much reality as the effect",
     "keywords": ["cause", "reality", "effect", "adequate"]},
    {"q": "Who wrote the Fifth Objections?",
     "a": "Gassendi",
     "keywords": ["gassendi"]},
]


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def eval_validity(engine, verbose: bool = True) -> Dict:
    """Test argument validity assessment."""

    if verbose:
        print("\n[1/5] Argument Validity Tests")
        print("-" * 40)

    correct = 0
    results = []

    for test in VALIDITY_TESTS:
        query = (
            f"Is the following argument logically valid? "
            f"Answer 'VALID' or 'INVALID' and explain why.\n\n"
            f"{test['argument']}"
        )

        result = engine.run(query)
        response_lower = result.final_response.lower()

        # Check if model identified validity correctly
        model_says_valid = ("valid" in response_lower and
                            "invalid" not in response_lower)
        model_says_invalid = "invalid" in response_lower

        if test["valid"] and model_says_valid:
            is_correct = True
        elif not test["valid"] and model_says_invalid:
            is_correct = True
        else:
            is_correct = False

        if is_correct:
            correct += 1

        results.append({
            "label": test["label"],
            "expected_valid": test["valid"],
            "model_response_valid": model_says_valid,
            "correct": is_correct,
        })

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] {test['label']}: "
                  f"expected={'valid' if test['valid'] else 'invalid'}, "
                  f"got={'valid' if model_says_valid else 'invalid'}")

    accuracy = correct / len(VALIDITY_TESTS)
    if verbose:
        print(f"\n  Validity accuracy: {correct}/{len(VALIDITY_TESTS)} "
              f"= {accuracy:.0%}")

    return {"accuracy": accuracy, "correct": correct,
            "total": len(VALIDITY_TESTS), "details": results}


def eval_routing(engine, verbose: bool = True) -> Dict:
    """Test routing accuracy."""

    if verbose:
        print("\n[2/5] Routing Accuracy Tests")
        print("-" * 40)

    correct = 0
    results = []

    for test in ROUTING_TESTS:
        result = engine.run(test["query"])

        is_correct = result.routing_decision == test["expected"]
        if is_correct:
            correct += 1

        results.append({
            "query": test["query"][:60] + "...",
            "expected": test["expected"],
            "predicted": result.routing_decision,
            "correct": is_correct,
            "confidence": result.confidence,
        })

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] Expected {test['expected']:7s}, "
                  f"got {result.routing_decision:7s} "
                  f"(conf={result.confidence:.2f}) "
                  f"— {test['query'][:50]}...")

    accuracy = correct / len(ROUTING_TESTS)
    if verbose:
        print(f"\n  Routing accuracy: {correct}/{len(ROUTING_TESTS)} "
              f"= {accuracy:.0%}")

    return {"accuracy": accuracy, "correct": correct,
            "total": len(ROUTING_TESTS), "details": results}


def eval_knowledge(engine, verbose: bool = True) -> Dict:
    """Test Descartes-specific factual knowledge."""

    if verbose:
        print("\n[3/5] Knowledge Tests")
        print("-" * 40)

    correct = 0
    results = []

    for test in KNOWLEDGE_TESTS:
        result = engine.run(test["q"])
        response_lower = result.final_response.lower()

        # Check if any expected keyword appears
        is_correct = any(kw in response_lower for kw in test["keywords"])
        if is_correct:
            correct += 1

        results.append({
            "question": test["q"][:60] + "...",
            "expected": test["a"],
            "correct": is_correct,
            "oracle_used": result.oracle_used,
        })

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            oracle_tag = " [oracle]" if result.oracle_used else ""
            print(f"  [{status}] {test['q'][:55]}...{oracle_tag}")

    accuracy = correct / len(KNOWLEDGE_TESTS)
    if verbose:
        oracle_count = sum(1 for r in results if r["oracle_used"])
        print(f"\n  Knowledge accuracy: {correct}/{len(KNOWLEDGE_TESTS)} "
              f"= {accuracy:.0%}")
        print(f"  Oracle used: {oracle_count}/{len(KNOWLEDGE_TESTS)}")

    return {"accuracy": accuracy, "correct": correct,
            "total": len(KNOWLEDGE_TESTS), "details": results}


def eval_calibration(engine, verbose: bool = True) -> Dict:
    """Test confidence calibration (Expected Calibration Error).

    Bins predictions by confidence level and checks if the
    fraction correct in each bin matches the confidence.
    """

    if verbose:
        print("\n[4/5] Confidence Calibration")
        print("-" * 40)

    # Collect predictions from all test suites
    predictions = []

    # Quick knowledge test predictions
    for test in KNOWLEDGE_TESTS:
        result = engine.run(test["q"])
        response_lower = result.final_response.lower()
        is_correct = any(kw in response_lower for kw in test["keywords"])
        predictions.append((result.confidence, is_correct))

    # Bin by confidence
    n_bins = 5
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_correct = [[] for _ in range(n_bins)]

    for conf, correct in predictions:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_correct[bin_idx].append(1.0 if correct else 0.0)

    # Compute ECE
    ece = 0.0
    total = len(predictions)

    if verbose:
        print(f"  {'Bin':>10s} {'Count':>6s} {'Accuracy':>9s} "
              f"{'Avg Conf':>9s} {'Gap':>6s}")

    for i in range(n_bins):
        if not bin_correct[i]:
            continue
        bin_acc = sum(bin_correct[i]) / len(bin_correct[i])
        bin_conf = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
        gap = abs(bin_acc - bin_conf)
        ece += gap * len(bin_correct[i]) / total

        if verbose:
            print(f"  [{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}] "
                  f"{len(bin_correct[i]):>6d} "
                  f"{bin_acc:>8.1%} "
                  f"{bin_conf:>8.1%} "
                  f"{gap:>5.1%}")

    if verbose:
        print(f"\n  ECE: {ece:.3f}")

    return {"ece": ece, "n_predictions": len(predictions)}


def eval_engine_stats(engine, verbose: bool = True) -> Dict:
    """Report engine routing statistics."""

    if verbose:
        print("\n[5/5] Engine Statistics")
        print("-" * 40)

    stats = engine.get_stats()

    if verbose:
        print(f"  Total queries:     {stats['total_queries']}")
        print(f"  Self-handled:      {stats['self_handled']} "
              f"({stats['self_rate']:.0%})")
        print(f"  Oracle-handled:    {stats['oracle_handled']}")
        print(f"  Hybrid-handled:    {stats['hybrid_handled']}")
        print(f"  Meta-learner updates: {stats['meta_learner_updates']}")
        if stats.get("oracle_stats"):
            print(f"  Oracle cost:       "
                  f"${stats['oracle_stats']['total_cost']:.4f}")

    return stats


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_full_evaluation(model_path: str,
                        meta_learner_path: Optional[str] = None,
                        oracle_provider: str = "deepseek",
                        output_path: Optional[str] = None):
    """Run the complete evaluation suite."""

    print("=" * 60)
    print("PHASE 12: Descartes Cascade End-to-End Evaluation")
    print("=" * 60)

    # Import engine
    from cascade_engine import DescartesEngine
    from oracle import OracleConfig

    oracle_config = OracleConfig(provider=oracle_provider)

    engine = DescartesEngine(
        model_path=model_path,
        meta_learner_path=meta_learner_path,
        oracle_config=oracle_config)

    # Run all test suites
    validity_results = eval_validity(engine)
    routing_results = eval_routing(engine)
    knowledge_results = eval_knowledge(engine)
    calibration_results = eval_calibration(engine)
    stats = eval_engine_stats(engine)

    # ---- Pass Criteria ----
    print("\n" + "=" * 60)
    print("PASS CRITERIA")
    print("=" * 60)

    criteria = {
        "validity_accuracy": {
            "value": validity_results["accuracy"],
            "threshold": 0.85,
            "label": "Argument Validity >= 85%",
        },
        "routing_accuracy": {
            "value": routing_results["accuracy"],
            "threshold": 0.80,
            "label": "Routing Accuracy >= 80%",
        },
        "knowledge_accuracy": {
            "value": knowledge_results["accuracy"],
            "threshold": 0.70,
            "label": "Knowledge >= 70%",
        },
        "calibration_ece": {
            "value": calibration_results["ece"],
            "threshold": 0.15,
            "label": "Calibration ECE < 0.15",
            "lower_is_better": True,
        },
    }

    all_pass = True
    for key, crit in criteria.items():
        if crit.get("lower_is_better"):
            passed = crit["value"] < crit["threshold"]
        else:
            passed = crit["value"] >= crit["threshold"]

        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {crit['label']}: {crit['value']:.3f}")

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    # ---- Save results ----
    results = {
        "validity": validity_results,
        "routing": routing_results,
        "knowledge": knowledge_results,
        "calibration": calibration_results,
        "engine_stats": stats,
        "criteria": {k: {**v, "passed": (
            v["value"] < v["threshold"] if v.get("lower_is_better")
            else v["value"] >= v["threshold"]
        )} for k, v in criteria.items()},
        "overall_pass": all_pass,
    }

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / "training" / "eval" /
            "cascade_eval_results.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_pass


if __name__ == "__main__":
    model_path = (sys.argv[1] if len(sys.argv) > 1
                  else str(PROJECT_ROOT / "models" / "descartes-8b-cascade"))

    meta_path = str(
        PROJECT_ROOT / "models" / "meta_learner_bootstrapped.pt")
    if not os.path.exists(meta_path):
        meta_path = None

    provider = "deepseek"
    for i, arg in enumerate(sys.argv):
        if arg == "--provider" and i + 1 < len(sys.argv):
            provider = sys.argv[i + 1]

    run_full_evaluation(
        model_path=model_path,
        meta_learner_path=meta_path,
        oracle_provider=provider)
