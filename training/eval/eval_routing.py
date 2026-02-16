"""
Test whether the meta-learner correctly routes queries.
Run AFTER bootstrap but BEFORE production deployment.

Uses the local Descartes model via Ollama to generate responses,
then checks if the meta-learner's routing decisions match
expected categories.

Pass Criteria: >= 80% routing accuracy on 12 held-out test queries.

Usage:
    python training/eval/eval_routing.py
    python training/eval/eval_routing.py --meta models/meta_learner_bootstrapped.pt
    python training/eval/eval_routing.py --local descartes:8b
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from signal_extractor_lite import LiteSignalExtractor
from meta_learner import MetaLearnerLite, ROUTING_LABELS

# Lazy import ollama
_ollama = None


def get_ollama():
    global _ollama
    if _ollama is None:
        import ollama as _ol
        _ollama = _ol
    return _ollama


# ============================================================
# ROUTING TEST CASES
# ============================================================

ROUTING_TESTS = [
    # (query, expected_routing)
    # SELF: Core Cartesian formalization/analysis
    ("Formalize the Cogito in Z3.", "SELF"),
    ("Decompose Arnauld's Circle in ASPIC+.", "SELF"),
    ("Check consistency of Trademark Argument premises.", "SELF"),
    ("What modal logic does the Real Distinction use?", "SELF"),

    # ORACLE: Broad philosophy, historical context
    ("What did Husserl say about Cartesian doubt?", "ORACLE"),
    ("How was Descartes received by the Jesuits?", "ORACLE"),
    ("Compare Descartes' doubt with Pyrrhonian skepticism.", "ORACLE"),
    ("What was Malebranche's occasionalist response?", "ORACLE"),

    # HYBRID: Cartesian core + external knowledge needed
    ("Is the Real Distinction identical to the zombie argument?", "HYBRID"),
    ("Can GWT be reconciled with substance dualism?", "HYBRID"),
    ("Formalize Elisabeth's objection alongside Kim's exclusion.", "HYBRID"),
    ("Does Kripke's identity argument parallel Descartes'?", "HYBRID"),
]


def eval_routing(meta_path: str, local_model: str = "descartes:8b",
                 save_results: bool = True) -> dict:
    """Run routing accuracy evaluation.

    Args:
        meta_path: Path to meta-learner checkpoint
        local_model: Ollama model name for local generation
        save_results: Whether to save results JSON

    Returns:
        Dict with accuracy, per-query results, pass/fail
    """
    # Load meta-learner
    meta = MetaLearnerLite(input_dim=11)
    ckpt = torch.load(meta_path, map_location='cpu', weights_only=False)
    meta.load_state_dict(ckpt["model_state"])
    meta.eval()

    extractor = LiteSignalExtractor()
    ol = get_ollama()

    correct = 0
    total = len(ROUTING_TESTS)
    per_query = []

    print(f"\nRouting Accuracy Evaluation")
    print(f"Meta-learner: {meta_path}")
    print(f"Local model:  {local_model}")
    print(f"Test queries: {total}")
    print()
    print(f"{'Query':<55} {'Expected':>8} {'Predicted':>9} {'Match':>5}")
    print("-" * 85)

    for query, expected in ROUTING_TESTS:
        try:
            # Generate response from local model
            resp = ol.chat(
                model=local_model,
                messages=[{"role": "user", "content": query}]
            )
            response_text = resp['message']['content']

            # Extract signals and predict routing
            signals = extractor.extract(response_text)
            with torch.no_grad():
                pred = meta(signals.to_tensor())

            predicted = pred["routing_decision"]
            confidence = pred["confidence"].item()
            error_type = pred["error_type"]
            match = predicted == expected
            correct += int(match)

            q_short = query[:52] + "..." if len(query) > 55 else query
            mark = "Y" if match else "X"
            print(f"{q_short:<55} {expected:>8} {predicted:>9} {mark:>5}")

            per_query.append({
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "confidence": round(confidence, 3),
                "error_type": error_type,
                "match": match,
                "response_length": len(response_text.split()),
            })

        except Exception as e:
            print(f"{'ERROR: ' + query[:46]:<55} {expected:>8} {'ERROR':>9} {'X':>5}")
            print(f"  -> {e}")
            per_query.append({
                "query": query,
                "expected": expected,
                "predicted": "ERROR",
                "match": False,
                "error": str(e),
            })

    accuracy = correct / total if total > 0 else 0.0
    passed = accuracy >= 0.80

    print()
    print(f"Routing accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"{'PASS' if passed else 'FAIL'} (threshold: 80%)")

    # Per-category breakdown
    categories = {"SELF": [], "ORACLE": [], "HYBRID": []}
    for r in per_query:
        if r["expected"] in categories:
            categories[r["expected"]].append(r.get("match", False))

    print(f"\nPer-category accuracy:")
    for cat, matches in categories.items():
        if matches:
            cat_acc = sum(matches) / len(matches)
            print(f"  {cat:>6}: {sum(matches)}/{len(matches)} = {cat_acc:.0%}")

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "pass": passed,
        "threshold": 0.80,
        "meta_path": meta_path,
        "local_model": local_model,
        "per_query": per_query,
        "per_category": {
            cat: {
                "correct": sum(m),
                "total": len(m),
                "accuracy": sum(m) / len(m) if m else 0
            }
            for cat, m in categories.items()
        },
    }

    if save_results:
        results_path = PROJECT_ROOT / "training" / "eval" / "routing_eval_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate meta-learner routing accuracy")
    parser.add_argument(
        "--meta", type=str, default=None,
        help="Path to meta-learner checkpoint (.pt)")
    parser.add_argument(
        "--local", type=str, default="descartes:8b",
        help="Local Ollama model name")
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to disk")

    args = parser.parse_args()

    # Find meta-learner checkpoint
    meta_path = args.meta
    if meta_path is None:
        for candidate in [
            PROJECT_ROOT / "models" / "meta_learner_bootstrapped.pt",
            PROJECT_ROOT / "models" / "meta_learner_bootstrap.pt",
            PROJECT_ROOT / "models" / "meta_learner_latest.pt",
        ]:
            if candidate.exists():
                meta_path = str(candidate)
                break

    if meta_path is None or not os.path.exists(meta_path):
        print("ERROR: No meta-learner checkpoint found.")
        print("Run training/bootstrap_meta.py first, or pass --meta PATH")
        sys.exit(1)

    eval_routing(meta_path, args.local, save_results=not args.no_save)


if __name__ == "__main__":
    main()
