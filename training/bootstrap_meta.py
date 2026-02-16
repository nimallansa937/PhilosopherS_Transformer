"""
Phase 9 (CASCADE): Bootstrap the meta-learner with synthetic feedback.

Method:
1. Run small model on held-out Descartes questions
2. Run oracle on same questions
3. Compare answers → ground truth labels
4. Pre-train meta-learner on these labels
5. Deploy with warm-started meta-learner

Cost: ~$2-5 (500 oracle calls via Ollama Cloud)
Time: ~2-4 hours (small model generation bottleneck)

Usage:
    python training/bootstrap_meta.py
    python training/bootstrap_meta.py --local descartes:8b --oracle deepseek-v3.1:671-cloud
    python training/bootstrap_meta.py --max-questions 100  # Quick test
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project dirs to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

import torch
from signal_extractor_lite import LiteSignalExtractor
from meta_learner import MetaLearnerLite
from feedback import MetaTrainer

# Lazy import ollama
_ollama = None


def get_ollama():
    global _ollama
    if _ollama is None:
        import ollama as _ol
        _ollama = _ol
    return _ollama


def compute_agreement(text_a: str, text_b: str) -> float:
    """Jaccard similarity on content words."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "and", "that", "this", "it", "with"}
    words_a = set(text_a.lower().split()) - stop
    words_b = set(text_b.lower().split()) - stop
    if not words_a or not words_b:
        return 0.5
    return len(words_a & words_b) / len(words_a | words_b)


def bootstrap(
    local_model: str = "descartes:8b",
    oracle_model: str = "deepseek-v3.1:671-cloud",
    questions_path: str = None,
    output_path: str = None,
    max_questions: int = 500
):
    if questions_path is None:
        questions_path = str(
            PROJECT_ROOT / "training" / "eval" / "bootstrap_questions.jsonl")
    if output_path is None:
        output_path = str(
            PROJECT_ROOT / "models" / "meta_learner_bootstrap.pt")

    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load questions
    if not os.path.exists(questions_path):
        print(f"ERROR: Questions file not found: {questions_path}")
        print("Run: python training/eval/generate_bootstrap_questions.py")
        sys.exit(1)

    with open(questions_path) as f:
        questions = [json.loads(line)["question"]
                     for line in f][:max_questions]

    print(f"Bootstrapping on {len(questions)} questions")
    print(f"  Local: {local_model}")
    print(f"  Oracle: {oracle_model}")

    ol = get_ollama()
    extractor = LiteSignalExtractor()
    meta = MetaLearnerLite(input_dim=11)
    trainer = MetaTrainer(meta, lr=5e-4, update_every=4)

    errors = 0

    for i, q in enumerate(questions):
        try:
            # Small model answer
            local_resp = ol.chat(
                model=local_model,
                messages=[{"role": "user", "content": q}]
            )
            local_text = local_resp['message']['content']

            # Oracle answer
            oracle_resp = ol.chat(
                model=oracle_model,
                messages=[{
                    "role": "user",
                    "content": f"Answer this philosophical question "
                               f"accurately and in detail:\n\n{q}"
                }]
            )
            oracle_text = oracle_resp['message']['content']

            # Extract signals from local response
            signals = extractor.extract(local_text)
            signal_tensor = signals.to_tensor()

            # Meta-learner prediction (will be random initially)
            meta.eval()
            with torch.no_grad():
                pred = meta(signal_tensor)

            # Compute ground truth from comparison
            agreement = compute_agreement(local_text, oracle_text)

            outcome = {
                "oracle_agreed": agreement > 0.6,
                "correction_magnitude": 1.0 - agreement,
                "z3_verified": None,
                "user_accepted": None,
            }

            trainer.record_and_maybe_train(
                pred["features"].detach(),
                pred["confidence"].item(),
                pred["routing_decision"],
                outcome
            )

        except Exception as e:
            errors += 1
            print(f"  [{i+1}] Error: {e}")
            if errors > 10:
                print("Too many errors. Aborting.")
                break
            continue

        if (i + 1) % 50 == 0:
            stats = trainer.get_stats()
            loss_str = (f"loss={stats['avg_loss']:.4f}"
                        if stats['avg_loss'] else "loss=N/A")
            print(f"  [{i+1}/{len(questions)}] "
                  f"updates={stats['updates']}, {loss_str}")

    # Save bootstrapped meta-learner
    trainer.save(output_path)

    print(f"\nBootstrap complete.")
    print(f"  Updates: {trainer.update_count}")
    print(f"  Buffer: {len(trainer.buffer.buffer)} examples")
    print(f"  Errors: {errors}")
    print(f"  Saved: {output_path}")

    # Also save as the standard name for Phase 9 validation
    standard_path = str(
        PROJECT_ROOT / "models" / "meta_learner_bootstrapped.pt")
    if output_path != standard_path:
        trainer.save(standard_path)
        print(f"  Also saved: {standard_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap meta-learner with Ollama")
    parser.add_argument(
        "--local", default="descartes:8b",
        help="Local model name in Ollama")
    parser.add_argument(
        "--oracle", default="deepseek-v3.1:671-cloud",
        help="Oracle model name in Ollama")
    parser.add_argument(
        "--max-questions", type=int, default=500,
        help="Maximum bootstrap questions to use")
    parser.add_argument(
        "--questions", default=None,
        help="Path to bootstrap questions JSONL")
    parser.add_argument(
        "--output", default=None,
        help="Output path for bootstrapped meta-learner")

    args = parser.parse_args()

    bootstrap(
        local_model=args.local,
        oracle_model=args.oracle,
        questions_path=args.questions,
        output_path=args.output,
        max_questions=args.max_questions,
    )
