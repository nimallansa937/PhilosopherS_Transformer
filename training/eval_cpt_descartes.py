"""
Phase 5 Validation: Compare base Qwen3-8B vs CPT model on
Descartes-specific and general text perplexity.

Usage:
    python training/eval_cpt_descartes.py [base_model] [cpt_model_path]

Defaults:
    base_model = Qwen/Qwen3-8B
    cpt_model  = models/descartes-8b-cpt
"""

import torch
import math
import sys
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_perplexity(model, tokenizer, texts, max_length=2048):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    return math.exp(total_loss / total_tokens)


# ---- Descartes-Specific Test Texts ----
DESCARTES_TEXTS = [
    "The Cogito -- I think, therefore I am -- is not a syllogism with "
    "a suppressed major premise, but rather an immediate intuition. "
    "Descartes clarifies in the Second Replies that the certainty of "
    "the Cogito does not depend on the prior knowledge of the major "
    "premise 'whatever thinks exists,' but is grasped by a simple "
    "act of mental intuition.",

    "The Real Distinction argument in the Sixth Meditation proceeds "
    "from clear and distinct perception to metaphysical possibility "
    "to actual distinctness. Because I can clearly and distinctly "
    "conceive of mind apart from body and body apart from mind, God "
    "could create them separately, therefore they are really distinct "
    "substances. This argument has a modal structure: conceivability "
    "entails possibility entails actual distinctness.",

    "Elisabeth's objection to Descartes concerns the interaction "
    "problem: if mind is unextended thinking substance and body is "
    "extended non-thinking substance, how can they causally interact? "
    "Extension seems required for contact, and contact seems required "
    "for causation. Descartes' responses in the correspondence invoke "
    "a primitive notion of mind-body union that does not reduce to "
    "either thought or extension alone.",

    "The Cartesian Circle objection, raised by Arnauld in the Fourth "
    "Objections, claims that Descartes' argument is circular: he uses "
    "clear and distinct perception to prove God's existence, then "
    "uses God's existence to validate clear and distinct perception. "
    "Descartes responds that the Cogito and the divine guarantee "
    "operate at different levels -- present certainty versus memory "
    "of past demonstrations.",
]

# ---- General Knowledge Test Texts ----
GENERAL_TEXTS = [
    "Photosynthesis converts carbon dioxide and water into glucose "
    "using light energy captured by chlorophyll in chloroplasts.",

    "Binary search operates on sorted arrays by repeatedly halving "
    "the search interval until the target is found or the interval "
    "is empty, achieving O(log n) time complexity.",
]


def evaluate(base_model_name: str, cpt_model_path: str):
    """Run full perplexity comparison."""

    print("=" * 60)
    print("PHASE 5 VALIDATION: CPT Perplexity Comparison")
    print("=" * 60)

    # ---- Load base model ----
    print(f"\nLoading base model: {base_model_name}...")
    base_tok = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)

    # ---- Load CPT model ----
    print(f"Loading CPT model: {cpt_model_path}...")
    cpt_tok = AutoTokenizer.from_pretrained(
        cpt_model_path, trust_remote_code=True)
    cpt_model = AutoModelForCausalLM.from_pretrained(
        cpt_model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)

    # ---- Evaluate ----
    print("\nComputing perplexities...")
    base_des = compute_perplexity(base_model, base_tok, DESCARTES_TEXTS)
    cpt_des = compute_perplexity(cpt_model, cpt_tok, DESCARTES_TEXTS)
    base_gen = compute_perplexity(base_model, base_tok, GENERAL_TEXTS)
    cpt_gen = compute_perplexity(cpt_model, cpt_tok, GENERAL_TEXTS)

    # ---- Report ----
    print(f"\n{'=' * 60}")
    print(f"{'Metric':<30} {'Base':>8} {'CPT':>8} {'Delta':>8}")
    print(f"{'=' * 60}")
    print(f"{'Descartes perplexity':<30} {base_des:>8.1f} {cpt_des:>8.1f} "
          f"{(cpt_des - base_des) / base_des * 100:>+7.1f}%")
    print(f"{'General perplexity':<30} {base_gen:>8.1f} {cpt_gen:>8.1f} "
          f"{(cpt_gen - base_gen) / base_gen * 100:>+7.1f}%")
    print(f"{'=' * 60}")

    # ---- Pass criteria ----
    des_improved = cpt_des < base_des
    gen_retained = (cpt_gen - base_gen) / base_gen < 0.15

    print(f"\nDescartes improved: {'PASS' if des_improved else 'FAIL'}")
    print(f"General retained (<15% increase): {'PASS' if gen_retained else 'FAIL'}")

    # ---- Save results ----
    results = {
        "base_model": base_model_name,
        "cpt_model": cpt_model_path,
        "descartes_ppl_base": round(base_des, 2),
        "descartes_ppl_cpt": round(cpt_des, 2),
        "descartes_delta_pct": round((cpt_des - base_des) / base_des * 100, 2),
        "general_ppl_base": round(base_gen, 2),
        "general_ppl_cpt": round(cpt_gen, 2),
        "general_delta_pct": round((cpt_gen - base_gen) / base_gen * 100, 2),
        "pass_descartes": des_improved,
        "pass_general": gen_retained,
        "overall_pass": des_improved and gen_retained,
    }

    out_path = PROJECT_ROOT / "training" / "eval" / "cpt_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    return des_improved and gen_retained


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    cpt = sys.argv[2] if len(sys.argv) > 2 else str(
        PROJECT_ROOT / "models" / "descartes-8b-cpt")
    evaluate(base, cpt)
