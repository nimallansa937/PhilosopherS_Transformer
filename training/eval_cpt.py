"""
Phase 5 Validation: Evaluate CPT model.

Compares philosophy perplexity and general capability retention
between base model and CPT-adapted model.

PASS CRITERIA:
1. Philosophy perplexity should DROP (lower = better)
2. General perplexity should NOT increase by more than 10%
3. Training loss should have converged

Usage:
    python training/eval_cpt.py --base meta-llama/Meta-Llama-3.1-70B --cpt models/philosopher-cpt-70b
"""

import torch
import math
import json
import argparse
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_perplexity(model, tokenizer, texts: List[str],
                       max_length: int = 2048) -> float:
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


# ---- Test texts ----

PHILOSOPHY_TEXTS = [
    "The hard problem of consciousness concerns the question of why and "
    "how physical processes in the brain give rise to subjective experience. "
    "Even if we could explain all the functional and behavioral aspects of "
    "consciousness, there would remain the further question of why these "
    "processes are accompanied by phenomenal experience at all.",

    "The zombie argument proceeds from the premise that it is conceivable "
    "that there exists a being physically identical to a conscious being "
    "but lacking any phenomenal experience. If conceivability entails "
    "metaphysical possibility, then such zombie worlds are possible, "
    "which contradicts the physicalist thesis that phenomenal properties "
    "supervene with metaphysical necessity on physical properties.",

    "Integrated Information Theory proposes that consciousness corresponds "
    "to integrated information, measured as phi. A system is conscious to "
    "the degree that it is both differentiated and integrated. The theory "
    "predicts that a photodiode, with minimal integration, has negligible "
    "consciousness, while the cerebral cortex, with its rich recurrent "
    "connectivity, generates high phi values.",

    "The knowledge argument asks us to imagine Mary, a brilliant scientist "
    "who has spent her entire life in a black-and-white room, studying "
    "the neurophysiology of color vision. She knows every physical fact "
    "about color perception. Yet when she finally leaves the room and "
    "sees red for the first time, she learns something new. Therefore, "
    "there are non-physical facts about consciousness.",

    "Global Workspace Theory holds that consciousness arises when "
    "information is broadcast globally across the cortex via a "
    "distributed network of neurons. Unconscious processing occurs "
    "in specialized modules, but when information gains access to "
    "the global workspace, it becomes available for report, decision-making, "
    "and flexible behavioral control.",
]

GENERAL_TEXTS = [
    "The process of photosynthesis converts carbon dioxide and water into "
    "glucose and oxygen using light energy from the sun. This occurs in "
    "the chloroplasts of plant cells, specifically in the thylakoid "
    "membranes where light-dependent reactions take place.",

    "To implement a binary search algorithm, first sort the array. Then "
    "compare the target value to the middle element. If the target is less "
    "than the middle element, search the left half; otherwise search the right. "
    "This gives O(log n) time complexity.",

    "The Treaty of Westphalia in 1648 established the principle of "
    "state sovereignty in international relations. It ended the Thirty "
    "Years War and created the modern concept of the nation-state, "
    "where each sovereign entity has authority within its borders.",

    "In macroeconomics, the Phillips curve describes an inverse "
    "relationship between unemployment and inflation. When unemployment "
    "is low, wages tend to rise faster, pushing up prices. However, "
    "the long-run Phillips curve is believed to be vertical.",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate CPT model")
    parser.add_argument("--base", type=str, default="meta-llama/Meta-Llama-3.1-70B",
                        help="Base model name/path")
    parser.add_argument("--cpt", type=str,
                        default=str(PROJECT_ROOT / "models" / "philosopher-cpt-70b"),
                        help="CPT model path")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 5 VALIDATION: CPT MODEL EVALUATION")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load base model
        print(f"\nLoading base model: {args.base}")
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )

        # Load CPT model
        print(f"Loading CPT model: {args.cpt}")
        cpt_model = AutoModelForCausalLM.from_pretrained(
            args.cpt,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )

        # Evaluate
        print("\nComputing perplexities...")

        base_phil_ppl = compute_perplexity(base_model, tokenizer, PHILOSOPHY_TEXTS)
        cpt_phil_ppl = compute_perplexity(cpt_model, tokenizer, PHILOSOPHY_TEXTS)

        base_gen_ppl = compute_perplexity(base_model, tokenizer, GENERAL_TEXTS)
        cpt_gen_ppl = compute_perplexity(cpt_model, tokenizer, GENERAL_TEXTS)

        # Results
        phil_change = (cpt_phil_ppl - base_phil_ppl) / base_phil_ppl * 100
        gen_change = (cpt_gen_ppl - base_gen_ppl) / base_gen_ppl * 100

        results = {
            "philosophy_perplexity": {
                "base": round(base_phil_ppl, 2),
                "cpt": round(cpt_phil_ppl, 2),
                "change_pct": round(phil_change, 2),
                "pass": cpt_phil_ppl < base_phil_ppl,
            },
            "general_perplexity": {
                "base": round(base_gen_ppl, 2),
                "cpt": round(cpt_gen_ppl, 2),
                "change_pct": round(gen_change, 2),
                "pass": gen_change <= 10.0,
            },
            "overall_pass": (cpt_phil_ppl < base_phil_ppl) and (gen_change <= 10.0),
        }

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"\nPhilosophy Perplexity:")
        print(f"  Base: {base_phil_ppl:.2f}")
        print(f"  CPT:  {cpt_phil_ppl:.2f} ({phil_change:+.1f}%)")
        print(f"  PASS: {'YES' if results['philosophy_perplexity']['pass'] else 'NO'}")

        print(f"\nGeneral Perplexity:")
        print(f"  Base: {base_gen_ppl:.2f}")
        print(f"  CPT:  {cpt_gen_ppl:.2f} ({gen_change:+.1f}%)")
        print(f"  PASS: {'YES' if results['general_perplexity']['pass'] else 'NO'}")

        print(f"\nOVERALL: {'PASS' if results['overall_pass'] else 'FAIL'}")

        # Save results
        results_path = PROJECT_ROOT / "training" / "eval" / "cpt_eval_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {results_path}")

    except ImportError as e:
        print(f"\nCannot run evaluation: {e}")
        print("Install with: pip install torch transformers")
        print("\nTest data is defined. Run this script on a GPU machine with models loaded.")


if __name__ == "__main__":
    main()
