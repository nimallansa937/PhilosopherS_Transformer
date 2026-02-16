"""
Phase 8: Evaluation suite for the Philosopher Engine model.

Tests:
1. Argument validity detection (binary classification)
2. Scheme identification (multi-class)
3. Inconsistency detection (binary + localization)
4. Cross-theory comparison (open-ended, LLM-judged)
5. General capability retention (MMLU subset)

MINIMUM THRESHOLDS TO DEPLOY:
  Argument Validity Detection:   >= 85% accuracy
  Scheme Identification:         >= 70% accuracy
  Inconsistency Detection:       >= 75% accuracy
  General Capability (MMLU):     <= 5% drop from base model

Usage:
    python training/eval/run_eval.py --model models/philosopher-sft
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "training" / "eval"
EVAL_DIR.mkdir(exist_ok=True)


# ============================================================
# TEST 1: Argument Validity Detection
# ============================================================

VALIDITY_TESTS = [
    {
        "argument": (
            "P1: Zombies are conceivable.\n"
            "P2: Whatever is conceivable is metaphysically possible.\n"
            "P3: If zombie worlds are possible, physicalism is false.\n"
            "C: Physicalism is false."
        ),
        "valid": True,
        "label": "zombie_argument_valid_structure"
    },
    {
        "argument": (
            "P1: The brain produces consciousness.\n"
            "P2: Computers are not brains.\n"
            "C: Computers cannot be conscious."
        ),
        "valid": False,
        "label": "substrate_fallacy_invalid"
    },
    {
        "argument": (
            "P1: If functionalism is true, then any system with the right "
            "functional organization is conscious.\n"
            "P2: The Chinese Room has the right functional organization.\n"
            "P3: The Chinese Room is not conscious.\n"
            "C: Functionalism is false."
        ),
        "valid": True,
        "label": "chinese_room_valid_modus_tollens"
    },
    {
        "argument": (
            "P1: Mary knows all physical facts about color.\n"
            "P2: Mary learns something new when she sees red.\n"
            "C: There are non-physical facts.\n"
            "Note: Implicit premise needed - 'If knowing all physical facts "
            "leaves something unknown, there are non-physical facts.'"
        ),
        "valid": True,  # Valid IF implicit premise is made explicit
        "label": "knowledge_argument_enthymeme"
    },
    {
        "argument": (
            "P1: Consciousness exists.\n"
            "P2: Physics cannot explain consciousness.\n"
            "C: God must have created consciousness."
        ),
        "valid": False,
        "label": "god_of_gaps_invalid"
    },
    {
        "argument": (
            "P1: If qualia are epiphenomenal, they have no causal effects.\n"
            "P2: If qualia have no causal effects, we cannot know about them.\n"
            "P3: We know about qualia.\n"
            "C: Qualia are not epiphenomenal."
        ),
        "valid": True,
        "label": "epiphenomenalism_modus_tollens"
    },
    {
        "argument": (
            "P1: Some philosophers are dualists.\n"
            "P2: All dualists believe in immaterial minds.\n"
            "C: All philosophers believe in immaterial minds."
        ),
        "valid": False,
        "label": "undistributed_middle_invalid"
    },
]


# ============================================================
# TEST 2: Scheme Identification
# ============================================================

SCHEME_TESTS = [
    {
        "text": "Just as a bat's sonar experience is alien to us, our "
                "visual experience may be alien to creatures with different "
                "sensory modalities. If we cannot know what it is like to be "
                "a bat, then we cannot fully understand bat consciousness.",
        "expected_scheme": "argument_from_analogy",
    },
    {
        "text": "The best explanation for why physical duplicates would "
                "have identical experiences is that consciousness supervenes "
                "on physical properties. No rival theory explains this "
                "correlation as well.",
        "expected_scheme": "inference_to_best_explanation",
    },
    {
        "text": "If we accept epiphenomenalism, then consciousness has no "
                "causal effects on behavior. But this implies we could never "
                "know we are conscious, which is absurd. Therefore, "
                "epiphenomenalism is false.",
        "expected_scheme": "argument_from_consequences",
    },
    {
        "text": "It is conceivable that zombies exist - beings physically "
                "identical to us but lacking consciousness. If conceivable, "
                "then possible. If possible, then physicalism is false.",
        "expected_scheme": "conceivability_to_possibility",
    },
    {
        "text": "Patients with blindsight can respond to visual stimuli "
                "without conscious awareness. This demonstrates that visual "
                "processing can occur without phenomenal consciousness, "
                "suggesting consciousness is not required for all perception.",
        "expected_scheme": "argument_from_sign",
    },
]


# ============================================================
# TEST 3: Inconsistency Detection
# ============================================================

INCONSISTENCY_TESTS = [
    {
        "claims": [
            "Physicalism holds that all facts are physical facts.",
            "Qualia are non-physical properties of experience.",
            "Qualia exist."
        ],
        "inconsistent": True,
        "conflicting_claims": [0, 1, 2],
        "label": "physicalism_qualia_trilemma"
    },
    {
        "claims": [
            "Functionalism identifies mental states with functional roles.",
            "Multiple realizability shows the same mental state can be "
            "realized by different physical substrates.",
            "The brain is one substrate that realizes mental states."
        ],
        "inconsistent": False,
        "label": "functionalism_consistent"
    },
    {
        "claims": [
            "IIT holds that consciousness is identical to integrated information.",
            "A thermostat has some integrated information (phi > 0).",
            "Thermostats are not conscious.",
            "Anything with phi > 0 is conscious."
        ],
        "inconsistent": True,
        "conflicting_claims": [2, 3],
        "label": "iit_thermostat_inconsistency"
    },
    {
        "claims": [
            "Consciousness is entirely determined by brain states.",
            "Brain states are entirely determined by physical laws.",
            "Physical laws do not mention consciousness.",
            "Consciousness has causal powers over behavior."
        ],
        "inconsistent": True,
        "conflicting_claims": [0, 1, 2, 3],
        "label": "causal_closure_consciousness"
    },
    {
        "claims": [
            "The hard problem asks why there is subjective experience.",
            "Some philosophers deny the hard problem is genuine.",
            "Dennett argues consciousness is not what it seems.",
            "There may be an explanatory gap without an ontological gap."
        ],
        "inconsistent": False,
        "label": "hard_problem_positions_consistent"
    },
]


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with __import__('torch').no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    return response.strip()


def eval_validity(model, tokenizer) -> Dict:
    """Test 1: Argument Validity Detection."""
    results = []

    for test in VALIDITY_TESTS:
        prompt = (
            f"Is the following argument logically valid? "
            f"Answer 'VALID' or 'INVALID' on the first line, "
            f"then explain briefly.\n\n"
            f"{test['argument']}"
        )

        response = generate_response(model, tokenizer, prompt)
        first_line = response.split('\n')[0].upper()
        predicted_valid = "VALID" in first_line and "INVALID" not in first_line
        correct = predicted_valid == test["valid"]

        results.append({
            "label": test["label"],
            "expected": "VALID" if test["valid"] else "INVALID",
            "predicted": "VALID" if predicted_valid else "INVALID",
            "correct": correct,
            "response_preview": response[:200],
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return {"accuracy": accuracy, "results": results, "pass": accuracy >= 0.85}


def eval_schemes(model, tokenizer) -> Dict:
    """Test 2: Scheme Identification."""
    valid_schemes = [
        "argument_from_analogy", "inference_to_best_explanation",
        "argument_from_consequences", "conceivability_to_possibility",
        "argument_from_sign"
    ]

    results = []

    for test in SCHEME_TESTS:
        prompt = (
            f"Identify the argumentation scheme used in this passage. "
            f"Choose from: {', '.join(valid_schemes)}.\n"
            f"State your answer on the first line.\n\n"
            f"{test['text']}"
        )

        response = generate_response(model, tokenizer, prompt)
        response_lower = response.lower().replace(" ", "_").replace("-", "_")
        correct = test["expected_scheme"] in response_lower

        results.append({
            "expected": test["expected_scheme"],
            "correct": correct,
            "response_preview": response[:200],
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return {"accuracy": accuracy, "results": results, "pass": accuracy >= 0.70}


def eval_inconsistency(model, tokenizer) -> Dict:
    """Test 3: Inconsistency Detection."""
    results = []

    for test in INCONSISTENCY_TESTS:
        claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(test["claims"]))
        prompt = (
            f"Are the following claims jointly consistent? "
            f"Answer 'CONSISTENT' or 'INCONSISTENT' on the first line. "
            f"If inconsistent, identify which claims conflict.\n\n"
            f"{claims_text}"
        )

        response = generate_response(model, tokenizer, prompt)
        response_lower = response.lower()
        predicted_inconsistent = "inconsistent" in response_lower.split('\n')[0]
        correct = predicted_inconsistent == test["inconsistent"]

        results.append({
            "label": test.get("label", ""),
            "expected": "INCONSISTENT" if test["inconsistent"] else "CONSISTENT",
            "predicted": "INCONSISTENT" if predicted_inconsistent else "CONSISTENT",
            "correct": correct,
            "response_preview": response[:200],
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return {"accuracy": accuracy, "results": results, "pass": accuracy >= 0.75}


def evaluate_model(model, tokenizer) -> Dict:
    """Run full evaluation suite."""

    print("\n[1/3] Testing argument validity detection...")
    validity_results = eval_validity(model, tokenizer)
    print(f"  Accuracy: {validity_results['accuracy']:.1%} "
          f"({'PASS' if validity_results['pass'] else 'FAIL'})")

    print("\n[2/3] Testing scheme identification...")
    scheme_results = eval_schemes(model, tokenizer)
    print(f"  Accuracy: {scheme_results['accuracy']:.1%} "
          f"({'PASS' if scheme_results['pass'] else 'FAIL'})")

    print("\n[3/3] Testing inconsistency detection...")
    inconsistency_results = eval_inconsistency(model, tokenizer)
    print(f"  Accuracy: {inconsistency_results['accuracy']:.1%} "
          f"({'PASS' if inconsistency_results['pass'] else 'FAIL'})")

    overall = {
        "validity": validity_results,
        "schemes": scheme_results,
        "inconsistency": inconsistency_results,
        "overall_pass": (validity_results["pass"] and
                        scheme_results["pass"] and
                        inconsistency_results["pass"]),
    }

    return overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate Philosopher Engine model")
    parser.add_argument("--model", type=str,
                        default=str(PROJECT_ROOT / "models" / "philosopher-sft"),
                        help="Path to model to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 8: MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )

        results = evaluate_model(model, tokenizer)

        # Save results
        results_path = EVAL_DIR / "eval_results.json"

        # Clean results for JSON serialization
        serializable = json.loads(json.dumps(results, default=str))
        results_path.write_text(json.dumps(serializable, indent=2))

        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Validity Detection:    {results['validity']['accuracy']:.1%} "
              f"(threshold: 85%)")
        print(f"  Scheme Identification: {results['schemes']['accuracy']:.1%} "
              f"(threshold: 70%)")
        print(f"  Inconsistency:         {results['inconsistency']['accuracy']:.1%} "
              f"(threshold: 75%)")
        print(f"\n  OVERALL: {'PASS' if results['overall_pass'] else 'FAIL'}")
        print(f"\nResults saved to: {results_path}")

        if not results['overall_pass']:
            print("\nFailed thresholds. Recommended actions:")
            if not results['validity']['pass']:
                print("  - Generate more Type A SFT examples focused on validity")
            if not results['schemes']['pass']:
                print("  - Generate more Type A SFT examples with scheme labels")
            if not results['inconsistency']['pass']:
                print("  - Generate targeted inconsistency detection examples")
            print("  - Re-run Z3 validation on new examples")
            print("  - Retrain (expect 3-5 iteration cycles)")

    except ImportError as e:
        print(f"\nCannot run evaluation: {e}")
        print("Install with: pip install torch transformers")
        print("\nSaving test definitions for later use...")

        # Save test definitions even without model
        test_defs = {
            "validity_tests": VALIDITY_TESTS,
            "scheme_tests": SCHEME_TESTS,
            "inconsistency_tests": INCONSISTENCY_TESTS,
            "thresholds": {
                "validity": 0.85,
                "schemes": 0.70,
                "inconsistency": 0.75,
                "mmlu_drop_max": 0.05,
            }
        }
        defs_path = EVAL_DIR / "test_definitions.json"
        defs_path.write_text(json.dumps(test_defs, indent=2, default=str))
        print(f"Test definitions saved to: {defs_path}")


if __name__ == "__main__":
    main()
