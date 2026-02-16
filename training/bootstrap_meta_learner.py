"""
Phase 9 (CASCADE): Bootstrap the meta-learner with synthetic
feedback data before deploying to production.

Replaces the simple temperature-scaling calibrator with a
proper neural meta-learner that learns from oracle interactions.

Method:
1. Run the small model on 500-1000 held-out Descartes questions
2. Simultaneously run the oracle on the same questions
3. Compare answers to generate ground-truth labels
4. Pre-train the meta-learner on these labels
5. Deploy with a warm-started meta-learner that improves online

The meta-learner is strictly more powerful than Platt scaling —
it predicts confidence, routing, AND error type from model
internals rather than just text-based confidence tags.

Cost estimate:
  500 oracle calls x ~500 tokens x $0.0002/1K = $0.05 oracle
  + ~2-4 hours GPU time = ~$2 GPU
  Total: ~$2-$3

Usage:
    python training/bootstrap_meta_learner.py [model_path] [--provider deepseek]
"""

import torch
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

from meta_learner import MetaLearner, MetaLearnerTrainer, ModelSignals
from signal_extractor import SignalExtractor
from oracle import OracleClient, OracleConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# BOOTSTRAP QUESTIONS — held-out Descartes evaluation set
# ============================================================

BOOTSTRAP_QUESTIONS = [
    # SELF-answerable (core Cartesian expertise)
    {"question": "Formalize the Cogito in Z3 as a strict inference.",
     "expected_routing": "SELF"},
    {"question": "What is the logical structure of the Real Distinction "
                 "argument?",
     "expected_routing": "SELF"},
    {"question": "Decompose the Trademark Argument into ASPIC+ premises.",
     "expected_routing": "SELF"},
    {"question": "Is the Cogito an inference or an intuition?",
     "expected_routing": "SELF"},
    {"question": "Formalize Arnauld's Cartesian Circle objection.",
     "expected_routing": "SELF"},
    {"question": "What is the role of God in validating clear and "
                 "distinct perception?",
     "expected_routing": "SELF"},
    {"question": "Reconstruct the Wax Argument as an elimination argument.",
     "expected_routing": "SELF"},
    {"question": "What are the three types of ideas in Meditation III?",
     "expected_routing": "SELF"},
    {"question": "Formalize substance dualism: mind and body have "
                 "different essential properties.",
     "expected_routing": "SELF"},
    {"question": "What is the dream argument and how does it differ "
                 "from the evil genius?",
     "expected_routing": "SELF"},

    # ORACLE-needed (broad philosophical knowledge)
    {"question": "What was Merleau-Ponty's critique of Cartesian "
                 "dualism?",
     "expected_routing": "ORACLE"},
    {"question": "How did Kant respond to the ontological argument "
                 "that Descartes also used?",
     "expected_routing": "ORACLE"},
    {"question": "Compare Descartes' method of doubt to Husserl's "
                 "phenomenological reduction.",
     "expected_routing": "ORACLE"},
    {"question": "What is the Cambridge Declaration on Consciousness "
                 "and how does it relate to Descartes' animal automata?",
     "expected_routing": "ORACLE"},
    {"question": "How did the Jesuits at La Fleche receive Descartes' "
                 "philosophy?",
     "expected_routing": "ORACLE"},
    {"question": "What does Daniel Dennett say about Cartesian "
                 "dualism in Consciousness Explained?",
     "expected_routing": "ORACLE"},
    {"question": "How did the condemnation of Descartes' works by "
                 "the Catholic Church affect early modern philosophy?",
     "expected_routing": "ORACLE"},
    {"question": "What is Gilbert Ryle's 'ghost in the machine' "
                 "critique?",
     "expected_routing": "ORACLE"},
    {"question": "How does Jaegwon Kim's exclusion argument apply "
                 "to Cartesian interactionism?",
     "expected_routing": "ORACLE"},
    {"question": "What does contemporary neuroscience say about the "
                 "pineal gland's actual function?",
     "expected_routing": "ORACLE"},

    # HYBRID (formal analysis + external knowledge)
    {"question": "Is the Real Distinction argument structurally "
                 "identical to Chalmers' zombie argument?",
     "expected_routing": "HYBRID"},
    {"question": "Can Descartes' causal adequacy principle be "
                 "reconciled with modern physicalism?",
     "expected_routing": "HYBRID"},
    {"question": "Does the Global Workspace Theory conflict with "
                 "substance dualism? Check formally.",
     "expected_routing": "HYBRID"},
    {"question": "Formalize both Descartes' ontological argument "
                 "and Kant's objection in Z3.",
     "expected_routing": "HYBRID"},
    {"question": "Connect Cartesian certainty to Bayesian predictive "
                 "confidence — is there a formal mapping?",
     "expected_routing": "HYBRID"},
]


def bootstrap(model_path: str,
              output_path: str,
              oracle_config: OracleConfig = None,
              max_questions: int = 500):
    """Bootstrap meta-learner from small-model vs oracle comparison."""

    print("=" * 60)
    print("PHASE 9: Meta-Learner Bootstrap")
    print("=" * 60)

    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    extractor = SignalExtractor(model, tokenizer)
    oracle = OracleClient(oracle_config or OracleConfig())

    meta = MetaLearner(hidden_dim=model.config.hidden_size)
    trainer = MetaLearnerTrainer(meta, lr=5e-4)  # Higher LR for bootstrap

    # Use built-in questions + any external questions file
    questions = BOOTSTRAP_QUESTIONS[:max_questions]

    # Also check for external questions file
    ext_path = PROJECT_ROOT / "training" / "eval" / "bootstrap_questions.jsonl"
    if ext_path.exists():
        with open(ext_path) as f:
            for line in f:
                if len(questions) >= max_questions:
                    break
                try:
                    q = json.loads(line.strip())
                    questions.append(q)
                except json.JSONDecodeError:
                    pass

    print(f"Bootstrapping on {len(questions)} questions...")
    print(f"Oracle provider: {oracle.config.provider}")

    system_prompt = (
        "You are a philosophical reasoning assistant specializing "
        "in Cartesian philosophy. Express confidence as "
        "[CONFIDENCE: 0.X] at the end of each response."
    )

    for i, q_data in enumerate(questions):
        question = q_data["question"]

        # Generate from small model
        extractor.clear()

        prompt = (f"<|system|>\n{system_prompt}\n"
                  f"<|user|>\n{question}\n<|assistant|>\n")
        inputs = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=4096
                          ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=0.3, do_sample=True,
                pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

        signals = extractor.extract_signals(
            inputs["input_ids"], outputs[0:1], response)

        # Get oracle answer
        oracle_response = oracle.query(question)

        # Compute agreement
        stop = {"the", "a", "an", "is", "are", "was", "were",
                "in", "on", "at", "to", "for", "of", "and"}
        words_s = set(response.lower().split()) - stop
        words_o = set(oracle_response.lower().split()) - stop
        agreement = (len(words_s & words_o) /
                     max(len(words_s | words_o), 1))

        # Record as training data
        with torch.no_grad():
            meta_out = meta(signals)

        outcome = {
            "oracle_agreed": agreement > 0.7,
            "correction_magnitude": 1.0 - agreement,
            "z3_verified": None,
            "user_accepted": None,
        }

        trainer.record_outcome(
            meta_out["features"],
            meta_out["confidence"].item(),
            meta_out["routing_decision"],
            outcome
        )

        if (i + 1) % 10 == 0:
            stats = {
                "questions": i + 1,
                "meta_updates": trainer.update_count,
                "oracle_cost": round(oracle.total_cost, 4),
            }
            if trainer.loss_history:
                stats["avg_loss"] = round(
                    sum(trainer.loss_history) /
                    len(trainer.loss_history), 4)
            print(f"  [{i+1}/{len(questions)}] {json.dumps(stats)}")

    # Save bootstrapped meta-learner
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(output_path)

    print(f"\n{'=' * 60}")
    print(f"BOOTSTRAP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Meta-learner updates: {trainer.update_count}")
    print(f"  Oracle calls: {oracle.total_calls}")
    print(f"  Oracle cost: ${oracle.total_cost:.2f}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    model_path = (sys.argv[1] if len(sys.argv) > 1
                  else str(PROJECT_ROOT / "models" / "descartes-8b-cascade"))

    output_path = str(
        PROJECT_ROOT / "models" / "meta_learner_bootstrapped.pt")

    provider = "deepseek"
    for i, arg in enumerate(sys.argv):
        if arg == "--provider" and i + 1 < len(sys.argv):
            provider = sys.argv[i + 1]

    oracle_config = OracleConfig(provider=provider)

    bootstrap(model_path, output_path,
              oracle_config=oracle_config,
              max_questions=len(BOOTSTRAP_QUESTIONS))
