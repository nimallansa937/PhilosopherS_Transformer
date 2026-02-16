"""
Descartes Philosopher Engine — Ollama Unified + Meta-Learner.

This is the production inference system. Both local specialist
and cloud oracle accessed through ollama.chat(). Meta-learner
routes queries and improves with every oracle interaction.

Usage:
    python inference/engine.py

Prerequisites:
    1. Ollama installed and running
    2. descartes:8b model imported (see ADDENDUM Part 1.2)
    3. Cloud oracle configured (see ADDENDUM Part 1.3)
    4. Meta-learner bootstrapped (optional but recommended)
"""

import torch
import json
import os
import sys
from typing import Optional, Dict
from dataclasses import dataclass
from pathlib import Path

# Add inference dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_extractor_lite import LiteSignalExtractor
from meta_learner import MetaLearnerLite
from feedback import MetaTrainer

# Lazy import ollama — not needed for testing
_ollama = None


def get_ollama():
    """Lazy import ollama to allow module loading without it installed."""
    global _ollama
    if _ollama is None:
        import ollama as _ol
        _ollama = _ol
    return _ollama


# ============================================================
# SYSTEM PROMPTS
# ============================================================

DESCARTES_SYSTEM = (
    "You are a philosophical reasoning assistant specializing in "
    "Cartesian philosophy, early modern rationalism, and the "
    "mind-body problem. You analyze arguments using ASPIC+ "
    "argumentation schemes and Z3 formal verification. You have "
    "deep expertise in Descartes' Meditations, the Objections and "
    "Replies, the Correspondence with Elisabeth, and the Principles "
    "of Philosophy."
)

ORACLE_SYSTEM = (
    "You are a philosophical knowledge oracle. A Descartes specialist "
    "is asking for information outside its training domain. Provide "
    "accurate, detailed philosophical knowledge with specific sources "
    "and positions. The specialist will integrate your knowledge with "
    "its own formal analysis."
)

INTEGRATION_TEMPLATE = (
    "You previously analyzed a question but needed additional "
    "philosophical knowledge. Integrate the oracle's response with "
    "your own Cartesian expertise. Preserve your formal analysis "
    "and strengthen it with the new information.\n\n"
    "ORIGINAL QUESTION:\n{query}\n\n"
    "YOUR INITIAL ANALYSIS:\n{initial}\n\n"
    "ADDITIONAL KNOWLEDGE:\n{oracle}\n\n"
    "Produce your final integrated answer."
)


# ============================================================
# ENGINE RESULT
# ============================================================

@dataclass
class EngineResult:
    """Complete result from one query."""
    query: str
    final_response: str
    confidence: float
    routing: str                # SELF, ORACLE, HYBRID
    error_type: str             # NONE, FACTUAL_GAP, etc.
    oracle_used: bool
    initial_response: str = ""
    oracle_query: Optional[str] = None
    oracle_response: Optional[str] = None


# ============================================================
# DESCARTES ENGINE (Pure Ollama)
# ============================================================

class DescartesEngine:
    """Production engine: Ollama local + cloud, meta-learner routing."""

    def __init__(self,
                 local_model: str = "descartes:8b",
                 oracle_model: str = "deepseek-v3.1:671-cloud",
                 meta_path: Optional[str] = None,
                 oracle_escalation: Optional[str] = "kimi-k2.5:cloud"):

        self.local_model = local_model
        self.oracle_model = oracle_model
        self.oracle_escalation = oracle_escalation

        # Signal extractor (text-only for pure Ollama)
        self.extractor = LiteSignalExtractor()

        # Meta-learner
        self.meta = MetaLearnerLite(input_dim=11)
        self.trainer = MetaTrainer(self.meta, lr=1e-4)

        if meta_path and os.path.exists(meta_path):
            self.trainer.load(meta_path)
            print(f"Loaded meta-learner ({self.trainer.update_count} updates)")

        self.meta.eval()

        # Stats
        self.stats = {
            "total": 0, "self": 0, "oracle": 0, "hybrid": 0,
            "oracle_calls": 0
        }

        print(f"Engine ready.")
        print(f"  Local:  {local_model}")
        print(f"  Oracle: {oracle_model}")
        print(f"  Escalation: {oracle_escalation}")
        print(f"  Meta-learner: {'warm' if meta_path else 'cold'} start")

    def _chat_local(self, messages: list) -> str:
        """Query the local Descartes model via Ollama."""
        ol = get_ollama()
        resp = ol.chat(model=self.local_model, messages=messages)
        return resp['message']['content']

    def _chat_oracle(self, messages: list,
                      escalate: bool = False) -> str:
        """Query the cloud oracle via Ollama."""
        ol = get_ollama()
        model = self.oracle_escalation if escalate else self.oracle_model
        resp = ol.chat(model=model, messages=messages)
        self.stats["oracle_calls"] += 1
        return resp['message']['content']

    def _build_oracle_query(self, query: str, initial: str,
                             error_type: str) -> str:
        """Shape the oracle query based on predicted error type."""

        if error_type == "FACTUAL_GAP":
            return (
                f"A Descartes specialist needs factual knowledge:\n\n"
                f"Question: {query}\n\n"
                f"Their partial analysis:\n{initial[:500]}\n\n"
                f"What factual information are they missing?"
            )
        elif error_type == "SCOPE_EXCEEDED":
            return (
                f"This question extends beyond Cartesian philosophy. "
                f"Provide the relevant broader context:\n\n{query}"
            )
        elif error_type == "REASONING_ERROR":
            return (
                f"Check this analysis for reasoning errors:\n\n"
                f"Question: {query}\n"
                f"Analysis: {initial[:500]}\n\n"
                f"Identify any errors and provide corrections."
            )
        elif error_type == "FORMALIZATION_ERROR":
            return (
                f"A specialist attempted to formalize this argument "
                f"but may have errors:\n\n"
                f"Question: {query}\n"
                f"Attempt: {initial[:500]}\n\n"
                f"Please verify the formal structure."
            )
        else:
            return query

    def run(self, query: str) -> EngineResult:
        """Full cascade pipeline."""

        self.stats["total"] += 1

        # ── Step 1: Local specialist generates ──
        initial = self._chat_local([
            {"role": "system", "content": DESCARTES_SYSTEM},
            {"role": "user", "content": query}
        ])

        # ── Step 2: Extract signals + meta-learner routes ──
        signals = self.extractor.extract(initial)
        signal_tensor = signals.to_tensor()

        with torch.no_grad():
            meta_out = self.meta(signal_tensor)

        confidence = meta_out["confidence"].item()
        routing = meta_out["routing_decision"]
        error_type = meta_out["error_type"]
        features = meta_out["features"]

        result = EngineResult(
            query=query,
            final_response=initial,
            confidence=confidence,
            routing=routing,
            error_type=error_type,
            oracle_used=False,
            initial_response=initial,
        )

        # ── Step 3: Route ──
        if routing == "SELF":
            self.stats["self"] += 1
            return result

        # ── Step 4: Oracle consultation ──
        oracle_query = self._build_oracle_query(
            query, initial, error_type)

        # Escalate to stronger model for SCOPE_EXCEEDED
        escalate = (error_type == "SCOPE_EXCEEDED"
                    and self.oracle_escalation is not None)

        oracle_response = self._chat_oracle([
            {"role": "system", "content": ORACLE_SYSTEM},
            {"role": "user", "content": oracle_query}
        ], escalate=escalate)

        result.oracle_used = True
        result.oracle_query = oracle_query
        result.oracle_response = oracle_response

        if routing == "ORACLE":
            self.stats["oracle"] += 1
        else:
            self.stats["hybrid"] += 1

        # ── Step 5: Integration pass ──
        integrated = self._chat_local([
            {"role": "system", "content": DESCARTES_SYSTEM},
            {"role": "user", "content": INTEGRATION_TEMPLATE.format(
                query=query,
                initial=initial,
                oracle=oracle_response
            )}
        ])

        result.final_response = integrated

        # ── Step 6: Feedback to meta-learner ──
        agreement = self._compute_agreement(initial, oracle_response)

        outcome = {
            "oracle_agreed": agreement > 0.6,
            "correction_magnitude": 1.0 - agreement,
            "z3_verified": None,
            "user_accepted": None,
        }

        self.trainer.record_and_maybe_train(
            features.detach(), confidence, routing, outcome)

        return result

    def record_user_feedback(self, accepted: bool):
        """Call when user gives explicit feedback (thumbs up/down)."""
        if self.trainer.buffer.buffer:
            last = self.trainer.buffer.buffer[-1]
            if accepted:
                last["true_confidence"] = min(
                    last["true_confidence"] + 0.1, 1.0)
            else:
                last["true_confidence"] = max(
                    last["true_confidence"] - 0.2, 0.0)

    def record_z3_result(self, verified: bool):
        """Call when Z3 verifies a formal claim from the response."""
        if self.trainer.buffer.buffer:
            last = self.trainer.buffer.buffer[-1]
            if verified:
                last["true_confidence"] = min(
                    last["true_confidence"] + 0.15, 1.0)
                last["true_error"] = 0  # NONE
            else:
                last["true_confidence"] = max(
                    last["true_confidence"] - 0.3, 0.0)
                last["true_error"] = 3  # FORMALIZATION_ERROR

    def _compute_agreement(self, text_a: str, text_b: str) -> float:
        """Quick semantic agreement: Jaccard on content words."""
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                "at", "to", "for", "of", "and", "that", "this", "it", "with"}
        wa = set(text_a.lower().split()) - stop
        wb = set(text_b.lower().split()) - stop
        if not wa or not wb:
            return 0.5
        return len(wa & wb) / len(wa | wb)

    def save(self, path: str):
        """Persist meta-learner state."""
        self.trainer.save(path)
        print(f"Saved (updates={self.trainer.update_count}, "
              f"buffer={len(self.trainer.buffer.buffer)})")

    def get_stats(self) -> Dict:
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "self_rate": f"{self.stats['self']/total:.1%}",
            "oracle_rate": f"{self.stats['oracle']/total:.1%}",
            "hybrid_rate": f"{self.stats['hybrid']/total:.1%}",
            "meta_learner": self.trainer.get_stats(),
        }


# ============================================================
# INTERACTIVE REPL
# ============================================================

def main():
    meta_path = None
    project_root = Path(__file__).resolve().parent.parent

    # Check standard locations for meta-learner
    for candidate in [
        project_root / "models" / "meta_learner_bootstrap.pt",
        project_root / "models" / "meta_learner_latest.pt",
        project_root / "models" / "meta_learner_bootstrapped.pt",
    ]:
        if candidate.exists():
            meta_path = str(candidate)
            break

    engine = DescartesEngine(
        local_model="descartes:8b",
        oracle_model="deepseek-v3.1:671-cloud",
        meta_path=meta_path,
        oracle_escalation="kimi-k2.5:cloud",
    )

    print("\n" + "=" * 60)
    print("DESCARTES PHILOSOPHER ENGINE")
    print("=" * 60)
    print("Commands:")
    print("  quit       — exit and save meta-learner")
    print("  stats      — show routing and meta-learner stats")
    print("  good/bad   — give feedback on last response")
    print("  z3:pass    — report Z3 verification passed")
    print("  z3:fail    — report Z3 verification failed")
    print("=" * 60)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        elif query == "quit":
            break
        elif query == "stats":
            print(json.dumps(engine.get_stats(), indent=2))
            continue
        elif query == "good":
            engine.record_user_feedback(True)
            print("  Recorded: positive feedback")
            continue
        elif query == "bad":
            engine.record_user_feedback(False)
            print("  Recorded: negative feedback")
            continue
        elif query == "z3:pass":
            engine.record_z3_result(True)
            print("  Recorded: Z3 verification passed")
            continue
        elif query == "z3:fail":
            engine.record_z3_result(False)
            print("  Recorded: Z3 verification failed")
            continue

        result = engine.run(query)

        print(f"\n[{result.routing}] "
              f"[conf={result.confidence:.2f}] "
              f"[error={result.error_type}] "
              f"[oracle={'yes' if result.oracle_used else 'no'}]")
        print(f"\n{result.final_response}")

    save_path = str(project_root / "models" / "meta_learner_latest.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    engine.save(save_path)
    print(f"\nFinal stats: {json.dumps(engine.get_stats(), indent=2)}")


if __name__ == "__main__":
    main()
