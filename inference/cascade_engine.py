"""
Phase 11 (CASCADE): Complete cascade inference engine V2 with
meta-learner feedback loop.

This is the production inference system. It:
1. Receives a philosophical query
2. Runs it through the small (trained) model
3. Extracts internal signals via SignalExtractor
4. Meta-learner predicts confidence, routing, and error type
5. Routes to oracle if needed (with error-type-aware query construction)
6. Runs integration pass through small model
7. Records feedback for online meta-learner training
8. Returns final response with full metadata

The meta-learner replaces text-based [CONFIDENCE] parsing with
learned confidence from model internals — more reliable and
calibrated through the feedback-forward loop.

Usage:
    python inference/cascade_engine.py [model_path] [--provider deepseek]
"""

import re
import torch
import json
import os
import sys
from typing import Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add inference dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta_learner import MetaLearner, MetaLearnerTrainer
from signal_extractor import SignalExtractor
from oracle import OracleClient, OracleConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CascadeResult:
    """Complete result from the cascade engine."""
    query: str
    final_response: str
    confidence: float
    routing_decision: str        # "SELF", "ORACLE", "HYBRID"
    error_type: str              # "NONE", "FACTUAL_GAP", etc.
    oracle_used: bool
    oracle_query: Optional[str] = None
    oracle_response: Optional[str] = None
    small_model_initial: str = ""
    iterations: int = 1
    oracle_cost: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "final_response": self.final_response,
            "confidence": round(self.confidence, 3),
            "routing_decision": self.routing_decision,
            "error_type": self.error_type,
            "oracle_used": self.oracle_used,
            "oracle_query": self.oracle_query,
            "iterations": self.iterations,
            "oracle_cost": round(self.oracle_cost, 6),
        }


class DescartesEngine:
    """Production Descartes Philosopher Engine with meta-learner routing."""

    def __init__(self,
                 model_path: str,
                 meta_learner_path: Optional[str] = None,
                 oracle_config: OracleConfig = None,
                 device: str = "auto"):

        # Load small model
        print(f"Loading Descartes model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
        self.model.eval()

        # Signal extractor (hooks into model)
        self.extractor = SignalExtractor(self.model, self.tokenizer)

        # Meta-learner
        self.meta = MetaLearner(
            hidden_dim=self.model.config.hidden_size,
            feature_dim=256
        )
        if meta_learner_path and os.path.exists(meta_learner_path):
            checkpoint = torch.load(meta_learner_path, weights_only=False)
            self.meta.load_state_dict(checkpoint["model_state"])
            print(f"Loaded meta-learner "
                  f"({checkpoint.get('update_count', '?')} updates)")
        self.meta.eval()

        # Online trainer (feedback loop)
        self.trainer = MetaLearnerTrainer(self.meta)

        # Oracle
        self.oracle = OracleClient(oracle_config or OracleConfig())

        # System prompt
        self.system_prompt = (
            "You are a philosophical reasoning assistant specializing "
            "in Cartesian philosophy, early modern rationalism, and "
            "the mind-body problem. You analyze arguments with formal "
            "rigor using ASPIC+ argumentation schemes and Z3 "
            "verification. You have deep expertise in Descartes' "
            "Meditations, the Objections and Replies, the "
            "Correspondence with Elisabeth, and the Principles of "
            "Philosophy.\n\n"
            "Express confidence as [CONFIDENCE: 0.X] at the end of "
            "each response. When requesting oracle help, output "
            "[ORACLE_REQUEST: <query>]."
        )

        # Stats
        self.total_queries = 0
        self.self_handled = 0
        self.oracle_handled = 0
        self.hybrid_handled = 0

        print("Engine ready (v2 with meta-learner).")

    def generate_with_signals(self, prompt: str,
                               max_new_tokens: int = 2048,
                               temperature: float = 0.3):
        """Generate response and extract internal signals."""

        self.extractor.clear()

        full_prompt = (
            f"<|system|>\n{self.system_prompt}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )

        inputs = self.tokenizer(
            full_prompt, return_tensors="pt",
            truncation=True, max_length=8192
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Extract signals from generation
        signals = self.extractor.extract_signals(
            inputs["input_ids"], outputs[0:1], response_text)

        return response_text, signals

    def _parse_text_signals(self, response: str) -> Dict:
        """Also parse text-based signals as fallback/complement."""

        conf_match = re.search(
            r'\[CONFIDENCE:\s*([\d.]+)\]', response)
        text_confidence = float(conf_match.group(1)) if conf_match else None

        oracle_match = re.search(
            r'\[ORACLE_REQUEST:\s*(.+?)\]', response, re.DOTALL)
        oracle_query = (oracle_match.group(1).strip()
                        if oracle_match else None)

        # Clean response (remove tags)
        clean = re.sub(r'\[CONFIDENCE:.*?\]', '', response)
        clean = re.sub(r'\[ORACLE_REQUEST:.*?\]', '', clean,
                       flags=re.DOTALL)
        clean = clean.strip()

        return {
            "text_confidence": text_confidence,
            "oracle_query": oracle_query,
            "clean_response": clean,
        }

    def _construct_oracle_query(self, original_query: str,
                                 small_response: str,
                                 error_type: str) -> str:
        """Construct oracle query based on predicted error type.

        The meta-learner tells us WHAT KIND of knowledge gap
        exists, so we can ask the oracle the right question.
        """

        if error_type == "FACTUAL_GAP":
            return (
                f"A Descartes specialist needs factual philosophical "
                f"knowledge to answer this question:\n\n"
                f"{original_query}\n\n"
                f"They've provided this partial analysis:\n"
                f"{small_response[:500]}\n\n"
                f"What factual information are they missing?"
            )

        elif error_type == "SCOPE_EXCEEDED":
            return (
                f"This question goes beyond Cartesian philosophy "
                f"into broader territory. Please provide the "
                f"relevant context:\n\n{original_query}"
            )

        elif error_type == "REASONING_ERROR":
            return (
                f"A specialist provided this analysis but may "
                f"have reasoning errors:\n\n"
                f"Question: {original_query}\n"
                f"Analysis: {small_response[:500]}\n\n"
                f"Please check the reasoning and provide corrections."
            )

        elif error_type == "FORMALIZATION_ERROR":
            return (
                f"A specialist attempted to formalize this argument "
                f"but may have formalization issues:\n\n"
                f"Question: {original_query}\n"
                f"Attempt: {small_response[:500]}\n\n"
                f"Please verify the formal structure."
            )

        else:
            return original_query

    def run(self, query: str) -> CascadeResult:
        """Execute the full cascade pipeline with meta-learner routing."""

        self.total_queries += 1

        # Step 1: Generate from small model + extract signals
        response, signals = self.generate_with_signals(query)
        text_parsed = self._parse_text_signals(response)

        # Step 2: Meta-learner decides confidence and routing
        with torch.no_grad():
            meta_output = self.meta(signals)

        meta_confidence = meta_output["confidence"].item()
        meta_routing = meta_output["routing_decision"]
        meta_error = meta_output["error_type"]
        cached_features = meta_output["features"]

        # If text-based oracle request exists, respect it
        # (the model was SFT-trained to emit these)
        if text_parsed["oracle_query"] and meta_routing == "SELF":
            meta_routing = "HYBRID"

        # Build result
        result = CascadeResult(
            query=query,
            final_response=text_parsed["clean_response"],
            confidence=meta_confidence,
            routing_decision=meta_routing,
            error_type=meta_error,
            oracle_used=False,
            small_model_initial=text_parsed["clean_response"],
        )

        # Step 3: Route based on meta-learner decision
        if meta_routing == "SELF":
            self.self_handled += 1
            return result

        # Step 4: Oracle consultation
        result.oracle_used = True

        # Use text-based oracle query if available, else construct from error
        oracle_query = (text_parsed["oracle_query"] or
                        self._construct_oracle_query(
                            query, text_parsed["clean_response"],
                            meta_error))
        result.oracle_query = oracle_query

        oracle_response = self.oracle.query(
            oracle_query,
            context=text_parsed["clean_response"],
            error_type=meta_error)
        result.oracle_response = oracle_response
        result.oracle_cost = self.oracle.total_cost

        if meta_routing == "ORACLE":
            self.oracle_handled += 1
        else:
            self.hybrid_handled += 1

        # Step 5: Integration pass through small model
        if meta_routing == "HYBRID":
            integration_prompt = (
                f"Integrate your formal analysis with additional "
                f"philosophical knowledge.\n\n"
                f"QUESTION: {query}\n\n"
                f"YOUR FORMAL ANALYSIS:\n"
                f"{text_parsed['clean_response']}\n\n"
                f"PHILOSOPHICAL CONTEXT:\n{oracle_response}\n\n"
                f"Produce a complete answer combining both."
            )
        else:  # ORACLE
            integration_prompt = (
                f"You previously answered a question but need to "
                f"integrate additional knowledge.\n\n"
                f"QUESTION: {query}\n\n"
                f"YOUR ANSWER:\n{text_parsed['clean_response']}\n\n"
                f"ADDITIONAL KNOWLEDGE:\n{oracle_response}\n\n"
                f"Produce your final integrated answer."
            )

        integrated, _ = self.generate_with_signals(integration_prompt)
        integrated_parsed = self._parse_text_signals(integrated)

        result.final_response = integrated_parsed["clean_response"]
        result.iterations = 2

        # Update confidence from integration pass
        if integrated_parsed["text_confidence"] is not None:
            result.confidence = integrated_parsed["text_confidence"]

        # Step 6: Record feedback for meta-learner training
        self._record_feedback(
            cached_features, meta_confidence, meta_routing,
            text_parsed["clean_response"], oracle_response
        )

        return result

    def _record_feedback(self, features, predicted_conf,
                          predicted_routing, small_response,
                          oracle_response):
        """Record outcome for meta-learner training.

        This is the FEEDBACK part of feedback-forward.
        """

        # Compute agreement between small model and oracle
        agreement = self._compute_agreement(
            small_response, oracle_response)

        # Compute correction magnitude
        correction_mag = 1.0 - agreement

        outcome = {
            "oracle_agreed": agreement > 0.7,
            "correction_magnitude": correction_mag,
            "z3_verified": None,  # Set later if Z3 was used
            "user_accepted": None,  # Set later from user feedback
        }

        self.trainer.record_outcome(
            features, predicted_conf,
            predicted_routing, outcome
        )

    def _compute_agreement(self, text_a: str, text_b: str) -> float:
        """Quick semantic agreement score between two responses.

        Simple version: Jaccard similarity on content words.
        Production version: use a sentence embedding model.
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "and",
            "that", "this", "it", "be", "as", "with", "by"
        }
        words_a = set(text_a.lower().split()) - stop_words
        words_b = set(text_b.lower().split()) - stop_words

        if not words_a or not words_b:
            return 0.5

        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def record_user_feedback(self, accepted: bool):
        """Call this when user gives explicit feedback.

        Updates the most recent buffer entry with user signal.
        """
        if self.trainer.buffer.buffer:
            last = self.trainer.buffer.buffer[-1]
            if accepted:
                last["true_confidence"] = min(
                    last["true_confidence"] + 0.1, 1.0)
            else:
                last["true_confidence"] = max(
                    last["true_confidence"] - 0.2, 0.0)

    def save_state(self, path: str):
        """Save meta-learner state for persistence across sessions."""
        self.trainer.save(path)
        print(f"Meta-learner saved ({self.trainer.update_count} updates, "
              f"buffer: {len(self.trainer.buffer.buffer)})")

    def get_stats(self) -> Dict:
        """Return comprehensive engine statistics."""
        return {
            "total_queries": self.total_queries,
            "self_handled": self.self_handled,
            "oracle_handled": self.oracle_handled,
            "hybrid_handled": self.hybrid_handled,
            "self_rate": round(
                self.self_handled / max(self.total_queries, 1), 3),
            "meta_learner_updates": self.trainer.update_count,
            "oracle_stats": self.oracle.get_stats(),
            "avg_meta_loss": (
                round(sum(self.trainer.loss_history) /
                      max(len(self.trainer.loss_history), 1), 4)
            ) if self.trainer.loss_history else None,
        }

    def interactive(self):
        """Interactive REPL for testing the cascade engine."""
        print("\n" + "=" * 60)
        print("DESCARTES PHILOSOPHER ENGINE — Interactive Mode (V2)")
        print("Commands: 'quit', 'stats', 'save', 'yes/no' (feedback)")
        print("=" * 60)

        last_result = None

        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                continue
            elif query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                print(json.dumps(self.get_stats(), indent=2))
                continue
            elif query.lower() == 'save':
                save_path = str(
                    PROJECT_ROOT / "models" / "meta_learner_state.pt")
                self.save_state(save_path)
                continue
            elif query.lower() in ('yes', 'y'):
                if last_result:
                    self.record_user_feedback(True)
                    print("  [Feedback recorded: accepted]")
                continue
            elif query.lower() in ('no', 'n'):
                if last_result:
                    self.record_user_feedback(False)
                    print("  [Feedback recorded: rejected]")
                continue

            result = self.run(query)
            last_result = result

            print(f"\n[Routing: {result.routing_decision}] "
                  f"[Confidence: {result.confidence:.2f}] "
                  f"[Error: {result.error_type}] "
                  f"[Oracle: {'Yes' if result.oracle_used else 'No'}]")
            print(f"\n{result.final_response}")

            if result.oracle_used:
                print(f"\n  [Oracle cost this call: "
                      f"${result.oracle_cost:.4f}]")

        print(f"\nSession stats:\n{json.dumps(self.get_stats(), indent=2)}")


if __name__ == "__main__":
    model_path = (sys.argv[1] if len(sys.argv) > 1
                  else str(PROJECT_ROOT / "models" / "descartes-8b-cascade"))

    meta_path = str(PROJECT_ROOT / "models" / "meta_learner_bootstrapped.pt")
    if not os.path.exists(meta_path):
        meta_path = None

    provider = "deepseek"
    for i, arg in enumerate(sys.argv):
        if arg == "--provider" and i + 1 < len(sys.argv):
            provider = sys.argv[i + 1]

    oracle_config = OracleConfig(provider=provider)

    engine = DescartesEngine(
        model_path=model_path,
        meta_learner_path=meta_path,
        oracle_config=oracle_config)

    engine.interactive()
