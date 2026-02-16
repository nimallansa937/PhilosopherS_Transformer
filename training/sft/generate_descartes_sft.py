"""
Phase 6 (CASCADE): Generate all SFT examples for Descartes cascade model.

Combines:
- Types A-D from LLM council (standard philosophical reasoning)
- Types E-G from self-play (cascade-specific behaviors)

Target: 6,000-10,000 total examples

Usage:
    python training/sft/generate_descartes_sft.py [--api-key KEY]
"""

import json
import os
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training" / "sft"))

from descartes_templates import (
    SYSTEM_PROMPT,
    TYPE_A_DESCARTES, TYPE_B_DESCARTES, TYPE_C_DESCARTES,
    TYPE_D_DESCARTES, TYPE_E_DESCARTES, TYPE_F_DESCARTES,
    TYPE_G_DESCARTES,
)

OUTPUT_DIR = PROJECT_ROOT / "training" / "sft" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LLM CLIENT ABSTRACTION
# ============================================================

class LLMClient:
    """Unified client for Claude / GPT-4 / Gemini / DeepSeek."""

    def __init__(self, provider: str = "deepseek", api_key: str = ""):
        self.provider = provider
        self.api_key = api_key or os.environ.get(
            {
                "deepseek": "DEEPSEEK_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(provider, "API_KEY"), ""
        )
        self._client = None

    def _init_client(self):
        if self._client:
            return
        if self.provider in ("deepseek", "openai"):
            from openai import OpenAI
            base_url = ("https://api.deepseek.com"
                        if self.provider == "deepseek" else None)
            self._client = OpenAI(
                api_key=self.api_key, base_url=base_url)
        elif self.provider == "claude":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """Generate a response from the LLM."""
        self._init_client()

        if self.provider in ("deepseek", "openai"):
            model = ("deepseek-chat" if self.provider == "deepseek"
                     else "gpt-4o")
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content

        elif self.provider == "claude":
            resp = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        return ""


# ============================================================
# EXAMPLE GENERATORS
# ============================================================

def generate_type_a(templates: List[Dict],
                    client: Optional[LLMClient] = None) -> List[Dict]:
    """Generate Type A (Argument Reconstruction) examples."""
    examples = []
    for t in templates:
        user_msg = t["user"]
        if client:
            response = client.generate(SYSTEM_PROMPT, user_msg)
        else:
            # Placeholder: list key elements for manual generation
            elements = "\n".join(f"  - {e}" for e in t.get("key_elements", []))
            response = (
                f"[PLACEHOLDER — LLM council should generate full response]\n"
                f"Key elements to cover:\n{elements}\n\n"
                f"[CONFIDENCE: 0.85]"
            )

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "type": "A",
                "philosopher": "Descartes",
                "z3_validated": False,
                "council_agreement": 0.0,
                "human_reviewed": False,
                "review_status": "pending",
            }
        })
    return examples


def generate_type_b(templates: List[Dict],
                    client: Optional[LLMClient] = None) -> List[Dict]:
    """Generate Type B (Critical Engagement) examples."""
    examples = []
    for t in templates:
        user_msg = t["user"]
        if client:
            response = client.generate(SYSTEM_PROMPT, user_msg)
        else:
            response = (
                f"[PLACEHOLDER — LLM council should generate]\n"
                f"Attack type: {t.get('attack_type', 'N/A')}\n"
                f"Target: {t.get('target', 'N/A')}\n\n"
                f"[CONFIDENCE: 0.80]"
            )
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "type": "B",
                "philosopher": "Descartes",
                "attack_type": t.get("attack_type", ""),
                "z3_validated": False,
                "council_agreement": 0.0,
                "human_reviewed": False,
                "review_status": "pending",
            }
        })
    return examples


def generate_type_c(templates: List[Dict],
                    client: Optional[LLMClient] = None) -> List[Dict]:
    """Generate Type C (Cross-Disciplinary) examples."""
    examples = []
    for t in templates:
        user_msg = t["user"]
        if client:
            response = client.generate(SYSTEM_PROMPT, user_msg)
        else:
            response = (
                f"[PLACEHOLDER — LLM council should generate]\n\n"
                f"[CONFIDENCE: 0.70]"
            )
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "type": "C",
                "philosopher": "Descartes",
                "z3_validated": False,
                "council_agreement": 0.0,
                "human_reviewed": False,
                "review_status": "pending",
            }
        })
    return examples


def generate_type_d(templates: List[Dict],
                    client: Optional[LLMClient] = None) -> List[Dict]:
    """Generate Type D (Comprehension) examples."""
    examples = []
    for t in templates:
        passage = t["passage"]
        source = t["source"]
        questions = "\n".join(t["questions"])
        user_msg = (
            f"Read this passage from Descartes ({source}):\n\n"
            f'"{passage}"\n\n'
            f"{questions}"
        )
        if client:
            response = client.generate(SYSTEM_PROMPT, user_msg)
        else:
            response = (
                f"[PLACEHOLDER — LLM council should generate]\n\n"
                f"[CONFIDENCE: 0.80]"
            )
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "type": "D",
                "philosopher": "Descartes",
                "source": source,
                "z3_validated": False,
                "council_agreement": 0.0,
                "human_reviewed": False,
                "review_status": "pending",
            }
        })
    return examples


def generate_type_e(templates: List[Dict],
                    client: Optional[LLMClient] = None) -> List[Dict]:
    """Generate Type E (Confidence Estimation) examples."""
    examples = []
    for t in templates:
        user_msg = t["user"]
        # Use the provided sketch or generate
        if "response_sketch" in t and t["response_sketch"]:
            response = t["response_sketch"]
        elif client:
            augmented_prompt = (
                f"{user_msg}\n\n"
                f"Important: End your response with [CONFIDENCE: X.X] "
                f"where X.X reflects your genuine confidence (0.0-1.0). "
                f"If you need external knowledge, include "
                f"[ORACLE_REQUEST: <specific query>]."
            )
            response = client.generate(SYSTEM_PROMPT, augmented_prompt)
        else:
            response = (
                f"[PLACEHOLDER — expected confidence: "
                f"{t.get('expected_confidence', 'N/A')}]\n"
                f"Routing: {t.get('routing', 'N/A')}\n"
                f"Reason: {t.get('reason', 'N/A')}\n\n"
                f"[CONFIDENCE: {t.get('expected_confidence', 0.5)}]"
            )

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "type": "E",
                "philosopher": "Descartes",
                "expected_confidence": t.get("expected_confidence"),
                "expected_routing": t.get("routing"),
                "z3_validated": False,
                "human_reviewed": False,
                "review_status": "pending",
            }
        })
    return examples


def generate_type_f(templates: List[Dict],
                    client: Optional[LLMClient] = None) -> List[Dict]:
    """Generate Type F (Routing Decision) examples."""
    examples = []
    for t in templates:
        user_msg = t["user"]
        routing = t["routing"]
        reason = t["reason"]

        if routing == "SELF":
            response = (
                f"I can handle this directly with my Cartesian expertise "
                f"and formal reasoning capabilities.\n\n"
                f"[Reason: {reason}]\n\n"
                f"[CONFIDENCE: 0.90]"
            )
        elif routing == "ORACLE":
            response = (
                f"This question goes beyond my specialized training in "
                f"Cartesian philosophy and formal reasoning. I need "
                f"broader knowledge.\n\n"
                f"[Reason: {reason}]\n\n"
                f"[ORACLE_REQUEST: {user_msg}]\n\n"
                f"[CONFIDENCE: 0.35]"
            )
        else:  # HYBRID
            response = (
                f"I can partially address this with my formal expertise, "
                f"but need oracle consultation for the broader context.\n\n"
                f"[Reason: {reason}]\n\n"
                f"[ORACLE_REQUEST: Provide context for: {user_msg}]\n\n"
                f"[CONFIDENCE: 0.55]"
            )

        if client:
            augmented_prompt = (
                f"{user_msg}\n\n"
                f"Decide whether to answer this yourself (SELF), "
                f"request oracle help (ORACLE), or combine both (HYBRID). "
                f"Explain your routing decision."
            )
            response = client.generate(SYSTEM_PROMPT, augmented_prompt)

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "type": "F",
                "philosopher": "Descartes",
                "expected_routing": routing,
                "z3_validated": False,
                "human_reviewed": False,
                "review_status": "pending",
            }
        })
    return examples


def generate_type_g(templates: List[Dict],
                    client: Optional[LLMClient] = None) -> List[Dict]:
    """Generate Type G (Oracle Integration) examples."""
    examples = []
    for t in templates:
        # Type G has a three-turn structure:
        # 1. Initial answer (with oracle request)
        # 2. Oracle response (simulated)
        # 3. Integrated final answer
        user_msg = (
            "You previously answered a question and requested oracle "
            "consultation. Integrate the oracle's response with your "
            "own expertise.\n\n"
            f"YOUR INITIAL ANSWER:\n{t['original_answer']}\n\n"
            f"ORACLE RESPONSE:\n{t['oracle_response']}\n\n"
            "Produce your final integrated answer. Update your "
            "confidence score."
        )
        response = t.get("integrated", "")
        if not response and client:
            response = client.generate(SYSTEM_PROMPT, user_msg)

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "type": "G",
                "philosopher": "Descartes",
                "z3_validated": False,
                "human_reviewed": False,
                "review_status": "pending",
            }
        })
    return examples


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_all(use_api: bool = False, provider: str = "deepseek"):
    """Generate complete SFT dataset."""

    print("=" * 60)
    print("PHASE 6: SFT Data Generation — Descartes Cascade")
    print("=" * 60)

    client = None
    if use_api:
        client = LLMClient(provider=provider)
        print(f"Using LLM API: {provider}")
    else:
        print("Generating TEMPLATE examples (no API).")
        print("Re-run with --api to use LLM council for full generation.")

    all_examples = []

    # Standard types (A-D)
    print("\n[1/7] Generating Type A (argument reconstruction)...")
    all_examples.extend(generate_type_a(TYPE_A_DESCARTES, client))

    print("[2/7] Generating Type B (critical engagement)...")
    all_examples.extend(generate_type_b(TYPE_B_DESCARTES, client))

    print("[3/7] Generating Type C (cross-disciplinary)...")
    all_examples.extend(generate_type_c(TYPE_C_DESCARTES, client))

    print("[4/7] Generating Type D (comprehension)...")
    all_examples.extend(generate_type_d(TYPE_D_DESCARTES, client))

    # Cascade types (E-G)
    print("[5/7] Generating Type E (confidence estimation)...")
    all_examples.extend(generate_type_e(TYPE_E_DESCARTES, client))

    print("[6/7] Generating Type F (routing decisions)...")
    all_examples.extend(generate_type_f(TYPE_F_DESCARTES, client))

    print("[7/7] Generating Type G (oracle integration)...")
    all_examples.extend(generate_type_g(TYPE_G_DESCARTES, client))

    # ---- Split into standard (A-D) and cascade (E-G) ----
    standard = [e for e in all_examples if e["metadata"]["type"] in "ABCD"]
    cascade = [e for e in all_examples if e["metadata"]["type"] in "EFG"]

    # Save combined
    all_path = OUTPUT_DIR / "descartes_sft_all.jsonl"
    with open(all_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Save split files for two-stage SFT
    std_path = OUTPUT_DIR / "descartes_sft_types_ABCD.jsonl"
    with open(std_path, 'w') as f:
        for ex in standard:
            f.write(json.dumps(ex) + "\n")

    cas_path = OUTPUT_DIR / "descartes_sft_types_EFG.jsonl"
    with open(cas_path, 'w') as f:
        for ex in cascade:
            f.write(json.dumps(ex) + "\n")

    # Stats
    type_counts = Counter(e["metadata"]["type"] for e in all_examples)

    print(f"\n{'=' * 60}")
    print(f"SFT DATA GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total examples: {len(all_examples)}")
    for t, c in sorted(type_counts.items()):
        print(f"  Type {t}: {c}")
    print(f"\nFiles:")
    print(f"  All:      {all_path}")
    print(f"  Standard: {std_path} ({len(standard)} examples)")
    print(f"  Cascade:  {cas_path} ({len(cascade)} examples)")

    if not use_api:
        print(f"\nNOTE: These are TEMPLATE examples. Run with --api flag "
              f"and API keys to generate full LLM council responses.")
        print(f"Then human-review with: python training/sft/review_interface.py")


if __name__ == "__main__":
    use_api = "--api" in sys.argv
    provider = "deepseek"
    for i, arg in enumerate(sys.argv):
        if arg == "--provider" and i + 1 < len(sys.argv):
            provider = sys.argv[i + 1]
    generate_all(use_api=use_api, provider=provider)
