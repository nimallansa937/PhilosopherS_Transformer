"""
Layer 4: Generate-Verify-Regenerate (GVR) Loop.

The core reasoning cycle that connects the LLM to formal verification.
In V3, the GVR loop operates per-claim within the cascade engine
rather than on full responses.

Original GVR (v1):
  Query → LLM generates full response → Z3 checks everything →
  if fail, re-generate everything → repeat up to N times

Modified GVR (v3):
  Query → LLM generates full response → CLAIM EXTRACTOR splits →
  each claim routed to verifier → failed claims get SELF-REPAIR →
  still-failed get ORACLE → INTEGRATION PASS

This module provides the standalone GVR loop for use outside
the cascade (e.g., during SFT data generation, evaluation).

Reference: PHILOSOPHER_ENGINE_ARCHITECTURE.md, Layer 4
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
import time


@dataclass
class GVRResult:
    """Result of a Generate-Verify-Regenerate cycle."""
    final_text: str
    attempts: int
    verified: bool
    verification_details: Dict = field(default_factory=dict)
    total_time_ms: float = 0.0
    history: List[Dict] = field(default_factory=list)


class GVRLoop:
    """Generate-Verify-Regenerate loop for philosophical reasoning.

    Takes a generation function and a verification function,
    and iterates until verification passes or max attempts reached.

    In V3, this is embedded in engine_v3.py at the claim level.
    This standalone version is used during:
    - SFT data generation (generate formalization pairs)
    - Evaluation (test model's formalization ability)
    - Interactive debugging
    """

    def __init__(self,
                 generate_fn: Callable[[str, Optional[str]], str],
                 verify_fn: Callable[[str], Dict],
                 max_attempts: int = 3):
        """
        Args:
            generate_fn: Function(query, feedback) -> generated_text
            verify_fn: Function(text) -> {"verified": bool, "details": dict}
            max_attempts: Maximum GVR iterations
        """
        self.generate_fn = generate_fn
        self.verify_fn = verify_fn
        self.max_attempts = max_attempts

    def run(self, query: str) -> GVRResult:
        """Execute the GVR loop.

        1. Generate initial response
        2. Verify against Z3/corpus/etc.
        3. If failed, regenerate with error feedback
        4. Repeat until verified or max attempts
        """
        start = time.monotonic()
        history = []
        feedback = None

        for attempt in range(1, self.max_attempts + 1):
            # Generate
            text = self.generate_fn(query, feedback)

            # Verify
            v_result = self.verify_fn(text)
            verified = v_result.get("verified", False)

            history.append({
                "attempt": attempt,
                "text_preview": text[:200],
                "verified": verified,
                "details": v_result.get("details", {}),
            })

            if verified:
                elapsed = (time.monotonic() - start) * 1000
                return GVRResult(
                    final_text=text,
                    attempts=attempt,
                    verified=True,
                    verification_details=v_result,
                    total_time_ms=elapsed,
                    history=history,
                )

            # Build feedback for next attempt
            feedback = self._build_feedback(v_result)

        # Max attempts reached without verification
        elapsed = (time.monotonic() - start) * 1000
        return GVRResult(
            final_text=text,
            attempts=self.max_attempts,
            verified=False,
            verification_details=v_result,
            total_time_ms=elapsed,
            history=history,
        )

    def _build_feedback(self, v_result: Dict) -> str:
        """Build error feedback string for next generation attempt."""
        details = v_result.get("details", {})
        parts = ["Your previous response had verification issues:"]

        if "errors" in details:
            for err in details["errors"]:
                parts.append(f"- {err}")

        if "unsat_core" in details:
            parts.append(
                f"Contradictory assertions: {details['unsat_core']}")

        if "suggestion" in details:
            parts.append(f"Suggestion: {details['suggestion']}")

        parts.append("Please revise your response to address these issues.")
        return "\n".join(parts)


class ClaimLevelGVR:
    """Claim-level GVR used in the V3 cascade engine.

    Instead of regenerating the entire response, this operates
    on individual claims — keeping verified parts and fixing
    only the broken parts.
    """

    def __init__(self,
                 formalize_fn: Callable[[str], str],
                 verify_fn: Callable[[str], Dict],
                 repair_fn: Callable[[str, Dict], str],
                 max_attempts: int = 2):
        self.formalize_fn = formalize_fn
        self.verify_fn = verify_fn
        self.repair_fn = repair_fn
        self.max_attempts = max_attempts

    def verify_claim(self, claim_text: str) -> Dict:
        """Run claim-level GVR: formalize → verify → repair if needed.

        Returns:
            {"verified": bool, "method": str, "artifact": str, ...}
        """
        # Step 1: Formalize claim to Z3
        z3_code = self.formalize_fn(claim_text)
        if not z3_code:
            return {"verified": False, "method": "formalization_failed"}

        # Step 2: Verify
        result = self.verify_fn(z3_code)

        if result.get("verified"):
            return {**result, "method": "z3_direct"}

        # Step 3: Repair loop
        for attempt in range(self.max_attempts):
            repaired = self.repair_fn(claim_text, result)
            if repaired:
                result = self.verify_fn(repaired)
                if result.get("verified"):
                    return {**result, "method": f"repair_attempt_{attempt+1}"}

        return {"verified": False, "method": "repair_exhausted",
                "details": result}
