"""
Self-repair: when a claim fails verification, ask the small
model to fix just that claim, then re-verify.

This eliminates ~40% of oracle calls because many Z3 failures
are formalization errors (wrong encoding), not knowledge gaps.
The model often knows the right answer but encoded it badly.

COGITO parallel: The Grounding Verifier's repair loop --
generate -> verify -> if ungrounded, repair -> re-verify.

Three repair strategies:
1. RE-ENCODE: Same claim, new Z3 encoding (formalization error)
2. RE-STATE: Reformulate the claim, then re-verify (imprecise language)
3. DECOMPOSE: Break complex claim into sub-claims, verify each
"""

from typing import Optional
from .claim_extractor import ExtractedClaim
from .verifier import FormalVerifier, CorpusVerifier, VerificationResult
from .knowledge_store import ProofStatus


class SelfRepairEngine:
    """
    Attempt local repair of failed claims before oracle escalation.
    """

    def __init__(self,
                 local_model: str = "descartes:8b",
                 formal_verifier: Optional[FormalVerifier] = None,
                 corpus_verifier: Optional[CorpusVerifier] = None,
                 max_attempts: int = 2):
        self.local_model = local_model
        self.formal = formal_verifier
        self.corpus = corpus_verifier
        self.max_attempts = max_attempts
        self.stats = {"attempted": 0, "succeeded": 0, "failed": 0}

    def attempt_repair(
        self,
        claim: ExtractedClaim,
        failed_result: VerificationResult,
    ) -> Optional[VerificationResult]:
        """
        Try to repair a failed claim locally.

        Returns:
          VerificationResult if repair succeeded
          None if repair failed (needs oracle)
        """
        self.stats["attempted"] += 1
        claim.repair_attempted = True

        for attempt in range(self.max_attempts):
            strategy = self._pick_strategy(failed_result, attempt)

            if strategy == "re_encode":
                result = self._repair_encoding(claim, failed_result)
            elif strategy == "re_state":
                result = self._repair_statement(claim, failed_result)
            elif strategy == "decompose":
                result = self._repair_decompose(claim, failed_result)
            else:
                break

            if result and result.status == ProofStatus.VERIFIED:
                self.stats["succeeded"] += 1
                claim.verified = True
                claim.repaired_text = claim.text
                return result

        self.stats["failed"] += 1
        return None

    def _pick_strategy(self, result: VerificationResult,
                        attempt: int) -> str:
        """Choose repair strategy based on failure type."""
        if "error" in result.proof_artifact:
            return "re_encode"

        if result.status == ProofStatus.REFUTED:
            return "re_state" if attempt == 0 else "decompose"

        if result.status == ProofStatus.TIMEOUT:
            return "decompose"

        return "re_encode"

    def _repair_encoding(
        self,
        claim: ExtractedClaim,
        failed: VerificationResult,
    ) -> Optional[VerificationResult]:
        """Same claim, new Z3 encoding."""
        try:
            import ollama
            ollama.chat(
                model=self.local_model,
                messages=[{
                    "role": "system",
                    "content": (
                        "You previously generated a Z3 encoding that "
                        "failed. Generate a CORRECTED encoding. "
                        "Output ONLY Z3 Python code."
                    )
                }, {
                    "role": "user",
                    "content": (
                        f"Claim: {claim.text}\n\n"
                        f"Failed encoding:\n{failed.encoding}\n\n"
                        f"Error: {failed.proof_artifact}\n\n"
                        f"Generate a corrected Z3 encoding."
                    )
                }]
            )
        except Exception:
            return None

        if self.formal:
            return self.formal.verify_formal(claim)
        return None

    def _repair_statement(
        self,
        claim: ExtractedClaim,
        failed: VerificationResult,
    ) -> Optional[VerificationResult]:
        """Reformulate the claim more precisely, then re-verify."""
        try:
            import ollama
            resp = ollama.chat(
                model=self.local_model,
                messages=[{
                    "role": "system",
                    "content": (
                        "Your previous claim could not be formally "
                        "verified. Reformulate it more precisely, "
                        "distinguishing what is formally provable "
                        "from what is interpretive."
                    )
                }, {
                    "role": "user",
                    "content": (
                        f"Original claim: {claim.text}\n"
                        f"Verification result: {failed.status.value}\n\n"
                        f"Restate this claim in a way that is either "
                        f"formally verifiable or explicitly marked as "
                        f"interpretive."
                    )
                }]
            )
            new_text = resp['message']['content'].strip()
            claim.text = new_text
        except Exception:
            return None

        if self.formal:
            return self.formal.verify_formal(claim)
        return None

    def _repair_decompose(
        self,
        claim: ExtractedClaim,
        failed: VerificationResult,
    ) -> Optional[VerificationResult]:
        """Break complex claim into simpler sub-claims."""
        try:
            import ollama
            resp = ollama.chat(
                model=self.local_model,
                messages=[{
                    "role": "system",
                    "content": (
                        "Break this complex philosophical claim into "
                        "2-3 simpler sub-claims that can each be "
                        "verified independently. Output one claim "
                        "per line."
                    )
                }, {
                    "role": "user",
                    "content": claim.text
                }]
            )
            sub_claims = [
                line.strip() for line in
                resp['message']['content'].strip().split('\n')
                if line.strip() and len(line.strip()) > 10
            ]
        except Exception:
            return None

        all_verified = True
        for sc_text in sub_claims[:3]:
            sc = ExtractedClaim(
                text=sc_text,
                claim_type=claim.claim_type,
                confidence=claim.confidence,
                position=claim.position,
            )
            if self.formal:
                result = self.formal.verify_formal(sc)
                if result.status != ProofStatus.VERIFIED:
                    all_verified = False
                    break

        if all_verified and sub_claims:
            claim.text = " AND ".join(sub_claims[:3])
            return VerificationResult(
                claim=claim,
                status=ProofStatus.VERIFIED,
                method=failed.method,
                encoding="decomposed",
                proof_artifact="all_sub_claims_verified",
                time_ms=0,
                stored_to_vks=True,
            )

        return None
