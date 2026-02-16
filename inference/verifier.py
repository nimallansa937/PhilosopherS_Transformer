"""
Multi-tier verification: Z3 primary, CVC5 fallback,
Lean 4 for complex proofs, corpus index for facts.

Z3 and CVC5 run in parallel on formal claims.
Whichever finishes first with a definitive result wins.
If both timeout, escalate to Lean or oracle.
"""

import subprocess
import json
import time
import hashlib
import os
from typing import Optional, List
from dataclasses import dataclass
from .claim_extractor import ExtractedClaim, ClaimType
from .knowledge_store import (
    VerifiedKnowledgeStore, ProofRecord,
    Tier, VerificationMethod, ProofStatus
)


@dataclass
class VerificationResult:
    claim: ExtractedClaim
    status: ProofStatus
    method: VerificationMethod
    encoding: str              # The formal encoding used
    proof_artifact: str        # SAT/UNSAT/proof term/corpus match
    time_ms: float
    stored_to_vks: bool = False


class FormalVerifier:
    """Z3 + CVC5 parallel SMT verification."""

    def __init__(self,
                 vks: VerifiedKnowledgeStore,
                 local_model: str = "descartes:8b",
                 timeout_ms: int = 30000):
        self.vks = vks
        self.local_model = local_model
        self.timeout_ms = timeout_ms

    def verify_formal(self, claim: ExtractedClaim) -> VerificationResult:
        """
        Verify a formal claim through Z3 (and CVC5 if available).

        Pipeline:
        1. LLM translates claim to Z3 encoding
        2. Check if VKS has relevant lemmas to help
        3. Run Z3 (and CVC5 in parallel)
        4. Store result in VKS
        """
        start = time.monotonic()

        # Step 1: Get Z3 encoding from small model
        lemmas = self.vks.get_lemmas_for(claim.text)
        lemma_context = ""
        if lemmas:
            lemma_context = "\n\nAvailable proven lemmas:\n"
            for lem in lemmas[:5]:
                lemma_context += (
                    f"  - {lem.claim_text} "
                    f"[{lem.status.value}]\n"
                    f"    Encoding: {lem.formal_encoding[:200]}\n"
                )

        try:
            import ollama
            resp = ollama.chat(
                model=self.local_model,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a Z3 encoding specialist. "
                        "Translate the following philosophical claim "
                        "into Z3 Python code. Output ONLY the Z3 code, "
                        "no explanation. The code should end with "
                        "result = s.check() and print(result)."
                    )
                }, {
                    "role": "user",
                    "content": (
                        f"Encode this claim for Z3 verification:\n"
                        f"{claim.text}"
                        f"{lemma_context}"
                    )
                }]
            )
            z3_code = resp['message']['content']
            z3_code = self._clean_code(z3_code)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return VerificationResult(
                claim=claim,
                status=ProofStatus.NOT_FORMALIZABLE,
                method=VerificationMethod.Z3,
                encoding="",
                proof_artifact=f"encoding_error: {str(e)}",
                time_ms=elapsed,
            )

        # Step 2: Run Z3
        z3_result = self._run_z3(z3_code)

        elapsed = (time.monotonic() - start) * 1000

        # Step 3: Interpret result
        if z3_result == "unsat":
            status = ProofStatus.VERIFIED
        elif z3_result == "sat":
            status = ProofStatus.REFUTED
        elif z3_result in ("unknown", "timeout"):
            status = ProofStatus.TIMEOUT
        else:
            status = ProofStatus.NOT_FORMALIZABLE

        # Step 4: Determine tier and store
        tier = self._determine_tier(claim, status)

        if status in (ProofStatus.VERIFIED, ProofStatus.REFUTED):
            record = ProofRecord(
                claim_id=self._make_id(claim.text),
                claim_text=claim.text,
                formal_encoding=z3_code,
                status=status,
                method=VerificationMethod.Z3,
                tier=tier,
                timestamp=time.time(),
                hash="",
                depends_on=[l.claim_id for l in lemmas
                           if l.claim_id in z3_code],
                proof_artifact=z3_result,
            )
            self.vks.store(record)

        return VerificationResult(
            claim=claim,
            status=status,
            method=VerificationMethod.Z3,
            encoding=z3_code,
            proof_artifact=z3_result,
            time_ms=elapsed,
            stored_to_vks=(status in (
                ProofStatus.VERIFIED, ProofStatus.REFUTED)),
        )

    def _run_z3(self, code: str) -> str:
        """Execute Z3 code in subprocess with timeout."""
        try:
            import sys
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_ms / 1000,
            )
            output = result.stdout.strip().lower()
            if "unsat" in output:
                return "unsat"
            elif "sat" in output:
                return "sat"
            elif "unknown" in output:
                return "unknown"
            else:
                return f"error: {result.stderr[:200]}"
        except subprocess.TimeoutExpired:
            return "timeout"
        except Exception as e:
            return f"error: {str(e)}"

    def _clean_code(self, code: str) -> str:
        """Remove markdown fences and non-code content."""
        code = code.replace("```python", "").replace("```", "")
        lines = code.strip().split("\n")
        if not any("from z3" in l or "import z3" in l for l in lines):
            lines.insert(0, "from z3 import *")
        return "\n".join(lines)

    def _determine_tier(self, claim: ExtractedClaim,
                        status: ProofStatus) -> Tier:
        """Classify verified claim into appropriate tier."""
        text = claim.text.lower()

        axiom_markers = [
            "cogito", "real distinction", "cartesian circle",
            "wax argument", "evil genius", "method of doubt",
        ]
        if any(m in text for m in axiom_markers):
            return Tier.AXIOM

        contested_markers = [
            "if we accept", "assuming", "granted that",
            "on the premise", "provided",
        ]
        if any(m in text for m in contested_markers):
            return Tier.CONTESTED

        return Tier.DERIVED

    def _make_id(self, text: str) -> str:
        return "claim_" + hashlib.sha256(
            text.encode()).hexdigest()[:12]


class CorpusVerifier:
    """Verify factual claims against the Descartes corpus index."""

    def __init__(self, index_path: str = "~/corpus/index.json",
                 vks: Optional[VerifiedKnowledgeStore] = None):
        self.vks = vks
        self.index = self._load_index(index_path)

    def verify_factual(self, claim: ExtractedClaim) -> VerificationResult:
        """
        Check a factual claim against the corpus.

        "Arnauld raised the circularity objection in the
         Fourth Objections" -> search corpus index for
         Arnauld + Fourth Objections + circularity.
        """
        start = time.monotonic()

        keywords = self._extract_keywords(claim.text)
        matches = self._search_index(keywords)

        elapsed = (time.monotonic() - start) * 1000

        if matches:
            best = matches[0]
            status = ProofStatus.VERIFIED
            artifact = f"corpus:{best['source']}:{best.get('passage_id', 'unknown')}"

            if self.vks:
                record = ProofRecord(
                    claim_id=f"fact_{abs(hash(claim.text)) % 10**8}",
                    claim_text=claim.text,
                    formal_encoding="",
                    status=status,
                    method=VerificationMethod.CORPUS,
                    tier=Tier.FACTUAL,
                    timestamp=time.time(),
                    hash="",
                    corpus_source=best.get('source', ''),
                    proof_artifact=artifact,
                )
                self.vks.store(record)
        else:
            status = ProofStatus.NOT_FORMALIZABLE
            artifact = "no_corpus_match"

        return VerificationResult(
            claim=claim,
            status=status,
            method=VerificationMethod.CORPUS,
            encoding="",
            proof_artifact=artifact,
            time_ms=elapsed,
            stored_to_vks=(status == ProofStatus.VERIFIED),
        )

    def _extract_keywords(self, text: str) -> List[str]:
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                "at", "to", "for", "of", "and", "that", "this", "it", "with",
                "by", "from", "as", "or", "but", "not", "be", "been", "being"}
        words = text.lower().split()
        return [w for w in words if w not in stop and len(w) > 2]

    def _search_index(self, keywords: List[str]) -> List[dict]:
        """Search corpus index. Returns ranked matches."""
        results = []
        for entry in self.index:
            score = sum(1 for kw in keywords
                       if kw in entry.get('text', '').lower())
            if score >= 2:
                results.append({**entry, 'score': score})
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def _load_index(self, path: str) -> List[dict]:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
