"""
Descartes Philosopher Engine V3.

Changes from V2 (Addendum A):
- Claim-level routing (not query-level)
- Verified Knowledge Store (persistent memory)
- Multi-tier verification (Z3 + corpus + soft)
- Self-repair before oracle escalation
- Per-claim annotations in output

The meta-learner from Addendum A still operates but now
receives per-claim feedback, not per-query feedback.

Usage:
    python -m inference.engine_v3
"""

import torch
import json
import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .knowledge_store import VerifiedKnowledgeStore, ProofStatus
from .claim_extractor import ClaimExtractor, ExtractedClaim, ClaimType
from .claim_router import ClaimRouter
from .verifier import FormalVerifier, CorpusVerifier, VerificationResult
from .self_repair import SelfRepairEngine
from .signal_extractor_lite import LiteSignalExtractor
from .meta_learner import MetaLearnerLite
from .feedback import MetaTrainer


DESCARTES_SYSTEM = (
    "You are a philosophical reasoning assistant specializing in "
    "Cartesian philosophy, early modern rationalism, and the "
    "mind-body problem. You analyze arguments using ASPIC+ "
    "argumentation schemes and Z3 formal verification."
)

ORACLE_SYSTEM = (
    "You are a philosophical knowledge oracle. A Descartes specialist "
    "needs help with specific claims it could not verify. Provide "
    "accurate philosophical knowledge for the specific claims listed."
)

INTEGRATION_TEMPLATE = (
    "Revise your response. Some claims were verified, some failed "
    "and have been corrected. Integrate the corrections while "
    "preserving all verified claims exactly.\n\n"
    "ORIGINAL QUESTION: {query}\n\n"
    "YOUR INITIAL RESPONSE: {initial}\n\n"
    "CLAIM-BY-CLAIM STATUS:\n{claim_status}\n\n"
    "ORACLE CORRECTIONS (if any):\n{oracle_corrections}\n\n"
    "Produce your final response. Mark verified claims with "
    "[VERIFIED]. Mark corrected claims with [CORRECTED]."
)


@dataclass
class EngineResultV3:
    query: str
    final_response: str
    claims: List[ExtractedClaim]
    vks_hits: int              # Claims answered from memory
    z3_verified: int           # Claims verified by Z3
    corpus_verified: int       # Claims verified by corpus
    soft_passed: int           # Interpretive claims
    self_repaired: int         # Claims fixed without oracle
    oracle_needed: int         # Claims that required oracle
    total_time_ms: float


class DescartesEngineV3:
    """Production engine with claim-level routing and VKS."""

    def __init__(self,
                 local_model: str = "descartes:8b",
                 oracle_model: str = "deepseek-v3.1:671-cloud",
                 vks_path: str = "~/models/vks.json",
                 meta_path: Optional[str] = None):

        self.local_model = local_model
        self.oracle_model = oracle_model

        # Knowledge Store
        self.vks = VerifiedKnowledgeStore(vks_path)
        if not self.vks.verify_integrity():
            raise RuntimeError("VKS integrity check failed!")

        # Components
        self.extractor = ClaimExtractor(local_model)
        self.router = ClaimRouter(self.vks)
        self.formal = FormalVerifier(self.vks, local_model)
        self.corpus = CorpusVerifier(vks=self.vks)
        self.repair = SelfRepairEngine(
            local_model, self.formal, self.corpus)

        # Meta-learner (from Addendum A, still used)
        self.signal_extractor = LiteSignalExtractor()
        self.meta = MetaLearnerLite(input_dim=11)
        self.trainer = MetaTrainer(self.meta)
        if meta_path and os.path.exists(os.path.expanduser(meta_path)):
            self.trainer.load(os.path.expanduser(meta_path))
        self.meta.eval()

        vks_stats = self.vks.get_stats()
        print(f"Engine V3 ready.")
        print(f"  VKS: {vks_stats['total']} records "
              f"({vks_stats['tiers']['AXIOM']} axioms)")
        print(f"  Meta-learner: {self.trainer.update_count} updates")

    def run(self, query: str) -> EngineResultV3:
        start = time.monotonic()

        # -- Step 1: Generate from local model --
        initial = self._chat_local(query)

        # -- Step 2: Extract and classify claims --
        claims = self.extractor.extract(initial)

        # -- Step 3: Route claims to verification backends --
        buckets = self.router.route(claims)

        # -- Step 4: Verify each bucket --
        failed_claims = []

        # 4a: VKS hits -- already done, claims marked verified
        vks_hits = len(buckets['vks_hit'])

        # 4b: Formal claims -> Z3
        z3_verified = 0
        for claim in buckets['z3']:
            result = self.formal.verify_formal(claim)
            if result.status == ProofStatus.VERIFIED:
                claim.verified = True
                claim.verification_method = 'z3'
                z3_verified += 1
            else:
                failed_claims.append((claim, result))

        # 4c: Factual claims -> corpus
        corpus_verified = 0
        for claim in buckets['corpus']:
            result = self.corpus.verify_factual(claim)
            if result.status == ProofStatus.VERIFIED:
                claim.verified = True
                claim.verification_method = 'corpus'
                corpus_verified += 1
            else:
                failed_claims.append((claim, result))

        # 4d: Soft pass claims already marked
        soft_passed = len(buckets['soft_pass'])

        # -- Step 5: Self-repair failed claims --
        still_failed = []
        self_repaired = 0

        for claim, failed_result in failed_claims:
            repair_result = self.repair.attempt_repair(
                claim, failed_result)
            if repair_result and repair_result.status == ProofStatus.VERIFIED:
                claim.verified = True
                claim.verification_method = 'self_repair'
                self_repaired += 1
            else:
                still_failed.append(claim)

        # -- Step 6: Oracle for remaining failures --
        oracle_corrections = ""
        oracle_needed = len(still_failed)

        if still_failed:
            try:
                import ollama
                failed_texts = "\n".join(
                    f"- [{c.claim_type.value}] {c.text}"
                    for c in still_failed
                )

                oracle_resp = ollama.chat(
                    model=self.oracle_model,
                    messages=[
                        {"role": "system", "content": ORACLE_SYSTEM},
                        {"role": "user", "content": (
                            f"The specialist could not verify these "
                            f"claims. Provide corrections:\n\n"
                            f"{failed_texts}\n\n"
                            f"Context (original question): {query}"
                        )}
                    ]
                )
                oracle_corrections = oracle_resp['message']['content']

                for claim in still_failed:
                    claim.verified = True  # oracle-corrected
                    claim.verification_method = 'oracle'
            except Exception as e:
                oracle_corrections = f"Oracle unavailable: {e}"

        # -- Step 7: Integration pass (if any claims were corrected) --
        if self_repaired > 0 or oracle_needed > 0:
            claim_status = "\n".join(
                f"- {c.text}: {c.verification_method}"
                for c in claims
            )

            final = self._chat_local_with_system(
                INTEGRATION_TEMPLATE.format(
                    query=query,
                    initial=initial,
                    claim_status=claim_status,
                    oracle_corrections=oracle_corrections or "None needed.",
                )
            )
        else:
            final = initial

        # -- Step 8: Feedback to meta-learner --
        try:
            signals = self.signal_extractor.extract(initial)
            signal_tensor = signals.to_tensor()
            with torch.no_grad():
                meta_out = self.meta(signal_tensor)

            z3_correct = z3_verified + vks_hits
            z3_total = z3_correct + oracle_needed + self_repaired
            z3_accuracy = z3_correct / max(z3_total, 1)

            self.trainer.record_and_maybe_train(
                meta_out["features"].detach(),
                meta_out["confidence"].item(),
                meta_out["routing_decision"],
                {
                    "z3_verified": z3_accuracy > 0.8,
                    "oracle_agreed": oracle_needed == 0,
                    "correction_magnitude": oracle_needed / max(len(claims), 1),
                    "user_accepted": None,
                }
            )
        except Exception:
            pass  # Non-critical; training continues on next query

        elapsed = (time.monotonic() - start) * 1000

        return EngineResultV3(
            query=query,
            final_response=final,
            claims=claims,
            vks_hits=vks_hits,
            z3_verified=z3_verified,
            corpus_verified=corpus_verified,
            soft_passed=soft_passed,
            self_repaired=self_repaired,
            oracle_needed=oracle_needed,
            total_time_ms=elapsed,
        )

    def _chat_local(self, query: str) -> str:
        import ollama
        resp = ollama.chat(
            model=self.local_model,
            messages=[
                {"role": "system", "content": DESCARTES_SYSTEM},
                {"role": "user", "content": query}
            ]
        )
        return resp['message']['content']

    def _chat_local_with_system(self, prompt: str) -> str:
        import ollama
        resp = ollama.chat(
            model=self.local_model,
            messages=[
                {"role": "system", "content": DESCARTES_SYSTEM},
                {"role": "user", "content": prompt}
            ]
        )
        return resp['message']['content']

    def save(self, meta_path: str):
        self.trainer.save(meta_path)

    def get_stats(self) -> Dict:
        return {
            "vks": self.vks.get_stats(),
            "repair": self.repair.stats,
            "meta_learner": self.trainer.get_stats(),
        }


# -- Interactive REPL --

def main():
    engine = DescartesEngineV3(
        local_model="descartes:8b",
        oracle_model="deepseek-v3.1:671-cloud",
        vks_path="~/models/vks.json",
        meta_path="~/models/meta_learner_latest.pt",
    )

    print("\n" + "=" * 60)
    print("DESCARTES PHILOSOPHER ENGINE V3")
    print("  claim-level routing | VKS memory | self-repair")
    print("=" * 60)
    print("Commands: quit, stats, vks, good, bad")
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
        elif query == "vks":
            stats = engine.vks.get_stats()
            print(f"VKS: {stats['total']} records")
            print(f"  Axioms:    {stats['tiers']['AXIOM']}")
            print(f"  Derived:   {stats['tiers']['DERIVED']}")
            print(f"  Contested: {stats['tiers']['CONTESTED']}")
            print(f"  Factual:   {stats['tiers']['FACTUAL']}")
            print(f"  Integrity: {'[ok]' if stats['integrity'] else '[FAIL]'}")
            continue
        elif query in ("good", "bad"):
            continue

        result = engine.run(query)

        print(f"\n[VKS:{result.vks_hits} Z3:{result.z3_verified} "
              f"CORPUS:{result.corpus_verified} SOFT:{result.soft_passed} "
              f"REPAIR:{result.self_repaired} ORACLE:{result.oracle_needed}] "
              f"({result.total_time_ms:.0f}ms)")
        print(f"\n{result.final_response}")


if __name__ == "__main__":
    main()
