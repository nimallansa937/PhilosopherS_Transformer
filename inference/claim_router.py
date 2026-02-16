"""
Route individual claims to appropriate verification backends.
This replaces query-level routing from Addendum A.

The meta-learner still operates, but now at claim level:
it decides confidence per-claim, not per-response.
"""

from typing import List, Dict, Optional
from .claim_extractor import ExtractedClaim, ClaimType
from .knowledge_store import VerifiedKnowledgeStore, ProofRecord


class ClaimRouter:
    """Routes each claim to the right verification backend."""

    def __init__(self, vks: VerifiedKnowledgeStore):
        self.vks = vks

    def route(self, claims: List[ExtractedClaim]) -> Dict[str, List[ExtractedClaim]]:
        """
        Sort claims into verification buckets.

        Returns dict with keys:
          'vks_hit'   -- already verified, skip
          'z3'        -- needs Z3/CVC5 verification
          'corpus'    -- needs corpus index lookup
          'soft_pass' -- interpretive/meta, no hard verification
        """
        buckets = {
            'vks_hit': [],
            'z3': [],
            'corpus': [],
            'soft_pass': [],
        }

        for claim in claims:
            # ALWAYS check VKS first, regardless of type
            existing = self.vks.lookup(claim.text)
            if existing and existing.status.value in ('verified', 'conditional'):
                claim.verified = True
                claim.verification_method = 'vks_cache'
                claim.vks_hit = True
                buckets['vks_hit'].append(claim)
                continue

            # Route by type
            if claim.claim_type == ClaimType.FORMAL:
                buckets['z3'].append(claim)

            elif claim.claim_type == ClaimType.FACTUAL:
                buckets['corpus'].append(claim)

            else:
                # INTERPRETIVE and META_PHILOSOPHICAL
                claim.verified = True  # soft pass
                claim.verification_method = 'soft_pass'
                buckets['soft_pass'].append(claim)

        return buckets
