"""
Extract individual claims from LLM response and classify by type.
Each type routes to a different verification backend.

COGITO parallel: GroundingVerifier extracts claims as
FACTUAL/AFFECTIVE/EPISTEMIC/META/BEHAVIORAL.
We extract as FORMAL/FACTUAL/INTERPRETIVE/META_PHILOSOPHICAL.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class ClaimType(Enum):
    FORMAL = "formal"
    """
    Logic claims: "X is valid", "Y is consistent",
    "Z follows from W", modal claims, entailment.
    -> Verification: Z3/CVC5 or VKS lookup
    """

    FACTUAL = "factual"
    """
    Historical/attributional: "Arnauld argued X",
    "In the Fourth Objections", "Descartes wrote to Elisabeth".
    -> Verification: Corpus index lookup
    """

    INTERPRETIVE = "interpretive"
    """
    Scholarly judgment: "Descartes intended X",
    "The best reading of this passage is Y",
    "This parallels Chalmers' argument".
    -> Verification: Soft pass (meta-learner confidence only)
    """

    META_PHILOSOPHICAL = "meta"
    """
    Claims about the argument's structure or significance:
    "This is the strongest objection to dualism",
    "The conceivability premise does the heavy lifting".
    -> Verification: Soft pass
    """


@dataclass
class ExtractedClaim:
    """One claim extracted from the LLM response."""
    text: str
    claim_type: ClaimType
    confidence: float          # Extractor's confidence in classification
    position: Tuple[int, int]  # (start_char, end_char) in original

    # Set after verification
    verified: Optional[bool] = None
    verification_method: Optional[str] = None
    vks_hit: bool = False      # Was this already in the knowledge store?
    repair_attempted: bool = False
    repaired_text: Optional[str] = None


# Patterns that strongly indicate formal claims
FORMAL_INDICATORS = [
    r'\b(valid|invalid)\b',
    r'\b(consistent|inconsistent)\b',
    r'\b(entails|follows from|implies)\b',
    r'\b(necessary|sufficient|iff)\b',
    r'\bS[45]\b',
    r'\b(modal|possible world|accessible world)\b',
    r'\b(countermodel|counterexample)\b',
    r'\b(conceivability|conceivable)\b.*\b(possibility|possible)\b',
    r'\b(ASPIC|defeasible|undercutter|rebuttal)\b',
    r'\bZ3\b',
    r'\b(distinct|identical|identity)\b.*\b(substance|property)\b',
]

# Patterns for factual claims
FACTUAL_INDICATORS = [
    r'\b(Arnauld|Gassendi|Hobbes|Elisabeth|Mersenne)\b.*\b(argued|wrote|objected|responded|claimed)\b',
    r'\b(in the|from the)\b.*\b(Meditations?|Objections?|Replies|Principles|Discourse|Correspondence)\b',
    r'\b(First|Second|Third|Fourth|Fifth|Sixth)\b.*\b(Meditation|Objection|Reply)\b',
    r'\b(historically|traditionally|according to)\b',
    r'\b(published|written|composed)\b.*\b\d{4}\b',
]


class ClaimExtractor:
    """
    Split LLM response into individual typed claims.

    Method: Use regex heuristics to classify each sentence.
    Production upgrade: use the local LLM itself to parse.
    """

    def __init__(self, local_model: str = "descartes:8b"):
        self.local_model = local_model

    def extract(self, response_text: str) -> List[ExtractedClaim]:
        """Extract and classify all claims from a response."""
        sentences = self._split_sentences(response_text)

        claims = []
        pos = 0

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                pos += len(sentence) + 1
                continue

            ctype = self._classify_sentence(sentence)
            conf = self._classification_confidence(sentence, ctype)

            claims.append(ExtractedClaim(
                text=sentence.strip(),
                claim_type=ctype,
                confidence=conf,
                position=(pos, pos + len(sentence)),
            ))
            pos += len(sentence) + 1

        return claims

    def _classify_sentence(self, sentence: str) -> ClaimType:
        """Classify a single sentence by claim type."""
        s_lower = sentence.lower()

        formal_score = sum(
            1 for pat in FORMAL_INDICATORS
            if re.search(pat, s_lower)
        )

        factual_score = sum(
            1 for pat in FACTUAL_INDICATORS
            if re.search(pat, sentence)  # case-sensitive for names
        )

        meta_phrases = [
            "strongest", "weakest", "key insight",
            "does the heavy lifting", "crucial", "the point is",
            "fundamentally", "in principle", "the real question",
        ]
        meta_score = sum(1 for p in meta_phrases if p in s_lower)

        scores = {
            ClaimType.FORMAL: formal_score,
            ClaimType.FACTUAL: factual_score,
            ClaimType.META_PHILOSOPHICAL: meta_score,
        }

        if max(scores.values()) == 0:
            return ClaimType.INTERPRETIVE

        return max(scores, key=scores.get)

    def _classification_confidence(self, sentence: str,
                                    ctype: ClaimType) -> float:
        """How confident are we in this classification?"""
        s_lower = sentence.lower()

        if ctype == ClaimType.FORMAL:
            hits = sum(1 for p in FORMAL_INDICATORS
                      if re.search(p, s_lower))
            return min(0.5 + hits * 0.15, 0.99)

        elif ctype == ClaimType.FACTUAL:
            hits = sum(1 for p in FACTUAL_INDICATORS
                      if re.search(p, sentence))
            return min(0.5 + hits * 0.2, 0.99)

        return 0.5  # Interpretive/meta are default, low confidence

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving logical structure."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s for s in sentences if s.strip()]
