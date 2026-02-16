"""
Text-level-only signal extraction for pure Ollama architecture.

When both local and oracle models run through Ollama's server,
you don't have access to hidden states. This extractor derives
uncertainty signals from response text alone.

Less accurate than the full extractor (signal_extractor.py) which
hooks into HuggingFace model internals, but production-viable.
The online feedback loop compensates — the lite meta-learner just
needs more training data to reach similar routing accuracy.

Use signal_extractor.py (full) when: HF local + Ollama cloud
Use this file (lite) when: pure Ollama on both sides
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List


HEDGE_WORDS = frozenset({
    "perhaps", "possibly", "might", "may", "could",
    "uncertain", "unclear", "debatable", "arguably",
    "not sure", "i believe", "it seems", "roughly",
    "approximately", "likely", "unlikely", "probably",
    "conceivably", "questionable", "speculative",
})

# Cartesian domain vocabulary — words the model SHOULD know well
DOMAIN_WORDS = frozenset({
    "cogito", "meditation", "descartes", "substance", "dualism",
    "extension", "thought", "res cogitans", "res extensa",
    "clear and distinct", "evil genius", "wax argument",
    "real distinction", "cartesian", "pineal", "elisabeth",
    "arnauld", "gassendi", "hobbes", "malebranche",
    "conceivability", "modal", "ontological", "trademark",
    "causal adequacy", "objective reality", "formal reality",
    "aspic", "z3", "formalization", "entailment",
})


@dataclass
class LiteSignals:
    """Text-only features for the lightweight meta-learner."""
    hedge_word_count: int
    hedge_word_density: float        # hedge_words / total_words
    repetition_rate: float
    response_length: int             # Word count
    sentence_count: int
    avg_sentence_length: float
    domain_word_density: float       # Fraction of words in domain vocab
    question_mark_count: int         # Self-questioning = uncertainty
    conditional_count: int           # "if", "would", "could" clauses
    negation_density: float          # "not", "no", "never" frequency
    lexical_diversity: float         # Unique words / total words

    def to_tensor(self) -> 'torch.Tensor':
        import torch
        return torch.tensor([
            self.hedge_word_count / 10.0,
            self.hedge_word_density,
            self.repetition_rate,
            self.response_length / 500.0,
            self.sentence_count / 20.0,
            self.avg_sentence_length / 30.0,
            self.domain_word_density,
            self.question_mark_count / 5.0,
            self.conditional_count / 10.0,
            self.negation_density,
            self.lexical_diversity,
        ], dtype=torch.float32)


class LiteSignalExtractor:
    """Extract uncertainty signals from response text only.

    Works with Ollama's server where you don't have access to
    hidden states. All 11 features are derived from text analysis.
    """

    def extract(self, response_text: str) -> LiteSignals:
        words = response_text.lower().split()
        total_words = max(len(words), 1)
        unique_words = len(set(words))

        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        hedge_count = sum(
            1 for hw in HEDGE_WORDS
            if hw in response_text.lower()
        )

        domain_count = sum(
            1 for w in words
            if w in DOMAIN_WORDS or any(dw in w for dw in DOMAIN_WORDS)
        )

        conditionals = sum(
            1 for w in words
            if w in {"if", "would", "could", "should", "might", "unless"}
        )

        negations = sum(
            1 for w in words
            if w in {"not", "no", "never", "neither", "nor", "cannot",
                     "don't", "doesn't", "isn't", "aren't",
                     "wasn't", "weren't"}
        )

        # 4-gram repetition rate
        if len(words) >= 20:
            ngrams = [tuple(words[i:i + 4])
                      for i in range(len(words) - 3)]
            rep_rate = 1.0 - len(set(ngrams)) / max(len(ngrams), 1)
        else:
            rep_rate = 0.0

        return LiteSignals(
            hedge_word_count=hedge_count,
            hedge_word_density=hedge_count / total_words,
            repetition_rate=rep_rate,
            response_length=total_words,
            sentence_count=len(sentences),
            avg_sentence_length=total_words / max(len(sentences), 1),
            domain_word_density=domain_count / total_words,
            question_mark_count=response_text.count('?'),
            conditional_count=conditionals,
            negation_density=negations / total_words,
            lexical_diversity=unique_words / total_words,
        )
