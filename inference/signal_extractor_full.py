"""
Full Signal Extractor — Hidden-State Feature Extraction.

Hooks into the HuggingFace model's forward pass to extract
internal uncertainty signals for the meta-learner:

- Hidden state statistics (mean, std of last layer activations)
- Per-token generation entropy (how uncertain each token choice was)
- Attention entropy (how dispersed attention is across heads)
- Hedge word detection (textual uncertainty markers)
- Repetition rate (sign of confabulation)
- Distribution similarity (is this query in-domain?)
- Out-of-distribution score (Mahalanobis distance proxy)

This extractor is MORE accurate than signal_extractor_lite.py
because it has access to model internals, but requires a
HuggingFace model running locally (not Ollama).

Use this when: HuggingFace local model + Ollama cloud oracle
Use signal_extractor_lite.py when: pure Ollama on both sides

11 features output (matching MetaLearner input_dim=11):
  hidden_mean, hidden_std, token_entropy, attention_entropy,
  hedge_ratio, repetition_ratio, response_length,
  ood_score, question_complexity, formal_claim_ratio,
  factual_claim_ratio

Reference: PHILOSOPHER_ENGINE_V3_UNIFIED_ARCHITECTURE.md, §8
"""

import torch
import numpy as np
import re
from typing import List, Optional
from dataclasses import dataclass


HEDGE_WORDS = frozenset({
    "perhaps", "possibly", "might", "may", "could",
    "uncertain", "unclear", "debatable", "arguably",
    "not sure", "i believe", "it seems", "roughly",
    "approximately", "likely", "unlikely", "probably",
    "conceivably", "questionable", "speculative",
})

# Indicators for claim type estimation
FORMAL_INDICATORS = re.compile(
    r'\b(valid|invalid|consistent|inconsistent|entails|follows|implies|'
    r'conceivable|conceivability|modal|S[45]|possible world|'
    r'necessarily|possibly|if and only if|iff|contradiction)\b',
    re.IGNORECASE
)
FACTUAL_INDICATORS = re.compile(
    r'\b(wrote|argued|published|stated|in the|from the|'
    r'Meditations?|Objections?|Replies|according to|'
    r'First|Second|Third|Fourth|Fifth|Sixth)\b',
    re.IGNORECASE
)


@dataclass
class FullSignals:
    """Features extracted from model internals + text analysis.

    All 11 features used by the full MetaLearner (input_dim=11).
    """
    hidden_mean: float           # Mean activation magnitude (normalized)
    hidden_std: float            # Activation variance (normalized)
    token_entropy: float         # Mean per-token entropy (normalized)
    attention_entropy: float     # Mean attention dispersion (normalized)
    hedge_ratio: float           # Hedge words / total words
    repetition_ratio: float      # 4-gram repetition rate
    response_length: float       # Token count (normalized by 500)
    ood_score: float             # Out-of-distribution estimate
    question_complexity: float   # Input complexity proxy
    formal_claim_ratio: float    # Fraction of formal-indicator sentences
    factual_claim_ratio: float   # Fraction of factual-indicator sentences

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for meta-learner input."""
        return torch.tensor([
            self.hidden_mean,
            self.hidden_std,
            self.token_entropy,
            self.attention_entropy,
            self.hedge_ratio,
            self.repetition_ratio,
            self.response_length,
            self.ood_score,
            self.question_complexity,
            self.formal_claim_ratio,
            self.factual_claim_ratio,
        ], dtype=torch.float32)


class FullSignalExtractor:
    """Hooks into HF model forward pass for hidden-state signals.

    Registers forward hooks on the last transformer layer to capture:
    1. Hidden state activations (mean + std)
    2. Attention patterns (entropy)
    3. Token-level logit distributions (entropy)

    Combined with text-level features for a complete 11-dim signal vector.
    """

    def __init__(self, model=None, tokenizer=None):
        """
        Args:
            model: HuggingFace CausalLM model (optional, set later)
            tokenizer: HuggingFace tokenizer (optional, set later)
        """
        self.model = model
        self.tokenizer = tokenizer
        self._hidden_states = []
        self._attention_maps = []
        self._hooks = []

        if model is not None:
            self._register_hooks()

    def set_model(self, model, tokenizer):
        """Set/update the model and register hooks."""
        self._remove_hooks()
        self.model = model
        self.tokenizer = tokenizer
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on the last transformer layer."""
        layers = None

        # Find layers — works for Qwen, LLaMA, Mistral, etc.
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                layers = self.model.model.layers

        if layers is not None and len(layers) > 0:
            last_layer = layers[-1]
            h = last_layer.register_forward_hook(self._capture_hidden)
            self._hooks.append(h)

            if hasattr(last_layer, 'self_attn'):
                h = last_layer.self_attn.register_forward_hook(
                    self._capture_attention)
                self._hooks.append(h)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _capture_hidden(self, module, input, output):
        """Capture hidden states from last layer."""
        if isinstance(output, tuple):
            self._hidden_states.append(output[0].detach())
        else:
            self._hidden_states.append(output.detach())

    def _capture_attention(self, module, input, output):
        """Capture attention weights if available."""
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None:
                self._attention_maps.append(attn_weights.detach())

    def clear(self):
        """Clear captured states for next generation."""
        self._hidden_states = []
        self._attention_maps = []

    def extract(self, response_text: str,
                input_ids: Optional[torch.Tensor] = None,
                generated_ids: Optional[torch.Tensor] = None,
                query_text: Optional[str] = None) -> FullSignals:
        """Extract all 11 signals after generation completes.

        If model is not attached (Ollama mode), falls back to
        text-only features with default values for hidden-state features.

        Args:
            response_text: The generated response text
            input_ids: Prompt token IDs (optional, for HF mode)
            generated_ids: Full sequence token IDs (optional, for HF mode)
            query_text: Original query (for complexity estimation)
        """
        words = response_text.lower().split()
        total_words = max(len(words), 1)
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # --- Hidden state features ---
        if self._hidden_states:
            last_hidden = self._hidden_states[-1]  # [batch, seq, hidden]
            prompt_len = input_ids.shape[-1] if input_ids is not None else 0

            if last_hidden.shape[1] > prompt_len:
                gen_hidden = last_hidden[:, prompt_len:, :]
            else:
                gen_hidden = last_hidden

            h_mean = gen_hidden.mean().item()
            h_std = gen_hidden.std().item()

            # Normalize (typical 8B model: mean ~0.01, std ~0.5)
            hidden_mean_norm = min(abs(h_mean) / 0.1, 1.0)
            hidden_std_norm = min(h_std / 2.0, 1.0)
        else:
            hidden_mean_norm = 0.5  # Default when no model attached
            hidden_std_norm = 0.5

        # --- Token entropy ---
        if (self.model is not None and generated_ids is not None
                and input_ids is not None):
            token_entropy = self._compute_token_entropy(
                input_ids, generated_ids)
        else:
            # Proxy from text: sentences with more hedge words -> higher entropy
            hedge_count = sum(1 for hw in HEDGE_WORDS
                              if hw in response_text.lower())
            token_entropy = min(hedge_count / 5.0, 1.0)

        # --- Attention entropy ---
        if self._attention_maps:
            attn = self._attention_maps[-1]  # [batch, heads, seq, seq]
            attn_probs = attn[0].mean(dim=0)  # [seq, seq]
            eps = 1e-10
            attn_ent = -torch.sum(
                attn_probs * torch.log(attn_probs + eps), dim=-1)
            attention_entropy = min(attn_ent.mean().item() / 8.0, 1.0)
        else:
            attention_entropy = 0.5  # Default

        # --- Text-level features ---
        hedge_count = sum(1 for hw in HEDGE_WORDS
                          if hw in response_text.lower())
        hedge_ratio = hedge_count / total_words

        # 4-gram repetition
        if len(words) >= 20:
            ngrams = [tuple(words[i:i + 4])
                      for i in range(len(words) - 3)]
            repetition_ratio = 1.0 - len(set(ngrams)) / max(len(ngrams), 1)
        else:
            repetition_ratio = 0.0

        response_length = total_words / 500.0

        # OOD score (proxy: inverse of domain word density)
        domain_words = {
            "cogito", "meditation", "descartes", "substance", "dualism",
            "extension", "thought", "res cogitans", "res extensa",
            "clear and distinct", "real distinction", "cartesian",
        }
        domain_count = sum(1 for w in words if w in domain_words)
        domain_density = domain_count / total_words
        ood_score = max(0.0, 1.0 - domain_density * 10)

        # Question complexity (proxy: word count + clause count)
        if query_text:
            q_words = len(query_text.split())
            q_clauses = query_text.count(',') + query_text.count(';') + 1
            question_complexity = min((q_words * q_clauses) / 200.0, 1.0)
        else:
            question_complexity = 0.5

        # Claim type ratios
        formal_sents = sum(
            1 for s in sentences if FORMAL_INDICATORS.search(s))
        factual_sents = sum(
            1 for s in sentences if FACTUAL_INDICATORS.search(s))
        total_sents = max(len(sentences), 1)

        formal_claim_ratio = formal_sents / total_sents
        factual_claim_ratio = factual_sents / total_sents

        return FullSignals(
            hidden_mean=hidden_mean_norm,
            hidden_std=hidden_std_norm,
            token_entropy=token_entropy,
            attention_entropy=attention_entropy,
            hedge_ratio=hedge_ratio,
            repetition_ratio=repetition_ratio,
            response_length=response_length,
            ood_score=ood_score,
            question_complexity=question_complexity,
            formal_claim_ratio=formal_claim_ratio,
            factual_claim_ratio=factual_claim_ratio,
        )

    def _compute_token_entropy(self, input_ids: torch.Tensor,
                               full_ids: torch.Tensor) -> float:
        """Compute mean per-token generation entropy."""
        try:
            with torch.no_grad():
                outputs = self.model(full_ids, output_hidden_states=False)
                logits = outputs.logits  # [1, seq_len, vocab_size]

            prompt_len = input_ids.shape[-1]
            gen_logits = logits[0, prompt_len - 1:-1, :]

            entropies = []
            for i in range(gen_logits.shape[0]):
                probs = torch.softmax(gen_logits[i], dim=-1)
                entropy = -torch.sum(
                    probs * torch.log(probs + 1e-10)).item()
                entropies.append(entropy)

            mean_ent = sum(entropies) / max(len(entropies), 1)
            # Normalize: typical range 2-10 for 8B models
            return min(mean_ent / 10.0, 1.0)
        except Exception:
            return 0.5

    def __del__(self):
        """Clean up hooks on destruction."""
        self._remove_hooks()
