"""
Signal Extractor: Hooks into the small model during generation
to extract uncertainty signals for the meta-learner.

Extracts WITHOUT modifying the model:
- Hidden state statistics (mean, std of last layer)
- Per-token generation entropy (how uncertain each token choice was)
- Attention entropy (how dispersed attention is)
- Hedge word detection (textual uncertainty markers)
- Repetition rate (sign of confabulation)
- Distribution similarity (is this query in-domain?)
"""

import torch
import numpy as np
import re
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta_learner import ModelSignals


HEDGE_WORDS = {
    "perhaps", "possibly", "might", "may", "could",
    "uncertain", "unclear", "debatable", "arguably",
    "not sure", "i believe", "it seems", "roughly",
    "approximately", "likely", "unlikely"
}


class SignalExtractor:
    """Hooks into model forward pass to extract uncertainty signals."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_states = []
        self.attention_maps = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on the last transformer layer."""

        # Find last layer — works for most HuggingFace models
        layers = None
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                layers = self.model.model.layers

        if layers is not None:
            last_layer = layers[-1]
            last_layer.register_forward_hook(self._capture_hidden)

            # Hook attention if accessible
            if hasattr(last_layer, 'self_attn'):
                last_layer.self_attn.register_forward_hook(
                    self._capture_attention)

    def _capture_hidden(self, module, input, output):
        """Capture hidden states from last layer."""
        if isinstance(output, tuple):
            self.hidden_states.append(output[0].detach())
        else:
            self.hidden_states.append(output.detach())

    def _capture_attention(self, module, input, output):
        """Capture attention weights."""
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None:
                self.attention_maps.append(attn_weights.detach())

    def clear(self):
        """Clear captured states for next generation."""
        self.hidden_states = []
        self.attention_maps = []

    def extract_signals(self,
                        input_ids: torch.Tensor,
                        generated_ids: torch.Tensor,
                        generated_text: str) -> ModelSignals:
        """Extract all signals after a generation is complete.

        Args:
            input_ids: The prompt token IDs
            generated_ids: The full sequence (prompt + generated)
            generated_text: Decoded generated text
        """

        # --- Hidden state statistics ---
        if self.hidden_states:
            # Use only the generated portion's hidden states
            last_hidden = self.hidden_states[-1]
            prompt_len = input_ids.shape[-1]

            if last_hidden.shape[1] > prompt_len:
                gen_hidden = last_hidden[:, prompt_len:, :]
            else:
                gen_hidden = last_hidden

            h_mean = gen_hidden.mean(dim=1).squeeze(0)  # [hidden_dim]
            h_std = gen_hidden.std(dim=1).squeeze(0)
        else:
            hidden_dim = 4096  # Default for 8B models
            h_mean = torch.zeros(hidden_dim)
            h_std = torch.ones(hidden_dim)

        # --- Token-level entropy ---
        token_entropies = self._compute_token_entropies(
            input_ids, generated_ids)

        # --- Attention entropy ---
        attn_entropy = self._compute_attention_entropy()

        # --- Textual signals ---
        hedge_count = sum(
            1 for hw in HEDGE_WORDS
            if hw in generated_text.lower()
        )

        rep_rate = self._compute_repetition_rate(generated_text)

        # --- Query similarity to training distribution ---
        # Proxy: use mean hidden state norm
        # (in-distribution inputs produce more typical activations)
        query_sim = self._compute_distribution_similarity(h_mean)

        # --- Topic embedding ---
        # Use the hidden state of the first generated token as
        # a rough topic embedding
        if (self.hidden_states and
                last_hidden.shape[1] > prompt_len):
            topic_emb = last_hidden[0, prompt_len, :256]
        else:
            topic_emb = torch.zeros(256)

        return ModelSignals(
            hidden_state_mean=h_mean,
            hidden_state_std=h_std,
            token_entropies=token_entropies,
            attention_entropy=attn_entropy,
            topic_embedding=topic_emb,
            hedge_word_count=hedge_count,
            repetition_rate=rep_rate,
            query_similarity=query_sim,
        )

    def _compute_token_entropies(self,
                                  input_ids: torch.Tensor,
                                  full_ids: torch.Tensor) -> List[float]:
        """Compute per-token generation entropy.

        High entropy tokens = model is uncertain about that token.
        Sustained high entropy = model is in uncertain territory.
        """
        entropies = []

        try:
            with torch.no_grad():
                outputs = self.model(full_ids, output_hidden_states=False)
                logits = outputs.logits  # [1, seq_len, vocab_size]

            prompt_len = input_ids.shape[-1]
            gen_logits = logits[0, prompt_len - 1:-1, :]

            for i in range(gen_logits.shape[0]):
                probs = torch.softmax(gen_logits[i], dim=-1)
                entropy = -torch.sum(
                    probs * torch.log(probs + 1e-10)).item()
                entropies.append(entropy)
        except Exception:
            entropies = [5.0]  # Default moderate entropy

        return entropies

    def _compute_attention_entropy(self) -> float:
        """Compute average attention entropy across heads.

        Dispersed attention (high entropy) -> model is uncertain,
        looking everywhere for relevant context.
        Focused attention (low entropy) -> model knows exactly
        what to attend to.
        """
        if not self.attention_maps:
            return 5.0  # Default moderate entropy

        last_attn = self.attention_maps[-1]  # [batch, heads, seq, seq]
        # Average across heads and positions
        attn_probs = last_attn[0].mean(dim=0)  # [seq, seq]

        # Entropy of each position's attention distribution
        eps = 1e-10
        entropies = -torch.sum(
            attn_probs * torch.log(attn_probs + eps), dim=-1)

        return entropies.mean().item()

    def _compute_repetition_rate(self, text: str) -> float:
        """Detect repeated phrases (sign of confabulation).

        Models that don't know the answer tend to repeat
        themselves or generate circular text.
        """
        words = text.lower().split()
        if len(words) < 20:
            return 0.0

        # Check 4-gram repetitions
        ngrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        unique_ratio = len(set(ngrams)) / max(len(ngrams), 1)

        return 1.0 - unique_ratio  # 0 = no repetition, 1 = all repeated

    def _compute_distribution_similarity(self,
                                          h_mean: torch.Tensor) -> float:
        """Estimate how close this input is to training distribution.

        Simple proxy: L2 norm of mean hidden state. In-distribution
        inputs produce activations with typical norms; OOD inputs
        produce unusually large or small norms.

        In production: replace with a proper OOD detector
        (Mahalanobis distance from training distribution statistics).
        """
        norm = h_mean.norm().item()
        # Typical norm for 8B model hidden states is ~30-60
        # Normalize to [0, 1] range
        typical_norm = 45.0
        deviation = abs(norm - typical_norm) / typical_norm
        return max(0.0, 1.0 - deviation)
