"""
Meta-Learner: Learned confidence estimation and routing for the
Descartes cascade engine.

Instead of parsing [CONFIDENCE: 0.X] tags from text output, the
meta-learner reads the model's internal state (hidden states,
attention entropy, token entropy) and predicts:
  1. Confidence score (calibrated probability of correctness)
  2. Routing decision (SELF / ORACLE / HYBRID)
  3. Error type (NONE / FACTUAL_GAP / REASONING_ERROR /
     FORMALIZATION_ERROR / SCOPE_EXCEEDED)

The meta-learner trains ONLINE from oracle feedback — every oracle
interaction produces a ground-truth label that improves future routing.

THREE VARIANTS:
  MetaLearner      — Original (~100K params), works with ModelSignals
  MetaLearnerFull  — Addendum (~12M params), 4160-dim flat tensor input
  MetaLearnerLite  — Addendum (~50K params), 11-dim text-only input

Use MetaLearnerFull with signal_extractor.py (HF hooks, hidden states)
Use MetaLearnerLite with signal_extractor_lite.py (text-only, pure Ollama)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from collections import deque


# ============================================================
# SHARED LABELS
# ============================================================

ROUTING_LABELS = ["SELF", "ORACLE", "HYBRID"]
ERROR_LABELS = [
    "NONE", "FACTUAL_GAP", "REASONING_ERROR",
    "FORMALIZATION_ERROR", "SCOPE_EXCEEDED"
]


# ============================================================
# MODEL SIGNALS (for original MetaLearner + SignalExtractor)
# ============================================================

@dataclass
class ModelSignals:
    """Raw signals extracted from the small model during generation.

    These are the inputs to the meta-learner. Each captures a
    different facet of model uncertainty.
    """
    hidden_state_mean: torch.Tensor   # Mean of last-layer hidden states
    hidden_state_std: torch.Tensor    # Std of last-layer hidden states
    token_entropies: List[float]      # Per-token generation entropy
    attention_entropy: float          # Average attention dispersion
    topic_embedding: torch.Tensor     # First generated token's hidden state
    hedge_word_count: int             # Count of uncertainty language
    repetition_rate: float            # 4-gram repetition ratio
    query_similarity: float           # Estimated proximity to training dist
    response_length: int = 0          # Number of generated tokens

    def to_tensor(self, target_dim: int = 4160) -> torch.Tensor:
        """Flatten all signals into a single feature vector.

        Used by MetaLearnerFull which takes a flat tensor input.
        """
        scalar_features = torch.tensor([
            self.attention_entropy,
            self.hedge_word_count / 10.0,
            self.repetition_rate,
            self.query_similarity,
            np.mean(self.token_entropies) if self.token_entropies else 5.0,
            np.std(self.token_entropies) if len(self.token_entropies) > 1 else 0.0,
            np.max(self.token_entropies) if self.token_entropies else 10.0,
            np.min(self.token_entropies) if self.token_entropies else 0.0,
            self.response_length / 500.0,
        ], dtype=torch.float32)

        combined = torch.cat([
            self.hidden_state_mean.float().flatten()[:4096],
            self.hidden_state_std.float().flatten()[:32],
            self.topic_embedding.float().flatten()[:16],
            scalar_features,
        ])

        # Pad or truncate to target_dim
        if combined.shape[0] < target_dim:
            combined = torch.nn.functional.pad(
                combined, (0, target_dim - combined.shape[0]))
        else:
            combined = combined[:target_dim]

        return combined


# ============================================================
# ORIGINAL META-LEARNER (from meta learner.md)
# Works with ModelSignals objects directly
# ============================================================

class MetaLearner(nn.Module):
    """Neural meta-learner for confidence estimation and routing.

    Architecture:
    - Feature encoder: compresses all signals into a fixed-size vector
    - Confidence head: predicts calibrated confidence [0, 1]
    - Routing head: predicts SELF/ORACLE/HYBRID distribution
    - Error head: predicts error type distribution

    The meta-learner is tiny (~100K parameters) and trains in real-time.
    """

    def __init__(self, hidden_dim: int = 4096, feature_dim: int = 256):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # Feature encoder — compresses model signals into fixed vector
        self.hidden_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        self.std_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        self.topic_encoder = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        # Scalar features: entropy stats + hedge + repetition + similarity
        self.scalar_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

        # Combined feature dimension
        combined_dim = feature_dim * 4

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self.routing_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # SELF, ORACLE, HYBRID
        )

        self.error_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # NONE, FACTUAL_GAP, REASONING_ERROR,
                               # FORMALIZATION_ERROR, SCOPE_EXCEEDED
        )

    def encode_signals(self, signals: ModelSignals) -> torch.Tensor:
        """Encode all model signals into a fixed feature vector."""

        # Hidden state statistics
        h_mean = signals.hidden_state_mean
        if h_mean.dim() == 1:
            h_mean = h_mean.unsqueeze(0)
        h_mean_feat = self.hidden_encoder(h_mean)

        h_std = signals.hidden_state_std
        if h_std.dim() == 1:
            h_std = h_std.unsqueeze(0)
        h_std_feat = self.std_encoder(h_std)

        # Topic embedding
        topic = signals.topic_embedding
        if topic.dim() == 1:
            topic = topic.unsqueeze(0)
        topic_feat = self.topic_encoder(topic)

        # Scalar features
        entropies = signals.token_entropies
        mean_entropy = np.mean(entropies) if entropies else 5.0
        max_entropy = max(entropies) if entropies else 10.0

        scalars = torch.tensor([[
            mean_entropy,
            max_entropy,
            signals.attention_entropy,
            float(signals.hedge_word_count),
            signals.repetition_rate,
        ]], dtype=torch.float32)

        if h_mean.is_cuda:
            scalars = scalars.to(h_mean.device)

        scalar_feat = self.scalar_encoder(scalars)

        # Concatenate all features
        combined = torch.cat([
            h_mean_feat, h_std_feat, topic_feat, scalar_feat
        ], dim=-1)

        return combined

    def forward(self, signals: ModelSignals) -> Dict:
        """Full forward pass: signals → confidence + routing + error."""

        combined = self.encode_signals(signals)
        features = self.trunk(combined)

        confidence = self.confidence_head(features)
        routing_logits = self.routing_head(features)
        error_logits = self.error_head(features)

        return {
            "confidence": confidence.squeeze(-1),
            "routing_logits": routing_logits,
            "routing_decision": ROUTING_LABELS[
                routing_logits.argmax(dim=-1).item()],
            "error_logits": error_logits,
            "error_type": ERROR_LABELS[
                error_logits.argmax(dim=-1).item()],
            "features": features,  # Cache for feedback loop
        }


# ============================================================
# META-LEARNER FULL (Addendum — for HF hooks + Ollama oracle)
# Input: 4160-dim flat tensor from ModelSignals.to_tensor()
# Parameters: ~12M
# ============================================================

class MetaLearnerFull(nn.Module):
    """Meta-learner for use with full signal extractor.

    Input: 4160-dim feature vector (hidden states + scalars)
    Parameters: ~12M
    Latency: <1ms on CPU
    """

    def __init__(self, input_dim: int = 4160,
                 feature_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        )

        # Confidence head: scalar [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Routing head: 3 classes (SELF=0, ORACLE=1, HYBRID=2)
        self.routing_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

        # Error type head: 5 classes
        self.error_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

        self.routing_labels = ROUTING_LABELS
        self.error_labels = ERROR_LABELS

    def forward(self, signal_tensor: torch.Tensor) -> Dict:
        """
        Args:
            signal_tensor: [batch, input_dim] or [input_dim]
        Returns:
            Dict with confidence, routing, error predictions
        """
        if signal_tensor.dim() == 1:
            signal_tensor = signal_tensor.unsqueeze(0)

        features = self.encoder(signal_tensor)

        conf = self.confidence_head(features).squeeze(-1)
        route_logits = self.routing_head(features)
        error_logits = self.error_head(features)

        route_idx = route_logits.argmax(dim=-1).item()
        error_idx = error_logits.argmax(dim=-1).item()

        return {
            "confidence": conf,
            "routing_logits": route_logits,
            "routing_decision": self.routing_labels[route_idx],
            "error_logits": error_logits,
            "error_type": self.error_labels[error_idx],
            "features": features,
        }


# ============================================================
# META-LEARNER LITE (Addendum — for pure Ollama, text-only)
# Input: 11-dim tensor from LiteSignals.to_tensor()
# Parameters: ~50K
# ============================================================

class MetaLearnerLite(nn.Module):
    """Meta-learner for use with lite (text-only) signal extractor.

    Input: 11-dim feature vector (text statistics)
    Parameters: ~50K (tiny)
    Latency: <0.1ms
    """

    def __init__(self, input_dim: int = 11):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.routing_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

        self.error_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
        )

        self.routing_labels = ROUTING_LABELS
        self.error_labels = ERROR_LABELS

    def forward(self, signal_tensor: torch.Tensor) -> Dict:
        if signal_tensor.dim() == 1:
            signal_tensor = signal_tensor.unsqueeze(0)

        features = self.encoder(signal_tensor)

        conf = self.confidence_head(features).squeeze(-1)
        route_logits = self.routing_head(features)
        error_logits = self.error_head(features)

        route_idx = route_logits.argmax(dim=-1).item()
        error_idx = error_logits.argmax(dim=-1).item()

        return {
            "confidence": conf,
            "routing_logits": route_logits,
            "routing_decision": self.routing_labels[route_idx],
            "error_logits": error_logits,
            "error_type": self.error_labels[error_idx],
            "features": features,
        }


# ============================================================
# LEGACY CLASSES (kept for backward compatibility with
# cascade_engine.py which imports them)
# ============================================================

class FeedbackBuffer:
    """Stores interaction outcomes for online meta-learner training.

    NOTE: The canonical version is in feedback.py (refactored).
    This version is kept for backward compatibility with
    cascade_engine.py. New code should use feedback.py.
    """

    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = 32

    def record(self,
               features: torch.Tensor,
               predicted_confidence: float,
               predicted_routing: str,
               actual_outcome: dict):
        """Record an interaction outcome."""

        true_confidence = self._compute_true_confidence(actual_outcome)
        true_routing = self._compute_true_routing(
            actual_outcome, true_confidence)
        true_error = self._compute_error_type(actual_outcome)

        self.buffer.append({
            "features": features.detach(),
            "true_confidence": true_confidence,
            "true_routing": true_routing,
            "true_error": true_error,
            "predicted_confidence": predicted_confidence,
            "predicted_routing": predicted_routing,
            "timestamp": torch.tensor(len(self.buffer)),
        })

    def _compute_true_confidence(self, outcome: dict) -> float:
        signals = []
        weights = []

        if outcome.get("z3_verified") is not None:
            signals.append(1.0 if outcome["z3_verified"] else 0.0)
            weights.append(3.0)

        if outcome.get("oracle_agreed") is not None:
            signals.append(1.0 if outcome["oracle_agreed"] else 0.0)
            weights.append(2.0)

        if outcome.get("correction_magnitude") is not None:
            signals.append(1.0 - min(outcome["correction_magnitude"], 1.0))
            weights.append(1.5)

        if outcome.get("user_accepted") is not None:
            signals.append(1.0 if outcome["user_accepted"] else 0.3)
            weights.append(0.5)

        if not signals:
            return 0.5

        return sum(s * w for s, w in zip(signals, weights)) / sum(weights)

    def _compute_true_routing(self, outcome: dict,
                               true_conf: float) -> int:
        if true_conf >= 0.85:
            return 0
        elif true_conf < 0.4:
            return 1
        else:
            return 2

    def _compute_error_type(self, outcome: dict) -> int:
        if outcome.get("z3_verified") is False:
            return 3
        if outcome.get("oracle_agreed") is False:
            mag = outcome.get("correction_magnitude", 0.5)
            if mag > 0.7:
                return 1
            else:
                return 2
        if outcome.get("correction_magnitude", 0) > 0.8:
            return 4
        return 0

    def sample_batch(self) -> Optional[dict]:
        if len(self.buffer) < self.batch_size:
            return None

        import random
        indices = list(range(len(self.buffer)))
        weights = [1.0 + i / len(self.buffer) for i in indices]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        sampled_idx = random.choices(indices, weights=probs,
                                      k=self.batch_size)

        batch = [self.buffer[i] for i in sampled_idx]

        return {
            "features": torch.stack([b["features"].squeeze(0)
                                     for b in batch]),
            "true_confidence": torch.tensor(
                [b["true_confidence"] for b in batch]),
            "true_routing": torch.tensor(
                [b["true_routing"] for b in batch], dtype=torch.long),
            "true_error": torch.tensor(
                [b["true_error"] for b in batch], dtype=torch.long),
        }


class MetaLearnerTrainer:
    """Online trainer for the meta-learner.

    NOTE: The canonical version is MetaTrainer in feedback.py.
    This version is kept for backward compatibility with
    cascade_engine.py. New code should use feedback.MetaTrainer.
    """

    def __init__(self, meta_learner: MetaLearner,
                 lr: float = 1e-4):
        self.meta = meta_learner
        self.buffer = FeedbackBuffer()
        self.optimizer = torch.optim.AdamW(
            meta_learner.parameters(), lr=lr)

        self.confidence_loss = nn.MSELoss()
        self.routing_loss = nn.CrossEntropyLoss()
        self.error_loss = nn.CrossEntropyLoss()

        self.w_conf = 2.0
        self.w_route = 1.5
        self.w_error = 1.0

        self.update_count = 0
        self.loss_history = deque(maxlen=1000)

    def record_outcome(self, features: torch.Tensor,
                       predicted_confidence: float,
                       predicted_routing: str,
                       actual_outcome: dict):
        """Record an interaction and maybe train."""

        self.buffer.record(features, predicted_confidence,
                          predicted_routing, actual_outcome)

        if len(self.buffer.buffer) >= 32 and len(self.buffer.buffer) % 8 == 0:
            self._train_step()

    def _train_step(self):
        """One gradient step on buffered feedback."""

        batch = self.buffer.sample_batch()
        if batch is None:
            return

        self.meta.train()
        self.optimizer.zero_grad()

        features = batch["features"]

        conf_pred = self.meta.confidence_head(features).squeeze(-1)
        route_pred = self.meta.routing_head(features)
        error_pred = self.meta.error_head(features)

        loss_conf = self.confidence_loss(
            conf_pred, batch["true_confidence"])
        loss_route = self.routing_loss(
            route_pred, batch["true_routing"])
        loss_error = self.error_loss(
            error_pred, batch["true_error"])

        total_loss = (self.w_conf * loss_conf +
                      self.w_route * loss_route +
                      self.w_error * loss_error)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        self.loss_history.append(total_loss.item())

        self.meta.eval()

        if self.update_count % 50 == 0:
            avg_loss = sum(self.loss_history) / len(self.loss_history)
            print(f"  [Meta-learner] Update {self.update_count}, "
                  f"avg loss: {avg_loss:.4f}")

    def save(self, path: str):
        """Save meta-learner state + buffer."""
        torch.save({
            "model_state": self.meta.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "buffer_size": len(self.buffer.buffer),
        }, path)

    def load(self, path: str):
        """Restore meta-learner state."""
        checkpoint = torch.load(
            path, map_location='cpu', weights_only=False)
        self.meta.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_count = checkpoint["update_count"]
