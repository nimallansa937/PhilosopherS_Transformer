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
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from collections import deque


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
            "routing_decision": ["SELF", "ORACLE", "HYBRID"][
                routing_logits.argmax(dim=-1).item()],
            "error_logits": error_logits,
            "error_type": [
                "NONE", "FACTUAL_GAP", "REASONING_ERROR",
                "FORMALIZATION_ERROR", "SCOPE_EXCEEDED"
            ][error_logits.argmax(dim=-1).item()],
            "features": features,  # Cache for feedback loop
        }


class FeedbackBuffer:
    """Stores interaction outcomes for online meta-learner training.

    The core of the feedback-forward mechanism. Every interaction
    produces a training signal that improves future routing.
    """

    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = 32

    def record(self,
               features: torch.Tensor,
               predicted_confidence: float,
               predicted_routing: str,
               actual_outcome: dict):
        """Record an interaction outcome.

        actual_outcome contains:
          - "oracle_agreed": bool (did oracle confirm small model?)
          - "z3_verified": bool | None (did Z3 confirm formal claims?)
          - "user_accepted": bool | None (did user accept answer?)
          - "correction_magnitude": float (how much oracle changed answer)
        """

        # Compute ground truth confidence from outcome signals
        true_confidence = self._compute_true_confidence(actual_outcome)

        # Compute ground truth routing
        true_routing = self._compute_true_routing(
            actual_outcome, true_confidence)

        # Compute error type
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
        """Derive true confidence from multiple outcome signals.

        Weighted combination:
        - Z3 verification: strongest signal (binary, reliable)
        - Oracle agreement: strong signal (but oracle can be wrong too)
        - User acceptance: weak signal (users accept wrong answers sometimes)
        """
        signals = []
        weights = []

        if outcome.get("z3_verified") is not None:
            signals.append(1.0 if outcome["z3_verified"] else 0.0)
            weights.append(3.0)  # Highest weight — Z3 is ground truth

        if outcome.get("oracle_agreed") is not None:
            signals.append(1.0 if outcome["oracle_agreed"] else 0.0)
            weights.append(2.0)

        if outcome.get("correction_magnitude") is not None:
            # Low correction = high confidence was justified
            signals.append(1.0 - min(outcome["correction_magnitude"], 1.0))
            weights.append(1.5)

        if outcome.get("user_accepted") is not None:
            signals.append(1.0 if outcome["user_accepted"] else 0.3)
            weights.append(0.5)  # Weakest signal

        if not signals:
            return 0.5  # No information

        return sum(s * w for s, w in zip(signals, weights)) / sum(weights)

    def _compute_true_routing(self, outcome: dict,
                               true_conf: float) -> int:
        """Derive what routing SHOULD have been.

        0=SELF, 1=ORACLE, 2=HYBRID
        """
        if true_conf >= 0.85:
            return 0  # Should have handled it alone
        elif true_conf < 0.4:
            return 1  # Should have deferred entirely
        else:
            return 2  # Hybrid was appropriate

    def _compute_error_type(self, outcome: dict) -> int:
        """Classify what went wrong (if anything).

        0=NONE, 1=FACTUAL_GAP, 2=REASONING_ERROR,
        3=FORMALIZATION_ERROR, 4=SCOPE_EXCEEDED
        """
        if outcome.get("z3_verified") is False:
            return 3  # Formalization error — Z3 rejected it

        if outcome.get("oracle_agreed") is False:
            mag = outcome.get("correction_magnitude", 0.5)
            if mag > 0.7:
                return 1  # Major factual gap
            else:
                return 2  # Reasoning error

        if outcome.get("correction_magnitude", 0) > 0.8:
            return 4  # Scope exceeded — question was out of domain

        return 0  # No error

    def sample_batch(self) -> Optional[dict]:
        """Sample a training batch from the buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        # Prioritized sampling: recent interactions weighted higher
        # because the small model may have been updated
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

    Runs in the background after each oracle consultation.
    Each oracle interaction produces a ground-truth label,
    which trains the meta-learner to be better calibrated
    for future queries.
    """

    def __init__(self, meta_learner: MetaLearner,
                 lr: float = 1e-4):
        self.meta = meta_learner
        self.buffer = FeedbackBuffer()
        self.optimizer = torch.optim.AdamW(
            meta_learner.parameters(), lr=lr)

        # Loss functions
        self.confidence_loss = nn.MSELoss()
        self.routing_loss = nn.CrossEntropyLoss()
        self.error_loss = nn.CrossEntropyLoss()

        # Loss weights — confidence matters most
        self.w_conf = 2.0
        self.w_route = 1.5
        self.w_error = 1.0

        # Training stats
        self.update_count = 0
        self.loss_history = deque(maxlen=1000)

    def record_outcome(self, features: torch.Tensor,
                       predicted_confidence: float,
                       predicted_routing: str,
                       actual_outcome: dict):
        """Record an interaction and maybe train."""

        self.buffer.record(features, predicted_confidence,
                          predicted_routing, actual_outcome)

        # Train every 8 new interactions
        if len(self.buffer.buffer) >= 32 and len(self.buffer.buffer) % 8 == 0:
            self._train_step()

    def _train_step(self):
        """One gradient step on buffered feedback."""

        batch = self.buffer.sample_batch()
        if batch is None:
            return

        self.meta.train()
        self.optimizer.zero_grad()

        # Forward through meta-learner heads directly from cached features
        features = batch["features"]

        conf_pred = self.meta.confidence_head(features).squeeze(-1)
        route_pred = self.meta.routing_head(features)
        error_pred = self.meta.error_head(features)

        # Compute losses
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
        checkpoint = torch.load(path)
        self.meta.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_count = checkpoint["update_count"]
