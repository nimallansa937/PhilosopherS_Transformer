"""
Feedback buffer and online trainer for the meta-learner.

THREE LEARNING SIGNALS:

Signal 1 — Oracle Agreement
  When oracle is consulted, compare small model's answer to oracle's.
  Agreement = confidence should have been high.
  Disagreement = confidence should have been low.
  Automatic ground truth without human annotation.

Signal 2 — Z3 Verification
  When small model makes formal claims, Z3 verifies independently.
  Z3 verdicts are perfect ground truth. The meta-learner learns
  which internal states correlate with correct formalizations.

Signal 3 — User Feedback
  If user corrects the model -> negative signal.
  If user accepts and continues -> weak positive signal.
  Weakest signal, but accumulates over time.

CONVERGENCE:
  The meta-learner typically converges to useful routing accuracy
  within 200-500 oracle interactions. Bootstrap provides the
  initial data points to start warm.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Dict
import json
import time
import random


class FeedbackBuffer:
    """Stores interaction outcomes for online training."""

    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = 32

    def record(self,
               features: torch.Tensor,
               predicted_confidence: float,
               predicted_routing: str,
               actual_outcome: Dict):
        """Record one interaction outcome.

        actual_outcome keys:
          oracle_agreed: bool | None
          z3_verified: bool | None
          user_accepted: bool | None
          correction_magnitude: float (0=identical, 1=completely different)
        """
        true_conf = self._derive_confidence(actual_outcome)
        true_route = self._derive_routing(true_conf)
        true_error = self._derive_error_type(actual_outcome)

        self.buffer.append({
            "features": features.detach().cpu(),
            "true_confidence": true_conf,
            "true_routing": true_route,
            "true_error": true_error,
            "predicted_confidence": predicted_confidence,
            "predicted_routing": predicted_routing,
            "timestamp": time.time(),
        })

    def _derive_confidence(self, outcome: Dict) -> float:
        """Weighted combination of outcome signals.

        Z3 > Oracle > Correction magnitude > User feedback
        """
        signals, weights = [], []

        if outcome.get("z3_verified") is not None:
            signals.append(1.0 if outcome["z3_verified"] else 0.0)
            weights.append(3.0)

        if outcome.get("oracle_agreed") is not None:
            signals.append(1.0 if outcome["oracle_agreed"] else 0.0)
            weights.append(2.0)

        if outcome.get("correction_magnitude") is not None:
            signals.append(
                1.0 - min(outcome["correction_magnitude"], 1.0))
            weights.append(1.5)

        if outcome.get("user_accepted") is not None:
            signals.append(1.0 if outcome["user_accepted"] else 0.3)
            weights.append(0.5)

        if not signals:
            return 0.5

        return sum(s * w for s, w in zip(signals, weights)) / sum(weights)

    def _derive_routing(self, true_conf: float) -> int:
        """What routing SHOULD have been. 0=SELF, 1=ORACLE, 2=HYBRID."""
        if true_conf >= 0.85:
            return 0
        elif true_conf < 0.4:
            return 1
        else:
            return 2

    def _derive_error_type(self, outcome: Dict) -> int:
        """Classify error type. 0=NONE through 4=SCOPE_EXCEEDED."""
        if outcome.get("z3_verified") is False:
            return 3  # Formalization error

        if outcome.get("oracle_agreed") is False:
            mag = outcome.get("correction_magnitude", 0.5)
            if mag > 0.7:
                return 1  # Factual gap
            else:
                return 2  # Reasoning error

        if outcome.get("correction_magnitude", 0) > 0.8:
            return 4  # Scope exceeded

        return 0

    def sample_batch(self) -> Optional[Dict]:
        """Sample training batch with recency bias."""
        if len(self.buffer) < self.batch_size:
            return None

        indices = list(range(len(self.buffer)))
        # Recency weight: newer interactions more important
        weights = [1.0 + i / len(self.buffer) for i in indices]
        total = sum(weights)
        probs = [w / total for w in weights]

        sampled = random.choices(indices, weights=probs,
                                 k=self.batch_size)

        batch = [self.buffer[i] for i in sampled]

        return {
            "features": torch.stack([
                b["features"].squeeze(0) if b["features"].dim() > 1
                else b["features"]
                for b in batch
            ]),
            "true_confidence": torch.tensor(
                [b["true_confidence"] for b in batch]),
            "true_routing": torch.tensor(
                [b["true_routing"] for b in batch], dtype=torch.long),
            "true_error": torch.tensor(
                [b["true_error"] for b in batch], dtype=torch.long),
        }

    def save(self, path: str):
        """Persist buffer to disk."""
        data = [{
            "features": b["features"].tolist(),
            "true_confidence": b["true_confidence"],
            "true_routing": b["true_routing"],
            "true_error": b["true_error"],
            "predicted_confidence": b["predicted_confidence"],
            "predicted_routing": b["predicted_routing"],
            "timestamp": b["timestamp"],
        } for b in self.buffer]

        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Restore buffer from disk."""
        with open(path) as f:
            data = json.load(f)

        for d in data:
            d["features"] = torch.tensor(d["features"])
            self.buffer.append(d)


class MetaTrainer:
    """Online trainer with feedback-forward loop.

    Runs a gradient step every N new interactions.
    The meta-learner gets better at routing with every
    oracle consultation — each consultation is a free
    training example.
    """

    def __init__(self, meta_learner: nn.Module,
                 lr: float = 1e-4,
                 update_every: int = 8):
        self.meta = meta_learner
        self.buffer = FeedbackBuffer()
        self.optimizer = torch.optim.AdamW(
            meta_learner.parameters(), lr=lr, weight_decay=0.01)

        self.conf_loss_fn = nn.MSELoss()
        self.route_loss_fn = nn.CrossEntropyLoss()
        self.error_loss_fn = nn.CrossEntropyLoss()

        # Loss weights
        self.w_conf = 2.0    # Confidence matters most
        self.w_route = 1.5   # Routing decision matters
        self.w_error = 1.0   # Error type is helpful but less critical

        self.update_every = update_every
        self.interactions_since_update = 0
        self.update_count = 0
        self.loss_history = deque(maxlen=1000)

    def record_and_maybe_train(self,
                                features: torch.Tensor,
                                predicted_confidence: float,
                                predicted_routing: str,
                                outcome: Dict):
        """Record outcome, train if enough new data."""

        self.buffer.record(features, predicted_confidence,
                           predicted_routing, outcome)

        self.interactions_since_update += 1

        if (self.interactions_since_update >= self.update_every
                and len(self.buffer.buffer) >= self.buffer.batch_size):
            self._train_step()
            self.interactions_since_update = 0

    def _train_step(self):
        """One gradient step on buffered feedback."""
        batch = self.buffer.sample_batch()
        if batch is None:
            return

        self.meta.train()
        self.optimizer.zero_grad()

        features = batch["features"]

        # Forward through shared encoder then heads
        encoded = self.meta.encoder(features)

        conf_pred = self.meta.confidence_head(encoded).squeeze(-1)
        route_pred = self.meta.routing_head(encoded)
        error_pred = self.meta.error_head(encoded)

        loss_conf = self.conf_loss_fn(
            conf_pred, batch["true_confidence"])
        loss_route = self.route_loss_fn(
            route_pred, batch["true_routing"])
        loss_error = self.error_loss_fn(
            error_pred, batch["true_error"])

        total = (self.w_conf * loss_conf +
                 self.w_route * loss_route +
                 self.w_error * loss_error)

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
        self.optimizer.step()

        self.meta.eval()
        self.update_count += 1
        self.loss_history.append(total.item())

        if self.update_count % 50 == 0:
            avg = sum(self.loss_history) / len(self.loss_history)
            print(f"  [Meta] update {self.update_count}, "
                  f"loss={avg:.4f}, buffer={len(self.buffer.buffer)}")

    def save(self, path: str):
        torch.save({
            "model_state": self.meta.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

        buffer_path = path.replace('.pt', '_buffer.json')
        self.buffer.save(buffer_path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.meta.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.update_count = ckpt["update_count"]

        buffer_path = path.replace('.pt', '_buffer.json')
        try:
            self.buffer.load(buffer_path)
        except FileNotFoundError:
            pass

    def get_stats(self) -> Dict:
        return {
            "updates": self.update_count,
            "buffer_size": len(self.buffer.buffer),
            "avg_loss": (sum(self.loss_history) / len(self.loss_history)
                         if self.loss_history else None),
            "interactions_pending": self.interactions_since_update,
        }
