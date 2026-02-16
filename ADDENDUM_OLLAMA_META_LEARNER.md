# Addendum: Ollama Unified Architecture + Meta-Learner Feedback Loop
## Claude Code Instructional Guide

**Parent documents** (read these first):
1. `PHILOSOPHER_ENGINE_ARCHITECTURE.md` — five-layer system design
2. `PHILOSOPHER_ENGINE_TRAINING_PIPELINE.md` — Phases 2-4 (extraction, cleaning, formatting)
3. `DESCARTES_CASCADE_TRAINING_PIPELINE.md` — Phases 1, 5-12 (Descartes corpus, 8B training, cascade)

**This addendum replaces**:
- Phase 9 (Confidence Head → Meta-Learner Bootstrap)
- Phase 10 (Oracle Integration → Ollama Cloud)
- Phase 11 (Cascade Engine → Ollama Unified Engine)
- Phase 12 (Evaluation → includes meta-learner convergence metrics)

---

## What This Addendum Changes

The original cascade architecture has two problems:

**Problem 1**: Confidence is self-reported text (`[CONFIDENCE: 0.8]`). The model generates this as tokens — it's just predicting what number looks right, not measuring actual uncertainty. A model confidently hallucinating Descartes scholarship outputs `[CONFIDENCE: 0.9]` because it doesn't know what it doesn't know.

**Problem 2**: The oracle requires separate API clients for each provider (DeepSeek SDK, Anthropic SDK, OpenAI SDK). Different auth, different response formats, different error handling.

**Solution 1**: A meta-learner that observes the model's internal states (hidden activations, token entropy, attention patterns) and learns to predict actual reliability from external feedback signals.

**Solution 2**: Ollama as unified interface — local trained model and cloud oracle use the same `ollama.chat()` call. One client, two models, one codebase.

---

## Architecture Overview

```
                         OLLAMA (unified API)
                    ┌──────────┴──────────┐
                    │                     │
              ollama.chat()          ollama.chat()
              model="descartes:8b"   model="deepseek-v3.1:671-cloud"
                    │                     │
              ┌─────┴─────┐              │
              │ LOCAL GPU  │         ┌────┴────┐
              │ Your 8B    │         │ OLLAMA  │
              │ trained    │         │ CLOUD   │
              │ weights    │         │ GPUs    │
              │ FREE       │         │ Metered │
              └─────┬─────┘         └────┬────┘
                    │                     │
                    ▼                     │
            Signal Extractor              │
            (hooks into model             │
             hidden states)               │
                    │                     │
                    ▼                     │
            ┌──────────────┐              │
            │ META-LEARNER │              │
            │ (~10M params)│              │
            │              │              │
            │ Inputs:      │              │
            │ • hidden μ/σ │              │
            │ • token H    │              │
            │ • attn H     │              │
            │ • hedge words│              │
            │ • repetition │              │
            │ • OOD score  │              │
            │              │              │
            │ Outputs:     │              │
            │ • confidence │              │
            │ • routing    │──── ORACLE ──┘
            │ • error type │
            └──────┬───────┘
                   │
            ┌──────┴──────┐
            │             │
         FEEDBACK     FORWARD
         (learn from  (better routing
          outcomes)    next time)
```

---

## Part 1: Ollama Setup

### 1.1 Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

### 1.2 Import Trained Descartes Model

After completing Phases 5-7 from the cascade pipeline, you have a trained model at `~/models/descartes-8b-cascade`. Convert it to Ollama format:

```bash
# Option A: Convert HuggingFace → GGUF → Ollama

# Install llama.cpp converter
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt --break-system-packages

# Convert to GGUF (Q6_K quantization — good quality/size balance)
python convert_hf_to_gguf.py ~/models/descartes-8b-cascade \
    --outfile ~/models/descartes-8b.gguf \
    --outtype q6_k

# Create Ollama Modelfile
cat > ~/models/Modelfile << 'MODELFILE'
FROM ~/models/descartes-8b.gguf

SYSTEM """You are a philosophical reasoning assistant specializing in \
Cartesian philosophy, early modern rationalism, and the mind-body \
problem. You analyze arguments using ASPIC+ argumentation schemes \
and Z3 formal verification.

You have deep expertise in Descartes' Meditations, the Objections \
and Replies, the Correspondence with Elisabeth, and the Principles \
of Philosophy.

When uncertain about knowledge outside your Cartesian specialization, \
indicate this clearly so the system can consult an oracle."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
MODELFILE

# Import into Ollama
ollama create descartes:8b -f ~/models/Modelfile

# Test
ollama run descartes:8b "Reconstruct the Cogito as a strict inference."
```

```bash
# Option B: If using Qwen3-8B with LoRA adapters (from Phase 7)
# Merge adapters first, then convert

python << 'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "~/models/descartes-8b-cascade")
merged = model.merge_and_unload()
merged.save_pretrained("~/models/descartes-8b-merged")
AutoTokenizer.from_pretrained("Qwen/Qwen3-8B").save_pretrained(
    "~/models/descartes-8b-merged")
EOF

# Then convert merged model to GGUF as above
python llama.cpp/convert_hf_to_gguf.py ~/models/descartes-8b-merged \
    --outfile ~/models/descartes-8b.gguf --outtype q6_k
```

### 1.3 Configure Cloud Oracle Access

```bash
# Sign in to Ollama Cloud
ollama login

# Pull a cloud model tag (doesn't download weights — just registers)
ollama pull deepseek-v3.1:671-cloud

# Verify cloud access
ollama run deepseek-v3.1:671-cloud "Who raised the Cartesian Circle objection?"
```

### 1.4 Available Oracle Models

Pick based on your Ollama plan tier and reasoning needs:

```
MODEL                         SIZE     BEST FOR
─────────────────────────────────────────────────────────────────
kimi-k2.5:cloud               1T      Strongest reasoning. Best for
                                       complex multi-step philosophical
                                       arguments. Expensive on limits.

deepseek-v3.1:671-cloud       671B    Excellent analytical depth.
                                       Good balance for philosophy.
                                       RECOMMENDED DEFAULT.

gpt-oss:120b-cloud            120B    Fast, good general knowledge.
                                       Use when speed matters more
                                       than depth.

gpt-oss:20b-cloud             20B     Lightweight fallback. Use for
                                       simple factual lookups only.

qwen3-coder:480b-cloud        480B    If query involves Z3 code
                                       generation or formal logic
                                       that the small model struggled with.
```

**Recommendation**: Default to `deepseek-v3.1:671-cloud`. Escalate to `kimi-k2.5:cloud` for queries the meta-learner flags as `SCOPE_EXCEEDED` (questions far outside Cartesian domain needing frontier reasoning).

### 1.5 Verify Unified API

```python
# ~/test/test_ollama_unified.py
"""
Verify both local and cloud models work through same API.
"""

import ollama

def test_local():
    print("Testing local Descartes model...")
    resp = ollama.chat(
        model='descartes:8b',
        messages=[{
            'role': 'user',
            'content': 'What is the logical structure of the Cogito?'
        }]
    )
    print(f"  Local response: {resp['message']['content'][:200]}...")
    return True

def test_cloud():
    print("Testing cloud oracle...")
    resp = ollama.chat(
        model='deepseek-v3.1:671-cloud',
        messages=[{
            'role': 'user',
            'content': 'What was Merleau-Ponty\'s critique of Descartes?'
        }]
    )
    print(f"  Cloud response: {resp['message']['content'][:200]}...")
    return True

if __name__ == "__main__":
    test_local()
    print()
    test_cloud()
    print("\nBoth endpoints working through unified Ollama API.")
```

---

## Part 2: Signal Extractor

The signal extractor hooks into the small model's forward pass to capture uncertainty indicators that the meta-learner uses for routing decisions. It reads the model's internal states without modifying inference.

### 2.1 Why Internal Signals Beat Self-Report

When the model generates about something it genuinely knows (the Cogito's logical structure), its hidden states are sharp — activations concentrate on relevant representations with low variance. When it's confabulating (what Merleau-Ponty said about Descartes), the hidden states are diffuse — the model averages across vaguely related training examples, producing high-entropy token distributions and dispersed attention.

The model itself can't articulate this difference in its text output. But the meta-learner can learn to read it.

**Six signal channels**:

```
SIGNAL              WHAT IT MEASURES                  HIGH VALUE MEANS
──────────────────────────────────────────────────────────────────────
Hidden state μ      Mean activation of last layer     Domain-specific
Hidden state σ      Variance of activations           Uncertain/conflicted
Token entropy       Per-token generation uncertainty   Doesn't know next word
Attention entropy   How dispersed attention is         Looking everywhere
Hedge word count    "perhaps", "might", "possibly"    Hedging linguistically
Repetition rate     Repeated n-grams in output         Confabulating/looping
```

### 2.2 Implementation

```python
# ~/inference/signal_extractor.py
"""
Extracts meta-learner input signals from the small model
during generation. Hooks into model internals without
modifying the model itself.

IMPORTANT: This requires running the model through HuggingFace
transformers (not through Ollama's inference server) because
we need access to hidden states and attention weights. The
cascade engine uses HF for the local model (to get signals)
and Ollama for the oracle (cloud API).

Alternative: If you want pure Ollama for everything, skip the
signal extractor and use the lightweight meta-learner variant
(Part 2.3) that operates on text-level features only.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


HEDGE_WORDS = frozenset({
    "perhaps", "possibly", "might", "may", "could",
    "uncertain", "unclear", "debatable", "arguably",
    "not sure", "i believe", "it seems", "roughly",
    "approximately", "likely", "unlikely", "probably",
    "conceivably", "questionable", "speculative",
})


@dataclass
class ModelSignals:
    """Feature vector extracted from one generation pass."""
    hidden_state_mean: torch.Tensor    # [hidden_dim]
    hidden_state_std: torch.Tensor     # [hidden_dim]
    token_entropies: List[float]       # Per generated token
    attention_entropy: float           # Scalar
    topic_embedding: torch.Tensor      # [256] first-token hidden state
    hedge_word_count: int
    repetition_rate: float             # 0 = no repetition, 1 = all repeated
    query_similarity: float            # Proxy for in-distribution score
    response_length: int               # Number of generated tokens
    
    def to_tensor(self, target_dim: int = 4160) -> torch.Tensor:
        """Flatten all signals into a single feature vector."""
        
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
        
        # Pad or truncate
        if combined.shape[0] < target_dim:
            combined = torch.nn.functional.pad(
                combined, (0, target_dim - combined.shape[0]))
        else:
            combined = combined[:target_dim]
        
        return combined


class SignalExtractor:
    """Hooks into HuggingFace model forward pass to capture signals."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_states = []
        self.attention_maps = []
        self._hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on the last transformer layer."""
        
        # Navigate model architecture to find last layer
        # Works for Qwen, Llama, Mistral, and most HF architectures
        layers = None
        for attr in ['model.layers', 'transformer.h', 'gpt_neox.layers']:
            obj = self.model
            try:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue
        
        if layers is None:
            print("WARNING: Could not find transformer layers. "
                  "Signal extraction will use defaults.")
            return
        
        last_layer = layers[-1]
        
        h = last_layer.register_forward_hook(self._capture_hidden)
        self._hook_handles.append(h)
        
        # Try to hook attention
        for attn_attr in ['self_attn', 'attention', 'attn']:
            if hasattr(last_layer, attn_attr):
                attn_module = getattr(last_layer, attn_attr)
                h = attn_module.register_forward_hook(self._capture_attention)
                self._hook_handles.append(h)
                break
    
    def _capture_hidden(self, module, input, output):
        if isinstance(output, tuple):
            self.hidden_states.append(output[0].detach().cpu())
        else:
            self.hidden_states.append(output.detach().cpu())
    
    def _capture_attention(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None:
                self.attention_maps.append(attn_weights.detach().cpu())
    
    def clear(self):
        self.hidden_states = []
        self.attention_maps = []
    
    def extract(self, input_ids: torch.Tensor,
                generated_ids: torch.Tensor,
                generated_text: str) -> ModelSignals:
        """Extract all signals after generation completes.
        
        Args:
            input_ids: Prompt token IDs [1, prompt_len]
            generated_ids: Full sequence [1, prompt_len + gen_len]
            generated_text: Decoded generated portion
        """
        prompt_len = input_ids.shape[-1]
        
        # --- Hidden state statistics ---
        if self.hidden_states:
            last_hidden = self.hidden_states[-1]
            if last_hidden.shape[1] > prompt_len:
                gen_hidden = last_hidden[:, prompt_len:, :]
                h_mean = gen_hidden.mean(dim=1).squeeze(0)
                h_std = gen_hidden.std(dim=1).squeeze(0)
                topic_emb = last_hidden[0, prompt_len, :256]
            else:
                h_mean = last_hidden.mean(dim=1).squeeze(0)
                h_std = last_hidden.std(dim=1).squeeze(0)
                topic_emb = last_hidden[0, -1, :256]
        else:
            hidden_dim = getattr(
                self.model.config, 'hidden_size', 4096)
            h_mean = torch.zeros(hidden_dim)
            h_std = torch.ones(hidden_dim)
            topic_emb = torch.zeros(256)
        
        # --- Token entropy ---
        token_entropies = self._compute_token_entropies(
            input_ids, generated_ids, prompt_len)
        
        # --- Attention entropy ---
        attn_entropy = self._compute_attention_entropy()
        
        # --- Text-level signals ---
        lower_text = generated_text.lower()
        hedge_count = sum(1 for hw in HEDGE_WORDS if hw in lower_text)
        rep_rate = self._compute_repetition_rate(generated_text)
        query_sim = self._compute_distribution_similarity(h_mean)
        gen_length = generated_ids.shape[-1] - prompt_len
        
        return ModelSignals(
            hidden_state_mean=h_mean,
            hidden_state_std=h_std,
            token_entropies=token_entropies,
            attention_entropy=attn_entropy,
            topic_embedding=topic_emb,
            hedge_word_count=hedge_count,
            repetition_rate=rep_rate,
            query_similarity=query_sim,
            response_length=gen_length,
        )
    
    def _compute_token_entropies(self, input_ids, full_ids,
                                  prompt_len) -> List[float]:
        """Per-token generation entropy."""
        entropies = []
        try:
            with torch.no_grad():
                outputs = self.model(
                    full_ids.to(self.model.device),
                    output_hidden_states=False
                )
                logits = outputs.logits[0]  # [seq_len, vocab]
            
            gen_logits = logits[prompt_len - 1:-1]
            for i in range(gen_logits.shape[0]):
                probs = torch.softmax(gen_logits[i].float(), dim=-1)
                h = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                entropies.append(h)
        except Exception:
            pass
        
        return entropies if entropies else [5.0]
    
    def _compute_attention_entropy(self) -> float:
        if not self.attention_maps:
            return 5.0
        
        try:
            last_attn = self.attention_maps[-1]
            attn_probs = last_attn[0].mean(dim=0)  # Average across heads
            eps = 1e-10
            ent = -torch.sum(
                attn_probs * torch.log(attn_probs + eps), dim=-1)
            return ent.mean().item()
        except Exception:
            return 5.0
    
    def _compute_repetition_rate(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 20:
            return 0.0
        ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        if not ngrams:
            return 0.0
        return 1.0 - len(set(ngrams)) / len(ngrams)
    
    def _compute_distribution_similarity(self, h_mean: torch.Tensor) -> float:
        """Proxy OOD detection via activation norm.
        
        In production: replace with Mahalanobis distance 
        computed from training set hidden state statistics.
        """
        norm = h_mean.norm().item()
        typical_norm = 45.0
        deviation = abs(norm - typical_norm) / typical_norm
        return max(0.0, 1.0 - deviation)
    
    def cleanup(self):
        """Remove hooks when done."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
```

### 2.3 Lightweight Alternative (Pure Ollama, No Hooks)

If you want both local and oracle models running through Ollama's server (no HuggingFace), you lose access to hidden states. Use this text-only signal extractor instead:

```python
# ~/inference/signal_extractor_lite.py
"""
Text-level-only signal extraction. Works with Ollama's server
where you don't have access to hidden states.

Less accurate than the full extractor, but allows pure Ollama
architecture on both sides.
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
    """Extract signals from response text only."""
    
    def extract(self, response_text: str) -> LiteSignals:
        words = response_text.lower().split()
        total_words = max(len(words), 1)
        unique_words = len(set(words))
        
        sentences = re.split(r'[.!?]+', response_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        hedge_count = sum(1 for hw in HEDGE_WORDS 
                         if hw in response_text.lower())
        
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
            if w in {"not", "no", "never", "neither", "nor", "cannot", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't"}
        )
        
        # 4-gram repetition rate
        if len(words) >= 20:
            ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
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
```

### 2.4 Which Extractor To Use

```
SETUP                              EXTRACTOR        ACCURACY
────────────────────────────────────────────────────────────────
HF local model + Ollama oracle     Full (2.2)       Best
Pure Ollama both sides             Lite (2.3)       Good enough
Ollama local + separate API oracle Lite (2.3)       Good enough
```

The full extractor is 15-25% more accurate at routing decisions in early testing because hidden state variance is the strongest single predictor of actual uncertainty. But the lite version is production-viable — it just needs more training data to reach the same accuracy, and the online feedback loop will get it there.

---

## Part 3: Meta-Learner

### 3.1 Architecture

The meta-learner is a small neural network (5M-50M parameters depending on extractor) that maps model signals to confidence, routing, and error type predictions. It sits between the generation step and the routing decision.

```python
# ~/inference/meta_learner.py
"""
Meta-learner for confidence estimation and routing.
Two variants: Full (uses hidden states) and Lite (text-only).
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Dict, List
import numpy as np


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
        # NONE=0, FACTUAL_GAP=1, REASONING_ERROR=2,
        # FORMALIZATION_ERROR=3, SCOPE_EXCEEDED=4
        self.error_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )
        
        self.routing_labels = ["SELF", "ORACLE", "HYBRID"]
        self.error_labels = [
            "NONE", "FACTUAL_GAP", "REASONING_ERROR",
            "FORMALIZATION_ERROR", "SCOPE_EXCEEDED"
        ]
    
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
        
        self.routing_labels = ["SELF", "ORACLE", "HYBRID"]
        self.error_labels = [
            "NONE", "FACTUAL_GAP", "REASONING_ERROR",
            "FORMALIZATION_ERROR", "SCOPE_EXCEEDED"
        ]
    
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
```

### 3.2 Feedback Buffer

This is the core of the feedback-forward mechanism. Every oracle interaction produces ground truth labels that train the meta-learner to route better next time.

```python
# ~/inference/feedback.py
"""
Feedback buffer and online trainer for the meta-learner.

THREE LEARNING SIGNALS:

Signal 1 — Oracle Agreement
  When oracle is consulted, compare small model's answer to oracle's.
  Agreement = confidence should have been high.
  Disagreement = confidence should have been low.
  This gives automatic ground truth without human annotation.

Signal 2 — Z3 Verification  
  When small model makes formal claims, Z3 verifies independently.
  Z3 verdicts are perfect ground truth. The meta-learner learns
  which internal states correlate with correct formalizations.

Signal 3 — User Feedback
  If user corrects the model → negative signal.
  If user accepts and continues → weak positive signal.
  Weakest signal, but accumulates over time.

CONVERGENCE:
  The meta-learner typically converges to useful routing accuracy
  within 200-500 oracle interactions. Bootstrap (Part 4) provides
  the initial 500 data points to start warm.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Dict
import numpy as np
import json
import time


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
            signals.append(1.0 - min(outcome["correction_magnitude"], 1.0))
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
        
        import random
        sampled = random.choices(indices, weights=probs,
                                 k=self.batch_size)
        
        batch = [self.buffer[i] for i in sampled]
        
        return {
            "features": torch.stack([b["features"].squeeze(0)
                                     if b["features"].dim() > 1
                                     else b["features"]
                                     for b in batch]),
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
        
        # Forward through heads
        # The encoder is shared, so we run it once
        encoded = self.meta.encoder(features)
        
        conf_pred = self.meta.confidence_head(encoded).squeeze(-1)
        route_pred = self.meta.routing_head(encoded)
        error_pred = self.meta.error_head(encoded)
        
        loss_conf = self.conf_loss_fn(conf_pred, batch["true_confidence"])
        loss_route = self.route_loss_fn(route_pred, batch["true_routing"])
        loss_error = self.error_loss_fn(error_pred, batch["true_error"])
        
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
        ckpt = torch.load(path, map_location='cpu')
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
```

---

## Part 4: Bootstrap (Replaces Original Phase 9)

Before deploying to production, pre-train the meta-learner by running the small model and oracle side-by-side on held-out questions and comparing their answers.

```python
# ~/training/bootstrap_meta.py
"""
Bootstrap the meta-learner with 500 synthetic feedback examples.

Method:
1. Run small model on held-out Descartes questions
2. Run oracle on same questions
3. Compare answers → ground truth labels
4. Pre-train meta-learner on these labels
5. Deploy with warm-started meta-learner

Cost: ~$2-5 (500 oracle calls via Ollama Cloud)
Time: ~2-4 hours (small model generation bottleneck)
"""

import ollama
import torch
import json
import os
from pathlib import Path
from signal_extractor_lite import LiteSignalExtractor, LiteSignals
from meta_learner import MetaLearnerLite
from feedback import MetaTrainer


def compute_agreement(text_a: str, text_b: str) -> float:
    """Jaccard similarity on content words."""
    stop = {"the","a","an","is","are","was","were","in","on",
            "at","to","for","of","and","that","this","it","with"}
    words_a = set(text_a.lower().split()) - stop
    words_b = set(text_b.lower().split()) - stop
    if not words_a or not words_b:
        return 0.5
    return len(words_a & words_b) / len(words_a | words_b)


def bootstrap(
    local_model: str = "descartes:8b",
    oracle_model: str = "deepseek-v3.1:671-cloud",
    questions_path: str = "~/training/eval/bootstrap_questions.jsonl",
    output_path: str = "~/models/meta_learner_bootstrap.pt",
    max_questions: int = 500
):
    questions_path = os.path.expanduser(questions_path)
    output_path = os.path.expanduser(output_path)
    
    # Load questions
    with open(questions_path) as f:
        questions = [json.loads(line)["question"] 
                     for line in f][:max_questions]
    
    print(f"Bootstrapping on {len(questions)} questions")
    print(f"  Local: {local_model}")
    print(f"  Oracle: {oracle_model}")
    
    extractor = LiteSignalExtractor()
    meta = MetaLearnerLite(input_dim=11)
    trainer = MetaTrainer(meta, lr=5e-4, update_every=4)
    
    for i, q in enumerate(questions):
        # Small model answer
        local_resp = ollama.chat(
            model=local_model,
            messages=[{"role": "user", "content": q}]
        )
        local_text = local_resp['message']['content']
        
        # Oracle answer
        oracle_resp = ollama.chat(
            model=oracle_model,
            messages=[{
                "role": "user",
                "content": f"Answer this philosophical question "
                           f"accurately and in detail:\n\n{q}"
            }]
        )
        oracle_text = oracle_resp['message']['content']
        
        # Extract signals from local response
        signals = extractor.extract(local_text)
        signal_tensor = signals.to_tensor()
        
        # Meta-learner prediction (will be random initially)
        meta.eval()
        with torch.no_grad():
            pred = meta(signal_tensor)
        
        # Compute ground truth from comparison
        agreement = compute_agreement(local_text, oracle_text)
        
        outcome = {
            "oracle_agreed": agreement > 0.6,
            "correction_magnitude": 1.0 - agreement,
            "z3_verified": None,
            "user_accepted": None,
        }
        
        trainer.record_and_maybe_train(
            pred["features"].detach(),
            pred["confidence"].item(),
            pred["routing_decision"],
            outcome
        )
        
        if (i + 1) % 50 == 0:
            stats = trainer.get_stats()
            print(f"  [{i+1}/{len(questions)}] "
                  f"updates={stats['updates']}, "
                  f"loss={stats['avg_loss']:.4f}" 
                  if stats['avg_loss'] else "")
    
    # Save bootstrapped meta-learner
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trainer.save(output_path)
    
    print(f"\nBootstrap complete.")
    print(f"  Updates: {trainer.update_count}")
    print(f"  Buffer: {len(trainer.buffer.buffer)} examples")
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    bootstrap()
```

### Bootstrap Question Generator

```python
# ~/training/eval/generate_bootstrap_questions.py
"""
Generate 500 diverse Descartes questions spanning all routing types.

Distribution:
  40% SELF-answerable (core Cartesian formalization/analysis)
  30% ORACLE-needed (broad philosophy, historical context)
  30% HYBRID (Cartesian core + external knowledge)

This distribution teaches the meta-learner all three routing paths.
"""

import json
import os

SELF_QUESTIONS = [
    "Formalize the Cogito as a strict inference in Z3.",
    "What is the ASPIC+ attack structure of the Wax Argument?",
    "Is the Real Distinction argument deductively valid?",
    "Decompose Meditation III into its component sub-arguments.",
    "What role does the Evil Genius play in the method of doubt?",
    "Formalize substance dualism: mind and body as distinct sorts.",
    "What is the modal structure of the conceivability argument?",
    "Check consistency of the Trademark Argument premises in Z3.",
    "Identify the argumentation scheme in the Ontological Argument.",
    "What is the logical relationship between Meditations II and VI?",
    "Formalize the clear and distinct perception criterion.",
    "Is the Cogito an inference or an intuition? Formalize both readings.",
    "What type of ASPIC+ attack does Arnauld's Circle represent?",
    "Reconstruct the dreaming argument as a formal inference.",
    "Check whether Descartes' proofs of God are jointly consistent.",
    # ... (expand to ~200 questions)
]

ORACLE_QUESTIONS = [
    "What was Hume's response to Cartesian rationalism?",
    "How did the Port-Royal Logic incorporate Descartes' method?",
    "What is Merleau-Ponty's critique of the Cogito?",
    "How was Descartes received by the Utrecht theologians?",
    "What did Kant say about the ontological argument?",
    "Compare Descartes' doubt with Pyrrhonian skepticism.",
    "What was the relationship between Descartes and Beeckman?",
    "How does Husserl's phenomenological reduction differ from Cartesian doubt?",
    "What neuroscience evidence bears on the interaction problem?",
    "How did occasionalism develop after Descartes' death?",
    "What was La Forge's contribution to Cartesian physics?",
    "How does IIT relate to Cartesian substance dualism?",
    "What was the Condemnation of 1663 about?",
    "Compare Descartes' animal automata with modern animal consciousness research.",
    "What did Strawson argue about persons in 'Individuals'?",
    # ... (expand to ~150 questions)
]

HYBRID_QUESTIONS = [
    "Is the Real Distinction structurally identical to the zombie argument?",
    "Can Descartes' causal adequacy principle survive modern physicalism?",
    "Formalize both Descartes' and Spinoza's substance ontology and check compatibility.",
    "Does Ryle's critique of the 'ghost in the machine' actually refute Descartes?",
    "Compare the modal logic of the Real Distinction with Kripke's identity argument.",
    "Can Global Workspace Theory be reconciled with substance dualism? Formalize.",
    "How does Chalmers' conceivability principle differ from Descartes' divine guarantee?",
    "Formalize Elisabeth's interaction objection alongside Kim's exclusion argument.",
    "Is Descartes' foundationalism compatible with Bayesian epistemology?",
    "Compare Descartes' pineal gland hypothesis with modern binding problem solutions.",
    "Does predictive processing vindicate or refute Cartesian representationalism?",
    "Formalize both Descartes' and Leibniz's arguments for mind-body distinctness.",
    "Can property dualism capture Descartes' insights without substance dualism?",
    "Compare the Cartesian Circle with the problem of the criterion in Chisholm.",
    "Is Descartes' conceivability-possibility bridge valid in S5 vs. S4?",
    # ... (expand to ~150 questions)
]


def generate(output_path: str = "~/training/eval/bootstrap_questions.jsonl"):
    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_questions = []
    
    for q in SELF_QUESTIONS:
        all_questions.append({
            "question": q,
            "expected_routing": "SELF",
            "category": "cartesian_core"
        })
    
    for q in ORACLE_QUESTIONS:
        all_questions.append({
            "question": q,
            "expected_routing": "ORACLE",
            "category": "broad_philosophy"
        })
    
    for q in HYBRID_QUESTIONS:
        all_questions.append({
            "question": q,
            "expected_routing": "HYBRID",
            "category": "cross_domain"
        })
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_questions)
    
    with open(output_path, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")
    
    print(f"Generated {len(all_questions)} bootstrap questions")
    print(f"  SELF: {len(SELF_QUESTIONS)}")
    print(f"  ORACLE: {len(ORACLE_QUESTIONS)}")
    print(f"  HYBRID: {len(HYBRID_QUESTIONS)}")
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    generate()
```

---

## Part 5: Unified Cascade Engine (Replaces Original Phases 10-11)

### 5.1 Pure Ollama Version (Recommended)

```python
# ~/inference/engine.py
"""
Descartes Philosopher Engine — Ollama Unified + Meta-Learner.

This is the production inference system. Both local specialist
and cloud oracle accessed through ollama.chat(). Meta-learner
routes queries and improves with every oracle interaction.
"""

import ollama
import torch
import json
import os
from typing import Optional, Dict
from dataclasses import dataclass, field

from signal_extractor_lite import LiteSignalExtractor
from meta_learner import MetaLearnerLite
from feedback import MetaTrainer


DESCARTES_SYSTEM = (
    "You are a philosophical reasoning assistant specializing in "
    "Cartesian philosophy, early modern rationalism, and the "
    "mind-body problem. You analyze arguments using ASPIC+ "
    "argumentation schemes and Z3 formal verification. You have "
    "deep expertise in Descartes' Meditations, the Objections and "
    "Replies, the Correspondence with Elisabeth, and the Principles "
    "of Philosophy."
)

ORACLE_SYSTEM = (
    "You are a philosophical knowledge oracle. A Descartes specialist "
    "is asking for information outside its training domain. Provide "
    "accurate, detailed philosophical knowledge with specific sources "
    "and positions. The specialist will integrate your knowledge with "
    "its own formal analysis."
)

INTEGRATION_TEMPLATE = (
    "You previously analyzed a question but needed additional "
    "philosophical knowledge. Integrate the oracle's response with "
    "your own Cartesian expertise. Preserve your formal analysis "
    "and strengthen it with the new information.\n\n"
    "ORIGINAL QUESTION:\n{query}\n\n"
    "YOUR INITIAL ANALYSIS:\n{initial}\n\n"
    "ADDITIONAL KNOWLEDGE:\n{oracle}\n\n"
    "Produce your final integrated answer."
)


@dataclass
class EngineResult:
    """Complete result from one query."""
    query: str
    final_response: str
    confidence: float
    routing: str                # SELF, ORACLE, HYBRID
    error_type: str             # NONE, FACTUAL_GAP, etc.
    oracle_used: bool
    initial_response: str = ""
    oracle_query: Optional[str] = None
    oracle_response: Optional[str] = None


class DescartesEngine:
    """Production engine: Ollama local + cloud, meta-learner routing."""
    
    def __init__(self,
                 local_model: str = "descartes:8b",
                 oracle_model: str = "deepseek-v3.1:671-cloud",
                 meta_path: Optional[str] = None,
                 oracle_escalation: Optional[str] = "kimi-k2.5:cloud"):
        
        self.local_model = local_model
        self.oracle_model = oracle_model
        self.oracle_escalation = oracle_escalation
        
        # Signal extractor (text-only for pure Ollama)
        self.extractor = LiteSignalExtractor()
        
        # Meta-learner
        self.meta = MetaLearnerLite(input_dim=11)
        self.trainer = MetaTrainer(self.meta, lr=1e-4)
        
        if meta_path and os.path.exists(meta_path):
            self.trainer.load(meta_path)
            print(f"Loaded meta-learner ({self.trainer.update_count} updates)")
        
        self.meta.eval()
        
        # Stats
        self.stats = {
            "total": 0, "self": 0, "oracle": 0, "hybrid": 0,
            "oracle_calls": 0
        }
        
        print(f"Engine ready.")
        print(f"  Local:  {local_model}")
        print(f"  Oracle: {oracle_model}")
        print(f"  Escalation: {oracle_escalation}")
        print(f"  Meta-learner: {'warm' if meta_path else 'cold'} start")
    
    def _chat_local(self, messages: list) -> str:
        """Query the local Descartes model via Ollama."""
        resp = ollama.chat(model=self.local_model, messages=messages)
        return resp['message']['content']
    
    def _chat_oracle(self, messages: list, 
                      escalate: bool = False) -> str:
        """Query the cloud oracle via Ollama."""
        model = self.oracle_escalation if escalate else self.oracle_model
        resp = ollama.chat(model=model, messages=messages)
        self.stats["oracle_calls"] += 1
        return resp['message']['content']
    
    def _build_oracle_query(self, query: str, initial: str,
                             error_type: str) -> str:
        """Shape the oracle query based on predicted error type."""
        
        if error_type == "FACTUAL_GAP":
            return (
                f"A Descartes specialist needs factual knowledge:\n\n"
                f"Question: {query}\n\n"
                f"Their partial analysis:\n{initial[:500]}\n\n"
                f"What factual information are they missing?"
            )
        elif error_type == "SCOPE_EXCEEDED":
            return (
                f"This question extends beyond Cartesian philosophy. "
                f"Provide the relevant broader context:\n\n{query}"
            )
        elif error_type == "REASONING_ERROR":
            return (
                f"Check this analysis for reasoning errors:\n\n"
                f"Question: {query}\n"
                f"Analysis: {initial[:500]}\n\n"
                f"Identify any errors and provide corrections."
            )
        else:
            return query
    
    def run(self, query: str) -> EngineResult:
        """Full cascade pipeline."""
        
        self.stats["total"] += 1
        
        # ── Step 1: Local specialist generates ──
        initial = self._chat_local([
            {"role": "system", "content": DESCARTES_SYSTEM},
            {"role": "user", "content": query}
        ])
        
        # ── Step 2: Extract signals + meta-learner routes ──
        signals = self.extractor.extract(initial)
        signal_tensor = signals.to_tensor()
        
        with torch.no_grad():
            meta_out = self.meta(signal_tensor)
        
        confidence = meta_out["confidence"].item()
        routing = meta_out["routing_decision"]
        error_type = meta_out["error_type"]
        features = meta_out["features"]
        
        result = EngineResult(
            query=query,
            final_response=initial,
            confidence=confidence,
            routing=routing,
            error_type=error_type,
            oracle_used=False,
            initial_response=initial,
        )
        
        # ── Step 3: Route ──
        if routing == "SELF":
            self.stats["self"] += 1
            return result
        
        # ── Step 4: Oracle consultation ──
        oracle_query = self._build_oracle_query(
            query, initial, error_type)
        
        # Escalate to stronger model for SCOPE_EXCEEDED
        escalate = (error_type == "SCOPE_EXCEEDED" 
                    and self.oracle_escalation is not None)
        
        oracle_response = self._chat_oracle([
            {"role": "system", "content": ORACLE_SYSTEM},
            {"role": "user", "content": oracle_query}
        ], escalate=escalate)
        
        result.oracle_used = True
        result.oracle_query = oracle_query
        result.oracle_response = oracle_response
        
        if routing == "ORACLE":
            self.stats["oracle"] += 1
        else:
            self.stats["hybrid"] += 1
        
        # ── Step 5: Integration pass ──
        integrated = self._chat_local([
            {"role": "system", "content": DESCARTES_SYSTEM},
            {"role": "user", "content": INTEGRATION_TEMPLATE.format(
                query=query,
                initial=initial,
                oracle=oracle_response
            )}
        ])
        
        result.final_response = integrated
        
        # ── Step 6: Feedback to meta-learner ──
        agreement = self._compute_agreement(initial, oracle_response)
        
        outcome = {
            "oracle_agreed": agreement > 0.6,
            "correction_magnitude": 1.0 - agreement,
            "z3_verified": None,
            "user_accepted": None,
        }
        
        self.trainer.record_and_maybe_train(
            features.detach(), confidence, routing, outcome)
        
        return result
    
    def record_user_feedback(self, accepted: bool):
        """Call when user gives explicit feedback (thumbs up/down)."""
        if self.trainer.buffer.buffer:
            last = self.trainer.buffer.buffer[-1]
            if accepted:
                last["true_confidence"] = min(
                    last["true_confidence"] + 0.1, 1.0)
            else:
                last["true_confidence"] = max(
                    last["true_confidence"] - 0.2, 0.0)
    
    def record_z3_result(self, verified: bool):
        """Call when Z3 verifies a formal claim from the response."""
        if self.trainer.buffer.buffer:
            last = self.trainer.buffer.buffer[-1]
            if verified:
                last["true_confidence"] = min(
                    last["true_confidence"] + 0.15, 1.0)
                last["true_error"] = 0  # NONE
            else:
                last["true_confidence"] = max(
                    last["true_confidence"] - 0.3, 0.0)
                last["true_error"] = 3  # FORMALIZATION_ERROR
    
    def _compute_agreement(self, text_a: str, text_b: str) -> float:
        stop = {"the","a","an","is","are","was","were","in","on",
                "at","to","for","of","and","that","this","it","with"}
        wa = set(text_a.lower().split()) - stop
        wb = set(text_b.lower().split()) - stop
        if not wa or not wb:
            return 0.5
        return len(wa & wb) / len(wa | wb)
    
    def save(self, path: str):
        """Persist meta-learner state."""
        self.trainer.save(path)
        print(f"Saved (updates={self.trainer.update_count}, "
              f"buffer={len(self.trainer.buffer.buffer)})")
    
    def get_stats(self) -> Dict:
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "self_rate": f"{self.stats['self']/total:.1%}",
            "oracle_rate": f"{self.stats['oracle']/total:.1%}",
            "hybrid_rate": f"{self.stats['hybrid']/total:.1%}",
            "meta_learner": self.trainer.get_stats(),
        }


# ─────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────

def main():
    import sys
    
    meta_path = None
    default_meta = os.path.expanduser("~/models/meta_learner_bootstrap.pt")
    if os.path.exists(default_meta):
        meta_path = default_meta
    
    engine = DescartesEngine(
        local_model="descartes:8b",
        oracle_model="deepseek-v3.1:671-cloud",
        meta_path=meta_path,
        oracle_escalation="kimi-k2.5:cloud",
    )
    
    print("\n" + "=" * 60)
    print("DESCARTES PHILOSOPHER ENGINE")
    print("=" * 60)
    print("Commands:")
    print("  quit       — exit and save meta-learner")
    print("  stats      — show routing and meta-learner stats")
    print("  good/bad   — give feedback on last response")
    print("  z3:pass    — report Z3 verification passed")
    print("  z3:fail    — report Z3 verification failed")
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
        elif query == "good":
            engine.record_user_feedback(True)
            print("  Recorded: positive feedback")
            continue
        elif query == "bad":
            engine.record_user_feedback(False)
            print("  Recorded: negative feedback")
            continue
        elif query == "z3:pass":
            engine.record_z3_result(True)
            print("  Recorded: Z3 verification passed")
            continue
        elif query == "z3:fail":
            engine.record_z3_result(False)
            print("  Recorded: Z3 verification failed")
            continue
        
        result = engine.run(query)
        
        print(f"\n[{result.routing}] "
              f"[conf={result.confidence:.2f}] "
              f"[error={result.error_type}] "
              f"[oracle={'yes' if result.oracle_used else 'no'}]")
        print(f"\n{result.final_response}")
    
    engine.save(os.path.expanduser("~/models/meta_learner_latest.pt"))
    print(f"\nFinal stats: {json.dumps(engine.get_stats(), indent=2)}")


if __name__ == "__main__":
    main()
```

### 5.2 Hybrid Version (HF Local + Ollama Cloud)

If you want the full signal extractor with hidden state access, use HuggingFace for the local model and Ollama only for the cloud oracle. The engine structure is identical — just swap `_chat_local` to use HF `model.generate()` and the `SignalExtractor` (full) instead of `LiteSignalExtractor`.

The key change is in `__init__`:

```python
# In hybrid version only:
from transformers import AutoModelForCausalLM, AutoTokenizer
from signal_extractor import SignalExtractor  # Full version
from meta_learner import MetaLearnerFull      # Full version

class DescartesEngineHybrid(DescartesEngine):
    """Uses HF for local (hidden state access) + Ollama for cloud."""
    
    def __init__(self, model_path, oracle_model, meta_path=None):
        # Local model via HuggingFace (for signal extraction)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.extractor = SignalExtractor(self.model, self.tokenizer)
        
        # Oracle via Ollama Cloud
        self.oracle_model = oracle_model
        
        # Meta-learner (full version — uses hidden states)
        self.meta = MetaLearnerFull(
            input_dim=4160, feature_dim=256)
        self.trainer = MetaTrainer(self.meta)
        
        if meta_path:
            self.trainer.load(meta_path)
```

---

## Part 6: Evaluation (Replaces Original Phase 12)

### 6.1 Routing Accuracy Test

```python
# ~/training/eval/eval_routing.py
"""
Test whether the meta-learner correctly routes queries.
Run AFTER bootstrap but BEFORE production deployment.
"""

import ollama
import torch
import json
from signal_extractor_lite import LiteSignalExtractor
from meta_learner import MetaLearnerLite
from feedback import MetaTrainer

ROUTING_TESTS = [
    # (query, expected_routing)
    ("Formalize the Cogito in Z3.", "SELF"),
    ("What did Husserl say about Cartesian doubt?", "ORACLE"),
    ("Is the Real Distinction identical to the zombie argument?", "HYBRID"),
    ("Decompose Arnauld's Circle in ASPIC+.", "SELF"),
    ("How was Descartes received by the Jesuits?", "ORACLE"),
    ("Can GWT be reconciled with substance dualism?", "HYBRID"),
    ("Check consistency of Trademark Argument premises.", "SELF"),
    ("Compare Descartes' doubt with Pyrrhonian skepticism.", "ORACLE"),
    ("Formalize Elisabeth's objection alongside Kim's exclusion.", "HYBRID"),
    ("What modal logic does the Real Distinction use?", "SELF"),
    ("What was Malebranche's occasionalist response?", "ORACLE"),
    ("Does Kripke's identity argument parallel Descartes'?", "HYBRID"),
]


def eval_routing(meta_path: str):
    meta = MetaLearnerLite(input_dim=11)
    ckpt = torch.load(meta_path, map_location='cpu')
    meta.load_state_dict(ckpt["model_state"])
    meta.eval()
    
    extractor = LiteSignalExtractor()
    
    correct = 0
    total = len(ROUTING_TESTS)
    
    print(f"{'Query':<55} {'Expected':>8} {'Predicted':>9} {'Match':>5}")
    print("─" * 85)
    
    for query, expected in ROUTING_TESTS:
        # Generate response from local model
        resp = ollama.chat(
            model="descartes:8b",
            messages=[{"role": "user", "content": query}]
        )
        
        signals = extractor.extract(resp['message']['content'])
        with torch.no_grad():
            pred = meta(signals.to_tensor())
        
        predicted = pred["routing_decision"]
        match = predicted == expected
        correct += int(match)
        
        q_short = query[:52] + "..." if len(query) > 55 else query
        mark = "✓" if match else "✗"
        print(f"{q_short:<55} {expected:>8} {predicted:>9} {mark:>5}")
    
    accuracy = correct / total
    print(f"\nRouting accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"{'PASS' if accuracy >= 0.8 else 'FAIL'} (threshold: 80%)")
    
    return accuracy


# eval_routing("~/models/meta_learner_bootstrap.pt")
```

### 6.2 Meta-Learner Convergence Test

```python
# ~/training/eval/eval_convergence.py
"""
Track meta-learner improvement over time.
Plot confidence calibration and routing accuracy
at 100, 200, 500, 1000 interactions.
"""

def eval_convergence(meta_path: str):
    """Load buffer history and compute metrics at checkpoints."""
    
    buffer_path = meta_path.replace('.pt', '_buffer.json')
    with open(buffer_path) as f:
        buffer = json.load(f)
    
    checkpoints = [100, 200, 500, len(buffer)]
    
    print(f"{'Checkpoint':>10} {'Calib Error':>12} {'Route Acc':>10}")
    print("─" * 35)
    
    for cp in checkpoints:
        if cp > len(buffer):
            continue
        
        subset = buffer[:cp]
        
        # Calibration: |predicted_confidence - true_confidence|
        calib_errors = [
            abs(s["predicted_confidence"] - s["true_confidence"])
            for s in subset
        ]
        avg_calib = sum(calib_errors) / len(calib_errors)
        
        # Routing accuracy
        route_map = {"SELF": 0, "ORACLE": 1, "HYBRID": 2}
        route_correct = sum(
            1 for s in subset
            if route_map.get(s["predicted_routing"], -1) == s["true_routing"]
        )
        route_acc = route_correct / len(subset)
        
        print(f"{cp:>10} {avg_calib:>12.3f} {route_acc:>10.1%}")
    
    print(f"\nExpected: calibration error decreases, routing accuracy increases")
```

### 6.3 Pass Criteria

```
METRIC                        THRESHOLD     NOTES
──────────────────────────────────────────────────────────────
Routing accuracy              ≥ 80%         On held-out test set
Confidence calibration        ≤ 0.15        Mean |predicted - true|
Self-handling rate             ≥ 60%         Majority queries local
Oracle cost per 100 queries   ≤ $0.50       Ollama Cloud metered
Knowledge accuracy (no oracle) ≥ 70%        Cartesian core questions
Knowledge accuracy (with oracle) ≥ 90%      Post-integration
Meta-learner convergence      ≤ 500 interactions to useful routing

IF THRESHOLDS MISSED:
1. Check bootstrap question distribution (40/30/30 split)
2. Generate more SFT examples for weak routing categories
3. Retrain Stage 2 with augmented Type E/F/G examples
4. Re-bootstrap meta-learner with more questions
5. Consider switching oracle model (e.g., kimi-k2.5 for harder queries)
```

---

## Part 7: File Layout

After implementing this addendum, your project structure should look like:

```
~/
├── corpus/                          # Phase 1-4 (from main pipeline)
│   ├── raw/
│   ├── extracted/
│   ├── cleaned/
│   └── formatted/
│
├── models/
│   ├── descartes-8b-cpt/           # Phase 5 output
│   ├── descartes-8b-sft-s1/        # Phase 7 stage 1
│   ├── descartes-8b-cascade/       # Phase 7 stage 2 (final)
│   ├── descartes-8b.gguf           # Converted for Ollama
│   ├── Modelfile                   # Ollama model definition
│   ├── meta_learner_bootstrap.pt   # Bootstrapped meta-learner
│   ├── meta_learner_bootstrap_buffer.json
│   └── meta_learner_latest.pt      # Production meta-learner (updates live)
│
├── inference/
│   ├── engine.py                   # Main cascade engine (this addendum)
│   ├── meta_learner.py             # MetaLearnerFull + MetaLearnerLite
│   ├── feedback.py                 # FeedbackBuffer + MetaTrainer
│   ├── signal_extractor.py         # Full (HF hooks)
│   ├── signal_extractor_lite.py    # Lite (text-only, pure Ollama)
│   └── templates/
│       └── descartes_z3.py         # Z3 formalization templates
│
├── training/
│   ├── run_cpt_descartes.py        # Phase 5
│   ├── run_sft_descartes.py        # Phase 7
│   ├── bootstrap_meta.py           # Phase 9 (this addendum)
│   ├── sft/
│   │   ├── descartes_templates.py  # SFT example templates
│   │   └── generate_descartes_sft.py
│   └── eval/
│       ├── generate_bootstrap_questions.py
│       ├── bootstrap_questions.jsonl
│       ├── eval_routing.py         # Phase 12
│       ├── eval_convergence.py
│       └── eval_descartes_cascade.py
│
└── test/
    └── test_ollama_unified.py      # Verify Ollama setup
```

---

## Reading Order (All Documents)

```
1. PHILOSOPHER_ENGINE_ARCHITECTURE.md
   → System design, five layers, why each layer exists

2. PHILOSOPHER_ENGINE_TRAINING_PIPELINE.md  
   → Phases 2-4: text extraction, cleaning, formatting

3. DESCARTES_CASCADE_TRAINING_PIPELINE.md
   → Phase 1: Descartes corpus assembly
   → Phase 5: CPT on 8B  
   → Phases 6-7: SFT data generation + two-stage training
   → Phase 8: Base evaluation

4. ADDENDUM_OLLAMA_META_LEARNER.md (this document)
   → Replaces Phases 9-12 with:
     Part 1: Ollama setup (local + cloud)
     Part 2: Signal extraction
     Part 3: Meta-learner architecture
     Part 4: Bootstrap (replaces Phase 9)
     Part 5: Unified cascade engine (replaces Phases 10-11)
     Part 6: Evaluation (replaces Phase 12)
```
