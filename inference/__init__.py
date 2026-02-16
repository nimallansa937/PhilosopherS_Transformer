"""
Philosopher Engine — Inference Package

Cascade architecture components:

ORIGINAL (HF-based):
- cascade_engine: HuggingFace orchestration engine (DescartesEngine)
- meta_learner: Neural routing + confidence estimation (MetaLearner)
- signal_extractor: Model internal signal hooks (SignalExtractor)
- oracle: Large model API client (OracleClient)

OLLAMA ADDENDUM (pure Ollama):
- engine: Ollama unified cascade engine (DescartesEngine as OllamaEngine)
- meta_learner: Full + Lite meta-learner variants
- signal_extractor_lite: Text-only signal extraction (LiteSignalExtractor)
- feedback: FeedbackBuffer + MetaTrainer (online learning)

- templates/: Z3 formalization templates
"""

# ── Original HF-based components ──
from .cascade_engine import DescartesEngine
from .meta_learner import (
    MetaLearner, FeedbackBuffer, MetaLearnerTrainer,
    ModelSignals, ROUTING_LABELS, ERROR_LABELS,
)
from .signal_extractor import SignalExtractor
from .oracle import OracleClient, OracleConfig

# ── Ollama addendum components ──
from .meta_learner import MetaLearnerFull, MetaLearnerLite
from .signal_extractor_lite import LiteSignalExtractor, LiteSignals
from .feedback import FeedbackBuffer as FeedbackBufferV2, MetaTrainer
from .engine import (
    DescartesEngine as OllamaEngine,
    EngineResult,
)
