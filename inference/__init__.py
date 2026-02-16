"""
Philosopher Engine -- Inference Package

Cascade architecture components:

ORIGINAL (HF-based):
- cascade_engine: HuggingFace orchestration engine (DescartesEngine)
- meta_learner: Neural routing + confidence estimation (MetaLearner)
- signal_extractor: Model internal signal hooks (SignalExtractor)
- oracle: Large model API client (OracleClient)

OLLAMA ADDENDUM (pure Ollama, Addendum A):
- engine: Ollama unified cascade engine (DescartesEngine as OllamaEngine)
- meta_learner: Full + Lite meta-learner variants
- signal_extractor_lite: Text-only signal extraction (LiteSignalExtractor)
- feedback: FeedbackBuffer + MetaTrainer (online learning)

VKS + MULTI-TIER VERIFICATION (Addendum B):
- knowledge_store: Verified Knowledge Store (persistent, hash-chained)
- seed_axioms: Foundational Cartesian axioms (run once)
- claim_extractor: Claim splitting + type classification
- claim_router: Per-claim VKS routing to verification backends
- verifier: Z3 + corpus multi-tier verification
- self_repair: Local repair before oracle escalation
- engine_v3: Production engine with claim-level routing

- templates/: Z3 formalization templates
"""

# -- Original HF-based components --
from .cascade_engine import DescartesEngine
from .meta_learner import (
    MetaLearner, FeedbackBuffer, MetaLearnerTrainer,
    ModelSignals, ROUTING_LABELS, ERROR_LABELS,
)
from .signal_extractor import SignalExtractor
from .oracle import OracleClient, OracleConfig

# -- Ollama addendum components (Addendum A) --
from .meta_learner import MetaLearnerFull, MetaLearnerLite
from .signal_extractor_lite import LiteSignalExtractor, LiteSignals
from .feedback import FeedbackBuffer as FeedbackBufferV2, MetaTrainer
from .engine import (
    DescartesEngine as OllamaEngine,
    EngineResult,
)

# -- VKS + Multi-tier verification (Addendum B) --
from .knowledge_store import (
    VerifiedKnowledgeStore, ProofRecord, ProofStatus,
    Tier, VerificationMethod,
)
from .claim_extractor import ClaimExtractor, ExtractedClaim, ClaimType
from .claim_router import ClaimRouter
from .verifier import FormalVerifier, CorpusVerifier, VerificationResult
from .self_repair import SelfRepairEngine
from .engine_v3 import DescartesEngineV3, EngineResultV3
