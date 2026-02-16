"""
Philosopher Engine — Inference Package

Cascade architecture components:
- cascade_engine: Main orchestration engine (DescartesEngine)
- meta_learner: Neural routing + confidence estimation (MetaLearner)
- signal_extractor: Model internal signal hooks (SignalExtractor)
- oracle: Large model API client (OracleClient)
- templates/: Z3 formalization templates
"""

from .cascade_engine import DescartesEngine
from .meta_learner import MetaLearner, FeedbackBuffer, MetaLearnerTrainer
from .signal_extractor import SignalExtractor
from .oracle import OracleClient, OracleConfig
