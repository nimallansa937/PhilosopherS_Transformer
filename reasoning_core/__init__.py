"""
Reasoning Core — Five-Layer Philosophical Reasoning Foundation.

Layer 1: Ontology (OWL 2 sorts, relations, theory commitments)
Layer 2: Argumentation (ASPIC+ knowledge base, Walton schemes)
Layer 3: Verification (Z3 modal/paraconsistent/defeasible, CVC5 parallel)
Layer 4: Bridge (GVR loop embedded in cascade engine)
Layer 5: Conceptual Spaces (geometric theory embeddings)

Cross-cutting: MicrotheoryManager (separate Z3 contexts per theory)
"""

from .ontology.core import OntologySorts, OntologyRelations
from .ontology.theories import TheoryCommitments, THEORY_NAMES
from .argumentation.aspic_engine import ASPICKnowledgeBase, AttackType, Rule
from .argumentation.walton_schemes import WaltonScheme, SCHEME_LIBRARY
from .verification.z3_engine import (
    ModalLogicEngine, ParaconsistentEngine,
    DefeasibleEngine, MicrotheoryManager,
)
from .spaces.conceptual_spaces import ConceptualSpace, TheoryEmbedding
