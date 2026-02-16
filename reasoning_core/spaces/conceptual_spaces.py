"""
Layer 5: Conceptual Spaces — Geometric Complement.

Embeds consciousness theories in a multi-dimensional space where
similarity = proximity. Handles graded judgments that resist
binary formalization ("how functionalist is IIT?").

Quality dimensions:
- Accessibility (can third-person science access it?)
- Integration (does it require information integration?)
- Higher-order (does consciousness require higher-order states?)
- Intentionality (is consciousness always about something?)
- Temporal (does it unfold over time?)
- Substrate (does the physical medium matter?)

14 theories of consciousness are embedded as vectors.
Distance metrics enable queries like "which theory is closest
to position X?" and "how similar are IIT and GWT?"

Reference: PHILOSOPHER_ENGINE_ARCHITECTURE.md, Layer 5
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Quality dimensions (0.0 = minimal, 1.0 = maximal)
DIMENSIONS = [
    "accessibility",      # Third-person accessibility of consciousness
    "integration",        # Requires information integration
    "higher_order",       # Requires higher-order representation
    "intentionality",     # Consciousness is always "about" something
    "temporal",           # Unfolds over time (vs. instantaneous)
    "substrate",          # Physical substrate matters (vs. multiple realizable)
]


@dataclass
class TheoryEmbedding:
    """A consciousness theory embedded as a vector in conceptual space."""
    name: str
    short_name: str
    vector: Dict[str, float]   # dimension_name -> value [0, 1]
    description: str = ""

    def as_list(self) -> List[float]:
        """Return vector as ordered list matching DIMENSIONS."""
        return [self.vector.get(d, 0.5) for d in DIMENSIONS]

    def __repr__(self):
        coords = ", ".join(f"{d[:3]}={v:.1f}"
                           for d, v in self.vector.items())
        return f"Theory({self.short_name}: {coords})"


# ============================================================
# PRE-DEFINED THEORY EMBEDDINGS
# ============================================================

THEORY_EMBEDDINGS: Dict[str, TheoryEmbedding] = {}


def _register(t: TheoryEmbedding):
    THEORY_EMBEDDINGS[t.short_name] = t
    return t


# Physicalism family
_register(TheoryEmbedding(
    "Type-A Physicalism", "TypeA",
    {"accessibility": 0.9, "integration": 0.3, "higher_order": 0.2,
     "intentionality": 0.4, "temporal": 0.5, "substrate": 0.9},
    "Consciousness = physical processes (identity theory)"
))

_register(TheoryEmbedding(
    "Type-B Physicalism", "TypeB",
    {"accessibility": 0.8, "integration": 0.4, "higher_order": 0.3,
     "intentionality": 0.5, "temporal": 0.5, "substrate": 0.8},
    "Phenomenal concepts, but identity holds"
))

# Functionalism
_register(TheoryEmbedding(
    "Functionalism", "FUNC",
    {"accessibility": 0.7, "integration": 0.5, "higher_order": 0.4,
     "intentionality": 0.6, "temporal": 0.6, "substrate": 0.2},
    "Mental states = functional roles (multiply realizable)"
))

# Dualism family
_register(TheoryEmbedding(
    "Substance Dualism (Descartes)", "SD",
    {"accessibility": 0.2, "integration": 0.5, "higher_order": 0.5,
     "intentionality": 0.8, "temporal": 0.6, "substrate": 0.1},
    "Mind and body are distinct substances"
))

_register(TheoryEmbedding(
    "Property Dualism", "PD",
    {"accessibility": 0.3, "integration": 0.5, "higher_order": 0.4,
     "intentionality": 0.7, "temporal": 0.5, "substrate": 0.3},
    "Mental properties are non-physical but supervene on physical"
))

# Integrated Information Theory
_register(TheoryEmbedding(
    "Integrated Information Theory", "IIT",
    {"accessibility": 0.4, "integration": 0.95, "higher_order": 0.3,
     "intentionality": 0.5, "temporal": 0.7, "substrate": 0.6},
    "Consciousness = integrated information (phi > 0)"
))

# Global Workspace Theory
_register(TheoryEmbedding(
    "Global Workspace Theory", "GWT",
    {"accessibility": 0.8, "integration": 0.6, "higher_order": 0.5,
     "intentionality": 0.5, "temporal": 0.8, "substrate": 0.3},
    "Consciousness = global broadcast to workspace"
))

# Higher-Order Theories
_register(TheoryEmbedding(
    "Higher-Order Thought Theory", "HOT",
    {"accessibility": 0.6, "integration": 0.4, "higher_order": 0.95,
     "intentionality": 0.7, "temporal": 0.6, "substrate": 0.3},
    "Conscious iff higher-order thought about it"
))

_register(TheoryEmbedding(
    "Higher-Order Perception Theory", "HOP",
    {"accessibility": 0.5, "integration": 0.4, "higher_order": 0.9,
     "intentionality": 0.6, "temporal": 0.6, "substrate": 0.3},
    "Conscious iff higher-order perception of it"
))

# Recurrent Processing Theory
_register(TheoryEmbedding(
    "Recurrent Processing Theory", "RPT",
    {"accessibility": 0.7, "integration": 0.7, "higher_order": 0.2,
     "intentionality": 0.4, "temporal": 0.8, "substrate": 0.5},
    "Consciousness requires recurrent neural processing"
))

# Attention Schema Theory
_register(TheoryEmbedding(
    "Attention Schema Theory", "AST",
    {"accessibility": 0.8, "integration": 0.5, "higher_order": 0.6,
     "intentionality": 0.5, "temporal": 0.7, "substrate": 0.3},
    "Consciousness = model of attention (schema)"
))

# Panpsychism
_register(TheoryEmbedding(
    "Panpsychism", "PAN",
    {"accessibility": 0.1, "integration": 0.6, "higher_order": 0.1,
     "intentionality": 0.3, "temporal": 0.5, "substrate": 0.5},
    "Consciousness is fundamental and ubiquitous"
))

# Eliminativism
_register(TheoryEmbedding(
    "Eliminativism", "ELIM",
    {"accessibility": 0.9, "integration": 0.2, "higher_order": 0.1,
     "intentionality": 0.2, "temporal": 0.3, "substrate": 0.9},
    "Folk-psychological consciousness doesn't exist"
))

# Mysterianism
_register(TheoryEmbedding(
    "Mysterianism", "MYST",
    {"accessibility": 0.1, "integration": 0.5, "higher_order": 0.5,
     "intentionality": 0.5, "temporal": 0.5, "substrate": 0.5},
    "Consciousness is real but cognitively closed to us"
))


class ConceptualSpace:
    """Multi-dimensional space for philosophical theory comparison.

    Supports:
    - Distance queries (how far apart are two theories?)
    - Nearest-neighbor (which theory is closest to position X?)
    - Dimensional analysis (which dimension separates theories most?)
    - Cluster detection (which theories form natural groups?)
    """

    def __init__(self, embeddings: Optional[Dict[str, TheoryEmbedding]] = None):
        self.embeddings = embeddings or dict(THEORY_EMBEDDINGS)

    def distance(self, theory_a: str, theory_b: str) -> float:
        """Euclidean distance between two theories."""
        if theory_a not in self.embeddings:
            raise ValueError(f"Unknown theory: {theory_a}")
        if theory_b not in self.embeddings:
            raise ValueError(f"Unknown theory: {theory_b}")

        va = self.embeddings[theory_a].as_list()
        vb = self.embeddings[theory_b].as_list()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(va, vb)))

    def nearest(self, target: str, n: int = 3) -> List[Tuple[str, float]]:
        """Find n theories closest to the target.

        Returns list of (theory_name, distance) sorted by distance.
        """
        if target not in self.embeddings:
            raise ValueError(f"Unknown theory: {target}")

        distances = []
        for name in self.embeddings:
            if name == target:
                continue
            d = self.distance(target, name)
            distances.append((name, d))

        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def nearest_to_vector(self, vector: Dict[str, float],
                          n: int = 3) -> List[Tuple[str, float]]:
        """Find theories closest to an arbitrary position vector."""
        v_list = [vector.get(d, 0.5) for d in DIMENSIONS]

        distances = []
        for name, emb in self.embeddings.items():
            e_list = emb.as_list()
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(v_list, e_list)))
            distances.append((name, d))

        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def separating_dimensions(self, theory_a: str,
                              theory_b: str) -> List[Tuple[str, float]]:
        """Find which dimensions separate two theories most.

        Returns list of (dimension, abs_difference) sorted by difference.
        """
        va = self.embeddings[theory_a].vector
        vb = self.embeddings[theory_b].vector

        diffs = []
        for dim in DIMENSIONS:
            diff = abs(va.get(dim, 0.5) - vb.get(dim, 0.5))
            diffs.append((dim, diff))

        diffs.sort(key=lambda x: x[1], reverse=True)
        return diffs

    def similarity(self, theory_a: str, theory_b: str) -> float:
        """Cosine similarity between two theories (0 to 1)."""
        va = self.embeddings[theory_a].as_list()
        vb = self.embeddings[theory_b].as_list()

        dot = sum(a * b for a, b in zip(va, vb))
        norm_a = math.sqrt(sum(a ** 2 for a in va))
        norm_b = math.sqrt(sum(b ** 2 for b in vb))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def all_pairwise_distances(self) -> Dict[Tuple[str, str], float]:
        """Compute all pairwise distances between theories."""
        names = list(self.embeddings.keys())
        result = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                result[(a, b)] = self.distance(a, b)
        return result

    def theory_profile(self, theory_name: str) -> Dict:
        """Get a human-readable profile of a theory's position."""
        if theory_name not in self.embeddings:
            raise ValueError(f"Unknown theory: {theory_name}")

        emb = self.embeddings[theory_name]
        nearest = self.nearest(theory_name, n=3)

        return {
            "name": emb.name,
            "description": emb.description,
            "position": dict(emb.vector),
            "nearest_theories": [
                {"name": n, "distance": round(d, 3)}
                for n, d in nearest
            ],
            "dominant_dimensions": [
                d for d, v in sorted(
                    emb.vector.items(), key=lambda x: x[1], reverse=True
                )[:3]
            ],
        }

    def get_stats(self) -> Dict:
        return {
            "theories": len(self.embeddings),
            "dimensions": len(DIMENSIONS),
            "dimension_names": DIMENSIONS,
        }
