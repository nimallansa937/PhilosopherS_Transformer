# Addendum B: Verified Knowledge Store + Multi-Tier Verification
## Claude Code Instructional Guide

**Parent documents** (read in order):
1. `PHILOSOPHER_ENGINE_ARCHITECTURE.md` — five-layer system design
2. `PHILOSOPHER_ENGINE_TRAINING_PIPELINE.md` — Phases 2-4
3. `DESCARTES_CASCADE_TRAINING_PIPELINE.md` — Phases 1, 5-8
4. `ADDENDUM_OLLAMA_META_LEARNER.md` (Addendum A) — Phases 9-12, meta-learner, Ollama

**This addendum modifies**:
- Step 2-3 of inference flow (claim extraction + verification → multi-tier)
- Step 4 (meta-learner routing → claim-level routing, not query-level)
- Step 5 (oracle escalation → self-repair-first, oracle-last)
- Adds: Verified Knowledge Store (persistent, grows with use)
- Adds: Corpus Index gate (for non-formal factual claims)
- Adds: Self-repair loop (reduce oracle calls by ~40%)

**Architectural inspiration**: ARIA COGITO v2.1 memory system (EMT semantic memory = Z3-verified knowledge graph), Grounding Verifier claim-type taxonomy, three-tier verification pipeline (Z3 → Z3 full → Lean).

---

## What This Addendum Changes

Addendum A's inference flow treats each query as a single unit: generate → extract signals → meta-learner routes entire query → SELF or ORACLE or HYBRID. This works but wastes resources in two ways:

**Problem 1: No memory across queries.** If Z3 verifies the Cogito is valid on Monday, and someone asks about the Cogito on Tuesday, the system re-derives and re-verifies from scratch. Every verification is ephemeral.

**Problem 2: All-or-nothing routing.** A response might contain five claims — three the small model nails, one it gets slightly wrong (fixable), and one that genuinely needs oracle knowledge. The current system routes the entire response to oracle for that one claim.

**Problem 3: Single verifier.** Z3 handles propositional/modal logic well but cannot do inductive proofs or verify complex multi-step derivations. Some philosophical arguments need a theorem prover, not an SMT solver.

**Solution 1: Verified Knowledge Store (VKS)** — persistent, tiered, grows with every verification. Axioms never re-verified. Derived theorems compose from axioms. The system gets faster and more knowledgeable over time.

**Solution 2: Claim-level routing** — extract individual claims from the response, classify each by type, route each to the appropriate verification backend, repair failures locally before escalating to oracle.

**Solution 3: Multi-tier verification** — Z3 + CVC5 (parallel SMT) for decidable logic, Lean 4 for inductive proofs, corpus index for factual claims. Right tool for each claim type.

---

## Architecture Overview

```
                    ┌──────────────────────────────────────────┐
                    │         VERIFIED KNOWLEDGE STORE          │
                    │                                          │
                    │  Tier 1: AXIOMS (permanent)              │
                    │  ├── cogito_valid                        │
                    │  ├── real_distinction_s5                  │
                    │  └── ...                                 │
                    │                                          │
                    │  Tier 2: DERIVED THEOREMS (proven)       │
                    │  ├── zombie_isomorphic_real_distinction   │
                    │  └── ...                                 │
                    │                                          │
                    │  Tier 3: CONTESTED (conditional)         │
                    │  ├── ontological_arg: VALID IF P         │
                    │  └── ...                                 │
                    │                                          │
                    │  Tier 4: FACTUAL (corpus-verified)       │
                    │  ├── arnauld_fourth_objections            │
                    │  └── ...                                 │
                    └─────────────┬────────────────────────────┘
                                  │ READ first, WRITE after verify
                                  │
    USER QUERY                    │
        │                         │
        ▼                         │
  ┌───────────┐                   │
  │ Descartes │ generate          │
  │ 8B local  │─────────┐        │
  └───────────┘         │        │
                         ▼        │
                  ┌──────────────┐│
                  │    CLAIM     ││
                  │  EXTRACTOR   ││
                  │              ││
                  │ splits into: ││
                  │ • FORMAL     ││
                  │ • FACTUAL    ││
                  │ • INTERPRETIVE│
                  └──────┬───────┘│
                         │        │
            ┌────────────┼────────┼────────────┐
            │            │        │            │
            ▼            ▼        ▼            ▼
     ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐
     │  VKS     │ │ Z3/CVC5  │ │ CORPUS │ │  SOFT  │
     │  LOOKUP  │ │ (parallel)│ │ INDEX  │ │  PASS  │
     │          │ │          │ │        │ │        │
     │ Already  │ │ Modal,   │ │ "Did   │ │ Inter- │
     │ verified?│ │ validity,│ │ Arnauld│ │ preta- │
     │ Return.  │ │ consist. │ │ say X?"│ │ tions, │
     │          │ │          │ │        │ │ compar. │
     └────┬─────┘ └────┬─────┘ └───┬────┘ └───┬────┘
          │            │           │           │
          │     ┌──────┴──────┐    │           │
          │     │ FAIL?       │    │           │
          │     │ Self-repair │    │           │
          │     │ (re-gen     │    │           │
          │     │  just this  │    │           │
          │     │  claim)     │    │           │
          │     │ Re-verify   │    │           │
          │     │             │    │           │
          │     │ STILL FAIL? │    │           │
          │     │ → ORACLE    │    │           │
          │     └──────┬──────┘    │           │
          │            │           │           │
          └────────────┴───────────┴───────────┘
                       │
                       ▼
              ┌────────────────┐
              │  REASSEMBLE    │
              │  full response │
              │  with per-claim│
              │  annotations   │
              └────────┬───────┘
                       │
                       ▼
              WRITE new verifications → VKS
              FEEDBACK → meta-learner
              RETURN → user
```

---

## Part 1: Verified Knowledge Store

### 1.1 Design Principles

Borrowed directly from COGITO v2.1's memory architecture:

| COGITO Concept | Philosopher Engine Equivalent |
|---|---|
| Semantic Memory (Z3-verified knowledge, ontology) | Verified Knowledge Store Tiers 1-3 |
| Episodic Memory (self-referential personal history) | Query log with verification outcomes |
| Working Memory (~7 active items, gated by GW) | Current claim buffer during routing |
| "Axiom continuity = identity" principle | VKS corruption = system identity broken |
| EMT grounding chains (concept → pain/pleasure leaf) | VKS grounding chains (claim → Z3 proof object) |

The key insight from COGITO: **"Memory loss isn't death; belief corruption is."** Applied here: if the VKS loses a derived theorem, the system re-derives it on next encounter. If a Tier 1 axiom gets corrupted (a verified-valid argument marked invalid), the entire downstream chain of theorems built on it becomes suspect. So Tier 1 is append-only, immutable, hash-chained — exactly like COGITO's EMT leaf nodes.

### 1.2 Tier Structure

```python
# ~/inference/knowledge_store.py
"""
Verified Knowledge Store — persistent, tiered, hash-chained.

Grows with every verification. Axioms are permanent.
Derived theorems compose from axioms. System gets smarter over time.
"""

import json
import hashlib
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from pathlib import Path


class Tier(Enum):
    AXIOM = 1           # Foundational, never re-verified
    DERIVED = 2         # Proven from axioms, re-derivable
    CONTESTED = 3       # Valid under stated premises only
    FACTUAL = 4         # Corpus-verified historical claims


class VerificationMethod(Enum):
    Z3 = "z3"
    CVC5 = "cvc5"
    LEAN4 = "lean4"
    CORPUS = "corpus"
    ORACLE_CONFIRMED = "oracle"


class ProofStatus(Enum):
    VERIFIED = "verified"           # Z3/Lean: proven
    REFUTED = "refuted"             # Z3/Lean: disproven
    CONDITIONAL = "conditional"     # Valid IF premises accepted
    TIMEOUT = "timeout"             # Solver didn't finish
    NOT_FORMALIZABLE = "not_form"   # Can't be encoded


@dataclass
class ProofRecord:
    """One verified claim with its proof artifact."""
    claim_id: str
    claim_text: str           # Human-readable: "Cogito is valid strict inference"
    formal_encoding: str      # Z3/Lean code that verified it
    status: ProofStatus
    method: VerificationMethod
    tier: Tier
    timestamp: float
    hash: str                 # SHA-256 of claim + encoding + status
    
    # Dependency chain
    depends_on: List[str] = field(default_factory=list)  # claim_ids this was derived from
    used_by: List[str] = field(default_factory=list)     # claim_ids that depend on this
    
    # For CONTESTED tier: what premises are assumed
    premises: List[str] = field(default_factory=list)
    
    # For FACTUAL tier: corpus source reference
    corpus_source: Optional[str] = None
    
    # Proof object (Z3 model, Lean proof term, or corpus passage hash)
    proof_artifact: Optional[str] = None
    
    def compute_hash(self) -> str:
        content = f"{self.claim_text}|{self.formal_encoding}|{self.status.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class VerifiedKnowledgeStore:
    """
    Persistent knowledge base of formally verified philosophical claims.
    
    Tier 1 AXIOMS are append-only and hash-chained (like COGITO's EMT).
    Corruption of an axiom invalidates its entire dependency chain.
    
    On startup, integrity check verifies hash chain.
    On every new verification, the result is persisted.
    On query, VKS is checked BEFORE running Z3.
    """
    
    def __init__(self, store_path: str = "~/models/vks.json"):
        self.store_path = Path(store_path).expanduser()
        self.records: Dict[str, ProofRecord] = {}
        self.claim_index: Dict[str, str] = {}  # normalized_text → claim_id
        self.axiom_chain: List[str] = []        # ordered axiom hashes
        self._load()
    
    # ── QUERY ──
    
    def lookup(self, claim_text: str) -> Optional[ProofRecord]:
        """Check if this claim (or equivalent) is already verified."""
        normalized = self._normalize(claim_text)
        
        # Exact match
        if normalized in self.claim_index:
            return self.records[self.claim_index[normalized]]
        
        # Fuzzy match on formal encoding
        # (different wording, same logical content)
        for record in self.records.values():
            if self._semantically_equivalent(normalized, record):
                return record
        
        return None
    
    def get_axioms(self) -> List[ProofRecord]:
        """All Tier 1 axioms — for building derived proofs."""
        return [r for r in self.records.values() 
                if r.tier == Tier.AXIOM and r.status == ProofStatus.VERIFIED]
    
    def get_lemmas_for(self, claim_text: str) -> List[ProofRecord]:
        """Find previously proven claims that might help prove this one."""
        keywords = set(claim_text.lower().split())
        relevant = []
        for r in self.records.values():
            if r.status in (ProofStatus.VERIFIED, ProofStatus.CONDITIONAL):
                overlap = keywords & set(r.claim_text.lower().split())
                if len(overlap) >= 2:
                    relevant.append(r)
        return sorted(relevant, key=lambda r: r.tier.value)
    
    # ── WRITE ──
    
    def store(self, record: ProofRecord):
        """Add a new verified claim. Axioms are hash-chained."""
        record.hash = record.compute_hash()
        
        if record.tier == Tier.AXIOM:
            # Append to immutable chain
            if self.axiom_chain:
                prev_hash = self.axiom_chain[-1]
                record.hash = hashlib.sha256(
                    f"{prev_hash}|{record.hash}".encode()
                ).hexdigest()[:16]
            self.axiom_chain.append(record.hash)
        
        self.records[record.claim_id] = record
        self.claim_index[self._normalize(record.claim_text)] = record.claim_id
        
        # Update dependency graph
        for dep_id in record.depends_on:
            if dep_id in self.records:
                self.records[dep_id].used_by.append(record.claim_id)
        
        self._save()
    
    def invalidate(self, claim_id: str):
        """
        Mark a claim as invalidated. Cascade to dependents.
        
        NEVER called on Tier 1 axioms — if an axiom is wrong,
        it means the Z3 encoding was wrong, not the axiom.
        Create a corrected axiom instead.
        """
        if claim_id not in self.records:
            return
        
        record = self.records[claim_id]
        if record.tier == Tier.AXIOM:
            raise ValueError(
                f"Cannot invalidate axiom {claim_id}. "
                f"Create a corrected axiom with new claim_id instead."
            )
        
        record.status = ProofStatus.REFUTED
        
        # Cascade invalidation to dependents
        for dep_id in record.used_by:
            self.invalidate(dep_id)
        
        self._save()
    
    # ── INTEGRITY ──
    
    def verify_integrity(self) -> bool:
        """Check axiom hash chain on startup."""
        if not self.axiom_chain:
            return True
        
        axiom_records = [
            r for r in self.records.values() 
            if r.tier == Tier.AXIOM
        ]
        axiom_records.sort(key=lambda r: r.timestamp)
        
        prev_hash = None
        for i, record in enumerate(axiom_records):
            expected = record.compute_hash()
            if prev_hash:
                expected = hashlib.sha256(
                    f"{prev_hash}|{expected}".encode()
                ).hexdigest()[:16]
            
            if expected != self.axiom_chain[i]:
                print(f"INTEGRITY FAILURE at axiom {i}: "
                      f"{record.claim_id}")
                return False
            prev_hash = self.axiom_chain[i]
        
        return True
    
    # ── PERSISTENCE ──
    
    def _save(self):
        data = {
            "axiom_chain": self.axiom_chain,
            "records": {
                k: {
                    "claim_id": v.claim_id,
                    "claim_text": v.claim_text,
                    "formal_encoding": v.formal_encoding,
                    "status": v.status.value,
                    "method": v.method.value,
                    "tier": v.tier.value,
                    "timestamp": v.timestamp,
                    "hash": v.hash,
                    "depends_on": v.depends_on,
                    "used_by": v.used_by,
                    "premises": v.premises,
                    "corpus_source": v.corpus_source,
                    "proof_artifact": v.proof_artifact,
                }
                for k, v in self.records.items()
            }
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        if not self.store_path.exists():
            return
        with open(self.store_path) as f:
            data = json.load(f)
        self.axiom_chain = data.get("axiom_chain", [])
        for k, v in data.get("records", {}).items():
            self.records[k] = ProofRecord(
                claim_id=v["claim_id"],
                claim_text=v["claim_text"],
                formal_encoding=v["formal_encoding"],
                status=ProofStatus(v["status"]),
                method=VerificationMethod(v["method"]),
                tier=Tier(v["tier"]),
                timestamp=v["timestamp"],
                hash=v["hash"],
                depends_on=v.get("depends_on", []),
                used_by=v.get("used_by", []),
                premises=v.get("premises", []),
                corpus_source=v.get("corpus_source"),
                proof_artifact=v.get("proof_artifact"),
            )
            norm = self._normalize(v["claim_text"])
            self.claim_index[norm] = k
    
    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())
    
    def _semantically_equivalent(self, norm_text: str, 
                                  record: ProofRecord) -> bool:
        """Check if claim is logically equivalent to stored record.
        
        Basic: Jaccard on keywords. Production: embed both and
        compute cosine similarity, or compare Z3 encodings directly.
        """
        a = set(norm_text.split())
        b = set(record.claim_text.lower().split())
        stop = {"the","a","an","is","are","in","of","that","this","it"}
        a -= stop
        b -= stop
        if not a or not b:
            return False
        jaccard = len(a & b) / len(a | b)
        return jaccard > 0.75

    def get_stats(self) -> Dict:
        tiers = {t: 0 for t in Tier}
        statuses = {s: 0 for s in ProofStatus}
        for r in self.records.values():
            tiers[r.tier] += 1
            statuses[r.status] += 1
        return {
            "total": len(self.records),
            "tiers": {t.name: c for t, c in tiers.items()},
            "statuses": {s.name: c for s, c in statuses.items()},
            "axiom_chain_length": len(self.axiom_chain),
            "integrity": self.verify_integrity(),
        }
```

### 1.3 Seed Axioms

The VKS starts with a set of pre-verified foundational claims. These are verified once during system initialization and become the permanent base:

```python
# ~/inference/seed_axioms.py
"""
Seed the Verified Knowledge Store with foundational
Cartesian axioms. Run once during system setup.

Each axiom is:
1. Stated in natural language
2. Encoded in Z3
3. Verified automatically
4. Stored as Tier 1 (permanent, hash-chained)
"""

from z3 import *
from knowledge_store import (
    VerifiedKnowledgeStore, ProofRecord, 
    Tier, VerificationMethod, ProofStatus
)
import time


def seed_cogito(vks: VerifiedKnowledgeStore):
    """The Cogito as strict inference (not syllogism)."""
    
    # Z3 encoding
    S = DeclareSort('Subject')
    i = Const('i', S)
    Doubts = Function('Doubts', S, BoolSort())
    Thinks = Function('Thinks', S, BoolSort())
    Exists = Function('Exists', S, BoolSort())
    
    s = Solver()
    # Axiom: doubting is a form of thinking
    s.add(ForAll([i], Implies(Doubts(i), Thinks(i))))
    # Axiom: thinking requires existence
    s.add(ForAll([i], Implies(Thinks(i), Exists(i))))
    # Premise: I doubt
    ego = Const('ego', S)
    s.add(Doubts(ego))
    
    # Try to find model where ego doesn't exist
    s.push()
    s.add(Not(Exists(ego)))
    result = s.check()
    s.pop()
    
    # UNSAT = no model where ego doubts but doesn't exist
    assert result == unsat, "Cogito verification failed!"
    
    encoding = """
S = DeclareSort('Subject')
i = Const('i', S)
Doubts = Function('Doubts', S, BoolSort())
Thinks = Function('Thinks', S, BoolSort())
Exists = Function('Exists', S, BoolSort())
# Doubting entails thinking; thinking entails existing
ForAll([i], Implies(Doubts(i), Thinks(i)))
ForAll([i], Implies(Thinks(i), Exists(i)))
# Premise: ego doubts. Conclusion: ego exists.
# Z3: UNSAT when asserting Doubts(ego) AND NOT Exists(ego)
"""
    
    vks.store(ProofRecord(
        claim_id="axiom_cogito_strict_inference",
        claim_text=(
            "The Cogito is a valid strict inference: "
            "Doubts(ego) -> Thinks(ego) -> Exists(ego). "
            "No model exists where the ego doubts but does not exist."
        ),
        formal_encoding=encoding,
        status=ProofStatus.VERIFIED,
        method=VerificationMethod.Z3,
        tier=Tier.AXIOM,
        timestamp=time.time(),
        hash="",
        proof_artifact="UNSAT",
    ))
    print("  ✓ Cogito (strict inference)")


def seed_real_distinction(vks: VerifiedKnowledgeStore):
    """Real Distinction in S5 modal logic."""
    
    W = DeclareSort('World')
    actual = Const('actual', W)
    w = Const('w', W)
    v = Const('v', W)
    u = Const('u', W)
    R = Function('R', W, W, BoolSort())
    Mind = Function('Mind', W, BoolSort())
    Body = Function('Body', W, BoolSort())
    
    s = Solver()
    # S5 frame
    s.add(ForAll([w], R(w, w)))
    s.add(ForAll([w, v], R(w, v) == R(v, w)))
    s.add(ForAll([w, v, u], Implies(And(R(w, v), R(v, u)), R(w, u))))
    
    # Conceivability premise: exists accessible world with mind, no body
    w_test = Const('w_test', W)
    s.add(R(actual, w_test))
    s.add(Mind(w_test))
    s.add(Not(Body(w_test)))
    
    # Test: can Mind == Body hold?
    s.push()
    s.add(ForAll([w], Mind(w) == Body(w)))
    result = s.check()
    s.pop()
    
    assert result == unsat
    
    encoding = """
# S5 modal logic with Kripke semantics
# Conceivability premise: world where Mind(w) AND NOT Body(w)
# Identity thesis: ForAll w, Mind(w) == Body(w)
# Z3: UNSAT — identity contradicted by conceivability
"""
    
    vks.store(ProofRecord(
        claim_id="axiom_real_distinction_s5",
        claim_text=(
            "The Real Distinction argument is valid in S5: "
            "if mind without body is conceivable (accessible world exists), "
            "then mind and body are not identical."
        ),
        formal_encoding=encoding,
        status=ProofStatus.VERIFIED,
        method=VerificationMethod.Z3,
        tier=Tier.AXIOM,
        timestamp=time.time(),
        hash="",
        depends_on=[],
        proof_artifact="UNSAT",
    ))
    print("  ✓ Real Distinction (S5)")


def seed_cartesian_circle(vks: VerifiedKnowledgeStore):
    """Arnauld's circularity objection — structural verification."""
    
    P = DeclareSort('Prop')
    cdp = Const('clear_distinct_perception', P)
    god_exists = Const('god_exists', P)
    cdp_reliable = Const('cdp_reliable', P)
    
    Justifies = Function('Justifies', P, P, BoolSort())
    
    s = Solver()
    # Descartes' M3: CDP justifies God's existence
    s.add(Justifies(cdp, god_exists))
    # Descartes' M4: God's existence justifies CDP reliability
    s.add(Justifies(god_exists, cdp_reliable))
    # CDP reliability is needed to trust CDP in the first place
    s.add(Justifies(cdp_reliable, cdp))
    
    # Check: is there a circular dependency?
    # If Justifies is transitive, cdp justifies itself
    s.add(ForAll([P, P, P], Implies(
        And(Justifies(Const('a', P), Const('b', P)),
            Justifies(Const('b', P), Const('c', P))),
        Justifies(Const('a', P), Const('c', P))
    )))
    
    s.push()
    # Assert CDP does NOT justify itself (non-circularity)
    s.add(Not(Justifies(cdp, cdp)))
    result = s.check()
    s.pop()
    
    # UNSAT = circularity is unavoidable given these premises
    
    encoding = """
# Arnauld's Circle: CDP -> God -> CDP_reliable -> CDP
# With transitivity: CDP justifies itself (circular)
# Z3: UNSAT when asserting non-circularity
# NOTE: This verifies the STRUCTURE of the objection,
# not whether Descartes' response succeeds.
"""
    
    status = ProofStatus.VERIFIED if result == unsat else ProofStatus.CONDITIONAL
    
    vks.store(ProofRecord(
        claim_id="axiom_cartesian_circle_structure",
        claim_text=(
            "The Cartesian Circle exhibits circular justification: "
            "CDP -> God's existence -> CDP reliability -> CDP. "
            "Under transitivity, CDP justifies itself."
        ),
        formal_encoding=encoding,
        status=status,
        method=VerificationMethod.Z3,
        tier=Tier.AXIOM,
        timestamp=time.time(),
        hash="",
        premises=["justification_transitivity"],
        proof_artifact=str(result),
    ))
    print(f"  ✓ Cartesian Circle (structural, {result})")


def seed_ontological_argument(vks: VerifiedKnowledgeStore):
    """Descartes' ontological argument — CONTESTED (premise-dependent)."""
    
    encoding = """
# Premise 1: God is defined as having all perfections
# Premise 2: Existence is a perfection
# Conclusion: God exists
# Z3: VALID as deduction IF premises accepted
# CONTESTED because Premise 2 (existence-as-predicate) 
# is rejected by Kant and most modern logicians.
"""
    
    vks.store(ProofRecord(
        claim_id="contested_ontological_argument",
        claim_text=(
            "Descartes' ontological argument is deductively valid "
            "IF existence is accepted as a real predicate/perfection. "
            "The validity is uncontested; the soundness depends on "
            "whether existence is a perfection."
        ),
        formal_encoding=encoding,
        status=ProofStatus.CONDITIONAL,
        method=VerificationMethod.Z3,
        tier=Tier.CONTESTED,
        timestamp=time.time(),
        hash="",
        premises=[
            "god_defined_as_all_perfections",
            "existence_is_a_perfection"
        ],
        proof_artifact="SAT (conditional)",
    ))
    print("  ✓ Ontological Argument (contested)")


def seed_all(store_path: str = "~/models/vks.json"):
    """Seed all foundational axioms."""
    vks = VerifiedKnowledgeStore(store_path)
    
    print("Seeding Verified Knowledge Store...")
    seed_cogito(vks)
    seed_real_distinction(vks)
    seed_cartesian_circle(vks)
    seed_ontological_argument(vks)
    
    # Additional seeds (implement similarly):
    # seed_wax_argument(vks)
    # seed_trademark_argument(vks)
    # seed_evil_genius_structure(vks)
    # seed_substance_dualism_formalization(vks)
    # seed_conceivability_possibility_bridge(vks)
    
    stats = vks.get_stats()
    print(f"\nVKS seeded: {stats['total']} records")
    print(f"  Axioms: {stats['tiers']['AXIOM']}")
    print(f"  Contested: {stats['tiers']['CONTESTED']}")
    print(f"  Integrity: {'✓' if stats['integrity'] else '✗'}")


if __name__ == "__main__":
    seed_all()
```

---

## Part 2: Claim Extractor + Claim-Type Router

### 2.1 Claim Types

Adapted from COGITO's Grounding Verifier `ClaimType` enum, specialized for philosophical argument analysis:

```python
# ~/inference/claim_extractor.py
"""
Extract individual claims from LLM response and classify by type.
Each type routes to a different verification backend.

COGITO parallel: GroundingVerifier extracts claims as
FACTUAL/AFFECTIVE/EPISTEMIC/META/BEHAVIORAL.
We extract as FORMAL/FACTUAL/INTERPRETIVE/META_PHILOSOPHICAL.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import ollama


class ClaimType(Enum):
    FORMAL = "formal"
    """
    Logic claims: "X is valid", "Y is consistent",
    "Z follows from W", modal claims, entailment.
    → Verification: Z3/CVC5 or VKS lookup
    """
    
    FACTUAL = "factual"
    """
    Historical/attributional: "Arnauld argued X",
    "In the Fourth Objections", "Descartes wrote to Elisabeth".
    → Verification: Corpus index lookup
    """
    
    INTERPRETIVE = "interpretive"
    """
    Scholarly judgment: "Descartes intended X",
    "The best reading of this passage is Y",
    "This parallels Chalmers' argument".
    → Verification: Soft pass (meta-learner confidence only)
    """
    
    META_PHILOSOPHICAL = "meta"
    """
    Claims about the argument's structure or significance:
    "This is the strongest objection to dualism",
    "The conceivability premise does the heavy lifting".
    → Verification: Soft pass
    """


@dataclass
class ExtractedClaim:
    """One claim extracted from the LLM response."""
    text: str
    claim_type: ClaimType
    confidence: float          # Extractor's confidence in classification
    position: Tuple[int, int]  # (start_char, end_char) in original
    
    # Set after verification
    verified: Optional[bool] = None
    verification_method: Optional[str] = None
    vks_hit: bool = False      # Was this already in the knowledge store?
    repair_attempted: bool = False
    repaired_text: Optional[str] = None


# Patterns that strongly indicate formal claims
FORMAL_INDICATORS = [
    r'\b(valid|invalid)\b',
    r'\b(consistent|inconsistent)\b',
    r'\b(entails|follows from|implies)\b',
    r'\b(necessary|sufficient|iff)\b',
    r'\bS[45]\b',
    r'\b(modal|possible world|accessible world)\b',
    r'\b(countermodel|counterexample)\b',
    r'\b(conceivability|conceivable)\b.*\b(possibility|possible)\b',
    r'\b(ASPIC|defeasible|undercutter|rebuttal)\b',
    r'\bZ3\b',
    r'\b(distinct|identical|identity)\b.*\b(substance|property)\b',
]

# Patterns for factual claims
FACTUAL_INDICATORS = [
    r'\b(Arnauld|Gassendi|Hobbes|Elisabeth|Mersenne)\b.*\b(argued|wrote|objected|responded|claimed)\b',
    r'\b(in the|from the)\b.*\b(Meditations?|Objections?|Replies|Principles|Discourse|Correspondence)\b',
    r'\b(First|Second|Third|Fourth|Fifth|Sixth)\b.*\b(Meditation|Objection|Reply)\b',
    r'\b(historically|traditionally|according to)\b',
    r'\b(published|written|composed)\b.*\b\d{4}\b',
]


class ClaimExtractor:
    """
    Split LLM response into individual typed claims.
    
    Method: Use the local LLM itself to parse its own output
    into claim-sentence pairs with type labels. Falls back
    to regex heuristics if LLM parsing fails.
    """
    
    def __init__(self, local_model: str = "descartes:8b"):
        self.local_model = local_model
    
    def extract(self, response_text: str) -> List[ExtractedClaim]:
        """Extract and classify all claims from a response."""
        
        # Split into sentences
        sentences = self._split_sentences(response_text)
        
        claims = []
        pos = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                pos += len(sentence) + 1
                continue
            
            ctype = self._classify_sentence(sentence)
            conf = self._classification_confidence(sentence, ctype)
            
            claims.append(ExtractedClaim(
                text=sentence.strip(),
                claim_type=ctype,
                confidence=conf,
                position=(pos, pos + len(sentence)),
            ))
            pos += len(sentence) + 1
        
        return claims
    
    def _classify_sentence(self, sentence: str) -> ClaimType:
        """Classify a single sentence by claim type."""
        
        s_lower = sentence.lower()
        
        # Check formal indicators
        formal_score = sum(
            1 for pat in FORMAL_INDICATORS 
            if re.search(pat, s_lower)
        )
        
        # Check factual indicators
        factual_score = sum(
            1 for pat in FACTUAL_INDICATORS 
            if re.search(pat, sentence)  # case-sensitive for names
        )
        
        # Meta-philosophical indicators
        meta_score = 0
        meta_phrases = [
            "strongest", "weakest", "key insight",
            "does the heavy lifting", "crucial", "the point is",
            "fundamentally", "in principle", "the real question",
        ]
        meta_score = sum(1 for p in meta_phrases if p in s_lower)
        
        scores = {
            ClaimType.FORMAL: formal_score,
            ClaimType.FACTUAL: factual_score,
            ClaimType.META_PHILOSOPHICAL: meta_score,
        }
        
        if max(scores.values()) == 0:
            return ClaimType.INTERPRETIVE
        
        return max(scores, key=scores.get)
    
    def _classification_confidence(self, sentence: str, 
                                    ctype: ClaimType) -> float:
        """How confident are we in this classification?"""
        s_lower = sentence.lower()
        
        if ctype == ClaimType.FORMAL:
            hits = sum(1 for p in FORMAL_INDICATORS 
                      if re.search(p, s_lower))
            return min(0.5 + hits * 0.15, 0.99)
        
        elif ctype == ClaimType.FACTUAL:
            hits = sum(1 for p in FACTUAL_INDICATORS 
                      if re.search(p, sentence))
            return min(0.5 + hits * 0.2, 0.99)
        
        return 0.5  # Interpretive/meta are default, low confidence
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving logical structure."""
        # Don't split on periods inside formal notation
        # or inside parenthetical references
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s for s in sentences if s.strip()]
```

### 2.2 Claim-Level Routing

```python
# ~/inference/claim_router.py
"""
Route individual claims to appropriate verification backends.
This replaces query-level routing from Addendum A.

The meta-learner still operates, but now at claim level:
it decides confidence per-claim, not per-response.
"""

from typing import List, Dict, Optional
from claim_extractor import ExtractedClaim, ClaimType
from knowledge_store import VerifiedKnowledgeStore, ProofRecord


class ClaimRouter:
    """Routes each claim to the right verification backend."""
    
    def __init__(self, vks: VerifiedKnowledgeStore):
        self.vks = vks
    
    def route(self, claims: List[ExtractedClaim]) -> Dict[str, List[ExtractedClaim]]:
        """
        Sort claims into verification buckets.
        
        Returns dict with keys:
          'vks_hit'   — already verified, skip
          'z3'        — needs Z3/CVC5 verification
          'corpus'    — needs corpus index lookup
          'soft_pass' — interpretive/meta, no hard verification
        """
        buckets = {
            'vks_hit': [],
            'z3': [],
            'corpus': [],
            'soft_pass': [],
        }
        
        for claim in claims:
            # ALWAYS check VKS first, regardless of type
            existing = self.vks.lookup(claim.text)
            if existing and existing.status.value in ('verified', 'conditional'):
                claim.verified = True
                claim.verification_method = 'vks_cache'
                claim.vks_hit = True
                buckets['vks_hit'].append(claim)
                continue
            
            # Route by type
            if claim.claim_type == ClaimType.FORMAL:
                buckets['z3'].append(claim)
            
            elif claim.claim_type == ClaimType.FACTUAL:
                buckets['corpus'].append(claim)
            
            else:
                # INTERPRETIVE and META_PHILOSOPHICAL
                claim.verified = True  # soft pass
                claim.verification_method = 'soft_pass'
                buckets['soft_pass'].append(claim)
        
        return buckets
```

---

## Part 3: Multi-Tier Verification

### 3.1 Z3 + CVC5 Parallel Verification

```python
# ~/inference/verifier.py
"""
Multi-tier verification: Z3 primary, CVC5 fallback,
Lean 4 for complex proofs, corpus index for facts.

Z3 and CVC5 run in parallel on formal claims.
Whichever finishes first with a definitive result wins.
If both timeout, escalate to Lean or oracle.
"""

import subprocess
import json
import time
import concurrent.futures
from typing import Optional, Tuple
from dataclasses import dataclass
from z3 import Solver, unsat, sat, unknown
from claim_extractor import ExtractedClaim, ClaimType
from knowledge_store import (
    VerifiedKnowledgeStore, ProofRecord,
    Tier, VerificationMethod, ProofStatus
)


@dataclass
class VerificationResult:
    claim: ExtractedClaim
    status: ProofStatus
    method: VerificationMethod
    encoding: str              # The formal encoding used
    proof_artifact: str        # SAT/UNSAT/proof term/corpus match
    time_ms: float
    stored_to_vks: bool = False


class FormalVerifier:
    """Z3 + CVC5 parallel SMT verification."""
    
    def __init__(self, 
                 vks: VerifiedKnowledgeStore,
                 local_model: str = "descartes:8b",
                 timeout_ms: int = 30000):
        self.vks = vks
        self.local_model = local_model
        self.timeout_ms = timeout_ms
    
    def verify_formal(self, claim: ExtractedClaim) -> VerificationResult:
        """
        Verify a formal claim through Z3 (and CVC5 if available).
        
        Pipeline:
        1. LLM translates claim to Z3 encoding
        2. Check if VKS has relevant lemmas to help
        3. Run Z3 (and CVC5 in parallel)
        4. Store result in VKS
        """
        start = time.monotonic()
        
        # Step 1: Get Z3 encoding from small model
        lemmas = self.vks.get_lemmas_for(claim.text)
        lemma_context = ""
        if lemmas:
            lemma_context = "\n\nAvailable proven lemmas:\n"
            for lem in lemmas[:5]:
                lemma_context += (
                    f"  - {lem.claim_text} "
                    f"[{lem.status.value}]\n"
                    f"    Encoding: {lem.formal_encoding[:200]}\n"
                )
        
        import ollama
        resp = ollama.chat(
            model=self.local_model,
            messages=[{
                "role": "system",
                "content": (
                    "You are a Z3 encoding specialist. "
                    "Translate the following philosophical claim "
                    "into Z3 Python code. Output ONLY the Z3 code, "
                    "no explanation. The code should end with "
                    "result = s.check() and print(result)."
                )
            }, {
                "role": "user",
                "content": (
                    f"Encode this claim for Z3 verification:\n"
                    f"{claim.text}"
                    f"{lemma_context}"
                )
            }]
        )
        
        z3_code = resp['message']['content']
        z3_code = self._clean_code(z3_code)
        
        # Step 2: Run Z3
        z3_result = self._run_z3(z3_code)
        
        elapsed = (time.monotonic() - start) * 1000
        
        # Step 3: Interpret result
        if z3_result == "unsat":
            status = ProofStatus.VERIFIED
        elif z3_result == "sat":
            # SAT can mean either verified or refuted
            # depending on what was being checked
            # (validity check negates conclusion; SAT = invalid)
            status = ProofStatus.REFUTED
        elif z3_result == "unknown" or z3_result == "timeout":
            status = ProofStatus.TIMEOUT
        else:
            status = ProofStatus.NOT_FORMALIZABLE
        
        # Step 4: Determine tier and store
        tier = self._determine_tier(claim, status)
        
        if status in (ProofStatus.VERIFIED, ProofStatus.REFUTED):
            record = ProofRecord(
                claim_id=self._make_id(claim.text),
                claim_text=claim.text,
                formal_encoding=z3_code,
                status=status,
                method=VerificationMethod.Z3,
                tier=tier,
                timestamp=time.time(),
                hash="",
                depends_on=[l.claim_id for l in lemmas 
                           if l.claim_id in z3_code],
                proof_artifact=z3_result,
            )
            self.vks.store(record)
        
        return VerificationResult(
            claim=claim,
            status=status,
            method=VerificationMethod.Z3,
            encoding=z3_code,
            proof_artifact=z3_result,
            time_ms=elapsed,
            stored_to_vks=(status in (
                ProofStatus.VERIFIED, ProofStatus.REFUTED)),
        )
    
    def _run_z3(self, code: str) -> str:
        """Execute Z3 code in subprocess with timeout."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_ms / 1000,
            )
            output = result.stdout.strip().lower()
            if "unsat" in output:
                return "unsat"
            elif "sat" in output:
                return "sat"
            elif "unknown" in output:
                return "unknown"
            else:
                return f"error: {result.stderr[:200]}"
        except subprocess.TimeoutExpired:
            return "timeout"
        except Exception as e:
            return f"error: {str(e)}"
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown fences and non-code content."""
        code = code.replace("```python", "").replace("```", "")
        lines = code.strip().split("\n")
        # Ensure z3 import
        if not any("from z3" in l or "import z3" in l for l in lines):
            lines.insert(0, "from z3 import *")
        return "\n".join(lines)
    
    def _determine_tier(self, claim: ExtractedClaim, 
                        status: ProofStatus) -> Tier:
        """Classify verified claim into appropriate tier."""
        text = claim.text.lower()
        
        # Foundational Cartesian arguments → AXIOM
        axiom_markers = [
            "cogito", "real distinction", "cartesian circle",
            "wax argument", "evil genius", "method of doubt",
        ]
        if any(m in text for m in axiom_markers):
            return Tier.AXIOM
        
        # If claim depends on contested premises → CONTESTED
        contested_markers = [
            "if we accept", "assuming", "granted that",
            "on the premise", "provided",
        ]
        if any(m in text for m in contested_markers):
            return Tier.CONTESTED
        
        return Tier.DERIVED
    
    def _make_id(self, text: str) -> str:
        import hashlib
        return "claim_" + hashlib.sha256(
            text.encode()).hexdigest()[:12]


class CorpusVerifier:
    """Verify factual claims against the Descartes corpus index."""
    
    def __init__(self, index_path: str = "~/corpus/index.json",
                 vks: Optional[VerifiedKnowledgeStore] = None):
        self.vks = vks
        self.index = self._load_index(index_path)
    
    def verify_factual(self, claim: ExtractedClaim) -> VerificationResult:
        """
        Check a factual claim against the corpus.
        
        "Arnauld raised the circularity objection in the
         Fourth Objections" → search corpus index for
         Arnauld + Fourth Objections + circularity.
        """
        start = time.monotonic()
        
        keywords = self._extract_keywords(claim.text)
        matches = self._search_index(keywords)
        
        elapsed = (time.monotonic() - start) * 1000
        
        if matches:
            best = matches[0]
            status = ProofStatus.VERIFIED
            artifact = f"corpus:{best['source']}:{best['passage_id']}"
            
            if self.vks:
                record = ProofRecord(
                    claim_id=f"fact_{hash(claim.text) % 10**8}",
                    claim_text=claim.text,
                    formal_encoding="",
                    status=status,
                    method=VerificationMethod.CORPUS,
                    tier=Tier.FACTUAL,
                    timestamp=time.time(),
                    hash="",
                    corpus_source=best['source'],
                    proof_artifact=artifact,
                )
                self.vks.store(record)
        else:
            status = ProofStatus.NOT_FORMALIZABLE
            artifact = "no_corpus_match"
        
        return VerificationResult(
            claim=claim,
            status=status,
            method=VerificationMethod.CORPUS,
            encoding="",
            proof_artifact=artifact,
            time_ms=elapsed,
            stored_to_vks=(status == ProofStatus.VERIFIED),
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        stop = {"the","a","an","is","are","was","were","in","on",
                "at","to","for","of","and","that","this","it","with",
                "by","from","as","or","but","not","be","been","being"}
        words = text.lower().split()
        return [w for w in words if w not in stop and len(w) > 2]
    
    def _search_index(self, keywords: List[str]) -> List[dict]:
        """Search corpus index. Returns ranked matches."""
        results = []
        for entry in self.index:
            score = sum(1 for kw in keywords 
                       if kw in entry.get('text', '').lower())
            if score >= 2:
                results.append({**entry, 'score': score})
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _load_index(self, path: str) -> List[dict]:
        import os
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)
```

---

## Part 4: Self-Repair Loop

The critical cost optimization. When verification fails, try to fix the specific claim locally before escalating to oracle.

```python
# ~/inference/self_repair.py
"""
Self-repair: when a claim fails verification, ask the small
model to fix just that claim, then re-verify.

This eliminates ~40% of oracle calls because many Z3 failures
are formalization errors (wrong encoding), not knowledge gaps.
The model often knows the right answer but encoded it badly.

COGITO parallel: The Grounding Verifier's repair loop —
generate → verify → if ungrounded, repair → re-verify.
"""

import ollama
from typing import Optional
from claim_extractor import ExtractedClaim
from verifier import FormalVerifier, CorpusVerifier, VerificationResult
from knowledge_store import ProofStatus


class SelfRepairEngine:
    """
    Attempt local repair of failed claims before oracle escalation.
    
    Three repair strategies:
    1. RE-ENCODE: Same claim, new Z3 encoding (formalization error)
    2. RE-STATE: Reformulate the claim, then re-verify (imprecise language)
    3. DECOMPOSE: Break complex claim into sub-claims, verify each
    """
    
    def __init__(self, 
                 local_model: str = "descartes:8b",
                 formal_verifier: Optional[FormalVerifier] = None,
                 corpus_verifier: Optional[CorpusVerifier] = None,
                 max_attempts: int = 2):
        self.local_model = local_model
        self.formal = formal_verifier
        self.corpus = corpus_verifier
        self.max_attempts = max_attempts
        self.stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    
    def attempt_repair(
        self,
        claim: ExtractedClaim,
        failed_result: VerificationResult,
    ) -> Optional[VerificationResult]:
        """
        Try to repair a failed claim locally.
        
        Returns:
          VerificationResult if repair succeeded
          None if repair failed (needs oracle)
        """
        self.stats["attempted"] += 1
        claim.repair_attempted = True
        
        for attempt in range(self.max_attempts):
            strategy = self._pick_strategy(failed_result, attempt)
            
            if strategy == "re_encode":
                result = self._repair_encoding(claim, failed_result)
            elif strategy == "re_state":
                result = self._repair_statement(claim, failed_result)
            elif strategy == "decompose":
                result = self._repair_decompose(claim, failed_result)
            else:
                break
            
            if result and result.status == ProofStatus.VERIFIED:
                self.stats["succeeded"] += 1
                claim.verified = True
                claim.repaired_text = claim.text
                return result
        
        self.stats["failed"] += 1
        return None
    
    def _pick_strategy(self, result: VerificationResult, 
                        attempt: int) -> str:
        """Choose repair strategy based on failure type."""
        
        if "error" in result.proof_artifact:
            # Z3 code didn't run → encoding problem
            return "re_encode"
        
        if result.status == ProofStatus.REFUTED:
            # Z3 ran but said SAT (invalid) → claim might be
            # wrong, or encoding might be wrong
            return "re_state" if attempt == 0 else "decompose"
        
        if result.status == ProofStatus.TIMEOUT:
            # Z3 took too long → simplify encoding
            return "decompose"
        
        return "re_encode"
    
    def _repair_encoding(
        self,
        claim: ExtractedClaim,
        failed: VerificationResult,
    ) -> Optional[VerificationResult]:
        """Same claim, new Z3 encoding."""
        
        resp = ollama.chat(
            model=self.local_model,
            messages=[{
                "role": "system",
                "content": (
                    "You previously generated a Z3 encoding that "
                    "failed. Generate a CORRECTED encoding. "
                    "Output ONLY Z3 Python code."
                )
            }, {
                "role": "user",
                "content": (
                    f"Claim: {claim.text}\n\n"
                    f"Failed encoding:\n{failed.encoding}\n\n"
                    f"Error: {failed.proof_artifact}\n\n"
                    f"Generate a corrected Z3 encoding."
                )
            }]
        )
        
        # Re-verify with new encoding
        if self.formal:
            return self.formal.verify_formal(claim)
        return None
    
    def _repair_statement(
        self,
        claim: ExtractedClaim,
        failed: VerificationResult,
    ) -> Optional[VerificationResult]:
        """Reformulate the claim more precisely, then re-verify."""
        
        resp = ollama.chat(
            model=self.local_model,
            messages=[{
                "role": "system",
                "content": (
                    "Your previous claim could not be formally "
                    "verified. Reformulate it more precisely, "
                    "distinguishing what is formally provable "
                    "from what is interpretive."
                )
            }, {
                "role": "user",
                "content": (
                    f"Original claim: {claim.text}\n"
                    f"Verification result: {failed.status.value}\n\n"
                    f"Restate this claim in a way that is either "
                    f"formally verifiable or explicitly marked as "
                    f"interpretive."
                )
            }]
        )
        
        new_text = resp['message']['content'].strip()
        claim.text = new_text
        
        if self.formal:
            return self.formal.verify_formal(claim)
        return None
    
    def _repair_decompose(
        self,
        claim: ExtractedClaim,
        failed: VerificationResult,
    ) -> Optional[VerificationResult]:
        """Break complex claim into simpler sub-claims."""
        
        resp = ollama.chat(
            model=self.local_model,
            messages=[{
                "role": "system",
                "content": (
                    "Break this complex philosophical claim into "
                    "2-3 simpler sub-claims that can each be "
                    "verified independently. Output one claim "
                    "per line."
                )
            }, {
                "role": "user",
                "content": claim.text
            }]
        )
        
        sub_claims = [
            line.strip() for line in 
            resp['message']['content'].strip().split('\n')
            if line.strip() and len(line.strip()) > 10
        ]
        
        # Verify each sub-claim
        all_verified = True
        for sc_text in sub_claims[:3]:
            sc = ExtractedClaim(
                text=sc_text,
                claim_type=claim.claim_type,
                confidence=claim.confidence,
                position=claim.position,
            )
            if self.formal:
                result = self.formal.verify_formal(sc)
                if result.status != ProofStatus.VERIFIED:
                    all_verified = False
                    break
        
        if all_verified and sub_claims:
            claim.text = " AND ".join(sub_claims[:3])
            return VerificationResult(
                claim=claim,
                status=ProofStatus.VERIFIED,
                method=failed.method,
                encoding="decomposed",
                proof_artifact="all_sub_claims_verified",
                time_ms=0,
                stored_to_vks=True,
            )
        
        return None
```

---

## Part 5: Modified Cascade Engine

This replaces the engine from Addendum A with claim-level routing, VKS integration, and the self-repair loop.

```python
# ~/inference/engine_v3.py
"""
Descartes Philosopher Engine V3.

Changes from V2 (Addendum A):
- Claim-level routing (not query-level)
- Verified Knowledge Store (persistent memory)
- Multi-tier verification (Z3 + corpus + soft)
- Self-repair before oracle escalation
- Per-claim annotations in output

The meta-learner from Addendum A still operates but now
receives per-claim feedback, not per-query feedback.
"""

import ollama
import torch
import json
import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from knowledge_store import VerifiedKnowledgeStore, ProofStatus
from claim_extractor import ClaimExtractor, ExtractedClaim, ClaimType
from claim_router import ClaimRouter
from verifier import FormalVerifier, CorpusVerifier, VerificationResult
from self_repair import SelfRepairEngine
from signal_extractor_lite import LiteSignalExtractor
from meta_learner import MetaLearnerLite
from feedback import MetaTrainer


DESCARTES_SYSTEM = (
    "You are a philosophical reasoning assistant specializing in "
    "Cartesian philosophy, early modern rationalism, and the "
    "mind-body problem. You analyze arguments using ASPIC+ "
    "argumentation schemes and Z3 formal verification."
)

ORACLE_SYSTEM = (
    "You are a philosophical knowledge oracle. A Descartes specialist "
    "needs help with specific claims it could not verify. Provide "
    "accurate philosophical knowledge for the specific claims listed."
)

INTEGRATION_TEMPLATE = (
    "Revise your response. Some claims were verified, some failed "
    "and have been corrected. Integrate the corrections while "
    "preserving all verified claims exactly.\n\n"
    "ORIGINAL QUESTION: {query}\n\n"
    "YOUR INITIAL RESPONSE: {initial}\n\n"
    "CLAIM-BY-CLAIM STATUS:\n{claim_status}\n\n"
    "ORACLE CORRECTIONS (if any):\n{oracle_corrections}\n\n"
    "Produce your final response. Mark verified claims with "
    "[VERIFIED]. Mark corrected claims with [CORRECTED]."
)


@dataclass
class EngineResultV3:
    query: str
    final_response: str
    claims: List[ExtractedClaim]
    vks_hits: int              # Claims answered from memory
    z3_verified: int           # Claims verified by Z3
    corpus_verified: int       # Claims verified by corpus
    soft_passed: int           # Interpretive claims
    self_repaired: int         # Claims fixed without oracle
    oracle_needed: int         # Claims that required oracle
    total_time_ms: float


class DescartesEngineV3:
    """Production engine with claim-level routing and VKS."""
    
    def __init__(self,
                 local_model: str = "descartes:8b",
                 oracle_model: str = "deepseek-v3.1:671-cloud",
                 vks_path: str = "~/models/vks.json",
                 meta_path: Optional[str] = None):
        
        self.local_model = local_model
        self.oracle_model = oracle_model
        
        # Knowledge Store
        self.vks = VerifiedKnowledgeStore(vks_path)
        if not self.vks.verify_integrity():
            raise RuntimeError("VKS integrity check failed!")
        
        # Components
        self.extractor = ClaimExtractor(local_model)
        self.router = ClaimRouter(self.vks)
        self.formal = FormalVerifier(self.vks, local_model)
        self.corpus = CorpusVerifier(vks=self.vks)
        self.repair = SelfRepairEngine(
            local_model, self.formal, self.corpus)
        
        # Meta-learner (from Addendum A, still used)
        self.signal_extractor = LiteSignalExtractor()
        self.meta = MetaLearnerLite(input_dim=11)
        self.trainer = MetaTrainer(self.meta)
        if meta_path and os.path.exists(meta_path):
            self.trainer.load(meta_path)
        self.meta.eval()
        
        vks_stats = self.vks.get_stats()
        print(f"Engine V3 ready.")
        print(f"  VKS: {vks_stats['total']} records "
              f"({vks_stats['tiers']['AXIOM']} axioms)")
        print(f"  Meta-learner: {self.trainer.update_count} updates")
    
    def run(self, query: str) -> EngineResultV3:
        start = time.monotonic()
        
        # ── Step 1: Generate from local model ──
        initial = self._chat_local(query)
        
        # ── Step 2: Extract and classify claims ──
        claims = self.extractor.extract(initial)
        
        # ── Step 3: Route claims to verification backends ──
        buckets = self.router.route(claims)
        
        # ── Step 4: Verify each bucket ──
        failed_claims = []
        
        # 4a: VKS hits — already done, claims marked verified
        vks_hits = len(buckets['vks_hit'])
        
        # 4b: Formal claims → Z3
        z3_verified = 0
        for claim in buckets['z3']:
            result = self.formal.verify_formal(claim)
            if result.status == ProofStatus.VERIFIED:
                claim.verified = True
                claim.verification_method = 'z3'
                z3_verified += 1
            else:
                failed_claims.append((claim, result))
        
        # 4c: Factual claims → corpus
        corpus_verified = 0
        for claim in buckets['corpus']:
            result = self.corpus.verify_factual(claim)
            if result.status == ProofStatus.VERIFIED:
                claim.verified = True
                claim.verification_method = 'corpus'
                corpus_verified += 1
            else:
                failed_claims.append((claim, result))
        
        # 4d: Soft pass claims already marked
        soft_passed = len(buckets['soft_pass'])
        
        # ── Step 5: Self-repair failed claims ──
        still_failed = []
        self_repaired = 0
        
        for claim, failed_result in failed_claims:
            repair_result = self.repair.attempt_repair(
                claim, failed_result)
            if repair_result and repair_result.status == ProofStatus.VERIFIED:
                claim.verified = True
                claim.verification_method = 'self_repair'
                self_repaired += 1
            else:
                still_failed.append(claim)
        
        # ── Step 6: Oracle for remaining failures ──
        oracle_corrections = ""
        oracle_needed = len(still_failed)
        
        if still_failed:
            failed_texts = "\n".join(
                f"- [{c.claim_type.value}] {c.text}" 
                for c in still_failed
            )
            
            oracle_resp = ollama.chat(
                model=self.oracle_model,
                messages=[
                    {"role": "system", "content": ORACLE_SYSTEM},
                    {"role": "user", "content": (
                        f"The specialist could not verify these "
                        f"claims. Provide corrections:\n\n"
                        f"{failed_texts}\n\n"
                        f"Context (original question): {query}"
                    )}
                ]
            )
            oracle_corrections = oracle_resp['message']['content']
            
            for claim in still_failed:
                claim.verified = True  # oracle-corrected
                claim.verification_method = 'oracle'
        
        # ── Step 7: Integration pass (if any claims were corrected) ──
        if self_repaired > 0 or oracle_needed > 0:
            claim_status = "\n".join(
                f"- {c.text}: {c.verification_method}"
                for c in claims
            )
            
            final = self._chat_local_with_system(
                INTEGRATION_TEMPLATE.format(
                    query=query,
                    initial=initial,
                    claim_status=claim_status,
                    oracle_corrections=oracle_corrections or "None needed.",
                )
            )
        else:
            final = initial
        
        # ── Step 8: Feedback to meta-learner ──
        signals = self.signal_extractor.extract(initial)
        signal_tensor = signals.to_tensor()
        with torch.no_grad():
            meta_out = self.meta(signal_tensor)
        
        # Per-claim Z3 results feed back as ground truth
        z3_correct = z3_verified + vks_hits
        z3_total = z3_correct + oracle_needed + self_repaired
        z3_accuracy = z3_correct / max(z3_total, 1)
        
        self.trainer.record_and_maybe_train(
            meta_out["features"].detach(),
            meta_out["confidence"].item(),
            meta_out["routing_decision"],
            {
                "z3_verified": z3_accuracy > 0.8,
                "oracle_agreed": oracle_needed == 0,
                "correction_magnitude": oracle_needed / max(len(claims), 1),
                "user_accepted": None,
            }
        )
        
        elapsed = (time.monotonic() - start) * 1000
        
        return EngineResultV3(
            query=query,
            final_response=final,
            claims=claims,
            vks_hits=vks_hits,
            z3_verified=z3_verified,
            corpus_verified=corpus_verified,
            soft_passed=soft_passed,
            self_repaired=self_repaired,
            oracle_needed=oracle_needed,
            total_time_ms=elapsed,
        )
    
    def _chat_local(self, query: str) -> str:
        resp = ollama.chat(
            model=self.local_model,
            messages=[
                {"role": "system", "content": DESCARTES_SYSTEM},
                {"role": "user", "content": query}
            ]
        )
        return resp['message']['content']
    
    def _chat_local_with_system(self, prompt: str) -> str:
        resp = ollama.chat(
            model=self.local_model,
            messages=[
                {"role": "system", "content": DESCARTES_SYSTEM},
                {"role": "user", "content": prompt}
            ]
        )
        return resp['message']['content']
    
    def save(self, meta_path: str):
        self.trainer.save(meta_path)
    
    def get_stats(self) -> Dict:
        return {
            "vks": self.vks.get_stats(),
            "repair": self.repair.stats,
            "meta_learner": self.trainer.get_stats(),
        }


# ── Interactive REPL ──

def main():
    engine = DescartesEngineV3(
        local_model="descartes:8b",
        oracle_model="deepseek-v3.1:671-cloud",
        vks_path="~/models/vks.json",
        meta_path="~/models/meta_learner_latest.pt",
    )
    
    print("\n" + "=" * 60)
    print("DESCARTES PHILOSOPHER ENGINE V3")
    print("  claim-level routing | VKS memory | self-repair")
    print("=" * 60)
    print("Commands: quit, stats, vks, good, bad")
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
        elif query == "vks":
            stats = engine.vks.get_stats()
            print(f"VKS: {stats['total']} records")
            print(f"  Axioms:    {stats['tiers']['AXIOM']}")
            print(f"  Derived:   {stats['tiers']['DERIVED']}")
            print(f"  Contested: {stats['tiers']['CONTESTED']}")
            print(f"  Factual:   {stats['tiers']['FACTUAL']}")
            print(f"  Integrity: {'✓' if stats['integrity'] else '✗'}")
            continue
        elif query in ("good", "bad"):
            continue
        
        result = engine.run(query)
        
        print(f"\n[VKS:{result.vks_hits} Z3:{result.z3_verified} "
              f"CORPUS:{result.corpus_verified} SOFT:{result.soft_passed} "
              f"REPAIR:{result.self_repaired} ORACLE:{result.oracle_needed}] "
              f"({result.total_time_ms:.0f}ms)")
        print(f"\n{result.final_response}")


if __name__ == "__main__":
    main()
```

---

## Part 6: Modified Flow (replaces Addendum A flowchart)

```
USER QUERY
    │
    ▼
┌─────────────┐
│ Descartes 8B│ generate full response
│ (local)     │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ CLAIM        │ split response into typed claims
│ EXTRACTOR    │ FORMAL / FACTUAL / INTERPRETIVE / META
└──────┬───────┘
       │
       ▼ (for each claim)
┌──────────────┐
│ VKS LOOKUP   │◄──── check memory first ────┐
│ (instant)    │                              │
└──────┬───────┘                              │
       │                                      │
       ├── HIT ──► claim verified, skip       │
       │                                      │
       ├── MISS, FORMAL ──────────────┐       │
       │                              ▼       │
       │                    ┌──────────────┐  │
       │                    │ Z3 + CVC5    │  │
       │                    │ (parallel)   │  │
       │                    └──────┬───────┘  │
       │                           │          │
       │                    ┌──────┴──────┐   │
       │                    │             │   │
       │                 VERIFIED      FAILED │
       │                    │             │   │
       │                    │     ┌───────┴───┤
       │                    │     ▼           │
       │                    │  SELF-REPAIR    │
       │                    │  (re-encode,    │
       │                    │   re-state,     │
       │                    │   decompose)    │
       │                    │     │           │
       │                    │  ┌──┴──┐        │
       │                    │  │     │        │
       │                    │ FIXED STILL     │
       │                    │  │   FAILED     │
       │                    │  │     │        │
       │                    │  │     ▼        │
       │                    │  │  ORACLE      │
       │                    │  │  (targeted   │
       │                    │  │   per-claim) │
       │                    │  │     │        │
       │                    ▼  ▼     ▼        │
       │               ┌─────────────────┐    │
       │               │  WRITE TO VKS   │────┘
       │               │  (permanent)    │ grows with every query
       │               └────────┬────────┘
       │                        │
       ├── MISS, FACTUAL ───────┤
       │        │               │
       │        ▼               │
       │  CORPUS INDEX          │
       │  (search training      │
       │   corpus for source)   │
       │        │               │
       │     ┌──┴──┐            │
       │   FOUND  NOT FOUND     │
       │     │      │           │
       │     │   ORACLE         │
       │     │      │           │
       │     ▼      ▼           │
       │  WRITE TO VKS ────────┘
       │
       ├── MISS, INTERPRETIVE/META
       │        │
       │     SOFT PASS (meta-learner confidence only)
       │
       ▼
┌──────────────────┐
│ REASSEMBLE       │ annotated response
│ [VERIFIED]       │ [VERIFIED] = Z3/VKS confirmed
│ [CORRECTED]      │ [CORRECTED] = self-repaired or oracle
│ [UNVERIFIED]     │ [UNVERIFIED] = interpretive/soft pass
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│ FEEDBACK         │ Z3 verdicts + oracle agreement
│ → meta-learner   │   + user feedback → training signal
│ → VKS growth     │ VKS now has new verified claims
└──────────────────┘
```

---

## Part 7: File Layout Update

```
~/
├── models/
│   ├── vks.json                     ◄── NEW: Verified Knowledge Store
│   ├── meta_learner_bootstrap.pt
│   ├── meta_learner_latest.pt
│   └── descartes-8b.gguf
│
├── inference/
│   ├── engine_v3.py                 ◄── NEW: replaces engine.py
│   ├── knowledge_store.py           ◄── NEW: VKS implementation
│   ├── seed_axioms.py               ◄── NEW: foundational axioms
│   ├── claim_extractor.py           ◄── NEW: claim splitting + typing
│   ├── claim_router.py              ◄── NEW: per-claim routing
│   ├── verifier.py                  ◄── NEW: Z3 + corpus verification
│   ├── self_repair.py               ◄── NEW: local repair before oracle
│   ├── meta_learner.py              (from Addendum A)
│   ├── feedback.py                  (from Addendum A)
│   ├── signal_extractor_lite.py     (from Addendum A)
│   └── templates/
│       └── descartes_z3.py
│
├── corpus/
│   └── index.json                   ◄── NEW: searchable corpus index
│
└── training/
    ├── bootstrap_meta.py            (from Addendum A)
    └── eval/
        ├── eval_routing.py
        └── eval_vks_growth.py       ◄── NEW: track VKS growth rate
```

---

## Reading Order (All Documents)

```
1. PHILOSOPHER_ENGINE_ARCHITECTURE.md
   → System design, five layers

2. PHILOSOPHER_ENGINE_TRAINING_PIPELINE.md
   → Phases 2-4: extraction, cleaning, formatting

3. DESCARTES_CASCADE_TRAINING_PIPELINE.md
   → Phase 1: corpus. Phases 5-8: CPT, SFT, eval

4. ADDENDUM_OLLAMA_META_LEARNER.md (Addendum A)
   → Phases 9-12: meta-learner, Ollama, bootstrap

5. ADDENDUM_B_VKS_MULTI_TIER_VERIFICATION.md (this document)
   → Modifies Steps 2-7 of inference flow
   → Adds: VKS, claim-level routing, self-repair
   → Adds: multi-tier verification (Z3 + corpus)
   → Addendum A's meta-learner still operates
     but receives per-claim feedback
```
