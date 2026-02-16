"""
Verified Knowledge Store — persistent, tiered, hash-chained.

Grows with every verification. Axioms are permanent.
Derived theorems compose from axioms. System gets smarter over time.

Architecture inspired by COGITO v2.1 EMT semantic memory:
  - Axioms = EMT leaf nodes (append-only, immutable, hash-chained)
  - Derived = proven compositions from axioms
  - Contested = valid under stated premises only
  - Factual = corpus-verified historical claims

Integrity: hash chain on axiom tier; corruption = identity broken.
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
        self.claim_index: Dict[str, str] = {}  # normalized_text -> claim_id
        self.axiom_chain: List[str] = []        # ordered axiom hashes
        self._load()

    # -- QUERY --

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
        """All Tier 1 axioms -- for building derived proofs."""
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

    # -- WRITE --

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

        NEVER called on Tier 1 axioms -- if an axiom is wrong,
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

    # -- INTEGRITY --

    def verify_integrity(self) -> bool:
        """Check axiom hash chain on startup."""
        if not self.axiom_chain:
            return True

        axiom_records = [
            r for r in self.records.values()
            if r.tier == Tier.AXIOM
        ]
        axiom_records.sort(key=lambda r: r.timestamp)

        if len(axiom_records) != len(self.axiom_chain):
            print(f"INTEGRITY FAILURE: {len(axiom_records)} axiom records "
                  f"vs {len(self.axiom_chain)} chain entries")
            return False

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

    # -- PERSISTENCE --

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
        stop = {"the", "a", "an", "is", "are", "in", "of", "that",
                "this", "it"}
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
