# PHILOSOPHER ENGINE V3: Unified Architecture
## Verified Cascade System for Cartesian Philosophical Reasoning

**Version:** 3.0
**Date:** February 16, 2026
**Project Owner:** Charith
**Status:** Complete Technical Specification

**This document supersedes and unifies:**
1. `PHILOSOPHER_ENGINE_ARCHITECTURE.md` — original five-layer design
2. `DESCARTES_CASCADE_TRAINING_PIPELINE.md` — cascade training (Phases 1-8)
3. `ADDENDUM_OLLAMA_META_LEARNER.md` — meta-learner + Ollama (Phases 9-12)
4. `ADDENDUM_B_VKS_MULTI_TIER_VERIFICATION.md` — VKS + claim routing + self-repair

---

## Executive Summary

The Philosopher Engine is a neuro-symbolic cascade system that formally verifies philosophical arguments in Cartesian philosophy and the mind-body problem. It combines a domain-specialized small language model (Descartes 8B) with a cloud oracle, Z3 formal verification, corpus-based fact checking, and a persistent Verified Knowledge Store — orchestrated by a meta-learner that routes individual claims to the appropriate verification backend.

The core architectural insight: no single component handles philosophical reasoning alone. The small model generates fluent domain-specific analysis. Z3 provides binary ground truth for formal claims. The corpus index verifies historical attributions. The oracle fills genuine knowledge gaps. The meta-learner learns which claims need which treatment. And the VKS remembers everything that's been verified, so the system never re-derives a settled result.

**What's novel (vs. existing cascade systems):**

| Feature | GPT-5 | RouteLLM | AutoMix | C3PO | Ours |
|---------|-------|----------|---------|------|------|
| Cascade routing | ✓ | ✓ | ✓ | ✓ | ✓ |
| Domain-specialized small model via CPT | ✗ | ✗ | ✗ | ✗ | ✓ |
| Hidden state signals for routing | ? | ✗ | ✗ | ✗ | ✓ |
| Formal verification in routing loop | ✗ | ✗ | ✗ | ✗ | ✓ |
| Persistent verified knowledge store | ✗ | ✗ | ✗ | ✗ | ✓ |
| Claim-level routing (not query-level) | ✗ | ✗ | ✗ | ✗ | ✓ |
| Self-repair before oracle escalation | ✗ | ✗ | ✗ | ✗ | ✓ |
| Error-type-aware oracle queries | ✗ | ✗ | ✗ | ✗ | ✓ |
| Integration pass (specialist has final word) | ✗ | ✗ | ✗ | ✗ | ✓ |

---

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PHILOSOPHER ENGINE V3                                 │
│                                                                              │
│  ┌─────────────────────────────── INFERENCE ──────────────────────────────┐  │
│  │                                                                        │  │
│  │   USER QUERY                                                           │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │   ┌───────────────┐    ┌────────────────┐    ┌─────────────────────┐   │  │
│  │   │  Descartes 8B  │───►│ CLAIM EXTRACTOR │───►│ VKS LOOKUP (first) │   │  │
│  │   │  (Ollama local)│    │ FORMAL/FACTUAL/ │    │ "Already verified?" │   │  │
│  │   └───────────────┘    │ INTERPRETIVE/META│    └────────┬────────────┘   │  │
│  │          │              └────────────────┘              │               │  │
│  │    Signal Extractor                           ┌────────┴────────┐      │  │
│  │    (hidden states,                            │                 │      │  │
│  │     entropy,                               HIT              MISS      │  │
│  │     attention)                          (skip!)          (verify)     │  │
│  │          │                                 │                 │         │  │
│  │          ▼                                 │    ┌────────────┴───┐     │  │
│  │   ┌──────────────┐                         │    │                │     │  │
│  │   │ META-LEARNER │                         │    ▼                ▼     │  │
│  │   │ per-claim    │                    ┌─────────────┐   ┌───────────┐  │  │
│  │   │ confidence   │                    │ Z3 + CVC5   │   │ CORPUS    │  │  │
│  │   └──────────────┘                    │ (parallel)  │   │ INDEX     │  │  │
│  │                                       └──────┬──────┘   └─────┬─────┘  │  │
│  │                                              │                │        │  │
│  │                                        ┌─────┴─────┐         │        │  │
│  │                                        │           │         │        │  │
│  │                                     VERIFIED    FAILED       │        │  │
│  │                                        │           │         │        │  │
│  │                                        │     ┌─────┴──────┐  │        │  │
│  │                                        │     │SELF-REPAIR │  │        │  │
│  │                                        │     │re-encode   │  │        │  │
│  │                                        │     │re-state    │  │        │  │
│  │                                        │     │decompose   │  │        │  │
│  │                                        │     └──┬───┬─────┘  │        │  │
│  │                                        │     FIXED  STILL    │        │  │
│  │                                        │        │   FAILED   │        │  │
│  │                                        │        │     │      │        │  │
│  │                                        │        │     ▼      │        │  │
│  │                                        │        │  ┌──────┐  │        │  │
│  │                                        │        │  │ORACLE│  │        │  │
│  │                                        │        │  │(cloud)│  │        │  │
│  │                                        │        │  └──┬───┘  │        │  │
│  │                                        ▼        ▼     ▼      ▼        │  │
│  │                                   ┌────────────────────────────┐       │  │
│  │                                   │       WRITE TO VKS        │       │  │
│  │                                   │    (permanent memory)     │       │  │
│  │                                   └────────────┬───────────────┘       │  │
│  │                                                │                      │  │
│  │                                                ▼                      │  │
│  │                                   ┌────────────────────────────┐       │  │
│  │                                   │  INTEGRATION PASS          │       │  │
│  │                                   │  Descartes 8B re-generates │       │  │
│  │                                   │  with verified + corrected │       │  │
│  │                                   │  claims. Specialist has    │       │  │
│  │                                   │  final word.              │       │  │
│  │                                   └────────────┬───────────────┘       │  │
│  │                                                │                      │  │
│  │                                                ▼                      │  │
│  │                                   ANNOTATED RESPONSE                  │  │
│  │                                   [VERIFIED] [CORRECTED] [UNVERIFIED] │  │
│  │                                                │                      │  │
│  │                                                ▼                      │  │
│  │                                   FEEDBACK → meta-learner training    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────── REASONING CORE (5 Layers) ────────────────────────┐  │
│  │                                                                        │  │
│  │  Layer 1          Layer 2          Layer 3          Layer 4   Layer 5  │  │
│  │  ONTOLOGY         ASPIC+           Z3 ENGINE        LLM       CONCEPT │  │
│  │  (OWL 2)          ARGUMENT         (Modal,          BRIDGE    SPACES  │  │
│  │                   STRUCTURE        Paracons.,       (GVR      (Geom.) │  │
│  │  Sorts,           Walton           Defeasible)      Loop)             │  │
│  │  Relations,       Schemes,                                            │  │
│  │  Theories         Attacks          NOW: multi-tier  NOW: cascade      │  │
│  │  [PRESERVED]      [PRESERVED]      + VKS + corpus   + claim routing   │  │
│  │                                    [MODIFIED]       [MODIFIED]         │  │
│  │                                                                 [PRES]│  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                  Microtheory Manager                            │   │  │
│  │  │  [Physicalism] [Functionalism] [Dualism] [IIT] [GWT] [HOT]    │   │  │
│  │  │  Separate Z3 contexts • Cross-theory consistency [PRESERVED]   │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────── PERSISTENT STORAGE ───────────────────────────────────┐  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │              VERIFIED KNOWLEDGE STORE (VKS)                      │  │  │
│  │  │                                                                  │  │  │
│  │  │  Tier 1: AXIOMS (permanent, hash-chained, append-only)          │  │  │
│  │  │  ├── cogito_valid (Doubts→Thinks→Exists, Z3: UNSAT)            │  │  │
│  │  │  ├── real_distinction_s5 (conceivability→non-identity, UNSAT)   │  │  │
│  │  │  ├── cartesian_circle_structure (CDP→God→CDP, circular)         │  │  │
│  │  │  └── ... grows with novel verifications                         │  │  │
│  │  │                                                                  │  │  │
│  │  │  Tier 2: DERIVED THEOREMS (proven from axioms, re-derivable)    │  │  │
│  │  │  ├── zombie_isomorphic_real_distinction                         │  │  │
│  │  │  └── ... grows as system encounters new questions               │  │  │
│  │  │                                                                  │  │  │
│  │  │  Tier 3: CONTESTED CLAIMS (valid under stated premises only)    │  │  │
│  │  │  ├── ontological_arg: VALID IF existence-is-predicate           │  │  │
│  │  │  └── ...                                                        │  │  │
│  │  │                                                                  │  │  │
│  │  │  Tier 4: FACTUAL (corpus-verified historical attributions)      │  │  │
│  │  │  ├── arnauld_fourth_objections_circularity                      │  │  │
│  │  │  └── ...                                                        │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  Corpus Index (searchable Descartes texts)                            │  │
│  │  Meta-Learner Checkpoints                                             │  │
│  │  Feedback Buffer (training signal accumulator)                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Part I: Reasoning Core (5 Layers)

### Layer 1: Philosophical Ontology — PRESERVED

**Full specification:** See `PHILOSOPHER_ENGINE_ARCHITECTURE.md`, Layer 1.

OWL 2 ontology defining the formal vocabulary for philosophy of mind. Sorts (World, Property, Subject, State, Experience, Process), relations (HasProperty, CausesState, Supervenes, AccessibleFrom), and theory commitment axioms (Physicalism, Functionalism, Dualism, IIT, GWT, HOT) are unchanged.

Key components preserved without modification:
- `OntologySorts` — Z3 sorts for the philosophical domain
- `OntologyRelations` — Z3 relations between entities
- `TheoryCommitments` — axiomatized positions for each theory
- OWL 2 taxonomy with Zalta's dual predication

The ontology defines what the system can talk about. Everything downstream — argumentation, verification, formalizations — uses this vocabulary.

---

### Layer 2: Argument Structure (ASPIC+ / AIF) — PRESERVED

**Full specification:** See `PHILOSOPHER_ENGINE_ARCHITECTURE.md`, Layer 2.

ASPIC+ knowledge base with Walton argumentation schemes. Captures how philosophical arguments are organized — premises, inference steps, attack types (rebuts, undermines, undercuts), and support relations.

Key components preserved without modification:
- `ASPICKnowledgeBase` — premises, rules, attack types, preferences
- Walton scheme templates — Analogy, Composition, Sign, Expert Opinion, Consequences, Best Explanation
- Zombie argument, Chinese Room, Knowledge Argument decompositions
- AIF export/import for interoperability

The argument layer structures what the system reasons about. It feeds formal claims to Layer 3 for verification and provides the structural backbone for the claim extractor.

---

### Layer 3: Verification Engine — MODIFIED

**Original specification:** `PHILOSOPHER_ENGINE_ARCHITECTURE.md`, Layer 3.
**What changed:** Z3 is no longer the sole verifier. Layer 3 is now a multi-tier verification engine backed by the Verified Knowledge Store. Claims are verified by the appropriate backend: Z3/CVC5 for formal logic, corpus index for historical facts, soft pass for interpretive claims. All results persist in the VKS.

#### 3.1 Multi-Logic Z3 Backend — PRESERVED

The three-mode Z3 engine is unchanged:

| Mode | Logic | Use Case |
|------|-------|----------|
| **Modal** | S5/S4/KB Kripke semantics | Conceivability arguments, possible worlds, necessity/possibility |
| **Paraconsistent** | Belnap 4-valued (T/F/Both/Neither) | Analyzing contradictory positions without explosion |
| **Defeasible** | Weighted soft constraints + priorities | Presumptive reasoning, ceteris paribus, defeating conditions |

Encodings preserved: `ModalLogicEngine`, `ParaconsistentEngine`, `DefeasibleEngine`, `MicrotheoryManager`, `UnsatCoreExtractor`, Maximal Consistent Subsets.

The full Z3 code (630+ lines) is in the original architecture document and remains the formal reasoning backbone.

#### 3.2 CVC5 Parallel Verification — NEW

CVC5 runs in parallel with Z3 on formal claims. Standard practice in formal verification: different solvers have different strengths, and running both catches cases where one times out but the other solves instantly.

```python
import concurrent.futures

def verify_parallel(z3_code: str, cvc5_code: str, 
                    timeout_s: float = 30.0) -> str:
    """Run Z3 and CVC5 in parallel. First definitive result wins."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        z3_future = pool.submit(run_z3, z3_code, timeout_s)
        cvc5_future = pool.submit(run_cvc5, cvc5_code, timeout_s)
        
        done, pending = concurrent.futures.wait(
            [z3_future, cvc5_future],
            return_when=concurrent.futures.FIRST_COMPLETED,
            timeout=timeout_s
        )
        
        for future in done:
            result = future.result()
            if result in ("sat", "unsat"):
                for p in pending:
                    p.cancel()
                return result
        
        # Both finished without definitive result
        return "timeout"
```

CVC5 adds ~5MB dependency and handles certain quantifier patterns and bitvector theories better than Z3. For our use case (modal logic, first-order philosophy), Z3 is primary. CVC5 is insurance.

#### 3.3 Corpus Index Verification — NEW

Factual claims about Descartes' texts ("Arnauld raised the circularity objection in the Fourth Objections") are verified against a searchable index of the training corpus.

```python
class CorpusVerifier:
    """Verify historical/attributional claims against source texts."""
    
    def __init__(self, index_path: str = "~/corpus/index.json"):
        self.index = self._load_index(index_path)
    
    def verify(self, claim: str) -> VerificationResult:
        keywords = self._extract_keywords(claim)
        matches = self._search_index(keywords)
        
        if matches and matches[0]['score'] >= 2:
            return VerificationResult(
                status=ProofStatus.VERIFIED,
                method=VerificationMethod.CORPUS,
                artifact=f"corpus:{matches[0]['source']}",
            )
        return VerificationResult(
            status=ProofStatus.NOT_FORMALIZABLE,
            method=VerificationMethod.CORPUS,
            artifact="no_corpus_match",
        )
```

The corpus index is built during Phase 2-3 of the training pipeline (text extraction and cleaning). Each entry contains source reference (e.g. "Meditations, Third Meditation, AT VII 40"), passage text, and keyword list. Index format is JSON for simplicity; production could use FAISS for semantic search.

#### 3.4 Verified Knowledge Store — NEW

The VKS is the system's persistent memory. Every verification result — Z3 proof, corpus match, oracle confirmation — gets stored permanently. On every new query, the VKS is checked first. If a claim (or its logical equivalent) was verified before, the stored result is returned instantly.

**Design principle from COGITO v2.1:** "Memory loss isn't death; belief corruption is." The VKS follows the same rule: losing a derived theorem just means re-deriving it next time. Corrupting a Tier 1 axiom (marking a verified-valid argument as invalid) breaks the entire dependency chain. So Tier 1 is append-only, immutable, hash-chained.

```python
class VerifiedKnowledgeStore:
    """
    Persistent, tiered, hash-chained knowledge base.
    Tier 1: AXIOMS — permanent, append-only, hash-chained
    Tier 2: DERIVED — proven from axioms, re-derivable
    Tier 3: CONTESTED — valid under stated premises only
    Tier 4: FACTUAL — corpus-verified historical claims
    """
    
    def lookup(self, claim_text: str) -> Optional[ProofRecord]:
        """Check VKS before running any verifier."""
        normalized = self._normalize(claim_text)
        if normalized in self.claim_index:
            return self.records[self.claim_index[normalized]]
        # Fuzzy match on Jaccard similarity > 0.75
        for record in self.records.values():
            if self._semantically_equivalent(normalized, record):
                return record
        return None
    
    def get_lemmas_for(self, claim: str) -> List[ProofRecord]:
        """Find proven claims to help prove new ones."""
        # Keyword overlap, sorted by tier (axioms first)
        ...
    
    def store(self, record: ProofRecord):
        """Write verified claim. Axioms are hash-chained."""
        if record.tier == Tier.AXIOM:
            # Append to immutable chain
            record.hash = chain_hash(self.axiom_chain[-1], record)
            self.axiom_chain.append(record.hash)
        self.records[record.claim_id] = record
        self._save()
    
    def invalidate(self, claim_id: str):
        """Cascade-invalidate a claim and all dependents."""
        # NEVER called on Tier 1 — axioms are immutable
        record = self.records[claim_id]
        record.status = ProofStatus.REFUTED
        for dep_id in record.used_by:
            self.invalidate(dep_id)
    
    def verify_integrity(self) -> bool:
        """Check axiom hash chain on startup."""
        ...
```

**Seed axioms** verified at system initialization:

| Axiom | Encoding | Z3 Result | Tier |
|-------|----------|-----------|------|
| Cogito (strict inference) | Doubts(ego) ∧ ¬Exists(ego) | UNSAT | 1 |
| Real Distinction (S5) | Conceivability + Identity thesis | UNSAT | 1 |
| Cartesian Circle (structure) | CDP→God→CDP + transitivity | UNSAT (circular) | 1 |
| Ontological Argument | Perfection + existence-as-predicate | SAT (conditional) | 3 |

The VKS grows with every query. After 1000 queries, the system has accumulated hundreds of verified theorems and factual attributions. Queries that would have required Z3 verification or oracle consultation become instant cache hits.

**Full ProofRecord schema, seed axiom code, and persistence implementation:** See `ADDENDUM_B_VKS_MULTI_TIER_VERIFICATION.md`, Parts 1-2.

---

### Layer 4: LLM Translation Bridge — MODIFIED

**Original specification:** `PHILOSOPHER_ENGINE_ARCHITECTURE.md`, Layer 4.
**What changed:** The GVR (Generate-Verify-Regenerate) loop is now embedded inside the cascade engine. The LLM is no longer a generic API call — it's a domain-specialized Descartes 8B model running locally via Ollama, with a cloud oracle for knowledge gaps. Routing happens at claim level, not query level.

#### 4.1 Generate-Verify-Regenerate Loop — PRESERVED (embedded in cascade)

The core GVR concept is unchanged: the LLM proposes formalizations, Z3 validates them, error feedback drives iterative refinement. What's different is that GVR now operates per-claim within a single response, not on the full response as a unit.

Original GVR (v1):
```
Query → LLM generates full response → Z3 checks everything → 
if fail, re-generate everything → repeat up to N times
```

Modified GVR (v3):
```
Query → LLM generates full response → CLAIM EXTRACTOR splits into 
individual claims → each claim routed to appropriate verifier → 
failed claims get SELF-REPAIR (targeted re-generation) → 
still-failed claims get ORACLE help → INTEGRATION PASS
```

The improvement: instead of throwing away the entire response when one claim fails, we keep the verified parts and fix only the broken parts. This preserves good reasoning while correcting errors.

#### 4.2 Descartes 8B Specialist — NEW

The small model is domain-specialized via Continued Pre-Training (CPT) and two-stage Supervised Fine-Tuning (SFT) on Cartesian philosophy.

```
Base: Qwen3-8B (8B parameters)
CPT:  100-500M tokens of Descartes corpus
      (Meditations, Objections/Replies, Correspondence, 
       rationalist tradition, Spinoza, Leibniz)
SFT Stage 1: Reasoning (Types A-D)
      A: Z3 formalization pairs
      B: ASPIC+ decomposition examples
      C: Cross-theory comparison chains
      D: Countermodel-driven revision traces
SFT Stage 2: Cascade (Types E-G)
      E: Confidence estimation training
      F: Oracle query formation
      G: Integration pass examples

Hardware: 1x A40 48GB (Vast.ai, ~$0.50/hr)
Cost:     $225-$735 total
Time:     96-165 hours part-time over 2-3 weeks
```

Why 8B instead of 70B: the Descartes corpus (100-500M tokens) provides 12.5+ tokens per parameter for an 8B model — full saturation. The same corpus gives a 70B model only 1.4 tokens/param — insufficient to shift weights meaningfully. The small model develops genuine domain expertise. The large model provides broad knowledge at inference time via the cascade.

**Full CPT/SFT specifications:** See `DESCARTES_CASCADE_TRAINING_PIPELINE.md`, Phases 5-7.

#### 4.3 Cloud Oracle — NEW

When the small model fails and self-repair also fails, the oracle provides broad knowledge. Accessed via Ollama Cloud for API uniformity.

```python
import ollama

# Same API for local and cloud — one client, two models
local_resp = ollama.chat(model="descartes:8b", messages=[...])
oracle_resp = ollama.chat(model="deepseek-v3.1:671-cloud", messages=[...])
```

The oracle receives targeted queries shaped by the error type:

| Error Type | Oracle Query Pattern |
|---|---|
| FACTUAL_GAP | "What factual information about {topic} is the specialist missing?" |
| REASONING_ERROR | "The specialist's argument has this flaw: {flaw}. What's the correct reasoning?" |
| FORMALIZATION_ERROR | "Check this Z3 encoding for errors: {encoding}" |
| SCOPE_EXCEEDED | "The question extends beyond Cartesian philosophy into {domain}. Provide relevant context." |

**Full Ollama setup, cloud configuration, and oracle integration:** See `ADDENDUM_OLLAMA_META_LEARNER.md`, Parts 1-2.

---

### Layer 5: Conceptual Spaces — PRESERVED

**Full specification:** See `PHILOSOPHER_ENGINE_ARCHITECTURE.md`, Layer 5.

Geometric complement to the logical layers. Embeds 14 consciousness theories in a multi-dimensional space where similarity = proximity. Handles graded judgments that resist binary formalization ("how functionalist is IIT?").

Key components preserved without modification:
- `ConceptualSpace` with quality dimensions (accessibility, integration, higher-order, intentionality, temporal, substrate)
- Theory embeddings for Physicalism, Functionalism, Dualism, IIT, GWT, HOT, RPT, AST, etc.
- Distance metrics and nearest-position queries
- Dimensional analysis for argument classification

The conceptual space provides the "soft" complement to Z3's "hard" verification. When a query involves comparing theoretical positions or assessing similarity between arguments, Layer 5 handles what Z3 cannot.

---

### Cross-Cutting: Microtheory Manager — PRESERVED

Separate Z3 contexts for competing philosophical positions. Enables cross-theory consistency checking without logical explosion (asserting both physicalist and dualist axioms in the same context would trivialize everything).

```python
class MicrotheoryManager:
    def __init__(self):
        self.theories = {
            "physicalism": Solver(),
            "property_dualism": Solver(), 
            "functionalism": Solver(),
            "IIT": Solver(),
            "GWT": Solver(),
            "HOT": Solver(),
        }
        # Each solver gets its own theory-specific axioms
        self._load_theory_axioms()
    
    def check_cross_theory_consistency(self, claim):
        """Test claim against each theory independently."""
        results = {}
        for name, solver in self.theories.items():
            solver.push()
            solver.add(claim)
            results[name] = solver.check()
            solver.pop()
        return results
```

---

## Part II: Inference Engine

### 5. Claim Extractor

The claim extractor replaces monolithic query routing. Instead of asking "should this entire query go to the oracle?", it asks "which specific claims in this response need which verification?"

#### 5.1 Claim Types

Adapted from COGITO's Grounding Verifier taxonomy, specialized for philosophical argument analysis:

| Type | Description | Verification Backend | Example |
|------|-------------|---------------------|---------|
| **FORMAL** | Logic claims: validity, consistency, entailment, modal | Z3/CVC5 or VKS lookup | "The Real Distinction is valid in S5" |
| **FACTUAL** | Historical/attributional claims | Corpus index | "Arnauld raised this in the Fourth Objections" |
| **INTERPRETIVE** | Scholarly judgment, readings, comparisons | Soft pass (confidence only) | "Descartes intended this as a response to Gassendi" |
| **META_PHILOSOPHICAL** | Structural/significance claims | Soft pass | "This is the strongest objection to substance dualism" |

#### 5.2 Classification Logic

```python
class ClaimExtractor:
    """Split LLM response into individual typed claims."""
    
    def extract(self, response_text: str) -> List[ExtractedClaim]:
        sentences = self._split_sentences(response_text)
        claims = []
        for sentence in sentences:
            ctype = self._classify(sentence)
            claims.append(ExtractedClaim(
                text=sentence,
                claim_type=ctype,
                confidence=self._confidence(sentence, ctype),
            ))
        return claims
    
    def _classify(self, sentence: str) -> ClaimType:
        """Regex-based classification with indicator patterns."""
        
        formal_score = count_matches(sentence, [
            r'\b(valid|invalid|consistent|inconsistent)\b',
            r'\b(entails|follows from|implies)\b',
            r'\bS[45]\b', r'\b(modal|possible world)\b',
            r'\b(conceivability|conceivable)\b.*\b(possibility|possible)\b',
        ])
        
        factual_score = count_matches(sentence, [
            r'\b(Arnauld|Gassendi|Hobbes|Elisabeth)\b.*\b(argued|wrote|objected)\b',
            r'\b(in the|from the)\b.*\b(Meditations?|Objections?|Replies)\b',
            r'\b(First|Second|Third|Fourth|Fifth|Sixth)\b.*\b(Meditation|Objection)\b',
        ])
        
        if formal_score > factual_score:
            return ClaimType.FORMAL
        elif factual_score > 0:
            return ClaimType.FACTUAL
        else:
            return ClaimType.INTERPRETIVE
```

**Full claim extractor with all indicator patterns and confidence scoring:** See `ADDENDUM_B_VKS_MULTI_TIER_VERIFICATION.md`, Part 2.

---

### 6. Claim Router

Routes each extracted claim to the appropriate verification backend, always checking the VKS first.

```python
class ClaimRouter:
    def route(self, claims: List[ExtractedClaim]) -> Dict[str, List]:
        buckets = {'vks_hit': [], 'z3': [], 'corpus': [], 'soft_pass': []}
        
        for claim in claims:
            # ALWAYS check VKS first
            existing = self.vks.lookup(claim.text)
            if existing and existing.status in (VERIFIED, CONDITIONAL):
                claim.verified = True
                claim.vks_hit = True
                buckets['vks_hit'].append(claim)
                continue
            
            # Route by type
            if claim.claim_type == FORMAL:
                buckets['z3'].append(claim)
            elif claim.claim_type == FACTUAL:
                buckets['corpus'].append(claim)
            else:
                claim.verified = True  # soft pass
                buckets['soft_pass'].append(claim)
        
        return buckets
```

The routing priority is always: VKS (instant) → appropriate verifier → self-repair → oracle (last resort).

---

### 7. Self-Repair Engine

When a claim fails verification, the self-repair engine tries to fix it locally before escalating to the oracle. This eliminates ~40% of oracle calls because many Z3 failures are formalization errors (wrong encoding), not genuine knowledge gaps.

#### 7.1 Three Repair Strategies

| Strategy | When | What It Does |
|----------|------|-------------|
| **RE-ENCODE** | Z3 code error, syntax failure | Keep the claim, generate a new Z3 encoding |
| **RE-STATE** | Z3 says REFUTED (SAT found) | Reformulate the claim more precisely, then re-verify |
| **DECOMPOSE** | Z3 timeout on complex claim | Break into 2-3 simpler sub-claims, verify each independently |

#### 7.2 Repair Flow

```python
class SelfRepairEngine:
    def attempt_repair(self, claim, failed_result, 
                       max_attempts=2) -> Optional[VerificationResult]:
        for attempt in range(max_attempts):
            strategy = self._pick_strategy(failed_result, attempt)
            
            if strategy == "re_encode":
                # Show model the failed encoding + error
                # Ask for corrected Z3 code
                result = self._repair_encoding(claim, failed_result)
            
            elif strategy == "re_state":
                # Ask model to reformulate claim more precisely
                # Distinguish formally provable from interpretive
                result = self._repair_statement(claim, failed_result)
            
            elif strategy == "decompose":
                # Break complex claim into simpler sub-claims
                # Verify each independently
                result = self._repair_decompose(claim, failed_result)
            
            if result and result.status == VERIFIED:
                return result
        
        return None  # repair failed → escalate to oracle
```

**COGITO parallel:** This mirrors the Grounding Verifier's repair loop from the LLM Wrapper spec — generate → verify → if ungrounded, repair → re-verify. The pattern is: never throw away the whole response for a local failure.

**Full self-repair implementation with all three strategies:** See `ADDENDUM_B_VKS_MULTI_TIER_VERIFICATION.md`, Part 4.

---

### 8. Meta-Learner

The meta-learner observes the model's internal states and learns to predict actual reliability. It replaces self-reported confidence (unreliable — models confidently hallucinate) with measured signals from hidden activations.

#### 8.1 Architecture

```
Input (11 features):
├── hidden_mean        — mean activation magnitude
├── hidden_std         — activation variance (diffuse = uncertain)
├── token_entropy      — per-token prediction uncertainty
├── attention_entropy  — attention dispersion
├── hedge_ratio        — fraction of hedging words
├── repetition_ratio   — word repetition rate
├── response_length    — token count (normalized)
├── ood_score          — distributional distance from training data
├── question_complexity — input complexity estimate
├── formal_claim_ratio — fraction of claims that are formal
└── factual_claim_ratio — fraction that are factual

Hidden layers: 128 → 64 → 32 (ReLU, dropout 0.3)

Output heads:
├── confidence    — scalar [0,1] (MSE loss, weight 2.0)
├── routing       — SELF/ORACLE/HYBRID (CE loss, weight 1.5)  
└── error_type    — 5-class (CE loss, weight 1.0)
```

Full model: ~12M parameters (7KB checkpoint). Lite model: ~50K parameters (text features only, no hidden state hooks).

#### 8.2 Three-Signal Feedback Loop

After every interaction, three signals train the meta-learner:

| Signal | Source | Strength | Weight |
|--------|--------|----------|--------|
| **Z3 verdicts** | Every Z3 verification | Binary ground truth, free, automatic | 3.0 |
| **Oracle agreement** | Every oracle consultation | Jaccard similarity on content words | 2.0 |
| **User feedback** | "good" / "bad" commands | Weak but accumulates | 0.5 |

The Z3 signal is uniquely powerful: no other cascade system has access to a formal verifier providing free binary ground truth labels. Each verification is a training example. The meta-learner converges to useful routing accuracy in ~200-500 interactions.

#### 8.3 Bootstrap

Before deployment, 500 warm-start examples from offline bootstrap:
1. Run 500 philosophical queries through the system
2. Record all signals + Z3 verdicts + oracle responses
3. Train meta-learner offline on this buffer
4. Deploy with pre-trained routing intelligence

**Full meta-learner architecture, signal extractor (full + lite), bootstrap procedure, and online training loop:** See `ADDENDUM_OLLAMA_META_LEARNER.md`, Parts 3-6.

---

### 9. Cascade Engine V3 — Complete Inference Pipeline

This is the main entry point. Orchestrates all components.

```python
class DescartesEngineV3:
    """Production inference engine."""
    
    def __init__(self):
        # Reasoning core
        self.vks = VerifiedKnowledgeStore("~/models/vks.json")
        self.formal = FormalVerifier(self.vks)
        self.corpus = CorpusVerifier(vks=self.vks)
        
        # Inference components
        self.extractor = ClaimExtractor()
        self.router = ClaimRouter(self.vks)
        self.repair = SelfRepairEngine(self.formal, self.corpus)
        
        # Meta-learner
        self.meta = MetaLearnerLite(input_dim=11)
        self.trainer = MetaTrainer(self.meta)
    
    def run(self, query: str) -> EngineResult:
        """Complete inference pipeline."""
        
        # ── Step 1: Generate ──
        initial = ollama.chat(
            model="descartes:8b", 
            messages=[
                {"role": "system", "content": DESCARTES_SYSTEM},
                {"role": "user", "content": query}
            ]
        )['message']['content']
        
        # ── Step 2: Extract claims ──
        claims = self.extractor.extract(initial)
        
        # ── Step 3: Route each claim ──
        buckets = self.router.route(claims)
        
        # ── Step 4: Verify ──
        failed = []
        
        # 4a: VKS hits — already verified, skip
        vks_hits = len(buckets['vks_hit'])
        
        # 4b: Formal claims → Z3/CVC5
        z3_ok = 0
        for claim in buckets['z3']:
            result = self.formal.verify_formal(claim)
            if result.status == VERIFIED:
                claim.verified = True
                z3_ok += 1
            else:
                failed.append((claim, result))
        
        # 4c: Factual claims → corpus index
        corpus_ok = 0
        for claim in buckets['corpus']:
            result = self.corpus.verify_factual(claim)
            if result.status == VERIFIED:
                claim.verified = True
                corpus_ok += 1
            else:
                failed.append((claim, result))
        
        # 4d: Soft pass — interpretive/meta already marked
        soft = len(buckets['soft_pass'])
        
        # ── Step 5: Self-repair failed claims ──
        still_failed = []
        repaired = 0
        for claim, result in failed:
            fix = self.repair.attempt_repair(claim, result)
            if fix and fix.status == VERIFIED:
                claim.verified = True
                repaired += 1
            else:
                still_failed.append(claim)
        
        # ── Step 6: Oracle for remaining failures ──
        oracle_corrections = ""
        if still_failed:
            failed_texts = "\n".join(
                f"- [{c.claim_type.value}] {c.text}" 
                for c in still_failed
            )
            oracle_corrections = ollama.chat(
                model="deepseek-v3.1:671-cloud",
                messages=[
                    {"role": "system", "content": ORACLE_SYSTEM},
                    {"role": "user", "content": (
                        f"Claims needing correction:\n{failed_texts}\n"
                        f"Context: {query}"
                    )}
                ]
            )['message']['content']
        
        # ── Step 7: Integration pass ──
        if repaired > 0 or still_failed:
            claim_status = "\n".join(
                f"- {c.text}: {c.verification_method}" 
                for c in claims
            )
            final = ollama.chat(
                model="descartes:8b",
                messages=[
                    {"role": "system", "content": DESCARTES_SYSTEM},
                    {"role": "user", "content": INTEGRATION_TEMPLATE.format(
                        query=query,
                        initial=initial,
                        claim_status=claim_status,
                        oracle_corrections=oracle_corrections or "None.",
                    )}
                ]
            )['message']['content']
        else:
            final = initial
        
        # ── Step 8: Feedback ──
        # Z3 + oracle + user signals → meta-learner training
        self.trainer.record_and_maybe_train(...)
        
        return EngineResult(
            final_response=final,
            vks_hits=vks_hits,
            z3_verified=z3_ok,
            corpus_verified=corpus_ok,
            soft_passed=soft,
            self_repaired=repaired,
            oracle_needed=len(still_failed),
        )
```

---

### 10. Complete Inference Flow

Step-by-step trace of a query through the full system:

```
USER: "Is the zombie argument structurally equivalent to
       Descartes' Real Distinction argument?"

STEP 1 — GENERATE
  Descartes 8B produces response with ~8 sentences containing:
  • "The zombie argument is valid in S5" [FORMAL]
  • "Both use conceivability-to-possibility bridge" [FORMAL]  
  • "Chalmers explicitly acknowledges this parallel" [FACTUAL]
  • "The arguments share identical modal structure" [FORMAL]
  • "Descartes argues in the Sixth Meditation..." [FACTUAL]
  • "The key difference is scope — zombies target..." [INTERPRETIVE]
  • "This makes the zombie argument the stronger..." [META]
  • "Under S4 the bridge principle is weaker..." [FORMAL]

STEP 2 — EXTRACT
  Claim extractor splits into 8 claims, classifies each.
  4 FORMAL, 2 FACTUAL, 1 INTERPRETIVE, 1 META.

STEP 3 — VKS LOOKUP
  "zombie argument valid in S5" → VKS HIT (Tier 2, derived theorem)
  "Real Distinction valid in S5" → VKS HIT (Tier 1, axiom)
  "Both use conceivability bridge" → VKS HIT (Tier 2, verified)
  "identical modal structure" → VKS MISS
  "Chalmers acknowledges parallel" → VKS MISS
  "Sixth Meditation" → VKS HIT (Tier 4, factual)
  
  4 hits, 4 misses. Already saved 4 verifier calls.

STEP 4 — VERIFY MISSES
  "identical modal structure" [FORMAL] → Z3:
    Encode both arguments in S5, check structural isomorphism
    → VERIFIED (UNSAT: no model distinguishes their structure)
    → WRITE to VKS Tier 2 as new derived theorem
  
  "Chalmers acknowledges" [FACTUAL] → Corpus:
    Search corpus for "Chalmers" + "Real Distinction" + "parallel"
    → NOT FOUND (Chalmers isn't in Descartes corpus)
    → Mark as failed
  
  "Under S4 bridge is weaker" [FORMAL] → Z3:
    Encode conceivability-possibility in S4
    → VERIFIED (SAT: countermodel exists in S4)
    → WRITE to VKS Tier 2
  
  "key difference is scope" [INTERPRETIVE] → SOFT PASS

STEP 5 — SELF-REPAIR
  "Chalmers acknowledges" failed corpus check.
  Strategy: RE-STATE (corpus miss, might be misphrased)
  Model reformulates: "The structural parallel between 
  zombie and Real Distinction arguments has been widely noted
  in contemporary philosophy of mind."
  Re-verify against corpus → still no match (corpus is Descartes-era)
  → REPAIR FAILED

STEP 6 — ORACLE
  Oracle receives: "The specialist couldn't verify: 'Chalmers 
  acknowledges the parallel between zombie and Real Distinction 
  arguments.' Is this accurate?"
  Oracle responds: Confirms with detail.
  → 1 oracle call total (vs. routing entire query in v1)

STEP 7 — INTEGRATION PASS
  Descartes 8B re-generates with all verified claims preserved 
  and oracle correction integrated.

STEP 8 — FEEDBACK
  Z3: 2 new verifications (modal structure + S4 bridge)
  Oracle: 1 call, agreed with original claim
  → Meta-learner receives training signals

FINAL OUTPUT:
  [VKS:4 Z3:2 CORPUS:0 SOFT:2 REPAIR:0 ORACLE:1] (1240ms)
  
  [VERIFIED] The zombie argument and Descartes' Real Distinction 
  share identical modal structure in S5...
  [VERIFIED] Both rely on the conceivability-to-possibility 
  bridge principle...
  [CORRECTED] Chalmers has noted this structural parallel...
  [VERIFIED] Under S4, the bridge principle is weaker...
  [UNVERIFIED] This makes the zombie argument the stronger of 
  the two formulations...
```

---

## Part III: Training Pipeline Summary

The full training pipeline produces all components needed for the inference engine.

### Phase Summary

| Phase | Task | Input | Output | Cost | Time |
|-------|------|-------|--------|------|------|
| 1 | Corpus Assembly | Descartes sources | 100-500M tokens | $0 | 15-25h |
| 2 | Text Extraction | Raw PDFs/scans | Clean plaintext | $0 | 5-10h |
| 3 | Cleaning | Raw text | Normalized corpus | $0 | 8-15h |
| 4 | CPT Formatting | Clean text | Tokenized JSONL | $0 | 3-5h |
| 5 | CPT Training (8B) | JSONL + Qwen3-8B | Domain-adapted model | $50-$200 | 24-40h |
| 6 | SFT Data Generation | Corpus + templates | 5K-10K examples | $50-$150 | 10-20h |
| 7 | Two-Stage SFT | SFT data + CPT model | Instruction-tuned model | $25-$100 | 8-15h |
| 8 | Base Evaluation | Trained model + benchmarks | Performance baseline | $10-$30 | 5-10h |
| 9 | Meta-Learner Bootstrap | 500 query traces | Pre-trained meta-learner | $30-$80 | 8-15h |
| 10 | VKS Seeding | Seed axioms + Z3 | Initialized knowledge store | $0 | 2-3h |
| 11 | Corpus Index Build | Clean corpus | Searchable JSON index | $0 | 2-3h |
| 12 | End-to-End Eval | Full system + benchmarks | Production metrics | $30-$80 | 10-15h |

**Total cost: $195-$720**
**Total time: 100-176 hours (2-3 weeks part-time)**

**Full phase specifications:**
- Phases 1-4: `DESCARTES_CASCADE_TRAINING_PIPELINE.md`
- Phases 5-8: `DESCARTES_CASCADE_TRAINING_PIPELINE.md`
- Phase 9: `ADDENDUM_OLLAMA_META_LEARNER.md`
- Phases 10-11: `ADDENDUM_B_VKS_MULTI_TIER_VERIFICATION.md`
- Phase 12: Both addendums

---

## Part IV: File Layout

```
~/
├── models/
│   ├── descartes-8b.gguf              # Trained Descartes specialist (GGUF)
│   ├── Modelfile                      # Ollama model definition
│   ├── vks.json                       # Verified Knowledge Store (persistent)
│   ├── meta_learner_bootstrap.pt      # Bootstrap-trained meta-learner
│   └── meta_learner_latest.pt         # Latest online-updated meta-learner
│
├── corpus/
│   ├── raw/                           # Phase 1: raw source texts
│   ├── clean/                         # Phase 3: normalized corpus
│   ├── formatted/                     # Phase 4: tokenized JSONL
│   └── index.json                     # Phase 11: searchable corpus index
│
├── inference/
│   ├── engine_v3.py                   # Main cascade engine (Part II, §9)
│   ├── knowledge_store.py             # VKS implementation (Part I, §3.4)
│   ├── seed_axioms.py                 # Foundational axiom seeding
│   ├── claim_extractor.py             # Claim splitting + typing (Part II, §5)
│   ├── claim_router.py                # Per-claim routing (Part II, §6)
│   ├── verifier.py                    # Z3 + CVC5 + corpus verification (Part I, §3)
│   ├── self_repair.py                 # Local repair before oracle (Part II, §7)
│   ├── meta_learner.py                # Meta-learner architecture (Part II, §8)
│   ├── signal_extractor_lite.py       # Text-based signal extraction
│   ├── signal_extractor_full.py       # Hidden-state signal extraction
│   ├── feedback.py                    # Three-signal feedback + online training
│   └── templates/
│       └── descartes_z3.py            # Z3 encoding templates
│
├── reasoning_core/
│   ├── ontology/
│   │   ├── core.py                    # Layer 1: sorts, relations
│   │   ├── theories.py                # Theory commitment axioms
│   │   └── owl_taxonomy.py            # OWL 2 layer
│   ├── argumentation/
│   │   ├── aspic_engine.py            # Layer 2: ASPIC+ knowledge base
│   │   ├── zombie_argument.py         # Zombie argument decomposition
│   │   └── walton_schemes.py          # Argumentation scheme templates
│   ├── verification/
│   │   ├── z3_engine.py               # Layer 3: modal/paracons/defeasible
│   │   ├── cvc5_engine.py             # CVC5 parallel verification
│   │   └── examples.py                # Verification examples
│   ├── bridge/
│   │   └── gvr_loop.py                # Layer 4: GVR (embedded in cascade)
│   └── spaces/
│       └── conceptual_spaces.py       # Layer 5: geometric complement
│
├── training/
│   ├── cpt/                           # Phase 5: continued pre-training
│   ├── sft/                           # Phases 6-7: supervised fine-tuning
│   ├── bootstrap_meta.py              # Phase 9: meta-learner bootstrap
│   └── eval/
│       ├── eval_base.py               # Phase 8: base model benchmarks
│       ├── eval_cascade.py            # Phase 12: end-to-end cascade eval
│       └── eval_vks_growth.py         # VKS accumulation metrics
│
└── data/
    ├── sft_examples/                  # Generated SFT training data
    ├── z3_templates/                  # Pre-formalized argument templates
    └── benchmarks/                    # Evaluation datasets
```

---

## Part V: Implementation Roadmap

### Updated Phases (incorporating all components)

#### Phase 1: Z3 Foundation + VKS Skeleton (Weeks 1-4)

| Task | Hours | Deliverable |
|------|-------|-------------|
| Implement `ModalLogicEngine` with S5 | 20 | Working modal logic in Z3 |
| Implement `ParaconsistentEngine` (Belnap 4-valued) | 15 | Paraconsistent reasoning |
| Formalize zombie argument, Chinese Room, Knowledge Argument | 25 | 3 verified arguments |
| Implement `MicrotheoryManager` with 4 theories | 20 | Cross-theory checking |
| Implement `UnsatCoreExtractor` + MCS | 15 | Error localization |
| **NEW:** Implement `VerifiedKnowledgeStore` skeleton | 10 | VKS read/write/integrity |
| **NEW:** Implement seed axioms (Cogito, Real Distinction, Circle) | 5 | 4+ seeded axioms |
| Write test suite | 10 | Automated validation |
| **Total** | **120** | **Working Layer 3 + VKS** |

#### Phase 2: ASPIC+ Structure + Claim Extraction (Weeks 5-7)

| Task | Hours | Deliverable |
|------|-------|-------------|
| Implement `ASPICKnowledgeBase` with all attack types | 25 | Working ASPIC+ engine |
| Implement 6 Walton scheme templates | 20 | Scheme instantiation |
| Decompose 10 major consciousness arguments | 30 | Structured argument corpus |
| **NEW:** Implement `ClaimExtractor` with type classification | 10 | Claim splitting + typing |
| **NEW:** Implement `ClaimRouter` | 5 | Per-claim routing logic |
| Build AIF export/import | 10 | Interoperability |
| **Total** | **100** | **Working Layer 2 + claim pipeline** |

#### Phase 3: Descartes Corpus + Training (Weeks 8-12)

| Task | Hours | Deliverable |
|------|-------|-------------|
| Corpus assembly (Meditations, O&R, Correspondence) | 15-25 | 100-500M tokens |
| Text extraction + cleaning + formatting | 16-30 | Tokenized JSONL |
| CPT Training on Qwen3-8B (A40 48GB) | 24-40 | Domain-adapted model |
| SFT Data Generation (Types A-G) | 10-20 | 5K-10K examples |
| Two-Stage SFT | 8-15 | Instruction-tuned + routing |
| **NEW:** Build corpus index for factual verification | 3-5 | Searchable JSON index |
| Base model evaluation | 5-10 | Performance baseline |
| **Total** | **81-145** | **Trained Descartes 8B + corpus index** |

#### Phase 4: Meta-Learner + Cascade Assembly (Weeks 13-15)

| Task | Hours | Deliverable |
|------|-------|-------------|
| Ollama setup (local + cloud) | 3-5 | Unified API working |
| Implement signal extractor (lite + full) | 10-15 | Feature extraction |
| Implement meta-learner architecture | 8-12 | ~12M param model |
| **NEW:** Implement `SelfRepairEngine` (3 strategies) | 8-12 | Local repair loop |
| **NEW:** Implement `CorpusVerifier` | 5-8 | Factual verification |
| Bootstrap meta-learner (500 queries) | 8-15 | Pre-trained routing |
| Build cascade engine V3 | 15-20 | Full orchestrated system |
| **Total** | **57-87** | **Complete inference engine** |

#### Phase 5: Integration + Evaluation (Weeks 16-18)

| Task | Hours | Deliverable |
|------|-------|-------------|
| Ontology + Conceptual Spaces | 40 | Layers 1 + 5 |
| End-to-end integration testing | 20-30 | Full pipeline validation |
| Cascade benchmarks (accuracy, cost, latency) | 10-15 | Performance metrics |
| **NEW:** VKS growth tracking (hit rate over time) | 5-8 | Accumulation curves |
| Expert review of 500 examples (spot check) | 50 | Quality validation |
| **Total** | **125-143** | **Production-ready system** |

### Totals

| Phase | Hours | Calendar Weeks | Cost |
|-------|-------|----------------|------|
| 1: Z3 + VKS | 120 | 4 | $0 |
| 2: ASPIC+ + Claims | 100 | 3 | $0 |
| 3: Corpus + Training | 81-145 | 5 | $125-$480 |
| 4: Meta-Learner + Cascade | 57-87 | 3 | $40-$110 |
| 5: Integration + Eval | 125-143 | 3 | $30-$130 |
| **Total** | **483-595** | **18** | **$195-$720** |

This is a ~10% increase in hours over the original plan (540h → 483-595h) for the addition of VKS, claim routing, self-repair, and corpus verification. The extra work pays for itself quickly through reduced oracle calls and cached verification results.

---

## Part VI: Evaluation Framework

### 6.1 Core Metrics

| Metric | Target | Measures |
|--------|--------|----------|
| **Formal Accuracy** | ≥90% agreement with expert Z3 encodings | Z3 encoding quality |
| **Factual Accuracy** | ≥95% on corpus-verifiable claims | Corpus index quality |
| **VKS Hit Rate** | ≥40% after 500 queries, ≥60% after 2000 | Memory effectiveness |
| **Self-Repair Rate** | ≥35% of failed claims repaired locally | Oracle cost reduction |
| **Oracle Call Rate** | ≤15% of all claims | Cascade efficiency |
| **End-to-End Accuracy** | ≥85% on expert-validated test set | System quality |
| **Latency (VKS hit)** | <100ms | Cache speed |
| **Latency (Z3 verify)** | <5s per claim | Verification speed |
| **Latency (full query)** | <15s (no oracle), <30s (with oracle) | User experience |

### 6.2 VKS Growth Evaluation

Track the knowledge store's accumulation over time:

```python
def eval_vks_growth(engine, test_queries, n_epochs=5):
    """Run test queries multiple times; measure VKS hit rate increase."""
    for epoch in range(n_epochs):
        hits, total = 0, 0
        for query in test_queries:
            result = engine.run(query)
            hits += result.vks_hits
            total += len(result.claims)
        
        hit_rate = hits / total
        print(f"Epoch {epoch}: VKS hit rate = {hit_rate:.1%} "
              f"({engine.vks.get_stats()['total']} records)")
    
    # Expected: Epoch 0: ~5%, Epoch 1: ~30%, Epoch 2: ~50%, 
    #           Epoch 3: ~60%, Epoch 4: ~65% (saturating)
```

### 6.3 Self-Repair Evaluation

Measure oracle cost savings from self-repair:

```python
def eval_repair_savings(engine, test_queries):
    """Compare oracle calls with and without self-repair."""
    # Run with repair enabled
    engine.repair.enabled = True
    oracle_with = sum(engine.run(q).oracle_needed for q in test_queries)
    
    # Run with repair disabled
    engine.repair.enabled = False
    oracle_without = sum(engine.run(q).oracle_needed for q in test_queries)
    
    savings = 1 - (oracle_with / max(oracle_without, 1))
    print(f"Self-repair saves {savings:.0%} of oracle calls")
    # Target: ≥35% reduction
```

### 6.4 Comparative Benchmarks

| Benchmark | What It Tests | Pass Condition |
|-----------|--------------|----------------|
| Cogito variants (20 phrasings) | VKS generalization | ≥18/20 VKS hits after seed |
| Arnauld's objections (10 claims) | Corpus verification | ≥8/10 corpus verified |
| Novel arguments (20 unseen) | Z3 formalization + meta-learner | ≥14/20 correctly routed |
| Cross-theory consistency (10 pairs) | Microtheory manager | 10/10 detect contradictions |
| Out-of-domain (10 non-Descartes) | Oracle routing | ≥8/10 correctly escalated |
| Adversarial (10 subtly wrong claims) | Z3 catching errors | ≥7/10 Z3 refutes |

---

## Appendix A: Comparison with COGITO Architecture

The Philosopher Engine borrows several design patterns from ARIA COGITO:

| COGITO Component | Philosopher Engine Equivalent | What We Took |
|---|---|---|
| EMT Semantic Memory (Z3-verified knowledge, ontology graph) | Verified Knowledge Store (Tiers 1-4) | Persistent, tiered, verified claims |
| EMT hash chains (immutable leaf history) | VKS axiom chain (append-only, hash-linked) | Corruption detection, integrity guarantees |
| "Axiom continuity = identity" principle | VKS corruption = system identity broken | Immutable axioms as foundation |
| Grounding Verifier (ClaimType taxonomy) | Claim Extractor (FORMAL/FACTUAL/INTERPRETIVE/META) | Per-claim classification and routing |
| Grounding Verifier repair loop (verify → repair → re-verify) | Self-Repair Engine (re-encode/re-state/decompose) | Fix locally before escalating |
| Three-tier verification (Z3 → Z3 full → Lean) | Multi-tier verification (Z3 + CVC5 / corpus / soft) | Right tool for each claim type |
| LLM Wrapper "speaks for, doesn't think for" COGITO | Integration pass: specialist has final word | Oracle provides knowledge, specialist integrates |
| Operative HOT (meta-free-energy over own processing) | Meta-learner (hidden state signals → routing) | Internal states reveal confidence model can't self-report |

---

## Appendix B: Key Dependencies

```
# requirements.txt

# Z3 / SMT
z3-solver>=4.12
# cvc5  # Optional: pip install cvc5 (for parallel SMT)

# LLM Interface
ollama>=0.2      # Unified local + cloud API

# ML (meta-learner)
torch>=2.1
numpy>=1.24

# Argumentation
owlready2>=0.46  # OWL 2 ontology

# Data Processing
jsonlines>=4.0

# Corpus Index
# (JSON-based for v1; optional FAISS for semantic search)
# faiss-cpu>=1.7  # Optional

# Training (GPU server only)
# transformers>=4.36
# peft>=0.7
# bitsandbytes>=0.41
# deepspeed>=0.12

# Testing
pytest>=7.4
```

---

## Appendix C: Document Reading Order

For anyone implementing this system, read the documents in this order:

```
1. THIS DOCUMENT (V3 Unified Architecture)
   → Complete system overview, all components, how they connect

2. PHILOSOPHER_ENGINE_ARCHITECTURE.md (original)
   → Full code for Layers 1, 2, 3 (Z3), 4 (GVR), 5
   → Argument templates library
   → 3310 lines of implementation detail

3. DESCARTES_CASCADE_TRAINING_PIPELINE.md
   → Phases 1-8: corpus assembly through base evaluation
   → Full CPT/SFT specifications and cost breakdowns

4. ADDENDUM_OLLAMA_META_LEARNER.md (Addendum A)
   → Ollama setup, signal extraction, meta-learner architecture
   → Bootstrap procedure, online training loop
   → Full code for meta-learner (~12M and ~50K param variants)

5. ADDENDUM_B_VKS_MULTI_TIER_VERIFICATION.md (Addendum B)
   → VKS implementation, seed axioms, claim extractor
   → Claim router, multi-tier verifier, self-repair engine
   → Modified cascade engine V3 with REPL

6. COGITO docs (for architectural inspiration, not implementation)
   → ARIA_COGITO_v2_1_MODIFIED_ARCHITECTURE.md
   → ARIA_COGITO_V2_1_LLM_WRAPPER_SPECIFICATION.md
   → ARIA_COGITO_COMPREHENSIVE_REFERENCE.md
```
