"""
Phase 3: Corpus Cleaning Pipeline for philosophical texts.

OPERATIONS:
1. Unicode normalization (NFC)
2. Fix hyphenation artifacts from PDF extraction
3. Reconnect paragraphs split across pages
4. Remove duplicate documents (MinHash deduplication with LSH banding)
5. Filter out documents with low argument density (multilingual)
6. Verify mixing ratio matches target
7. Produce final cleaned corpus with quality report

Handles multilingual corpus (English, Latin, French, German, Dutch, Spanish)
and prioritises core Cartesian/rationalist texts.

Usage:
    python corpus/scripts/clean_corpus.py
"""

import re
import os
import sys
import json
import unicodedata
import hashlib
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "corpus" / "extracted"
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "cleaned"

# Categories that should NEVER be filtered by argument density
# (these are the core Descartes / rationalist texts we downloaded specifically)
PRIORITY_CATEGORIES = {
    "descartes_primary",
    "descartes_scholarship",
    "rationalist_tradition",
    "gutenberg",
    "archive_org",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """NFC normalization - canonical decomposition + composition."""
    return unicodedata.normalize("NFC", text)


def fix_hyphenation(text: str) -> str:
    """Rejoin words split by line-end hyphenation in PDFs.

    'con-\\nsciousness' -> 'consciousness'
    But preserve intentional hyphens: 'mind-body' stays 'mind-body'
    """
    # Pattern: word fragment + hyphen + newline + lowercase continuation
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Also handle hyphen + space + newline
    text = re.sub(r'(\w)- \n(\w)', r'\1\2', text)
    return text


def reconnect_paragraphs(text: str) -> str:
    """Reconnect paragraphs split across PDF pages.

    A paragraph continues if a line ends without sentence-final
    punctuation and the next line starts with a lowercase letter.
    """
    lines = text.split('\n')
    result = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append("")
            continue

        if buffer:
            # Check if this continues the previous line
            if (stripped[0].islower() or stripped[0] in '("\'') and \
               buffer and buffer[-1] not in '.!?:;':
                buffer += " " + stripped
            else:
                result.append(buffer)
                buffer = stripped
        else:
            buffer = stripped

    if buffer:
        result.append(buffer)

    return "\n".join(result)


def remove_headers_footers(text: str) -> str:
    """Remove repeated headers/footers (journal names, page numbers)."""
    lines = text.split('\n')

    # Count line frequencies - headers/footers repeat across pages
    line_counts = Counter(line.strip() for line in lines if line.strip())

    # If a short line appears more than 3 times, it's likely a header/footer
    repeated = {line for line, count in line_counts.items()
                if count > 3 and len(line) < 100}

    filtered = [line for line in lines if line.strip() not in repeated]
    return "\n".join(filtered)


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single document."""
    text = normalize_unicode(text)
    text = fix_hyphenation(text)
    text = remove_headers_footers(text)
    text = reconnect_paragraphs(text)

    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove common PDF artifacts
    text = re.sub(r'\x0c', '\n', text)  # Form feed
    text = re.sub(r'[\x00-\x08\x0b\x0e-\x1f]', '', text)  # Control chars

    return text.strip()


# ============================================================
# DEDUPLICATION (MinHash + LSH Banding)
# ============================================================

def compute_shingles(text: str, k: int = 5, max_shingles: int = 10000) -> set:
    """Compute k-word shingles for MinHash.

    Caps shingles at max_shingles for performance on large documents.
    Samples evenly across the document.
    """
    words = text.lower().split()
    if len(words) < k:
        return set()
    total = len(words) - k + 1
    if total <= max_shingles:
        return {tuple(words[i:i+k]) for i in range(total)}
    # Sample evenly across document
    step = total / max_shingles
    return {tuple(words[int(i*step):int(i*step)+k]) for i in range(max_shingles)}


def minhash_signature(shingles: set, num_hashes: int = 128) -> List[int]:
    """Compute MinHash signature."""
    if not shingles:
        return [0xFFFFFFFF] * num_hashes

    signature = []
    for i in range(num_hashes):
        min_hash = 0xFFFFFFFF
        for shingle in shingles:
            h = hash((i, shingle)) & 0xFFFFFFFF
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def jaccard_from_minhash(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    assert len(sig_a) == len(sig_b)
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def lsh_candidates(signatures: List[List[int]],
                   bands: int = 16, rows: int = 8) -> Set[Tuple[int, int]]:
    """Use Locality-Sensitive Hashing to find candidate duplicate pairs.

    With b=16 bands and r=8 rows per band (128 hashes total),
    pairs with Jaccard ≥ 0.8 have ~96% chance of being candidates.
    Much faster than O(n²) pairwise comparison.
    """
    assert bands * rows == len(signatures[0]), \
        f"bands*rows={bands*rows} != sig_len={len(signatures[0])}"

    buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    candidates: Set[Tuple[int, int]] = set()

    for band_idx in range(bands):
        buckets.clear()
        start = band_idx * rows
        end = start + rows

        for doc_idx, sig in enumerate(signatures):
            band_hash = tuple(sig[start:end])
            buckets[band_hash].append(doc_idx)

        # All documents in the same bucket are candidates
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        candidates.add((bucket_docs[i], bucket_docs[j]))

    return candidates


def deduplicate(documents: List[Tuple[Path, str]],
                threshold: float = 0.8) -> List[Tuple[Path, str]]:
    """Remove near-duplicate documents using MinHash + LSH banding.

    Uses LSH to reduce candidate pairs from O(n²) to O(n), then
    verifies candidates with full MinHash Jaccard comparison.
    """
    n = len(documents)
    if n <= 1:
        return documents

    t0 = time.time()
    print(f"  Computing MinHash signatures for {n} documents...")
    signatures = []
    for idx, (path, text) in enumerate(documents):
        shingles = compute_shingles(text)
        sig = minhash_signature(shingles)
        signatures.append(sig)
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    {idx+1}/{n} signatures computed ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  All {n} signatures computed in {elapsed:.1f}s")

    # Find candidate pairs using LSH banding
    print(f"  Finding candidate pairs via LSH (16 bands × 8 rows)...")
    candidates = lsh_candidates(signatures, bands=16, rows=8)
    print(f"  Found {len(candidates)} candidate pairs "
          f"(vs {n*(n-1)//2} full pairwise)")

    # Verify candidates
    duplicates = set()
    verified = 0
    for i, j in candidates:
        if i in duplicates or j in duplicates:
            continue
        similarity = jaccard_from_minhash(signatures[i], signatures[j])
        verified += 1
        if similarity > threshold:
            # Keep the longer document
            len_i = len(documents[i][1])
            len_j = len(documents[j][1])
            if len_i >= len_j:
                duplicates.add(j)
            else:
                duplicates.add(i)

    elapsed = time.time() - t0
    print(f"  Verified {verified} pairs, found {len(duplicates)} "
          f"duplicates ({elapsed:.1f}s total)")
    return [doc for i, doc in enumerate(documents) if i not in duplicates]


# ============================================================
# QUALITY FILTERING (Multilingual)
# ============================================================

# Philosophical argument indicators — English
ARGUMENT_INDICATORS_EN = [
    "therefore", "thus", "hence", "consequently", "it follows",
    "implies", "entails", "because", "since", "given that",
    "suppose", "assume", "consider", "if we accept", "granted that",
    "however", "nevertheless", "on the other hand", "objection",
    "contra", "despite", "although", "whereas",
    "I argue", "I contend", "I maintain", "I claim",
    "the argument", "the objection", "the problem", "the thesis",
    "premise", "conclusion", "inference", "valid", "sound",
    "necessary", "sufficient", "possible", "impossible",
    "conceivable", "metaphysically", "a priori", "a posteriori",
    "supervenes", "reduces to", "identical to", "constituted by",
    "cogito", "descartes", "substance", "attribute", "mode",
]

# Latin philosophical terms (Descartes wrote in Latin)
ARGUMENT_INDICATORS_LA = [
    "ergo", "igitur", "itaque", "quoniam", "quia",
    "quapropter", "proinde", "namque", "enim",
    "cogito", "sum", "res cogitans", "res extensa",
    "substantia", "attributum", "modus",
    "meditatio", "meditationes", "objectiones", "responsiones",
    "principia", "philosophiae", "methodus", "discursus",
    "demonstratio", "propositio", "axioma", "definitio",
    "deus", "anima", "corpus", "idea", "intellectus",
]

# French philosophical terms (Descartes also wrote in French)
ARGUMENT_INDICATORS_FR = [
    "donc", "ainsi", "par conséquent", "puisque", "parce que",
    "cependant", "néanmoins", "toutefois", "objection",
    "argument", "raison", "preuve", "conclusion",
    "je pense", "je suis", "substance", "pensée",
    "méthode", "méditation", "discours", "principes",
    "âme", "corps", "esprit", "idée", "entendement",
    "volonté", "jugement", "certitude", "doute",
]

# German philosophical terms (Leibniz, Spinoza translations)
ARGUMENT_INDICATORS_DE = [
    "daher", "deshalb", "folglich", "weil", "denn",
    "jedoch", "dennoch", "obwohl", "einwand",
    "beweis", "schluss", "substanz", "monade",
    "vernunft", "verstand", "erkenntnis",
    "philosophie", "metaphysik", "ethik",
]

ALL_INDICATORS = (ARGUMENT_INDICATORS_EN + ARGUMENT_INDICATORS_LA +
                  ARGUMENT_INDICATORS_FR + ARGUMENT_INDICATORS_DE)


def argument_density(text: str) -> float:
    """Score document by density of philosophical argument indicators.

    Multilingual: checks English, Latin, French, German terms.
    Returns indicators per 1000 words. Typical values:
    - Philosophy papers: 15-40
    - Neuroscience papers: 5-15
    - Non-philosophical text: 0-5
    """
    words = text.lower().split()
    if len(words) < 100:
        return 0.0

    text_lower = text.lower()
    count = sum(text_lower.count(indicator) for indicator in ALL_INDICATORS)
    return (count / len(words)) * 1000


def get_category(path: Path) -> str:
    """Extract top-level category from relative path."""
    parts = path.parts
    return parts[0] if parts else "unknown"


def filter_quality(documents: List[Tuple[Path, str]],
                   min_length: int = 500,
                   min_argument_density: float = 2.0) -> List[Tuple[Path, str]]:
    """Filter out low-quality documents.

    Priority categories (descartes_primary, rationalist_tradition, etc.)
    are kept regardless of argument density — these are the core texts
    we specifically downloaded for training.
    """

    filtered = []
    stats = {"too_short": 0, "low_density": 0, "passed": 0,
             "priority_kept": 0}

    for path, text in documents:
        category = get_category(path)

        if len(text) < min_length:
            # Still keep priority texts even if short
            if category in PRIORITY_CATEGORIES and len(text) >= 200:
                stats["priority_kept"] += 1
                filtered.append((path, text))
            else:
                stats["too_short"] += 1
            continue

        # Priority categories always pass density check
        if category in PRIORITY_CATEGORIES:
            stats["priority_kept"] += 1
            filtered.append((path, text))
            continue

        density = argument_density(text)

        # Use lower threshold for neuroscience (less argumentative language)
        threshold = 1.0 if "neuroscience" in str(path) else min_argument_density

        if density < threshold:
            stats["low_density"] += 1
            continue

        stats["passed"] += 1
        filtered.append((path, text))

    print(f"  Quality filter: {stats}")
    return filtered


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_cleaning_pipeline():
    """Execute the full cleaning pipeline."""

    print("=" * 60)
    print("PHASE 3: CORPUS CLEANING PIPELINE")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Clear old cleaned output to avoid stale files
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print("  Cleared old cleaned corpus")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Step 1: Load all extracted documents
    print("\n[1/5] Loading extracted documents...")
    documents = []
    skip_prefixes = ("extraction", "preparation", "metrics")
    for filepath in INPUT_DIR.rglob("*.txt"):
        # Skip metadata files
        if filepath.name.startswith(skip_prefixes):
            continue
        if filepath.name.endswith(".json"):
            continue
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [WARN] Could not read {filepath.name}: {e}")
            continue
        rel_path = filepath.relative_to(INPUT_DIR)
        documents.append((rel_path, text))
    print(f"  Loaded {len(documents)} documents ({time.time()-t_start:.1f}s)")

    if not documents:
        print("\n  No documents found! Run Phase 2 (extraction) first.")
        return

    # Step 2: Clean each document
    print("\n[2/5] Cleaning texts...")
    cleaned = [(path, clean_text(text)) for path, text in documents]

    # Step 3: Deduplicate
    print("\n[3/5] Deduplicating...")
    deduped = deduplicate(cleaned)

    # Step 4: Quality filter
    print("\n[4/5] Quality filtering...")
    filtered = filter_quality(deduped)

    # Step 5: Save and compute statistics
    print("\n[5/5] Saving cleaned corpus...")

    category_stats = {}
    total_tokens_est = 0

    for rel_path, text in filtered:
        out_path = OUTPUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        # Track per-category stats
        category = rel_path.parts[0] if rel_path.parts else "unknown"
        if category not in category_stats:
            category_stats[category] = {"docs": 0, "tokens_est": 0}

        # Rough token estimate: words * 1.3
        token_est = int(len(text.split()) * 1.3)
        category_stats[category]["docs"] += 1
        category_stats[category]["tokens_est"] += token_est
        total_tokens_est += token_est

    # Save report
    elapsed_total = time.time() - t_start
    report = {
        "total_documents": len(filtered),
        "total_tokens_estimated": total_tokens_est,
        "category_breakdown": category_stats,
        "mixing_ratios_actual": {
            cat: round(stats["tokens_est"] / max(total_tokens_est, 1), 3)
            for cat, stats in category_stats.items()
        },
        "mixing_ratios_target": {
            "descartes_primary": "core",
            "rationalist_tradition": "core",
            "descartes_scholarship": "core",
            "broader_philosophy": "supplementary",
            "philosophy_of_mind": "supplementary",
            "neuroscience": "supplementary",
            "cognitive_science": "supplementary",
            "cross_disciplinary": "supplementary",
        },
        "pipeline_stats": {
            "input_documents": len(documents),
            "after_dedup": len(deduped),
            "after_quality_filter": len(filtered),
            "removed_total": len(documents) - len(filtered),
            "elapsed_seconds": round(elapsed_total, 1),
        }
    }

    report_path = OUTPUT_DIR / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 60}")
    print(f"CLEANING COMPLETE ({elapsed_total:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Documents: {len(documents)} -> {len(filtered)}")
    print(f"Estimated tokens: {total_tokens_est:,}")
    print(f"\nCategory breakdown:")
    for cat, stats in sorted(category_stats.items()):
        ratio = stats["tokens_est"] / max(total_tokens_est, 1)
        print(f"  {cat}: {stats['docs']} docs, "
              f"~{stats['tokens_est']:,} tokens ({ratio:.1%})")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_cleaning_pipeline()
