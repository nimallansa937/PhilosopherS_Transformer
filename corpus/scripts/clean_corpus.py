"""
Phase 3: Corpus Cleaning Pipeline for philosophical texts.

OPERATIONS:
1. Unicode normalization (NFC)
2. Fix hyphenation artifacts from PDF extraction
3. Reconnect paragraphs split across pages
4. Remove duplicate documents (MinHash deduplication)
5. Filter out documents with low argument density
6. Verify mixing ratio matches target
7. Produce final cleaned corpus with quality report

Usage:
    python corpus/scripts/clean_corpus.py
"""

import re
import os
import json
import unicodedata
import hashlib
from pathlib import Path
from collections import Counter
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "corpus" / "extracted"
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "cleaned"


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
# DEDUPLICATION (MinHash)
# ============================================================

def compute_shingles(text: str, k: int = 5) -> set:
    """Compute k-word shingles for MinHash."""
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}


def minhash_signature(shingles: set, num_hashes: int = 128) -> List[int]:
    """Compute MinHash signature."""
    if not shingles:
        return [float('inf')] * num_hashes

    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
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


def deduplicate(documents: List[Tuple[Path, str]],
                threshold: float = 0.8) -> List[Tuple[Path, str]]:
    """Remove near-duplicate documents using MinHash."""

    print(f"  Computing MinHash signatures for {len(documents)} documents...")
    signatures = []
    for path, text in documents:
        shingles = compute_shingles(text)
        sig = minhash_signature(shingles)
        signatures.append(sig)

    # Find duplicates
    duplicates = set()
    total_comparisons = len(documents) * (len(documents) - 1) // 2
    comparisons_done = 0

    for i in range(len(documents)):
        if i in duplicates:
            continue
        for j in range(i + 1, len(documents)):
            if j in duplicates:
                continue
            similarity = jaccard_from_minhash(signatures[i], signatures[j])
            if similarity > threshold:
                # Keep the longer document
                len_i = len(documents[i][1])
                len_j = len(documents[j][1])
                if len_i >= len_j:
                    duplicates.add(j)
                else:
                    duplicates.add(i)
            comparisons_done += 1

        # Progress update every 1000 documents
        if i % 100 == 0 and i > 0:
            print(f"    Processed {i}/{len(documents)} documents...")

    print(f"  Found {len(duplicates)} duplicates (threshold={threshold})")
    return [doc for i, doc in enumerate(documents) if i not in duplicates]


# ============================================================
# QUALITY FILTERING
# ============================================================

# Philosophical argument indicators
ARGUMENT_INDICATORS = [
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
    "supervenes", "reduces to", "identical to", "constituted by"
]


def argument_density(text: str) -> float:
    """Score document by density of philosophical argument indicators.

    Returns indicators per 1000 words. Typical values:
    - Philosophy papers: 15-40
    - Neuroscience papers: 5-15
    - Non-philosophical text: 0-5
    """
    words = text.lower().split()
    if len(words) < 100:
        return 0.0

    count = sum(text.lower().count(indicator) for indicator in ARGUMENT_INDICATORS)
    return (count / len(words)) * 1000


def filter_quality(documents: List[Tuple[Path, str]],
                   min_length: int = 500,
                   min_argument_density: float = 3.0) -> List[Tuple[Path, str]]:
    """Filter out low-quality documents."""

    filtered = []
    stats = {"too_short": 0, "low_density": 0, "passed": 0}

    for path, text in documents:
        if len(text) < min_length:
            stats["too_short"] += 1
            continue

        density = argument_density(text)

        # Use lower threshold for neuroscience (less argumentative language)
        threshold = 1.5 if "neuroscience" in str(path) else min_argument_density

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all extracted documents
    print("\n[1/5] Loading extracted documents...")
    documents = []
    for filepath in INPUT_DIR.rglob("*.txt"):
        # Skip metadata files
        if filepath.name.startswith("extraction"):
            continue
        text = filepath.read_text(encoding="utf-8", errors="replace")
        rel_path = filepath.relative_to(INPUT_DIR)
        documents.append((rel_path, text))
    print(f"  Loaded {len(documents)} documents")

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
    report = {
        "total_documents": len(filtered),
        "total_tokens_estimated": total_tokens_est,
        "category_breakdown": category_stats,
        "mixing_ratios_actual": {
            cat: round(stats["tokens_est"] / max(total_tokens_est, 1), 3)
            for cat, stats in category_stats.items()
        },
        "mixing_ratios_target": {
            "philosophy_of_mind": 0.40,
            "neuroscience": 0.20,
            "broader_philosophy": 0.15,
            "cognitive_science": 0.15,
            "cross_disciplinary": 0.10,
        },
        "pipeline_stats": {
            "input_documents": len(documents),
            "after_dedup": len(deduped),
            "after_quality_filter": len(filtered),
            "removed_total": len(documents) - len(filtered),
        }
    }

    report_path = OUTPUT_DIR / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 60}")
    print(f"CLEANING COMPLETE")
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
