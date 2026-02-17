"""
Phase 11: Build searchable corpus index for factual verification.

The corpus index enables the CorpusVerifier (inference/verifier.py) to
check FACTUAL claims like "Arnauld raised the circularity objection in
the Fourth Objections" against the actual training corpus.

Each index entry contains:
  - source: origin text reference (e.g. "Meditations, Third Meditation")
  - passage: short text chunk (128-512 tokens)
  - keywords: extracted keywords for BM25-style search
  - category: corpus category (descartes_primary, philosophy_of_mind, etc.)

Index format: JSON for simplicity. Production could use FAISS for
semantic search, but keyword search is sufficient for factual claims
about attributions and textual references.

Usage:
    python corpus/scripts/build_corpus_index.py
    python corpus/scripts/build_corpus_index.py --input corpus/extracted
    python corpus/scripts/build_corpus_index.py --output corpus/index.json

Reference: PHILOSOPHER_ENGINE_V3_UNIFIED_ARCHITECTURE.md, §3.3 / Phase 11
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text: str,
               chunk_size: int = 512,
               overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks of roughly `chunk_size` words.

    Uses sentence-aware splitting: chunks break at sentence
    boundaries to avoid splitting mid-thought.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        sent_len = len(words)

        if current_len + sent_len > chunk_size and current_chunk:
            chunk_text_str = ' '.join(current_chunk)
            if len(chunk_text_str.strip()) > 50:  # Skip tiny chunks
                chunks.append(chunk_text_str.strip())

            # Overlap: keep last few sentences
            overlap_words = 0
            overlap_start = len(current_chunk)
            for i in range(len(current_chunk) - 1, -1, -1):
                w = len(current_chunk[i].split())
                if overlap_words + w > overlap:
                    break
                overlap_words += w
                overlap_start = i

            current_chunk = current_chunk[overlap_start:]
            current_len = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sentence)
        current_len += sent_len

    # Final chunk
    if current_chunk:
        chunk_text_str = ' '.join(current_chunk)
        if len(chunk_text_str.strip()) > 50:
            chunks.append(chunk_text_str.strip())

    return chunks


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

# Philosophical stopwords (common but not meaningful for search)
STOPWORDS: Set[str] = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'this', 'that', 'these', 'those',
    'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him',
    'her', 'us', 'them', 'my', 'your', 'his', 'their', 'our',
    'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'when', 'while',
    'as', 'which', 'who', 'whom', 'what', 'where', 'how', 'why',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'also', 'very', 'just', 'because',
    'about', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'into', 'out', 'up', 'down', 'over', 'under',
    'again', 'further', 'once', 'here', 'there', 'any', 'same',
    'however', 'therefore', 'thus', 'hence', 'moreover', 'indeed',
    'rather', 'quite', 'still', 'yet', 'already', 'even', 'well',
}

# High-value philosophical terms (boosted in keyword ranking)
PHILOSOPHY_TERMS: Set[str] = {
    'cogito', 'descartes', 'meditation', 'substance', 'mind', 'body',
    'soul', 'thought', 'extension', 'god', 'existence', 'doubt',
    'perception', 'clear', 'distinct', 'idea', 'innate', 'wax',
    'dream', 'deceiver', 'evil', 'genius', 'pineal', 'gland',
    'arnauld', 'gassendi', 'hobbes', 'mersenne', 'elisabeth',
    'princess', 'objection', 'reply', 'meditation', 'principle',
    'discourse', 'method', 'passions', 'rules', 'direction',
    'spinoza', 'leibniz', 'malebranche', 'occasionalism',
    'consciousness', 'qualia', 'zombie', 'physicalism', 'dualism',
    'functionalism', 'phenomenal', 'intentionality', 'supervenience',
    'causal', 'adequacy', 'formal', 'reality', 'objective',
    'conceivability', 'possibility', 'necessity', 'modal', 'possible',
    'world', 'identity', 'distinction', 'attribute', 'mode',
    'cartesian', 'circle', 'trademark', 'ontological', 'argument',
    'premise', 'conclusion', 'valid', 'invalid', 'proof',
}


def extract_keywords(text: str, max_keywords: int = 30) -> List[str]:
    """Extract ranked keywords from a text chunk.

    Uses frequency + philosophy term boosting.
    Returns the top `max_keywords` unique terms.
    """
    # Tokenize
    words = re.findall(r'\b[a-z][a-z\'-]*[a-z]\b|\b[a-z]\b',
                       text.lower())

    # Count frequencies, exclude stopwords
    counts: Counter = Counter()
    for word in words:
        if word not in STOPWORDS and len(word) > 2:
            counts[word] += 1

    # Boost philosophy terms
    scored = {}
    for word, count in counts.items():
        score = count
        if word in PHILOSOPHY_TERMS:
            score *= 3  # 3x boost for domain terms
        scored[word] = score

    # Return top N by score
    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return [word for word, _ in ranked[:max_keywords]]


# ============================================================
# SOURCE DETECTION
# ============================================================

def detect_source(filepath: Path, chunk_text: str) -> str:
    """Infer the source reference from filepath and content.

    Maps file names and directory structure to source citations.
    """
    parts = filepath.parts
    filename = filepath.stem.lower()

    # Detect by parent directory category
    category = "unknown"
    for part in parts:
        part_lower = part.lower()
        if part_lower in ('descartes_primary', 'rationalist_tradition',
                          'philosophy_of_mind', 'contemporary_responses',
                          'formal_logic', 'descartes_scholarship',
                          'broader_philosophy', 'cognitive_science',
                          'cross_disciplinary', 'neuroscience'):
            category = part_lower
            break

    # Detect specific Descartes works
    descartes_works = {
        'meditation': 'Meditations on First Philosophy',
        'discourse': 'Discourse on the Method',
        'principles': 'Principles of Philosophy',
        'passions': 'Passions of the Soul',
        'rules': 'Rules for the Direction of the Mind',
        'objections': 'Objections and Replies',
        'correspondence': 'Correspondence',
    }

    for key, title in descartes_works.items():
        if key in filename:
            # Try to detect specific meditation/objection number
            match = re.search(r'(first|second|third|fourth|fifth|sixth|'
                              r'[ivx]+|[1-6])', filename)
            if match:
                return f"{title}, {match.group()}"
            return title

    # SEP articles
    if 'sep' in str(filepath).lower():
        return f"SEP: {filename.replace('-', ' ').replace('_', ' ').title()}"

    # arXiv papers
    if 'arxiv' in str(filepath).lower():
        return f"arXiv: {filename}"

    # Generic: use filename
    return f"{category}: {filename.replace('_', ' ').replace('-', ' ')}"


def detect_category(filepath: Path) -> str:
    """Detect corpus category from directory structure."""
    parts = [p.lower() for p in filepath.parts]

    categories = [
        'descartes_primary', 'rationalist_tradition',
        'philosophy_of_mind', 'contemporary_responses',
        'formal_logic', 'descartes_scholarship',
        'broader_philosophy', 'cognitive_science',
        'cross_disciplinary', 'neuroscience',
    ]
    for cat in categories:
        if cat in parts:
            return cat

    # Fallback: check for SEP / arXiv
    if 'sep' in parts:
        return 'philosophy_of_mind'
    if 'arxiv' in parts:
        return 'cross_disciplinary'

    return 'unknown'


# ============================================================
# INDEX BUILDER
# ============================================================

def build_index(input_dir: Path,
                chunk_size: int = 512,
                overlap: int = 64,
                verbose: bool = True) -> List[Dict]:
    """Build the corpus index from extracted text files.

    Walks the input directory, chunks every .txt file, extracts
    keywords, and builds index entries.
    """
    entries = []
    file_count = 0
    chunk_count = 0

    text_extensions = {'.txt', '.text', '.md'}

    for filepath in sorted(input_dir.rglob('*')):
        if filepath.suffix.lower() not in text_extensions:
            continue
        if filepath.stat().st_size < 100:  # Skip tiny files
            continue

        file_count += 1
        try:
            text = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            if verbose:
                print(f"  [WARN] Could not read {filepath}: {e}")
            continue

        source = detect_source(filepath, text)
        category = detect_category(filepath)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(chunks):
            keywords = extract_keywords(chunk)
            entries.append({
                "id": f"{filepath.stem}_{i:04d}",
                "source": source,
                "passage": chunk[:2000],  # Cap passage length
                "keywords": keywords,
                "category": category,
                "file": str(filepath.relative_to(input_dir)),
                "chunk_index": i,
            })
            chunk_count += 1

        if verbose and file_count % 50 == 0:
            print(f"  Processed {file_count} files, "
                  f"{chunk_count} chunks so far...")

    if verbose:
        print(f"\n  Total: {file_count} files -> {chunk_count} chunks")

    return entries


def save_index(entries: List[Dict], output_path: Path):
    """Save the corpus index as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    index = {
        "version": 1,
        "total_entries": len(entries),
        "categories": list(set(e["category"] for e in entries)),
        "entries": entries,
    }

    output_path.write_text(json.dumps(index, indent=2, ensure_ascii=False),
                           encoding='utf-8')


# ============================================================
# SEARCH (for testing and CorpusVerifier integration)
# ============================================================

def search_index(index_path: Path,
                 query: str,
                 max_results: int = 5) -> List[Dict]:
    """BM25-style keyword search over the corpus index.

    Simple but effective for factual claim verification:
    "Arnauld raised the circularity objection" →
    search for ["arnauld", "circularity", "objection"]
    """
    with open(index_path) as f:
        index = json.load(f)

    query_terms = set(re.findall(r'\b[a-z]+\b', query.lower()))
    query_terms -= STOPWORDS

    results = []
    for entry in index["entries"]:
        # Score: keyword overlap + philosophy term boost
        entry_keywords = set(entry["keywords"])
        overlap = query_terms & entry_keywords
        if not overlap:
            continue

        score = len(overlap)
        # Boost for philosophy-relevant matches
        for term in overlap:
            if term in PHILOSOPHY_TERMS:
                score += 1

        results.append({
            "source": entry["source"],
            "passage": entry["passage"][:300] + "...",
            "score": score,
            "matched_keywords": list(overlap),
            "category": entry["category"],
        })

    results.sort(key=lambda x: -x["score"])
    return results[:max_results]


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build searchable corpus index (Phase 11)")
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "corpus" / "extracted"),
        help="Input directory with extracted text files")
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "corpus" / "index.json"),
        help="Output path for the JSON index")
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Words per chunk (default: 512)")
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="Overlap words between chunks (default: 64)")
    parser.add_argument(
        "--test-query", type=str, default=None,
        help="Run a test query after building the index")

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("PHASE 11: Build Corpus Index")
    print("=" * 60)
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {output_path}")
    print(f"  Chunk size: {args.chunk_size} words")
    print(f"  Overlap:    {args.overlap} words")

    if not input_dir.exists():
        # Fallback: try cleaned/ or raw/ directories
        fallbacks = [
            PROJECT_ROOT / "corpus" / "cleaned",
            PROJECT_ROOT / "corpus" / "raw",
        ]
        for fb in fallbacks:
            if fb.exists():
                input_dir = fb
                print(f"  Fallback:   {input_dir}")
                break
        else:
            print(f"\n  [ERROR] No input directory found. "
                  f"Run text extraction first.")
            sys.exit(1)

    print()
    entries = build_index(
        input_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    if not entries:
        print("  [WARN] No entries generated. Check input directory.")
        sys.exit(1)

    save_index(entries, output_path)

    # Statistics
    categories = Counter(e["category"] for e in entries)
    print(f"\n  Index saved to: {output_path}")
    print(f"  Total entries:  {len(entries)}")
    print(f"  Categories:")
    for cat, count in categories.most_common():
        print(f"    {cat}: {count}")

    # Test query
    if args.test_query:
        print(f"\n  Test query: '{args.test_query}'")
        results = search_index(output_path, args.test_query)
        for i, r in enumerate(results, 1):
            print(f"    [{i}] score={r['score']} "
                  f"source={r['source']} "
                  f"matched={r['matched_keywords']}")

    print(f"\n{'=' * 60}")
    print("Phase 11 complete.")


if __name__ == "__main__":
    main()
