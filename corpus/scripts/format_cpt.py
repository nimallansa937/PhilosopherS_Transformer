"""
Phase 4: Format cleaned corpus into tokenized JSONL for continued pre-training.

FORMAT: JSONL where each line is one document.
SPLIT: 95% train, 5% validation.
KEY RULE: Each document is a single continuous sequence.
No chunking - preserve long-range argumentative structure.
If a document exceeds max context length, split at
paragraph boundaries (never mid-sentence).

Supports: Llama 3.1 (default), Mixtral

Usage:
    python corpus/scripts/format_cpt.py
"""

import json
import os
import random
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "corpus" / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "formatted"

# Configuration
MAX_SEQ_LENGTH = 8192      # Tokens per training sequence
TRAIN_SPLIT = 0.95
SEED = 42

# Optional: use real tokenizer
TOKENIZER = None
try:
    from transformers import AutoTokenizer
    # Uncomment and set your model to use real tokenizer:
    # TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B")
    # print("Using real tokenizer for token counting")
except ImportError:
    pass


def estimate_tokens(text: str) -> int:
    """Estimate token count.

    Uses real tokenizer if available, otherwise rough heuristic.
    """
    if TOKENIZER is not None:
        return len(TOKENIZER.encode(text))
    # Rough estimate: ~1.3 tokens per whitespace-delimited word
    return int(len(text.split()) * 1.3)


def split_at_paragraph(text: str, max_tokens: int) -> List[str]:
    """Split long documents at paragraph boundaries.

    NEVER split mid-sentence - philosophical arguments lose
    coherence when cut at arbitrary points.
    """
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def format_corpus():
    """Format the entire corpus into training JSONL."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 4: CPT DATA FORMATTING")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH} tokens")
    print()

    # Collect all documents
    all_docs = []
    for filepath in INPUT_DIR.rglob("*.txt"):
        # Skip report files
        if filepath.name.startswith("cleaning"):
            continue
        text = filepath.read_text(encoding="utf-8")
        if len(text.strip()) < 50:
            continue
        category = filepath.relative_to(INPUT_DIR).parts[0]
        all_docs.append({
            "text": text,
            "category": category,
            "source": str(filepath.relative_to(INPUT_DIR))
        })

    print(f"Loaded {len(all_docs)} documents")

    if not all_docs:
        print("\nNo documents found! Run Phase 3 (cleaning) first.")
        return

    # Split long documents
    all_sequences = []
    split_count = 0
    for doc in all_docs:
        tokens = estimate_tokens(doc["text"])

        if tokens <= MAX_SEQ_LENGTH:
            all_sequences.append(doc)
        else:
            chunks = split_at_paragraph(doc["text"], MAX_SEQ_LENGTH)
            split_count += 1
            for i, chunk in enumerate(chunks):
                all_sequences.append({
                    "text": chunk,
                    "category": doc["category"],
                    "source": f"{doc['source']}__chunk_{i}"
                })

    print(f"After splitting: {len(all_sequences)} sequences "
          f"({split_count} docs were split)")

    # Shuffle
    random.seed(SEED)
    random.shuffle(all_sequences)

    # Split train/val
    split_idx = int(len(all_sequences) * TRAIN_SPLIT)
    train_seqs = all_sequences[:split_idx]
    val_seqs = all_sequences[split_idx:]

    # Write JSONL
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    total_train_tokens = 0
    with open(train_path, 'w', encoding='utf-8') as f:
        for seq in train_seqs:
            f.write(json.dumps({"text": seq["text"]}) + "\n")
            total_train_tokens += estimate_tokens(seq["text"])

    total_val_tokens = 0
    with open(val_path, 'w', encoding='utf-8') as f:
        for seq in val_seqs:
            f.write(json.dumps({"text": seq["text"]}) + "\n")
            total_val_tokens += estimate_tokens(seq["text"])

    # Compute category distribution
    category_dist = {}
    for seq in train_seqs:
        cat = seq["category"]
        if cat not in category_dist:
            category_dist[cat] = 0
        category_dist[cat] += 1

    # Write dataset card
    card = {
        "name": "philosopher-engine-cpt",
        "description": "Continued pre-training corpus for philosophical reasoning",
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "train_tokens_estimated": total_train_tokens,
        "val_tokens_estimated": total_val_tokens,
        "total_tokens_estimated": total_train_tokens + total_val_tokens,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_split": TRAIN_SPLIT,
        "seed": SEED,
        "category_distribution": category_dist,
    }

    card_path = OUTPUT_DIR / "dataset_card.json"
    card_path.write_text(json.dumps(card, indent=2))

    print(f"\n{'=' * 60}")
    print(f"FORMATTING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_seqs)} sequences, ~{total_train_tokens:,} tokens")
    print(f"  Val:   {len(val_seqs)} sequences, ~{total_val_tokens:,} tokens")
    print(f"  Total: ~{total_train_tokens + total_val_tokens:,} tokens")
    print(f"\nCategory distribution (train):")
    for cat, count in sorted(category_dist.items()):
        pct = count / len(train_seqs) * 100
        print(f"  {cat}: {count} sequences ({pct:.1f}%)")
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"Dataset card:   {card_path}")


if __name__ == "__main__":
    format_corpus()
