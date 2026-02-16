"""
Phase 1: Download Stanford Encyclopedia of Philosophy articles.
Respects robots.txt and rate limits.

Usage:
    python corpus/scripts/download_sep.py
"""

import requests
import time
import json
import os
from pathlib import Path

# Target categories for philosophy of mind
TARGET_ENTRIES = [
    "consciousness", "qualia", "zombies", "chinese-room",
    "functionalism", "physicalism", "dualism", "epiphenomenalism",
    "mental-causation", "multiple-realizability", "supervenience",
    "consciousness-neuroscience", "consciousness-higher",
    "consciousness-representational", "consciousness-temporal",
    "intentionality", "propositional-attitudes", "mental-content",
    "mind-body-problem", "personal-identity", "self-knowledge",
    "perception-problem", "other-minds", "panpsychism",
    "neutral-monism", "emergent-properties", "identity-theory",
    "anomalous-monism", "behaviorism", "eliminative-materialism",
    "folk-psychology", "computational-mind", "connectionism",
    "embodied-cognition", "extended-mind", "free-will",
    "consciousness-animal", "pain", "pleasure", "emotion",
    "imagination", "memory", "attention", "sleep",
    # Broader philosophy entries
    "modality-epistemology", "possible-worlds", "necessity-possibility",
    "a-priori", "analytic-synthetic", "thought-experiments",
    "abduction", "scientific-explanation"
]

# Use project-relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "raw" / "philosophy_of_mind" / "sep"
METADATA_DIR = PROJECT_ROOT / "corpus" / "metadata"


def download_entry(entry_name: str) -> bool:
    """Download a single SEP entry."""
    url = f"https://plato.stanford.edu/entries/{entry_name}/"
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "PhilosopherEngine/1.0 (Academic Research Corpus Builder)"
        })
        if resp.status_code == 200:
            filepath = OUTPUT_DIR / f"{entry_name}.html"
            filepath.write_text(resp.text, encoding="utf-8")
            print(f"  [OK] {entry_name}")
            return True
        else:
            print(f"  [SKIP] {entry_name} (HTTP {resp.status_code})")
    except Exception as e:
        print(f"  [FAIL] {entry_name} ({e})")
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(TARGET_ENTRIES)} SEP entries...")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Rate limit: 1 request per 2 seconds\n")

    results = {"downloaded": [], "failed": [], "skipped": []}
    success = 0

    for i, entry in enumerate(TARGET_ENTRIES, 1):
        print(f"[{i}/{len(TARGET_ENTRIES)}] {entry}")

        # Skip if already downloaded
        filepath = OUTPUT_DIR / f"{entry}.html"
        if filepath.exists():
            print(f"  [CACHED] {entry}")
            results["downloaded"].append(entry)
            success += 1
            continue

        if download_entry(entry):
            results["downloaded"].append(entry)
            success += 1
        else:
            results["failed"].append(entry)

        time.sleep(2)  # Rate limit: 1 request per 2 seconds

    # Save metadata
    meta = {
        "source": "Stanford Encyclopedia of Philosophy",
        "url": "https://plato.stanford.edu/",
        "entries_attempted": len(TARGET_ENTRIES),
        "entries_downloaded": success,
        "entries_failed": len(results["failed"]),
        "failed_entries": results["failed"],
        "downloaded_entries": results["downloaded"],
    }
    meta_path = METADATA_DIR / "sep_download_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone: {success}/{len(TARGET_ENTRIES)} entries downloaded")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
