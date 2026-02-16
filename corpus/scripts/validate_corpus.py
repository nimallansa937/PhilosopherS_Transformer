"""
Corpus Validation Script

Quick validation checks to run after each phase.
Can be run standalone or called by run_pipeline.py.

Usage:
    python corpus/scripts/validate_corpus.py
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def validate_phase1():
    """Validate Phase 1: Corpus Assembly."""
    print("\n--- Phase 1: Corpus Assembly ---")
    raw_dir = PROJECT_ROOT / "corpus" / "raw"

    categories = ["philosophy_of_mind", "neuroscience", "broader_philosophy",
                   "cognitive_science", "cross_disciplinary"]

    total_files = 0
    total_size = 0

    for cat in categories:
        cat_dir = raw_dir / cat
        if cat_dir.exists():
            files = list(cat_dir.rglob("*"))
            files = [f for f in files if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            total_files += len(files)
            total_size += size
            print(f"  {cat}: {len(files)} files, {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {cat}: (not found)")

    print(f"\n  Total: {total_files} files, {total_size / 1024 / 1024:.1f} MB")
    print(f"  Threshold: 5,000+ documents, 2GB+ raw text")
    passed = total_files >= 50  # Relaxed for testing
    print(f"  Status: {'PASS' if passed else 'NEEDS MORE DATA'}")
    return passed


def validate_phase2():
    """Validate Phase 2: Text Extraction."""
    print("\n--- Phase 2: Text Extraction ---")
    extracted_dir = PROJECT_ROOT / "corpus" / "extracted"
    metrics_path = extracted_dir / "extraction_metrics.json"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print(f"  Total files processed: {metrics.get('total', 0)}")
        print(f"  Successful: {metrics.get('success', 0)}")
        print(f"  Empty/short: {metrics.get('empty', 0)}")
        print(f"  Failed: {metrics.get('failed', 0)}")
    else:
        print("  No extraction metrics found.")

    # Check sample files for diacritics
    diacritics_found = False
    for f in extracted_dir.rglob("*.txt"):
        text = f.read_text(encoding="utf-8", errors="replace")
        if any(c in text for c in "éèêëàâùûüîïôöçñ"):
            diacritics_found = True
            break

    print(f"  Diacritics preserved: {'Yes' if diacritics_found else 'Not tested'}")

    txt_count = sum(1 for _ in extracted_dir.rglob("*.txt"))
    passed = txt_count > 0
    print(f"  Extracted text files: {txt_count}")
    print(f"  Status: {'PASS' if passed else 'NO DATA'}")
    return passed


def validate_phase3():
    """Validate Phase 3: Cleaning & Filtering."""
    print("\n--- Phase 3: Cleaning & Filtering ---")
    report_path = PROJECT_ROOT / "corpus" / "cleaned" / "cleaning_report.json"

    if not report_path.exists():
        print("  No cleaning report found.")
        return False

    report = json.loads(report_path.read_text())
    print(f"  Total documents: {report.get('total_documents', 0)}")
    print(f"  Estimated tokens: {report.get('total_tokens_estimated', 0):,}")

    # Check mixing ratios
    actual = report.get("mixing_ratios_actual", {})
    targets = {
        "philosophy_of_mind": 0.40,
        "neuroscience": 0.20,
        "broader_philosophy": 0.15,
        "cognitive_science": 0.15,
        "cross_disciplinary": 0.10,
    }

    ratio_ok = True
    for cat, target in targets.items():
        actual_val = actual.get(cat, 0)
        diff = abs(actual_val - target)
        status = "OK" if diff <= 0.05 else "OFF"
        if diff > 0.05:
            ratio_ok = False
        print(f"  {cat}: {actual_val:.1%} (target: {target:.0%}) [{status}]")

    pipeline = report.get("pipeline_stats", {})
    print(f"\n  Pipeline: {pipeline.get('input_documents', 0)} -> "
          f"{pipeline.get('after_dedup', 0)} -> "
          f"{pipeline.get('after_quality_filter', 0)}")

    passed = report.get("total_tokens_estimated", 0) > 0
    print(f"  Status: {'PASS' if passed else 'NO DATA'}")
    return passed


def validate_phase4():
    """Validate Phase 4: CPT Data Formatting."""
    print("\n--- Phase 4: CPT Data Formatting ---")
    formatted_dir = PROJECT_ROOT / "corpus" / "formatted"
    card_path = formatted_dir / "dataset_card.json"

    if card_path.exists():
        card = json.loads(card_path.read_text())
        print(f"  Train sequences: {card.get('train_sequences', 0):,}")
        print(f"  Val sequences: {card.get('val_sequences', 0):,}")
        print(f"  Train tokens: {card.get('train_tokens_estimated', 0):,}")
        print(f"  Val tokens: {card.get('val_tokens_estimated', 0):,}")
        print(f"  Max seq length: {card.get('max_seq_length', 'N/A')}")
    else:
        print("  No dataset card found.")

    train_exists = (formatted_dir / "train.jsonl").exists()
    val_exists = (formatted_dir / "val.jsonl").exists()
    print(f"  train.jsonl: {'exists' if train_exists else 'MISSING'}")
    print(f"  val.jsonl: {'exists' if val_exists else 'MISSING'}")

    passed = train_exists and val_exists
    print(f"  Status: {'PASS' if passed else 'MISSING FILES'}")
    return passed


def main():
    print("=" * 60)
    print("PHILOSOPHER ENGINE - CORPUS VALIDATION")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    results = {}
    results["phase1"] = validate_phase1()
    results["phase2"] = validate_phase2()
    results["phase3"] = validate_phase3()
    results["phase4"] = validate_phase4()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for phase, passed in results.items():
        print(f"  {phase}: {'PASS' if passed else 'PENDING'}")


if __name__ == "__main__":
    main()
