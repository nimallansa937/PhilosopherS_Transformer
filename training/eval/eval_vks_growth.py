"""
VKS Growth Evaluation — Track knowledge accumulation over time.

Measures:
1. VKS hit rate increase across repeated query epochs
2. Self-repair savings (oracle calls with vs. without repair)
3. Per-tier accumulation statistics
4. Time-to-answer improvement as VKS fills

Expected behavior:
  Epoch 0: ~5% hit rate  (VKS mostly empty)
  Epoch 1: ~30% hit rate (first pass stored many results)
  Epoch 2: ~50% hit rate (fuzzy matching catches paraphrases)
  Epoch 3: ~60% hit rate (approaching saturation)
  Epoch 4: ~65% hit rate (diminishing returns)

Targets (from V3 Architecture §6.1):
  - VKS hit rate >= 40% after 500 queries
  - VKS hit rate >= 60% after 2000 queries
  - Self-repair >= 35% of failed claims
  - Oracle calls <= 15% of all claims

Usage:
    python training/eval/eval_vks_growth.py

Reference: PHILOSOPHER_ENGINE_V3_UNIFIED_ARCHITECTURE.md, §6.2
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Test queries covering different claim types and topics
TEST_QUERIES = [
    # Formal claims (should hit VKS after first pass)
    "Is the Cogito a valid deductive argument?",
    "Is the Real Distinction argument valid in S5 modal logic?",
    "Can the zombie argument be formalized using the same modal structure?",
    "Is the Cartesian Circle formally circular?",

    # Factual claims (should hit corpus)
    "What did Arnauld object in the Fourth Objections?",
    "In which Meditation does Descartes introduce the wax argument?",
    "Who raised the interaction problem for substance dualism?",

    # Interpretive claims (soft pass, VKS for repeated queries)
    "Is the Cogito an inference or an intuition?",
    "How does Descartes' method of doubt differ from Pyrrhonian skepticism?",
    "What is the relationship between conceivability and possibility?",

    # Cross-theory (microtheory manager + VKS)
    "Is the zombie argument compatible with functionalism?",
    "Does IIT entail panpsychism?",
    "Can a physicalist accept the knowledge argument?",

    # Complex multi-claim queries
    "Compare the Real Distinction argument with the zombie argument. "
    "Are they structurally equivalent?",
    "Evaluate the Trademark Argument. Is the causal adequacy principle "
    "defensible?",

    # Paraphrase variants (tests fuzzy VKS matching)
    "Does the Cogito constitute a logically valid argument?",
    "Can mind and body be shown to be distinct using S5?",
    "Is Descartes' reasoning about God's existence circular?",
    "What was Arnauld's main criticism of the Meditations?",
    "Is the conceivability of zombies sufficient for their possibility?",
]


def eval_vks_growth(n_epochs: int = 5, verbose: bool = True):
    """Run test queries multiple epochs; measure VKS hit rate increase.

    This is the main VKS growth evaluation. Requires a running
    Ollama instance with descartes:8b loaded.
    """
    try:
        from inference.engine_v3 import DescartesEngineV3
    except ImportError:
        print("ERROR: Cannot import DescartesEngineV3.")
        print("Run from project root: python training/eval/eval_vks_growth.py")
        sys.exit(1)

    # Initialize engine with fresh VKS
    vks_path = os.path.expanduser("~/models/vks_eval.json")
    engine = DescartesEngineV3(
        local_model="descartes:8b",
        oracle_model="deepseek-v3.1:671-cloud",
        vks_path=vks_path,
    )

    results = []
    total_queries = len(TEST_QUERIES)

    print("=" * 60)
    print("VKS GROWTH EVALUATION")
    print(f"  Queries per epoch: {total_queries}")
    print(f"  Epochs: {n_epochs}")
    print("=" * 60)

    for epoch in range(n_epochs):
        epoch_start = time.monotonic()
        hits = 0
        total_claims = 0
        z3_count = 0
        corpus_count = 0
        soft_count = 0
        repair_count = 0
        oracle_count = 0
        latencies = []

        for i, query in enumerate(TEST_QUERIES):
            try:
                result = engine.run(query)

                n_claims = len(result.claims)
                total_claims += n_claims
                hits += result.vks_hits
                z3_count += result.z3_verified
                corpus_count += result.corpus_verified
                soft_count += result.soft_passed
                repair_count += result.self_repaired
                oracle_count += result.oracle_needed
                latencies.append(result.total_time_ms)

                if verbose:
                    print(f"  [{epoch}:{i+1:02d}] "
                          f"VKS:{result.vks_hits} Z3:{result.z3_verified} "
                          f"CORP:{result.corpus_verified} "
                          f"SOFT:{result.soft_passed} "
                          f"REP:{result.self_repaired} "
                          f"ORA:{result.oracle_needed} "
                          f"({result.total_time_ms:.0f}ms)")

            except Exception as e:
                if verbose:
                    print(f"  [{epoch}:{i+1:02d}] ERROR: {e}")
                continue

        epoch_time = (time.monotonic() - epoch_start)

        # Calculate metrics
        hit_rate = hits / max(total_claims, 1)
        oracle_rate = oracle_count / max(total_claims, 1)
        repair_rate = repair_count / max(
            repair_count + oracle_count, 1)
        avg_latency = (sum(latencies) / max(len(latencies), 1))

        vks_stats = engine.vks.get_stats()

        epoch_result = {
            "epoch": epoch,
            "total_claims": total_claims,
            "vks_hits": hits,
            "hit_rate": round(hit_rate, 4),
            "z3_verified": z3_count,
            "corpus_verified": corpus_count,
            "soft_passed": soft_count,
            "self_repaired": repair_count,
            "oracle_needed": oracle_count,
            "oracle_rate": round(oracle_rate, 4),
            "repair_rate": round(repair_rate, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "vks_total_records": vks_stats['total'],
            "vks_tiers": vks_stats['tiers'],
            "epoch_time_s": round(epoch_time, 1),
        }
        results.append(epoch_result)

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"  VKS hit rate:  {hit_rate:.1%} "
              f"(target: >=40% by epoch 2)")
        print(f"  Oracle rate:   {oracle_rate:.1%} "
              f"(target: <=15%)")
        print(f"  Repair rate:   {repair_rate:.1%} "
              f"(target: >=35%)")
        print(f"  Avg latency:   {avg_latency:.0f}ms")
        print(f"  VKS records:   {vks_stats['total']}")
        print(f"  Epoch time:    {epoch_time:.1f}s\n")

    # Save results
    results_path = str(PROJECT_ROOT / "training" / "eval" /
                       "vks_growth_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Final assessment
    print("\n" + "=" * 60)
    print("ASSESSMENT")
    print("=" * 60)

    final = results[-1]
    checks = [
        ("VKS hit rate >= 40%", final["hit_rate"] >= 0.40),
        ("Oracle rate <= 15%", final["oracle_rate"] <= 0.15),
        ("Repair rate >= 35%", final["repair_rate"] >= 0.35),
    ]

    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    return results


def eval_repair_savings(verbose: bool = True):
    """Compare oracle calls with and without self-repair enabled.

    Measures the % reduction in oracle calls from self-repair.
    Target: >= 35% reduction.
    """
    try:
        from inference.engine_v3 import DescartesEngineV3
    except ImportError:
        print("ERROR: Cannot import DescartesEngineV3.")
        sys.exit(1)

    print("=" * 60)
    print("SELF-REPAIR SAVINGS EVALUATION")
    print("=" * 60)

    # Run WITH repair
    print("\n--- Phase 1: With self-repair ---")
    engine = DescartesEngineV3(
        local_model="descartes:8b",
        oracle_model="deepseek-v3.1:671-cloud",
        vks_path=os.path.expanduser("~/models/vks_repair_test.json"),
    )

    oracle_with = 0
    total_with = 0
    for query in TEST_QUERIES[:10]:  # Subset for speed
        try:
            result = engine.run(query)
            oracle_with += result.oracle_needed
            total_with += len(result.claims)
            if verbose:
                print(f"  Oracle: {result.oracle_needed}/{len(result.claims)}")
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")

    # Run WITHOUT repair (disable it)
    print("\n--- Phase 2: Without self-repair ---")
    engine.repair.max_attempts = 0  # Disable repair

    oracle_without = 0
    total_without = 0
    for query in TEST_QUERIES[:10]:
        try:
            result = engine.run(query)
            oracle_without += result.oracle_needed
            total_without += len(result.claims)
            if verbose:
                print(f"  Oracle: {result.oracle_needed}/{len(result.claims)}")
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")

    if oracle_without > 0:
        savings = 1 - (oracle_with / oracle_without)
    else:
        savings = 0.0

    print(f"\n--- Results ---")
    print(f"  Oracle calls WITH repair:    {oracle_with}")
    print(f"  Oracle calls WITHOUT repair: {oracle_without}")
    print(f"  Savings: {savings:.0%}")
    print(f"  Target: >= 35%")
    print(f"  Status: {'PASS' if savings >= 0.35 else 'FAIL'}")

    return {"oracle_with": oracle_with,
            "oracle_without": oracle_without,
            "savings": savings}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VKS Growth Evaluation")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of query epochs (default: 3)")
    parser.add_argument("--repair-test", action="store_true",
                        help="Run self-repair savings test instead")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")
    args = parser.parse_args()

    if args.repair_test:
        eval_repair_savings(verbose=not args.quiet)
    else:
        eval_vks_growth(n_epochs=args.epochs, verbose=not args.quiet)
