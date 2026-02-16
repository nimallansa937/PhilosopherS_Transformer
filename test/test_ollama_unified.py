"""
Verify both local and cloud models work through same Ollama API.

Tests:
  1. Local Descartes model responds to Cartesian queries
  2. Cloud oracle responds to broad philosophy queries
  3. Both use identical ollama.chat() interface

Prerequisites:
  1. Ollama installed and running
  2. descartes:8b model imported (see ADDENDUM Part 1.2)
  3. Cloud oracle configured (see ADDENDUM Part 1.3)

Usage:
    python test/test_ollama_unified.py
    python test/test_ollama_unified.py --local descartes:8b
    python test/test_ollama_unified.py --oracle deepseek-v3.1:671-cloud
    python test/test_ollama_unified.py --local-only
    python test/test_ollama_unified.py --cloud-only
"""

import sys
import argparse
import time


def test_local(model: str = "descartes:8b") -> bool:
    """Test local Descartes model via Ollama."""
    import ollama

    print(f"Testing local model: {model}")
    print("-" * 40)

    try:
        start = time.time()
        resp = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': 'What is the logical structure of the Cogito?'
            }]
        )
        elapsed = time.time() - start

        content = resp['message']['content']
        print(f"  Response ({elapsed:.1f}s): {content[:200]}...")
        print(f"  Length: {len(content)} chars")
        print(f"  PASS: Local model responding")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        print()
        print("  Troubleshooting:")
        print("    1. Is Ollama running? (ollama serve)")
        print(f"    2. Is '{model}' imported? (ollama list)")
        print("    3. See ADDENDUM Part 1.2 for import instructions")
        return False


def test_cloud(model: str = "deepseek-v3.1:671-cloud") -> bool:
    """Test cloud oracle via Ollama."""
    import ollama

    print(f"Testing cloud oracle: {model}")
    print("-" * 40)

    try:
        start = time.time()
        resp = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': "What was Merleau-Ponty's critique of Descartes?"
            }]
        )
        elapsed = time.time() - start

        content = resp['message']['content']
        print(f"  Response ({elapsed:.1f}s): {content[:200]}...")
        print(f"  Length: {len(content)} chars")
        print(f"  PASS: Cloud oracle responding")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        print()
        print("  Troubleshooting:")
        print("    1. Are you logged in? (ollama login)")
        print(f"    2. Is '{model}' available? (ollama pull {model})")
        print("    3. Check your Ollama Cloud subscription tier")
        print("    4. See ADDENDUM Part 1.3 for setup instructions")
        return False


def test_api_consistency(local_model: str = "descartes:8b",
                         oracle_model: str = "deepseek-v3.1:671-cloud") -> bool:
    """Verify both models use identical API structure."""
    import ollama

    print("Testing API consistency")
    print("-" * 40)

    query = "What is the Cogito?"
    messages = [{'role': 'user', 'content': query}]

    try:
        local_resp = ollama.chat(model=local_model, messages=messages)
        cloud_resp = ollama.chat(model=oracle_model, messages=messages)

        # Both should have same response structure
        local_keys = set(local_resp.keys())
        cloud_keys = set(cloud_resp.keys())

        # Check required keys present
        required = {'message', 'model'}
        local_has = required.issubset(local_keys)
        cloud_has = required.issubset(cloud_keys)

        print(f"  Local response keys: {sorted(local_keys)}")
        print(f"  Cloud response keys: {sorted(cloud_keys)}")
        print(f"  Local has required keys: {local_has}")
        print(f"  Cloud has required keys: {cloud_has}")

        # Both message objects should have 'role' and 'content'
        local_msg = local_resp['message']
        cloud_msg = cloud_resp['message']

        msg_consistent = (
            'role' in local_msg and 'content' in local_msg and
            'role' in cloud_msg and 'content' in cloud_msg
        )

        print(f"  Message structure consistent: {msg_consistent}")

        passed = local_has and cloud_has and msg_consistent
        print(f"  {'PASS' if passed else 'FAIL'}: "
              f"API structure consistent across local and cloud")
        return passed

    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_meta_learner_integration() -> bool:
    """Test that meta-learner components import and work."""
    print("Testing meta-learner integration")
    print("-" * 40)

    try:
        from pathlib import Path
        sys.path.insert(0, str(
            Path(__file__).resolve().parent.parent / "inference"))

        from signal_extractor_lite import LiteSignalExtractor
        from meta_learner import MetaLearnerLite, ROUTING_LABELS, ERROR_LABELS

        # Test signal extraction
        extractor = LiteSignalExtractor()
        test_text = ("The Cogito establishes that thinking is the one "
                     "indubitable fact. Descartes argues that even if an "
                     "evil genius deceives us about everything, the very "
                     "act of doubting proves existence.")
        signals = extractor.extract(test_text)
        tensor = signals.to_tensor()

        print(f"  Signal extractor: OK ({tensor.shape[0]} features)")

        # Test meta-learner forward pass
        import torch
        meta = MetaLearnerLite(input_dim=11)
        meta.eval()
        with torch.no_grad():
            output = meta(tensor)

        conf = output["confidence"].item()
        routing = output["routing_decision"]
        error = output["error_type"]

        print(f"  Meta-learner:     OK (conf={conf:.2f}, "
              f"route={routing}, error={error})")
        print(f"  Routing labels:   {ROUTING_LABELS}")
        print(f"  Error labels:     {ERROR_LABELS}")
        print(f"  PASS: Meta-learner components working")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Ollama unified API for Philosopher Engine")
    parser.add_argument(
        "--local", type=str, default="descartes:8b",
        help="Local Ollama model name")
    parser.add_argument(
        "--oracle", type=str, default="deepseek-v3.1:671-cloud",
        help="Cloud oracle Ollama model name")
    parser.add_argument(
        "--local-only", action="store_true",
        help="Only test local model")
    parser.add_argument(
        "--cloud-only", action="store_true",
        help="Only test cloud oracle")
    parser.add_argument(
        "--meta-only", action="store_true",
        help="Only test meta-learner integration (no Ollama needed)")

    args = parser.parse_args()

    print("=" * 50)
    print("OLLAMA UNIFIED API TEST")
    print("=" * 50)
    print()

    results = {}

    if args.meta_only:
        results["meta_learner"] = test_meta_learner_integration()
    elif args.local_only:
        results["local"] = test_local(args.local)
        print()
        results["meta_learner"] = test_meta_learner_integration()
    elif args.cloud_only:
        results["cloud"] = test_cloud(args.oracle)
    else:
        results["local"] = test_local(args.local)
        print()
        results["cloud"] = test_cloud(args.oracle)
        print()
        results["api_consistency"] = test_api_consistency(
            args.local, args.oracle)
        print()
        results["meta_learner"] = test_meta_learner_integration()

    # Summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_pass = True
    for test_name, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {test_name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed. Ollama unified API is ready.")
        if not args.local_only and not args.cloud_only and not args.meta_only:
            print("Both local and cloud models work through "
                  "identical ollama.chat() interface.")
    else:
        print("\nSome tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
