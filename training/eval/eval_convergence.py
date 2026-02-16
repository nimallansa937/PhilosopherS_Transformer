"""
Track meta-learner improvement over time.

Analyzes the feedback buffer history to compute:
  - Confidence calibration error at checkpoints
  - Routing accuracy at checkpoints
  - Error type classification accuracy

Shows whether the meta-learner converges to useful routing
within the expected 200-500 oracle interactions.

Pass Criteria:
  - Calibration error <= 0.15 by 500 interactions
  - Routing accuracy >= 80% by 500 interactions
  - Decreasing calibration error trend
  - Increasing routing accuracy trend

Usage:
    python training/eval/eval_convergence.py
    python training/eval/eval_convergence.py --meta models/meta_learner_bootstrapped.pt
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def eval_convergence(meta_path: str, save_results: bool = True) -> dict:
    """Load buffer history and compute metrics at checkpoints.

    Args:
        meta_path: Path to meta-learner .pt checkpoint
                   (buffer is at same path with _buffer.json suffix)
        save_results: Whether to save results JSON

    Returns:
        Dict with metrics at each checkpoint
    """
    buffer_path = meta_path.replace('.pt', '_buffer.json')

    try:
        with open(buffer_path) as f:
            buffer = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Buffer not found at {buffer_path}")
        print("The meta-learner must have been trained with feedback data.")
        sys.exit(1)

    total_interactions = len(buffer)
    print(f"\nConvergence Evaluation")
    print(f"Buffer: {buffer_path}")
    print(f"Total interactions: {total_interactions}")
    print()

    if total_interactions == 0:
        print("ERROR: Empty buffer, no interactions to evaluate.")
        return {"error": "empty_buffer"}

    # Standard checkpoints + final
    checkpoints = [50, 100, 200, 500, 1000]
    checkpoints = [cp for cp in checkpoints if cp <= total_interactions]
    if total_interactions not in checkpoints:
        checkpoints.append(total_interactions)
    checkpoints.sort()

    print(f"{'Checkpoint':>10} {'Calib Error':>12} {'Route Acc':>10} "
          f"{'Error Acc':>10} {'Avg Conf':>10}")
    print("-" * 57)

    route_map = {"SELF": 0, "ORACLE": 1, "HYBRID": 2}
    results_per_cp = []

    for cp in checkpoints:
        subset = buffer[:cp]

        # --- Calibration error: |predicted_confidence - true_confidence| ---
        calib_errors = []
        for s in subset:
            pred_conf = s.get("predicted_confidence", 0.5)
            true_conf = s.get("true_confidence", 0.5)
            calib_errors.append(abs(pred_conf - true_conf))
        avg_calib = sum(calib_errors) / len(calib_errors)

        # --- Routing accuracy ---
        route_correct = 0
        route_total = 0
        for s in subset:
            pred_route = route_map.get(s.get("predicted_routing"), -1)
            true_route = s.get("true_routing", -2)
            if pred_route >= 0:
                route_total += 1
                if pred_route == true_route:
                    route_correct += 1
        route_acc = route_correct / max(route_total, 1)

        # --- Error type accuracy ---
        error_correct = 0
        error_total = 0
        for s in subset:
            # Error type prediction is harder — track if available
            if "predicted_error" in s and "true_error" in s:
                error_total += 1
                if s["predicted_error"] == s["true_error"]:
                    error_correct += 1
        error_acc = error_correct / max(error_total, 1) if error_total > 0 else None

        # --- Average confidence ---
        avg_conf = sum(s.get("predicted_confidence", 0.5)
                       for s in subset) / len(subset)

        error_str = f"{error_acc:.1%}" if error_acc is not None else "N/A"
        print(f"{cp:>10} {avg_calib:>12.3f} {route_acc:>10.1%} "
              f"{error_str:>10} {avg_conf:>10.2f}")

        results_per_cp.append({
            "checkpoint": cp,
            "calibration_error": round(avg_calib, 4),
            "routing_accuracy": round(route_acc, 4),
            "error_type_accuracy": round(error_acc, 4) if error_acc is not None else None,
            "avg_confidence": round(avg_conf, 4),
        })

    # --- Trend analysis ---
    print()
    if len(results_per_cp) >= 2:
        first = results_per_cp[0]
        last = results_per_cp[-1]

        calib_improving = last["calibration_error"] < first["calibration_error"]
        route_improving = last["routing_accuracy"] > first["routing_accuracy"]

        print(f"Calibration trend: {'IMPROVING' if calib_improving else 'DEGRADING'} "
              f"({first['calibration_error']:.3f} -> {last['calibration_error']:.3f})")
        print(f"Routing trend:     {'IMPROVING' if route_improving else 'DEGRADING'} "
              f"({first['routing_accuracy']:.1%} -> {last['routing_accuracy']:.1%})")
    else:
        calib_improving = None
        route_improving = None
        print("Not enough checkpoints for trend analysis.")

    # --- Pass/fail criteria ---
    final = results_per_cp[-1]
    calib_pass = final["calibration_error"] <= 0.15
    route_pass = final["routing_accuracy"] >= 0.80

    print()
    print(f"Final calibration error: {final['calibration_error']:.3f} "
          f"{'PASS' if calib_pass else 'FAIL'} (threshold: <= 0.15)")
    print(f"Final routing accuracy:  {final['routing_accuracy']:.1%} "
          f"{'PASS' if route_pass else 'FAIL'} (threshold: >= 80%)")

    overall_pass = calib_pass and route_pass
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        print("\nRemediation steps:")
        if not calib_pass:
            print("  1. Increase bootstrap questions (target 500+)")
            print("  2. Check feedback signal quality (Z3 verdicts improve calibration)")
        if not route_pass:
            print("  1. Check bootstrap question distribution (40/30/30 split)")
            print("  2. Generate more SFT examples for weak routing categories")
            print("  3. Re-bootstrap meta-learner with more questions")

    # Expected convergence
    print(f"\nExpected: calibration error decreases, "
          f"routing accuracy increases over interactions.")

    results = {
        "total_interactions": total_interactions,
        "checkpoints": results_per_cp,
        "calibration_pass": calib_pass,
        "routing_pass": route_pass,
        "overall_pass": overall_pass,
        "calibration_improving": calib_improving,
        "routing_improving": route_improving,
        "meta_path": meta_path,
        "buffer_path": buffer_path,
    }

    if save_results:
        results_path = (PROJECT_ROOT / "training" / "eval" /
                        "convergence_eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate meta-learner convergence over time")
    parser.add_argument(
        "--meta", type=str, default=None,
        help="Path to meta-learner checkpoint (.pt)")
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to disk")

    args = parser.parse_args()

    # Find meta-learner checkpoint
    meta_path = args.meta
    if meta_path is None:
        for candidate in [
            PROJECT_ROOT / "models" / "meta_learner_bootstrapped.pt",
            PROJECT_ROOT / "models" / "meta_learner_bootstrap.pt",
            PROJECT_ROOT / "models" / "meta_learner_latest.pt",
        ]:
            if candidate.exists():
                meta_path = str(candidate)
                break

    if meta_path is None:
        print("ERROR: No meta-learner checkpoint found.")
        print("Run training/bootstrap_meta.py first, or pass --meta PATH")
        sys.exit(1)

    eval_convergence(meta_path, save_results=not args.no_save)


if __name__ == "__main__":
    main()
