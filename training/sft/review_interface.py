"""
Phase 6: Terminal-based human review interface for SFT examples.

Presents each example and allows: approve, edit, reject, skip.
Tracks review progress and inter-annotator agreement.

Usage:
    python training/sft/review_interface.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = PROJECT_ROOT / "training" / "sft" / "examples" / "sft_examples_raw.jsonl"
OUTPUT = PROJECT_ROOT / "training" / "sft" / "examples" / "sft_examples_reviewed.jsonl"


def load_examples():
    """Load examples from JSONL file."""
    examples = []
    if INPUT.exists():
        with open(INPUT, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

    # Also load any previously reviewed examples for merging
    reviewed = {}
    if OUTPUT.exists():
        with open(OUTPUT, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    ex_id = ex.get("metadata", {}).get("id", "")
                    if ex_id:
                        reviewed[ex_id] = ex

    # Merge review status into raw examples
    for ex in examples:
        ex_id = ex.get("metadata", {}).get("id", "")
        if ex_id in reviewed:
            ex["metadata"] = reviewed[ex_id]["metadata"]
            ex["messages"] = reviewed[ex_id]["messages"]

    return examples


def save_examples(examples):
    """Save examples to JSONL file."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def display_example(example, index, total):
    """Display a single example for review."""
    meta = example["metadata"]
    msgs = example["messages"]

    print(f"\n{'=' * 60}")
    print(f"Example {index}/{total}")
    print(f"{'=' * 60}")
    print(f"ID: {meta['id']}  |  Type: {meta['type']}  |  "
          f"Philosopher: {meta['philosopher']}")
    print(f"Z3 Validated: {meta.get('z3_validated', 'N/A')}  |  "
          f"Council Agreement: {meta.get('council_agreement', 0):.2f}")
    print(f"Difficulty Tier: {meta.get('tier', 'N/A')}")
    print(f"\n--- SYSTEM PROMPT (truncated) ---")
    print(f"{msgs[0]['content'][:200]}...")
    print(f"\n--- USER PROMPT ---")
    print(f"{msgs[1]['content'][:500]}...")
    print(f"\n--- ASSISTANT RESPONSE ---")
    response = msgs[2]['content']
    if len(response) > 800:
        print(f"{response[:800]}...")
        print(f"\n  [... {len(response) - 800} more characters ...]")
    else:
        print(response)


def review_session():
    """Run an interactive review session."""
    examples = load_examples()

    if not examples:
        print("No examples found. Run generate_examples.py first.")
        print(f"Expected file: {INPUT}")
        return

    # Filter for unreviewed
    pending = [(i, e) for i, e in enumerate(examples)
               if not e.get("metadata", {}).get("human_reviewed")]

    total = len(examples)
    reviewed_total = total - len(pending)

    print("=" * 60)
    print("SFT EXAMPLE REVIEW INTERFACE")
    print("=" * 60)
    print(f"\nTotal examples: {total}")
    print(f"Already reviewed: {reviewed_total}")
    print(f"Pending review: {len(pending)}")
    print(f"\nCommands:")
    print(f"  [a] Approve   - Mark as good for training")
    print(f"  [e] Edit      - Modify the response (opens editor)")
    print(f"  [r] Reject    - Mark as bad / exclude from training")
    print(f"  [f] Full view - Show complete response text")
    print(f"  [s] Skip      - Skip for now")
    print(f"  [q] Quit      - Save progress and exit")

    reviewed_count = 0

    for idx, (orig_idx, example) in enumerate(pending):
        display_example(example, idx + 1, len(pending))

        while True:
            try:
                action = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                action = "q"

            if action == 'a':
                example["metadata"]["human_reviewed"] = True
                example["metadata"]["review_status"] = "approved"
                reviewed_count += 1
                print("  -> APPROVED")
                break

            elif action == 'e':
                print("\nEnter corrected response (type END on a new line to finish):")
                lines = []
                while True:
                    try:
                        line = input()
                        if line.strip() == "END":
                            break
                        lines.append(line)
                    except (EOFError, KeyboardInterrupt):
                        break
                new_response = "\n".join(lines)
                if new_response.strip():
                    example["messages"][2]["content"] = new_response
                    example["metadata"]["human_reviewed"] = True
                    example["metadata"]["review_status"] = "edited"
                    reviewed_count += 1
                    print("  -> EDITED")
                else:
                    print("  -> Edit cancelled (empty response)")
                    continue
                break

            elif action == 'r':
                example["metadata"]["human_reviewed"] = True
                example["metadata"]["review_status"] = "rejected"
                reviewed_count += 1
                print("  -> REJECTED")
                break

            elif action == 'f':
                print(f"\n--- FULL RESPONSE ---")
                print(example["messages"][2]["content"])
                continue

            elif action == 's':
                print("  -> SKIPPED")
                break

            elif action == 'q':
                save_examples(examples)
                print(f"\nSession saved. Reviewed {reviewed_count} examples this session.")
                print(f"Total reviewed: {reviewed_total + reviewed_count}/{total}")
                print(f"Saved to: {OUTPUT}")
                return

            else:
                print("  Invalid command. Use: a/e/r/f/s/q")

    # Save all
    save_examples(examples)

    print(f"\n{'=' * 60}")
    print(f"REVIEW SESSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Reviewed this session: {reviewed_count}")
    print(f"Total reviewed: {reviewed_total + reviewed_count}/{total}")
    print(f"Saved to: {OUTPUT}")

    # Summary
    statuses = {}
    for ex in examples:
        status = ex.get("metadata", {}).get("review_status", "pending")
        statuses[status] = statuses.get(status, 0) + 1
    print(f"\nStatus breakdown: {statuses}")


if __name__ == "__main__":
    review_session()
