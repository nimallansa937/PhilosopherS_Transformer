"""
Phase 7: Supervised Fine-Tuning with LoRA on the CPT-adapted model.

Uses the CPT model as base (not the original Llama) and trains LoRA
adapters on the curated SFT examples.

Prerequisites:
    pip install torch transformers peft trl datasets wandb
    huggingface-cli login
    wandb login

Launch:
    python training/run_sft.py
    # Or with specific model:
    python training/run_sft.py --base models/philosopher-cpt-70b
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults
DEFAULT_BASE = str(PROJECT_ROOT / "models" / "philosopher-cpt-70b")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "models" / "philosopher-sft")
DEFAULT_SFT_DATA = str(PROJECT_ROOT / "training" / "sft" / "examples" / "sft_examples_reviewed.jsonl")

# LoRA config - lower rank than CPT (SFT needs less capacity)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


def format_chat(example, tokenizer=None):
    """Format SFT example into chat template."""
    messages = example["messages"]

    # Filter out rejected examples
    metadata = example.get("metadata", {})
    if metadata.get("review_status") == "rejected":
        return {"text": ""}

    # Try to use tokenizer's chat template if available
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            return {"text": text}
        except Exception:
            pass

    # Fallback: simple format
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    return {"text": "\n".join(parts)}


def main():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE,
                        help="Base model path (CPT model)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output directory for SFT model")
    parser.add_argument("--data", type=str, default=DEFAULT_SFT_DATA,
                        help="Path to reviewed SFT examples")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 7: SUPERVISED FINE-TUNING")
    print("=" * 60)
    print(f"  Base model: {args.base}")
    print(f"  SFT data:   {args.data}")
    print(f"  Output:     {args.output}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Load dataset
    print("\nLoading SFT dataset...")
    if not Path(args.data).exists():
        # Fall back to raw examples if reviewed don't exist
        fallback = str(PROJECT_ROOT / "training" / "sft" / "examples" / "sft_examples_raw.jsonl")
        if Path(fallback).exists():
            print(f"  Reviewed data not found, using raw: {fallback}")
            args.data = fallback
        else:
            print(f"  ERROR: No SFT data found at {args.data}")
            print("  Run Phase 6 (generate_examples.py) first.")
            return

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(lambda ex: format_chat(ex, tokenizer))
    dataset = dataset.filter(lambda x: len(x["text"]) > 50)
    print(f"  Examples after filtering: {len(dataset)}")

    if len(dataset) == 0:
        print("  ERROR: No valid examples after filtering.")
        return

    # Split 95/5
    split = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])} examples")
    print(f"  Eval:  {len(split['test'])} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="philosopher-sft",
    )

    # Train
    print("\nStarting SFT training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=4096,
        packing=False,  # Don't pack SFT examples
    )

    trainer.train()
    trainer.save_model(args.output)

    print(f"\n{'=' * 60}")
    print(f"SFT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
