"""
Phase 5 (CASCADE): Full CPT on Qwen3-8B with Descartes corpus.
Single A40 48GB or A6000 Ada. No multi-GPU, no QLoRA needed.

Expected: 8-15 hours training, $5-$10 GPU cost.

Usage:
    python training/run_cpt_descartes.py

Vast.ai setup:
    pip install torch transformers accelerate datasets wandb \
        flash-attn bitsandbytes sentencepiece --break-system-packages
    huggingface-cli login
    wandb login
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os
import sys
from pathlib import Path

# ---- Configuration ----
MODEL_NAME = "Qwen/Qwen3-8B"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "descartes-8b-cpt")
DATA_DIR = str(PROJECT_ROOT / "corpus" / "formatted")


def main():
    print("=" * 60)
    print("PHASE 5: CPT Training — Qwen3-8B on Descartes Corpus")
    print("=" * 60)

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    print(f"Loading model from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"VRAM estimate: {total_params * 2 / 1e9:.1f} GB (bf16)")
    # ~16 GB model + ~32 GB optimizer = ~48 GB -> fits A40 48GB

    # ---- Dataset ----
    print(f"\nLoading dataset from {DATA_DIR}...")
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run Phases 1-4 first.")
        sys.exit(1)

    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": val_path
    })

    print(f"  Train: {len(dataset['train']):,} sequences")
    print(f"  Val:   {len(dataset['validation']):,} sequences")

    # ---- Tokenize ----
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=8192,
            padding=False,
            return_special_tokens_mask=True
        )

    print("Tokenizing...")
    tokenized = dataset.map(
        tokenize, batched=True,
        remove_columns=["text"], num_proc=4
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # ---- Training Arguments ----
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # 4 epochs — more than 70B because 8B absorbs more per epoch
        num_train_epochs=4,

        # Larger batches fit because model is small
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        # Effective batch: 8 x 4 = 32 sequences

        # Low LR to avoid catastrophic forgetting
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        bf16=True,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,

        eval_strategy="steps",
        eval_steps=250,

        logging_steps=10,
        report_to="wandb",
        run_name="descartes-8b-cpt",

        gradient_checkpointing=True,
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    # ---- Estimate time and cost ----
    n_samples = len(tokenized["train"])
    total_steps = (n_samples * 4) // 32  # samples x epochs / effective_batch
    hours_est = total_steps / 2 / 3600   # ~2 steps/sec on A40
    cost_est = hours_est * 0.55
    print(f"\nTraining plan:")
    print(f"  Samples:   {n_samples:,}")
    print(f"  Steps:     {total_steps:,}")
    print(f"  Est. time: {hours_est:.1f} hours")
    print(f"  Est. cost: ${cost_est:.2f} (at $0.55/hr)")

    # ---- Train ----
    print("\nStarting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nCPT complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
