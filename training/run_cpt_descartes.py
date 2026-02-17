"""
Phase 5 (CASCADE): Full CPT on Qwen3-8B with Descartes corpus.
Single A40 48GB or A6000 Ada. No multi-GPU, no QLoRA needed.

Expected: ~1 hour training, ~$0.55 GPU cost on A40.

Usage:
    python training/run_cpt_descartes.py

Vast.ai quick-start (copy-paste into fresh terminal):
    See bottom of this file or README for full setup commands.
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
import time
from pathlib import Path

# ---- Configuration ----
MODEL_NAME = "Qwen/Qwen3-8B"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "descartes-8b-cpt")
DATA_DIR = str(PROJECT_ROOT / "corpus" / "formatted")

# VRAM Budget (A40 48GB):
#   Model bf16:          ~16.4 GB
#   Optimizer (AdamW):   ~32.8 GB (2x model for momentum + variance)
#   Activations (grad ckpt): ~2-4 GB
#   Total:               ~51-53 GB -> TIGHT on 48GB
#
# Solution: gradient_checkpointing=True + per_device_batch=2 + grad_accum=16
# Effective batch stays 32, but peak VRAM drops to ~40-44GB.
# If still OOM, reduce per_device_batch to 1 + grad_accum to 32.


def main():
    print("=" * 60)
    print("PHASE 5: CPT Training - Qwen3-8B on Descartes Corpus")
    print("=" * 60)

    # ---- Check GPU ----
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. CPT requires GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    print(f"Loading model from {MODEL_NAME}...")

    # Use flash_attention_2 if available, fall back to sdpa
    attn_impl = "sdpa"  # PyTorch native, always available
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("  Using: flash_attention_2 (fastest)")
    except ImportError:
        print("  flash-attn not installed, using PyTorch SDPA (still fast)")
        print("  To install: MAX_JOBS=4 pip install flash-attn --no-build-isolation")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Model VRAM: ~{total_params * 2 / 1e9:.1f} GB (bf16)")

    # ---- Dataset ----
    print(f"\nLoading dataset from {DATA_DIR}...")
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found!")
        print(f"Upload train.jsonl and val.jsonl to {DATA_DIR}/")
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

    # ---- Determine batch size for available VRAM ----
    # A40 48GB: batch=2, accum=16 -> effective 32 (safe)
    # A100 80GB: batch=4, accum=8 -> effective 32 (comfortable)
    # H100 80GB: batch=8, accum=4 -> effective 32 (fast)
    if gpu_mem >= 75:
        per_device_batch = 4
        grad_accum = 8
    elif gpu_mem >= 45:
        per_device_batch = 2
        grad_accum = 16
    else:
        per_device_batch = 1
        grad_accum = 32
    effective_batch = per_device_batch * grad_accum
    print(f"\nBatch config: {per_device_batch} x {grad_accum} = "
          f"{effective_batch} effective")

    # ---- Check if wandb is available ----
    report_to = "none"
    try:
        import wandb
        if wandb.api.api_key:
            report_to = "wandb"
            print("Logging to: Weights & Biases")
        else:
            print("wandb installed but not logged in. Logging disabled.")
            print("  To enable: wandb login")
    except (ImportError, AttributeError):
        print("wandb not installed. Logging to console only.")
        print("  To enable: pip install wandb && wandb login")

    # ---- Training Arguments ----
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # 4 epochs for thorough domain absorption on 8B model
        num_train_epochs=4,

        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,

        # Low LR to avoid catastrophic forgetting
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        bf16=True,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        eval_strategy="steps",
        eval_steps=500,

        logging_steps=10,
        report_to=report_to,
        run_name="descartes-8b-cpt",

        gradient_checkpointing=True,

        # Prevents OOM from optimizer state fragmentation
        optim="adamw_torch_fused",

        # DataLoader workers
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
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
    total_steps = (n_samples * 4) // effective_batch
    # A40: ~1.5-2.5 steps/sec with batch=2, grad_ckpt, 8192 seq
    steps_per_sec = 2.0
    hours_est = total_steps / steps_per_sec / 3600
    cost_est = hours_est * 0.55

    print(f"\n{'=' * 60}")
    print(f"TRAINING PLAN")
    print(f"{'=' * 60}")
    print(f"  Model:         {MODEL_NAME}")
    print(f"  Samples:       {n_samples:,}")
    print(f"  Epochs:        4")
    print(f"  Batch size:    {effective_batch} (effective)")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Est. time:     {hours_est:.1f} hours")
    print(f"  Est. cost:     ${cost_est:.2f} (at $0.55/hr)")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # ---- Train ----
    t0 = time.time()
    print("\nStarting training...")
    trainer.train()

    elapsed = (time.time() - t0) / 3600
    print(f"\nTraining complete in {elapsed:.2f} hours")

    # ---- Save final model ----
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # ---- Save training summary ----
    import json
    summary = {
        "model": MODEL_NAME,
        "output_dir": OUTPUT_DIR,
        "train_samples": n_samples,
        "val_samples": len(tokenized["validation"]),
        "epochs": 4,
        "effective_batch_size": effective_batch,
        "total_steps": total_steps,
        "training_hours": round(elapsed, 2),
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem, 1),
    }
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
