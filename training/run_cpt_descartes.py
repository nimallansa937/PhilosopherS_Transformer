"""
Phase 5 (CASCADE): Full CPT on Qwen3-8B with Descartes corpus.

VRAM budget for full-parameter CPT on 8B model:
  Model bf16:           ~16.4 GB
  Optimizer (AdamW):    ~32.8 GB (2x model for momentum + variance)
  Activations (grad ckpt, seq=2048, batch=1): ~4-6 GB
  Total:                ~53-55 GB
  -> Fits A100 80GB with headroom. Tight on A40 48GB.

Usage:
    python training/run_cpt_descartes.py

    # Override seq length or batch size:
    MAX_SEQ=4096 python training/run_cpt_descartes.py
    BATCH=2 ACCUM=16 python training/run_cpt_descartes.py
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
import json
from pathlib import Path

# ---- Configuration ----
MODEL_NAME = "Qwen/Qwen3-8B"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "descartes-8b-cpt")
DATA_DIR = str(PROJECT_ROOT / "corpus" / "formatted")

# Sequence length: 2048 is safe for 80GB, override with MAX_SEQ env var
MAX_SEQ = int(os.environ.get("MAX_SEQ", "2048"))


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
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("  Using: flash_attention_2 (fastest)")
    except ImportError:
        print("  Using: PyTorch SDPA (still fast)")

    # IMPORTANT: Do NOT use device_map="auto" for training.
    # It splits model across CPU/GPU which causes backward pass errors.
    # Load directly to GPU with .to("cuda").
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to("cuda")

    # Enable gradient checkpointing BEFORE creating optimizer
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Model VRAM: ~{total_params * 2 / 1e9:.1f} GB (bf16)")

    # Print actual VRAM usage after model load
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after model load: {alloc_gb:.1f} GB / {gpu_mem:.1f} GB "
          f"({gpu_mem - alloc_gb:.1f} GB free)")

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
    print(f"Tokenizing (max_length={MAX_SEQ})...")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ,
            padding=False,
            return_special_tokens_mask=True
        )

    tokenized = dataset.map(
        tokenize, batched=True,
        remove_columns=["text"], num_proc=4
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # ---- Determine batch size for available VRAM ----
    # After model load, remaining VRAM is for optimizer + activations.
    # Optimizer takes ~32.8 GB (AdamW states for 8B params).
    # With grad checkpointing, activations scale linearly with seq_len.
    # Safe defaults: batch=1, accum=32 for any 80GB GPU with 8B model.
    per_device_batch = int(os.environ.get("BATCH", "1"))
    grad_accum = int(os.environ.get("ACCUM", "32"))
    effective_batch = per_device_batch * grad_accum
    print(f"\nBatch config: {per_device_batch} x {grad_accum} = "
          f"{effective_batch} effective (seq_len={MAX_SEQ})")

    # ---- Check if wandb is available ----
    report_to = "none"
    try:
        import wandb
        if wandb.api.api_key:
            report_to = "wandb"
            print("Logging to: Weights & Biases")
        else:
            print("wandb not configured. Logging to console only.")
    except (ImportError, AttributeError):
        print("wandb not available. Logging to console only.")

    # ---- Training Arguments ----
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # 4 epochs for thorough domain absorption
        num_train_epochs=4,

        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,

        # Low LR to avoid catastrophic forgetting
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        bf16=True,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,  # 2 checkpoints to save disk

        eval_strategy="steps",
        eval_steps=500,

        logging_steps=10,
        report_to=report_to,
        run_name="descartes-8b-cpt",

        gradient_checkpointing=True,

        # Fused optimizer reduces VRAM fragmentation
        optim="adamw_torch_fused",

        # DataLoader
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        # Max sequence length for position embeddings
        max_grad_norm=1.0,
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
    # With batch=1 + grad_ckpt + seq=2048 on A100: ~1.5-2 steps/sec
    steps_per_sec = 1.5
    hours_est = total_steps / steps_per_sec / 3600
    cost_est = hours_est * 0.75  # A100 rate

    print(f"\n{'=' * 60}")
    print(f"TRAINING PLAN")
    print(f"{'=' * 60}")
    print(f"  Model:         {MODEL_NAME}")
    print(f"  Samples:       {n_samples:,}")
    print(f"  Epochs:        4")
    print(f"  Seq length:    {MAX_SEQ}")
    print(f"  Batch size:    {effective_batch} (effective)")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Est. time:     {hours_est:.1f} hours")
    print(f"  Est. cost:     ${cost_est:.2f} (at $0.75/hr)")
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
    summary = {
        "model": MODEL_NAME,
        "output_dir": OUTPUT_DIR,
        "train_samples": n_samples,
        "val_samples": len(tokenized["validation"]),
        "epochs": 4,
        "max_seq_length": MAX_SEQ,
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
