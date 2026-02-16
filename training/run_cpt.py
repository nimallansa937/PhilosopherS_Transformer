"""
Phase 5 (Option A): Continued Pre-Training on Llama 3.1 70B.

Uses DeepSpeed ZeRO-3 for distributed training across 4x A100 80GB.
Budget: $10K-$20K

Environment: 4x A100 80GB (e.g., Lambda Labs, RunPod, AWS p4d.24xlarge)

Prerequisites:
    pip install torch transformers accelerate deepspeed datasets wandb bitsandbytes flash-attn
    huggingface-cli login
    wandb login

Launch:
    deepspeed --num_gpus=4 training/run_cpt.py
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
from pathlib import Path

# ---- Configuration ----
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "philosopher-cpt-70b")
DATA_DIR = str(PROJECT_ROOT / "corpus" / "formatted")
DS_CONFIG = str(PROJECT_ROOT / "training" / "ds_config.json")

TRAINING_CONFIG = {
    # Learning rate: LOW to avoid catastrophic forgetting
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,

    # Batch size: effective = per_device * gradient_accum * num_gpus
    # 2 * 4 * 4 = 32 sequences per step
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,

    # Duration: 1-3 epochs over the corpus
    "num_train_epochs": 2,
    "max_steps": -1,  # Let epochs determine

    # Precision: bf16 for A100s
    "bf16": True,
    "tf32": True,

    # Checkpointing
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,

    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 250,

    # Logging
    "logging_steps": 10,
    "report_to": "wandb",
    "run_name": "philosopher-cpt-70b",

    # Memory optimization
    "gradient_checkpointing": True,
    "deepspeed": DS_CONFIG,

    # Output
    "output_dir": OUTPUT_DIR,
}


def main():
    print("=" * 60)
    print("PHASE 5: CONTINUED PRE-TRAINING (Full 70B)")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl"
    })
    print(f"  Train: {len(dataset['train'])} sequences")
    print(f"  Val:   {len(dataset['validation'])} sequences")

    # Tokenize
    print("Tokenizing...")
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=8192,
            padding=False,
            return_special_tokens_mask=True
        )

    tokenized = dataset.map(tokenize, batched=True,
                            remove_columns=["text"],
                            num_proc=8)

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Data collator for causal LM (shifts labels by 1)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training
    print("\nStarting training...")
    args = TrainingArguments(**TRAINING_CONFIG)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nCPT complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
