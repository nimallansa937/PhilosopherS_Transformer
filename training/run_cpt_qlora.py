"""
Phase 5 (Option B): QLoRA CPT - fits on a single A100 80GB or 2x A100 40GB.
Trains ~100M adapter parameters instead of all 70B.
Budget: $2K-$5K

Prerequisites:
    pip install torch transformers peft trl datasets wandb bitsandbytes flash-attn
    huggingface-cli login
    wandb login

Launch:
    python training/run_cpt_qlora.py
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "philosopher-cpt-qlora")
DATA_DIR = str(PROJECT_ROOT / "corpus" / "formatted")

# QLoRA quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config - target attention + MLP layers
lora_config = LoraConfig(
    r=64,                    # Rank - higher = more capacity
    lora_alpha=128,          # Scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


def main():
    print("=" * 60)
    print("PHASE 5: CONTINUED PRE-TRAINING (QLoRA)")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl"
    })
    print(f"  Train: {len(dataset['train'])} sequences")
    print(f"  Val:   {len(dataset['validation'])} sequences")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,      # Higher LR for LoRA adapters
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="philosopher-cpt-qlora",
    )

    print("\nStarting QLoRA training...")
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=8192,
        packing=True,  # Pack short sequences together for efficiency
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nQLoRA CPT complete. Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
