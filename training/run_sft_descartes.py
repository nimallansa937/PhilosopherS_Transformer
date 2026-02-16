"""
Phase 7 (CASCADE): Two-Stage SFT for Descartes cascade model.

Stage 1: Philosophical reasoning (Types A-D) -- 3 epochs
Stage 2: Cascade behaviors (Types E-G) -- 2 epochs

Two-stage prevents cascade routing from interfering
with core reasoning capability.

Usage:
    python training/run_sft_descartes.py
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPT_MODEL = str(PROJECT_ROOT / "models" / "descartes-8b-cpt")
STAGE1_OUT = str(PROJECT_ROOT / "models" / "descartes-8b-sft-s1")
STAGE2_OUT = str(PROJECT_ROOT / "models" / "descartes-8b-cascade")

SFT_DIR = PROJECT_ROOT / "training" / "sft" / "examples"

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


def run_sft_stage(base_path: str, data_path: str, output_dir: str,
                  run_name: str, epochs: int = 3):
    """Run one SFT stage with LoRA."""

    print(f"\nLoading tokenizer from {base_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {base_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable / total:.2%})")

    print(f"Loading data from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Format into chat template
    def format_chat(example):
        messages = example.get("messages", [])
        meta = example.get("metadata", {})
        if meta.get("review_status") == "rejected":
            return {"text": ""}

        parts = []
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        parts.append("<|end|>")
        return {"text": "\n".join(parts)}

    dataset = dataset.map(format_chat)
    dataset = dataset.filter(lambda x: len(x["text"]) > 50)

    print(f"  {len(dataset)} examples after filtering")

    split = dataset.train_test_split(test_size=0.05, seed=42)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
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
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=4096,
        packing=False,
    )

    trainer.train()

    # Merge LoRA weights back into base model for next stage
    print("Merging LoRA weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nStage complete. Merged model saved to {output_dir}")


if __name__ == "__main__":
    SFT_STANDARD = str(SFT_DIR / "descartes_sft_types_ABCD.jsonl")
    SFT_CASCADE = str(SFT_DIR / "descartes_sft_types_EFG.jsonl")

    # Check files exist
    for p in [SFT_STANDARD, SFT_CASCADE]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run Phase 6 first.")
            sys.exit(1)

    print("=" * 60)
    print("PHASE 7: Two-Stage SFT — Descartes Cascade")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("STAGE 1: Core Philosophical Reasoning (Types A-D)")
    print("=" * 60)
    run_sft_stage(CPT_MODEL, SFT_STANDARD, STAGE1_OUT,
                  "descartes-sft-stage1", epochs=3)

    print("\n" + "=" * 60)
    print("STAGE 2: Cascade Behaviors (Types E-G)")
    print("=" * 60)
    run_sft_stage(STAGE1_OUT, SFT_CASCADE, STAGE2_OUT,
                  "descartes-sft-stage2", epochs=2)

    print("\n" + "=" * 60)
    print("TWO-STAGE SFT COMPLETE")
    print(f"Final model: {STAGE2_OUT}")
    print("=" * 60)
