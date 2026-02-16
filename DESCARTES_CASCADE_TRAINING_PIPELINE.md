# Philosopher Engine: Cascade Architecture Training Pipeline
## Claude Code Instructional Guide — Descartes Focus

This guide replaces and extends the original Training Pipeline with a cascade architecture optimized for Descartes' philosophy. Instead of training a 70B-685B model on a corpus too small to shift its weights, we fully saturate a small model (8B) with Cartesian expertise and route to an untrained large model at inference time for broad knowledge.

**Reconciliation**: Phases 1-4 adapt for Descartes-focused corpus. Phase 5 changes to 8B CPT. Phases 6-7 add cascade-specific SFT. New Phases 9-12 build the oracle routing and inference orchestrator.

---

## Pipeline Overview

```
Phase 1:  Corpus Assembly (Descartes)  → Raw texts + context       [ADAPTED]
Phase 2:  Text Extraction              → Clean plaintext            [UNCHANGED]
Phase 3:  Cleaning & Filtering         → Normalized corpus          [UNCHANGED]
Phase 4:  CPT Data Formatting          → Tokenized JSONL            [UNCHANGED]
Phase 5:  CPT Training (8B)            → Domain-adapted small model [CHANGED]
Phase 6:  SFT Data Generation          → Reasoning + cascade examples[ADAPTED]
Phase 7:  Two-Stage SFT                → Instruction-tuned + routing [ADAPTED]
Phase 8:  Base Model Evaluation        → Benchmark small model alone [ADAPTED]
Phase 9:  Confidence Head Training     → Uncertainty-aware routing   [NEW]
Phase 10: Oracle Integration           → Large model API bridge      [NEW]
Phase 11: Cascade Inference Engine     → Full orchestrated system    [NEW]
Phase 12: End-to-End Evaluation        → Cascade benchmarks          [NEW]
```

**Total estimated cost**: $200-$800 (down from $8K-$28K)
**Total estimated time**: 2-3 weeks part-time
**Required infrastructure**: 1x A40 48GB on Vast.ai (~$0.40-$0.60/hr)

---

## Architecture Diagram

```
User Query
    │
    ▼
┌──────────────────────────────┐
│  SMALL MODEL (Qwen3-8B)     │
│  Trained on Descartes corpus │
│                              │
│  Strengths:                  │
│  ├─ Z3 formalization         │
│  ├─ Cartesian argument       │
│  │  decomposition            │
│  ├─ Substance dualism logic  │
│  ├─ Cogito verification      │
│  ├─ Mind-body interaction    │
│  │  formalization            │
│  └─ ASPIC+ scheme mapping    │
└─────────┬────────────────────┘
          │
    Confidence check
    [CONFIDENCE: 0.X]
          │
   ┌──────┴──────┐
   │ ≥ 0.7       │ < 0.7
   ▼              ▼
 Answer     ┌──────────────────────────┐
 directly   │  ORACLE (DeepSeek/Claude) │
            │  Untrained, via API       │
            │                           │
            │  Strengths:               │
            │  ├─ Broad Descartes       │
            │  │  scholarship           │
            │  ├─ Historical context    │
            │  ├─ Cross-philosopher     │
            │  │  comparison            │
            │  ├─ Contemporary debates  │
            │  └─ Empirical neuroscience│
            └──────────┬───────────────┘
                       │
                       ▼
              Small model integrates
              oracle's knowledge with
              its formal expertise
                       │
                       ▼
                 Final response
                 [CONFIDENCE: updated]
```

---

## Phase 1 (ADAPTED): Corpus Assembly — Descartes Focus

### Session Goal
Assemble a Descartes-centered philosophical corpus. Target: 50M-200M tokens with Descartes as concentrated core embedded in broader rationalist and philosophy-of-mind context.

### Why Descartes Is Good for This

Descartes is an ideal first target because his arguments have unusually clean logical structure. The Cogito is a strict inference. The Real Distinction argument is modal. The Trademark argument is deductive. The Wax argument is abductive. His Meditations move step-by-step with explicit premises and conclusions — exactly what ASPIC+ decomposition and Z3 verification are designed for. Many other philosophers require extensive interpretive work before formalization. Descartes gives you arguments that are already halfway to formal.

### Claude Code Instructions

```
You are assembling a Descartes-focused philosophical corpus for 
continued pre-training. The target is 50M-200M tokens.

MIXING RATIO:
- 25% Descartes primary texts + scholarship (concentrated core)
- 20% Rationalist tradition (Spinoza, Leibniz, Malebranche)
- 20% Philosophy of mind (mind-body problem, substance dualism responses)
- 15% History of philosophy / Early Modern period
- 10% Formal logic + argumentation theory
- 10% Contemporary responses to Cartesian arguments

CRITICAL: Only original published texts. Never LLM-generated content.
Descartes' works are public domain — full texts available freely.

Create directory structure at ~/corpus/ and track all sources.
```

### Source Inventory

```yaml
# corpus_config_descartes.yaml

corpus_target_tokens: 100_000_000  # 100M tokens target

sources:

  # ============================================================
  # TIER 1: DESCARTES PRIMARY TEXTS (Public Domain)
  # ============================================================
  
  descartes_primary:
    type: "gutenberg_and_archive"
    estimated_tokens: 5_000_000
    category: "descartes_primary"
    priority: 1
    texts:
      # Major works — all public domain
      - title: "Meditations on First Philosophy"
        notes: >
          Core text. Six meditations + objections and replies.
          The Objections/Replies are critical — they contain 
          Descartes' most rigorous argument formulations in 
          response to Arnauld, Gassendi, Hobbes, etc.
          Get BOTH Latin original and English translations.
        sources:
          - "Project Gutenberg"
          - "Early Modern Texts (earlymoderntexts.com) — Jonathan Bennett's edition"
          - "Internet Archive — Cottingham/Stoothoff/Murdoch translation"
        
      - title: "Discourse on the Method"
        notes: "Parts 1-6. Part 4 contains the Cogito reasoning."
        
      - title: "Principles of Philosophy"
        notes: >
          Parts 1-2 especially. Contains Descartes' most systematic
          metaphysics — substance/attribute/mode ontology maps 
          directly to OWL 2 class hierarchies.
          
      - title: "The Passions of the Soul"
        notes: >
          Descartes' theory of mind-body interaction via pineal gland.
          His most detailed account of how mental causation works.
          
      - title: "Rules for the Direction of the Mind"
        notes: "Early methodological work. Shows Descartes' formalist tendencies."
        
      - title: "Correspondence (selected)"
        notes: >
          Letters to Princess Elisabeth are essential — she presses 
          him on how mind-body interaction is possible, and his 
          responses reveal the deep difficulties in his position.
          Also letters to Mersenne, Arnauld, More, Regius.
        sources:
          - "Cottingham et al., Philosophical Writings of Descartes Vol. 3"
          - "earlymoderntexts.com selections"

  # ============================================================
  # TIER 1: OBJECTIONS AND REPLIES
  # ============================================================
  
  objections_and_replies:
    type: "gutenberg_and_archive"
    estimated_tokens: 3_000_000
    category: "descartes_primary"
    priority: 1
    notes: >
      The seven sets of Objections and Replies are arguably MORE 
      important than the Meditations themselves for formal analysis. 
      They contain:
      - Arnauld's Cartesian Circle objection (logical structure)
      - Gassendi's materialist counterarguments
      - Hobbes' nominalist/physicalist objections
      - Caterus' ontological argument analysis
      - Bourdin's methodological critiques
      Each objection-reply pair is a self-contained argument 
      with explicit premises, attacks, and defenses — perfect 
      ASPIC+ training data.

  # ============================================================
  # TIER 2: RATIONALIST TRADITION (Context)
  # ============================================================
  
  rationalist_tradition:
    type: "gutenberg_and_archive"
    estimated_tokens: 15_000_000
    category: "rationalist_tradition"
    priority: 2
    texts:
      - "Spinoza — Ethics (geometrical method = formal structure)"
      - "Leibniz — Monadology, Discourse on Metaphysics"
      - "Malebranche — The Search After Truth (occasionalism)"
      - "Arnauld — On True and False Ideas (anti-Malebranche)"
      - "Princess Elisabeth — Correspondence with Descartes"
      - "Regius — Notes on the Program"
    notes: >
      Spinoza's Ethics is especially valuable because it's written 
      in axiomatic-deductive form (definitions, axioms, propositions, 
      demonstrations) — essentially pre-formalized philosophy.

  # ============================================================
  # TIER 2: DESCARTES SCHOLARSHIP
  # ============================================================
  
  descartes_scholarship:
    type: "pdf_download"
    estimated_tokens: 20_000_000
    category: "descartes_primary"
    priority: 1
    notes: >
      Open-access papers on Descartes from PhilPapers, JSTOR, 
      academia.edu. Focus on:
      - Cartesian Circle literature (Arnauld, Doney, Frankfurt, etc.)
      - Real Distinction argument analysis
      - Mind-body problem in Descartes
      - Cogito interpretations (inference vs. performance vs. intuition)
      - Cartesian epistemology (clear and distinct perception)
      - Descartes and mechanism
      SEP entries: "Descartes' Epistemology", "Descartes' Ethics",
      "Descartes' Life and Works", "Descartes' Modal Metaphysics",
      "Descartes' Ontological Argument", "Descartes and the Pineal Gland"

  # ============================================================
  # TIER 2: PHILOSOPHY OF MIND
  # ============================================================
  
  philosophy_of_mind:
    type: "mixed"
    estimated_tokens: 20_000_000
    category: "philosophy_of_mind"
    priority: 2
    notes: >
      Focus on the mind-body problem lineage FROM Descartes:
      - Ryle — The Concept of Mind (direct response to Descartes)
      - Churchland — Matter and Consciousness
      - Kim — Mind in a Physical World (interaction problem)
      - Chalmers — The Conscious Mind (neo-property dualism)
      - Searle — Minds, Brains and Science
      - Nagel — "What Is It Like to Be a Bat?"
      - Block, Fodor, Putnam on functionalism
      - Jackson — "What Mary Didn't Know"
      
      SEP entries on: mind-body problem, mental causation, 
      substance dualism, property dualism, interaction problem,
      overdetermination, epiphenomenalism, occasionalism

  # ============================================================
  # TIER 3: FORMAL LOGIC AND ARGUMENTATION
  # ============================================================

  formal_logic:
    type: "mixed"
    estimated_tokens: 10_000_000
    category: "formal_logic"
    priority: 2
    notes: >
      Texts that bridge formal logic and philosophical argument:
      - Walton — Argumentation Schemes
      - Bencivenga — "Descartes' Cogito in a Formal Reconstruction"
      - Nolan — formal reconstructions of ontological argument
      - Beyssade — formal structure of Meditations
      - Formal epistemology papers on foundationalism
      - Modal logic textbooks (accessible chapters)
      - Z3/SMT tutorial papers and documentation

  # ============================================================  
  # TIER 3: CONTEMPORARY RESPONSES
  # ============================================================

  contemporary_responses:
    type: "mixed"
    estimated_tokens: 10_000_000
    category: "contemporary_responses"
    priority: 3
    notes: >
      Modern engagement with Cartesian themes:
      - Cottingham — Cartesian Reflections, The Rational Emotive
      - Williams — Descartes: The Project of Pure Enquiry
      - Wilson — Descartes (Routledge Arguments of the Philosophers)
      - Hatfield — Routledge Guidebook to Descartes' Meditations
      - Broughton — Descartes' Method of Doubt
      - IIT and GWT connections to Cartesian consciousness
      - Predictive processing and the Cartesian brain
```

### Directory Structure

```bash
# Execute in Claude Code:

mkdir -p ~/corpus/{raw,extracted,cleaned,formatted}
mkdir -p ~/corpus/raw/{descartes_primary,rationalist_tradition,philosophy_of_mind,formal_logic,contemporary_responses,descartes_scholarship}
mkdir -p ~/corpus/metadata

cat > ~/corpus/manifest.json << 'EOF'
{
  "created": "2026-02-16",
  "project": "Philosopher Engine — Descartes Focus",
  "target_tokens": 100000000,
  "mixing_ratio": {
    "descartes_primary": 0.25,
    "rationalist_tradition": 0.20,
    "philosophy_of_mind": 0.20,
    "descartes_scholarship": 0.15,
    "formal_logic": 0.10,
    "contemporary_responses": 0.10
  },
  "sources": [],
  "total_documents": 0,
  "total_tokens_estimated": 0,
  "status": "assembling"
}
EOF
```

### Download Scripts

```python
# ~/corpus/scripts/download_descartes.py
"""
Download Descartes' primary texts from public domain sources.
All works are 350+ years old — fully in public domain worldwide.
"""

import requests
import time
import os
from pathlib import Path

OUTPUT_DIR = Path(os.path.expanduser(
    "~/corpus/raw/descartes_primary"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Early Modern Texts — Jonathan Bennett's cleaned editions
# These are the best freely available philosophical editions:
# modernized English, philosophical precision, clear formatting
EMT_BASE = "https://www.earlymoderntexts.com/assets/pdfs"

DESCARTES_TEXTS = {
    "meditations": f"{EMT_BASE}/descartes1641.pdf",
    "objections_replies_1-2": f"{EMT_BASE}/descartes1641o1.pdf",
    "objections_replies_3": f"{EMT_BASE}/descartes1641o3.pdf",
    "objections_replies_4": f"{EMT_BASE}/descartes1641o4.pdf",
    "objections_replies_5": f"{EMT_BASE}/descartes1641o5.pdf",
    "objections_replies_6": f"{EMT_BASE}/descartes1641o6.pdf",
    "objections_replies_7": f"{EMT_BASE}/descartes1641o7.pdf",
    "discourse_on_method": f"{EMT_BASE}/descartes1637.pdf",
    "principles_of_philosophy": f"{EMT_BASE}/descartes1644.pdf",
    "passions_of_the_soul": f"{EMT_BASE}/descartes1649.pdf",
    "rules_for_direction": f"{EMT_BASE}/descartes1628.pdf",
    "correspondence_elisabeth": f"{EMT_BASE}/descartes1643e.pdf",
    "comments_on_a_broadsheet": f"{EMT_BASE}/descartes1648.pdf",
    "search_for_truth": f"{EMT_BASE}/descartes1701.pdf",
}

# Rationalist context — also from EMT
RATIONALIST_TEXTS = {
    "spinoza_ethics": f"{EMT_BASE}/spinoza1665.pdf",
    "leibniz_monadology": f"{EMT_BASE}/leibniz1714b.pdf",
    "leibniz_discourse_metaphysics": f"{EMT_BASE}/leibniz1686a.pdf",
    "malebranche_search_truth": f"{EMT_BASE}/malebranche1674.pdf",
    "arnauld_true_false_ideas": f"{EMT_BASE}/arnauld1683.pdf",
}

# SEP entries
SEP_ENTRIES = [
    "descartes", "descartes-epistemology", "descartes-ethics",
    "descartes-modal", "descartes-ontological", "descartes-pineal",
    "descartes-works", "dualism", "substance-dualism",
    "mind-body", "mental-causation", "other-minds",
    "consciousness", "qualia", "physicalism", "functionalism",
    "cognitive-science", "epistemology-foundational",
    "rationalism-empiricism", "spinoza-modal", "leibniz-mind",
    "occasionalism", "epiphenomenalism", "interaction-problem",
    "personal-identity", "self-knowledge",
]


def download_file(url: str, output_path: Path) -> bool:
    """Download a file with retry logic."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                output_path.write_bytes(resp.content)
                print(f"  OK: {output_path.name}")
                return True
            else:
                print(f"  HTTP {resp.status_code}: {url}")
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return False


def download_sep_entry(entry: str, output_dir: Path) -> bool:
    """Download Stanford Encyclopedia entry as HTML."""
    url = f"https://plato.stanford.edu/entries/{entry}/"
    output_path = output_dir / f"sep_{entry}.html"
    return download_file(url, output_path)


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING DESCARTES CORPUS")
    print("=" * 60)
    
    # Descartes primary texts
    print("\n[1/3] Descartes primary texts (Early Modern Texts PDFs)...")
    success = 0
    for name, url in DESCARTES_TEXTS.items():
        path = OUTPUT_DIR / f"{name}.pdf"
        if download_file(url, path):
            success += 1
        time.sleep(1)
    print(f"  {success}/{len(DESCARTES_TEXTS)} Descartes texts downloaded")
    
    # Rationalist tradition
    rat_dir = Path(os.path.expanduser(
        "~/corpus/raw/rationalist_tradition"))
    rat_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[2/3] Rationalist tradition texts...")
    success = 0
    for name, url in RATIONALIST_TEXTS.items():
        path = rat_dir / f"{name}.pdf"
        if download_file(url, path):
            success += 1
        time.sleep(1)
    print(f"  {success}/{len(RATIONALIST_TEXTS)} rationalist texts downloaded")
    
    # SEP entries
    sep_dir = Path(os.path.expanduser(
        "~/corpus/raw/descartes_scholarship/sep"))
    sep_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[3/3] Stanford Encyclopedia entries...")
    success = 0
    for entry in SEP_ENTRIES:
        if download_sep_entry(entry, sep_dir):
            success += 1
        time.sleep(2)  # Respect rate limits
    print(f"  {success}/{len(SEP_ENTRIES)} SEP entries downloaded")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    total = len(DESCARTES_TEXTS) + len(RATIONALIST_TEXTS) + len(SEP_ENTRIES)
    print(f"Total files attempted: {total}")
    print(f"\nNext: Run Phase 2 (text extraction)")
```

### Validation Checkpoint

```bash
echo "=== Descartes Corpus Assembly Status ==="
echo "Primary texts:"
find ~/corpus/raw/descartes_primary -type f | wc -l
echo "Rationalist tradition:"
find ~/corpus/raw/rationalist_tradition -type f | wc -l
echo "Philosophy of mind:"
find ~/corpus/raw/philosophy_of_mind -type f | wc -l
echo "Scholarship:"
find ~/corpus/raw/descartes_scholarship -type f | wc -l
echo "Formal logic:"
find ~/corpus/raw/formal_logic -type f | wc -l
echo "Contemporary:"
find ~/corpus/raw/contemporary_responses -type f | wc -l
echo ""
echo "Total files:"
find ~/corpus/raw -type f | wc -l
echo "Total size:"
du -sh ~/corpus/raw
```

**Minimum to proceed**: Descartes primary texts complete (14 PDFs), 20+ SEP entries, 2GB+ total.

---

## Phases 2-4: UNCHANGED

Run exactly as specified in the original Training Pipeline document. The only difference is the input directory structure (Descartes categories instead of generic philosophy-of-mind categories). The cleaning pipeline, deduplication, quality filtering, and formatting scripts work identically.

Refer to original `PHILOSOPHER_ENGINE_TRAINING_PIPELINE.md` Phases 2-4.

---

## Phase 5 (CHANGED): CPT Training on 8B Model

### Session Goal
Run continued pre-training on Qwen3-8B using the Descartes corpus. Full fine-tuning on a single A40 48GB.

### Why 8B Instead of 70B

Your Descartes corpus is approximately 100M tokens. The ratio of tokens-to-parameters determines how deeply the model absorbs domain knowledge:

```
Llama 70B  + 100M tokens = 1.4 tokens/parameter   → barely nudges weights
Qwen3-8B   + 100M tokens = 12.5 tokens/parameter  → genuine domain shift
```

The 8B model will develop real Cartesian fluency. The 70B model would be a waste of money on this corpus size.

### Vast.ai Setup

```bash
# Search for: 1x A40 48GB, Belgium or Australia (cheapest)
# Target: $0.40-$0.60/hr
# Look for listings like:
#   Belgium 2x A40 45GB @ $0.577/hr (use 1 GPU)
#   Australia 1x A40 45GB @ $0.605/hr

# SSH into Vast.ai instance, then:

# Install dependencies
pip install torch transformers accelerate datasets wandb \
    flash-attn bitsandbytes sentencepiece --break-system-packages

# Clone your corpus (upload via SCP or download from cloud storage)
# scp -r ~/corpus/formatted/ vastai:~/corpus/formatted/

huggingface-cli login  # For Qwen3-8B access
wandb login            # For training monitoring
```

### Training Script

```python
# ~/training/run_cpt_descartes.py
"""
Full CPT on Qwen3-8B with Descartes corpus.
Single A40 48GB. No multi-GPU, no QLoRA needed.

Expected: 8-15 hours training, $5-$10 GPU cost.
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

MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = os.path.expanduser("~/models/descartes-8b-cpt")
DATA_DIR = os.path.expanduser("~/corpus/formatted")


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    # ~16 GB model + ~32 GB optimizer = ~48 GB → fits A40 48GB
    
    dataset = load_dataset("json", data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl"
    })
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=8192,
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
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # 4 epochs — more than 70B because 8B absorbs more per epoch
        num_train_epochs=4,
        
        # Larger batches fit because model is small
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        # Effective batch: 8 × 4 = 32 sequences
        
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
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )
    
    # Estimate time and cost
    n_samples = len(tokenized["train"])
    total_steps = (n_samples * 4) // 32  # samples × epochs / effective_batch
    hours_est = total_steps / 2 / 3600   # ~2 steps/sec on A40
    cost_est = hours_est * 0.55
    print(f"\nTraining plan:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Steps: {total_steps:,}")
    print(f"  Est. time: {hours_est:.1f} hours")
    print(f"  Est. cost: ${cost_est:.2f} (at $0.55/hr)")
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nCPT complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

### Validation

```python
# ~/training/eval_cpt_descartes.py
"""
Validate CPT model on Descartes-specific and general text.
"""

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model, tokenizer, texts, max_length=2048):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    return math.exp(total_loss / total_tokens)


DESCARTES_TEXTS = [
    "The Cogito — I think, therefore I am — is not a syllogism with "
    "a suppressed major premise, but rather an immediate intuition. "
    "Descartes clarifies in the Second Replies that the certainty of "
    "the Cogito does not depend on the prior knowledge of the major "
    "premise 'whatever thinks exists,' but is grasped by a simple "
    "act of mental intuition.",

    "The Real Distinction argument in the Sixth Meditation proceeds "
    "from clear and distinct perception to metaphysical possibility "
    "to actual distinctness. Because I can clearly and distinctly "
    "conceive of mind apart from body and body apart from mind, God "
    "could create them separately, therefore they are really distinct "
    "substances. This argument has a modal structure: conceivability "
    "entails possibility entails actual distinctness.",

    "Elisabeth's objection to Descartes concerns the interaction "
    "problem: if mind is unextended thinking substance and body is "
    "extended non-thinking substance, how can they causally interact? "
    "Extension seems required for contact, and contact seems required "
    "for causation. Descartes' responses in the correspondence invoke "
    "a primitive notion of mind-body union that does not reduce to "
    "either thought or extension alone.",

    "The Cartesian Circle objection, raised by Arnauld in the Fourth "
    "Objections, claims that Descartes' argument is circular: he uses "
    "clear and distinct perception to prove God's existence, then "
    "uses God's existence to validate clear and distinct perception. "
    "Descartes responds that the Cogito and the divine guarantee "
    "operate at different levels — present certainty versus memory "
    "of past demonstrations.",
]

GENERAL_TEXTS = [
    "Photosynthesis converts carbon dioxide and water into glucose "
    "using light energy captured by chlorophyll in chloroplasts.",

    "Binary search operates on sorted arrays by repeatedly halving "
    "the search interval until the target is found or the interval "
    "is empty, achieving O(log n) time complexity.",
]


def evaluate(base_model_name, cpt_model_path):
    print("Loading base model...")
    base_tok = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    
    print("Loading CPT model...")
    cpt_tok = AutoTokenizer.from_pretrained(
        cpt_model_path, trust_remote_code=True)
    cpt_model = AutoModelForCausalLM.from_pretrained(
        cpt_model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    
    base_des = compute_perplexity(base_model, base_tok, DESCARTES_TEXTS)
    cpt_des = compute_perplexity(cpt_model, cpt_tok, DESCARTES_TEXTS)
    base_gen = compute_perplexity(base_model, base_tok, GENERAL_TEXTS)
    cpt_gen = compute_perplexity(cpt_model, cpt_tok, GENERAL_TEXTS)
    
    print(f"\n{'='*50}")
    print(f"{'Metric':<30} {'Base':>8} {'CPT':>8} {'Delta':>8}")
    print(f"{'='*50}")
    print(f"{'Descartes perplexity':<30} {base_des:>8.1f} {cpt_des:>8.1f} "
          f"{(cpt_des-base_des)/base_des*100:>+7.1f}%")
    print(f"{'General perplexity':<30} {base_gen:>8.1f} {cpt_gen:>8.1f} "
          f"{(cpt_gen-base_gen)/base_gen*100:>+7.1f}%")
    print(f"{'='*50}")
    
    # Pass criteria
    des_improved = cpt_des < base_des
    gen_retained = (cpt_gen - base_gen) / base_gen < 0.15
    
    print(f"\nDescartes improved: {'PASS' if des_improved else 'FAIL'}")
    print(f"General retained:   {'PASS' if gen_retained else 'FAIL'}")
    
    return des_improved and gen_retained


# Usage:
# evaluate("Qwen/Qwen3-8B", "~/models/descartes-8b-cpt")
```

### Phase 5 Cost

```
HARDWARE: 1x A40 48GB @ $0.55/hr
CORPUS:   100M tokens × 4 epochs = 400M token-passes
SPEED:    ~15K tokens/sec on A40 for 8B
TIME:     400M / 15K / 3600 ≈ 7.4 hours
COST:     7.4 × $0.55 = $4.07

With overhead (3x buffer): ~$12-$15
```

---

## Phase 6 (ADAPTED): SFT Data Generation — Descartes

### Session Goal
Generate 5K-10K SFT examples covering both standard philosophical reasoning (Types A-D) and cascade-specific behaviors (Types E-G).

### Descartes-Specific Example Templates

```python
# ~/training/sft/descartes_templates.py
"""
SFT templates specialized for Descartes' arguments.
"""

SYSTEM_PROMPT = """You are a philosophical reasoning assistant \
specializing in Cartesian philosophy, early modern rationalism, \
and the mind-body problem. You analyze arguments with formal rigor \
using ASPIC+ argumentation schemes and Z3 verification. You have \
deep expertise in Descartes' Meditations, the Objections and Replies, \
the Correspondence with Elisabeth, and the Principles of Philosophy.

You have access to an oracle for broad philosophical knowledge beyond \
your Cartesian specialization. Request oracle consultation when needed.

Express confidence as [CONFIDENCE: 0.X] at the end of each response. \
When requesting oracle help, output [ORACLE_REQUEST: <query>]."""


# ============================================================
# TYPE A: ARGUMENT RECONSTRUCTION (Descartes-specific)
# ============================================================

TYPE_A_DESCARTES = [
    {
        "user": "Reconstruct the logical structure of the Cogito as "
                "presented in the Second Meditation.",
        "key_elements": [
            "Whether it's inference or intuition",
            "Role of the Evil Genius hypothesis",
            "Scope: what exactly is established (existence, not nature)",
            "Distinction from syllogistic reading",
        ]
    },
    {
        "user": "Reconstruct the Real Distinction argument from "
                "Meditation VI. Identify each premise, the modal "
                "inference, and the theological guarantee that "
                "makes it work.",
        "key_elements": [
            "Clear and distinct conceivability premise",
            "Divine guarantee (God could create them apart)",
            "Modal step: conceivability → possibility → actuality",
            "Role of the earlier proof of God's existence",
            "Difference from the argument in Meditation II",
        ]
    },
    {
        "user": "Formalize the Trademark Argument (Meditation III) "
                "for God's existence in ASPIC+ structure.",
        "key_elements": [
            "Causal adequacy principle",
            "Formal vs objective reality distinction",
            "Idea of infinite substance",
            "Only infinite substance can cause idea of infinity",
            "Strict rule: cause must have at least as much reality",
        ]
    },
    {
        "user": "Reconstruct the Cartesian Circle as identified by "
                "Arnauld. Show both the circular structure and "
                "Descartes' proposed escape.",
        "key_elements": [
            "C&D perception → God exists → C&D perception is reliable",
            "Arnauld's charge in Fourth Objections",
            "Memory vs. present perception distinction",
            "Whether the escape actually works",
        ]
    },
    {
        "user": "Formalize the Wax Argument from Meditation II. "
                "What is the conclusion and what argumentation "
                "scheme does it use?",
        "key_elements": [
            "Argument from elimination (not senses, not imagination)",
            "Conclusion: bodies known through intellect alone",
            "Implicit inference to best explanation",
            "Role in broader project of establishing mind's primacy",
        ]
    },
]


# ============================================================
# TYPE B: CRITICAL ENGAGEMENT (Descartes-specific)
# ============================================================

TYPE_B_DESCARTES = [
    {
        "user": "Present Elisabeth's interaction problem objection "
                "to Descartes, then give Descartes' best defense.",
        "attack_type": "undermine",
        "target": "mind-body causal interaction premise",
    },
    {
        "user": "Present the strongest materialist objection to the "
                "Real Distinction argument. Can a physicalist accept "
                "the conceivability premise but deny the conclusion?",
        "attack_type": "undercut",
        "target": "conceivability-to-separability bridge",
    },
    {
        "user": "Gassendi objects that the Cogito only proves that "
                "thinking occurs, not that a thinking SUBSTANCE exists. "
                "Evaluate this objection.",
        "attack_type": "undermine",
        "target": "substance inference from Cogito",
    },
    {
        "user": "Hobbes argues in the Third Objections that 'I think' "
                "does not entail 'I am a thinking thing' — thinking "
                "might be a property of a material body. How would "
                "Descartes respond?",
        "attack_type": "rebut",
        "target": "immateriality conclusion",
    },
    {
        "user": "The Cartesian Circle: does Descartes' memory "
                "distinction actually solve Arnauld's objection? "
                "Present the strongest case for and against.",
        "attack_type": "undercut",
        "target": "epistemic bootstrapping",
    },
]


# ============================================================
# TYPE C: CROSS-DISCIPLINARY (Descartes + Neuroscience)
# ============================================================

TYPE_C_DESCARTES = [
    {
        "user": "Connect Descartes' pineal gland hypothesis to modern "
                "neuroscience. Was he entirely wrong, or did he "
                "identify a real problem that neuroscience still "
                "hasn't solved?",
    },
    {
        "user": "Descartes argued that animals are automata without "
                "consciousness. How does this compare to contemporary "
                "evidence on animal consciousness from neuroscience?",
    },
    {
        "user": "The Global Workspace Theory says consciousness arises "
                "from information broadcast across brain areas. Could "
                "Descartes accept GWT, or does it conflict with his "
                "substance dualism? Formalize the compatibility check.",
    },
    {
        "user": "Descartes' 'clear and distinct perception' criterion "
                "for truth — does it have an analog in predictive "
                "processing / Bayesian brain theories? Connect "
                "Cartesian certainty to predictive confidence.",
    },
]


# ============================================================
# TYPE D: COMPREHENSION (Passage-based)
# ============================================================

TYPE_D_DESCARTES = [
    {
        "passage": "But what then am I? A thing that thinks. What is "
                   "that? A thing that doubts, understands, affirms, "
                   "denies, is willing, is unwilling, and also imagines "
                   "and has sensory perceptions.",
        "source": "Meditation II",
        "questions": [
            "What is the logical status of this enumeration?",
            "Is Descartes defining thinking or giving examples?",
            "Does 'sensory perception' belong on this list if senses "
            "have been doubted?",
            "How would Ryle critique this passage?",
        ]
    },
    {
        "passage": "I knew that I was a substance whose whole essence "
                   "or nature is simply to think, and which does not "
                   "require any place, or depend on any material thing, "
                   "in order to exist.",
        "source": "Discourse on the Method, Part IV",
        "questions": [
            "What argument licenses 'whose WHOLE essence is to think'?",
            "Is this stronger than what the Cogito actually establishes?",
            "How does this relate to the Real Distinction argument?",
            "How would a property dualist reinterpret this passage?",
        ]
    },
]


# ============================================================
# TYPE E: CONFIDENCE ESTIMATION (Cascade-specific)
# ============================================================

TYPE_E_DESCARTES = [
    {
        "user": "Formalize the Cogito in Z3 as a strict inference.",
        "expected_confidence": 0.95,
        "routing": "SELF",
        "reason": "Core formalization task, directly trained on this.",
        "response_sketch": """The Cogito can be formalized as:

```python
from z3 import *

# Sorts
Agent = DeclareSort('Agent')

# Predicates
Thinks = Function('Thinks', Agent, BoolSort())
Exists = Function('Exists', Agent, BoolSort())
Doubts = Function('Doubts', Agent, BoolSort())

# The thinker
I = Const('I', Agent)
s = Solver()

# Premise: I am doubting (a form of thinking)
s.add(Doubts(I))

# Strict rule: doubting entails thinking
a = Const('a', Agent)
s.add(ForAll([a], Implies(Doubts(a), Thinks(a))))

# Strict rule: thinking entails existing (the Cogito)
s.add(ForAll([a], Implies(Thinks(a), Exists(a))))

# Verify: does I exist follow?
s.push()
s.add(Not(Exists(I)))
result = s.check()  # Expected: UNSAT
# UNSAT confirms: Exists(I) is entailed
```

The Cogito is a strict (deductive) inference, not defeasible. The 
argumentation scheme is modus ponens applied twice: Doubts(I) → 
Thinks(I) → Exists(I).

Note: this formalization captures the SYLLOGISTIC reading. Descartes 
himself preferred the INTUITION reading (the Cogito is not an 
inference from a general premise but an immediate grasp). The 
intuition reading is harder to formalize — it would require 
epistemic logic operators for immediate certainty.

[CONFIDENCE: 0.95]"""
    },
    {
        "user": "How does Descartes' substance dualism compare to "
                "Strawson's person theory in 'Individuals'?",
        "expected_confidence": 0.45,
        "routing": "ORACLE",
        "reason": "Strawson comparison is outside Cartesian core expertise.",
        "response_sketch": """Descartes' substance dualism holds that mind 
and body are distinct substances — mind is unextended thinking substance, 
body is extended non-thinking substance. I can formalize this:

Mind ≠ Body (distinct substances)
Essential property of Mind = Thought
Essential property of Body = Extension
Mind can exist without Body (Real Distinction)

However, I am less certain about the details of Strawson's position 
in 'Individuals.' I know Strawson argues that persons are logically 
primitive — not reducible to either minds or bodies — but I'd need 
oracle consultation for the precise comparison and whether Strawson's 
framework constitutes a genuine alternative to Descartes or merely 
redescribes the problem.

[ORACLE_REQUEST: What is Strawson's person theory in 'Individuals' 
and how does it specifically differ from Cartesian dualism? Does 
Strawson address Descartes directly?]

[CONFIDENCE: 0.45]"""
    },
    {
        "user": "Check whether the Trademark Argument, the Ontological "
                "Argument, and the Cogito form a consistent set of "
                "premises when formalized together.",
        "expected_confidence": 0.85,
        "routing": "SELF",
        "reason": "Consistency checking is Z3's core function.",
    },
]


# ============================================================
# TYPE F: ROUTING DECISIONS (Cascade-specific)
# ============================================================

TYPE_F_DESCARTES = [
    {
        "user": "Formalize the Real Distinction argument in Z3 with "
                "S5 modal logic.",
        "routing": "SELF",
        "reason": "Z3 formalization is core small-model competency."
    },
    {
        "user": "What was Merleau-Ponty's critique of Cartesian "
                "dualism in Phenomenology of Perception?",
        "routing": "ORACLE",
        "reason": "Phenomenological tradition outside Cartesian core."
    },
    {
        "user": "Is the Real Distinction argument structurally identical "
                "to Chalmers' zombie argument?",
        "routing": "HYBRID",
        "reason": "Small model formalizes both arguments in Z3. Oracle "
                  "provides Chalmers' own commentary on the connection."
    },
    {
        "user": "Decompose Arnauld's Fourth Objections into ASPIC+ "
                "attack structure.",
        "routing": "SELF",
        "reason": "ASPIC+ decomposition is core training task."
    },
    {
        "user": "How did the reception of Descartes differ between "
                "France and the Netherlands in the 1640s-1650s?",
        "routing": "ORACLE",
        "reason": "Historical context question, not formal analysis."
    },
    {
        "user": "Can Descartes' causal adequacy principle (Trademark "
                "argument) be reconciled with modern physicalism? "
                "Check for joint satisfiability.",
        "routing": "HYBRID",
        "reason": "Small model does Z3 consistency check. Oracle "
                  "provides contemporary physicalist positions on "
                  "causation to formalize."
    },
]


# ============================================================
# TYPE G: ORACLE INTEGRATION (Cascade-specific)
# ============================================================

TYPE_G_DESCARTES = [
    {
        "original_answer": (
            "Descartes' Real Distinction uses modal reasoning: clear "
            "and distinct conceivability → God can separate → actually "
            "distinct. I can formalize this in S5. But I'm uncertain "
            "how closely this maps to Chalmers' zombie argument. "
            "[CONFIDENCE: 0.5]"
        ),
        "oracle_response": (
            "Chalmers himself acknowledges the structural parallel in "
            "The Conscious Mind (1996, p. 146). Both arguments use "
            "conceivability-to-possibility bridges. The key difference: "
            "Descartes requires a divine guarantee (God can create mind "
            "without body), while Chalmers relies on a general modal "
            "principle (ideal conceivability entails metaphysical "
            "possibility). Chalmers' bridge is weaker — it doesn't "
            "need God — but also more vulnerable to Type-B physicalist "
            "objections about a posteriori necessity."
        ),
        "integrated": (
            "The Real Distinction argument and the zombie argument are "
            "structurally parallel but differ in their modal bridge.\n\n"
            "Shared structure:\n"
            "Both: Conceive(Mind without Body/P without C) → "
            "Possible(Mind without Body/P without C) → "
            "Not-Identical(Mind, Body / Phenomenal, Physical)\n\n"
            "In Z3, both formalize as:\n"
            "```python\n"
            "# Shared modal structure\n"
            "w_test = Const('w_test', World)\n"
            "s.add(R(actual, w_test))  # Accessible world\n"
            "s.add(A(w_test))          # Property A obtains\n"
            "s.add(Not(B(w_test)))     # Property B absent\n"
            "# Test: is this consistent with A=B? → UNSAT\n"
            "```\n\n"
            "Key difference in the bridge premise:\n"
            "- Descartes: Conceivability + Divine omnipotence → Possibility\n"
            "  (STRICT rule: God can create anything conceivable)\n"
            "- Chalmers: Ideal conceivability → Metaphysical possibility\n"
            "  (DEFEASIBLE rule: vulnerable to a posteriori necessity)\n\n"
            "This means the Cartesian version is actually STRONGER if "
            "you accept the divine guarantee — it's a strict inference "
            "rather than a defeasible one. The zombie argument trades "
            "theological commitment for broader acceptability, but gains "
            "vulnerability to Type-B objections that don't apply to "
            "Descartes (who pre-empts them with God's veracity).\n\n"
            "[CONFIDENCE: 0.9]"
        ),
    },
]
```

### SFT Data Generation Script

```python
# ~/training/sft/generate_descartes_sft.py
"""
Generate all SFT examples for Descartes cascade model.

Combines:
- Types A-D from LLM council (standard philosophical reasoning)
- Types E-G from self-play (cascade-specific behaviors)

Target: 6,000-10,000 total examples
"""

import json
import os
from pathlib import Path
from typing import List, Dict

# Import templates
from descartes_templates import *

OUTPUT_DIR = Path(os.path.expanduser("~/training/sft/examples"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_from_templates(templates: List[Dict], 
                            example_type: str,
                            llm_client=None) -> List[Dict]:
    """Generate SFT examples from templates using LLM council."""
    
    examples = []
    
    for template in templates:
        # In production: send to Claude + GPT-4 + Gemini
        # Each generates a full response
        # Cross-validate for accuracy
        # Z3 validate any formal claims
        
        user_content = template.get("user", "")
        if not user_content and "passage" in template:
            user_content = (
                f"Read this passage from Descartes ({template['source']}):\n\n"
                f'"{template["passage"]}"\n\n'
                + "\n".join(template["questions"])
            )
        
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ""}  # LLM generates
            ],
            "metadata": {
                "type": example_type,
                "philosopher": "Descartes",
                "z3_validated": False,
                "council_agreement": 0.0,
                "human_reviewed": False,
            }
        }
        examples.append(example)
    
    return examples


def generate_all():
    """Generate complete SFT dataset."""
    
    all_examples = []
    
    # Standard types (A-D)
    print("Generating Type A (reconstruction)...")
    all_examples.extend(
        generate_from_templates(TYPE_A_DESCARTES, "A"))
    
    print("Generating Type B (critical engagement)...")
    all_examples.extend(
        generate_from_templates(TYPE_B_DESCARTES, "B"))
    
    print("Generating Type C (cross-disciplinary)...")
    all_examples.extend(
        generate_from_templates(TYPE_C_DESCARTES, "C"))
    
    print("Generating Type D (comprehension)...")
    all_examples.extend(
        generate_from_templates(TYPE_D_DESCARTES, "D"))
    
    # Cascade types (E-G)
    print("Generating Type E (confidence estimation)...")
    all_examples.extend(
        generate_from_templates(TYPE_E_DESCARTES, "E"))
    
    print("Generating Type F (routing decisions)...")
    all_examples.extend(
        generate_from_templates(TYPE_F_DESCARTES, "F"))
    
    print("Generating Type G (oracle integration)...")
    all_examples.extend(
        generate_from_templates(TYPE_G_DESCARTES, "G"))
    
    # Save
    output_path = OUTPUT_DIR / "descartes_sft_all.jsonl"
    with open(output_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    
    # Stats
    from collections import Counter
    type_counts = Counter(e["metadata"]["type"] for e in all_examples)
    
    print(f"\nTotal examples: {len(all_examples)}")
    for t, c in sorted(type_counts.items()):
        print(f"  Type {t}: {c}")
    print(f"\nSaved to: {output_path}")
    
    print(f"\nNOTE: These are TEMPLATES. Run the LLM council to "
          f"generate full responses, then human-review before training.")


if __name__ == "__main__":
    generate_all()
```

---

## Phase 7 (ADAPTED): Two-Stage SFT

### Session Goal
Fine-tune in two stages: (1) core reasoning, (2) cascade behaviors.

```python
# ~/training/run_sft_descartes.py
"""
Two-stage SFT for Descartes cascade model.

Stage 1: Philosophical reasoning (Types A-D) — 3 epochs
Stage 2: Cascade behaviors (Types E-G) — 2 epochs

Two-stage prevents cascade routing from interfering 
with core reasoning capability.
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import os

CPT_MODEL = os.path.expanduser("~/models/descartes-8b-cpt")
STAGE1_OUT = os.path.expanduser("~/models/descartes-8b-sft-s1")
STAGE2_OUT = os.path.expanduser("~/models/descartes-8b-cascade")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


def run_sft_stage(base_path, data_path, output_dir,
                  run_name, epochs=3):
    """Run one SFT stage with LoRA."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() 
                    if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Format into chat template
    def format_chat(example):
        messages = example.get("messages", [])
        # Filter rejected examples
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
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nStage complete. Merged model saved to {output_dir}")


if __name__ == "__main__":
    SFT_STANDARD = os.path.expanduser(
        "~/training/sft/examples/descartes_sft_types_ABCD.jsonl")
    SFT_CASCADE = os.path.expanduser(
        "~/training/sft/examples/descartes_sft_types_EFG.jsonl")
    
    print("=" * 60)
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
    print("TRAINING COMPLETE")
    print(f"Final model: {STAGE2_OUT}")
    print("=" * 60)
```

---

## Phase 9 (NEW): Confidence Head Training

### Session Goal
The model already outputs confidence scores from Type E SFT training. Phase 9 calibrates these scores against ground truth.

```python
# ~/training/calibrate_confidence.py
"""
Calibrate confidence scores so that when the model says 
[CONFIDENCE: 0.8], it's actually correct 80% of the time.

Method: Generate responses on held-out Descartes questions,
extract confidence scores, compare to human judgments,
train a simple temperature-scaling calibrator.
"""

import json
import numpy as np
from typing import List, Tuple


def extract_confidence(response: str) -> float:
    """Parse [CONFIDENCE: X.X] from model output."""
    import re
    match = re.search(r'\[CONFIDENCE:\s*([\d.]+)\]', response)
    if match:
        return float(match.group(1))
    return 0.5  # Default if not found


def calibrate(predictions: List[Tuple[float, bool]]) -> callable:
    """Platt scaling: fit sigmoid to confidence → accuracy mapping.
    
    Args:
        predictions: list of (confidence_score, was_correct) pairs
    
    Returns:
        Calibration function that maps raw confidence to calibrated
    """
    from scipy.optimize import minimize_scalar
    
    confs = np.array([p[0] for p in predictions])
    correct = np.array([p[1] for p in predictions], dtype=float)
    
    # Temperature scaling: calibrated = sigmoid(conf / T)
    def nll(T):
        scaled = 1 / (1 + np.exp(-(confs - 0.5) / max(T, 0.01)))
        eps = 1e-7
        return -np.mean(
            correct * np.log(scaled + eps) + 
            (1 - correct) * np.log(1 - scaled + eps)
        )
    
    result = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded')
    T_opt = result.x
    
    print(f"Optimal temperature: {T_opt:.3f}")
    
    def calibrated_confidence(raw_conf: float) -> float:
        return 1 / (1 + np.exp(-(raw_conf - 0.5) / T_opt))
    
    return calibrated_confidence


# After calibration, save T_opt and use in the inference engine
# to adjust routing thresholds. If T_opt > 1, the model is 
# overconfident (lower the oracle routing threshold).
# If T_opt < 1, the model is underconfident (raise it).
```

---

## Phase 10 (NEW): Oracle Integration

### Session Goal
Build the bridge between the small model and the large model API.

```python
# ~/inference/oracle.py
"""
Oracle integration — connects the small Descartes model 
to a large model API for broad knowledge retrieval.

Supports: DeepSeek API, Claude API, OpenAI API.
Choose based on cost and quality for your use case.
"""

import os
import json
import re
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class OracleConfig:
    """Configuration for the oracle (large model) backend."""
    provider: str = "deepseek"  # "deepseek", "claude", "openai"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.3
    
    # Cost tracking
    input_cost_per_1k: float = 0.0001   # DeepSeek is very cheap
    output_cost_per_1k: float = 0.0002
    
    # Routing thresholds (calibrated in Phase 9)
    confidence_threshold: float = 0.7  # Below this → oracle
    hybrid_threshold: float = 0.85      # Below this but above 0.7 → hybrid


class OracleClient:
    """Client for querying the large model oracle."""
    
    def __init__(self, config: OracleConfig = None):
        self.config = config or OracleConfig()
        self.total_cost = 0.0
        self.total_calls = 0
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate API client."""
        api_key = os.environ.get(self.config.api_key_env, "")
        
        if self.config.provider == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
        elif self.config.provider == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.config.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
    
    def query(self, oracle_request: str, 
              context: str = "") -> str:
        """Send a query to the oracle and return the response.
        
        Args:
            oracle_request: The specific question for the oracle
            context: The small model's partial answer for context
        """
        system_msg = (
            "You are a philosophical knowledge oracle. A specialist "
            "AI focused on Descartes and formal reasoning is asking "
            "you for information outside its training domain. Provide "
            "accurate, detailed philosophical knowledge. Be specific "
            "about sources and positions. The specialist will integrate "
            "your knowledge with its own formal analysis."
        )
        
        user_msg = oracle_request
        if context:
            user_msg = (
                f"CONTEXT (from specialist model):\n{context}\n\n"
                f"QUESTION:\n{oracle_request}"
            )
        
        self.total_calls += 1
        
        if self.config.provider in ("deepseek", "openai"):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            text = response.choices[0].message.content
            
            # Track cost
            in_tok = response.usage.prompt_tokens
            out_tok = response.usage.completion_tokens
            cost = (in_tok * self.config.input_cost_per_1k / 1000 +
                    out_tok * self.config.output_cost_per_1k / 1000)
            self.total_cost += cost
            
        elif self.config.provider == "claude":
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}]
            )
            text = response.content[0].text
            
            cost = (response.usage.input_tokens * 0.003 / 1000 +
                    response.usage.output_tokens * 0.015 / 1000)
            self.total_cost += cost
        
        return text
    
    def get_stats(self) -> Dict:
        return {
            "total_calls": self.total_calls,
            "total_cost": round(self.total_cost, 4),
            "avg_cost_per_call": round(
                self.total_cost / max(self.total_calls, 1), 6
            )
        }
```

---

## Phase 11 (NEW): Cascade Inference Engine

### Session Goal
Build the complete inference pipeline that orchestrates small model + oracle.

```python
# ~/inference/cascade_engine.py
"""
The complete cascade inference engine for the Descartes 
Philosopher Engine.

This is the production inference system. It:
1. Receives a philosophical query
2. Runs it through the small (trained) model
3. Parses confidence and oracle requests
4. Routes to oracle if needed
5. Runs integration pass through small model
6. Returns final response with metadata
"""

import re
import torch
import json
from typing import Optional, Dict
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from oracle import OracleClient, OracleConfig


@dataclass
class CascadeResult:
    """Complete result from the cascade engine."""
    query: str
    final_response: str
    confidence: float
    routing_decision: str        # "SELF", "ORACLE", "HYBRID"
    oracle_used: bool
    oracle_query: Optional[str] = None
    oracle_response: Optional[str] = None
    small_model_initial: str = ""
    iterations: int = 1
    total_tokens: int = 0
    oracle_cost: float = 0.0


class DescartesEngine:
    """The production Descartes Philosopher Engine."""
    
    def __init__(self, 
                 model_path: str,
                 oracle_config: OracleConfig = None,
                 confidence_threshold: float = 0.7,
                 device: str = "auto"):
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
        self.model.eval()
        
        self.oracle = OracleClient(oracle_config or OracleConfig())
        self.threshold = confidence_threshold
        
        self.system_prompt = (
            "You are a philosophical reasoning assistant specializing "
            "in Cartesian philosophy, early modern rationalism, and "
            "the mind-body problem. You analyze arguments using ASPIC+ "
            "schemes and Z3 verification.\n\n"
            "Express confidence as [CONFIDENCE: 0.X].\n"
            "When needing external knowledge, output "
            "[ORACLE_REQUEST: <query>]."
        )
        
        print("Engine ready.")
    
    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.3) -> str:
        """Generate from the small model."""
        
        full_prompt = (
            f"<|system|>\n{self.system_prompt}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )
        
        inputs = self.tokenizer(
            full_prompt, return_tensors="pt",
            truncation=True, max_length=8192
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def parse_response(self, response: str) -> Dict:
        """Extract confidence and oracle requests from response."""
        
        # Extract confidence
        conf_match = re.search(
            r'\[CONFIDENCE:\s*([\d.]+)\]', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        # Extract oracle request
        oracle_match = re.search(
            r'\[ORACLE_REQUEST:\s*(.+?)\]', response, re.DOTALL)
        oracle_query = oracle_match.group(1).strip() if oracle_match else None
        
        # Clean response (remove tags)
        clean = re.sub(r'\[CONFIDENCE:.*?\]', '', response)
        clean = re.sub(r'\[ORACLE_REQUEST:.*?\]', '', clean, flags=re.DOTALL)
        clean = clean.strip()
        
        return {
            "confidence": confidence,
            "oracle_query": oracle_query,
            "clean_response": clean,
            "needs_oracle": oracle_query is not None or confidence < self.threshold
        }
    
    def run(self, query: str) -> CascadeResult:
        """Execute the full cascade pipeline."""
        
        result = CascadeResult(query=query, final_response="",
                               confidence=0.0, routing_decision="SELF",
                               oracle_used=False)
        
        # Step 1: Initial generation from small model
        initial_response = self.generate(query)
        parsed = self.parse_response(initial_response)
        
        result.small_model_initial = parsed["clean_response"]
        result.confidence = parsed["confidence"]
        
        # Step 2: Routing decision
        if not parsed["needs_oracle"]:
            # High confidence — return directly
            result.routing_decision = "SELF"
            result.final_response = parsed["clean_response"]
            result.confidence = parsed["confidence"]
            return result
        
        # Step 3: Oracle consultation
        result.oracle_used = True
        oracle_query = parsed["oracle_query"] or query
        result.oracle_query = oracle_query
        
        oracle_response = self.oracle.query(
            oracle_query,
            context=parsed["clean_response"]
        )
        result.oracle_response = oracle_response
        result.oracle_cost = self.oracle.total_cost
        
        if parsed["confidence"] < 0.4:
            result.routing_decision = "ORACLE"
        else:
            result.routing_decision = "HYBRID"
        
        # Step 4: Integration pass through small model
        integration_prompt = (
            f"You previously answered a question and requested "
            f"oracle consultation. Integrate the oracle's response "
            f"with your own expertise.\n\n"
            f"ORIGINAL QUESTION: {query}\n\n"
            f"YOUR INITIAL ANSWER:\n{parsed['clean_response']}\n\n"
            f"ORACLE RESPONSE:\n{oracle_response}\n\n"
            f"Produce your final integrated answer. Update your "
            f"confidence score."
        )
        
        integrated = self.generate(integration_prompt)
        integrated_parsed = self.parse_response(integrated)
        
        result.final_response = integrated_parsed["clean_response"]
        result.confidence = integrated_parsed["confidence"]
        result.iterations = 2
        
        return result
    
    def interactive(self):
        """Interactive REPL for testing."""
        print("\n" + "=" * 60)
        print("DESCARTES PHILOSOPHER ENGINE — Interactive Mode")
        print("Type 'quit' to exit, 'stats' for oracle usage stats")
        print("=" * 60)
        
        while True:
            query = input("\n> ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                print(json.dumps(self.oracle.get_stats(), indent=2))
                continue
            elif not query:
                continue
            
            result = self.run(query)
            
            print(f"\n[Routing: {result.routing_decision}]"
                  f"[Confidence: {result.confidence:.2f}]"
                  f"[Oracle: {'Yes' if result.oracle_used else 'No'}]")
            print(f"\n{result.final_response}")
        
        print(f"\nSession stats: {json.dumps(self.oracle.get_stats())}")


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/models/descartes-8b-cascade")
    
    engine = DescartesEngine(model_path)
    engine.interactive()
```

---

## Phase 12 (NEW): Cascade Evaluation

### Descartes-Specific Benchmark

```python
# ~/training/eval/eval_descartes_cascade.py
"""
End-to-end evaluation of the Descartes cascade engine.

Tests:
1. Cartesian argument validity (Z3-verifiable)
2. Objection identification (ASPIC+ attack types)
3. Routing accuracy (does it correctly route to oracle?)
4. Integration quality (does oracle info improve answers?)
5. Descartes-specific knowledge (facts about texts/positions)
"""

VALIDITY_TESTS = [
    {
        "argument": (
            "P1: I think.\n"
            "P2: Whatever thinks, exists.\n"
            "C: I exist."
        ),
        "valid": True,
        "label": "cogito_syllogistic"
    },
    {
        "argument": (
            "P1: I can clearly and distinctly conceive mind without body.\n"
            "P2: Whatever I can C&D conceive, God can create.\n"
            "P3: If God can create A without B, A and B are distinct.\n"
            "C: Mind and body are distinct substances."
        ),
        "valid": True,
        "label": "real_distinction"
    },
    {
        "argument": (
            "P1: I have an idea of a perfect being.\n"
            "P2: I am imperfect.\n"
            "C: A perfect being must exist to cause my idea."
        ),
        "valid": False,  # Missing: causal adequacy principle as explicit premise
        "label": "trademark_incomplete"
    },
    {
        "argument": (
            "P1: The senses sometimes deceive.\n"
            "P2: Whatever sometimes deceives cannot be trusted.\n"
            "C: Nothing known through the senses is certain."
        ),
        "valid": True,  # Valid but unsound (P2 is too strong)
        "label": "dream_argument_valid_unsound"
    },
    {
        "argument": (
            "P1: Mind is unextended.\n"
            "P2: Body is extended.\n"
            "C: Mind cannot causally interact with body."
        ),
        "valid": False,  # Missing: "causal interaction requires extension"
        "label": "interaction_problem_missing_premise"
    },
]

ROUTING_TESTS = [
    {"query": "Formalize the Cogito in Z3.", 
     "expected": "SELF"},
    {"query": "What did Husserl say about the Cartesian Meditations?", 
     "expected": "ORACLE"},
    {"query": "Is the Real Distinction structurally identical to "
              "the zombie argument?",
     "expected": "HYBRID"},
    {"query": "Decompose Arnauld's circularity objection into "
              "ASPIC+ attack structure.",
     "expected": "SELF"},
    {"query": "How was Descartes received by the Jesuits at La Flèche?",
     "expected": "ORACLE"},
]

KNOWLEDGE_TESTS = [
    {"q": "In which Meditation does Descartes present the Wax Argument?",
     "a": "Second Meditation"},
    {"q": "Who raised the Cartesian Circle objection?",
     "a": "Arnauld"},
    {"q": "What is the name of Descartes' correspondent who pressed "
          "the interaction problem?",
     "a": "Princess Elisabeth of Bohemia"},
    {"q": "What gland did Descartes identify as the seat of "
          "mind-body interaction?",
     "a": "Pineal gland"},
    {"q": "Which Objection set is by Hobbes?",
     "a": "Third Objections"},
]

PASS_CRITERIA = """
MINIMUM THRESHOLDS:

Argument Validity:      ≥ 85%
Routing Accuracy:       ≥ 80%
Knowledge (no oracle):  ≥ 70%
Knowledge (with oracle): ≥ 90%
Integration Quality:    Human judges ≥ 4/5

If thresholds missed → iterate:
1. Analyze failure patterns
2. Generate targeted SFT examples
3. Retrain Stage 1 or Stage 2 as appropriate
4. Re-calibrate confidence threshold
"""
```

---

## Quick Reference: Phase Dependencies

```
Phase 1  (Corpus Assembly)    → nothing (start here)
Phase 2  (Text Extraction)    → Phase 1
Phase 3  (Cleaning)           → Phase 2
Phase 4  (Formatting)         → Phase 3
Phase 5  (CPT 8B)             → Phase 4 + Vast.ai A40
Phase 6  (SFT Data Gen)       → Philosopher Engine arch doc
Phase 7  (Two-Stage SFT)      → Phase 5 + Phase 6
Phase 8  (Base Eval)          → Phase 7
Phase 9  (Confidence Calib)   → Phase 8
Phase 10 (Oracle Setup)       → API key for DeepSeek/Claude
Phase 11 (Cascade Engine)     → Phase 9 + Phase 10
Phase 12 (End-to-End Eval)    → Phase 11

Phases 6 + 10 can run IN PARALLEL with Phases 1-5.
Phase 10 requires no training — just API setup.
```

---

## Cost Summary

| Phase | Compute | API Costs | Human Time |
|-------|---------|-----------|------------|
| 1-4 (Data) | Minimal (CPU) | $0 | 20-30 hrs |
| 5 (CPT 8B) | $10-$15 GPU | $0 | 3-5 hrs |
| 6 (SFT Data) | Minimal | $200-$500 LLM API | 40-80 hrs |
| 7 (Two-Stage SFT) | $5-$10 GPU | $0 | 3-5 hrs |
| 8-9 (Eval + Calib) | $5 GPU | $50-$100 API | 10-15 hrs |
| 10-11 (Oracle + Engine) | Minimal | $0 (setup only) | 10-15 hrs |
| 12 (Final Eval) | $5 GPU | $50-$100 API | 10-15 hrs |
| **Total** | **$25-$35** | **$300-$700** | **96-165 hrs** |

**Grand total: $325-$735**

Compare to original pipeline: **$6,700-$25,500**
**Savings: 95-97%**

---

## Appendix: Descartes → Z3 Formalization Templates

These templates are referenced by the small model during inference and used in SFT training data generation.

```python
# ~/inference/templates/descartes_z3.py
"""
Ready-to-use Z3 templates for Descartes' core arguments.
Small model references these during formalization tasks.
"""

from z3 import *


def template_cogito():
    """The Cogito: I think, therefore I am."""
    Agent = DeclareSort('Agent')
    Thinks = Function('Thinks', Agent, BoolSort())
    Exists = Function('Exists', Agent, BoolSort())
    Doubts = Function('Doubts', Agent, BoolSort())
    
    I = Const('I', Agent)
    a = Const('a', Agent)
    s = Solver()
    
    s.add(Doubts(I))  # I am doubting
    s.add(ForAll([a], Implies(Doubts(a), Thinks(a))))   # Doubting → Thinking
    s.add(ForAll([a], Implies(Thinks(a), Exists(a))))   # Thinking → Existing
    
    # Test: is Not(Exists(I)) consistent with above?
    s.push()
    s.add(Not(Exists(I)))
    result = s.check()  # UNSAT → Exists(I) is entailed
    s.pop()
    
    return s, result


def template_real_distinction():
    """The Real Distinction: mind and body are distinct substances.
    Uses S5 modal logic (conceivability → possibility → distinctness).
    """
    World = DeclareSort('World')
    R = Function('R', World, World, BoolSort())
    
    Mind = Function('Mind', World, BoolSort())
    Body = Function('Body', World, BoolSort())
    
    actual = Const('actual', World)
    w_mind_only = Const('w_mind_only', World)
    w_body_only = Const('w_body_only', World)
    
    w, v, u = Consts('w v u', World)
    s = Solver()
    
    # S5 frame
    s.add(ForAll([w], R(w, w)))
    s.add(ForAll([w, v], Implies(R(w, v), R(v, w))))
    s.add(ForAll([w, v, u], Implies(And(R(w, v), R(v, u)), R(w, u))))
    
    # Actual world: both mind and body
    s.add(Mind(actual))
    s.add(Body(actual))
    
    # Conceivability → accessible worlds where they come apart
    s.add(R(actual, w_mind_only))
    s.add(Mind(w_mind_only))
    s.add(Not(Body(w_mind_only)))
    
    s.add(R(actual, w_body_only))
    s.add(Not(Mind(w_body_only)))
    s.add(Body(w_body_only))
    
    # Test: is Mind = Body consistent with separability?
    s.push()
    s.add(ForAll([w], Mind(w) == Body(w)))  # Identity thesis
    result = s.check()  # UNSAT → they are distinct
    s.pop()
    
    return s, result


def template_cartesian_circle():
    """The Cartesian Circle: does Descartes' reasoning contain
    a vicious circularity?
    
    Formalizes: C&D perception → God exists → C&D reliable
    as a potential circular dependency.
    """
    # Propositions
    CDP_Reliable = Bool('CDP_Reliable')   # Clear & distinct perception is reliable
    God_Exists = Bool('God_Exists')       # God exists
    God_Not_Deceiver = Bool('God_Not_Deceiver')
    CDP_Used = Bool('CDP_Used_For_God')   # C&D perception used to prove God
    
    s = Solver()
    
    # Descartes' argument structure:
    # 1. C&D perception used to prove God exists
    s.add(Implies(CDP_Reliable, God_Exists))
    s.add(CDP_Used)
    
    # 2. God's existence validates C&D perception
    s.add(Implies(God_Exists, God_Not_Deceiver))
    s.add(Implies(God_Not_Deceiver, CDP_Reliable))
    
    # This creates: CDP_Reliable → God_Exists → CDP_Reliable
    # Is this actually circular? Check if CDP_Reliable has 
    # independent support or is purely self-referential.
    
    # Arnauld's objection: without independent support, 
    # CDP_Reliable is ungrounded
    s.push()
    s.add(Not(CDP_Reliable))  # Assume C&D is NOT reliable
    result_without = s.check()  # SAT → the circle doesn't force reliability
    s.pop()
    
    # Descartes' defense: present perception is self-guaranteeing
    Present_CDP = Bool('Present_CDP')  # Currently having C&D perception
    s.add(Present_CDP)
    s.add(Implies(Present_CDP, CDP_Reliable))  # Present C&D is certain
    
    s.push()
    s.add(Not(CDP_Reliable))
    result_with_present = s.check()  # UNSAT → with present-perception axiom, reliability holds
    s.pop()
    
    return s, result_without, result_with_present
```
