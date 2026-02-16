"""
Philosopher Engine: Cascade Training Pipeline Orchestrator

Master script to run the full 12-phase cascade training pipeline.
Each phase can be run independently or sequentially.

Usage:
    python run_pipeline.py --phase 1       # Run Phase 1 only
    python run_pipeline.py --phase 1-4     # Run Phases 1 through 4
    python run_pipeline.py --phase 5-12    # Run training + cascade phases
    python run_pipeline.py --phase all     # Run all phases
    python run_pipeline.py --status        # Show pipeline status
    python run_pipeline.py --validate 9    # Run validation for Phase 9

Phase Dependencies (Cascade Architecture):
    Phase 1  (Assembly)       -> nothing (start here)
    Phase 2  (Extraction)     -> Phase 1
    Phase 3  (Cleaning)       -> Phase 2
    Phase 4  (Formatting)     -> Phase 3
    Phase 5  (CPT 8B)         -> Phase 4 + GPU access
    Phase 6  (SFT Data)       -> Phases 1-3 (can run parallel to 4-5)
    Phase 7  (Two-Stage SFT)  -> Phase 5 + Phase 6
    Phase 8  (Base Eval)      -> Phase 7
    Phase 9  (Meta-Learner)   -> Phase 7 + Ollama (local + cloud)
    Phase 10 (Oracle Setup)   -> Ollama Cloud configured
    Phase 11 (Cascade Engine) -> Phase 9 + Phase 10
    Phase 12 (End-to-End)     -> Phase 11 (routing + convergence eval)

Architecture:
    Small Model (Qwen3-8B)  — fully trained on Descartes corpus
    Oracle (DeepSeek/Claude) — untrained, via Ollama Cloud or API
    Meta-Learner (Lite ~50K / Full ~12M) — routes queries + online learning
    Ollama unified API for both local and cloud models (Phases 9-12)
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent

PHASES = {
    1: {
        "name": "Corpus Assembly",
        "scripts": [
            "corpus/scripts/download_sep.py",
            "corpus/scripts/download_arxiv.py",
            "corpus/scripts/download_corpus.py",
        ],
        "description": "Download Descartes + context philosophical texts (SEP, arXiv, Gutenberg, Archive.org)",
        "depends_on": [],
        "validation": "check_phase1",
        "status_tag": "ADAPTED",
    },
    2: {
        "name": "Text Extraction",
        "scripts": ["corpus/scripts/extract_text.py"],
        "description": "Convert PDFs/HTMLs to clean plaintext",
        "depends_on": [1],
        "validation": "check_phase2",
        "status_tag": "UNCHANGED",
    },
    3: {
        "name": "Cleaning & Filtering",
        "scripts": ["corpus/scripts/clean_corpus.py"],
        "description": "Normalize, deduplicate, filter corpus",
        "depends_on": [2],
        "validation": "check_phase3",
        "status_tag": "UNCHANGED",
    },
    4: {
        "name": "CPT Data Formatting",
        "scripts": ["corpus/scripts/format_cpt.py"],
        "description": "Tokenize and format for training",
        "depends_on": [3],
        "validation": "check_phase4",
        "status_tag": "UNCHANGED",
    },
    5: {
        "name": "CPT Training (Qwen3-8B)",
        "scripts": ["training/run_cpt_descartes.py"],
        "description": "Continued pre-training on Qwen3-8B (requires GPU)",
        "depends_on": [4],
        "validation": "check_phase5",
        "status_tag": "CHANGED",
    },
    6: {
        "name": "SFT Data Generation",
        "scripts": ["training/sft/generate_descartes_sft.py"],
        "description": "Generate Types A-G SFT examples via LLM council",
        "depends_on": [],  # Can run parallel to 4-5
        "validation": "check_phase6",
        "status_tag": "ADAPTED",
    },
    7: {
        "name": "Two-Stage SFT",
        "scripts": ["training/run_sft_descartes.py"],
        "description": "Stage 1: reasoning (A-D), Stage 2: cascade (E-G)",
        "depends_on": [5, 6],
        "validation": "check_phase7",
        "status_tag": "ADAPTED",
    },
    8: {
        "name": "Base Model Evaluation",
        "scripts": ["training/eval/eval_cpt_descartes.py"],
        "description": "Benchmark small model alone (perplexity + domain)",
        "depends_on": [7],
        "validation": "check_phase8",
        "status_tag": "ADAPTED",
    },
    9: {
        "name": "Meta-Learner Bootstrap",
        "scripts": ["training/bootstrap_meta.py"],
        "description": "Bootstrap routing meta-learner via Ollama (local vs oracle)",
        "depends_on": [7],
        "validation": "check_phase9",
        "status_tag": "NEW — Ollama",
    },
    10: {
        "name": "Oracle Integration (Ollama)",
        "scripts": [],  # No scripts — validates Ollama setup
        "description": "Configure and verify Ollama local + cloud oracle access",
        "depends_on": [],
        "validation": "check_phase10",
        "status_tag": "NEW — Ollama",
    },
    11: {
        "name": "Cascade Inference Engine (Ollama)",
        "scripts": [],  # Engine is a library, not a script
        "description": "Full cascade: Ollama local + meta-learner + Ollama cloud oracle",
        "depends_on": [9, 10],
        "validation": "check_phase11",
        "status_tag": "NEW — Ollama",
    },
    12: {
        "name": "End-to-End Evaluation",
        "scripts": [
            "training/eval/eval_routing.py",
            "training/eval/eval_convergence.py",
        ],
        "description": "Routing accuracy + meta-learner convergence evaluation",
        "depends_on": [11],
        "validation": "check_phase12",
        "status_tag": "NEW — Ollama",
    },
}


# ============================================================
# VALIDATION CHECKS — Phases 1-4 (Data Pipeline)
# ============================================================

def check_phase1() -> dict:
    """Validate Phase 1: Corpus Assembly."""
    raw_dir = PROJECT_ROOT / "corpus" / "raw"
    if not raw_dir.exists():
        return {"files": 0, "size_mb": 0, "pass": False,
                "threshold": "50+ files (Descartes corpus)"}

    file_count = sum(1 for _ in raw_dir.rglob("*") if _.is_file())
    total_size = sum(
        f.stat().st_size for f in raw_dir.rglob("*") if f.is_file())

    status = {
        "files": file_count,
        "size_mb": round(total_size / (1024 * 1024), 1),
        "categories": {},
    }

    for cat_dir in raw_dir.iterdir():
        if cat_dir.is_dir():
            cat_files = sum(1 for _ in cat_dir.rglob("*") if _.is_file())
            status["categories"][cat_dir.name] = cat_files

    status["pass"] = file_count >= 50
    status["threshold"] = "50+ files (Descartes corpus)"
    return status


def check_phase2() -> dict:
    """Validate Phase 2: Text Extraction."""
    extracted_dir = PROJECT_ROOT / "corpus" / "extracted"
    metrics_path = extracted_dir / "extraction_metrics.json"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = {"total": 0, "success": 0, "failed": 0}

    txt_files = sum(1 for _ in extracted_dir.rglob("*.txt")
                    if _.name != "extraction_metrics.json"
                    ) if extracted_dir.exists() else 0

    status = {
        "extracted_files": txt_files,
        "metrics": metrics,
        "pass": txt_files > 0,
        "threshold": "extracted text files exist",
    }
    return status


def check_phase3() -> dict:
    """Validate Phase 3: Cleaning & Filtering."""
    cleaned_dir = PROJECT_ROOT / "corpus" / "cleaned"
    report_path = cleaned_dir / "cleaning_report.json"

    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {"total_documents": 0, "total_tokens_estimated": 0}

    status = {
        "documents": report.get("total_documents", 0),
        "tokens_estimated": report.get("total_tokens_estimated", 0),
        "mixing_ratios": report.get("mixing_ratios_actual", {}),
        "pass": report.get("total_tokens_estimated", 0) > 0,
        "threshold": "cleaned docs with token count > 0",
    }
    return status


def check_phase4() -> dict:
    """Validate Phase 4: CPT Data Formatting."""
    formatted_dir = PROJECT_ROOT / "corpus" / "formatted"
    card_path = formatted_dir / "dataset_card.json"

    if card_path.exists():
        card = json.loads(card_path.read_text())
    else:
        card = {}

    train_path = formatted_dir / "train.jsonl"
    val_path = formatted_dir / "val.jsonl"

    status = {
        "train_exists": train_path.exists(),
        "val_exists": val_path.exists(),
        "train_sequences": card.get("train_sequences", 0),
        "val_sequences": card.get("val_sequences", 0),
        "total_tokens": card.get("total_tokens_estimated", 0),
        "pass": train_path.exists() and val_path.exists(),
        "threshold": "train.jsonl and val.jsonl exist",
    }
    return status


# ============================================================
# VALIDATION CHECKS — Phase 5 (CPT Training)
# ============================================================

def check_phase5() -> dict:
    """Validate Phase 5: CPT Training on Qwen3-8B."""
    models_dir = PROJECT_ROOT / "models"

    cpt_model = models_dir / "descartes-8b-cpt"
    eval_path = (PROJECT_ROOT / "training" / "eval" /
                 "cpt_eval_results.json")

    status = {
        "model_exists": cpt_model.exists(),
        "eval_results_exist": eval_path.exists(),
        "pass": cpt_model.exists(),
        "threshold": "descartes-8b-cpt model checkpoint exists",
    }

    if eval_path.exists():
        try:
            status["eval_results"] = json.loads(eval_path.read_text())
        except json.JSONDecodeError:
            pass

    return status


# ============================================================
# VALIDATION CHECKS — Phases 6-7 (SFT)
# ============================================================

def check_phase6() -> dict:
    """Validate Phase 6: SFT Data Generation (Types A-G)."""
    examples_dir = PROJECT_ROOT / "training" / "sft" / "examples"

    files = {
        "all": examples_dir / "descartes_sft_all.jsonl",
        "standard": examples_dir / "descartes_sft_types_ABCD.jsonl",
        "cascade": examples_dir / "descartes_sft_types_EFG.jsonl",
    }

    counts = {}
    for name, path in files.items():
        if path.exists():
            with open(path) as f:
                counts[name] = sum(1 for _ in f)
        else:
            counts[name] = 0

    status = {
        "total_examples": counts.get("all", 0),
        "standard_examples": counts.get("standard", 0),
        "cascade_examples": counts.get("cascade", 0),
        "files_exist": all(p.exists() for p in files.values()),
        "pass": counts.get("all", 0) > 0,
        "threshold": "SFT JSONL files exist with examples",
    }
    return status


def check_phase7() -> dict:
    """Validate Phase 7: Two-Stage SFT."""
    models_dir = PROJECT_ROOT / "models"

    cascade_model = models_dir / "descartes-8b-cascade"
    stage1_model = models_dir / "descartes-8b-sft-s1"

    status = {
        "cascade_model_exists": cascade_model.exists(),
        "stage1_model_exists": stage1_model.exists(),
        "pass": cascade_model.exists(),
        "threshold": "descartes-8b-cascade model checkpoint exists",
    }
    return status


# ============================================================
# VALIDATION CHECKS — Phase 8 (Base Eval)
# ============================================================

def check_phase8() -> dict:
    """Validate Phase 8: Base Model Evaluation."""
    eval_path = (PROJECT_ROOT / "training" / "eval" /
                 "cpt_eval_results.json")

    status = {
        "results_exist": eval_path.exists(),
        "pass": False,
        "threshold": "CPT eval shows Descartes PPL decrease, general PPL < 15% increase",
    }

    if eval_path.exists():
        try:
            results = json.loads(eval_path.read_text())
            status["results"] = results
            status["pass"] = results.get("overall_pass", False)
        except json.JSONDecodeError:
            pass

    return status


# ============================================================
# VALIDATION CHECKS — Phase 9 (Meta-Learner Bootstrap)
# ============================================================

def check_phase9() -> dict:
    """Validate Phase 9: Meta-Learner Bootstrap."""
    meta_path = PROJECT_ROOT / "models" / "meta_learner_bootstrapped.pt"
    report_path = (PROJECT_ROOT / "training" / "eval" /
                   "bootstrap_report.json")

    status = {
        "meta_learner_exists": meta_path.exists(),
        "report_exists": report_path.exists(),
        "pass": meta_path.exists(),
        "threshold": "meta_learner_bootstrapped.pt exists",
    }

    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            status["total_questions"] = report.get("total_questions", 0)
            status["routing_accuracy"] = report.get("routing_accuracy", 0)
            status["oracle_cost"] = report.get("oracle_cost", 0)
        except json.JSONDecodeError:
            pass

    return status


# ============================================================
# VALIDATION CHECKS — Phase 10 (Oracle Integration)
# ============================================================

def check_phase10() -> dict:
    """Validate Phase 10: Ollama local + cloud oracle access."""
    import shutil
    import subprocess as sp

    # Check Ollama is installed
    ollama_installed = shutil.which("ollama") is not None

    # Check Ollama models available
    local_model_ready = False
    cloud_model_ready = False
    available_models = []

    if ollama_installed:
        try:
            result = sp.run(
                ["ollama", "list"], capture_output=True, text=True,
                timeout=10)
            if result.returncode == 0:
                output = result.stdout.lower()
                available_models = [
                    line.split()[0]
                    for line in output.strip().split('\n')[1:]
                    if line.strip()
                ]
                local_model_ready = any(
                    "descartes" in m for m in available_models)
                cloud_model_ready = any(
                    "cloud" in m for m in available_models)
        except Exception:
            pass

    # Fallback: also check for legacy API keys
    legacy_providers = {
        "deepseek": "DEEPSEEK_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    legacy_available = {
        name: bool(os.environ.get(env, "") and len(os.environ.get(env, "")) > 10)
        for name, env in legacy_providers.items()
    }
    any_legacy = any(legacy_available.values())

    # Pass if Ollama is set up OR legacy API keys exist
    ollama_ready = ollama_installed and (local_model_ready or cloud_model_ready)
    overall_pass = ollama_ready or any_legacy

    status = {
        "ollama_installed": ollama_installed,
        "local_model_ready": local_model_ready,
        "cloud_model_ready": cloud_model_ready,
        "available_models": available_models[:10],  # Limit display
        "legacy_api_keys": legacy_available,
        "pass": overall_pass,
        "threshold": "Ollama installed with models, or legacy API key configured",
    }

    if ollama_ready:
        status["active_provider"] = "ollama"
    elif any_legacy:
        for name, is_set in legacy_available.items():
            if is_set:
                status["active_provider"] = name
                break

    return status


# ============================================================
# VALIDATION CHECKS — Phase 11 (Cascade Engine)
# ============================================================

def check_phase11() -> dict:
    """Validate Phase 11: Cascade Inference Engine is ready."""
    # Check all required components exist (original + Ollama + VKS)
    components = {
        # Original HF-based
        "cascade_engine": (PROJECT_ROOT / "inference" /
                           "cascade_engine.py"),
        "meta_learner_module": (PROJECT_ROOT / "inference" /
                                "meta_learner.py"),
        "signal_extractor": (PROJECT_ROOT / "inference" /
                             "signal_extractor.py"),
        "signal_extractor_full": (PROJECT_ROOT / "inference" /
                                  "signal_extractor_full.py"),
        "oracle_module": (PROJECT_ROOT / "inference" / "oracle.py"),
        "z3_templates": (PROJECT_ROOT / "inference" / "templates" /
                         "descartes_z3.py"),
        # Ollama addendum (Addendum A)
        "engine_ollama": (PROJECT_ROOT / "inference" / "engine.py"),
        "signal_extractor_lite": (PROJECT_ROOT / "inference" /
                                  "signal_extractor_lite.py"),
        "feedback_module": (PROJECT_ROOT / "inference" / "feedback.py"),
        # VKS + Multi-tier verification (Addendum B)
        "knowledge_store": (PROJECT_ROOT / "inference" /
                            "knowledge_store.py"),
        "seed_axioms": (PROJECT_ROOT / "inference" / "seed_axioms.py"),
        "claim_extractor": (PROJECT_ROOT / "inference" /
                            "claim_extractor.py"),
        "claim_router": (PROJECT_ROOT / "inference" /
                         "claim_router.py"),
        "verifier": (PROJECT_ROOT / "inference" / "verifier.py"),
        "self_repair": (PROJECT_ROOT / "inference" / "self_repair.py"),
        "engine_v3": (PROJECT_ROOT / "inference" / "engine_v3.py"),
        # Reasoning Core (V3 Unified Architecture, Layers 1-5)
        "ontology_core": (PROJECT_ROOT / "reasoning_core" /
                          "ontology" / "core.py"),
        "ontology_theories": (PROJECT_ROOT / "reasoning_core" /
                              "ontology" / "theories.py"),
        "aspic_engine": (PROJECT_ROOT / "reasoning_core" /
                         "argumentation" / "aspic_engine.py"),
        "walton_schemes": (PROJECT_ROOT / "reasoning_core" /
                           "argumentation" / "walton_schemes.py"),
        "z3_engine": (PROJECT_ROOT / "reasoning_core" /
                      "verification" / "z3_engine.py"),
        "cvc5_engine": (PROJECT_ROOT / "reasoning_core" /
                        "verification" / "cvc5_engine.py"),
        "gvr_loop": (PROJECT_ROOT / "reasoning_core" /
                     "bridge" / "gvr_loop.py"),
        "conceptual_spaces": (PROJECT_ROOT / "reasoning_core" /
                              "spaces" / "conceptual_spaces.py"),
    }

    component_status = {k: v.exists() for k, v in components.items()}
    all_present = all(component_status.values())

    # Ollama-essential components (subset)
    ollama_components = {
        "engine_ollama", "meta_learner_module",
        "signal_extractor_lite", "feedback_module"
    }
    ollama_ready = all(
        component_status.get(k, False) for k in ollama_components)

    # VKS components (Addendum B)
    vks_components = {
        "knowledge_store", "claim_extractor", "claim_router",
        "verifier", "self_repair", "engine_v3"
    }
    vks_ready = all(
        component_status.get(k, False) for k in vks_components)

    # Reasoning Core components (V3 Unified Architecture)
    reasoning_core_components = {
        "ontology_core", "ontology_theories", "aspic_engine",
        "walton_schemes", "z3_engine", "cvc5_engine",
        "gvr_loop", "conceptual_spaces"
    }
    reasoning_core_ready = all(
        component_status.get(k, False) for k in reasoning_core_components)

    # Check model and meta-learner
    model_exists = (PROJECT_ROOT / "models" /
                    "descartes-8b-cascade").exists()
    meta_exists = (PROJECT_ROOT / "models" /
                   "meta_learner_bootstrapped.pt").exists()

    # Check oracle API
    oracle_ready = check_phase10()["pass"]

    # Check VKS store
    vks_exists = (PROJECT_ROOT / "models" / "vks.json").exists()

    status = {
        "components": component_status,
        "all_components_present": all_present,
        "ollama_components_present": ollama_ready,
        "vks_components_present": vks_ready,
        "reasoning_core_present": reasoning_core_ready,
        "vks_store_exists": vks_exists,
        "model_exists": model_exists,
        "meta_learner_exists": meta_exists,
        "oracle_configured": oracle_ready,
        "pass": ollama_ready and vks_ready,
        "threshold": "Ollama + VKS inference modules present",
    }

    # Full readiness requires model + meta-learner + oracle + VKS + reasoning core
    status["fully_ready"] = (ollama_ready and vks_ready and
                             reasoning_core_ready and
                             model_exists and meta_exists and
                             oracle_ready and vks_exists)

    return status


# ============================================================
# VALIDATION CHECKS — Phase 12 (End-to-End Evaluation)
# ============================================================

def check_phase12() -> dict:
    """Validate Phase 12: End-to-End Cascade Evaluation."""
    # Check both new eval results and legacy
    routing_path = (PROJECT_ROOT / "training" / "eval" /
                    "routing_eval_results.json")
    convergence_path = (PROJECT_ROOT / "training" / "eval" /
                        "convergence_eval_results.json")
    legacy_path = (PROJECT_ROOT / "training" / "eval" /
                   "cascade_eval_results.json")

    status = {
        "routing_results_exist": routing_path.exists(),
        "convergence_results_exist": convergence_path.exists(),
        "legacy_results_exist": legacy_path.exists(),
        "pass": False,
        "threshold": ("routing >= 80%, calibration error <= 0.15"),
    }

    # Check routing eval results
    if routing_path.exists():
        try:
            routing_results = json.loads(routing_path.read_text())
            status["routing_accuracy"] = routing_results.get("accuracy", 0)
            status["routing_pass"] = routing_results.get("pass", False)
        except json.JSONDecodeError:
            pass

    # Check convergence eval results
    if convergence_path.exists():
        try:
            conv_results = json.loads(convergence_path.read_text())
            status["calibration_pass"] = conv_results.get(
                "calibration_pass", False)
            status["convergence_pass"] = conv_results.get(
                "overall_pass", False)
        except json.JSONDecodeError:
            pass

    # Check legacy results
    if legacy_path.exists():
        try:
            results = json.loads(legacy_path.read_text())
            status["legacy_results"] = {
                "validity": results.get("validity", {}).get(
                    "accuracy", 0),
                "routing": results.get("routing", {}).get(
                    "accuracy", 0),
                "knowledge": results.get("knowledge", {}).get(
                    "accuracy", 0),
                "calibration_ece": results.get("calibration", {}).get(
                    "ece", 1.0),
            }
        except json.JSONDecodeError:
            pass

    # Overall pass: new eval OR legacy
    status["pass"] = (
        status.get("routing_pass", False) or
        status.get("convergence_pass", False) or
        (legacy_path.exists() and
         json.loads(legacy_path.read_text()).get("overall_pass", False)
         if legacy_path.exists() else False)
    )

    return status


# ============================================================
# VALIDATORS REGISTRY
# ============================================================

VALIDATORS = {
    "check_phase1": check_phase1,
    "check_phase2": check_phase2,
    "check_phase3": check_phase3,
    "check_phase4": check_phase4,
    "check_phase5": check_phase5,
    "check_phase6": check_phase6,
    "check_phase7": check_phase7,
    "check_phase8": check_phase8,
    "check_phase9": check_phase9,
    "check_phase10": check_phase10,
    "check_phase11": check_phase11,
    "check_phase12": check_phase12,
}


# ============================================================
# PIPELINE EXECUTION
# ============================================================

def run_phase(phase_num: int, extra_args: list = None):
    """Run a single phase."""
    phase = PHASES[phase_num]
    tag = phase.get("status_tag", "")
    tag_str = f" [{tag}]" if tag else ""

    print(f"\n{'=' * 60}")
    print(f"PHASE {phase_num}: {phase['name'].upper()}{tag_str}")
    print(f"{'=' * 60}")
    print(f"Description: {phase['description']}")

    # Check dependencies
    for dep in phase["depends_on"]:
        validator = VALIDATORS[PHASES[dep]["validation"]]
        result = validator()
        if not result["pass"]:
            print(f"\n  WARNING: Phase {dep} ({PHASES[dep]['name']}) "
                  f"has not completed successfully.")
            print(f"  Threshold: {result['threshold']}")
            response = input(
                "  Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("  Skipping phase.")
                return False

    # Handle phases with no scripts (setup/validation only)
    if not phase["scripts"]:
        print(f"\n  Phase {phase_num} has no scripts to run.")
        print(f"  This phase validates configuration/setup.")

    # Run scripts
    for script in phase["scripts"]:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            print(f"\n  WARNING: Script not found: {script}")
            print(f"  Full path: {script_path}")
            response = input(
                "  Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                return False
            continue

        print(f"\nRunning: {script}")
        print("-" * 40)

        cmd = [sys.executable, str(script_path)]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode != 0:
            print(f"\nERROR: {script} exited with code "
                  f"{result.returncode}")
            return False

    # Validate
    print(f"\nValidating Phase {phase_num}...")
    validator = VALIDATORS[phase["validation"]]
    result = validator()
    passed = result["pass"]
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print(f"  Threshold: {result['threshold']}")

    # Show extra info for cascade phases
    if phase_num == 9 and "total_questions" in result:
        print(f"  Bootstrap questions: {result['total_questions']}")
        print(f"  Routing accuracy: {result.get('routing_accuracy', 'N/A')}")
        print(f"  Oracle cost: ${result.get('oracle_cost', 0):.4f}")
    elif phase_num == 10:
        ollama_ok = "OK" if result.get("ollama_installed") else "--"
        local_ok = "OK" if result.get("local_model_ready") else "--"
        cloud_ok = "OK" if result.get("cloud_model_ready") else "--"
        print(f"    [{ollama_ok}] Ollama installed")
        print(f"    [{local_ok}] Local model (descartes:8b)")
        print(f"    [{cloud_ok}] Cloud oracle")
        models = result.get("available_models", [])
        if models:
            print(f"    Models: {', '.join(models[:5])}")
        # Show legacy API keys as fallback
        for provider, avail in result.get("legacy_api_keys", {}).items():
            if avail:
                print(f"    [OK] Legacy: {provider} API key")
    elif phase_num == 12:
        if "routing_accuracy" in result:
            print(f"  Routing accuracy: {result['routing_accuracy']:.1%}")
        if "routing_pass" in result:
            print(f"  Routing:     {'PASS' if result['routing_pass'] else 'FAIL'}")
        if "calibration_pass" in result:
            print(f"  Calibration: {'PASS' if result['calibration_pass'] else 'FAIL'}")
        if "legacy_results" in result:
            r = result["legacy_results"]
            print(f"  (Legacy) Validity:  {r.get('validity', 0):.1%}")
            print(f"  (Legacy) Knowledge: {r.get('knowledge', 0):.1%}")

    return passed


def show_status():
    """Show current pipeline status."""
    print("=" * 60)
    print("PHILOSOPHER ENGINE — CASCADE PIPELINE STATUS")
    print(f"{'=' * 60}")
    print(f"Project:  {PROJECT_ROOT}")
    print(f"Date:     {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Arch:     Qwen3-8B (trained) + Oracle API (untrained)")
    print()

    # Group phases
    groups = [
        ("DATA PIPELINE", [1, 2, 3, 4]),
        ("MODEL TRAINING", [5, 6, 7, 8]),
        ("CASCADE SYSTEM", [9, 10, 11, 12]),
    ]

    total_pass = 0
    total_phases = len(PHASES)

    for group_name, phase_nums in groups:
        print(f"  --- {group_name} ---")
        for num in phase_nums:
            phase = PHASES[num]
            validator = VALIDATORS[phase["validation"]]
            result = validator()
            icon = "PASS" if result["pass"] else "----"
            tag = phase.get("status_tag", "")
            tag_str = f" [{tag}]" if tag else ""

            if result["pass"]:
                total_pass += 1

            deps = (f" (depends: {phase['depends_on']})"
                    if phase['depends_on'] else "")
            print(f"    Phase {num:>2}: [{icon}] "
                  f"{phase['name']}{tag_str}{deps}")

            # Show key metrics
            for key in ["files", "documents", "tokens_estimated",
                        "total_examples", "train_sequences",
                        "total_questions"]:
                if key in result and result[key]:
                    print(f"              {key}: {result[key]:,}")

        print()

    # Summary
    print(f"  Progress: {total_pass}/{total_phases} phases complete")

    # Check full cascade readiness
    p11 = check_phase11()
    if p11.get("fully_ready"):
        print(f"  CASCADE: READY FOR DEPLOYMENT")
    elif p11.get("all_components_present"):
        missing = []
        if not p11.get("model_exists"):
            missing.append("trained model")
        if not p11.get("meta_learner_exists"):
            missing.append("meta-learner")
        if not p11.get("oracle_configured"):
            missing.append("oracle API key")
        print(f"  CASCADE: Components present, missing: "
              f"{', '.join(missing)}")
    else:
        print(f"  CASCADE: Not yet assembled")

    print()


def parse_phase_range(phase_str: str) -> list:
    """Parse phase specification: '1', '1-4', '5-12', 'all'.

    Also supports named groups:
        'data'    -> Phases 1-4
        'train'   -> Phases 5-8
        'cascade' -> Phases 9-12
    """
    named_groups = {
        "all": list(PHASES.keys()),
        "data": [1, 2, 3, 4],
        "train": [5, 6, 7, 8],
        "cascade": [9, 10, 11, 12],
    }

    if phase_str.lower() in named_groups:
        return named_groups[phase_str.lower()]

    if "-" in phase_str:
        start, end = phase_str.split("-")
        return list(range(int(start), int(end) + 1))

    return [int(phase_str)]


def main():
    parser = argparse.ArgumentParser(
        description="Philosopher Engine: Cascade Training Pipeline"
    )
    parser.add_argument(
        "--phase", type=str, default=None,
        help=("Phase(s) to run: '1', '1-4', '5-12', 'all', "
              "'data', 'train', 'cascade'"))
    parser.add_argument(
        "--status", action="store_true",
        help="Show pipeline status")
    parser.add_argument(
        "--validate", type=int, default=None,
        help="Run validation for a specific phase")
    parser.add_argument(
        "--provider", type=str, default="deepseek",
        help="Oracle provider for Phases 9-12 (deepseek/claude/openai)")
    parser.add_argument(
        "--api", action="store_true",
        help="Pass --api flag to sub-scripts (for Phase 6 LLM council)")

    args, extra = parser.parse_known_args()

    if args.status:
        show_status()
        return

    if args.validate is not None:
        if args.validate not in PHASES:
            print(f"Unknown phase: {args.validate}")
            print(f"Valid phases: {list(PHASES.keys())}")
            sys.exit(1)

        phase = PHASES[args.validate]
        validator = VALIDATORS[phase["validation"]]
        result = validator()
        print(f"Phase {args.validate} ({phase['name']}): "
              f"{'PASS' if result['pass'] else 'FAIL'}")
        print(json.dumps(result, indent=2, default=str))
        return

    if args.phase:
        phases = parse_phase_range(args.phase)

        # Validate all phase numbers
        invalid = [p for p in phases if p not in PHASES]
        if invalid:
            print(f"Unknown phase(s): {invalid}")
            print(f"Valid phases: {list(PHASES.keys())}")
            sys.exit(1)

        print("=" * 60)
        print("PHILOSOPHER ENGINE — CASCADE TRAINING PIPELINE")
        print("=" * 60)
        print(f"Running phases: {phases}")
        print(f"Oracle provider: {args.provider}")

        # Build extra args for sub-scripts
        extra_args = []
        if args.provider != "deepseek":
            extra_args.extend(["--provider", args.provider])
        if args.api:
            extra_args.append("--api")
        extra_args.extend(extra)

        for phase_num in phases:
            success = run_phase(phase_num, extra_args)
            if not success:
                print(f"\nPhase {phase_num} did not pass validation.")
                if phase_num < max(phases):
                    response = input(
                        "Continue to next phase? (y/N): "
                    ).strip().lower()
                    if response != 'y':
                        break

        print("\nPipeline run complete.")
        show_status()
    else:
        parser.print_help()
        print("\n")
        show_status()


if __name__ == "__main__":
    main()
