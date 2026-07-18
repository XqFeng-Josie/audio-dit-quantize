"""Shared filesystem paths for audio-dit-quantize scripts.

The research repo owns data/eval/gen/results.  The external LongCat model repo
is still discovered through ``LONGCAT_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_REPRO_DIR = _resolve_env_path(
    "SEED_REPRO_DIR",
    REPO_ROOT,
)
LONGCAT_DIR = _resolve_env_path(
    "LONGCAT_DIR",
    SEED_REPRO_DIR.parent / "LongCat-AudioDiT",
)
DATA_DIR = _resolve_env_path(
    "SEED_DATA_DIR",
    SEED_REPRO_DIR / "data" / "seedtts_testset",
)
GEN_DIR = _resolve_env_path("SEED_GEN_DIR", SEED_REPRO_DIR / "gen")
RESULTS_DIR = _resolve_env_path("SEED_RESULTS_DIR", SEED_REPRO_DIR / "results")
EVAL_DIR = _resolve_env_path("SEED_EVAL_DIR", SEED_REPRO_DIR / "eval" / "seed-tts-eval")
WAVLM_CKPT = _resolve_env_path(
    "WAVLM_CKPT",
    SEED_REPRO_DIR / "eval" / "ckpt" / "wavlm_large_finetune.pth",
)
FLATQUANT_REF_DIR = _resolve_env_path(
    "FLATQUANT_REF_DIR",
    SEED_REPRO_DIR / "vendor" / "flatquant_ref",
)
CALIB_LST = (REPO_ROOT / "data" / "calib_heldout_hardlike32.lst").resolve()

# ONE canonical location for the fixed best-config calibration models, shared by every script that loads
# them (generate_step_axis, w4a4_deploy_quality, w4a4_deploy_check_numerics). Override the dir with
# SEED_MODELS_DIR (e.g. point at an existing calibration dir instead of copying the 5–15 GB files).
MODELS_DIR = _resolve_env_path("SEED_MODELS_DIR", REPO_ROOT / "models")


def bc_model_path(model_dir: str) -> Path:
    """Canonical fixed best-config model path for a HF model_dir (1B -> bc_1b_model.pt, 3.5B -> bc_3p5b_model.pt)."""
    tag = "3p5b" if ("3.5" in model_dir or "3p5" in model_dir) else "1b"
    return MODELS_DIR / f"bc_{tag}_model.pt"


def qgptq_model_path(model_dir: str) -> Path:
    """Canonical calibrated QuaRot-GPTQ model path (1B -> qgptq_1b_model.pt, 3.5B -> qgptq_3p5b_model.pt).
    Same calibrate-once/reuse pattern as svd_model_path."""
    tag = "3p5b" if ("3.5" in model_dir or "3p5" in model_dir) else "1b"
    return MODELS_DIR / f"qgptq_{tag}_model.pt"


def svd_model_path(model_dir: str) -> Path:
    """Canonical calibrated SVDQuant model path (1B -> svd_1b_model.pt, 3.5B -> svd_3p5b_model.pt).
    Calibrate once (single-GPU) + save, then load for sharded multi-GPU generation — the same
    calibrate-once/reuse pattern as bc_model_path, giving reproducibility + shardability."""
    tag = "3p5b" if ("3.5" in model_dir or "3p5" in model_dir) else "1b"
    return MODELS_DIR / f"svd_{tag}_model.pt"


SETS = {"zh": "zh/meta.lst", "en": "en/meta.lst", "hard": "zh/hardcase.lst"}
