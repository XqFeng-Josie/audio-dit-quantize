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


SETS = {"zh": "zh/meta.lst", "en": "en/meta.lst", "hard": "zh/hardcase.lst"}
