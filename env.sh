#!/usr/bin/env bash
# Source this once per shell before running any experiment:  source env.sh
# Sets the paths every script expects. paths.py auto-resolves data/eval/gen/results
# under SEED_REPRO_DIR; the model code (audiodit + batch_inference) lives in the
# EXTERNAL LongCat repo, and the int4 deploy kernels need the cu130 toolchain.

AQ="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export SEED_REPRO_DIR="$AQ"                                    # this folder = data/eval/gen/results root
export LONGCAT_DIR="${LONGCAT_DIR:-$HOME/workspace/LongCat-AudioDiT}"   # external: AudioDiTModel + batch_inference.infer_one
export SEED_EVAL_DIR="$AQ/eval/seed-tts-eval"                  # ASR/SIM harness (score_seedtts_asr.sh cd's here)
export SEED_DATA_DIR="${SEED_DATA_DIR:-$AQ/data/seedtts_testset}"
export SEED_GEN_DIR="${SEED_GEN_DIR:-$AQ/gen}"
export SEED_RESULTS_DIR="${SEED_RESULTS_DIR:-$AQ/results}"
export WAVLM_CKPT="${WAVLM_CKPT:-$AQ/eval/ckpt/wavlm_large_finetune.pth}"
export FLATQUANT_REF_DIR="$AQ/vendor/flatquant_ref"           # FlatQuant algo + int4 deploy kernels (incl. HP fp32 kron)
export PYTHON_BIN="${PYTHON_BIN:-$AQ/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  export PYTHON_BIN="$(command -v python3.13 || command -v python3 || command -v python)"
fi
export GPU_INDEX="${GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}"
export DEVICE="${DEVICE:-cuda:0}"

# Python implementation files live under src/audio_dit_quantize; vendor gives `flatquant`+`deploy`;
# LongCat gives `audiodit`+`batch_inference`.
export PYTHONPATH="$AQ/src:$AQ/vendor/flatquant_ref:$LONGCAT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# int4 CUTLASS deploy kernels were built against cu130 — prepend if present (override CUDA_HOME to change)
if [ -d /usr/local/cuda-13.3 ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

echo "[env] SEED_REPRO_DIR = $SEED_REPRO_DIR"
echo "[env] LONGCAT_DIR    = $LONGCAT_DIR  $( [ -d "$LONGCAT_DIR" ] && echo '(ok)' || echo '(MISSING — clone LongCat-AudioDiT)')"
echo "[env] PYTHON_BIN     = $PYTHON_BIN"
echo "[env] PYTHONPATH     = src : vendor/flatquant_ref : LongCat"
