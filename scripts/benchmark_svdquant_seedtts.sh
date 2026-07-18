#!/usr/bin/env bash
# SVDQuant W4A4 quality run under the current order-free per-item-seeded protocol.
#
# What this script does (same calibrate-once/reuse + sharded-gen pattern as flatquant):
#   1. Calibrate SVDQuant ONCE on a single GPU with fixed data/calib_heldout_hardlike32.lst and save a
#      canonical models/svd_{1b,3p5b}_model.pt (SVDQuant calibration is non-deterministic, so the model
#      must be saved+reused for reproducibility AND so every generation shard uses the SAME draw).
#   2. Generate Seed-TTS wavs by loading that model, item-sharded across the GPU range (GPUS).
#   3. Automatically evaluate generated wavs with scripts/evaluate_seedtts_metrics.sh.
#
# Usage:
#   bash scripts/benchmark_svdquant_seedtts.sh [1b|3.5b|both] [zh,en,hard]
# Examples:
#   bash scripts/benchmark_svdquant_seedtts.sh 1b "zh en hard"
#   GPUS=0,1,2,3 bash scripts/benchmark_svdquant_seedtts.sh 1b "zh en hard"   # calibrate 1 GPU, gen 4-way
#   LIMIT=1 RANK=32 bash scripts/benchmark_svdquant_seedtts.sh 1b hard
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark_svdquant_seedtts.sh 1b "zh en hard"
#
# Default eval metrics: zh/hard -> CER, en -> WER; plus MOS (UTMOS + DNSMOS) + WavLM SIM on all sets.
#   (SIM needs eval/ckpt/wavlm_large_finetune.pth; override with e.g. EVAL_METRICS="wer cer mos".)
#
# Common env knobs:
#   RANK=32             SVD low-rank residual rank  (⚠ baked into svd_*.pt; delete the .pt to change it)
#   CALIB_SEED=0        calibration RNG seed        (⚠ baked into svd_*.pt; delete the .pt to change it)
#   SVD_ROWS=2048       calib activation rows per linear (legacy runs used 512; ⚠ baked into svd_*.pt)
#   SVD_ITERS=          low-rank refinement iters; empty = paper-best 100 w/ early stop, 1 = legacy
#                       one-shot ablation (⚠ baked into svd_*.pt)
#   ⚠ svd_*.pt files calibrated BEFORE the paper-best alignment (one-shot SVD, asym act quant,
#     19-candidate α grid) are stale — delete them to re-calibrate with the aligned config.
#   BASE=1024           per-item generation seed base
#   LIMIT=0             items per set; 0 means full set
#   GPUS=0,1,2,3        GPUs for item-sharded generation + parallel eval (calibration stays single-GPU)
#   EVAL_METRICS=...    metrics passed to evaluate_seedtts_metrics.sh
#
# Naming convention (aligned across all quality benchmarks):
#   gen wavs -> gen/paired/<tag>/<set>/*.wav ; metrics -> results/<tag>_<set>_<metric>.txt
#   gen tag == metric prefix; 3.5B adds the _3.5b suffix (tag=svd / svd_3.5b).
#
# Required data:
#   data/seedtts_testset/zh/meta.lst and data/seedtts_testset/en/meta.lst
#   eval/ckpt/wavlm_large_finetune.pth when running SIM
#   Install with: bash scripts/download_seedtts_testset.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"   # GPU-range knob (GPUS/CUDA_VISIBLE_DEVICES); eval parallelizes
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
sets_csv="${2:-zh,en,hard}"
limit="${LIMIT:-0}"
base="${BASE:-1024}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer mos sim}}"

run_one() {
  local label="$1" model_dir="$2" tag="$3" model_pt="$4"   # tag = gen subdir under paired/ AND metric prefix

  # First GPU of the range is used for the (single-GPU) calibration.
  local pool; read -r -a pool <<< "$(gpu_pool)"
  local cgpu=""; [ "${#pool[@]}" -ge 1 ] && cgpu="${pool[0]}"

  # Calibrate ONCE (single-GPU) if the canonical model is absent, then fan out generation with --load_model
  # so every shard uses the SAME (non-deterministic) draw. Same pattern as flatquant's canonical model.
  if [ ! -f "$model_pt" ]; then
    echo "=== [SVDQuant/$label] calibrated model $model_pt absent -> calibrate once (single-GPU${cgpu:+ =$cgpu}) + save ==="
    CUDA_VISIBLE_DEVICES="${cgpu:-${CUDA_VISIBLE_DEVICES:-0}}" "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
      --mode svdquant --tag "$tag" --model "$model_pt" --save_model --calibrate_only \
      --model_dir "$model_dir" --rank "${RANK:-32}" --calib_seed "${CALIB_SEED:-0}" \
      --svd_rows "${SVD_ROWS:-2048}" ${SVD_ITERS:+--svd_iters "$SVD_ITERS"} --device cuda:0
  else
    echo "=== [SVDQuant/$label] reusing calibrated model $model_pt ==="
  fi

  gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>  — load the calibrated model, generate a shard
    "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
      --mode svdquant --tag "$tag" --model "$model_pt" --load_model \
      --sets "$1" --offset "$2" --limit "$3" --base "$base" --model_dir "$model_dir" --device "$DEVICE"
  }
  echo "=== [SVDQuant/$label] generate sets=$sets_csv tag=$tag (sharded) ==="
  run_gen_parallel gen_cb "${sets_csv//,/ }" "$limit"

  echo "=== [SVDQuant/$label] evaluate ==="
  bash scripts/evaluate_seedtts_metrics.sh "gen/paired/$tag" "$tag" "${sets_csv//,/ }" "$eval_metrics"
}

case "$which_model" in
  1b|1B)
    run_one 1B meituan-longcat/LongCat-AudioDiT-1B svd "$SEED_MODELS_DIR/svd_1b_model.pt"
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B svd_3.5b "$SEED_MODELS_DIR/svd_3p5b_model.pt"
    ;;
  both)
    run_one 1B   meituan-longcat/LongCat-AudioDiT-1B   svd      "$SEED_MODELS_DIR/svd_1b_model.pt"
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B svd_3.5b "$SEED_MODELS_DIR/svd_3p5b_model.pt"
    ;;
  *)
    echo "usage: bash scripts/benchmark_svdquant_seedtts.sh [1b|3.5b|both] [zh,en,hard]" >&2
    exit 2
    ;;
esac
