#!/usr/bin/env bash
# QuaRot-GPTQ W4A4 (paper-best: Hadamard rotation + GPTQ weights with --w_clip MSE search) quality
# run under the order-free per-item-seeded protocol. Sibling of benchmark_svdquant_seedtts.sh —
# same calibrate-once/reuse + sharded-gen pattern, only --mode quarot_gptq.
#
# The training-free QuaRot-RTN baseline stays in benchmark_quarot_seedtts.sh; THIS script is the
# paper-headline config (fake_quant/README: GPTQ weights, percdamp 0.01, w_clip MSE grid search,
# per-token sym A4). Calibration uses the same fixed data/calib_heldout_hardlike32.lst and the same
# 64x2 block-0 capture recipe as flatquant_best, so the calib-data budget matches across methods.
#
# What this script does:
#   1. Calibrate QuaRot-GPTQ ONCE on a single GPU (capture + sequential per-block GPTQ) and save a
#      canonical models/qgptq_{1b,3p5b}_model.pt (capture is RNG-driven -> save+reuse for
#      reproducibility AND so every generation shard uses the SAME draw).
#   2. Generate Seed-TTS wavs by loading that model, item-sharded across the GPU range (GPUS).
#   3. Automatically evaluate generated wavs with scripts/evaluate_seedtts_metrics.sh.
#
# Usage:
#   bash scripts/benchmark_quarot_gptq_seedtts.sh [1b|3.5b|both] [zh,en,hard]
# Examples:
#   bash scripts/benchmark_quarot_gptq_seedtts.sh 1b "zh en hard"
#   GPUS=0,1,2,3 bash scripts/benchmark_quarot_gptq_seedtts.sh 1b "zh en hard"
#   LIMIT=1 bash scripts/benchmark_quarot_gptq_seedtts.sh 1b hard
#
# Common env knobs:
#   CALIB_SEED=0        calibration RNG seed        (⚠ baked into qgptq_*.pt; delete the .pt to change it)
#   BASE=1024           per-item generation seed base
#   LIMIT=0             items per set; 0 means full set
#   GPUS=0,1,2,3        GPUs for item-sharded generation + parallel eval (calibration stays single-GPU)
#   EVAL_METRICS=...    metrics passed to evaluate_seedtts_metrics.sh
#
# Naming convention (aligned across all quality benchmarks):
#   gen wavs -> gen/paired/<tag>/<set>/*.wav ; metrics -> results/<tag>_<set>_<metric>.txt
#   gen tag == metric prefix; 3.5B adds the _3.5b suffix (tag=quarot_gptq / quarot_gptq_3.5b).
# Required data/ckpt: same as benchmark_fp32_seedtts.sh (bash scripts/download_seedtts_testset.sh).
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

  if [ ! -f "$model_pt" ]; then
    echo "=== [QuaRot-GPTQ/$label] calibrated model $model_pt absent -> calibrate once (single-GPU${cgpu:+ =$cgpu}) + save ==="
    CUDA_VISIBLE_DEVICES="${cgpu:-${CUDA_VISIBLE_DEVICES:-0}}" "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
      --mode quarot_gptq --tag "$tag" --model "$model_pt" --save_model --calibrate_only \
      --model_dir "$model_dir" --calib_seed "${CALIB_SEED:-0}" --device cuda:0
  else
    echo "=== [QuaRot-GPTQ/$label] reusing calibrated model $model_pt ==="
  fi

  gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>  — load the calibrated model, generate a shard
    "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
      --mode quarot_gptq --tag "$tag" --model "$model_pt" --load_model \
      --sets "$1" --offset "$2" --limit "$3" --base "$base" --model_dir "$model_dir" --device "$DEVICE"
  }
  echo "=== [QuaRot-GPTQ/$label] generate sets=$sets_csv tag=$tag (sharded) ==="
  run_gen_parallel gen_cb "${sets_csv//,/ }" "$limit"

  echo "=== [QuaRot-GPTQ/$label] evaluate ==="
  bash scripts/evaluate_seedtts_metrics.sh "gen/paired/$tag" "$tag" "${sets_csv//,/ }" "$eval_metrics"
}

case "$which_model" in
  1b|1B)
    run_one 1B meituan-longcat/LongCat-AudioDiT-1B quarot_gptq "$SEED_MODELS_DIR/qgptq_1b_model.pt"
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B quarot_gptq_3.5b "$SEED_MODELS_DIR/qgptq_3p5b_model.pt"
    ;;
  both)
    run_one 1B   meituan-longcat/LongCat-AudioDiT-1B   quarot_gptq      "$SEED_MODELS_DIR/qgptq_1b_model.pt"
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B quarot_gptq_3.5b "$SEED_MODELS_DIR/qgptq_3p5b_model.pt"
    ;;
  *)
    echo "usage: bash scripts/benchmark_quarot_gptq_seedtts.sh [1b|3.5b|both] [zh,en,hard]" >&2
    exit 2
    ;;
esac
