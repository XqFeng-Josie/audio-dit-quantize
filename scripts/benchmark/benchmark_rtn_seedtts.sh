#!/usr/bin/env bash
# RTN W4A4 (naive round-to-nearest, no rotation/calibration) Seed-TTS QUALITY with the order-free protocol.
# Sibling of benchmark_{fp32,int8,quarot}_seedtts.sh — same generator (generate_seedtts) + same eval, only
# --mode rtn. RTN is training-free + deterministic (per-channel/per-token symmetric fake-quant, no calib),
# so there is no model to save/load and generation shards cleanly across GPUs — like fp32/int8/quarot.
#
# ⚠ RTN is the naive W4A4 LOWER-BOUND baseline (no Hadamard, no learned transform, no calibration): its
#   quality is expected to collapse. It exists as the control that FlatQuant/SVDQuant/QuaRot improve on.
#
# What this script does:
#   1. Generate Seed-TTS wavs with RTN W4A4 (generate_seedtts --mode rtn).
#   2. Evaluate with scripts/evaluate_seedtts_metrics.sh.
#
# Usage:
#   bash scripts/benchmark/benchmark_rtn_seedtts.sh [1b|3.5b|both] [zh,en,hard]
# Examples:
#   bash scripts/benchmark/benchmark_rtn_seedtts.sh 1b "zh en hard"
#   GPUS=0,1,2,3 bash scripts/benchmark/benchmark_rtn_seedtts.sh 1b "zh en hard"   # item-shard gen 4-way + parallel eval
#   LIMIT=1 bash scripts/benchmark/benchmark_rtn_seedtts.sh 1b hard
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark/benchmark_rtn_seedtts.sh both "zh en hard"
#
# Default eval metrics: zh/hard -> CER, en -> WER; plus MOS (UTMOS + DNSMOS) + WavLM SIM on all sets.
#   (SIM needs eval/ckpt/wavlm_large_finetune.pth; override with e.g. EVAL_METRICS="wer cer mos".)
#
# Multi-GPU: set GPUS="0,1,2,3" to shard generation by item across those GPUs and parallelize eval.
# Env knobs: LIMIT=0 (0=full set), BASE=1024 (per-item seed base), EVAL_METRICS=...
#
# Naming convention (aligned across all quality benchmarks):
#   gen wavs -> gen/paired/<tag>/<set>/*.wav ; metrics -> results/<tag>_<set>_<metric>.txt
#   gen tag == metric prefix; 3.5B adds the _3.5b suffix (tag=rtn / rtn_3.5b).
# Required data/ckpt: same as benchmark_fp32_seedtts.sh (bash scripts/setup/download_seedtts_testset.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"   # GPU-range knob (GPUS/CUDA_VISIBLE_DEVICES) + item-shard fan-out
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
sets_csv="${2:-zh,en,hard}"
limit="${LIMIT:-0}"
base="${BASE:-1024}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer mos sim}}"

run_one() {
  local label="$1" model_dir="$2" tag="$3"          # tag = gen subdir under paired/ AND metric prefix

  # RTN is deterministic (no calibration draw) -> generation shards cleanly across the GPU pool.
  gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>
    "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
      --mode rtn --tag "$tag" --base "$base" \
      --sets "$1" --offset "$2" --limit "$3" \
      --model_dir "$model_dir" --device "$DEVICE"
  }
  echo "=== [rtn/$label] generate sets=$sets_csv tag=$tag ==="
  run_gen_parallel gen_cb "${sets_csv//,/ }" "$limit"

  echo "=== [rtn/$label] evaluate ==="
  bash scripts/evaluate_seedtts_metrics.sh "gen/paired/$tag" "$tag" "${sets_csv//,/ }" "$eval_metrics"
}

case "$which_model" in
  1b|1B)
    run_one 1B meituan-longcat/LongCat-AudioDiT-1B rtn
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B rtn_3.5b
    ;;
  both)
    run_one 1B   meituan-longcat/LongCat-AudioDiT-1B   rtn
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B rtn_3.5b
    ;;
  *)
    echo "usage: bash scripts/benchmark/benchmark_rtn_seedtts.sh [1b|3.5b|both] [zh,en,hard]" >&2
    exit 2
    ;;
esac
