#!/usr/bin/env bash
# Reproduce LongCat-AudioDiT fp32 Seed-TTS quality with the current order-free protocol.
#
# What this script does:
#   1. Generate Seed-TTS wavs with the fp32 LongCat-AudioDiT model.
#   2. Automatically evaluate generated wavs with scripts/evaluate_seedtts_metrics.sh.
#
# Usage:
#   bash scripts/benchmark/benchmark_fp32_seedtts.sh [1b|3.5b|both] [zh,en,hard]
# Examples:
#   bash scripts/benchmark/benchmark_fp32_seedtts.sh 1b "zh en hard"
#   LIMIT=1 bash scripts/benchmark/benchmark_fp32_seedtts.sh 1b hard
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark/benchmark_fp32_seedtts.sh 1b "zh en hard"
#
# Default eval metrics: zh/hard -> CER, en -> WER; plus MOS (UTMOS + DNSMOS) + WavLM SIM on all sets.
#   (SIM needs eval/ckpt/wavlm_large_finetune.pth; override with e.g. EVAL_METRICS="wer cer mos".)
#
# Common env knobs:
#   LIMIT=0          items per set; 0 means full set
#   BASE=1024        per-item generation seed base
#   EVAL_METRICS=... metrics passed to evaluate_seedtts_metrics.sh
#
# Naming convention (aligned across all quality benchmarks):
#   gen wavs -> gen/paired/<tag>/<set>/*.wav
#   metrics  -> results/<tag>_<set>_<metric>.txt
#   the gen tag and the metric prefix are the SAME token; 3.5B adds the _3.5b suffix (tag=fp32 / fp32_3.5b).
#
# Required data:
#   data/seedtts_testset/zh/meta.lst and data/seedtts_testset/en/meta.lst
#   eval/ckpt/wavlm_large_finetune.pth when running SIM
#   Install with: bash scripts/setup/download_seedtts_testset.sh
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

  # fp32 has no calibration -> generation shards cleanly across the GPU pool (item sharding).
  gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>
    "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
      --mode fp32 --tag "$tag" --base "$base" \
      --sets "$1" --offset "$2" --limit "$3" \
      --model_dir "$model_dir" --device "$DEVICE"
  }
  echo "=== [fp32/$label] generate sets=$sets_csv tag=$tag ==="
  run_gen_parallel gen_cb "${sets_csv//,/ }" "$limit"

  echo "=== [fp32/$label] evaluate ==="
  bash scripts/evaluate_seedtts_metrics.sh "gen/paired/$tag" "$tag" "${sets_csv//,/ }" "$eval_metrics"
}

case "$which_model" in
  1b|1B)
    run_one 1B meituan-longcat/LongCat-AudioDiT-1B fp32
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B fp32_3.5b
    ;;
  both)
    run_one 1B   meituan-longcat/LongCat-AudioDiT-1B   fp32
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B fp32_3.5b
    ;;
  *)
    echo "usage: bash scripts/benchmark/benchmark_fp32_seedtts.sh [1b|3.5b|both] [zh,en,hard]" >&2
    exit 2
    ;;
esac
