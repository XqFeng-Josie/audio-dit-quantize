#!/usr/bin/env bash
# Reproduce LongCat-AudioDiT INT8 (W8A8 fake-quant) Seed-TTS QUALITY with the order-free protocol.
# Sibling of benchmark_fp32_seedtts.sh — same generator (generate_seedtts) + same eval, only --mode int8.
# (Distinct from benchmark_int8_efficiency.sh, which measures LATENCY via torchao real W8A8, not quality.)
#
# What this script does:
#   1. Generate Seed-TTS wavs with the INT8 W8A8 fake-quant LongCat-AudioDiT model (generate_seedtts --mode int8).
#   2. Evaluate with scripts/evaluate_seedtts_metrics.sh -> results/pf_int8[_3.5b]_<set>_<metric>.txt.
#
# Usage:
#   bash scripts/benchmark_int8_seedtts.sh [1b|3.5b|both] [zh,en,hard]
# Examples:
#   bash scripts/benchmark_int8_seedtts.sh 1b "zh en hard"
#   LIMIT=1 bash scripts/benchmark_int8_seedtts.sh 1b hard
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark_int8_seedtts.sh both "zh en hard"
#
# Default eval metrics: zh/hard -> CER, en -> WER, all requested sets -> MOS (UTMOS + DNSMOS). Add sim via EVAL_METRICS.
# Env knobs: LIMIT=0 (0=full set), BASE=1024 (per-item seed base), EVAL_METRICS=...
# Outputs: gen/paired/<tag>/<set>/*.wav ; results/<result_prefix>_*.txt
# Required data/ckpt: same as benchmark_fp32_seedtts.sh (bash scripts/download_seedtts_testset.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"
cd "$SEED_REPRO_DIR"

which_model="${1:-both}"
sets_csv="${2:-zh,en,hard}"
limit="${LIMIT:-0}"
base="${BASE:-1024}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer mos}}"

run_one() {
  local label="$1" model_dir="$2" tag="$3" result_prefix="$4"

  echo "=== [int8/$label] generate sets=$sets_csv tag=$tag ==="
  "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
    --mode int8 \
    --tag "$tag" \
    --base "$base" \
    --sets "$sets_csv" \
    --limit "$limit" \
    --model_dir "$model_dir" \
    --device "$DEVICE"

  echo "=== [int8/$label] evaluate ==="
  bash scripts/evaluate_seedtts_metrics.sh "gen/paired/$tag" "$result_prefix" "${sets_csv//,/ }" "$eval_metrics"
}

case "$which_model" in
  1b|1B)
    run_one 1B meituan-longcat/LongCat-AudioDiT-1B int8 pf_int8
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B int8_3.5b pf_int8_3.5b
    ;;
  both)
    run_one 1B meituan-longcat/LongCat-AudioDiT-1B int8 pf_int8
    run_one 3.5B meituan-longcat/LongCat-AudioDiT-3.5B int8_3.5b pf_int8_3.5b
    ;;
  *)
    echo "usage: bash scripts/benchmark_int8_seedtts.sh [1b|3.5b|both] [zh,en,hard]" >&2
    exit 2
    ;;
esac
