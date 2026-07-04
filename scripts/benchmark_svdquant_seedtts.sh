#!/usr/bin/env bash
# SVDQuant W4A4 quality run under the current order-free per-item-seeded protocol.
#
# What this script does:
#   1. Calibrate SVDQuant with fixed data/calib_heldout_hardlike32.lst.
#   2. Generate Seed-TTS wavs with the calibrated W4A4 model.
#   3. Automatically evaluate generated wavs with scripts/evaluate_seedtts_metrics.sh.
#
# Usage:
#   bash scripts/benchmark_svdquant_seedtts.sh [1b|3.5b] [zh,en,hard]
# Examples:
#   bash scripts/benchmark_svdquant_seedtts.sh 1b "zh en hard"
#   LIMIT=1 RANK=32 bash scripts/benchmark_svdquant_seedtts.sh 1b hard
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark_svdquant_seedtts.sh 1b "zh en hard"
#
# Default eval metrics:
#   zh/hard -> CER, en -> WER, all requested sets -> MOS (UTMOS + DNSMOS).
#   Add SIM with EVAL_METRICS="wer cer mos sim".
#
# Common env knobs:
#   RANK=32             SVD low-rank residual rank
#   BASE=1024           per-item generation seed base
#   LIMIT=0             items per set; 0 means full set
#   TAG=svd             generated wav tag under gen/paired/
#   RESULT_PREFIX=...   metric filename prefix under results/
#   EVAL_METRICS=...    metrics passed to evaluate_seedtts_metrics.sh
#
# Outputs:
#   gen/paired/<tag>/<set>/*.wav
#   results/<result_prefix>_*.txt
#
# Required data:
#   data/seedtts_testset/zh/meta.lst and data/seedtts_testset/en/meta.lst
#   eval/ckpt/wavlm_large_finetune.pth when running SIM
#   Install with: bash scripts/download_seedtts_testset.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
sets_csv="${2:-zh,en,hard}"
limit="${LIMIT:-0}"
base="${BASE:-1024}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer mos}}"

case "$which_model" in
  1b|1B)
    model_dir=meituan-longcat/LongCat-AudioDiT-1B
    tag="${TAG:-svd}"
    result_prefix="${RESULT_PREFIX:-pf_svdquant}"
    ;;
  3.5b|3.5B|3p5b|3P5B)
    model_dir=meituan-longcat/LongCat-AudioDiT-3.5B
    tag="${TAG:-svd_3.5b}"
    result_prefix="${RESULT_PREFIX:-pf_svdquant_3.5b}"
    ;;
  *)
    echo "usage: bash scripts/benchmark_svdquant_seedtts.sh [1b|3.5b] [zh,en,hard]" >&2
    exit 2
    ;;
esac

echo "=== [SVDQuant/$which_model] generate sets=$sets_csv tag=$tag ==="
"$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts \
  --mode svdquant \
  --tag "$tag" \
  --base "$base" \
  --sets "$sets_csv" \
  --limit "$limit" \
  --model_dir "$model_dir" \
  --rank "${RANK:-32}" \
  --device "$DEVICE"

echo "=== [SVDQuant/$which_model] evaluate ==="
bash scripts/evaluate_seedtts_metrics.sh "gen/paired/$tag" "$result_prefix" "${sets_csv//,/ }" "$eval_metrics"
