#!/usr/bin/env bash
# FlatQuant best-config quality run:
# per-block reconstruction + LWC + LAC + add_diag + Kronecker, hardlike32 calib.
#
# What this script does:
#   1. Calibrate FlatQuant best-config with fixed data/calib_heldout_hardlike32.lst.
#   2. Generate Seed-TTS wavs with the calibrated W4A4 model.
#   3. Automatically evaluate generated wavs with scripts/evaluate_seedtts_metrics.sh.
#
# Usage:
#   bash scripts/benchmark_flatquant_best_seedtts.sh [1b|3.5b] [zh,en,hard]
# Examples:
#   bash scripts/benchmark_flatquant_best_seedtts.sh 1b "zh en hard"
#   LIMIT=1 STEPS=20 bash scripts/benchmark_flatquant_best_seedtts.sh 1b hard
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark_flatquant_best_seedtts.sh 1b "zh en hard"
#
# Default eval metrics:
#   zh/hard -> CER, en -> WER, all requested sets -> MOS (UTMOS + DNSMOS).
#   Add SIM with EVAL_METRICS="wer cer mos sim".
#
# Common env knobs:
#   LOSS=mse             per-block reconstruction loss: mse or chanbal
#   CALIB_SEED=0         calibration RNG seed
#   BASE=1024            per-item generation seed base
#   MAX_SEQS=64          captured calibration sequences
#   PER_ITEM_KEEP=2      kept denoising states per calibration item
#   STEPS=200            optimization steps per block
#   MB=4                 calibration minibatch size
#   LIMIT=0              items per set; 0 means full set
#   OUT_SUBDIR=flat_best generated wav subdir under gen/
#   RESULT_PREFIX=...    metric filename prefix under results/
#
# Outputs:
#   gen/<out_subdir>/<set>/*.wav
#   results/<result_prefix>_*.txt
#
# Required data:
#   data/seedtts_testset/zh/meta.lst and data/seedtts_testset/en/meta.lst
#   Install with: bash scripts/download_seedtts_testset.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
sets_csv="${2:-zh,en,hard}"
loss="${LOSS:-mse}"
limit="${LIMIT:-0}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer mos}}"

case "$which_model" in
  1b|1B)
    model_dir=meituan-longcat/LongCat-AudioDiT-1B
    out_subdir="${OUT_SUBDIR:-flat_best}"
    result_prefix="${RESULT_PREFIX:-pf_flat_best}"
    ;;
  3.5b|3.5B|3p5b|3P5B)
    model_dir=meituan-longcat/LongCat-AudioDiT-3.5B
    out_subdir="${OUT_SUBDIR:-flat_best_3.5b}"
    result_prefix="${RESULT_PREFIX:-pf_flat_best_3.5b}"
    ;;
  *)
    echo "usage: bash scripts/benchmark_flatquant_best_seedtts.sh [1b|3.5b] [zh,en,hard]" >&2
    exit 2
    ;;
esac

echo "=== [FlatQuant best/$which_model] generate out=$out_subdir sets=$sets_csv loss=$loss ==="
"$PYTHON_BIN" -m audio_dit_quantize.flatquant_best \
  --model_dir "$model_dir" \
  --out_subdir "$out_subdir" \
  --sets "$sets_csv" \
  --limit "$limit" \
  --calib_seed "${CALIB_SEED:-0}" \
  --base "${BASE:-1024}" \
  --max_seqs "${MAX_SEQS:-64}" \
  --per_item_keep "${PER_ITEM_KEEP:-2}" \
  --steps "${STEPS:-200}" \
  --mb "${MB:-4}" \
  --loss "$loss" \
  --device "$DEVICE"

eval_sets="${sets_csv//,/ }"
[ "$sets_csv" = "all" ] && eval_sets="zh en hard"

echo "=== [FlatQuant best/$which_model] evaluate ==="
bash scripts/evaluate_seedtts_metrics.sh "gen/$out_subdir" "$result_prefix" "$eval_sets" "$eval_metrics"
