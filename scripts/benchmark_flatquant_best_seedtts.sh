#!/usr/bin/env bash
# FlatQuant best-config quality run:
# per-block reconstruction + LWC + LAC + add_diag + Kronecker, hardlike32 calib.
#
# What this script does:
#   1. Reuse (or, if absent, calibrate-once-and-save) the ONE canonical best-config model
#      models/bc_{1b,3p5b}_model.pt so its W4A4 numbers match the step-axis 'full' baseline exactly.
#   2. Generate Seed-TTS wavs with that W4A4 model.
#   3. Automatically evaluate generated wavs with scripts/evaluate_seedtts_metrics.sh.
#
# Usage:
#   bash scripts/benchmark_flatquant_best_seedtts.sh [1b|3.5b|both] [zh,en,hard]
# Examples:
#   bash scripts/benchmark_flatquant_best_seedtts.sh 1b "zh en hard"
#   LIMIT=1 STEPS=20 bash scripts/benchmark_flatquant_best_seedtts.sh 1b hard
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark_flatquant_best_seedtts.sh 1b "zh en hard"
#
# Default eval metrics: zh/hard -> CER, en -> WER; plus MOS (UTMOS + DNSMOS) + WavLM SIM on all sets.
#   SIM is on by default so this run doubles as the step-axis paired-ΔSIM baseline (flat_best per-item SIM).
#   (SIM needs eval/ckpt/wavlm_large_finetune.pth; override with e.g. EVAL_METRICS="wer cer mos".)
#
# Common env knobs:
#   LOSS=mse             per-block reconstruction loss: mse (canonical) or chanbal (ablation)
#   CALIB_SEED=0         calibration RNG seed
#   BASE=1024            per-item generation seed base
#   MAX_SEQS=64          captured calibration sequences
#   PER_ITEM_KEEP=2      kept denoising states per calibration item
#   STEPS=200            optimization steps per block
#   MB=4                 calibration minibatch size
#   LIMIT=0              items per set; 0 means full set
#
# Naming convention (aligned across all quality benchmarks):
#   gen wavs -> gen/paired/<tag>/<set>/*.wav ; metrics -> results/<tag>_<set>_<metric>.txt
#   gen tag == metric prefix; 3.5B adds the _3.5b suffix (tag=flat_best / flat_best_3.5b).
#
# Canonical-model consistency (default best-config, LOSS=mse):
#   - models/bc_*.pt present -> --load_model (bit-identical to step-axis 'full')
#   - models/bc_*.pt absent  -> --save_model (calibrate once with the same recipe and become its producer)
#   Ablations (LOSS!=mse) are a different recipe: calibrate fresh and DO NOT touch the canonical model.
#
# Required data:
#   data/seedtts_testset/zh/meta.lst and data/seedtts_testset/en/meta.lst
#   eval/ckpt/wavlm_large_finetune.pth when running SIM
#   Install with: bash scripts/download_seedtts_testset.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"   # GPU-range knob + item-shard fan-out (calibration stays single-GPU)
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
sets_csv="${2:-zh,en,hard}"
loss="${LOSS:-mse}"
limit="${LIMIT:-0}"
base="${BASE:-1024}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer mos sim}}"

run_model() {
  local label="$1" model_dir="$2" tag="$3" model_pt="$4"   # tag = gen subdir under paired/ AND metric prefix
  local out_subdir="paired/$tag"
  local eval_sets="${sets_csv//,/ }"
  [ "$sets_csv" = "all" ] && eval_sets="zh en hard"

  # First GPU of the range is used for the (single-GPU) calibration.
  local pool; read -r -a pool <<< "$(gpu_pool)"
  local cgpu=""; [ "${#pool[@]}" -ge 1 ] && cgpu="${pool[0]}"

  if [ "$loss" = "mse" ]; then
    # Default best-config: ONE canonical models/bc_*.pt shared with step-axis 'full'. Calibrate it once
    # on a single GPU if absent, then fan out GENERATION across the whole GPU pool with --load_model.
    if [ ! -f "$model_pt" ]; then
      echo "[FlatQuant best/$label] canonical model $model_pt absent -> calibrate once (single-GPU${cgpu:+ =$cgpu}) + save"
      CUDA_VISIBLE_DEVICES="${cgpu:-${CUDA_VISIBLE_DEVICES:-0}}" "$PYTHON_BIN" -m audio_dit_quantize.flatquant_best \
        --model_dir "$model_dir" --model "$model_pt" --save_model --calibrate_only \
        --calib_seed "${CALIB_SEED:-0}" --max_seqs "${MAX_SEQS:-64}" --per_item_keep "${PER_ITEM_KEEP:-2}" \
        --steps "${STEPS:-200}" --mb "${MB:-4}" --loss mse --device cuda:0
    else
      echo "[FlatQuant best/$label] reusing canonical model $model_pt (matches step-axis 'full')"
    fi
    gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>  — load the canonical model, generate a shard
      "$PYTHON_BIN" -m audio_dit_quantize.flatquant_best \
        --model_dir "$model_dir" --model "$model_pt" --load_model \
        --out_subdir "$out_subdir" --sets "$1" --offset "$2" --limit "$3" \
        --base "$base" --device "$DEVICE"
    }
    echo "=== [FlatQuant best/$label] generate out=$out_subdir sets=$sets_csv (sharded) ==="
    run_gen_parallel gen_cb "$eval_sets" "$limit"
  else
    # Ablation (loss!=mse): a different recipe, not the canonical model. Item sharding would recalibrate
    # per shard (different draws), so calibrate + generate in ONE single-GPU process.
    echo "=== [FlatQuant best/$label] ablation loss=$loss: fresh calibration + gen, single-GPU${cgpu:+ =$cgpu} ==="
    CUDA_VISIBLE_DEVICES="${cgpu:-${CUDA_VISIBLE_DEVICES:-0}}" "$PYTHON_BIN" -m audio_dit_quantize.flatquant_best \
      --model_dir "$model_dir" --out_subdir "$out_subdir" --sets "$sets_csv" --limit "$limit" \
      --calib_seed "${CALIB_SEED:-0}" --base "$base" --max_seqs "${MAX_SEQS:-64}" \
      --per_item_keep "${PER_ITEM_KEEP:-2}" --steps "${STEPS:-200}" --mb "${MB:-4}" --loss "$loss" --device cuda:0
  fi

  echo "=== [FlatQuant best/$label] evaluate ==="
  bash scripts/evaluate_seedtts_metrics.sh "gen/$out_subdir" "$tag" "$eval_sets" "$eval_metrics"
}

case "$which_model" in
  1b|1B)
    run_model 1B meituan-longcat/LongCat-AudioDiT-1B flat_best "$SEED_MODELS_DIR/bc_1b_model.pt"
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_model 3.5B meituan-longcat/LongCat-AudioDiT-3.5B flat_best_3.5b "$SEED_MODELS_DIR/bc_3p5b_model.pt"
    ;;
  both)
    run_model 1B   meituan-longcat/LongCat-AudioDiT-1B   flat_best      "$SEED_MODELS_DIR/bc_1b_model.pt"
    run_model 3.5B meituan-longcat/LongCat-AudioDiT-3.5B flat_best_3.5b "$SEED_MODELS_DIR/bc_3p5b_model.pt"
    ;;
  *)
    echo "usage: bash scripts/benchmark_flatquant_best_seedtts.sh [1b|3.5b|both] [zh,en,hard]" >&2
    exit 2
    ;;
esac
