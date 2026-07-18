#!/usr/bin/env bash
# Step-axis activation-precision experiment: early / late vs the W4A4 best-config baseline.
# The SAME fixed best-config W4A4 model generates each set, gating ONLY which ODE steps quantize
# activations (weights stay int4). Then eval WER/CER/SIM/MOS and run the per-item paired ΔSIM bootstrap
# (early|late vs baseline) — the key result: LATE recovers timbre/SIM, EARLY (equal budget) does not.
#
# The BASELINE is the standard W4A4 best-config = "full" (all 15 steps quantize activations). It is
# already produced by benchmark_flatquant_best_seedtts.sh, which (after the F2 fix) LOADS the same
# canonical models/bc_*.pt with the same base=1024 per-item seeds — i.e. it IS "full". So this script no
# longer regenerates full by default (that was a redundant full-set gen pass); it defaults to {early,late}
# and compares them against the flat_best per-item SIM. `full` remains a valid config if passed explicitly.
#
# ONE fixed calibration model (models/bc_{1b,3p5b}_model.pt) is LOADED so every config shares the same
# calibration — that identity is the whole point of the control. Produce it once with MODE=calibrate.
#
# REQUIRED before the ΔSIM bootstrap: run flatquant_best WITH per-item SIM to make the baseline, e.g.
#   EVAL_METRICS="wer cer mos sim" bash scripts/benchmark/benchmark_flatquant_best_seedtts.sh 1b
# (override the baseline metric prefix with BASELINE_PREFIX; default flat_best / flat_best_3.5b).
#
# Usage:
#   bash scripts/benchmark/benchmark_step_axis_seedtts.sh [1b|3.5b|both] ["zh en hard"] ["early late"]
#   MODE=calibrate bash scripts/benchmark/benchmark_step_axis_seedtts.sh 1b        # one-time: produce the fixed model
#   LIMIT=50 bash scripts/benchmark/benchmark_step_axis_seedtts.sh 1b hard "late"  # quick smoke
# Env: LIMIT=0 (items/set; 0=full)  BASE=1024 (per-item seed)  DEVICE=cuda:0  MODE=calibrate (produce model)
#      BASELINE_PREFIX=flat_best  (metric prefix of the "full" W4A4 baseline; _3.5b auto-appended for 3.5B)
#
# Naming convention (aligned across all quality benchmarks):
#   gen wavs -> gen/paired/step_axis[_3.5b]/<cfg>/<set>/*.wav
#   metrics  -> results/step_<cfg>[_3.5b]_<set>_<metric>.txt   (1B untagged; 3.5B adds _3.5b)
# Model-scoped paths keep 1B and 3.5B from colliding (skip-existing would otherwise make the second
# model reuse the first model's cached wavs, and result files would overwrite each other).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"   # GPU-range knob + item-shard fan-out (calibration stays single-GPU)
cd "$ROOT_DIR"

WHICH="${1:-1b}"; SETS="${2:-zh en hard}"; CONFIGS="${3:-early late}"
LIMIT="${LIMIT:-0}"; BASE="${BASE:-1024}"
declare -A MODEL=( [1b]="meituan-longcat/LongCat-AudioDiT-1B" [3.5b]="meituan-longcat/LongCat-AudioDiT-3.5B" )
SETSCSV=$(echo "$SETS" | tr -s ' ' ','); CSV=$(echo "$CONFIGS" | tr -s ' ' ',')

run_model() {
  local which="$1" md="$2"
  local GENSUB TAG
  case "$which" in
    3.5b) GENSUB="paired/step_axis_3.5b"; TAG="_3.5b" ;;
    *)    GENSUB="paired/step_axis";      TAG="" ;;
  esac
  local pool; read -r -a pool <<< "$(gpu_pool)"
  local cgpu=""; [ "${#pool[@]}" -ge 1 ] && cgpu="${pool[0]}"

  # one-time: produce the fixed best-config calibration model (calib_seed 0, CALIB_LST, asym+LAC+add_diag).
  # Single-GPU (calibration is not sharded); `--sets ""` = calibrate + save only, no generation.
  if [ "${MODE:-}" = "calibrate" ]; then
    echo "[step-axis/$which] calibrating the FIXED best-config model (one-time, single-GPU${cgpu:+ =$cgpu}) ..."
    CUDA_VISIBLE_DEVICES="${cgpu:-${CUDA_VISIBLE_DEVICES:-0}}" "$PYTHON_BIN" -m audio_dit_quantize.generate_step_axis \
      --model_dir "$md" --out_subdir "$GENSUB" --calibrate --sets "" --device cuda:0
  fi

  # generation: load the ONE fixed model, run all configs for each item shard, fanned across the GPU pool.
  gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>
    "$PYTHON_BIN" -m audio_dit_quantize.generate_step_axis \
      --model_dir "$md" --base "$BASE" --out_subdir "$GENSUB" \
      --sets "$1" --configs "$CSV" --offset "$2" --limit "$3" --device "$DEVICE"
  }
  echo "[step-axis/$which] generating {$CONFIGS} x {$SETS} from the ONE fixed model (base=$BASE, limit=$LIMIT) ..."
  run_gen_parallel gen_cb "$SETS" "$LIMIT"

  local c s A B
  for c in $CONFIGS; do
    echo "[step-axis/$which] eval $c (wer cer sim mos) ..."
    bash "$ROOT_DIR/scripts/evaluate_seedtts_metrics.sh" "$SEED_GEN_DIR/$GENSUB/$c" "step_${c}${TAG}" "$SETS" "wer cer mos sim"
  done

  # Baseline "full" = the W4A4 best-config, owned by benchmark_flatquant_best (same canonical bc_*.pt +
  # same base=1024 seeds => numerically the 'full' config). Its per-item SIM is the paired reference.
  local base_prefix="${BASELINE_PREFIX:-flat_best}${TAG}"
  echo "[step-axis/$which] ===== paired ΔSIM vs baseline '$base_prefix' (= W4A4 full; per-item bootstrap) ====="
  for c in $CONFIGS; do
    [ "$c" = "full" ] && continue
    for s in $SETS; do
      A="$SEED_RESULTS_DIR/${base_prefix}_${s}_sim.txt"; B="$SEED_RESULTS_DIR/step_${c}${TAG}_${s}_sim.txt"
      if [ ! -f "$A" ]; then
        echo "  [$which $s: $c] SKIP — baseline per-item SIM $A missing." >&2
        echo "        produce it with: EVAL_METRICS=\"wer cer mos sim\" bash scripts/benchmark/benchmark_flatquant_best_seedtts.sh $which" >&2
        continue
      fi
      if [ -f "$B" ]; then
        "$PYTHON_BIN" -m audio_dit_quantize.paired_bootstrap --metric sim --a "$A" --b "$B" --labels full "$c" \
          2>/dev/null | sed "s/^/  [$which $s: $c vs full] /"
      fi
    done
  done
}

case "$WHICH" in
  1b|1B)                 run_model 1b   "${MODEL[1b]}" ;;
  3.5b|3.5B|3p5b|3P5B)   run_model 3.5b "${MODEL[3.5b]}" ;;
  both)                  run_model 1b "${MODEL[1b]}"; run_model 3.5b "${MODEL[3.5b]}" ;;
  *) echo "usage: $0 [1b|3.5b|both] [\"zh en hard\"] [\"full early late\"]" >&2; exit 2 ;;
esac
echo "[step-axis] DONE"
