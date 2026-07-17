#!/usr/bin/env bash
# Step-axis activation-precision experiment: full / early / late on the FIXED best-config W4A4 model.
# Same model + same per-item seed generate each set three ways, gating ONLY which ODE steps quantize
# activations (weights stay int4). Then eval WER/CER/SIM/MOS and run the per-item paired ΔSIM bootstrap
# (early|late vs full) — the key result: LATE recovers timbre/SIM, EARLY (equal budget) does not.
# See docs/quality-metrics-matrix.md.
#
# ONE fixed calibration model (models/bc_{1b,3p5b}_model.pt) is LOADED so full/early/late share the same
# calibration — that identity is the whole point of the control. Produce it once with MODE=calibrate
# (or copy an existing bc_*.pt into models/). Quality gens are contention-tolerant (unlike latency).
#
# Usage:
#   bash scripts/benchmark_step_axis_seedtts.sh [1b|3.5b] ["zh en hard"] ["full early late"]
#   MODE=calibrate bash scripts/benchmark_step_axis_seedtts.sh 1b        # one-time: produce the fixed model
#   LIMIT=50 bash scripts/benchmark_step_axis_seedtts.sh 1b hard "full late"   # quick smoke
# Env: LIMIT=0 (items/set; 0=full)  BASE=1024 (per-item seed)  DEVICE=cuda:0
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/env.sh"
cd "$ROOT_DIR"

WHICH="${1:-1b}"; SETS="${2:-zh en hard}"; CONFIGS="${3:-full early late}"
LIMIT="${LIMIT:-0}"; BASE="${BASE:-1024}"
declare -A MODEL=( [1b]="meituan-longcat/LongCat-AudioDiT-1B" [3.5b]="meituan-longcat/LongCat-AudioDiT-3.5B" )
MD="${MODEL[$WHICH]:?usage: $0 [1b|3.5b] [\"zh en hard\"] [\"full early late\"]}"
SETSCSV=$(echo "$SETS" | tr -s ' ' ','); CSV=$(echo "$CONFIGS" | tr -s ' ' ',')
G=("$PYTHON_BIN" -m audio_dit_quantize.generate_step_axis --model_dir "$MD" --base "$BASE")

# one-time: produce the fixed best-config calibration model (calib_seed 0, CALIB_LST, asym+LAC+add_diag)
if [ "${MODE:-}" = "calibrate" ]; then
  echo "[step-axis] calibrating the FIXED best-config model (one-time) ..."
  "${G[@]}" --calibrate
fi

echo "[step-axis] generating {$CONFIGS} x {$SETS} from the ONE fixed model (base=$BASE, limit=$LIMIT) ..."
"${G[@]}" --sets "$SETSCSV" --configs "$CSV" --limit "$LIMIT"

for c in $CONFIGS; do
  echo "[step-axis] eval $c (wer cer sim mos) ..."
  bash "$ROOT_DIR/scripts/evaluate_seedtts_metrics.sh" "$SEED_GEN_DIR/step_axis/$c" "step_$c" "$SETS" "wer cer mos sim"
done

echo "[step-axis] ===== paired ΔSIM vs full (per-item bootstrap; the key result) ====="
for c in $CONFIGS; do
  [ "$c" = "full" ] && continue
  for s in $SETS; do
    A="$SEED_RESULTS_DIR/step_full_${s}_sim.txt"; B="$SEED_RESULTS_DIR/step_${c}_${s}_sim.txt"
    if [ -f "$A" ] && [ -f "$B" ]; then
      "$PYTHON_BIN" -m audio_dit_quantize.paired_bootstrap --metric sim --a "$A" --b "$B" --labels full "$c" \
        2>/dev/null | sed "s/^/  [$s: $c vs full] /"
    fi
  done
done
echo "[step-axis] DONE — matrix in docs/quality-metrics-matrix.md"
