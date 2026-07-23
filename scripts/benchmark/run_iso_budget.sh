#!/usr/bin/env bash
# Iso-budget activation-bit reallocation (docs/results-consolidated.md §4): sharded gen + eval.
#
# Usage:
#   bash scripts/benchmark/run_iso_budget.sh [1b|3.5b] [configs_csv]
#     configs_csv default: iso_late,iso_early,iso_ffn,iso_attn
#   Knobs (env): GPUS="0,1,2,3" GPU pool (scripts/gpu_parallel.sh); LIMIT=N items/set;
#                SETS_CSV=zh,en,hard; EVAL=0 to skip metrics (e.g. smoke runs).
#
# Sharding: run_gen_parallel splits each set into contiguous item ranges (one per GPU); per-item
# seed = base + GLOBAL index, so sharded output is byte-identical to a single-GPU run. Each shard
# worker loads the fixed bc model once and runs ALL configs over its item range (resume-safe:
# existing wavs are skipped).
#
# Smoke first (identity + A3-collapse probe, no eval):
#   LIMIT=2 EVAL=0 bash scripts/benchmark/run_iso_budget.sh 1b uniform4,iso_late,iso_early,iso_ffn,iso_attn
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
configs_csv="${2:-iso_late,iso_early,iso_ffn,iso_attn}"
sets_csv="${SETS_CSV:-zh,en,hard}"
limit="${LIMIT:-0}"
do_eval="${EVAL:-1}"

case "$which_model" in
  1b)   model_dir="meituan-longcat/LongCat-AudioDiT-1B";   sub="iso_budget";      sfx="" ;;
  3.5b) model_dir="meituan-longcat/LongCat-AudioDiT-3.5B"; sub="iso_budget_3.5b"; sfx="_3.5b" ;;
  *) echo "usage: $0 [1b|3.5b] [configs_csv]"; exit 1 ;;
esac

gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>
  "$PYTHON_BIN" -m audio_dit_quantize.generate_iso_budget \
    --model_dir "$model_dir" --out_subdir "$sub" \
    --configs "$configs_csv" --sets "$1" --offset "$2" --limit "$3" \
    --device "$DEVICE"
}
echo "=== [iso/$which_model] generate configs=$configs_csv sets=$sets_csv ==="
run_gen_parallel gen_cb "${sets_csv//,/ }" "$limit"

if [ "$do_eval" = "1" ]; then
  for c in ${configs_csv//,/ }; do
    echo "=== [iso/$which_model] evaluate $c ==="
    bash scripts/evaluate_seedtts_metrics.sh "gen/$sub/$c" "${c}${sfx}" "${sets_csv//,/ }"
  done
fi
