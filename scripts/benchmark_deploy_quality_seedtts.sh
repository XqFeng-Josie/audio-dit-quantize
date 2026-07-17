#!/usr/bin/env bash
# Deploy-quality recovery experiment (§2B): real W4A4 deploy layer, fp16-glue (fast, degrades) vs
# fp32-glue (recovers ~fake-quant quality). Chains gen (w4a4_deploy_quality) + eval (evaluate_seedtts_metrics.sh)
# into one command, matching benchmark_{fp32,int8,step_axis}_seedtts.sh. Both models, any of zh/en/hard.
#
# What each glue means (the knob = int4-GEMM's surrounding Kron transform + per-token scale + dequant):
#   --glue fp32  = 准档: fp32 glue reproduces fake-quant quality (the shippable, quality-recovered path)
#   --glue fp16  = 快档: fp16 glue is fast but degrades (1B-Hard ~13.3% WER, 3.5B collapses) — the control
# int4 GEMM itself is integer-exact; quality is decided entirely by glue precision (see docs/efficiency.md §2B).
#
# Usage:
#   bash scripts/benchmark_deploy_quality_seedtts.sh [1b|3.5b|both] [zh,en,hard] [fp32,fp16]
# Examples:
#   bash scripts/benchmark_deploy_quality_seedtts.sh both hard              # §2B canonical: both models, Hard, both glues
#   bash scripts/benchmark_deploy_quality_seedtts.sh 1b  "zh en hard"       # 1B, full-set
#   bash scripts/benchmark_deploy_quality_seedtts.sh 3.5b hard fp32         # 3.5B, Hard, fp32-glue only (准档)
#   LIMIT=8 bash scripts/benchmark_deploy_quality_seedtts.sh 1b hard        # quick smoke
#
# Env knobs:
#   LIMIT=0            items per set (0 = full set)
#   BASE=1024          per-item generation seed base
#   EVAL_METRICS="wer cer sim"   en->WER, zh/hard->CER, all->SIM (deploy §2B is CER/SIM; no MOS by default)
#   MODEL=path         override the fixed best-config .pt (single-model runs; default = models/bc_{1b,3p5b}_model.pt)
#
# Model selection: the DiT is loaded from the self-contained fixed best-config pickle
#   models/bc_1b_model.pt (1b) / models/bc_3p5b_model.pt (3.5b); the keyword picks it (override dir with SEED_MODELS_DIR).
#
# Outputs:
#   gen/dep_<tag>_<glue>/<set>/*.wav
#   results/dep_<tag>_<glue>_<set>_<metric>.txt        (tag = bc for 1B, bc35 for 3.5B)
# Required data/ckpt: same as benchmark_fp32_seedtts.sh (bash scripts/download_seedtts_testset.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"
cd "$SEED_REPRO_DIR"

which_model="${1:-both}"
sets_csv="${2:-zh,en,hard}"
glues="${3:-fp32 fp16}"; glues="${glues//,/ }"
limit="${LIMIT:-0}"
base="${BASE:-1024}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer sim}}"

run_model() {
  local label="$1" model_dir="$2" tag="$3"        # tag: bc (1B) | bc35 (3.5B)
  local model_arg=(--model_dir "$model_dir")
  [ -n "${MODEL:-}" ] && model_arg=(--model "$MODEL")

  for glue in $glues; do
    local genroot="dep_${tag}_${glue}"            # dep_bc_fp32 / dep_bc35_fp16

    echo "=== [deploy/$label glue=$glue] generate sets=$sets_csv ==="
    for s in ${sets_csv//,/ }; do
      "$PYTHON_BIN" -m audio_dit_quantize.efficiency.w4a4_deploy_quality \
        "${model_arg[@]}" \
        --set "$s" \
        --glue "$glue" \
        --base "$base" \
        --limit "$limit" \
        --out_subdir "$genroot/$s" \
        --device "$DEVICE"
    done

    echo "=== [deploy/$label glue=$glue] evaluate ==="
    bash scripts/evaluate_seedtts_metrics.sh "gen/$genroot" "$genroot" "${sets_csv//,/ }" "$eval_metrics"
  done
}

case "$which_model" in
  1b|1B)
    run_model 1B meituan-longcat/LongCat-AudioDiT-1B bc
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_model 3.5B meituan-longcat/LongCat-AudioDiT-3.5B bc35
    ;;
  both)
    run_model 1B   meituan-longcat/LongCat-AudioDiT-1B   bc
    run_model 3.5B meituan-longcat/LongCat-AudioDiT-3.5B bc35
    ;;
  *)
    echo "usage: bash scripts/benchmark_deploy_quality_seedtts.sh [1b|3.5b|both] [zh,en,hard] [fp32,fp16]" >&2
    exit 2
    ;;
esac
