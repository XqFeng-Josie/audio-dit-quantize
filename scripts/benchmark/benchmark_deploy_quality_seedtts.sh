#!/usr/bin/env bash
# Deploy-quality recovery experiment (§2B): real W4A4 deploy layer, fp16-glue (fast, degrades) vs
# fp32-glue (recovers ~fake-quant quality). Chains gen (w4a4_deploy_quality) + eval (evaluate_seedtts_metrics.sh)
# into one command, matching benchmark_{fp32,int8,step_axis}_seedtts.sh. Both models, any of zh/en/hard.
#
# What each glue means (the knob = int4-GEMM's surrounding Kron transform + per-token scale + dequant):
#   --glue fp32  = 准档: fp32 glue reproduces fake-quant quality (the shippable, quality-recovered path)
#   --glue fp16  = 快档: fp16 glue is fast but degrades (1B-Hard ~13.3% WER, 3.5B collapses) — the control
# int4 GEMM itself is integer-exact; quality is decided entirely by glue precision.
#
# Usage:
#   bash scripts/benchmark/benchmark_deploy_quality_seedtts.sh [1b|3.5b|both] [zh,en,hard] [fp32,fp16]
# Examples:
#   bash scripts/benchmark/benchmark_deploy_quality_seedtts.sh both hard              # both models, Hard, both glues
#   bash scripts/benchmark/benchmark_deploy_quality_seedtts.sh 1b  "zh en hard"       # 1B, full-set
#   bash scripts/benchmark/benchmark_deploy_quality_seedtts.sh 3.5b hard fp32         # 3.5B, Hard, fp32-glue only (准档)
#   LIMIT=8 bash scripts/benchmark/benchmark_deploy_quality_seedtts.sh 1b hard        # quick smoke
#
# Env knobs:
#   LIMIT=0            items per set (0 = full set)
#   BASE=1024          per-item generation seed base
#   EVAL_METRICS="wer cer mos sim"  en->WER, zh/hard->CER, all->MOS(UTMOS+DNSMOS)+WavLM SIM (needs wavlm ckpt)
#   MODEL=path         override the fixed best-config .pt (single-model runs; default = models/bc_{1b,3p5b}_model.pt)
#
# Model selection: the DiT is loaded from the self-contained fixed best-config pickle
#   models/bc_1b_model.pt (1b) / models/bc_3p5b_model.pt (3.5b); the keyword picks it (override dir with SEED_MODELS_DIR).
#
# Naming convention (aligned across all quality benchmarks):
#   gen wavs -> gen/paired/<tag>/<set>/*.wav ; metrics -> results/<tag>_<set>_<metric>.txt
#   tag = dep_<glue> (1B) / dep_3.5b_<glue> (3.5B), e.g. dep_fp32, dep_3.5b_fp16.
# Required data/ckpt: same as benchmark_fp32_seedtts.sh (bash scripts/setup/download_seedtts_testset.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"   # GPU-range knob + item-shard fan-out (loads a fixed bc_*.pt)
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
sets_csv="${2:-zh,en,hard}"
glues="${3:-fp32 fp16}"; glues="${glues//,/ }"
limit="${LIMIT:-0}"
base="${BASE:-1024}"
eval_metrics="${EVAL_METRICS:-${METRICS:-wer cer mos sim}}"

run_model() {
  local label="$1" model_dir="$2" mtok="$3"       # mtok: "" (1B) | "3.5b"
  local model_arg=(--model_dir "$model_dir")
  [ -n "${MODEL:-}" ] && model_arg=(--model "$MODEL")

  # Deploy loads a self-contained fixed bc_*.pt (deterministic wrap), so generation shards cleanly.
  for glue in $glues; do
    local tag="dep${mtok:+_$mtok}_${glue}"         # dep_fp32 / dep_3.5b_fp16

    gen_cb() {   # cb <sets_csv> <offset> <limit> <gpu>  — w4a4_deploy_quality is single-set, so loop the csv
      local s
      for s in ${1//,/ }; do
        "$PYTHON_BIN" -m audio_dit_quantize.efficiency.w4a4_deploy_quality \
          "${model_arg[@]}" --set "$s" --glue "$glue" --base "$base" \
          --offset "$2" --limit "$3" --out_subdir "paired/$tag/$s" --device "$DEVICE"
      done
    }
    echo "=== [deploy/$label glue=$glue] generate sets=$sets_csv tag=$tag (sharded) ==="
    run_gen_parallel gen_cb "${sets_csv//,/ }" "$limit"

    echo "=== [deploy/$label glue=$glue] evaluate ==="
    bash scripts/evaluate_seedtts_metrics.sh "gen/paired/$tag" "$tag" "${sets_csv//,/ }" "$eval_metrics"
  done
}

case "$which_model" in
  1b|1B)
    run_model 1B meituan-longcat/LongCat-AudioDiT-1B ""
    ;;
  3.5b|3.5B|3p5b|3P5B)
    run_model 3.5B meituan-longcat/LongCat-AudioDiT-3.5B 3.5b
    ;;
  both)
    run_model 1B   meituan-longcat/LongCat-AudioDiT-1B   ""
    run_model 3.5B meituan-longcat/LongCat-AudioDiT-3.5B 3.5b
    ;;
  *)
    echo "usage: bash scripts/benchmark/benchmark_deploy_quality_seedtts.sh [1b|3.5b|both] [zh,en,hard] [fp32,fp16]" >&2
    exit 2
    ;;
esac
