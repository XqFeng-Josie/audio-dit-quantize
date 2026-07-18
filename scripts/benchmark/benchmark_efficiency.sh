#!/usr/bin/env bash
# Measure LongCat-AudioDiT W4A4 deploy efficiency (latency / RTF / VRAM) vs fp32 / fp16, and the deploy
# quality-latency tradeoff (fp16-glue vs fp32-glue). See docs/efficiency.md for the full design + results.
#
# ⚠️ LATENCY MEASUREMENT NEEDS AN IDLE GPU — run nothing else on the GPU (cudagraph timing is
#    contention-sensitive). Each config runs sequentially; output-validity guard flags degenerate (NaN/zero) runs.
#
# Usage:
#   bash scripts/benchmark/benchmark_efficiency.sh [1b|3.5b|both] [N]
# Examples:
#   bash scripts/benchmark/benchmark_efficiency.sh 1b           # 1B, N=10
#   bash scripts/benchmark/benchmark_efficiency.sh both 10      # 1B + 3.5B
#
# Configs per model (quant × execution mode):
#   fp32 / fp16 / W4A4-fp16glue  × {eager, cudagraph-only, compile+cudagraph}
#   W4A4-fp32glue (quality-recovered) × {compile+cudagraph}   (fused fp32-glue excluded: negative result + needs vendor-kron hp patch)
# Requires the built int4 kernels (see README: apply patch, then `python setup.py build_ext --inplace`).
#
# Outputs: results/eff/<model>_<config>.txt (full profiler output) + results/eff/progress.log
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/env.sh"

WHICH="${1:-1b}"; N="${2:-10}"
OUT="$SEED_RESULTS_DIR/eff"; mkdir -p "$OUT"; PROG="$OUT/progress.log"; : > "$PROG"
COMMON="--guidance_method apg --steps 16 --runs $N --warmup 2"
declare -A MODEL=( [1b]="meituan-longcat/LongCat-AudioDiT-1B" [3.5b]="meituan-longcat/LongCat-AudioDiT-3.5B" )
case "$WHICH" in 1b) MODELS=(1b);; 3.5b) MODELS=(3.5b);; both) MODELS=(1b 3.5b);; *) echo "usage: $0 [1b|3.5b|both] [N]"; exit 1;; esac

run () {  # $1=model-key $2=tag $3..=flags
  local mk=$1 tag=$2; shift 2
  echo "[$(date +%H:%M:%S)] START ${mk}_${tag}" | tee -a "$PROG"
  "$PYTHON_BIN" -m audio_dit_quantize.efficiency.profile_efficiency --model_dir "${MODEL[$mk]}" $COMMON "$@" \
      > "$OUT/${mk}_${tag}.txt" 2>&1 || true
  local med bad
  med=$(awk '/^medium/{print $4" ±"$5}' "$OUT/${mk}_${tag}.txt" | head -1)
  bad=$(grep -cE "INVALID|Traceback|Error" "$OUT/${mk}_${tag}.txt" || true)
  echo "[$(date +%H:%M:%S)] DONE  ${mk}_${tag}  medium=${med:-NA}  issues=$bad" | tee -a "$PROG"
}

for mk in "${MODELS[@]}"; do
  run $mk fp32_eager            --precision fp32
  run $mk fp32_cgonly           --precision fp32 --cudagraph
  run $mk fp32_compilecg        --precision fp32 --compile --inductor-cudagraph
  run $mk fp16_eager            --fp16
  run $mk fp16_cgonly           --fp16 --cudagraph
  run $mk fp16_compilecg        --fp16 --compile --inductor-cudagraph
  run $mk w4a4fp16g_eager       --fp16 --w4a4-deploy
  run $mk w4a4fp16g_cgonly      --fp16 --w4a4-deploy --cudagraph
  run $mk w4a4fp16g_compilecg   --fp16 --w4a4-deploy --compile --inductor-cudagraph
  run $mk w4a4fp32g_compilecg   --fp16 --w4a4-hp-deploy --compile --inductor-cudagraph
  # NOTE: the FUSED fp32-glue variant (--w4a4-hp-fused-deploy) is a measured-NEGATIVE result (fusing the fp32
  # Triton kron made it ~2x SLOWER than unfused, so it's not in the §2A table) AND it needs a vendor-kron
  # modification (kronecker_matmul's `hp=True` flag in online_trans.py + kron_matmul.py) that pristine FlatQuant
  # lacks. Left out of the default run. To reproduce it, patch the vendor kron, then add:
  #   run $mk w4a4fp32g_fused_cg  --fp16 --w4a4-hp-fused-deploy --compile --inductor-cudagraph
done
echo "[$(date +%H:%M:%S)] ALL DONE — see $OUT/ and docs/efficiency.md" | tee -a "$PROG"
