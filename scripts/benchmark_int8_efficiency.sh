#!/usr/bin/env bash
# Measure LongCat-AudioDiT REAL INT8 (torchao W8A8) deploy efficiency — the INT8 efficiency sub-story,
# kept SEPARATE from the W4A4 main line (scripts/benchmark_efficiency.sh). See docs/efficiency.md.
#
# The whole point is to isolate TWO effects that both move latency at bs=1:
#   (a) the COMPILE lever  — fp32-eager vs fp32-compile shows what torch.compile alone buys;
#   (b) the INT8-KERNEL gain — fp32-compile vs int8-compile, same compile, so any delta is the W8A8
#       tensor-core kernel (torchao pre-quantizes the DiT once -> fair, not per-run).
# torchao INT8's eager kernels are unfused/slow, so INT8 REQUIRES --compile (inductor fusion). The
# manual-graph path (--cudagraph) is the WRONG combo for INT8; the right cudagraph arm is
# --compile --inductor-cudagraph (needs the rotary keystone fix, applied automatically in the profiler).
#
# NOTE — the fp32/fp16 configs below OVERLAP scripts/benchmark_efficiency.sh ON PURPOSE, not redundancy:
# INT8 is a SEPARATE compile stack whose absolute ms must NOT be mixed with the §2A main table (doc §1.4,
# "ratio-comparable only, within-stack"). So the INT8 ratios (INT8 vs fp32, compile-gain vs kernel-gain)
# need fp32/fp16 DENOMINATORS measured in THIS SAME session / GPU state — you cannot borrow §2A's 677.
# That is why this script is self-contained rather than INT8-only.
#
# HEADLINE (scale-dependent — don't read this as "INT8 is always faster"):
#   1B  = launch-bound  -> COMPILE is the lever; INT8 itself is ~+25% SLOWER (negative). The cudagraph
#         arm (int8_compilecg) is what can pull 1B back once launch overhead is removed.
#   3.5B = compute-bound -> INT8 is a real 1.17–2.42x speedup + ~2.9x less VRAM (positive).
#
# ⚠️ LATENCY MEASUREMENT NEEDS AN IDLE GPU — run nothing else on it (cudagraph/compile timing is
#    contention-sensitive). Configs run sequentially; the output-validity guard flags NaN/zero runs.
#
# Usage:
#   bash scripts/benchmark_int8_efficiency.sh [1b|3.5b|both] [N]
# Examples:
#   bash scripts/benchmark_int8_efficiency.sh 3.5b        # 3.5B, N=10 (the model where INT8 wins)
#   bash scripts/benchmark_int8_efficiency.sh both 10     # full sub-story, 1B + 3.5B
#
# Configs per model (quant × execution mode):
#   fp32-eager                  (original baseline; anchors the compile gain)
#   fp32-compile                (compile-only reference; isolates the compile lever from the INT8 gain)
#   fp16-compile+inductor-cg    (best-mode fp16 anchor — where INT8 lands relative to fp16)
#   int8-compile                (real W8A8 tensor cores, no cudagraphs)
#   int8-compile+inductor-cg    (INT8 + launch-overhead removed — the arm that can turn 1B positive)
#
# Prereq: torchao installed in the venv (the W4A4 main line does NOT need it).
# Outputs: results/eff_int8/<model>_<config>.txt + results/eff_int8/progress.log
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/env.sh"

WHICH="${1:-3.5b}"; N="${2:-10}"
"$PYTHON_BIN" -c "import torchao" 2>/dev/null || {
  echo "ERROR: torchao not importable in $PYTHON_BIN — real W8A8 needs it. Install with: $PYTHON_BIN -m pip install torchao"
  echo "       (the W4A4 main line, scripts/benchmark_efficiency.sh, does NOT depend on torchao.)"; exit 1; }

OUT="$SEED_RESULTS_DIR/eff_int8"; mkdir -p "$OUT"; PROG="$OUT/progress.log"; : > "$PROG"
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
  # KNOWN LIMITATION: torchao-int8 ⊗ inductor-cudagraph is unsupported on the cu130/torch-2.12 stack —
  # torch's INT8 `_int_mm` needs M>16 but the bs=1 timestep/global projections have M=1, and this only
  # trips on the cudagraph partition path (so int8_compile / no-cg is fine). Documented in
  # docs/efficiency.md LIMITATIONS. Flag it as EXPECTED-FAIL (not a broken setup); kept in the run so a
  # future torch that lifts the M>16 limit is auto-detected here and yields real numbers.
  if [ -z "$med" ] && grep -qE "greater than 16|_int_mm" "$OUT/${mk}_${tag}.txt"; then
    echo "[$(date +%H:%M:%S)] DONE  ${mk}_${tag}  EXPECTED-FAIL (torchao-int8 ⊗ inductor-cudagraph unsupported on cu130 — see docs/efficiency.md LIMITATIONS)" | tee -a "$PROG"
  else
    echo "[$(date +%H:%M:%S)] DONE  ${mk}_${tag}  medium=${med:-NA}  issues=$bad" | tee -a "$PROG"
  fi
}

for mk in "${MODELS[@]}"; do
  run $mk fp32_eager        --precision fp32
  run $mk fp32_compile      --precision fp32 --compile
  run $mk fp16_compilecg    --fp16          --compile --inductor-cudagraph
  run $mk int8_compile      --torchao-int8  --compile
  run $mk int8_compilecg    --torchao-int8  --compile --inductor-cudagraph   # EXPECTED-FAIL on cu130 (see run(): _int_mm M>16)
done
echo "[$(date +%H:%M:%S)] ALL DONE — see $OUT/ and docs/efficiency.md" | tee -a "$PROG"
