#!/usr/bin/env bash
# Phase 0 / GATE-B driver — calibration-data SENSITIVITY study (docs/experiments.md §6).
#
# Question: does the CHOICE of the 32-item calibration set move dev metrics by more than
# calibration-seed noise?  This gate decides the whole data-selection line.
#
# Jobs (all independent -> wave-scheduled across the GPU pool, one GPU each):
#   fp32                          dev-split fp32 reference (paired-Δ anchor), generated ONCE (runs LAST)
#   rand<SET_SIZE>_s{0..K-1}      FlatQuant frozen-best calibrated on K random pool draws, calib_seed 0
#   rand<SET_SIZE>_s0_cs{c}       SAME draw s0 under other calib seeds  -> seed-noise baseline
# Each job: calibrate (in-process, ablation-style — no model files) -> generate dev sets -> evaluate.
# Idempotent: finished jobs (all text-metric files present) are skipped; partially generated wav dirs
# resume (existing wavs are skipped by the generators).
#
# Usage:
#   bash scripts/calib/phase0_sensitivity.sh [1b|3.5b]
# Typical:
#   GPUS=0,1,2,3,4,5,6,7 K=10 bash scripts/calib/phase0_sensitivity.sh 1b
#   LIMIT=2 K=2 SEED_REPEATS="1" EVAL_METRICS="cer wer" bash scripts/calib/phase0_sensitivity.sh 1b  # smoke
#
# Env knobs:
#   K=10                 number of random calibration draws
#   SET_SIZE=32          items per calibration set
#   SEED_REPEATS="1 2 3" extra calib seeds re-running draw s0 (seed-noise baseline; "" = none)
#   POOL_LST=data/calib_pool/pool_v1.lst
#   SETS_DEV="zh_dev hard_dev en_dev"
#   EVAL_METRICS="wer cer sim"   (mos off by default — post-hoc computable from the kept wavs)
#   STEPS/MB/MAX_SEQS/PER_ITEM_KEEP  FlatQuant budget (defaults = frozen paper-best 200/4/64/2)
#   BASE=1024 LIMIT=0 GPUS=...
#   STATUS_INTERVAL=120  console heartbeat: every N s print one line per RUNNING job with its
#                        newest log line (capture x/32 -> [pb] block i/24 loss -> gen 45% -> eval)
#
# PROTOCOL: contribution knobs stay OFF (loss=mse, no region weights). Dev-list per-item seeds pair
# only with other dev runs (see dev_split.py note). Analysis: phase0_collect (aggregate CSV) +
# paired_bootstrap on per-item files vs the p0_fp32 reference.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"
cd "$SEED_REPRO_DIR"

which_model="${1:-1b}"
case "$which_model" in
  1b|1B)   MODEL_DIR="meituan-longcat/LongCat-AudioDiT-1B";   msuf="" ;;
  3.5b|3.5B|3p5b) MODEL_DIR="meituan-longcat/LongCat-AudioDiT-3.5B"; msuf="_3.5b" ;;
  *) echo "usage: bash scripts/calib/phase0_sensitivity.sh [1b|3.5b]" >&2; exit 2 ;;
esac

K="${K:-10}"
SET_SIZE="${SET_SIZE:-32}"
SEED_REPEATS="${SEED_REPEATS-1 2 3}"
POOL_LST="${POOL_LST:-data/calib_pool/pool_v1.lst}"
SETS_DEV="${SETS_DEV:-zh_dev hard_dev en_dev}"
# mos dropped from the default (2026-07-19): least calibration-sensitive metric, and wavs are kept
# so MOS is post-hoc computable anytime via evaluate_seedtts_metrics.sh <gen_root> <tag> <sets> mos
EVAL_METRICS="${EVAL_METRICS:-wer cer sim}"
BASE="${BASE:-1024}"; LIMIT="${LIMIT:-0}"
STEPS="${STEPS:-200}"; MB="${MB:-4}"; MAX_SEQS="${MAX_SEQS:-64}"; PER_ITEM_KEEP="${PER_ITEM_KEEP:-2}"
sets_csv="${SETS_DEV// /,}"

[ -f "$POOL_LST" ] || { echo "[p0] candidate pool missing: $POOL_LST (build: python -m audio_dit_quantize.calib.pool build)" >&2; exit 1; }

# 0) frozen dev split (materialize-only when manifests exist) + K deterministic draws
"$PYTHON_BIN" -m audio_dit_quantize.calib.dev_split
mkdir -p data/calib_pool/sets logs/p0
for (( k=0; k<K; k++ )); do
  "$PYTHON_BIN" -m audio_dit_quantize.calib.pool sample --pool "$POOL_LST" --n "$SET_SIZE" \
    --seed "$k" --out "data/calib_pool/sets/rand${SET_SIZE}_s${k}.lst"
done

# job spec: name|kind|calib_lst|calib_seed
# fp32 goes LAST: it is much faster than a flat job, so pairing it into an early wave leaves its
# GPU idle behind the wave barrier for hours; flat-only waves are duration-homogeneous.
JOBS=()
for (( k=0; k<K; k++ )); do
  JOBS+=("rand${SET_SIZE}_s${k}|flat|data/calib_pool/sets/rand${SET_SIZE}_s${k}.lst|0")
done
for c in $SEED_REPEATS; do
  JOBS+=("rand${SET_SIZE}_s0_cs${c}|flat|data/calib_pool/sets/rand${SET_SIZE}_s0.lst|$c")
done
JOBS+=("fp32|fp32||")

job_done() {  # all per-set text-metric result files present AND valid -> job complete
  local tag="$1" s m f
  for s in $SETS_DEV; do
    case "$s" in en*) m=wer ;; *) m=cer ;; esac
    f="$SEED_RESULTS_DIR/${tag}_${s}_${m}.txt"
    [ -s "$f" ] || return 1
    if grep -qi 'nan' "$f"; then
      # poisoned leftover from a failed eval (e.g. ASR OOM on a shared GPU) — remove it so the
      # resume re-runs this job instead of silently skipping it
      echo "[p0] WARN: $f contains nan -> deleting, job ${tag} will re-run" >&2
      rm -f "$f"; return 1
    fi
  done
  return 0
}

run_job() {  # run_job <name> <kind> <lst> <cseed> <gpu>   (CUDA_VISIBLE_DEVICES already set)
  local name="$1" kind="$2" lst="$3" cseed="$4" gpu="$5"
  local tag="p0_${name}${msuf}" gen_root
  if [ "$kind" = "fp32" ]; then
    gen_root="gen/paired/$tag"
    "$PYTHON_BIN" -m audio_dit_quantize.generate_seedtts --mode fp32 --tag "$tag" \
      --sets "$sets_csv" --base "$BASE" --limit "$LIMIT" --model_dir "$MODEL_DIR" --device cuda:0
  else
    gen_root="gen/$tag"
    # calibrate+generate in ONE process (per-run draw, no model persistence), frozen best-config recipe
    "$PYTHON_BIN" -m audio_dit_quantize.flatquant_best --model_dir "$MODEL_DIR" \
      --calib_lst "$lst" --calib_seed "$cseed" --out_subdir "$tag" --sets "$sets_csv" \
      --base "$BASE" --limit "$LIMIT" --max_seqs "$MAX_SEQS" --per_item_keep "$PER_ITEM_KEEP" \
      --steps "$STEPS" --mb "$MB" --loss mse --device cuda:0
  fi
  GPUS="$gpu" EVAL_JOBS=1 bash scripts/evaluate_seedtts_metrics.sh "$gen_root" "$tag" "$SETS_DEV" "$EVAL_METRICS"
}

read -r -a POOL <<< "$(gpu_pool)"
NG="${#POOL[@]}"
[ "$NG" -ge 1 ] || { echo "[p0] no usable GPUs (GPUS / GPU_MIN_FREE_GIB)" >&2; exit 1; }
STATUS_INTERVAL="${STATUS_INTERVAL:-120}"
echo "[p0] ${#JOBS[@]} job(s) over $NG GPU(s) [pool: ${POOL[*]}] | model=$MODEL_DIR sets=$SETS_DEV"
echo "[p0] budget: steps=$STEPS mb=$MB max_seqs=$MAX_SEQS per_item_keep=$PER_ITEM_KEEP | K=$K set_size=$SET_SIZE seed_repeats='$SEED_REPEATS'"
echo "[p0] queue (= complete, will skip | * pending):"
for spec in "${JOBS[@]}"; do
  IFS='|' read -r jn _ _ _ <<< "$spec"
  if job_done "p0_${jn}${msuf}"; then echo "  = ${jn}"; else echo "  * ${jn}"; fi
done

_now() { date +%H:%M:%S; }
_min() { echo $(( ($(date +%s) - $1) / 60 )); }
_last_line() {  # newest non-empty line of a log (tqdm \r-aware), truncated for one-line status
  tail -c 4096 "$1" 2>/dev/null | tr '\r' '\n' | grep -E '\S' | tail -1 | cut -c1-110
}

run_t0=$(date +%s); n_done=0; n_skip=0; wave=0; fail=0; failed_jobs=()
i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  wave=$(( wave + 1 ))
  pids=(); names=(); logs=(); gpus=(); starts=(); w=0
  while [ "$w" -lt "$NG" ] && [ "$i" -lt "${#JOBS[@]}" ]; do
    IFS='|' read -r jname jkind jlst jseed <<< "${JOBS[$i]}"
    i=$(( i + 1 ))
    if job_done "p0_${jname}${msuf}"; then
      echo "[p0 $(_now)] skip ${jname} (results complete)"
      n_skip=$(( n_skip + 1 ))
      continue
    fi
    gpu="${POOL[$w]}"; log="logs/p0/${jname}${msuf}.log"
    echo "[p0 $(_now)] wave $wave: launch ${jname} -> GPU $gpu  (follow: tail -f $log)"
    ( export CUDA_VISIBLE_DEVICES="$gpu" DEVICE="cuda:0"
      echo "[job] ${jname} gpu=${gpu} kind=${jkind} calib_lst=${jlst:-none} calib_seed=${jseed:-none} start=$(date '+%F %T')"
      run_job "$jname" "$jkind" "$jlst" "$jseed" "$gpu"
      echo "[job] ${jname} end=$(date '+%F %T')" ) >"$log" 2>&1 &
    pids+=("$!"); names+=("$jname"); logs+=("$log"); gpus+=("$gpu"); starts+=("$(date +%s)")
    w=$(( w + 1 ))
  done
  [ "${#pids[@]}" -eq 0 ] && continue
  # heartbeat while the wave runs: one status line per live job + overall progress
  while :; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { alive=1; break; }; done
    [ "$alive" -eq 0 ] && break
    sleep "$STATUS_INTERVAL"
    for k2 in "${!pids[@]}"; do
      kill -0 "${pids[$k2]}" 2>/dev/null || continue
      echo "[p0 $(_now)] RUN  ${names[$k2]}@gpu${gpus[$k2]} $(_min "${starts[$k2]}")min :: $(_last_line "${logs[$k2]}")"
    done
    echo "[p0 $(_now)] progress: done $n_done + skip $n_skip of ${#JOBS[@]} | wave $wave | total elapsed $(_min "$run_t0")min"
  done
  for k2 in "${!pids[@]}"; do
    if wait "${pids[$k2]}"; then
      n_done=$(( n_done + 1 ))
      summary=""
      for f in "$SEED_RESULTS_DIR"/p0_"${names[$k2]}"${msuf}_*_cer.txt \
               "$SEED_RESULTS_DIR"/p0_"${names[$k2]}"${msuf}_*_wer.txt; do
        [ -f "$f" ] || continue
        summary+="$(basename "$f" .txt | sed "s/^p0_${names[$k2]}${msuf}_//")=$(tail -1 "$f" | grep -oE '[0-9.]+%?' | head -1)  "
      done
      echo "[p0 $(_now)] DONE ${names[$k2]} ($(_min "${starts[$k2]}")min)  ${summary}"
    else
      fail=1; failed_jobs+=("${names[$k2]}")
      echo "[p0 $(_now)] FAILED ${names[$k2]} after $(_min "${starts[$k2]}")min — last 20 log lines (${logs[$k2]}):" >&2
      tail -c 8192 "${logs[$k2]}" | tr '\r' '\n' | tail -20 >&2
    fi
  done
done
if [ "$fail" -ne 0 ]; then
  echo "[p0] ERROR: ${#failed_jobs[@]} job(s) failed: ${failed_jobs[*]}" >&2
  echo "[p0] full logs in logs/p0/ — rerun the SAME command to resume just the failed job(s)" >&2
  exit 1
fi

"$PYTHON_BIN" -m audio_dit_quantize.calib.phase0_collect --prefix "p0" --suffix "$msuf" \
  --sets "$SETS_DEV" --out "$SEED_RESULTS_DIR/p0_summary${msuf}.csv"
echo "P0 DONE"
