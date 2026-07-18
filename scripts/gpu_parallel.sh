#!/usr/bin/env bash
# Shared GPU-range + parallel helpers, sourced by the benchmark launchers and the eval entry point so
# inference (generation) and evaluation obey the SAME "which GPUs to use" knob and split work the same way.
#
# GPU range selection (one uniform knob):
#   GPUS="0,2,3"        explicit physical GPU indices to use (honored as-is, no free-memory filter).
#   (unset)             fall back to CUDA_VISIBLE_DEVICES if set, else all GPUs; keep only those with
#                       >= GPU_MIN_FREE_GIB (default 8) free memory (nvidia-smi).
#
# Generation is parallelized by ITEM SHARD: each set's items are split into contiguous ranges (one per
# GPU) via --offset/--limit. Per-item seed is base + GLOBAL index, so a sharded gen is byte-for-byte the
# same as a single-GPU full gen (order-free / paired protocol preserved). One-time quantization
# calibration is NOT sharded — the launcher does it once on a single GPU, then fans out generation.
#
# This file only defines functions; it does not run anything on source.

# Echo space-separated usable PHYSICAL GPU indices (empty if none / no nvidia-smi).
gpu_pool() {
  if [ -n "${GPUS:-}" ]; then
    # Explicit user range: trust it, do not memory-filter.
    echo "${GPUS//,/ }" | tr -s ' ' | sed 's/^ //; s/ $//'
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local min_free_mib allow idx free out=""
  min_free_mib="$(awk -v g="${GPU_MIN_FREE_GIB:-8}" 'BEGIN{printf "%d", g*1024}')"
  allow="${CUDA_VISIBLE_DEVICES:-}"
  declare -A allowset=()
  if [ -n "$allow" ]; then
    local x _a; IFS=',' read -r -a _a <<< "$allow"; for x in "${_a[@]}"; do allowset["$x"]=1; done
  fi
  while IFS=',' read -r idx free; do
    idx="${idx// /}"; free="${free// /}"
    [ -n "$idx" ] || continue
    if [ -n "$allow" ] && [ -z "${allowset[$idx]:-}" ]; then continue; fi
    if [ "${free:-0}" -ge "$min_free_mib" ]; then out="$out $idx"; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
  echo "${out# }"
}

# Echo the meta list path for a Seed set (used to count items for sharding).
set_meta() {
  case "$1" in
    zh)   echo "$SEED_DATA_DIR/zh/meta.lst" ;;
    en)   echo "$SEED_DATA_DIR/en/meta.lst" ;;
    hard) echo "$SEED_DATA_DIR/zh/hardcase.lst" ;;
    *)    echo ""; return 1 ;;
  esac
}

# Echo the effective item count of a set (capped by a >0 user limit).
set_count() {
  local s="$1" ulimit="${2:-0}" meta n
  meta="$(set_meta "$s")" || { echo 0; return 1; }
  n="$(wc -l < "$meta" 2>/dev/null | tr -d ' ')"; [ -n "$n" ] || n=0
  if [ "$ulimit" -gt 0 ] && [ "$ulimit" -lt "$n" ]; then n="$ulimit"; fi
  echo "$n"
}

# Echo "<offset> <len>" for each of k contiguous shards of N items (sizes differ by at most 1).
shard_ranges() {
  local N="$1" k="$2" q r i off len
  q=$(( N / k )); r=$(( N % k )); off=0
  for (( i=0; i<k; i++ )); do
    if [ "$i" -lt "$r" ]; then len=$(( q + 1 )); else len=$q; fi
    echo "$off $len"; off=$(( off + len ))
  done
}

# run_gen_parallel <callback> <sets_space> <user_limit>
#   callback signature:  cb <sets_csv> <offset> <limit> <gpu>
#     - single-GPU / no pool : called ONCE with the full sets_csv, offset 0, the user limit, gpu="" ->
#                              cb uses the ambient DEVICE (identical to the old one-process behaviour).
#     - multi-GPU            : called once per (single set, shard) with CUDA_VISIBLE_DEVICES + DEVICE=cuda:0
#                              exported for that worker; cb should pass --device "$DEVICE".
#   The callback owns the actual python command (it differs per method), so calibration/model handling
#   stays in the launcher. Ordered log replay + failure propagation, one shard per GPU per wave.
run_gen_parallel() {
  local cb="$1" sets="$2" ulimit="${3:-0}"
  local POOL; read -r -a POOL <<< "$(gpu_pool)"; local NG="${#POOL[@]}"

  if [ "$NG" -le 1 ]; then
    ( "$cb" "$(echo "$sets" | tr ' ' ',')" 0 "$ulimit" "" )
    return $?
  fi

  local jobs=() s cnt k off len
  for s in $sets; do
    cnt="$(set_count "$s" "$ulimit")"
    if [ "$cnt" -le 0 ]; then echo "[gen] WARN: set $s has 0 items, skipped" >&2; continue; fi
    k="$NG"; if [ "$k" -gt "$cnt" ]; then k="$cnt"; fi
    while read -r off len; do jobs+=("$s|$off|$len"); done < <(shard_ranges "$cnt" "$k")
  done
  local njobs="${#jobs[@]}"
  if [ "$njobs" -eq 0 ]; then echo "[gen] nothing to generate" >&2; return 0; fi

  echo "[gen] parallel: $njobs item-shard job(s) over $NG GPU(s) [pool: ${POOL[*]}]  (calibration, if any, was single-GPU)"
  local i=0 fail=0
  while [ "$i" -lt "$njobs" ]; do
    local pids=() logs=() descs=() w=0 job js jo jl gpu log rest
    while [ "$w" -lt "$NG" ] && [ "$i" -lt "$njobs" ]; do
      job="${jobs[$i]}"; js="${job%%|*}"; rest="${job#*|}"; jo="${rest%%|*}"; jl="${rest#*|}"
      gpu="${POOL[$w]}"; log="$(mktemp)"
      ( export CUDA_VISIBLE_DEVICES="$gpu" DEVICE="cuda:0"; "$cb" "$js" "$jo" "$jl" "$gpu" ) >"$log" 2>&1 &
      pids+=("$!"); logs+=("$log"); descs+=("$js[$jo:+$jl]@gpu$gpu")
      i=$(( i + 1 )); w=$(( w + 1 ))
    done
    local k2
    for k2 in "${!pids[@]}"; do
      if ! wait "${pids[$k2]}"; then fail=1; fi
      echo "===== [gen:${descs[$k2]}] ====="; cat "${logs[$k2]}"; rm -f "${logs[$k2]}"
    done
    if [ "$fail" -eq 1 ]; then echo "[gen] ERROR: a shard worker failed (see above)." >&2; return 1; fi
  done
}
