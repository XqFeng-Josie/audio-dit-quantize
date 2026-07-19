#!/usr/bin/env bash
# Evaluate generated Seed-TTS wavs with text metrics and MOS.
#
# This is the single metric entry point used by all benchmark_*.sh scripts.
# Normally you do not need to run it by hand because generation launchers call it
# automatically after generation.
#
# Defaults:
#   zh/hard -> CER, en -> WER, all sets -> UTMOS + DNSMOS + WavLM SIM
#   (SIM needs eval/ckpt/wavlm_large_finetune.pth; pass an explicit metric list to drop it)
# Usage:
#   bash scripts/evaluate_seedtts_metrics.sh <gen_root> <result_prefix> ["zh en hard"] ["wer cer mos"]
# Examples:
#   bash scripts/evaluate_seedtts_metrics.sh gen/paired/fp32 pf_fp32 "zh en hard"
#   bash scripts/evaluate_seedtts_metrics.sh gen/paired/fp32 pf_fp32 hard "cer mos"
#   bash scripts/evaluate_seedtts_metrics.sh gen/paired/fp32 pf_fp32 "zh en hard" "wer cer mos sim"
#
# Metric names:
#   wer   run ASR WER on en
#   cer   run ASR CER on zh/hard
#   mos   run MOS backends selected by MOS_METRICS, default "utmos dnsmos"
#   sim   optional WavLM speaker similarity
#   all   shorthand for "wer cer mos sim"
#
# Common env knobs:
#   MOS_METRICS="utmos dnsmos"  MOS backends to run
#   MOS_LIMIT=0                 limit wavs for MOS; 0 means full set
#   DEVICE=cuda:0               device for ASR/SIM/UTMOS where applicable
#   ASR_DEVICE=cuda:0           override ASR device; defaults to DEVICE
#   ASR_BATCH_SIZE=auto         auto-tune ASR batch from GPU memory; set an int to override
#   ASR_BATCH_SIZE_S=auto       auto-tune FunASR zh batch_size_s; set an int to override
#   ASR_GPU_MEM_GIB=40          override detected GPU memory for ASR auto-tuning
#
# Acceleration (resource-aware; sets zh/en/hard are independent -> evaluated in parallel across GPUs):
#   GPUS="0,2,3"        which physical GPUs to use (shared with the generation launchers). Unset -> use
#                       CUDA_VISIBLE_DEVICES (or all) filtered by free memory. Single GPU -> sequential.
#   EVAL_JOBS=auto      parallel workers: auto = min(#sets, #usable GPUs); 1 = force sequential; or an int
#   GPU_MIN_FREE_GIB=8  when GPUS is unset, a GPU joins the pool only if it has >= this much free memory
#   (single usable GPU -> sequential, identical to the old behaviour; each worker gets one GPU + an even
#    share of CPU threads via OMP_NUM_THREADS to avoid oversubscription.)
#
# Outputs:
#   results/<result_prefix>_<set>_cer.txt or results/<result_prefix>_<set>_wer.txt
#   results/<result_prefix>_<set>_utmos.txt
#   results/<result_prefix>_<set>_dnsmos.txt
#   results/<result_prefix>_<set>_sim.txt when sim is requested
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"
source "$ROOT_DIR/scripts/gpu_parallel.sh"   # gpu_pool() — the shared GPU-range knob (GPUS / CUDA_VISIBLE_DEVICES)

gen_root="${1:?usage: evaluate_seedtts_metrics.sh <gen_root> <result_prefix> [sets] [metrics]}"
result_prefix="${2:?usage: evaluate_seedtts_metrics.sh <gen_root> <result_prefix> [sets] [metrics]}"
sets="${3:-zh en hard}"
metrics="${4:-${METRICS:-wer cer mos sim}}"

[ "$sets" = "all" ] && sets="zh en hard"
sets="${sets//,/ }"
metrics="${metrics//,/ }"
case " $metrics " in
  *" all "*) metrics="wer cer mos sim" ;;
esac
case " $metrics " in
  *" asr "*) metrics="$metrics wer cer" ;;
esac

case "$gen_root" in
  /*) ;;
  *) gen_root="$SEED_REPRO_DIR/$gen_root" ;;
esac

mkdir -p "$SEED_RESULTS_DIR"

# Preserve any user-pinned ASR mem hint BEFORE the global auto-detect below fills it in, so parallel
# workers can re-detect per assigned GPU without inheriting the first GPU's value.
USER_ASR_GPU_MEM_GIB="${ASR_GPU_MEM_GIB:-}"

detect_asr_gpu_mem() {
  [ -z "${ASR_GPU_MEM_GIB:-}" ] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local asr_device="${ASR_DEVICE:-$DEVICE}"
  case "$asr_device" in
    cuda|cuda:*) ;;
    *) return 0 ;;
  esac

  local logical_index=0
  case "$asr_device" in
    cuda:*) logical_index="${asr_device#cuda:}" ;;
  esac

  local visible="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX:-0}}"
  local gpu_index="${GPU_INDEX:-$logical_index}"
  IFS=',' read -r -a visible_gpus <<< "$visible"
  if [ "$logical_index" -lt "${#visible_gpus[@]}" ]; then
    gpu_index="${visible_gpus[$logical_index]}"
  fi

  local mem_mib
  mem_mib="$(nvidia-smi -i "$gpu_index" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
  if [ -n "$mem_mib" ]; then
    export ASR_GPU_MEM_GIB
    ASR_GPU_MEM_GIB="$(awk -v m="$mem_mib" 'BEGIN { printf "%.3f", m / 1024 }')"
  fi
}

detect_asr_gpu_mem

want_metric() {
  case " $metrics " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

make_wav_res_ref_text() {
  local meta="$1"
  local gen_dir="$2"
  local wav_res_ref_text="$gen_dir/wav_res_ref_text"

  (
    cd "$SEED_EVAL_DIR"
    "$PYTHON_BIN" get_wav_res_ref_text.py "$meta" "$gen_dir" "$wav_res_ref_text"
  )
}

score_text_metric() {
  local meta="$1"
  local gen_dir="$2"
  local lang="$3"
  local score_file="$4"
  local wav_res_ref_text="$gen_dir/wav_res_ref_text"
  local merge="$gen_dir/asr_merge.out"
  local run_wer_py="$SEED_REPRO_DIR/src/audio_dit_quantize/seedtts_asr.py"

  make_wav_res_ref_text "$meta" "$gen_dir"
  # Guard (audit F4): if no (gen wav <-> ref) pairs were produced — e.g. generated wav filenames do
  # not match the meta utt ids, or generation wrote nothing — the ASR merge is empty and average_wer.py
  # computes mean([]) = nan, writing "WER: nan%" and exiting 0. set -e cannot catch a nan, so the
  # pipeline would report success on a non-result. Fail loudly instead.
  if [ ! -s "$wav_res_ref_text" ]; then
    echo "[metrics] ERROR: no (gen wav <-> ref) pairs for $gen_dir (empty $wav_res_ref_text) — generated wav filenames do not match meta utt ids, or generation produced no wavs." >&2
    exit 1
  fi
  (
    cd "$SEED_EVAL_DIR"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX:-0}}" "$PYTHON_BIN" "$run_wer_py" "$wav_res_ref_text" "$merge" "$lang"
    "$PYTHON_BIN" average_wer.py "$merge" "$score_file"
  )
  echo "---- $score_file ----"
  tail -1 "$score_file"
  # Second guard: a nan can still slip through if every utterance failed ASR (empty merge). Do not
  # leave a "nan%" file masquerading as a valid metric.
  if grep -qi 'nan' "$score_file"; then
    echo "[metrics] ERROR: $score_file contains nan (no validly scored utterances in $gen_dir)." >&2
    exit 1
  fi
}

score_sim_metric() {
  local meta="$1"
  local gen_dir="$2"
  local score_file="$3"

  make_wav_res_ref_text "$meta" "$gen_dir"
  "$PYTHON_BIN" -m audio_dit_quantize.seedtts_similarity "$gen_dir/wav_res_ref_text" "$score_file" "$DEVICE"
}

score_mos_metrics() {
  local gen_dir="$1"
  local setname="$2"
  local mos_metrics="${MOS_METRICS:-utmos dnsmos}"
  local mos_limit="${MOS_LIMIT:-0}"

  for mos_metric in $mos_metrics; do
    local score_file="$SEED_RESULTS_DIR/${result_prefix}_${setname}_${mos_metric}.txt"
    "$PYTHON_BIN" -m audio_dit_quantize.seedtts_mos "$mos_metric" "$gen_dir" "$score_file" \
      --limit "$mos_limit" \
      --device "$DEVICE"
  done
}

# Fail early (before any heavy ASR/MOS work) if SIM is requested but the WavLM checkpoint is absent.
if want_metric sim && [ ! -f "${WAVLM_CKPT:-}" ]; then
  echo "[metrics] ERROR: SIM requested but WavLM checkpoint not found: ${WAVLM_CKPT:-<unset>}" >&2
  echo "        get it with: bash scripts/setup/download_seedtts_testset.sh   (or export WAVLM_CKPT=/path/to/wavlm_large_finetune.pth)" >&2
  echo "        or drop SIM:  EVAL_METRICS=\"wer cer mos\" bash scripts/evaluate_seedtts_metrics.sh ..." >&2
  exit 1
fi

run_set() {
  local setname="$1" gpu="${2:-}"
  local meta lang text_metric metric_suffix
  case "$setname" in
    zh)            meta="$SEED_DATA_DIR/zh/meta.lst";              lang=zh; text_metric=cer ;;
    en)            meta="$SEED_DATA_DIR/en/meta.lst";              lang=en; text_metric=wer ;;
    hard)          meta="$SEED_DATA_DIR/zh/hardcase.lst";          lang=zh; text_metric=cer ;;
    zh_dev)        meta="$SEED_DATA_DIR/zh/meta_dev.lst";          lang=zh; text_metric=cer ;;
    en_dev)        meta="$SEED_DATA_DIR/en/meta_dev.lst";          lang=en; text_metric=wer ;;
    hard_dev)      meta="$SEED_DATA_DIR/zh/hardcase_dev.lst";      lang=zh; text_metric=cer ;;
    zh_heldtest)   meta="$SEED_DATA_DIR/zh/meta_heldtest.lst";     lang=zh; text_metric=cer ;;
    en_heldtest)   meta="$SEED_DATA_DIR/en/meta_heldtest.lst";     lang=en; text_metric=wer ;;
    hard_heldtest) meta="$SEED_DATA_DIR/zh/hardcase_heldtest.lst"; lang=zh; text_metric=cer ;;
    *) echo "[metrics] unknown set: $setname (frozen dev/heldtest splits: python -m audio_dit_quantize.calib.dev_split)" >&2; exit 2 ;;
  esac
  metric_suffix="${setname}_${text_metric}"
  local gen_dir="$gen_root/$setname"
  [ -d "$gen_dir" ] || { echo "[metrics] missing generated dir: $gen_dir" >&2; exit 1; }

  # Pin this worker to its assigned GPU (subshell-local); re-detect ASR mem for THIS GPU and split CPU
  # threads across workers so N parallel evals do not oversubscribe the cores.
  if [ -n "$gpu" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu"; export DEVICE="cuda:0"; export ASR_DEVICE="cuda:0"
    ASR_GPU_MEM_GIB="$USER_ASR_GPU_MEM_GIB"; detect_asr_gpu_mem
    if [ -z "${OMP_NUM_THREADS:-}" ] && command -v nproc >/dev/null 2>&1; then
      export OMP_NUM_THREADS="$(( ( $(nproc) + jobs - 1 ) / jobs ))"
    fi
    echo "[metrics] set=$setname -> physical GPU $gpu (DEVICE=cuda:0, OMP_NUM_THREADS=${OMP_NUM_THREADS:-default})"
  fi

  # SIM/MOS run BEFORE the ASR text metric: the text metric's nan guard exits the worker on a
  # transient Whisper failure, and anything ordered after it never runs (this silently dropped
  # en_dev SIM for every run of two whole experiment rounds — root-caused 2026-07-20).
  if want_metric sim; then
    score_sim_metric "$meta" "$gen_dir" "$SEED_RESULTS_DIR/${result_prefix}_${setname}_sim.txt"
  fi
  if want_metric mos; then
    score_mos_metrics "$gen_dir" "$setname"
  fi
  if want_metric "$text_metric"; then
    score_text_metric "$meta" "$gen_dir" "$lang" "$SEED_RESULTS_DIR/${result_prefix}_${metric_suffix}.txt"
  fi
}

# ── resolve parallelism from live GPU state (shared GPUS / CUDA_VISIBLE_DEVICES knob) ──
read -r -a GPU_POOL <<< "$(gpu_pool)"
NGPU="${#GPU_POOL[@]}"
set_arr=($sets); NSETS="${#set_arr[@]}"
eval_jobs="${EVAL_JOBS:-auto}"
if [ "$eval_jobs" = "auto" ]; then
  if [ "$NGPU" -gt 1 ]; then jobs=$(( NGPU < NSETS ? NGPU : NSETS )); else jobs=1; fi
else
  jobs="$eval_jobs"
  if [ "$jobs" -lt 1 ]; then jobs=1; fi
  if [ "$NGPU" -gt 0 ] && [ "$jobs" -gt "$NGPU" ]; then jobs="$NGPU"; fi   # never more workers than usable GPUs
  if [ "$jobs" -gt "$NSETS" ]; then jobs="$NSETS"; fi
fi

if [ "$jobs" -le 1 ]; then
  # Sequential (single usable GPU or forced): live output, uses the ambient DEVICE — identical to the
  # old behaviour (no GPU pinning, no OMP override).
  for setname in $sets; do ( run_set "$setname" "" ); done
else
  echo "[metrics] parallel eval: $NSETS set(s) over $jobs GPU worker(s) [usable pool: ${GPU_POOL[*]}]"
  # Wave scheduling: at most `jobs` sets concurrently, one per distinct GPU; capture each worker's
  # output to a log and replay in order so the parallel logs stay readable.
  i=0; fail=0
  while [ "$i" -lt "$NSETS" ]; do
    pids=(); names=(); logs=(); w=0
    while [ "$w" -lt "$jobs" ] && [ "$i" -lt "$NSETS" ]; do
      s="${set_arr[$i]}"; gpu="${GPU_POOL[$(( w % NGPU ))]}"; log="$(mktemp)"
      ( run_set "$s" "$gpu" ) >"$log" 2>&1 &
      pids+=("$!"); names+=("$s"); logs+=("$log")
      i=$(( i + 1 )); w=$(( w + 1 ))
    done
    for k in "${!pids[@]}"; do
      if ! wait "${pids[$k]}"; then fail=1; fi
      echo "===== [eval:${names[$k]}] ====="; cat "${logs[$k]}"; rm -f "${logs[$k]}"
    done
    if [ "$fail" -eq 1 ]; then
      echo "[metrics] ERROR: a parallel eval worker failed (see above)." >&2
      exit 1
    fi
  done
fi
