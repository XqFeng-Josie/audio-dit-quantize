#!/usr/bin/env bash
# Evaluate generated Seed-TTS wavs with text metrics and MOS.
#
# This is the single metric entry point used by all benchmark_*.sh scripts.
# Normally you do not need to run it by hand because generation launchers call it
# automatically after generation.
#
# Defaults:
#   zh/hard -> CER, en -> WER, all sets -> UTMOS + DNSMOS
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
# Outputs:
#   results/<result_prefix>_<set>_cer.txt or results/<result_prefix>_<set>_wer.txt
#   results/<result_prefix>_<set>_utmos.txt
#   results/<result_prefix>_<set>_dnsmos.txt
#   results/<result_prefix>_<set>_sim.txt when sim is requested
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"

gen_root="${1:?usage: evaluate_seedtts_metrics.sh <gen_root> <result_prefix> [sets] [metrics]}"
result_prefix="${2:?usage: evaluate_seedtts_metrics.sh <gen_root> <result_prefix> [sets] [metrics]}"
sets="${3:-zh en hard}"
metrics="${4:-${METRICS:-wer cer mos}}"

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
  (
    cd "$SEED_EVAL_DIR"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX:-0}}" "$PYTHON_BIN" "$run_wer_py" "$wav_res_ref_text" "$merge" "$lang"
    "$PYTHON_BIN" average_wer.py "$merge" "$score_file"
  )
  echo "---- $score_file ----"
  tail -1 "$score_file"
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

for setname in $sets; do
  case "$setname" in
    zh)
      meta="$SEED_DATA_DIR/zh/meta.lst"
      lang=zh
      text_metric=cer
      metric_suffix=zh_cer
      ;;
    en)
      meta="$SEED_DATA_DIR/en/meta.lst"
      lang=en
      text_metric=wer
      metric_suffix=en_wer
      ;;
    hard)
      meta="$SEED_DATA_DIR/zh/hardcase.lst"
      lang=zh
      text_metric=cer
      metric_suffix=hard_cer
      ;;
    *)
      echo "[metrics] unknown set: $setname" >&2
      exit 2
      ;;
  esac

  gen_dir="$gen_root/$setname"
  [ -d "$gen_dir" ] || { echo "[metrics] missing generated dir: $gen_dir" >&2; exit 1; }

  if want_metric "$text_metric"; then
    score_text_metric "$meta" "$gen_dir" "$lang" "$SEED_RESULTS_DIR/${result_prefix}_${metric_suffix}.txt"
  fi
  if want_metric sim; then
    score_sim_metric "$meta" "$gen_dir" "$SEED_RESULTS_DIR/${result_prefix}_${setname}_sim.txt"
  fi
  if want_metric mos; then
    score_mos_metrics "$gen_dir" "$setname"
  fi
done
