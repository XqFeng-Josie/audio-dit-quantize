#!/usr/bin/env bash
# Install the Seed-TTS eval assets expected by the benchmark scripts.
#
# The benchmark code expects:
#   data/seedtts_testset/zh/meta.lst
#   data/seedtts_testset/en/meta.lst
#   data/seedtts_testset/zh/hardcase.lst
#   data/seedtts_testset/{zh,en}/{prompt-wavs,wavs}/*.wav
#   eval/ckpt/wavlm_large_finetune.pth                 # only needed when running SIM
#
# Usage:
#   bash scripts/download_seedtts_testset.sh
#
# Fast local install when another repo already has the assets:
#   SOURCE_DIR=/home/xiaoqin_feng/workspace/seed_repro/data/seedtts_testset \
#   WAVLM_SOURCE=/home/xiaoqin_feng/workspace/seed_repro/eval/ckpt/wavlm_large_finetune.pth \
#     bash scripts/download_seedtts_testset.sh
#
# Google Drive sources:
#   Seed-TTS test set: 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP
#   WavLM SIM ckpt:    1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP
#
# Common env knobs:
#   SOURCE_DIR=/path/to/seedtts_testset      copy an existing unpacked dataset
#   WAVLM_SOURCE=/path/to/wavlm_large_...pth copy an existing WavLM checkpoint
#   ARCHIVE=/path/to/seedtts_testset.tar     reuse/write the downloaded dataset tar
#   FORCE=1                                  reinstall even if target exists
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"

seedtts_file_id="1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"
wavlm_file_id="1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
data_target="$SEED_DATA_DIR"
wavlm_target="$WAVLM_CKPT"
archive="${ARCHIVE:-$SEED_REPRO_DIR/data/seedtts_testset.tar}"
source_dir="${SOURCE_DIR:-}"
wavlm_source="${WAVLM_SOURCE:-}"

download_gdrive() {
  local file_id="$1"
  local out_file="$2"

  if command -v gdown >/dev/null 2>&1; then
    gdown "https://drive.google.com/uc?id=$file_id" -O "$out_file"
  else
    "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=$file_id" -O "$out_file"
  fi
}

install_seedtts_testset() {
  if [ -f "$data_target/zh/meta.lst" ] && [ -f "$data_target/en/meta.lst" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[data] Seed-TTS test set already installed at $data_target"
    echo "[data] set FORCE=1 to reinstall it"
    return
  fi

  mkdir -p "$(dirname "$data_target")"

  if [ -n "$source_dir" ]; then
    [ -d "$source_dir" ] || { echo "[data] SOURCE_DIR not found: $source_dir" >&2; exit 1; }
    mkdir -p "$data_target"
    cp -a "$source_dir"/. "$data_target"/
  else
    mkdir -p "$(dirname "$archive")"
    download_gdrive "$seedtts_file_id" "$archive"
    tar -xf "$archive" -C "$(dirname "$data_target")"
  fi

  test -f "$data_target/zh/meta.lst"
  test -f "$data_target/en/meta.lst"
  test -f "$data_target/zh/hardcase.lst"

  echo "[data] installed Seed-TTS test set at $data_target"
  echo "[data] wav files: $(find "$data_target" -type f -name '*.wav' | wc -l)"
}

install_wavlm_ckpt() {
  if [ -f "$wavlm_target" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[ckpt] WavLM checkpoint already installed at $wavlm_target"
    echo "[ckpt] set FORCE=1 to reinstall it"
    return
  fi

  mkdir -p "$(dirname "$wavlm_target")"

  if [ -n "$wavlm_source" ]; then
    [ -f "$wavlm_source" ] || { echo "[ckpt] WAVLM_SOURCE not found: $wavlm_source" >&2; exit 1; }
    cp -a "$wavlm_source" "$wavlm_target"
  else
    download_gdrive "$wavlm_file_id" "$wavlm_target"
  fi

  test -f "$wavlm_target"
  echo "[ckpt] installed WavLM checkpoint at $wavlm_target"
  echo "[ckpt] size bytes: $(wc -c < "$wavlm_target")"
}

install_seedtts_testset
install_wavlm_ckpt
