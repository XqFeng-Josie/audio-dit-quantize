#!/usr/bin/env bash
# Install the Seed-TTS test set expected by the benchmark scripts.
#
# The benchmark code expects:
#   data/seedtts_testset/zh/meta.lst
#   data/seedtts_testset/en/meta.lst
#   data/seedtts_testset/zh/hardcase.lst
#   data/seedtts_testset/{zh,en}/{prompt-wavs,wavs}/*.wav
#
# Usage:
#   bash scripts/download_seedtts_testset.sh
#
# Fast local install when another repo already has the data:
#   SOURCE_DIR=/home/xiaoqin_feng/workspace/seed_repro/data/seedtts_testset \
#     bash scripts/download_seedtts_testset.sh
#
# Google Drive source:
#   id = 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP
#
# Common env knobs:
#   SOURCE_DIR=/path/to/seedtts_testset  copy an existing unpacked dataset
#   ARCHIVE=/path/to/seedtts_testset.tar reuse/write the downloaded tar
#   FORCE=1                              reinstall even if target exists
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/env.sh"

file_id="1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"
target="$SEED_DATA_DIR"
archive="${ARCHIVE:-$SEED_REPRO_DIR/data/seedtts_testset.tar}"
source_dir="${SOURCE_DIR:-}"

if [ -f "$target/zh/meta.lst" ] && [ -f "$target/en/meta.lst" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[data] Seed-TTS test set already installed at $target"
  echo "[data] set FORCE=1 to reinstall"
  exit 0
fi

mkdir -p "$(dirname "$target")"

if [ -n "$source_dir" ]; then
  [ -d "$source_dir" ] || { echo "[data] SOURCE_DIR not found: $source_dir" >&2; exit 1; }
  mkdir -p "$target"
  cp -a "$source_dir"/. "$target"/
else
  command -v gdown >/dev/null 2>&1 || {
    echo "[data] gdown is required. Install requirements first: pip install -r requirements.txt" >&2
    exit 127
  }
  mkdir -p "$(dirname "$archive")"
  gdown "https://drive.google.com/uc?id=$file_id" -O "$archive"
  tar -xf "$archive" -C "$(dirname "$target")"
fi

test -f "$target/zh/meta.lst"
test -f "$target/en/meta.lst"
test -f "$target/zh/hardcase.lst"

echo "[data] installed Seed-TTS test set at $target"
echo "[data] wav files: $(find "$target" -type f -name '*.wav' | wc -l)"
