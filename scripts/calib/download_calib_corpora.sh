#!/usr/bin/env bash
# Download the CLEAN calibration-pool source corpora (docs/experiments.md §4.3, roadmap E0):
#   - AISHELL-3   (openslr 93, ~19 GB tgz): zh multi-speaker TTS corpus -> zh prompt wavs + transcripts.
#                 Speaker-disjoint from the Seed-TTS zh test set (L2 isolation).
#   - LibriTTS dev-clean (openslr 60, ~1.2 GB): en corpus -> en prompt wavs (Seed-TTS en uses
#                 Common Voice, so LibriTTS is disjoint).
#
# Layout after this script:
#   data/calib_corpora/aishell3/{train,test}/...      (wav + content.txt + spk-info)
#   data/calib_corpora/LibriTTS/dev-clean/...         (spk/chapter/*.wav + *.normalized.txt)
#
# Idempotent: skips anything already extracted. Resumes partial downloads (wget -c).
# Env knobs:
#   KEEP_ARCHIVE=0   delete each .tgz after successful extraction (default; disk is tight)
#   MIRRORS="..."    override the openslr mirror list (space-separated base URLs)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/data/calib_corpora"
mkdir -p "$DEST"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"
# CN mirror first (this box is in CN); then main + US.
MIRRORS="${MIRRORS:-https://openslr.magicdatatech.com/resources https://www.openslr.org/resources https://us.openslr.org/resources}"

fetch() {  # fetch <resource_path> <out_file>  -> tries each mirror with resume
    local res="$1" out="$2" m
    for m in $MIRRORS; do
        echo "[fetch] trying $m/$res"
        if wget -c --timeout=60 --tries=3 -O "$out" "$m/$res"; then
            return 0
        fi
        echo "[fetch] mirror failed: $m (will try next)"
    done
    return 1
}

# ── LibriTTS dev-clean (small, first) ─────────────────────────────────────────
if [ -d "$DEST/LibriTTS/dev-clean" ] && [ -n "$(find "$DEST/LibriTTS/dev-clean" -name '*.wav' -print -quit 2>/dev/null)" ]; then
    echo "[libritts] already extracted -> skip"
else
    TGZ="$DEST/dev-clean.tar.gz"
    fetch "60/dev-clean.tar.gz" "$TGZ" || { echo "[libritts] DOWNLOAD FAILED"; exit 1; }
    echo "[libritts] extracting ..."
    tar -xzf "$TGZ" -C "$DEST" || { echo "[libritts] EXTRACT FAILED (corrupt archive? delete $TGZ and rerun)"; exit 1; }
    [ "$KEEP_ARCHIVE" = "1" ] || rm -f "$TGZ"
    echo "[libritts] done: $(find "$DEST/LibriTTS/dev-clean" -name '*.wav' | wc -l) wavs"
fi

# ── AISHELL-3 (~19 GB) ────────────────────────────────────────────────────────
if [ -d "$DEST/aishell3" ] && [ -n "$(find "$DEST/aishell3" -name '*.wav' -print -quit 2>/dev/null)" ]; then
    echo "[aishell3] already extracted -> skip"
else
    TGZ="$DEST/data_aishell3.tgz"
    fetch "93/data_aishell3.tgz" "$TGZ" || { echo "[aishell3] DOWNLOAD FAILED"; exit 1; }
    echo "[aishell3] extracting (this takes a while) ..."
    mkdir -p "$DEST/aishell3"
    tar -xzf "$TGZ" -C "$DEST/aishell3" --strip-components=0 || { echo "[aishell3] EXTRACT FAILED (corrupt archive? delete $TGZ and rerun)"; exit 1; }
    # the archive may nest everything under data_aishell3/ — flatten if so
    if [ -d "$DEST/aishell3/data_aishell3" ]; then
        mv "$DEST/aishell3/data_aishell3"/* "$DEST/aishell3/" && rmdir "$DEST/aishell3/data_aishell3"
    fi
    [ "$KEEP_ARCHIVE" = "1" ] || rm -f "$TGZ"
    echo "[aishell3] done: $(find "$DEST/aishell3" -name '*.wav' | wc -l) wavs"
fi

echo "== summary =="
du -sh "$DEST"/* 2>/dev/null
echo "CORPORA DONE"
