#!/usr/bin/env bash
# E1 calibration-pool scoring pass (FP model, per-sample intermediate signals).
# Scores the whole clean pool once per scale; offline analysis then correlates
# set-level scores with the known dual-scale set rankings (docs §1.5).
#
# Usage:  bash scripts/calib/score_pool.sh <1b|3.5b> [offset] [limit]
#   Resumable: existing raw npz are skipped, so re-running after a crash (or
#   sharding by offset/limit on several GPUs into the same out dir) is safe.
#   Final CSV: data/calib_pool/scores_v3/<tag>/pool_scores.csv
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT_DIR/env.sh"
cd "$SEED_REPRO_DIR"

case "${1:-1b}" in
  1b)   model_dir="meituan-longcat/LongCat-AudioDiT-1B";   tag=1b ;;
  3.5b) model_dir="meituan-longcat/LongCat-AudioDiT-3.5B"; tag=3p5b ;;
  *) echo "usage: $0 <1b|3.5b> [offset] [limit]"; exit 1 ;;
esac

"$PYTHON_BIN" -m audio_dit_quantize.calib.score_pool score \
  --model_dir "$model_dir" \
  --out "data/calib_pool/scores_v3/$tag" \
  --offset "${2:-0}" --limit "${3:-0}" \
  --device "$DEVICE"
