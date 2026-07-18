#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/env.sh" >/dev/null

missing=0

check_file() {
  local f="$1"
  if [ ! -e "$f" ]; then
    echo "[preflight] MISSING: $f"
    missing=1
  else
    echo "[preflight] OK: $f"
  fi
}

echo "[preflight] python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "[preflight] torch runtime"
"$PYTHON_BIN" - <<'PY'
import torch
print('torch', torch.__version__)
print('torch_cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device0', torch.cuda.get_device_name(0))
PY

echo "[preflight] imports"
"$PYTHON_BIN" - <<'PY'
import importlib
mods = [
    'audiodit',
    'batch_inference',
    'flatquant',
    'deploy',
    'fast_hadamard_transform',
    'audio_dit_quantize.generate_seedtts',
]
for m in mods:
    importlib.import_module(m)
    print('import OK:', m)
PY

check_file "$SEED_DATA_DIR/zh/meta.lst"
check_file "$SEED_DATA_DIR/en/meta.lst"
check_file "$SEED_DATA_DIR/zh/hardcase.lst"
check_file "$WAVLM_CKPT"
if ls "$FLATQUANT_REF_DIR"/deploy/_CUDA*.so >/dev/null 2>&1; then
  echo "[preflight] OK: $FLATQUANT_REF_DIR/deploy/_CUDA*.so"
else
  echo "[preflight] MISSING: $FLATQUANT_REF_DIR/deploy/_CUDA*.so"
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  echo "[preflight] FAIL: missing required files"
  exit 2
fi

echo "[preflight] PASS"
