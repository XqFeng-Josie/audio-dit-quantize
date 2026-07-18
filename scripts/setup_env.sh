#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

WITH_ASSETS="${WITH_ASSETS:-1}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

log() {
  echo "[setup] $*"
}

pick_python() {
  if command -v python3.13 >/dev/null 2>&1; then
    echo "python3.13"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  command -v python
}

ensure_venv() {
  if [ ! -x "$PYTHON_BIN" ]; then
    local py
    py="$(pick_python)"
    log "creating venv with $py"
    "$py" -m venv .venv
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
}

detect_driver_cuda() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local line
  line="$(nvidia-smi 2>/dev/null | grep -m1 'CUDA Version' || true)"
  if [ -z "$line" ]; then
    return 1
  fi
  echo "$line" | sed -E 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/'
}

pick_torch_index() {
  local drv="${1:-}"
  local major minor

  if [ -z "$drv" ]; then
    echo "https://download.pytorch.org/whl/cu124"
    return
  fi

  major="${drv%%.*}"
  minor="${drv##*.}"

  if [ "$major" -ge 13 ]; then
    echo "https://download.pytorch.org/whl/cu128"
    return
  fi

  if [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; then
    echo "https://download.pytorch.org/whl/cu128"
    return
  fi
  if [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then
    echo "https://download.pytorch.org/whl/cu126"
    return
  fi
  if [ "$major" -eq 12 ] && [ "$minor" -ge 4 ]; then
    echo "https://download.pytorch.org/whl/cu124"
    return
  fi
  if [ "$major" -eq 12 ] && [ "$minor" -ge 1 ]; then
    echo "https://download.pytorch.org/whl/cu121"
    return
  fi
  echo "https://download.pytorch.org/whl/cu118"
}

install_python_deps() {
  local req_no_torch
  req_no_torch="$(mktemp)"
  grep -Ev '^(torch|torchaudio)==|^# --- core' requirements.txt > "$req_no_torch"

  log "installing repo requirements (excluding torch/torchaudio)"
  "$PYTHON_BIN" -m pip install -r "$req_no_torch"
  rm -f "$req_no_torch"

  local drv idx
  drv="$(detect_driver_cuda || true)"
  idx="$(pick_torch_index "$drv")"
  log "driver CUDA: ${drv:-unknown}, installing torch stack from $idx"
  "$PYTHON_BIN" -m pip install --upgrade --index-url "$idx" torch==2.12.0 torchaudio==2.11.0
}

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"
  local extra_args="${3:-}"

  if [ -d "$target_dir/.git" ]; then
    log "repo exists: $target_dir"
    return
  fi

  log "cloning $repo_url -> $target_dir"
  git clone $extra_args "$repo_url" "$target_dir"
}

apply_patch_if_needed() {
  local repo_dir="$1"
  local patch_file="$2"

  if git -C "$repo_dir" apply --check "$patch_file" >/dev/null 2>&1; then
    log "applying patch $(basename "$patch_file") in $repo_dir"
    git -C "$repo_dir" apply "$patch_file"
    return
  fi

  if git -C "$repo_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    log "patch already applied: $(basename "$patch_file")"
    return
  fi

  log "patch state unclear for $(basename "$patch_file"); please inspect manually"
}

pick_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME" ]; then
    echo "$CUDA_HOME"
    return
  fi

  local cand
  cand="$(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V | tail -n1 || true)"
  if [ -n "$cand" ] && [ -d "$cand" ]; then
    echo "$cand"
    return
  fi

  if [ -d /usr/local/cuda ]; then
    echo "/usr/local/cuda"
    return
  fi

  echo ""
}

setup_repos_and_kernels() {
  clone_if_missing "https://github.com/meituan-longcat/LongCat-AudioDiT.git" "../LongCat-AudioDiT"
  clone_if_missing "https://github.com/BytedanceSpeech/seed-tts-eval" "eval/seed-tts-eval"
  clone_if_missing "https://github.com/ruikangliu/FlatQuant" "vendor/flatquant_ref" "--recursive"

  apply_patch_if_needed "eval/seed-tts-eval" "../../patches/seed_tts_eval.patch"
  git -C vendor/flatquant_ref submodule update --init --recursive
  apply_patch_if_needed "vendor/flatquant_ref" "../../patches/flatquant_cudagraph_stream.patch"

  "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps vendor/flatquant_ref/third-party/fast-hadamard-transform

  local cuda_home
  cuda_home="$(pick_cuda_home)"
  if [ -z "$cuda_home" ]; then
    log "no CUDA toolkit found under /usr/local/cuda*; cannot build FlatQuant kernels"
    return 1
  fi

  export CUDA_HOME="$cuda_home"
  export PATH="$CUDA_HOME/bin:$PATH"

  if [ -d "$CUDA_HOME/targets/x86_64-linux/include/cccl" ]; then
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include/cccl${CPATH:+:$CPATH}"
  fi

  log "note: FlatQuant setup.py may print a non-fatal editable-build warning for fast_hadamard_transform"
  log "building FlatQuant deploy extension with CUDA_HOME=$CUDA_HOME"
  (
    cd vendor/flatquant_ref
    PIP_NO_BUILD_ISOLATION=1 "$PYTHON_BIN" setup.py build_ext --inplace
  )
}

run_preflight() {
  log "running preflight"
  bash scripts/preflight_env.sh
}

install_assets_if_requested() {
  if [ "$WITH_ASSETS" = "1" ]; then
    log "installing Seed-TTS eval assets"
    bash scripts/download_seedtts_testset.sh
  else
    log "skipping asset install (WITH_ASSETS=$WITH_ASSETS)"
  fi
}

main() {
  ensure_venv
  install_python_deps
  setup_repos_and_kernels
  install_assets_if_requested
  run_preflight
  log "done"
}

main "$@"
