# Audio DiT quantization

## Setup


```bash
# 0) venv + this repo's pinned deps (single source of truth: requirements.txt)
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Model repo — scripts import `audiodit`/`batch_inference` from it via PYTHONPATH.
#    Weights auto-download from HF on first use (meituan-longcat/LongCat-AudioDiT-{1B,3.5B}).
git clone https://github.com/meituan-longcat/LongCat-AudioDiT.git ~/workspace/LongCat-AudioDiT

# 2) Eval harness + data assets (all gitignored — fetch these)
git clone https://github.com/BytedanceSpeech/seed-tts-eval eval/seed-tts-eval
git -C eval/seed-tts-eval apply ../../patches/seed_tts_eval.patch   # py3.13 / torch2.12 / transformers5.11 fixes
#    W4A4 needs the real int4 kernel. Clone WITH submodules (cutlass + fast-hadamard-transform), else the
#    build fails on `No module named 'fast_hadamard_transform'` and all W4A4 configs die at import.
git clone --recursive https://github.com/ruikangliu/FlatQuant vendor/flatquant_ref
git -C vendor/flatquant_ref submodule update --init --recursive   # if you cloned without --recursive
#    For CUDA-graph-correct W4A4: apply the default-stream→current-stream fix, then build the kernels.
git -C vendor/flatquant_ref apply ../../patches/flatquant_cudagraph_stream.patch
cd vendor/flatquant_ref && CUDA_HOME=/usr/local/cuda-13.3 python setup.py build_ext --inplace && cd ../..
#    (build_ext also `pip install -e`s third-party/fast-hadamard-transform; takes a few min — CUTLASS int4 GEMM)
#    Build with the SAME python that RUNS the code (the .venv), not a stray system/conda python.
#    CUDA 13.3 gotchas (both hit during a fresh build here):
#      - fast-hadamard 'No module named torch' -> pip build isolation; install it first WITHOUT isolation:
#          .venv/bin/python -m pip install --no-build-isolation --no-deps vendor/flatquant_ref/third-party/fast-hadamard-transform
#      - CUTLASS 'cuda/std/utility: No such file' -> CUDA 13.3 puts CCCL headers under .../include/cccl/;
#          add it:  export CPATH=/usr/local/cuda-13.3/targets/x86_64-linux/include/cccl:$CPATH  before build_ext.
#    Fallback: a prebuilt deploy/_CUDA*.so + fast_hadamard_transform*.so from a matching env
#    (same python 3.13 + torch 2.12+cu130) are ABI-compatible — drop them into deploy/ and .venv site-packages.
#    Seed test sets -> data/seedtts_testset/   (gdrive 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP, a tar)
#    WavLM SIM ckpt  -> eval/ckpt/wavlm_large_finetune.pth   (gdrive 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP)

```

Install the Seed-TTS eval assets before running benchmarks. This installs the
test set and the WavLM checkpoint used by optional SIM evaluation:

```bash
# Download from Google Drive into data/seedtts_testset/ and eval/ckpt/
bash scripts/download_seedtts_testset.sh

# Or reuse existing local unpacked assets:
SOURCE_DIR=/home/xiaoqin_feng/workspace/seed_repro/data/seedtts_testset \
WAVLM_SOURCE=/home/xiaoqin_feng/workspace/seed_repro/eval/ckpt/wavlm_large_finetune.pth \
  bash scripts/download_seedtts_testset.sh

test -f data/seedtts_testset/zh/meta.lst
test -f data/seedtts_testset/en/meta.lst
test -f eval/ckpt/wavlm_large_finetune.pth
```

```bash
source env.sh          # sets PYTHONPATH + repo/data/eval/gen/results paths
```

`seedtts_similarity.py` needs `eval/ckpt/wavlm_large_finetune.pth`; alternatively export
`WAVLM_CKPT=/path/to/wavlm_large_finetune.pth`.

## Run

Layout:

```text
scripts/                 shell launchers for experiments/evaluation
src/audio_dit_quantize/  importable Python package: quantization, generation, eval helpers
data/ eval/ vendor/      external assets, gitignored where large
gen/ results/            generated audio and metric outputs
```

Each experiment line has its own launcher; method logic stays in separate Python modules.
Generation launchers run evaluation automatically after generation. The default
evaluation metrics are CER/WER plus MOS (`utmos` + `dnsmos`).

```bash
# 1) Reproduce the LongCat-AudioDiT Seed-TTS benchmark at fp32.
bash scripts/benchmark_fp32_seedtts.sh 1b
bash scripts/benchmark_fp32_seedtts.sh 3.5b

# 2) FlatQuant W4A4, paper-aligned best-config:
#    per-block + LWC + LAC + add_diag + learned Kronecker.
#    Quant calibration is fixed at data/calib_heldout_hardlike32.lst.
bash scripts/benchmark_flatquant_best_seedtts.sh 1b
bash scripts/benchmark_flatquant_best_seedtts.sh 3.5b

# 3) SVDQuant W4A4 under the current order-free paired protocol.
#    Quant calibration is fixed at data/calib_heldout_hardlike32.lst.
#    Seeds: CALIB_SEED=0 for quant calibration, BASE=1024 for generation.
bash scripts/benchmark_svdquant_seedtts.sh 1b
bash scripts/benchmark_svdquant_seedtts.sh 3.5b

# 4) Efficiency (latency / RTF / VRAM) — W4A4 deploy vs fp32/fp16, and the deploy
#    quality-latency tradeoff. ⚠️ NEEDS AN IDLE GPU (cudagraph timing is contention-sensitive).
#    Needs the built int4 kernels (Setup step 2: apply patch, then build_ext).
bash scripts/benchmark_efficiency.sh 1b            # or: both 10
python -m audio_dit_quantize.efficiency.intrinsic_efficiency  # Layer-1 (BitOps/compression, no GPU)
#    Full design + results + code inventory: docs/efficiency.md

# 5) Step-axis activation precision (full / early / late) — the late-step SIM lever.
#    ONE fixed best-config model; full/early/late only change WHICH ODE steps quantize activations.
#    Key result: LATE recovers timbre/SIM (paired ΔSIM), EARLY (equal budget) does not.
MODE=calibrate bash scripts/benchmark_step_axis_seedtts.sh 1b   # one-time: produce models/bc_1b_model.pt
bash scripts/benchmark_step_axis_seedtts.sh 1b "zh en hard" "full early late"
#    Full matrix + method: docs/quality-metrics-matrix.md
```

Useful smoke-test knob:

```bash
LIMIT=1 bash scripts/benchmark_fp32_seedtts.sh 1b hard
LIMIT=1 bash scripts/benchmark_flatquant_best_seedtts.sh 1b hard
LIMIT=1 bash scripts/benchmark_svdquant_seedtts.sh 1b hard
```

Metric evaluation is a single entry point. By default it computes CER for `zh`
and `hard`, WER for `en`, and MOS (`utmos` + `dnsmos`) for every requested set:

```bash
bash scripts/evaluate_seedtts_metrics.sh gen/paired/fp32 pf_fp32 "zh en hard"
```

Optional speaker similarity can be attached to the same generation run:

```bash
EVAL_METRICS="wer cer mos sim" bash scripts/benchmark_fp32_seedtts.sh 1b "zh en hard"
```

ASR metric evaluation auto-tunes batch parameters from GPU memory by default.
Override only when needed:

```bash
ASR_BATCH_SIZE=32 ASR_BATCH_SIZE_S=300 bash scripts/evaluate_seedtts_metrics.sh gen/paired/fp32 pf_fp32 zh "cer"
```
