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
git clone https://github.com/ruikangliu/FlatQuant vendor/flatquant_ref
#    For CUDA-graph-correct W4A4: apply the default-stream→current-stream fix, then rebuild the kernels.
git -C vendor/flatquant_ref apply ../../patches/flatquant_cudagraph_stream.patch
#    (cd vendor/flatquant_ref && CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace)
#    Seed test sets -> data/seedtts_testset/   (gdrive 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP, a tar)
#    WavLM SIM ckpt  -> eval/ckpt/wavlm_large_finetune.pth   (gdrive 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP)

```

```bash
source env.sh          # sets PYTHONPATH + SEED_REPRO_DIR + LONGCAT_DIR + cu130 toolchain
```
1. Reproduce the LongCat-AudioDiT Seed-TTS benchmark at fp32

2. FlatQuant W4A4, paper-aligned best-config 
<!-- Per-block reconstruction + LWC + LAC + add_diag + learned Kronecker transform, `hardlike32` calibration. -->

