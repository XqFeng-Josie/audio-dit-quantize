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

Install the Seed-TTS test set before running any generation benchmark:

```bash
# Download from Google Drive into data/seedtts_testset/
bash scripts/download_seedtts_testset.sh

# Or reuse an existing local unpacked copy:
SOURCE_DIR=/home/xiaoqin_feng/workspace/seed_repro/data/seedtts_testset \
  bash scripts/download_seedtts_testset.sh

test -f data/seedtts_testset/zh/meta.lst
test -f data/seedtts_testset/en/meta.lst
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
bash scripts/benchmark_svdquant_seedtts.sh 1b
bash scripts/benchmark_svdquant_seedtts.sh 3.5b
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
