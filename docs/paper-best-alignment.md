# Paper-best configuration alignment (FlatQuant / SVDQuant / QuaRot)

Date: 2026-07-17. Sources: `vendor/flatquant_ref`, `vendor/deepcompressor_ref`, `vendor/quarot_ref`
(+ papers in `vendor/papers/`). Every official value below carries a file:line in the vendor repo.

Goal: all three W4A4 methods run at **their own paper-best configuration** so the method-selection
comparison is not a strawman. Legacy (pre-alignment) behaviour stays reachable via flags for
ablation rows. Shared across methods: same target linears (`self_attn|cross_attn|ffn` GEMMs),
same fixed calib list `data/calib_heldout_hardlike32.lst`, same per-item-seeded generation protocol.

## QuaRot — upgraded RTN → GPTQ (paper headline: `fake_quant/README.md:37`)

| Parameter | Official paper-best | Ours now | Note |
|---|---|---|---|
| Weight algorithm | GPTQ (percdamp 0.01, blocksize 128, groupsize -1, act_order off) | same (`quarot_linear._GPTQ`) | `--mode quarot_gptq`; RTN kept as `--mode quarot` baseline |
| Weight clip | `--w_clip` MSE grid search (norm 2.4, grid 100, maxshrink 0.8) | same (`_find_wscale(mse=True)`) | |
| Sym INT4 clamp | **[-8, 7]** (`quant_utils.sym_quant`) | same (`_sym_qdq`) — was [-7,7] | fixed for weights AND activations (also affects legacy RTN mode) |
| Activations | per-token dynamic symmetric, clip 1.0 | same | |
| Rotation | randomized Hadamard, fixed seed | same | |
| GPTQ Hessian | rotated fp input, sequential per-layer, act quant off during calib, TF32 off | same (`calibrate_gptq`, per-block) | calib = 64×2 block-0 capture (flatquant recipe) for cross-method budget parity; official = wikitext2 128×2048 |
| Global residual rotation | fused via computational invariance | per-linear rotation only | **architectural**: AdaLN breaks the invariance; documented, not alignable |

## SVDQuant — upgraded one-shot → iterative (deepcompressor `configs/svdquant/{__default__,int4}.yaml`)

| Parameter | Official paper-best | Ours now | Note |
|---|---|---|---|
| Low-rank refinement | num_iters 100, early_stop, best-so-far; alternating SVD(W′−qw) ↔ qw=Q(W′−LR); OutputsError with act quant active; SVD float64 | same (`SVDQuantLinear._refine`) | `--svd_iters 1` = legacy one-shot ablation |
| Smoothing grid | 39 candidates: (0,0) + (α,0) + (α,1−α), α∈{1..19}/20 (`beta:-2`, num_grids 20) | same | was 19 two-sided only; smoke test shows (α,0) family actually wins |
| Act quant | per-group-64 dynamic **symmetric** sint4 (no zero-point; = Nunchaku deploy format) | same default | `--svd_asym` = legacy per-group-zero-point ablation |
| Weights | per-group-64 symmetric INT4, rank 32, fp16 low-rank branch | unchanged (already matched) | |
| GPTQ residual overlay | NOT in headline SVDQ row (separate `gptq.yaml`) | omitted | correct |
| Static activation shift | only post-nonlinearity inputs, folded into bias | omitted | minor, documented |
| Calib rows/linear | 128 prompts × all timesteps | `--svd_rows` default 2048 (was 512) | ⚠ old `models/svd_*.pt` are STALE — delete to recalibrate |

## FlatQuant — verified, 3 deviations fixed (`scripts/llama-2/llama-2-7b/w4a4kv4.sh` + `flatquant/train_utils.py`)

Already exactly matching: trans/diag lr 5e-3, LWC/LAC lr 5e-2 (=lr×10), AdamW defaults, cosine
schedule, loss/loss.detach() normalization, LWC/LAC init 4.0, W4 per-channel sym, random-orthogonal
Kronecker init, block-level output-MSE objective.

| Parameter | Official paper-best | Ours now | Note |
|---|---|---|---|
| Act symmetry | per-token **symmetric** (`a_asym` default False) | symmetric default — was asym | `--a_asym` = legacy; sym also matches the deploy int4 kron kernel |
| diag_scale init | sq_style, α=0.3: `w_smax^0.7 / x_smax^0.3` from calib absmax | implemented (`begin_smax`/`init_diag_scale`) — was ones | `--no_diag_init` = legacy |
| Input propagation | drift-free: FP inputs/targets per block (`train_utils.py:155`) | drift-free default — was BRECQ-style quantized drift | `--drift` = legacy (deploy-realistic error accumulation) |
| Cosine eta_min | flat_lr×1e-3 | added | |
| Budget | 480 steps × bsz4 over 128 seqs / layer | 200 steps × mb4 over 64 seqs (kept; audio-DiT adaptation) | raise via `--steps/--max_seqs` if quality-limited |
| Transform sharing | module-level (ln_trans shared q/k/v etc.) | per-linear | structural DiT adaptation, documented |

⚠ The recorded `models/bc_*.pt` were calibrated with the legacy config (asym + drift + ones-diag).
Reproduce them with `--a_asym --drift --no_diag_init`; new paper-best models should be calibrated
fresh (and step-axis / deploy tooling that loads bc_*.pt keeps working — old pickles still load).

## Stale-artifact checklist before re-running comparisons

1. delete `models/svd_*.pt` (legacy one-shot/asym calibration)
2. recalibrate FlatQuant canonical models under the new defaults (or pass legacy flags to reproduce)
3. QuaRot-GPTQ canonical models: `bash scripts/benchmark/benchmark_quarot_gptq_seedtts.sh` creates `models/qgptq_*.pt`
4. keep legacy rows (quarot-RTN, svd one-shot, flatquant-drift) as labeled ablations, not headline numbers
