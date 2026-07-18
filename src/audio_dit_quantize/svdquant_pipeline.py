"""SVDQuant W4A4 calibration helpers for the LongCat-AudioDiT DiT (faithful-core, fake-quant).

This module is the CALIBRATION library for SVDQuant — capture per-linear inputs across the denoising
trajectory (fp model) -> wrap with SVDQuantLinear -> per-linear calibrate (grid-search smoothing alpha
by output MSE + one-shot SVD low-rank + group-64 INT4). See svdquant_linear.py for the method itself.

Generation is NOT done here: it runs through the shared, protocol-correct entry point
``generate_seedtts.py --mode svdquant`` (per-item seed base+idx, order-free/paired, NaN guard, GPU-range
aware), which imports ``load_calib_items`` / ``calibrate`` / ``load_items`` / ``DATA`` / ``SETS`` below.
A100 = fake-quant quality study (Nunchaku INT4 kernels would give real speed; not wired here).

Note: SVDQuant re-calibrates in-process and has no model save/load, so its generation is single-GPU
(item sharding would recalibrate per shard); see scripts/benchmark_svdquant_seedtts.sh.
"""
import os, time
import torch
from tqdm import tqdm

from batch_inference import infer_one
from . import svdquant_linear as sq
from .paths import CALIB_LST, DATA_DIR, SETS

DATA = str(DATA_DIR)


def load_items(lst):
    d = os.path.dirname(lst)
    out = []
    for line in open(lst):
        line = line.strip()
        if line:
            p = line.split("|")
            out.append((p[0], p[1], os.path.join(d, p[2]), p[3]))
    return out


def load_calib_items():
    if not CALIB_LST.exists():
        raise FileNotFoundError(f"fixed calibration list not found: {CALIB_LST}")
    calib = load_items(CALIB_LST)
    print(f"[calib] list = {len(calib)} items from {CALIB_LST}")
    return calib


@torch.no_grad()
def capture_inputs(model, tok, dev, calib_items, rows_per_linear):
    targets = list(sq._target_linears(model.transformer))
    store = {id(lin): [] for _, _, lin in targets}
    got = {id(lin): 0 for _, _, lin in targets}

    def mk(lin):
        def hook(mod, inp):
            if got[id(lin)] >= rows_per_linear:
                return
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            take = min(x.shape[0], rows_per_linear - got[id(lin)])
            idx = torch.randperm(x.shape[0], device=x.device)[:take]
            store[id(lin)].append(x[idx].float().cpu())
            got[id(lin)] += take
        return hook

    hooks = [lin.register_forward_pre_hook(mk(lin)) for _, _, lin in targets]
    for uid, pt, pwa, gt in tqdm(calib_items, desc="capture svd calib", dynamic_ncols=True):
        infer_one(gt, pt, pwa, model, tok, dev, nfe=16, cfg_strength=4.0, guidance_method="apg")
    for h in hooks:
        h.remove()
    return store


def calibrate(model, tok, dev, calib_items, rows_per_linear, rank, a_bits=4,
              num_iters=sq.NUM_ITERS, a_sym=True):
    print(f"[calib] capturing inputs on {len(calib_items)} prompts ...")
    store = capture_inputs(model, tok, dev, calib_items, rows_per_linear)
    wrapped = sq.wrap_dit(model, w_bits=4, a_bits=a_bits, rank=rank, num_iters=num_iters, a_sym=a_sym)
    print(f"[calib] wrapped {len(wrapped)} linears; SVDQuant calibrate "
          f"((α,β) grid + iterative low-rank, num_iters={num_iters}, a_sym={a_sym}) ...")
    t0 = time.time()
    iterator = tqdm(wrapped, desc="svd calibrate", dynamic_ncols=True)
    for i, (parent, attr, sql) in enumerate(iterator):
        bufs = store.get(id(sql.linear), [])
        if not bufs:
            # no captured activations -> degenerate fallback (identity smoothing wins the grid)
            X = torch.zeros(1, sql.in_features, device=dev)
        else:
            X = torch.cat(bufs, 0).to(dev)
        sql.calibrate(X)
        del X
        store[id(sql.linear)] = None
        iterator.set_postfix_str(f"a={sql.alpha:.2f} b={sql.beta:.2f}")
        if (i + 1) % 40 == 0:
            torch.cuda.empty_cache()
            tqdm.write(f"[calib]  {i+1}/{len(wrapped)} (alpha={sql.alpha:.2f} beta={sql.beta:.2f})  {time.time()-t0:.0f}s")
    torch.cuda.empty_cache()
    print(f"[calib] done in {time.time()-t0:.0f}s")
    return model
