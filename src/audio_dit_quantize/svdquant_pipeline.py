"""Calibrate + generate with SVDQuant W4A4 on the LongCat-AudioDiT DiT (faithful-core, fake-quant).

Pipeline: capture per-linear inputs across the denoising trajectory (fp model) -> wrap with
SVDQuantLinear -> per-linear calibrate (grid-search smoothing alpha by output MSE + one-shot SVD
low-rank + group-64 INT4) -> generate Seed sets (same infer_one + seed 1024 as everything else).

See svdquant_linear.py for the method + which details match deepcompressor and which are simplified.
A100 = fake-quant quality study (Nunchaku INT4 kernels would give real speed; not wired here).

Usage:
  python -m audio_dit_quantize.svdquant_pipeline \
      --out_subdir svdquant_full/sq --limit 0 [--model_dir ...] [--rank 32]
"""
import argparse, os, time
import torch, soundfile as sf

import audiodit  # noqa
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
from . import svdquant_linear as sq
from .paths import CALIB_LST, DATA_DIR, GEN_DIR, SETS

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
    for uid, pt, pwa, gt in calib_items:
        infer_one(gt, pt, pwa, model, tok, dev, nfe=16, cfg_strength=4.0, guidance_method="apg")
    for h in hooks:
        h.remove()
    return store


def calibrate(model, tok, dev, calib_items, rows_per_linear, rank, a_bits=4):
    print(f"[calib] capturing inputs on {len(calib_items)} prompts ...")
    store = capture_inputs(model, tok, dev, calib_items, rows_per_linear)
    wrapped = sq.wrap_dit(model, w_bits=4, a_bits=a_bits, rank=rank)
    print(f"[calib] wrapped {len(wrapped)} linears; SVDQuant calibrate (alpha grid + SVD) ...")
    t0 = time.time()
    for i, (parent, attr, sql) in enumerate(wrapped):
        bufs = store.get(id(sql.linear), [])
        if not bufs:
            # no captured activations -> fall back to alpha=0.5 with weight-only stats
            X = torch.zeros(1, sql.in_features, device=dev)
        else:
            X = torch.cat(bufs, 0).to(dev)
        sql.calibrate(X)
        del X
        store[id(sql.linear)] = None
        if (i + 1) % 40 == 0:
            torch.cuda.empty_cache()
            print(f"[calib]  {i+1}/{len(wrapped)} (alpha={sql.alpha:.2f})  {time.time()-t0:.0f}s")
    torch.cuda.empty_cache()
    print(f"[calib] done in {time.time()-t0:.0f}s")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_subdir", required=True)
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calib_rows", type=int, default=512)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--a_bits", type=int, default=4, help="activation bits (4=W4A4 default, 8=W4A8)")
    ap.add_argument("--sets", default="zh,en,hard", help="comma list of Seed sets to generate")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    genroot = os.path.join(str(GEN_DIR), args.out_subdir)

    calib = load_calib_items()
    calibrate(model, tok, dev, calib, args.calib_rows, args.rank, a_bits=args.a_bits)

    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    want = [s.strip() for s in args.sets.split(",") if s.strip()]
    for name in want:
        rel = SETS[name]
        items = load_items(os.path.join(DATA, rel))
        if args.limit:
            items = items[: args.limit]
        outdir = os.path.join(genroot, name)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        for i, (uid, pt, pwa, gt) in enumerate(items):
            op = os.path.join(outdir, f"{uid}.wav")
            if os.path.exists(op):
                continue
            try:
                wav = infer_one(gt, pt, pwa, model, tok, dev, 16, 4.0, "apg")
                sf.write(op, wav, model.config.sampling_rate)
            except Exception as e:
                print(f"[{name} {i+1}] ERR {uid}: {e}")
        print(f"[{name}] {len(items)} items in {time.time()-t0:.0f}s -> {outdir}")
    print("DONE")


if __name__ == "__main__":
    main()
