"""Calibrate + generate W4A4 on LongCat-AudioDiT with the *best-config* FlatQuant:
**per-BLOCK reconstruction** (OmniQuant/BRECQ-style sequential block output MSE) +
**LAC** (learnable activation clipping) + **add_diag** (learnable per-input-channel scaling),
on top of LWC + the learnable Kronecker transform.

Per-block protocol (standard PTQ block reconstruction):
  1. capture block-0 input x + the *shared* per-forward conditioning (t, cond, mask, cond_mask,
     rope, cond_rope, adaln_global_out) across a few calib generations (timestep + content spread).
  2. wrap all target linears (lwc+lac+add_diag).
  3. for each block in order:
       fp_out  = fp_block(inps)                 # true fp target on the (drifted) quantized input
       train block's quant params to match fp_out (block output MSE)
       freeze; inps = quant_block(inps)         # advance — errors accumulate, as at deploy

Usage:
  python -m audio_dit_quantize.flatquant_best \
      --model_dir meituan-longcat/LongCat-AudioDiT-3.5B --out_subdir flatquant_pb_3.5b \
      --sets hard --limit 0
"""
import argparse, math, os, time
import torch, soundfile as sf
from tqdm import tqdm

import audiodit  # noqa
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
from . import flatquant_layers as fq
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


# ── recursive device mover (handles tensors / tuples / lists / dicts / None) ──
def _move(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(_move(o, dev) for o in obj)
    if isinstance(obj, list):
        return [_move(o, dev) for o in obj]
    if isinstance(obj, dict):
        return {k: _move(v, dev) for k, v in obj.items()}
    return obj


# ── capture block-0 input + shared conditioning across calib generations ──────
@torch.no_grad()
def capture_block_inputs(model, tok, dev, calib_items, max_seqs, per_item_keep, nfe=16):
    block0 = model.transformer.blocks[0]
    cur = []  # current item's (x_cpu, cond_kwargs_cpu, prompt_dur) captures
    state = {"pd": 0}  # prompt latent length (prompt/generation-region boundary) of the current item

    def hook(_mod, _args, kwargs):
        kw = dict(kwargs)
        x = kw.pop("x").detach().float().cpu()
        cond = {k: _move(v, "cpu") for k, v in kw.items()}   # t, cond, mask, cond_mask, rope, cond_rope, adaln_global_out
        cur.append((x, cond, state["pd"]))

    # monkeypatch encode_prompt_audio (returns (latent, prompt_dur)) to record the boundary per item
    _orig_enc = model.encode_prompt_audio
    def _enc(pa):
        lat, pd = _orig_enc(pa); state["pd"] = int(pd); return lat, pd

    store = []
    try:
        model.encode_prompt_audio = _enc
        for it in tqdm(calib_items, desc="capture calib", dynamic_ncols=True):
            cur.clear()
            h = block0.register_forward_pre_hook(hook, with_kwargs=True)
            try:
                infer_one(it[3], it[1], it[2], model, tok, dev, nfe=nfe, cfg_strength=4.0, guidance_method="apg")
            finally:
                h.remove()
            if not cur:
                continue
            # strided subset across this item's ODE trajectory (timestep spread)
            n = len(cur)
            k = min(per_item_keep, n)
            idx = [round(i * (n - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
            for j in sorted(set(idx)):
                store.append(cur[j])
            if len(store) >= max_seqs:
                break
    finally:
        del model.encode_prompt_audio     # restore the class method
    return store[:max_seqs]


def _block_wrappers(block):
    return [m for m in block.modules() if isinstance(m, fq.FlatQuantLinear)]


def _chanbal_weight(fp_outs, dev):
    """Per-(hidden)channel inverse-variance weight for the block-output MSE — the per-block analog of
    the docs/11 channel-balanced loss (normalize each output channel by its variance so high-variance
    channels don't dominate). Computed once per block from the fp target, mean-normalized to keep scale."""
    s = ss = None; n = 0
    for f in fp_outs:
        ff = f.reshape(-1, f.shape[-1]).double()
        s = ff.sum(0) if s is None else s + ff.sum(0)
        ss = (ff ** 2).sum(0) if ss is None else ss + (ff ** 2).sum(0)
        n += ff.shape[0]
    var = (ss / n - (s / n) ** 2).clamp(min=1e-6)
    w = (1.0 / var).float()
    return (w / w.mean()).to(dev)          # [dim]


def calibrate_perblock(model, store, dev, steps=200, mb=4, lr=5e-3, clip_lr=5e-2, loss_type="mse",
                       region_gen_w=1.0, region_prompt_w=1.0):
    assert loss_type in ("mse", "chanbal")
    blocks = model.transformer.blocks
    for p in model.parameters():              # only the per-block quant params train; freeze the rest
        p.requires_grad_(False)
    inps = [x for (x, _, _) in store]         # block-0 inputs (CPU)
    conds = [c for (_, c, _) in store]        # shared conditioning per seq (CPU)
    pds = [pd for (_, _, pd) in store]        # prompt/generation-region boundary per seq
    n = len(inps)
    # per-TOKEN region weight (SIM<-prompt region / CER<-generation region): up-weight one span of the
    # block-output MSE. gen region = tokens [pd:] (denoised synthesis, drives CER/coverage); prompt region
    # = [:pd] (voice conditioning, drives SIM). mean-normalized to preserve loss scale.
    region = (region_gen_w != 1.0 or region_prompt_w != 1.0)
    if region:
        wtoks = []
        for j in range(n):
            T = inps[j].shape[1]
            pd = min(max(int(pds[j]), 0), T)
            w = torch.full((T,), float(region_prompt_w), dtype=torch.float32)
            w[pd:] = float(region_gen_w)
            wtoks.append((w / w.mean().clamp(min=1e-6)).to(dev))
    else:
        wtoks = None
    print(f"[pb] {n} calib seqs, {len(blocks)} blocks, {steps} steps x mb{mb} each"
          + (f" | region gen_w={region_gen_w} prompt_w={region_prompt_w}" if region else ""))
    t0 = time.time()
    for bi, blk in enumerate(blocks):
        ws = _block_wrappers(blk)
        if not ws:
            continue
        # 1) full-precision block target on the current (drifted) input
        for w in ws:
            w.enable_quant = False
        with torch.no_grad():
            fp_outs = [blk(x=_move(inps[j], dev), **_move(conds[j], dev)).detach().float().cpu()
                       for j in range(n)]
        for w in ws:
            w.enable_quant = True
        cbw = _chanbal_weight(fp_outs, dev) if loss_type == "chanbal" else None   # [dim] or None
        # 2) train block's quant params to match fp_out (block output MSE, optionally chan-balanced)
        trans_p, clip_p = [], []
        for w in ws:
            if w.use_trans:
                trans_p += list(w.trans.parameters())
            if w.lwc:
                clip_p += [w.clip_w_max, w.clip_w_min]
            if w.lac:
                clip_p += [w.aq.clip_factor_a_max, w.aq.clip_factor_a_min]
        for p in trans_p + clip_p:
            p.requires_grad_(True)
        opt = torch.optim.AdamW([{"params": trans_p, "lr": lr},
                                 {"params": clip_p, "lr": clip_lr}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for s in range(steps):
            perm = torch.randperm(n)[:mb].tolist()
            opt.zero_grad()
            for w in ws:
                w.set_wq_cache()        # one differentiable qweight reused across the mini-batch
            loss = 0.0
            for j in perm:
                out = blk(x=_move(inps[j], dev), **_move(conds[j], dev))
                tgt = fp_outs[j].to(dev)
                se = (out - tgt) ** 2
                if wtoks is not None:
                    se = wtoks[j][None, :, None] * se       # per-token region weight (broadcast over channels)
                loss = loss + (se.mean() if cbw is None else (cbw * se).mean())
            loss = loss / len(perm)
            (loss / loss.detach().clamp(min=1e-12)).backward()
            opt.step(); sched.step()
            for w in ws:
                w.clear_wq_cache()
        # 3) freeze, then advance inputs through the now-quantized block
        for w in ws:
            w.freeze()
        with torch.no_grad():
            inps = [blk(x=_move(inps[j], dev), **_move(conds[j], dev)).detach().float().cpu()
                    for j in range(n)]
        del fp_outs, opt
        torch.cuda.empty_cache()
        if (bi + 1) % 4 == 0 or bi == len(blocks) - 1:
            print(f"[pb]  block {bi+1}/{len(blocks)} ({time.time()-t0:.0f}s)")
    print(f"[pb] done in {time.time()-t0:.0f}s")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_subdir", required=True)
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--sets", default="hard", help="comma list of {zh,en,hard} or 'all'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_seqs", type=int, default=64)
    ap.add_argument("--per_item_keep", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--clip_lr", type=float, default=5e-2)
    ap.add_argument("--loss", default="mse", choices=["mse", "chanbal"],
                    help="per-block reconstruction objective: mse (default) or chanbal "
                         "(per-hidden-channel inverse-variance weighted block-output MSE)")
    ap.add_argument("--region_gen_w", type=float, default=1.0,
                    help="per-token weight for the GENERATION region ([prompt_dur:]) in the block MSE. "
                         ">1 up-weights synthesis (LEAD 2, targets CER/coverage). 1.0 = off.")
    ap.add_argument("--region_prompt_w", type=float, default=1.0,
                    help="per-token weight for the PROMPT region ([:prompt_dur]). >1 up-weights the voice "
                         "conditioning (targets SIM/timbre). 1.0 = off.")
    ap.add_argument("--a_bits", type=int, default=4)
    ap.add_argument("--a_sym", action="store_true")
    ap.add_argument("--no_lac", action="store_true")
    ap.add_argument("--no_diag", action="store_true")
    ap.add_argument("--base", type=int, default=1024,
                    help="per-item seed = base + idx (order-free protocol; pairs with fp32/flat gens)")
    ap.add_argument("--calib_seed", type=int, default=None,
                    help="pin the calibration draw (capture noise + training randperm) for reproducibility; "
                         "None = uncontrolled RNG (legacy behaviour). Set it so zh/en/hard from one launch share one calib.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    genroot = os.path.join(str(GEN_DIR), args.out_subdir)
    if args.calib_seed is not None:
        torch.manual_seed(args.calib_seed); torch.cuda.manual_seed_all(args.calib_seed)
        print(f"[pb] calibration pinned to seed {args.calib_seed}")

    if not CALIB_LST.exists():
        raise FileNotFoundError(f"fixed calibration list not found: {CALIB_LST}")
    calib = load_items(CALIB_LST)
    print(f"[pb] calib = {len(calib)} items from {CALIB_LST}")
    print(f"[pb] capturing block-0 activations on {len(calib)} prompts ...")
    store = capture_block_inputs(model, tok, dev, calib, args.max_seqs, args.per_item_keep)
    print(f"[pb] captured {len(store)} sequences")

    fq.wrap_dit(model, w_bits=4, a_bits=args.a_bits, use_trans=True, lwc=True,
                a_sym=args.a_sym, lac=not args.no_lac, add_diag=not args.no_diag)
    calibrate_perblock(model, store, dev, steps=args.steps, mb=args.mb,
                       lr=args.lr, clip_lr=args.clip_lr, loss_type=args.loss,
                       region_gen_w=args.region_gen_w, region_prompt_w=args.region_prompt_w)

    sel = list(SETS) if args.sets == "all" else [s.strip() for s in args.sets.split(",")]
    for name in sel:
        items = load_items(os.path.join(DATA, SETS[name]))
        if args.limit:
            items = items[: args.limit]
        outdir = os.path.join(genroot, name)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        iterator = tqdm(
            enumerate(items),
            total=len(items),
            desc=f"gen flat/{name}",
            dynamic_ncols=True,
        )
        for idx, (uid, pt, pwa, gt) in iterator:
            op = os.path.join(outdir, f"{uid}.wav")
            if os.path.exists(op):
                iterator.set_postfix_str("skip existing")
                continue
            torch.manual_seed(args.base + idx); torch.cuda.manual_seed(args.base + idx)  # per-item, order-free
            try:
                wav = infer_one(gt, pt, pwa, model, tok, dev, 16, 4.0, "apg")
                sf.write(op, wav, model.config.sampling_rate)
                iterator.set_postfix_str(uid[:32])
            except Exception as e:
                tqdm.write(f"[{name} {idx}] ERR {uid}: {e}")
        print(f"[{name}] {len(items)} items in {time.time()-t0:.0f}s -> {outdir}")
    print("DONE")


if __name__ == "__main__":
    main()
