"""Calibrate + generate W4A4 on LongCat-AudioDiT with the *best-config* FlatQuant:
**per-BLOCK reconstruction** (OmniQuant/BRECQ-style sequential block output MSE) +
**LAC** (learnable activation clipping) + **add_diag** (learnable per-input-channel scaling),
on top of LWC + the learnable Kronecker transform.

Per-block protocol (official FlatQuant train_utils.py, drift-free by default):
  1. capture block-0 input x + the *shared* per-forward conditioning (t, cond, mask, cond_mask,
     rope, cond_rope, adaln_global_out) across a few calib generations (timestep + content spread).
  2. wrap all target linears (lwc+lac+add_diag).
  3. for each block in order:
       fp_out  = fp_block(inps)   # fp target; also accumulates act absmax for the sq_style diag init
       init diag_scale (sq_style, alpha=0.3) from calib absmax   [official diag_init]
       train block's quant params to match fp_out (block output MSE)
       freeze; inps = fp_out      # official: propagate the FP output — NO quantization drift
     (--drift restores the legacy BRECQ/OmniQuant-style advance through the QUANTIZED block,
      i.e. deploy-realistic error accumulation; the recorded bc_* models used that protocol.)

Usage:
  python -m audio_dit_quantize.flatquant_best \
      --model_dir meituan-longcat/LongCat-AudioDiT-3.5B --out_subdir flatquant_pb_3.5b \
      --sets hard --limit 0
"""
import argparse, json, math, os, time
import numpy as np
import torch, soundfile as sf
from tqdm import tqdm

import audiodit  # noqa
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
from . import flatquant_layers as fq
from .paths import CALIB_LST, DATA_DIR, GEN_DIR, SETS, bc_model_path

DATA = str(DATA_DIR)


def _valid_wav(wav):
    """Finite + not degenerate all-(near)zero. Guards against a broken W4A4 draw silently writing
    NaN/all-zero wavs that the metric harness would then score as a real generation (audit F3)."""
    arr = np.asarray(wav)
    return arr.size > 0 and np.isfinite(arr).all() and float(np.abs(arr).max()) > 1e-6


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
def capture_block_inputs(model, tok, dev, calib_items, max_seqs, per_item_keep, nfe=16,
                         snap_place="spread", return_steps=False):
    """return_steps=True additionally returns the per-sequence NETWORK-STEP index (capture index // 2,
    because CFG runs cond+uncond forwards per ODE step — same step convention as generate_step_axis).
    Default return shape is unchanged (callers that unpack 3-tuples are unaffected)."""
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

    store, steps_of = [], []
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
            # snapshot subset of this item's ODE trajectory. "spread" (canonical) = endpoint-inclusive
            # stride; note that at per_item_keep=2 this means ONLY the first+last step — the middle
            # steps are never seen by calibration. "early"/"late" = the first/last k steps (L2 placement).
            n = len(cur)
            k = min(per_item_keep, n)
            if snap_place == "early":
                idx = list(range(k))
            elif snap_place == "late":
                idx = list(range(n - k, n))
            else:  # spread
                idx = [round(i * (n - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
            for j in sorted(set(idx)):
                store.append(cur[j])
                steps_of.append(j // 2)   # capture index -> network step (2 CFG forwards per step)
            if len(store) >= max_seqs:
                break
    finally:
        del model.encode_prompt_audio     # restore the class method
    return (store[:max_seqs], steps_of[:max_seqs]) if return_steps else store[:max_seqs]


def _block_wrappers(block):
    return [m for m in block.modules() if isinstance(m, fq.FlatQuantLinear)]


def _chanbal_weight(fp_outs, dev, invert=True):
    """Per-(hidden)channel variance-based weight for the block-output MSE, computed once per block from
    the fp target, mean-normalized to keep scale. invert=True -> inverse-variance (chanbal, docs/11:
    high-variance channels don't dominate). invert=False -> variance (varw, M3): EMPHASIZE high-variance
    channels — the reversed-direction hypothesis (§4.1: 1/var hurt at per-block; test if var helps)."""
    s = ss = None; n = 0
    for f in fp_outs:
        ff = f.reshape(-1, f.shape[-1]).double()
        s = ff.sum(0) if s is None else s + ff.sum(0)
        ss = (ff ** 2).sum(0) if ss is None else ss + (ff ** 2).sum(0)
        n += ff.shape[0]
    var = (ss / n - (s / n) ** 2).clamp(min=1e-6)
    w = (1.0 / var if invert else var).float()
    return (w / w.mean()).to(dev)          # [dim]


def calibrate_perblock(model, store, dev, steps=200, mb=4, lr=5e-3, clip_lr=5e-2, loss_type="mse",
                       region_gen_w=1.0, region_prompt_w=1.0, snap_w=None, chan_w=None,
                       drift=False, diag_alpha=0.3,
                       diag_init=True, stats_out=None, stats_meta=None):
    """stats_out: optional JSON path — records per-block first/last/min training loss. These are
    cheap calibration-set-quality signals for the P1 proxy-feature regression (computable without
    any generation/eval). Pure logging: the training math is untouched."""
    assert loss_type in ("mse", "chanbal", "varw", "percw")
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
    # per-SEQUENCE snapshot weight (L-B, perception-guided loss: up-weight snapshots from the ODE steps
    # the task says matter — late steps carry SIM). Mean-normalized to preserve loss scale.
    if snap_w is not None:
        assert len(snap_w) == n, f"snap_w len {len(snap_w)} != n seqs {n}"
        _m = sum(snap_w) / n
        snap_w = [float(w) / max(_m, 1e-6) for w in snap_w]
    # per-(block, hidden-channel) PERCEPTUAL weight (L-C): fixed guidance from calib.spk_saliency,
    # mean-normalized per block exactly like _chanbal_weight so percw is comparable with chanbal/varw.
    if loss_type == "percw":
        assert chan_w is not None, "loss_type percw needs chan_w (--chan_w_file)"
        assert chan_w.shape[0] == len(blocks), \
            f"chan_w has {chan_w.shape[0]} blocks, model has {len(blocks)}"
        chan_w = [(w / w.mean().clamp(min=1e-12)).float().to(dev) for w in chan_w]
    print(f"[pb] {n} calib seqs, {len(blocks)} blocks, {steps} steps x mb{mb} each"
          + f" | {'DRIFT (legacy)' if drift else 'fp-propagation (official)'}"
          + (f" | sq_style diag init a={diag_alpha}" if diag_init else "")
          + (f" | region gen_w={region_gen_w} prompt_w={region_prompt_w}" if region else "")
          + (f" | snap_w norm range [{min(snap_w):.3f}, {max(snap_w):.3f}]" if snap_w is not None else ""))
    t0 = time.time()
    stats = []
    for bi, blk in enumerate(blocks):
        ws = _block_wrappers(blk)
        if not ws:
            continue
        # 1) full-precision block target on the current input; the same fp pass collects each
        #    linear's activation absmax for the official sq_style diag_scale init
        for w in ws:
            w.enable_quant = False
            if diag_init:
                w.begin_smax()
        with torch.no_grad():
            fp_outs = [blk(x=_move(inps[j], dev), **_move(conds[j], dev)).detach().float().cpu()
                       for j in range(n)]
        for w in ws:
            w.enable_quant = True
            if diag_init:
                w.init_diag_scale(alpha=diag_alpha)
        if loss_type == "percw":
            cbw = chan_w[bi]                                    # [dim] perceptual guidance (L-C)
        else:
            cbw = (_chanbal_weight(fp_outs, dev, invert=(loss_type == "chanbal"))
                   if loss_type in ("chanbal", "varw") else None)   # [dim] or None
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
        # eta_min = flat_lr*1e-3: official train_utils.py cosine floor
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-3)
        blk_losses = []
        for s in range(steps):
            perm = torch.randperm(n)[:mb].tolist()
            opt.zero_grad()
            # grad explicitly enabled: some callers run under @torch.no_grad() (generate_step_axis)
            with torch.enable_grad():
                for w in ws:
                    w.set_wq_cache()        # one differentiable qweight reused across the mini-batch
                loss = 0.0
                for j in perm:
                    out = blk(x=_move(inps[j], dev), **_move(conds[j], dev))
                    tgt = fp_outs[j].to(dev)
                    se = (out - tgt) ** 2
                    if wtoks is not None:
                        se = wtoks[j][None, :, None] * se       # per-token region weight (broadcast over channels)
                    contrib = se.mean() if cbw is None else (cbw * se).mean()
                    if snap_w is not None:
                        contrib = snap_w[j] * contrib           # per-sequence snapshot-step weight (L-B)
                    loss = loss + contrib
                loss = loss / len(perm)
                blk_losses.append(float(loss.detach()))
                (loss / loss.detach().clamp(min=1e-12)).backward()
            opt.step(); sched.step()
            for w in ws:
                w.clear_wq_cache()
        # 3) freeze, then advance inputs to the next block
        for w in ws:
            w.freeze()
        if drift:
            # legacy: advance through the QUANTIZED block — errors accumulate, as at deploy
            with torch.no_grad():
                inps = [blk(x=_move(inps[j], dev), **_move(conds[j], dev)).detach().float().cpu()
                        for j in range(n)]
        else:
            # official FlatQuant: propagate the FP output (train_utils.py fp_inps<->fp_outs swap)
            inps = fp_outs
        del fp_outs, opt
        torch.cuda.empty_cache()
        k = min(20, len(blk_losses))
        stats.append({"block": bi, "loss_first": blk_losses[0],
                      "loss_last": sum(blk_losses[-k:]) / k, "loss_min": min(blk_losses)})
        print(f"[pb]  block {bi+1}/{len(blocks)} loss {stats[-1]['loss_first']:.3e}"
              f"->{stats[-1]['loss_last']:.3e} ({time.time()-t0:.0f}s)")
    print(f"[pb] done in {time.time()-t0:.0f}s")
    if stats_out:
        payload = {"meta": stats_meta or {}, "blocks": stats,
                   "sum_loss_last": sum(b["loss_last"] for b in stats),
                   "sum_loss_first": sum(b["loss_first"] for b in stats)}
        os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
        with open(stats_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[pb] block-loss stats -> {stats_out}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_subdir", required=True)
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--sets", default="hard", help="comma list of {zh,en,hard} or 'all'")
    ap.add_argument("--limit", type=int, default=0, help="items per set from --offset; 0 = to end of set")
    ap.add_argument("--offset", type=int, default=0,
                    help="start item index per set (multi-GPU item sharding). Seed = base + GLOBAL index "
                         "(offset+local) so a sharded gen is identical to a full gen. Generation only; "
                         "calibration is unaffected (run it once, single-GPU, with --load_model here).")
    ap.add_argument("--max_seqs", type=int, default=64)
    ap.add_argument("--per_item_keep", type=int, default=2,
                    help="activation snapshots kept per calib prompt. 2 = the canonical recipe that produced the "
                         "fixed bc_*.pt models (so --save_model reproduces them + --load_model matches).")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--clip_lr", type=float, default=5e-2)
    ap.add_argument("--loss", default="mse", choices=["mse", "chanbal", "varw", "percw"],
                    help="per-block reconstruction objective: mse (default) / chanbal "
                         "(per-hidden-channel inverse-variance weighted) / varw (M3: variance-weighted, "
                         "the reversed-direction channel-sensitivity hypothesis) / percw (L-C: "
                         "speaker-perceptual saliency weights from --chan_w_file)")
    ap.add_argument("--chan_w_file", default=None,
                    help="[percw] npz from calib.spk_saliency with sal [n_blocks, d]")
    ap.add_argument("--region_gen_w", type=float, default=1.0,
                    help="per-token weight for the GENERATION region ([prompt_dur:]) in the block MSE. "
                         ">1 up-weights synthesis (LEAD 2, targets CER/coverage). 1.0 = off.")
    ap.add_argument("--region_prompt_w", type=float, default=1.0,
                    help="per-token weight for the PROMPT region ([:prompt_dur]). >1 up-weights the voice "
                         "conditioning (targets SIM/timbre). 1.0 = off.")
    ap.add_argument("--snap_place", default="spread", choices=["spread", "early", "late"],
                    help="which ODE steps the per-item activation snapshots come from. spread (canonical) "
                         "= endpoint-inclusive stride (at per_item_keep=2: first+last step ONLY); "
                         "early/late = the first/last per_item_keep steps (L2 placement arms).")
    ap.add_argument("--snap_late_w", type=float, default=1.0,
                    help="L-B (perception-guided loss): per-sequence weight for snapshots from LATE5 "
                         "network steps (10-14; at per_item_keep=2 spread = the last-step snapshot) in the "
                         "block MSE. Continuous dose version of the L2 'late' placement SIM lever. 1.0 = off.")
    ap.add_argument("--snap_early_w", type=float, default=1.0,
                    help="per-sequence weight for EARLY5-step (0-4) snapshots — the mirrored control arm "
                         "of --snap_late_w. 1.0 = off.")
    ap.add_argument("--w_bits", type=int, default=4,
                    help="weight bit-width for all target linears. 16 = leave weights full-precision "
                         "(transform still folded; LWC disabled) -> the W16A4 attribution corner: "
                         "quantize ONLY activations.")
    ap.add_argument("--w8_blocks", default="",
                    help="comma list of transformer block indices whose linears calibrate with INT8 "
                         "weights instead of --w_bits (W1 depth-aligned weight mixed precision), "
                         "e.g. '8,19,20,21,22,23'. Empty = off.")
    ap.add_argument("--a_bits", type=int, default=4)
    ap.add_argument("--a_asym", action="store_true",
                    help="legacy per-token ASYMMETRIC act quant ([0,15]). Default is now SYMMETRIC "
                         "([-8,7]) — the official paper-best AND what the deploy int4 kernel does. "
                         "The recorded bc_* models were calibrated with asym.")
    ap.add_argument("--a_sym", action="store_true",
                    help="deprecated no-op (symmetric is now the default; use --a_asym for legacy)")
    ap.add_argument("--drift", action="store_true",
                    help="legacy propagation: advance calib inputs through the QUANTIZED block "
                         "(BRECQ-style, deploy-realistic error accumulation). Default is now the "
                         "official FlatQuant drift-free protocol (fp inputs/targets per block).")
    ap.add_argument("--diag_alpha", type=float, default=0.3,
                    help="sq_style diag_scale init exponent (official diag_alpha default)")
    ap.add_argument("--no_diag_init", action="store_true",
                    help="skip the sq_style diag init (legacy: diag_scale starts at ones)")
    ap.add_argument("--no_lac", action="store_true")
    ap.add_argument("--no_diag", action="store_true")
    ap.add_argument("--base", type=int, default=1024,
                    help="per-item seed = base + idx (order-free protocol; pairs with fp32/flat gens)")
    ap.add_argument("--calib_lst", default=None,
                    help="quant-calibration list path (default: SEED_CALIB_LST env or paths.CALIB_LST)")
    ap.add_argument("--calib_seed", type=int, default=None,
                    help="pin the calibration draw (capture noise + training randperm) for reproducibility; "
                         "None = uncontrolled RNG (legacy behaviour). Set it so zh/en/hard from one launch share one calib.")
    ap.add_argument("--model", default=None,
                    help="canonical fixed-model path (default: from --model_dir; override dir with SEED_MODELS_DIR)")
    ap.add_argument("--load_model", action="store_true",
                    help="skip calibration and LOAD --model — its W4A4 numbers then match the step-axis 'full' baseline exactly")
    ap.add_argument("--save_model", action="store_true",
                    help="after calibration, torch.save to --model — makes this the ONE producer of the canonical fixed model")
    ap.add_argument("--calibrate_only", action="store_true",
                    help="calibrate (+ --save_model) then EXIT before generation. Lets a multi-GPU launcher do the "
                         "single-GPU calibration once, then fan out sharded generation with --load_model.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model_path = args.model or str(bc_model_path(args.model_dir))
    genroot = os.path.join(str(GEN_DIR), args.out_subdir)

    if args.load_model:
        # consistency mode: reuse the ONE fixed model (produced once by --save_model) instead of re-calibrating
        print(f"[pb] loading fixed calibrated model from {model_path}")
        model = torch.load(model_path, weights_only=False, map_location=dev); model.eval()
        tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    else:
        model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
        model.vae.to_half(); model.eval()
        tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        if args.calib_seed is not None:
            torch.manual_seed(args.calib_seed); torch.cuda.manual_seed_all(args.calib_seed)
            print(f"[pb] calibration pinned to seed {args.calib_seed}")
        calib_lst = args.calib_lst or str(CALIB_LST)
        if not os.path.exists(calib_lst):
            raise FileNotFoundError(f"calibration list not found: {calib_lst} "
                                    "(pass --calib_lst or set SEED_CALIB_LST)")
        calib = load_items(calib_lst)
        print(f"[pb] calib = {len(calib)} items from {calib_lst}")
        print(f"[pb] capturing block-0 activations on {len(calib)} prompts ...")
        store, snap_steps = capture_block_inputs(model, tok, dev, calib, args.max_seqs, args.per_item_keep,
                                                 snap_place=args.snap_place, return_steps=True)
        print(f"[pb] captured {len(store)} sequences (snap_place={args.snap_place}; "
              f"snapshot steps present: {sorted(set(snap_steps))})")
        snap_w = None
        if args.snap_late_w != 1.0 or args.snap_early_w != 1.0:
            snap_w = [args.snap_early_w if s < 5 else (args.snap_late_w if s >= 10 else 1.0)
                      for s in snap_steps]
            print(f"[pb] L-B snapshot weights: early5 x{args.snap_early_w} / mid x1 / late5 x{args.snap_late_w} "
                  f"({sum(1 for s in snap_steps if s >= 10)} late / {sum(1 for s in snap_steps if s < 5)} early "
                  f"/ {sum(1 for s in snap_steps if 5 <= s < 10)} mid seqs)")
        if args.a_sym:
            print("[pb] note: --a_sym is deprecated (symmetric is now the default)")
        # w_bits=16 -> weights stay fp (WeightQuantizer no-ops at 16 bits); LWC off so the clip
        # doesn't lossily modify unquantized weights. Transforms/LAC still train (they serve A4).
        fq.wrap_dit(model, w_bits=args.w_bits, a_bits=args.a_bits, use_trans=True,
                    lwc=(args.w_bits < 16),
                    a_sym=not args.a_asym, lac=not args.no_lac, add_diag=not args.no_diag)
        if args.w8_blocks.strip():
            w8 = sorted({int(t) for t in args.w8_blocks.split(",") if t.strip()})
            nb = len(model.transformer.blocks)
            bad = [i for i in w8 if not (0 <= i < nb)]
            if bad:
                raise SystemExit(f"--w8_blocks indices out of range (model has {nb} blocks): {bad}")
            n_up = 0
            for i in w8:
                for w in _block_wrappers(model.transformer.blocks[i]):
                    w.wq.configure(8, perchannel=True, sym=True, mse=False)
                    n_up += 1
            print(f"[pb] W1 mixed precision: {n_up} linears in blocks {w8} calibrate at INT8 weights "
                  f"(rest {args.w_bits}-bit)")
        chan_w = None
        if args.loss == "percw":
            if not args.chan_w_file:
                raise SystemExit("--loss percw requires --chan_w_file (produce it with "
                                 "python -m audio_dit_quantize.calib.spk_saliency)")
            chan_w = torch.from_numpy(np.load(args.chan_w_file)["sal"])
            print(f"[pb] percw perceptual channel weights from {args.chan_w_file}: {tuple(chan_w.shape)}")
        calibrate_perblock(model, store, dev, steps=args.steps, mb=args.mb,
                           lr=args.lr, clip_lr=args.clip_lr, loss_type=args.loss,
                           region_gen_w=args.region_gen_w, region_prompt_w=args.region_prompt_w,
                           snap_w=snap_w, chan_w=chan_w, drift=args.drift, diag_alpha=args.diag_alpha,
                           diag_init=not args.no_diag_init and not args.no_diag,
                           stats_out=os.path.join(genroot, "calib_block_losses.json"),
                           stats_meta={"calib_lst": str(calib_lst), "calib_seed": args.calib_seed,
                                       "steps": args.steps, "mb": args.mb,
                                       "max_seqs": args.max_seqs, "per_item_keep": args.per_item_keep,
                                       "snap_place": args.snap_place,
                                       "snap_late_w": args.snap_late_w, "snap_early_w": args.snap_early_w,
                                       "w_bits": args.w_bits,
                                       "w8_blocks": args.w8_blocks,
                                       "loss": args.loss, "chan_w_file": args.chan_w_file,
                                       "model_dir": args.model_dir,
                                       "n_calib_items": len(calib), "n_seqs": len(store)})
        if args.save_model:
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save(model, model_path)
            print(f"[pb] saved canonical fixed model -> {model_path}")

    if args.calibrate_only:
        print("[pb] calibrate_only: model ready, skipping generation")
        print("DONE")
        return

    sel = list(SETS) if args.sets == "all" else [s.strip() for s in args.sets.split(",")]
    for name in sel:
        _all = load_items(os.path.join(DATA, SETS[name]))
        items = _all[args.offset:(args.offset + args.limit) if args.limit else None]
        outdir = os.path.join(genroot, name)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        n_invalid = n_err = 0
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
            gidx = args.offset + idx                                                      # global index -> shard-invariant seed
            torch.manual_seed(args.base + gidx); torch.cuda.manual_seed(args.base + gidx)  # per-item, order/shard-free
            try:
                wav = infer_one(gt, pt, pwa, model, tok, dev, 16, 4.0, "apg")
                if not _valid_wav(wav):
                    n_invalid += 1
                    tqdm.write(f"[{name} {idx}] INVALID {uid}: non-finite or all-zero output (not written)")
                    continue
                sf.write(op, wav, model.config.sampling_rate)
                iterator.set_postfix_str(uid[:32])
            except Exception as e:
                n_err += 1
                tqdm.write(f"[{name} {idx}] ERR {uid}: {e}")
        print(f"[{name}] {len(items)} items in {time.time()-t0:.0f}s -> {outdir}"
              + (f"  [WARN invalid={n_invalid} err={n_err}]" if (n_invalid or n_err) else ""))
    print("DONE")


if __name__ == "__main__":
    main()
