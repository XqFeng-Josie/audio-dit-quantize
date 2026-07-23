"""E1 calibration-pool scoring pass: per-sample intermediate signals on the FP model.

Tests whether the literature's per-sample calibration-selection signals predict
calibration-set quality in TTS (docs/results-consolidated.md §1.5, paper §4):

  S1  trajectory motion  (S2Q-VDiT C_diff, NeurIPS'25):   ||x_{t+1}-x_t||^2 / ||x_t||^2
  S2  activation energy  (S2Q-VDiT C_quant / QuantVGGT NFDS stats, ICLR'26):
      input mean-square at the quantized sub-module boundaries

each in the papers' GENERIC form (all steps / all module classes) and in a
TASK-ALIGNED form (late steps x FFN — where our diagnosis localises the W4A4
damage). Validation is offline: set-level score (member mean) vs the known
dual-scale set rankings; only signals that survive get a top/bottom-set arm.

One GPU pass per model scale captures everything the offline analysis needs
(any window / pass / region re-aggregates from raw, no GPU re-run):

  per pool item, FP model, canonical generation path (infer_one, nfe=16,
  cfg 4.0, apg  ->  15 Euler updates x 2 transformer forwards cond+uncond):
    - full ODE latent trajectory via a patched odeint_euler; generation-region
      norms ||x_t||^2 (nfe states) and ||x_{t+1}-x_t||^2 (nfe-1 diffs).
      The prompt region is overwritten in-place inside the ODE fn (noise
      interpolation, then zeroed for the uncond pass), so trajectory stats are
      GENERATION-REGION ONLY by construction.
    - per {step} x {pass cond/uncond} x {block} x {module class}: input
      sum, sum-of-squares, amax, numel. Classes are the sub-module inputs
      feeding the quantized linears (wrap_dit targets): self_attn / ffn = the
      modulated norm, cross_attn = cross_attn_norm(x) (q stream),
      cross_attn_cond = cross_attn_norm_c(cond) (k/v text stream). o-proj and
      FFN-mid inputs are internal to the sub-modules and not separately scored.
      Full-sequence AND generation-region (x[:, pd:]) variants.

Step windows MUST stay aligned with generate_step_axis (NSTEPS=15):
late5 = steps 10-14, early5 = steps 0-4.

HONEST USE: scores are computed on the FULL pool before looking at any ranking;
predictions pre-registered in docs/results-consolidated.md; both scales;
correlations reported as-is whether or not any signal survives.

Usage (experiment machine, after `source env.sh`):
  python -m audio_dit_quantize.calib.score_pool score \
      --model_dir meituan-longcat/LongCat-AudioDiT-1B \
      --out data/calib_pool/scores_v3/1b [--offset 0 --limit 0]
  python -m audio_dit_quantize.calib.score_pool agg \
      --out data/calib_pool/scores_v3/1b

`score` is resumable (items with an existing raw npz are skipped) and shards
with --offset/--limit across GPUs into the same --out. `agg` rebuilds the CSV
from raw/ offline (default: cond pass, generation region).
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import audiodit.modeling_audiodit as MDL
from audiodit import AudioDiTModel
from batch_inference import infer_one
from transformers import AutoTokenizer

from ..flatquant_best import _valid_wav, load_items
from ..paths import REPO_ROOT

CLS = ("self_attn", "cross_attn", "ffn", "cross_attn_cond")
GEN_CLS = 3                      # classes 0..2 live on the hidden stream and have a gen region
STATS = ("sum_sq", "sum", "amax", "numel")
LATE5, EARLY5 = range(10, 15), range(0, 5)   # generate_step_axis NSTEPS=15 windows

POOL_DEFAULT = REPO_ROOT / "data" / "calib_pool" / "pool_v1.lst"
META_DEFAULT = REPO_ROOT / "data" / "calib_pool" / "pool_v1_meta.csv"


# ── capture state ─────────────────────────────────────────────────────────────
class _Cap:
    """Per-item capture buffers. full/gen: [steps, pass, block, class, stat] float64 on GPU
    (accumulated with +=, so numel doubling flags a double-fire)."""

    def __init__(self, nsteps, nblocks, dev):
        self.nsteps, self.nblocks, self.dev = nsteps, nblocks, dev
        self.full = torch.zeros(nsteps, 2, nblocks, len(CLS), len(STATS), dtype=torch.float64, device=dev)
        self.gen = torch.zeros(nsteps, 2, nblocks, GEN_CLS, len(STATS), dtype=torch.float64, device=dev)
        self.reset()

    def reset(self):
        self.full.zero_(); self.gen.zero_()
        self.fires = 0          # transformer forward count this item
        self.pd = 0             # prompt latent frames (hidden-stream gen boundary)
        self.dur = None         # hidden seq len, checked against the trajectory
        self.traj = None        # [nfe, 1, dur, latent_dim]

    def put(self, buf, s, p, b, c, x):
        xf = x.detach().to(torch.float64)
        buf[s, p, b, c, 0] += (xf * xf).sum()
        buf[s, p, b, c, 1] += xf.sum()
        buf[s, p, b, c, 2] = torch.maximum(buf[s, p, b, c, 2], xf.abs().amax())
        buf[s, p, b, c, 3] += xf.numel()


def _install_hooks(model, cap, nfe):
    """Transformer pass counter + per-block sub-module input hooks. Returns handles."""
    handles = []

    def tf_hook(_m, _a):
        cap.fires += 1
        if cap.fires > 2 * (nfe - 1):
            raise RuntimeError(f"transformer fired {cap.fires} > expected {2 * (nfe - 1)} "
                               "(cfg/step structure changed?)")
    handles.append(model.transformer.register_forward_pre_hook(tf_hook))

    def make_hook(b, ci):
        def h(_m, args, kwargs):
            s, p = divmod(cap.fires - 1, 2)          # fn() runs cond then uncond each Euler step
            x = kwargs.get("x", args[0] if args else None)
            if cap.dur is None:
                cap.dur = x.shape[1]
            cap.put(cap.full, s, p, b, ci, x)
            cap.put(cap.gen, s, p, b, ci, x[:, cap.pd:])
            if ci == 1:                              # cross_attn also exposes the k/v text stream
                cond = kwargs.get("cond", args[1] if len(args) > 1 else None)
                cap.put(cap.full, s, p, b, 3, cond)
        return h

    for b, block in enumerate(model.transformer.blocks):
        for ci, name in enumerate(CLS[:GEN_CLS]):
            mod = getattr(block, name, None)
            if mod is not None:
                handles.append(mod.register_forward_pre_hook(make_hook(b, ci), with_kwargs=True))
    return handles


# ── score pass ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_score(args):
    dev = torch.device(args.device)
    items = load_items(args.pool)
    lo, hi = args.offset, len(items) if args.limit <= 0 else min(len(items), args.offset + args.limit)
    todo = [(i, items[i]) for i in range(lo, hi)]

    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"[score] loading FP model {args.model_dir}")
    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    nblocks = len(model.transformer.blocks)
    nsteps = args.nfe - 1                      # Euler updates = network steps (= step_axis NSTEPS)
    cap = _Cap(nsteps, nblocks, dev)
    handles = _install_hooks(model, cap, args.nfe)

    # patch odeint_euler to stash the full latent trajectory the model discards
    _ode = MDL.odeint_euler
    def ode_stash(fn, y0, t):
        out = _ode(fn, y0, t)
        if cap.traj is not None:
            raise RuntimeError("odeint fired twice in one item")
        cap.traj = out.detach()
        return out
    MDL.odeint_euler = ode_stash

    # record the prompt/gen boundary per item (same pattern as capture_block_inputs)
    _enc = model.encode_prompt_audio
    def enc(pa):
        lat, pd = _enc(pa); cap.pd = int(pd); return lat, pd
    model.encode_prompt_audio = enc

    done, skipped, failed = 0, 0, []
    try:
        for idx, it in tqdm(todo, desc="score pool", dynamic_ncols=True):
            uid = it[0]
            out_npz = os.path.join(raw_dir, f"{uid}.npz")
            if os.path.exists(out_npz) and not args.overwrite:
                skipped += 1
                continue
            cap.reset()
            torch.manual_seed(args.seed + idx); torch.cuda.manual_seed_all(args.seed + idx)
            try:
                wav = infer_one(it[3], it[1], it[2], model, tok, dev,
                                nfe=args.nfe, cfg_strength=4.0, guidance_method="apg")
                _check_item(cap, args.nfe, uid)
                traj = cap.traj.to(torch.float64)            # [nfe, 1, dur, ld]
                g = traj[:, 0, cap.pd:, :]                   # gen region only (prompt region is
                d = g[1:] - g[:-1]                           #  mutated in-place by the ODE fn)
                np.savez_compressed(
                    out_npz,
                    full=cap.full.cpu().numpy(), gen=cap.gen.cpu().numpy(),
                    traj_norm_sq=(g * g).sum(dim=(1, 2)).cpu().numpy(),
                    traj_diff_sq=(d * d).sum(dim=(1, 2)).cpu().numpy(),
                    traj_numel=np.array(g.shape[1] * g.shape[2]),
                    meta=np.array(json.dumps({
                        "uid": uid, "idx": idx, "seed": args.seed + idx, "nfe": args.nfe,
                        "pd": cap.pd, "dur": int(cap.dur), "nblocks": nblocks,
                        "cls": CLS, "stats": STATS, "wav_ok": bool(_valid_wav(wav)),
                        "wav_dur_s": float(len(np.asarray(wav).ravel())) / model.config.sampling_rate,
                        "model_dir": args.model_dir,
                    })))
                done += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                failed.append(uid); print(f"[score] OOM on {uid}, skipped")
            except Exception as e:  # noqa: BLE001 — record and continue, rerun picks failures up
                failed.append(uid); print(f"[score] FAILED {uid}: {e}")
    finally:
        for h in handles:
            h.remove()
        MDL.odeint_euler = _ode
        del model.encode_prompt_audio

    build = {"model_dir": args.model_dir, "pool": args.pool, "nfe": args.nfe, "seed": args.seed,
             "offset": lo, "end": hi, "done": done, "skipped_existing": skipped, "failed": failed,
             "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, f"build_{lo}_{hi}.json"), "w") as f:
        json.dump(build, f, ensure_ascii=False, indent=2)
    print(f"[score] done={done} skipped={skipped} failed={len(failed)} -> {raw_dir}")
    if failed:
        print(f"[score] failed uids: {failed}")
    run_agg(args)


def _check_item(cap, nfe, uid):
    """Fail fast if the capture doesn't match the expected generation structure."""
    exp = 2 * (nfe - 1)
    if cap.fires != exp:
        raise RuntimeError(f"{uid}: transformer fired {cap.fires}, expected {exp}")
    if cap.traj is None or cap.traj.shape[0] != nfe:
        raise RuntimeError(f"{uid}: trajectory missing or wrong length "
                           f"({None if cap.traj is None else tuple(cap.traj.shape)})")
    if not (0 < cap.pd < cap.traj.shape[2]):
        raise RuntimeError(f"{uid}: bad prompt boundary pd={cap.pd} dur={cap.traj.shape[2]}")
    if cap.traj.shape[2] != cap.dur:
        raise RuntimeError(f"{uid}: hidden seq len {cap.dur} != latent dur {cap.traj.shape[2]}")
    # every (step, pass, block, hidden-class) cell must have fired exactly once:
    # numel == dur * width, identical within a class (width differs per class)
    numel = cap.full[..., 3]
    if (numel[..., :GEN_CLS] <= 0).any():
        raise RuntimeError(f"{uid}: missing sub-module hook fires")
    for c in range(GEN_CLS):
        n = numel[..., c]
        if (n != n[0, 0, 0]).any():
            raise RuntimeError(f"{uid}: inconsistent numel in class {CLS[c]} (double fire?)")
    if not torch.isfinite(cap.full).all() or not torch.isfinite(cap.traj).all():
        raise RuntimeError(f"{uid}: non-finite capture")


# ── offline aggregation ───────────────────────────────────────────────────────
def _window_msq(buf, steps, p, classes):
    """Mean over window steps/blocks/classes of per-fire mean-square (sum_sq/numel)."""
    sl = buf[list(steps), p][:, :, list(classes), :]          # [w, block, cls, stat]
    return float((sl[..., 0] / sl[..., 3]).mean())


def run_agg(args):
    p = {"cond": 0, "uncond": 1}[getattr(args, "pass_", "cond")]
    region = getattr(args, "region", "gen")
    raw_dir = os.path.join(args.out, "raw")
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".npz"))
    if not files:
        print("[agg] no raw npz found"); return

    meta_rows = {}
    if os.path.exists(args.meta):
        with open(args.meta, newline="") as f:
            meta_rows = {r["uid"]: r for r in csv.DictReader(f)}

    windows = {"late5": LATE5, "early5": EARLY5, "all": range(0, 15)}
    cols = (["uid", "lang", "style", "idx", "seed", "pd", "dur", "gen_len", "wav_ok", "wav_dur_s"]
            + [f"cdiff_{w}" for w in windows]
            + [f"{c}_msq_{w}" for c in ("ffn", "sattn", "xattn", "allmod") for w in windows])
    rows = []
    for fn in files:
        z = np.load(os.path.join(raw_dir, fn))
        meta = json.loads(z["meta"].item())
        if meta["nfe"] != 16:
            raise RuntimeError(f"{fn}: nfe={meta['nfe']} but windows assume 15 network steps")
        buf = z["gen"] if region == "gen" else z["full"]
        norm_sq, diff_sq = z["traj_norm_sq"], z["traj_diff_sq"]
        m = meta_rows.get(meta["uid"], {})
        row = {"uid": meta["uid"], "lang": m.get("lang", ""), "style": m.get("style", ""),
               "idx": meta["idx"], "seed": meta["seed"], "pd": meta["pd"], "dur": meta["dur"],
               "gen_len": meta["dur"] - meta["pd"], "wav_ok": int(meta["wav_ok"]),
               "wav_dur_s": round(meta["wav_dur_s"], 2)}
        for w, steps in windows.items():
            # C_diff of network step t: motion it produced, normalised by the state it reached
            # (paper form: ||x_t - x_{t-1}||^2 / ||x_t||^2)
            row[f"cdiff_{w}"] = float(np.mean([diff_sq[t] / norm_sq[t + 1] for t in steps]))
            row[f"ffn_msq_{w}"] = _window_msq(buf, steps, p, [2])
            row[f"sattn_msq_{w}"] = _window_msq(buf, steps, p, [0])
            row[f"xattn_msq_{w}"] = _window_msq(buf, steps, p, [1])
            row[f"allmod_msq_{w}"] = _window_msq(buf, steps, p, [0, 1, 2])
        rows.append(row)

    tag = "" if (p == 0 and region == "gen") else f"_{getattr(args, 'pass_', 'cond')}_{region}"
    out_csv = os.path.join(args.out, f"pool_scores{tag}.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader(); wr.writerows(rows)
    bad = [r["uid"] for r in rows if not r["wav_ok"]]
    print(f"[agg] {len(rows)} samples -> {out_csv} (pass={getattr(args, 'pass_', 'cond')}, region={region})")
    if bad:
        print(f"[agg] WARNING invalid wavs (score kept, flag wav_ok=0): {bad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="GPU scoring pass over the pool (resumable, shardable)")
    sc.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    sc.add_argument("--pool", default=str(POOL_DEFAULT))
    sc.add_argument("--meta", default=str(META_DEFAULT))
    sc.add_argument("--out", required=True, help="e.g. data/calib_pool/scores_v3/1b")
    sc.add_argument("--nfe", type=int, default=16)
    sc.add_argument("--seed", type=int, default=0, help="item i draws noise with seed+i")
    sc.add_argument("--offset", type=int, default=0)
    sc.add_argument("--limit", type=int, default=0, help="0 = to end of pool")
    sc.add_argument("--device", default="cuda:0")
    sc.add_argument("--overwrite", action="store_true", help="re-score items with existing raw npz")
    sc.set_defaults(fn=run_score)

    ag = sub.add_parser("agg", help="rebuild pool_scores.csv from raw/ (offline)")
    ag.add_argument("--out", required=True)
    ag.add_argument("--meta", default=str(META_DEFAULT))
    ag.add_argument("--pass", dest="pass_", choices=["cond", "uncond"], default="cond")
    ag.add_argument("--region", choices=["gen", "full"], default="gen")
    ag.set_defaults(fn=run_agg)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
