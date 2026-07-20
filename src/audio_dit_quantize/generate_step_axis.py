"""Step-axis activation-precision experiment (full / early / late) on the FIXED best-config W4A4 model.

WHAT: the same fixed best-config FlatQuant W4A4 model (one calibration) generates each Seed set three ways,
gating ONLY the ODE-step at which activations are quantized (weights stay int4 the whole time):
  full  = all 15 Euler steps quantize activations                        (the W4A4 baseline)
  early = first 5 steps (0-4) run fp activation, the rest int4           (equal-budget control)
  late  = last  5 steps (10-14) run fp activation, the rest int4         (the method — recovers timbre/SIM)
The per-step gate flips `flatquant_layers._ACT_QUANT` via a transformer forward-pre-hook
(step = fwd // 2, because CFG runs 2 forwards — cond+uncond — per Euler step).

MODULE AXIS (M1, docs §3.4): configs xattn / sattn / ffnfp keep ONE block class (cross_attn /
self_attn / ffn) at fp activation across ALL steps via the per-linear `_act_on` flag; late_xattn
combines both axes (the candidate task-aligned recipe). Same fixed model, so every config pairs
item-wise with `full` free of calibration-instance noise.

FIXED CALIBRATION MODEL (ONE model, shared with the efficiency deploy experiments): models/bc_{1b,3p5b}_model.pt
  - default: LOAD it, so full/early/late compare the SAME model — that identity is the whole point of the control.
  - --calibrate: PRODUCE it once (best-config recipe: per-block recon + LWC + LAC + add_diag, calib_seed 0,
    CALIB_LST) and torch.save. GPU non-determinism makes each calibration a slightly different draw, so
    calibrate ONCE and reuse the saved file everywhere (do not re-calibrate per run).

Run (after `source env.sh`):
  # one-time: produce the fixed model (or copy an existing bc_*.pt to models/)
  python -m audio_dit_quantize.generate_step_axis --model_dir meituan-longcat/LongCat-AudioDiT-1B --calibrate
  # generate the three configs, then evaluate each config dir with the standard harness:
  python -m audio_dit_quantize.generate_step_axis --model_dir meituan-longcat/LongCat-AudioDiT-1B \
      --sets zh,en,hard --configs full,early,late
  bash scripts/evaluate_seedtts_metrics.sh gen/step_axis/late step_late "zh en hard"
  # NOTE: pass a model-scoped --out_subdir for 3.5B (the benchmark script uses step_axis_3.5b) so a
  # 3.5B run does not reuse the 1B run's cached wavs (skip-existing) or overwrite its result files.
"""
import argparse, os, time
import numpy as np
import torch, soundfile as sf

import audiodit  # noqa
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
from . import flatquant_layers as fq
from . import flatquant_best as pb          # load_items / capture_block_inputs / calibrate_perblock
from .paths import CALIB_LST, DATA_DIR, GEN_DIR, SETS, bc_model_path

DATA = str(DATA_DIR)
NSTEPS, FPS = 15, 2   # 15 Euler steps; 2 network forwards/step (CFG cond+uncond)
# config = (fp-activation ODE steps, fp-activation module classes). Step axis and module axis are
# orthogonal gates on the SAME loaded model: steps flip fq._ACT_QUANT per forward (hook), module
# classes flip the per-linear _act_on flag (M1, docs §3.4). Weights stay int4 in every config.
CONFIGS = {
    "full":  (set(), ()),                     # every step quantizes activation (W4A4 baseline)
    "early": (set(range(0, 5)), ()),          # steps 0-4 fp activation  (equal-budget control)
    "late":  (set(range(10, NSTEPS)), ()),    # steps 10-14 fp activation (the step-axis method)
    "mid":   (set(range(5, 10)), ()),         # steps 5-9 fp activation   (position control)
    "noact": (set(range(NSTEPS)), ()),        # all steps fp activation   (weight-only W4 ceiling)
    # M1 module axis (all steps quantize; ONE block class runs fp activation)
    "xattn": (set(), ("cross_attn",)),        # mechanism hypothesis target (repeat-collapse §4.7)
    "sattn": (set(), ("self_attn",)),         # control
    "ffnfp": (set(), ("ffn",)),               # control (largest param share)
    # candidate combined recipe: late steps for SIM + cross-attn for intelligibility
    "late_xattn": (set(range(10, NSTEPS)), ("cross_attn",)),
}


def set_act_fp_modules(model, classes):
    """M1 module-axis gate: every FlatQuantLinear whose module path crosses one of `classes`
    (self_attn / cross_attn / ffn) runs fp activation (weight stays int4) via the per-linear
    _act_on flag. Returns the act-fp fraction of wrapped params (budget bookkeeping)."""
    import re
    pat = re.compile(r"\.(%s)\." % "|".join(classes)) if classes else None
    tot = fp = 0
    for name, m in model.transformer.named_modules():
        if isinstance(m, fq.FlatQuantLinear):
            off = bool(pat and pat.search("." + name + "."))
            m._act_on = not off
            n = m.linear.weight.numel()
            tot += n
            fp += n if off else 0
    return fp / max(tot, 1)

def _valid_wav(wav):
    """Finite + not degenerate all-(near)zero. A protected/quantized step combination that silently
    emits NaN/all-zero must not be written, or the paired ΔSIM would be computed on garbage (audit F3)."""
    arr = np.asarray(wav)
    return arr.size > 0 and np.isfinite(arr).all() and float(np.abs(arr).max()) > 1e-6


_st = {"fwd": 0, "protect": set()}
def _hook(*_a):
    step = _st["fwd"] // FPS
    fq._ACT_QUANT = step not in _st["protect"]   # protected step -> fp activation (weight stays int4)
    _st["fwd"] += 1


@torch.no_grad()
def gen_config(model, tok, dev, items, base, outdir, nfe, cfg, guid, protect):
    """Generate one config (one protect-set) for one set. Per-item seed = base + idx (order-free)."""
    os.makedirs(outdir, exist_ok=True)
    _st["protect"] = protect
    t0 = time.time()
    n_invalid = n_err = 0
    for idx, (uid, pt, pwa, gt) in enumerate(items):
        op = os.path.join(outdir, f"{uid}.wav")
        if os.path.exists(op):
            continue
        _st["fwd"] = 0
        torch.manual_seed(base + idx); torch.cuda.manual_seed(base + idx)
        try:
            wav = infer_one(gt, pt, pwa, model, tok, dev, nfe, cfg, guid)
            if not _valid_wav(wav):
                n_invalid += 1
                print(f"  [{idx}] INVALID {uid}: non-finite or all-zero output (not written)", flush=True)
                continue
            sf.write(op, wav, model.config.sampling_rate)
        except Exception as e:
            n_err += 1
            print(f"  [{idx}] ERR {uid}: {e}", flush=True)
    fq._ACT_QUANT = True
    if n_invalid or n_err:
        print(f"  [WARN] {outdir}: invalid={n_invalid} err={n_err}", flush=True)
    return time.time() - t0


@torch.no_grad()
def calibrate_bestconfig(model_dir, dev, calib_seed=0, max_seqs=64, per_item_keep=2, steps=200, mb=4):
    """Best-config calibration (faithful to flatquant_best.main): per-block recon + LWC + LAC + add_diag, asym."""
    model = AudioDiTModel.from_pretrained(model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    torch.manual_seed(calib_seed); torch.cuda.manual_seed_all(calib_seed)
    print(f"[bc] calibration pinned to seed {calib_seed}", flush=True)
    if not CALIB_LST.exists():
        raise FileNotFoundError(f"fixed calibration list not found: {CALIB_LST}")
    calib = pb.load_items(CALIB_LST)
    print(f"[bc] calib = {len(calib)} items from {CALIB_LST}", flush=True)
    store = pb.capture_block_inputs(model, tok, dev, calib, max_seqs, per_item_keep)
    fq.wrap_dit(model, w_bits=4, a_bits=4, use_trans=True, lwc=True, a_sym=False, lac=True, add_diag=True)
    pb.calibrate_perblock(model, store, dev, steps=steps, mb=mb)   # per-block mse; freezes each block
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--model", default=None,
                    help="fixed best-config model path (default: models/bc_{1b,3p5b}_model.pt from --model_dir). "
                         "LOADED by default so full/early/late share ONE calibration.")
    ap.add_argument("--calibrate", action="store_true",
                    help="produce the fixed model instead of loading: calibrate best-config and torch.save to --model.")
    ap.add_argument("--sets", default="zh,en,hard", help="comma list of {zh,en,hard}")
    ap.add_argument("--configs", default="full,early,late", help="comma list of " + ",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=0, help="items per set; 0 = full set")
    ap.add_argument("--offset", type=int, default=0, help="start item index (seed shifts by offset too)")
    ap.add_argument("--out_subdir", default="step_axis")
    # calibration knobs (only used with --calibrate)
    ap.add_argument("--calib_seed", type=int, default=0)
    ap.add_argument("--max_seqs", type=int, default=64)
    ap.add_argument("--per_item_keep", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--mb", type=int, default=4)
    # generation knobs
    ap.add_argument("--base", type=int, default=1024, help="per-item seed = base + offset + idx (order-free)")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg_strength", type=float, default=4.0)
    ap.add_argument("--guidance", default="apg")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model_path = args.model or str(bc_model_path(args.model_dir))

    if args.calibrate:
        model, tok = calibrate_bestconfig(args.model_dir, dev, args.calib_seed,
                                          args.max_seqs, args.per_item_keep, args.steps, args.mb)
        torch.set_grad_enabled(False)
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save(model, model_path)
        print(f"[bc] saved fixed calibrated model -> {model_path}", flush=True)
    else:
        if not os.path.exists(model_path):
            raise SystemExit(
                f"fixed calibrated model not found: {model_path}\n"
                f"  produce it once with:  python -m audio_dit_quantize.generate_step_axis "
                f"--model_dir {args.model_dir} --calibrate\n"
                f"  (or pass --model /path/to/an/existing/bc_*.pt)")
        print(f"[bc] loading fixed calibrated model from {model_path}", flush=True)
        model = torch.load(model_path, weights_only=False).to(dev); model.eval()
        tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        torch.set_grad_enabled(False)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config '{c}'; choose from {list(CONFIGS)}")
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]

    h = model.transformer.register_forward_pre_hook(_hook)
    for setname in sets:
        _all = pb.load_items(os.path.join(DATA, SETS[setname]))
        items = _all[args.offset:(args.offset + args.limit) if args.limit else None]
        eff_base = args.base + args.offset   # seed = base+offset+idx -> matches a single full-set run's per-item seeds
        for c in configs:
            protect, mods = CONFIGS[c]
            frac = set_act_fp_modules(model, mods)
            outdir = os.path.join(str(GEN_DIR), args.out_subdir, c, setname)
            dt = gen_config(model, tok, dev, items, eff_base, outdir,
                            args.nfe, args.cfg_strength, args.guidance, protect)
            print(f"[gen] {setname}/{c}: protect-steps {sorted(protect) or '[]'} "
                  f"mods-fp {list(mods) or '[]'} (act-fp param frac {frac:.2f}) "
                  f"-> {len(items)} items in {dt:.0f}s -> {outdir}", flush=True)
    set_act_fp_modules(model, ())   # reset module gates
    h.remove()
    print("STEP-AXIS GEN DONE")


if __name__ == "__main__":
    main()
