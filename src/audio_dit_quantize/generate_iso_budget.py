"""Iso-budget activation-bit reallocation on the FIXED best-config W4A4 model (§3.2 fairness arm).

§3.2's protections ADD activation budget on top of W4A4 (late = 33% fp, ffnfp = 50% fp, ...).
This experiment holds the TOTAL param-weighted average activation bit-width at EXACTLY 4.0 and
only REALLOCATES bits -- the true "precision allocation at equal cost" test:

  step axis   : iso_late  = late 5 steps @A6, other 10 steps @A3   (5*6+10*3)/15 = 4.0
                iso_early = early 5 steps @A6 (mirror control)                  = 4.0
  module axis : iso_ffn   = FFN @A5, attn @A3   (1B exactly 4.0; 3.5B 3.9474, see below)
                iso_attn  = FFN @A3, attn @A5   (1B exactly 4.0; 3.5B 4.0526)
  baseline    : the existing step-axis `full` run (uniform A4) -- zero new generation.
  identity    : uniform4 (all A4) must reproduce `full` bit-exact; smoke-check only.

MODULE-AXIS CONTRACT (pair-symmetric, decided 2026-07-23): the FFN param fraction is exactly 1/2
on 1B (ffn ratio 4.0) but 9/19 on 3.5B (ffn ratio 3.6: per block 2*3.6d^2 ffn vs 8d^2 attn), and
9a+10b=76 has NO integer two-level solution besides a=b=4 -- an exact per-arm 4.0 module split is
impossible on 3.5B (per-linear/per-block compensation or step-mixed schedules would inject
arbitrary subsets/step structure; rejected). Contract: PAIR MEAN exactly 4.0, per-arm deviation
<= 1/19 bit (3.5B: iso_ffn 3.9474 / iso_attn 4.0526). The predicted winner iso_ffn is the
LOWER-budget arm, so an iso_ffn win is conservative; an iso_attn win is budget-confounded.
Step-axis arms are composition-independent (exactly 4.0 at every scale).

Protocol identical to generate_step_axis: same fixed calibrated model, per-item seed =
base + offset + idx, same sets/sharding, wav validity guard. Weights stay INT4 everywhere;
only ActivationQuantizer bit-widths are re-stamped per ODE step. Per-token dynamic act quant
means changing bits needs no recalibration (scales are computed per forward; the learned LAC
clip factors are scalar amax/amin multipliers, bits-independent, shared by all arms).

DOCUMENTED HANDICAP (pre-registered, docs/results-consolidated.md §4): transforms/clips were
calibrated at uniform A4, so the non-uniform arms run off their calibrated operating point while
the uniform-A4 baseline runs on it. The two arms of each axis are symmetric to each other (fair
pair); arm-vs-baseline carries the handicap -- an arm that still beats `full` is a strong result,
an arm that loses is ambiguous (allocation vs mis-calibration).

Usage (experiment machine, after `source env.sh`; fixed bc model must exist):
  python -m audio_dit_quantize.generate_iso_budget --model_dir meituan-longcat/LongCat-AudioDiT-1B \
      --sets zh,en,hard --configs iso_late,iso_early,iso_ffn,iso_attn
  bash scripts/evaluate_seedtts_metrics.sh gen/iso_budget/iso_late iso_late "zh en hard"
"""
import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

from batch_inference import infer_one
from . import flatquant_best as pb
from . import flatquant_layers as fq
from .generate_step_axis import FPS, NSTEPS, TAGS, _valid_wav, tag_linears
from .paths import DATA_DIR, GEN_DIR, SETS, bc_model_path
from flatquant.quant_utils import get_qmin_qmax

DATA = str(DATA_DIR)
_L, _E = frozenset(range(10, NSTEPS)), frozenset(range(0, 5))

# policy(step, tag) -> activation bits for every wrapped linear of that module class at that step
CONFIGS = {
    "iso_late":  lambda step, tag: 6 if step in _L else 3,
    "iso_early": lambda step, tag: 6 if step in _E else 3,
    "iso_ffn":   lambda step, tag: 5 if tag == "ffn" else 3,
    "iso_attn":  lambda step, tag: 3 if tag == "ffn" else 5,
    "uniform4":  lambda step, tag: 4,   # identity control: must match step-axis `full` bit-exact
}
MIRROR = {"iso_late": "iso_early", "iso_early": "iso_late",
          "iso_ffn": "iso_attn", "iso_attn": "iso_ffn", "uniform4": "uniform4"}


def avg_bits(policy, counts):
    """Param-weighted mean activation bits over all steps and module classes."""
    tot = sum(counts.values()) or 1
    return sum(counts[t] * policy(s, t) for s in range(NSTEPS) for t in TAGS) / (NSTEPS * tot)


def _set_bits(aq, b):
    if aq.bits != b:
        aq.bits = b
        aq.q_max, aq.q_min = get_qmin_qmax(b, aq.sym)


_st = {"fwd": 0, "policy": None, "last": None}


def _hook(module, *_a):
    step = min(_st["fwd"] // FPS, NSTEPS - 1)
    key = tuple(_st["policy"](step, t) for t in TAGS)
    if key != _st["last"]:                   # re-stamp only when this step's bit map changes
        for m in module.modules():
            if isinstance(m, fq.FlatQuantLinear):
                _set_bits(m.aq, _st["policy"](step, m._mod_tag))
        _st["last"] = key
    _st["fwd"] += 1


@torch.no_grad()
def gen_config(model, tok, dev, items, base, outdir, nfe, cfg, guid, policy):
    """Generate one iso-budget config for one set. Per-item seed = base + idx (order-free)."""
    os.makedirs(outdir, exist_ok=True)
    _st["policy"], _st["last"] = policy, None
    for m in model.transformer.modules():    # bit policies replace the binary gates: force gates open
        if isinstance(m, fq.FlatQuantLinear):
            m._act_on = True
    fq._ACT_QUANT = True
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
    for m in model.transformer.modules():    # restore uniform A4 (no cross-config contamination)
        if isinstance(m, fq.FlatQuantLinear):
            _set_bits(m.aq, 4)
    if n_invalid or n_err:
        print(f"  [WARN] {outdir}: invalid={n_invalid} err={n_err}", flush=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--model", default=None,
                    help="fixed best-config model path (default: models/bc_{1b,3p5b}_model.pt from --model_dir)")
    ap.add_argument("--sets", default="zh,en,hard", help="comma list of {zh,en,hard}")
    ap.add_argument("--configs", default="iso_late,iso_early,iso_ffn,iso_attn",
                    help="comma list of " + ",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=0, help="items per set; 0 = full set")
    ap.add_argument("--offset", type=int, default=0, help="start item index (seed shifts by offset too)")
    ap.add_argument("--out_subdir", default="iso_budget")
    ap.add_argument("--base", type=int, default=1024, help="per-item seed = base + offset + idx (order-free)")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg_strength", type=float, default=4.0)
    ap.add_argument("--guidance", default="apg")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model_path = args.model or str(bc_model_path(args.model_dir))

    if args.nfe - 1 != NSTEPS:
        raise SystemExit(f"--nfe {args.nfe} inconsistent with NSTEPS={NSTEPS} (policies assume 15 network steps)")
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config '{c}'; choose from {list(CONFIGS)}")
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]

    if not os.path.exists(model_path):
        raise SystemExit(f"fixed calibrated model not found: {model_path} "
                         "(produce it with generate_step_axis --calibrate)")
    print(f"[iso] loading fixed calibrated model from {model_path}", flush=True)
    model = torch.load(model_path, weights_only=False).to(dev); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    torch.set_grad_enabled(False)

    counts = tag_linears(model)
    untagged = [n for n, m in model.transformer.named_modules()
                if isinstance(m, fq.FlatQuantLinear) and m._mod_tag is None]
    if untagged:
        raise SystemExit(f"wrapped linears without module tag (policy would be undefined): {untagged}")
    for c in configs:
        b = avg_bits(CONFIGS[c], counts)
        pair = (b + avg_bits(CONFIGS[MIRROR[c]], counts)) / 2
        print(f"[iso] {c}: param-weighted avg activation bits = {b:.4f} "
              f"(mirror {MIRROR[c]}: pair mean = {pair:.4f})", flush=True)
        if abs(pair - 4.0) > 1e-3 or abs(b - 4.0) > 1 / 19 + 1e-6:
            raise SystemExit(f"config {c} breaks the pair-symmetric iso contract "
                             f"(avg bits {b:.4f}, pair mean {pair:.4f})")

    h = model.transformer.register_forward_pre_hook(_hook)
    try:
        for setname in sets:
            _all = pb.load_items(os.path.join(DATA, SETS[setname]))
            items = _all[args.offset:(args.offset + args.limit) if args.limit else None]
            eff_base = args.base + args.offset
            for c in configs:
                outdir = os.path.join(str(GEN_DIR), args.out_subdir, c, setname)
                dt = gen_config(model, tok, dev, items, eff_base, outdir,
                                args.nfe, args.cfg_strength, args.guidance, CONFIGS[c])
                print(f"[gen] {setname}/{c}: avg bits {avg_bits(CONFIGS[c], counts):.3f} "
                      f"-> {len(items)} items in {dt:.0f}s -> {outdir}", flush=True)
    finally:
        h.remove()
        for m in model.transformer.modules():
            if isinstance(m, fq.FlatQuantLinear):
                _set_bits(m.aq, 4)
    print("ISO-BUDGET GEN DONE")


if __name__ == "__main__":
    main()
