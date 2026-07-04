"""Per-item-seeded Seed-TTS generation for fp32, SVDQuant, and simple baselines.

Before each item: torch.manual_seed(base + idx) -> item idx always gets the same diffusion init
noise regardless of generation order OR config. So all configs generate each utterance with
IDENTICAL noise -> clean paired comparison (noise + order removed as confounds).

modes: fp32 | int8 (W8A8) | rtn (naive W4A4) | svdquant (W4A4)
SVDQuant calibration always uses data/calib_heldout_hardlike32.lst.

Usage:
  python -m audio_dit_quantize.generate_seedtts --mode fp32 --tag fp32 --base 1024
  ... --mode int8 --tag int8 ; --mode rtn --tag rtn ; --mode svdquant --tag svd
"""
import argparse, os, time
import torch, soundfile as sf
import audiodit  # noqa
from tqdm import tqdm
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
from . import svdquant_pipeline as rs     # svdquant calibrate + load_items/SETS/DATA
from . import flatquant_layers as fq      # rtn wrap
from .precision import apply_precision   # int8
from .paths import GEN_DIR


def prep(mode, model, tok, dev, rank):
    if mode == "fp32":
        return
    if mode == "int8":
        apply_precision(model, "int8"); return
    if mode == "rtn":
        wrapped = fq.wrap_dit(model, w_bits=4, a_bits=4, use_trans=False, lwc=False)
        for m in wrapped:
            m.freeze()
        return
    if mode == "svdquant":
        calib = rs.load_calib_items()
        rs.calibrate(model, tok, dev, calib, 512, rank, a_bits=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fp32", "int8", "rtn", "svdquant"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", type=int, default=1024, help="per-item seed = base + item_index")
    ap.add_argument("--sets", default="hard", help="comma list of Seed sets (zh,en,hard) — per-item seeded")
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--nfe", type=int, default=16, help="ODE time points; forwards=2*(nfe-1) under CFG/APG")
    ap.add_argument("--guidance", default="apg", choices=["cfg", "apg"])
    ap.add_argument("--cfg_strength", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="limit items per set; 0 = full set")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    prep(args.mode, model, tok, dev, args.rank)

    for setname in [s.strip() for s in args.sets.split(",") if s.strip()]:
        items = rs.load_items(os.path.join(rs.DATA, rs.SETS[setname]))
        if args.limit:
            items = items[:args.limit]
        outdir = os.path.join(str(GEN_DIR), "paired", args.tag, setname)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        iterator = tqdm(
            enumerate(items),
            total=len(items),
            desc=f"gen {args.tag}/{setname}",
            dynamic_ncols=True,
        )
        for idx, (uid, pt, pwa, gt) in iterator:
            op = os.path.join(outdir, f"{uid}.wav")
            if os.path.exists(op):
                iterator.set_postfix_str("skip existing")
                continue
            torch.manual_seed(args.base + idx); torch.cuda.manual_seed(args.base + idx)   # per-item, order-independent
            try:
                wav = infer_one(gt, pt, pwa, model, tok, dev, args.nfe, args.cfg_strength, args.guidance)
                sf.write(op, wav, model.config.sampling_rate)
                iterator.set_postfix_str(uid[:32])
            except Exception as e:
                tqdm.write(f"[{args.tag}/{setname} {idx}] ERR {uid}: {e}")
        print(f"[{args.tag}/{setname}] {len(items)} items in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    print("GEN DONE")


if __name__ == "__main__":
    main()
