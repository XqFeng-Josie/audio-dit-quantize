"""Per-item-seeded Hard-set generation for any quant config (order-independent, cross-config paired).

Before each item: torch.manual_seed(base + idx) -> item idx always gets the same diffusion init
noise regardless of generation order OR config. So all configs generate each utterance with
IDENTICAL noise -> clean paired comparison (noise + order removed as confounds).

modes: fp32 | int8 (W8A8) | rtn (naive W4A4) | flatquant (W4A4, trained) | svdquant (W4A4)

Usage:
  PYTHONPATH=~/workspace/LongCat-AudioDiT python run_paired_hard.py --mode fp32 --tag fp32 --base 1024
  ... --mode int8 --tag int8 ; --mode rtn --tag rtn ; --mode flatquant --tag flat ; --mode svdquant --tag svd
"""
import argparse, os, time
import torch, soundfile as sf
import audiodit  # noqa
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
import run_svdquant as rs          # svdquant calibrate + load_items/SETS/DATA
import run_flatquant as rf         # flatquant calibrate
import flatquant_dit as fq         # rtn wrap
from precision import apply_precision   # int8
from paths import GEN_DIR


def prep(mode, model, tok, dev, calib_prompts, rank, calib_lst=None, flat_loss="mse",
         calib_rows=512, calib_steps=200):
    if mode == "fp32":
        return
    if mode == "int8":
        apply_precision(model, "int8"); return
    if mode == "rtn":
        wrapped = fq.wrap_dit(model, w_bits=4, a_bits=4, use_trans=False, lwc=False)
        for m in wrapped:
            m.freeze()
        return
    calib = (rs.load_items(os.path.expanduser(calib_lst)) if calib_lst
             else rs.load_items(os.path.join(rs.DATA, rs.SETS["zh"]))[:calib_prompts])
    print(f"[prep] calib = {len(calib)} items from {'--calib_lst '+calib_lst if calib_lst else 'Seed-ZH[:%d]'%calib_prompts}", flush=True)
    if mode == "flatquant":
        rf.calibrate(model, tok, dev, calib, calib_rows, calib_steps, a_bits=4, loss_type=flat_loss)
    elif mode == "svdquant":
        rs.calibrate(model, tok, dev, calib, 512, rank, a_bits=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fp32", "int8", "rtn", "flatquant", "svdquant"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", type=int, default=1024, help="per-item seed = base + item_index")
    ap.add_argument("--sets", default="hard", help="comma list of Seed sets (zh,en,hard) — per-item seeded")
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--calib_prompts", type=int, default=6)
    ap.add_argument("--calib_rows", type=int, default=512)
    ap.add_argument("--calib_steps", type=int, default=200)
    ap.add_argument("--calib_lst", default=None, help="custom calibration .lst (overrides Seed-ZH[:6])")
    ap.add_argument("--nfe", type=int, default=16, help="ODE time points; forwards=2*(nfe-1) under CFG/APG")
    ap.add_argument("--guidance", default="apg", choices=["cfg", "apg"])
    ap.add_argument("--cfg_strength", type=float, default=4.0)
    ap.add_argument("--flat_loss", default="mse", choices=["mse", "outlier", "chanbal", "selective_q", "timebal"],
                    help="FlatQuant calibration objective (docs/11 loss-design study)")
    ap.add_argument("--limit", type=int, default=0, help="limit items per set; 0 = full set")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    prep(args.mode, model, tok, dev, args.calib_prompts, args.rank, calib_lst=args.calib_lst,
         flat_loss=args.flat_loss, calib_rows=args.calib_rows, calib_steps=args.calib_steps)

    for setname in [s.strip() for s in args.sets.split(",") if s.strip()]:
        items = rs.load_items(os.path.join(rs.DATA, rs.SETS[setname]))
        if args.limit:
            items = items[:args.limit]
        outdir = os.path.join(str(GEN_DIR), "paired", args.tag, setname)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        for idx, (uid, pt, pwa, gt) in enumerate(items):
            op = os.path.join(outdir, f"{uid}.wav")
            if os.path.exists(op):
                continue
            torch.manual_seed(args.base + idx); torch.cuda.manual_seed(args.base + idx)   # per-item, order-independent
            try:
                wav = infer_one(gt, pt, pwa, model, tok, dev, args.nfe, args.cfg_strength, args.guidance)
                sf.write(op, wav, model.config.sampling_rate)
            except Exception as e:
                print(f"[{args.tag}/{setname} {idx}] ERR {uid}: {e}")
        print(f"[{args.tag}/{setname}] {len(items)} items in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    print("GEN DONE")


if __name__ == "__main__":
    main()
