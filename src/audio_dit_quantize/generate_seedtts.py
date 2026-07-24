"""Per-item-seeded Seed-TTS generation for fp32, SVDQuant, and simple baselines.

Before each item: torch.manual_seed(base + idx) -> item idx always gets the same diffusion init
noise regardless of generation order OR config. So all configs generate each utterance with
IDENTICAL noise -> clean paired comparison (noise + order removed as confounds).

modes: fp32 | int8 (W8A8) | rtn (naive W4A4) | svdquant (W4A4) | quarot (W4A4 Hadamard, training-free)
      | quarot_gptq (W4A4 Hadamard + GPTQ weights, the QuaRot paper-best config)
      | gptq (W4A4 GPTQ weights, NO rotation — ladder ablation) | smoothquant (W4A4 scale migration)
Calibrated modes (svdquant/quarot_gptq/gptq/smoothquant) use --calib_lst (default: SEED_CALIB_LST env
or paths.CALIB_LST).

Usage:
  python -m audio_dit_quantize.generate_seedtts --mode fp32 --tag fp32 --base 1024
  ... --mode int8 --tag int8 ; --mode rtn --tag rtn ; --mode svdquant --tag svd
"""
import argparse, os, time
import numpy as np
import torch, soundfile as sf
import audiodit  # noqa
from tqdm import tqdm
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from batch_inference import infer_one
from . import svdquant_pipeline as rs     # svdquant calibrate + load_items/SETS/DATA
from . import flatquant_layers as fq      # rtn wrap
from . import quarot_linear as qr         # quarot wrap (training-free Hadamard)
from .precision import apply_precision   # int8
from .paths import GEN_DIR


def _valid_wav(wav):
    """True iff the synthesized waveform is finite and not a degenerate all-(near)zero signal.
    A quantized/deploy path that silently produces NaN/inf or all-zeros must NOT be written to disk,
    or the metric harness would score garbage as if it were a real generation (see docs audit F3)."""
    arr = np.asarray(wav)
    return arr.size > 0 and np.isfinite(arr).all() and float(np.abs(arr).max()) > 1e-6


def prep(mode, model, tok, dev, rank, calib_seed, w_clip_ratio=1.0,
         svd_rows=2048, svd_iters=None, svd_asym=False, calib_lst=None, sq_alpha=0.5):
    if mode == "fp32":
        return
    if mode == "int8":
        apply_precision(model, "int8"); return
    if mode == "rtn":
        wrapped = fq.wrap_dit(model, w_bits=4, a_bits=4, use_trans=False, lwc=False)
        for m in wrapped:
            m.freeze()
        return
    if mode == "quarot":
        # QuaRot-RTN W4A4: fixed Hadamard rotation, training-free + deterministic -> wrap+freeze (no
        # calibration, no save/load; a sharded gen is identical to a full gen, like fp32/int8/rtn).
        wrapped = qr.wrap_dit(model, w_bits=4, a_bits=4, w_clip_ratio=w_clip_ratio, freeze=True)
        print(f"[quarot] wrapped+froze {len(wrapped)} W4A4 linears (Hadamard rotation, training-free)", flush=True)
        return
    if mode in ("quarot_gptq", "gptq"):
        # QuaRot-GPTQ W4A4 (paper-best): Hadamard rotation + GPTQ weights with --w_clip MSE search.
        # mode "gptq" = the ladder ablation: IDENTICAL pipeline (solver, clip search, sym per-token
        # A4) with rotation disabled (R = I) — isolates the Hadamard's contribution.
        # Calib protocol matches flatquant_best (same CALIB_LST, same 64x2 block-0 capture recipe)
        # so the calibration-data budget is identical across the learned/calibrated methods.
        from .flatquant_best import capture_block_inputs
        if calib_seed is not None:
            torch.manual_seed(calib_seed)
            torch.cuda.manual_seed_all(calib_seed)
            print(f"[prep] {mode} calibration pinned to seed {calib_seed}", flush=True)
        calib = rs.load_calib_items(calib_lst)
        store = capture_block_inputs(model, tok, dev, calib, max_seqs=64, per_item_keep=2)
        print(f"[{mode}] captured {len(store)} sequences", flush=True)
        wrapped = qr.wrap_dit(model, w_bits=4, a_bits=4, w_clip_ratio=w_clip_ratio, freeze=False,
                              rotate=(mode == "quarot_gptq"))
        print(f"[{mode}] wrapped {len(wrapped)} linears (rotate={mode == 'quarot_gptq'}); "
              f"sequential per-block GPTQ ...", flush=True)
        qr.calibrate_gptq(model, store, dev, w_clip_mse=True)
        return
    if mode == "smoothquant":
        # SmoothQuant W4A4 (scale-migration baseline, alpha=sq_alpha): fp amax pass -> fold s
        # into weights -> W4 per-out-channel sym + A4 per-token sym (same primitives as quarot).
        from .flatquant_best import capture_block_inputs
        from . import smoothquant_linear as sql
        if calib_seed is not None:
            torch.manual_seed(calib_seed)
            torch.cuda.manual_seed_all(calib_seed)
            print(f"[prep] smoothquant calibration pinned to seed {calib_seed}", flush=True)
        calib = rs.load_calib_items(calib_lst)
        store = capture_block_inputs(model, tok, dev, calib, max_seqs=64, per_item_keep=2)
        print(f"[smoothquant] captured {len(store)} sequences", flush=True)
        wrapped = sql.wrap_dit(model, w_bits=4, a_bits=4, alpha=sq_alpha)
        print(f"[smoothquant] wrapped {len(wrapped)} linears (alpha={sq_alpha}); calibrating ...", flush=True)
        sql.calibrate_smoothquant(model, store, dev)
        return
    if mode == "svdquant":
        if calib_seed is not None:
            torch.manual_seed(calib_seed)
            torch.cuda.manual_seed_all(calib_seed)
            print(f"[prep] svdquant calibration pinned to seed {calib_seed}", flush=True)
        calib = rs.load_calib_items(calib_lst)
        from .svdquant_linear import NUM_ITERS
        rs.calibrate(model, tok, dev, calib, svd_rows, rank, a_bits=4,
                     num_iters=svd_iters if svd_iters is not None else NUM_ITERS,
                     a_sym=not svd_asym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["fp32", "int8", "rtn", "svdquant", "quarot", "quarot_gptq",
                             "gptq", "smoothquant"])
    ap.add_argument("--sq_alpha", type=float, default=0.5,
                    help="[smoothquant] migration strength alpha (paper default 0.5)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--w_clip_ratio", type=float, default=1.0, help="quarot weight absmax clip ratio (QuaRot~0.9)")
    ap.add_argument("--base", type=int, default=1024, help="per-item seed = base + item_index")
    ap.add_argument("--sets", default="hard", help="comma list of Seed sets (zh,en,hard) — per-item seeded")
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--svd_rows", type=int, default=2048,
                    help="[svdquant] calib activation rows per linear (legacy runs used 512)")
    ap.add_argument("--svd_iters", type=int, default=None,
                    help="[svdquant] low-rank refinement iters; default = paper NUM_ITERS (100, "
                         "early-stopped); 1 = legacy one-shot SVD ablation")
    ap.add_argument("--svd_asym", action="store_true",
                    help="[svdquant] legacy per-group ASYMMETRIC act quant (paper-best/deployable is symmetric)")
    ap.add_argument("--calib_seed", type=int, default=None,
                    help="pin quant calibration randomness; None leaves calibration RNG uncontrolled")
    ap.add_argument("--calib_lst", default=None,
                    help="quant-calibration list path (default: SEED_CALIB_LST env or paths.CALIB_LST)")
    ap.add_argument("--nfe", type=int, default=16, help="ODE time points; forwards=2*(nfe-1) under CFG/APG")
    ap.add_argument("--guidance", default="apg", choices=["cfg", "apg"])
    ap.add_argument("--cfg_strength", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="items per set from --offset; 0 = to end of set")
    ap.add_argument("--offset", type=int, default=0,
                    help="start item index per set (for multi-GPU item sharding). Per-item seed stays "
                         "base + GLOBAL index (offset+local), so a sharded run is identical to a full run.")
    ap.add_argument("--device", default="cuda:0")
    # Calibrated-model persistence (svdquant / quarot_gptq): calibrate once (single-GPU) + save, then
    # load for sharded multi-GPU generation. Without this, each process would re-calibrate its own draw.
    ap.add_argument("--model", default=None,
                    help="calibrated model path (default: models/{svd,qgptq}_{1b,3p5b}_model.pt by mode)")
    ap.add_argument("--save_model", action="store_true", help="[svdquant|quarot_gptq] after calibration, torch.save the model to --model")
    ap.add_argument("--load_model", action="store_true", help="[svdquant|quarot_gptq] skip calibration and LOAD --model (for sharded gen)")
    ap.add_argument("--calibrate_only", action="store_true", help="[svdquant|quarot_gptq] calibrate (+--save_model) then EXIT before generation")
    args = ap.parse_args()
    dev = torch.device(args.device)

    calibrated_mode = args.mode in ("svdquant", "quarot_gptq", "gptq", "smoothquant")
    if calibrated_mode and (args.load_model or args.save_model or args.calibrate_only):
        from .paths import gptq_model_path, qgptq_model_path, sq_model_path, svd_model_path
        path_fn = {"svdquant": svd_model_path, "quarot_gptq": qgptq_model_path,
                   "gptq": gptq_model_path, "smoothquant": sq_model_path}[args.mode]
        model_path = args.model or str(path_fn(args.model_dir))

    if calibrated_mode and args.load_model:
        # consistency mode: reuse the ONE calibrated model (produced once by --save_model) — every shard
        # loads the SAME draw, so a sharded run is internally consistent (calibration is non-deterministic
        # unless --calib_seed is pinned, and even then one shared model avoids any per-shard drift).
        print(f"[{args.mode}] loading calibrated model from {model_path}", flush=True)
        model = torch.load(model_path, weights_only=False, map_location=dev); model.eval()
        tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    else:
        model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
        model.vae.to_half(); model.eval()
        tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        prep(args.mode, model, tok, dev, args.rank, args.calib_seed, args.w_clip_ratio,
             svd_rows=args.svd_rows, svd_iters=args.svd_iters, svd_asym=args.svd_asym,
             calib_lst=args.calib_lst, sq_alpha=args.sq_alpha)
        if calibrated_mode and (args.save_model or args.calibrate_only):
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save(model, model_path)
            print(f"[{args.mode}] saved calibrated model -> {model_path}", flush=True)

    if calibrated_mode and args.calibrate_only:
        print(f"[{args.mode}] calibrate_only: model ready, skipping generation")
        print("GEN DONE")
        return

    for setname in [s.strip() for s in args.sets.split(",") if s.strip()]:
        _all = rs.load_items(os.path.join(rs.DATA, rs.SETS[setname]))
        items = _all[args.offset:(args.offset + args.limit) if args.limit else None]
        outdir = os.path.join(str(GEN_DIR), "paired", args.tag, setname)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        n_invalid = n_err = 0
        iterator = tqdm(
            enumerate(items),
            total=len(items),
            desc=f"gen {args.tag}/{setname}",
            dynamic_ncols=True,
        )
        for idx, (uid, pt, pwa, gt) in iterator:
            gidx = args.offset + idx                       # global index -> seed is shard-invariant
            op = os.path.join(outdir, f"{uid}.wav")
            if os.path.exists(op):
                iterator.set_postfix_str("skip existing")
                continue
            torch.manual_seed(args.base + gidx); torch.cuda.manual_seed(args.base + gidx)   # per-item, order/shard-independent
            try:
                wav = infer_one(gt, pt, pwa, model, tok, dev, args.nfe, args.cfg_strength, args.guidance)
                if not _valid_wav(wav):
                    n_invalid += 1
                    tqdm.write(f"[{args.tag}/{setname} {idx}] INVALID {uid}: non-finite or all-zero output (not written)")
                    continue
                sf.write(op, wav, model.config.sampling_rate)
                iterator.set_postfix_str(uid[:32])
            except Exception as e:
                n_err += 1
                tqdm.write(f"[{args.tag}/{setname} {idx}] ERR {uid}: {e}")
        print(f"[{args.tag}/{setname}] {len(items)} items in {time.time()-t0:.0f}s -> {outdir}"
              + (f"  [WARN invalid={n_invalid} err={n_err}]" if (n_invalid or n_err) else ""), flush=True)
    print("GEN DONE")


if __name__ == "__main__":
    main()
