"""Capture the E2 probe set ONCE and persist it, so every candidate scoring run reuses it.

The probe list comes from `calib.pool probe` (unused speakers + fresh hardlike; zero overlap
with pool candidates and test). Capture cost ~5 min GPU; the saved .pt is loaded by
`sensitivity.score --probe_capture`.

Usage:
  python -m audio_dit_quantize.calib.sensitivity.probe_capture \
      --probe_lst data/calib_pool/probe/probe_v1.lst \
      --out data/calib_pool/probe/probe_capture.pt [--seed 777]
"""
import argparse, time

import torch

import audiodit  # noqa
from audiodit import AudioDiTModel
from transformers import AutoTokenizer

from ...flatquant_best import load_items
from .probes import capture_tagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_lst", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--per_item_keep", type=int, default=2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()
    items = load_items(args.probe_lst)
    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    tagged = capture_tagged(model, tok, dev, items, per_item_keep=args.per_item_keep)
    torch.save({"tagged": tagged,
                "meta": {"probe_lst": args.probe_lst, "model_dir": args.model_dir,
                         "seed": args.seed, "per_item_keep": args.per_item_keep,
                         "n_items": len(items), "n_seqs": len(tagged)}}, args.out)
    print(f"[probe-cap] {len(tagged)} seqs -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
