"""E2 CLI: score ONE calibration candidate list without training (docs §3.4 E2).

Pipeline (one GPU, ~10-15 min, capture-dominated):
  load fp model -> capture per-item states (infer_one per item, per_item_keep snapshots)
  -> grad probes on the UNQUANTIZED path (per-item influence + per-channel Fisher)
  -> wrap W4A4 (frozen best-config flags) -> init block losses (deterministic loss_first)

Outputs:
  <out>.json  meta + set-level aggregates + per-item scores  (the P2 selection inputs)
  <out>.npz   init_loss[blocks, seqs], fisher[blocks, dim]   (the GATE-A weight inputs)

Usage (after `source env.sh`):
  python -m audio_dit_quantize.calib.sensitivity.score \
      --calib_lst data/calib_pool/sets/rand32_s0.lst \
      --out data/calib_pool/scores/rand32_s0 [--probes 2] [--seed 0] [--device cuda:0]
"""
import argparse, json, os, time

import numpy as np
import torch

import audiodit  # noqa
from audiodit import AudioDiTModel
from transformers import AutoTokenizer

from ... import flatquant_layers as fq
from ...flatquant_best import load_items
from .probes import (aggregate_per_item, capture_tagged, grad_probes, init_losses,
                     transfer_and_coverage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_lst", required=True)
    ap.add_argument("--out", required=True, help="output stem; writes <out>.json + <out>.npz")
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--per_item_keep", type=int, default=2)
    ap.add_argument("--probes", type=int, default=2, help="Rademacher probes per captured seq")
    ap.add_argument("--region", default="gen", choices=["gen", "prompt", "all"],
                    help="task-proxy token region (GATE-B: content pathway = gen)")
    ap.add_argument("--seed", type=int, default=0, help="capture noise + probe RNG (mirrors calib_seed)")
    ap.add_argument("--probe_capture", default=None,
                    help="probe_capture.pt from sensitivity.probe_capture — enables the v2 scores "
                         "(sum_transfer_loss, mean_coverage: candidate stats vs hard-like probe set)")
    ap.add_argument("--diag_alpha", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    items = load_items(args.calib_lst)
    print(f"[e2] {len(items)} items from {args.calib_lst}")
    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half(); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    tagged = capture_tagged(model, tok, dev, items, per_item_keep=args.per_item_keep)
    print(f"[e2] captured {len(tagged)} seqs ({time.time()-t0:.0f}s)")

    infl, infl_tok, fisher = grad_probes(model, tagged, dev,
                                         n_probes=args.probes, seed=args.seed, region=args.region)
    print(f"[e2] grad probes done ({time.time()-t0:.0f}s)")

    fq.wrap_dit(model, w_bits=4, a_bits=4, use_trans=True, lwc=True,
                a_sym=True, lac=True, add_diag=True)          # frozen best-config flags
    L = init_losses(model, tagged, dev, diag_alpha=args.diag_alpha)
    print(f"[e2] init losses done ({time.time()-t0:.0f}s)")

    T = cov = None
    if args.probe_capture:
        pc = torch.load(args.probe_capture, weights_only=False)
        print(f"[e2] probe: {pc['meta']['n_seqs']} seqs from {pc['meta']['probe_lst']}")
        T, cov = transfer_and_coverage(model, tagged, pc["tagged"], dev,
                                       diag_alpha=args.diag_alpha)
        print(f"[e2] transfer+coverage done ({time.time()-t0:.0f}s)")

    per_item = {}
    for key, vec in (("influence", infl), ("influence_per_token", infl_tok),
                     ("loss_init", L.mean(axis=0))):
        for uid, v in aggregate_per_item(tagged, vec).items():
            per_item.setdefault(uid, {})[key] = v
    payload = {
        "meta": {"calib_lst": args.calib_lst, "model_dir": args.model_dir, "seed": args.seed,
                 "per_item_keep": args.per_item_keep, "probes": args.probes, "region": args.region,
                 "n_items": len(items), "n_seqs": len(tagged), "proxy": "last_block_output",
                 "wrap": "w4a4 trans lwc sym lac diag (frozen best-config)"},
        "set_scores": {
            "sum_loss_init": float(L.sum(axis=0).mean()),      # per-seq block-sum, averaged
            "mean_loss_init": float(L.mean()),
            "mean_influence": float(np.mean(infl)),
            "mean_influence_per_token": float(np.mean(infl_tok)),
            **({"sum_transfer_loss": float(T.sum(axis=0).mean()),   # LOW = good (S prepares P well)
                "mean_transfer_loss": float(T.mean()),
                "mean_coverage": float(np.mean(cov))}               # HIGH = good (S spans P's range)
               if T is not None else {}),
        },
        "per_item": per_item,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    extra = {"transfer_loss": T, "coverage": cov} if T is not None else {}
    np.savez_compressed(args.out + ".npz", init_loss=L, fisher=fisher,
                        seq_uids=np.array([t[0] for t in tagged]), **extra)
    print(f"[e2] set_scores: { {k: round(v, 4) for k, v in payload['set_scores'].items()} }")
    print(f"[e2] -> {args.out}.json / .npz  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
