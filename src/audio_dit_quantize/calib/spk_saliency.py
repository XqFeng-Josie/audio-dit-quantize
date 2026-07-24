"""L-C offline pass: per-(block, hidden-channel) SPEAKER-perceptual saliency (TTS-Q contribution 3).

For each calibration item the FP model generates ONCE with gradients enabled ONLY through the
LAST Euler step: a patched odeint_euler runs steps 0..n-2 under the model's native no_grad,
then re-enables grad for the final step, so one graph spans

  last-step transformer forwards (cond+uncond, APG combine) -> Euler update -> gen-region slice
  -> fp32 VAE decode -> 24k->16k resample -> eval WavLM speaker embedding
  -> cos-sim vs the prompt-wav embedding  (the SIM metric itself)

One backward per item; saliency[b, c] = mean over items/passes/frames of |d sim / d h_b[.., c]|
where h_b is block b's output at the last step.  This replaces M3's variance guidance with the
PERCEPTUAL guidance the task actually scores.

Pre-registered choices: last network step only (endpoint of the late5 window the task says
carries SIM, cheapest exact chain); |grad| magnitude; fp32 VAE (no to_half) for exact decode
gradients; per-item seed = 1024 + idx (order-free, matches the generation protocol convention).

Output npz: sal [n_blocks, d] float64 (already item-averaged), n_items, meta json string.
Consumed by:  flatquant_best --loss percw --chan_w_file <npz>

Usage (single GPU, ~30-60 min for 32 items):
  python -m audio_dit_quantize.calib.spk_saliency \
      --model_dir meituan-longcat/LongCat-AudioDiT-1B \
      --calib_lst data/calib_pool/sets/rand32_s4.lst \
      --out data/calib_pool/spk_saliency_1b.npz
"""
import argparse
import json
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.functional import resample
from tqdm import tqdm

import audiodit  # noqa
from audiodit import AudioDiTModel
import audiodit.modeling_audiodit as MDL
from transformers import AutoTokenizer
from batch_inference import infer_one

from .. import seedtts_similarity as ss     # WavLM init_model + torchaudio stubs + CKPT
from ..flatquant_best import load_items


class _St:
    def __init__(self):
        self.armed = False
        self.final = None       # grad-enabled final latent [B, T, latent_dim]
        self.pd = 0             # prompt/gen boundary of the current item
        self.blocks = []        # [(block_idx, out_tensor_with_retained_grad)] for the last step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--calib_lst", default="data/calib_pool/sets/rand32_s4.lst")
    ap.add_argument("--out", default="data/calib_pool/spk_saliency_1b.npz")
    ap.add_argument("--limit", type=int, default=0, help="items; 0 = all")
    ap.add_argument("--base", type=int, default=1024, help="per-item seed = base + idx")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--target", default="sim", choices=["sim", "energy"],
                    help="saliency target scalar: sim = WavLM speaker cos-sim vs prompt wav (the "
                         "perceptual metric, default); energy = mean squared waveform amplitude — a "
                         "NON-perceptual control to test speaker-specificity of the channel profile "
                         "(same chain up to the wav, no embedder). Use a distinct --out.")
    ap.add_argument("--max_sec", type=float, default=0.0,
                    help="if >0, truncate wavs to this many seconds before the speaker embedder "
                         "(OOM guard; 0 = exact metric chain, default)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)   # NOTE: vae stays fp32 (no to_half)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    sr = model.config.sampling_rate
    for p in model.parameters():
        p.requires_grad_(False)      # input-saliency only: no param grads, saves memory

    sv = None
    if args.target == "sim":
        sv = ss.init_model("wavlm_large", checkpoint=ss.CKPT).to(dev).eval()
        for p in sv.parameters():
            p.requires_grad_(False)
        # the eval wrapper severs INPUT grads in two places (inference defaults): ECAPA.get_feat wraps
        # extraction in no_grad unless update_extract; s3prl WavLM wraps its CNN feature extractor in
        # no_grad when feature_grad_mult == 0. Re-enable both — params stay frozen regardless.
        n_u = n_f = 0
        for m in [sv] + list(sv.modules()):
            if hasattr(m, "update_extract"):
                m.update_extract = True; n_u += 1
            if hasattr(m, "feature_grad_mult"):
                m.feature_grad_mult = 1.0; n_f += 1
        print(f"[sal] grad-enabled eval chain: update_extract on {n_u}, feature_grad_mult on {n_f} module(s)")

    items = load_items(args.calib_lst)
    if args.limit:
        items = items[:args.limit]

    st = _St()
    blocks = model.transformer.blocks
    nb, d = len(blocks), None

    def mk_hook(bi):
        def hook(_m, _a, out):
            if st.armed:
                assert torch.is_tensor(out), f"block {bi} output is not a tensor"
                out.retain_grad()
                st.blocks.append((bi, out))
        return hook
    handles = [b.register_forward_hook(mk_hook(i)) for i, b in enumerate(blocks)]

    _ode = MDL.odeint_euler
    def ode_lastgrad(fn, y0, t):
        ys, y = [y0], y0
        n = len(t) - 1
        for i in range(n):
            dt = t[i + 1] - t[i]
            if i < n - 1:
                y = y + fn(t[i], y) * dt
                ys.append(y)
            else:
                with torch.enable_grad():                    # overrides forward's @no_grad
                    x_leaf = y.detach().requires_grad_(True)
                    # fn writes the prompt region in-place each step; that is illegal on views of a
                    # LEAF, so hand it a non-leaf clone (identical values, in-place-safe, same graph)
                    x = x_leaf.clone()
                    st.armed = True
                    try:
                        v = fn(t[i], x)
                    finally:
                        st.armed = False
                    yg = x + v * dt
                st.final = yg
                y = yg.detach()
                ys.append(y)
        return torch.stack(ys)
    MDL.odeint_euler = ode_lastgrad

    _enc = model.encode_prompt_audio
    def enc(pa):
        lat, pd = _enc(pa); st.pd = int(pd); return lat, pd
    model.encode_prompt_audio = enc

    sal = None      # [nb, d] float64 accumulated per-item means
    n_ok, fails = 0, []
    try:
        for idx, (uid, pt, pwa, gt) in enumerate(tqdm(items, desc="spk saliency", dynamic_ncols=True)):
            st.final, st.blocks = None, []
            torch.manual_seed(args.base + idx); torch.cuda.manual_seed(args.base + idx)
            try:
                infer_one(gt, pt, pwa, model, tok, dev, args.nfe, 4.0, "apg")
                assert st.final is not None and st.blocks, "last-step graph not captured"
                # exact metric chain: gen-region slice -> fp32 decode -> 16k -> target scalar
                emb_p = None
                if args.target == "sim":
                    pw, _ = librosa.load(pwa, sr=16000)
                    pw = torch.from_numpy(pw).float().to(dev)
                    if args.max_sec > 0:
                        pw = pw[: int(16000 * args.max_sec)]
                    with torch.no_grad():
                        emb_p = sv(pw.unsqueeze(0))
                with torch.enable_grad():
                    lat = st.final[:, st.pd:].permute(0, 2, 1).float()
                    wav = model.vae.decode(lat).squeeze(1)               # [1, T_sr], grad-connected
                    wav16 = resample(wav, orig_freq=sr, new_freq=16000)
                    if args.max_sec > 0:
                        wav16 = wav16[:, : int(16000 * args.max_sec)]
                    if args.target == "sim":
                        sim = F.cosine_similarity(sv(wav16), emb_p).squeeze()
                    else:                                                # energy control
                        sim = wav16.pow(2).mean()
                    sim.backward()
                per_item = {}
                for bi, out in st.blocks:                    # 2 passes (cond+uncond) per block
                    if out.grad is None:
                        continue
                    g = out.grad.abs().mean(dim=(0, 1)).double().cpu().numpy()   # [d]
                    per_item.setdefault(bi, []).append(g)
                assert len(per_item) == nb, f"grads reached {len(per_item)}/{nb} blocks"
                if sal is None:
                    d = len(next(iter(per_item.values()))[0])
                    sal = np.zeros((nb, d), dtype=np.float64)
                for bi, gs in per_item.items():
                    sal[bi] += np.mean(gs, axis=0)
                n_ok += 1
                if idx == 0:
                    print(f"\n[sal] item0 {args.target}={float(sim.detach()):.4g}  grad-mag per-block "
                          f"min/max={sal.min():.2e}/{sal.max():.2e}  d={d}", flush=True)
            except Exception as e:
                fails.append((uid, repr(e)))
                print(f"\n[sal] FAIL {uid}: {e}", flush=True)
            finally:
                st.final, st.blocks = None, []
                torch.cuda.empty_cache()
    finally:
        MDL.odeint_euler = _ode
        del model.encode_prompt_audio
        for h in handles:
            h.remove()

    if not n_ok:
        raise SystemExit(f"no items succeeded ({len(fails)} failures)")
    sal /= n_ok
    meta = {"model_dir": args.model_dir, "calib_lst": args.calib_lst, "n_items": n_ok,
            "n_fail": len(fails), "fails": fails[:10], "base": args.base, "nfe": args.nfe,
            "max_sec": args.max_sec, "step": "last", "reduce": "mean|grad|", "target": args.target}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, sal=sal, meta=json.dumps(meta))
    rng = sal.mean(axis=1)
    print(f"[sal] saved {args.out}: sal[{nb},{sal.shape[1]}], {n_ok} items ok, {len(fails)} failed; "
          f"per-block mean saliency range [{rng.min():.3e}, {rng.max():.3e}]")


if __name__ == "__main__":
    main()
