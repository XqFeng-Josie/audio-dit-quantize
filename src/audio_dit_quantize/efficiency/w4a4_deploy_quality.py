"""Best-config real-int4 deploy, GLUE-precision configurable — for the "both quality + efficiency" question.

This module is BOTH the deploy implementation (imported by w4a4_deploy_check_numerics) AND a runnable
end-to-end quality eval — see main() at the bottom:
  python -m audio_dit_quantize.efficiency.w4a4_deploy_quality --model bc_1b_model.pt --set hard --glue fp32 --out_subdir dep_bc_hard_fp32


Unlike the per-linear SYMMETRIC deploy (no LAC, no add_diag), this handles the BEST-CONFIG
FlatQuant frozen linear: ASYMMETRIC activation + LAC (learnable activation clip) + add_diag. The int4 GEMM is
integer-exact, so we do it as a float matmul of the codes (== the real CUTLASS int4b_t kernel, which accumulates
in int32); glue precision (Kron transform + per-token scale/zero + dequant) is done in `glue_dtype`. This lets us
answer: does best-config's fp16 GLUE keep quality (unlike per-linear's 18.2%)?

Faithful to the fake-quant get_scale_zero (asym branch) + ActivationQuantizer(lac) + SVDDecomposeTransMatrix(add_diag):
  xf   = x * diag_scale                         (add_diag, BEFORE the transform)
  xt   = kronecker_matmul(xf, L, R)             (Kron transform, in glue_dtype)
  xmax,xmin = per-token amax/amin(xt); clamp 0 into range
  if lac:  xmax *= sigmoid(clip_a_max); xmin *= sigmoid(clip_a_min)
  scale = (xmax - xmin)/15;  zero = round(-xmin/scale)          (asym int4, q in [0,15])
  q_x  = clamp(round(xt/scale) + zero, 0, 15)                   (codes from the glue_dtype grid)
  y    = scale_x*scale_w * [ (q_x-8) @ q_w^T + (8 - zero)·rowsum(q_w) ]   (signed-kernel form + asym correction)
The int GEMM (q_x-8)@q_w and the correction are integer-exact (fp32 accumulate, no overflow); only the transform,
scale/zero, and dequant scales carry `glue_dtype`. glue_dtype=fp32 ⇒ Gate-0 == best-config fake-quant (~1e-6).

Real-kernel LATENCY is measured separately (profile_efficiency.py / w4a4_deploy_fp32glue.py) — the codes here are
bit-identical to what the signed int4 kernel + correction would produce, so quality transfers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flatquant.flat_utils import kronecker_matmul
from .. import flatquant_layers as fq


@torch.no_grad()
def _asym_lac_scale_zero(xt2, lac, clip_a_max, clip_a_min, g):
    """Replicate ActivationQuantizer.get_scale_zero (asym) + LAC, in dtype g. xt2 = [M, K]. Returns scale,zero [M,1].
    clip_a_max/clip_a_min are the raw (pre-sigmoid) LAC factors (or None when lac is off)."""
    xmax = xt2.amax(1, keepdim=True)
    xmin = xt2.amin(1, keepdim=True)
    z = torch.zeros_like(xmax)
    xmax = torch.maximum(xmax, z)
    xmin = torch.minimum(xmin, z)
    if lac:
        xmax = xmax * torch.sigmoid(clip_a_max.to(g))
        xmin = xmin * torch.sigmoid(clip_a_min.to(g))
    tmp = (xmin == 0) & (xmax == 0)
    xmin = torch.where(tmp, -torch.ones_like(xmin), xmin)
    xmax = torch.where(tmp, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / 15.0
    zero = torch.round(-xmin / scale)
    return scale, zero


class W4A4BCDeployLinear(nn.Module):
    """Real-int4 deploy of a FROZEN best-config FlatQuantLinear, glue in glue_dtype (int GEMM exact float matmul)."""
    def __init__(self, fql, glue_dtype=torch.float32):
        super().__init__()
        assert getattr(fql, "_frozen", False), "fql must be frozen"
        assert fql.use_trans and hasattr(fql.trans, "matrix_left"), "needs eval-mode Kron transform"
        self.g = glue_dtype
        self.lac = fql.lac
        dev = fql.linear.weight.device
        self.register_buffer("mL", fql.trans.matrix_left.detach().float())
        self.register_buffer("mR", fql.trans.matrix_right.detach().float())
        ds = (fql.trans.diag_scale.detach().float()
              if (getattr(fql.trans, "add_diag", False) and getattr(fql.trans, "use_diag", True)) else None)
        self.register_buffer("diag_scale", ds)                                # [K] or None
        if self.lac:
            self.register_buffer("clip_a_max", fql.aq.clip_factor_a_max.detach().float())
            self.register_buffer("clip_a_min", fql.aq.clip_factor_a_min.detach().float())
        # weight int4 codes (signed, fp32-exact from the frozen dequant weight) + per-channel scale
        N = fql.linear.weight.shape[0]
        wscale = fql.wq.scale.reshape(N, 1).float()                           # [N,1]
        q_w = torch.clamp(torch.round(fql.linear.weight.detach().float() / wscale), -8, 7)  # [N,K] signed
        self.register_buffer("q_w", q_w)                                      # [N,K]
        self.register_buffer("wscale", wscale.reshape(1, N))                  # [1,N]
        self.register_buffer("rowsum_qw", q_w.sum(dim=1).reshape(1, N))       # [1,N] = Σ_k q_w[n,k]
        self.bias = fql.linear.bias
        self.K = q_w.shape[1]
        self._act_on = True                                                  # parity w/ the fake-quant gate

    @torch.no_grad()
    def forward(self, x):
        g = self.g
        sh = x.shape
        xf = x.reshape(-1, sh[-1]).to(g)
        if self.diag_scale is not None:
            xf = xf * self.diag_scale.to(g)                                   # add_diag BEFORE transform
        xt = kronecker_matmul(xf, self.mL.to(g), self.mR.to(g)).reshape(-1, self.K)   # Kron transform (glue dtype)
        act_on = fq._ACT_QUANT and getattr(self, "_act_on", True)
        if not act_on:                                                       # act-quant gate off -> fp activation
            y = F.linear(xt.float(), (self.q_w * self.wscale.reshape(-1, 1)).float(),
                         self.bias.float() if self.bias is not None else None)
            return y.to(x.dtype).reshape(*sh[:-1], -1)
        scale_x, zero_x = _asym_lac_scale_zero(
            xt, self.lac, self.clip_a_max if self.lac else None,
            self.clip_a_min if self.lac else None, g)                          # [M,1], [M,1] in glue dtype
        q_x = torch.clamp(torch.round(xt / scale_x) + zero_x, 0, 15)         # codes [0,15] from glue grid
        q_x_s = (q_x - 8.0).float()                                          # signed [-8,7] for the int4 kernel
        gemm = q_x_s @ self.q_w.t().float()                                  # [M,N] exact int GEMM (fp32 accumulate)
        corr = (8.0 - zero_x.float()) @ self.rowsum_qw.float()               # [M,N] asym zero-point correction
        y = (gemm + corr) * scale_x.to(g).float() * self.wscale.float()       # dequant (scales carry glue precision)
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*sh[:-1], -1)


@torch.no_grad()
def bc_fakequant_ref(fql, x):
    """The best-config fake-quant reference = the actual frozen forward (trans + aq[asym+LAC] + baked weight), fp32."""
    xt = fql._apply_xtrans(x.float())
    xq = fql.aq(xt).float()
    b = fql.linear.bias.float() if fql.linear.bias is not None else None
    return F.linear(xq, fql.linear.weight.float(), b)


def wrap_dit_bc_deploy(model, glue_dtype=torch.float32):
    """Swap every FROZEN best-config FlatQuantLinear under the DiT for a W4A4BCDeployLinear (glue_dtype).
    Returns the count swapped. Skips linears whose in/out are not %32 (int4 GEMM constraint) — those stay fake-quant."""
    from ..flatquant_layers import FlatQuantLinear
    n = 0
    for parent in model.transformer.modules():
        for attr, child in list(parent.named_children()):
            if isinstance(child, FlatQuantLinear) and getattr(child, "_frozen", False):
                if child.linear.in_features % 32 or child.linear.out_features % 32:
                    continue
                setattr(parent, attr, W4A4BCDeployLinear(child, glue_dtype).to(child.linear.weight.device))
                n += 1
    return n


# ── end-to-end QUALITY eval (run as `python -m audio_dit_quantize.efficiency.w4a4_deploy_quality`) ──
# Gen with the deploy path (asym+LAC+add_diag, glue fp32 or fp16) and write wavs; score SIM/WER separately.
#   fp32-glue -> reproduces the best-config fake-quant (recovered; int4 GEMM exact, Gate-0 ~1e-6)
#   fp16-glue -> the FAST path; does best-config keep quality (unlike per-linear's 18.2%)?
# Loads the saved best-config model (one calibration) and swaps FlatQuantLinears for W4A4BCDeployLinear.
# Real-kernel latency is measured separately (profile_efficiency + w4a4_deploy_fp32glue); numeric exactness
# is checked by w4a4_deploy_check_numerics.
@torch.no_grad()
def main():
    import argparse, os, time
    import soundfile as sf
    import audiodit  # noqa
    from transformers import AutoTokenizer
    from batch_inference import infer_one
    from ..flatquant_best import load_items
    from ..paths import DATA_DIR, GEN_DIR, SETS, bc_model_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B",
                    help="picks the fixed model default (1B -> models/bc_1b_model.pt, 3.5B -> bc_3p5b_model.pt)")
    ap.add_argument("--model", default=None,
                    help="fixed best-config model path (default: from --model_dir; override dir with SEED_MODELS_DIR)")
    ap.add_argument("--set", default="hard")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--glue", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--base", type=int, default=1024)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--cfg_strength", type=float, default=4.0)
    ap.add_argument("--guidance", default="apg")
    ap.add_argument("--out_subdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    gdt = torch.float16 if args.glue == "fp16" else torch.float32
    model_path = args.model or str(bc_model_path(args.model_dir))

    model = torch.load(model_path, weights_only=False, map_location=dev); model.eval()
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    n = wrap_dit_bc_deploy(model, glue_dtype=gdt)
    print(f"[dep] swapped {n} linears -> W4A4BCDeployLinear (glue={args.glue}); model={model_path}", flush=True)

    _all = load_items(os.path.join(str(DATA_DIR), SETS[args.set]))
    items = _all[args.offset : (args.offset + args.limit) if args.limit else None]
    eff_base = args.base + args.offset
    outdir = os.path.join(str(GEN_DIR), args.out_subdir)
    os.makedirs(outdir, exist_ok=True)
    print(f"[dep] gen {len(items)} {args.set} items (glue={args.glue})", flush=True)
    t0 = time.time()
    for idx, (uid, pt, pwa, gt) in enumerate(items):
        op = os.path.join(outdir, f"{uid}.wav")
        if os.path.exists(op):
            continue
        torch.manual_seed(eff_base + idx); torch.cuda.manual_seed(eff_base + idx)
        try:
            wav = infer_one(gt, pt, pwa, model, tok, dev, args.nfe, args.cfg_strength, args.guidance)
            sf.write(op, wav, model.config.sampling_rate)
        except Exception as e:
            print(f"  [{idx}] ERR {uid}: {e}", flush=True)
    print(f"[dep] DONE {len(items)} in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    print("DEPLOY-BC-QUALITY DONE")


if __name__ == "__main__":
    main()
