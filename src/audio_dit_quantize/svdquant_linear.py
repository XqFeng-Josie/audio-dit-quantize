"""Apply SVDQuant (W4A4) to the LongCat-AudioDiT DiT — faithful-core fake-quant quality study.

SVDQuant (Liu et al., ICLR'25). Per linear y = X Wᵀ (W is [out,in]):
  1. SMOOTH   : per-in-channel factor s migrates activation outliers into the weight,
                X Wᵀ = (X/s)(W·s)ᵀ.  s_j = max|X_j|^α / max|W_:,j|^(1-α).  α grid-searched
                per linear to minimise output MSE (deepcompressor: GridSearch / OutputsError).
  2. LOW-RANK : SVD of W' = W·diag(s) → keep rank-r  L=U[:, :r]Σ[:r] (out×r), R=Vᵀ[:r] (r×in)
                in fp16; residual Wres = W' - L R has outliers absorbed → safe to 4-bit.
  3. QUANTIZE : Wres → INT4 **per-group (g=64) symmetric**; X' = X/s → INT4 **per-group (g=64)
                dynamic asymmetric** (zero-point absorbs the activation shift). Low-rank fp16.
                y = INT4(X')·INT4(Wres)ᵀ  +  X'(L R)ᵀ  + b.

Granularity / rank / smoothing match deepcompressor `configs/svdquant/{int4,__default__}.yaml`
(wgts sint4 group64; ipts sint4 group64 dynamic; rank 32; smooth GridSearch α). **Faithfully
omitted** (documented simplifications vs the official optimiser): the iterative low-rank
refinement (num_iters=100, OutputsError) — we use one-shot SVD; optional GPTQ refinement; and
the separate static activation-shift (we fold it into per-group dynamic *asymmetric* act quant).

Contrast with FlatQuant (flatquant_layers.py): FlatQuant learns Kronecker rotations (per-layer
gradient training); SVDQuant absorbs outliers with closed-form SVD + an α search (no grad
training). Both fake-quant here; SVDQuant's Nunchaku INT4 kernels DO support A100, so it is the
one W4A4 route that could become a real A100 speedup later (see 04-flatquant.md / 07-matrix).
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 64          # deepcompressor int4.yaml group_shapes [1,64]
RANK = 32           # __default__.yaml low_rank.rank


def _pad_to_group(n, g=GROUP):
    return (g - n % g) % g


@torch.no_grad()
def quant_weight_group_sym(w, bits=4, group=GROUP):
    """Per-group (along in-dim) symmetric fake-quant. w: [out, in] -> dequantized [out, in]."""
    qmax = 2 ** (bits - 1) - 1                    # int4->7, int8->127
    out, inf = w.shape
    pad = _pad_to_group(inf, group)
    if pad:
        w = F.pad(w, (0, pad))
    wg = w.reshape(out, -1, group)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    q = torch.clamp(torch.round(wg / scale), -qmax, qmax)
    deq = (q * scale).reshape(out, -1)
    return deq[:, :inf] if pad else deq


@torch.no_grad()
def quant_act_group_asym(x, bits=4, group=GROUP):
    """Per-group dynamic asymmetric fake-quant along last dim. x: [..., in]. Zero-point folds
    in the activation shift. Returns dequantized x (same shape)."""
    qmax = 2 ** bits - 1                          # int4->15, int8->255 (asymmetric, unsigned)
    *lead, inf = x.shape
    pad = _pad_to_group(inf, group)
    xf = F.pad(x, (0, pad)) if pad else x
    xg = xf.reshape(*lead, -1, group)
    xmin = xg.amin(dim=-1, keepdim=True)
    xmax = xg.amax(dim=-1, keepdim=True)
    scale = ((xmax - xmin) / qmax).clamp(min=1e-8)
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(xg / scale) + zero, 0, qmax)
    deq = ((q - zero) * scale).reshape(*lead, -1)
    return deq[..., :inf] if pad else deq


class SVDQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, w_bits=4, a_bits=4, rank=RANK,
                 alpha_grid=None, group=GROUP):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.w_bits, self.a_bits, self.group = w_bits, a_bits, group
        self.rank = min(rank, self.in_features, self.out_features)
        # GridSearch over α (deepcompressor num_grids=20, centred 0.5)
        self.alpha_grid = alpha_grid if alpha_grid is not None else [i / 20 for i in range(1, 20)]
        self.register_buffer("smooth", torch.ones(self.in_features, device=linear.weight.device))
        self.L = None; self.R = None; self.res_wq = None
        self.alpha = None
        self._calibrated = False

    @torch.no_grad()
    def _build(self, alpha, W, a_absmax, w_absmax):
        s = (a_absmax.pow(alpha) / w_absmax.pow(1 - alpha)).clamp(min=1e-5)   # [in]
        Wp = W * s.view(1, -1)
        U, S, Vh = torch.linalg.svd(Wp, full_matrices=False)
        r = self.rank
        L = (U[:, :r] * S[:r]); R = Vh[:r, :]
        res = Wp - L @ R
        res_q = quant_weight_group_sym(res, self.w_bits, self.group)
        return s, L, R, res_q

    @torch.no_grad()
    def calibrate(self, X_calib: torch.Tensor):
        """X_calib: [N, in] sample of this linear's inputs across timesteps. Grid-search α to
        minimise output MSE (||X Wᵀ - ŷ||), then freeze the chosen smoothing + low-rank + INT4."""
        W = self.linear.weight.data.float()                       # [out, in]
        Xc = X_calib.float()
        w_absmax = W.abs().amax(dim=0).clamp(min=1e-5)           # [in]
        a_absmax = Xc.abs().amax(dim=0).clamp(min=1e-5)          # [in]
        y_fp = Xc @ W.t()                                         # reference output

        best = None
        for alpha in self.alpha_grid:
            s, L, R, res_q = self._build(alpha, W, a_absmax, w_absmax)
            xs = Xc / s.view(1, -1)
            xq = quant_act_group_asym(xs, self.a_bits, self.group)
            y_q = xq @ res_q.t() + (xs @ R.t()) @ L.t()
            err = (y_q - y_fp).pow(2).mean().item()
            if best is None or err < best[0]:
                best = (err, alpha, s, L, R, res_q)
        _, self.alpha, s, L, R, res_q = best
        self.smooth.copy_(s.to(self.smooth.dtype))
        dt = self.linear.weight.dtype
        self.L = L.to(dt).contiguous(); self.R = R.to(dt).contiguous()
        self.res_wq = res_q.to(dt).contiguous()
        self.linear.weight = None                                # free fp weight
        self._calibrated = True

    def forward(self, x):
        assert self._calibrated, "call calibrate() first"
        s = self.smooth.to(x.dtype)
        xs = x / s
        low = F.linear(F.linear(xs, self.R), self.L)             # X'(L R)ᵀ  fp16 branch
        xq = quant_act_group_asym(xs, self.a_bits, self.group).to(x.dtype)
        main = F.linear(xq, self.res_wq)                         # INT4 group residual (fake)
        out = main + low
        if self.linear.bias is not None:
            out = out + self.linear.bias
        return out


# ── same target-linear selection as FlatQuant (attn/cross_attn/ffn GEMMs) ────────────
def _target_linears(transformer):
    seen = set()
    for bname, block in transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in sub.named_children():
                if isinstance(child, nn.Linear) and id(child) not in seen:
                    seen.add(id(child))
                    yield sub, attr, child


def wrap_dit(model, w_bits=4, a_bits=4, rank=RANK, group=GROUP):
    """Wrap target linears with SVDQuantLinear (NOT yet calibrated — caller feeds activations)."""
    wrapped = []
    for parent, attr, lin in list(_target_linears(model.transformer)):
        sq = SVDQuantLinear(lin, w_bits, a_bits, rank=rank, group=group).to(lin.weight.device)
        setattr(parent, attr, sq)
        wrapped.append((parent, attr, sq))
    return wrapped
