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

Granularity / rank / smoothing / refinement match deepcompressor `configs/svdquant/{int4,__default__}.yaml`
(wgts sint4 group64 sym; ipts sint4 group64 dynamic sym; rank 32; smooth GridSearch with the full
39-candidate (α,β) grid: identity + pure-activation (α,0) + two-sided (α,1−α); iterative low-rank
refinement num_iters=100 with early_stop, SVD in float64, OutputsError objective with activation
quant active — `calib/lowrank.py`). **Still omitted** (documented): the optional GPTQ residual
overlay (`svdquant/gptq.yaml` — NOT part of the headline SVDQ row) and the static activation
shift for post-nonlinearity inputs (`shift_activations`; minor, affects only ffn down-proj).

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
NUM_ITERS = 100     # __default__.yaml low_rank.num_iters (early_stop usually exits much sooner)


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
def quant_act_group_sym(x, bits=4, group=GROUP):
    """Per-group dynamic SYMMETRIC fake-quant along last dim (deepcompressor sint4, no zero-point —
    the deployable Nunchaku INT4 format). scale = group absmax / qmax, round clamped to the full
    signed range [-(qmax+1), qmax]. x: [..., in] -> dequantized (same shape)."""
    qmax = 2 ** (bits - 1) - 1                    # int4->7
    *lead, inf = x.shape
    pad = _pad_to_group(inf, group)
    xf = F.pad(x, (0, pad)) if pad else x
    xg = xf.reshape(*lead, -1, group)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    q = torch.clamp(torch.round(xg / scale), -(qmax + 1), qmax)
    deq = (q * scale).reshape(*lead, -1)
    return deq[..., :inf] if pad else deq


@torch.no_grad()
def quant_act_group_asym(x, bits=4, group=GROUP):
    """LEGACY (pre paper-best alignment): per-group dynamic asymmetric fake-quant along last dim.
    Kept for the a_sym=False ablation only — the official deepcompressor int4 recipe (and the
    deployable Nunchaku kernel) has NO per-group zero-point; use quant_act_group_sym instead."""
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
                 smooth_grid=None, group=GROUP, num_iters=NUM_ITERS, a_sym=True):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.w_bits, self.a_bits, self.group = w_bits, a_bits, group
        self.rank = min(rank, self.in_features, self.out_features)
        self.num_iters = num_iters
        self.a_sym = a_sym          # paper-best: symmetric (sint4, no zero-point); False = legacy ablation
        # Smoothing GridSearch over (α, β) exponents of s = |X|maxᵃ / |W|maxᵇ — the official beta:-2
        # expansion (config/smooth.py): identity (0,0) + pure-activation (α,0) + two-sided (α,1−α),
        # α ∈ {1/20..19/20} -> 39 candidates.
        if smooth_grid is None:
            choices = [i / 20 for i in range(1, 20)]
            smooth_grid = [(0.0, 0.0)] + [(a, 0.0) for a in choices] + [(a, 1.0 - a) for a in choices]
        self.smooth_grid = smooth_grid
        self.register_buffer("smooth", torch.ones(self.in_features, device=linear.weight.device))
        self.L = None; self.R = None; self.res_wq = None
        self.alpha = None; self.beta = None
        self._calibrated = False

    def _quant_act(self, x):
        fn = quant_act_group_sym if self.a_sym else quant_act_group_asym
        return fn(x, self.a_bits, self.group)

    @torch.no_grad()
    def _refine(self, Wp, Xs, y_fp, num_iters):
        """Alternating low-rank refinement (deepcompressor calib/lowrank.py, compensate=False):
        qw=0; repeat { SVD(Wp - qw) -> rank-r (L,R); qw = Q(Wp - L R) }, scoring each iterate by
        the REAL output error with activation quant active (OutputsError, degree 2); keep the
        best-so-far and stop on the first error increase (early_stop). num_iters=1 == one-shot SVD.
        Returns (err, L, R, res_q)."""
        r = self.rank
        xq = self._quant_act(Xs)
        qw = torch.zeros_like(Wp)
        best = None
        for _ in range(num_iters):
            U, S, Vh = torch.linalg.svd((Wp - qw).double(), full_matrices=False)   # official: float64
            L = (U[:, :r] * S[:r]).float()
            R = Vh[:r, :].float()
            qw = quant_weight_group_sym(Wp - L @ R, self.w_bits, self.group)       # scales recomputed
            y = xq @ qw.t() + (Xs @ R.t()) @ L.t()
            err = (y - y_fp).pow(2).sum().item()
            if best is None or err <= best[0]:
                best = (err, L, R, qw)
            else:
                break                                                              # early_stop
        return best

    @torch.no_grad()
    def calibrate(self, X_calib: torch.Tensor):
        """X_calib: [N, in] sample of this linear's inputs across timesteps.
        Stage 1: grid-search (α, β) by output MSE with the low-rank branch active (one-shot SVD per
        candidate — allow_low_rank). Stage 2: full iterative refinement on the winner. Freeze."""
        W = self.linear.weight.data.float()                       # [out, in]
        Xc = X_calib.float()
        w_absmax = W.abs().amax(dim=0).clamp(min=1e-5)           # [in]
        a_absmax = Xc.abs().amax(dim=0).clamp(min=1e-5)          # [in]
        y_fp = Xc @ W.t()                                         # reference output

        best = None
        for alpha, beta in self.smooth_grid:
            s = (a_absmax.pow(alpha) / w_absmax.pow(beta)).clamp(min=1e-5)   # [in]; (0,0) -> ones
            err, _, _, _ = self._refine(W * s.view(1, -1), Xc / s.view(1, -1), y_fp, num_iters=1)
            if best is None or err < best[0]:
                best = (err, alpha, beta, s)
        _, self.alpha, self.beta, s = best
        _, L, R, res_q = self._refine(W * s.view(1, -1), Xc / s.view(1, -1), y_fp,
                                      num_iters=self.num_iters)
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
        xq = self._quant_act(xs).to(x.dtype)
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


def wrap_dit(model, w_bits=4, a_bits=4, rank=RANK, group=GROUP, num_iters=NUM_ITERS, a_sym=True):
    """Wrap target linears with SVDQuantLinear (NOT yet calibrated — caller feeds activations).
    num_iters=1 + a_sym=False reproduces the legacy one-shot/asym variant as an ablation."""
    wrapped = []
    for parent, attr, lin in list(_target_linears(model.transformer)):
        sq = SVDQuantLinear(lin, w_bits, a_bits, rank=rank, group=group,
                            num_iters=num_iters, a_sym=a_sym).to(lin.weight.device)
        setattr(parent, attr, sq)
        wrapped.append((parent, attr, sq))
    return wrapped
