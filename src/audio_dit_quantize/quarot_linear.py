"""Apply QuaRot (W4A4) to the LongCat-AudioDiT DiT — self-contained fake-quant quality study.

Two variants:
  * QuaRot-RTN  (training-free): rotate + round-to-nearest, `wrap_dit(freeze=True)`.
  * QuaRot-GPTQ (paper-best, fake_quant/README headline config): rotate + GPTQ weights with
    MSE clip search — `wrap_dit(freeze=False)` then `calibrate_gptq(model, store, dev)`.
    GPTQ settings vendored from spcl/QuaRot fake_quant (gptq_utils.py / quant_utils.py):
    percdamp 0.01, blocksize 128, groupsize -1 (per-out-channel), act_order off, sym weights,
    --w_clip MSE grid search (norm 2.4, grid 100, maxshrink 0.8), sym INT4 clamp [-8, 7].
    Protocol mirrors gptq_fwrd: sequential per-block — Hessians accumulated on the ROTATED
    fp inputs of the current block, GPTQ + freeze, then inputs advance through the
    weight-quantized block with activations still fp (act quant only active at deploy).

QuaRot (Ashkboos et al., NeurIPS'24): rotate weights+activations with an orthonormal Hadamard
matrix so outlier channels are *destroyed* (energy spread across all channels), making both safe
to quantize to 4-bit. For one linear y = x Wᵀ (W is [out,in]) we conjugate the contraction
(in) dimension by an orthonormal R (R Rᵀ = I):

    y = x Wᵀ = (x R)(W R)ᵀ          # exact, since R Rᵀ = I

then quantize the *rotated* activation (xR, per-token sym INT4) and the *rotated* weight (WR,
per-out-channel sym INT4). The rotation is the SAME for x and W of a given linear, so it cancels.

Differences vs the LLM QuaRot and vs our FlatQuant/SVDQuant (deliberate, documented):
  * PER-LINEAR rotation only. The LLM QuaRot fuses one global rotation through the *residual
    stream* via computational invariance (RMSNorm scale folded into the next linear). A DiT uses
    AdaLN (per-timestep dynamic scale/shift) — the residual-stream trick does NOT transfer (γ/β
    can't be baked into weights). So we rotate each target linear's input independently, which is
    the part that DOES transfer (cf. QuaRot's own online Hadamards before down_proj / on v/o).
  * RTN variant is TRAINING-FREE (fixed randomized Hadamard, no calibration) — the cheap W4A4
    baseline. The GPTQ variant (paper headline; LLaMA-2-7B: RTN 6.76 vs GPTQ 6.10 PPL) needs
    one calibration pass to accumulate per-linear Hessians (see calibrate_gptq below).
  * Symmetric INT4 for both W and A (rotated activations are ~symmetric/Gaussian).

Kernel/HW note: this is FAKE-QUANT (rotate in fp32, dequantize, fp32 GEMM) — a QUALITY study.
A *real* W4A4 speedup needs INT4 Tensor Cores, which exist on A100/Ada (sm_80/89, QuaRot's
CUTLASS int4 kernel) but were DROPPED on H100/Hopper (sm_90 wgmma has no INT4) — so the real
QuaRot speed path is A100/Ada, not H100. The dense rotation here is O(n²); a deployment would use
the fast Walsh-Hadamard transform (O(n log n)). Self-contained: no flatquant_ref dependency.
"""
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Hadamard construction (vendored: get_had12 verbatim from spcl/QuaRot) ────────────
def _is_pow2(n):
    return (n & (n - 1) == 0) and n > 0


def _had12():
    return torch.tensor([
        [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
        [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
        [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
        [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
        [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
        [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
        [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
        [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
        [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
        [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
    ], dtype=torch.float64)


def _sylvester(m):
    """Dense 2^k Hadamard of size m (m a power of 2): H Hᵀ = m·I."""
    H = torch.ones((1, 1), dtype=torch.float64)
    while H.shape[0] < m:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def _unnormalized_hadamard(n):
    """±1 Hadamard of order n (H Hᵀ = n·I), via had12 ⊗ Sylvester or pure Sylvester."""
    if n % 12 == 0 and _is_pow2(n // 12):
        return torch.kron(_had12(), _sylvester(n // 12))
    if _is_pow2(n):
        return _sylvester(n)
    raise ValueError(
        f"no Hadamard factorization for in_features={n} "
        f"(need n = 12·2^m or 2^m); extend with had20/had28 if a new dim appears")


# rotation matrices are shared across all linears with the same in_features (one per dim)
_R_CACHE = {}


def get_rotation(n, device, seed_signs=True):
    """Orthonormal rotation R (R Rᵀ = I) for contraction dim n, cached per (n, device).
    R = diag(s) · Ĥ / √n with a fixed random sign vector s (randomized Hadamard)."""
    key = (n, str(device))
    R = _R_CACHE.get(key)
    if R is None:
        H = _unnormalized_hadamard(n)                      # [n,n], H Hᵀ = n·I (float64)
        if seed_signs:
            g = torch.Generator().manual_seed(0x5EED + n)  # deterministic per dim
            s = (torch.randint(0, 2, (n,), generator=g, dtype=torch.float64) * 2 - 1)
            H = s.view(-1, 1) * H                           # sign rows -> still orthogonal
        R = (H / (n ** 0.5)).to(torch.float32).to(device)  # orthonormal
        _R_CACHE[key] = R
    return R


def _sym_qdq(x, scale, qmax):
    """Symmetric quant-dequant with the official QuaRot clamp: scale = amax/qmax but the round is
    clamped into the FULL signed range [-(qmax+1), qmax] (quant_utils.sym_quant), i.e. [-8, 7]."""
    return torch.clamp(torch.round(x / scale), -(qmax + 1), qmax) * scale


@torch.no_grad()
def _find_wscale(w, bits=4, clip_ratio=1.0, mse=False, norm=2.4, grid=100, maxshrink=0.8):
    """Per-out-channel symmetric weight scale [out, 1]. mse=True runs the official --w_clip
    grid search (quant_utils.find_params: p = 1 - i/grid, keep min L^2.4 error per row)."""
    qmax = 2 ** (bits - 1) - 1
    amax = (w.abs().amax(dim=1) * clip_ratio).clamp(min=1e-5)
    scale = amax / qmax
    if mse:
        best = torch.full_like(amax, float("inf"))
        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid
            s1 = (p * amax) / qmax
            err = (_sym_qdq(w, s1.unsqueeze(1), qmax) - w).abs_().pow_(norm).sum(1)
            m = err < best
            best[m] = err[m]
            scale[m] = s1[m]
    return scale.unsqueeze(1)


@torch.no_grad()
def _quant_weight_perchannel_sym(w, bits=4, clip_ratio=1.0, mse=False):
    """Per-output-channel symmetric fake-quant. w: [out, in] -> dequantized [out, in]."""
    qmax = 2 ** (bits - 1) - 1
    return _sym_qdq(w, _find_wscale(w, bits, clip_ratio, mse=mse), qmax)


@torch.no_grad()
def _quant_act_pertoken_sym(x, bits=4):
    """Per-token (last-dim) symmetric dynamic fake-quant. x: [..., in]."""
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return _sym_qdq(x, amax / qmax, qmax)


# ── GPTQ solver (vendored logic: spcl/QuaRot fake_quant/gptq_utils.py, paper-best settings) ──
class _GPTQ:
    """Hessian accumulator + fasterquant for ONE linear, operating on the ROTATED in-dim."""

    def __init__(self, columns, dev):
        self.H = torch.zeros((columns, columns), device=dev)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp):
        """inp: [tokens, in] ROTATED fp activation of one calib sequence (counts as 1 sample,
        matching the official add_batch where a [B,T,C] batch counts tmp=B)."""
        self.H *= self.nsamples / (self.nsamples + 1)
        self.nsamples += 1
        x = math.sqrt(2 / self.nsamples) * inp.t().float()
        self.H += x @ x.t()

    @torch.no_grad()
    def quantize(self, W, scale, qmax, blocksize=128, percdamp=0.01):
        """fasterquant, groupsize=-1 / act_order=False (headline defaults). W: [out, in] fp32
        (rotated), consumed; returns dequantized Q. scale: [out, 1] fixed per-row sym scale."""
        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        cols = W.shape[1]
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(cols, device=W.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)
        Q = torch.zeros_like(W)
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = _sym_qdq(w.unsqueeze(1), scale, qmax).flatten()
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1
            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        if torch.isnan(Q).any():
            raise ValueError("GPTQ produced NaN weights")
        return Q


class QuaRotLinear(nn.Module):
    def __init__(self, linear: nn.Linear, w_bits=4, a_bits=4, w_clip_ratio=1.0):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.w_bits, self.a_bits, self.w_clip_ratio = w_bits, a_bits, w_clip_ratio
        self._frozen = False
        self._collecting = False    # GPTQ Hessian-accumulation mode (calibration only)
        self._gptq = None
        self.act_quant = True       # off only while advancing calib inputs (official protocol)

    def _rotate(self, t):
        """Apply the orthonormal rotation to the contraction (last) dim, in fp32."""
        R = get_rotation(self.in_features, t.device)
        return t.float() @ R

    @torch.no_grad()
    def _qrot_weight(self):
        wr = self._rotate(self.linear.weight)                       # rotate W's in-dim
        return _quant_weight_perchannel_sym(wr, self.w_bits, self.w_clip_ratio)

    def forward(self, x):
        if self._collecting:
            # Hessian on the ROTATED input; forward stays exact fp (rotation would cancel anyway)
            self._gptq.add_batch(self._rotate(x).reshape(-1, self.in_features))
            return F.linear(x, self.linear.weight, self.linear.bias)
        xr = self._rotate(x)
        xq = _quant_act_pertoken_sym(xr, self.a_bits).to(x.dtype) if self.act_quant else xr.to(x.dtype)
        if self._frozen:
            return F.linear(xq, self.linear.weight, self.linear.bias)   # weight pre-rotated+quant
        wq = self._qrot_weight().to(x.dtype)
        return F.linear(xq, wq, self.linear.bias)

    @torch.no_grad()
    def freeze(self):
        """Bake rotated+quantized weight in-place (avoids a 2nd full-size copy -> OK on 3.5B)."""
        self.linear.weight.data = self._qrot_weight().to(self.linear.weight.dtype)
        self._frozen = True

    @torch.no_grad()
    def gptq_freeze(self, w_clip_mse=True, blocksize=128, percdamp=0.01):
        """GPTQ-quantize the rotated weight against the accumulated Hessian and bake it in-place.
        w_clip_mse=True = the official --w_clip MSE scale search (paper-best)."""
        assert self._gptq is not None and self._gptq.nsamples > 0, "no Hessian accumulated"
        qmax = 2 ** (self.w_bits - 1) - 1
        wr = self._rotate(self.linear.weight)                       # fp32 [out, in]
        scale = _find_wscale(wr, self.w_bits, self.w_clip_ratio, mse=w_clip_mse)
        Q = self._gptq.quantize(wr, scale, qmax, blocksize=blocksize, percdamp=percdamp)
        self.linear.weight.data = Q.to(self.linear.weight.dtype)
        self._gptq = None
        self._frozen = True

    @torch.no_grad()
    def invariance_error(self, x):
        """Self-check: ||(rotate(x))·(rotate(W))ᵀ - x·Wᵀ|| relative error (should be ~0, fp32)."""
        ref = F.linear(x.float(), self.linear.weight.float())
        wr = self._rotate(self.linear.weight)
        got = F.linear(self._rotate(x), wr)
        return (got - ref).norm() / ref.norm().clamp(min=1e-12)


# ── same target-linear selection as FlatQuant/SVDQuant (attn/cross_attn/ffn GEMMs) ──
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


def wrap_dit(model, w_bits=4, a_bits=4, w_clip_ratio=1.0, freeze=True):
    """Wrap target DiT linears with QuaRotLinear. freeze=True bakes RTN weights immediately
    (training-free QuaRot-RTN); freeze=False leaves them fp for calibrate_gptq (QuaRot-GPTQ)."""
    wrapped = []
    for parent, attr, lin in list(_target_linears(model.transformer)):
        qr = QuaRotLinear(lin, w_bits, a_bits, w_clip_ratio).to(lin.weight.device)
        if freeze:
            qr.freeze()
        setattr(parent, attr, qr)
        wrapped.append(qr)
    return wrapped


# ── recursive device mover (mirror of flatquant_best._move; kept local so this module stays
#    importable without the audiodit/batch_inference deps that flatquant_best pulls in) ──────
def _move(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(_move(o, dev) for o in obj)
    if isinstance(obj, list):
        return [_move(o, dev) for o in obj]
    if isinstance(obj, dict):
        return {k: _move(v, dev) for k, v in obj.items()}
    return obj


@torch.no_grad()
def calibrate_gptq(model, store, dev, w_clip_mse=True, blocksize=128, percdamp=0.01):
    """Sequential per-block QuaRot-GPTQ (official gptq_fwrd protocol, paper-best settings).

    store: [(x_cpu, cond_kwargs_cpu, prompt_dur)] captured block-0 inputs + shared conditioning
    (flatquant_best.capture_block_inputs format). For each block in order: forward the calib
    seqs with fp weights while every wrapper accumulates the Hessian of its ROTATED input,
    GPTQ+freeze each linear, then advance the inputs through the weight-quantized block with
    activation quant OFF (official: act quant is configured only after all GPTQ finishes).
    """
    import time
    tf32_m, tf32_c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False   # official gptq_utils
    t0 = time.time()
    try:
        blocks = model.transformer.blocks
        inps = [x for (x, _, _) in store]
        conds = [c for (_, c, _) in store]
        n = len(inps)
        print(f"[gptq] {n} calib seqs, {len(blocks)} blocks (percdamp={percdamp}, "
              f"blocksize={blocksize}, w_clip_mse={w_clip_mse})")
        for bi, blk in enumerate(blocks):
            ws = [m for m in blk.modules() if isinstance(m, QuaRotLinear)]
            if not ws:
                continue
            for w in ws:
                w._gptq = _GPTQ(w.in_features, dev)
                w._collecting = True
            for j in range(n):
                blk(x=_move(inps[j], dev), **_move(conds[j], dev))
            for w in ws:
                w._collecting = False
                w.gptq_freeze(w_clip_mse=w_clip_mse, blocksize=blocksize, percdamp=percdamp)
            for w in ws:
                w.act_quant = False
            inps = [blk(x=_move(inps[j], dev), **_move(conds[j], dev)).detach().float().cpu()
                    for j in range(n)]
            for w in ws:
                w.act_quant = True
            torch.cuda.empty_cache()
            if (bi + 1) % 4 == 0 or bi == len(blocks) - 1:
                print(f"[gptq]  block {bi+1}/{len(blocks)} ({time.time()-t0:.0f}s)")
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32_m, tf32_c
    print(f"[gptq] done in {time.time()-t0:.0f}s")
    return model
