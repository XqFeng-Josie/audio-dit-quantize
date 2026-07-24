"""SmoothQuant W4A4 baseline (scale-migration family; Xiao et al., ICML'23).

Per-linear activation->weight difficulty migration: with per-in-channel calibration
statistics amax_x (activations, from a fp calibration pass) and amax_w (weights),

    s_j = amax_x_j^alpha / amax_w_j^(1-alpha)        (alpha = 0.5, paper default)
    y   = (x / s) @ (W * s)^T + b                    (exact identity in fp)

then W*s is quantized per-out-channel symmetric INT4 and x/s per-token symmetric INT4 —
the SAME quantizer primitives as quarot_linear, so the baseline ladder
(RTN -> SmoothQuant -> GPTQ -> QuaRot-GPTQ -> FlatQuant) differs only in the transform.

Original SmoothQuant is a W8A8 method; at W4A4 it is expected to degrade — that is the
point of the table row (smoothing alone vs rotation vs learned transforms).

Calibration protocol matches the other calibrated baselines: same capture
(flatquant_best.capture_block_inputs, 64x2, block-0 inputs + shared conditioning), one
sequential fp pass over blocks — amax_x is collected on EXACT fp activations (official
SmoothQuant semantics: statistics from the fp model), then each block's wrappers freeze.

Self-contained: imports only the quantizer primitives + target selection from
quarot_linear (shared single source of truth); no flatquant_ref dependency.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quarot_linear import (_move, _quant_act_pertoken_sym,
                            _quant_weight_perchannel_sym, _target_linears)


class SmoothQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, w_bits=4, a_bits=4, alpha=0.5):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.w_bits, self.a_bits, self.alpha = w_bits, a_bits, alpha
        self._frozen = False
        self._collecting = False
        self.act_quant = True
        # buffer so it follows .to(device) and survives torch.save/load of the whole model
        self.register_buffer("s", torch.ones(linear.in_features))
        self.register_buffer("x_amax", torch.zeros(linear.in_features))

    def forward(self, x):
        if self._collecting:
            with torch.no_grad():
                cur = x.detach().float().abs().reshape(-1, self.in_features).amax(dim=0)
                torch.maximum(self.x_amax, cur, out=self.x_amax)
        if not self._frozen:
            return F.linear(x, self.linear.weight, self.linear.bias)   # exact fp until frozen
        xs = x.float() / self.s
        xq = _quant_act_pertoken_sym(xs, self.a_bits) if self.act_quant else xs
        return F.linear(xq.to(x.dtype), self.linear.weight, self.linear.bias)

    @torch.no_grad()
    def freeze(self):
        """Compute s from the collected amax_x + weight amax, fold s into W, quantize, bake."""
        assert self.x_amax.max() > 0, "freeze() before any calibration forward"
        w = self.linear.weight.float()                                  # [out, in]
        w_amax = w.abs().amax(dim=0).clamp(min=1e-5)                    # per-IN-channel
        x_amax = self.x_amax.clamp(min=1e-5)
        self.s = (x_amax.pow(self.alpha) / w_amax.pow(1 - self.alpha)).clamp(min=1e-5)
        wq = _quant_weight_perchannel_sym(w * self.s, self.w_bits)      # fold, then W4 sym
        self.linear.weight.data = wq.to(self.linear.weight.dtype)
        self._frozen = True

    @torch.no_grad()
    def invariance_error(self, x):
        """Self-check: ||(x/s)·(W·s)ᵀ − x·Wᵀ|| relative error BEFORE quantization (~fp roundoff).
        Call only pre-freeze (needs the unfolded weight)."""
        assert not self._frozen
        w = self.linear.weight.float()
        w_amax = w.abs().amax(dim=0).clamp(min=1e-5)
        x_amax = self.x_amax.clamp(min=1e-5)
        s = (x_amax.pow(self.alpha) / w_amax.pow(1 - self.alpha)).clamp(min=1e-5)
        ref = F.linear(x.float(), w)
        got = F.linear(x.float() / s, w * s)
        return (got - ref).norm() / ref.norm().clamp(min=1e-12)


def wrap_dit(model, w_bits=4, a_bits=4, alpha=0.5):
    """Wrap the same target linears as the other baselines. NOT frozen — calibration first."""
    wrapped = []
    for parent, attr, lin in list(_target_linears(model.transformer)):
        sq = SmoothQuantLinear(lin, w_bits, a_bits, alpha).to(lin.weight.device)
        setattr(parent, attr, sq)
        wrapped.append(sq)
    return wrapped


@torch.no_grad()
def calibrate_smoothquant(model, store, dev):
    """Sequential per-block calibration: collect per-channel amax on EXACT fp forwards (the
    collect pass doubles as the input-advance pass — un-frozen wrappers compute exact fp), then
    freeze the block. store: flatquant_best.capture_block_inputs format."""
    t0 = time.time()
    blocks = model.transformer.blocks
    inps = [x for (x, _, _) in store]
    conds = [c for (_, c, _) in store]
    n = len(inps)
    n_lin = 0
    print(f"[smoothquant] {n} calib seqs, {len(blocks)} blocks")
    for bi, blk in enumerate(blocks):
        ws = [m for m in blk.modules() if isinstance(m, SmoothQuantLinear)]
        if not ws:
            continue
        for w in ws:
            w._collecting = True
        outs = [blk(x=_move(inps[j], dev), **_move(conds[j], dev)).detach().float().cpu()
                for j in range(n)]
        for w in ws:
            w._collecting = False
            w.freeze()
        n_lin += len(ws)
        inps = outs
        torch.cuda.empty_cache()
        if (bi + 1) % 8 == 0 or bi == len(blocks) - 1:
            print(f"[smoothquant]  block {bi+1}/{len(blocks)} ({time.time()-t0:.0f}s)")
    print(f"[smoothquant] froze {n_lin} linears (alpha={ws[0].alpha}) in {time.time()-t0:.0f}s")
