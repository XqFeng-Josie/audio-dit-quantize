"""Apply FlatQuant to the LongCat-AudioDiT DiT (W4A4), per FLATQUANT_METHOD.md.

Reuses the reference repo's tested transform + quantizers (flatquant_ref/):
  - SVDDecomposeTransMatrix : learnable invertible Kronecker transform P = P_L ⊗ P_R
  - WeightQuantizer / ActivationQuantizer : W4 per-channel sym / A4 per-token

Each target nn.Linear is wrapped by FlatQuantLinear, which is mathematically equivalent to
the dense linear (verified invariant: P(x) @ (P_invt(W))ᵀ == x @ Wᵀ) but quantizes the
*flattened* activation/weight. The best-config pipeline trains these wrappers with
sequential per-block reconstruction, then freezes them.

A100 note: W4A4 has no fast kernel here -> this is a QUALITY study (fake-quant), slower.
"""
import sys, os
from pathlib import Path
from .paths import FLATQUANT_REF_DIR

_fq_pkg = Path(FLATQUANT_REF_DIR) / "flatquant"
if not (_fq_pkg / "__init__.py").exists():
    raise ModuleNotFoundError(
        "FlatQuant reference package not found. Expected "
        f"{_fq_pkg}. Sync vendor/flatquant_ref to this machine, "
        "or set FLATQUANT_REF_DIR=/path/to/FlatQuant before running."
    )
sys.path.insert(0, str(FLATQUANT_REF_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from flatquant.trans_utils import SVDDecomposeTransMatrix, SVDSingleTransMatrix
from flatquant.quant_utils import WeightQuantizer, ActivationQuantizer

# Global activation-quant gate for the ODE step-axis diagnostic (diag_step_axis.py). Default True =
# normal W4A4 (no behavior change). Set False to skip per-token activation quantization on the next
# forward(s) -> those ODE steps run fp activation (weight stays W4). Weights are shared across all ODE
# steps, so only the activation quant is step-varying; this gate is what makes step-phase precision switchable.
_ACT_QUANT = True


def _factorize(n):
    """Return (a,b), a*b=n, a<=b, a closest to sqrt(n) (for Kronecker factor sizes)."""
    a = int(n ** 0.5)
    while a > 1 and n % a != 0:
        a -= 1
    return (a, n // a)


class FlatQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, w_bits=4, a_bits=4, use_trans=True, lwc=True,
                 a_sym=False, lac=False, add_diag=False):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.use_trans = use_trans
        self.lwc = lwc
        self.lac = lac

        if use_trans:
            a, b = _factorize(self.in_features)
            # full transform if not factorizable (prime in-dim); else cheap Kronecker.
            # add_diag => learnable per-input-channel scaling diag(c) fused into the transform
            # (the FlatQuant "+scaling" ablation step; folds into weight at freeze for the inv path).
            self.trans = SVDSingleTransMatrix(self.in_features) if a == 1 \
                else SVDDecomposeTransMatrix(a, b, add_diag=add_diag)

        self.wq = WeightQuantizer(); self.wq.configure(w_bits, perchannel=True, sym=True, mse=False)
        # a_sym=True => SYMMETRIC int4 act (maxabs/7, [-8,7]) — matches the real deploy kron kernel exactly,
        # so transforms calibrated under it transfer faithfully to deploy. Default False (asym, [0,15]) is the
        # original sim that produced the recorded numbers but is NOT what the int4 kernels actually do.
        # lac => learnable activation clipping (per-tensor clip_factor on amax/amin, trained jointly).
        self.aq = ActivationQuantizer(bits=a_bits, sym=a_sym, lac=lac, groupsize=-1)

        if lwc:
            o = linear.weight.shape[0]
            self.clip_w_max = nn.Parameter(torch.ones((o, 1)) * 4.0)
            self.clip_w_min = nn.Parameter(torch.ones((o, 1)) * 4.0)
            self.sig = nn.Sigmoid()

        self._frozen = False
        self.enable_quant = True   # set False to bypass quant (true-fp passthrough) for per-block fp targets
        self._wq_cache = None      # per-step cached (differentiable) qweight; only the act path varies per item

    def _wclip(self, w):
        wmin = w.min(1, keepdim=True)[0] * self.sig(self.clip_w_min)
        wmax = w.max(1, keepdim=True)[0] * self.sig(self.clip_w_max)
        return torch.clamp(w, min=wmin, max=wmax)

    def _qweight(self):
        w = self.linear.weight
        if self.use_trans:
            w = self.trans(w, inv_t=True)          # fuse P^{-1} into weight
        if self.lwc:
            w = self._wclip(w)
        self.wq.find_params(w)
        return self.wq(w)

    def forward(self, x):
        if not self.enable_quant and not self._frozen:
            # true-fp passthrough (original weight intact pre-freeze): used to capture the
            # full-precision block output as the per-block reconstruction target.
            return F.linear(x, self.linear.weight, self.linear.bias)
        if self._frozen:
            xt = self._apply_xtrans(x)
            # ACT-QUANT gate (default on): global _ACT_QUANT (per-step, set by a forward hook) AND a
            # per-linear self._act_on (per-layer-group). When either is off, skip the per-token activation
            # quantizer for THIS forward -> fp activation, weight stays W4. Weights are shared across ODE
            # steps so only the activation quant is step-varying; this gate makes step×layer precision
            # switchable for the mixed-precision diagnostics. No behavior change when both are True.
            act_on = _ACT_QUANT and getattr(self, "_act_on", True)
            xq = (self.aq(xt) if act_on else xt).to(x.dtype)
            return F.linear(xq, self.linear.weight, self.linear.bias)
        xt = self._apply_xtrans(x)
        xq = self.aq(xt)
        # per-block training reuses one differentiable wq across the mini-batch (weight path is item-independent);
        # the expensive Kron-on-weight + clip + quant then runs once per optimizer step, not once per item.
        wq = self._wq_cache if self._wq_cache is not None else self._qweight()
        return F.linear(xq, wq, self.linear.bias)

    def set_wq_cache(self):
        self._wq_cache = self._qweight()

    def clear_wq_cache(self):
        self._wq_cache = None

    def _apply_xtrans(self, x):
        return self.trans(x) if self.use_trans else x

    def trainable_parameters(self):
        ps = []
        if self.use_trans:
            ps += list(self.trans.parameters())   # incl. diag_scale when add_diag=True
        if self.lwc:
            ps += [self.clip_w_max, self.clip_w_min]
        if self.lac:
            ps += [self.aq.clip_factor_a_max, self.aq.clip_factor_a_min]
        return ps

    @torch.no_grad()
    def freeze(self):
        if self.use_trans and hasattr(self.trans, "to_eval_mode"):
            self.trans.to_eval_mode()
        wq = self._qweight().detach()
        # Replace the original fp32 weight IN-PLACE (it's unused in frozen forward) instead of
        # keeping a 2nd full-size copy — avoids doubling weight memory (OOM on 3.5B). Same math.
        self.linear.weight.data = wq
        if self.lwc:                       # clips are baked into wq now
            del self.clip_w_max, self.clip_w_min
        self._frozen = True


# ── target-linear selection in the DiT block ────────────────────────────────
def _target_linears(transformer):
    """Yield (immediate_parent, attr, linear) for every nn.Linear under the blocks'
    attn / cross_attn / ffn sub-modules (the GEMM-heavy projections + FFN). Skips adaLN
    modulation, time_mlp, embedders, proj_out (sensitive / tiny FLOPs). Dedup by id."""
    import re
    seen = set()
    for bname, block in transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in sub.named_children():
                if isinstance(child, nn.Linear) and id(child) not in seen:
                    seen.add(id(child))
                    yield sub, attr, child


def wrap_dit(model, w_bits=4, a_bits=4, use_trans=True, lwc=True, a_sym=False, lac=False, add_diag=False):
    wrapped = []
    for parent, attr, lin in list(_target_linears(model.transformer)):
        fq = FlatQuantLinear(lin, w_bits, a_bits, use_trans=use_trans, lwc=lwc,
                             a_sym=a_sym, lac=lac, add_diag=add_diag).to(lin.weight.device)
        setattr(parent, attr, fq)
        wrapped.append(fq)
    return wrapped


# Backward-compat for pickled models: best-config models saved when this module was the top-level
# `flatquant_dit` module (seed_repro) pickle their layers as `flatquant_dit.FlatQuantLinear`. Register
# the old name as an alias for this module so `torch.load(<whole model>, weights_only=False)` resolves it.
import sys as _sys
_sys.modules.setdefault("flatquant_dit", _sys.modules[__name__])
