"""Precision / quantization variants for LongCat-AudioDiT inference: fp32 / int8.

int8  — W8A8 int8 fake-quant on every DiT nn.Linear: weight per-output-channel + activation
        per-token (symmetric, dynamic). Quality study (matmul in fp32 — isolates int8 rounding).
        INT8 is the real-speedup target on A100 (it HAS int8 tensor cores); the *speed* path is
        torchao int8-dynamic + compile on a pre-quantized model (roadmap Phase 4), not this fake-quant.

(bf16 removed — it was quality-lossless but no A100 speedup, only VRAM; not a quant target.
 fp8 removed — A100 sm_80 has no FP8 tensor cores. INT8 + INT4/FlatQuant are the quant tracks.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── INT8 W8A8 fake-quant ────────────────────────────────────────────────────
def _fake_int8_weight(w):
    """Per-output-channel symmetric int8 round-trip (w: [out, in])."""
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.clamp(torch.round(w / scale), -127, 127) * scale


def _fake_int8_act(x):
    """Per-token (per last-dim row) symmetric dynamic int8 round-trip."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


class Int8EmuLinear(nn.Module):
    """nn.Linear with W8A8 int8 fake-quant: weight per-channel (pre-quantized once),
    activation per-token dynamic. Matmul runs in the weight dtype (isolates int8 rounding)."""
    def __init__(self, lin: nn.Linear, quant_act: bool = True):
        super().__init__()
        self.lin = lin
        self.quant_act = quant_act
        with torch.no_grad():
            self.lin.weight.copy_(_fake_int8_weight(self.lin.weight))

    def forward(self, x):
        if self.quant_act:
            x = _fake_int8_act(x)
        return self.lin(x)


def _swap_linears_int8(module, quant_act=True):
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8EmuLinear(child, quant_act=quant_act))
            n += 1
        else:
            n += _swap_linears_int8(child, quant_act=quant_act)
    return n


def apply_precision(model, precision: str) -> dict:
    """Mutate the model for the requested precision (DiT-only). Returns an info dict.

    fp32  — unchanged.
    int8  — every DiT nn.Linear fake-quantized to W8A8 (naive; if a layer hurts quality, exclude
            adaLN/time-embed/proj_out by applying Int8EmuLinear selectively).
    """
    info = {"precision": precision}
    if precision == "fp32":
        return info
    if precision == "int8":
        info["int8_linears_quantized"] = _swap_linears_int8(model.transformer, quant_act=True)
        return info
    raise ValueError(f"unknown precision {precision!r}")
