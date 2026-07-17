"""Feasibility probe: wire FlatQuant's REAL W4A4 deploy kernels into LongCat-AudioDiT end-to-end.

Goal = confirm we can replace DiT linears with the deployed W4A4 path (OnlineTrans Kron-transform+quant
-> Linear4bit int4 GEMM + dequant) and run a real generation, so we can later MEASURE end-to-end
latency + peak VRAM (paper-grade efficiency). Latency/VRAM only -> values don't matter (random transforms).

Two stages: (1) unit-test the wrapper on a synthetic [1,T,K] forward; (2) wrap the 1B DiT and run one
infer_one. Robust: skips un-wrappable linears (in_features not %32 etc.), reports coverage + where it breaks.

Run (after `source env.sh`; standalone self-test of the wrapper + one 1B generation):
  python -m audio_dit_quantize.efficiency.w4a4_deploy_fp16glue
"""
import re
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

import deploy
from deploy.nn import Linear4bit
from deploy.functional.online_trans import kronecker_matmul, get_decompose_dim


# ---------------------------------------------------------------------------
# torch.library.custom_op wrapping the WHOLE W4A4 forward body as ONE opaque op
# ---------------------------------------------------------------------------
# Why wrap here (not kron / GEMM separately): kronecker_matmul RETURNS and
# Linear4bit CONSUMES a `PackedQuantizedTensor` (a plain Python class, not a
# Tensor subclass) — it can't cross a custom_op boundary, so the only place
# where ONLY plain Tensors cross is the outer forward. Wrapping it makes
# torch.compile/Dynamo see ONE trusted opaque leaf (no graph-break on the
# pybind CUTLASS calls), and inductor can then fuse the surrounding fp16 glue
# ([fused pre] -> opaque w4a4 op -> [fused post]) and capture the whole thing
# in a cudagraph. (Inductor canNOT fuse epilogues INTO the int4 kernels — a
# FallbackKernel doesn't fuse — so the int4 linear stays ~4 kernels; the win
# over the manual cudagraph is glue-fusion only.) Functional (mutates_args=()):
# every inner kernel allocates its own output, nothing is mutated in place, no
# host sync -> cudagraph-safe by default, no cudagraph_unsafe tag needed.
try:
    @torch.library.custom_op("flatquant::w4a4_linear", mutates_args=(), device_types="cuda")
    def w4a4_linear(
        x: Tensor,                 # [.., K] activations
        L: Tensor,                 # [a, a] fp16  kron-left
        R: Tensor,                 # [b, b] fp16  kron-right
        weight: Tensor,            # [N, K//2] uint8  packed int4 weight (l4.weight)
        weight_scales: Tensor,     # [N, 1] fp16      (l4.weight_scales)
        bias: Optional[Tensor],    # [N] fp16 or None (l4.bias)
        out_features: int,         # N (static)
        clip_a_max: float,         # act-quant clip; kron kernel scales xmax by sigmoid(clip). large => no clip
        clip_a_min: float,
    ) -> Tensor:
        in_dtype = x.dtype
        sh = x.shape
        xf = x.reshape(1, -1, sh[-1]).half() if x.dim() != 3 else x.half()
        packed = kronecker_matmul(xf, [L, R], clip_a_max, clip_a_min)  # Triton Kron + sym-int4 act-quant -> Packed
        qx, sx = packed.quantized_x, packed.scales_x
        y = deploy.matmul(qx, weight)                     # CUTLASS int4 GEMM -> int32
        y = deploy.sym_dequant(y, sx, weight_scales)      # int32 -> fp16
        if bias is not None:
            y = y + bias
        return y.reshape(*sh[:-1], out_features).to(in_dtype)

    @w4a4_linear.register_fake
    def _w4a4_linear_fake(x, L, R, weight, weight_scales, bias, out_features, clip_a_max, clip_a_min):
        # meta path: no compute, output dtype = activation dtype (NOT int4); inherit device/fake-mode from x.
        return x.new_empty((*x.shape[:-1], out_features), dtype=x.dtype)
except Exception:
    pass  # already registered in this process (re-import) — torch.ops.flatquant.w4a4_linear stays valid


class W4A4DeployLinear(nn.Module):
    """Deployed FlatQuant W4A4 linear: online Kron transform+quant (scalar clips to dodge the Triton-3.7
    buffer-tensor bug) -> int4 GEMM -> sym_dequant. Random transform (latency/VRAM only).

    forward routes through the `flatquant::w4a4_linear` custom_op so torch.compile sees one opaque,
    graph-break-free, cudagraph-safe node (lets it join the inductor fused+cudagraph path)."""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        K, N = lin.in_features, lin.out_features
        dev = lin.weight.device
        a, b = get_decompose_dim(K)
        # Identity transforms (not random): same kernel cost (identity Kron matmul is still a full
        # matmul, so latency/VRAM are representative) but the output stays FINITE across all 240/320
        # stacked layers -> the output-validity guard can certify cudagraph correctness end-to-end.
        # (Random transforms explode to NaN over 240 layers even in eager, masking real bugs.)
        self.register_buffer("L", torch.eye(a, dtype=torch.float16, device=dev))
        self.register_buffer("R", torch.eye(b, dtype=torch.float16, device=dev))
        wscale = (lin.weight.detach().abs().amax(dim=1, keepdim=True) / 7).half()
        linh = nn.Linear(K, N, bias=lin.bias is not None).half().to(dev)
        linh.weight.data = lin.weight.detach().half()
        if lin.bias is not None:
            linh.bias.data = lin.bias.detach().half()
        self.l4 = Linear4bit.from_float(linh, weight_scales=wscale).to(dev)
        self.out_features = N
        # act-quant clip passed to the kron kernel (kernel does xmax*=sigmoid(clip)). 1.0 -> sigmoid≈0.731
        # (a 27% range shrink — the deploy default). For real-quality runs from_flatquant raises this so
        # sigmoid≈1 (no clip), matching the lac=False calibration. Irrelevant for latency-only (garbage values).
        self.clip_a_max = 1.0
        self.clip_a_min = 1.0

    def forward(self, x):
        return torch.ops.flatquant.w4a4_linear(
            x, self.L, self.R, self.l4.weight, self.l4.weight_scales, self.l4.bias, self.out_features,
            self.clip_a_max, self.clip_a_min,
        )

    @classmethod
    def from_flatquant(cls, fql, clip_a=8.0):
        """Build a deploy W4A4 linear from a CALIBRATED+FROZEN FlatQuantLinear (flatquant_layers.py).
        Weight + Kronecker transform transfer EXACTLY (verified, bit-exact weight); the ONLY difference vs
        the fake-quant simulation is the activation quantizer (deploy = symmetric int4; fake-quant = asym).
        clip_a large (default 8 -> sigmoid≈1) neutralizes the kron act-clip to match the lac=False calib."""
        assert getattr(fql, "_frozen", False), "fql must be frozen (rf.calibrate calls freeze())"
        lin = fql.linear
        K, N = lin.in_features, lin.out_features
        dev = lin.weight.device
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        # 1) learned Kronecker factors (eval-mode) -> deploy kron L,R (as-is; kernel transposes L internally)
        if not hasattr(fql.trans, "matrix_left"):
            raise NotImplementedError("single full-transform (prime in-dim) not supported by 2-factor kron")
        L = fql.trans.matrix_left.detach().half().to(dev)
        R = fql.trans.matrix_right.detach().half().to(dev)
        a, b = L.shape[0], R.shape[0]
        ad, bd = get_decompose_dim(K)
        assert (a, b) == (ad, bd), f"factorization mismatch {(a,b)} vs deploy {(ad,bd)}"
        assert a * b == K
        self.register_buffer("L", L)
        self.register_buffer("R", R)
        # 2) weight: reuse the FROZEN dequant weight + the TRAINING per-row scale (recovers int bit-exact)
        wscale = fql.wq.scale.reshape(N, 1).half().to(dev)   # = maxabs(clipped, transformed W) / 7
        linh = nn.Linear(K, N, bias=lin.bias is not None).half().to(dev)
        linh.weight.data = lin.weight.detach().half()        # frozen dequant W = wscale * q_int
        if lin.bias is not None:
            linh.bias.data = lin.bias.detach().half()
        self.l4 = Linear4bit.from_float(linh, weight_scales=wscale).to(dev)
        self.out_features = N
        self.clip_a_max = float(clip_a)
        self.clip_a_min = float(clip_a)
        return self


def wrap_dit_w4a4_calibrated(model, clip_a=8.0):
    """Swap each CALIBRATED FlatQuantLinear in the DiT for a real-int4 deploy W4A4 linear.
    Run AFTER rf.calibrate(model, ...) (which wraps + freezes FlatQuantLinear). Returns (wrapped, skipped).
    Linears whose in/out aren't %32 (deploy kernel constraint) are left as the fake-quant FlatQuantLinear
    -> the run is then a HYBRID; report coverage so the deploy delta is attributable."""
    from .. import flatquant_layers as fq
    wrapped, skipped = 0, []
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in list(sub.named_children()):
                if isinstance(child, fq.FlatQuantLinear):
                    lin = child.linear
                    if lin.in_features % 32 != 0 or lin.out_features % 32 != 0:
                        skipped.append((f"{bname}.{attr}", f"in{lin.in_features}/out{lin.out_features} not %32"))
                        continue
                    try:
                        setattr(sub, attr, W4A4DeployLinear.from_flatquant(child, clip_a=clip_a))
                        wrapped += 1
                    except Exception as e:
                        skipped.append((f"{bname}.{attr}", f"{type(e).__name__}:{str(e)[:50]}"))
    return wrapped, skipped


def wrappable(lin):
    return isinstance(lin, nn.Linear) and lin.in_features % 32 == 0 and lin.out_features % 32 == 0


def wrap_dit_w4a4(model):
    wrapped, skipped = 0, []
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in list(sub.named_children()):
                if isinstance(child, nn.Linear):
                    if wrappable(child):
                        try:
                            setattr(sub, attr, W4A4DeployLinear(child))
                            wrapped += 1
                        except Exception as e:
                            skipped.append((f"{bname}.{attr}", f"{type(e).__name__}:{str(e)[:40]}"))
                    else:
                        skipped.append((f"{bname}.{attr}", f"in{child.in_features}/out{child.out_features} not %32"))
    return wrapped, skipped


def main():
    dev = torch.device("cuda:0")

    # ---- stage 1: unit-test the wrapper on a synthetic forward ----
    print("=== stage 1: wrapper unit test ===")
    for K, N in [(1536, 1536), (1536, 6144), (6144, 1536)]:
        lin = nn.Linear(K, N, bias=False).to(dev)
        w = W4A4DeployLinear(lin).to(dev)
        x = torch.randn(2, 300, K, device=dev)  # [B,T,K] like the DiT
        y = w(x)
        ok = tuple(y.shape) == (2, 300, N) and torch.isfinite(y).all().item()
        print(f"  K={K}->N={N}: out {tuple(y.shape)} finite={torch.isfinite(y).all().item()} {'OK' if ok else 'BAD'}")

    # ---- stage 2: wrap the real 1B DiT + run one infer_one ----
    print("\n=== stage 2: wrap 1B DiT + infer_one ===")
    import audiodit  # noqa
    from quant_sensitivity import load_model
    import run_svdquant as rs
    from batch_inference import infer_one
    from ..paths import DATA_DIR, SETS
    import os

    model, tok = load_model("meituan-longcat/LongCat-AudioDiT-1B", dev)
    n_wrapped, skipped = wrap_dit_w4a4(model)
    print(f"  wrapped {n_wrapped} linears; skipped {len(skipped)}")
    for s in skipped[:8]:
        print(f"    skip {s[0]} ({s[1]})")

    items = rs.load_items(os.path.join(str(DATA_DIR), SETS["hard"]))[:1]
    uid, pt, pwa, gt = items[0]
    try:
        wav = infer_one(gt, pt, pwa, model, tok, dev, 16, 4.0, "apg")
        print(f"  infer_one OK: wav shape {getattr(wav,'shape',len(wav))} -> END-TO-END W4A4 RUNS")
    except Exception as exc:
        import traceback
        print(f"  infer_one FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
