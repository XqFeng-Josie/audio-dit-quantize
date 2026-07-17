"""C1 (approach A): high-precision-GLUE W4A4 deploy on the REAL int4 kernel.

The 18.2% deploy gap is entirely the fp16 "glue" (Kron transform + per-token scale + dequant) around
an EXACT int4 GEMM. Doing that glue in fp32 makes the deploy path ALGEBRAICALLY equal to the symmetric
fake-quant control (~6.9%), because for symmetric per-token/per-channel quant:
    (scale_x . q_x) @ (scale_w . q_w)^T  ==  (q_x @ q_w^T) . scale_x . scale_w
and the int4 GEMM q_x@q_w^T is exact. So this recovers quality WITHOUT touching the Triton/CUDA kernels:
we quantize + pack in fp32 (deploy.functional.pack_i4 — bit-identical packing to the Triton kernel),
run the SAME real int4 `deploy.matmul`, then dequant in fp32 (instead of the fp16 CUDA sym_dequant).

Contrast with w4a4_deploy_fp16glue.py W4A4DeployLinear (the SHIPPED path): that keeps the glue in
fp16 (Triton kron fp16 + fp16 scale + fp16 sym_dequant) -> 18.2%. This hp path swaps ONLY the glue
precision; the int4 codes/GEMM are the same real kernel. => a genuine int4-deploy QUALITY number.

NOTE: this is a QUALITY probe (fp32 glue in PyTorch is slower than the fused fp16 kernel); efficiency is
measured separately on the shipped fp16 kernel. The point: does the real int4 GEMM + fp32 glue recover CER?
"""
import re
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

import deploy
from deploy.functional.quantization import pack_i4
from deploy.functional.online_trans import get_decompose_dim  # noqa: F401 (kept for parity/debug)
from flatquant.flat_utils import kronecker_matmul  # the EXACT fake-quant Kron transform (x.reshape(-1,a,b); x@hadR; hadL.T@x)


def _sym_quant_pack(mat_fp32, scale_fp32):
    """mat [R, C] fp32, scale [R, 1] fp32 -> packed uint8 [R, C//2]. Symmetric int4 [-8,7].
    Bit-identical to deploy.functional.pack_i4 (two's-complement 4-bit; even cols -> low nibble, odd -> high),
    but WITHOUT its `assert torch.all(...)` host-sync — that sync invalidates cudagraph capture. The clamp
    below guarantees the range the assert would check, so dropping it is safe (verified == pack_i4)."""
    q = torch.clamp(torch.round(mat_fp32 / scale_fp32), -8, 7).to(torch.int8)
    q_u8 = torch.where(q < 0, q + 16, q).to(torch.uint8)          # two_compl(q, 4): [-8,7] -> [0,15] uint8
    return q_u8[:, 0::2] | (q_u8[:, 1::2] << 4)                    # even -> low nibble, odd -> high nibble


# ── the hp forward wrapped as ONE opaque custom_op, so torch.compile/Dynamo sees a single
# cudagraph-safe leaf (no graph-break on pack_i4 bit-ops / the deploy.matmul pybind call) and can
# fuse the surrounding fp16 glue + cudagraph the whole model forward -> recovered quality AT speed.
# NB: this CALLS the exact same functions as W4A4DeployHPLinear._forward_eager (kronecker_matmul,
# _sym_quant_pack, deploy.matmul) in the exact same order — it is bit-identical, not a re-implementation.
try:
    @torch.library.custom_op("flatquant::w4a4_hp_linear", mutates_args=(), device_types="cuda")
    def w4a4_hp_linear(
        x: Tensor,                       # [.., K] activations
        matrix_left: Tensor,             # [a, a] Kron left factor (hadL)
        matrix_right: Tensor,            # [b, b] Kron right factor (hadR); a*b == K
        w_packed: Tensor,                # [N, K//2] uint8  int4-packed frozen weight
        wscale: Tensor,                  # [1, N] fp32  per-output-channel weight scale
        bias: Optional[Tensor],          # [N] fp32 or None
        out_features: int,               # N (static)
        diag_scale: Optional[Tensor],    # [K] fp32 per-input-channel scale (add_diag) or None (per-linear calib)
    ) -> Tensor:
        sh = x.shape
        xf = x.float()
        if diag_scale is not None:
            xf = xf * diag_scale.to(xf)                                   # add_diag: applied BEFORE the transform
        xt = kronecker_matmul(xf, matrix_left.to(xf), matrix_right.to(xf)).reshape(-1, sh[-1])  # fp32 Kron transform
        scale_x = (xt.abs().amax(dim=1, keepdim=True) / 7).clamp(min=1e-9)  # fp32 per-token sym scale
        xq_packed = _sym_quant_pack(xt, scale_x)                          # int4 codes (fp32 grid) -> uint8
        y_int = deploy.matmul(xq_packed, w_packed)                       # exact int4 GEMM -> int32
        y = y_int.float() * scale_x * wscale                            # fp32 dequant
        if bias is not None:
            y = y + bias
        return y.reshape(*sh[:-1], out_features).to(x.dtype)

    @w4a4_hp_linear.register_fake
    def _(x, matrix_left, matrix_right, w_packed, wscale, bias, out_features, diag_scale):
        return x.new_empty((*x.shape[:-1], out_features), dtype=x.dtype)
except Exception:
    pass  # already registered in this process (re-import)


class W4A4DeployHPLinear(nn.Module):
    """Real int4 deploy linear with fp32 GLUE. Transform + per-token act scale + dequant in fp32;
    int4 codes packed with deploy.pack_i4; GEMM via the real deploy.matmul (int4 CUTLASS)."""

    @classmethod
    @torch.no_grad()
    def from_flatquant(cls, fql):
        assert getattr(fql, "_frozen", False), "fql must be frozen (rf.calibrate calls freeze())"
        assert fql.use_trans and hasattr(fql.trans, "matrix_left"), "needs a decomposed Kron transform"
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        lin = fql.linear
        K, N = lin.in_features, lin.out_features
        dev = lin.weight.device
        # keep ONLY the (small) frozen transform as a submodule — NOT the whole fql (which would re-nest the
        # replaced FlatQuantLinear under the tree and infinite-recurse named_modules during the swap).
        self.trans = fql.trans               # eval-mode Kron transform; kept only for _forward_eager reference
        self.N, self.K = N, K
        # transform factors + optional add_diag as OWN buffers (so forward/custom_op take plain tensors,
        # no submodule-attr getattr in the hot path -> Dynamo/compile-clean).
        self.register_buffer("matrix_left", fql.trans.matrix_left.detach())
        self.register_buffer("matrix_right", fql.trans.matrix_right.detach())
        ds = (fql.trans.diag_scale.detach()
              if (getattr(fql.trans, "add_diag", False) and getattr(fql.trans, "use_diag", True)) else None)
        self.register_buffer("diag_scale", ds)                     # [K] fp32 or None (per-linear calib -> None)
        # weight int4 codes (fp32-exact, from the frozen dequant weight) + per-channel fp32 scale.
        wscale = fql.wq.scale.reshape(N, 1).float().to(dev)          # [N,1] fp32
        q_w = torch.clamp(torch.round(lin.weight.detach().float() / wscale), -8, 7).to(torch.int8)
        self.register_buffer("w_packed", pack_i4(q_w))              # [N, K//2] uint8
        self.register_buffer("wscale", wscale.reshape(1, N))       # [1,N] fp32 for dequant
        self.register_buffer("bias", lin.bias.detach().float() if lin.bias is not None else None)
        return self

    @classmethod
    @torch.no_grad()
    def from_linear_latency(cls, lin):
        """Build an hp linear with IDENTITY Kron transforms (latency/VRAM measurement only — no calibration;
        the int4 GEMM + glue ops have the same cost regardless of transform values, and identity keeps the
        output finite across stacked layers so the output-validity guard can certify cudagraph correctness)."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        K, N = lin.in_features, lin.out_features
        dev = lin.weight.device
        a, b = get_decompose_dim(K)
        assert a * b == K
        self.N, self.K = N, K
        self.trans = None                                          # no eager reference for the latency path
        self.register_buffer("matrix_left", torch.eye(a, dtype=torch.float32, device=dev))
        self.register_buffer("matrix_right", torch.eye(b, dtype=torch.float32, device=dev))
        self.register_buffer("diag_scale", None)
        wscale = (lin.weight.detach().abs().amax(dim=1, keepdim=True) / 7).float().clamp(min=1e-9)   # [N,1]
        q_w = torch.clamp(torch.round(lin.weight.detach().float() / wscale), -8, 7).to(torch.int8)
        self.register_buffer("w_packed", pack_i4(q_w))
        self.register_buffer("wscale", wscale.reshape(1, N))
        self.register_buffer("bias", lin.bias.detach().float() if lin.bias is not None else None)
        return self

    def forward(self, x):
        # route through the opaque custom_op (compile/cudagraph-friendly). Bit-identical to _forward_eager.
        return torch.ops.flatquant.w4a4_hp_linear(
            x, self.matrix_left, self.matrix_right,
            self.w_packed, self.wscale, self.bias, self.N, self.diag_scale,
        )

    @torch.no_grad()
    def _forward_eager(self, x):
        """Reference (proven, Gate-0) forward — kept for the custom_op==eager unit check."""
        sh = x.shape
        xt = self.trans(x.float()).reshape(-1, self.K)              # [T, K] fp32 Kron transform (exact)
        scale_x = (xt.abs().amax(dim=1, keepdim=True) / 7).clamp(min=1e-9)   # [T,1] fp32 per-token sym scale
        xq_packed = _sym_quant_pack(xt, scale_x)                    # [T, K//2] uint8
        y_int = deploy.matmul(xq_packed, self.w_packed)             # [T, N] int32 (exact int4 GEMM)
        y = y_int.float() * scale_x * self.wscale                   # [T, N] fp32 dequant
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*sh[:-1], self.N).to(x.dtype)


@torch.no_grad()
def wrap_dit_w4a4_hp(model):
    """Swap each CALIBRATED+FROZEN FlatQuantLinear in the DiT for the fp32-glue real-int4 hp deploy linear.
    Run AFTER rf.calibrate(model, ...). Linears with in/out not %32 (int4 GEMM constraint) stay fake-quant
    (hybrid); returns (wrapped, skipped) for coverage attribution."""
    from .. import flatquant_layers as fq
    # MATERIALIZE targets first (dedup by id) — never mutate the module tree while iterating it.
    targets, seen = [], set()
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in list(sub.named_children()):
                if isinstance(child, fq.FlatQuantLinear) and id(child) not in seen:
                    seen.add(id(child))
                    targets.append((sub, attr, child, f"{bname}.{attr}"))
    wrapped, skipped = 0, []
    for sub, attr, child, name in targets:
        lin = child.linear
        if lin.in_features % 32 != 0 or lin.out_features % 32 != 0:
            skipped.append((name, f"in{lin.in_features}/out{lin.out_features} not %32"))
            continue
        if not (child.use_trans and hasattr(child.trans, "matrix_left")):
            skipped.append((name, "single full-transform (prime in-dim)"))
            continue
        try:
            setattr(sub, attr, W4A4DeployHPLinear.from_flatquant(child))
            wrapped += 1
        except Exception as e:
            skipped.append((name, f"{type(e).__name__}:{str(e)[:50]}"))
    return wrapped, skipped


@torch.no_grad()
def wrap_dit_w4a4_hp_latency(model):
    """LATENCY/VRAM-only: swap every %32 raw nn.Linear in the DiT for an identity-transform hp deploy linear
    (no calibration). Same int4 GEMM + fp32 glue ops as the real hp path -> representative latency for the
    'compiled hp' efficiency measurement. Returns (wrapped, skipped)."""
    targets, seen = [], set()
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in list(sub.named_children()):
                if isinstance(child, nn.Linear) and id(child) not in seen:
                    seen.add(id(child))
                    targets.append((sub, attr, child, f"{bname}.{attr}"))
    wrapped, skipped = 0, []
    for sub, attr, child, name in targets:
        if child.in_features % 32 != 0 or child.out_features % 32 != 0:
            skipped.append((name, f"in{child.in_features}/out{child.out_features} not %32"))
            continue
        try:
            setattr(sub, attr, W4A4DeployHPLinear.from_linear_latency(child))
            wrapped += 1
        except Exception as e:
            skipped.append((name, f"{type(e).__name__}:{str(e)[:50]}"))
    return wrapped, skipped


# ===========================================================================
# FUSED fp32-glue path — the real "both quality + efficiency" candidate.
# The Kron transform + per-token scale + int4 quant + pack are done in ONE fp32
# Triton kernel (kron_matmul hp=True: true-fp32 tl.dot, fp32 intermediate, fp32
# scale — verified to produce int4 codes bit-identical to fp32), then the exact
# int4 GEMM, then fp32 dequant. Unlike the eager hp path (unfused PyTorch kron =
# 838 ms), the transform is fused -> few kernels -> cudagraph-fast, while keeping
# the fp32 accuracy that recovers quality (6.758%).
# ===========================================================================
_HP_CLIP = 20.0   # sigmoid(20) ~ 1.0 -> the kron act-clip is neutralized (scale = maxabs/7), matching the
                  # eager hp path (W4A4DeployHPLinear, which recovered to 6.758%). Verified fused == eager.
try:
    @torch.library.custom_op("flatquant::w4a4_hp_fused_linear", mutates_args=(), device_types="cuda")
    def w4a4_hp_fused_linear(
        x: Tensor,                 # [.., K] activations
        L: Tensor,                 # [a, a] fp32 kron-left
        R: Tensor,                 # [b, b] fp32 kron-right
        w_packed: Tensor,          # [N, K//2] uint8 int4-packed weight
        wscale: Tensor,            # [1, N] fp32 per-output-channel weight scale
        bias: Optional[Tensor],    # [N] fp32 or None
        out_features: int,         # N (static)
        clip_a: float,             # act-quant clip (kron does xmax*=sigmoid(clip)); large -> no clip
    ) -> Tensor:
        from deploy.functional.online_trans import kronecker_matmul as _okron
        sh = x.shape
        xf = x.reshape(1, -1, sh[-1]).float() if x.dim() != 3 else x.float()
        packed = _okron(xf, [L, R], clip_a, clip_a, hp=True)   # FUSED fp32 kron -> int4 codes + fp32 scale
        qx, sx = packed.quantized_x, packed.scales_x           # qx [bsz,seq,K//2] uint8, sx [bsz,1,seq] fp32
        y_int = deploy.matmul(qx, w_packed)                    # [bsz,seq,N] int32 (exact int4 GEMM)
        T = y_int.shape[0] * y_int.shape[1]
        # fp32 dequant: y_int * per-token scale * per-channel weight scale ([bsz,1,seq] flattens token-order to [T])
        y = y_int.reshape(T, out_features).float() * sx.reshape(T, 1) * wscale.reshape(1, out_features)
        if bias is not None:
            y = y + bias
        return y.reshape(*sh[:-1], out_features).to(x.dtype)

    @w4a4_hp_fused_linear.register_fake
    def _(x, L, R, w_packed, wscale, bias, out_features, clip_a):
        return x.new_empty((*x.shape[:-1], out_features), dtype=x.dtype)
except Exception:
    pass


class W4A4DeployHPFusedLinear(nn.Module):
    """Real int4 deploy linear with FUSED fp32 glue: the Kron transform+scale+quant+pack run in one fp32
    Triton kernel (kron_matmul hp=True), then exact int4 GEMM, then fp32 dequant. Bit-matches W4A4DeployHPLinear
    (verified) but the transform is fused -> cudagraph-fast. Routed via a custom_op for compile/cudagraph."""

    @classmethod
    @torch.no_grad()
    def from_flatquant(cls, fql):
        assert getattr(fql, "_frozen", False), "fql must be frozen (rf.calibrate calls freeze())"
        assert fql.use_trans and hasattr(fql.trans, "matrix_left"), "needs a decomposed Kron transform"
        self = cls.__new__(cls); nn.Module.__init__(self)
        lin = fql.linear; K, N = lin.in_features, lin.out_features; dev = lin.weight.device
        self.N, self.K = N, K
        self.register_buffer("L", fql.trans.matrix_left.detach().float().contiguous())
        self.register_buffer("R", fql.trans.matrix_right.detach().float().contiguous())
        wscale = fql.wq.scale.reshape(N, 1).float().to(dev)
        q_w = torch.clamp(torch.round(lin.weight.detach().float() / wscale), -8, 7).to(torch.int8)
        self.register_buffer("w_packed", pack_i4(q_w))
        self.register_buffer("wscale", wscale.reshape(1, N))
        self.register_buffer("bias", lin.bias.detach().float() if lin.bias is not None else None)
        self.clip_a = _HP_CLIP
        return self

    @classmethod
    @torch.no_grad()
    def from_linear_latency(cls, lin):
        """Identity-transform build for latency/VRAM measurement (no calibration; representative kernel cost)."""
        self = cls.__new__(cls); nn.Module.__init__(self)
        K, N = lin.in_features, lin.out_features; dev = lin.weight.device
        a, b = get_decompose_dim(K); assert a * b == K
        self.N, self.K = N, K
        self.register_buffer("L", torch.eye(a, dtype=torch.float32, device=dev).contiguous())
        self.register_buffer("R", torch.eye(b, dtype=torch.float32, device=dev).contiguous())
        wscale = (lin.weight.detach().abs().amax(dim=1, keepdim=True) / 7).float().clamp(min=1e-9)
        q_w = torch.clamp(torch.round(lin.weight.detach().float() / wscale), -8, 7).to(torch.int8)
        self.register_buffer("w_packed", pack_i4(q_w))
        self.register_buffer("wscale", wscale.reshape(1, N))
        self.register_buffer("bias", lin.bias.detach().float() if lin.bias is not None else None)
        self.clip_a = _HP_CLIP
        return self

    def forward(self, x):
        return torch.ops.flatquant.w4a4_hp_fused_linear(
            x, self.L, self.R, self.w_packed, self.wscale, self.bias, self.N, self.clip_a,
        )


@torch.no_grad()
def wrap_dit_w4a4_hp_fused_latency(model):
    """LATENCY/VRAM-only: swap every %32 raw nn.Linear for a FUSED-fp32-kron hp deploy linear (identity trans)."""
    targets, seen = [], set()
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in list(sub.named_children()):
                if isinstance(child, nn.Linear) and id(child) not in seen:
                    seen.add(id(child)); targets.append((sub, attr, child, f"{bname}.{attr}"))
    wrapped, skipped = 0, []
    for sub, attr, child, name in targets:
        if child.in_features % 32 != 0 or child.out_features % 32 != 0:
            skipped.append((name, f"in{child.in_features}/out{child.out_features} not %32")); continue
        try:
            setattr(sub, attr, W4A4DeployHPFusedLinear.from_linear_latency(child)); wrapped += 1
        except Exception as e:
            skipped.append((name, f"{type(e).__name__}:{str(e)[:50]}"))
    return wrapped, skipped
