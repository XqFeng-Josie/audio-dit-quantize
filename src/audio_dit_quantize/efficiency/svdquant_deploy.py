"""End-to-end deploy of Nunchaku's REAL SVDQuant W4A4 kernels into LongCat-AudioDiT.

Mirrors w4a4_deploy_fp16glue.py (FlatQuant W4A4DeployLinear) but wraps each DiT linear with
Nunchaku's SVDQW4A4Linear (fused smooth+quant+lora-down -> int4 GEMM+lora-up), so we can MEASURE
end-to-end latency + peak VRAM with the SAME profile_efficiency.py harness as fp32/INT8/FlatQuant-W4A4.

MUST run in the isolated `nunchaku` conda env (torch 2.11 + nunchaku 1.2.1 prebuilt cu13.0 wheel).
Launch (clean import env; put the repo src + vendor + LongCat on PYTHONPATH):
  cd ~/workspace/audio-dit-quantize && PYTHONNOUSERSITE=1 \
    PYTHONPATH=src:vendor/flatquant_ref:$HOME/workspace/LongCat-AudioDiT \
    ~/miniconda3/envs/nunchaku/bin/python -m audio_dit_quantize.efficiency.profile_efficiency \
    --model_dir meituan-longcat/LongCat-AudioDiT-1B --svdquant-deploy --runs 5 --warmup 2

Random/real weights both fine for latency (depends on shapes). SVDQW4A4Linear is bf16-native and wants
a 3D [B,T,C] input, so the wrapper casts to bf16 + reshapes to 3D and casts the output back (same shape
contract as W4A4DeployLinear). Target = the same 240/320 self_attn/cross_attn/ffn linears we quantize
elsewhere (in/out divisible by the group size 64 — all DiT dims 1536/6144/2560/9216 qualify).
"""
import re
import torch
import torch.nn as nn

from nunchaku.models.linear import SVDQW4A4Linear

DT = torch.bfloat16
RANK = 32


class SVDQuantDeployLinear(nn.Module):
    """Deployed SVDQuant W4A4 linear: build SVDQW4A4Linear from the fp32 nn.Linear (one-time SVD +
    INT4 pack at setup), then forward = bf16 3D in -> fused SVDQuant W4A4 -> cast back to in dtype."""
    def __init__(self, lin: nn.Linear, rank: int = RANK):
        super().__init__()
        dev = lin.weight.device
        src = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None).to(dev, DT)
        src.weight.data = lin.weight.detach().to(DT)
        if lin.bias is not None:
            src.bias.data = lin.bias.detach().to(DT)
        self.q = SVDQW4A4Linear.from_linear(src, rank=rank).to(dev)
        self.out_features = lin.out_features

    def forward(self, x):
        in_dtype = x.dtype
        sh = x.shape
        sm = getattr(self, "smooth", None)          # Tier-2 (from_calibrated) carries our calibrated smoothing s
        xs = x if sm is None else x / sm.to(x.dtype)  # X/s -> the smoothed activations SVDQuant int4-quantizes
        xf = xs.reshape(1, -1, sh[-1]).to(DT) if xs.dim() != 3 else xs.to(DT)
        y = self.q(xf)
        return y.reshape(*sh[:-1], self.out_features).to(in_dtype)

    @classmethod
    def from_calibrated(cls, sql, *, rank: int = RANK):
        """TIER-2: bridge a CALIBRATED svdquant_dit.SVDQuantLinear into Nunchaku's REAL int4 kernel,
        carrying OUR calibrated smoothing s + low-rank + int4 residual. The deploy-vs-fakequant delta is
        then PURELY kernel numerics (+ Nunchaku's activation-quant format) — the from_flatquant philosophy.

        Robust to Nunchaku's internal field layout (uses ONLY the public from_linear): SVDQuant's identity
        is  X Wᵀ = (X/s)(W·diag(s))ᵀ.  We hand Nunchaku the PRE-SMOOTHED weight Wp = W·diag(s) (it does its
        own SVD + int4 pack on Wp — deterministic, ~= our L,R,res) and DIVIDE activations by s in forward().
        So the int4 path quantizes the SAME smoothed activations (outliers removed) + smoothed-weight residual
        our fake-quant did, with no surgery on Nunchaku's smooth/lora/qweight fields.

        Wp is reconstructed as L@R + res_wq (the calibrated weight is freed). res_wq already sits on the int4
        grid, so Nunchaku's re-quant of it is near-idempotent -> negligible double-quant. Residual mismatch:
        our act-quant is per-group-64 dynamic ASYMMETRIC; Nunchaku's is typically symmetric -> that is a real,
        measurable sub-gap (report it, like FlatQuant's sym-vs-asym +0.076), NOT a wiring bug.
        """
        assert getattr(sql, "_calibrated", False), "sql must be a CALIBRATED svdquant_dit.SVDQuantLinear"
        dev = sql.smooth.device
        s = sql.smooth.detach().float()                                                   # [in]
        Wp = (sql.L.detach().float() @ sql.R.detach().float()) + sql.res_wq.detach().float()  # [out,in] ~= W·diag(s)
        bias = sql.linear.bias
        in_f, out_f = sql.in_features, sql.out_features
        src = nn.Linear(in_f, out_f, bias=bias is not None).to(dev, DT)
        src.weight.data.copy_(Wp.to(DT))
        if bias is not None:
            src.bias.data.copy_(bias.detach().to(DT))
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.q = SVDQW4A4Linear.from_linear(src, rank=min(rank, in_f, out_f)).to(dev)
        obj.register_buffer("smooth", s.to(DT).to(dev))
        obj.out_features = out_f
        obj._calibrated = True
        return obj


@torch.no_grad()
def selftest_bridge(sql, n_tokens: int = 64, seed: int = 0) -> float:
    """Wiring guard for from_calibrated on ONE calibrated SVDQuantLinear: rel(deploy(x), fakequant(x)) on a
    random x. Small (a few %) = correctly wired, residual is kernel numerics (the measurement). ~>0.5 =
    a wiring/smoothing-convention bug (e.g. x*s vs x/s, transposed L/R) — NOT kernel degradation."""
    dev = sql.smooth.device
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(1, n_tokens, sql.in_features, device=dev, generator=g)
    ref = sql(x).float()                                                  # fake-quant reference
    dep = SVDQuantDeployLinear.from_calibrated(sql)(x).float()            # Tier-2 real int4
    return (dep - ref).norm().item() / max(ref.norm().item(), 1e-12)


@torch.no_grad()
def selftest_dit_bridge(model, n: int = 8):
    """Run selftest_bridge on n SVDQuantLinears sampled across depth; print rel + an OK/!!WIRING? flag.
    Call AFTER run_svdquant.calibrate and BEFORE wrap_dit_svdquant_from_calibrated, to catch a wiring bug
    before the (expensive) full generation. Prints a FULL traceback on the first failure. Returns [(name, rel)]."""
    import traceback
    import svdquant_dit as sqd
    sqls = [(f"{bname}.{attr}", child)
            for bname, block in model.transformer.named_modules() if re.search(r"\.(self_attn|cross_attn|ffn)$", bname)
            for sub in block.modules()
            for attr, child in sub.named_children()
            if isinstance(child, sqd.SVDQuantLinear)
            and child.in_features % 64 == 0 and child.out_features % 64 == 0]
    if not sqls:
        print("  [selftest] no wrappable calibrated SVDQuantLinear found"); return []
    step = max(1, len(sqls) // n)
    out, first_err = [], True
    for name, sql in sqls[::step][:n]:
        try:
            r = selftest_bridge(sql)
            flag = "OK" if r < 0.5 else "!!WIRING?  (NOT kernel degradation — check x/s + L,R shapes)"
        except Exception as e:
            r = float("nan"); flag = f"ERR {type(e).__name__}: {str(e)[:60]}"
            if first_err:
                traceback.print_exc(); first_err = False
        out.append((name, r))
        print(f"  [selftest] {name:40s} rel(deploy,fakequant)={r:.4f}  {flag}", flush=True)
    return out


def wrappable(lin):
    return isinstance(lin, nn.Linear) and lin.in_features % 64 == 0 and lin.out_features % 64 == 0


def wrap_dit_svdquant(model, rank: int = RANK):
    wrapped, skipped = 0, []
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in list(sub.named_children()):
                if isinstance(child, nn.Linear):
                    if wrappable(child):
                        try:
                            setattr(sub, attr, SVDQuantDeployLinear(child, rank=rank))
                            wrapped += 1
                        except Exception as e:
                            skipped.append((f"{bname}.{attr}", f"{type(e).__name__}:{str(e)[:50]}"))
                    else:
                        skipped.append((f"{bname}.{attr}", f"in{child.in_features}/out{child.out_features} not %64"))
    return wrapped, skipped


def wrap_dit_svdquant_from_calibrated(model, rank: int = RANK):
    """TIER-2: swap each CALIBRATED svdquant_dit.SVDQuantLinear in the DiT for a from_calibrated deploy
    bridge (real Nunchaku int4 carrying our calibrated smoothing). Run AFTER run_svdquant.calibrate(model,...).
    Non-%64 linears are left as the fake-quant SVDQuantLinear -> HYBRID run; report coverage so the deploy
    delta stays attributable. Returns (wrapped, skipped)."""
    import svdquant_dit as sqd
    wrapped, skipped = 0, []
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in list(sub.named_children()):
                if isinstance(child, sqd.SVDQuantLinear):
                    if child.in_features % 64 != 0 or child.out_features % 64 != 0:
                        skipped.append((f"{bname}.{attr}", f"in{child.in_features}/out{child.out_features} not %64"))
                        continue
                    try:
                        setattr(sub, attr, SVDQuantDeployLinear.from_calibrated(child, rank=rank))
                        wrapped += 1
                    except Exception as e:
                        skipped.append((f"{bname}.{attr}", f"{type(e).__name__}:{str(e)[:50]}"))
    return wrapped, skipped


def _named_sqls(model):
    """Ordered [(name, parent, attr, SVDQuantLinear)] of the wrapped target linears (deterministic order)."""
    import svdquant_dit as sqd
    out = []
    for bname, block in model.transformer.named_modules():
        if not re.search(r"\.(self_attn|cross_attn|ffn)$", bname):
            continue
        for sub in block.modules():
            for attr, child in sub.named_children():
                if isinstance(child, sqd.SVDQuantLinear):
                    out.append((f"{bname}.{attr}", sub, attr, child))
    return out


def save_calib(model, path):
    """Serialize the calibrated SVDQuant transforms (smooth/L/R/res_wq/alpha/bias) so a re-run skips the
    ~14-min capture+SVD. Save AFTER run_svdquant.calibrate, BEFORE the bridge, so a later crash keeps it."""
    blob = []
    for name, _, _, sql in _named_sqls(model):
        blob.append(dict(
            name=name, in_f=sql.in_features, out_f=sql.out_features, alpha=float(sql.alpha),
            smooth=sql.smooth.detach().cpu(), L=sql.L.detach().cpu(), R=sql.R.detach().cpu(),
            res_wq=sql.res_wq.detach().cpu(),
            bias=(sql.linear.bias.detach().cpu() if sql.linear.bias is not None else None)))
    torch.save(blob, path)
    return len(blob)


def load_calib(model, path, dev):
    """Rebuild calibrated SVDQuantLinears from a save_calib cache: wrap fresh nn.Linears, then fill tensors.
    Positional match with a per-linear (in,out) assertion to catch any structural drift."""
    import svdquant_dit as sqd
    blob = torch.load(path, map_location="cpu")
    sqd.wrap_dit(model)                                   # wrap all targets (uncalibrated SVDQuantLinear)
    sqls = _named_sqls(model)
    assert len(sqls) == len(blob), f"cache has {len(blob)} linears but model has {len(sqls)}"
    for (name, _, _, sql), rec in zip(sqls, blob):
        assert (sql.in_features, sql.out_features) == (rec["in_f"], rec["out_f"]), \
            f"dim mismatch at {name}: {(sql.in_features, sql.out_features)} vs cache {(rec['in_f'], rec['out_f'])}"
        sql.smooth.copy_(rec["smooth"].to(dev, sql.smooth.dtype))
        sql.L = rec["L"].to(dev); sql.R = rec["R"].to(dev); sql.res_wq = rec["res_wq"].to(dev)
        if rec["bias"] is not None and sql.linear.bias is not None:
            sql.linear.bias.data.copy_(rec["bias"].to(dev, sql.linear.bias.dtype))
        sql.alpha = rec["alpha"]; sql._calibrated = True
        sql.linear.weight = None
    return len(blob)
