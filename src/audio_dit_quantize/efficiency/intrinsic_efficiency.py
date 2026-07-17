"""Intrinsic (hardware/kernel-INDEPENDENT) efficiency metrics for LongCat-AudioDiT quantization.

WHY: a wall-clock latency table is unfair to W4A4 — on A100 the deployed FlatQuant W4A4 is ~2x
SLOWER than fp32 because of the online Kron-transform LAUNCH overhead (not the method); the INT4
GEMM core itself is 1.1-3.6x faster (see docs/03 sec 3.5). So we report efficiency in two layers
(docs/03 sec 3.6): this script emits LAYER 1 = the intrinsic format efficiency (compression ratio,
BitOps, weight memory traffic) that is a pure function of shape x bitwidth — identical on any GPU /
kernel and immune to kernel maturity. Layer 2 (measured latency / peak VRAM) lives in results/
e2e_efficiency.txt + logs/int8_efficiency_sweep.log.

WHAT IT COUNTS: only the quantized DiT linears = the nn.Linear under each block's .self_attn /
.cross_attn / .ffn (matches scripts/quant_sensitivity.py::target_linears and
w4a4_deploy_fp16glue.py::wrap_dit_w4a4). Per block = 10 linears: self_attn {to_q,to_k,to_v,
to_out}, cross_attn {to_q,to_k,to_v,to_out} (all dim x dim, since heads*dim_head == dim and the text
context is projected to dim before cross-attn), ffn {ff.0 up: dim->ffn_inner, ff.3 down: ffn_inner->dim}.
EXCLUDES adaLN modulation, time-embed, embedders, proj_out, text_conv, norms, VAE, text-encoder.
The inventory reproduces the real-model enumeration to the parameter: 1B 905.97M, 3.5B 3187.67M.

BitOps = MACs x bit_w x bit_a (Baskin et al., UNIQ; an ASIC area/power proxy). MACs/forward over the
quantized linears = (sum in*out) * T = quant_params * T. Default T=150 frames (representative Seed
voice-clone seq length: measured ~105 normal / ~171 hard; codec hop=2048 compresses heavily, so it
is NOT ~1500). BitOps/traffic scale linearly in T; weight bytes are T-independent.

  *** GUARDRAIL ***  The BitOps reduction (W4A4 = 16x fewer than fp16) is an ASIC-area/power proxy,
  NOT a GPU wall-clock target. On A100 the INT4 tensor-core peak is only ~4x FP16 (1248 vs 312
  TOPS/TFLOPS) — that ~4x is the realistic GPU ceiling, and it is what the measured int4 GEMM core
  (1.1-3.6x) realizes. Never advertise 16x as an achievable A100 speedup.

Run (no GPU, no weight load — reads only config.json):
  python3 scripts/intrinsic_efficiency.py [--seq 150] [--out results/intrinsic_efficiency.txt]
"""
import argparse
import glob
import json
import os

HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = [
    ("LongCat-AudioDiT-1B", "models--meituan-longcat--LongCat-AudioDiT-1B"),
    ("LongCat-AudioDiT-3.5B", "models--meituan-longcat--LongCat-AudioDiT-3.5B"),
]
# (label, weight bits, activation bits). W4A16 == INT8 in BitOps (4*16 == 8*8) but int4 memory.
PRECISIONS = [("fp32", 32, 32), ("fp16", 16, 16), ("INT8", 8, 8), ("W4A16", 4, 16), ("W4A4", 4, 4)]
FORWARDS_PER_GEN = 30  # 15 Euler steps x 2 (CFG cond+uncond); nfe=16 -> 2*(nfe-1)


def load_config(hub_dir: str) -> dict:
    hits = glob.glob(os.path.join(HF_HUB, hub_dir, "snapshots", "*", "config.json"))
    if not hits:
        raise FileNotFoundError(f"config.json not found under {hub_dir}")
    with open(hits[0]) as f:
        return json.load(f)


def quant_linears(cfg: dict):
    """Return (count, total_quant_params, breakdown) for the quantized DiT linears from config dims."""
    dim = cfg["dit_dim"]
    depth = cfg["dit_depth"]
    heads = cfg["dit_heads"]
    dim_head = dim // heads
    assert heads * dim_head == dim, f"heads*dim_head ({heads}*{dim_head}) != dim ({dim})"
    ffn_inner = int(round(dim * cfg["dit_ff_mult"]))
    attn = dim * dim                       # to_q/to_k/to_v/to_out, self + cross = 8 per block
    ffn = dim * ffn_inner + ffn_inner * dim  # up + down = 2 per block
    per_block_params = 8 * attn + ffn
    per_block_linears = 8 + 2
    return (
        per_block_linears * depth,
        per_block_params * depth,
        dict(dim=dim, depth=depth, heads=heads, dim_head=dim_head, ffn_inner=ffn_inner),
    )


def fmt_gb(nbytes: float) -> float:
    return nbytes / 1e9  # decimal GB, standard for model size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=150, help="representative DiT sequence length (frames)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "intrinsic_efficiency.txt"))
    args = ap.parse_args()
    T = args.seq

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Intrinsic (hardware-independent) efficiency — LongCat-AudioDiT quantized DiT linears")
    emit(f"# T={T} frames, {FORWARDS_PER_GEN} DiT forwards/generation. BitOps = MACs*bit_w*bit_a (ASIC proxy).")
    emit(f"# GUARDRAIL: BitOps 16x reduction != GPU speedup; A100 INT4 peak is ~4x FP16 (real GPU ceiling).")
    emit("")

    for label, hub in MODELS:
        cfg = load_config(hub)
        n_lin, n_params, d = quant_linears(cfg)
        macs_fwd = n_params * T  # = sum(in*out)*T
        emit(f"== {label} ==  dim={d['dim']} depth={d['depth']} heads={d['heads']}x{d['dim_head']} "
             f"ffn_inner={d['ffn_inner']}")
        emit(f"   quantized linears: {n_lin}  ({d['depth']}x[8 attn dim*dim + 2 ffn])   "
             f"quant params: {n_params/1e6:.2f} M")
        emit(f"   MACs/forward (T={T}): {macs_fwd/1e9:.2f} GMAC")
        emit("")
        hdr = (f"   {'precision':<8} {'W/A':>6} {'weight GB':>10} {'compr vs fp32':>14} "
               f"{'compr vs fp16':>14} {'BitOps/fwd Tb':>14} {'BitOps vs fp16':>15} "
               f"{'wt traffic/gen GB':>18}")
        emit(hdr)
        emit("   " + "-" * (len(hdr) - 3))
        wbytes = {bw: n_params * (bw / 8.0) for _, bw, _ in PRECISIONS}
        fp32_b = wbytes[32]
        fp16_b = wbytes[16]
        bitops = {}
        for pname, bw, ba in PRECISIONS:
            bitops[pname] = macs_fwd * bw * ba  # bit-operations per forward
        fp16_bitops = bitops["fp16"]
        for pname, bw, ba in PRECISIONS:
            wb = wbytes[bw]
            traffic_gen = fmt_gb(wb) * FORWARDS_PER_GEN  # weights re-read every forward (flow model, bs=1)
            emit(f"   {pname:<8} {f'{bw}/{ba}':>6} {fmt_gb(wb):>10.3f} {fp32_b/wb:>13.1f}x "
                 f"{fp16_b/wb:>13.1f}x {bitops[pname]/1e12:>14.2f} {fp16_bitops/bitops[pname]:>14.1f}x "
                 f"{traffic_gen:>18.1f}")
        emit("")

    emit("# Layer 2 (measured, kernel-dependent) lives in results/e2e_efficiency.txt + "
         "logs/int8_efficiency_sweep.log; see docs/03 sec 3.6.")

    outp = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {outp}")


if __name__ == "__main__":
    main()
