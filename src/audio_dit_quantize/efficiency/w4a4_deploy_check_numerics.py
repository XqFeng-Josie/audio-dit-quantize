"""GATE 0 for the best-config deploy (w4a4_deploy_quality.py): verify the asym+LAC+add_diag deploy math is
(a) numerically EQUAL to the best-config fake-quant with fp32 glue (~1e-6 rel-err -> the packing/transform/
asym-correction are correct), and (b) shows the fp16-glue error (the quantity we then test end-to-end).

Math identity (holds on ANY input, so random activations suffice to test correctness):
  fake-quant:  y = scale_x*(q_x - zero) @ (scale_w*q_w)^T
  deploy:      y = scale_x*scale_w*[ (q_x-8) @ q_w^T + (8-zero)*rowsum(q_w) ] = scale_x*scale_w*(q_x-zero)@q_w^T   ✓

Run (after `source env.sh`; loads the saved best-config model; ~1 min, small GPU footprint):
  python -m audio_dit_quantize.efficiency.w4a4_deploy_check_numerics --model models/bc_1b_model.pt --sample 24
"""
import argparse, torch
import audiodit  # noqa
from .. import flatquant_layers as fq
from ..flatquant_layers import FlatQuantLinear
from .w4a4_deploy_quality import W4A4BCDeployLinear, bc_fakequant_ref


def relerr(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def main():
    from ..paths import bc_model_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B",
                    help="picks the fixed model default (1B -> models/bc_1b_model.pt, 3.5B -> bc_3p5b_model.pt)")
    ap.add_argument("--model", default=None,
                    help="fixed best-config model path (default: from --model_dir; override dir with SEED_MODELS_DIR)")
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model_path = args.model or str(bc_model_path(args.model_dir))

    model = torch.load(model_path, weights_only=False, map_location=dev); model.eval()
    fqls = [m for m in model.transformer.modules()
            if isinstance(m, FlatQuantLinear) and getattr(m, "_frozen", False)
            and m.linear.in_features % 32 == 0 and m.linear.out_features % 32 == 0]
    print(f"[gate0] {len(fqls)} frozen best-config linears; testing {min(args.sample, len(fqls))}")
    step = max(1, len(fqls) // args.sample)
    picked = fqls[::step][: args.sample]

    fq._ACT_QUANT = True
    r32, r16 = [], []
    lac = add_diag = 0
    for fql in picked:
        lac += int(fql.lac)
        add_diag += int(getattr(fql.trans, "add_diag", False) and getattr(fql.trans, "use_diag", True))
        K = fql.linear.in_features
        x = torch.randn(1, args.tokens, K, device=dev) * 2.0            # arbitrary activations (identity holds on any x)
        ref = bc_fakequant_ref(fql, x)
        d32 = W4A4BCDeployLinear(fql, torch.float32).to(dev)(x)
        d16 = W4A4BCDeployLinear(fql, torch.float16).to(dev)(x)
        r32.append(relerr(d32, ref))
        r16.append(relerr(d16, ref))
    import statistics as st
    print(f"[gate0] linears with lac={lac}/{len(picked)}  add_diag={add_diag}/{len(picked)}")
    print(f"[gate0] fp32-glue vs fake-quant  rel-err: mean {st.mean(r32):.2e}  max {max(r32):.2e}   "
          f"{'PASS (exact) ✅' if max(r32) < 1e-4 else 'FAIL ❌ (math mismatch)'}")
    print(f"[gate0] fp16-glue vs fake-quant  rel-err: mean {st.mean(r16):.2e}  max {max(r16):.2e}   "
          f"(this is the per-linear fp16 glue error; end-to-end quality tested separately)")


if __name__ == "__main__":
    main()
