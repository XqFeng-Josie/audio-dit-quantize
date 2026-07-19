"""Per-item paired bootstrap for the order-free paired protocol: ΔSIM (timbre) or ΔCER/ΔWER (content).

A = baseline (e.g. `full`), B = method (e.g. `late`). Pairs items by uid, bootstraps the mean paired
difference (B − A), and reports the CI + P. PURE post-hoc analysis of the official per-item scores —
it never regenerates or re-evaluates anything.

VALIDITY: A and B must be generated under the SAME per-item seed base (e.g. 1024) so each uid shares
identical generation noise (true pairing) — as `generate_step_axis.py` does for full/early/late. The CI
is conditional on that fixed protocol + fixed calibration draw; it makes no calibration-run-to-run claim.

File formats (auto by --metric):
  sim      seedtts_similarity.py:  <gen_wav>|<prompt_wav>\t<score>       higher = better
  cer/wer  average_wer.py:         <wav>\t<score>\t<ref>\t<hyp>\t...     lower  = better

Usage (after `source env.sh`):
  python -m audio_dit_quantize.paired_bootstrap --metric sim \
      --a results/step_axis/full_hard_sim.txt --b results/step_axis/late_hard_sim.txt --labels full late
  # -> mean ΔSIM, 95% CI, P(late higher), SIG/ns
"""
import argparse, os
import numpy as np


def _uid(field):
    """uid from a score-file first field: strip a trailing |prompt, dirname, and .wav."""
    uid = os.path.basename(field.split("|")[0])
    return uid[:-4] if uid.endswith(".wav") else uid


def parse_per_item(path, metric):
    """Return {uid: score}. SIM lines have '|'; CER/WER lines are tab wav<TAB>score<TAB>..."""
    out = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith(("SIM:", "WER:", "CER:", "utt\t")):
            continue
        if metric == "sim" and "|" not in line:
            continue
        f = line.split("\t")
        if len(f) < 2:
            continue
        try:
            out[_uid(f[0])] = float(f[1])
        except ValueError:
            continue
    return out


def paired_bootstrap(a, b, higher_better, n_boot=10000, seed=0, ci=95.0, scale=1.0):
    """a, b: 1D arrays of paired per-item scores (same order). Returns dict of stats."""
    d = (b - a) * scale                       # per-item paired diff (B - A)
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    p_better = float((boot > 0).mean()) if higher_better else float((boot < 0).mean())
    sig = (lo > 0) if higher_better else (hi < 0)         # CI excludes 0, B better
    sig_worse = (hi < 0) if higher_better else (lo > 0)   # CI excludes 0, B worse
    return dict(mean=d.mean(), lo=lo, hi=hi, p_better=p_better, sig=sig, sig_worse=sig_worse,
                std=d.std(), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["sim", "cer", "wer"], required=True)
    ap.add_argument("--a", required=True, help="baseline per-item file (e.g. full)")
    ap.add_argument("--b", required=True, help="method per-item file (e.g. late/early)")
    ap.add_argument("--labels", nargs=2, default=["A(baseline)", "B(method)"])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ci", type=float, default=95.0)
    args = ap.parse_args()

    higher_better = args.metric == "sim"
    A, B = parse_per_item(args.a, args.metric), parse_per_item(args.b, args.metric)
    uids = sorted(set(A) & set(B))
    if not uids:
        raise SystemExit("no shared uids between the two files (not paired / wrong --metric?)")
    a = np.array([A[u] for u in uids]); b = np.array([B[u] for u in uids])
    scale = 1.0 if args.metric == "sim" else 100.0
    unit = "" if args.metric == "sim" else "%"
    r = paired_bootstrap(a, b, higher_better, args.n_boot, args.seed, args.ci, scale)

    dirw = "higher" if higher_better else "lower"
    print(f"paired bootstrap  metric={args.metric}  n={r['n']} items  "
          f"(A-only {len(set(A) - set(B))}, B-only {len(set(B) - set(A))} dropped)")
    print(f"  {args.labels[0]:18} mean = {a.mean():.4f}{unit}")
    print(f"  {args.labels[1]:18} mean = {b.mean():.4f}{unit}")
    print(f"  Δ (B−A) = {r['mean']*(1 if args.metric=='sim' else 1):+.4f}{unit}   "
          f"{args.ci:.0f}% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    print(f"  P({args.labels[1]} {dirw}=better) = {r['p_better']:.3f}   per-item std = {r['std']:.4f}")
    verdict = (f"SIGNIFICANT — {args.labels[1]} better, CI excludes 0 ✅" if r["sig"] else
               f"SIGNIFICANT — {args.labels[1]} WORSE, CI excludes 0 ⚠️" if r["sig_worse"] else
               "ns — CI crosses 0")
    print(f"  VERDICT: {verdict} (fixed protocol + fixed calib draw)")


if __name__ == "__main__":
    main()
