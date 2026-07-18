"""Collect Phase 0 (GATE-B) results into one CSV + print the gate readout.

Scans results/ for  <prefix>_<run><suffix>_<set>_<metric>.txt  aggregate files (written by
evaluate_seedtts_metrics.sh), parses the final "NAME: value" line of each, and emits:
  - a long-format CSV (run, group, set, metric, value)
  - a console readout per (set, metric):
        fp32 reference | random-draw mean±std (between-SET variance) | s0 seed-repeat std (NOISE)
    GATE-B asks: between-set std >> seed std, and the (best-worst) oracle gap worth chasing.

This reads only the aggregate numbers; per-item significance goes through paired_bootstrap.py
on the per-item score files (asr merge / sim lists) against the p0_fp32 reference.

Usage:
  python -m audio_dit_quantize.calib.phase0_collect --prefix p0 --sets "zh_dev hard_dev en_dev" \
      --out results/p0_summary.csv
"""
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..paths import RESULTS_DIR

_VAL_RE = re.compile(r"([A-Za-z_]+)\s*:\s*(-?[0-9.]+)")
METRICS = ("cer", "wer", "sim", "utmos", "dnsmos")


def parse_value(path):
    """Last 'NAME: <float>' in the file (aggregate line is written last)."""
    val = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = _VAL_RE.search(line)
        if m:
            try:
                val = float(m.group(2).rstrip("."))
            except ValueError:
                pass
    return val


def classify(run):
    if run == "fp32":
        return "fp32"
    if re.fullmatch(r"rand\d+_s\d+", run):
        return "random_draw"
    if re.fullmatch(r"rand\d+_s\d+_cs\d+", run):
        return "seed_repeat"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="p0")
    ap.add_argument("--suffix", default="", help="model suffix baked into tags (e.g. _3.5b)")
    ap.add_argument("--sets", default="zh_dev hard_dev en_dev")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sets = args.sets.replace(",", " ").split()
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.prefix}_summary{args.suffix}.csv"

    rows = []
    pat = re.compile(rf"^{re.escape(args.prefix)}_(?P<run>.+?){re.escape(args.suffix)}"
                     rf"_(?P<set>{'|'.join(map(re.escape, sets))})_(?P<met>{'|'.join(METRICS)})\.txt$")
    for f in sorted(Path(RESULTS_DIR).glob(f"{args.prefix}_*.txt")):
        m = pat.match(f.name)
        if not m:
            continue
        v = parse_value(f)
        if v is None:
            print(f"[collect] WARN no value parsed from {f.name}", file=sys.stderr)
            continue
        rows.append({"run": m["run"], "group": classify(m["run"]),
                     "set": m["set"], "metric": m["met"], "value": v})
    if not rows:
        print("[collect] no result files matched — nothing to summarize", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "group", "set", "metric", "value"])
        w.writeheader(); w.writerows(rows)
    print(f"[collect] {len(rows)} values -> {out}\n")

    # gate readout
    by = defaultdict(dict)
    for r in rows:
        by[(r["set"], r["metric"])][r["run"]] = r["value"]
    print(f"{'set':<10}{'metric':<8}{'fp32':>8}{'rand mean±std':>18}{'oracle gap':>12}{'seed std':>10}{'ratio':>7}")
    for (s, met), runs in sorted(by.items()):
        fp = runs.get("fp32")
        draws = [v for r, v in runs.items() if classify(r) == "random_draw"]
        s0 = [v for r, v in runs.items()
              if classify(r) == "seed_repeat" or re.fullmatch(r"rand\d+_s0", r)]
        if not draws:
            continue
        dstd = np.std(draws, ddof=1) if len(draws) > 1 else float("nan")
        sstd = np.std(s0, ddof=1) if len(s0) > 1 else float("nan")
        ratio = dstd / sstd if sstd and sstd == sstd and sstd > 0 else float("nan")
        print(f"{s:<10}{met:<8}"
              + (f"{fp:>8.3f}" if fp is not None else f"{'-':>8}")
              + f"{np.mean(draws):>11.3f}±{dstd:<5.3f}"
              + f"{max(draws)-min(draws):>11.3f}"
              + f"{sstd:>10.3f}{ratio:>7.2f}")
    print("\n[collect] GATE-B: 'ratio' = between-set std / seed std — >>1 with a meaningful oracle "
          "gap opens the data-selection line; ~1 means content choice is inside calibration noise.")


if __name__ == "__main__":
    main()
