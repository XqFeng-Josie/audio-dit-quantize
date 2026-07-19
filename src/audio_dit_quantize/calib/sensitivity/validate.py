"""E2 validation gate: do the scorer's set-level scores retro-predict real outcomes?

Reads <scores_dir>/*.json (from sensitivity.score) and the corresponding runs' dev metrics
from results/, then reports Spearman rank correlations. E2 passes its gate (docs §3.4) only
if a score family significantly rank-predicts hard-dev CER across the labeled sets
(10 random + 4 contrast = 14 as of 2026-07-20).

List-name -> result-tag mapping (extend _tag_of for new runs):
  rand32_s<k>      -> p0_rand32_s<k>
  ctr_<x>_s1000    -> p1c_<x>

Usage:
  python -m audio_dit_quantize.calib.sensitivity.validate --scores_dir data/calib_pool/scores
"""
import argparse, glob, json, os, re

import numpy as np

from ...paths import RESULTS_DIR


def _tag_of(list_stem):
    m = re.fullmatch(r"rand(\d+)_s(\d+)", list_stem)
    if m:
        return f"p0_{list_stem}"
    m = re.fullmatch(r"ctr_([a-z0-9]+)_s\d+", list_stem)
    if m:
        return f"p1c_{m.group(1)}"
    return None


def _metric_value(tag, setname, metric):
    p = os.path.join(str(RESULTS_DIR), f"{tag}_{setname}_{metric}.txt")
    if not os.path.exists(p):
        return None
    vals = []
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("utt\t") or re.match(r"^(WER|CER|SIM):", line):
            continue
        f = line.split("\t")
        if len(f) >= 2:
            try:
                vals.append(float(f[1]))
            except ValueError:
                pass
    if not vals:
        return None
    v = float(np.mean(vals))
    return v * 100 if metric in ("cer", "wer") else v


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", default="data/calib_pool/scores")
    ap.add_argument("--targets", default="hard_dev:cer,zh_dev:cer,en_dev:wer",
                    help="comma list of <set>:<metric> outcome axes")
    args = ap.parse_args()
    targets = [t.split(":") for t in args.targets.split(",")]

    rows = []
    for p in sorted(glob.glob(os.path.join(args.scores_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        tag = _tag_of(stem)
        if tag is None:
            print(f"[val] skip {stem} (no tag mapping)")
            continue
        payload = json.load(open(p, encoding="utf-8"))
        row = {"stem": stem, "tag": tag, **payload["set_scores"]}
        ok = True
        for s, m in targets:
            v = _metric_value(tag, s, m)
            if v is None:
                ok = False
                print(f"[val] skip {stem}: missing {tag}_{s}_{m}")
                break
            row[f"{s}:{m}"] = v
        if ok:
            rows.append(row)
    if len(rows) < 5:
        print(f"[val] only {len(rows)} scored+labeled sets — need >=5 for a meaningful rank test")
        return
    score_keys = [k for k in rows[0] if k.startswith(("sum_", "mean_"))]
    print(f"[val] n={len(rows)} sets: {', '.join(r['stem'] for r in rows)}")
    print(f"{'score':<26}" + "".join(f"{s}:{m:>4}".rjust(16) for s, m in targets)
          + "   (|rho|>~0.53 = p<.05 at n=14)")
    for k in score_keys:
        x = np.array([r[k] for r in rows])
        line = f"{k:<26}"
        for s, m in targets:
            y = np.array([r[f'{s}:{m}'] for r in rows])
            line += f"{_spearman(x, y):>+16.2f}"
        print(line)
    print("\n[val] gate: a score family with strong |rho| vs hard_dev:cer -> E2 usable for P2;"
          " sign convention: CER lower is better, so a GOOD 'high score = good set' signal shows rho < 0.")


if __name__ == "__main__":
    main()
