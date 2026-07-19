"""Offline error-type mechanism attribution for W4A4 quantization (docs §4.7).

Reads the per-item ref/hyp text already sitting in results/*_{hard,zh}_dev_cer.txt
(zero GPU), char-aligns each pair, and classifies every edit operation:

  substitutions  homophone (same pinyin+tone: ASR char-choice noise, acoustically
                 correct) / tone-only / initial-confusion (z-zh, c-ch, s-sh, n-l,
                 f-h, l-r; same final) / nasal-final (an-ang, en-eng, in-ing, ...;
                 same initial) / other
  deletions      inside a tandem-repeat span (repeat-collapse) / other
  insertions     echo of a nearby ref char (stutter) / other

hard_dev items are split into three structural categories: twister (uid
raokouling-*), repeat (some gram tandem-repeated >= 3x in the ref), plain.

Conventions (see §4.7): rates are corpus-level (total edits / total ref CJK chars,
long items weigh more); the official aggregate is the unweighted per-item mean
(sanity-checked here: item mean reproduces it exactly, rank corr between the two
conventions rho=0.92). "Strict CER" drops homophone substitutions.

Parsing gotcha (docs §5.2 2026-07-19): these files declare 8 header columns but
data rows carry 7 (no wav_res) — always index from the row END.

Usage (after `source env.sh`, from the repo root):
  python -m audio_dit_quantize.calib.error_taxonomy                 # §4.7 tables
  python -m audio_dit_quantize.calib.error_taxonomy \
      --extra_runs p2_rule_r0 p2_rule_r1                            # + pre-registered check
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from pypinyin import Style, pinyin

CJK = re.compile(r"[一-鿿]")

FP32 = "p0_fp32"
RAND = [f"p0_rand32_s{k}" for k in range(10)]
CTR = ["p1c_en0", "p1c_en12", "p1c_hard0", "p1c_hard16"]
BASE_RUNS = [FP32, "int8"] + RAND + CTR
W4A4 = RAND + CTR                                   # the 14 runs used for correlations

TYPES = ["sub_homophone", "sub_tone", "sub_initial", "sub_nasal", "sub_other",
         "del_repeat", "del_other", "ins_echo", "ins_other"]
INIT_CONF = {frozenset(p) for p in
             [("z", "zh"), ("c", "ch"), ("s", "sh"), ("n", "l"), ("f", "h"), ("l", "r")]}

_py_cache = {}


def _py(ch):
    if ch not in _py_cache:
        ini = pinyin(ch, style=Style.INITIALS, heteronym=False, strict=False)[0][0]
        fin = pinyin(ch, style=Style.FINALS_TONE3, heteronym=False, strict=False)[0][0]
        _py_cache[ch] = (ini, re.sub(r"\d", "", fin), (re.findall(r"\d", fin) or ["0"])[0])
    return _py_cache[ch]


def classify_sub(r, h):
    ri, rf, rt = _py(r)
    hi, hf, ht = _py(h)
    if (ri, rf, rt) == (hi, hf, ht):
        return "sub_homophone"
    if (ri, rf) == (hi, hf):
        return "sub_tone"
    if rf == hf and frozenset((ri, hi)) in INIT_CONF:
        return "sub_initial"
    if ri == hi and (rf + "g" == hf or hf + "g" == rf):
        return "sub_nasal"
    return "sub_other"


def norm(s):
    return "".join(CJK.findall(s or ""))


def align(ref, hyp):
    """Levenshtein backtrace -> list of (op, ref_pos, ref_char, hyp_char)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i-1, j-1] + (ref[i-1] != hyp[j-1]), D[i-1, j] + 1, D[i, j-1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i-1, j-1] + (ref[i-1] != hyp[j-1]):
            if ref[i-1] != hyp[j-1]:
                ops.append(("sub", i-1, ref[i-1], hyp[j-1]))
            i, j = i-1, j-1
        elif i > 0 and D[i, j] == D[i-1, j] + 1:
            ops.append(("del", i-1, ref[i-1], ""))
            i -= 1
        else:
            ops.append(("ins", i, "", hyp[j-1]))
            j -= 1
    return ops


def repeat_mask(ref, max_k=8):
    """mask[i]=True iff ref[i] lies inside a tandem repeat (gram of len k, twice+)."""
    mask = np.zeros(len(ref), bool)
    for k in range(1, max_k + 1):
        for j in range(len(ref) - 2 * k + 1):
            if ref[j:j+k] == ref[j+k:j+2*k]:
                mask[j:j+2*k] = True
    return mask


def max_tandem(ref, max_k=8):
    best = 1
    for k in range(1, max_k + 1):
        j = 0
        while j + k <= len(ref):
            r = 1
            while ref[j:j+k] == ref[j+k*r:j+k*(r+1)]:
                r += 1
            best = max(best, r)
            j += k if r == 1 else k * r
    return best


def category(uid, ref):
    if uid.startswith("raokouling"):
        return "twister"
    return "repeat" if max_tandem(ref) >= 3 else "plain"


def load(results_dir, run, setname):
    """[(uid, ref, hyp)] from a per-item CER/WER file. Index from the row END (7 vs 8 cols)."""
    metric = "wer" if setname.startswith("en") else "cer"
    rows = []
    for line in open(f"{results_dir}/{run}_{setname}_{metric}.txt"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7 or p[0] in ("utt",) or p[0].startswith(("WER", "SIM")):
            continue
        uid = os.path.basename(p[0]).replace(".wav", "")
        rows.append((uid, norm(p[-5]), norm(p[-4])))
    return rows


def official_and_item_mean(results_dir, run, setname):
    metric = "wer" if setname.startswith("en") else "cer"
    vals, agg = [], None
    for line in open(f"{results_dir}/{run}_{setname}_{metric}.txt"):
        p = line.rstrip("\n").split("\t")
        if p[0].startswith("WER"):
            agg = float(p[0].split()[-1].rstrip("%"))
        elif len(p) >= 7 and p[0] != "utt":
            vals.append(float(p[-6]))
    return agg, 100 * float(np.mean(vals))


def analyze(results_dir, run, setname, ref_cache):
    """-> (rates dict incl. CER/CER_strict, {uid: (nchars, per-type counts)})"""
    per_item, tot, N, edits = {}, {t: 0 for t in TYPES}, 0, 0
    for uid, ref, hyp in load(results_dir, run, setname):
        if not ref:
            continue
        if uid not in ref_cache:
            ref_cache[uid] = repeat_mask(ref)
        rmask = ref_cache[uid]
        cnt = {t: 0 for t in TYPES}
        for op, pos, rc, hc in align(ref, hyp):
            edits += 1
            if op == "sub":
                cnt[classify_sub(rc, hc)] += 1
            elif op == "del":
                cnt["del_repeat" if rmask[pos] else "del_other"] += 1
            else:
                cnt["ins_echo" if hc in ref[max(0, pos-3):pos+3] else "ins_other"] += 1
        for t in TYPES:
            tot[t] += cnt[t]
        per_item[uid] = (len(ref), cnt)
        N += len(ref)
    rates = {t: 100.0 * tot[t] / N for t in TYPES}
    rates["CER"] = 100.0 * edits / N
    rates["CER_strict"] = rates["CER"] - rates["sub_homophone"]
    return rates, per_item


def collect(results_dir, setname, runs):
    ref_cache, rates, items = {}, {}, {}
    for r in runs:
        try:
            rates[r], items[r] = analyze(results_dir, r, setname, ref_cache)
        except FileNotFoundError:
            print(f"[skip] {r} {setname}: file missing")
    return pd.DataFrame(rates).T, items


def boot_delta(items_a, items_b, uids, n_boot=10000, seed=0):
    """Corpus-level ΔCER (b−a) on a uid subset; bootstrap over items. -> (Δ, lo, hi, sig)"""
    rng = np.random.default_rng(seed)
    uids = [u for u in uids if u in items_a and u in items_b]
    na = np.array([items_a[u][0] for u in uids], float)
    ea = np.array([sum(items_a[u][1].values()) for u in uids], float)
    eb = np.array([sum(items_b[u][1].values()) for u in uids], float)
    idx = rng.integers(0, len(uids), (n_boot, len(uids)))
    d = 100 * (eb[idx].sum(1) - ea[idx].sum(1)) / na[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return 100 * (eb.sum() - ea.sum()) / na.sum(), lo, hi, ("SIG" if lo * hi > 0 else "ns")


def cat_cer(items_of_run, uids):
    n = sum(items_of_run[u][0] for u in uids if u in items_of_run)
    e = sum(sum(items_of_run[u][1].values()) for u in uids if u in items_of_run)
    return 100.0 * e / n


def pooled_items(items_by_run, runs, ref_items):
    """Average the runs' per-item edit counts into one pseudo-run (for pooled Δ)."""
    return {u: (ref_items[u][0],
                {"x": float(np.mean([sum(items_by_run[r][u][1].values()) for r in runs]))})
            for u in ref_items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--extra_runs", nargs="*", default=[],
                    help="extra result prefixes (e.g. p2_rule_r0) to append to all tables "
                         "and run the §4.7 pre-registered repeat-category check on")
    ap.add_argument("--out_dir", default=None, help="optionally dump the raw tables as CSV")
    args = ap.parse_args()
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", "{:.3f}".format)
    runs = BASE_RUNS + [r for r in args.extra_runs if r not in BASE_RUNS]

    print("#### sanity: official aggregate == unweighted per-item mean ####")
    for r in [FP32, "int8"] + args.extra_runs:
        try:
            agg, m = official_and_item_mean(args.results_dir, r, "hard_dev")
            print(f"  {r}: official={agg:.3f}  item-mean={m:.3f}")
        except FileNotFoundError:
            pass

    H, Hi = collect(args.results_dir, "hard_dev", runs)
    Z, Zi = collect(args.results_dir, "zh_dev", runs)
    rand = [r for r in RAND if r in H.index]

    print("\n#### T1: type rates, hard_dev (% of ref chars) ####")
    t1 = pd.DataFrame({
        "fp32": H.loc[FP32], "int8": H.loc["int8"],
        "rand_mean": H.loc[rand].mean(), "rand_std": H.loc[rand].std(),
        "n_rand>fp32": (H.loc[rand] > H.loc[FP32]).sum().astype(int),
        "s8_best": H.loc["p0_rand32_s8"], "s5_worst": H.loc["p0_rand32_s5"],
        "hard16": H.loc["p1c_hard16"],
        **{r: H.loc[r] for r in args.extra_runs if r in H.index}})
    print(t1)

    print("\n#### T2: type rate vs total CER / strict CER, spearman across 14 W4A4 runs ####")
    w = H.loc[[r for r in W4A4 if r in H.index]]
    print(pd.DataFrame({"rho_vs_CER": w[TYPES].corrwith(w["CER"], method="spearman"),
                        "rho_vs_strict": w[TYPES].corrwith(w["CER_strict"], method="spearman")}
                       ).round(2))

    cats = {uid: category(uid, norm(ref) or ref)
            for uid, ref, hyp in load(args.results_dir, FP32, "hard_dev")}
    by_cat = {c: [u for u, cc in cats.items() if cc == c] for c in ("twister", "repeat", "plain")}

    print("\n#### T3: corpus CER by item category ####")
    t3 = pd.DataFrame({r: {c: cat_cer(Hi[r], uids) for c, uids in by_cat.items()}
                       for r in runs if r in Hi}).T
    print(pd.DataFrame({
        "n_items": {c: len(u) for c, u in by_cat.items()},
        "fp32": t3.loc[FP32], "int8": t3.loc["int8"],
        "rand_mean": t3.loc[rand].mean(), "rand_std": t3.loc[rand].std(),
        "rand_min": t3.loc[rand].min(), "rand_max": t3.loc[rand].max(),
        "hard16": t3.loc["p1c_hard16"],
        **{r: t3.loc[r] for r in args.extra_runs if r in t3.index}}))

    print("\n#### T3b: paired Δ on categories (bootstrap over items) ####")
    pooled = pooled_items(Hi, rand, Hi[FP32])
    for name, a, b in [("rand-pooled − fp32", Hi[FP32], pooled),
                       ("hard16 − hard0", Hi["p1c_hard0"], Hi["p1c_hard16"]),
                       ("s8 − s5 (best−worst)", Hi["p0_rand32_s5"], Hi["p0_rand32_s8"]),
                       ("int8 − fp32", Hi[FP32], Hi["int8"])]:
        row = "   ".join(f"{c}: Δ={p:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}"
                         for c, uids in by_cat.items()
                         for p, lo, hi, sig in [boot_delta(a, b, uids)])
        print(f"  {name:22s} {row}")

    print("\n#### T4: zh_dev type rates (improvement + language-damage axes) ####")
    print(pd.DataFrame({
        "fp32": Z.loc[FP32], "int8": Z.loc["int8"],
        "rand_mean": Z.loc[rand].mean(), "rand_std": Z.loc[rand].std(),
        "n_rand<fp32": (Z.loc[rand] < Z.loc[FP32]).sum().astype(int),
        "en0": Z.loc["p1c_en0"], "en12": Z.loc["p1c_en12"],
        **{r: Z.loc[r] for r in args.extra_runs if r in Z.index}}))

    if args.extra_runs:
        print("\n#### PRE-REGISTERED CHECK (§4.7): repeat-category CER vs rand distribution ####")
        rep = by_cat["repeat"]
        rmean, rmin, rmax = t3.loc[rand, "repeat"].agg(["mean", "min", "max"])
        print(f"  reference: rand mean {rmean:.3f}  min {rmin:.3f}  max {rmax:.3f}  "
              f"hard16 {t3.loc['p1c_hard16', 'repeat']:.3f}  (prediction: extra < mean, "
              f"twister NOT expected to move)")
        for r in args.extra_runs:
            if r not in Hi:
                continue
            p, lo, hi, sig = boot_delta(Hi[FP32], Hi[r], rep)
            pct = (t3.loc[rand, "repeat"] < t3.loc[r, "repeat"]).mean() * 100
            print(f"  {r}: repeat CER {t3.loc[r, 'repeat']:.3f} "
                  f"(beats {100-pct:.0f}% of rand draws)  Δvs fp32 {p:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}"
                  f"   twister {t3.loc[r, 'twister']:.3f} (rand mean {t3.loc[rand, 'twister'].mean():.3f})")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for tag, df in [("hard_types", H), ("hard_categories", t3), ("zh_types", Z)]:
            df.to_csv(f"{args.out_dir}/taxonomy_{tag}.csv")
        print(f"\nCSV -> {args.out_dir}/taxonomy_*.csv")


if __name__ == "__main__":
    main()
