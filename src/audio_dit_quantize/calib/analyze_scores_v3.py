"""E1 offline analysis: set-level scores vs known dual-scale set rankings (pre-registered §1.6)."""
import csv, itertools, os
import numpy as np

AQ = "/data/xiaoqinfeng/code/audio-dit-quantize"
SCORES = {"1b": f"{AQ}/data/calib_pool/scores_v3/1b/pool_scores.csv",
          "3p5b": f"{AQ}/data/calib_pool/scores_v3/3p5b/pool_scores.csv"}
SETS_DIR = f"{AQ}/data/calib_pool/sets"

# §1.1 per-set metrics: zh CER, en WER, hard CER (dev, vs-full protocol tables)
MET = {
    "1b": {"s0": (1.255, 7.475, 3.422), "s1": (1.214, 6.875, 3.257), "s2": (1.330, 7.244, 3.074),
           "s3": (1.367, 6.991, 3.366), "s4": (1.157, 6.859, 2.985), "s5": (1.372, 7.489, 3.250),
           "s6": (1.343, 7.426, 2.806), "s7": (1.309, 7.121, 2.806), "s8": (1.335, 6.644, 3.129),
           "s9": (1.168, 6.853, 3.148)},
    "3p5b": {"s0": (1.166, 6.086, 2.346), "s1": (1.267, 6.289, 2.489), "s2": (1.357, 6.282, 2.708),
             "s3": (1.185, 6.473, 2.455), "s4": (1.162, 6.210, 2.602), "s5": (1.473, 6.325, 2.739),
             "s6": (1.358, 6.168, 2.416), "s7": (1.136, 6.174, 2.300), "s8": (1.281, 6.192, 2.615),
             "s9": (1.154, 6.091, 2.396)},
}
DOC_RANK = {"1b": {"s0": 8, "s1": 5, "s2": 6, "s3": 8, "s4": 1, "s5": 10, "s6": 7, "s7": 3, "s8": 4, "s9": 2},
            "3p5b": {"s0": 2, "s1": 6, "s2": 9, "s3": 6, "s4": 4, "s5": 10, "s6": 4, "s7": 1, "s8": 6, "s9": 2}}

SIDS = [f"s{i}" for i in range(10)]


def rankdata(x):
    x = np.asarray(x, float); n = len(x)
    order = np.argsort(x); r = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def perm_p(a, b, n_iter=20000, seed=0):
    obs = abs(spearman(a, b)); rng = np.random.default_rng(seed); c = 0
    for _ in range(n_iter):
        if abs(spearman(a, rng.permutation(b))) >= obs - 1e-12:
            c += 1
    return (c + 1) / (n_iter + 1)


def load_scores(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c.split("_")[0] in ("cdiff", "ffn", "sattn", "xattn", "allmod")]
    return {r["uid"]: {c: float(r[c]) for c in cols} for r in rows}, cols, rows


def set_members(sid):
    with open(f"{SETS_DIR}/rand32_{sid}.lst") as f:
        return [ln.split("|")[0] for ln in f if ln.strip()]


def composite_rank(scale):
    per_axis = [rankdata([MET[scale][s][a] for s in SIDS]) for a in range(3)]
    comp = np.mean(per_axis, axis=0)
    return {s: comp[i] for i, s in enumerate(SIDS)}


for scale in ("1b", "3p5b"):
    scores, cols, rows = load_scores(SCORES[scale])
    bad = [r["uid"] for r in rows if r["wav_ok"] != "1"]
    print(f"\n===== {scale}: {len(rows)} samples, wav_ok fails: {bad or 'none'} =====")

    comp = composite_rank(scale)
    doc = DOC_RANK[scale]
    chk = spearman([comp[s] for s in SIDS], [doc[s] for s in SIDS])
    print(f"recomputed composite vs doc rank agreement rho={chk:.3f}")

    members = {s: set_members(s) for s in SIDS}
    miss = {s: [u for u in m if u not in scores] for s, m in members.items()}
    assert not any(miss.values()), f"missing uids in scores: {miss}"

    print(f"{'score':<18} {'rho_comp':>8} {'p':>7} | {'rho_zh':>7} {'rho_en':>7} {'rho_hard':>8} | set-mean spread(CV%)")
    y_comp = [comp[s] for s in SIDS]
    for c in cols:
        sm = [np.mean([scores[u][c] for u in members[s]]) for s in SIDS]
        rho = spearman(sm, y_comp); p = perm_p(sm, y_comp)
        ax = [spearman(sm, [MET[scale][s][a] for s in SIDS]) for a in range(3)]
        pool_vals = [scores[u][c] for u in scores]
        cv = 100 * np.std(sm) / np.mean(sm)
        pool_cv = 100 * np.std(pool_vals) / np.mean(pool_vals)
        print(f"{c:<18} {rho:>8.2f} {p:>7.3f} | {ax[0]:>7.2f} {ax[1]:>7.2f} {ax[2]:>8.2f} | "
              f"{cv:.1f}% (pool {pool_cv:.0f}%)")

# cross-scale per-sample score stability + pool cdiff curve
s1, cols, _ = load_scores(SCORES["1b"])
s2, _, _ = load_scores(SCORES["3p5b"])
common = sorted(set(s1) & set(s2))
print(f"\n===== cross-scale per-sample score stability (n={len(common)}) =====")
for c in cols:
    a = [s1[u][c] for u in common]; b = [s2[u][c] for u in common]
    print(f"{c:<18} spearman={spearman(a, b):>6.2f}")

print("\n===== pool-level cdiff windows (mean over 376) =====")
for scale, sc in (("1b", s1), ("3p5b", s2)):
    for w in ("early5", "all", "late5"):
        v = [sc[u][f"cdiff_{w}"] for u in sc]
        print(f"{scale} cdiff_{w}: mean={np.mean(v):.4f}")
