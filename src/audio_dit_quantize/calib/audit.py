"""Contamination audit for a calibration list vs the Seed-TTS test sets (docs/experiments.md §4.3).

Checks (L-levels per docs/experiments.md — a pool must pass ALL to be usable):
  A. no wav path resolves inside data/seedtts_testset (audio reuse, incl. legacy-style lists)
  B. no prompt-wav basename collides with a test prompt basename
  C. no calib prompt/target TEXT equals a test prompt/target text (normalized exact match)
  D. no containment either way for normalized texts >= 10 chars (near-dup guard)
  E. intra-list hygiene: unique uids, unique target texts, wav files exist, 4 non-empty fields
     (duplicate prompt WAVs are reported as info only — hardlike items may share prompts)

Exit code 0 = clean, 1 = violations found (prints each).

Usage:
  python -m audio_dit_quantize.calib.audit --lst data/calib_pool/pool_v1.lst
"""
import argparse, os, re, sys
from pathlib import Path

from ..paths import DATA_DIR

_NORM_RE = re.compile(r"[^0-9a-z一-鿿]+")


def norm(t):
    return _NORM_RE.sub("", t.lower())


def load_test_refs():
    """(prompt_basenames, norm prompt texts, norm target texts) across zh/en/hard."""
    basenames, ptexts, ttexts = set(), set(), set()
    for rel in ("zh/meta.lst", "zh/hardcase.lst", "en/meta.lst"):
        p = Path(DATA_DIR) / rel
        if not p.exists():
            print(f"[audit] WARN missing test list {p} — its items are NOT checked")
            continue
        for line in open(p, encoding="utf-8"):
            f = line.rstrip("\n").split("|")
            if len(f) < 4:
                continue
            basenames.add(os.path.basename(f[2]))
            ptexts.add(norm(f[1]))
            ttexts.add(norm(f[3]))
    return basenames, ptexts, ttexts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lst", required=True)
    args = ap.parse_args()
    lst = Path(args.lst).resolve()
    lst_dir = lst.parent
    test_root = Path(DATA_DIR).resolve()
    tb, tp, tt = load_test_refs()
    test_texts = tp | tt

    bad, info = [], []
    uids, ttexts_seen, wavs_seen = set(), set(), {}
    n = 0
    for ln, line in enumerate(open(lst, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        n += 1
        f = line.split("|")
        if len(f) < 4 or any(not x.strip() for x in f[:4]):
            bad.append(f"L{ln}: malformed/empty field: {line[:80]}")
            continue
        uid, ptext, wav, ttext = f[0], f[1], f[2], f[3]
        if uid in uids:
            bad.append(f"L{ln}: duplicate uid {uid}")
        uids.add(uid)
        wav_abs = (lst_dir / wav).resolve() if not os.path.isabs(wav) else Path(wav)
        if not wav_abs.exists():
            bad.append(f"L{ln}: missing wav {wav_abs}")
        if test_root in wav_abs.parents:
            bad.append(f"L{ln}: [A] wav inside seedtts_testset: {wav}")
        if os.path.basename(wav) in tb:
            bad.append(f"L{ln}: [B] prompt basename collides with a test prompt: {os.path.basename(wav)}")
        for label, txt in (("prompt", ptext), ("target", ttext)):
            nt = norm(txt)
            if nt in test_texts:
                bad.append(f"L{ln}: [C] {label} text equals a test text: {txt[:40]}")
            elif len(nt) >= 10:
                for ref in test_texts:
                    if len(ref) >= 10 and (nt in ref or ref in nt):
                        bad.append(f"L{ln}: [D] {label} text contains/contained-by a test text: {txt[:40]}")
                        break
        nt = norm(ttext)
        if nt in ttexts_seen:
            bad.append(f"L{ln}: duplicate target text: {ttext[:40]}")
        ttexts_seen.add(nt)
        wavs_seen.setdefault(wav, []).append(uid)

    shared = {w: u for w, u in wavs_seen.items() if len(u) > 1}
    if shared:
        info.append(f"{len(shared)} prompt wavs shared by multiple items (expected for hardlike): "
                    + ", ".join(f"{w}x{len(u)}" for w, u in list(shared.items())[:5]) + " ...")

    print(f"[audit] {lst.name}: {n} items | test refs: {len(tb)} prompt wavs, {len(test_texts)} texts")
    for m in info:
        print(f"[audit] info: {m}")
    if bad:
        print(f"[audit] FAIL — {len(bad)} violation(s):")
        for m in bad[:50]:
            print("  " + m)
        if len(bad) > 50:
            print(f"  ... and {len(bad)-50} more")
        sys.exit(1)
    print("[audit] PASS — no test-set contamination detected")


if __name__ == "__main__":
    main()
