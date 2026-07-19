"""Build the CLEAN calibration candidate pool (docs/experiments.md §4.3, roadmap E0).

Sources (downloaded by scripts/download_calib_corpora.sh, L2 speaker-disjoint from Seed-TTS):
  zh prompts + zh normal target texts : AISHELL-3 (data/calib_corpora/aishell3, 218 spk, char+pinyin transcripts)
  en prompts + en target texts        : LibriTTS dev-clean (data/calib_corpora/LibriTTS/dev-clean)
  zh hardlike target texts            : deterministic confusable-syllable generator (this file)

Design (measured 2026-07-18, see docs/experiments.md §4.3.1):
  prompt duration target = Seed-TTS test prompts (zh p10/p50/p90 = 4.1/4.5/5.5s, en 3.4/4.5/6.0s)
    -> acceptance windows AFTER librosa trim(top_db=35): zh [4.0, 9.0]s, en [3.0, 9.0]s.
    Acceptance is measured on the TRIMMED clip (raw-duration preselection alone loses most
    candidates: AISHELL-3 edge silence routinely shaves ~1s).
  target text lengths: zh-normal 15-35 chars (test p10-p90 18-28), en 40-90 chars,
    zh-hardlike 25-90 chars (hard set p50=51, long tail)
  cross-pair protocol (mirrors the test sets): each item's target text is from a DIFFERENT
    speaker than its prompt (per-item exclusion); every prompt utterance backs at most one
    normal item; target texts unique within the pool.

Outputs (default data/calib_pool/):
  pool_v1.lst        uid|prompt_text|wavs/<utt>.wav|target_text  (paths relative to the lst dir)
  pool_v1_meta.csv   per-item task-side feature matrix (speaker/gender/accent, durations, text
                     difficulty features) — this is the P1 regression feature table
  pool_v1_build.json build args + composition for reproducibility
  wavs/              trimmed 24 kHz mono prompt clips

Subset sampler for GATE-B (writes wav paths relative to the OUTPUT lst's dir):
  python -m audio_dit_quantize.calib.pool sample --pool data/calib_pool/pool_v1.lst \
      --n 32 --seed 7 --out data/calib_pool/sets/rand32_s7.lst

Run the overlap audit (calib/audit.py) after every build; a pool is usable only if it passes.
"""
import argparse, csv, json, os, re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..paths import REPO_ROOT

CORPORA = REPO_ROOT / "data" / "calib_corpora"
POOL_DIR = REPO_ROOT / "data" / "calib_pool"

# pinyin initials, longest first so zh/ch/sh match before z/c/s
_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k",
             "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]
_PY_RE = re.compile(r"^[a-z]+[1-5]$")
_CJK_RE = re.compile(r"[一-鿿]")


def _split_syllable(py):
    """'zhuang1' -> ('zh','uang'); tone digit dropped. Returns (initial, final)."""
    base = py[:-1] if py[-1].isdigit() else py
    for ini in _INITIALS:
        if base.startswith(ini) and len(base) > len(ini):
            return ini, base[len(ini):]
    return "", base          # zero-initial syllable (a/o/e/ai/er...)


# ── AISHELL-3 parsing ─────────────────────────────────────────────────────────
def parse_aishell3():
    """Returns (utts, spk_info).
    utts: {utt_id: {"spk","wav","text","pys"}} — text = chars joined, pys = pinyin list.
    """
    utts, spk_info = {}, {}
    info_path = CORPORA / "aishell3" / "spk-info.txt"
    if info_path.exists():
        for line in open(info_path, encoding="utf-8", errors="ignore"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) >= 4:
                spk_info[f[0]] = {"age": f[1], "gender": f[2], "accent": f[3]}
    for split in ("train", "test"):
        content = CORPORA / "aishell3" / split / "content.txt"
        if not content.exists():
            continue
        for line in open(content, encoding="utf-8"):
            toks = line.split()
            if len(toks) < 3 or not toks[0].endswith(".wav"):
                continue
            fid = toks[0][:-4]
            spk = fid[:7]
            chars, pys = [], []
            for t in toks[1:]:
                if _PY_RE.match(t):
                    pys.append(t)
                elif _CJK_RE.search(t):
                    chars.append(t)
            # strict 1:1 char<->pinyin alignment (a rare token carries >1 CJK char; drop those
            # lines rather than guessing — downstream feature/generator code indexes pairwise)
            if not chars or any(len(c) != 1 for c in chars) or len(chars) != len(pys):
                continue
            wav = CORPORA / "aishell3" / split / "wav" / spk / f"{fid}.wav"
            if wav.exists():
                utts[fid] = {"spk": spk, "wav": wav, "text": "".join(chars), "pys": pys}
    return utts, spk_info


# ── LibriTTS parsing ──────────────────────────────────────────────────────────
def parse_libritts():
    """{utt_id: {"spk","wav","text"}} from dev-clean *.normalized.txt."""
    utts = {}
    root = CORPORA / "LibriTTS" / "dev-clean"
    for txt in root.glob("*/*/*.normalized.txt"):
        wav = txt.with_name(txt.name.replace(".normalized.txt", ".wav"))
        if not wav.exists():
            continue
        text = " ".join(txt.read_text(encoding="utf-8", errors="ignore").split())
        if text:
            utts[wav.stem] = {"spk": txt.parts[-3], "wav": wav, "text": text}
    return utts


# ── text feature extraction (P1 task-side features) ───────────────────────────
def zh_text_features(pys):
    base = [p[:-1] if p[-1].isdigit() else p for p in pys]
    inis, fins = zip(*(_split_syllable(p) for p in pys)) if pys else ((), ())
    cnt = defaultdict(int)
    for b in base:
        cnt[b] += 1
    return {
        "n_syll": len(pys),
        "n_uniq_syll": len(set(pys)),
        "n_uniq_syll_base": len(set(base)),
        "max_syll_repeat": max(cnt.values()) if cnt else 0,
        "n_uniq_initials": len(set(i for i in inis if i)),
        "n_uniq_finals": len(set(fins)),
    }


_EMPTY_ZH_FEATS = {k: "" for k in
                   ("n_syll", "n_uniq_syll", "n_uniq_syll_base", "max_syll_repeat",
                    "n_uniq_initials", "n_uniq_finals")}


# ── hardlike generator (deterministic, confusable-syllable tongue twisters) ───
_CONF_INITIALS = [("z", "zh"), ("c", "ch"), ("s", "sh"), ("n", "l"), ("l", "r"), ("f", "h")]
_CONF_FINALS = [("in", "ing"), ("an", "ang"), ("en", "eng"), ("ian", "iang"), ("uan", "uang")]
_CONNECT = ["，", "、", "的", "了", "不", "是", "又", "再"]


def build_char_inventory(a3_utts, min_freq=5):
    """char -> most common base syllable, from AISHELL-3 transcripts (freq-filtered)."""
    freq = defaultdict(lambda: defaultdict(int))
    for u in a3_utts.values():
        for ch, py in zip(u["text"], u["pys"]):
            base = py[:-1] if py[-1].isdigit() else py
            freq[ch][base] += 1
    inv = {}
    for ch, m in freq.items():
        base, n = max(m.items(), key=lambda kv: kv[1])
        if n >= min_freq:
            inv[ch] = base
    return inv


def _conf_variants(s):
    """Confusable variants of a base syllable (initial and final swaps)."""
    ini, fin = _split_syllable(s + "1")
    out = set()
    for a, b in _CONF_INITIALS:
        if ini == a:
            out.add(b + fin)
        elif ini == b:
            out.add(a + fin)
    for a, b in _CONF_FINALS:
        if fin == a and ini:
            out.add(ini + b)
        elif fin == b and ini:
            out.add(ini + a)
    return out


_FUNC_CHARS = set("的了是不在就都把被地得着过也与及等和之其")


def _harvest_words(a3_utts, min_freq=10):
    """Frequent 2-char sequences from transcripts (crude word proxy) -> (syl1, syl2) base.
    Bigrams containing function chars are dropped — they are mostly cross-word-boundary junk
    ('的七') and break the twister templates ('的'+'的七' -> '的的')."""
    cnt, syl = defaultdict(int), {}
    for u in a3_utts.values():
        t, p = u["text"], u["pys"]
        for i in range(len(t) - 1):
            w = t[i:i + 2]
            if w[0] in _FUNC_CHARS or w[1] in _FUNC_CHARS:
                continue
            cnt[w] += 1
            if w not in syl:
                b = lambda x: x[:-1] if x[-1].isdigit() else x
                syl[w] = (b(p[i]), b(p[i + 1]))
    return {w: syl[w] for w, c in cnt.items() if c >= min_freq}


_TWIST_VERBS = ["拿", "搬", "数", "买", "卖", "换", "抬", "挑", "拦", "捞", "补", "扛"]
_SURNAMES = "张王李赵刘孙周吴郑陈冯蒋沈韩杨朱秦"


def gen_hardlike(a3_utts, inv, rng, n_items):
    """Generate hardlike target texts matching the REAL hardcase taxonomy (measured 2026-07-18,
    docs/experiments.md §4.3.1): word-level tongue twisters (A), phrase/sentence repetition (B),
    long concatenations (C) — NOT char-run soup (real hardcase max_char_run p50 = 1).
    Returns [(text, pys_base, substyle)]; pys are toneless for A, real transcript pinyin for B/C."""
    words = _harvest_words(a3_utts)
    by_sylpair = defaultdict(list)
    for w, (s1, s2) in words.items():
        by_sylpair[(s1, s2)].append(w)
    # confusable word pairs: same syllables (homophone-ish) or ONE syllable swapped confusably
    wpairs = []
    for w, (s1, s2) in sorted(words.items()):
        for cand in by_sylpair[(s1, s2)]:
            if cand > w:
                wpairs.append((w, cand))
        for v1 in sorted(_conf_variants(s1)):
            wpairs += [(w, c) for c in by_sylpair.get((v1, s2), ())]
        for v2 in sorted(_conf_variants(s2)):
            wpairs += [(w, c) for c in by_sylpair.get((s1, v2), ())]
    sents = sorted(u["text"] for u in a3_utts.values() if 6 <= len(u["text"]) <= 16)
    pys_of = {u["text"]: u["pys"] for u in a3_utts.values()}

    def syls(text):          # toneless syllables via char inventory (best effort)
        return [inv[c] for c in text if c in inv]

    def style_a():           # word-level confusable twister, syntactic templates
        wA, wB = wpairs[rng.integers(0, len(wpairs))]
        v = _TWIST_VERBS[rng.integers(0, len(_TWIST_VERBS))]
        s1 = "老" + _SURNAMES[rng.integers(0, len(_SURNAMES))]
        s2 = "老" + _SURNAMES[rng.integers(0, len(_SURNAMES))]
        t = rng.integers(0, 3)
        if t == 0:
            text = f"{s1}{v}{wA}，{s2}{v}{wB}，{s1}的{wA}{v}不过{s2}的{wB}。"
        elif t == 1:
            text = f"{wA}{v}{wB}，{wB}{v}{wA}，{wA}{v}得了{wB}，{wB}{v}不了{wA}。"
        else:
            text = (f"{s1}拿{wA}换{s2}的{wB}，{s2}拿{wB}换{s1}的{wA}，"
                    f"换来换去{wA}还是{wA}，{wB}还是{wB}。")
        return text, syls(text)

    def style_b():           # phrase/sentence repetition (dominant real-hardcase pattern)
        base = sents[rng.integers(0, len(sents))]
        n = int(rng.integers(4, 7))   # real hardcase phrase_rep p50 = 5
        if rng.random() < 0.5:
            text = (base + "。") * n
            pys = pys_of.get(base, []) * n
        else:
            k = int(rng.integers(4, min(10, len(base)) + 1))
            frag, tail = base[:k], sents[rng.integers(0, len(sents))]
            text = (frag + "，") * n + tail + "。"
            pys = pys_of.get(base, [])[:k] * n + pys_of.get(tail, [])
        return text, pys

    def style_c():           # long multi-clause concatenation (high-uniqueness tail)
        parts, pys, total = [], [], 0
        while total < int(rng.integers(60, 121)):
            s = sents[rng.integers(0, len(sents))]
            parts.append(s); pys += pys_of.get(s, []); total += len(s)
        return "，".join(parts) + "。", pys

    n_a = n_items // 3
    n_c = n_items // 4
    n_b = n_items - n_a - n_c
    out, used = [], set()
    for gen, sub, quota in ((style_a, "twister", n_a), (style_b, "repeat", n_b),
                            (style_c, "concat", n_c)):
        got = 0
        while got < quota:
            text, pys = gen()
            if text in used:
                continue
            used.add(text)
            out.append((text, pys, sub)); got += 1
    return out


# ── audio helpers ─────────────────────────────────────────────────────────────
def trim_and_write(src, dst, sr=24000, top_db=35):
    """Load -> mono 24k -> trim edge silence -> write. Returns (dur_s, rms_db)."""
    import librosa, soundfile as sf
    y, _ = librosa.load(str(src), sr=sr, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < sr:                                  # degenerate trim -> keep original
        yt = y
    rms = float(np.sqrt(np.mean(yt ** 2)) + 1e-12)
    sf.write(str(dst), yt, sr)
    return len(yt) / sr, 20 * np.log10(rms)


def _speaker_order(spks, spk_info, rng):
    """Gender round-robin over shuffled per-gender speaker lists (balance without quotas)."""
    groups = defaultdict(list)
    for spk in sorted(spks):
        groups[spk_info.get(spk, {}).get("gender", "unknown")].append(spk)
    for g in groups:
        rng.shuffle(groups[g])
    order, gs = [], sorted(groups)
    while any(groups[g] for g in gs):
        for g in gs:
            if groups[g]:
                order.append(groups[g].pop())
    return order


def select_prompts(utts, durs, spk_info, rng, quota, per_spk, lo, hi, target, wavdir,
                   max_probe_per_spk=6):
    """Accept prompts by TRIMMED duration: walk speakers in stratified order; per speaker probe
    up to `max_probe_per_spk` candidates (raw duration nearest target+0.8s expected-trim offset),
    trim+write each, keep those landing in [lo, hi] until `per_spk` accepted; continue through
    the speaker list until `quota` prompts are accepted.  Returns {utt_id: (wname, dur, rms)}."""
    by_spk = defaultdict(list)
    for uid, u in utts.items():
        d = durs.get(uid)
        if d is not None and lo <= d <= hi + 2.5:     # generous raw prefilter; trim only shortens
            by_spk[u["spk"]].append((abs(d - (target + 0.8)), uid))
    accepted, probed = {}, 0
    for spk in _speaker_order(by_spk.keys(), spk_info, rng):
        got = 0
        for _, uid in sorted(by_spk[spk])[:max_probe_per_spk]:
            if got >= per_spk or len(accepted) >= quota:
                break
            wname = f"{uid}.wav"
            dur, rms = trim_and_write(utts[uid]["wav"], wavdir / wname)
            probed += 1
            if lo <= dur <= hi:
                accepted[uid] = (wname, dur, rms); got += 1
            else:
                (wavdir / wname).unlink(missing_ok=True)
        if len(accepted) >= quota:
            break
    print(f"[pool]   accepted {len(accepted)}/{quota} prompts "
          f"({probed} probed, {len(by_spk)} speakers available)")
    return accepted


def _pair_texts(texts, prompt_spk, prompt_text, used):
    """First unused text whose speaker differs from the prompt's and text differs from the
    prompt transcript. Returns index or None."""
    for i, (text, _pys, _src, spk) in enumerate(texts):
        if i in used or spk == prompt_spk or text == prompt_text:
            continue
        return i
    return None


# ── build ─────────────────────────────────────────────────────────────────────
def build(args):
    import soundfile as sf
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.out_dir); wavdir = outdir / "wavs"
    wavdir.mkdir(parents=True, exist_ok=True)

    print("[pool] parsing AISHELL-3 ...")
    a3, spk_info = parse_aishell3()
    print(f"[pool]   {len(a3)} utts, {len(set(u['spk'] for u in a3.values()))} speakers")
    print("[pool] parsing LibriTTS dev-clean ...")
    lt = parse_libritts()
    print(f"[pool]   {len(lt)} utts, {len(set(u['spk'] for u in lt.values()))} speakers")

    print("[pool] scanning raw durations (header reads) ...")
    def scan(utts):
        out = {}
        for uid, u in utts.items():
            try:
                i = sf.info(str(u["wav"])); out[uid] = i.frames / i.samplerate
            except Exception:
                pass
        return out
    a3_durs, lt_durs = scan(a3), scan(lt)

    print("[pool] selecting zh prompts (trim-validated) ...")
    zh_quota = args.zh_speakers * args.zh_per_spk
    zh_acc = select_prompts(a3, a3_durs, spk_info, rng, zh_quota, args.zh_per_spk,
                            lo=4.0, hi=9.0, target=4.5, wavdir=wavdir)
    print("[pool] selecting en prompts (trim-validated) ...")
    en_quota = args.en_speakers * args.en_per_spk
    en_acc = select_prompts(lt, lt_durs, {}, rng, en_quota, args.en_per_spk,
                            lo=3.0, hi=9.0, target=4.5, wavdir=wavdir)

    # ── target text pools: (text, pys, src_utt, spk); per-item speaker exclusion at pairing ──
    zh_texts, seen = [], set()
    for uid, u in a3.items():
        if not (args.zh_text_lo <= len(u["text"]) <= args.zh_text_hi) or u["text"] in seen:
            continue
        seen.add(u["text"])
        zh_texts.append((u["text"] + "。", u["pys"], uid, u["spk"]))
    rng.shuffle(zh_texts)
    en_texts, seen = [], set()
    for uid, u in lt.items():
        if not (args.en_text_lo <= len(u["text"]) <= args.en_text_hi) or u["text"] in seen:
            continue
        seen.add(u["text"])
        en_texts.append((u["text"], None, uid, u["spk"]))
    rng.shuffle(en_texts)

    print(f"[pool] generating {args.zh_hard} hardlike texts ...")
    inv = build_char_inventory(a3)
    hard_texts = gen_hardlike(a3, inv, rng, args.zh_hard)

    # ── assemble rows (pairing with per-item cross-speaker exclusion) ──
    rows, dropped = [], defaultdict(int)

    def add_item(uid, style, putt, corpus, acc, text, pys, tsrc, substyle=""):
        src_u = a3[putt] if corpus == "aishell3" else lt[putt]
        wname, dur, rms = acc[putt]
        ptext = src_u["text"] + ("。" if corpus == "aishell3" else "")
        info = spk_info.get(src_u["spk"], {})
        feats = zh_text_features(pys) if pys else dict(_EMPTY_ZH_FEATS)
        rows.append({
            "uid": uid, "lang": style.split("_")[0], "style": style, "hard_substyle": substyle,
            "prompt_wav": f"wavs/{wname}", "prompt_text": ptext,
            "spk": src_u["spk"], "gender": info.get("gender", ""),
            "age": info.get("age", ""), "accent": info.get("accent", ""),
            "prompt_src_utt": putt, "prompt_dur_s": f"{dur:.2f}", "prompt_rms_db": f"{rms:.1f}",
            "n_chars_prompt": len(src_u["text"]),
            "target_text": text, "target_src": tsrc, "n_chars_target": len(text),
            "uniq_char_ratio": f"{len(set(text)) / max(1, len(text)):.3f}",
            **{k: str(v) for k, v in feats.items()},
        })

    used_zh_texts = set()
    for i, putt in enumerate(zh_acc):
        j = _pair_texts(zh_texts, a3[putt]["spk"], a3[putt]["text"] + "。", used_zh_texts)
        if j is None:
            dropped["zh_no_text"] += 1
            continue
        used_zh_texts.add(j)
        text, pys, tsrc, _ = zh_texts[j]
        add_item(f"zhn_{i:04d}", "zh_normal", putt, "aishell3", zh_acc, text, pys, tsrc)

    hard_prompts = list(zh_acc); rng.shuffle(hard_prompts)
    for i, (text, pys, sub) in enumerate(hard_texts):
        if not hard_prompts:
            dropped["hard_no_prompt"] += 1
            continue
        putt = hard_prompts[i % len(hard_prompts)]
        add_item(f"zhh_{i:04d}", "zh_hardlike", putt, "aishell3", zh_acc, text, pys,
                 "generator", substyle=sub)

    used_en_texts = set()
    for i, putt in enumerate(en_acc):
        j = _pair_texts(en_texts, lt[putt]["spk"], lt[putt]["text"], used_en_texts)
        if j is None:
            dropped["en_no_text"] += 1
            continue
        used_en_texts.add(j)
        text, _, tsrc, _ = en_texts[j]
        add_item(f"enn_{i:04d}", "en_normal", putt, "libritts", en_acc, text, None, tsrc)

    # prune wavs that ended up unused (accepted prompt whose pairing failed)
    used_wavs = {r["prompt_wav"].split("/")[-1] for r in rows}
    for acc in (zh_acc, en_acc):
        for wname, _, _ in acc.values():
            if wname not in used_wavs:
                (wavdir / wname).unlink(missing_ok=True)

    if dropped:
        print(f"[pool] WARN dropped items: {dict(dropped)}")

    # ── outputs ──
    tag = args.tag
    lst_path = outdir / f"{tag}.lst"
    with open(lst_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['uid']}|{r['prompt_text']}|{r['prompt_wav']}|{r['target_text']}\n")
    meta_path = outdir / f"{tag}_meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    comp = defaultdict(int)
    for r in rows:
        comp[r["style"]] += 1
    build_info = {"args": vars(args), "composition": dict(comp),
                  "n_items": len(rows), "n_speakers": len({r["spk"] for r in rows}),
                  "dropped": dict(dropped),
                  "sources": {"zh": "AISHELL-3 (openslr93)", "en": "LibriTTS dev-clean (openslr60)"}}
    (outdir / f"{tag}_build.json").write_text(json.dumps(build_info, indent=2, ensure_ascii=False))
    print(f"[pool] {len(rows)} items -> {lst_path}")
    print(f"[pool] composition: {dict(comp)} | speakers: {build_info['n_speakers']}")
    print(f"[pool] meta -> {meta_path}")
    print("[pool] NOW RUN: python -m audio_dit_quantize.calib.audit --lst", lst_path)


# ── single-factor contrast pair builder (P1 hypothesis confirmation, §3.3-3) ──
_FACTORS = {
    # factor: (base-pool uid prefixes, swap-in uid prefix, pair name stem)
    "en":       (("zhn_", "zhh_"), "enn_", "en"),
    "hardlike": (("zhn_",),        "zhh_", "hard"),
}


def contrast(args):
    """Build a PAIR of calibration lists that differ in exactly ONE composition factor:
    set A = `n` items from the base pool (factor absent); set B = the same A minus `swap`
    random items, plus `swap` items of the factor style. The (n−swap) shared items make the
    comparison a designed single-factor contrast (max power at 2 jobs, docs §3.3-3)."""
    base_pre, swap_pre, stem = _FACTORS[args.factor]
    pool = Path(args.pool)
    lines = [l.rstrip("\n") for l in open(pool, encoding="utf-8") if l.strip()]
    order = {l.split("|")[0]: i for i, l in enumerate(lines)}
    by_uid = {l.split("|")[0]: l for l in lines}
    base = sorted([u for u in by_uid if u.startswith(base_pre)], key=order.get)
    swap_in = sorted([u for u in by_uid if u.startswith(swap_pre)], key=order.get)
    rng = np.random.default_rng(args.seed)
    A = sorted(rng.choice(base, size=args.n, replace=False), key=order.get)
    keep = sorted(rng.choice(A, size=args.n - args.swap, replace=False), key=order.get)
    B = sorted(list(keep) + list(rng.choice(swap_in, size=args.swap, replace=False)), key=order.get)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for tag, sel in ((f"ctr_{stem}0_s{args.seed}", A), (f"ctr_{stem}{args.swap}_s{args.seed}", B)):
        out = out_dir / f"{tag}.lst"
        with open(out, "w", encoding="utf-8") as f:
            for u in sel:
                p = by_uid[u].split("|")
                wav_abs = (pool.parent / p[2]).resolve()
                p[2] = os.path.relpath(wav_abs, out_dir.resolve())
                f.write("|".join(p) + "\n")
        comp = defaultdict(int)
        for u in sel:
            comp[u.split("_")[0]] += 1
        print(f"[contrast] {out.name}: {dict(comp)}")
    print(f"[contrast] shared items: {len(keep)}/{args.n} (single factor = {args.swap} x {args.factor})")


# ── subset sampler (GATE-B random draws) ──────────────────────────────────────
def sample(args):
    pool = Path(args.pool)
    lines = [l.rstrip("\n") for l in open(pool, encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(lines), size=args.n, replace=False)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in sorted(idx):
            p = lines[i].split("|")
            # rewrite wav path relative to the OUTPUT lst dir (load_items resolves against it)
            wav_abs = (pool.parent / p[2]).resolve()
            p[2] = os.path.relpath(wav_abs, out.parent.resolve())
            f.write("|".join(p) + "\n")
    print(f"[sample] {args.n} items (seed {args.seed}) {pool.name} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the candidate pool")
    b.add_argument("--out_dir", default=str(POOL_DIR))
    b.add_argument("--tag", default="pool_v1")
    b.add_argument("--seed", type=int, default=20260718)
    b.add_argument("--zh_speakers", type=int, default=120)
    b.add_argument("--zh_per_spk", type=int, default=2)
    b.add_argument("--zh_hard", type=int, default=60)
    b.add_argument("--en_speakers", type=int, default=40, help="dev-clean has 40 speakers")
    b.add_argument("--en_per_spk", type=int, default=2)
    b.add_argument("--zh_text_lo", type=int, default=15)
    b.add_argument("--zh_text_hi", type=int, default=35)
    b.add_argument("--en_text_lo", type=int, default=40)
    b.add_argument("--en_text_hi", type=int, default=90)
    s = sub.add_parser("sample", help="draw a random calibration subset from a pool lst")
    s.add_argument("--pool", required=True)
    s.add_argument("--n", type=int, default=32)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("contrast", help="build a single-factor contrast PAIR of calibration lists")
    c.add_argument("--pool", required=True)
    c.add_argument("--factor", required=True, choices=sorted(_FACTORS))
    c.add_argument("--n", type=int, default=32)
    c.add_argument("--swap", type=int, required=True, help="how many base items are replaced by factor items in set B")
    c.add_argument("--seed", type=int, default=1000)
    c.add_argument("--out_dir", default=None, help="default: <pool_dir>/sets")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "sample":
        sample(args)
    else:
        if args.out_dir is None:
            args.out_dir = str(Path(args.pool).parent / "sets")
        contrast(args)


if __name__ == "__main__":
    main()
