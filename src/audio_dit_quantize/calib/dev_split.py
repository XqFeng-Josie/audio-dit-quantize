"""FROZEN dev/heldtest split of the Seed-TTS test sets (docs/experiments.md §4.3, roadmap E0).

Three-tier data governance: the calib CANDIDATE POOL supplies calibration data; the DEV split below
is the only selection/comparison signal during development; the HELDTEST remainder is untouched until
final reporting.

Split (decided 2026-07-18, seed 20260718):  zh 300/2020, hard 100/400, en 200/1088.

Freeze semantics:
  - The committed uid manifests under data/splits/*_dev_uids.txt are the source of truth.
  - If a manifest exists, this script only MATERIALIZES the meta_dev/heldtest .lst files from it
    (needed on fresh machines — seedtts_testset/ is gitignored). It never resamples.
  - First run (no manifests) samples once and writes the manifests; commit them.
  - --force_resample discards existing manifests (protocol change — never do this mid-study).

Materialized files (next to the originals, so relative wav paths keep resolving):
  zh/meta_dev.lst zh/meta_heldtest.lst zh/hardcase_dev.lst zh/hardcase_heldtest.lst
  en/meta_dev.lst en/meta_heldtest.lst

PROTOCOL NOTE: generation seeds each item as base + index-IN-THE-GENERATED-LIST. Dev-list indices
differ from full-list indices, so dev runs pair ONLY with other dev runs (fp32 dev reference
included) — never compare a dev run against an old full-set run per-item.
"""
import argparse

import numpy as np

from ..paths import DATA_DIR, REPO_ROOT

SPLIT_SEED = 20260718
SPLITS_DIR = REPO_ROOT / "data" / "splits"
# name -> (source rel path, dev rel path, heldtest rel path, n_dev)   (fixed order = fixed RNG stream)
SPLITS = {
    "zh":   ("zh/meta.lst",     "zh/meta_dev.lst",     "zh/meta_heldtest.lst",     300),
    "hard": ("zh/hardcase.lst", "zh/hardcase_dev.lst", "zh/hardcase_heldtest.lst", 100),
    "en":   ("en/meta.lst",     "en/meta_dev.lst",     "en/meta_heldtest.lst",     200),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force_resample", action="store_true",
                    help="DANGER: discard the frozen manifests and resample (protocol change)")
    args = ap.parse_args()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SPLIT_SEED)

    for name, (src_rel, dev_rel, held_rel, n_dev) in SPLITS.items():
        src = DATA_DIR / src_rel
        if not src.exists():
            raise FileNotFoundError(f"{src} — download the Seed-TTS test set first")
        lines = [l.rstrip("\n") for l in open(src, encoding="utf-8") if l.strip()]
        uid_of = [l.split("|")[0] for l in lines]
        manifest = SPLITS_DIR / f"{name}_dev_uids.txt"

        if manifest.exists() and not args.force_resample:
            dev_uids = set(manifest.read_text(encoding="utf-8").split())
            missing = dev_uids - set(uid_of)
            if missing:
                raise RuntimeError(f"{manifest}: {len(missing)} manifest uids not in {src} "
                                   f"(test-set files changed?) e.g. {sorted(missing)[:3]}")
            action = "materialized from frozen manifest"
        else:
            idx = rng.choice(len(lines), size=n_dev, replace=False)
            dev_uids = {uid_of[i] for i in idx}
            manifest.write_text("\n".join(sorted(dev_uids)) + "\n", encoding="utf-8")
            action = "SAMPLED (new manifest written — commit data/splits/)"

        dev = [l for l, u in zip(lines, uid_of) if u in dev_uids]
        held = [l for l, u in zip(lines, uid_of) if u not in dev_uids]
        (DATA_DIR / dev_rel).write_text("\n".join(dev) + "\n", encoding="utf-8")
        (DATA_DIR / held_rel).write_text("\n".join(held) + "\n", encoding="utf-8")
        print(f"[split] {name}: dev {len(dev)} / heldtest {len(held)} (of {len(lines)}) — {action}")
    print("[split] done — dev sets usable as zh_dev/hard_dev/en_dev in gen+eval")


if __name__ == "__main__":
    main()
