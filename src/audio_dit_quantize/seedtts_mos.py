"""No-reference MOS evaluation for generated Seed-TTS wavs.

Run one MOS backend per process. UTMOS uses torch.hub, while DNSMOS uses the
speechmos package; keeping them in separate invocations avoids package conflicts.

Usage:
  python -m audio_dit_quantize.seedtts_mos utmos  <gen_dir> <out_file>
  python -m audio_dit_quantize.seedtts_mos dnsmos <gen_dir> <out_file>
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def load_16khz(path: str) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = np.nan_to_num(wav.astype("float32"))
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return np.clip(wav, -1.0, 1.0)


def list_wavs(gen_dir: str, limit: int) -> list[str]:
    wavs = sorted(glob.glob(os.path.join(gen_dir, "*.wav")))
    if limit:
        wavs = wavs[:limit]
    if not wavs:
        raise FileNotFoundError(f"no wav files found under {gen_dir}")
    return wavs


def score_utmos(wavs: list[str], device: str) -> str:
    import torch

    dev = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("tarepan/SpeechMOS:main", "utmos22_strong", trust_repo=True).to(dev).eval()
    values = []
    for wav_path in tqdm(wavs, desc="utmos", dynamic_ncols=True):
        wav = load_16khz(wav_path)
        with torch.no_grad():
            score = model(torch.from_numpy(wav).unsqueeze(0).to(dev), 16000).item()
        values.append(score)
    return f"UTMOS: {np.mean(values):.3f}\n(n={len(values)})\n"


def score_dnsmos(wavs: list[str]) -> str:
    from speechmos import dnsmos

    ovrl, p808 = [], []
    skipped = 0
    for wav_path in tqdm(wavs, desc="dnsmos", dynamic_ncols=True):
        try:
            scores = dnsmos.run(load_16khz(wav_path), 16000)
            ovrl.append(float(scores["ovrl_mos"]))
            p808.append(float(scores["p808_mos"]))
        except Exception as exc:
            skipped += 1
            tqdm.write(f"[dnsmos] skip {os.path.basename(wav_path)}: {exc}")
    if not ovrl:
        raise RuntimeError("DNSMOS produced no valid scores")
    if skipped:
        print(f"[dnsmos] skipped {skipped} wavs", flush=True)
    return f"DNSMOS_ovrl: {np.mean(ovrl):.3f}\nDNSMOS_p808: {np.mean(p808):.3f}\n(n={len(ovrl)})\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-TTS no-reference MOS evaluation")
    parser.add_argument("metric", choices=["utmos", "dnsmos"])
    parser.add_argument("gen_dir")
    parser.add_argument("out_file")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    wavs = list_wavs(args.gen_dir, args.limit)
    if args.metric == "utmos":
        result = score_utmos(wavs, args.device)
    else:
        result = score_dnsmos(wavs)

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
