"""Single-GPU SIM eval for one Seed test set.

Computes WavLM-large speaker cosine similarity between each generated wav and
its prompt wav, replicating seed-tts-eval's verification_pair_list_v2.py but
without the cwd `select.py` stdlib-shadowing and without the multi-GPU split.

Usage: python -m audio_dit_quantize.seedtts_similarity <wav_res_ref_text> <out_score_file> [device]
  wav_res_ref_text lines: gen_wav_path|prompt_wav_path|infer_text

Speed knobs:
  SIM_AUDIO_WORKERS=8       parallel CPU audio load/resample workers
  SIM_BATCH_SIZE=1          exact by default; >1 batches same-length tensors
  SIM_PAD_BATCH=0           set 1 to batch padded variable-length tensors
  SIM_EMB_CACHE_DIR=...     cache exact per-audio embeddings across tags
"""
import sys
import os
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .paths import EVAL_DIR, WAVLM_CKPT

# torchaudio 2.11 removed several APIs that s3prl's hub imports at module load.
# wavlm_large doesn't use them, but `import s3prl.hub` pulls in every upstream,
# so we stub the missing pieces just enough to let the imports succeed.
import types
import torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **k: None
try:
    import torchaudio.sox_effects  # noqa: F401
except ModuleNotFoundError:
    _sox = types.ModuleType("torchaudio.sox_effects")
    _sox.apply_effects_tensor = lambda tensor, sample_rate, effects, *a, **k: (tensor, sample_rate)
    _sox.apply_effects_file = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sox_effects unavailable"))
    sys.modules["torchaudio.sox_effects"] = _sox
    torchaudio.sox_effects = _sox

SV_DIR = os.path.join(str(EVAL_DIR), "thirdparty", "UniSpeech", "downstreams", "speaker_verification")
# append (not insert): keep stdlib ahead of the repo's select.py
sys.path.append(SV_DIR)

from verification import init_model  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchaudio.functional import resample  # noqa: E402
import tqdm  # noqa: E402

CKPT = str(WAVLM_CKPT)


def parse_args():
    parser = argparse.ArgumentParser(description="Seed-TTS WavLM speaker similarity eval")
    parser.add_argument("pair_file")
    parser.add_argument("score_file")
    parser.add_argument("device", nargs="?", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--audio-workers", type=int, default=int(os.environ.get("SIM_AUDIO_WORKERS", "4")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SIM_BATCH_SIZE", "1")))
    parser.add_argument("--pad-batch", action="store_true", default=os.environ.get("SIM_PAD_BATCH", "0") == "1")
    parser.add_argument("--cache-dir", default=os.environ.get("SIM_EMB_CACHE_DIR", ""))
    parser.add_argument("--torch-threads", type=int, default=int(os.environ.get("SIM_TORCH_THREADS", "0")))
    return parser.parse_args()


def read_pairs(pair_file):
    pairs = []
    with open(pair_file) as f:
        for line in f:
            e = line.strip().split("|")
            if len(e) >= 2:
                pairs.append((e[0], e[1]))
    return pairs


def model_fingerprint():
    ckpt = Path(CKPT)
    if not ckpt.exists():
        return f"wavlm_large|{CKPT}"
    st = ckpt.stat()
    return f"wavlm_large|{ckpt.resolve()}|{st.st_size}|{st.st_mtime_ns}|seedtts_similarity_v2_exact"


def audio_fingerprint(path):
    p = Path(path)
    st = p.stat()
    key = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{model_fingerprint()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def cache_path(cache_dir, path):
    if not cache_dir:
        return None
    return Path(cache_dir) / f"{audio_fingerprint(path)}.pt"


def load_cached_embedding(cache_dir, path):
    cpath = cache_path(cache_dir, path)
    if cpath is None or not cpath.exists():
        return None
    try:
        return torch.load(cpath, map_location="cpu")
    except Exception as exc:
        print(f"[sim] cache read failed for {path}: {exc}", flush=True)
        return None


def save_cached_embedding(cache_dir, path, emb):
    cpath = cache_path(cache_dir, path)
    if cpath is None:
        return
    try:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_suffix(".tmp")
        torch.save(emb.cpu(), tmp)
        os.replace(tmp, cpath)
    except Exception as exc:
        print(f"[sim] cache write failed for {path}: {exc}", flush=True)


def load_audio(path):
    wav, sr = librosa.load(path, sr=None, mono=False)
    if len(wav.shape) == 2:
        wav = wav[0, :]
    wav = torch.from_numpy(wav).float().unsqueeze(0)
    if sr != 16000:
        wav = resample(wav, orig_freq=sr, new_freq=16000)
    return path, wav.squeeze(0).contiguous()


def load_audio_safe(path):
    try:
        return path, load_audio(path)[1], None
    except Exception as exc:
        return path, None, exc


def embed_batch(model, batch, device, allow_padding):
    max_len = max(wav.numel() for _, wav in batch)
    if not allow_padding and any(wav.numel() != max_len for _, wav in batch):
        raise RuntimeError("non-padded batch contains variable-length waveforms")
    wavs = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, (_, wav) in enumerate(batch):
        wavs[i, : wav.numel()] = wav
    wavs = wavs.to(device, non_blocking=True)
    with torch.inference_mode():
        embs = model(wavs)
    return [(path, emb.detach().cpu()) for (path, _), emb in zip(batch, embs)]


def embed_with_fallback(model, batch, device, allow_padding):
    try:
        return embed_batch(model, batch, device, allow_padding)
    except Exception as exc:
        if len(batch) == 1:
            print(f"ERR {batch[0][0]}: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(batch) // 2
        return (
            embed_with_fallback(model, batch[:mid], device, allow_padding)
            + embed_with_fallback(model, batch[mid:], device, allow_padding)
        )


def flush_same_length_buckets(model, buckets, device, batch_size, embeddings, cache_dir):
    for length in list(buckets):
        bucket = buckets[length]
        while len(bucket) >= batch_size:
            batch = bucket[:batch_size]
            del bucket[:batch_size]
            for path, emb in embed_with_fallback(model, batch, device, allow_padding=False):
                embeddings[path] = emb
                save_cached_embedding(cache_dir, path, emb)
        if not bucket:
            del buckets[length]


def compute_embeddings(paths, args):
    cache_dir = args.cache_dir
    if args.pad_batch and cache_dir:
        print("[sim] disabling embedding cache because SIM_PAD_BATCH=1 is batch-composition dependent", flush=True)
        cache_dir = ""
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    embeddings = {}
    missing = []
    for path in paths:
        emb = load_cached_embedding(cache_dir, path)
        if emb is None:
            missing.append(path)
        else:
            embeddings[path] = emb

    print(
        f"[sim] unique_audio={len(paths)} cache_hits={len(embeddings)} "
        f"to_embed={len(missing)} batch_size={args.batch_size} "
        f"pad_batch={int(args.pad_batch)} audio_workers={args.audio_workers}",
        flush=True,
    )
    if not missing:
        return embeddings

    model = init_model("wavlm_large", checkpoint=CKPT).to(args.device)
    model.eval()

    loaded = []
    workers = max(1, args.audio_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, wav, exc in tqdm.tqdm(
            pool.map(load_audio_safe, missing),
            total=len(missing),
            desc="load",
        ):
            if exc is not None:
                print(f"ERR {path}: {exc}", flush=True)
                continue
            loaded.append((path, wav))

    batch_size = max(1, args.batch_size)
    if args.pad_batch:
        loaded.sort(key=lambda item: item[1].numel())
        batches = [loaded[i:i + batch_size] for i in range(0, len(loaded), batch_size)]
        iterator = tqdm.tqdm(batches, desc="embed")
        for batch in iterator:
            for path, emb in embed_with_fallback(model, batch, args.device, allow_padding=True):
                embeddings[path] = emb
        return embeddings

    buckets = {}
    for path, wav in tqdm.tqdm(loaded, desc="embed"):
        length = wav.numel()
        buckets.setdefault(length, []).append((path, wav))
        flush_same_length_buckets(model, buckets, args.device, batch_size, embeddings, cache_dir)
    for length in list(buckets):
        bucket = buckets.pop(length)
        for path, emb in embed_with_fallback(model, bucket, args.device, allow_padding=False):
            embeddings[path] = emb
            save_cached_embedding(cache_dir, path, emb)
    return embeddings


def main():
    args = parse_args()
    pairs = read_pairs(args.pair_file)
    valid_pairs = []
    for gen_wav, prompt_wav in pairs:
        if os.path.exists(gen_wav) and os.path.exists(prompt_wav):
            valid_pairs.append((gen_wav, prompt_wav))
    paths = sorted({p for pair in valid_pairs for p in pair})
    print(
        f"[sim] pairs={len(pairs)} valid_pairs={len(valid_pairs)} "
        f"device={args.device} cache_dir={args.cache_dir or '<none>'}",
        flush=True,
    )

    embeddings = compute_embeddings(paths, args)
    scores = []
    with open(args.score_file, "w") as fout:
        for gen_wav, prompt_wav in tqdm.tqdm(valid_pairs, desc="score"):
            if gen_wav not in embeddings or prompt_wav not in embeddings:
                continue
            sim = F.cosine_similarity(
                embeddings[gen_wav].unsqueeze(0),
                embeddings[prompt_wav].unsqueeze(0),
            )
            s = sim.item()
            scores.append(s)
            fout.write(f"{gen_wav}|{prompt_wav}\t{s}\n")
            fout.flush()
        if not scores:
            raise RuntimeError(f"no valid SIM scores for {args.pair_file}")
        avg = sum(scores) / len(scores)
        fout.write(f"SIM: {avg}\n")
    print(f"SIM: {round(avg, 4)}  (n={len(scores)})")


if __name__ == "__main__":
    main()
