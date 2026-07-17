import os
import sys
import subprocess

from tqdm import tqdm
import jiwer
from zhon.hanzi import punctuation
import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel


punctuation_all = punctuation + string.punctuation

wav_res_text_path = ""
res_path = ""
lang = ""
device = os.environ.get("ASR_DEVICE", os.environ.get("DEVICE", "cuda:0"))
ASR_BATCH_SIZE = 1
ASR_BATCH_SIZE_SOURCE = "default"
ASR_BATCH_SIZE_S = 300
ASR_BATCH_SIZE_S_SOURCE = "default"


def cuda_total_gib(dev):
    if not dev.startswith("cuda"):
        return 0.0
    env_mem = os.environ.get("ASR_GPU_MEM_GIB", "").strip()
    if env_mem:
        try:
            return float(env_mem)
        except ValueError:
            pass
    try:
        import torch

        if not torch.cuda.is_available():
            return nvidia_smi_total_gib(dev)
        index = torch.device(dev).index
        if index is None:
            index = torch.cuda.current_device()
        return torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    except Exception:
        return nvidia_smi_total_gib(dev)


def nvidia_smi_total_gib(dev):
    try:
        index = torch_cuda_index_from_device(dev)
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        values = [float(line.strip()) for line in out.splitlines() if line.strip()]
        if not values:
            return 0.0
        index = min(max(index, 0), len(values) - 1)
        return values[index] / 1024
    except Exception:
        return 0.0


def torch_cuda_index_from_device(dev):
    try:
        parsed = dev.split(":", 1)
        logical_index = int(parsed[1]) if len(parsed) == 2 and parsed[1] else 0
    except Exception:
        logical_index = 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parts = [p.strip() for p in visible.split(",") if p.strip()]
        if logical_index < len(parts):
            try:
                return int(parts[logical_index])
            except ValueError:
                return logical_index
    return logical_index


def auto_asr_batch_size(lang_name, dev):
    total_gib = cuda_total_gib(dev)
    if lang_name == "zh":
        if total_gib >= 75:
            return 96
        if total_gib >= 39:
            return 64
        if total_gib >= 23:
            return 48
        if total_gib >= 15:
            return 32
        if total_gib >= 7.5:
            return 16
        return 4
    if lang_name == "en":
        if total_gib >= 75:
            return 12
        if total_gib >= 39:
            return 8
        if total_gib >= 23:
            return 6
        if total_gib >= 15:
            return 4
        if total_gib >= 10:
            return 2
        return 1
    return 1


def auto_asr_batch_size_s(lang_name, dev):
    if lang_name != "zh":
        return 300
    total_gib = cuda_total_gib(dev)
    if total_gib >= 75:
        return 900
    if total_gib >= 39:
        return 600
    if total_gib >= 23:
        return 450
    if total_gib >= 15:
        return 300
    return 180


def env_int_or_auto(name, auto_value):
    raw = os.environ.get(name, "auto").strip().lower()
    if raw == "auto":
        return auto_value, "auto"
    return int(raw), "env"


def load_en_model():
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    return processor, model


def load_zh_model():
    return AutoModel(model="paraformer-zh", hub="hf", disable_update=True)


def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = jiwer.process_words(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures.wer
    subs = measures.substitutions / len(ref_list)
    dele = measures.deletions / len(ref_list)
    inse = measures.insertions / len(ref_list)
    return raw_truth, raw_hypo, wer, subs, dele, inse


def read_params(path):
    params = []
    for line in open(path).readlines():
        line = line.strip()
        if len(line.split("|")) == 2:
            wav_res_path, text_ref = line.split("|")
        elif len(line.split("|")) == 3:
            wav_res_path, _, text_ref = line.split("|")
        elif len(line.split("|")) == 4:
            wav_res_path, _, text_ref, _ = line.split("|")
        else:
            raise NotImplementedError

        if os.path.exists(wav_res_path):
            params.append((wav_res_path, text_ref))
    return params


def write_result(fout, wav_res_path, text_ref, transcription):
    raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
    fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
    fout.flush()


def transcribe_en_batch(processor, model, batch):
    wavs = []
    for wav_res_path, _ in batch:
        wav, sr = sf.read(wav_res_path)
        if len(getattr(wav, "shape", ())) == 2:
            wav = wav[:, 0]
        if sr != 16000:
            wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
        wavs.append(np.asarray(wav, dtype=np.float32))
    input_features = processor(
        wavs, sampling_rate=16000, return_tensors="pt", padding=True
    ).input_features
    input_features = input_features.to(device).to(model.dtype)
    predicted_ids = model.generate(input_features, language="english", task="transcribe")
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)


def transcribe_zh_batch(model, batch):
    wavs = [wav_res_path for wav_res_path, _ in batch]
    res = model.generate(input=wavs[0] if len(wavs) == 1 else wavs, batch_size_s=ASR_BATCH_SIZE_S)
    if isinstance(res, dict):
        res = [res]
    return [zhconv.convert(x["text"], "zh-cn") for x in res]


def empty_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe_with_fallback(batch, fn):
    try:
        return fn(batch)
    except Exception:
        if len(batch) == 1:
            raise
        empty_cuda_cache()
        mid = len(batch) // 2
        return transcribe_with_fallback(batch[:mid], fn) + transcribe_with_fallback(batch[mid:], fn)


def run_asr():
    global wav_res_text_path, res_path, lang, device
    global ASR_BATCH_SIZE, ASR_BATCH_SIZE_SOURCE, ASR_BATCH_SIZE_S, ASR_BATCH_SIZE_S_SOURCE

    wav_res_text_path = sys.argv[1]
    res_path = sys.argv[2]
    lang = sys.argv[3]
    device = os.environ.get("ASR_DEVICE", os.environ.get("DEVICE", "cuda:0"))
    ASR_BATCH_SIZE, ASR_BATCH_SIZE_SOURCE = env_int_or_auto(
        "ASR_BATCH_SIZE",
        auto_asr_batch_size(lang, device),
    )
    ASR_BATCH_SIZE_S, ASR_BATCH_SIZE_S_SOURCE = env_int_or_auto(
        "ASR_BATCH_SIZE_S",
        auto_asr_batch_size_s(lang, device),
    )

    params = read_params(wav_res_text_path)
    batch_size = max(1, ASR_BATCH_SIZE)
    print(
        f"[asr] lang={lang} items={len(params)} batch_size={batch_size} "
        f"({ASR_BATCH_SIZE_SOURCE}) batch_size_s={ASR_BATCH_SIZE_S} "
        f"({ASR_BATCH_SIZE_S_SOURCE}) device={device} "
        f"gpu_mem={cuda_total_gib(device):.1f}GiB",
        flush=True,
    )

    if lang == "en":
        processor, model = load_en_model()
        fn = lambda batch: transcribe_en_batch(processor, model, batch)
    elif lang == "zh":
        model = load_zh_model()
        fn = lambda batch: transcribe_zh_batch(model, batch)
    else:
        raise NotImplementedError

    with open(res_path, "w") as fout:
        for i in tqdm(range(0, len(params), batch_size), desc=f"asr {lang}", dynamic_ncols=True):
            batch = params[i:i + batch_size]
            transcriptions = transcribe_with_fallback(batch, fn)
            for (wav_res_path, text_ref), transcription in zip(batch, transcriptions):
                write_result(fout, wav_res_path, text_ref, transcription)


if __name__ == "__main__":
    run_asr()
