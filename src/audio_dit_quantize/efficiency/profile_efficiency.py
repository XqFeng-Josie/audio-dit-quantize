"""Baseline efficiency profiler for LongCat-AudioDiT.

Measures, for voice-cloning inference:
  - param counts + dtype + memory footprint per component (DiT / text-encoder / VAE)
  - end-to-end latency and Real-Time Factor (RTF = synth_time / audio_seconds)
  - per-stage latency breakdown (text-enc / VAE-enc / DiT sampling / VAE-dec)
  - number of DiT forward passes and per-forward latency
  - peak VRAM

Usage:
  python profile_efficiency.py --model_dir meituan-longcat/LongCat-AudioDiT-1B \
      --guidance_method apg --steps 16 --runs 5
"""
import argparse, time, statistics
import torch, librosa
import audiodit  # noqa: F401  (registers with transformers)
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from ..precision import apply_precision
from ..paths import LONGCAT_DIR

PROMPT_WAV = str(LONGCAT_DIR / "assets" / "prompt.wav")
PROMPT_TXT = "小偷却一点也不气馁，继续在抽屉里翻找。"
# a few gen texts of increasing length to probe RTF vs duration
GEN_TEXTS = {
    "short": "今天天气不错。",
    "medium": "今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。",
    "long": "今天晴暖转阴雨，空气质量优至良，空气相对湿度较低，出行请注意携带雨具，"
            "傍晚气温下降明显，体感偏凉，建议适当增添衣物以防着凉感冒。",
}


def fmt_params(n):
    return f"{n/1e6:.1f}M" if n < 1e9 else f"{n/1e9:.3f}B"


def component_report(model):
    comps = {
        "DiT transformer": model.transformer,
        "text encoder (umt5)": model.text_encoder,
        "Wav-VAE (enc+dec)": model.vae,
    }
    print("\n=== Component footprint ===")
    print(f"{'component':<24}{'params':>10}{'dtype':>12}{'MB':>10}")
    total = 0
    for name, mod in comps.items():
        n = sum(p.numel() for p in mod.parameters())
        total += n
        dtypes = {str(p.dtype).replace('torch.', '') for p in mod.parameters()}
        mb = sum(p.numel() * p.element_size() for p in mod.parameters()) / 1e6
        print(f"{name:<24}{fmt_params(n):>10}{','.join(sorted(dtypes)):>12}{mb:>10.1f}")
    print(f"{'TOTAL':<24}{fmt_params(total):>10}")


class StageTimer:
    """Wrap methods; accumulate CUDA-event pairs without forcing sync mid-run."""
    def __init__(self):
        self.events = {}   # stage -> list[(start, end)]
        self.counts = {}

    def wrap(self, obj, attr, stage):
        orig = getattr(obj, attr)
        self.events.setdefault(stage, [])
        self.counts.setdefault(stage, 0)

        def wrapped(*a, **k):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = orig(*a, **k)
            e.record()
            self.events[stage].append((s, e))
            self.counts[stage] += 1
            return out
        setattr(obj, attr, wrapped)

    def summary_ms(self):
        torch.cuda.synchronize()
        res = {}
        for stage, pairs in self.events.items():
            res[stage] = sum(s.elapsed_time(e) for s, e in pairs)
        return res, dict(self.counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meituan-longcat/LongCat-AudioDiT-1B")
    ap.add_argument("--guidance_method", default="apg", choices=["cfg", "apg"])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--cfg_strength", type=float, default=4.0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1,
                    help="batch B identical utterances per generation (throughput / arithmetic-intensity "
                         "axis). Raises M=B*seq toward compute-bound, where W4A4's int4 GEMM advantage shows. "
                         "Reports amortized RTF (lat / total-audio) — lower = higher throughput.")
    ap.add_argument("--mode", default="clone", choices=["clone", "tts"],
                    help="clone = voice cloning (with prompt audio); tts = zero-shot TTS (no prompt)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"],
                    help="fp32 baseline; int8 = W8A8 fake-quant (quality)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the DiT (Phase 2 utilization; warmup absorbs compile time)")
    ap.add_argument("--cudagraph", action="store_true",
                    help="capture transformer.forward into a CUDA graph and replay it (manual; removes "
                         "launch overhead — the launch-bound bottleneck at bs=1). Works for fp16/int8/w4a4 "
                         "incl. opaque CUTLASS/Triton quant kernels. Compatible with both cfg and apg "
                         "(only the pure transformer is captured; the guidance loop stays eager).")
    ap.add_argument("--inductor-cudagraph", action="store_true", dest="inductor_cudagraph",
                    help="use with --compile: enable inductor cudagraphs (mode='reduce-overhead' unless "
                         "--compile-mode overrides). The right cudagraph path for torchao INT8 (whose eager "
                         "kernels are unfused/slow). Needs the rotary keystone fix, applied here automatically.")
    ap.add_argument("--compile-mode", default=None,
                    help="override torch.compile mode (e.g. default / reduce-overhead / max-autotune / "
                         "max-autotune-no-cudagraphs). Default: no-cudagraphs, or reduce-overhead with "
                         "--inductor-cudagraph. Use 'default' vs 'reduce-overhead' for a fast cudagraph A/B.")
    ap.add_argument("--torchao-int8", action="store_true", dest="torchao_int8",
                    help="REAL W8A8 via torchao (efficiency; pre-quantizes DiT once -> fair). Use with --compile "
                         "to hit A100 INT8 tensor cores. (Distinct from --precision int8, which is fake-quant for quality.)")
    ap.add_argument("--torchao-fp8", action="store_true", dest="torchao_fp8",
                    help="REAL fp8 W8A8 (e4m3) via torchao float8. H100/Ada (sm_89+) ONLY — A100 sm_80 has no fp8 "
                         "compute. Use with --compile. The H100-native low-bit arm (parallels --torchao-int8).")
    ap.add_argument("--fp16", action="store_true", help="cast the DiT to fp16 (fair baseline for W4A4, whose kernels are fp16)")
    ap.add_argument("--svdquant-deploy", action="store_true", dest="svdquant_deploy",
                    help="wrap DiT linears with Nunchaku's real SVDQuant W4A4 fused kernels (run in the "
                         "nunchaku conda env; see scripts/svdquant_deploy.py)")
    ap.add_argument("--w4a4-deploy", action="store_true", dest="w4a4_deploy",
                    help="REAL W4A4: wrap every DiT linear with FlatQuant's deploy int4 GEMM + online Kron transform "
                         "(efficiency end-to-end; latency/VRAM only, random transforms). Needs the built `deploy` pkg.")
    ap.add_argument("--w4a4-hp-deploy", action="store_true", dest="w4a4_hp_deploy",
                    help="REAL W4A4 with fp32 GLUE (the quality-recovered path, w4a4_deploy_fp32glue.py) — same int4 GEMM, "
                         "glue in fp32; wrapped as one custom_op so --compile --inductor-cudagraph can fuse+capture it. "
                         "Latency/VRAM only (identity transforms). Tests whether recovered-quality can ALSO be fast.")
    ap.add_argument("--w4a4-hp-fused-deploy", action="store_true", dest="w4a4_hp_fused_deploy",
                    help="REAL W4A4, fp32 glue, FUSED Kron kernel (kron_matmul hp=True: transform+scale+quant+pack in "
                         "one fp32 Triton kernel) + int4 GEMM + fp32 dequant. The candidate for recovered-quality AT speed "
                         "(vs the unfused eager hp path). Latency/VRAM only (identity transforms).")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    clone = args.mode == "clone"

    dev = torch.device(args.device)
    model = AudioDiTModel.from_pretrained(args.model_dir).to(dev)
    model.vae.to_half()
    model.eval()
    info = apply_precision(model, args.precision)
    if args.fp16:
        model.transformer.half(); info["fp16"] = True
    if args.w4a4_deploy:
        from .w4a4_deploy_fp16glue import wrap_dit_w4a4
        import gc
        n_w, sk = wrap_dit_w4a4(model)
        gc.collect(); torch.cuda.empty_cache()
        info["w4a4_deploy"] = f"{n_w} linears wrapped, {len(sk)} skipped"
    if getattr(args, "w4a4_hp_fused_deploy", False):
        from .w4a4_deploy_fp32glue import wrap_dit_w4a4_hp_fused_latency
        import gc
        n_w, sk = wrap_dit_w4a4_hp_fused_latency(model)
        gc.collect(); torch.cuda.empty_cache()
        info["w4a4_hp_fused_deploy"] = f"{n_w} linears wrapped, {len(sk)} skipped"
    if getattr(args, "w4a4_hp_deploy", False):
        from .w4a4_deploy_fp32glue import wrap_dit_w4a4_hp_latency
        import gc
        n_w, sk = wrap_dit_w4a4_hp_latency(model)
        gc.collect(); torch.cuda.empty_cache()
        info["w4a4_hp_deploy"] = f"{n_w} linears wrapped, {len(sk)} skipped"
    if getattr(args, "svdquant_deploy", False):
        from .svdquant_deploy import wrap_dit_svdquant
        import gc
        n_w, sk = wrap_dit_svdquant(model)
        gc.collect(); torch.cuda.empty_cache()
        info["svdquant_deploy"] = f"{n_w} linears wrapped, {len(sk)} skipped"
    if args.torchao_int8:
        # Real INT8 W8A8 — pre-quantize the DiT ONCE (fairness), then (ideally) --compile.
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        quantize_(model.transformer, Int8DynamicActivationInt8WeightConfig())
        info["torchao_int8"] = True
    if getattr(args, "torchao_fp8", False):
        # Real fp8 W8A8 (e4m3) — H100/Ada sm_89+ only. Pre-quantize ONCE (fairness), then --compile.
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        quantize_(model.transformer, Float8DynamicActivationFloat8WeightConfig())
        info["torchao_fp8"] = True
    if args.compile:
        # Compile the DiT (the 97% hot path). Bucket/pad seq-len or expect a few recompiles.
        # Default "max-autotune-no-cudagraphs": autotuned INT8 triton kernels (tensor cores)
        # WITHOUT the CUDA-graphs wrapper. Historically the DiT reused a persistent rotary buffer
        # (rotary_embed `_cos`/`_sin`) reassigned across forwards, which cudagraphs flagged as an
        # overwritten static output and aborted — hence no-cudagraphs. That keystone is now fixed
        # (cudagraph_dit.patch_rotary_for_cudagraph); pass --inductor-cudagraph to enable the
        # WITH-cudagraphs path and remove launch overhead on top of the fused kernels.
        if args.inductor_cudagraph:
            from .cudagraph_dit import patch_rotary_for_cudagraph, clone_dict_output
            p0 = next((p for p in model.transformer.parameters() if p.dtype.is_floating_point), None)
            cg_dtype = p0.dtype if p0 is not None else torch.float16
            patch_rotary_for_cudagraph(model.transformer, cg_dtype, dev)
            mode = args.compile_mode or "reduce-overhead"
            model.transformer = torch.compile(model.transformer, mode=mode)
            clone_dict_output(model.transformer)  # fix cond/uncond output aliasing (L619)
        else:
            mode = args.compile_mode or "max-autotune-no-cudagraphs"
            model.transformer = torch.compile(model.transformer, mode=mode)
        info["compiled"] = mode + (" (cudagraphs)" if args.inductor_cudagraph else "")
    if args.cudagraph:
        if args.compile:
            raise SystemExit("--cudagraph (manual graph) and --compile are mutually exclusive; for "
                             "compile+cudagraph use --compile --inductor-cudagraph.")
        from .cudagraph_dit import wrap_dit_cudagraph
        p0 = next((p for p in model.transformer.parameters() if p.dtype.is_floating_point), None)
        cg_dtype = p0.dtype if p0 is not None else torch.float16
        wrap_dit_cudagraph(model, cg_dtype, dev, warmup=max(args.warmup, 3))
        info["cudagraph"] = str(cg_dtype).replace("torch.", "")
    tok = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

    print(f"model={args.model_dir}  precision={args.precision} {info}  mode={args.mode}  "
          f"compile={args.compile}  guidance={args.guidance_method}  steps={args.steps}  "
          f"runs={args.runs} (warmup {args.warmup})  device={args.device}")
    component_report(model)

    audio, _ = librosa.load(PROMPT_WAV, sr=model.config.sampling_rate, mono=True)
    prompt_wav = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)

    sr = model.config.sampling_rate
    hop = model.config.latent_hop
    # rough prompt latent frames
    prompt_frames = audio.shape[-1] // hop

    B = max(1, args.batch)
    print(f"\n=== Latency / RTF ({'voice cloning' if clone else 'zero-shot TTS'}, batch={B}) ===")
    print(f"{'case':<8}{'gen_frames':>11}{'audio_s':>9}{'lat_med':>10}{'±std':>8}{'min':>9}{'max':>9}{'RTF':>8}{'peakVRAM_GB':>13}")

    for case, gen_text in GEN_TEXTS.items():
        # clone: prompt_text + gen_text fed to encoder, prompt audio prepended.
        # tts: gen_text only, no prompt audio.
        full_text = f"{PROMPT_TXT} {gen_text}" if clone else gen_text
        inp = tok([full_text], padding="longest", return_tensors="pt")
        # batch B identical utterances (throughput axis): replicate along batch dim.
        ids_b = inp.input_ids.repeat(B, 1)
        am_b = inp.attention_mask.repeat(B, 1)
        prompt_b = prompt_wav.repeat(B, 1, 1) if clone else None
        # duration heuristic: ~0.21s/zh-char for gen part, in latent frames (+ prompt if cloning)
        gen_frames = int(len([c for c in gen_text]) * 0.21 * sr / hop)
        duration = (prompt_frames + gen_frames) if clone else gen_frames

        def run_once():
            with torch.no_grad():
                return model(
                    input_ids=ids_b, attention_mask=am_b,
                    prompt_audio=prompt_b,
                    duration=duration, steps=args.steps,
                    cfg_strength=args.cfg_strength, guidance_method=args.guidance_method,
                )

        for _ in range(args.warmup):
            run_once()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)

        lats = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = run_once()
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
        peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9
        # total audio synthesized this batch = per-sample length * B; RTF amortized over it.
        per_sample_s = out.waveform.shape[-1] / sr
        audio_s = per_sample_s * B
        lat = statistics.median(lats)
        lat_std = statistics.stdev(lats) if len(lats) > 1 else 0.0   # dispersion across runs
        lat_min, lat_max = min(lats), max(lats)
        rtf = (lat / 1000) / audio_s
        # OUTPUT-VALIDITY GUARD: a degenerate (all-zero / NaN / inf) waveform means the
        # graph did not compute correctly (e.g. cudagraph-incompatible custom kernels that
        # produce zeros under capture) -> the latency is NOT a valid measurement. Flag it.
        wf = out.waveform.float()
        wmax = wf.abs().max().item()
        finite = bool(torch.isfinite(wf).all().item())
        flag = "" if (finite and wmax > 1e-6) else f"  ⚠️INVALID(max={wmax:.2g},finite={finite})"
        print(f"{case:<8}{gen_frames:>11}{audio_s:>9.2f}{lat:>10.1f}{lat_std:>8.1f}{lat_min:>9.1f}{lat_max:>9.1f}"
              f"{rtf:>8.3f}{peak_gb:>13.2f}{flag}   (N={len(lats)})")

    # ── per-stage breakdown on the medium case ──────────────────────────
    print("\n=== Per-stage breakdown (medium case, 1 instrumented run) ===")
    st = StageTimer()
    st.wrap(model, "encode_text", "text_encode")
    st.wrap(model, "encode_prompt_audio", "vae_encode")
    st.wrap(model.transformer, "forward", "dit_sampling")
    st.wrap(model.vae, "decode", "vae_decode")

    gen_text = GEN_TEXTS["medium"]
    inp = tok([f"{PROMPT_TXT} {gen_text}" if clone else gen_text], padding="longest", return_tensors="pt")
    gen_frames = int(len(gen_text) * 0.21 * sr / hop)
    duration = (prompt_frames + gen_frames) if clone else gen_frames
    with torch.no_grad():
        out = model(input_ids=inp.input_ids, attention_mask=inp.attention_mask,
                    prompt_audio=prompt_wav if clone else None, duration=duration, steps=args.steps,
                    cfg_strength=args.cfg_strength, guidance_method=args.guidance_method)
    times, counts = st.summary_ms()
    audio_s = out.waveform.squeeze().shape[-1] / sr
    total = sum(times.values())
    print(f"{'stage':<16}{'calls':>7}{'ms':>10}{'%':>8}")
    for stage in ["text_encode", "vae_encode", "dit_sampling", "vae_decode"]:
        ms = times.get(stage, 0.0)
        print(f"{stage:<16}{counts.get(stage,0):>7}{ms:>10.1f}{100*ms/total:>8.1f}")
    print(f"{'(sum of stages)':<16}{'':>7}{total:>10.1f}")
    dit_calls = counts.get("dit_sampling", 0)
    if dit_calls:
        print(f"\nDiT forwards: {dit_calls} (= {args.steps} steps x "
              f"{'2 (cond+uncond)' if args.cfg_strength>=1e-5 else '1'})  "
              f"per-forward: {times['dit_sampling']/dit_calls:.1f} ms")


if __name__ == "__main__":
    main()
