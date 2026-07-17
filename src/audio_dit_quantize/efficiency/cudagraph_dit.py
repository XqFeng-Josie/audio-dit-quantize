"""CUDA-graph capture for the LongCat-AudioDiT transformer (launch-overhead removal).

Why this exists
---------------
At batch-size-1 the DiT is *launch-bound*: each of the ~30 forwards fires
hundreds of tiny kernel launches (240/320 linears x {quant, transform, GEMM,
dequant} for W4A4), and Python dispatch + per-kernel launch latency dominate
over compute. INT4's ~4x tensor-core peak is invisible in this regime. The
standard fix is to capture the whole DiT forward into a CUDA graph once and
*replay* it — collapsing all launches into a single graph launch.

`torch.compile(mode="max-autotune")` (which enables cudagraphs) ABORTS on this
DiT: the rotary `_cos`/`_sin` tables are a persistent module attribute that gets
*reassigned* on rebuild, which cudagraphs flags as an overwritten static output
(buffer aliasing). This module fixes that at the source and adds a manual,
precision-agnostic cudagraph wrapper that works uniformly for fp16 / INT8
(torchao) / W4A4 (FlatQuant or SVDQuant) — including the opaque CUTLASS / Triton
quant kernels, which inductor cannot trace but a manual graph captures verbatim.

Two pieces
----------
1. ``patch_rotary_for_cudagraph`` — the *keystone*: build the rotary tables ONCE
   to a generous max length (never rebuilt afterwards, so their address is
   stable across replays) and make ``forward`` return a fresh ``.clone()`` so the
   returned cos/sin never alias the persistent buffer.

2. ``CudaGraphRunner`` / ``wrap_dit_cudagraph`` — replace ``transformer.forward``
   with a shape-keyed capture+replay wrapper: first call at a given input shape
   captures a graph (after a few side-stream warmups); subsequent calls copy the
   dynamic inputs into the static buffers and ``graph.replay()``.

Usage (from profile_efficiency.py): ``wrap_dit_cudagraph(model, dtype, device)``
after the precision / quant wrapping is applied. Then run as usual — the first
warmup forward of each latency case captures, the rest replay.
"""
from __future__ import annotations
import torch


# ----------------------------------------------------------------------------
# 1. Keystone: make the rotary embedding cudagraph-safe
# ----------------------------------------------------------------------------
def patch_rotary_for_cudagraph(transformer, dtype: torch.dtype, device: torch.device,
                               max_len: int = 8192) -> None:
    """Build rotary cos/sin ONCE to ``max_len`` and freeze ``forward`` so it
    never rebuilds (stable address) and never aliases the buffer (returns a
    clone). This is the prerequisite for ANY cudagraph capture of the DiT.

    ``max_len`` is deliberately generous (covers every eval sequence length);
    the table is tiny (max_len x head_dim x {cos,sin}).
    """
    rot = transformer.rotary_embed
    # Build once, outside any graph, at the activation dtype. After this the
    # _cos/_sin tensors keep a fixed allocation; we must never call _build again.
    rot._build(max(max_len, rot.max_position_embeddings), device, dtype)
    frozen_len = rot._cached_len

    def forward_frozen(x: torch.Tensor, seq_len: int | None = None):
        if seq_len is None:
            seq_len = x.shape[1]
        # No rebuild. Slice the stable buffer and clone so the result is a fresh
        # graph-internal tensor (never aliases the persistent _cos/_sin).
        assert seq_len <= frozen_len, (
            f"rotary frozen at {frozen_len} but got seq_len={seq_len}; "
            f"raise max_len in patch_rotary_for_cudagraph")
        cos = rot._cos[:seq_len]
        sin = rot._sin[:seq_len]
        if cos.dtype != x.dtype:
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)
        return cos.clone(), sin.clone()

    rot.forward = forward_frozen


# ----------------------------------------------------------------------------
# 2. Manual cudagraph capture+replay of transformer.forward
# ----------------------------------------------------------------------------
# Tensor kwargs that may change in VALUE across forwards (copied into the static
# input buffers on every replay). Constant-valued ones are copied too — cheap,
# and robust if the caller mutates them.
_TENSOR_KWARGS = ("x", "text", "text_len", "time", "mask", "cond_mask", "latent_cond")


class _GraphEntry:
    __slots__ = ("graph", "static_in", "static_out")

    def __init__(self, graph, static_in, static_out):
        self.graph = graph
        self.static_in = static_in      # name -> persistent input tensor
        self.static_out = static_out    # dict returned by orig forward (static)


class CudaGraphRunner:
    """Drop-in replacement for ``transformer.forward`` that captures one CUDA
    graph per distinct input-shape signature and replays it.

    All 30 forwards of a generation (15 Euler steps x cond/uncond) share one
    shape signature here, because ``neg_text = zeros_like(text_condition)`` — so
    a single graph is captured per latency case and replayed 30x.
    """

    def __init__(self, transformer, warmup: int = 3, verbose: bool = True):
        self.transformer = transformer
        self.orig_forward = transformer.forward
        self.warmup = warmup
        self.verbose = verbose
        self.cache: dict = {}
        self.n_capture = 0
        self.n_replay = 0

    @staticmethod
    def _sig(kwargs):
        parts = []
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                parts.append((k, tuple(v.shape), str(v.dtype)))
            else:
                parts.append((k, v))
        return tuple(parts)

    def _capture(self, kwargs) -> _GraphEntry:
        # Persistent static inputs (clone the live tensors); non-tensors are
        # baked into the captured call.
        static_in = {}
        call_kwargs = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                static_in[k] = v.clone()
                call_kwargs[k] = static_in[k]
            else:
                call_kwargs[k] = v

        # Warm up on a side stream so cuBLAS/CUTLASS workspaces are allocated
        # before capture (required by the CUDA graph API).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                self.orig_forward(**call_kwargs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = self.orig_forward(**call_kwargs)

        self.n_capture += 1
        if self.verbose:
            shp = tuple(static_in["x"].shape)
            print(f"[cudagraph] captured graph #{self.n_capture} for x={shp}")
        return _GraphEntry(g, static_in, static_out)

    def __call__(self, **kwargs):
        sig = self._sig(kwargs)
        entry = self.cache.get(sig)
        if entry is None:
            entry = self._capture(kwargs)
            self.cache[sig] = entry

        for k, buf in entry.static_in.items():
            v = kwargs[k]
            if torch.is_tensor(v):
                buf.copy_(v)
        entry.graph.replay()
        self.n_replay += 1

        # Clone outputs: the static buffers are overwritten on the next replay,
        # and within one Euler step the cond + uncond calls share this graph, so
        # the caller MUST get independent copies of pred and null_pred.
        out = entry.static_out
        return {
            "last_hidden_state": out["last_hidden_state"].clone(),
            "hidden_state": None if out["hidden_state"] is None
            else out["hidden_state"].clone(),
        }


def wrap_dit_cudagraph(model, dtype: torch.dtype, device: torch.device,
                       warmup: int = 3, max_len: int = 8192) -> CudaGraphRunner:
    """Apply the rotary keystone fix and replace transformer.forward with a
    cudagraph capture+replay runner. Returns the runner (for stats)."""
    patch_rotary_for_cudagraph(model.transformer, dtype, device, max_len=max_len)
    runner = CudaGraphRunner(model.transformer, warmup=warmup)
    model.transformer.forward = runner
    return runner


# ----------------------------------------------------------------------------
# 3. Inductor-cudagraph (torch.compile reduce-overhead) compatibility fix
# ----------------------------------------------------------------------------
def clone_dict_output(transformer) -> None:
    """Make `transformer.forward` clone its dict-tensor outputs so they escape
    inductor's cudagraph-trees static memory.

    Past the rotary keystone, inductor's automatic cudagraph (reduce-overhead /
    max-autotune) hits a SECOND blocker: the guidance loop holds BOTH the cond
    `pred` and uncond `null_pred`, which the cudagraph returns from the same
    static buffer -> 'accessing tensor output ... overwritten by a subsequent
    run' (modeling_audiodit.py:619). Cloning the outputs here (outside the graph
    region, exactly what the manual CudaGraphRunner does) resolves it. Apply
    AFTER torch.compile so the clone wraps the compiled call."""
    inner = transformer.forward

    def forward_cloned(*a, **k):
        out = inner(*a, **k)
        if isinstance(out, dict):
            return {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in out.items()}
        return out

    transformer.forward = forward_cloned
