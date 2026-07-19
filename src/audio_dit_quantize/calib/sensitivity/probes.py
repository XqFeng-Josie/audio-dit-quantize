"""Pure computation layer of the E2 scorer. No CLI, no file IO (except caller-provided tensors).

All functions take an AudioDiT model + "tagged" captured states and return numpy/torch results.
Reuses the frozen-baseline capture (flatquant_best.capture_block_inputs) called per item so each
captured state keeps its item uid — the baseline capture API is NOT modified.
"""
from collections import defaultdict

import numpy as np
import torch

from ... import flatquant_layers as fq
from ...flatquant_best import _move, capture_block_inputs


def capture_tagged(model, tok, dev, items, per_item_keep=2):
    """[(uid, x_cpu, cond_cpu, prompt_dur)] — one capture call per item so uids stay attached.
    Same total inference cost as the baseline bulk capture (one infer_one per item either way)."""
    tagged = []
    for it in items:
        store = capture_block_inputs(model, tok, dev, [it],
                                     max_seqs=per_item_keep, per_item_keep=per_item_keep)
        for (x, cond, pd) in store:
            tagged.append((it[0], x, cond, pd))
    return tagged


def _wrappers(block):
    return [m for m in block.modules() if isinstance(m, fq.FlatQuantLinear)]


@torch.no_grad()
def init_losses(model, tagged, dev, diag_alpha=0.3):
    """Per-(block, seq) quant-at-init reconstruction loss under fp input propagation — the
    deterministic analog of calibrate_perblock's step-0 loss (same wrap flags, same sq_style
    diag init, fp-propagated inputs). Model must already be fq.wrap_dit-wrapped.
    Returns L [n_blocks, n_seqs] (float64 numpy)."""
    blocks = model.transformer.blocks
    inps = [x for (_, x, _, _) in tagged]
    conds = [c for (_, _, c, _) in tagged]
    n = len(inps)
    L = np.zeros((len(blocks), n))
    for bi, blk in enumerate(blocks):
        ws = _wrappers(blk)
        for w in ws:
            w.enable_quant = False
            w.begin_smax()
        fp = [blk(x=_move(inps[j], dev), **_move(conds[j], dev)).float() for j in range(n)]
        for w in ws:
            w.enable_quant = True
            w.init_diag_scale(alpha=diag_alpha)
        for j in range(n):
            q = blk(x=_move(inps[j], dev), **_move(conds[j], dev)).float()
            L[bi, j] = float(((q - fp[j]) ** 2).mean())
        for w in ws:
            w.enable_quant = False        # leave the model in fp mode
        inps = [f.cpu() for f in fp]      # official drift-free propagation: next block sees fp
        del fp
        torch.cuda.empty_cache()
    return L


def grad_probes(model, tagged, dev, n_probes=2, seed=0, region="gen"):
    """Hutchinson-style sensitivity of a task proxy w.r.t. every block's output, on the FP path.

    Proxy scalar per probe: sum over REGION tokens of (last-block output ⊙ Rademacher u).
    region: 'gen' (tokens [pd:], content pathway — the GATE-B battleground), 'prompt' ([:pd]),
    or 'all'.

    Returns (influence[n_seqs], influence_per_token[n_seqs], fisher[n_blocks, dim]):
      influence  = mean over probes of sum-sq gradient at ALL block outputs for that seq
      fisher     = per-channel mean of squared gradients (last block row is trivial — exclude
                   downstream, see package docstring)
    Model may be wrapped (quant must be DISABLED) or unwrapped."""
    blocks = model.transformer.blocks
    for blk in blocks:                      # ensure fp path if wrapped
        for w in _wrappers(blk):
            w.enable_quant = False
    gcpu = torch.Generator().manual_seed(seed)
    dim = None
    fisher = None
    infl = np.zeros(len(tagged))
    infl_tok = np.zeros(len(tagged))
    for j, (_uid, x, cond, pd) in enumerate(tagged):
        xg = _move(x, dev)
        cd = _move(cond, dev)
        T = xg.shape[1]
        pd = min(max(int(pd), 0), T)
        if region == "gen":
            lo, hi = pd, T
        elif region == "prompt":
            lo, hi = 0, pd
        else:
            lo, hi = 0, T
        if hi <= lo:                        # degenerate region -> whole seq
            lo, hi = 0, T
        for _p in range(n_probes):
            with torch.enable_grad():
                h = xg
                outs = []
                for blk in blocks:
                    h = blk(x=h, **cd)
                    h.retain_grad()
                    outs.append(h)
                u = (torch.randint(0, 2, h.shape, generator=gcpu).float() * 2 - 1).to(dev)
                s = (h[:, lo:hi, :] * u[:, lo:hi, :]).sum()
                s.backward()
            if fisher is None:
                dim = outs[0].shape[-1]
                fisher = torch.zeros(len(blocks), dim, dtype=torch.float64)
            for bi, o in enumerate(outs):
                g2 = (o.grad.detach().float() ** 2)
                fisher[bi] += g2.sum(dim=(0, 1)).double().cpu()
                infl[j] += float(g2.sum())
            for o in outs:
                o.grad = None
            del outs, h, u
        infl[j] /= n_probes
        infl_tok[j] = infl[j] / max(1, T)
        torch.cuda.empty_cache()
    fisher /= max(1, len(tagged) * n_probes)
    return infl, infl_tok, fisher.numpy()


def aggregate_per_item(tagged, per_seq_values):
    """{uid: mean over that item's captured seqs} for any per-seq score vector."""
    acc, cnt = defaultdict(float), defaultdict(int)
    for (uid, _x, _c, _pd), v in zip(tagged, per_seq_values):
        acc[uid] += float(v)
        cnt[uid] += 1
    return {u: acc[u] / cnt[u] for u in acc}
