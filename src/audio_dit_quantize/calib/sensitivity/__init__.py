"""E2: the candidate-set SCORER (docs/experiments.md §3.4 E2, design decided 2026-07-20).

Goal: predict whether a calibration list will produce a good quantized model WITHOUT running
the full calibrate+generate+evaluate loop (~10-15 GPU-min vs ~2 GPU-h per candidate).

One FP pass over a candidate list yields three score families:
  1. init block losses  — quant-at-init reconstruction difficulty (the `sum_loss_first` signal,
     computed deterministically over all captured seqs instead of a noisy first minibatch)
  2. per-item influence — squared gradient norm of a generation-region task proxy w.r.t. each
     block output, aggregated per calibration item (model-side selection score for P2)
  3. per-channel Fisher — the same gradients aggregated per block-output channel
     (the sensitivity weights for the GATE-A loss-weighting line)

Layering: probes.py = pure math (importable by GATE-A / P2); score.py = CLI for one list;
validate.py = rank-correlation of scores vs the 14 labeled runs (10 random + 4 contrast).

v1 limitations (documented in output meta): task proxy target = LAST BLOCK's residual-stream
output on generation-region tokens (Rademacher probe), not the final projection head — the last
block's own Fisher row is therefore trivial and excluded from GATE-A use.
"""
