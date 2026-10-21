# D5 G2 model, overfit and short pilot

Date: 2026-10-21. Validation-only work; `test.npz` was never loaded. This is not
G3 and does not establish reconstruction acceptance.

## Implemented models

| Model | File | Prior | Purpose |
|---|---|---|---|
| A | `unrolling/general_pgd.py` (existing D4) | shared 4-convolution CNN | control |
| B | `unrolling/hsfaut.py` + `unrolling/filter_split.py` | shared CNN | HSFAU-style filter split |
| C | `unrolling/hsfaut.py` + `priors/sfat.py` | 3-level spatial-frequency aggregation | SFAT candidate |

All models expose `[B,21,H,W]`, reprojection, global spectrum, residual and
per-stage residual diagnostics. No classifier, class label, face ROI or spectral
basis is present. The filter split reduces exactly to the validated A/At at unit
weights.

## Critical implementation correction

The first C/B pilot used a wrong `FilterSplitPhysics.lipschitz_bound`: it summed
the already channel-summed spectrum again, giving 441 instead of 21. That made the
data-consistency step 21x too small. The traceable symptom was a much worse
measurement residual in B/C than in A at the same stages. The bound is now
`sum_b w_b^2 |K_b|^2`, matches `SinglePlanePhysics` exactly at unit weights, and
has a regression test. Earlier buggy B/C runs are not used as evidence.

## Single-sample overfit

Scene: `balloons`, fixed full scene, no augmentation, truth reconstruction loss.

| Model | Steps | Initial loss | Final loss | Reload max abs |
|---|---:|---:|---:|---:|
| C, 3 stages | 1000 | 0.12578 | 0.03518 | 0.0 |
| C, 5 stages | 600 | 0.11538 | 0.03908 | 0.0 |
| A, 6 stages | 600 | 0.04824 | 0.03349 | 0.0 |

The loss, gradients, all stage parameters and checkpoint reload work. However the
fit is not tight: a Charbonnier floor around 0.03 remains. This is evidence of the
ill-posed single-plane problem and limited prior capacity, not a claim of successful
overfitting. It is not used to justify a longer G3 run.

## Short pilot

Budget: 10 epochs x 100 optimizer steps = 1000 steps, batch 1, 256x256, 5% training
measurement noise, shared geometric augmentation. Selection used validation loss
only. Values are unweighted scene means over 9 validation scenes.

| Run | Params | Best epoch | PSNR dB | SSIM | SAM deg | Cube rel L2 | HF rel L2 | Grad rel L2 | Measurement rel L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | 98,139 | 10 | **28.24** | **0.868** | **12.32** | **0.330** | 0.9741 | 0.8335 | 0.0484 |
| B6 | 98,139 | 10 | **28.24** | **0.868** | **12.32** | **0.330** | 0.9741 | 0.8335 | 0.0484 |
| C6 | 60,699 | 9 | 27.96 | 0.863 | 14.24 | 0.342 | 0.9748 | 0.8387 | 0.0450 |

A6 and B6 agree to numerical precision, confirming the split formulation and unit
filter. C6 does not improve over B6 at this budget: PSNR is 0.28 dB lower and SAM
is 1.9 degrees worse. C also has 37% fewer parameters because the SFAT prior is
smaller than the CNN prior; this is not a matched-capacity comparison.

Per-source results are in `g2_pilot_summary.json`. Both sources show the same
pattern: high-frequency relative error remains about 0.975 and spatial-gradient
error about 0.84, so the pilot has not recovered texture/detail. Global spectrum
error is better than cube error but does not certify local spectra.

## Resources and resume

| Model | Size | Batch | Step ms | Peak CUDA |
|---|---:|---:|---:|---:|
| A6 | 256 | 1/2 | 64.6 / 118.9 | 1.10 / 2.14 GiB |
| B6 | 256 | 1/2 | 75.2 / 152.4 | 1.23 / 2.39 GiB |
| C6 | 256 | 1/2 | 93.2 / 165.6 | 1.22 / 2.34 GiB |
| C6 | 512 | 1 | 302.4 | 4.75 GiB |

Checkpoint resume was tested: 3 epochs, then resume from `last.pt` to 5 epochs.
Epochs 4-5 continued with the correct scheduler state, the best epoch 2 remained
selected, and no state was reset.

## Decision

G2 is **executed but not accepted**:

- mechanics, full-state output, gradients and resume pass;
- single-sample loss decreases but does not tightly fit;
- the short pilot does not show SFAT gain over the CNN prior;
- high-frequency and local structural recovery remain weak.

Do not start G3 on this evidence. The next G2 iteration must first make a
matched-capacity A/B/C comparison (same stages and approximately same parameter
count) and establish whether SFAT helps at all. Only then freeze a formal training
configuration. The frozen test set remains untouched.
