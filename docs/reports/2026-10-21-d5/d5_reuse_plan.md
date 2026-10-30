# D5 G0 code reuse and gap plan

Date: 2026-10-21. Based on the actual working tree, not only the D4 reports.
Classification, face proposals, compression and joint optimization remain out of
scope.

## Actual code table

| Existing file | Role | Current tests | Reuse | D5 gap |
|---|---|---|---|---|
| `src/lcd_reconstruction/operators/measured_psf.py` | NumPy 21-band full convolution and explicit adjoint with integer-centred crop | `tests/test_d2_numerics.py` | High; numerical reference | No filter split, no spatial-spectral response, no nonnegative mode |
| `src/lcd_reconstruction/unrolling/spatial_pgd.py` | GPU FFT full convolution/adjoint, Lipschitz bound, low-rank PGD | `test_d2_numerics.py` | High; keep validated A/At | Zero-padded crop is not exactly diagonalised; no learned frequency update |
| `src/lcd_reconstruction/unrolling/full_spectral.py` | `SinglePlanePhysics`, full 21-channel PGD, reconstruction loss | `test_d3_independent.py`, `test_d4_general.py` | High; physics and loss base | Shared single CNN proximal only; no multiscale/SFAT |
| `src/lcd_reconstruction/unrolling/general_pgd.py` | D4 general 21-channel PGD, 6/8 stages | `test_d4_general.py` | High; Model A control | No filter split, no SFAT, no resume |
| `src/lcd_reconstruction/rgb_frontend.py` | Minimal RGB demosaic and YuNet proposals | `test_d4_general.py` | None for D5 | Deliberately out of scope |
| `experiments/d4/prepare.py` | Whole CAVE scenes, grouped split, measured PSF integration | split test in `test_d4_general.py` | Medium; adapt, do not overwrite D4 protocol | Hard-coded 31-scene D4 split; no KAIST/official loaders |
| `experiments/d4/train.py` | GPU training, validation selection, fixed metrics | 15-test regression | Medium; reuse metric definitions | No resume, run metadata, AMP, last checkpoint, atomic writes |
| `experiments/d4/diagnostics.py` | Post-selection panels and mismatch tests | executed historically | Low/medium | No D5 protocol or fresh test split |
| `experiments/d4/evaluate_uwa.py`, `evaluate_harvard_sample.py` | External source diagnostics | executed historically | Low | No frozen D5 test evaluator |
| `docs/architecture.md`, `docs/method_principles.md` | Approved N1 interface | documentation | High | D5 implementation status must be added after execution |

## Verified capabilities

- N1 already returns the full `[B,21,H,W]` cube, reprojection, global spectrum and
  residual (`general_pgd.py:24-32`).
- The physics operator supports autograd, an explicit adjoint and a Lipschitz
  bound; CPU/GPU/dtype parity and directional derivatives are already tested.
- Two real CUDA runs exist. D4 used scene-level groups, validation-only checkpoint
  selection and never loaded the final-reserve arrays.
- The current public regression is 15 passed, including full-state output and
  signed even-kernel adjoint tests.
- No HSFAU/SFAT, filter-split or learned frequency-update code exists in this
  repository. The public MIT SDI clone is local only and was not copied.

## Minimal D5 additions

1. `src/lcd_reconstruction/operators/filter_split.py`
   - Explicit `J_b = H_b X_b`, `y = sum_b w_b J_b`, with per-band spatial weights.
   - Reuse the validated convolution A/At; analytic filter update only for constant
     weights. Keep zero-boundary full-convolution semantics explicit.
2. `src/lcd_reconstruction/unrolling/hsfaut.py`
   - Stage-wise data consistency plus learned proximal update.
   - Frequency-domain step only as a validated preconditioner under circular
     assumptions; otherwise use the matched iterative operator.
3. `src/lcd_reconstruction/priors/sfat.py`
   - Three-level U-shaped spatial-frequency aggregation prior with channel
     attention and FFT magnitude branch. First version: 21 channels, 3 stages.
4. `src/lcd_reconstruction/datasets/general_scenes.py`
   - CAVE 21-band loader, official KAIST/official-CAVE `.mat` inspection and
     conversion, source-group IDs, frozen split manifest.
5. `experiments/d5/preflight.py`, `prepare.py`, `train.py`, `evaluate.py`
   - Preflight checks inputs/operator/tests/GPU/disk.
   - Atomic `last`/`best`, optimizer/scheduler/AMP/RNG state, unique run ID.
   - Frozen independent evaluation with per-scene artifacts.
6. Tests:
   - `tests/test_d5_physics.py`: 10-item forward/adjoint contract.
   - `tests/test_d5_model.py`: stage participation, gradients, full-state output.
   - `tests/test_d5_protocol.py`: grouped splits, no test leakage, band mapping.

## Explicit non-goals

- Do not replace the measured PSF with the authors' 28-channel PSF or filter.
- Do not call an adapted 21-channel model a line-by-line HSFAUT reproduction.
- Do not use the 205 `cave_1024_28` files as independent scene groups.
- Do not use development-test results for checkpoint selection.
- Do not update the Meta Submodule pin during D5 development.

## G0 decision

The local GPU software environment and D4 code base are usable, but D5 is not yet
cleared to train. The first blocking items are the 28-to-21 band policy, official
source grouping, and the D5 forward validation. G1 addresses these before any
training run.
