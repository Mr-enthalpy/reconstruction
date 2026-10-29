# D5.3 — dispersion-encoding mechanism experiments

Scripts for the D5.3 block. They reuse the D5 trainer and the D5.2 PSF
provenance checks; nothing here is a second training framework.

| File | Purpose |
|---|---|
| `configs/d5_3_a6.json` | one uniform budget for every arm: 400 epochs, 100 steps/epoch, `patience=60`, `eval_every=5` |
| `counterfactual_spectra.py` | builds `X_plus`/`X_minus` with identical per-pixel wideband sums and different spectral distributions |
| `encoding_observability.py` | operator-level (no network) measurement difference, analytic check and noise comparison |
| `reconstruction_response.py` | frozen reconstructor response to the same pairs, with identity / noise-only / achromatic controls |
| `otf_controls.py` | OTF of the chromatic set, the mean kernel and the 550 nm kernel |
| `summarize_seeds.py` | paired multi-seed summary from finished run directories |

## Why the construction works

With `sum_b delta_X_b = 0` at every pixel, a forward model that applies one
kernel to all bands returns the same measurement for both cubes. The chromatic
difference reduces to

```
delta_Y(f) = sum_b (K_b(f) - mean_b K_b(f)) * delta_X_b(f)
```

so the whole experiment separates "one blur for all wavelengths" from
"wavelength-dependent blur" without any network in the loop. The network part
is a separate question: whether the frozen reconstructor turns a visible
measurement difference into a correct spectral difference.

`q <= min(X_i, X_j)` is enforced pointwise; that is what keeps both cubes
non-negative and the wideband sums equal.

## Seeds and budget

Every run passes `--seed`; the effective seed is written to `execution.json`
with `seed_source=cli`. The config's `seed` field is only a default.

All six S1 arms share the same 400-epoch cap and cosine schedule. A run that
reaches the cap with its best epoch inside the final `patience` window is
reported as `budget_censored`; the uniform budget comparison is still valid,
but "converged capability" is not claimed.

## Commands

```bash
python3 experiments/d5/train.py --prepared <prepared512> \
  --config experiments/d5_3/configs/d5_3_a6.json --seed 20261021 \
  --kernel-file <kernels.npz> --output <runs/d5_3/s1c_s20261021> --device cuda

python3 experiments/d5_3/counterfactual_spectra.py --prepared <prepared512> \
  --output counterfactual_protocol.json

python3 experiments/d5_3/encoding_observability.py --prepared <prepared512> \
  --kernel-file <kernels.npz> --output encoding_observability.json --device cuda \
  --sigma-chromatic <measured> --sigma-achromatic <measured>

python3 experiments/d5_3/reconstruction_response.py --prepared <prepared512> \
  --kernel-dir <kernel dir> --runs <run dir> [--runs <run dir> ...] \
  --output reconstruction_response.json --device cuda

python3 experiments/d5_3/summarize_seeds.py --runs <runs/d5_3> \
  --output multiseed_comparison.json
```

## Rules the scripts enforce

- only the `val` split is read; `test.npz` is never opened;
- checkpoints are used exactly as selected by `min_validation_loss`, with their
  own recorded metrics; nothing is re-selected;
- a kernel file that does not match a checkpoint's stored operator is refused;
- seed-level (n = 3) statistics are the primary evidence; nine scenes inside one
  seed are direction evidence only;
- validation loss is not compared across arms (each arm's own forward RMS is
  both the noise scale and the data-loss normaliser).
