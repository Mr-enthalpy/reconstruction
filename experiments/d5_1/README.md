# D5.1 PSF support and integration sweep

D5.1 reuses D5 data/training code and changes only the measured-PSF numerical
strategy. Do not select against the frozen test set.

## Scripts

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path

python -X utf8 experiments/d5_1/psf_sweep.py --psf <spectral21-root> --output <new-sweep>
python -X utf8 experiments/d5_1/repeatability.py --psf <spectral21-root> --output <repeatability.json>
python -X utf8 experiments/d5_1/truncation.py --psf <spectral21-root> --cave <cave-npz> --output <truncation.json>
python -X utf8 experiments/d5_1/scale_consistency.py --psf <spectral21-root> --cave <cave-npz> --output <scale.json>
python -X utf8 experiments/d5_1/local_spectra.py --prepared <prepared512> --run <run> --kernel-file <kernel> --output <spectra.json>
```

Training uses the D5 entry point with an explicit kernel file:

```powershell
python -X utf8 experiments/d5/prepare.py ... --size 512 --output <prepared512>
python -X utf8 experiments/d5/train.py --prepared <prepared512> --output <run> `
  --config experiments/d5/config_pilot_a6_long.json `
  --kernel-file <sweep>/kernels/baseline_shape.npz
```

`psf_sweep.py` writes `kernels/{id}_shape.npz` and `{id}_amplitude.npz`; training
uses shape-only unless an amplitude experiment is explicitly declared.

## Status

The D5.1 report is in `docs/reports/2026-10-21-d5.1/`. Current decision: no lower
integration factor is adopted; baseline 512/8 remains a numerical comparison, not
a frozen physical calibration. G3 has not started.
