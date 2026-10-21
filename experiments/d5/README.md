# D5 general reconstruction

D5 uses the existing D4 worktree plus a measured-PSF filter split and an
HSFAU-style/SFAT candidate. It never trains on Face labels or the frozen test set.
See `docs/reports/2026-10-21-d5/` for evidence and current acceptance status.

## Entry points

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path

python -X utf8 experiments/d5/prepare.py --cave <cave-npz-dir> `
  --watercolors <watercolors_ms.zip> --kaist-train <official-nonselected> `
  --kaist-test <official-selected> --psf <spectral21-root> `
  --output <new-prepared> --size 256 --include-1024

python -X utf8 experiments/d5/preflight.py --prepared <prepared> `
  --output <new-forward-validation.json> --sizes 128,256,512 --batches 1,2

python -X utf8 experiments/d5/psf_modes.py --psf <spectral21-root> `
  --output <new-psf-comparison.json>

python -X utf8 experiments/d5/train.py --prepared <prepared> `
  --output <new-run> --config experiments/d5/config_pilot_c6_long.json

python -X utf8 experiments/d5/benchmark_model.py --prepared <prepared> `
  --config experiments/d5/config_pilot_c6_long.json --output <new-benchmark.json>

python -X utf8 experiments/d5/summarize.py --run name=<validation.json> `
  --output <new-summary.json>
```

`--overfit` with `--overfit-steps` runs the fixed single-scene check. `--resume`
restores model, optimizer, scheduler, epoch, RNG, best metric and sampler state.

## Models and configs

| Config | Model | Stages | Prior |
|---|---|---:|---|
| `config_pilot_a.json` | A, D4 PGD | 3 | shared CNN |
| `config_pilot_b.json` | B, filter split | 3 | shared CNN |
| `config_pilot_c.json` | C, filter split | 3 | SFAT |
| `config_pilot_c5.json` | C | 5 | SFAT |
| `config_overfit_a6.json` | A | 6 | shared CNN |
| `config_pilot_a6_long.json` | A | 6 | shared CNN |
| `config_pilot_b6_long.json` | B | 6 | shared CNN |
| `config_pilot_c6_long.json` | C | 6 | SFAT |

`config_resume_c.json` is the short resume test configuration. The pilot configs
are not formal G3 configs: G2 was not accepted, so no formal run is authorized.

## Test policy

`test.npz` is not loaded by `train.py`. G4 evaluation must load it only after the
checkpoint and metric protocol are frozen. Validation selection uses only the
validation objective. No development-test or test scores are used for tuning.
