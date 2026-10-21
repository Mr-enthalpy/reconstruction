# D5 — HSFAU-inspired general spatial-spectral reconstruction

Started 2026-10-21 on branch `codex/d5-hsfaut-reconstruction` from D4 commit
`809e6664661fdfd741665c0acd39fec73e4a0c78`. This report records G0 and G1 only.
G2 model work and any formal training are not yet executed.

## Status

| Stage | Status | Evidence |
|---|---|---|
| G0 environment and code reuse audit | executed | `d5_environment_audit.md`, `d5_reuse_plan.md` |
| G1 official data access and single-file validation | executed | sample hashes below |
| G1 frozen split and prepared tensors | executed | `data_split_manifest.json`, `prepared_checks.json` |
| G1 forward/adjoint contract | executed | `forward_validation.json`, 21 tests passed |
| G1 PSF mode analysis | executed | `psf_mode_comparison.json`, `psf_policy.md` |
| G1 operator benchmark | executed | `compute_benchmark.json` |
| G2 model, overfit and pilot | not started | none |
| G3 formal training | not started | none |
| G4 frozen test | not started | none |

## G0 facts

- Local GPU: RTX 3060 Ti 8 GiB, PyTorch 2.7.1+cu128, BF16 available.
- Disk D: 675.8 GiB free; CPU 12 cores / 20 threads; RAM 31.7 GiB.
- Existing D4 regression: 15 passed in 4.61 s.
- D5 branch now: 21 passed in 7.64 s after adding D5 tests.
- No shared GPU allocation was verified. G2 is local-first; 512x512 model runs
  still require a measured benchmark.
- The official data are stored outside Git under a machine-local D5 data root.
  Public artifacts contain only hashes, shapes and aggregate results.

## G1 data protocol (frozen)

The prepared set contains 62 independent scenes at 21 x 256 x 256:

| Split | CAVE | KAIST | Total |
|---|---:|---:|---:|
| train | 23 | 16 | 39 |
| validation | 5 | 4 | 9 |
| test | 4 | 10 | 14 |

D4 train + development-test scenes are D5 training data. D4 validation remains D5
validation. The CAVE test set is three never-trained reserve scenes plus the 8-bit
`watercolors` stress scene. KAIST uses the official non-selected 20-scene set for
16/4 train/validation and the official selected 10-scene set as test. Test IDs were
frozen before any D5 training.

Official data validation:

| Sample | SHA256 | Variable | Shape | Dtype |
|---|---|---|---|---|
| `cave_1024_28_scene1.mat` | `8EA19395871017643EA42068236CF4DD1140E84109E4D45A41E71D3DB18C7CAD` | `img_expand` | 1024x1024x28 | uint16 |
| `kaist_selected_scene1.mat` | `3E3EDDD69101D24EF66EDE0A76DB6415A78EDCB82FA36AB8367C8B49A4CEFC1D` | `HSI_crop_586` | 586x586x28 | float64 |

All 20 non-selected KAIST scenes are 2704x3376x28 float64. All 10 selected test
scenes are 586x586x28 float64. D5 center-crops non-square scenes to a square before
resizing, clips small negative KAIST values to zero, and linearly interpolates the
assumed uniform 28 samples over 450-650 nm to 21 bands. This interpolation is an
assumption, not verified wavelength metadata.

The 205-file official CAVE set is not used for primary D5 training because it lacks
a source-scene map; treating each file as independent would leak groups. The local
31-scene CAVE set supplies direct 21-band data.

## G1 operator and tests

- 16 numerical checks passed, including single-band impulse, 21-band superposition,
  CPU/GPU parity, forward/adjoint inner product, FP64 directional derivative,
  asymmetric even kernel, four boundary impulses, batch consistency, and measured
  64 x 64 kernels against a SciPy reference.
- Measured-kernel errors: adjoint 4.4e-7, FFT vs SciPy 1.3e-7, CPU/GPU 1.2e-7.
- All operator benchmarks at 128/256/512, batch 1/2, and 512/1024 PSF support
  completed without OOM. At 512x512, batch 2 with 64 x 64 kernels used about
  848 MiB peak and 10.5 ms for forward+adjoint. This is operator-only, not model
  throughput.

## Critical PSF finding

8 x 8 integration raises the mean off-diagonal correlation between band kernels
from 0.521 to 0.921; the achromatic mean rises from 0.735 to 0.961. The 99% energy
radius is about 32 simulation pixels, so a 256x256 scene has a 32-pixel border halo.
These facts do not invalidate the operator as a declared simulation, but they mean
D5 cannot claim strong wavelength-code exploitation from results under this
integrated operator alone.

## Next gate

G2 must first use the existing D4 PGD as Model A, then add the HSFAU-style
filter-split and SFAT prior only after the G1 contracts. Required before formal
training: single-sample overfit, stage/gradient checks, checkpoint/resume test,
10-20 epoch pilot, and a measured 256x256 model memory benchmark. No classification
or compression work enters D5.

## Reproduction

Commands use the D5 prepared directory and an explicit research Python. Local paths
are intentionally omitted from this public report.

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -X utf8 experiments/d5/prepare.py --cave <local-cave> --watercolors <local-zip> `
  --kaist-train <official-nonselected> --kaist-test <official-selected> `
  --psf <spectral21-root> --output <new-prepared> --size 256 --include-1024
python -X utf8 experiments/d5/preflight.py --prepared <prepared> --output <new-json> --sizes 128,256,512 --batches 1,2
python -X utf8 experiments/d5/psf_modes.py --psf <spectral21-root> --output <new-json>
python -X utf8 -m pytest -q -p no:cacheprovider
```

These commands were executed on 2026-10-21; the placeholders identify required
local inputs and are not claimed as a fresh-clone one-liner.
