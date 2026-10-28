# D5.2 server training and mechanism experiments

D5.2 reuses `experiments/d5` for data preparation, training and metrics. It adds
formal configs, checkpoint resume repair and mechanism controls. The frozen test
set is not uploaded or loaded during training.

## S0 checks

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -X utf8 experiments/d5_2/resume_equivalence.py --output <resume.json>
python -X utf8 -m pytest -q -p no:cacheprovider
```

On the server, use an explicit CUDA device and the uploaded prepared directory.

## Single-card A6

```bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=<server-root>/code/reconstruction/src
python3 -X utf8 experiments/d5/train.py \
  --prepared <server-root>/data/prepared512 \
  --output <server-root>/runs/a6_formal_s1_gpu3 \
  --config experiments/d5_2/configs/a6_formal.json
```

`configs/a6_formal.json` uses 6 stages, batch 1, 100 steps/epoch, 300 epochs,
patience 40 and validation-only selection. `eval_every=5` still limits the expensive
SSIM/FFT validation metrics, but full metrics are now also computed whenever validation
loss sets a new best and on the final epoch, so the selected checkpoint always carries
its own metrics. Each checkpoint records `selection` (rule, epoch, value, metrics) and
`psf_provenance`; `validation.json` re-checks the recorded metrics against a fresh pass.

## PSF provenance

Kernel identity is the canonical array digest (shape + float64 C-order bytes), because
two `.npz` containers of the same array hash differently. Checkpoints store that digest
plus the digests of the physics buffers (`kernel_fft`, `weights`, `L`).

- `train.py --resume` refuses to continue when `--kernel-file` does not match the
  checkpoint operator, so two matched arms cannot be cross-wired.
- `evaluate.py --kernel-file` must match the checkpoint operator. A deliberate
  substitution requires `--allow-psf-mismatch`, which keeps the requested operator,
  records both digests, marks `formal_result=false` and is refused on the test split.
- Checkpoints without a provenance record (before `8c50c88`) are still checked
  operator-vs-operator and reported as `matched-legacy-no-record`.

## Matched-arm comparison

```bash
python3 -X utf8 experiments/d5_2/compare_arms.py \
  --chromatic <run>/s1_chromatic_20261028 --achromatic <run>/s3_achromatic_20261028 \
  --output comparison.json
```

It compares each arm's rule-selected checkpoint on the same scenes, the trajectory at
matching epochs, and the within-arm plateau spread. It never re-selects a checkpoint.

## Resume

```bash
python3 -X utf8 experiments/d5/train.py ... \
  --resume <run>/last.pt
```

`last.pt` now contains post-update best/stale, scheduler state, `next_epoch`,
`global_step` and RNG state. A resume must be tested against an uninterrupted run.

## Achromatic control

Construct the mean kernel from the 21 baseline shape-only kernels, replicate it to
21 bands and train A6 from scratch:

```bash
python3 -c "import numpy as np; k=np.load('.../kernels.npz')['kernels']; \
m=k.mean(0,keepdims=True); np.savez_compressed('.../kernels_achromatic.npz', \
kernels=np.repeat(m,21,axis=0))"

python3 -X utf8 experiments/d5/train.py ... \
  --kernel-file .../kernels_achromatic.npz \
  --output <run>/a6_achromatic_s1_gpu5
```

This is a matched retraining control, not an inference-time kernel substitution.

## Configs

- `configs/a6_formal.json`: A6 CNN-PGD, 98,139 params.
- `configs/a_large_formal.json`: width 96 / depth 5, about 285k params.
- `configs/c6_matched_formal.json`: SFAT width 64, 101,211 params, 6 stages.

Do not begin G4 or load `test.npz` until the model, checkpoint, metrics and
configuration are frozen.
