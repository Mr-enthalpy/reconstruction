# Reproduce the D2 pilot

Create an independent Python environment and install this repository with
`python -m pip install -e ".[d2]"`. The recorded run used Python 3.13.5 and CPU
PyTorch 2.7.1. Use the environment's explicit Python path for every command.
No Meta checkout, hardware SDK or sibling imports are required.

```text
python -m pytest -q -p no:cacheprovider
python experiments/d2/cave_inventory.py --archive <official-complete-zip> --output <new-inventory-dir>
python experiments/d2/prepare.py --uwa <D1-production-dir> --cave <reviewed-inventory-dir> --psf <local-measurement-dir> --output <new-prepared-dir>
python experiments/d2/run.py --prepared <prepared-dir> --output <new-run-dir>
```

Output directories must be new. Keep all data outside Git. The CAVE inventory
requires `cave_review_decisions.json`, keyed by scene ID, with a `decision` value
of `accepted_non_face` only after inspecting content; other decisions are excluded.
This is a task annotation, not an official CAVE label. Retain reasons and reviewer.
The full archive source and local-research license evidence are in the report.

The UWA input is the frozen D1 `dataset_manifest.csv`, `splits.json` and sample NPZs.
Only accepted, non-augmented paired face/non-face acquisitions are consumed.
Each sample contains `cube` and `metadata_json`. No face files are redistributed.
The PSF input is the verified `spectral21_20261018_reset01` output layout:
`roi_extent_review/roi1024_<nm>.npz` and matched
`received/R21Dark{Pre,Post}01/probe_<001..003>_mean.npy`.
These explicit readers consume measured outputs, not sibling implementation code.

## Fixed experiment

`config.json` was fixed before model fitting. Train/val/test are isolated by UWA
subject and CAVE scene. One replayable view per training ROI, no validation/test
augmentation. All 21 bands share geometry and amplitude. The 32×32 state is a small
prototype; preservation of full ROI semantics does not establish photometric
calibration. Initial D1 scores remain historical; D2 changes preprocessing and
augmentation, so its matched no-augmentation control is the immediate comparator.

Input is divided by its nominal bit range, spatially averaged, then scaled to
grand mean 0.2. This energy-matched convention does not equate reflectance and DN.
Flat illumination and equal band weights are assumptions. Signed dark/background
corrected PSFs are unit-sum normalized per band and block summed by 16 sensor
pixels. Zero-boundary convolution and its exact adjoint reuse D1's crop convention.
No Bayer model, measured absolute throughput or validated scene/sensor pixel map.

For each protocol, train three 591-parameter models: measured PSF with augmentation,
achromatic PSF with augmentation, measured PSF without augmentation. Use rank four
from original training spectra, three PGD stages, a shared two-convolution proximal
module, 30 fixed epochs, and the final epoch. No test-selected checkpoint.
Logistic heads use training-only standardization, balanced loss, fixed L2=0.01.
Encoded heads use an 8×8 pooled scalar observation; oracle and three-band proxy
see privileged spectral data. The proxy is not a calibrated RGB camera.

`results.json` contains train/val/test and per-domain metrics, confusion matrices,
loss histories, spectral/map/reprojection error and fixed perturbation probes.
Single-class CAVE evaluation reports negative recall, not binary BA.
`*_outputs.npz` exposes spectral maps, global spectra and reprojections;
checkpoints, individual predictions and sample IDs stay local.
Perturbations are robustness probes without retraining or model selection.
No C2/C3 head or high-resolution recovery requirement is introduced.
