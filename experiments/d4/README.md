# D4 general reconstruction

Run from this independent repository with an explicit research Python environment
containing the `d2` extras. CUDA PyTorch is required for declared training.
Keep existing D2/D3 environments and results intact.

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -X utf8 experiments/d4/prepare.py --cave <validated-CAVE-cubes> --psf <spectral21_20261018_reset01> --output <new-prepared-directory> --size 128
python -X utf8 experiments/d4/train.py --prepared <prepared-directory> --output <new-GPU-run-directory>
python -X utf8 experiments/d4/diagnostics.py --prepared <prepared-directory> --run <GPU-run-directory> --output <new-diagnostics-directory>
python -X utf8 -m pytest -q -p no:cacheprovider
```

Replace `python` with the environment's absolute interpreter. No sibling imports
or parent-directory assumptions. Output directories must not exist. Inputs remain
read-only. `--cpu-smoke` executes two epochs and labels results as an execution
test, not a reconstruction quality acceptance.

The loader verifies 31 uint16 full CAVE scenes, selects no semantic face ROIs, and
retains three reserve records without loading their arrays into the prepared tensor.
17/5/6 train/validation/development-test scenes preserve prior held groups; three
further scenes are reserved for final integration. Prior content inspection is
disclosed. This small collection is insufficient for strong generalization claims.

Whole scenes are area-resampled to128² and divided by65535 (no scene-wise energy
normalization). All21 bands share training-only full-scene flips/quarter-turns
and global illumination scaling before simulation. No independent overlapping
optical patch assumption. PSF crop512 is signed-normalized and integrated to64
with assumed sensor scale8; zero boundaries/unit weights/no Bayer are explicit.
Neither scale nor weights are calibrated camera parameters.

Six PGD stages share a four-convolution 21→64→64→64→21 prior. Full21 state,
no spectral basis/classifier. Config: maximum400 epochs, validation selection,
patience60, cosine LR, 0.5% training-RMS Gaussian measurement noise.
Curves must determine adequacy; those settings do not prove convergence.
Development-test evaluation follows checkpoint selection, with fixed100-step
Landweber and1%/5% noise checks. Final-reserve arrays are unavailable to training.
Weights and per-scene outputs stay private. Two GPU runs, UWA/Harvard diagnostics
and model/support-mismatch tests have now executed; see the dated D4 report.
Larger-data training and general reconstruction quality acceptance remain open.

## Capacity comparison and post-selection commands

The single larger-capacity candidate was fixed before development-test inspection.
It uses eight stages, width96 and depth5; select checkpoints/candidates using only
the validation objective. Actual selection was epoch244 of this candidate.

```powershell
python -X utf8 experiments/d4/train.py --prepared <prepared-directory> --output <new-capacity-run> --config experiments/d4/config_capacity.json
python -X utf8 experiments/d4/diagnostics.py --prepared <prepared-directory> --run <selected-run> --output <new-diagnostics> --psf <spectral21_20261018_reset01>
python -X utf8 experiments/d4/evaluate_uwa.py --prepared <prepared-directory> --run <selected-run> --archive <audited-UWA-v1.1.zip> --splits <frozen-D1-splits.json> --output <new-UWA-diagnostic>
python -X utf8 experiments/d4/evaluate_harvard_sample.py --prepared <prepared-directory> --run <selected-run> --sample <verified-img3.mat> --output <new-Harvard-diagnostic>
```

The source evaluators verify the audited archive/sample SHA256 and never fit a
model. Harvard reports both fixed native/gain10 cases, not a selected gain.
The optional1024 support test retains scale8 and signed normalization; it changes
input physics, not trained weights. None of these diagnostics reads final-reserve
arrays or establishes a retrained achromatic comparison.

`rgb_smoke.py --cave <validated-CAVE-cubes> --model <licensed-YuNet.onnx>
--output <new-private-directory>` checks the mature R interface on Oracle
three-band proxies through a synthetic Bayer mosaic. It does not simulate UDC
diffraction, calibrate RGB response or evaluate detector accuracy.
