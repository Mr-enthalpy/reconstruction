# D3 independent reconstruction and classification

Use the independent environment documented in D2 (`pip install -e ".[d2]"`).
There is no hardware or Meta package dependency. All paths are explicit.

```text
python -m pytest -q -p no:cacheprovider
python experiments/d3/prepare.py --d2 <D2-prepared> --uwa <D1-production> --cave <D2-cave-full> --psf <measurement-root> --output <new-prepared>
python experiments/d3/run.py --prepared <prepared> --output <new-run>
python experiments/d3/diagnostics.py --prepared <prepared> --run <run> --output <new-diagnostics>
```

Scripts refuse existing output directories. Use UTF-8 mode on Windows. Original
archives, D1/D2 data and measurements are read only. No data/model is redistributed.
The D3 report records actual results; `config.json` was fixed before fitting.

## Data

Reuse D2 accepted original paired UWA ROIs and 26 CAVE negatives. Verify D1 NPZ
hashes, reproduce D2 numerical input and check all 21 raw PSF source hashes.
Three newly reviewed CAVE face crops include two physical-person regions and one
printed-face region from two source images. Label means recognizable human face,
not skin or liveness. The two face scenes are conservatively grouped train-only
because their independence is unverified. There is no held-out CAVE face claim.
The script's coordinates were inspected on full source images and crop previews.

Preserve UWA subject and original CAVE scene splits. A single shared-band affine
view per training sample uses the existing guarded D2 transform. A draw that cuts
ROI corners is rejected deterministically (up to 100 attempts); actual seed and
rejection count are stored. No test or validation augmentation. Recompute target
means from the transformed cube. Keep original and derived counts separate.

Bit-range scaling and equal grand mean 0.2 are a numerical convention. They do not
calibrate 8-bit processed intensity against 16-bit approximate reflectance. No
bandwise jitter, quantization repair or silent band filling is used. Existing
registration/illumination limitations persist. `bit_scaled` is an additional
diagnostic copy before energy matching, not radiance.

## Model and independent heads

`FullSpectralPGD` maintains all 21 spatial channels with no fitted basis or spectral
bottleneck. Three PGD steps use D1/D2 FFT forward/adjoint, followed by a bounded
data step. The shared 21->32->21 convolutional proximal has 12,152 total model
parameters. State resolution is 32x32; there is no claim of full-sensor recovery.

`train_n1` receives scenes, training indices, kernels and configuration only.
Reconstruction loss combines map MSE, spatial gradient MSE, pixel spectral cosine
error (target norm >1e-3), and normalized measurement MSE. Train 60 fixed epochs,
final checkpoint, one seed, no test selection. Main noise is zero; explicit seeded
1%/5% train-measurement-RMS Gaussian noise is a sensitivity, not a calibrated
camera noise model. No classification loss enters N1.

N1 is frozen and its complete state hash verified unchanged through C1 fitting.
C1 is balanced logistic regression on a 4x4 area-pooled 21-band cube (336 features,
337 weights including intercept), L2=0.01 and training-only standardization.
Oracle fits/uses true cubes; Reconstruction fits/uses detached N1 cubes;
Frozen-head fits true cubes and applies the same weights and normalization to N1.
The global spectrum, three-band proxy, intensity-only and encoded-plane heads are
explicit additional baselines. They do not replace N1's spatial output.

Repeated protocols use measured/achromatic kernels and a no-augmentation control.
All output maps, global spectra, observations, reprojections, checkpoints and
classifier weights are local NPZ/PT files. Individual errors stay private.
`results.json` is aggregate; `private_predictions.json` is not publishable.

## Interpretation

The simulation retains signed, shape-normalized measured kernels, equal band
weights, 16 sensor pixels per simulation pixel and zero boundaries. Each entire
candidate scene is convolved before pooling; overlapping subpatches are not
invented as independent observations. 1024-support, weights, clipped-kernel and
hypothetical RGGB perturbations are fixed mismatch probes, not calibrated models.
The RGGB example uses broad Gaussian responses and does not identify this sensor.

The diagnostic near-null perturbation is a numerical illustration, with its CG
status reported explicitly. One scalar 32x32 observation has fewer dimensions than
21x32x32 unknowns. Recovery therefore depends on priors; small residuals do not
prove correct spectra. Report band errors, spatial gradients, spectral angles,
negative outputs and domain dependence alongside classification.
