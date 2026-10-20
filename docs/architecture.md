# Independent RGB, general reconstruction and downstream classification

## D4: current architecture and acceptance

```text
                         one UDC RAW observation
                           /                 \
             R: RGB recovery             N1: full-field
             mature face proposals       spatial-spectral reconstruction
                           \                 /
                             aligned regions
                                    |
                         C: RGB + spectral task
```

R and N1 run independently. Face proposals select downstream regions; they do not
condition N1 execution, training eligibility, supervision or output. Classification
consumes reconstruction and does not select its checkpoint. General full scenes,
not face-classification ROI lists, supply N1's training distribution.

N1 exports `spectral_cube_hat [B,21,H,W]`, `reprojected_measurement`,
`global_spectrum` (derived), and residual/quality diagnostics. No imposed low-rank
or global-average bottleneck. Select checkpoints using reconstruction validation;
report PSNR with declared range, SSIM, local/band/mean spectral errors, SAM,
structure/frequency errors, residual, unseen-scene/domain behavior and resources.
Reconstruction quality is the explanatory foundation of the sensing claim.
Poor reconstruction cannot be excused by calling it an intermediate task.

R exports RGB, boxes/scores, coordinates and observation identity. Use mature
preprocessing/detection and evaluate restoration and recall/localization separately.
Demosaicing alone is not UDC deblurring or calibrated colour recovery.
Current N1 physics is scalar shape-only, not Bayer RAW: R must not interpret it
as a CFA mosaic. Fusion requires one explicitly modelled `y_raw=A_raw(X)+noise`
for both branches. Independent ideal RGB is Oracle, not same-RAW recovery.

C is deferred until N1 independent acceptance. Compare RGB-only, RGB+N1,
Oracle RGB+Oracle spectral and RAW controls with matched groups/budgets and frozen
N1. Final task scenes cannot enter N1 supervision. Preserve detector misses and
false positives. Compression/joint training require later comparative evidence.

The sections below describe the D3 implementation preserved as historical context.
[D4 status](reports/2026-10-20-d4/README.md) owns current execution evidence.

## D3 baseline architecture (historical)

```text
True hyperspectral scene X [B,21,H,W]
  -> explicit measured-PSF approximate forward A -> encoded observation y
  -> spatial-spectral Deep Unrolling reconstructor N1
  -> reconstructed scene X_hat [B,21,H,W]
  -> independently trained task classifier C1
```

N1 is a scene reconstructor independent of Face labels. Its primary output is the
spatial-spectral matrix at an explicitly declared working resolution. It also
exports the spatial-mean 21-point spectrum, A(X_hat), residuals and quality status.
A global curve or arbitrary embedding is not a substitute for this primary output.
Low-rank states, compressed spectral taps and joint task losses are later ablations,
not requirements for the baseline. This does not require full sensor resolution.

Train reconstruction first with scene supervision and measurement consistency,
without classification loss. Validate spatial structure, local spectral shape,
amplitude errors and reprojection, including zero-input/prior dependence. Freeze
N1 before training the classifier. Evaluate real hyperspectral scenes, reconstructed
scenes, direct encoded observations and achromatic-PSF controls on fixed groups.
Oracle spectra have privileged information; a three-band proxy is not calibrated RGB.
The two questions are scene recovery fidelity and spectral information useful to
classification. Reconstruction metrics alone do not establish task benefit.

## Data and source separation

Review CAVE real-face regions as positive candidates as well as non-face negatives.
Photos, statues and toys require explicit separate semantics; no automatic positive
label. Keep every crop from a source scene in one split and preserve the frozen UWA
subject groups. A new dataset protocol must be versioned, not overwrite D2 decisions.
Few CAVE face scenes may still be insufficient for source-balanced evaluation.

Cleaning and common spatial/brightness augmentation may reduce domain cues. All
bands share geometry; never alter spectral shape independently to hide source.
Report registration, quantization and exceptional files, and rerun a source-only
diagnostic after processing. Common normalization does not calibrate UWA intensity
against CAVE approximate reflectance. Data sources remain explicit in evaluation.

## Physical limits and implementation status

The existing D2 code is a completed low-rank 32x32 pilot, not implementation or
acceptance of the revised general scene baseline. Preserve its measured results.
D3 now implements and tests the separate full 21-channel baseline at 32x32, with
reconstruction-only training and frozen downstream heads. See the dated
[D3 report](reports/2026-10-20-d3/README.md) for actual results and limitations.
This bounded implementation does not establish general recoverability or task gain.

The current forward uses shape-normalized measured kernels with assumed spectral
weights, spatial scale, shift invariance and boundary conditions. Bayer response
and absolute throughput remain uncalibrated. Exact numerical adjoints do not prove
optical validity or unique recovery. Generality is limited to validated scene/model
conditions. No hardware services, CASSI substitution or new handoff framework.
