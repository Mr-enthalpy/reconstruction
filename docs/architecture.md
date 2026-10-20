# Spatial-spectral reconstruction and independent task evaluation

## Approved baseline architecture

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
The next baseline must be implemented and tested separately before claiming it runs.
No training, relabeling or new dataset build accompanies this architecture revision.

The current forward uses shape-normalized measured kernels with assumed spectral
weights, spatial scale, shift invariance and boundary conditions. Bayer response
and absolute throughput remain uncalibrated. Exact numerical adjoints do not prove
optical validity or unique recovery. Generality is limited to validated scene/model
conditions. No hardware services, CASSI substitution or new handoff framework.
