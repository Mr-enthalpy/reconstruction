# S1/S2 — observable differences and frozen-network response

Status on 2026-10-31: construction, operator, scalar Oracle and frozen-network
sweeps completed. Coefficient-map inversion is still running; this document does
not substitute detection for recovery. All results are validation-only simulation
with the measured signed, shape-only PSFs from the single laboratory prototype.

## Construction and scope

`natural_spectral_basis.json` uses 39 training scenes, 2,048 sampled neighbour
pairs per scene, three projected unit-norm spectral directions and training-derived
25/50/75-percentile shape-change scales. Nine validation scenes each contribute
27 fixed direction/mask/scale conditions. No test archive was loaded.

Among 243 constructed pairs, the heuristic distribution checks classify 232 as
training-statistics-compatible and 11 as near-distribution. None is classified
as stress-test. These checks do not establish that the pairs are genuine natural
samples. The achieved half-pair perturbation norm relative to the base cube has
minimum 0.001035, median 0.012544 and maximum 0.053520. Per-case SAM, bounds,
effective area and shrinkage are retained in `natural_counterfactual_summary.json`.
The maximum per-pixel broadband difference between the two cubes is 2.132e-14.

## Q1: additional measurement information

The maximum full-forward versus restricted-forward difference-identity error is
4.743e-13. The maximum achromatic difference norm is 7.487e-15. This confirms the
numerical zero-sum control and a chromatic response under the existing linear,
zero-boundary convolution. It is not a new hardware measurement.

All arms use the same absolute sigma, equal to noise fraction times 3.465871334.
This is the simulation's numerical scale, not calibrated sensor DN noise.
Independent plus/minus noise has variance 2 sigma squared. The single-observation
scalar Oracle knows the true baseline cube, spectral direction and complete
spatial template. Its SNR differs from the paired-difference matched-filter SNR
by the observation count and signal amplitude; the two are separately recorded.

At 0.1% noise, 25/27 fixed direction/mask/scale groups have ideal single-template
SNR at least 3 in at least eight of nine scenes. At 0.5% noise, 16/27 do.
These counts describe the chosen conditions; they are neither independent trials
nor a significance test. The remaining groups fail this descriptive detection
threshold. Noiseless SNR is reported as undefined, with the scalar estimate
reported separately. Achromatic templates are marked unidentifiable and are
never divided by a near-zero template energy.

Each condition uses one deterministic noise realization (a pair of independent
draws where two observations are compared), shared across the matched arms.
The theoretical scalar standard error assumes the stated independent white
Gaussian noise. The SNR counts are not empirical repeated-detection success
probabilities, and model seeds do not create additional independent noise trials.

No unconditional SNR range is used. `observability_strata.json` lists each fixed
direction/mask/scale at each nonzero noise level, with all nine scene values.
`observability_noise_sweep.json` also retains per-case pixel RMS SNR, paired
known-template SNR and descriptive frequency groups. Cropped-image FFT bins
are not singular values of a circular imaging operator.

## Q3: frozen A6 response is largely incorrect or noise-driven

The three Chromatic, three Achromatic-mean and three Achromatic-550 checkpoints
are the previously selected D5.3 best checkpoints. Their original bytes and PSF
buffers were checked; no checkpoint was reselected. The 243 pairs, three noise
levels and nine models produce 6,561 response records. The sweep completed in
4,423.57 seconds, with measured peak CUDA allocation 1.174 GiB.

The table below gives scene-balanced averages over all 27 fixed conditions within
each scene, with the range across the three Chromatic seeds. It is a descriptive
overview; every direction/mask/scale, dataset and seed remains separate in
`frozen_a6_strata.json`, and all scalar case values remain in
`frozen_a6_noise_response.json`.

| Noise fraction | Response norm / truth norm | Direction cosine | Relative L2 | Noise-only norm / truth norm |
|---|---:|---:|---:|---:|
| 0 | 0.03647–0.03713 | 0.01002–0.01696 | 0.99990–1.00034 | 0 |
| 0.001 | 0.16054–0.16848 | 0.00464–0.00804 | 1.02102–1.02397 | 0.15087–0.15943 |
| 0.005 | 0.75594–0.79878 | 0.00120–0.00213 | 1.32881–1.35959 | 0.75358–0.79660 |

The large response at 0.5% noise is almost entirely reproduced by two noisy
observations of the unchanged scene. It is not successful recovery. Signed
projection gains remain approximately 0.00073–0.00122, despite the much larger
output norms. No complete condition group passes the diagnostic recovery gate
for any model: 0/729 groups across all arms, seeds and noise levels. This is a
negative finding, not merely a failure to prove a positive aggregate average.

Achromatic models likewise fail. Their noiseless response norms are about
2.1e-5–5.2e-5 of the truth norm: finite float32 forward/network differences,
not evidence of an achromatic spectral channel. Their noisy responses coincide
with noise-only responses. Different noise sensitivity across trained arms does
not by itself establish better exploitation of wavelength-dependent encoding.

## What remains unresolved

The network outputs have poor directional agreement even for these smaller,
training-statistics-derived perturbations. This does not prove that all natural
spectral changes are unrecoverable. The three PCA directions and three masks
cover a limited construction family, and the distribution labels are heuristic.
Lower-noise inputs shift the inference distribution relative to 0.5%-noise
training, so noiseless failure alone cannot identify prior capacity as the cause.

The shared target sigma exactly matches Chromatic training. Achromatic-mean
trained at sigma 0.017336426 and Achromatic-550 at 0.017294910; the common target
is respectively about 0.041% lower and 0.199% higher. These small absolute-noise
mismatches are retained. The raw frozen run's `inference_noise_shift` flag tracked
the nominal fraction only; the checkpoint `sigma_noise` values are authoritative
for absolute alignment. This metadata qualification does not change any measured
response or require recomputation.

The roughly 3.7% noiseless response magnitude here must not be called an
improvement over D5.3's roughly 1.9%: the perturbation families and truth norms
differ, and directional recovery remains poor in both constructions.

Known-template detection supplies much stronger priors than A6 receives. The
restricted coefficient-map inverse must finish before assigning a recoverability
gap. Neither PSNR nor these detection statistics justify a spectral/material
task claim. The unmatched frequency-response control remains an open issue for
interpreting D5.3 spatial reconstruction advantages.

## Traceability

Raw operator result SHA256:
`1cbfc46e89f26684dd5af9702ff337a3405064ab06572c3ccc2572e20f3736e4`.
Raw frozen result SHA256:
`2eade8aef2e88e997d9c5fd8a38dd7818800e408af87da35d6c9e7bad2b11cc9`.
The raw frozen result was recovered from the persisted managed workspace after
lease expiry and verified byte-for-byte, without recomputation. Compact JSONs
carry input manifest, PSF, basis, configuration, checkpoint and execution-source
identities. Frozen inference ran source `aba3c10`; the operator sweep records
its own source identity. Earlier construction ran with uncommitted D5.4 source
on base `ef07c1b`; its source-file hashes, rather than that base alone, identify
the executed implementation.

`basis_reproduction_check.json` additionally verifies the original basis against
the committed fitting algorithm. Maximum direction difference is 1.388e-16;
amplitude quantiles, mean shape and spectral bounds reproduce exactly. The
original basis remains the input to every sweep. Cross-platform construction
summaries retain identical categorical labels; the largest SAM-summary rounding
difference is 8.538e-7 degrees, with relative-norm differences at 1.333e-15.
