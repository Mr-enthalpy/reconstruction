# Restricted coefficient-map inversion — execution in progress

The full validation inverse is running under committed policy `04b5804` and
runner `34e65eb65dc055d029b72456c22299ab48a51e62`. No final validation
recoverability category is assigned yet. This document separates completed
training/numerical diagnostics from the pending scientific result.

## Operator and numerical meaning

The restricted map is B(z) = A(v z), and its adjoint is the spectral projection
of A*. The implementation uses exact spectral distributivity to collapse the
21 fixed kernels for this particular direction, then calls the existing
`FFTMeasuredOperator`. Tests compare both forward and adjoint against the full
21-band zero-padded linear convolution, including even-sized asymmetric crops.
This computational identity introduces neither a new physical kernel nor a
circular boundary model.

The solver uses float64 SciPy conjugate gradients on B*B + lambda D*D + mu I,
zero initialization, relative normal-equation residual tolerance 1e-6 and a
uniform maximum of 1,000 iterations. D*D uses non-wrapping differences. The
preconditioner is the exact inverse of the positive Neumann regularizer, evaluated
by orthonormal DCT-II. A small normal residual establishes numerical convergence
of this regularized objective; it does not establish low reconstruction error.

An achromatic zero-sum suboperator is explicitly marked unobservable and returns
the zero coefficient estimate, with relative error 1. It is not reported as a
successful converged reconstruction. Every other result retains convergence,
iterations, final normal residual, measurement residual and reconstruction error.

## Frozen training-only parameter selection

All three directions on the first three sorted training scenes (balloons, beads,
cd), local mask and middle amplitude were used. These are three original CAVE
scenes, not nine independent scenes; the small, source-limited development set is
a limitation. Each noise level uses the same eight-candidate lambda/mu grid.
Noise is matched by level, without using validation reconstruction outcomes.

The final selection chooses minimum mean relative L2 among candidates converged
on all nine training cases. This reliability rule was recorded before reading
the final grid outcomes or running validation inversion. Raw unconstrained
minima and failed candidates are retained.

| Noise fraction | Selected lambda | Selected mu | Mean training relative L2 | Converged cases |
|---|---:|---:|---:|---:|
| 0 | 1e-4 | 1e-8 | 0.099971 | 9/9 |
| 0.001 | 1 | 1e-8 | 0.916401 | 9/9 |
| 0.005 | 1 | 1e-4 | 1.249358 | 9/9 |

At zero noise the unconstrained minimum is 0.067646, but only 3/9 cases converge;
it was not selected. The noisy optima reach the largest lambda in the fixed grid.
The grid is not expanded after observing this. At target 0.5% noise, even the
selected mean error exceeds the zero-estimator reference of 1. This is a negative
training diagnostic, not a universal bound on possible estimators.

## Retained numerical failures and budget history

The first unpreconditioned 100-iteration training grid selected an unconverged
candidate (0/9); its original result remains `training_regularisation.json`.
A regularizer-preconditioned 100-iteration pilot converged at high noise but
remained unconverged for some low-noise candidates. Before any validation
coefficient-map result, every arm and candidate received the same 1,000-iteration
cap and unchanged tolerance/objective. The final three training grids are
`training_regularisation_final_*.json`. They retain all candidates and per-case
metrics; hashed raw outputs retain full residual histories.

These changes address numerical termination. They do not justify interpreting
an earlier censored estimate as a converged physical limit. No promising
validation arm was selectively extended.

## Frequency and boundary diagnostics

`conditioning_probes.json` evaluates finite, cropped B on constants, a local
smooth field and several cosine frequencies. A constant zero-sum spectral
perturbation produces essentially boundary-only chromatic response: for direction
0 the interior energy fraction is 5.744e-15. Interior DC blindness is consistent
with equal kernel sums; finite-image boundaries can still carry a response.
The local smooth probe and different cosine probes have very different gains.

These are descriptive directional gains, not a singular-value decomposition or
an exact condition-number estimate. Zero padding, cropping and regularization
must remain explicit. The results cannot be used to claim that every spatial
frequency is identifiable or that frequency-response confounding in D5.3 is
eliminated.

## Pending comparison

The first-level scalar Oracle knows the complete true template and baseline;
the second level knows only the true baseline and the chosen spectral direction,
with no mask or true coefficient initialization. If a complete second-level
condition passes the registered gate, the third level will replace the true
cube baseline by a fixed A6 estimate from an additional clean reference exposure.
That third level is explicitly a privileged two-observation diagnostic. It will
not be described as deployable single-shot spectral reconstruction.
