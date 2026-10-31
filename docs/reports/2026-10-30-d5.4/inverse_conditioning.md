# Restricted coefficient-map inversion

The full validation inverse completed under committed policy `04b5804` and
runner `34e65eb65dc055d029b72456c22299ab48a51e62`: 2,187 arm/noise/case records,
5,347.03 seconds, peak CUDA allocation 0.706 GiB. The conditional baseline
substitution diagnostic is running separately. No model was trained.

## Validation result with true baseline and direction

| Noise fraction | Scene-balanced relative L2 | Direction cosine | Correlation | Converged Chromatic cases | Passing complete strata |
|---|---:|---:|---:|---:|---:|
| 0 | 0.074716 | 0.996265 | 0.997223 | 243/243 | 27/27, each 9/9 scenes |
| 0.001 | 0.951518 | 0.396527 | 0.407341 | 243/243 | 0/27 |
| 0.005 | 1.445389 | 0.153774 | 0.157111 | 243/243 | 0/27 |

These are descriptive means: average the 27 fixed conditions within each original
scene, then the nine scenes. They are not 243 independent samples. CAVE and KAIST
remain separate in `restricted_inverse_summary.json`, along with worst cases;
every scalar case is in `restricted_inverse_results.json` and every complete
stratum in `diagnostic_gates.json`.

At zero noise, the known-direction coefficient map is recoverable by this
regularized estimator throughout the tested construction family. Frozen A6 fails
the same perturbation-family response gate. This supports a limited noiseless
recoverability gap, with the Oracle's true baseline/direction and float64
computation explicitly privileged. A6's noiseless input is outside its training
noise distribution, so the gap does not isolate neural capacity as the cause.

At both nonzero noise levels, numerical convergence is complete but the recovery
gate fails. At the target 0.5%, the mean error is worse than returning zero. Thus
the result is a detection-to-map-recovery gap for this fixed estimator and prior,
not a proof of information-theoretic impossibility. The finite regularization
grid and three-scene CAVE-only development selection remain material limitations.

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

`numerical_budget_audit.json` explicitly marks every training-grid case with
`budget_censored`. The initial unpreconditioned grid has 69/72 censored cases;
the final grids have 6/72, 12/72 and 12/72 at zero, 0.1% and 0.5% noise. Only
fully converged candidates were selected, and all 729 Chromatic validation solves
converged. The audit also records an earlier pilot metadata limitation: a config
digest taken at completion can reflect a subsequent edit rather than the loaded
100-iteration configuration. Original pilot bytes remain unchanged; they are
not used as final validation evidence.

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

## Conditional baseline-substitution diagnostic

The first-level scalar Oracle knows the complete true template and baseline;
the second level knows only the true baseline and the chosen spectral direction,
with no mask or true coefficient initialization. All 27 zero-noise strata passed,
so the predeclared third level evaluates all of them with the fixed Chromatic
seed 20261021 estimate from an additional clean reference exposure in place of
the true cube baseline. It keeps the direction, regularization and solver budget.

This is a baseline-substitution sensitivity test with two privileged observations,
not deployable single-shot reconstruction. In particular, because the clean
reference measurement is granted, directly subtracting it in measurement space
would already supply the same residual as the true-baseline Oracle (apart from
floating-point rounding). Therefore any failure after inserting the A6 baseline
would identify avoidable baseline-estimation bias in this diagnostic pipeline;
it would not establish a fundamental failure caused by removing cube ground truth.
This qualification follows from the design before the third-level scientific
results are inspected. No favourable conditions or alternate baselines are selected.
