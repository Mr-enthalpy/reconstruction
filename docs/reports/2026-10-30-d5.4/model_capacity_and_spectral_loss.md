# S4 — not executed

S2/S3 have not yet established a recoverability gap. No new A6, A-large or C6 training
is justified by the current implementation or performance probe.

Source inspection: A6 retains 21 channels, applies physical forward/adjoint updates,
and a shared CNN proximal; the D5.3 loss configuration uses spatial-gradient 0.2,
spectral-shape 0.05 and data-consistency 0.02. This is a code observation, not an
experimental diagnosis that any coefficient causes spectral failure. Training-gradient
and local-spectrum diagnostics have not been performed. Inference below the training
0.5% RMS noise level is a noise-distribution shift.
