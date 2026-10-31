# S4 — not executed

S2 is complete and negative for frozen-network differential recovery. S3 validation
coefficient-map inversion is still running. A recoverability gap has not yet been
assigned. A-large and C6-SFAT remain unexecuted; no A6 was retrained.

Source inspection: A6 retains 21 channels, applies physical forward/adjoint updates,
and a shared CNN proximal; the D5.3 loss configuration uses spatial-gradient 0.2,
spectral-shape 0.05 and data-consistency 0.02. This is a code observation, not an
experimental diagnosis that any coefficient causes spectral failure. Training-gradient
and local-spectrum diagnostics have not been performed. Inference below the training
0.5% RMS noise level is a noise-distribution shift.

## Actual training objective and update path

The D5 training entry imports `hsfaut.reconstruction_loss` for every arm, including
A6. It does **not** call the older MSE wrapper in `general_pgd.py`. The actual
intensity term is mean Charbonnier error, sqrt(error squared + epsilon squared)
minus epsilon, with epsilon 1e-3. The additional terms are:

- 0.2 times mean squared spatial-gradient error, averaged across x/y axes;
- 0.05 times mean one-minus-spectral-cosine over pixels with target spectral norm
  above 1e-3; this is not the angle in degrees used by the SAM evaluation metric;
- 0.02 times mean squared measurement residual divided by that training arm's
  fixed forward-RMS scale squared.

The loss supervises the full 21-channel cube and local pixel spectra. Code alone
does not support the claim that it rewards only broadband structure. There is no
independent per-band normalization in this loss. Cross-arm scalar losses remain
incomparable because the measurement normalization differs between training arms.

A6 initializes with A*(y)/L, performs six physical gradient updates with learned
bounded scalar steps, and uses one shared 21-to-64-to-21 convolutional proximal
module across stages. Its 3x3 convolutional layers mix all spectral channels;
there is no PCA output or forced low-rank state. A final physical update precedes
the full [B,21,H,W] output. `L` is the existing physical Lipschitz bound, not a
trainable spectral-direction preconditioner.

These observations identify quantities to inspect, not causes of failure. Actual
training-sample output-gradient decompositions and local spectrum errors have
not been run in this D5.4 stage. No spectral loss coefficient has been changed.
No architecture is declared capacity-limited from the response results alone.

Source references: `experiments/d5/train.py`,
`src/lcd_reconstruction/unrolling/hsfaut.py`,
`src/lcd_reconstruction/unrolling/general_pgd.py`, and the frozen
`docs/reports/2026-10-29-d5.3/seed_protocol.json`. The D5.4 branch has not modified
these training/model implementations.
