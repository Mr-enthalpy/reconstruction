# Method principles

Use explicit forward and adjoint operators inside a spatial-spectral Deep Unrolling
reconstructor. The first baseline exposes X_hat [B,21,H,W], its spatial mean and
measurement reprojection. Declare the spatial resolution and valid domain.
Do not impose a single global spectral output or a low-rank bottleneck before
establishing the scene reconstruction baseline.

Train N1 without task labels or classification loss. Freeze it for downstream
classification. This separates reconstruction fidelity from evidence of spectral
task gain. Later compression or joint learning requires an explicit comparison.
Keep model capacity proportional to independent samples; do not tune on test data.

Preserve source, subject, acquisition and scene isolation. Both labels should be
reviewed within each source where possible; report remaining source/label imbalance.
Use shared band geometry and physically stated brightness assumptions. Source
normalization is not radiometric calibration. Preserve achromatic PSF controls.

The existing D2 low-rank pilot and its scores are retained as historical results.
They do not validate the revised baseline, absolute radiometry, general face
recognition or information gain beyond RGB.
