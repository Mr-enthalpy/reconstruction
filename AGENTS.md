# Research implementation rules

This independent repository owns offline measured-PSF simulation, dataset protocols,
general spatial-spectral reconstruction, and independent task evaluation.
Current D4 priority is standalone reconstruction quality, not Face / Non-face
classification. N1 data eligibility and model selection must not depend on class
labels, face proposals, class balance or downstream scores.

Preserve originals and frozen subject/acquisition/scene splits. Keep raw/derived
faces, previews, models and machine configuration outside Git. Use explicit paths;
never import sibling repository internals. The active package is lcd_reconstruction
and must import without optional training dependencies.

Treat measured kernels as shape-only approximations until radiometry, Bayer,
spatial scales and field dependence are validated. Test an explicit forward/adjoint
pair. Numerical identity is not optical validation. Exposure-dependent DN is not
absolute spectral throughput.

All bands share augmentation geometry. Fit bases/models only on training groups;
no test selection. Separate original and augmented counts. Report source confounding.
Three narrow bands are not calibrated RGB. Preserve achromatic-PSF controls.

N1's approved baseline is task-independent spatial-spectral reconstruction with
primary output [B,21,H,W], plus a mean spectrum, reprojection and quality diagnostics.
Train it without classification loss; freeze it for independent downstream task
comparisons. Low-rank compression/global taps are later ablations, not baseline
requirements. The existing D2 low-rank pilot remains historical evidence.

Review CAVE real-face ROIs as positive candidates as well as negatives; do not infer
labels from source. Keep scene groups intact and version revised annotations.
Cleaning/augmentation must preserve spectra and be checked for remaining source cues.
No hardware, mask optimization, services, schema registries or C2/C3 placeholders.
Existing initialization placeholders do not require implementation.

R (RGB recovery and mature face proposals) and N1 consume the SAME raw observation
in a future integrated experiment. They run independently; proposals select
downstream C inputs, not N1's training targets or field of view. Scalar shape-only
N1 simulations cannot establish same-RAW RGB fusion. Oracle RGB is explicitly
privileged. Do not crop measured observations without checking PSF support.
Reconstruction is the explanatory foundation: spatial and spectral quality must
be measured seriously even though reconstruction metrics are not the ultimate
research story. Preserve an untouched final integration holdout before training.
Do not expose its truth through validation, normalization fitting or augmentation.

Run no-hardware tests in an explicit independent research environment. Commit here
before any validated Meta pin change. Never deploy this environment to acquisition.
