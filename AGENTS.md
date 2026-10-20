# Research implementation rules

This independent repository owns offline measured-PSF simulation, dataset protocols,
small spatial-spectral unrolling prototypes, and task evaluation. Current D2 scope
is fixed-region Face / Non-face, not general face detection or identity recognition.

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

N1 exposes spatial spectral maps, global 21-point curves and reprojections. Small
D2 unrolling is explicitly authorized, not a general training platform or a
PSNR-driven high-resolution reconstruction target. No hardware, mask optimization,
network services, schema registries or C2/C3 placeholders. Existing initialization
placeholders do not require implementation.

Run no-hardware tests in an explicit independent research environment. Commit here
before any validated Meta pin change. Never deploy this environment to acquisition.
