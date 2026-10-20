# reconstruction

Offline research for measured-PSF spectral sensing in a UDC-like laboratory system.
D2 studies fixed-region Face / Non-face with public hyperspectral scenes, explicit
approximate forward/adjoint operators and a small spatial-spectral unrolled estimator.
Task validity matters more than reconstruction PSNR. No hardware is operated and
no Meta checkout or sibling Python package is required.

The initial multi-exposure operator-handoff architecture was a proposal. D2 uses
existing dense measured ROI PSFs through explicit operators without a new handoff
framework. See [current architecture](docs/architecture.md) and [rules](AGENTS.md).

Keep datasets and outputs outside Git. UWA restrictions apply to raw and derived
faces. CAVE positive and negative candidates require content review; cross-source classification is not
general face recognition evidence. Optional experiment dependencies belong in an
independent research environment, never in the acquisition environment.

See the [D2 evidence report](docs/reports/2026-10-20-d2/README.md) and
[reproduction commands](experiments/d2/README.md). The bounded pilot runs, but
source confounding and achromatic controls do not establish a spectral task gain.

The approved next baseline separates **scene reconstruction** from **spectral task
gain**: N1 exports a spatial-spectral matrix and is trained without classification
loss; a frozen N1 feeds an independent classifier. See [architecture](docs/architecture.md).
This revision is a design decision; D2 code/results remain the earlier low-rank pilot.
