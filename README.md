# reconstruction

Offline research for measured-PSF spectral sensing in a UDC-like laboratory system.
D4 prioritizes general spatial-spectral reconstruction independent of face tasks.
R (RGB recovery/proposals) and N1 consume the same raw observation in the intended
integrated system; C independently consumes their aligned outputs. Reconstruction
quality is a necessary explanatory foundation, not an expendable intermediate.
No hardware is operated and
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
This architecture was adopted after D2; D3 execution is documented below. D2 results remain the earlier low-rank pilot.

## D3 execution status

The [D3 report](docs/reports/2026-10-20-d3/README.md) records an actually trained
full 21-channel spatial-state PGD model and independent Oracle / Reconstruction /
Frozen-head classification. [Commands](experiments/d3/README.md) reproduce the
bounded pilot with authorized local inputs. Source bias remains and the task score
matches achromatic/intensity controls; general recovery and spectral task gain
are not established. Historical D2 results remain unchanged.

## D4 active work

See the [D4 status](docs/reports/2026-10-20-d4/README.md) and
[run instructions](experiments/d4/README.md). The general scene loader, full-state
PGD and training entry do not consume classification labels or face ROIs. Two CUDA
training runs and independent diagnostics are complete. The selected model still
has substantial local spectral/high-frequency error; general quality acceptance
and downstream C training remain deferred.
Scalar physics does not establish calibrated same-RAW RGB+spectral recovery.
