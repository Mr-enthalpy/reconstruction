# Real UDC migration acceptance

The present runnable adapter accepts one finite signed float scalar observation
`[1,1,H,W]`. R repeats that intensity into three channels; frozen N1 processes
the full field into `[1,21,H,W]`. Integer xywh regions use half-open full-field
pixel bounds. Only downstream C consumes region features. Empty candidate sets
retain the full reconstruction. Invalid coordinates, trainable N1 and nonfinite
outputs fail explicitly. No label input or classification gradient reaches N1.

Before introducing a real-device adapter, the acquisition owner must provide:

| Input | Required evidence and interpretation |
| --- | --- |
| Original Bayer RAW | CFA orientation, bit depth, active area, black/white levels, exposure, gain and immutable acquisition identity |
| Display state | Pattern/state and timing tied to the exposure; no inferred dynamic code |
| Spectral response | Wavelength grid, sensor colour response and radiometric units; distinguish throughput from normalized PSF shape |
| Wavelength PSFs | Calibration provenance, pixel centres, support, normalization, field positions, wavelength uncertainty and repeatability |
| Spatial mapping | Sensor/display/lens geometry, crop/resize transform and ROI alignment; preserve convolution support before cropping |
| Noise | Measured dark/read/shot behaviour by exposure/gain, clipping and invalid pixels; simulation sigma is not sensor DN |
| Evaluation separation | Independent calibration, training, validation and untouched test acquisitions with leakage audit |

The reconstruction owner then implements and tests a forward/adjoint pair for
that explicit sensor observation, connects the independent R and N1 branches,
and evaluates reconstruction before task claims. The trainer must consume the
declared operator and observation type; the current A6 weights are not assumed
compatible with Bayer RAW. Numerical adjoint and interface tests are no-hardware
checks. A hardware smoke check establishes execution, and a controlled scientific
experiment is separately required to establish physical or task information gain.

Model identity, observation hash, candidate source, coordinate convention,
intensity handling, completion/failure state and classifier provenance accompany
each output. Large arrays and private acquisition metadata remain outside Git.
No acquisition host deployment or optical-member change is part of this work.
