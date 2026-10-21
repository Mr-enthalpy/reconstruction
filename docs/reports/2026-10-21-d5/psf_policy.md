# D5 G1 PSF policy

Date: 2026-10-21. Source: `spectral21_20261018_reset01`, acquisition commit
`18dd598852b1d44ea80b4f6d53e1d2d5656f9e9b`. Raw measurements remain read-only.

## Primary mode: S, signed shape-only

For each nominal band 450-650 nm / 10 nm:

1. Read `roi_extent_review/roi1024_{nm}.npz` and its metadata.
2. Subtract the matched pre/post dark mean over the same 1024 ROI.
3. Estimate a border background from the outer 40-pixel ring and subtract it.
4. Take the central 512 x 512 sensor ROI.
5. Normalize by the signed sum.
6. Integrate 8 x 8 sensor blocks to a 64 x 64 simulation kernel.
7. Use unit spectral weights, zero-boundary full convolution and integer-centred
   same crop.

This is a traceable shape-only numerical model. It is not absolute spectral
throughput, not a Bayer RAW response and not a nonnegative optical PSF.

Recorded properties:

| Item | Value |
|---|---|
| Bands | 21, 450-650 nm |
| Signed negative fraction | 0.745-0.874 |
| Integrated kernels | 21 x 64 x 64, each sum 1 |
| 99% absolute-energy radius | median about 32 simulation pixels / 256 sensor pixels |
| 512-support vs central 512 of 1024-support | relative L2 about 5.8e-8 after normalization |

## Critical G1 finding: integration reduces band separability

The 8 x 8 block integration is numerically exact for the chosen simulation scale,
but it substantially increases similarity between wavelength kernels:

| Comparison | Mean off-diagonal correlation |
|---|---:|
| Signed 512-support kernels | 0.521 |
| Integrated 64 x 64 kernels | 0.921 |
| Achromatic mean kernel, 512 support | 0.735 |
| Achromatic mean kernel, integrated | 0.961 |

This is measured-kernel evidence, not a network result. It means the current 8:1
spatial scale may remove much of the wavelength-dependent structure before D5
training. A D5 model that performs well under this operator is not evidence of
strong intrinsic wavelength coding. The 512/1024 support comparison is stable,
so this is primarily an integration/scale issue, not a crop-origin issue.

## Sensitivity mode: P, nonnegative projection

Mode P clips negative values, then renormalizes each integrated kernel. It is a
sensitivity analysis only:

- Relative L2 change from Mode S: 0.0010-0.0176 by band.
- Absolute energy removed by clipping: 0.1%-3.4%.
- Negative fraction after clipping remains 0.26-0.75 because clipping is applied
  to the integrated pixels, not to a reconstructed continuous PSF.

Mode P is not used for D5 primary training or test metrics. It does not establish
a physically valid nonnegative PSF. A future calibrated estimate needs an explicit
background/noise model and a separate versioned analysis.

## Boundary and halo

The operator uses zero-padded full convolution followed by the central crop. The
99% energy radius is about 32 simulation pixels, so a 256 x 256 scene loses some
outer-context support within 32 pixels of each border (12.5% of width). Training
patches must retain this halo; do not crop observations or truths independently.
The 1024-support run is an input-physics sensitivity test, not a new calibration.

## Data-side consequences

- CAVE uses direct 450-650 nm bands from the local 31-scene set, `/65535`, no
  per-scene normalization.
- KAIST official data carries 28 samples and no embedded wavelength vector. D5
  assumes uniform 28 samples over 450-650 nm and linearly interpolates to 21 bands.
  This is a declared assumption, not verified metadata.
- KAIST selected test scenes contain small negative values; D5 clips them to zero
  and records the pre-clip negative fraction.
- `watercolors` is an 8-bit grayscale-in-RGB CAVE scene with constant alpha 255.
  It is a separate quantization stress test, not pooled into the 16-bit CAVE claim
  without disclosure.
- The official `cave_1024_28` set is not used for primary D5 training because its
  205 files lack an original-source-scene map and the public reference treats them
  as independent inputs. Using them would risk group leakage.

## Decision

G1 is numerically passed for the primary Mode S operator. D5 may proceed to G2
model and pilot work, but no D5 result may be described as calibrated RAW, absolute
radiometry, or proof that the 8:1 integrated operator preserves useful wavelength
coding. The G1 separability finding must be carried into the G4 interpretation.
