# Method Principles

The preferred learned method family is physics-guided deep unrolling.

A reconstruction stage should preserve this structure:

```text
data consistency:
  uses A and A*

learned prior / proximal:
  uses a learned spatial-spectral prior

diagnostics:
  reports residual and reconstruction quality
```

Conceptual form:

```text
x^0 = A*(y) or classical warm start

for k = 1..K:
    z^k = data_consistency_update(x^k, y, A, A*)
    x^{k+1} = learned_or_variational_prior(z^k, metadata)
```

The repository should avoid black-box `frames -> spectra` networks as the mainline. Such models may appear only as explicit baselines.

## Multi-Band Principle

The repository must not be hard-coded to three spectral bands. Mainline modules should support arbitrary `L` and should use terms such as `spectral`, `multi_band`, `wavelength_grid`, and `band_index` instead of RGB-specific names.

## Multi-Frame Principle

The repository should support multiple coded exposures of the same static target. `T` is the number of coded observations, exposures, or measurement frames. It is not video time by default.

Allowed future concepts include frame-index conditioning, mask-sequence conditioning, multi-exposure fusion, and operator-aware frame weighting.

Motion compensation, optical flow, temporal consistency losses for moving scenes, video restoration priors, and dynamic-scene assumptions are not mainline concepts.
