# Architecture

`reconstruction` is the inverse-problem and evaluation repository for static-scene, multi-exposure, multi-band coded spectral reconstruction under published LCD-forward operators.

It is not a single-network implementation repository. It is a bounded reconstruction platform whose main research route is physics-guided deep unrolling, with classical and variational baselines as first-class diagnostic references.

## Mainline Problem

```text
Given:
  y = multi-frame coded measurements of one static scene
  A = published forward operator from LCD_forward
  A* = matching adjoint operator
  metadata = wavelengths, mask sequence identity, exposure/frame identity, calibration provenance

Recover:
  x = multi-band spectral scene
```

Preferred tensor convention:

```text
x: [B, L, H, W]
y: [B, T, 1, H, W]
```

`B` is batch index, `L` is spectral band index, `T` is coded exposure or measurement-frame index, and `H, W` are spatial dimensions.

`T` is coded measurement index, not video time.

## Operator Contract Direction

The reconstruction stack should consume operator APIs rather than dense PSF tensors:

```python
A.forward(x)
A.adjoint(y)
```

Dense PSF materialization is reserved for small debug cases, parity tests, legacy comparisons, and baseline sanity checks. It must not become the primary data contract.

## Package Layout

The active package is `src/lcd_reconstruction/`.

The package is intentionally skeletal during initialization. Subpackages document ownership boundaries and future intent without committing solver APIs, model classes, loaders, metrics, or external schemas.
