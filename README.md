# reconstruction

`reconstruction` is a bounded operator-consuming reconstruction platform for static-scene, multi-exposure, multi-band coded spectral reconstruction under published LCD-forward operators.

The repository consumes `OperatorHandoff` outputs from `LCD_forward` and `MeasuredResponseHandoff` outputs from `optic_system`. It does not own hardware control, LCD mask families, operator modelling, or raw capture generation.

The mainline inverse problem is:

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

`T` is a coded measurement index, not video time.

Deep unrolling is the preferred learned method route. Classical and variational baselines are included as first-class diagnostic references because they diagnose the operator and establish reconstruction reference levels.

This repository is currently architecture-only. It contains no solver implementation, network implementation, dataset loader, or final external handoff schema.

## Scope

This repository owns inverse-problem formulation, reconstruction baselines, physics-guided deep unrolling designs, learned or variational priors, evaluation protocols, metrics, diagnostics, ablation runners, and comparison reports.

It must not own camera control, LCD display control, TLS control, synchronized acquisition, raw capture generation, mask-family definitions, mask rendering, LCD mask-to-operator surrogate modelling, peak-cluster extraction, operator package generation, or canonical operator diagnostics.

## Initial Phase

Phase 0 initializes the repository shape only:

```text
- architecture documentation
- placeholder source package
- placeholder handoff directories
- import-only test
- no reconstruction algorithms
- no final schemas
- no external repository imports
```
