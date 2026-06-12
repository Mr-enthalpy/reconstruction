# Cross-Repository Boundary

`reconstruction` consumes published outputs from upstream repositories. It does not generate those products.

## Inputs

From `LCD_forward`:

```text
OperatorHandoff
  -> forward operator A
  -> adjoint operator A*
  -> operator metadata
  -> sparse peak-cluster / shift-patch operator package
  -> OTF / H-matrix diagnostics
  -> mask sequence identity
  -> operator provenance
```

From `optic_system`:

```text
MeasuredResponseHandoff
  -> target measurements
  -> calibration metadata
  -> exposure/frame metadata
  -> wavelength metadata when available
  -> acquisition provenance
```

From `lcd_mask_families`:

```text
MaskInstanceSpec / MaskSequenceSpec identity
  -> only when needed for provenance, conditioning analysis, or reporting
```

`reconstruction` may reference mask identity, but it must not render LCD masks as a primary responsibility.

## Outputs

Conceptual output categories:

```text
ReconstructionResult
EvaluationReport
ReconstructionDrivenProposal
```

These are conceptual handoff categories only. Final external schemas should not be defined before producer and consumer contracts stabilize.

## Non-Ownership

This repository must not own hardware acquisition, mask-family generation, mask rendering, LCD mask-to-operator modelling, peak-cluster extraction from raw PSF surveys, H-matrix computation as an `LCD_forward` source product, or final external handoff schemas before contracts exist.
