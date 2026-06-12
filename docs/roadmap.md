# Roadmap

## Phase 0: Architecture Initialization

```text
- Create repository structure.
- Add README, AGENTS, and architecture docs.
- Add placeholder source package.
- Add handoff placeholder directories.
- Add import-only test.
- Add no solver implementation.
```

## Phase 1: Handoff Contract Alignment

```text
- Align with LCD_forward OperatorHandoff once it exists.
- Align with optic_system MeasuredResponseHandoff once it exists.
- Document expected fields.
- Do not implement fragile loaders against unstable files.
```

## Phase 2: Classical Operator-Consuming Baselines

```text
- Add A* y backprojection.
- Add ridge / Tikhonov.
- Add CG / LSQR.
- Add residual diagnostics.
```

## Phase 3: Variational Baselines

```text
- Add spectral smoothness.
- Add TV.
- Add low-rank spectral prior.
```

## Phase 4: Deep Unrolling Mainline

```text
- Add physics-guided unrolling.
- Use A and A* in each stage.
- Add learned prior / proximal modules.
- Keep architecture multi-band and multi-exposure.
```

## Phase 5: Evaluation and Reporting

```text
- Add benchmark runners.
- Add method comparison reports.
- Add ablations across operators, mask sequences, and reconstruction methods.
```
