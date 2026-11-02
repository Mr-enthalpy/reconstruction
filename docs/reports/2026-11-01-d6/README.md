# D6 — computational-imaging prototype and migration preparation

Started 2026-11-01 from completed D5.4 commit
`aee8675e4554e320e4992395325024cdffe6b0c7`, with a clean member worktree.
Branch: `codex/d6-g4-prototype`. The user explicitly authorized G4 and prototype
integration, retiring local spectral-difference recovery as an entry condition.
Prior negative evidence remains unchanged in the D5.4 archive.

## Current execution status

- G4 protocol prepared before test-array access; nine existing checkpoint
  identities and all 14 scenes fixed. No new N1 training or model selection.
- Exact inference export completed remotely. Local FFT regeneration was rejected
  by the byte-level physics gate; original checkpoint buffers are being retrieved.
  This is an engineering portability issue, not a test outcome.
- R/N1/C simulation adapter, group-isolated downstream feasibility and migration
  contracts are in development. No same-Bayer-RAW or downstream gain claim yet.

`g4_protocol.json` fixes inputs, models, noise, metrics, descriptive seed/scene
stability and one-time access semantics. Its Git commit precedes test evaluation.
The test byte hash was established without decoding the cube. Once G4 is complete,
these scenes are a consumed reconstruction holdout, not a fresh future test.

The LCD-near-pupil prototype is not optically equivalent to a real UDC. Current
work tests methods and data flow under an approximate measured-PSF simulation.
Three-channel scalar proxies and privileged independent RGB must be labelled
separately from future Bayer RAW. Downstream benefits, if observed, are empirical
representation benefits and do not establish extra physical spectral information.
