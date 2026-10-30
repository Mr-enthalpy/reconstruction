# D5.4 — spatial-spectral differential recoverability

Started 2026-10-30 from reconstruction main `ef07c1b0befbf28d6938c4e67006c8b4f5f8e37d`.
The member working tree was clean. Development branch: `codex/d5.4-spectral-recoverability`.
Meta pins and the acquisition member are outside this change.

## Current status (in progress)

- S0: published SNR strata recomputed; nine server checkpoint paths found. Original
  checkpoint byte hashes are being verified. See `d5_3_evidence_amendment.md`.
- S1: training-only PCA completed after train/manifest hash checks; 243 validation
  constructions completed. Diagnostic labels are not proof of in-distribution membership.
- S2: implementation ready; full operator and frozen-network sweep pending.
- S3: seven mathematical tests passed; one training-scene performance probe completed.
  Regularisation selection and formal inversion are pending. Probe outputs are not results.
- S4: not executed; no demonstrated capability gap yet.
- G4: not opened; no test-loading interface exists in D5.4.

The local no-hardware suite passed 120 tests using the designated research environment.
Earlier sandbox runs failed on temporary-directory ACLs; those are not scientific failures.

The protocol is `experiments/d5_4/configs/diagnostics.json`. Use module entry points
with repository `src` on PYTHONPATH. All data and output locations are explicit CLI
arguments. Original data, checkpoints, machine configuration and run logs remain outside Git.

No D5.4 recovery claim is established by implementation, synthetic tests, or the benchmark.
