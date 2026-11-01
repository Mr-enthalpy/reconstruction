# D5.4 — spatial-spectral differential recoverability

Started 2026-10-30 from reconstruction main `ef07c1b0befbf28d6938c4e67006c8b4f5f8e37d`.
The member working tree was clean. Development branch: `codex/d5.4-spectral-recoverability`.
Meta pins and the acquisition member are outside this change.

## Current status (completed diagnostics; finalized 2026-11-01)

- S0: published SNR strata recomputed; all nine checkpoints' original bytes,
  selection epochs and physics buffers verified. See `d5_3_evidence_amendment.md`.
- S1: training-only PCA completed after train/manifest hash checks; 243 validation
  constructions completed. Diagnostic labels are not proof of in-distribution membership.
- S2: full operator, known-template scalar Oracle and nine-model frozen-network
  sweeps completed. No frozen-model condition group passes the recovery gate.
  See `spectral_response_analysis.md` and the scalar case/stratum JSONs.
- S3: training-only noise-specific regularisation selection completed and committed
  before validation inversion. Full true-baseline inversion completed: all 27
  noiseless strata pass in 9/9 scenes; no noisy stratum passes. The predeclared
  baseline-substitution sensitivity diagnostic completed all 243 noiseless cases:
  no recovery passes, with 81 cases explicitly budget-censored.
- S4: not executed; a noiseless privileged gap exists, but target-noise recovery
  and a causal neural-capacity bottleneck have not been demonstrated.
- G4: not opened; no test-loading interface exists in D5.4.

The local no-hardware suite passed 128 tests using the designated research environment.
Earlier sandbox runs failed on temporary-directory ACLs; those are not scientific failures.

The protocol is `experiments/d5_4/configs/diagnostics.json`. Use module entry points
with repository `src` on PYTHONPATH. All data and output locations are explicit CLI
arguments. Original data, checkpoints, machine configuration and run logs remain outside Git.

Known-template detection succeeds for some conditions. A true-baseline,
known-direction coefficient map is recoverable without noise, but fails the
registered gate at 0.1% and 0.5% noise with the fixed estimator. Frozen A6 largely
produces incorrectly directed or noise-driven differences at all three levels.
This establishes a conditional distinction between detection and recovery; it
does not establish task-level spectral benefit or a neural-capacity cause.

Replacing the true baseline with A6's estimate introduces measurement bias whose
median norm is 83.25 times the perturbation signal. This two-exposure sensitivity
test fails even in its 162 converged cases. It is not a single-shot impossibility
result: subtracting the granted clean reference directly would avoid that bias.

Read `scientific_decision.md` for the noise-conditional B/C interpretation and G4
recommendation, `inverse_conditioning.md` for all three prior levels, and
`execution_history.md` for historical failures versus completed work. The artifact
manifest inventories final report bytes, execution commits and shared provenance.
S4 remains unexecuted; its absence is a research decision, not unfinished training.
