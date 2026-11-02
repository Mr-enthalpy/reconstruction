# D6 — computational-imaging prototype and migration preparation

Started 2026-11-01 from completed D5.4 commit
`aee8675e4554e320e4992395325024cdffe6b0c7`, with a clean member worktree.
Branch: `codex/d6-g4-prototype`. The user explicitly authorized G4 and prototype
integration, retiring local spectral-difference recovery as an entry condition.
Prior negative evidence remains unchanged in the D5.4 archive.

## Current execution status

- On 2026-11-02, G4 completed all 126 pairs (nine frozen models, 14 scenes),
  after protocol commit `f56192d34985db075972bbc6452cf271931c5a9f`.
  Exact original physics and every model-state tensor passed byte verification.
- Summary serialization failed on a NumPy integer after all scene metrics had
  been written. The original partial `g4_results.json` is preserved as evidence;
  use [g4_results_recovered.json](g4_results_recovered.json) for valid results.
  [Recovery audit](g4_serialization_audit.json) records no repeated inference and
  no reopened test cubes. The access ledger records completion after recovery.
- Independent R/N1 and aligned C interfaces passed a validation-scene smoke
  check. R is a scalar intensity proxy; C was an untrained wiring fixture.
  This is not classifier acceptance. See [receipt](pipeline_smoke.json).
- [Downstream audit](downstream_audit.json) found six legacy CAVE test ROIs
  overlapping N1 training scenes. A new independent task protocol is outstanding.
  [Migration requirements](migration_contract.md) distinguish present adapters
  from future calibrated RAW inputs. No hardware experiment was performed.

## Frozen G4 findings

Means first average 14 scenes within each seed, then the three seeds. These are
simulation reconstruction metrics, not independent physical spectral evidence.

| Arm | PSNR dB | SSIM | SAM degrees | Spatial-gradient relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Chromatic A6 | 27.43879 | 0.919969 | 12.73281 | 0.554347 |
| Mean-kernel achromatic | 27.14555 | 0.910409 | 12.84735 | 0.603541 |
| 550-nm achromatic | 27.04027 | 0.907035 | 12.86514 | 0.622271 |

SSIM, spatial-gradient error and measurement residual pass the frozen descriptive
stability rule against both controls. PSNR, SAM, cube error, mean-spectrum error
and high-frequency error do not. The +0.29325 dB PSNR difference against the mean
control has only 11/14, 12/14 and 14/14 agreeing scenes; it is direction with
limited stability, not a stable win. SAM's mean direction is favourable here but
does not support a spectral-recovery claim. Source-specific summaries, per-band
errors, all scene values and the three worst scenes per metric/model/source are
preserved in the recovered result. No case, including watercolors, was excluded.

The holdout is consumed. No checkpoint, threshold, architecture or noise setting
was selected from these results. D5.4's negative local-recovery findings remain
unchanged. Three seeds are the stability unit; the 126 rows are not independent
replications. This is neither a real UDC experiment nor downstream task evidence.

## Reproduction and next acceptance boundary

Verification on 2026-11-02: 144 no-hardware tests passed in the independent
research environment, including exact-state restoration, coordinate rejection,
full-field branch isolation, empty candidates and recovery from preserved metrics.
The first sandbox test attempts hit Windows temporary-directory ACL errors;
the successful full suite ran outside the sandbox with an explicit scratch path.
No hardware smoke check or scientific mechanism experiment was performed.

Use the owning repository's independent research Python environment and
`PYTHONPATH=src`. The original formal command used `experiments.d6.g4`,
`--mode formal --allow-frozen-test`, the committed protocol hash in
`g4_access.json`, and the exact bundle/physics/input hashes in that protocol.
Do not rerun formal access: the exclusive ledger is intentional. Summary recovery
is implemented in `experiments/d6/recover_summary.py` and reads saved metrics only.
`experiments.d6.smoke` opens validation data only and saves private arrays outside
Git; its two candidate boxes are fixed geometry, not detector output.

The next task experiment needs full-frame observations, independently sourced
candidate geometry, a versioned untouched final holdout, N1 exposure audits,
source-stratified positives/hard negatives and comparable classifier capacity.
The existing scalar proxy cannot be named RGB-only. A calibrated or privileged
independent RGB arm must be explicitly identified before such a comparison.

`g4_protocol.json` fixes inputs, models, noise, metrics, descriptive seed/scene
stability and one-time access semantics. Its Git commit precedes test evaluation.
The test byte hash was established without decoding the cube. Once G4 is complete,
these scenes are a consumed reconstruction holdout, not a fresh future test.

The LCD-near-pupil prototype is not optically equivalent to a real UDC. Current
work tests methods and data flow under an approximate measured-PSF simulation.
Three-channel scalar proxies and privileged independent RGB must be labelled
separately from future Bayer RAW. Downstream benefits, if observed, are empirical
representation benefits and do not establish extra physical spectral information.
