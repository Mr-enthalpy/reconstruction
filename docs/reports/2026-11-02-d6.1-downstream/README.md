# D6.1 downstream data and protocol — 2026-11-02

Decision: **READY_FOR_DEVELOPMENT**. Development ready: **true**.
Final test ready: **false**; `final_holdout_status = not_available`.
Started from `27784b9fd600421b1007b83e97ec2de00bc69808` on the dedicated D6.1 branch.
No classifier training, GPU training, image review or G4 cube access occurred.

| Source | Role | Face ROIs | Non-face ROIs | Conservative groups |
| --- | --- | ---: | ---: | ---: |
| CAVE | development_train | 3 | 15 | 16 |
| CAVE | development_val | 0 | 5 | 5 |
| CAVE | development_test | 0 | 6 | 6 |
| UWA | development_train | 39 | 39 | 28 |
| UWA | development_val | 10 | 10 | 7 |
| UWA | development_test | 9 | 9 | 6 |

These are existing historically exposed development groups, not verified fresh
independent groups. Native full-field files are available for all admitted ROIs;
the complete inventory also retains excluded/review-only D1 and CAVE candidates.
See [source/exposure audit](source_exposure_analysis.md) for distinct ROI,
acquisition and subject/family counts and missing/unknown evidence.

Face semantics remain consistent: recognizable photographed faces are positive.
The three original CAVE positive coordinates and historical Agent reviewer
status are preserved privately, with both scenes in one train-only family.
There are **zero confirmed hard negatives**; legacy semantic acceptance is not
difficulty annotation. Ambiguous source records remain queued, not relabelled.

All old D1/D2/D3 task tests and all14 consumed G4 scenes are ineligible for a new
final set. Six CAVE legacy task-test ROIs overlap A6 training; CAVE validation
and task-test have no positive group. UWA is mapped to original subject/session
evidence, but common background and cross-ID person independence remain unknown.

The next agent may implement C development training under the frozen
[classification protocol](classification_protocol_v1.md), after the declared
loader/feature/hash checks. Its result must be labelled historical development
feasibility; final-test, detector and hard-negative performance are not ready.
No new external data is assumed. [Data gaps](data_gap_report.md) specify minimum
new sources and the acquisition/identity/calibration/isolation fields required
for future Axon20 data. The device has not been used or validated in this phase.

Full per-sample manifest, annotations, review queue and local paths remain in
the private output directory. Published JSON contains aggregate statistics,
protocols and digests only. Inputs and all D2–D6 reports are unchanged; Meta
submodule pins are unchanged. See `input_inventory.json` and `protocol_lock.json`.

## Reproduce without opening images

Use the reconstruction research environment, `PYTHONPATH=src` and CPU only.
`python -m experiments.d6_1.audit_sources --help` lists required explicit inputs.
Provide D1 manifest/split/review/oracle/crop-rejection records, the UWA source
inventory/archive/receipt, CAVE inventory/review/full-field directory, D2/D3
manifests, D4 manifest/UWA diagnostic metadata, the N1 manifest, G4 protocol/access
ledger and D6 audit. Both output directories must be new. Point `--output` at
a staging report directory and `--private-output` at a Git-ignored private path.
Do not supply any G4 array file. No local data root is hardcoded in the modules.

For a frozen run, use `python -m experiments.d6_1.verify_frozen --report REPORT
--manifest PRIVATE_MANIFEST --lock-sha256 RECORDED_LOCK_HASH`. Record the lock
hash from the committed report; do not recompute it to accept a changed lock.
The checker verifies all public protocol bytes, implementation source hashes
and private sample bytes. A changed sample/protocol requires a new explicit
version; rerunning the builder cannot overwrite an existing output directory.

Verification:29 new synthetic metadata tests cover all15 requested failure modes;
the complete CPU/no-hardware suite passed173 tests. Native-file and manifest
hash checks, deterministic replay and publication privacy scans are recorded
in the accompanying validation receipt. No research classifier was trained.
