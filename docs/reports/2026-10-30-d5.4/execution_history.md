# Execution history and current evidence

Experiments ran on 2026-10-30/31; final retrieval and reporting finished 2026-11-01.
Historical obstacles below are not current scientific
failures, and completed outputs are not inferred merely from a job log or lease.

1. Started from clean member main `ef07c1b0befbf28d6938c4e67006c8b4f5f8e37d`.
   Meta/acquisition worktree deviations were inspected and preserved. No pin moved.
2. S0 recomputed D5.3 published statistics. Later, original bytes of all nine best
   checkpoints were checked and loaded through the existing PSF provenance gate.
   No D5.3 model was retrained. Its only changed report text is the qualified
   equivalence wording, with a dated amendment; measured JSON records are unchanged.
3. S1 used the training/validation archives after byte checks. Initial execution
   recorded uncommitted source hashes on the base revision. A later independent
   reproduction with committed fitting code matched the statistical quantities
   to floating-point precision, without replacing the original basis.
4. The first 100-iteration training-only inverse grids exposed nonconvergence.
   Their outputs and censoring are retained. Before validation inversion, the
   numerical implementation and uniform 1,000-iteration cap were frozen; the
   final noise-specific selections use only fully converged training candidates.
5. The frozen nine-model sweep ran source `aba3c10`, completed all 6,561 response
   records, and survived lease expiry in persistent storage. Its raw result was
   retrieved through the authorized job-log channel and checked against the
   remote SHA256. A log size limit required bounded chunks; no file-transfer
   or direct-SSH restriction was bypassed.
6. An attempted parallel inverse source-deployment command was rejected by
   automatic approval. That lease was released without running the proposed
   inverse jobs. A local uncompleted timing attempt is not a scientific result.
   The entire final inverse then ran locally under source `34e65eb`, completed
   all 2,187 records, and used identical frozen parameters throughout.
7. Basis byte hashes differed across Windows and the server because of CRLF/LF.
   The exact two byte variants were verified, not accepted by numerical tolerance.
   D5.4 JSON files are now tracked with `-text`, preserving their local evidence
   bytes. The earlier remote byte variant remains explicitly recorded. All
   previously recorded JSON semantic content was checked unchanged when enabling
   this attribute. D5.3 validation artifacts were not renormalized.
8. After the noiseless inverse gate passed, formal resource approval permitted
   downloading the pinned public source archive, checking its digest and extracting
   it into a new directory as separate steps. No security setting changed.
   The conditional baseline-substitution run uses source
   `037d20664ad144e9f503bcfbb2ac75928ca7bed2`, a fixed seed/checkpoint, every passing
   stratum and three fixed case partitions. Its one-case benchmark was 24.03 s,
   peak CUDA allocation 850,962,944 bytes; it was not used to select scientific
   conditions. Final completion is determined by verified full result files and
   the coverage/provenance checks in `summarize_reduced.py`.
9. On 2026-11-01 the original lease was expired but all three partitions had
   completed 81 cases. A CPU-only lease recovered the existing result files;
   all three SHA256 values matched and the reducer verified all 243 expected
   scene/condition keys. The CPU lease was released immediately after recovery.
   No experiments were rerun. There are 162 converged and 81 budget-censored
   solves, with no passing cases. Current evidence is complete; there is no
   remaining resource or file-retrieval blocker.

The frozen implementation config's historical `status` string says
`implementation_protocol_not_yet_executed_on_research_data`; it describes its
creation state, not current execution. The shared input helper also retains a
legacy test-policy string mentioning D5.3. These captured fields are preserved
for provenance. The enclosing result statuses, this history and the D5.4 README
give actual execution status; D5.4 never accesses the test archive. The final
regularization policy supplements the config's original selection wording with
the pre-validation all-cases-converged requirement. Neither metadata correction
retroactively changes the executed numerical configuration.

Local observed runtime: Python 3.13.5, torch 2.7.1+cu128, NumPy 2.5.4, SciPy 1.18.1.
The current baseline-substitution container was observed as Python 3.11.13,
torch 2.7.1+cu128, NumPy 2.3.2, SciPy 1.17.1. These latter observations are not
retroactively claimed as a separately captured version record for the earlier
frozen-network container. The local no-hardware suite passed 128 tests; pytest
was not claimed to run inside the remote container.

Private resource configuration, detailed job transport logs, raw archives,
checkpoints and full numerical histories remain in authorized ignored storage.
Only curated scalar evidence, source identities and reports are versioned.
No frozen test archive was loaded, copied, uploaded or downloaded. No new model
training, acquisition-code change or Meta integration was performed.
