# D5.4 diagnostic entry points

Use the independent research Python environment, with repository `src` on
`PYTHONPATH`. All archive/model paths are supplied by CLI. No entry point accepts
the frozen test split. No entry point trains a neural model.

The report is `docs/reports/2026-10-30-d5.4/`. Its execution records identify
different historical source commits; running today's source is a new diagnostic,
not a retroactive claim about those executions. Use a fresh output directory.
Do not overwrite prepared data or published experiment results.

## Main sequence

1. `evidence_audit` recomputes the published D5.3 report evidence without loading
   original datasets or weights. Original checkpoint-byte verification is separate
   and recorded in `checkpoint_registry.json`.
2. `natural_spectral_basis` accepts `--prepared --config --protocol --output` and
   fits only training spectra. Keep the original basis bytes for subsequent stages.
3. `noise_sweep --mode natural` evaluates construction; `--mode operator` evaluates
   full forward identities, frequency diagnostics and known-template scalar Oracles.
   Both accept `--prepared --kernel-file --basis --config --protocol --output`.
4. `inverse_conditioning` accepts the same paths plus `--noise-fraction` and
   `--device`. The recorded training-only selection is frozen in
   `regularisation_policy.json`; never select or expand the grid from validation.
5. `noise_sweep --mode inverse --regularisation-policy <policy>` evaluates all
   fixed validation cases with the true baseline and known direction. It does not
   receive a true spatial map as initialization.
6. `noise_sweep --mode frozen --run-dirs <nine-runs> --checkpoint-provenance <registry>`
   evaluates the existing best checkpoints, without selecting them again.

`run_sequence` is a thin convenience wrapper for the same explicit data/run/output
roots. `--code-commit` records the pinned source revision when running a Git
tarball. Its `inverse` stage uses the report's frozen noise-specific policy.
`--shard-count`/`--shard-index` partition deterministic case numbers; they do not
change the seed, solver cap or scientific conditions. No scheduler is implemented.

## Reduction and gates

- `compact_results --input <raw> --kind operator|oracle|inverse|frozen --output <new>`
  preserves scalar case values in a columnar table. Source SHA256 and provenance
  are retained; full CG histories stay with the hashed raw result.
- `decision_summary --operator <raw> --inverse <raw> --frozen <raw> --basis <basis>
  --output <new>` checks completed runs, physical identity, original-scene coverage
  and the fixed recovery gate. `--basis` can prove exact LF/CRLF transport
  equivalence; it does not permit numerical changes or arbitrary hash aliases.
- `summarize_frozen` keeps seeds separate and provides CAVE/KAIST and full-condition
  summaries. `summarize_d5_4` is a generic per-stratum scalar reducer.
- `merge_inverse` accepts only complete, mutually compatible fixed partitions.
  The completed D5.4 true-baseline sweep ultimately ran without partitioning;
  an earlier attempted remote deployment was blocked and its lease released.

## Conditional baseline substitution

`run_sequence --stage reduced-benchmark` runs one timing/memory probe, labelled
benchmark-only. `--stage reduced` runs the predeclared conditions that pass the
true-baseline inverse gate, using the fixed Chromatic seed 20261021. The direct
`reduced_oracle` entry also accepts explicit input, policy and checkpoint paths.

This diagnostic grants an extra clean reference exposure and replaces the true
cube baseline with its A6 estimate. It is not single-shot deployment. Directly
subtracting the clean reference measurement would already supply the ideal
residual, so failures diagnose avoidable baseline-substitution bias. They do not
establish a fundamental consequence of removing cube ground truth.

`summarize_reduced --inputs <all-raw-partitions> --inverse <compact-inverse>
--output <new>` verifies exact coverage of every gated scene/condition and matching
source provenance, then retains all failures and budget censoring. Do not treat
one completed partition as a complete scientific result.

All final comparisons remain validation-only. S4 and G4 are separate decisions;
script completion never authorizes new training or access to the frozen test.
