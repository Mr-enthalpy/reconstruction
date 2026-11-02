# Frozen D6.1 classification protocol

`classification_protocol_v1.json` is the exact executable specification; the
protocol lock binds it, source audits, code and private sample manifest hashes.
This version permits development experiments only; final testing is disabled.
No classifier was trained while preparing this protocol.

The task is recognizable Face/Non-face, including printed faces. R and N1 use
the same scalar full-field observation. N1 is the pre-existing fixed first A6
seed, with original state hashes, frozen evaluation mode and no label input.
Process the full512 field before native-to-working ROI selection. UWA uses
original1024 bands downsampled as a full field, never enlarged32x32 ROI data.

B0 uses22 intensity features; B1 adds21 reconstructed band means. Their affine
heads have23 and44 parameters. A mandatory44-parameter scalar spatial-feature
control addresses capacity; merely beating B0 is insufficient to attribute a
benefit to the N1 representation. Train-only weighted normalization, CPU optimizer,
three seeds, budgets, patience, checkpoint rule and threshold0.5 are frozen in
JSON. No tuning against historical development-test scores within this version.

Report pooled and within-source metrics, group and ROI weights, source/class
cross tables, source-only prevalence baseline and separate source-identifiability
probe. CAVE single-class validation/test permits only negative recall/error;
balanced accuracy, macro-F1, AUROC and AUPRC are undefined there. Hard-negative
FPR is undefined until typed reviewed negatives exist. Candidate boxes are
privileged fixed annotations, not evidence of detector recall. Repeated scalar
channels cannot be called RGB. Any gain is development representation evidence,
not physically identified spectral information, liveness or real UDC performance.

Before a future final test, create a new protocol version with independent
unused sources and a once-only access ledger. Never relabel D3/G4 consumed data
as fresh. Every failure, reuse or error-driven development must be recorded.
