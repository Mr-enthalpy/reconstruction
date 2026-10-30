# D5.3 evidence audit for D5.4

Base: `ef07c1b0befbf28d6938c4e67006c8b4f5f8e37d`. No D5.3 experiment was rerun or overwritten.
Machine-readable audit: `evidence_audit.json`; source hashes and historically recorded
data/kernel/checkpoint SHA256 values are retained there. A recorded hash is not a fresh
verification of the original bytes.

Subsequent byte verification is recorded in `checkpoint_registry.json` (SHA256
`900be1d9991b2e7cbdb033d68ff44819dbfb141069044f837fc15442c6356aa7`).
All nine best checkpoints were found and loaded through the existing strict PSF
provenance gate. All best epochs match the published validation files. The six
S1 checkpoint hashes match D5.3's response registry; the three 550 nm hashes are
newly recorded byte observations, not a claim that an earlier weight hash existed.
The earlier audit's `checkpoint_bytes_checked: false` describes its earlier read-only
report audit, superseded by this explicitly separate original-byte check.

The current D5.3 main report already contains the corrected SNR strata. The older
0.03–0.32 / 1.9–73 shorthand is explicitly identified as covering only a subset;
it cannot summarise all 675 cases. There are 75 configuration records, each a mean
over nine scenes. Recomputed ranges by strength are approximately:

| Strength | Pixel RMS SNR | Known-template matched-filter SNR |
|---|---|---|
| 0.25 | 0.0038–0.357 | 1.92–182.75 |
| 0.50 | 0.0089–0.714 | 4.55–365.49 |
| 1.00 | 0.0098–1.428 | 5.02–730.98 |

These ranges still mix masks and band spacings within a strength. Full mask, band-pair
and spacing strata are in the audit; they are configuration-mean ranges, not case extrema.
Independent paired measurements have noise variance 2 sigma². A known difference
template supports ideal detection, not blind recovery of an unknown spectral cube.

Achromatic-550's three validation reports exist. Historical inability to launch within
the first lease is preserved as history; the later second lease completed the runs.
Only the tested reference-kernel change is covered: the relative direction did not flip.
Neither reference matches the chromatic operator at every frequency.

One remaining wording issue is D5.3 scientific_conclusions.md table D's “550 nm and
mean kernel equivalent — supported (small magnitude)”. The body reports differences;
small differences do not establish equivalence without an equivalence margin and test.
D5.4 therefore treats equivalence as unestablished. The previous files remain intact.

Spatial SSIM/gradient/residual passed the registered rule; PSNR/HF did not pass all
clauses, and SAM went in the opposite direction. Frozen A6 failed on the old constructed
perturbations. This is validation-only, shape-only simulation evidence. It does not
establish hardware throughput, material/classification benefit, or universal impossibility
of spectral recovery. Noise sensitivity, distribution shift, conditioning and the
reconstructor's contribution remain open.
