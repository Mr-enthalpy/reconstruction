# Source and exposure audit

All D1 non-augmented candidates were audited, not just the116 paired D3 UWA ROIs.
CAVE includes29 D3 ROIs plus four source-level review entries. Candidate counts
are not independent observations. No new identity inference was performed.

| Source | Original candidates | Acquisition/scenes | Conservative groups | Admitted development ROIs |
| --- | ---: | ---: | ---: | ---: |
| CAVE | 33 | 32 | 30 | 29 |
| UWA | 206 | 145 | 78 | 116 |

All116 D3 UWA rows map back through sample hash and native ROI to D1 acquisitions,
the original subject split and the verified UWA archive. Their absence from the
A6 manifest is explicitly scoped; D2/D3 reconstructor and task exposures remain.
D4's16 subject-level whole-field diagnostic records are joined by subject,
including all sessions conservatively. Cross-ID identity aliases and common
background independence remain unverified; exact file hashes cannot establish them.

CAVE family joins reproduce18 task-train and6 task-test ROIs in A6 training
families, plus5 in A6 model-selection validation. The four additional CAVE
source records belong to consumed G4 test scenes; their source cubes were not
opened. All14 G4 scenes, including KAIST, remain consumed; KAIST has no admitted
task labels. Source files are hashed, UWA PNG headers and CAVE NPY headers checked,
and D1 metadata read without decoding pixels. No G4 tensor archive was opened.

Machine-readable counts and nonexclusive exclusion reasons are in
`source_exposure_audit.json`; input availability/digests are in `input_inventory.json`.
Full identities, ancestry, source paths and per-sample audit stay in the private
output. Zero candidates have verified fresh final independence. Unmatched or
additional D1 candidates are review-only, never promoted by absence of evidence.
