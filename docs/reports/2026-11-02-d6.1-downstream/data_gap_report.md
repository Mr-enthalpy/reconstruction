# Data gaps and minimum next evidence

Development can use the legacy admitted groups; this does not create a new
independent final set. CAVE validation and development-test each lack any
positive group, and the only CAVE positive family is train-only. At least one
new independent positive family per validation and final partition is needed
before either can support CAVE within-source two-class assessment; never move
the existing face family to fill this gap.

Fresh final data: zero eligible positive or negative groups. A bare engineering
floor is at least two new independent source groups supporting each class, with
at least two independently reviewed hard-negative groups. Groups may contain
both classes, but individual ROIs never count as independent groups. This is
a minimum for non-singleton reporting, not a powered scientific sample size;
set a justified larger acquisition budget in a new preregistration. Cross-source
hard-negative claims need confirmed examples from at least two sources; currently
both UWA and CAVE have zero confirmed types. Augmentation cannot fill these gaps.

All admitted D3 ROIs have verified native full-field sources. Legacy32x32
arrays are ineligible inputs. Review-only records with missing/invalid ROI or
unverified full-field compatibility are counted in the audit; unavailable
derivatives are explicitly recorded privately, not reconstructed or invented.
Cross-ID UWA aliases and shared-background independence are unverified. D1/D2/D3
task tests, D4 diagnostics and all14 G4 scenes cannot be a new final reserve.

For future Axon20 data, record device/sensor/calibration version, acquisition
and session ID, subject/family aliases privately, original file SHA256, timestamp,
display state/timing, exposure/gain, native resolution/orientation/CFA, black and
white levels, radiometric units, wavelength/PSF/noise calibration provenance,
candidate coordinates and transforms, label semantics/reviewer/version,
presentation and confirmed difficulty type, consent/license/use restrictions,
all prior training/selection/test access, and a locked source-group split.
Keep all sessions/derivatives of the same subject and related scenes in one
partition. Separate calibration/development/final acquisitions; hold final
truth and arrays behind a separate access ledger. No Axon20 data was acquired
or assumed available here, and no phone specification is inferred.
