# Repository boundaries

`optic_system` owns hardware, calibrations and immutable real measurements.
`reconstruction` reads explicit local copies of PSF/metadata and public datasets;
it neither imports sibling internals nor changes acquisition code or data.
The Meta Repository coordinates research and records approved version pins.

Current outputs are local numerical arrays, model checkpoints and audit reports.
Raw/derived face data and checkpoints remain private. Only source and aggregate
reports belong in Git. There is no dependency on an unadmitted modelling repository,
operator registry, handoff schema or remote service.
