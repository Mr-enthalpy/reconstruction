# Evidence-led next steps

## Current D4 priority

1. General whole-scene data independent of face labels/ROIs. Preserve prior held
   groups and a separate final integration reserve; expand legally accessible data.
2. GPU training and validation of full-channel N1. Independently measure spatial,
   spectral and measurement fidelity, unseen-scene generalization and resources.
3. Mature R preprocessing/detection and a consistent shared RAW convention.
4. Only after reconstruction acceptance, independently evaluate C on RGB/spectral
   inputs, including detector failures and source confounding.
5. Compression and joint optimization follow evidence against this full-state baseline.

Detailed current status: [D4 report](reports/2026-10-20-d4/README.md).
Two independent GPU training runs and domain/model-mismatch diagnostics are now
complete. General reconstruction quality is not accepted: local spectral and
high-frequency errors remain large. Keep classification and compression deferred;
next address general-scene data coverage and physical recoverability with a frozen
training/validation protocol, preserving the final integration reserve.
The following D2/D3 sequence is retained as historical context, not current
instructions to prioritize classification-selected training data.

The completed [D2 pilot](reports/2026-10-20-d2/README.md) remains historical evidence.
The [revised architecture](architecture.md) separates spatial-spectral scene recovery
from downstream classification. The original design sequence follows; actual D3 execution is recorded below.

1. Review CAVE face and non-face ROIs; retain scene isolation and frozen UWA subject
   groups. Version the new protocol and audit source/label balance and sample limits.
2. Check whether shared cleaning and augmentation reduce source cues without
   changing real spectral shapes or leaking groups.
3. Build the independent spatial-spectral unrolling reconstruction baseline, with
   inspectable scene matrices and reconstruction-only supervision. Validate local
   spectra, spatial structure, amplitude and measurement consistency.
4. Freeze the reconstructor and compare independent classifiers on common splits,
   including oracle scenes, encoded observations and achromatic controls.
5. Only then assess low-rank compression, global/local spectral taps or other
   intermediate coding choices against the uncompressed baseline.

No hardware work, larger training platform or speculative classifier modules are
part of this documentation update. Progress is recorded once in dated reports.

## D3 evidence (2026-10-20)

Steps 1-4 have a bounded implementation and actual runs: reviewed CAVE face ROIs,
shared preprocessing/source checks, full-channel N1, and frozen classifier protocols.
The [D3 report](reports/2026-10-20-d3/README.md) owns detailed acceptance. There are
no independent CAVE positive test scenes, source prediction remains easy, and
achromatic/intensity controls match the task score. Hold compression and joint
fine-tuning pending a data/physical-model research decision; do not interpret
completed software runs as completed scientific acceptance.
