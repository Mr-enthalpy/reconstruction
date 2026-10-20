# Evidence-led next steps

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
