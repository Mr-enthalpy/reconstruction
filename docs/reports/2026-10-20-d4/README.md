# D4 — independent general reconstruction, work in progress

**Status: implementation and CPU execution checks completed; GPU training and
reconstruction quality acceptance NOT completed. Do not start C training from
this status.** D2/D3 reports, models and acquisition outputs remain unchanged.

## Why D3 did not fully decouple the task

D3's forward/adjoint tests and full-channel PGD are reusable. Its reconstruction
loss had no classification labels or classifier gradients. However, its input
dataset still consisted of classification-selected UWA/CAVE ROIs, with task
splits, class-oriented exclusions, equal-energy scene normalization and32² working
resolution. Its12,152-parameter/60-epoch CPU pilot was not general reconstruction
acceptance. The source classifier remained perfectly predictive on that test
subset, and task scores matched achromatic/intensity controls. Those results
cannot validate a general imaging inverse.

D4 changes data eligibility and interfaces, not the historical experimental record.
R and N1 independently consume a shared observation in the intended system; C
consumes their aligned outputs. Reconstruction quality is necessary in its own
right, even though task-level sensing remains the eventual research claim.

## Implemented and actually run

- Independent `codex/d4-general-reconstruction` branch from D3 `f572d1e`.
- General CAVE loader: complete21×512×512 scenes, uint16 validation, nominal
 450–650nm order, fixed `/65535` range, no face label/ROI input.
- 17 train,5 validation,6 development-test scenes;3 additional scenes reserved.
  Existing development holds remain held. `face` and `photo_and_face` stay together
  as a conservative source group; chart/toy reserve scenes stay together.
- Final reserve: `egyptian_statue`, `chart_and_stuffed_toy`, `stuffed_toys`.
  Previously inspected for content, but absent from D2/D3 N1 fitting/evaluation.
  D4 records file hashes without placing these arrays in its prepared tensor.
  This is an untouched-by-training reserve, not a never-viewed blind collection.
- `watercolors` remains excluded for its unresolved8-bit RGBA format, not semantics.
  Other scenes are not rejected because of their subject matter.
- Real21 PSFs and matched pre/post darks were read from
  `spectral21_20261018_reset01`; source hashes/geometry/correction are recorded.
  Signed512 support is integrated at8 sensor pixels per simulation pixel.
  Unit spectral weights, zero boundaries and whole-scene128² scale remain assumed.
- General PGD: full21-channel state,6 stages, shared4-layer64-channel prior,
  **98,139 parameters**; no low-rank basis, detector, classifier or task feedback.
- Validation-only checkpoint selection, cosine LR, finite-gradient checks,
  fixed Landweber comparison, full spatial/spectral/reprojection metrics, Gaussian
  noise sensitivities and private output export.
- Minimal R: explicit OpenCV Bayer demosaicing, observation hash/identity, full
  coordinates, optional supplied YuNet weights. Scalar input cannot masquerade as
  Bayer. No UDC deblurring or colour calibration claimed.

### Executed verification

Repository test command using the existing independent D2 interpreter:
`python -X utf8 -m pytest -q -p no:cacheprovider`: **14 passed in3.03s**.
An initial unrestricted discovery encountered an inaccessible temporary pytest
cache directory; tests are now scoped to `tests/`, with cache disabled for this
restricted environment. No files or permissions were removed/changed to bypass it.

Coverage includes previous numerical adjoints, directional derivatives, full-state
optimizer steps, shared/replayable augmentation, fixed-range PSNR, all four Bayer
channel orders, missing model failure and explicit scalar rejection. Synthetic
Bayer tests validate coordinate/channel semantics, not real-camera restoration.
The original test run did not execute pretrained detection. A subsequent local
check downloaded the official MIT-licensed OpenCV YuNet2026may model
(SHA256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`).
`rgb_smoke.py` ran it after demosaicing an Oracle650/550/450 proxy mosaic:
1 candidate in `face`,2 in `photo_and_face`,0 in `balloons`, threshold0.9.
This is an interface check on three known scenes, not recall/localization or
UDC-degradation acceptance. Private previews and source identities remain local.
References: [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
[model licence](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE).

Preparation succeeded on actual local CAVE/PSF inputs. A separate
`--cpu-smoke` run completed two epochs, checkpoint reload and evaluation in20.51s.
CPU median inference was70.96ms/scene at128². This is execution evidence only;
two epochs are not a quality or convergence result. Development-test metrics from
that smoke are retained locally, not used to tune the frozen training config.
The three final-reserve arrays were not loaded by either preparation or training.

Prepared manifest SHA256:
`de334816a9b7a4dd381f193336a9db891b47db404a6b3c5c503c052c77d544bb`

Physics record SHA256:
`a39f2423e06a01e96dffe495b3d026a71428b35252fa6225c87812b3c8781d22`

## Data expansion and remaining scientific limits

CAVE alone is a small collection, with related objects and uncertain scene
independence beyond file grouping. Do not count transforms as new scenes.
UWA was deliberately not used as the general N1 training backbone; its intensity
domain and face-specific preprocessing remain separate from CAVE reflectance.

[Harvard's official database](https://vision.seas.harvard.edu/hyperspec/download.html)
permits non-commercial research and describes31 bands from420–720nm with motion
masks. It is a promising general-scene expansion with nominal21-band overlap.
Its download page lists50 daylight and27 mixed-light files, while the overview
says25 additional files; actual counts and archive README must be verified.
Only an incomplete archive prefix has been downloaded; no Harvard cube has been
added to this experiment. The prefix's README confirms27 mixed-light images,
`ref [H,W,31]`, `lbl` with zero marking invalid pixels, and a camera-sensitivity
vector in `calib.txt`; it permits research/academic use and asks for the cited paper.
Full archive integrity and sample contents remain unverified. Its
irradiance domain must not silently be treated as CAVE reflectance.

[ARAD's official entry](https://github.com/boazarad/ARAD_1K) remains an alternative
pending access and actual project-use terms. Prior D0 registration/permission
limitations remain unresolved; it is not part of the measured training inventory.

Larger-data training, new-domain evaluation, model-mismatch/PSF support sensitivity,
local spectral plots, and serious convergence/overfit analysis remain incomplete.
Scalar signed shape-only physics does not prove unique recovery, absolute spectral
throughput, Bayer fidelity or real-camera performance.

## GPU setup and next execution

The machine reports RTX3060Ti8GB. The existing research environment has
PyTorch2.7.1+cpu. A separate D4 venv creation failed during ensurepip under the
sandbox. The subsequent approved-path CUDA setup request did not execute because
automatic approval review reported the workspace out of credits. This is a review
availability failure, not a finding that installation is unsafe. No workaround,
system Python modification or replacement of the historical environment occurred.

The user subsequently confirmed credits restored. Approval now succeeds; the
remaining delay is the official3.27GB CUDA wheel transfer, which was slow and
intermittent. A bounded local range downloader retains completed chunks and will
accept the assembled wheel only against the official index SHA256
`500ad5b670483f62d4052e41948a3fb19e8c8de65b99f8d418d879cbb15a82d6`.
Non-Torch dependencies have been installed in the separate D4 environment from
the existing wheel cache. GPU installation/training must still be confirmed by
execution logs, not inferred from the resumed downloader.

Required continuation:

1. Complete independent CUDA environment from the official PyTorch distribution;
   verify device kernels and repeat numerical tests on GPU.
2. Verify/access larger general data and its radiometric/scene contracts; version
   any expanded splits before training, without touching the final reserve.
3. Run the declared128²,6-stage reconstruction experiment with validation selection.
   Review curves and quality before deciding whether capacity/training need changes.
4. Complete spatial/spectral/noise/mismatch/unseen-domain evidence and failure plots.
5. Validate R with actual licensed pretrained weights and independent detector
   annotations; do not call the demosaic-only interface a recovered UDC RGB system.
6. Only then decide whether frozen N1 and independently trained C are justified.

No hardware connection, acquisition-code edit, Meta gitlink update or publication
was performed for D4. Code is local work in progress; training/weights are not
accepted merely because the interface or CPU execution checks work.
