# D5 G0 environment audit

Date: 2026-10-21. Scope: local development host and read-only official dataset
access. No remote deployment, no training, no hardware operation.

## Git state

| Repository | Branch | HEAD | Working tree |
|---|---|---|---|
| `reconstruction` | `codex/d5-hsfaut-reconstruction` | `809e6664661fdfd741665c0acd39fec73e4a0c78` | clean before this report |
| Meta workspace | `codex/admit-reconstruction` | unchanged | reconstruction gitlink still `b6668a5`; no pin update |

The D5 branch starts from the published D4 commit. The existing Meta gitlink
deviation is preserved and is not part of D5.

## Local compute

| Item | Observed |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8192 MiB, driver 630.20, compute capability 8.6 |
| CUDA runtime | PyTorch 2.7.1+cu128, CUDA 12.8, CUDA available |
| BF16 | supported |
| CPU | Intel Core i7-12700K, 12 cores / 20 logical processors |
| RAM | 31.7 GiB |
| Disk | C: 190.3 GiB free; D: 675.8 GiB free |
| Research Python | `outputs/d4_general_20261020/.venv/Scripts/python.exe`, Python 3.13.5 |
| Packages | numpy 2.5.4, scipy 1.18.1, torch 2.7.1+cu128, opencv 5.0.0 |
| Existing regression | `python -X utf8 -m pytest -q -p no:cacheprovider` -> 15 passed in 4.61 s |

The D4 selected model used 1.02 GiB peak CUDA allocation and 12.53 ms median
single-scene inference at 128x128. Those are historical measurements, not a D5
budget guarantee.

## GPU budget and limits

- Confirmed budget: one local 8 GiB GPU. It is sufficient for D5 numerical tests
  and a 128x128 pilot; 256x256/512x512 Model B/C runs must be benchmarked first.
- No shared-cluster allocation was requested or verified in this audit. The
  2026-10-21 server snapshot is stale and is not treated as available capacity.
- Formal D5 training is therefore local-first. If a 512x512 or larger configuration
  does not fit, report the measured OOM and request a coordinated GPU window; do
  not crop PSF support or drop bands to manufacture a fit.
- No Slurm/PBS scheduler is used on the local host. Training entry points must run
  as explicit single-process commands with logs and resumable checkpoints.

## Official dataset access

The research lead supplied and authorized these NJU Box links:

- `https://box.nju.edu.cn/d/9a5edf3953ad4135bfcf/` -> `cave_1024_28/`, 205 `.mat`
- `https://box.nju.edu.cn/d/da41fc2138444874ac46/` -> `KAIST_non_selected/`, 20 `.mat`
- `https://box.nju.edu.cn/d/b4da0dd885c042ef84fe/` -> `KAIST_selected_1.3_resize/`, 10 `.mat`

Only directory listings and two single-file validation downloads were performed.
No bulk download has occurred. The files are stored outside Git under
`D:\udc-data\d5\official\samples` on this host; that path is machine-local and must
not be published.

Observed sample structure:

| Sample | SHA256 | Variable | Shape | Dtype | Range |
|---|---|---|---|---|---|
| `cave_1024_28_scene1.mat` | `8EA19395871017643EA42068236CF4DD1140E84109E4D45A41E71D3DB18C7CAD` | `img_expand` | 1024x1024x28 | uint16 | 0 to 63446 |
| `kaist_selected_scene1.mat` | `3E3EDDD69101D24EF66EDE0A76DB6415A78EDCB82FA36AB8367C8B49A4CEFC1D` | `HSI_crop_586` | 586x586x28 | float64 | 0.00956 to 0.73764 |

Both are finite MATLAB 5.0 files. They carry 28 spectral samples and no embedded
wavelength vector. `cave_1024_28` is consumed by the public HSFAU reference code as
205 independent 1024x1024 inputs with random crops; it does not expose an original
CAVE source-scene map. Therefore it cannot be used for grouped D5 training until a
source mapping is established. The KAIST selected files are scene-level test inputs.

## Unverified or blocked

- Official 28-channel wavelength centers, spectral units and 28-to-21 policy are
  not verified yet.
- CAVE `cave_1024_28` source-scene grouping is unknown; 205 files must not be
  treated as 205 independent scenes.
- KAIST access terms beyond research use were not independently re-read in this
  audit.
- Shared GPU/container environment and PyTorch compatibility remain unverified.
- No D5 training has been executed.
