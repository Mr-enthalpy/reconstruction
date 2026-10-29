"""D5.3 frozen-reconstructor response to fixed-wideband counterfactual pairs.

For each trained arm the two counterfactual cubes are pushed through the arm's
own operator and then through the frozen reconstructor. The question is not
"do the outputs differ" but "does the reconstructed difference point in the
direction of the true spectral difference, and how much of it is recovered".

Three controls decide whether a response means anything:

* ``identity``      - the same measurement twice must give the same output;
* ``noise_only``    - two independent noise draws of the same scene give the
                      spurious response floor of that arm;
* ``achromatic``    - with one kernel per band both cubes have the same
                      measurement, so the response must be exactly zero and the
                      recoverable error floor is ``0.5 * ||delta_X||``.

No checkpoint is re-selected here; each arm contributes its own
minimum-validation-loss ``best.pt``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lcd_reconstruction.operators.psf_provenance import (
    PSFProvenanceError, array_digest, file_digest, physics_buffers_from_model,
    physics_buffers_from_state, resolve_psf_policy)
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from lcd_reconstruction.unrolling.hsfaut import HSFAUReconstructor

import counterfactual_spectra as cf


def build_model(cfg, kernels, hw):
    if cfg["model"] == "A":
        return GeneralSpectralPGD(kernels, hw, cfg["stages"], cfg["width"], cfg["depth"])
    return HSFAUReconstructor(kernels, hw, stages=cfg["stages"], width=cfg["width"],
                              depth=cfg["depth"],
                              prior="cnn" if cfg["model"] == "B" else "sfat",
                              sfat_width=cfg["sfat_width"], sfat_levels=cfg["sfat_levels"])


def checkpoint_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arm(run_dir: Path, kernel_dir: Path, device: str):
    run_dir = Path(run_dir)
    execution = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    record = checkpoint.get("psf_provenance") or {}
    kernel_name = record.get("kernel_file")
    if not kernel_name:
        raise PSFProvenanceError(f"{run_dir.name}: checkpoint has no PSF provenance record")
    kernel_path = Path(kernel_dir) / kernel_name
    if not kernel_path.exists():
        raise FileNotFoundError(kernel_path)
    if record.get("kernel_sha256_file") and \
            record["kernel_sha256_file"] != file_digest(kernel_path):
        raise PSFProvenanceError(f"{run_dir.name}: kernel file content changed since training")
    kernels = np.load(kernel_path)["kernels"]
    model = build_model(checkpoint["config"], kernels, tuple(checkpoint["hw"])).to(device)
    policy = resolve_psf_policy(record, physics_buffers_from_model(model),
                                physics_buffers_from_state(checkpoint["state"]),
                                allow_mismatch=False, split="val",
                                requested_array=array_digest(kernels))
    model.load_state_dict(checkpoint["state"])
    if physics_buffers_from_model(model) != policy["stored"]:
        raise PSFProvenanceError(f"{run_dir.name}: operator changed after loading")
    model.eval().requires_grad_(False)
    sigma = execution.get("train_noise_std_absolute")
    if sigma is None and execution.get("measured_forward_rms"):
        sigma = (float(execution["measured_forward_rms"]) *
                 float(execution["train_noise_rms_fraction"]))
    return model, dict(
        arm=run_dir.name.split("_s")[0], run_dir=run_dir.name,
        seed=checkpoint["config"]["seed"], epoch=checkpoint.get("epoch"),
        best_epoch=checkpoint.get("best_epoch"),
        checkpoint_sha256=checkpoint_sha(run_dir / "best.pt"),
        kernel_file=kernel_name,
        kernel_sha256_array=record["kernel_sha256_array"],
        kernel_sha256_file=record["kernel_sha256_file"],
        sigma_noise=None if sigma is None else float(sigma),
        selection_rule=(checkpoint.get("selection") or {}).get("rule"),
        psf_policy=policy["policy"])


def radial_masks(hw):
    fy = np.fft.fftfreq(hw[0])[:, None]
    fx = np.fft.fftfreq(hw[1])[None, :]
    return np.hypot(fy, fx) >= 0.25


def response_metrics(delta_hat, delta, mask, bands):
    i, j = bands
    selected = mask > 0
    denominator = float(np.linalg.norm(delta)) or 1e-30
    hat_norm = float(np.linalg.norm(delta_hat))
    cosine = float((delta_hat * delta).sum() / max(hat_norm * denominator, 1e-30))
    contrast_hat = delta_hat[i] - delta_hat[j]
    contrast = delta[i] - delta[j]
    meaningful = selected & (np.abs(contrast) > 0.1 * float(np.abs(contrast[selected]).max()))
    direction = float(np.mean(np.sign(contrast_hat[meaningful]) ==
                              np.sign(contrast[meaningful]))) if meaningful.any() else None
    gain = float(np.mean(contrast_hat[meaningful] / contrast[meaningful])) \
        if meaningful.any() else None
    high = radial_masks(delta.shape[-2:])
    hf_hat = np.fft.fft2(delta_hat)[..., high]
    hf_true = np.fft.fft2(delta)[..., high]
    return dict(
        response_fraction=hat_norm / denominator,
        delta_error_relative=float(np.linalg.norm(delta_hat - delta)) / denominator,
        cosine=cosine,
        band_contrast_direction_accuracy=direction,
        band_contrast_gain=gain,
        high_frequency_fraction=float(np.linalg.norm(hf_hat) /
                                      max(np.linalg.norm(hf_true), 1e-30)),
        masked_l2_ratio=float(np.linalg.norm(delta_hat[..., selected]) /
                              max(np.linalg.norm(delta[..., selected]), 1e-30)))


def reconstruct(model, x, device):
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))[None].to(device)
        measurement = model.physics(tensor)
        output = model(measurement)
    return output["spectral_cube_hat"][0].cpu().numpy()


def case_block(model, case, device, sigma, rng):
    x_plus, x_minus = case["x_plus"], case["x_minus"]
    bands = case["bands"]
    delta = x_plus.astype(np.float64) - x_minus.astype(np.float64)
    hat_plus = reconstruct(model, x_plus, device)
    hat_minus = reconstruct(model, x_minus, device)
    delta_hat = hat_plus.astype(np.float64) - hat_minus.astype(np.float64)
    result = dict(
        scene_id=case["scene_id"], dataset_id=case["dataset_id"],
        bands_nm=case["bands_nm"], mask=case["mask"], strength=case["strength"],
        truth_delta_l2=float(np.linalg.norm(delta)),
        response=response_metrics(delta_hat, delta, case["mask_array"], bands))

    measurement = model.physics(torch.from_numpy(x_plus)[None].to(device))
    first = model(measurement)["spectral_cube_hat"]
    second = model(measurement)["spectral_cube_hat"]
    result["identity_max_abs"] = float((first - second).abs().max())

    if sigma:
        shape = tuple(measurement.shape)
        draws = [torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(device) *
                 float(sigma) for _ in range(2)]
        with torch.no_grad():
            out_a = model(measurement + draws[0])["spectral_cube_hat"][0].cpu().numpy()
            out_b = model(measurement + draws[1])["spectral_cube_hat"][0].cpu().numpy()
        result["noise_only"] = response_metrics(out_a.astype(np.float64) - out_b.astype(np.float64),
                                                delta, case["mask_array"], bands)
    return result


def summarise(rows):
    def median(values):
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None

    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    summary = {}
    for arm, group in by_arm.items():
        summary[arm] = dict(
            cases=len(group),
            median_response_fraction=median([r["response"]["response_fraction"] for r in group]),
            median_delta_error_relative=median([r["response"]["delta_error_relative"]
                                                for r in group]),
            median_cosine=median([r["response"]["cosine"] for r in group]),
            median_direction_accuracy=median([r["response"]["band_contrast_direction_accuracy"]
                                              for r in group]),
            median_band_contrast_gain=median([r["response"]["band_contrast_gain"]
                                              for r in group]),
            median_high_frequency_fraction=median([r["response"]["high_frequency_fraction"]
                                                   for r in group]),
            median_noise_only_fraction=median([r["noise_only"]["response_fraction"] for r in group
                                               if "noise_only" in r]),
            max_identity_abs=max((r["identity_max_abs"] for r in group), default=None),
            median_truth_delta_l2=median([r["truth_delta_l2"] for r in group]))
    by_seed = {}
    for row in rows:
        key = f"{row['arm']}:{row['seed']}"
        by_seed.setdefault(key, []).append(row)
    per_seed = {key: dict(
        cases=len(group),
        median_response_fraction=median([r["response"]["response_fraction"] for r in group]),
        median_cosine=median([r["response"]["cosine"] for r in group]),
        median_direction_accuracy=median([r["response"]["band_contrast_direction_accuracy"]
                                          for r in group]),
        median_noise_only_fraction=median([r["noise_only"]["response_fraction"] for r in group
                                           if "noise_only" in r]))
        for key, group in by_seed.items()}
    by_mask = {}
    for row in rows:
        by_mask.setdefault((row["arm"], row["mask"]), []).append(row)
    per_mask = {f"{arm}|{mask}": dict(
        cases=len(group),
        median_response_fraction=median([r["response"]["response_fraction"] for r in group]))
        for (arm, mask), group in by_mask.items()}
    return dict(by_arm=summary, by_seed=per_seed, by_mask=per_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--kernel-dir", type=Path, required=True)
    ap.add_argument("--runs", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--pairs", default=";".join(f"{a},{b}" for a, b in cf.default_pairs()))
    ap.add_argument("--strengths", default="0.5")
    ap.add_argument("--masks", default="uniform,edge,stripes")
    ap.add_argument("--max-cases", type=int)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = [tuple(int(v) for v in p.split(",")) for p in args.pairs.split(";") if p]
    strengths = [float(v) for v in args.strengths.split(",") if v]
    masks = [m for m in args.masks.split(",") if m]
    scenes = [s for s in args.scenes.split(",") if s]

    arms = []
    rows = []
    for run_dir in args.runs:
        model, meta = load_arm(Path(run_dir), args.kernel_dir, args.device)
        arms.append(meta)
        rng = np.random.default_rng(args.seed)
        for case in cf.iter_cases(args.prepared, scenes, pairs, strengths, masks,
                                  args.seed, args.max_cases):
            row = case_block(model, case, args.device, meta["sigma_noise"], rng)
            row["arm"] = meta["arm"]
            row["seed"] = meta["seed"]
            rows.append(row)

    document = dict(
        protocol="D5.3-frozen-reconstruction-response",
        statement="The reconstructor is frozen; only its own operator and the two "
                  "counterfactual cubes vary. A response counts only if it beats the "
                  "arm's own noise-only floor and follows the true spectral difference.",
        arms=arms,
        cases=rows,
        summary=summarise(rows),
        prior_only=dict(response_fraction=0.0,
                        reason="a measurement-independent prior baseline cannot respond to "
                               "delta_X; its contrast is exactly zero by construction"),
        indistinguishability=dict(
            statement="with one kernel per band the two cubes have the same measurement, "
                      "so any single estimate has error >= 0.5*||delta_X|| for a symmetric "
                      "error; the measured achromatic response must be zero",
            floor_relative_to_delta=0.5),
        notes=["validation scenes only; test is never loaded",
               "no checkpoint is re-selected by this script"])
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document["summary"]["by_arm"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
