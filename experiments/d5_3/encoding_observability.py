"""D5.3 operator-level observability of zero-sum spectral changes.

For the fixed-wideband counterfactual pairs this script evaluates the physics
operator only (no network):

* an achromatic operator (one kernel for all bands) must return the *same*
  measurement for both cubes, and the numeric residual is reported;
* the chromatic difference is checked against the analytic relation
  ``delta_Y = sum_b (K_b - mean_b K_b) * delta_X_b``;
* the difference is compared with measurement noise, per pixel and per radial
  frequency bin, using the two independent-noise convention (variance 2 sigma^2).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lcd_reconstruction.operators.psf_provenance import array_digest
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics

import counterfactual_spectra as cf

BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]


def load_kernels(path: Path):
    return np.load(path)["kernels"].astype(np.float32)


def achromatic_from(kernels):
    return np.repeat(kernels.mean(0, keepdims=True), kernels.shape[0], axis=0)


def single_band_from(kernels, nm: int):
    return np.repeat(kernels[cf.band_index(nm)][None], kernels.shape[0], axis=0)


def radial_geometry(hw):
    fy = np.fft.fftfreq(hw[0])[:, None]
    fx = np.fft.fftfreq(hw[1])[None, :]
    radius = np.hypot(fy, fx)
    masks = [((radius >= low) & (radius < high)) for low, high in BINS]
    return masks


def spectral_weights(kernels):
    """|K_b(f) - mean_b K_b(f)| integrated over the kernel support (float64)."""
    k = np.asarray(kernels, dtype=np.float64)
    return np.abs(k - k.mean(0, keepdims=True)).sum(axis=(1, 2))


def noise_statistics(delta_y, sigma, hw):
    """Per-pixel and per-radial-bin detection statistics for two noise draws."""
    delta = np.asarray(delta_y, dtype=np.float64).reshape(hw)
    variance = 2.0 * float(sigma) ** 2
    per_pixel = np.abs(delta) / np.sqrt(variance)
    spectrum = np.fft.fft2(delta)
    masks = radial_geometry(hw)
    total = float(delta.size)
    radial = {}
    for (low, high), mask in zip(BINS, masks):
        count = int(mask.sum())
        signal = float((np.abs(spectrum[mask]) ** 2).sum())
        # A white field of variance v has per-coefficient variance v * N.
        noise = variance * total * count
        radial[f"{low:.1f}-{high:.1f}"] = dict(
            coefficient_count=count,
            signal_energy=signal,
            expected_noise_energy=noise,
            snr=float(np.sqrt(signal / max(noise, 1e-30))),
        )
    return dict(
        sigma=float(sigma),
        energy_snr=float(np.linalg.norm(delta) /
                         max(np.sqrt(variance) * np.sqrt(delta.size), 1e-30)),
        per_pixel_snr_mean=float(per_pixel.mean()),
        per_pixel_snr_max=float(per_pixel.max()),
        detectable_pixel_fraction=float((per_pixel > 2.0).mean()),
        radial=radial,
    )


def evaluate_cases(cases, kernels, kernels_achrom, kernels_550, hw, device,
                   sigma_chromatic=None, sigma_achromatic=None, chunk=6):
    ops = {name: SinglePlanePhysics(kernels_, hw).to(device).eval()
           for name, kernels_ in (("chromatic", kernels), ("achromatic", kernels_achrom),
                                  ("achromatic550", kernels_550))}
    mean = kernels.mean(0, keepdims=True)
    diff_op = SinglePlanePhysics(kernels - mean, hw).to(device).eval()
    for op in list(ops.values()) + [diff_op]:
        for parameter in op.parameters():
            parameter.requires_grad_(False)

    rows = []
    with torch.no_grad():
        while True:
            block = list(itertools.islice(cases, chunk))
            if not block:
                break
            x_plus = torch.from_numpy(np.stack([c["x_plus"] for c in block])).to(device)
            x_minus = torch.from_numpy(np.stack([c["x_minus"] for c in block])).to(device)
            y = {}
            for name, op in ops.items():
                y[name] = (op(x_plus) - op(x_minus)).squeeze(1).cpu().numpy()
            y_ref = ops["chromatic"](x_plus).squeeze(1).cpu().numpy()
            analytic = (diff_op(x_plus) - diff_op(x_minus)).squeeze(1).cpu().numpy()
            for index, case in enumerate(block):
                reference = float(np.linalg.norm(y_ref[index]))
                delta_chromatic = y["chromatic"][index]
                delta_norm = float(np.linalg.norm(delta_chromatic))
                row = dict(
                    scene_id=case["scene_id"], dataset_id=case["dataset_id"],
                    bands_nm=case["bands_nm"], mask=case["mask"], strength=case["strength"],
                    chromatic=dict(
                        delta_l2=delta_norm,
                        measurement_l2=reference,
                        relative=delta_norm / max(reference, 1e-30)),
                    achromatic=dict(
                        max_abs=float(np.abs(y["achromatic"][index]).max()),
                        l2=float(np.linalg.norm(y["achromatic"][index])),
                        relative_to_chromatic=float(np.linalg.norm(y["achromatic"][index]) /
                                                    max(delta_norm, 1e-30))),
                    achromatic550=dict(
                        max_abs=float(np.abs(y["achromatic550"][index]).max()),
                        l2=float(np.linalg.norm(y["achromatic550"][index])),
                        relative_to_chromatic=float(np.linalg.norm(y["achromatic550"][index]) /
                                                    max(delta_norm, 1e-30))),
                    analytic=dict(
                        max_abs_error=float(np.abs(analytic[index] - delta_chromatic).max()),
                        relative_to_measurement=float(
                            np.linalg.norm(analytic[index] - delta_chromatic) /
                            max(reference, 1e-30)),
                        relative_to_delta=float(
                            np.linalg.norm(analytic[index] - delta_chromatic) /
                            max(delta_norm, 1e-30))),
                )
                if sigma_chromatic:
                    row["noise_chromatic"] = noise_statistics(delta_chromatic, sigma_chromatic, hw)
                if sigma_achromatic:
                    row["noise_achromatic"] = noise_statistics(y["achromatic"][index],
                                                              sigma_achromatic, hw)
                rows.append(row)
    return rows


def summarise(rows):
    def average(values):
        return float(np.mean(values)) if values else None

    grouped = {}
    for row in rows:
        key = (tuple(row["bands_nm"]), row["mask"], row["strength"])
        grouped.setdefault(key, []).append(row)
    per_configuration = []
    for key, group in sorted(grouped.items()):
        per_configuration.append(dict(
            bands_nm=list(key[0]), mask=key[1], strength=key[2], cases=len(group),
            mean_relative_chromatic=average([r["chromatic"]["relative"] for r in group]),
            max_achromatic_abs=average([r["achromatic"]["max_abs"] for r in group]),
            mean_analytic_relative_to_measurement=average(
                [r["analytic"]["relative_to_measurement"] for r in group]),
            mean_energy_snr=average([r["noise_chromatic"]["energy_snr"] for r in group
                                     if "noise_chromatic" in r]),
            mean_detectable_fraction=average([
                r["noise_chromatic"]["detectable_pixel_fraction"] for r in group
                if "noise_chromatic" in r]),
        ))
    by_band_pair = {}
    for row in rows:
        by_band_pair.setdefault(tuple(row["bands_nm"]), []).append(row["chromatic"]["relative"])
    return dict(
        cases=len(rows),
        per_configuration=per_configuration,
        mean_relative_by_band_pair={str(k): average(v) for k, v in sorted(by_band_pair.items())},
        max_achromatic_abs_residual=max((r["achromatic"]["max_abs"] for r in rows), default=None),
        max_achromatic550_abs_residual=max((r["achromatic550"]["max_abs"] for r in rows),
                                           default=None),
        max_analytic_relative_to_measurement=max(
            (r["analytic"]["relative_to_measurement"] for r in rows), default=None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--kernel-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--pairs", default=";".join(f"{a},{b}" for a, b in cf.default_pairs()))
    ap.add_argument("--strengths", default="0.25,0.5,1.0")
    ap.add_argument("--masks", default="uniform,block,edge,stripes,chroma")
    ap.add_argument("--sigma-chromatic", type=float)
    ap.add_argument("--sigma-achromatic", type=float)
    ap.add_argument("--max-cases", type=int)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = [tuple(int(v) for v in p.split(",")) for p in args.pairs.split(";") if p]
    strengths = [float(v) for v in args.strengths.split(",") if v]
    masks = [m for m in args.masks.split(",") if m]
    scenes = [s for s in args.scenes.split(",") if s]

    case_iter = cf.iter_cases(args.prepared, scenes, pairs, strengths, masks, args.seed,
                              args.max_cases)
    first = next(case_iter, None)
    if first is None:
        raise ValueError("no counterfactual cases were generated")
    kernels = load_kernels(args.kernel_file)
    kernels_achrom = achromatic_from(kernels)
    kernels_550 = single_band_from(kernels, 550)
    hw = (int(first["x_plus"].shape[-2]), int(first["x_plus"].shape[-1]))
    rows = evaluate_cases(itertools.chain([first], case_iter), kernels, kernels_achrom,
                          kernels_550, hw, args.device,
                          args.sigma_chromatic, args.sigma_achromatic)
    document = dict(
        protocol="D5.3-encoding-observability",
        operator="y = sum_b K_b * X_b (zero-padded convolution, all-1 spectral weights)",
        identity="delta_Y(f) = sum_b (K_b(f) - mean_b K_b(f)) * delta_X_b(f); "
                 "identically zero for any single-kernel-per-band operator",
        rank_note="at each spatial frequency the map from 21 band coefficients to one "
                  "measurement is rank 1; dispersion changes the band weights per frequency",
        kernels=dict(
            chromatic=dict(file=args.kernel_file.name, array_sha256=array_digest(kernels)),
            achromatic=dict(kind="mean over 21 bands", array_sha256=array_digest(kernels_achrom)),
            achromatic550=dict(kind="550 nm band replicated", array_sha256=array_digest(kernels_550)),
            band_sensitivity_l1=spectral_weights(kernels).tolist()),
        noise=dict(sigma_chromatic=args.sigma_chromatic,
                   sigma_achromatic=args.sigma_achromatic,
                   convention="two independent draws: variance 2*sigma^2"),
        cases=rows,
        summary=summarise(rows),
        notes=["operator-level result, no network involved",
               "a non-zero achromatic residual means a bug in the forward, not information",
               "only the val split is read; test is never loaded"])
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(dict(cases=len(rows), summary=document["summary"]), indent=2))


if __name__ == "__main__":
    sys.exit(main())
