"""D5 G1 PSF mode comparison: signed shape-only versus nonnegative sensitivity."""
import argparse
import json
from pathlib import Path
import numpy as np

from lcd_reconstruction.datasets.general_scenes import WAVELENGTHS
from lcd_reconstruction.operators.measured_psf_source import load_measured_kernel


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / den) if den > 0 else 1.0


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def radial_energy(kernel):
    h, w = kernel.shape
    yy, xx = np.indices((h, w))
    radius = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    energy = np.abs(kernel)
    total = energy.sum()
    out = {}
    for frac in (0.5, 0.9, 0.99):
        target = frac * total
        bins = np.arange(0, radius.max() + 1.0, 1.0)
        cumulative = np.array([energy[radius <= r].sum() for r in bins])
        out[f"radius_{int(frac * 100)}pct"] = float(bins[np.searchsorted(cumulative, target)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scale", type=int, default=8)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    signed, integrated, signed1024, integrated1024, audits = [], [], [], [], []
    for nm in WAVELENGTHS:
        k, ki, audit = load_measured_kernel(args.psf, int(nm), 512, args.scale)
        k1024, ki1024, _ = load_measured_kernel(args.psf, int(nm), 1024, args.scale)
        signed.append(k)
        integrated.append(ki)
        signed1024.append(k1024)
        integrated1024.append(ki1024)
        audits.append(audit)
    signed = np.asarray(signed)
    integrated = np.asarray(integrated)
    signed1024 = np.asarray(signed1024)
    integrated1024 = np.asarray(integrated1024)

    corr_before = np.array([[corr(a, b) for b in signed] for a in signed])
    corr_after = np.array([[corr(a, b) for b in integrated] for a in integrated])
    off = ~np.eye(21, dtype=bool)
    average = signed.mean(0)
    average_integrated = integrated.mean(0)
    achromatic_before = [corr(k, average) for k in signed]
    achromatic_after = [corr(k, average_integrated) for k in integrated]

    nonnegative = np.maximum(integrated, 0)
    nonnegative = nonnegative / nonnegative.sum((1, 2), keepdims=True)
    mode_p = dict(
        negative_fraction=[float((k < 0).mean()) for k in integrated],
        relative_l2_change=[rel(p, k) for p, k in zip(nonnegative, integrated)],
        removed_absolute_energy_fraction=[
            float((np.abs(k).sum() - p.sum()) / np.abs(k).sum())
            for p, k in zip(nonnegative, integrated)],
    )

    center512 = integrated1024[:, 32:96, 32:96]
    center512 = center512 / center512.sum((1, 2), keepdims=True)
    support_compare = [rel(a, b) for a, b in zip(integrated, center512)]

    result = dict(
        protocol="D5-G1-psf-modes",
        scale=args.scale,
        primary_mode="S_signed_shape_only",
        wavelength_nm=[int(v) for v in WAVELENGTHS],
        band_pair_correlation=dict(
            before_integration_mean=float(corr_before[off].mean()),
            before_integration_min=float(corr_before[off].min()),
            after_integration_mean=float(corr_after[off].mean()),
            after_integration_min=float(corr_after[off].min()),
            achromatic_before_mean=float(np.mean(achromatic_before)),
            achromatic_after_mean=float(np.mean(achromatic_after)),
        ),
        integration=dict(
            integrated_sum=[float(k.sum()) for k in integrated],
            negative_fraction_before=[float((k < 0).mean()) for k in signed],
            negative_fraction_after=[float((k < 0).mean()) for k in integrated],
            note="8x8 sum is exactly block integration; information change is assessed by band-pair correlation",
        ),
        support=dict(
            signed_512=[radial_energy(k) for k in signed],
            integrated_64=[radial_energy(k) for k in integrated],
            support512_vs_central512_of_1024_relative_l2=support_compare,
        ),
        mode_p_nonnegative_projection=mode_p,
        audit=audits,
        note="Mode P is a sensitivity analysis, not a calibrated nonnegative PSF; 512 and 1024 supports remain signed.",
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(
        corr_before=result["band_pair_correlation"]["before_integration_mean"],
        corr_after=result["band_pair_correlation"]["after_integration_mean"],
        mode_p_change=float(np.mean(mode_p["relative_l2_change"])),
        support_change=float(np.mean(support_compare)),
    )))


if __name__ == "__main__":
    main()


