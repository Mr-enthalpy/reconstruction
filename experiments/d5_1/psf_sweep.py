"""D5.1 support/integration sweep from one corrected 1024 PSF source."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from lcd_reconstruction.datasets.general_scenes import WAVELENGTHS
from lcd_reconstruction.operators.measured_psf_source import (
    block_integrate, crop_center, load_corrected_psf, normalize_sum,
)
from lcd_reconstruction.operators.psf_statistics import (
    normalization_processing_consistency, projection_summary,
    signed_absolute_radius_descriptive,
)

CONFIGS = [
    dict(id="baseline", support=512, integration=8),
    dict(id="S1", support=256, integration=4),
    dict(id="S2", support=192, integration=2),
    dict(id="S3", support=256, integration=2),
    dict(id="S4", support=192, integration=1),
    dict(id="S5", support=256, integration=1),
]


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / den) if den > 0 else 1.0


def rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / np.sqrt((np.linalg.norm(a) ** 2 + np.linalg.norm(b) ** 2) / 2))


def otf_metrics(kernels):
    mags = np.abs(np.fft.rfft2(kernels, axes=(-2, -1)))
    h, w = mags.shape[-2:]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(kernels.shape[-1])[None, :]
    radius = np.hypot(fy, fx)
    masks = dict(low=radius < .10, mid=(radius >= .10) & (radius < .30), high=radius >= .30)
    pairs = [(0, 10), (10, 20), (0, 20)]
    values = {}
    for name, mask in masks.items():
        values[name] = float(np.mean([
            rel(mags[a][mask], mags[b][mask]) for a, b in pairs]))
    matrix = mags.reshape(21, -1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    cumulative = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    rank99 = int(np.searchsorted(cumulative, .99) + 1)
    total = np.abs(mags).sum()
    return dict(relative_l2_by_frequency=values,
                effective_rank_99=rank99,
                singular_values=[float(v) for v in singular[:10]],
                high_frequency_abs_energy_fraction=float(np.abs(mags[..., masks["high"]]).sum() / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    (args.output / "kernels").mkdir(parents=True)
    corrected = {}
    source_audit = {}
    for nm in WAVELENGTHS:
        corrected[int(nm)], source_audit[int(nm)] = load_corrected_psf(args.psf, int(nm))
    result = dict(
        protocol="D5.1-support-integration-sweep",
        source="spectral21_20261018_reset01",
        source_roi_xywh=[866, 554, 1024, 1024],
        common_origin="fixed central 1024 ROI; no per-band shift",
        normalization_primary="signed shape-only (unit sum)",
        amplitude_mode="signed block sum with no post-truncation renormalization",
        configs=[],
        source_audit=source_audit,
    )
    for config in CONFIGS:
        support, factor = config["support"], config["integration"]
        shapes, amplitudes, audits = [], [], []
        for nm in WAVELENGTHS:
            cropped = crop_center(corrected[int(nm)], support)
            amp = block_integrate(cropped, factor, "sum")
            shape = normalize_sum(amp)
            if not np.isfinite(shape).all():
                raise ValueError(f"invalid kernel {config['id']} {nm}")
            shapes.append(shape)
            amplitudes.append(amp)
            audits.append(dict(
                wavelength_nm=int(nm),
                kernel_sum_amplitude=float(amp.sum()),
                kernel_abs_sum_amplitude=float(np.abs(amp).sum()),
                kernel_sum_shape=float(shape.sum()),
                negative_fraction_shape=float((shape < 0).mean()),
                projection=projection_summary(shape),
                signed_absolute_radius=signed_absolute_radius_descriptive(shape),
            ))
        shapes = np.asarray(shapes, dtype=np.float32)
        amplitudes = np.asarray(amplitudes, dtype=np.float32)
        corr_matrix = np.array([[corr(a, b) for b in shapes] for a in shapes])
        off = ~np.eye(21, dtype=bool)
        cross = dict(
            adjacent_mean=float(np.mean([corr(shapes[i], shapes[i + 1]) for i in range(20)])),
            all_offdiag_mean=float(corr_matrix[off].mean()),
            all_offdiag_min=float(corr_matrix[off].min()),
            pair_450_550=corr(shapes[0], shapes[10]),
            pair_450_650=corr(shapes[0], shapes[20]),
            pair_550_650=corr(shapes[10], shapes[20]),
            relative_l2_450_550=rel(shapes[0], shapes[10]),
            relative_l2_450_650=rel(shapes[0], shapes[20]),
            relative_l2_550_650=rel(shapes[10], shapes[20]),
        )
        offset = (1024 - support) // 2
        config_result = dict(
            **config,
            kernel_hw=[support // factor, support // factor],
            crop_xywh_sensor=[866 + offset, 554 + offset, support, support],
            crop_origin_shared=[offset, offset],
            scene_to_sensor_scale_assumption=f"one simulation pixel = {factor} sensor pixels; assumed, not calibrated",
            support_abs_mass_fraction=[float(np.abs(crop_center(corrected[int(nm)], support)).sum() /
                                             np.abs(corrected[int(nm)]).sum()) for nm in WAVELENGTHS],
            support_signed_mass_fraction=[float(crop_center(corrected[int(nm)], support).sum() /
                                               corrected[int(nm)].sum()) for nm in WAVELENGTHS],
            bands=audits,
            band_correlation_matrix=np.round(corr_matrix, 6).tolist(),
            cross_band=cross,
            otf=otf_metrics(shapes),
            normalization_processing_consistency=dict(
                comparison="normalized 512-support 8x kernel versus central 64x64 of normalized 1024-support 8x kernel",
                **normalization_processing_consistency(
                    normalize_sum(block_integrate(crop_center(corrected[550], 512), 8)),
                    normalize_sum(block_integrate(corrected[550], 8))[32:96, 32:96])),
            kernel_shape_sha256=sha_array(shapes),
            kernel_amplitude_sha256=sha_array(amplitudes),
        )
        result["configs"].append(config_result)
        np.savez_compressed(args.output / "kernels" / f"{config['id']}_shape.npz", kernels=shapes)
        np.savez_compressed(args.output / "kernels" / f"{config['id']}_amplitude.npz", kernels=amplitudes)
    (args.output / "psf_support_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([dict(id=c["id"], kernel=c["kernel_hw"],
                           corr_off=c["cross_band"]["all_offdiag_mean"],
                           rel=c["cross_band"]["relative_l2_450_550"],
                           rank=c["otf"]["effective_rank_99"])
                      for c in result["configs"]]))


if __name__ == "__main__":
    main()


