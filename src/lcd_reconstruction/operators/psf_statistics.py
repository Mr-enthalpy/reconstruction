"""D5.1 PSF statistics with explicit projection and normalization semantics."""
import numpy as np


def relative_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def negative_fraction(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    return float((kernel < 0).mean())


def negative_absolute_mass_fraction(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    denom = np.abs(kernel).sum()
    return float(np.abs(kernel[kernel < 0]).sum() / denom) if denom > 0 else 0.0


def nonnegative_projection(kernel):
    return np.maximum(np.asarray(kernel, dtype=np.float64), 0.0)


def projection_summary(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    clipped = nonnegative_projection(kernel)
    clipped_norm = clipped / clipped.sum() if clipped.sum() > 0 else clipped
    reference_norm = kernel / kernel.sum() if kernel.sum() != 0 else kernel
    return dict(
        negative_fraction_before_projection=negative_fraction(kernel),
        negative_fraction_after_projection=negative_fraction(clipped),
        negative_fraction_after_shape_renormalization=negative_fraction(clipped_norm),
        negative_absolute_mass_fraction_before_renormalization=negative_absolute_mass_fraction(kernel),
        removed_absolute_mass_fraction_before_renormalization=float(
            (np.abs(kernel).sum() - clipped.sum()) / np.abs(kernel).sum()
        ) if np.abs(kernel).sum() > 0 else 0.0,
        shape_relative_l2_change=relative_l2(clipped_norm, reference_norm),
        kernel_sum_before=float(kernel.sum()),
        kernel_sum_after_projection=float(clipped.sum()),
        kernel_sum_after_shape_renormalization=float(clipped_norm.sum()),
    )


def signed_absolute_radius_descriptive(kernel, fractions=(0.5, 0.9, 0.99)):
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("expected 2D kernel")
    h, w = kernel.shape
    yy, xx = np.indices((h, w))
    radius = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    energy = np.abs(kernel)
    total = energy.sum()
    bins = np.arange(0, radius.max() + 1.0, 1.0)
    cumulative = np.array([energy[radius <= r].sum() for r in bins])
    out = {}
    for fraction in fractions:
        out[f"radius_{int(fraction * 100)}pct"] = float(
            bins[np.searchsorted(cumulative, fraction * total)])
    out["physical_energy_enclosure"] = False
    out["note"] = "descriptive signed-absolute radial mass; not a physical PSF energy enclosure"
    return out


def normalization_processing_consistency(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("consistency comparison requires equal shapes")
    return dict(
        relative_l2=relative_l2(a, b),
        status="processing consistency only",
        note="renormalized central-support comparison; not evidence that support truncation is lossless",
    )
