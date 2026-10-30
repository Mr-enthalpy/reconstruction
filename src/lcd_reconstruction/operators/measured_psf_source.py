"""Read-only measured PSF source handling for D5/D5.1."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corrected_psf(psf_root, nm):
    psf_root = Path(psf_root)
    roi_path = psf_root / f"roi_extent_review/roi1024_{nm}.npz"
    with np.load(roi_path) as z:
        meta = json.loads(str(z["metadata_json"]))
        if (meta["wavelength_nm"] != nm or meta["roi_xywh"] != [866, 554, 1024, 1024]
                or meta["normalization"] != "none"):
            raise ValueError("PSF geometry or normalization mismatch")
        probe = {201.6: 1, 403.2: 2, 1612.8: 3}[meta["camera_settings"]["requested_exposure_us"]]
        dark_paths = [psf_root / f"received/R21Dark{phase}01/probe_{probe:03d}_mean.npy"
                      for phase in ("Pre", "Post")]
        dark = sum(np.load(p)[554:1578, 866:1890].astype(float)
                   for p in dark_paths) / 2
        corrected = z["psf"].astype(float) - dark
    border = np.ones((1024, 1024), bool)
    border[40:-40, 40:-40] = False
    corrected -= corrected[border].mean()
    if not np.isfinite(corrected).all() or corrected.sum() <= 0:
        raise ValueError("invalid corrected PSF")
    audit = dict(
        wavelength_nm=nm, roi_sha256=sha(roi_path),
        dark_sha256=[sha(p) for p in dark_paths],
        roi_xywh=[866, 554, 1024, 1024], coordinate_frame="sensor_full_frame",
        background_policy="matched pre/post dark mean plus outer-40px residual border mean",
        raw_signed_sum=float(corrected.sum()),
        raw_negative_fraction=float((corrected < 0).mean()),
    )
    return corrected, audit


def crop_center(kernel, support):
    kernel = np.asarray(kernel)
    h, w = kernel.shape[-2:]
    if support > min(h, w) or support < 1:
        raise ValueError("invalid support")
    oy, ox = (h - support) // 2, (w - support) // 2
    return kernel[oy:oy + support, ox:ox + support]


def block_integrate(kernel, factor, reduction="sum"):
    kernel = np.asarray(kernel, dtype=float)
    if factor < 1 or kernel.shape[-2] % factor or kernel.shape[-1] % factor:
        raise ValueError("factor must divide both kernel dimensions")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be sum or mean")
    h, w = kernel.shape[-2] // factor, kernel.shape[-1] // factor
    blocks = kernel.reshape(h, factor, w, factor)
    return blocks.sum((1, 3)) if reduction == "sum" else blocks.mean((1, 3))


def normalize_sum(kernel):
    kernel = np.asarray(kernel, dtype=float)
    total = kernel.sum()
    if not np.isfinite(kernel).all() or total <= 0:
        raise ValueError("kernel must be finite with positive sum")
    return kernel / total


def load_measured_kernel(psf_root, nm, support=512, scale=8):
    corrected, audit = load_corrected_psf(psf_root, nm)
    signed = crop_center(corrected, support)
    raw_signed_sum = float(signed.sum())
    signed = normalize_sum(signed)
    integrated = block_integrate(signed, scale, "sum")
    audit.update(dict(
        support_sensor_pixels=support,
        raw_signed_sum=raw_signed_sum,
        negative_fraction=float((signed < 0).mean()),
        integrated_kernel_shape=list(integrated.shape),
    ))
    return signed.astype(np.float32), integrated.astype(np.float32), audit
