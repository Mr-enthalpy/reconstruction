"""Read-only measured PSF source handling for D5 Mode S and Mode P."""
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


def load_measured_kernel(psf_root, nm, support=512, scale=8):
    psf_root = Path(psf_root)
    if support % scale:
        raise ValueError("support must be divisible by scale")
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
    offset = (1024 - support) // 2
    signed = corrected[offset:offset + support, offset:offset + support]
    if not np.isfinite(signed).all() or signed.sum() <= 0:
        raise ValueError("invalid corrected PSF")
    raw_signed_sum = float(signed.sum())
    signed = signed / raw_signed_sum
    integrated = signed.reshape(support // scale, scale, support // scale, scale).sum((1, 3))
    audit = dict(wavelength_nm=nm, support_sensor_pixels=support,
                 roi_sha256=sha(roi_path),
                 dark_sha256=[sha(p) for p in dark_paths],
                 raw_signed_sum=raw_signed_sum,
                 negative_fraction=float((signed < 0).mean()),
                 integrated_kernel_shape=list(integrated.shape))
    return signed.astype(np.float32), integrated.astype(np.float32), audit

