"""Fixed-range D5 reconstruction metrics; no test-dependent rescaling."""
import cv2
import numpy as np


def _relative(a, b):
    return float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-12))


def reconstruction_metrics(prediction, truth, measurement, projected, include_ssim=True):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    measurement = np.asarray(measurement, dtype=np.float64).squeeze()
    projected = np.asarray(projected, dtype=np.float64).squeeze()
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("expected matching [21,H,W] arrays")
    error = prediction - truth
    valid = np.linalg.norm(truth, axis=0) > 1e-3
    cosine = (prediction * truth).sum(0) / np.maximum(
        np.linalg.norm(prediction, axis=0) * np.linalg.norm(truth, axis=0), 1e-12)
    # SSIM is by far the most expensive term (a Gaussian blur per band), so cadence
    # evaluations may skip it. The key is then present but None, which is not a
    # measured zero: consumers must skip it rather than average it.
    ssim = []
    if include_ssim:
        for a, b in zip(prediction, truth):
            blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
            ma, mb = blur(a), blur(b)
            va, vb = blur(a * a) - ma * ma, blur(b * b) - mb * mb
            cov = blur(a * b) - ma * mb
            ssim.append(float((((2 * ma * mb + .01 ** 2) * (2 * cov + .03 ** 2)) /
                               ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2))
                               )[5:-5, 5:-5].mean()))
    gy, gx = np.meshgrid(np.fft.fftfreq(error.shape[-2]),
                         np.fft.fftfreq(error.shape[-1]), indexing="ij")
    high = np.hypot(gy, gx) >= .25
    grad_error = np.concatenate([np.diff(error, axis=-1).ravel(),
                                 np.diff(error, axis=-2).ravel()])
    grad_truth = np.concatenate([np.diff(truth, axis=-1).ravel(),
                                 np.diff(truth, axis=-2).ravel()])
    return dict(
        psnr_unit_range_db=float(-10 * np.log10(max(float((error * error).mean()), 1e-12))),
        ssim_band_mean=(float(np.mean(ssim)) if ssim else None),
        sam_deg=float(np.degrees(np.arccos(np.clip(cosine[valid], -1, 1))).mean()),
        valid_spectral_fraction=float(valid.mean()),
        cube_relative_l2=_relative(error, truth),
        global_spectrum_relative_l2=_relative(error.mean((1, 2)), truth.mean((1, 2))),
        band_rmse=np.sqrt((error * error).mean((1, 2))).tolist(),
        band_relative_l2=[_relative(e, t) for e, t in zip(error, truth)],
        high_frequency_relative_l2=_relative(
            np.fft.fft2(error)[..., high], np.fft.fft2(truth)[..., high]),
        spatial_gradient_relative_l2=_relative(grad_error, grad_truth),
        measurement_relative_l2=_relative(projected - measurement, measurement),
        negative_fraction=float((prediction < 0).mean()),
    )
