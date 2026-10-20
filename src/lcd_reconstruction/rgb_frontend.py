"""Minimal R interface: explicit Bayer demosaicing and optional OpenCV YuNet.

No spectral reconstruction, face ground truth, UDC deblurring or learned colour
calibration is hidden here. Scalar N1 observations must not be passed as Bayer.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np


def recover_rgb(raw, *, cfa, black_level, white_level, observation_id):
    if cfa not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        raise ValueError("explicit four-letter Bayer layout required; scalar is not Bayer")
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype not in (np.uint8, np.uint16) or min(raw.shape) < 4:
        raise ValueError("integer [H,W] raw mosaic required")
    if not observation_id or not 0 <= black_level < white_level <= np.iinfo(raw.dtype).max:
        raise ValueError("observation identity and valid camera levels required")
    # Explicit 4-letter names avoid OpenCV's historical 2-letter channel ambiguity.
    rgb = cv2.cvtColor(raw, getattr(cv2, f"COLOR_Bayer{cfa}2RGB")).astype(np.float32)
    rgb = (rgb-black_level)/(white_level-black_level)
    return dict(rgb_image=rgb, candidate_boxes=None, candidate_scores=None,
                metadata=dict(observation_id=observation_id, cfa=cfa,
                              raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                              coordinates="sensor pixel xywh; full image, no crop/resize",
                              recovery="OpenCV bilinear demosaic only; no UDC deblurring",
                              radiometry="black/white scaled; no calibrated colour matrix",
                              detector_status="not_run", unclipped=True))


def propose_faces(result, model_path, threshold=.9):
    """Use an explicitly supplied, licensed YuNet model; no automatic downloads."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rgb = result["rgb_image"]
    h, w = rgb.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(path), "", (w, h), threshold, .3, 5000)
    # This explicit clipping/quantization is for detection only, not N1 or saved RGB.
    bgr = np.ascontiguousarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)[..., ::-1])
    _, faces = detector.detect(bgr)
    result = dict(result)
    result["candidate_boxes"] = np.empty((0, 4), np.float32) if faces is None else faces[:, :4].copy()
    result["candidate_scores"] = np.empty((0,), np.float32) if faces is None else faces[:, -1].copy()
    result["metadata"] = dict(result["metadata"], detector_status="executed_not_accuracy_validated",
                              detector="OpenCV YuNet", score_threshold=threshold,
                              model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result
