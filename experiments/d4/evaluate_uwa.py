"""Unseen-domain diagnostic on whole UWA frames, without face ROI/label filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from train import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.archive.open("rb") as f:
        archive_sha = hashlib.file_digest(f, "sha256").hexdigest()
    if archive_sha != "34f78b92f496285d8f32374f94a777fae82896b433085d8e0fe40096715b9c53":
        raise ValueError("not the audited UWA v1.1 archive")
    subjects = sorted(s for s, split in json.loads(args.splits.read_text())["subject_to_split"].items()
                      if split == "test")
    if len(subjects) != 16:
        raise ValueError("expected 16 frozen D1 test subjects")
    checkpoint = torch.load(args.run/"best.pt", map_location=args.device, weights_only=True)
    cfg = checkpoint["config"]
    k = np.load(args.prepared/"kernels.npz")["kernels"]
    model = GeneralSpectralPGD(k, tuple(checkpoint["hw"]), cfg["stages"],
                               cfg["hidden"], cfg["depth"]).to(args.device)
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    torch.set_num_threads(4)
    rows = []
    with zipfile.ZipFile(args.archive) as archive, torch.no_grad():
        for subject in subjects:
            bands, hashes, limit = [], [], []
            # One predetermined session per subject; no image/content-based selection.
            for band in range(6, 27):
                raw = archive.read(f"UWA HSFD V1.1/HyperSpec_Face_Session1/{subject}/F_{band:02d}.png")
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                if image.shape != (1024, 1024) or image.dtype != np.uint8:
                    raise ValueError("unexpected UWA frame")
                hashes.append(hashlib.sha256(raw).hexdigest())
                limit.append(float((image == 255).mean()))
                bands.append(cv2.resize(image.astype(np.float32)/255,
                             tuple(reversed(checkpoint["hw"])), interpolation=cv2.INTER_AREA))
            x = torch.from_numpy(np.stack(bands)[None]).to(args.device)
            y = model.physics(x)
            out = model(y)
            rows.append(dict(subject_id=subject, session=1, band_sha256=hashes,
                             source_255_fraction=limit,
                             metrics=metrics(out["spectral_cube_hat"], x, out["reprojected_measurement"], y)))
    aggregate = {key: float(np.mean([r["metrics"][key] for r in rows])) for key in
                 ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "map_relative_l2",
                  "spatial_gradient_relative_l2", "measurement_relative_l2", "global_spectrum_relative_l2"]}
    (args.output/"private_samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output/"aggregate.json").write_text(json.dumps(dict(
        dataset="UWA HSFD v1.1", independent_subjects=len(subjects), acquisitions=len(rows),
        scene_selection="all frozen D1 test subjects, first session, full frame; no ROI/label filter",
        model_fit_on_uwa=False, final_cave_reserve_accessed=False,
        archive_sha256=archive_sha,
        checkpoint_sha256=hashlib.sha256((args.run/"best.pt").read_bytes()).hexdigest(),
        normalization="DN / 255; processed intensity, not CAVE reflectance or calibrated radiance",
        limitations=["Uncorrected sequential-band motion", "255 is not proven saturation",
                     "Whole-frame resizing changes assumed magnification",
                     "Some source acquisitions were evaluated historically in D3; no D4 N1 fitting"],
        metrics=aggregate), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
