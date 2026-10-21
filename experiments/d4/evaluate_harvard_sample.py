"""One independently acquired Harvard scene: format/range diagnostic, not a benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.io import loadmat
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from train import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(a.sample.read_bytes()).hexdigest()
    if digest != "d1d59fb149bbf127a354ef8893d801eeb06f21266c7d8af8b34b213609a863b6":
        raise ValueError("expected independently downloaded img3.mat")
    z = loadmat(a.sample, verify_compressed_data_integrity=True)
    if z["ref"].shape != (1040, 1392, 31) or not np.isfinite(z["ref"]).all():
        raise ValueError("invalid Harvard cube")
    # Fixed centred valid truth region, THEN whole-field forward simulation.
    # Never crop a previously convolved sensor observation.
    roi = (440, 264, 512, 512)
    if not (z["lbl"][264:776, 440:952] != 0).all():
        raise ValueError("predeclared central ROI includes invalid/moving pixels")
    cp = torch.load(a.run/"best.pt", map_location="cuda", weights_only=True)
    cfg = cp["config"]
    model = GeneralSpectralPGD(np.load(a.prepared/"kernels.npz")["kernels"],
                              tuple(cp["hw"]), cfg["stages"], cfg["hidden"], cfg["depth"]).cuda()
    model.load_state_dict(cp["state"])
    model.eval().requires_grad_(False)
    cube = np.stack([cv2.resize(z["ref"][264:776, 440:952, b].astype(np.float32),
                     tuple(reversed(cp["hw"])), interpolation=cv2.INTER_AREA) for b in range(3, 24)])
    cases = {}
    with torch.no_grad():
        # Both fixed gains reported. Neither is tuned or used to select a model.
        for gain in [1.0, 10.0]:
            x = torch.from_numpy((cube*gain)[None]).cuda()
            y = model.physics(x)
            out = model(y)
            cases[str(gain)] = metrics(out["spectral_cube_hat"], x, out["reprojected_measurement"], y)
            np.savez_compressed(a.output/f"private_gain{gain:g}.npz", truth=x.cpu().numpy(),
                observation=y.cpu().numpy(), reconstruction=out["spectral_cube_hat"].cpu().numpy())
    report = dict(scene_count=1, source="Harvard artificial/mixed illumination img3",
                  sample_sha256=digest, crop_xywh=roi, source_shape=list(z["ref"].shape),
                  nominal_wavelengths_nm=list(range(450, 651, 10)), all_roi_pixels_valid=True,
                  source_range=[float(z["ref"].min()), float(z["ref"].max())],
                  crop_mean_native=float(cube.mean()), cases=cases,
                  checkpoint_sha256=hashlib.sha256((a.run/"best.pt").read_bytes()).hexdigest(),
                  interpretation="As-provided camera-dependent float intensity; not CAVE reflectance. "
                  "No sensitivity-vector correction or absolute radiometric calibration assumed. "
                  "Gain10 is an explicit synthetic illumination/range sensitivity, not a calibrated conversion. "
                  "Fixed PSNR range1 can flatter dim data; read relative errors and SAM too. "
                  "One sample cannot establish domain generalization; full archive not acquired.")
    (a.output/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
