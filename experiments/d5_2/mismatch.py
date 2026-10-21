"""D5.2 inference-time PSF mismatch diagnostics; no retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from lcd_reconstruction.unrolling.hsfaut import HSFAUReconstructor
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics
from metrics import reconstruction_metrics


def build_model(cfg, kernels, hw):
    if cfg["model"] == "A":
        return GeneralSpectralPGD(kernels, hw, cfg["stages"], cfg["width"], cfg["depth"])
    return HSFAUReconstructor(kernels, hw, stages=cfg["stages"], width=cfg["width"],
                              depth=cfg["depth"],
                              prior="cnn" if cfg["model"] == "B" else "sfat",
                              sfat_width=cfg["sfat_width"], sfat_levels=cfg["sfat_levels"])


def aggregate(rows, method):
    selected = [r[method] for r in rows]
    keys = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "cube_relative_l2",
            "high_frequency_relative_l2", "spatial_gradient_relative_l2",
            "measurement_relative_l2"]
    return {key: float(np.mean([r[key] for r in selected])) for key in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--correct-kernel", type=Path, required=True)
    ap.add_argument("--alternative-kernel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    val = np.load(args.prepared / "val.npz")["cube"]
    correct = np.load(args.correct_kernel)["kernels"]
    alternative = np.load(args.alternative_kernel)["kernels"]
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    cfg = checkpoint["config"]
    model = build_model(cfg, correct, tuple(checkpoint["hw"])).cuda()
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    alt_physics = SinglePlanePhysics(torch.tensor(alternative), tuple(checkpoint["hw"])).cuda()
    rows = []
    with torch.no_grad():
        for record in records:
            if record["split"] != "val":
                continue
            x = torch.from_numpy(val[record["array_index"]:record["array_index"] + 1]).cuda()
            y_correct = model.physics(x)
            y_alt = alt_physics(x)
            matched = model(y_correct)
            mismatched = model(y_alt)
            rows.append(dict(
                scene_id=record["scene_id"], dataset_id=record["dataset_id"],
                matched=reconstruction_metrics(
                    matched["spectral_cube_hat"][0].cpu().numpy(), x[0].cpu().numpy(),
                    y_correct[0].cpu().numpy(), matched["reprojected_measurement"][0].cpu().numpy()),
                mismatch=reconstruction_metrics(
                    mismatched["spectral_cube_hat"][0].cpu().numpy(), x[0].cpu().numpy(),
                    y_alt[0].cpu().numpy(), alt_physics(mismatched["spectral_cube_hat"])[0].cpu().numpy()),
            ))
    result = dict(protocol="D5.2-inference-mismatch", validation_only=True,
                  correct_kernel=str(args.correct_kernel),
                  alternative_kernel=str(args.alternative_kernel),
                  scenes=rows, aggregate=dict(matched=aggregate(rows, "matched"),
                                              mismatch=aggregate(rows, "mismatch")),
                  note="fixed model and fixed truth; this is calibration sensitivity, not a retrained achromatic ablation")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"]))


if __name__ == "__main__":
    main()
