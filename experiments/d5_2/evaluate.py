"""Evaluate a frozen D5.2 checkpoint; test loading requires an explicit flag."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from lcd_reconstruction.unrolling.hsfaut import HSFAUReconstructor
from metrics import reconstruction_metrics


def build_model(cfg, kernels, hw):
    if cfg["model"] == "A":
        return GeneralSpectralPGD(kernels, hw, cfg["stages"], cfg["width"], cfg["depth"])
    return HSFAUReconstructor(kernels, hw, stages=cfg["stages"], width=cfg["width"],
                              depth=cfg["depth"],
                              prior="cnn" if cfg["model"] == "B" else "sfat",
                              sfat_width=cfg["sfat_width"], sfat_levels=cfg["sfat_levels"])


def aggregate(rows):
    keys = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "cube_relative_l2",
            "global_spectrum_relative_l2", "high_frequency_relative_l2",
            "spatial_gradient_relative_l2", "measurement_relative_l2"]
    out = {}
    for source in ("CAVE", "KAIST", "ALL"):
        selected = rows if source == "ALL" else [r for r in rows if r["dataset_id"] == source]
        out[source] = {key: float(np.mean([r["metrics"][key] for r in selected])) for key in keys}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-file", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("frozen test requires --allow-test")
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(args.prepared / f"{args.split}.npz")["cube"]
    kernels = np.load(args.kernel_file)["kernels"]
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    cfg = checkpoint["config"]
    model = build_model(cfg, kernels, tuple(checkpoint["hw"])).to(args.device)
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    rows = []
    with torch.no_grad():
        for record in records:
            if record["split"] != args.split:
                continue
            x = torch.from_numpy(data[record["array_index"]:record["array_index"] + 1]).to(args.device)
            y = model.physics(x)
            out = model(y)
            rows.append(dict(scene_id=record["scene_id"], dataset_id=record["dataset_id"],
                             metrics=reconstruction_metrics(
                                 out["spectral_cube_hat"][0].cpu().numpy(),
                                 x[0].cpu().numpy(), y[0].numpy(),
                                 out["reprojected_measurement"][0].cpu().numpy())))
    result = dict(protocol="D5.2-frozen-evaluation", split=args.split,
                  allow_test=args.allow_test, checkpoint=str(args.checkpoint),
                  kernel_sha256=__import__("hashlib").sha256(args.kernel_file.read_bytes()).hexdigest(),
                  scenes=rows, aggregate=aggregate(rows),
                  watercolors=next((r for r in rows if r["scene_id"] == "watercolors"), None),
                  note="metrics are fixed-range; no test-dependent rescaling")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"]))


if __name__ == "__main__":
    main()
