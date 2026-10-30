"""D5.2 radial-frequency and per-band diagnostics on validation scenes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from lcd_reconstruction.unrolling.hsfaut import HSFAUReconstructor
from lcd_reconstruction.operators.psf_provenance import (
    PSFProvenanceError, array_digest, physics_buffers_from_model,
    physics_buffers_from_state, resolve_psf_policy)
from metrics import reconstruction_metrics

BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]


def build_model(cfg, kernels, hw):
    if cfg["model"] == "A":
        return GeneralSpectralPGD(kernels, hw, cfg["stages"], cfg["width"], cfg["depth"])
    return HSFAUReconstructor(kernels, hw, stages=cfg["stages"], width=cfg["width"],
                              depth=cfg["depth"],
                              prior="cnn" if cfg["model"] == "B" else "sfat",
                              sfat_width=cfg["sfat_width"], sfat_levels=cfg["sfat_levels"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    val = np.load(args.prepared / "val.npz")["cube"]
    kernels = np.load(args.kernel_file)["kernels"]
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    cfg = checkpoint["config"]
    model = build_model(cfg, kernels, tuple(checkpoint["hw"])).cuda()
    policy = resolve_psf_policy(checkpoint.get("psf_provenance"),
                                physics_buffers_from_model(model),
                                physics_buffers_from_state(checkpoint["state"]),
                                allow_mismatch=False, split="val",
                                requested_array=array_digest(kernels))
    model.load_state_dict(checkpoint["state"])
    if physics_buffers_from_model(model) != policy["stored"]:
        raise PSFProvenanceError("--kernel-file does not match the checkpoint operator")
    model.eval().requires_grad_(False)
    rows = []
    with torch.no_grad():
        for record in records:
            if record["split"] != "val":
                continue
            x = torch.from_numpy(val[record["array_index"]:record["array_index"] + 1]).cuda()
            y = model.physics(x)
            out = model(y)
            prediction = out["spectral_cube_hat"][0].cpu().numpy()
            truth = x[0].cpu().numpy()
            error = prediction - truth
            f_error = np.fft.fft2(error)
            f_truth = np.fft.fft2(truth)
            fy = np.fft.fftfreq(error.shape[-2])[:, None]
            fx = np.fft.fftfreq(error.shape[-1])[None, :]
            radius = np.hypot(fy, fx)
            radial = {}
            for low, high in BINS:
                mask = (radius >= low) & (radius < high)
                radial[f"{low:.1f}-{high:.1f}"] = float(
                    np.linalg.norm(f_error[..., mask]) /
                    max(np.linalg.norm(f_truth[..., mask]), 1e-12))
            rows.append(dict(scene_id=record["scene_id"], dataset_id=record["dataset_id"],
                             metrics=reconstruction_metrics(
                                 prediction, truth, y[0].cpu().numpy(),
                                 out["reprojected_measurement"][0].cpu().numpy()),
                             radial_frequency_relative_l2=radial))
    keys = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "cube_relative_l2",
            "high_frequency_relative_l2", "spatial_gradient_relative_l2",
            "measurement_relative_l2"]
    aggregate = {}
    for source in ("CAVE", "KAIST", "ALL"):
        selected = rows if source == "ALL" else [r for r in rows if r["dataset_id"] == source]
        aggregate[source] = {
            "metrics": {key: float(np.mean([r["metrics"][key] for r in selected])) for key in keys},
            "radial_frequency_relative_l2": {
                band: float(np.mean([r["radial_frequency_relative_l2"][band] for r in selected]))
                for band, _ in [(f"{a:.1f}-{b:.1f}", None) for a, b in BINS]},
            "band_relative_l2": np.mean(
                [r["metrics"]["band_relative_l2"] for r in selected], axis=0).tolist(),
        }
    args.output.write_text(json.dumps(dict(protocol="D5.2-frequency-diagnostics",
                                           psf=dict(policy=policy["policy"],
                                                    formal_result=policy["formal_result"],
                                                    kernel_sha256_array=array_digest(kernels),
                                                    recorded_kernel_file=policy["recorded_kernel_file"]),
                                           scenes=rows, aggregate=aggregate), indent=2),
                           encoding="utf-8")
    print(json.dumps(aggregate["ALL"]))


if __name__ == "__main__":
    main()
