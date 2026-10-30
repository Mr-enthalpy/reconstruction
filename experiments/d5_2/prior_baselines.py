"""D5.2 low-cost non-unrolling baselines on validation only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics
from metrics import reconstruction_metrics


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
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    train = np.load(args.prepared / "train.npz")["cube"]
    val = np.load(args.prepared / "val.npz")["cube"]
    kernels = np.load(args.prepared / "kernels.npz")["kernels"]
    op = SinglePlanePhysics(torch.tensor(kernels), tuple(val.shape[-2:])).to(args.device)
    mean_spectrum = torch.tensor(train.mean((0, 2, 3)), dtype=torch.float32, device=args.device)
    L = op.lipschitz_bound().clamp_min(1e-8)
    rows = {"landweber100": [], "training_mean_spectrum": []}
    with torch.no_grad():
        for record in records:
            if record["split"] != "val":
                continue
            x = torch.from_numpy(val[record["array_index"]:record["array_index"] + 1]).to(args.device)
            y = op(x)
            land = op.adjoint(y) / L
            for _ in range(100):
                land -= op.adjoint(op(land) - y) / L
            prior = y[:, 0][:, None] * mean_spectrum[None, :, None, None] / mean_spectrum.sum()
            for name, estimate in (("landweber100", land), ("training_mean_spectrum", prior)):
                rows[name].append(dict(
                    scene_id=record["scene_id"], dataset_id=record["dataset_id"],
                    metrics=reconstruction_metrics(
                        estimate[0].cpu().numpy(), x[0].cpu().numpy(), y[0].numpy(),
                        op(estimate)[0].numpy())))
    result = dict(protocol="D5.2-prior-baselines", validation_only=True,
                  mean_spectrum=mean_spectrum.cpu().numpy().tolist(),
                  methods={name: aggregate(values) for name, values in rows.items()})
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: result["methods"][name]["ALL"] for name in result["methods"]}))


if __name__ == "__main__":
    main()

