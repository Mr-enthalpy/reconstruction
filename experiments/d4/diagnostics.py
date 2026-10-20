"""Post-selection diagnostics; no fitting, no reserved-final access."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics
from train import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    results = json.loads((args.run/"results.json").read_text(encoding="utf-8"))
    history = json.loads((args.run/"history.json").read_text(encoding="utf-8"))
    k = np.load(args.prepared/"kernels.npz")["kernels"]
    checkpoint = torch.load(args.run/"best.pt", map_location=args.device, weights_only=True)
    cfg = checkpoint["config"]
    model = GeneralSpectralPGD(k, tuple(checkpoint["hw"]), cfg["stages"],
                               cfg["hidden"], cfg["depth"]).to(args.device)
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    alternative = {
        "achromatic_input_mismatch": np.repeat(k.mean(0, keepdims=True), 21, axis=0),
        "assumed_efficiency_ramp_0.8_to_1.2": k*np.linspace(.8, 1.2, 21, dtype=np.float32)[:, None, None],
        "negative_tail_clipped_input_mismatch": np.maximum(k, 0)/np.maximum(k, 0).sum((1, 2), keepdims=True)
    }
    # These deliberately changed input physics do NOT replace the original signed kernel.
    ops = {n: SinglePlanePhysics(torch.from_numpy(v), tuple(checkpoint["hw"])).to(args.device)
           for n, v in alternative.items()}
    summary = {}
    scalar_metrics = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "map_relative_l2",
                      "spatial_gradient_relative_l2", "global_spectrum_relative_l2",
                      "measurement_relative_l2", "high_frequency_relative_l2", "negative_fraction"]
    for split in ["train", "val", "development_test"]:
        rows = [r for r in results["results"] if r["split"] == split]
        summary[split] = {method: {name: float(np.mean([r[method][name] for r in rows]))
                                  for name in scalar_metrics} for method in ["n1", "landweber100"]}
        summary[split]["scenes"] = len(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ["train_loss", "validation_loss"]:
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
    ax.axvline(checkpoint["epoch"], color="black", ls=":", label="selected epoch")
    ax.set(xlabel="Epoch", ylabel="Reconstruction objective", yscale="log")
    ax.legend();fig.tight_layout();fig.savefig(args.output/"training_curve.png", dpi=160);plt.close(fig)
    mismatch = {}
    spectrum_records = []
    torch.manual_seed(987)
    with torch.no_grad():
        for f in sorted(args.run.glob("private_*.npz")):
            z = np.load(f)
            truth, estimate, observation = z["truth"][0], z["reconstruction"][0], z["observation"][0, 0]
            x = torch.from_numpy(z["truth"]).to(args.device)
            per_scene = {}
            for name, op in ops.items():
                y = op(x)
                out = model(y)
                per_scene[name] = metrics(out["spectral_cube_hat"], x, out["reprojected_measurement"], y)
                actual_residual = torch.linalg.vector_norm(op(out["spectral_cube_hat"])-y)/torch.linalg.vector_norm(y)
                per_scene[name]["true_changed_operator_residual"] = float(actual_residual)
            zero = model(torch.zeros_like(model.physics(x)))
            per_scene["zero_observation_prior"] = dict(
                output_rms=float(zero["spectral_cube_hat"].square().mean().sqrt()),
                global_spectrum=zero["global_spectrum"].cpu().numpy()[0].tolist())
            mismatch[f.stem.removeprefix("private_")] = per_scene
            # Same fixed physical input range for true/recovered grayscale, no image-wise gain.
            fig, axes = plt.subplots(2, 3, figsize=(10, 7))
            for col, (a, title) in enumerate([(truth[10], "Truth 550nm"), (observation, "Scalar measurement"),
                                               (estimate[10], "N1 550nm")]):
                im = axes[0, col].imshow(a, cmap="gray", vmin=0, vmax=1 if col != 1 else None)
                axes[0, col].set_title(title);axes[0, col].axis("off")
                fig.colorbar(im, ax=axes[0, col], shrink=.65)
            axes[1, 0].plot(range(450, 651, 10), truth.mean((1, 2)), label="truth")
            axes[1, 0].plot(range(450, 651, 10), estimate.mean((1, 2)), label="N1")
            axes[1, 0].set_title("Global spectrum");axes[1, 0].legend()
            im = axes[1, 1].imshow(np.sqrt(((estimate-truth)**2).mean(0)), cmap="magma", vmin=0)
            axes[1, 1].set_title("Spectral RMSE map");fig.colorbar(im, ax=axes[1, 1], shrink=.65)
            h, w = truth.shape[1:]
            local = []
            for index, (py, px) in enumerate([(h//4, w//4), (h//2, w//2), (3*h//4, 3*w//4)]):
                axes[1, 2].plot(range(450, 651, 10), truth[:, py, px], color=f"C{index}", ls="-")
                axes[1, 2].plot(range(450, 651, 10), estimate[:, py, px], color=f"C{index}", ls="--")
                local.append(dict(x=px, y=py, truth=truth[:, py, px].tolist(), n1=estimate[:, py, px].tolist()))
            axes[1, 2].set_title("Fixed pixels: truth solid / N1 dashed")
            for ax in [axes[1, 0], axes[1, 2]]:
                ax.set_xlabel("Wavelength (nm)")
            fig.suptitle(f.stem+" — approximate-reflectance units; shape-only simulation")
            fig.tight_layout();fig.savefig(args.output/f"{f.stem}_diagnostic.png", dpi=150);plt.close(fig)
            spectrum_records.append(dict(scene=f.stem, pixels=local))
    (args.output/"summary.json").write_text(json.dumps(dict(
        status=results["status"], metrics=summary, mismatch=mismatch,
        checkpoint_sha256=hashlib.sha256((args.run/"best.pt").read_bytes()).hexdigest(),
        note="Changed-input mismatch, not a retrained achromatic comparison. Final reserve not read."),
        indent=2), encoding="utf-8")
    (args.output/"private_local_spectra.json").write_text(json.dumps(spectrum_records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
