"""Extract fixed-pixel local spectra from a frozen A6 validation checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--kernel-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    val = np.load(args.prepared / "val.npz")["cube"]
    kernels = np.load(args.kernel_file)["kernels"]
    checkpoint = torch.load(args.run / "best.pt", map_location="cuda", weights_only=True)
    cfg = checkpoint["config"]
    model = GeneralSpectralPGD(kernels, tuple(checkpoint["hw"]), cfg["stages"],
                               cfg["width"], cfg["depth"]).cuda()
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    rows = []
    with torch.no_grad():
        for record in records:
            if record["split"] != "val":
                continue
            x = torch.from_numpy(val[record["array_index"]:record["array_index"] + 1]).cuda()
            y = model.physics(x)
            out = model(y)["spectral_cube_hat"]
            truth = x[0].cpu().numpy()
            estimate = out[0].cpu().numpy()
            h, w = truth.shape[-2:]
            pixels = [(h // 4, w // 4), (h // 2, w // 2), (3 * h // 4, 3 * w // 4)]
            local = []
            for py, px in pixels:
                t, e = truth[:, py, px], estimate[:, py, px]
                cosine = float((t * e).sum() / max(np.linalg.norm(t) * np.linalg.norm(e), 1e-12))
                local.append(dict(y=py, x=px, truth=t.tolist(), estimate=e.tolist(),
                                  sam_deg=float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))))
            rows.append(dict(scene_id=record["scene_id"], dataset_id=record["dataset_id"],
                             pixels=local))
    args.output.write_text(json.dumps(dict(protocol="D5.1-local-spectra", run=args.run.name,
                                           scenes=rows), indent=2), encoding="utf-8")
    print(json.dumps(dict(scenes=len(rows), mean_sam=float(np.mean(
        [p["sam_deg"] for r in rows for p in r["pixels"]])))))


if __name__ == "__main__":
    main()
