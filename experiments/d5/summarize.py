"""Aggregate D5 validation outputs without scene-wise test selection."""
import argparse
import json
from pathlib import Path
import numpy as np

KEYS = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "cube_relative_l2",
        "global_spectrum_relative_l2", "high_frequency_relative_l2",
        "spatial_gradient_relative_l2", "measurement_relative_l2", "negative_fraction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="name=path-to-validation.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = []
    for item in args.run:
        name, path = item.split("=", 1)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        grouped = {}
        for source in ["CAVE", "KAIST"]:
            rows = [r["metrics"] for r in data["results"] if r["dataset_id"] == source]
            if rows:
                grouped[source] = {key: float(np.mean([r[key] for r in rows])) for key in KEYS}
        all_rows = [r["metrics"] for r in data["results"]]
        grouped["ALL"] = {key: float(np.mean([r[key] for r in all_rows])) for key in KEYS}
        execution = data["execution"]
        runs.append(dict(name=name, model=execution["model"], parameters=execution["parameters"],
                         best_epoch=data["best_epoch"],
                         best_validation_loss=data["best_validation_loss"],
                         seconds=data["seconds"], metrics=grouped,
                         config=execution["config"]))
    args.output.write_text(json.dumps(dict(protocol="D5-G2-validation-summary", runs=runs),
                                     indent=2), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("name", "model", "parameters", "best_epoch")}
                      for r in runs]))


if __name__ == "__main__":
    main()
