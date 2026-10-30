"""Paired comparison of two matched D5.2 arms on the same validation scenes.

Both arms must have been trained from the same code, data, configuration and
seed, differing only in the PSF kernel file. The comparison has three parts:

1. primary: each arm's rule-selected best checkpoint on the same scenes;
2. secondary: the metric trajectory at matching epoch numbers;
3. an uncertainty scale: the within-arm spread of a metric over its plateau.

No checkpoint is re-selected here; the arms' own ``min_validation_loss`` rule
decides which weights are compared.
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "cube_relative_l2",
        "global_spectrum_relative_l2", "high_frequency_relative_l2",
        "spatial_gradient_relative_l2", "measurement_relative_l2"]
LOWER_IS_BETTER = {"sam_deg", "cube_relative_l2", "global_spectrum_relative_l2",
                   "high_frequency_relative_l2", "spatial_gradient_relative_l2",
                   "measurement_relative_l2"}


def load_arm(directory):
    directory = Path(directory)
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    return dict(directory=str(directory), validation=validation, history=history,
                execution=execution)


def best_metrics(arm):
    selection = arm["validation"].get("selection") or {}
    metrics = selection.get("metrics_in_checkpoint")
    if metrics and "ALL" in metrics:
        return metrics
    rows = [r for r in arm["validation"]["results"] if r]
    return {"ALL": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in KEYS}}


def per_scene(arm):
    return {r["scene_id"]: r for r in arm["validation"]["results"]}


def trajectory(arm):
    return {r["epoch"]: r["metrics"]["ALL"] for r in arm["history"] if "metrics" in r}


def plateau_spread(rows, window=40):
    recent = rows[-window:]
    return {key: float(np.std([r[key] for r in recent], ddof=1)) if len(recent) > 2 else None
            for key in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromatic", type=Path, required=True)
    ap.add_argument("--achromatic", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arms = dict(chromatic=load_arm(args.chromatic), achromatic=load_arm(args.achromatic))

    summary = {}
    for name, arm in arms.items():
        best = best_metrics(arm)["ALL"]
        summary[name] = dict(
            directory=arm["directory"],
            best_epoch=arm["validation"].get("best_epoch"),
            best_validation_loss=arm["validation"].get("best_validation_loss"),
            epochs_completed=arm["history"][-1]["epoch"],
            global_steps=arm["history"][-1]["global_step"],
            kernel_sha256_array=arm["execution"].get("kernel_sha256_array"),
            kernel_sha256_file=arm["execution"].get("kernel_sha256"),
            best_metrics=best)

    scenes = per_scene(arms["chromatic"])
    other = per_scene(arms["achromatic"])
    common = sorted(set(scenes) & set(other))
    paired = []
    for scene in common:
        row = dict(scene_id=scene, dataset_id=scenes[scene]["dataset_id"], metrics={})
        for key in KEYS:
            c = scenes[scene]["metrics"][key]
            a = other[scene]["metrics"][key]
            row["metrics"][key] = dict(chromatic=c, achromatic=a, delta=c - a)
        paired.append(row)

    aggregate = {}
    for key in KEYS:
        c = summary["chromatic"]["best_metrics"][key]
        a = summary["achromatic"]["best_metrics"][key]
        better = "chromatic" if ((c < a) if key in LOWER_IS_BETTER else (c > a)) else "achromatic"
        wins = sum(1 for row in paired
                   if ((row["metrics"][key]["delta"] < 0) if key in LOWER_IS_BETTER
                       else (row["metrics"][key]["delta"] > 0)))
        aggregate[key] = dict(chromatic=c, achromatic=a, delta=c - a,
                              relative_delta=(c - a) / a if a else None,
                              better=better,
                              scenes_favouring_chromatic=wins, scenes_compared=len(paired))

    chromatic_traj = trajectory(arms["chromatic"])
    achromatic_traj = trajectory(arms["achromatic"])
    shared = sorted(set(chromatic_traj) & set(achromatic_traj))
    traj_rows = [dict(epoch=e,
                      **{key: dict(chromatic=chromatic_traj[e][key],
                                   achromatic=achromatic_traj[e][key],
                                   delta=chromatic_traj[e][key] - achromatic_traj[e][key])
                         for key in KEYS}) for e in shared]

    chrom_rows = [chromatic_traj[e] for e in sorted(chromatic_traj)]
    achrom_rows = [achromatic_traj[e] for e in sorted(achromatic_traj)]
    result = dict(
        protocol="D5.2-matched-arm-comparison",
        selection_rule="each arm keeps its own minimum-validation-loss checkpoint",
        arms=summary,
        comparison=dict(paired_scenes=paired, aggregate=aggregate,
                        trajectory_overlap_epochs=shared, trajectory=traj_rows),
        plateau_spread=dict(window_metric_rows=40,
                            chromatic=plateau_spread(chrom_rows),
                            achromatic=plateau_spread(achrom_rows)),
        notes=["no checkpoint is re-selected from this comparison",
               "differences below the plateau spread are not treated as demonstrated"])
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: dict(chromatic=round(v["chromatic"], 4),
                              achromatic=round(v["achromatic"], 4),
                              delta=round(v["delta"], 4), better=v["better"])
                      for k, v in aggregate.items()}, indent=2))
    print(json.dumps({name: dict(best_epoch=summary[name]["best_epoch"],
                                 epochs=summary[name]["epochs_completed"])
                      for name in summary}, indent=2))


if __name__ == "__main__":
    main()
