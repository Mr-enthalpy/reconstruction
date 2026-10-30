"""D5.3 multi-seed summary of the matched chromatic/achromatic arms.

Reads finished run directories only. Each arm's own ``minimum validation loss``
checkpoint is used; this script never re-selects a checkpoint and never mixes a
metric from one epoch with a metric from another.

Statistics are reported at two levels:

* seed level (n = 3): the primary evidence, mean/sd/min/max of the paired
  difference chromatic - achromatic;
* scene level (9 per seed): direction consistency only. Nine scenes inside one
  seed are not nine independent experiments.

A run whose best epoch lies inside the last ``patience`` epochs, or whose
validation loss was still improving over its final 60 epochs, is marked
``budget_censored``: at that budget its converged capability is not established.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

KEYS = ["psnr_unit_range_db", "ssim_band_mean", "sam_deg", "cube_relative_l2",
        "global_spectrum_relative_l2", "high_frequency_relative_l2",
        "spatial_gradient_relative_l2", "measurement_relative_l2"]
# Per-band entries are lists of 21 values, so they cannot live in KEYS. They are
# carried separately below; band_seeds.py is the reducer that applies the
# pre-registered stability rule band by band.
PER_BAND = ("band_relative_l2", "band_rmse")
LOWER_IS_BETTER = {"sam_deg", "cube_relative_l2", "global_spectrum_relative_l2",
                   "high_frequency_relative_l2", "spatial_gradient_relative_l2",
                   "measurement_relative_l2"}
ARM_PATTERN = re.compile(r"^(?P<arm>[a-z0-9]+)_s(?P<seed>\d+)$")


def better(metric, value, other):
    return value < other if metric in LOWER_IS_BETTER else value > other


def per_band_curve(rows):
    """Band-by-band scene mean and sd from the per-scene rows.

    ``band_relative_l2`` is a mean of per-scene ratios ``||error_b||/||truth_b||``,
    not a ratio of summed norms, so this curve is not comparable to
    ``cube_relative_l2``. Keeping it here stops the reducer from discarding a
    dimension that is already present in every validation.json.
    """
    out = {}
    for metric in PER_BAND:
        stack = np.stack([np.asarray(row["metrics"][metric], dtype=float) for row in rows])
        out[metric] = dict(scene_mean=stack.mean(axis=0).tolist(),
                           scene_sd=(stack.std(axis=0, ddof=1).tolist()
                                     if stack.shape[0] > 1 else None))
    return out


def load_run(directory: Path):
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    selection = validation.get("selection") or {}
    metrics = (selection.get("metrics_in_checkpoint") or {})
    recomputed = False
    if "ALL" not in metrics:
        rows = validation["results"]
        metrics = {source: {key: float(np.mean([r["metrics"][key] for r in rows
                                                if source == "ALL"
                                                or r["dataset_id"] == source]))
                            for key in KEYS}
                   for source in ("CAVE", "KAIST", "ALL")}
        recomputed = True
    final = history[-1]
    plateau_rows = [r for r in history if "metrics" in r][-40:]
    epochs_completed = int(final["epoch"])
    best_epoch = int(selection.get("best_epoch", validation.get("best_epoch", 0)))
    patience = int(execution["config"]["patience"])
    stopped_by_patience = bool(final.get("stale", 0) >= patience)
    censored = bool((not stopped_by_patience) and
                    (epochs_completed - best_epoch) <= patience)
    last = history[-60:]
    slope = None
    if len(last) > 3:
        epochs = np.array([r["epoch"] for r in last], dtype=float)
        values = np.array([r["validation_loss"] for r in last], dtype=float)
        slope = float(np.polyfit(epochs, values, 1)[0])
    match = ARM_PATTERN.match(directory.name)
    return dict(
        directory=directory.name,
        arm=match.group("arm") if match else directory.name,
        seed=execution["config"]["seed"],
        epochs_completed=epochs_completed,
        global_steps=final["global_step"],
        best_epoch=best_epoch,
        best_validation_loss=validation.get("best_validation_loss"),
        final_validation_loss=final["validation_loss"],
        stale=final.get("stale"),
        stopped_by_patience=stopped_by_patience,
        budget_censored=censored,
        validation_loss_slope_last_60=slope,
        kernel_sha256_array=execution.get("kernel_sha256_array"),
        kernel_file=(validation.get("psf_provenance") or {}).get("kernel_file"),
        psf_protocol=(validation.get("psf_provenance") or {}).get("protocol"),
        psf_kernel_array=(validation.get("psf_provenance") or {}).get("kernel_sha256_array"),
        measured_forward_rms=execution.get("measured_forward_rms"),
        train_noise_std_absolute=execution.get("train_noise_std_absolute"),
        selection_rule=selection.get("rule"),
        metrics_recomputed=recomputed,
        metrics=metrics,
        per_band=per_band_curve(validation["results"]),
        plateau_std={key: float(np.std([r["metrics"]["ALL"][key] for r in plateau_rows], ddof=1))
                     for key in KEYS} if len(plateau_rows) > 2 else {},
        scenes={r["scene_id"]: r["metrics"] for r in validation["results"]},
        scene_datasets={r["scene_id"]: r["dataset_id"] for r in validation["results"]})


def cross_seed(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if values.size > 1 else None,
                minimum=float(values.min()), maximum=float(values.max()), n=int(values.size))


def analyse(runs_root: Path, reference_arm: str = "s3a", primary_arm: str = "s1c"):
    runs = []
    for directory in sorted(runs_root.iterdir()):
        if not directory.is_dir() or not (directory / "validation.json").exists():
            continue
        if not ARM_PATTERN.match(directory.name):
            continue
        runs.append(load_run(directory))
    if not runs:
        raise ValueError(f"no run directories found under {runs_root}")

    by_arm = {}
    for run in runs:
        by_arm.setdefault(run["arm"], {})[run["seed"]] = run
    for arm in (primary_arm, reference_arm):
        if arm not in by_arm:
            raise ValueError(f"both {primary_arm} (primary) and {reference_arm} (reference) "
                             f"runs are required; found arms {sorted(by_arm)}")
    seeds = sorted(set(by_arm[primary_arm]) & set(by_arm[reference_arm]))

    pairs = []
    for seed in seeds:
        chrom, achrom = by_arm[primary_arm][seed], by_arm[reference_arm][seed]
        deltas = {key: chrom["metrics"]["ALL"][key] - achrom["metrics"]["ALL"][key] for key in KEYS}
        dataset_deltas = {}
        for source in ("CAVE", "KAIST"):
            dataset_deltas[source] = {
                key: chrom["metrics"][source][key] - achrom["metrics"][source][key]
                for key in KEYS}
        scene_rows = []
        for scene in sorted(set(chrom["scenes"]) & set(achrom["scenes"])):
            scene_rows.append(dict(
                scene_id=scene, dataset_id=chrom["scene_datasets"][scene],
                metrics={key: dict(chromatic=chrom["scenes"][scene][key],
                                   achromatic=achrom["scenes"][scene][key],
                                   delta=chrom["scenes"][scene][key] - achrom["scenes"][scene][key])
                         for key in KEYS}))
        pairs.append(dict(seed=seed, deltas=deltas, dataset_deltas=dataset_deltas,
                          scenes=scene_rows,
                          per_band={metric: dict(
                              chromatic=chrom["per_band"][metric]["scene_mean"],
                              achromatic=achrom["per_band"][metric]["scene_mean"],
                              delta=[c - a for c, a in zip(chrom["per_band"][metric]["scene_mean"],
                                                           achrom["per_band"][metric]["scene_mean"])])
                              for metric in PER_BAND},
                          censored=[chrom["budget_censored"], achrom["budget_censored"]]))

    aggregate = {}
    for key in KEYS:
        values = [pair["deltas"][key] for pair in pairs]
        statistics = cross_seed(values)
        direction = "chromatic" if (statistics["mean"] < 0) == (key in LOWER_IS_BETTER) else "achromatic"
        consistent = all(better(key, value, 0.0) for value in values)
        scene_direction = []
        for pair in pairs:
            wins = sum(1 for row in pair["scenes"]
                       if better(key, row["metrics"][key]["delta"], 0.0))
            scene_direction.append(wins)
        aggregate[key] = dict(
            per_seed=values, across_seeds=statistics,
            direction=direction,
            sign_consistent_across_seeds=consistent,
            scenes_favouring_chromatic=scene_direction, scenes_compared=len(pairs[0]["scenes"]),
            exceeds_seed_sd=bool(statistics["sd"] is not None and
                                 abs(statistics["mean"]) > statistics["sd"]),
            stable=bool(consistent and statistics["sd"] is not None and
                        abs(statistics["mean"]) > statistics["sd"] and
                        min(scene_direction) >= 8))

    censored = {f"{run['arm']}:{run['seed']}": run["budget_censored"] for run in runs}
    return dict(
        protocol="D5.3-multiseed-comparison",
        primary_arm=primary_arm,
        reference_arm=reference_arm,
        selection_rule="each arm keeps its own minimum-validation-loss checkpoint; "
                       "no checkpoint or metric is re-selected here",
        runs={run["directory"]: dict(arm=run["arm"], seed=run["seed"],
                                     epochs_completed=run["epochs_completed"],
                                     best_epoch=run["best_epoch"],
                                     best_validation_loss=run["best_validation_loss"],
                                     stale=run["stale"],
                                     stopped_by_patience=run["stopped_by_patience"],
                                     budget_censored=run["budget_censored"],
                                     validation_loss_slope_last_60=run["validation_loss_slope_last_60"],
                                     kernel_file=run["kernel_file"],
                                     kernel_sha256_array=run["kernel_sha256_array"],
                                     psf_protocol=run["psf_protocol"],
                                     psf_kernel_array=run["psf_kernel_array"],
                                     measured_forward_rms=run["measured_forward_rms"],
                                     train_noise_std_absolute=run["train_noise_std_absolute"],
                                     selection_rule=run["selection_rule"],
                                     metrics_recomputed=run["metrics_recomputed"],
                                     plateau_std=run["plateau_std"],
                                     per_band=run["per_band"],
                                     metrics=run["metrics"]["ALL"])
                 for run in runs},
        seeds=seeds,
        pairs=pairs,
        aggregate=aggregate,
        budget_censored=censored,
        any_censored=any(censored.values()),
        notes=["seed level n = 3 is the primary evidence",
               "nine scenes inside one seed are not independent experiments",
               "plateau epoch-to-epoch std is a training-stability diagnostic, not seed variance",
               "a run whose best epoch lies in its final patience window is budget_censored",
               "per_band curves are scene means of ||error_b||/||truth_b||, a mean of "
               "per-scene ratios and not comparable to cube_relative_l2",
               "per-band stability is not evaluated here; use band_seeds.py for that",
               "validation loss is not compared across arms: each arm uses its own forward "
               "RMS as both the noise scale and the data-loss normaliser"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference-arm", default="s3a",
                    help="arm compared against the primary arm; default s3a (mean kernel)")
    ap.add_argument("--primary-arm", default="s1c",
                    help="arm whose deltas are reported; default s1c (chromatic)")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    document = analyse(args.runs, args.reference_arm, args.primary_arm)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps({key: dict(mean=round(value["across_seeds"]["mean"], 4),
                                sd=None if value["across_seeds"]["sd"] is None
                                else round(value["across_seeds"]["sd"], 4),
                                stable=value["stable"])
                      for key, value in document["aggregate"].items()}, indent=2))
    print(json.dumps(dict(any_censored=document["any_censored"],
                          runs={name: run["budget_censored"]
                                for name, run in document["runs"].items()}), indent=2))


if __name__ == "__main__":
    main()
