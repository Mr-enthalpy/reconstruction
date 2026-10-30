"""Per-band relative-L2 report from finished ``validation.json`` files.

``reconstruction_metrics`` already returns ``band_relative_l2`` and ``band_rmse``
for every validation scene (21 values, 450-650 nm in 10 nm steps), but the
aggregate blocks in ``validation.json`` keep only eight scalars. This script
rebuilds the missing aggregate from the per-scene rows, so it works on runs
written before the aggregate ever carried per-band means.

Definitions are fixed here and echoed into the output document:

``scene_mean[b] = mean over scenes of ||error_b|| / ||truth_b||``

That is a mean of per-scene ratios. It is *not* ``sum||error_b|| / sum||truth_b||``
and it is therefore not comparable to ``cube_relative_l2``; it also weights every
band equally regardless of how much energy that band carries.

Both metrics are errors, so a negative delta means the first condition is lower,
i.e. better. Nothing here re-selects a checkpoint, opens a data file or touches a
PSF kernel: it reads the JSON it is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BAND_START, BAND_STOP, BAND_STEP = 450, 650, 10
METRICS = ("band_relative_l2", "band_rmse")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def band_labels(start=BAND_START, stop=BAND_STOP, step=BAND_STEP):
    labels = list(range(start, stop + 1, step))
    if len(labels) != 21:
        raise ValueError(f"{start}-{stop} step {step} nm gives {len(labels)} bands, expected 21")
    return labels


def load_scenes(path: Path, expected_bands: int):
    """Return {scene_id: {dataset_id, metric: array}} from one validation.json."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = document.get("results")
    if not rows:
        raise ValueError(f"{path}: no 'results' rows; this is not a D5 validation.json")
    scenes = {}
    for row in rows:
        scene = row.get("scene_id")
        if not scene:
            raise ValueError(f"{path}: a result row has no scene_id")
        if scene in scenes:
            raise ValueError(f"{path}: duplicate scene_id {scene!r}")
        metrics = row.get("metrics") or {}
        entry = dict(dataset_id=row.get("dataset_id"), metrics={})
        for name in METRICS:
            if name not in metrics:
                raise ValueError(f"{path}: scene {scene!r} is missing {name!r}")
            values = np.asarray(metrics[name], dtype=np.float64).ravel()
            if values.size != expected_bands:
                raise ValueError(f"{path}: scene {scene!r} has {values.size} {name} values, "
                                 f"expected {expected_bands}")
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: scene {scene!r} has non-finite {name}")
            if (values < 0).any():
                raise ValueError(f"{path}: scene {scene!r} has negative {name}")
            entry["metrics"][name] = values
        scenes[scene] = entry
    return dict(scenes=scenes,
                provenance=dict(path=str(path), sha256=sha256_file(path),
                                scenes=len(scenes),
                                best_epoch=document.get("best_epoch"),
                                best_validation_loss=document.get("best_validation_loss"),
                                recorded_selection=((document.get("selection") or {})
                                                    .get("best_epoch"))))


def summarise(scenes, name):
    """Band-by-band scene statistics for one condition, plus CAVE/KAIST splits."""
    ids = sorted(scenes)
    if not ids:
        raise ValueError(f"{name}: no scenes")

    def block(selected):
        if not selected:
            return None
        out = {}
        for metric in METRICS:
            stack = np.stack([scenes[s]["metrics"][metric] for s in selected])
            out[metric] = dict(
                scene_mean=stack.mean(axis=0).tolist(),
                scene_sd=(stack.std(axis=0, ddof=1).tolist() if stack.shape[0] > 1 else None),
                scene_min=stack.min(axis=0).tolist(),
                scene_max=stack.max(axis=0).tolist(),
                scenes=int(stack.shape[0]))
        return out

    datasets = sorted({scenes[s]["dataset_id"] for s in ids if scenes[s]["dataset_id"]})
    return dict(name=name, scenes=ids, by_dataset={d: block([s for s in ids
                                                             if scenes[s]["dataset_id"] == d])
                                                   for d in datasets},
                ALL=block(ids))


def paired(condition, reference):
    """Elementwise ``condition - reference`` per band, paired by scene_id."""
    if condition["scenes"] != reference["scenes"]:
        missing = sorted(set(reference["scenes"]) - set(condition["scenes"]))
        extra = sorted(set(condition["scenes"]) - set(reference["scenes"]))
        raise ValueError(f"scene sets differ for {condition['name']} vs {reference['name']}; "
                         f"only in reference {missing}, only in condition {extra}. "
                         "Refusing to silently intersect.")
    ids = reference["scenes"]
    out = dict(condition=condition["name"], reference=reference["name"], scenes=len(ids),
               delta=dict())
    for metric in METRICS:
        a = np.stack([condition["_scenes"][s]["metrics"][metric] for s in ids])
        b = np.stack([reference["_scenes"][s]["metrics"][metric] for s in ids])
        delta = a - b
        mean = delta.mean(axis=0)
        out["delta"][metric] = dict(
            delta_mean=mean.tolist(),
            delta_sd=(delta.std(axis=0, ddof=1).tolist() if delta.shape[0] > 1 else None),
            delta_min=delta.min(axis=0).tolist(),
            delta_max=delta.max(axis=0).tolist(),
            scenes_delta_negative=[int((delta[:, b] < 0).sum()) for b in range(delta.shape[1])],
            scenes_delta_positive=[int((delta[:, b] > 0).sum()) for b in range(delta.shape[1])],
            exceeds_scene_sd=(None if delta.shape[0] < 2 else
                              [bool(abs(mean[b]) > delta[:, b].std(ddof=1))
                               for b in range(delta.shape[1])]),
            full_agreement=[bool(len({bool(v < 0) for v in delta[:, b]}) == 1)
                            for b in range(delta.shape[1])],
            mean_over_bands=float(mean.mean()),
            sd_of_scene_mean=(None if delta.shape[0] < 2 else
                              float(delta.mean(axis=1).std(ddof=1))),
            scenes_with_negative_scene_mean=int((delta.mean(axis=1) < 0).sum()))
    return out


def load_runs(root: Path):
    conditions = []
    for directory in sorted(Path(root).iterdir()):
        candidate = directory / "validation.json"
        if directory.is_dir() and candidate.exists():
            conditions.append((directory.name, candidate))
    if not conditions:
        raise ValueError(f"no <run>/validation.json found under {root}")
    return conditions


def parse_condition(item: str):
    if "=" not in item:
        raise ValueError(f"--condition expects name=path, got {item!r}")
    name, path = item.split("=", 1)
    if not name or not path:
        raise ValueError(f"--condition expects name=path, got {item!r}")
    return name, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--reference", help="condition name used as the baseline for deltas")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bands", default=f"{BAND_START}:{BAND_STOP}:{BAND_STEP}")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start, stop, step = (int(v) for v in args.bands.split(":"))
    labels = band_labels(start, stop, step)

    pairs = [parse_condition(item) for item in args.condition]
    if args.runs:
        pairs.extend(load_runs(args.runs))
    if not pairs:
        raise SystemExit("give at least one --condition NAME=PATH or --runs DIRECTORY")
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate condition names: {names}")

    loaded = {}
    for name, path in pairs:
        loaded[name] = load_scenes(path, len(labels))

    reference = args.reference or names[0]
    if reference not in loaded:
        raise ValueError(f"--reference {reference!r} is not one of {names}")

    summaries = []
    for name, _ in pairs:
        summary = summarise(loaded[name]["scenes"], name)
        summary["provenance"] = loaded[name]["provenance"]
        summaries.append(summary)

    deltas = []
    if len(names) > 1:
        base = dict(name=reference, scenes=sorted(loaded[reference]["scenes"]),
                    _scenes=loaded[reference]["scenes"])
        for name, _ in pairs:
            if name == reference:
                continue
            current = dict(name=name, scenes=sorted(loaded[name]["scenes"]),
                           _scenes=loaded[name]["scenes"])
            deltas.append(paired(current, base))

    document = dict(
        protocol="D5-per-band-relative-l2",
        bands_nm=labels,
        definitions=dict(
            scene_mean="mean over validation scenes of ||error_b|| / ||truth_b||; "
                       "a mean of per-scene ratios, not sum||error_b||/sum||truth_b||",
            scene_sd="standard deviation of the per-scene ratios, ddof=1",
            delta="condition - reference, band by band, paired by scene_id",
            sign="both metrics are errors, so a negative delta means the condition is better",
            comparability="per-band relative L2 weights every band equally regardless of band "
                          "energy, so it is not comparable to cube_relative_l2 and cannot be "
                          "summed or averaged with it",
            source="computed from validation.json results[]; aggregate blocks are not used"),
        conditions=summaries,
        deltas=deltas,
        notes=["validation split only; this script never opens a data, kernel or checkpoint file",
               "no checkpoint is re-selected here"])
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(dict(bands=len(labels), conditions=names, reference=reference,
                          deltas=[d["condition"] for d in deltas]), indent=2))


if __name__ == "__main__":
    main()
