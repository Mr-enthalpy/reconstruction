"""Per-band relative-L2 across the D5.3 seeds, paired within each seed.

The band axis is a separate dimension from the eight scalar metrics in
``multiseed_comparison.json``: ``band_relative_l2`` is a mean of per-scene ratios
``||error_b|| / ||truth_b||``, so it weights every band equally regardless of that
band's energy and it is not comparable to ``cube_relative_l2``.

Statistical unit is the seed, not the scene. Nine scenes inside one seed are used
only for the direction-consistency clause, exactly as in ``summarize_seeds.py``.

A band is called ``stable`` only when all three pre-registered clauses hold:

1. every seed has the same sign (the chromatic arm is lower);
2. ``|delta_mean| > sd`` of the three seed-level deltas (ddof = 1);
3. every seed has at least eight of nine scenes moving in that direction.

The scalar reduction also used ``max(cross-seed sd, plateau epoch-to-epoch std)``.
That plateau term is not available per band: at acquisition time ``evaluate_metrics``
recorded only the eight scalar metrics per cadence epoch, so no per-band
epoch-to-epoch series exists in ``history.json``. This script therefore compares
against the cross-seed sd alone and reports ``plateau_available: false`` rather than
silently substituting something else.

Nothing here re-selects a checkpoint, opens a data file, or touches a kernel: it
reads the ``validation.json`` documents it is given.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import band_report  # noqa: E402

ARM_PATTERN = re.compile(r"^(?P<arm>[A-Za-z][A-Za-z0-9]*)_s(?P<seed>\d+)$")
METRICS = band_report.METRICS


def discover(root: Path, expected_bands: int):
    runs = {}
    for directory in sorted(Path(root).iterdir()):
        if not directory.is_dir() or not (directory / "validation.json").exists():
            continue
        match = ARM_PATTERN.match(directory.name)
        if not match:
            continue
        loaded = band_report.load_scenes(directory / "validation.json", expected_bands)
        runs[directory.name] = dict(arm=match["arm"], seed=int(match["seed"]),
                                    scenes=loaded["scenes"], provenance=loaded["provenance"])
    if not runs:
        raise ValueError(f"no <arm>_s<seed>/validation.json found under {root}")
    return runs


def require_same_scenes(named_scenes):
    names = sorted(named_scenes)
    reference = names[0]
    base = set(named_scenes[reference])
    for name in names[1:]:
        current = set(named_scenes[name])
        if current != base:
            raise ValueError(f"scene sets differ between {reference} and {name}; "
                             f"only in {reference}: {sorted(base - current)}, "
                             f"only in {name}: {sorted(current - base)}. "
                             "Refusing to silently intersect.")
    return sorted(base)


def band_statistics(chromatic, achromatic, scenes, expected_bands):
    """Paired per-band deltas for one seed: arrays shaped (scenes, bands)."""
    out = {}
    for metric in METRICS:
        a = np.stack([chromatic[s]["metrics"][metric] for s in scenes])
        b = np.stack([achromatic[s]["metrics"][metric] for s in scenes])
        if a.shape != (len(scenes), expected_bands) or b.shape != a.shape:
            raise ValueError(f"{metric}: unexpected shape {a.shape} / {b.shape}")
        out[metric] = dict(chromatic_scene_mean=a.mean(axis=0).tolist(),
                           achromatic_scene_mean=b.mean(axis=0).tolist(),
                           delta=a - b)
    return out


def summarise(runs, chromatic_arm, achromatic_arm, labels, multiseed=None):
    expected_bands = len(labels)
    by_arm = {}
    for name, run in runs.items():
        by_arm.setdefault(run["arm"], {})[run["seed"]] = (name, run)
    for arm in (chromatic_arm, achromatic_arm):
        if arm not in by_arm:
            raise ValueError(f"no runs for arm {arm!r}; found {sorted(by_arm)}")
    seeds = sorted(set(by_arm[chromatic_arm]) & set(by_arm[achromatic_arm]))
    if not seeds:
        raise ValueError("the two arms share no seed")
    only_chromatic = sorted(set(by_arm[chromatic_arm]) - set(by_arm[achromatic_arm]))
    only_achromatic = sorted(set(by_arm[achromatic_arm]) - set(by_arm[chromatic_arm]))

    scene_ids = require_same_scenes({name: run["scenes"] for name, run in runs.items()})

    censored = {}
    if multiseed:
        document = json.loads(Path(multiseed).read_text(encoding="utf-8"))
        for name, run in runs.items():
            record = (document.get("runs") or {}).get(name)
            if record:
                censored[name] = record.get("budget_censored")

    pairs = []
    for seed in seeds:
        chrom_name, chrom = by_arm[chromatic_arm][seed]
        achrom_name, achrom = by_arm[achromatic_arm][seed]
        stats = band_statistics(chrom["scenes"], achrom["scenes"], scene_ids, expected_bands)
        pairs.append(dict(seed=seed, chromatic=chrom_name, achromatic=achrom_name,
                          statistics=stats))

    bands = []
    for index, band in enumerate(labels):
        record = dict(band_nm=band, metrics={})
        for metric in METRICS:
            deltas = np.array([pair["statistics"][metric]["delta"][:, index]
                               for pair in pairs])            # (seeds, scenes)
            chrom_curve = np.array([pair["statistics"][metric]["chromatic_scene_mean"][index]
                                    for pair in pairs])
            achrom_curve = np.array([pair["statistics"][metric]["achromatic_scene_mean"][index]
                                     for pair in pairs])
            seed_mean = deltas.mean(axis=1)
            scene_wins = [int((row < 0).sum()) for row in deltas]
            mean = float(seed_mean.mean())
            sd = float(seed_mean.std(ddof=1)) if len(pairs) > 1 else None
            exceeds = bool(sd is not None and abs(mean) > sd)
            same_sign = bool(all(value < 0 for value in seed_mean)
                             or all(value > 0 for value in seed_mean))
            reasons = []
            if not same_sign:
                reasons.append("seed signs disagree")
            if not exceeds:
                reasons.append("|delta_mean| does not exceed the cross-seed sd")
            if min(scene_wins) < 8:
                reasons.append("a seed has fewer than 8/9 scenes in that direction")
            record["metrics"][metric] = dict(
                chromatic_scene_mean_across_seeds=float(chrom_curve.mean()),
                achromatic_scene_mean_across_seeds=float(achrom_curve.mean()),
                chromatic_scene_mean_sd_across_seeds=(float(chrom_curve.std(ddof=1))
                                                      if len(pairs) > 1 else None),
                achromatic_scene_mean_sd_across_seeds=(float(achrom_curve.std(ddof=1))
                                                       if len(pairs) > 1 else None),
                delta_per_seed=seed_mean.tolist(),
                delta_mean=mean,
                delta_sd_across_seeds=sd,
                scenes_favouring_chromatic=scene_wins,
                scenes_compared=len(scene_ids),
                seeds_with_negative_delta=int((seed_mean < 0).sum()),
                exceeds_cross_seed_sd=exceeds,
                plateau_available=False,
                stable=not reasons,
                unstable_reasons=reasons)
        bands.append(record)

    per_seed_overall = []
    for pair in pairs:
        record = dict(seed=pair["seed"], chromatic=pair["chromatic"],
                      achromatic=pair["achromatic"], metrics={})
        for metric in METRICS:
            delta = pair["statistics"][metric]["delta"]
            seed_mean = delta.mean(axis=0)
            record["metrics"][metric] = dict(
                mean_over_bands=float(seed_mean.mean()),
                scenes_with_negative_mean=int((delta.mean(axis=1) < 0).sum()),
                scenes_compared=int(delta.shape[0]))
        per_seed_overall.append(record)

    summary = {}
    for metric in METRICS:
        records = [band["metrics"][metric] for band in bands]
        summary[metric] = dict(
            bands_chromatic_lower=int(sum(1 for r in records
                                          if r["chromatic_scene_mean_across_seeds"]
                                          < r["achromatic_scene_mean_across_seeds"])),
            bands_with_all_seeds_negative=int(sum(1 for r in records
                                                  if r["seeds_with_negative_delta"] == len(pairs))),
            bands_exceeding_cross_seed_sd=int(sum(1 for r in records
                                                  if r["exceeds_cross_seed_sd"])),
            bands_stable=int(sum(1 for r in records if r["stable"])),
            bands_unstable=[band["band_nm"] for band in bands
                            if not band["metrics"][metric]["stable"]],
            mean_over_bands_across_seeds=float(np.mean([r["delta_mean"] for r in records])),
            across_band_sd=(float(np.std([r["delta_mean"] for r in records], ddof=1))
                            if len(records) > 1 else None),
            largest_improvement_band=int(
                min(bands, key=lambda b: b["metrics"][metric]["delta_mean"])["band_nm"]),
            smallest_improvement_band=int(
                max(bands, key=lambda b: b["metrics"][metric]["delta_mean"])["band_nm"]))

    return dict(
        protocol="D5.3-per-band-relative-l2",
        arms=dict(chromatic=chromatic_arm, achromatic=achromatic_arm),
        seeds=seeds,
        bands_nm=list(labels),
        definitions=dict(
            scene_mean="mean over the nine validation scenes of ||error_b|| / ||truth_b||, "
                       "a mean of per-scene ratios rather than sum||error_b||/sum||truth_b||",
            delta="chromatic minus achromatic, paired by scene_id inside each seed",
            sign="both metrics are errors, so a negative delta means the chromatic arm is better",
            unit="the statistical unit is the seed (n = %d); scenes are direction evidence only"
                 % len(pairs),
            stable="all seeds share the sign, |delta_mean| > cross-seed sd (ddof=1), and every "
                   "seed has at least 8/9 scenes in that direction",
            plateau="per-band plateau std is not available: cadence epochs recorded only the "
                    "eight scalar metrics, so only the cross-seed sd is used",
            comparability="per-band relative L2 weights every band equally; it is not "
                          "comparable to cube_relative_l2 and cannot be summed with it"),
        runs={name: dict(arm=run["arm"], seed=run["seed"],
                         sha256=run["provenance"]["sha256"],
                         best_epoch=run["provenance"]["best_epoch"],
                         budget_censored=censored.get(name))
              for name, run in runs.items()},
        pairs_per_seed=per_seed_overall,
        bands=bands,
        summary=summary,
        notes=["only the validation split is read; no data, kernel or checkpoint file is opened",
               "unpaired seeds are reported, never pooled: only_chromatic=%s only_achromatic=%s"
               % (only_chromatic, only_achromatic),
               "budget_censored is copied from multiseed_comparison.json when given, otherwise null"])


def labels_from_argument(bands: str):
    """Accept a band count (``"21"``) or an explicit nm span (``"450:650:10"``)."""
    if ":" in bands:
        return band_report.band_labels(*(int(v) for v in bands.split(":")))
    count = int(bands)
    if count < 1:
        raise ValueError(f"band count must be positive, got {count}")
    return list(range(band_report.BAND_START,
                      band_report.BAND_START + band_report.BAND_STEP * count,
                      band_report.BAND_STEP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding <arm>_s<seed>/validation.json run directories")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chromatic-arm", default="s1c")
    ap.add_argument("--achromatic-arm", default="s3a")
    ap.add_argument("--multiseed", type=Path,
                    help="optional multiseed_comparison.json, used only for budget_censored")
    ap.add_argument("--bands", default="21", help="number of bands, or start:stop:step in nm")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = labels_from_argument(args.bands)

    runs = discover(args.runs, len(labels))
    document = summarise(runs, args.chromatic_arm, args.achromatic_arm, labels,
                         args.multiseed)
    document["source"] = dict(runs=Path(args.runs).as_posix(),
                              sha256={name: run["provenance"]["sha256"]
                                      for name, run in runs.items()})
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(dict(seeds=document["seeds"], bands=len(labels),
                          summary={k: {kk: vv for kk, vv in v.items()
                                       if kk in ("bands_stable", "bands_chromatic_lower",
                                                 "bands_exceeding_cross_seed_sd",
                                                 "bands_unstable", "mean_over_bands_across_seeds")}
                                   for k, v in document["summary"].items()}),
                     indent=2))


if __name__ == "__main__":
    main()
