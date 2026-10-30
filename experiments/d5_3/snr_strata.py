"""Stratified SNR summary of the D5.3 observability experiment.

``encoding_observability.json`` keeps 75 configuration records (5 band pairs x 5
masks x 3 strengths, nine scenes each = 675 cases). Reporting a single min-max
over all of them merges three different things: the perturbation strength, the
spectral distance between the perturbed bands, and the spatial mask. Those have
different physical meaning, so this script reports them separately.

Every range here is a range of **configuration means over the nine validation
scenes**, not of individual cases. The per-case records were left in the
acquisition workspace; the committed JSON carries only these aggregates, so a
per-case extreme cannot be quoted from it and is not invented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STRENGTHS = (0.25, 0.5, 1.0)
SEPARATION_BUCKETS = ((10, 10, "adjacent (10 nm)"),
                      (20, 40, "near (20-40 nm)"),
                      (50, 10_000, "distant (>= 50 nm)"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def separation(record):
    low, high = record["bands_nm"]
    return abs(high - low)


def bucket_of(record):
    distance = separation(record)
    for low, high, name in SEPARATION_BUCKETS:
        if low <= distance <= high:
            return name
    raise ValueError(f"unexpected band separation {distance}")


def block(records):
    """Summary of one stratum, over configuration means."""
    if not records:
        raise ValueError("empty stratum: every strength must have all separation buckets")
    pixel = [r["mean_energy_snr"] for r in records]
    matched = [r["mean_matched_filter_snr"] for r in records]
    fraction = [r["mean_detectable_fraction"] for r in records]
    return dict(
        configurations=len(records),
        cases=sum(int(r["cases"]) for r in records),
        per_pixel_snr=dict(min=min(pixel), max=max(pixel)),
        matched_filter_snr=dict(min=min(matched), max=max(matched)),
        detectable_fraction=dict(min=min(fraction), max=max(fraction)))


def analyse(document, source_path=None, source_sha256=None):
    records = document["summary"]["per_configuration"]
    if not records:
        raise ValueError("no per_configuration records")
    strengths = sorted({float(r["strength"]) for r in records})
    masks = sorted({r["mask"] for r in records})
    pairs = sorted({tuple(r["bands_nm"]) for r in records})

    by_strength_separation = {}
    for strength in strengths:
        buckets = {}
        for _, _, name in SEPARATION_BUCKETS:
            subset = [r for r in records
                      if float(r["strength"]) == strength and bucket_of(r) == name]
            # A missing bucket is null, never a range computed from no
            # configurations; the counts above say how many records there were.
            buckets[name] = block(subset) if subset else None
        by_strength_separation[f"{strength:g}"] = buckets

    merged_by_separation = {}
    for _, _, name in SEPARATION_BUCKETS:
        subset = [r for r in records if bucket_of(r) == name]
        merged_by_separation[name] = block(subset) if subset else None

    strata = dict(
        overall=block(records),
        by_strength={f"{s:g}": block([r for r in records if float(r["strength"]) == s])
                     for s in strengths},
        by_mask={m: block([r for r in records if r["mask"] == m]) for m in masks},
        by_band_pair={"-".join(str(b) for b in pair): block(
            [r for r in records if tuple(r["bands_nm"]) == pair]) for pair in pairs},
        by_separation=merged_by_separation,
        by_strength_and_separation=by_strength_separation)

    noise = document.get("noise") or {}
    return dict(
        protocol="D5.3-snr-strata",
        source=dict(path=source_path, sha256=source_sha256),
        unit="ranges are of configuration means over nine validation scenes, "
             "not of individual cases",
        counts=dict(configurations=len(records),
                    cases=sum(int(r["cases"]) for r in records),
                    band_pairs=len({tuple(r["bands_nm"]) for r in records}),
                    masks=len(masks), strengths=len(strengths)),
        noise=noise,
        strata=strata,
        notes=["per-case records were not retained in the committed JSON, so no "
               "per-case extreme is quoted",
               "the matched-filter SNR assumes the exact 21-band difference pattern is "
               "known; it is a detectability ceiling, not an achievable blind recovery",
               "a single-kernel-per-band operator is identically blind to these changes, "
               "so every number here is a chromatic-minus-achromatic operator difference"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="encoding_observability.json produced by encoding_observability.py")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    document = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyse(document, args.input.as_posix(), sha256_file(args.input))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result["strata"]["by_strength"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
