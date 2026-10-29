"""Reduce the large D5.3 result JSONs to report-sized artifacts.

The per-case files carry one record per scene x band pair x mask x strength
(hundreds of entries, megabytes). The report needs the definitions, the
verification totals and the per-configuration aggregates; the raw files stay in
the workspace. Nothing is recomputed or re-selected here, only aggregated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reduce_protocol(document):
    cases = document["cases"]
    grouped = {}
    for case in cases:
        key = (tuple(case["bands_nm"]), case["mask"], case["strength"])
        grouped.setdefault(key, []).append(case)
    per_configuration = []
    for (bands, mask, strength), group in sorted(grouped.items()):
        stats = [case["statistics"] for case in group]
        per_configuration.append(dict(
            bands_nm=list(bands), mask=mask, strength=strength, cases=len(group),
            mean_q_fraction_of_wideband=float(np.mean([s["q_fraction_of_wideband"] for s in stats])),
            mean_touched_fraction=float(np.mean([s["touched_fraction"] for s in stats])),
            mean_q_max=float(np.mean([s["q_max"] for s in stats])),
            max_wideband_relative_residual=float(max(
                s["verification"]["wideband_relative_residual"] for s in stats)),
            min_value=float(min(s["verification"]["nonnegative_minimum"] for s in stats)),
            mean_delta_l2=float(np.mean([s["verification"]["delta_l2"] for s in stats])),
        ))
    return dict(
        protocol=document["protocol"], definition=document["definition"],
        bands_nm=document["bands_nm"], seed=document["seed"],
        strengths=document["strengths"], masks=document["masks"],
        pairs_nm=document["pairs_nm"], scenes=document["scenes"],
        identity=document["identity"], verification=document["verification"],
        per_configuration=per_configuration,
        case_count=len(cases),
        note="per-case records stay in the run workspace; this file is the aggregated protocol")


def reduce_observability(document):
    return {key: document[key] for key in
            ("protocol", "operator", "identity", "rank_note", "kernels", "noise",
             "summary", "notes")} | {"case_count": len(document["cases"]),
                                     "note": "per-case difference records stay in the workspace"}


def merge_responses(documents):
    arms = []
    summary = dict(by_arm={}, by_seed={}, by_mask={})
    for document in documents:
        arms.extend(document["arms"])
        for section in ("by_arm", "by_seed", "by_mask"):
            summary[section].update(document["summary"][section])
    return dict(
        protocol=documents[0]["protocol"],
        statement=documents[0]["statement"],
        arms=arms,
        summary=summary,
        prior_only=documents[0]["prior_only"],
        indistinguishability=documents[0]["indistinguishability"],
        notes=documents[0]["notes"] + ["per-case records stay in the run workspace"],
        case_count=sum(len(document["cases"]) for document in documents))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["protocol", "observability", "response"], required=True)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if args.kind == "protocol":
        reduced = reduce_protocol(documents[0])
    elif args.kind == "observability":
        reduced = reduce_observability(documents[0])
    else:
        reduced = merge_responses(documents)
    args.output.write_text(json.dumps(reduced, indent=2), encoding="utf-8")
    print(json.dumps(dict(kind=args.kind, output=args.output.name,
                          bytes=args.output.stat().st_size)))


if __name__ == "__main__":
    main()
