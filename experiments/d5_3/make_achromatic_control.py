"""Write the achromatic control kernels used by D5.3.

Two constructions are supported and kept explicit so a report never has to
guess which one produced a file:

``mean``  the arithmetic mean of the 21 measured kernels, replicated;
``band``  one measured band (default 550 nm) replicated.

Both are shape-only controls: they keep the same zero-sum convention as the
measured set and change only the wavelength dependence.

The script deliberately depends only on numpy so it can run without the package
on ``PYTHONPATH``; the array digest is the same canonical digest used by
``lcd_reconstruction.operators.psf_provenance`` (shape + float64 C-order bytes),
and a unit test pins that equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BANDS = tuple(range(450, 660, 10))


def array_digest(array) -> str:
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    h = hashlib.sha256()
    h.update(repr(tuple(a.shape)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def build(kernels, kind="band", band_nm=550):
    kernels = np.asarray(kernels, dtype=np.float32)
    if kernels.ndim != 3 or kernels.shape[0] != len(BANDS):
        raise ValueError(f"expected [{len(BANDS)},kh,kw], got {kernels.shape}")
    if kind == "mean":
        control = np.repeat(kernels.mean(0, keepdims=True), kernels.shape[0], axis=0)
        description = "arithmetic mean of the 21 measured kernels, replicated 21 times"
    elif kind == "band":
        if band_nm not in BANDS:
            raise ValueError(f"{band_nm} nm is not a D5.3 band")
        control = np.repeat(kernels[BANDS.index(band_nm)][None], kernels.shape[0], axis=0)
        description = f"{band_nm} nm measured kernel, replicated 21 times"
    else:
        raise ValueError(f"unknown kind {kind}")
    return control.astype(np.float32), description


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--kind", choices=["mean", "band"], default="band")
    ap.add_argument("--band-nm", type=int, default=550)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = np.load(args.kernel_file)["kernels"].astype(np.float32)
    control, description = build(source, args.kind, args.band_nm)
    np.savez_compressed(args.output, kernels=control)
    report = dict(
        protocol="D5.3-achromatic-control-kernel",
        source=str(args.kernel_file),
        source_array_sha256=array_digest(source),
        kind=args.kind,
        description=description,
        output=args.output.name,
        output_array_sha256=array_digest(control),
        output_file_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        shape=list(control.shape),
        band_sums=dict(minimum=float(control.sum(axis=(1, 2)).min()),
                       maximum=float(control.sum(axis=(1, 2)).max())),
        note="a single-kernel-per-band operator has exactly zero response to a "
             "zero-sum spectral change; this file is a frequency-response control, "
             "not an information control")
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
