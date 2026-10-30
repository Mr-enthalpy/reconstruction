"""D5.3 OTF controls for the three kernel conditions.

Compares the measured chromatic kernels with their arithmetic mean and with the
550 nm kernel replicated over all bands. This answers one narrow question: is
the arithmetic-mean kernel an unusually weak or unusually strong blur, i.e. a
frequency-response confound for the chromatic/achromatic comparison?

It cannot answer whether dispersion carries spectral information: every
single-kernel-per-band operator has zero response to a zero-sum spectral change
(see ``encoding_observability.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lcd_reconstruction.operators.psf_provenance import array_digest

BIN_EDGES = np.arange(0.0, 0.5001, 0.05)


def load(path: Path):
    return np.load(path)["kernels"].astype(np.float32)


def fft_size(hw):
    return 1 << (max(hw) - 1).bit_length()


def otf_magnitude(kernels, size):
    return np.abs(np.fft.fft2(kernels, s=(size, size)))


def radial_profile(magnitude, size):
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.hypot(fy, fx)
    profile = {}
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (radius >= low) & (radius < high)
        profile[f"{low:.2f}-{high:.2f}"] = float(magnitude[..., mask].mean()) if mask.any() else 0.0
    return profile


def band_statistics(magnitude, size):
    """Per-band spread of the OTF at each radial bin."""
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.hypot(fy, fx)
    spread = {}
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (radius >= low) & (radius < high)
        if not mask.any():
            spread[f"{low:.2f}-{high:.2f}"] = None
            continue
        values = magnitude[..., mask].mean(axis=1)
        spread[f"{low:.2f}-{high:.2f}"] = dict(
            band_min=float(values.min()), band_max=float(values.max()),
            band_std=float(values.std()), band_range=float(values.max() - values.min()))
    return spread


def describe(name, kernels, size):
    magnitude = otf_magnitude(kernels, size)
    wide = np.asarray(kernels, dtype=np.float64)
    return dict(
        name=name,
        array_sha256=array_digest(kernels),
        bands=int(kernels.shape[0]),
        kernel_shape=list(kernels.shape[1:]),
        band_sums=[float(v) for v in wide.sum(axis=(1, 2))],
        dc_gain=float(magnitude[:, 0, 0].mean()),
        radial_profile=radial_profile(magnitude, size),
        band_spread=band_statistics(magnitude, size),
        zero_sum_sensitivity=float(np.abs(wide - wide.mean(0, keepdims=True))
                                   .sum(axis=(1, 2)).max()))


def summarise(conditions, threshold=0.25):
    key = min(conditions["chromatic"]["radial_profile"],
              key=lambda k: abs(float(k.split("-")[0]) - threshold))
    profile = {name: condition["radial_profile"][key] for name, condition in conditions.items()}
    spread = {name: condition["band_spread"][key] for name, condition in conditions.items()}
    return dict(
        reference_bin=key,
        mean_otf_per_condition=profile,
        band_spread_per_condition=spread,
        ratio_mean_to_chromatic=float(profile["achromatic"] /
                                      max(profile["chromatic"], 1e-30)),
        ratio_550_to_chromatic=float(profile["achromatic550"] /
                                     max(profile["chromatic"], 1e-30)),
        note="the chromatic entry is the per-band average |OTF| at this bin, not the "
             "OTF of the mean kernel; the ratios compare achromatic blur strength "
             "with the average chromatic blur strength")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromatic-kernels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--band-nm", type=int, default=550)
    ap.add_argument("--hw", type=int, default=512)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    chromatic = load(args.chromatic_kernels)
    mean = np.repeat(chromatic.mean(0, keepdims=True), chromatic.shape[0], axis=0)
    band = (args.band_nm - 450) // 10
    if not 0 <= band < chromatic.shape[0]:
        raise ValueError("band index out of range")
    single = np.repeat(chromatic[band][None], chromatic.shape[0], axis=0)
    size = fft_size((args.hw, args.hw))
    conditions = {
        "chromatic": describe("measured chromatic", chromatic, size),
        "achromatic": describe("mean kernel replicated", mean, size),
        "achromatic550": describe(f"{args.band_nm} nm replicated", single, size),
    }
    document = dict(
        protocol="D5.3-otf-controls",
        fft_size=size,
        reference_hw=args.hw,
        conditions=conditions,
        summary=summarise(conditions),
        spectral_map_rank=1,
        rank_note="at each spatial frequency the forward map from 21 band coefficients to "
                  "the single measurement is rank 1; dispersion changes the band weights as "
                  "a function of spatial frequency, it does not create 21 sensors",
        limits=["signed shape-only PSF approximation, no radiometric/Bayer/field calibration",
                "no claim about real hardware resolution"],
        notes=["achromatic and achromatic550 have exactly zero zero-sum sensitivity by "
               "construction; that is arithmetic, not an empirical result",
               "this analysis cannot show that dispersion adds spectral information"])
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
