"""D5.1 scene-grid versus sensor-grid convolution consistency."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from lcd_reconstruction.datasets.general_scenes import WAVELENGTHS
from lcd_reconstruction.operators.measured_psf_source import (
    block_integrate, load_corrected_psf, normalize_sum,
)
from lcd_reconstruction.unrolling.spatial_pgd import FFTMeasuredOperator

SCENES = ("beads", "cloth", "feathers")


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def high_rel(a, b):
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    mask = np.hypot(fy, fx) >= .25
    return rel(fa[mask], fb[mask])


def measure(kernels, scene):
    op = FFTMeasuredOperator(torch.tensor(kernels, dtype=torch.float32), scene.shape[-2:]).cuda()
    with torch.no_grad():
        out = op(torch.tensor(scene[None], dtype=torch.float32, device="cuda"))
    del op
    torch.cuda.empty_cache()
    return out[0].cpu().numpy()


def downsample(a, factor):
    h, w = a.shape[-2:]
    if h % factor or w % factor:
        raise ValueError("factor must divide scene")
    return a.reshape(*a.shape[:-2], h // factor, factor, w // factor, factor).mean((-3, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--cave", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = np.asarray([
        normalize_sum(load_corrected_psf(args.psf, int(nm))[0]) for nm in WAVELENGTHS
    ], np.float32)
    scenes = {}
    for name in SCENES:
        with np.load(args.cave / f"{name}.npz") as z:
            scenes[name] = z["cube"].astype(np.float32) / 65535.0
    result = dict(protocol="D5.1-scene-scale-consistency",
                  reference="1024 sensor-support kernel, unit sum, 512x512 scene",
                  scenes=list(SCENES),
                  note="high-resolution simulation then measurement downsample versus scene/kernel resize",
                  cases=[])
    for factor in (1, 2, 4):
        case = dict(factor=factor, scenes={})
        for name, cube in scenes.items():
            high = measure(reference, cube)
            high_down = downsample(high, factor)
            small_scene = downsample(cube, factor)
            small_kernel = np.asarray([
                normalize_sum(block_integrate(k, factor, "sum")) for k in reference
            ], np.float32)
            low = measure(small_kernel, small_scene)
            case["scenes"][name] = dict(
                relative_l2=rel(low, high_down),
                high_frequency_relative_l2=high_rel(low, high_down),
                high_res_rms=float(np.sqrt((high_down ** 2).mean())),
                low_res_rms=float(np.sqrt((low ** 2).mean())),
            )
        case["mean_relative_l2"] = float(np.mean([v["relative_l2"] for v in case["scenes"].values()]))
        case["mean_high_frequency_relative_l2"] = float(np.mean(
            [v["high_frequency_relative_l2"] for v in case["scenes"].values()]))
        result["cases"].append(case)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([{k: c[k] for k in ("factor", "mean_relative_l2",
                                        "mean_high_frequency_relative_l2")} for c in result["cases"]]))


if __name__ == "__main__":
    main()

