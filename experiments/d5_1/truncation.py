"""D5.1 finite-support forward error at a common sensor coordinate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from lcd_reconstruction.datasets.general_scenes import WAVELENGTHS
from lcd_reconstruction.operators.measured_psf_source import (
    block_integrate, crop_center, load_corrected_psf, normalize_sum,
)
from lcd_reconstruction.unrolling.spatial_pgd import FFTMeasuredOperator

CONFIGS = [
    dict(id="baseline", support=512, integration=8),
    dict(id="S1", support=256, integration=4),
    dict(id="S2", support=192, integration=2),
    dict(id="S3", support=256, integration=2),
    dict(id="S4", support=192, integration=1),
    dict(id="S5", support=256, integration=1),
]
SCENES = ("beads", "cloth", "feathers")


def rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def high_frequency_rel(a, b):
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    mask = np.hypot(fy, fx) >= .25
    return rel(fa[mask], fb[mask])


def border_center_rel(a, b):
    error = a - b
    h, w = error.shape
    border = np.zeros_like(error, bool)
    border[:32] = border[-32:] = True
    border[:, :32] = border[:, -32:] = True
    center = ~border
    return float(np.linalg.norm(error[border]) / max(np.linalg.norm(b[border]), 1e-15)), \
        float(np.linalg.norm(error[center]) / max(np.linalg.norm(b[center]), 1e-15))


def expand_to_sensor(kernel, factor):
    if factor == 1:
        return np.asarray(kernel, dtype=np.float32)
    return np.repeat(np.repeat(np.asarray(kernel, dtype=np.float32) / (factor ** 2),
                               factor, axis=0), factor, axis=1)


def measure(kernels, scene):
    op = FFTMeasuredOperator(torch.tensor(kernels, dtype=torch.float32), scene.shape[-2:]).cuda()
    with torch.no_grad():
        out = op(torch.tensor(scene[None], dtype=torch.float32, device="cuda"))
    del op
    torch.cuda.empty_cache()
    return out[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--cave", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    corrected = {int(nm): load_corrected_psf(args.psf, int(nm))[0] for nm in WAVELENGTHS}
    references = {
        "shape_only": np.asarray([normalize_sum(corrected[int(nm)]) for nm in WAVELENGTHS], np.float32),
        "amplitude_preserving": np.asarray([corrected[int(nm)] for nm in WAVELENGTHS], np.float32),
    }
    scenes = {}
    for name in SCENES:
        with np.load(args.cave / f"{name}.npz") as z:
            cube = z["cube"].astype(np.float32) / 65535.0
        scenes[name] = cube
    reference_measurements = {
        mode: {name: measure(kernels, cube) for name, cube in scenes.items()}
        for mode, kernels in references.items()
    }
    result = dict(protocol="D5.1-forward-truncation",
                  reference={"shape_only": "1024-support signed 1x kernel, unit sum",
                             "amplitude_preserving": "1024-support signed raw-amplitude kernel, no normalization"},
                  scene_size=[512, 512],
                  scenes=list(SCENES),
                  scene_note="training scenes reused for numerical forward audit; no frozen test",
                  configs=[],
                  limitations=["a finite kernel is expanded by block replication / f^2 to the common sensor grid",
                               "the reference is not an independent physical measurement; it is a numerical 1024-support operator"])
    for config in CONFIGS:
        support, factor = config["support"], config["integration"]
        amp = np.asarray([block_integrate(crop_center(corrected[int(nm)], support), factor, "sum")
                          for nm in WAVELENGTHS], np.float32)
        shape = np.asarray([normalize_sum(k) for k in amp], np.float32)
        variants = {}
        for mode, kernels in (("shape_only", shape), ("amplitude_preserving", amp)):
            expanded = np.asarray([expand_to_sensor(k, factor) for k in kernels], np.float32)
            per_scene = {}
            per_band = []
            for name, cube in scenes.items():
                candidate = measure(expanded, cube)
                reference_measure = reference_measurements[mode][name]
                border, center = border_center_rel(candidate, reference_measure)
                per_scene[name] = dict(
                    relative_l2=rel(candidate, reference_measure),
                    high_frequency_relative_l2=high_frequency_rel(candidate, reference_measure),
                    border_relative_l2=border,
                    center_relative_l2=center,
                    candidate_rms=float(np.sqrt((candidate ** 2).mean())),
                    reference_rms=float(np.sqrt((reference_measure ** 2).mean())),
                )
                band_errors = []
                for b, nm in enumerate(WAVELENGTHS):
                    ref_b = measure(references[mode][b:b + 1], cube[b:b + 1])
                    cand_b = measure(expanded[b:b + 1], cube[b:b + 1])
                    band_errors.append(rel(cand_b, ref_b))
                per_band.append(dict(scene=name, per_band_relative_l2=band_errors,
                                     wavelength_spread=float(np.std(band_errors) /
                                                             max(np.mean(band_errors), 1e-15))))
            variants[mode] = dict(
                kernel_hw=list(expanded.shape[-2:]),
                kernel_sum=[float(k.sum()) for k in expanded],
                scenes=per_scene,
                per_band=per_band,
                mean_relative_l2=float(np.mean([v["relative_l2"] for v in per_scene.values()])),
                mean_high_frequency_relative_l2=float(np.mean(
                    [v["high_frequency_relative_l2"] for v in per_scene.values()])),
            )
        result["configs"].append(dict(**config, variants=variants))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([{k: c[k] for k in ("id", "support", "integration")} | {
        m: round(c["variants"][m]["mean_relative_l2"], 5) for m in ("shape_only", "amplitude_preserving")}
        for c in result["configs"]]))


if __name__ == "__main__":
    main()


