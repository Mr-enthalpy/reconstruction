"""D5.1 same-wavelength repeatability, Bayer phase and 1x/2x/4x checks."""
import argparse
import json
from pathlib import Path
import numpy as np

from lcd_reconstruction.operators.measured_psf_source import block_integrate, normalize_sum


def rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.sqrt((np.linalg.norm(a) ** 2 + np.linalg.norm(b) ** 2) / 2))


def high_rel(a, b):
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    mask = np.hypot(fy, fx) >= .25
    return rel(fa[mask], fb[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.psf
    dark = json.loads((root / "received/R21DarkPre01/verified_dark.json").read_text(encoding="utf-8"))
    settings = {w: q for q in dark["settings"] for w in q["matching_wavelength_labels_nm"]}
    runs = {}
    for run in ("R21RefPre32", "R21RefRepeat32", "R21RefPost32",
                "R21RefPre650", "R21RefRepeat650", "R21RefPost650"):
        for path in sorted((root / "received" / run / "analysis").glob("roi_*.npz")):
            with np.load(path) as z:
                meta = json.loads(str(z["metadata_json"]))
                nm = int(meta["wavelength_nm"])
                mean = z["psf"].astype(float)
                setting = settings[nm]
                dark_mean = np.load(root / "received/R21DarkPre01" / setting["mean_file"])
                mean = mean - dark_mean[810:1322, 1122:1634]
                edge = np.concatenate([mean[:40].ravel(), mean[-40:].ravel(),
                                       mean[40:-40, :40].ravel(), mean[40:-40, -40:].ravel()]).mean()
                mean -= edge
                runs[(run, nm)] = mean

    result = dict(
        protocol="D5.1-reference-repeatability",
        source="matched pre-dark plus residual 40px border; raw reference ROI",
        reference_pairs={
            450: ["R21RefPre32", "R21RefRepeat32", "R21RefPost32"],
            550: ["R21RefPre32", "R21RefRepeat32", "R21RefPost32"],
            650: ["R21RefPre650", "R21RefRepeat650", "R21RefPost650"],
        },
        limitations=["only 450/550/650 have independent repeats",
                     "post references are not a third independent pre/repeat pair",
                     "reference ROI is 512x512; 1024 native support is not available for these repeats"],
    )
    same, cross = {}, {}
    for factor in (1, 2, 4, 8):
        if 512 % factor:
            continue
        kernels = {}
        for nm, names in result["reference_pairs"].items():
            kernels[nm] = {}
            for name in names:
                kernels[nm][name] = normalize_sum(
                    block_integrate(runs[(name, nm)], factor, "sum"))
        same[str(factor)] = {}
        for nm, names in result["reference_pairs"].items():
            a, b = kernels[nm][names[0]], kernels[nm][names[1]]
            c = kernels[nm][names[2]]
            same[str(factor)][str(nm)] = dict(
                pre_repeat_relative_l2=rel(a, b),
                pre_post_relative_l2=rel(a, c),
                pre_repeat_high_frequency_relative_l2=high_rel(a, b),
                pre_post_high_frequency_relative_l2=high_rel(a, c),
            )
        cross[str(factor)] = {}
        for a, b in ((450, 550), (450, 650), (550, 650)):
            cross[str(factor)][f"{a}_{b}"] = rel(
                kernels[a][result["reference_pairs"][a][0]],
                kernels[b][result["reference_pairs"][b][0]])
    phase = {}
    for nm in (450, 550, 650):
        k = runs[("R21RefPre32" if nm != 650 else "R21RefPre650", nm)]
        p = [k[0::2, 0::2], k[0::2, 1::2], k[1::2, 0::2], k[1::2, 1::2]]
        block2_raw = block_integrate(k[:512, :512], 2, "sum")
        block2 = block2_raw / block2_raw.sum()
        p_norm = [v / v.sum() for v in p]
        phase[str(nm)] = dict(
            phase_means=[float(v.mean()) for v in p],
            phase_std_over_mean=[float(v.std() / max(abs(v.mean()), 1e-12)) for v in p],
            phase_relative_l2_to_2x_mean=[rel(v, block2) for v in p_norm],
            two_x_phase_identity_error=rel(block2_raw, sum(p)),
            phase_pairwise_correlation=[[float(np.corrcoef(p[i].ravel(), p[j].ravel())[0, 1])
                                         for j in range(4)] for i in range(4)],
        )
    shift = {}
    for nm in (450, 550, 650):
        k = runs[("R21RefPre32" if nm != 650 else "R21RefPre650", nm)]
        a = normalize_sum(block_integrate(k[:510, :510], 2, "sum"))
        b = normalize_sum(block_integrate(k[1:511, 1:511], 2, "sum"))
        shift[str(nm)] = rel(a, b)
    metrics = json.loads((root / "reference_review_final/reference_metrics.json").read_text(encoding="utf-8"))
    result.update(same_wavelength=same, cross_wavelength=cross,
                  bayer_native_phase=phase, bayer_2x_origin_shift=shift,
                  reference_metrics=metrics["entries"],
                  gate=dict(
                      native_1x_phase_fails_if="phase_relative_l2_to_2x_mean exceeds same-wavelength repeat envelope",
                      two_x_origin_shift_fails_if="relative_l2 exceeds the same-wavelength repeat envelope",
                  ))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(
        same_2x={k: round(v["pre_repeat_relative_l2"], 5) for k, v in same["2"].items()},
        cross_2x={k: round(v, 5) for k, v in cross["2"].items()},
        phase_2x={k: round(float(np.mean(v["phase_relative_l2_to_2x_mean"])), 5)
                  for k, v in phase.items()},
        shift_2x={k: round(v, 5) for k, v in shift.items()})))


if __name__ == "__main__":
    main()





