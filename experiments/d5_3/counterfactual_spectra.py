"""D5.3 fixed-wideband-integral counterfactual scenes.

Two cubes are built so that every pixel has the *same* 21-band sum but a
different spectral distribution:

    X_plus[i]  = X[i] + q        X_plus[j]  = X[j] - q
    X_minus[i] = X[i] - q        X_minus[j] = X[j] + q

with ``0 <= q <= min(X[i], X[j])`` pointwise. The bound keeps both cubes
non-negative and makes the spectral difference exactly ``2q (e_i - e_j)``.

Consequence used by every downstream script: an operator that applies one
kernel to all 21 bands (mean kernel or a single-wavelength kernel) gives the
same measurement for both cubes, because ``sum_b delta_X_b = 0``. A
wavelength-dependent operator does not, and its difference is exactly

    delta_Y(f) = sum_b (K_b(f) - mean_b K_b(f)) * delta_X_b(f).

Nothing here claims these cubes are natural images; they are a constraint-
satisfying construction used to test the operator, not a data distribution.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

BANDS = tuple(range(450, 660, 10))


def band_index(nm: int) -> int:
    nm = int(nm)
    if nm not in BANDS:
        raise ValueError(f"{nm} nm is not one of the {len(BANDS)} D5.3 bands {BANDS}")
    return BANDS.index(nm)


def _normalise(values, percentile: float = 99.0):
    scale = float(np.percentile(values, percentile))
    return np.clip(values / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)


def mask_uniform(hw, **_):
    return np.ones(hw, dtype=np.float32)


def mask_block(hw, rng, fraction: float = 0.25, **_):
    h, w = hw
    sh, sw = max(1, int(round(h * fraction))), max(1, int(round(w * fraction)))
    y = int(rng.integers(0, h - sh + 1))
    x = int(rng.integers(0, w - sw + 1))
    mask = np.zeros(hw, dtype=np.float32)
    mask[y:y + sh, x:x + sw] = 1.0
    return mask


def mask_edge(hw, cube, **_):
    """Smooth mask on wideband edges: keeps q where spatial structure exists."""
    gy, gx = np.gradient(cube.sum(0, dtype=np.float64))
    return _normalise(np.hypot(gy, gx))


def mask_stripes(hw, period: float = 8.0, angle_deg: float = 0.0, **_):
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w]
    theta = np.deg2rad(angle_deg)
    coord = xx * np.cos(theta) + yy * np.sin(theta)
    return (0.5 * (1.0 + np.cos(2 * np.pi * coord / period))).astype(np.float32)


def mask_chroma(hw, cube, i, j, **_):
    """Smooth mask where the two target bands already differ."""
    return _normalise(np.abs(cube[i].astype(np.float64) - cube[j].astype(np.float64)))


MASK_BUILDERS = {
    "uniform": mask_uniform,
    "block": mask_block,
    "edge": mask_edge,
    "stripes": mask_stripes,
    "chroma": mask_chroma,
}
MASK_PARAMS = {
    "uniform": {},
    "block": dict(fraction=0.25),
    "edge": {},
    "stripes": dict(period=8.0),
    "chroma": {},
}


def build_mask(name, cube, i, j, rng, params=None):
    if name not in MASK_BUILDERS:
        raise ValueError(f"unknown mask {name}; choose from {sorted(MASK_BUILDERS)}")
    kwargs = dict(MASK_PARAMS.get(name, {}))
    kwargs.update(params or {})
    mask = MASK_BUILDERS[name](cube.shape[-2:], cube=cube, i=i, j=j, rng=rng, **kwargs)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape != tuple(cube.shape[-2:]) or not np.isfinite(mask).all():
        raise ValueError("mask must be finite and match the spatial shape")
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("mask must lie in [0, 1]")
    return mask


def amplitude(cube, i, j, mask, strength):
    """q = strength * mask * min(X_i, X_j), clipped to the non-negativity bound."""
    if not 0.0 < float(strength) <= 1.0:
        raise ValueError("strength must lie in (0, 1]")
    cap = np.minimum(cube[i], cube[j])
    q = np.clip(float(strength) * np.asarray(mask, np.float32) * cap, 0.0, cap)
    return q.astype(np.float32)


def build_pair(cube, i, j, mask, strength):
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[0] != len(BANDS):
        raise ValueError(f"expected [21,H,W], got {cube.shape}")
    if i == j:
        raise ValueError("i and j must differ")
    q = amplitude(cube, i, j, mask, strength)
    x_plus = cube.copy()
    x_minus = cube.copy()
    x_plus[i] += q
    x_plus[j] -= q
    x_minus[i] -= q
    x_minus[j] += q
    return dict(x_plus=x_plus, x_minus=x_minus, q=q, bands=(int(i), int(j)),
                mask=np.asarray(mask, dtype=np.float32))


def verify_pair(pair, rtol: float = 1e-6):
    """Hard checks: non-negativity and identical per-pixel wideband sums."""
    x_plus, x_minus = pair["x_plus"], pair["x_minus"]
    if x_plus.shape != x_minus.shape:
        raise ValueError("counterfactual cubes must share a shape")
    if x_plus.min() < 0 or x_minus.min() < 0:
        raise ValueError("counterfactual cubes must stay non-negative")
    wide_plus = x_plus.sum(0, dtype=np.float64)
    wide_minus = x_minus.sum(0, dtype=np.float64)
    scale = max(float(np.abs(wide_plus).max()), 1e-12)
    residual = float(np.abs(wide_plus - wide_minus).max())
    if residual > rtol * scale:
        raise ValueError(f"wideband per-pixel sum changed by {residual} (scale {scale})")
    delta = x_plus.astype(np.float64) - x_minus.astype(np.float64)
    return dict(
        wideband_max_abs_residual=residual,
        wideband_relative_residual=residual / scale,
        nonnegative_minimum=float(min(x_plus.min(), x_minus.min())),
        delta_l2=float(np.linalg.norm(delta)),
        delta_l2_per_band=[float(np.linalg.norm(delta[b])) for b in range(delta.shape[0])],
    )


def case_statistics(case):
    q, mask = case["q"], case["mask_array"]
    touched = mask > 0
    return dict(
        q_l1=float(q.sum()),
        q_max=float(q.max()),
        q_mean_over_mask=float(q[touched].mean()) if touched.any() else 0.0,
        q_fraction_of_wideband=float(q.sum() / max(case["wideband_sum"] * q.size, 1e-12)),
        touched_pixels=int(touched.sum()),
        touched_fraction=float(touched.mean()),
        verification=verify_pair(case),
    )


def make_case(cube, scene_id, dataset_id, i, j, mask_name, mask, strength):
    pair = build_pair(cube, i, j, mask, strength)
    case = dict(scene_id=scene_id, dataset_id=dataset_id,
                bands_nm=[int(BANDS[i]), int(BANDS[j])],
                mask=mask_name, mask_params=MASK_PARAMS.get(mask_name, {}),
                strength=float(strength), mask_fraction=float((mask > 0).mean()),
                bands=pair["bands"], mask_array=pair["mask"], q=pair["q"],
                x_plus=pair["x_plus"], x_minus=pair["x_minus"],
                wideband_sum=float(cube.sum(0, dtype=np.float64).mean()))
    return case


def _case_rng(seed, scene_id, i, j, mask_name, strength):
    base = zlib.crc32(str(scene_id).encode("utf-8"))
    names = sorted(MASK_BUILDERS)
    return np.random.default_rng([seed, base, i, j, names.index(mask_name),
                                  int(round(float(strength) * 1000))])


def default_pairs():
    return [(450, 460), (480, 620), (490, 510), (450, 650), (530, 570)]


def default_strengths():
    return [0.25, 0.5, 1.0]


def build_cases(prepared: Path, scenes, pairs, strengths, masks, seed: int,
                max_cases: int | None = None):
    return list(iter_cases(prepared, scenes, pairs, strengths, masks, seed, max_cases))


def iter_cases(prepared: Path, scenes, pairs, strengths, masks, seed: int,
               max_cases: int | None = None):
    """Yield counterfactual cases one at a time; never materialises the split."""
    records = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    selected = [r for r in records if r["split"] == "val"]
    if scenes != ["all"]:
        wanted = set(scenes)
        selected = [r for r in selected if r["scene_id"] in wanted]
    if not selected:
        raise ValueError("no validation scenes selected")
    val = np.load(prepared / "val.npz")["cube"]
    emitted = 0
    for record in selected:
        cube = np.asarray(val[record["array_index"]], dtype=np.float32)
        for i_nm, j_nm in pairs:
            i, j = band_index(i_nm), band_index(j_nm)
            for mask_name in masks:
                for strength in strengths:
                    rng = _case_rng(seed, record["scene_id"], i, j, mask_name, strength)
                    mask = build_mask(mask_name, cube, i, j, rng)
                    yield make_case(cube, record["scene_id"], record["dataset_id"],
                                    i, j, mask_name, mask, strength)
                    emitted += 1
                    if max_cases is not None and emitted >= max_cases:
                        return


def protocol_document(cases, scenes, pairs, strengths, masks, seed):
    entries = []
    for case in cases:
        stats = case_statistics(case)
        entries.append(dict(scene_id=case["scene_id"], dataset_id=case["dataset_id"],
                            bands_nm=case["bands_nm"], mask=case["mask"], strength=case["strength"],
                            mask_params=case["mask_params"], statistics=stats))
    residuals = [e["statistics"]["verification"]["wideband_relative_residual"] for e in entries]
    return dict(
        protocol="D5.3-fixed-wideband-counterfactual",
        definition="X_plus[i]=X[i]+q, X_plus[j]=X[j]-q, X_minus[i]=X[i]-q, "
                   "X_minus[j]=X[j]+q, 0<=q<=min(X[i],X[j])",
        bands_nm=list(BANDS), seed=int(seed), strengths=strengths, masks=masks,
        pairs_nm=[[int(a), int(b)] for a, b in pairs], scenes=scenes,
        identity="sum_b X_plus[b] == sum_b X_minus[b] at every pixel",
        cases=entries,
        verification=dict(cases=len(entries),
                          max_wideband_relative_residual=max(residuals) if residuals else None,
                          all_nonnegative=all(e["statistics"]["verification"]["nonnegative_minimum"] >= 0
                                              for e in entries),
                          all_wideband_sums_equal=all(
                              e["statistics"]["verification"]["wideband_relative_residual"] <= 1e-6
                              for e in entries)),
        notes=["constraint-satisfying construction, not a natural-image distribution",
               "an achromatic operator maps X_plus and X_minus to the same measurement",
               "only the val split is read; test is never loaded"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--pairs", default=";".join(f"{a},{b}" for a, b in default_pairs()))
    ap.add_argument("--strengths", default=",".join(str(s) for s in default_strengths()))
    ap.add_argument("--masks", default="uniform,block,edge,stripes,chroma")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--max-cases", type=int)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = [tuple(int(v) for v in item.split(",")) for item in args.pairs.split(";") if item]
    strengths = [float(v) for v in args.strengths.split(",") if v]
    masks = [m for m in args.masks.split(",") if m]
    scenes = [s for s in args.scenes.split(",") if s]
    cases = build_cases(args.prepared, scenes, pairs, strengths, masks, args.seed,
                        args.max_cases)
    document = protocol_document(cases, scenes, pairs, strengths, masks, args.seed)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(dict(cases=len(document["cases"]),
                          verification=document["verification"])))


if __name__ == "__main__":
    main()
