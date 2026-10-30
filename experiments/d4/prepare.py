"""Whole-scene CAVE inputs; no face labels/ROIs and no final-holdout array loading."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

VALIDATION = {"fake_and_real_tomatoes", "fake_and_real_lemon_slices",
              "fake_and_real_lemons", "fake_and_real_sushi", "fake_and_real_strawberries"}
DEVELOPMENT_TEST = {"superballs", "cloth", "flowers", "real_and_fake_apples", "feathers", "beads"}
# Previously inspected for data quality, but not trained/evaluated in D2/D3 N1.
# Keep related toy/chart scenes together. This is not a never-viewed collection.
FINAL = {"egyptian_statue", "chart_and_stuffed_toy", "stuffed_toys"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(scene):
    if scene in FINAL:
        return "reserved_final"
    if scene in VALIDATION:
        return "val"
    if scene in DEVELOPMENT_TEST:
        return "development_test"
    return "train"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cave", type=Path, required=True)
    p.add_argument("--psf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=128)
    a = p.parse_args()
    if a.size < 16:
        raise ValueError("working size must be >=16")
    a.output.mkdir(parents=True, exist_ok=False)
    records, arrays = [], []
    for f in sorted(a.cave.glob("*.npz")):
        scene = f.stem
        if scene == "watercolors":
            raise ValueError("unresolved RGBA/8-bit source must not enter this loader")
        split = split_for(scene)
        r = dict(scene_id=scene, source_group_id="toy_chart_family" if scene in
                 {"chart_and_stuffed_toy", "stuffed_toys"} else
                 "face_family" if scene in {"face", "photo_and_face"} else scene,
                 split=split, source_sha256=sha(f), dataset_id="CAVE",
                 radiometric_type="approximate reflectance",
                 normalization="uint16 / 65535; no per-scene mean normalization",
                 wavelengths_nm=list(range(450, 651, 10)), array_index=None)
        if split != "reserved_final":
            with np.load(f) as z:
                cube = z["cube"]
                if cube.dtype != np.uint16 or cube.shape != (21, 512, 512):
                    raise ValueError(f"unexpected cube: {f}")
                if not np.array_equal(z["wavelengths_nm"], np.arange(450, 651, 10)):
                    raise ValueError(f"unexpected wavelength order: {f}")
                cube = np.stack([cv2.resize(b.astype(np.float32)/65535,
                                 (a.size, a.size), interpolation=cv2.INTER_AREA) for b in cube])
            r["array_index"] = len(arrays)
            arrays.append(cube)
        records.append(r)
    if len(records) != 31 or {r["scene_id"] for r in records if r["split"] == "reserved_final"} != FINAL:
        raise ValueError("expected 31 validated CAVE scenes, including three reserved scenes")
    groups = {}
    for r in records:
        if groups.setdefault(r["source_group_id"], r["split"]) != r["split"]:
            raise ValueError("group leakage")
    kernels, audit = [], []
    for nm in range(450, 651, 10):
        f = a.psf / f"roi_extent_review/roi1024_{nm}.npz"
        with np.load(f) as z:
            m = json.loads(str(z["metadata_json"]))
            if m["wavelength_nm"] != nm or m["roi_xywh"] != [866, 554, 1024, 1024] or m["normalization"] != "none":
                raise ValueError("PSF geometry or normalization mismatch")
            probe = {201.6: 1, 403.2: 2, 1612.8: 3}[m["camera_settings"]["requested_exposure_us"]]
            dark_paths = [a.psf / f"received/R21Dark{phase}01/probe_{probe:03d}_mean.npy"
                          for phase in ("Pre", "Post")]
            dark = sum(np.load(p)[554:1578, 866:1890].astype(float) for p in dark_paths)/2
            corrected = z["psf"].astype(float)-dark
        border = np.ones((1024, 1024), bool)
        border[40:-40, 40:-40] = False
        corrected -= corrected[border].mean()
        k = corrected[256:768, 256:768]
        if not np.isfinite(k).all() or k.sum() <= 0:
            raise ValueError("invalid corrected PSF")
        k = k/k.sum()
        kernels.append(k.reshape(64, 8, 64, 8).sum((1, 3)))
        audit.append(dict(wavelength_nm=nm, roi_sha256=sha(f),
                          dark_sha256=[sha(p) for p in dark_paths],
                          signed_sum=float(corrected.sum()),
                          negative_fraction=float((k < 0).mean())))
    np.savez_compressed(a.output/"scenes.npz", cube=np.stack(arrays))
    np.savez_compressed(a.output/"kernels.npz", kernels=np.asarray(kernels, np.float32))
    (a.output/"manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (a.output/"physics.json").write_text(json.dumps(dict(
        experiment_id="spectral21_20261018_reset01", psf_audit=audit,
        sensor_pixels_per_simulation_pixel=8, support_sensor_pixels=512,
        spectral_weights=[1]*21, signed_kernels=True,
        boundary="zero full convolution, integer-centred same crop",
        scene_to_sensor_scale="assumed; whole scene resized to working grid",
        absolute_radiometry=False, bayer=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
