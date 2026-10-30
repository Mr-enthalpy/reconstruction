"""Freeze D5 scene groups, 21-band inputs and measured-PSF operators."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from lcd_reconstruction.datasets.general_scenes import (
    CAVE_TEST, KAIST_TEST_IDS, KAIST_TRAIN_IDS, KAIST_VAL_IDS, WAVELENGTHS,
    cave_group, cave_split, load_cave_npz, load_official_mat, load_watercolors_zip,
)
from lcd_reconstruction.operators.measured_psf_source import load_measured_kernel


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def add(records, arrays, cube, **fields):
    fields = dict(fields)
    fields["array_index"] = len(arrays)
    fields["shape"] = list(cube.shape)
    fields["min"] = float(cube.min())
    fields["max"] = float(cube.max())
    if not np.isfinite(cube).all():
        raise ValueError(f"nonfinite scene {fields['scene_id']}")
    arrays.append(cube.astype(np.float32, copy=False))
    records.append(fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cave", type=Path, required=True)
    ap.add_argument("--watercolors", type=Path, required=True)
    ap.add_argument("--kaist-train", type=Path, required=True)
    ap.add_argument("--kaist-test", type=Path, required=True)
    ap.add_argument("--psf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--support", type=int, default=512)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--include-1024", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    train, val, test = [], [], []
    records = []

    cave_files = sorted(args.cave.glob("*.npz"))
    if len(cave_files) != 31:
        raise ValueError(f"expected 31 validated CAVE scenes, got {len(cave_files)}")
    for path in cave_files:
        scene = path.stem
        split = cave_split(scene)
        cube = load_cave_npz(path, args.size)
        target = train if split == "train" else val if split == "val" else test
        add(records, target, cube, scene_id=scene, dataset_id="CAVE",
            source_group_id=cave_group(scene), split=split,
            source_sha256=sha(path),
            normalization="uint16 / 65535; no per-scene mean normalization",
            band_policy="direct CAVE bands 450-650 nm / 10 nm",
            radiometric_type="approximate reflectance")

    wc = load_watercolors_zip(args.watercolors, args.size)
    add(records, test, wc, scene_id="watercolors", dataset_id="CAVE",
        source_group_id="watercolors", split="test",
        source_sha256=sha(args.watercolors),
        normalization="uint8 / 255; 8-bit source, constant alpha=255, grayscale RGB",
        band_policy="direct CAVE bands 450-650 nm / 10 nm",
        radiometric_type="approximate reflectance, 8-bit quantized")

    train_files = {int(p.stem.removeprefix("scene")): p for p in args.kaist_train.glob("*.mat")}
    test_files = {int(p.stem.removeprefix("scene")): p for p in args.kaist_test.glob("*.mat")}
    if tuple(sorted(train_files)) != tuple(sorted(KAIST_TRAIN_IDS + KAIST_VAL_IDS)):
        raise ValueError(f"unexpected KAIST non-selected ids: {sorted(train_files)}")
    if tuple(sorted(test_files)) != tuple(sorted(KAIST_TEST_IDS)):
        raise ValueError(f"unexpected KAIST selected ids: {sorted(test_files)}")
    for scene_id in KAIST_TRAIN_IDS + KAIST_VAL_IDS:
        path = train_files[scene_id]
        split = "train" if scene_id in KAIST_TRAIN_IDS else "val"
        cube, meta = load_official_mat(path, args.size, return_meta=True)
        target = train if split == "train" else val
        add(records, target, cube, scene_id=f"kaist_{scene_id}", dataset_id="KAIST",
            source_group_id=f"kaist_{scene_id}", split=split,
            source_sha256=sha(path),
            normalization="native relative intensity; no per-scene normalization",
            band_policy="linear interpolation from assumed uniform 28 samples over 450-650 nm",
            radiometric_type="relative intensity", source_meta=meta)
    for scene_id in KAIST_TEST_IDS:
        path = test_files[scene_id]
        cube, meta = load_official_mat(path, args.size, return_meta=True)
        add(records, test, cube, scene_id=f"kaist_{scene_id}", dataset_id="KAIST",
            source_group_id=f"kaist_{scene_id}", split="test",
            source_sha256=sha(path),
            normalization="native relative intensity; no per-scene normalization",
            band_policy="linear interpolation from assumed uniform 28 samples over 450-650 nm",
            radiometric_type="relative intensity", source_meta=meta)

    groups = {}
    for r in records:
        if groups.setdefault(r["source_group_id"], r["split"]) != r["split"]:
            raise ValueError(f"group leakage: {r['source_group_id']}")
    if {r["scene_id"] for r in records if r["dataset_id"] == "CAVE" and r["split"] == "test"} != (CAVE_TEST | {"watercolors"}):
        raise ValueError("CAVE test set does not match the frozen declaration")
    for name, arrays in (("train", train), ("val", val), ("test", test)):
        np.savez_compressed(args.output / f"{name}.npz", cube=np.stack(arrays))

    kernels, audit = [], []
    for nm in WAVELENGTHS:
        _, k, a = load_measured_kernel(args.psf, int(nm), args.support, args.scale)
        kernels.append(k)
        audit.append(a)
    kernels = np.asarray(kernels)
    np.savez_compressed(args.output / "kernels.npz", kernels=kernels)
    if args.include_1024:
        k1024 = []
        for nm in WAVELENGTHS:
            _, k, _ = load_measured_kernel(args.psf, int(nm), 1024, args.scale)
            k1024.append(k)
        np.savez_compressed(args.output / "kernels1024.npz", kernels=np.asarray(k1024))

    (args.output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    split_manifest = dict(
        protocol="D5-v1",
        frozen_before_training=True,
        cave=dict(train=[r["scene_id"] for r in records if r["dataset_id"] == "CAVE" and r["split"] == "train"],
                  val=[r["scene_id"] for r in records if r["dataset_id"] == "CAVE" and r["split"] == "val"],
                  test=[r["scene_id"] for r in records if r["dataset_id"] == "CAVE" and r["split"] == "test"]),
        kaist=dict(train=[r["scene_id"] for r in records if r["dataset_id"] == "KAIST" and r["split"] == "train"],
                   val=[r["scene_id"] for r in records if r["dataset_id"] == "KAIST" and r["split"] == "val"],
                   test=[r["scene_id"] for r in records if r["dataset_id"] == "KAIST" and r["split"] == "test"]),
        cave_test_note="three never-trained reserve scenes plus 8-bit watercolors stress scene",
        d4_note="D4 development-test scenes are training data in D5; D4 results are not reused as D5 test",
        source="CAVE local 31-scene set; KAIST official non-selected/selected split; no cave_1024_28 grouping assumed",
    )
    (args.output / "data_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8")
    physics = dict(
        experiment_id="spectral21_20261018_reset01",
        working_size=args.size, psf_mode="S_signed_shape_only",
        sensor_pixels_per_simulation_pixel=args.scale,
        support_sensor_pixels=args.support,
        integrated_kernel_shape=[args.support // args.scale, args.support // args.scale],
        spectral_weights=[1] * 21, signed_kernels=True,
        boundary="zero full convolution, integer-centred same crop",
        scene_to_sensor_scale="assumed; whole scene resized to working grid",
        absolute_radiometry=False, bayer=False,
        kaist_band_policy="linear interpolation; uniform 28 samples over 450-650 nm is an assumption, not metadata",
        watercolors_policy="8-bit grayscale bands with constant alpha; reported separately from 16-bit CAVE",
        psf_audit=audit,
    )
    (args.output / "physics.json").write_text(json.dumps(physics, indent=2), encoding="utf-8")
    counts = {s: {d: sum(1 for r in records if r["split"] == s and r["dataset_id"] == d)
                  for d in ("CAVE", "KAIST")} for s in ("train", "val", "test")}
    checks = dict(counts=counts, total=len(records), shape=[21, args.size, args.size],
                  kernels=kernels.shape, kernel_sha256=sha(args.output / "kernels.npz"),
                  manifest_sha256=sha(args.output / "manifest.json"),
                  physics_sha256=sha(args.output / "physics.json"),
                  cave_source_sha256={p.stem: sha(p) for p in cave_files},
                  watercolors_sha256=sha(args.watercolors))
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()


