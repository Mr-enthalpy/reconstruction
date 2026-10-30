"""D5 general-scene loaders with explicit band, source and split contracts."""
from pathlib import Path
import re
import zipfile
import cv2
import numpy as np
import scipy.io as sio

WAVELENGTHS = np.arange(450, 651, 10, dtype=np.int64)
OFFICIAL_28_WAVELENGTHS = np.linspace(450.0, 650.0, 28)

CAVE_VALIDATION = {
    "fake_and_real_tomatoes", "fake_and_real_lemon_slices", "fake_and_real_lemons",
    "fake_and_real_sushi", "fake_and_real_strawberries",
}
CAVE_DEVELOPMENT_TEST = {
    "superballs", "cloth", "flowers", "real_and_fake_apples", "feathers", "beads",
}
CAVE_RESERVED_FINAL = {"egyptian_statue", "chart_and_stuffed_toy", "stuffed_toys"}
CAVE_WATERCOLORS = "watercolors"
CAVE_TEST = CAVE_RESERVED_FINAL | {CAVE_WATERCOLORS}

KAIST_TRAIN_IDS = (2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20)
KAIST_VAL_IDS = (25, 26, 28, 29)
KAIST_TEST_IDS = (1, 3, 7, 11, 21, 22, 23, 24, 27, 30)


def cave_split(scene_id):
    if scene_id in CAVE_TEST:
        return "test"
    if scene_id in CAVE_VALIDATION:
        return "val"
    if scene_id in CAVE_DEVELOPMENT_TEST:
        return "train"
    return "train"


def cave_group(scene_id):
    if scene_id in {"face", "photo_and_face"}:
        return "face_family"
    if scene_id in {"chart_and_stuffed_toy", "stuffed_toys"}:
        return "toy_chart_family"
    return scene_id


def resize_cube(cube, size):
    if size < 16:
        raise ValueError("working size must be >=16")
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[0] != 21 or not np.isfinite(cube).all():
        raise ValueError("expected finite [21,H,W] cube")
    if cube.shape[-2:] == (size, size):
        return cube
    return np.stack([cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA) for b in cube])


def load_cave_npz(path, size):
    with np.load(path) as z:
        cube = z["cube"]
        nm = z["wavelengths_nm"]
        if cube.dtype != np.uint16 or cube.shape != (21, 512, 512):
            raise ValueError(f"unexpected CAVE cube: {path}")
        if not np.array_equal(nm, WAVELENGTHS):
            raise ValueError(f"unexpected CAVE wavelength order: {path}")
    return resize_cube(cube.astype(np.float32) / 65535.0, size)


def load_watercolors_zip(path, size):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError(f"corrupt zip: {path}")
        bands = {}
        for name in z.namelist():
            m = re.search(r"_ms_(\d\d)\.png$", name, re.I)
            if not m or name.startswith("__MACOSX"):
                continue
            index = int(m.group(1))
            image = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
            if image.shape != (512, 512, 4) or image.dtype != np.uint8:
                raise ValueError(f"unexpected watercolors band: {name}")
            if not np.all(image[..., 3] == 255):
                raise ValueError(f"non-opaque watercolors alpha: {name}")
            if not (np.array_equal(image[..., 0], image[..., 1]) and
                    np.array_equal(image[..., 0], image[..., 2])):
                raise ValueError(f"watercolors band is not grayscale RGB: {name}")
            bands[index] = image[..., 0]
        if sorted(bands) != list(range(1, 32)):
            raise ValueError("expected 31 watercolors bands")
        cube = np.stack([bands[b] for b in range(6, 27)])
    return resize_cube(cube.astype(np.float32) / 255.0, size)


def interp_28_to_21(cube):
    cube = np.asarray(cube, dtype=np.float32)
    if cube.shape[-1] != 28:
        raise ValueError("expected 28 spectral samples")
    out = np.empty(cube.shape[:-1] + (21,), dtype=np.float32)
    for j, nm in enumerate(WAVELENGTHS):
        pos = (nm - OFFICIAL_28_WAVELENGTHS[0]) / (
            OFFICIAL_28_WAVELENGTHS[1] - OFFICIAL_28_WAVELENGTHS[0])
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, 27)
        w = float(pos - i0)
        out[..., j] = cube[..., i0] * (1.0 - w) + cube[..., i1] * w
    return out


def load_official_mat(path, size, return_meta=False):
    data = sio.loadmat(path)
    keys = [k for k in data if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"expected one variable in {path}, got {keys}")
    cube = np.asarray(data[keys[0]])
    if cube.ndim != 3 or cube.shape[-1] != 28 or not np.isfinite(cube).all():
        raise ValueError(f"expected finite [H,W,28] in {path}")
    source_shape = tuple(cube.shape)
    negative_fraction = float((cube < 0).mean())
    if cube.dtype == np.uint16:
        cube = cube.astype(np.float32) / 65535.0
    else:
        cube = np.maximum(cube.astype(np.float32), 0.0)
    h, w = cube.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cube = cube[y0:y0 + side, x0:x0 + side]
    cube = interp_28_to_21(cube).transpose(2, 0, 1)
    out = resize_cube(cube, size)
    if return_meta:
        return out, dict(source_shape=list(source_shape),
                         center_square_crop=[x0, y0, side, side],
                         negative_fraction_before_clip=negative_fraction,
                         negative_clipped=True)
    return out
