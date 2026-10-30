import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import make_achromatic_control as mac  # noqa: E402
from lcd_reconstruction.operators.psf_provenance import array_digest  # noqa: E402


def kernels(seed=0):
    rng = np.random.default_rng(seed)
    k = rng.random((21, 6, 6), dtype=np.float32) * 0.02
    return (k / k.sum(axis=(1, 2), keepdims=True)).astype(np.float32)


def test_array_digest_matches_the_package_implementation():
    k = kernels(1)
    assert mac.array_digest(k) == array_digest(k)


def test_band_control_replicates_the_550_nm_kernel():
    k = kernels(2)
    control, description = mac.build(k, "band", 550)
    assert np.array_equal(control, np.repeat(k[10][None], 21, axis=0))
    assert np.allclose(control.sum(axis=(1, 2)), k[10].sum(axis=(0, 1)))
    assert "550" in description
    assert control.dtype == np.float32


def test_mean_control_replicates_the_arithmetic_mean():
    k = kernels(3)
    control, description = mac.build(k, "mean")
    assert np.array_equal(control, np.repeat(k.mean(0, keepdims=True), 21, axis=0))
    assert "mean" in description


def test_band_must_exist_and_kind_must_be_known():
    k = kernels(4)
    with pytest.raises(ValueError):
        mac.build(k, "band", 555)
    with pytest.raises(ValueError):
        mac.build(k, "median")


def test_shape_is_validated():
    with pytest.raises(ValueError):
        mac.build(np.zeros((20, 5, 5), np.float32))
