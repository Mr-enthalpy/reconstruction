import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import counterfactual_spectra as cf  # noqa: E402


def cube(seed=0, hw=(24, 24)):
    rng = np.random.default_rng(seed)
    return (rng.random((21,) + hw, dtype=np.float32) * 0.6 + 0.05).astype(np.float32)


def test_band_index_matches_the_21_band_grid():
    assert cf.band_index(450) == 0
    assert cf.band_index(550) == 10
    assert cf.band_index(650) == 20
    with pytest.raises(ValueError):
        cf.band_index(455)


def test_pair_keeps_wideband_sum_and_non_negativity():
    data = cube()
    i, j = cf.band_index(490), cf.band_index(510)
    mask = cf.build_mask("uniform", data, i, j, np.random.default_rng(1))
    pair = cf.build_pair(data, i, j, mask, 1.0)
    verify = cf.verify_pair(pair)
    assert pair["x_plus"].min() >= 0 and pair["x_minus"].min() >= 0
    assert verify["wideband_relative_residual"] <= 1e-6
    assert verify["wideband_max_abs_residual"] >= 0


def test_difference_is_exactly_two_q_on_the_two_bands():
    data = cube(3)
    i, j = cf.band_index(450), cf.band_index(650)
    mask = cf.build_mask("stripes", data, i, j, np.random.default_rng(2))
    pair = cf.build_pair(data, i, j, mask, 0.5)
    delta = pair["x_plus"].astype(np.float64) - pair["x_minus"].astype(np.float64)
    q = pair["q"].astype(np.float64)
    assert np.allclose(delta[i], 2 * q, atol=1e-6)
    assert np.allclose(delta[j], -2 * q, atol=1e-6)
    others = [b for b in range(21) if b not in (i, j)]
    assert np.abs(delta[others]).max() == 0


def test_amplitude_respects_the_non_negativity_bound():
    data = cube(7)
    i, j = cf.band_index(530), cf.band_index(570)
    mask = np.ones(data.shape[-2:], dtype=np.float32)
    q = cf.amplitude(data, i, j, mask, 1.0)
    cap = np.minimum(data[i], data[j])
    assert np.all(q <= cap + 1e-7)
    assert np.all(q >= 0)


def test_invalid_constructions_are_refused():
    data = cube(11)
    i, j = 0, 1
    mask = np.ones(data.shape[-2:], dtype=np.float32)
    with pytest.raises(ValueError):
        cf.build_pair(data, i, i, mask, 0.5)
    with pytest.raises(ValueError):
        cf.amplitude(data, i, j, mask, 0.0)
    with pytest.raises(ValueError):
        cf.amplitude(data, i, j, mask, 1.5)
    with pytest.raises(ValueError):
        cf.build_mask("triangle", data, i, j, np.random.default_rng(0))


def test_masks_are_bounded_and_shape_safe():
    data = cube(13)
    i, j = 2, 17
    for name in sorted(cf.MASK_BUILDERS):
        mask = cf.build_mask(name, data, i, j, np.random.default_rng(4))
        assert mask.shape == data.shape[-2:]
        assert mask.min() >= 0 and mask.max() <= 1
        assert np.isfinite(mask).all()
    assert cf.build_mask("uniform", data, i, j, np.random.default_rng(0)).min() == 1


def test_mask_generation_is_reproducible_across_processes():
    """Scene keys must not depend on Python's salted hash()."""
    data = cube(17)
    first = cf._case_rng(20261021, "kaist_28", 4, 8, "block", 0.5).integers(0, 1000, 3)
    second = cf._case_rng(20261021, "kaist_28", 4, 8, "block", 0.5).integers(0, 1000, 3)
    other = cf._case_rng(20261021, "kaist_29", 4, 8, "block", 0.5).integers(0, 1000, 3)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    a = cf.build_mask("block", data, 4, 8, np.random.default_rng([1, 2]))
    b = cf.build_mask("block", data, 4, 8, np.random.default_rng([1, 2]))
    assert np.array_equal(a, b)


def test_broken_pair_is_rejected():
    data = cube(19)
    i, j = 1, 5
    mask = np.ones(data.shape[-2:], dtype=np.float32)
    pair = cf.build_pair(data, i, j, mask, 0.5)
    pair["x_minus"][j] += 1e-2
    with pytest.raises(ValueError):
        cf.verify_pair(pair)


def test_negative_pair_is_rejected():
    data = cube(23)
    i, j = 1, 5
    mask = np.ones(data.shape[-2:], dtype=np.float32)
    pair = cf.build_pair(data, i, j, mask, 0.5)
    pair["x_plus"][i] = -1.0
    with pytest.raises(ValueError):
        cf.verify_pair(pair)
