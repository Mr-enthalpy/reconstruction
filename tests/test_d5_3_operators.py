import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import counterfactual_spectra as cf  # noqa: E402
import encoding_observability as eo  # noqa: E402
import otf_controls  # noqa: E402
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics  # noqa: E402

HW = (32, 32)


def kernels(seed=0, count=21):
    rng = np.random.default_rng(seed)
    k = rng.random((count, 5, 5), dtype=np.float32) * 0.02 + 0.001
    return k / k.sum(axis=(1, 2), keepdims=True)


def cube(seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((21,) + HW, dtype=np.float32) * 0.5 + 0.1).astype(np.float32)


def pair_case(seed=1):
    data = cube(seed)
    i, j = cf.band_index(490), cf.band_index(510)
    mask = cf.build_mask("edge", data, i, j, np.random.default_rng(0))
    return cf.build_pair(data, i, j, mask, 0.8)


def test_achromatic_operator_cannot_see_a_zero_sum_change():
    data = pair_case()
    k = kernels()
    achrom = eo.achromatic_from(k)
    op = SinglePlanePhysics(achrom, HW)
    with torch.no_grad():
        delta = (op(torch.from_numpy(data["x_plus"])[None]) -
                 op(torch.from_numpy(data["x_minus"])[None])).squeeze().numpy()
    reference = float(np.abs(op(torch.from_numpy(data["x_plus"])[None]).numpy()).max())
    assert np.abs(delta).max() <= 1e-5 * max(reference, 1e-6)


def test_single_band_kernel_is_also_blind():
    data = pair_case(2)
    k = kernels(3)
    op = SinglePlanePhysics(eo.single_band_from(k, 550), HW)
    with torch.no_grad():
        plus = op(torch.from_numpy(data["x_plus"])[None]).numpy()
        delta = (op(torch.from_numpy(data["x_minus"])[None]).numpy() - plus).squeeze()
    assert np.abs(delta).max() <= 1e-5 * max(float(np.abs(plus).max()), 1e-6)


def test_chromatic_difference_matches_the_analytic_relation():
    """The chromatic difference equals the kernel-deviation term plus the
    (float32-tiny) residual wideband mismatch. Both terms are checked."""
    data = pair_case(4)
    k = kernels(5).astype(np.float64)
    mean = k.mean(0, keepdims=True)
    op = SinglePlanePhysics(k, HW)
    op_diff = SinglePlanePhysics(k - mean, HW)
    op_mean = SinglePlanePhysics(np.repeat(mean, k.shape[0], axis=0), HW)
    plus = torch.from_numpy(data["x_plus"].astype(np.float64))[None]
    minus = torch.from_numpy(data["x_minus"].astype(np.float64))[None]
    with torch.no_grad():
        direct = (op(plus) - op(minus)).squeeze()
        analytic = op_diff(plus - minus).squeeze()
        wideband_term = op_mean(plus - minus).squeeze()
    scale = float(direct.norm())
    assert scale > 1e-6
    assert float((direct - analytic - wideband_term).norm()) / scale < 1e-10
    # The construction zeroes the per-pixel wideband sum only to float32 precision.
    assert float(wideband_term.norm()) / scale < 1e-5


def test_band_sensitivity_is_zero_only_for_achromatic_sets():
    k = kernels(6)
    chromatic = eo.spectral_weights(k)
    achromatic = eo.spectral_weights(eo.achromatic_from(k))
    assert chromatic.shape == (21,)
    assert chromatic.max() > 0
    assert np.allclose(achromatic, 0, atol=1e-6)


def test_noise_statistics_reports_bins_and_two_draw_variance():
    rng = np.random.default_rng(7)
    delta = rng.standard_normal(HW) * 0.01
    stats = eo.noise_statistics(delta, 0.01, HW)
    assert stats["sigma"] == 0.01
    assert set(stats["radial"]) == {"0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"}
    expected = np.linalg.norm(delta) / (np.sqrt(2) * 0.01 * np.sqrt(delta.size))
    assert stats["energy_snr"] == pytest.approx(expected, rel=1e-6)
    assert stats["matched_filter_snr"] == pytest.approx(expected * np.sqrt(delta.size), rel=1e-6)
    assert stats["per_pixel_snr"] == pytest.approx(expected, rel=1e-12)
    assert stats["matched_filter_snr"] > stats["per_pixel_snr"]
    assert 0.0 <= stats["detectable_pixel_fraction"] <= 1.0


def test_otf_control_reports_zero_sensitivity_for_achromatic_kernels():
    k = kernels(8)
    size = otf_controls.fft_size(HW)
    chromatic = otf_controls.describe("chromatic", k, size)
    achromatic = otf_controls.describe("achromatic", eo.achromatic_from(k), size)
    single = otf_controls.describe("550", eo.single_band_from(k, 550), size)
    assert chromatic["zero_sum_sensitivity"] > 0
    assert achromatic["zero_sum_sensitivity"] == pytest.approx(0.0, abs=1e-6)
    assert single["zero_sum_sensitivity"] == pytest.approx(0.0, abs=1e-6)
    assert chromatic["band_spread"]["0.20-0.25"]["band_range"] > 0
    assert achromatic["band_spread"]["0.20-0.25"]["band_range"] == pytest.approx(0.0, abs=1e-6)
