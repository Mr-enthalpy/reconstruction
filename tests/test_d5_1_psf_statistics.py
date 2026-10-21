import numpy as np
import pytest

from lcd_reconstruction.operators.measured_psf_source import block_integrate, crop_center
from lcd_reconstruction.operators.psf_statistics import (
    negative_absolute_mass_fraction, negative_fraction,
    normalization_processing_consistency, projection_summary,
    signed_absolute_radius_descriptive,
)


def test_projection_statistics_are_pre_and_post_renormalization():
    kernel = np.array([[-0.2, -0.1], [0.3, 0.4]])
    summary = projection_summary(kernel)
    assert negative_fraction(kernel) == pytest.approx(0.5)
    assert negative_absolute_mass_fraction(kernel) == pytest.approx(0.3)
    assert summary["negative_fraction_before_projection"] == pytest.approx(0.5)
    assert summary["negative_fraction_after_projection"] == 0.0
    assert summary["negative_fraction_after_shape_renormalization"] == 0.0
    assert summary["negative_absolute_mass_fraction_before_renormalization"] == pytest.approx(0.3)
    assert summary["removed_absolute_mass_fraction_before_renormalization"] == pytest.approx(0.3)
    assert summary["kernel_sum_after_projection"] == pytest.approx(0.7)


def test_projection_does_not_mix_normalized_mass_with_raw_mass():
    kernel = np.array([[-0.4, 0.1, 0.2, 0.3]])
    summary = projection_summary(kernel)
    assert summary["removed_absolute_mass_fraction_before_renormalization"] == pytest.approx(0.4)
    assert summary["kernel_sum_after_shape_renormalization"] == pytest.approx(1.0)


def test_2x2_integration_is_cfa_phase_invariant_sum():
    pattern = np.array([[1.0, 2.0], [3.0, 4.0]])
    tile = np.tile(pattern, (8, 8))
    assert block_integrate(tile, 2, "sum").shape == (8, 8)
    assert np.allclose(block_integrate(tile, 2, "sum"), 10.0)
    assert np.allclose(block_integrate(tile, 2, "mean"), 2.5)


def test_center_crop_has_one_shared_origin():
    kernel = np.arange(1024, dtype=float).reshape(32, 32)
    a = crop_center(kernel, 16)
    b = crop_center(kernel, 16)
    assert np.array_equal(a, b)
    assert a.shape == (16, 16)


def test_support_comparison_is_not_physical_truncation_evidence():
    a = np.ones((4, 4))
    b = np.ones((4, 4)) * 2
    result = normalization_processing_consistency(a, b)
    assert result["status"] == "processing consistency only"
    assert "not evidence" in result["note"]
    with pytest.raises(ValueError):
        normalization_processing_consistency(a, np.ones((2, 2)))


def test_signed_radius_is_descriptive_only():
    result = signed_absolute_radius_descriptive(np.ones((8, 8)))
    assert result["physical_energy_enclosure"] is False
    assert "descriptive" in result["note"]
