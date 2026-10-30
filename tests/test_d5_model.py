import torch
import pytest

from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics
from lcd_reconstruction.unrolling.hsfaut import HSFAUReconstructor, reconstruction_loss


def test_filter_split_reduces_to_validated_operator_at_unit_weights():
    torch.manual_seed(11)
    kernel = torch.randn(21, 4, 6) * 0.01
    x = torch.rand(2, 21, 16, 16)
    model = HSFAUReconstructor(kernel, (16, 16), stages=2, prior="cnn", width=24, depth=3)
    reference = SinglePlanePhysics(kernel, (16, 16))
    assert torch.allclose(model.physics(x), reference(x), atol=1e-5)
    assert torch.allclose(model.physics.lipschitz_bound(), reference.lipschitz_bound(), rtol=1e-5)
    y = torch.randn(2, 1, 16, 16)
    assert torch.allclose((model.physics(x) * y).sum(),
                          (x * model.physics.adjoint(y)).sum(), atol=1e-5)


def test_full_state_sfat_stages_and_gradients():
    torch.manual_seed(12)
    kernel = torch.randn(21, 4, 6) * 0.01
    model = HSFAUReconstructor(kernel, (16, 16), stages=3, prior="sfat",
                               sfat_width=16, sfat_levels=3)
    target = torch.rand(2, 21, 16, 16)
    observation = model.physics(target)
    out = model(observation)
    assert out["spectral_cube_hat"].shape == target.shape
    assert out["global_spectrum"].shape == (2, 21)
    assert len(out["stage_residuals"]) == 3
    assert torch.equal(out["reprojected_measurement"], model.physics(out["spectral_cube_hat"]))
    loss, parts = reconstruction_loss(out, target, observation.square().mean().sqrt())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert parts["total"] > 0
    assert not hasattr(model, "classifier") and not hasattr(model, "basis")

