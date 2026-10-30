import numpy as np
import pytest
torch = pytest.importorskip("torch")
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics, FullSpectralPGD, scene_loss
from lcd_reconstruction.operators.measured_psf import MeasuredPSFOperator


def test_batch_physics_directional_derivative_and_d1():
    torch.manual_seed(51)
    k = torch.randn(21, 4, 6, dtype=torch.float64)
    op = SinglePlanePhysics(k, (7, 9))
    x = torch.randn(2, 21, 7, 9, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 1, 7, 9, dtype=torch.float64)
    assert torch.allclose((op(x)*y).sum(), (x*op.adjoint(y)).sum(), atol=1e-10)
    old = MeasuredPSFOperator(k.numpy(), np.ones(21))
    assert np.allclose(op(x)[0, 0].detach(), old.forward(x[0].detach().numpy()), atol=1e-10)
    d = torch.randn_like(x)
    f = lambda z: .5*(op(z)-y).square().sum()
    gradient, = torch.autograd.grad(f(x), x)
    numerical = (f(x+1e-6*d)-f(x-1e-6*d))/2e-6
    assert torch.allclose(numerical, (gradient*d).sum(), rtol=1e-7, atol=1e-5)


def test_full_state_no_basis_and_frozen_outputs():
    torch.manual_seed(8)
    k = torch.rand(21, 4, 4); k /= k.sum((-2, -1), keepdim=True)
    model = FullSpectralPGD(k, hw=(8, 8))
    assert not hasattr(model, "basis")
    target = torch.rand(2, 21, 8, 8)
    y = model.physics(target)
    out = model(y)
    assert out["spectral_map"].shape == target.shape
    assert out["global_spectrum"].shape == (2, 21)
    assert torch.allclose(out["measurement_reprojection"], model.physics(out["spectral_map"]))
    scene_loss(out, target, y.square().mean().sqrt()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.requires_grad_(False).eval()
    with torch.no_grad():
        frozen = model(y)
    assert not frozen["spectral_map"].requires_grad
    with pytest.raises(ValueError):
        model.physics.adjoint(y[:, 0])


def test_classifier_fit_ignores_held_out_labels_and_transfer_is_frozen():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/"experiments/d3/run.py"
    spec = importlib.util.spec_from_file_location("d3_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = np.array([[-2., 1.], [-1., 2.], [1., 1.], [2., 2.], [50., -50.]])
    labels = np.array([0, 0, 1, 1, 0]); train = np.array([True]*4+[False])
    head = module.fit_head(x, labels, train)
    labels[-1] = 1
    changed = module.fit_head(x, labels, train)
    for key in head:
        assert np.array_equal(head[key], changed[key])
    assert np.allclose(head["mu"], x[:4].mean(0))
    before = {k:v.copy() for k,v in head.items()}
    module.predict(head, x*.7)
    for key in head:
        assert np.array_equal(head[key], before[key])
