import importlib.util
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD, reconstruction_loss
from lcd_reconstruction.rgb_frontend import recover_rgb, propose_faces


def load_script(name):
    path = Path(__file__).resolve().parents[1]/"experiments"/"d4"/f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"d4_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_full_state_optimizer_and_adjoint_with_signed_even_kernel():
    torch.manual_seed(6)
    kernel = torch.randn(21, 4, 6)*.01
    model = GeneralSpectralPGD(kernel, (16, 16), stages=2, hidden=24, depth=3)
    target = torch.rand(2, 21, 16, 16)
    y = model.physics(target)
    test_y = torch.randn_like(y)
    assert torch.allclose((y*test_y).sum(), (target*model.physics.adjoint(test_y)).sum(), atol=1e-5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(2):
        opt.zero_grad()
        out = model(y)
        assert out["spectral_cube_hat"].shape == target.shape
        assert out["global_spectrum"].shape == (2, 21)
        assert torch.equal(out["reprojected_measurement"], model.physics(out["spectral_cube_hat"]))
        reconstruction_loss(out, target, y.square().mean().sqrt()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert not hasattr(model, "classifier") and not hasattr(model, "basis")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cpu_forward_adjoint_direction_and_training_parity():
    torch.manual_seed(46)
    kernel = torch.randn(21, 4, 6, dtype=torch.float64)*.01
    from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics
    cpu = SinglePlanePhysics(kernel, (17, 19))
    gpu = SinglePlanePhysics(kernel.cuda(), (17, 19))
    x = torch.randn(2, 21, 17, 19, dtype=torch.float64)
    y = torch.randn(2, 1, 17, 19, dtype=torch.float64)
    assert torch.allclose(cpu(x), gpu(x.cuda()).cpu(), atol=1e-10, rtol=1e-9)
    assert torch.allclose(cpu.adjoint(y), gpu.adjoint(y.cuda()).cpu(), atol=1e-10, rtol=1e-9)
    z = x.cuda().requires_grad_()
    d = torch.randn_like(z)
    f = lambda v: .5*(gpu(v)-y.cuda()).square().sum()
    gradient, = torch.autograd.grad(f(z), z)
    finite_difference = (f(z+1e-6*d)-f(z-1e-6*d))/2e-6
    assert torch.allclose(finite_difference, (gradient*d).sum(), rtol=1e-6, atol=1e-5)
    model = GeneralSpectralPGD(kernel.float(), (17, 19), stages=2, hidden=24).cuda()
    target = x.float().cuda()
    observation = model.physics(target)
    loss = reconstruction_loss(model(observation), target, observation.square().mean().sqrt())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_scene_splits_do_not_filter_on_face_labels():
    mod = load_script("prepare")
    assert mod.split_for("face") == mod.split_for("photo_and_face") == "train"
    assert mod.split_for("chart_and_stuffed_toy") == mod.split_for("stuffed_toys") == "reserved_final"
    assert not mod.FINAL & mod.VALIDATION and not mod.FINAL & mod.DEVELOPMENT_TEST


def test_common_augmentation_replays_and_metrics_use_fixed_range():
    mod = load_script("train")
    plane = torch.rand(1, 16, 16)
    x = plane.repeat(21, 1, 1)*torch.arange(1, 22)[:, None, None]
    a, b = mod.augment(x, 3, [.9, 1.1]), mod.augment(x, 3, [.9, 1.1])
    assert torch.equal(a, b)
    assert torch.allclose(a[20], 21*a[0])
    truth = torch.ones(1, 21, 16, 16)*.5
    met = mod.metrics(truth+.1, truth, torch.ones(1), torch.ones(1))
    assert met["psnr_unit_range_db"] == pytest.approx(20, abs=1e-4)
    assert met["measurement_relative_l2"] == 0


@pytest.mark.parametrize("cfa", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_rgb_bayer_channel_order_and_no_false_detection_claim(cfa):
    raw = np.empty((16, 16), np.uint16)
    for idx, letter in enumerate(cfa):
        raw[idx//2::2, idx%2::2] = {"R": 3000, "G": 2000, "B": 1000}[letter]
    out = recover_rgb(raw, cfa=cfa, black_level=0, white_level=4000, observation_id="synthetic")
    assert np.allclose(out["rgb_image"][2:-2, 2:-2], [.75, .5, .25])
    assert out["candidate_boxes"] is None
    with pytest.raises(ValueError):
        recover_rgb(raw, cfa="scalar", black_level=0, white_level=4000, observation_id="x")
    with pytest.raises(FileNotFoundError):
        propose_faces(out, "missing_model.onnx")
