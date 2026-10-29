import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import reconstruction_response as rr  # noqa: E402
from lcd_reconstruction.operators.psf_provenance import (  # noqa: E402
    PSFProvenanceError, build_psf_record)
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD  # noqa: E402

HW = (16, 16)


def tiny_kernels(seed=0):
    rng = np.random.default_rng(seed)
    k = rng.random((21, 4, 4), dtype=np.float32) * 0.01 + 0.001
    return k / k.sum(axis=(1, 2), keepdims=True)


def make_run(tmp_path, store_other_operator=False, drop_record=False):
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    k = tiny_kernels()
    np.savez_compressed(kernel_dir / "kernels.npz", kernels=k)
    model = GeneralSpectralPGD(k, HW, stages=2, hidden=24, depth=3)
    record = build_psf_record(model, k, kernel_dir / "kernels.npz", HW)
    if store_other_operator:
        other = GeneralSpectralPGD(tiny_kernels(9), HW, stages=2, hidden=24, depth=3)
        state = other.state_dict()
    else:
        state = model.state_dict()
    run = tmp_path / "s1c_s20261021"
    run.mkdir()
    payload = dict(state=state,
                   config=dict(seed=20261021, model="A", stages=2, width=24, depth=3),
                   hw=list(HW), epoch=3, best_epoch=3,
                   selection=dict(rule="min_validation_loss"),
                   psf_provenance=None if drop_record else record)
    torch.save(payload, run / "best.pt")
    (run / "execution.json").write_text(json.dumps(dict(
        seed=20261021, train_noise_std_absolute=0.004,
        kernel_sha256_array=record["kernel_sha256_array"])), encoding="utf-8")
    return run, kernel_dir


def test_load_arm_accepts_the_matching_kernel(tmp_path):
    run, kernel_dir = make_run(tmp_path)
    model, meta = rr.load_arm(run, kernel_dir, "cpu")
    assert meta["arm"] == "s1c"
    assert meta["seed"] == 20261021
    assert meta["psf_policy"] == "matched"
    assert meta["sigma_noise"] == pytest.approx(0.004)
    assert meta["kernel_file"] == "kernels.npz"
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_load_arm_refuses_a_changed_kernel_file(tmp_path):
    run, kernel_dir = make_run(tmp_path)
    np.savez_compressed(kernel_dir / "kernels.npz", kernels=tiny_kernels(5))
    with pytest.raises(PSFProvenanceError):
        rr.load_arm(run, kernel_dir, "cpu")


def test_load_arm_refuses_a_checkpoint_without_provenance(tmp_path):
    run, kernel_dir = make_run(tmp_path, drop_record=True)
    with pytest.raises(PSFProvenanceError):
        rr.load_arm(run, kernel_dir, "cpu")


def test_load_arm_refuses_an_operator_that_contradicts_the_record(tmp_path):
    run, kernel_dir = make_run(tmp_path, store_other_operator=True)
    with pytest.raises(PSFProvenanceError):
        rr.load_arm(run, kernel_dir, "cpu")


def test_response_metrics_recovers_a_perfect_difference():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal((21, 32, 32))
    mask = np.zeros((32, 32), np.float32)
    mask[8:24, 8:24] = 1.0
    metrics = rr.response_metrics(delta, delta, mask, (3, 7))
    assert metrics["response_fraction"] == pytest.approx(1.0)
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["delta_error_relative"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["band_contrast_direction_accuracy"] == 1.0
    assert metrics["band_contrast_gain"] == pytest.approx(1.0)


def test_response_metrics_detects_an_opposite_difference():
    rng = np.random.default_rng(1)
    delta = rng.standard_normal((21, 16, 16))
    mask = np.ones((16, 16), np.float32)
    metrics = rr.response_metrics(-delta, delta, mask, (0, 1))
    assert metrics["cosine"] == pytest.approx(-1.0)
    assert metrics["band_contrast_direction_accuracy"] == 0.0
    assert metrics["band_contrast_gain"] == pytest.approx(-1.0)


def test_response_metrics_marks_a_zero_response():
    rng = np.random.default_rng(2)
    delta = rng.standard_normal((21, 8, 8))
    mask = np.ones((8, 8), np.float32)
    metrics = rr.response_metrics(np.zeros_like(delta), delta, mask, (2, 4))
    assert metrics["response_fraction"] == 0.0
    assert metrics["delta_error_relative"] == pytest.approx(1.0)
    assert metrics["band_contrast_direction_accuracy"] == 0.0
