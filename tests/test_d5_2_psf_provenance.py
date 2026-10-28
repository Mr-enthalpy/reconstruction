import numpy as np
import pytest
import torch

from lcd_reconstruction.operators.psf_provenance import (
    PSFProvenanceError, array_digest, build_psf_record, is_physics_key,
    physics_buffers_from_model, physics_buffers_from_state, resolve_psf_policy)
from lcd_reconstruction.unrolling.general_pgd import GeneralSpectralPGD
from lcd_reconstruction.unrolling.hsfaut import HSFAUReconstructor

HW = (16, 16)


def chromatic():
    rng = np.random.default_rng(3)
    return (rng.random((21, 5, 5), dtype=np.float32) * .01 + .002)


def achromatic():
    k = chromatic()
    return np.repeat(k.mean(0, keepdims=True), 21, axis=0)


def model_a(kernels):
    return GeneralSpectralPGD(kernels, HW, stages=2, hidden=24, depth=3)


def model_c(kernels):
    return HSFAUReconstructor(kernels, HW, stages=2, width=24, depth=3, prior="cnn")


def test_physics_keys_cover_both_architectures():
    a_keys = {k for k in model_a(chromatic()).state_dict() if is_physics_key(k)}
    c_keys = {k for k in model_c(chromatic()).state_dict() if is_physics_key(k)}
    assert a_keys == {"physics.operator.kernel_fft", "L"}
    assert c_keys == {"physics.kernel_fft", "physics.weights", "L"}


def test_operator_digest_is_stable_and_kernel_sensitive():
    first = physics_buffers_from_model(model_a(chromatic()))
    second = physics_buffers_from_model(model_a(chromatic()))
    other = physics_buffers_from_model(model_a(achromatic()))
    assert first == second
    assert first["physics.operator.kernel_fft"] != other["physics.operator.kernel_fft"]
    assert array_digest(chromatic()) != array_digest(achromatic())


def test_array_digest_ignores_container_but_not_values():
    kernels = chromatic()
    assert array_digest(kernels) == array_digest(kernels.astype(np.float64))
    changed = kernels.copy()
    changed[0, 0, 0] += 1e-6
    assert array_digest(changed) != array_digest(kernels)


def test_record_roundtrip_matches(tmp_path):
    kernels = chromatic()
    path = tmp_path / "kernels.npz"
    np.savez_compressed(path, kernels=kernels)
    model = model_a(kernels)
    record = build_psf_record(model, kernels, path, HW)
    assert record["kernel_sha256_array"] == array_digest(kernels)
    policy = resolve_psf_policy(record, physics_buffers_from_model(model),
                                physics_buffers_from_state(model.state_dict()),
                                requested_array=array_digest(kernels))
    assert policy["policy"] == "matched"
    assert policy["formal_result"] is True
    assert policy["use_checkpoint_operator"] is True


def test_wrong_kernel_is_refused():
    chromatic_model = model_a(chromatic())
    with pytest.raises(PSFProvenanceError):
        resolve_psf_policy(None, physics_buffers_from_model(model_a(achromatic())),
                           physics_buffers_from_state(chromatic_model.state_dict()),
                           requested_array=array_digest(achromatic()))


def test_declared_mismatch_uses_the_requested_operator():
    kernels = achromatic()
    model = model_a(kernels)
    stored = physics_buffers_from_state(model_a(chromatic()).state_dict())
    policy = resolve_psf_policy(None, physics_buffers_from_model(model), stored,
                                allow_mismatch=True, split="val",
                                requested_array=array_digest(kernels))
    assert policy["policy"] == "declared-mismatch"
    assert policy["formal_result"] is False
    assert policy["use_checkpoint_operator"] is False
    kept = {k: v for k, v in model_a(chromatic()).state_dict().items()
            if not is_physics_key(k)}
    model.load_state_dict(kept, strict=False)
    assert physics_buffers_from_model(model) == policy["requested"]


def test_declared_mismatch_is_refused_on_test_split():
    with pytest.raises(PSFProvenanceError):
        resolve_psf_policy(None, physics_buffers_from_model(model_a(achromatic())),
                           physics_buffers_from_state(model_a(chromatic()).state_dict()),
                           allow_mismatch=True, split="test",
                           requested_array=array_digest(achromatic()))


def test_legacy_checkpoint_is_not_a_formal_result():
    model = model_a(chromatic())
    policy = resolve_psf_policy(None, physics_buffers_from_model(model),
                                physics_buffers_from_state(model.state_dict()))
    assert policy["policy"] == "matched-legacy-no-record"
    assert policy["formal_result"] is False
    assert policy["use_checkpoint_operator"] is True


def test_tampered_record_is_always_fatal(tmp_path):
    kernels = chromatic()
    path = tmp_path / "kernels.npz"
    np.savez_compressed(path, kernels=kernels)
    model = model_a(kernels)
    record = build_psf_record(model, kernels, path, HW)
    record["physics_buffers"] = {"physics.operator.kernel_fft": "0" * 64, "L": "0" * 64}
    with pytest.raises(PSFProvenanceError):
        resolve_psf_policy(record, physics_buffers_from_model(model),
                           physics_buffers_from_state(model.state_dict()),
                           allow_mismatch=True, requested_array=array_digest(kernels))
