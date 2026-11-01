import copy
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from experiments.d6.inference_bundle import restore, state_digests
from lcd_reconstruction.operators.psf_provenance import (
    array_digest, physics_buffers_from_model, is_physics_key, PSFProvenanceError)
from experiments.d5_3.reconstruction_response import build_model


def small_bundle():
    torch.manual_seed(12)
    kernels = np.random.default_rng(2).normal(size=(21, 4, 4)).astype('float32')
    cfg = dict(model='A', stages=2, width=24, depth=2)
    model = build_model(cfg, kernels, (16, 16)).eval()
    state = model.state_dict()
    item = dict(kernels=torch.from_numpy(kernels), config=cfg, hw=[16,16],
                parameters={k:v.clone() for k,v in state.items() if not is_physics_key(k)},
                original_state_digests=state_digests(state),
                psf_provenance=dict(physics_buffers=physics_buffers_from_model(model),
                                    kernel_sha256_array=array_digest(kernels)))
    return model, item


def test_inference_bundle_reproduces_original_outputs_and_all_state():
    original, item = small_bundle()
    restored = restore(item)
    y = torch.randn(1,1,16,16)
    with torch.no_grad():
        assert torch.equal(original(y)['spectral_cube_hat'], restored(y)['spectral_cube_hat'])
    assert all(not p.requires_grad for p in restored.parameters())


@pytest.mark.parametrize('target', ['kernel', 'parameter', 'missing'])
def test_inference_bundle_rejects_any_state_substitution(target):
    _, item = small_bundle()
    item = copy.deepcopy(item)
    if target == 'kernel': item['kernels'][0,0,0] += 1
    elif target == 'parameter': item['parameters']['step_logits'][0] += 1
    else: item['parameters'].pop('step_logits')
    with pytest.raises(PSFProvenanceError): restore(item)
