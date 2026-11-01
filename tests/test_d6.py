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


def test_original_physics_restore_checks_every_buffer():
    original,item=small_bundle()
    physics={k:v.clone() for k,v in original.state_dict().items() if is_physics_key(k)}
    assert state_digests(restore(item,original_physics=physics).state_dict())==state_digests(original.state_dict())
    physics['L']+=.01
    with pytest.raises(PSFProvenanceError,match='buffer digest'):
        restore(item,original_physics=physics)


@pytest.mark.parametrize('target', ['kernel', 'parameter', 'missing'])
def test_inference_bundle_rejects_any_state_substitution(target):
    _, item = small_bundle()
    item = copy.deepcopy(item)
    if target == 'kernel': item['kernels'][0,0,0] += 1
    elif target == 'parameter': item['parameters']['step_logits'][0] += 1
    else: item['parameters'].pop('step_logits')
    with pytest.raises(PSFProvenanceError): restore(item)


def test_g4_denies_access_before_reading_protocol_or_data(tmp_path):
    from experiments.d6.g4 import read_protocol
    with pytest.raises(ValueError,match='authorization'):
        read_protocol(tmp_path/'missing.json','untrusted')


def test_g4_noise_is_fixed_by_scene_not_arm_or_model_seed():
    from experiments.d6.g4 import shared_noise
    p=dict(noise_seed=20261101,absolute_sigma=.017329356670379638)
    a=shared_noise((32,32),1,p)
    assert np.array_equal(a,shared_noise((32,32),1,p))
    assert not np.array_equal(a,shared_noise((32,32),2,p))


def test_g4_stability_uses_seeds_and_scene_direction_not_pooled_samples():
    from experiments.d6.g4 import summarize,METRICS
    models=[dict(run_dir=f'{a}_{s}',arm=a,seed=s) for a in ('s1c','s3a','s3a550') for s in (1,2,3)]
    rows=[]
    for m in models:
        for i in range(14):
            metrics={k:float(1+(m['arm']=='s1c')) for k in METRICS}
            metrics.update(band_rmse=[1.]*21,band_relative_l2=[1.]*21)
            rows.append(dict(**m,scene_id=str(i),dataset_id='CAVE' if i<4 else 'KAIST',metrics=metrics))
    out=summarize(rows,dict(models=models,scene_direction_minimum=dict(ALL=13,CAVE=4,KAIST=9)))
    c=out['paired_comparisons'][0]
    assert len(c['seed_means'])==3 and c['scene_direction_counts']==[14]*3
    assert c['stability_passed']
    with pytest.raises(ValueError,match='coverage'):
        summarize(rows[:-1],dict(models=models))
