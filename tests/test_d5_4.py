import numpy as np
import pytest
from experiments.d5_4.natural_spectral_basis import zero_sum, fit, load_split
from experiments.d5_4.natural_counterfactual import construct
from experiments.d5_4.oracle_detection import detect
from experiments.d5_4.restricted_inverse_operator import RestrictedOperator, solve, grad_normal


def fixture():
    rng = np.random.default_rng(10)
    cubes = rng.uniform(.1,.9,(2,21,12,12))
    records = [dict(split='train', scene_id=str(i), array_index=i) for i in range(2)]
    cfg = dict(seed=1, minimum_band_sum=.021, pixels_per_scene=50,
               basis_directions=3, amplitude_quantiles=[.25,.5,.75])
    return cubes, records, cfg


def test_training_only_reproducible():
    x,r,c = fixture()
    a,b = fit(x,r,c),fit(x,r,c)
    assert a == b
    r[0]['split']='val'
    with pytest.raises(ValueError): fit(x,r,c)
    with pytest.raises(ValueError): load_split('nonexistent','test',{})


def test_weighted_null_projection():
    v = np.arange(21.)
    w = np.linspace(1,3,21)
    assert abs(zero_sum(v,w)@w)<1e-13
    with pytest.raises(ValueError): zero_sum(np.ones(21))


def test_neumann_preconditioner_is_exact_for_regularizer():
    from experiments.d5_4.restricted_inverse_operator import regularizer_inverse
    rng=np.random.default_rng(22); z=rng.normal(size=(17,23))
    rhs=.7*grad_normal(z)+.003*z
    np.testing.assert_allclose(regularizer_inverse(rhs,.7,.003),z,atol=2e-12)


@pytest.mark.parametrize('mask',['smooth','texture','local'])
def test_counterfactual_bounds_and_sum(mask):
    x,r,c = fixture()
    basis=fit(x,r,c)
    args=(x[0],basis['directions'][0],10.,mask,basis)
    plus,minus,z,s=construct(*args)
    assert plus.min()>=0 and minus.min()>=0
    assert plus.max()<=1 and minus.max()<=1
    np.testing.assert_allclose(plus.sum(0),minus.sum(0),rtol=0,atol=2e-14)
    np.testing.assert_array_equal(construct(*args)[0],plus)


def test_scalar_oracle_and_noise_variance():
    t=np.arange(1.,10.)
    assert detect(t,2.5*t,0)['alpha_hat']==pytest.approx(2.5)
    assert detect(np.zeros(9),t,1)['status']=='unidentifiable'
    single=detect(t,t,.1)
    pair=detect(t,t,.1,observations=2)
    assert pair['alpha_standard_error']**2==pytest.approx(2*single['alpha_standard_error']**2)
    rng=np.random.default_rng(0)
    errors=rng.normal(0,.1,(10000,9))@t/(t@t)
    assert errors.var()==pytest.approx(single['alpha_standard_error']**2,rel=.05)


def test_adjoint_inverse_and_achromatic():
    rng=np.random.default_rng(3)
    v=zero_sum(rng.normal(size=21))
    # Identity coefficient sub-operator, through the real full-frame implementation.
    k=np.zeros((21,3,3)); k[:,1,1]=v
    b=RestrictedOperator(k,v,(8,9))
    z=rng.normal(size=(8,9)); r=rng.normal(size=(8,9))
    assert np.vdot(b.forward(z),r)==pytest.approx(np.vdot(z,b.adjoint(r)),abs=1e-12)
    estimate, report=solve(b,b.forward(z),0,1e-10,rtol=1e-9)
    assert report['converged']
    np.testing.assert_allclose(estimate,z,atol=1e-8)
    a=RestrictedOperator(np.ones((21,3,3))/9,v,(8,9))
    assert np.linalg.norm(a.forward(z))<1e-12
    assert np.vdot(z,grad_normal(z))>=0
    # Nontrivial even support exercises asymmetric crop and its true adjoint.
    b=RestrictedOperator(rng.normal(size=(21,4,4)),v,(8,9))
    assert np.vdot(b.forward(z),r)==pytest.approx(np.vdot(z,b.adjoint(r)),abs=1e-10)


def test_full_forward_difference_and_direction_metrics():
    import torch
    from experiments.d5_4.restricted_inverse_operator import differential_metrics
    rng=np.random.default_rng(4)
    v=zero_sum(rng.normal(size=21)); z=rng.normal(size=(8,9))
    x=rng.normal(size=(21,8,9)); delta=v[:,None,None]*z
    b=RestrictedOperator(rng.normal(size=(21,4,4)),v,(8,9))
    yp=b.physics(torch.tensor((x+delta)[None]))
    ym=b.physics(torch.tensor((x-delta)[None]))
    np.testing.assert_allclose((yp-ym)[0,0].numpy(),b.forward(2*z),atol=1e-12)
    residual=torch.tensor(rng.normal(size=(1,1,8,9)))
    full_adjoint=(b.physics.adjoint(residual)[0]*torch.tensor(v[:,None,None])).sum(0).numpy()
    np.testing.assert_allclose(b.adjoint(residual[0,0].numpy()),full_adjoint,atol=1e-12)
    m=differential_metrics(-delta,delta,v)
    assert m['cosine']==pytest.approx(-1)
    assert m['correct_direction_fraction']==0
    assert m['local_relative_l2']==pytest.approx(2)


def test_loader_opens_only_requested_train_archive(tmp_path,monkeypatch):
    import json
    import experiments.d5_4.natural_spectral_basis as module
    records=[dict(split='train',scene_id=str(i),array_index=i) for i in range(39)]
    records += [dict(split='test',scene_id='frozen',array_index=0)]
    (tmp_path/'manifest.json').write_text(json.dumps(records))
    opened=[]
    class Cubes:
        shape=(39,21,512,512)
        def __len__(self): return 39
    class Archive:
        def __enter__(self): return {'cube':Cubes()}
        def __exit__(self,*args): pass
    def fake_load(path,**kwargs):
        opened.append(path.name)
        assert path.name=='train.npz'
        return Archive()
    monkeypatch.setattr(module,'digest',lambda path:'expected')
    monkeypatch.setattr(module.np,'load',fake_load)
    _,selected=load_split(tmp_path,'train',dict(train_npz_sha256='expected',manifest_sha256='expected'))
    assert opened==['train.npz'] and len(selected)==39


def test_scientific_gate_rejects_nonconvergence_and_incomplete_scenes():
    from experiments.d5_4.decision_summary import group
    rows=[dict(scene_id=str(i),dataset_id='CAVE',direction=0,mask='local',level=1,
        noise_fraction=.005,arm='chromatic',inverse=dict(metrics=dict(relative_l2=.2,cosine=.95),
        convergence=dict(converged=True))) for i in range(9)]
    assert group(rows,'inverse')[0]['gate_passed']
    rows[0]['inverse']['convergence']['converged']=False
    rows[1]['inverse']['convergence']['converged']=False
    assert not group(rows,'inverse')[0]['gate_passed']
    with pytest.raises(ValueError): group(rows[:-1],'inverse')
    rows[0]['scene_id']=rows[1]['scene_id']
    with pytest.raises(ValueError): group(rows,'inverse')


def test_reduced_oracle_requires_completed_chromatic_gate():
    from experiments.d5_4.reduced_oracle import passing_strata
    rows=[dict(scene_id=str(i),dataset_id='CAVE',direction=0,mask='local',level=1,
        noise_fraction=0,arm=arm,inverse=dict(metrics=dict(relative_l2=.2,cosine=.95),
        convergence=dict(converged=True))) for arm in ('chromatic','achromatic_mean') for i in range(9)]
    assert passing_strata(dict(status='completed',rows=rows))=={(0,'local',1,0)}
    with pytest.raises(ValueError): passing_strata(dict(status='completed_shard',rows=rows))
    for row in rows:
        if row['arm']=='chromatic': row['inverse']['metrics']['relative_l2']=1.1
    assert not passing_strata(dict(status='completed',rows=rows))


def test_partition_merge_rejects_missing_and_incompatible_sources(tmp_path):
    import json
    from experiments.d5_4.merge_inverse import merge
    paths=[tmp_path/f'{i}.json' for i in range(2)]
    docs=[dict(status='completed_shard',shard=dict(index=i,count=2),
        provenance=dict(kernel='same'),rows=[]) for i in range(2)]
    for path,doc in zip(paths,docs): path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='missing or duplicate'): merge([paths[1]])
    docs[1]['provenance']['kernel']='different'
    paths[1].write_text(json.dumps(docs[1]))
    with pytest.raises(ValueError,match='provenance'): merge(paths)
    docs[1]['provenance']['kernel']='same'
    paths[1].write_text(json.dumps(docs[1]))
    with pytest.raises(ValueError,match='unique'): merge(paths)


def test_basis_transport_proof_accepts_only_line_endings(tmp_path):
    import hashlib
    from experiments.d5_4.basis_identity import verify_basis
    path=tmp_path/'basis.json'; raw=b'{\n  "direction": [1, -1]\n}\n'; path.write_bytes(raw)
    hashes=[hashlib.sha256(x).hexdigest() for x in (raw,raw.replace(b'\n',b'\r\n'))]
    assert len(verify_basis(path,hashes)['accepted_byte_sha256'])==2
    with pytest.raises(ValueError): verify_basis(path,[hashlib.sha256(raw.replace(b'-1',b'-2')).hexdigest()])
