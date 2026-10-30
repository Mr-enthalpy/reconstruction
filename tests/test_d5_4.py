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
    m=differential_metrics(-delta,delta,v)
    assert m['cosine']==pytest.approx(-1)
    assert m['correct_direction_fraction']==0
    assert m['local_relative_l2']==pytest.approx(2)
