import numpy as np
import pytest
torch=pytest.importorskip('torch')
from lcd_reconstruction.operators.measured_psf import MeasuredPSFOperator
from lcd_reconstruction.unrolling.spatial_pgd import FFTMeasuredOperator,SpatialSpectralPGD
from lcd_reconstruction.datasets.controlled_views import make_view

def test_fft_matches_d1_and_adjoint():
    rng=np.random.default_rng(23);k=rng.normal(size=(21,4,6));x=rng.normal(size=(21,7,8));y=rng.normal(size=(7,8))
    old=MeasuredPSFOperator(k,np.ones(21));op=FFTMeasuredOperator(torch.tensor(k),(7,8))
    assert np.allclose(op(torch.tensor(x)).numpy(),old.forward(x),atol=1e-11)
    assert np.allclose(op.adjoint(torch.tensor(y)).numpy(),old.adjoint(y),atol=1e-11)
    assert torch.autograd.gradcheck(lambda z: op(z),(torch.tensor(x,requires_grad=True),),fast_mode=True)

def test_view_shared_replay_no_crop():
    base=np.arange(1024,dtype=np.float32).reshape(32,32)
    x=np.stack([base*(b+1) for b in range(21)])
    a,m=make_view(x,11);b,n=make_view(x,11)
    assert np.array_equal(a,b) and m==n and m['corners_inside']
    for j in range(21): assert np.allclose(a[j],a[0]*(j+1),rtol=2e-6,atol=.01)

def test_outputs_reprojection_gradient():
    torch.manual_seed(7);k=torch.rand(21,4,4);k/=k.sum((-2,-1),keepdim=True)
    basis=torch.linalg.qr(torch.randn(21,4)).Q
    model=SpatialSpectralPGD(k,basis,hw=(8,8));y=torch.rand(2,8,8)
    out=model(y);assert out['spectral_map'].shape==(2,21,8,8)
    assert torch.allclose(out['global_spectrum'],out['spectral_map'].mean((-2,-1)))
    full=FFTMeasuredOperator(k,(8,8))
    assert torch.allclose(out['measurement_reprojection'],full(out['spectral_map']),atol=1e-5)
    loss=(out['measurement_reprojection']-y).square().mean()+out['spectral_map'].square().mean()
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
