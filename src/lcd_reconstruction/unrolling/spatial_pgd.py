"""Small low-rank PGD unfolding with a measured linear convolution operator."""
import torch
from torch import nn
import torch.nn.functional as F


class FFTMeasuredOperator(nn.Module):
    """Full zero-padded convolution followed by the D1 integer-centred crop."""
    def __init__(self, kernels, hw):
        super().__init__()
        k = torch.as_tensor(kernels)
        if k.ndim != 3 or not torch.isfinite(k).all():
            raise ValueError('finite [channels,kh,kw] kernels required')
        self.hw = tuple(hw)
        self.origin = (k.shape[-2] // 2, k.shape[-1] // 2)
        self.fft_hw = tuple(1 << (a+b-2).bit_length() for a,b in zip(hw,k.shape[-2:]))
        self.register_buffer('kernel_fft', torch.fft.rfft2(k, s=self.fft_hw))

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.hw or x.shape[-3] != self.kernel_fft.shape[0]:
            raise ValueError('input shape differs from operator contract')
        full = torch.fft.irfft2((torch.fft.rfft2(x,s=self.fft_hw)*self.kernel_fft).sum(-3),s=self.fft_hw)
        oy,ox=self.origin;h,w=self.hw
        return full[...,oy:oy+h,ox:ox+w]

    def adjoint(self, y):
        if tuple(y.shape[-2:]) != self.hw:
            raise ValueError('measurement shape differs from operator contract')
        oy,ox=self.origin;h,w=self.hw;fh,fw=self.fft_hw
        padded=F.pad(y,(ox,fw-w-ox,oy,fh-h-oy))
        full=torch.fft.irfft2(torch.fft.rfft2(padded,s=self.fft_hw).unsqueeze(-3)*self.kernel_fft.conj(),s=self.fft_hw)
        return full[...,:h,:w]

    def lipschitz_bound(self):
        return self.kernel_fft.abs().square().sum(0).max().real


class SpatialSpectralPGD(nn.Module):
    def __init__(self,kernels,basis,hw=(32,32),stages=3,hidden=8):
        super().__init__()
        basis=torch.as_tensor(basis,dtype=torch.float32)
        self.register_buffer('basis',basis)
        effective=torch.einsum('lr,lhw->rhw',basis,torch.as_tensor(kernels,dtype=torch.float32))
        self.physics=FFTMeasuredOperator(effective,hw)
        self.register_buffer('lipschitz',self.physics.lipschitz_bound().clamp_min(1e-8))
        rank=basis.shape[1]
        self.prox=nn.Sequential(nn.Conv2d(rank,hidden,3,padding=1),nn.Tanh(),nn.Conv2d(hidden,rank,3,padding=1))
        nn.init.zeros_(self.prox[-1].weight);nn.init.zeros_(self.prox[-1].bias)
        self.step_logits=nn.Parameter(torch.zeros(stages))

    def forward(self,y):
        c=self.physics.adjoint(y)/self.lipschitz
        for a in self.step_logits:
            c=c-(1.8*a.sigmoid()/self.lipschitz)*self.physics.adjoint(self.physics(c)-y)
            c=c+0.1*self.prox(c)
        # A final bounded data step cannot increase the least-squares objective.
        c=c-self.physics.adjoint(self.physics(c)-y)/self.lipschitz
        spectral_map=torch.einsum('lr,brhw->blhw',self.basis,c)
        reprojection=self.physics(c)
        return dict(spectral_map=spectral_map,global_spectrum=spectral_map.mean((-2,-1)),
                    measurement_reprojection=reprojection,coefficient_maps=c,
                    negative_fraction=(spectral_map<0).float().mean((-3,-2,-1)))
