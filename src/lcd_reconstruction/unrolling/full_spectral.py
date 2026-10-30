"""Full-channel scene reconstruction; no task labels or fitted spectral basis."""
import torch
from torch import nn
from .spatial_pgd import FFTMeasuredOperator


class SinglePlanePhysics(nn.Module):
    """D1/D2 convolution with an explicit singleton measurement-channel axis."""
    def __init__(self, kernels, hw):
        super().__init__()
        if kernels.shape[0] != 21:
            raise ValueError("21 measured bands required")
        self.operator = FFTMeasuredOperator(kernels, hw)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("expected [batch,21,H,W]")
        return self.operator(x).unsqueeze(1)

    def adjoint(self, y):
        if y.ndim != 4 or y.shape[1] != 1:
            raise ValueError("expected [batch,1,H,W]")
        return self.operator.adjoint(y[:, 0])

    def lipschitz_bound(self):
        return self.operator.lipschitz_bound()


class FullSpectralPGD(nn.Module):
    """Three explicit data steps; shared 21->32->21 spatial-spectral proximal."""
    def __init__(self, kernels, hw=(32, 32), stages=3, hidden=32):
        super().__init__()
        if hidden < 21:
            raise ValueError("D3 baseline does not use a spectral bottleneck")
        self.physics = SinglePlanePhysics(torch.as_tensor(kernels, dtype=torch.float32), hw)
        self.register_buffer("L", self.physics.lipschitz_bound().clamp_min(1e-8))
        self.prox = nn.Sequential(nn.Conv2d(21, hidden, 3, padding=1), nn.Tanh(),
                                  nn.Conv2d(hidden, 21, 3, padding=1))
        nn.init.zeros_(self.prox[-1].weight)
        nn.init.zeros_(self.prox[-1].bias)
        self.step_logits = nn.Parameter(torch.zeros(stages))

    def forward(self, y):
        x = self.physics.adjoint(y) / self.L
        for step in self.step_logits:
            x = x - 1.8 * step.sigmoid() / self.L * self.physics.adjoint(self.physics(x) - y)
            x = x + .1 * self.prox(x)
        x = x - self.physics.adjoint(self.physics(x) - y) / self.L
        reprojection = self.physics(x)
        return dict(spectral_map=x, global_spectrum=x.mean((-2, -1)),
                    measurement_reprojection=reprojection, residual=reprojection-y)


def scene_loss(out, target, measurement_scale):
    """Reconstruction-only objective; intensity, structure, spectral shape and data."""
    x = out["spectral_map"]
    mse = (x-target).square().mean()
    spatial = sum((torch.diff(x, dim=d)-torch.diff(target, dim=d)).square().mean()
                  for d in [-1, -2]) / 2
    valid = torch.linalg.vector_norm(target, dim=1) > 1e-3
    cosine = torch.nn.functional.cosine_similarity(x, target, dim=1, eps=1e-8)
    shape = (1-cosine[valid]).mean() if valid.any() else mse*0
    data = out["residual"].square().mean() / measurement_scale**2
    return mse + .2*spatial + .05*shape + .02*data
