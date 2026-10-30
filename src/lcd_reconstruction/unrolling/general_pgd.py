"""Full-channel PGD reconstruction with no classifier or dataset-label interface."""
import torch
from torch import nn
from .full_spectral import SinglePlanePhysics


class GeneralSpectralPGD(nn.Module):
    """Shared spatial-spectral proximal, full 21-channel state at every stage."""
    def __init__(self, kernels, hw, stages=6, hidden=64, depth=4):
        super().__init__()
        if stages < 1 or hidden < 21 or depth < 2:
            raise ValueError("positive stages, hidden >= 21, depth >= 2 required")
        self.physics = SinglePlanePhysics(torch.as_tensor(kernels, dtype=torch.float32), hw)
        self.register_buffer("L", self.physics.lipschitz_bound().clamp_min(1e-8))
        layers = [nn.Conv2d(21, hidden, 3, padding=1), nn.GELU()]
        for _ in range(depth-2):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(hidden, 21, 3, padding=1)]
        self.prox = nn.Sequential(*layers)
        nn.init.zeros_(self.prox[-1].weight)
        nn.init.zeros_(self.prox[-1].bias)
        self.step_logits = nn.Parameter(torch.zeros(stages))

    def forward(self, measurement):
        x = self.physics.adjoint(measurement) / self.L
        for step in self.step_logits:
            x = x - 1.8*step.sigmoid()/self.L*self.physics.adjoint(self.physics(x)-measurement)
            x = x + .1*self.prox(x)
        x = x - self.physics.adjoint(self.physics(x)-measurement)/self.L
        projected = self.physics(x)
        return dict(spectral_cube_hat=x, reprojected_measurement=projected,
                    global_spectrum=x.mean((-2, -1)), residual=projected-measurement)


def reconstruction_loss(out, truth, measurement_scale):
    # Reuse the tested reconstruction-only D3 objective, not its task data loader.
    from .full_spectral import scene_loss
    return scene_loss(dict(spectral_map=out["spectral_cube_hat"],
                           residual=out["residual"]), truth, measurement_scale)
