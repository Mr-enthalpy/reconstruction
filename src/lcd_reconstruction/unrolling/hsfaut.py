"""HSFAU-style full-state unfolding for D5, without task or classifier interfaces."""
import torch
from torch import nn

from ..priors.sfat import SFAT
from .filter_split import FilterSplitPhysics


class CNNProx(nn.Module):
    def __init__(self, channels=21, width=64, depth=4):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(width, channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class HSFAUReconstructor(nn.Module):
    def __init__(self, kernels, hw, stages=3, width=64, depth=4,
                 prior="sfat", sfat_width=32, sfat_levels=3):
        super().__init__()
        if stages < 1:
            raise ValueError("positive stages required")
        self.physics = FilterSplitPhysics(kernels, hw)
        self.register_buffer("L", self.physics.lipschitz_bound().clamp_min(1e-8))
        if prior == "sfat":
            self.prox = SFAT(21, sfat_width, sfat_levels)
        elif prior == "cnn":
            self.prox = CNNProx(21, width, depth)
        else:
            raise ValueError("prior must be cnn or sfat")
        self.step_logits = nn.Parameter(torch.zeros(stages))

    def forward(self, measurement):
        x = self.physics.adjoint(measurement) / self.L
        diagnostics = []
        for step in self.step_logits:
            residual = self.physics(x) - measurement
            diagnostics.append(float(residual.square().mean().detach()))
            x = x - 1.8 * step.sigmoid() / self.L * self.physics.adjoint(residual)
            x = x + 0.1 * self.prox(x)
        x = x - self.physics.adjoint(self.physics(x) - measurement) / self.L
        projected = self.physics(x)
        return dict(spectral_cube_hat=x, reprojected_measurement=projected,
                    global_spectrum=x.mean((-2, -1)), residual=projected - measurement,
                    stage_residuals=diagnostics)


def reconstruction_loss(out, truth, measurement_scale, lambda_grad=0.2,
                        lambda_spectral=0.05, lambda_data=0.02, epsilon=1e-3):
    x = out["spectral_cube_hat"]
    charbonnier = (torch.sqrt((x - truth).square() + epsilon ** 2) - epsilon).mean()
    spatial = sum((torch.diff(x, dim=d) - torch.diff(truth, dim=d)).square().mean()
                  for d in (-1, -2)) / 2
    valid = torch.linalg.vector_norm(truth, dim=1) > 1e-3
    cosine = torch.nn.functional.cosine_similarity(x, truth, dim=1, eps=1e-8)
    spectral = (1 - cosine[valid]).mean() if valid.any() else charbonnier * 0
    data = out["residual"].square().mean() / measurement_scale ** 2
    total = charbonnier + lambda_grad * spatial + lambda_spectral * spectral + lambda_data * data
    return total, dict(total=float(total.detach()), charbonnier=float(charbonnier.detach()),
                       spatial=float(spatial.detach()), spectral=float(spectral.detach()),
                       data=float(data.detach()))
