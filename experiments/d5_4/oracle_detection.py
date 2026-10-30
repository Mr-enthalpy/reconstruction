"""Known baseline AND known template scalar least squares; not blind imaging."""
import numpy as np


def detect(template, residual, sigma, observations=1, energy_floor=1e-20):
    t = np.asarray(template, dtype=np.float64).ravel()
    r = np.asarray(residual, dtype=np.float64).ravel()
    if observations not in (1,2) or sigma < 0 or t.shape != r.shape:
        raise ValueError('invalid observation or noise contract')
    energy = float(t@t)
    if not np.isfinite(energy) or energy <= energy_floor:
        return dict(status='unidentifiable', template_energy=energy, alpha_hat=None)
    return dict(status='estimated', alpha_hat=float(t@r/energy), template_energy=energy,
                alpha_standard_error=float(sigma*np.sqrt(observations/energy)),
                observations=observations,
                convention='paired residual estimates amplitude difference, not half amplitude')
