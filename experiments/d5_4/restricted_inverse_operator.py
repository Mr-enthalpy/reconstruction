"""Oracle coefficient-map inverse through the existing zero-boundary physics."""
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics


class RestrictedOperator:
    def __init__(self, kernels, direction, hw, device='cpu'):
        self.hw = tuple(hw)
        self.device = device
        effective = np.einsum('b,bhw->hw', np.asarray(direction), np.asarray(kernels))
        self.unobservable = np.linalg.norm(effective) <= 1e-12 * np.linalg.norm(kernels)
        self.v = torch.as_tensor(direction, dtype=torch.float64, device=device)
        if self.v.shape != (21,) or not torch.isfinite(self.v).all():
            raise ValueError('finite 21-band direction required')
        self.physics = SinglePlanePhysics(torch.as_tensor(kernels, dtype=torch.float64,
                                                         device=device), self.hw)

    def forward(self, z):
        with torch.no_grad():
            z = torch.as_tensor(np.asarray(z).reshape(self.hw), dtype=torch.float64, device=self.device)
            return self.physics((self.v[:,None,None]*z)[None])[0,0].cpu().numpy()

    def adjoint(self, residual):
        with torch.no_grad():
            r = torch.as_tensor(np.asarray(residual).reshape(self.hw), dtype=torch.float64, device=self.device)
            back = self.physics.adjoint(r[None,None])[0]
            return (back*self.v[:,None,None]).sum(0).cpu().numpy()


def grad_normal(z):
    """D*D for forward differences without periodic wrap (Neumann boundary)."""
    result = np.zeros_like(z)
    dy, dx = np.diff(z, axis=0), np.diff(z, axis=1)
    result[:-1] -= dy
    result[1:] += dy
    result[:,:-1] -= dx
    result[:,1:] += dx
    return result


def solve(operator, residual, lam, mu, rtol=1e-6, maxiter=300):
    if lam < 0 or mu <= 0:
        raise ValueError('nonnegative gradient and positive ridge regularisation required')
    if operator.unobservable:
        return np.zeros(operator.hw), dict(converged=False, status='unobservable',
            cg_info=None, iterations=0, normal_residual_history=[],lambda_grad=lam,mu=mu,
            measurement_residual=float(np.linalg.norm(residual)))
    size = int(np.prod(operator.hw))
    def normal(flat):
        z = flat.reshape(operator.hw)
        return (operator.adjoint(operator.forward(z))+lam*grad_normal(z)+mu*z).ravel()
    rhs = operator.adjoint(residual).ravel()
    history = []
    matrix = LinearOperator((size,size), matvec=normal, dtype=np.float64)
    z, info = cg(matrix, rhs, x0=np.zeros(size), rtol=rtol, atol=0., maxiter=maxiter,
                 callback=lambda x: history.append(float(np.linalg.norm(normal(x)-rhs))))
    z = z.reshape(operator.hw)
    return z, dict(converged=info == 0, cg_info=int(info), iterations=len(history),
                   normal_residual_history=history, lambda_grad=lam, mu=mu,
                   measurement_residual=float(np.linalg.norm(operator.forward(z)-residual)),
                   initialization='zero; no GT coefficients supplied')


def metrics(estimate, truth):
    e, t = np.asarray(estimate).ravel(), np.asarray(truth).ravel()
    norm = np.linalg.norm(t)
    if norm < 1e-14:
        return dict(status='zero_truth_unidentifiable')
    en = np.linalg.norm(e)
    return dict(relative_l2=float(np.linalg.norm(e-t)/norm),
                improvement_over_zero=float(1-np.linalg.norm(e-t)/norm),
                response_fraction=float(en/norm),
                cosine=float(e@t/(en*norm)) if en>1e-14 else None,
                projection_gain=float(e@t/(norm*norm)),
                correlation=float(np.corrcoef(e,t)[0,1]) if e.std()>1e-14 and t.std()>1e-14 else None)


def differential_metrics(estimate, truth, direction):
    result=metrics(estimate,truth)
    v=np.asarray(direction)
    ep=np.einsum('b,bhw->hw',v,estimate)
    tp=np.einsum('b,bhw->hw',v,truth)
    affected=abs(tp)>.1*np.max(abs(tp))
    result['direction_projection']=metrics(ep,tp)
    result['correct_direction_fraction']=float((np.sign(ep[affected])==np.sign(tp[affected])).mean()) if affected.any() else None
    result['local_relative_l2']=float(np.linalg.norm((estimate-truth)[:,affected])/
        np.linalg.norm(truth[:,affected])) if affected.any() else None
    return result
