"""Bound the shared spatial coefficient, never clip individual bands."""
import numpy as np
from scipy.ndimage import gaussian_filter


def construct(cube, direction, scale, mask_name, basis, seed=20261030):
    x = np.asarray(cube, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    if x.shape[0] != 21 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('finite unit-range cube required')
    if abs(v.sum()) > 1e-12 or abs(np.linalg.norm(v)-1) > 1e-10 or scale <= 0:
        raise ValueError('normalised zero-sum direction and positive scale required')
    h, w = x.shape[-2:]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    if mask_name == 'smooth':
        mask = gaussian_filter(rng.normal(size=(h, w)), max(h, w)/32)
    elif mask_name == 'texture':
        mask = gaussian_filter(x.sum(0), 2)
        mask -= mask.mean()
    elif mask_name == 'local':
        cy, cx = rng.uniform(.25, .75, 2) * [h, w]
        mask = np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*(min(h,w)/8)**2))
    else:
        raise ValueError('unknown spatial mask')
    mask /= max(np.max(abs(mask)), 1e-30)
    # scale refers to relative spectral shape, not an arbitrary raw alpha.
    proposed = scale * x.sum(0) * mask
    active = abs(v) > 0
    bound = (np.minimum(x[active], 1-x[active])/abs(v[active,None,None])).min(0)
    z = np.sign(proposed) * np.minimum(abs(proposed), bound*(1-1e-12))
    delta = v[:,None,None] * z
    plus, minus = x+delta, x-delta
    total = x.sum(0)
    valid = total > .021
    shape = np.concatenate([a[:,valid].T / total[valid,None] for a in (plus, minus)])
    directions = np.asarray(basis['directions'])
    scores = (shape-np.asarray(basis['mean_shape'])) @ directions.T
    radius = np.linalg.norm(scores/np.asarray(basis['score_std']), axis=1)
    bounds = np.asarray(basis['shape_bounds'])
    outside = np.any((shape < bounds[0]) | (shape > bounds[1]), axis=1)
    out_fraction = float(outside.mean()) if outside.size else 1.
    rel = float(np.linalg.norm(delta)/max(np.linalg.norm(x),1e-30))
    relative_pixel = np.linalg.norm(delta[:,valid],axis=0)/np.maximum(np.linalg.norm(x[:,valid],axis=0),1e-30)
    shape_change = np.linalg.norm(delta[:,valid],axis=0)/total[valid]
    cosine = (plus[:,valid]*minus[:,valid]).sum(0)/np.maximum(
        np.linalg.norm(plus[:,valid],axis=0)*np.linalg.norm(minus[:,valid],axis=0),1e-30)
    sam = np.degrees(np.arccos(np.clip(cosine,-1,1)))
    q = lambda a: np.quantile(a,[0,.25,.5,.75,1]).tolist() if a.size else None
    summary = dict(relative_perturbation=rel, nonzero_area=float((abs(z)>1e-12).mean()),
        shrunk_fraction=float((abs(z)<abs(proposed)*(1-1e-10)).mean()),
        broadband_error=float(abs(plus.sum(0)-minus.sum(0)).max()),
        pixel_relative_perturbation_quantiles=q(relative_pixel),
        pair_sam_degrees_quantiles=q(sam),
        natural_shape_percentile_quantiles=q(np.searchsorted(basis['natural_shape_change_reference'],shape_change)/101),
        natural_pair_sam_percentile_quantiles=q(np.searchsorted(basis['natural_sam_reference'],sam)/101),
        valid_pixel_fraction=float(valid.mean()),
        boundary_pixel_fraction=float(((plus.min(0)<1e-6)|(minus.min(0)<1e-6)|
                                      (plus.max(0)>1-1e-6)|(minus.max(0)>1-1e-6)).mean()),
        shape_outside_fraction=out_fraction,
        pca_outside_99_fraction=float((radius>basis['score_radius_quantiles'][1]).mean()) if radius.size else 1.,
        status='constructible' if rel>1e-8 else 'unconstructible',
        distribution_label='stress-test / clearly synthetic')
    # Conservative diagnostic labels; never equate PCA origin with in-distribution.
    if out_fraction <= .05 and summary['pca_outside_99_fraction'] <= .05:
        summary['distribution_label'] = 'training-statistics-compatible'
    elif out_fraction <= .2 and summary['pca_outside_99_fraction'] <= .2:
        summary['distribution_label'] = 'near-distribution'
    return plus, minus, z, summary
