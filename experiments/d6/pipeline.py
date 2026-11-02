"""Simulation-only, full-field frozen N1 and independent scalar-proxy R.

This adapter makes no Bayer/RGB recovery or detector accuracy claim. Candidate
boxes are supplied independently; labels are never accepted by either branch.
"""
import hashlib
import numpy as np
import torch


def scalar_proxy(measurement):
    """Repeat signed intensity without clipping, fitting or interpreting a CFA."""
    return measurement.detach().repeat(1, 3, 1, 1)


def run_pipeline(measurement, n1, boxes, *, observation_id, model_sha256,
                 candidate_source, classifier=None, classifier_id=None):
    if not observation_id or not candidate_source or not model_sha256:
        raise ValueError('observation, model and candidate provenance required')
    if measurement.ndim != 4 or measurement.shape[:2] != (1, 1):
        raise ValueError('one full-field scalar observation [1,1,H,W] required')
    if not measurement.is_floating_point() or not torch.isfinite(measurement).all():
        raise ValueError('finite floating-point scalar observation required')
    if n1.training or any(p.requires_grad for p in n1.parameters()):
        raise ValueError('N1 must be frozen and in evaluation mode')
    if classifier is not None and not classifier_id:
        raise ValueError('classifier provenance required')
    h, w = measurement.shape[-2:]
    raw_boxes = np.asarray(boxes)
    if raw_boxes.ndim != 2 or raw_boxes.shape[1] != 4:
        raise ValueError('candidate boxes must be [N,4] integer xywh')
    if not np.isfinite(raw_boxes).all() or not np.equal(raw_boxes, np.floor(raw_boxes)).all():
        raise ValueError('candidate boxes must be finite integers')
    boxes = raw_boxes.astype(np.int64)
    for x, y, bw, bh in boxes:
        if min(x, y) < 0 or min(bw, bh) <= 0 or x+bw > w or y+bh > h:
            raise ValueError('candidate outside full-field pixel coordinates')
    with torch.no_grad():
        # Independent branch: no reconstructed cube is an input to R.
        proxy = scalar_proxy(measurement)
        # Reconstruct the entire observation before any ROI slicing.
        out = n1(measurement)
        cube = out['spectral_cube_hat']
        projected = out['reprojected_measurement']
        if cube.shape != (1, 21, h, w) or projected.shape != measurement.shape:
            raise ValueError('N1 spatial/spectral output contract violated')
        if not torch.isfinite(cube).all() or not torch.isfinite(projected).all():
            raise ValueError('nonfinite N1 output')
        features = []
        for x, y, bw, bh in boxes:
            r = proxy[0, :, y:y+bh, x:x+bw]
            n = cube[0, :, y:y+bh, x:x+bw]
            features.append(torch.cat((r.mean((-2,-1)), n.mean((-2,-1)))))
        features = torch.stack(features) if features else cube.new_empty((0, 24))
        scores = None if classifier is None else classifier(features.detach())
        if scores is not None:
            if scores.ndim != 2 or scores.shape[0] != len(boxes) or not torch.isfinite(scores).all():
                raise ValueError('classifier must return finite [N,classes] scores')
            scores = scores.detach().cpu().numpy()
        residual = float(torch.linalg.vector_norm(projected-measurement) /
                         torch.linalg.vector_norm(measurement).clamp_min(1e-12))
    return dict(proxy=proxy.cpu().numpy(), cube=cube.cpu().numpy(),
                features=features.detach().cpu().numpy(), scores=scores, boxes=boxes,
                metadata=dict(status='completed', scope='simulation-only',
                    observation_id=observation_id, model_sha256=model_sha256,
                    observation_sha256=hashlib.sha256(measurement.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                    candidate_source=candidate_source, classifier_id=classifier_id,
                    classifier_status='not_run' if classifier is None else 'executed',
                    coordinates='full-field pixel xywh; half-open integer bounds; no resize',
                    r_input='same scalar observation; three identical signed intensity channels',
                    r_claim='not Bayer, calibrated RGB, deblurred RGB or independent spectral information',
                    feature_order='3 proxy channel means then 21 reconstructed band means',
                    measurement_relative_l2=residual))
