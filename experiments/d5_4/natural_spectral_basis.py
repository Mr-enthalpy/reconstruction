"""Scene-balanced training-only spectral statistics; never opens test data."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.d5_4.evidence_audit import digest


def zero_sum(v, weights=None):
    v = np.asarray(v, dtype=np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != v.shape or not np.isfinite(w).all() or np.dot(w, w) == 0:
        raise ValueError('invalid spectral weights')
    v = v - w * (np.dot(w, v) / np.dot(w, w))
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('degenerate zero-sum direction')
    v /= norm
    if v[np.argmax(abs(v))] < 0:
        v = -v
    return v


def load_split(prepared, split, expected):
    if split not in ('train', 'val'):
        raise ValueError('test access forbidden')
    prepared = Path(prepared)
    name = split + '.npz'
    for filename, sha in ((name, expected[split + '_npz_sha256']),
                          ('manifest.json', expected['manifest_sha256'])):
        if digest(prepared / filename) != sha:
            raise ValueError('D5.3 provenance mismatch: ' + filename)
    records = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    records = [r for r in records if r['split'] == split]
    if len({r['scene_id'] for r in records}) != len(records):
        raise ValueError('expected unique original scenes, not augmentations')
    with np.load(prepared / name, allow_pickle=False) as archive:
        cubes = archive['cube']
    if cubes.shape != (39 if split == 'train' else 9, 21, 512, 512):
        raise ValueError('unexpected data shape')
    if len(records) != len(cubes):
        raise ValueError('manifest count mismatch')
    return cubes, records


def fit(cubes, records, config):
    if any(r['split'] != 'train' for r in records):
        raise ValueError('basis fitting accepts training scenes only')
    rng = np.random.default_rng(config['seed'])
    samples, changes, angles, counts = [], [], [], {}
    for r in sorted(records, key=lambda r: r['scene_id']):
        cube = np.asarray(cubes[r['array_index']], dtype=np.float64)
        total = cube.sum(0)
        valid = np.isfinite(cube).all(0) & (cube.min(0) >= 0) & (cube.max(0) <= 1)
        valid &= total > config['minimum_band_sum']
        # Neighbour pairs quantify local natural shape variation; no val fitting.
        pair = valid[:, :-1] & valid[:, 1:]
        indices = np.flatnonzero(pair)
        if len(indices) < config['pixels_per_scene']:
            raise ValueError('insufficient valid samples in ' + r['scene_id'])
        chosen = rng.choice(indices, config['pixels_per_scene'], replace=False)
        shapes = cube / np.maximum(total, 1e-30)
        a = shapes[:, :, :-1].reshape(21, -1)[:, chosen].T
        b = shapes[:, :, 1:].reshape(21, -1)[:, chosen].T
        samples.append(a)
        changes.append(np.linalg.norm(a - b, axis=1))
        angles.append(np.degrees(np.arccos(np.clip((a*b).sum(1)/
            (np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)), -1, 1))))
        counts[r['scene_id']] = len(chosen)
    x = np.concatenate(samples)
    center = x.mean(0)
    _, singular, vt = np.linalg.svd(x - center, full_matrices=False)
    directions = np.stack([zero_sum(v) for v in vt[:config['basis_directions']]])
    scores = (x - center) @ directions.T
    std = scores.std(0).clip(1e-12)
    radius = np.linalg.norm(scores / std, axis=1)
    delta = np.concatenate(changes)
    return dict(status='completed_training_statistics', seed=config['seed'],
                fitting_parameters={k:config[k] for k in ('seed','pixels_per_scene','basis_directions',
                    'minimum_band_sum','amplitude_quantiles')},
                sampling=counts, directions=directions.tolist(), mean_shape=center.tolist(),
                score_std=std.tolist(), score_radius_quantiles=np.quantile(radius, [.95, .99]).tolist(),
                shape_bounds=np.quantile(x, [.005, .995], axis=0).tolist(),
                natural_shape_change_quantiles=np.quantile(delta, config['amplitude_quantiles']).tolist(),
                natural_shape_change_reference=np.quantile(delta, np.linspace(0, 1, 101)).tolist(),
                natural_sam_reference=np.quantile(np.concatenate(angles), np.linspace(0,1,101)).tolist(),
                singular_values=singular.tolist(),
                caveat='PCA provenance does not prove distribution membership')


def main():
    p = argparse.ArgumentParser()
    for name in ('prepared', 'config', 'protocol', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    expected = json.loads(a.protocol.read_text())['data']
    cubes, records = load_split(a.prepared, 'train', expected)
    result = fit(cubes, records, cfg)
    result['provenance'] = dict(inputs=expected, config_sha256=digest(a.config))
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
