"""Lossless inference-state export, with exact reconstruction of physics buffers.

Original checkpoints remain immutable. The derived bundle omits optimizer state
and redundant FFT buffers; every model state tensor must reproduce its original
digest before inference. A mismatched buffer is an error, never a tolerated shift.
"""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from experiments.d5_3.reconstruction_response import build_model, load_arm
from lcd_reconstruction.operators.psf_provenance import (
    file_digest, array_digest, tensor_digest, is_physics_key,
    physics_buffers_from_model, resolve_psf_policy, PSFProvenanceError)


def state_digests(state):
    return {k: tensor_digest(v) for k, v in state.items()}


def restore(item, device='cpu'):
    kernels = item['kernels'].numpy()
    model = build_model(item['config'], kernels, tuple(item['hw']))
    resolve_psf_policy(item['psf_provenance'], physics_buffers_from_model(model),
                       item['psf_provenance']['physics_buffers'], split='test',
                       requested_array=array_digest(kernels))
    state = model.state_dict()
    expected_parameters = {k for k in state if not is_physics_key(k)}
    if set(item['parameters']) != expected_parameters:
        raise PSFProvenanceError('incomplete or unexpected inference parameters')
    state.update(item['parameters'])
    model.load_state_dict(state, strict=True)
    if state_digests(model.state_dict()) != item['original_state_digests']:
        raise PSFProvenanceError('derived inference state differs from original checkpoint')
    return model.to(device).eval().requires_grad_(False)


def export(runs, kernel_dir, registry):
    items = []
    for expected in registry['arms']:
        directory = runs / expected['run_dir']
        if file_digest(directory / 'best.pt') != expected['checkpoint_sha256']:
            raise ValueError('original checkpoint hash differs from frozen registry')
        model, metadata = load_arm(directory, kernel_dir, 'cpu')
        if metadata != expected:
            raise ValueError('original checkpoint provenance differs from frozen registry')
        checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        with np.load(kernel_dir / metadata['kernel_file'], allow_pickle=False) as a:
            kernels = np.array(a['kernels'], copy=True)
        state = model.state_dict()
        item = dict(metadata=metadata, config=checkpoint['config'], hw=checkpoint['hw'],
                    psf_provenance=checkpoint['psf_provenance'],
                    original_state_digests=state_digests(state),
                    kernels=torch.from_numpy(kernels),
                    parameters={k: v.clone() for k, v in state.items() if not is_physics_key(k)})
        restore(item)  # Complete exact-state equality, including regenerated FFT/L.
        items.append(item)
    return dict(format='d6-lossless-inference-v1', models=items,
                note='Original checkpoint SHA256 is distinct from this derived bundle SHA256.')


def main():
    p = argparse.ArgumentParser()
    for key in ('runs', 'kernel-dir', 'registry', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    result = export(a.runs, a.kernel_dir, json.loads(a.registry.read_bytes()))
    with a.output.open('xb') as f:
        torch.save(result, f)
    print(json.dumps(dict(models=len(result['models']), sha256=file_digest(a.output),
                         bytes=a.output.stat().st_size)))


if __name__ == '__main__':
    main()
