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


def restore(item, device='cpu', original_physics=None):
    kernels = item['kernels'].numpy()
    model = build_model(item['config'], kernels, tuple(item['hw']))
    if array_digest(kernels) != item['psf_provenance']['kernel_sha256_array']:
        raise PSFProvenanceError('kernel array differs from original checkpoint')
    if original_physics is not None:
        # Cross-platform FFT initialization need not be byte-identical. Restore
        # the ORIGINAL stored buffers, never an approximate replacement. Their
        # digests, shape and every final state tensor are checked below.
        state = model.state_dict()
        expected = {k for k in state if is_physics_key(k)}
        if set(original_physics) != expected:
            raise PSFProvenanceError('incomplete original physics buffers')
        for k, value in original_physics.items():
            restored = value.expand_as(state[k]).clone() if value.shape != state[k].shape else value
            if tensor_digest(restored) != item['original_state_digests'][k]:
                raise PSFProvenanceError('original physics buffer digest differs')
            state[k] = restored
        model.load_state_dict(state, strict=True)
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


def export(runs, kernel_dir, registry, physics=None):
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
        if physics is not None:
            buffers = {}
            for key, value in state.items():
                if not is_physics_key(key):continue
                # Exact repeated channels in achromatic arms are stored once.
                if value.ndim > 0 and value.shape[0] == 21 and torch.equal(value,value[:1].expand_as(value)):
                    value=value[:1].clone()
                buffers[key]=value.clone()
            if metadata['arm'] in physics:
                if state_digests(physics[metadata['arm']]) != state_digests(buffers):
                    raise PSFProvenanceError('same-arm physics differs across model seeds')
            else:physics[metadata['arm']]=buffers
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
    p.add_argument('--physics-output',type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.physics_output and a.physics_output.exists():raise FileExistsError(a.physics_output)
    torch.set_num_threads(2)
    physics={} if a.physics_output else None
    result = export(a.runs, a.kernel_dir, json.loads(a.registry.read_bytes()),physics)
    with a.output.open('xb') as f:
        torch.save(result, f)
    print(json.dumps(dict(models=len(result['models']), sha256=file_digest(a.output),
                         bytes=a.output.stat().st_size)))
    if a.physics_output:
        with a.physics_output.open('xb') as f:torch.save(physics,f)
        print(json.dumps(dict(physics_sha256=file_digest(a.physics_output),bytes=a.physics_output.stat().st_size)))


if __name__ == '__main__':
    main()
