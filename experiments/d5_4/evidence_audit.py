"""Recompute published SNR strata without loading data or model weights."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from experiments.d5_3.snr_strata import analyse


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(source):
    def read(name):
        return json.loads((source / name).read_text(encoding='utf-8'))
    snr = analyse(read('encoding_observability.json'))
    runs = read('s3_multiseed.json')['runs']
    response = read('reconstruction_response.json')
    return dict(status='published_evidence_recomputed_not_artifact_revalidated',
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                sources={p.name: digest(p) for p in source.glob('*.json')},
                snr=snr, data_and_kernel_digests={key:read('seed_protocol.json')[key]
                                               for key in ('data','arms')},
                recorded_checkpoint_provenance=response['arms'],
                achromatic550={k: dict(best_epoch=v['best_epoch'],
                    validation_present=(source / 'runs' / k / 'validation.json').exists(),
                    checkpoint_bytes_checked=False)
                    for k, v in runs.items() if k.startswith('s3a550_')},
                limitations=['No original data or checkpoint bytes checked by this audit.',
                            '675 cases are 75 configuration means across nine scenes.',
                            'Known-template detection is not blind reconstruction.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
