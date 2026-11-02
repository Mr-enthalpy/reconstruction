"""Verify published protocol, implementation and private sample bytes on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.d6_1.validate_groups import digest, load_development


def verify(report, manifest, expected_lock_sha256):
    lock_path=report/'protocol_lock.json'
    if digest(lock_path)!=expected_lock_sha256:
        raise ValueError('protocol lock changed')
    lock=json.loads(lock_path.read_bytes())
    for name,expected in lock['public_file_sha256'].items():
        if Path(name).name!=name or digest(report/name)!=expected:
            raise ValueError('frozen report changed')
    for name,expected in lock['implementation_sha256_lf'].items():
        if Path(name).name!=name:
            raise ValueError('invalid source name')
        actual=hashlib.sha256((Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if actual!=expected:
            raise ValueError('frozen implementation changed')
    return {role:len(load_development(manifest,lock_path,role,expected_lock_sha256=expected_lock_sha256))
            for role in ('development_train','development_val','development_test')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--lock-sha256',required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.report,a.manifest,a.lock_sha256)))


if __name__=='__main__':main()
