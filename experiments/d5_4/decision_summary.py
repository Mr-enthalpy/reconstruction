"""Evaluate complete strata against the recorded diagnostic recovery gate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from experiments.d5_4.evidence_audit import digest
from experiments.d5_4.basis_identity import verify_basis


def recovered(metric):
    return metric.get('relative_l2',float('inf'))<.5 and (metric.get('cosine') or -1)>.8


def group(rows,kind):
    groups=defaultdict(list)
    for row in rows:
        model=row.get('model',{})
        key=(row['direction'],row['mask'],row['level'],row['noise_fraction'],
             row.get('arm',model.get('arm')),model.get('seed'))
        groups[key].append(row)
    result=[]
    for key,items in sorted(groups.items(),key=lambda kv:str(kv[0])):
        if len(items)!=9 or len({r['scene_id'] for r in items})!=9:
            raise ValueError('incomplete scene stratum')
        cases=[]
        for r in items:
            if kind=='inverse':
                m=r['inverse']['metrics']; conv=r['inverse']['convergence']['converged']
                passed=bool(conv and recovered(m))
            elif kind=='frozen':
                m=r['response']; conv=None; passed=recovered(m)
            else:
                oracle=r['single_oracle']
                snr=oracle.get('expected_amplitude_snr')
                # Descriptive ideal known-template threshold, not a blind recovery gate.
                passed=oracle['status']=='estimated' and (r['noise_fraction']==0 or snr is not None and snr>=3)
                m=oracle; conv=None
            cases.append(dict(scene_id=r['scene_id'],dataset_id=r['dataset_id'],passed=passed,
                converged=conv,metrics=m))
        result.append(dict(direction=key[0],mask=key[1],level=key[2],noise_fraction=key[3],
            arm=key[4],model_seed=key[5],passed_scenes=sum(c['passed'] for c in cases),
            gate_passed=sum(c['passed'] for c in cases)>=8,cases=cases))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('operator','inverse','frozen','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--basis',type=Path)
    a=p.parse_args()
    docs={k:json.loads(getattr(a,k).read_text()) for k in ('operator','inverse','frozen')}
    if any(d['status']!='completed' for d in docs.values()):
        raise ValueError('benchmark or incomplete run cannot support final decision')
    basis_proof=None
    if a.basis:
        basis_proof=verify_basis(a.basis,[d['provenance']['basis_sha256'] for d in docs.values()])
    for field in ('kernel_array_sha256',) if a.basis else ('kernel_array_sha256','basis_sha256'):
        if len({d['provenance'][field] for d in docs.values()})!=1:
            raise ValueError('cross-run physical/basis provenance mismatch')
    result=dict(status='completed_diagnostic_gate_evaluation',
        basis_transport_proof=basis_proof,
        source_sha256={k:digest(getattr(a,k)) for k in docs},
        operator=group(docs['operator']['rows'],'operator'),
        inverse=group(docs['inverse']['rows'],'inverse'),
        frozen=group(docs['frozen']['rows'],'frozen'),
        interpretation='Gate applies to complete strata of nine original scenes. Model seeds remain distinct. '
            'A failed inverse gate is not a proof of physical impossibility. Known-template detection is privileged.')
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__': main()
