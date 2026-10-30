"""Finite, prespecified training-only regularisation sweep."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from experiments.d5_4.natural_spectral_basis import load_split
from experiments.d5_4.natural_counterfactual import construct
from experiments.d5_4.restricted_inverse_operator import RestrictedOperator, solve, metrics
from experiments.d5_4.evidence_audit import digest
from lcd_reconstruction.operators.psf_provenance import array_digest


def main():
    p=argparse.ArgumentParser()
    for name in ('prepared','kernel-file','basis','config','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--noise-fraction',type=float,choices=[0.,.001,.005],default=.005)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(4)
    cfg=json.loads(a.config.read_text()); basis=json.loads(a.basis.read_text())
    protocol=json.loads(a.protocol.read_text())
    cubes,records=load_split(a.prepared,'train',protocol['data'])
    kernels=np.load(a.kernel_file)['kernels']
    if array_digest(kernels)!=protocol['arms']['s1c']['kernel_array_sha256']:
        raise ValueError('PSF provenance mismatch')
    rows=[]
    for record in sorted(records,key=lambda r:r['scene_id'])[:3]:
        x=cubes[record['array_index']]
        for index,v in enumerate(basis['directions']):
            _,_,z,_=construct(x,v,basis['natural_shape_change_quantiles'][1],'local',basis,cfg['seed'])
            b=RestrictedOperator(kernels,v,x.shape[-2:],a.device)
            rng=np.random.default_rng(cfg['seed']+index)
            residual=b.forward(z)+rng.normal(0,a.noise_fraction*cfg['reference_forward_rms'],z.shape)
            for lam,mu in itertools.product(cfg['inverse']['lambda_grid'],cfg['inverse']['mu_grid']):
                estimate,report=solve(b,residual,lam,mu,cfg['inverse']['rtol'],cfg['inverse']['maxiter'])
                rows.append(dict(scene_id=record['scene_id'],direction=index,lam=lam,mu=mu,
                    metrics=metrics(estimate,z),solver=report))
                print(json.dumps(dict(completed=len(rows))),flush=True)
    candidates=[]
    for lam,mu in itertools.product(cfg['inverse']['lambda_grid'],cfg['inverse']['mu_grid']):
        subset=[r for r in rows if r['lam']==lam and r['mu']==mu]
        candidates.append(dict(lam=lam,mu=mu,mean_error=float(np.mean([r['metrics']['relative_l2'] for r in subset])),
            converged_cases=sum(r['solver']['converged'] for r in subset)))
    result=dict(status='completed',noise_fraction=a.noise_fraction,selection=min(candidates,key=lambda r:r['mean_error']),
        candidates=candidates,rows=rows,config_sha256=digest(a.config),basis_sha256=digest(a.basis),
        inputs=protocol['data'],kernel_sha256=digest(a.kernel_file),
        caveat='Fixed-iteration estimator selected on training only; nonconvergence is retained.')
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__': main()
