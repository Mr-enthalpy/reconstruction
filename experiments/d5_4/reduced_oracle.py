"""Conditional baseline-prior removal; includes a privileged clean reference exposure.

Only the previous full-scene gate selects conditions. Never select a model seed,
mask, regularizer or direction by this diagnostic's outcomes.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from experiments.d5_4.evidence_audit import digest
from experiments.d5_4.decision_summary import group
from experiments.d5_4.natural_spectral_basis import load_split
from experiments.d5_4.noise_sweep import cases
from experiments.d5_4.restricted_inverse_operator import RestrictedOperator, solve, metrics
from experiments.d5_3.reconstruction_response import load_arm
from lcd_reconstruction.operators.psf_provenance import array_digest


def passing_strata(inverse):
    if inverse['status']!='completed':
        raise ValueError('completed full inverse required')
    return {(r['direction'],r['mask'],r['level'],r['noise_fraction'])
        for r in group(inverse['rows'],'inverse')
        if r['arm']=='chromatic' and r['gate_passed']}


def main():
    p=argparse.ArgumentParser()
    for name in ('prepared','kernel-file','basis','config','protocol','inverse',
                 'regularisation-policy','checkpoint-provenance','run-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda'); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(4)
    source={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    inverse=json.loads(a.inverse.read_text()); selected=passing_strata(inverse)
    cfg=json.loads(a.config.read_text()); basis=json.loads(a.basis.read_text())
    protocol=json.loads(a.protocol.read_text()); policy=json.loads(a.regularisation_policy.read_text())
    if digest(a.basis)!=inverse['provenance']['basis_sha256']:
        raise ValueError('basis differs from completed inverse')
    if policy!=inverse['provenance']['regularisation_policy']:
        raise ValueError('regularization changed after gate')
    registry=json.loads(a.checkpoint_provenance.read_text())['arms']
    expected=next(r for r in registry if r['run_dir']=='s1c_s20261021')
    if a.run_dir.name!=expected['run_dir'] or digest(a.run_dir/'best.pt')!=expected['checkpoint_sha256']:
        raise ValueError('fixed model seed/checkpoint mismatch')
    model,model_provenance=load_arm(a.run_dir,a.kernel_file.parent,a.device)
    kernels=np.load(a.kernel_file)['kernels']
    if array_digest(kernels)!=inverse['provenance']['kernel_array_sha256']:
        raise ValueError('physical kernel differs')
    cubes,records=load_split(a.prepared,'val',protocol['data'])
    rows=[]; started=time.perf_counter(); baseline_cache={}
    for index,(label,x,plus,minus,z,v) in enumerate(cases(cubes,records,basis,cfg)):
        noises=[f for f in cfg['noise_fractions'] if (label['direction'],label['mask'],label['level'],f) in selected]
        if not noises: continue
        op=RestrictedOperator(kernels,v,x.shape[-2:],a.device)
        scene=label['scene_id']
        if scene not in baseline_cache:
            with torch.no_grad():
                clean_reference=model.physics(torch.as_tensor(x[None],dtype=torch.float32,device=a.device))
                base=model(clean_reference)['spectral_cube_hat'][0].cpu().numpy().astype(np.float64)
                bias=op.physics(torch.as_tensor((x-base)[None],dtype=torch.float64,device=a.device))[0,0].cpu().numpy()
            baseline_cache[scene]=(bias,metrics(base,x))
        bias,baseline_metrics=baseline_cache[scene]
        for fraction in noises:
            sigma=fraction*cfg['reference_forward_rms']
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],index,int(fraction*1e6)]))
            # Algebraic y_plus - A(base); X0 is used only by the simulation and scoring.
            residual=bias+op.forward(z)+rng.normal(0,sigma,z.shape)
            params=policy['by_noise_fraction'][str(fraction)]
            estimate,convergence=solve(op,residual,params['lam'],params['mu'],
                cfg['inverse']['rtol'],cfg['inverse']['maxiter'])
            rows.append(dict(**label,noise_fraction=fraction,arm='chromatic',
                correction_metrics=metrics(estimate,z),convergence=convergence,
                baseline_cube_metrics=baseline_metrics,baseline_measurement_bias_norm=float(np.linalg.norm(bias)),
                true_perturbation_measurement_norm=float(np.linalg.norm(op.forward(z))),
                expected_noise_norm=float(sigma*np.sqrt(z.size))))
        print(json.dumps(dict(case=index+1,completed=len(rows),elapsed_seconds=time.perf_counter()-started)),flush=True)
    result=dict(status='completed' if selected else 'not_executed_gate_not_met',
        prior='known training direction; A6 baseline from extra clean reference exposure; no GT cube or mask given to solver',
        observation_count=2,reference_noise_fraction=0,
        limitation='Privileged two-exposure differential diagnostic, not deployable single-shot reconstruction. Baseline-estimation error can dominate the correction.',
        selected_strata=[list(k) for k in sorted(selected)],rows=rows,elapsed_seconds=time.perf_counter()-started,
        provenance=dict(code_commit=commit,source_files=source,inputs=protocol['data'],
            inverse_sha256=digest(a.inverse),basis_sha256=digest(a.basis),kernel_sha256=digest(a.kernel_file),
            config=cfg,regularisation_policy=policy,model=model_provenance))
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,separators=(',',':'),allow_nan=False)


if __name__=='__main__': main()
