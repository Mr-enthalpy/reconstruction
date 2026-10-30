"""Explicit-path D5.4 diagnostic runner. No test or training entry point.

Each case retains its original scene; reducers must not pool scenes x seeds.
Inverse uses a known baseline and direction, but a zero initial coefficient map.
"""
import argparse
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from experiments.d5_4.evidence_audit import digest
from experiments.d5_4.natural_spectral_basis import load_split
from experiments.d5_4.natural_counterfactual import construct
from experiments.d5_4.restricted_inverse_operator import RestrictedOperator, solve, metrics, differential_metrics
from experiments.d5_4.oracle_detection import detect
from experiments.d5_3.encoding_observability import noise_statistics
from lcd_reconstruction.operators.psf_provenance import array_digest


def cases(cubes, records, basis, cfg):
    for record in sorted(records, key=lambda r:r['scene_id']):
        x = cubes[record['array_index']]
        for direction, mask, level in itertools.product(range(len(basis['directions'])),cfg['masks'],range(3)):
            v = np.asarray(basis['directions'][direction])
            plus, minus, z, summary = construct(x,v,basis['natural_shape_change_quantiles'][level],
                                               mask,basis,cfg['seed'])
            label = dict(scene_id=record['scene_id'],dataset_id=record['dataset_id'],
                         direction=direction,mask=mask,level=level,construction=summary)
            yield label, x, plus, minus, z, v


def run(args):
    cfg = json.loads(args.config.read_text())
    basis = json.loads(args.basis.read_text())
    protocol = json.loads(args.protocol.read_text())
    cubes, records = load_split(args.prepared,args.split,protocol['data'])
    if basis['provenance']['inputs'] != protocol['data']:
        raise ValueError('basis input provenance differs')
    if digest(args.kernel_file) != protocol['arms']['s1c'].get('kernel_file_sha256',
            'f07d9bbb7a4ff870433e8e72dcd19298d642e639c499afe1057a97eeb9206d4b'):
        raise ValueError('measured PSF file digest mismatch')
    with np.load(args.kernel_file) as archive:
        kernels = archive['kernels']
    if array_digest(kernels) != protocol['arms']['s1c']['kernel_array_sha256']:
        raise ValueError('PSF array digest mismatch')
    controls = dict(chromatic=kernels,achromatic_mean=np.repeat(kernels.mean(0,keepdims=True),21,0),
                    achromatic_550=np.repeat(kernels[10:11],21,0))
    models = []
    if args.mode == 'frozen':
        from experiments.d5_3.reconstruction_response import load_arm
        if not args.run_dirs:
            raise ValueError('frozen mode requires explicit run directories')
        expected_arms = json.loads(args.checkpoint_provenance.read_text())['arms']
        expected = {r['run_dir']:r for r in expected_arms}
        for directory in args.run_dirs:
            if directory.name not in expected or digest(directory/'best.pt') != expected[directory.name]['checkpoint_sha256']:
                raise ValueError('checkpoint not in frozen D5.3 registry')
            model, provenance = load_arm(directory,args.kernel_file.parent,args.device)
            models.append((model,provenance))
    rows=[]
    started=time.perf_counter()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for case_index,(label,x,plus,minus,z,v) in enumerate(cases(cubes,records,basis,cfg)):
        if args.max_cases is not None and case_index>=args.max_cases:
            break
        if args.mode == 'natural':
            rows.append(label)
            continue
        for arm,k in controls.items():
            if args.mode == 'frozen':
                break
            op=RestrictedOperator(k,v,x.shape[-2:],args.device)
            delta_y=op.forward(2*z)
            with torch.no_grad():
                px=op.physics(torch.as_tensor(plus[None],dtype=torch.float64,device=args.device))[0,0].cpu().numpy()
                mx=op.physics(torch.as_tensor(minus[None],dtype=torch.float64,device=args.device))[0,0].cpu().numpy()
            identity_error=float(np.linalg.norm((px-mx)-delta_y))
            relative_norm=float(np.linalg.norm(delta_y)/max(np.linalg.norm((px+mx)/2),1e-30))
            for fraction in cfg['noise_fractions']:
                sigma=fraction*cfg['reference_forward_rms']
                rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],case_index,int(fraction*1e6)]))
                # Same independent noise pair across arms; one shared absolute sigma.
                n1=rng.normal(0,sigma,z.shape); n2=rng.normal(0,sigma,z.shape)
                row=dict(**label,arm=arm,noise_fraction=fraction,sigma=sigma,
                         delta_measurement_norm=float(np.linalg.norm(delta_y)),
                         forward_difference_identity_error=identity_error,
                         relative_measurement_difference=relative_norm,
                         frequency_diagnostic='FFT of cropped output; not circular-operator singular values')
                row['snr']=noise_statistics(delta_y,sigma,z.shape) if sigma else dict(
                    per_pixel_snr=None,matched_filter_snr=None,reason='noiseless; SNR undefined')
                template=delta_y/2
                row['single_oracle']=detect(template,template+n1,sigma)
                row['paired_oracle']=detect(template,delta_y+n1-n2,sigma,2)
                for name,truth_amplitude in (('single_oracle',1.),('paired_oracle',2.)):
                    oracle=row[name]
                    oracle['true_amplitude']=truth_amplitude
                    if oracle['alpha_hat'] is not None:
                        oracle['amplitude_relative_error']=abs(oracle['alpha_hat']-truth_amplitude)/truth_amplitude
                        oracle['expected_amplitude_snr']=truth_amplitude/oracle['alpha_standard_error'] if sigma else None
                if sigma:
                    fy=np.fft.fftfreq(z.shape[0])[:,None]; fx=np.fft.fftfreq(z.shape[1])[None,:]
                    high=np.hypot(fy,fx)>=.5
                    spectrum=np.fft.fft2(delta_y)
                    energy=float(np.sum(abs(spectrum[high])**2))
                    noise_energy=float(2*sigma*sigma*z.size*high.sum())
                    row['snr']['radial']['0.5-corners']=dict(coefficient_count=int(high.sum()),
                        signal_energy=energy,expected_noise_energy=noise_energy,snr=float(np.sqrt(energy/noise_energy)))
                if args.mode=='inverse':
                    lam,mu=args.lam,args.mu
                    if args.regularisation_policy:
                        policy=json.loads(args.regularisation_policy.read_text())
                        selected=policy['by_noise_fraction'][str(fraction)]
                        lam,mu=selected['lam'],selected['mu']
                    if lam is None or mu is None:
                        raise ValueError('inverse requires explicit training-selected lambda and mu')
                    estimate,convergence=solve(op,template+n1,lam,mu,
                        cfg['inverse']['rtol'],args.maxiter or cfg['inverse']['maxiter'])
                    row['inverse']=dict(metrics=metrics(estimate,z),convergence=convergence,
                        prior='known true baseline and known spectral direction; unknown spatial map',
                        expected_noise_norm=float(sigma*np.sqrt(z.size)))
                rows.append(row)
        for model,provenance in models:
            with torch.no_grad():
                tx=torch.as_tensor(np.stack([plus,minus,x]),dtype=torch.float32,device=args.device)
                measurements=[model.physics(tx[i:i+1]) for i in range(3)]
                for fraction in cfg['noise_fractions']:
                    sigma=fraction*cfg['reference_forward_rms']
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],case_index,int(fraction*1e6)]))
                    noise=[torch.as_tensor(rng.normal(0,sigma,measurements[0].shape),
                                           dtype=torch.float32,device=args.device) for _ in range(2)]
                    outputs=[model(measurements[i]+noise[i])['spectral_cube_hat'][0].cpu().numpy()
                             for i in range(2)]
                    null=[model(measurements[2]+n)['spectral_cube_hat'][0].cpu().numpy() for n in noise]
                    delta=plus-minus
                    rows.append(dict(**label,model=provenance,noise_fraction=fraction,
                        response=differential_metrics(outputs[0]-outputs[1],delta,v),
                        noise_only=differential_metrics(null[0]-null[1],delta,v),
                        inference_noise_shift=fraction!=.005))
        print(json.dumps(dict(case=case_index+1,elapsed_seconds=time.perf_counter()-started)),flush=True)
    return dict(status='benchmark_only' if args.max_cases else 'completed',mode=args.mode,split=args.split,
        elapsed_seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        provenance=dict(code_commit=os.environ.get('D54_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_files={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
            config=cfg,config_sha256=digest(args.config),basis_sha256=digest(args.basis),
            regularisation_policy=None if not args.regularisation_policy else json.loads(args.regularisation_policy.read_text()),
            inputs=protocol['data'],kernel_file_sha256=digest(args.kernel_file),
            kernel_array_sha256=array_digest(kernels)),rows=rows)


def main():
    p=argparse.ArgumentParser()
    for name in ('prepared','kernel-file','basis','config','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['natural','operator','inverse','frozen'],required=True)
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-cases',type=int)
    p.add_argument('--maxiter',type=int)
    p.add_argument('--lam',type=float)
    p.add_argument('--mu',type=float)
    p.add_argument('--regularisation-policy',type=Path)
    p.add_argument('--run-dirs',type=Path,nargs='+')
    p.add_argument('--checkpoint-provenance',type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    result=run(a)
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
