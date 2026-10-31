"""Thin sequential CLI around existing diagnostics; no scheduler or resource API."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.d5_4.evidence_audit import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('prepared','runs','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--stage',choices=['benchmark','operator','tune','inverse','frozen','reduced','reduced-benchmark'],required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.environ['PYTHONPATH']=str(root/'src')+os.pathsep+str(root)
    os.environ['D54_CODE_COMMIT']=a.code_commit
    os.environ['OPENBLAS_NUM_THREADS']='4'
    os.environ['OMP_NUM_THREADS']='4'
    a.output.mkdir(parents=True,exist_ok=True)
    report=root/'docs/reports/2026-10-30-d5.4'
    old=root/'docs/reports/2026-10-29-d5.3'
    cfg=root/'experiments/d5_4/configs/diagnostics.json'
    common=['--prepared',str(a.prepared),'--kernel-file',str(a.prepared/'kernels.npz'),
            '--basis',str(report/'natural_spectral_basis.json'),'--config',str(cfg),
            '--protocol',str(old/'seed_protocol.json'),'--device',a.device]
    def launch(module,extra):
        subprocess.run([sys.executable,'-X','utf8','-m','experiments.d5_4.'+module,*common,*extra],check=True)
    if a.stage in ('reduced','reduced-benchmark'):
        options=['--max-cases','1'] if a.stage=='reduced-benchmark' else []
        launch('reduced_oracle',['--inverse',str(report/'restricted_inverse_results.json'),
            '--regularisation-policy',str(report/'regularisation_policy.json'),
            '--checkpoint-provenance',str(report/'checkpoint_registry.json'),
            '--run-dir',str(a.runs/'s1c_s20261021'),
            '--shard-index',str(a.shard_index),'--shard-count',str(a.shard_count),
            '--output',str(a.output/f'{a.stage}.json'),*options])
    elif a.stage in ('benchmark','frozen'):
        # Verify all old registered hashes, then validate physics buffers through load_arm.
        sys.path.insert(0,str(root/'src'))
        from experiments.d5_3.reconstruction_response import load_arm
        expected={r['run_dir']:r for r in json.loads((old/'reconstruction_response.json').read_text())['arms']}
        registry=[]
        for arm in ('s1c','s3a','s3a550'):
            for seed in (20261021,20261022,20261023):
                directory=a.runs/f'{arm}_s{seed}'
                if directory.name in expected and digest(directory/'best.pt')!=expected[directory.name]['checkpoint_sha256']:
                    raise ValueError('published checkpoint digest mismatch')
                model,meta=load_arm(directory,a.prepared,'cpu')
                validation=json.loads((old/'runs'/directory.name/'validation.json').read_text())
                if meta['best_epoch']!=validation['best_epoch']:
                    raise ValueError('checkpoint selection epoch differs from published report')
                del model
                registry.append(meta)
        registry_path=a.output/'checkpoint_registry.json'
        if registry_path.exists():
            if json.loads(registry_path.read_text())['arms']!=registry:
                raise ValueError('checkpoint registry changed')
        else:
            registry_path.write_text(json.dumps(dict(arms=registry,code_commit=a.code_commit),indent=2))
        suffix=['--max-cases','1'] if a.stage=='benchmark' else []
        launch('noise_sweep',['--mode','frozen','--checkpoint-provenance',str(registry_path),
            '--output',str(a.output/f'{a.stage}_a6.json'),*suffix,
            '--run-dirs',*[str(a.runs/r['run_dir']) for r in registry]])
    elif a.stage=='tune':
        launch('inverse_conditioning',['--output',str(a.output/'training_regularisation.json')])
    else:
        options=[]
        if a.stage=='inverse':
            options=['--regularisation-policy',str(report/'regularisation_policy.json'),
                '--shard-index',str(a.shard_index),'--shard-count',str(a.shard_count)]
        launch('noise_sweep',['--mode',a.stage,'--output',str(a.output/f'{a.stage}.json'),*options])


if __name__=='__main__': main()
