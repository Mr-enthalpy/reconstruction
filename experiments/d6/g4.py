"""One frozen G4 evaluation; reuse D5 metrics and exact D5.3 inference states."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from experiments.d5.metrics import reconstruction_metrics
from experiments.d6.inference_bundle import restore
from lcd_reconstruction.operators.psf_provenance import file_digest

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('psnr_unit_range_db', 'ssim_band_mean', 'sam_deg', 'cube_relative_l2',
           'global_spectrum_relative_l2', 'high_frequency_relative_l2',
           'spatial_gradient_relative_l2', 'measurement_relative_l2')
HIGHER = {'psnr_unit_range_db', 'ssim_band_mean'}


def read_protocol(path, expected_sha, allow_test=False):
    if not allow_test:
        raise ValueError('G4 test access requires explicit authorization flag')
    if file_digest(path) != expected_sha:
        raise ValueError('frozen protocol bytes differ')
    protocol = json.loads(path.read_bytes())
    if protocol['status'] != 'frozen_before_test':
        raise ValueError('protocol is not frozen')
    for name, sha in protocol['implementation_sha256_lf'].items():
        actual = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != sha:
            raise ValueError('implementation differs from frozen protocol: '+name)
    return protocol


def load_bundle(path, physics_path, protocol):
    if file_digest(path) != protocol['inference_bundle_sha256']:
        raise ValueError('inference bundle bytes differ')
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle['format'] != 'd6-lossless-inference-v1':
        raise ValueError('unsupported inference bundle')
    if [x['metadata'] for x in bundle['models']] != protocol['models']:
        raise ValueError('model selection differs from frozen protocol')
    if file_digest(physics_path) != protocol['original_physics_sha256']:
        raise ValueError('original physical buffers differ')
    physics = torch.load(physics_path,map_location='cpu',weights_only=True)
    # Check every model before the test archive is opened.
    for item in bundle['models']:
        restore(item,original_physics=physics[item['metadata']['arm']])
    bundle['original_physics']=physics
    return bundle


def check_inputs(prepared, protocol, split):
    for name in ('manifest.json', split+'.npz'):
        if file_digest(prepared/name) != protocol['input_sha256'][name]:
            raise ValueError('input bytes differ: '+name)
    records = json.loads((prepared/'manifest.json').read_bytes())
    groups = {}
    for r in records:
        if groups.setdefault((r['dataset_id'],r['scene_id']),r['split']) != r['split']:
            raise ValueError('scene crosses reconstruction splits')
    selected = [r for r in records if r['split']==split]
    if len(selected) != (14 if split=='test' else 9):
        raise ValueError('unexpected original scene count')
    if sorted(r['array_index'] for r in selected) != list(range(len(selected))):
        raise ValueError('invalid array indices')
    return selected


def shared_noise(hw, scene_index, protocol):
    rng = np.random.default_rng(np.random.SeedSequence([protocol['noise_seed'],scene_index]))
    return rng.normal(0,protocol['absolute_sigma'],(1,1,*hw)).astype(np.float32)


def summarize(rows, protocol):
    summaries=[]; comparisons=[]
    for model in protocol['models']:
        selected=[r for r in rows if r['run_dir']==model['run_dir']]
        if len(selected)!=14 or len({r['scene_id'] for r in selected})!=14:
            raise ValueError('incomplete G4 model/scene coverage')
        for source in ('ALL','CAVE','KAIST'):
            rs=[r for r in selected if source=='ALL' or r['dataset_id']==source]
            means={k:float(np.mean([r['metrics'][k] for r in rs])) for k in METRICS}
            bands={k:np.mean([r['metrics'][k] for r in rs],axis=0).tolist()
                   for k in ('band_rmse','band_relative_l2')}
            worst={k:[dict(scene_id=r['scene_id'],value=r['metrics'][k]) for r in
                      sorted(rs,key=lambda r:r['metrics'][k],reverse=k not in HIGHER)[:3]]
                   for k in METRICS}
            summaries.append(dict(run_dir=model['run_dir'],arm=model['arm'],seed=model['seed'],
                                  source=source,scenes=len(rs),mean=means,bands=bands,worst=worst))
    seeds=sorted({m['seed'] for m in protocol['models']})
    for control in ('s3a','s3a550'):
        for source in ('ALL','CAVE','KAIST'):
            for metric in METRICS:
                seed_deltas=[]; directional_counts=[]; paired=[]
                for seed in seeds:
                    cr={r['scene_id']:r for r in rows if r['arm']=='s1c' and r['seed']==seed
                        and (source=='ALL' or r['dataset_id']==source)}
                    ar={r['scene_id']:r for r in rows if r['arm']==control and r['seed']==seed
                        and (source=='ALL' or r['dataset_id']==source)}
                    if cr.keys()!=ar.keys():raise ValueError('unpaired scene sets')
                    delta={s:cr[s]['metrics'][metric]-ar[s]['metrics'][metric] for s in sorted(cr)}
                    seed_deltas.append(float(np.mean(list(delta.values()))));paired.append(delta)
                mean=float(np.mean(seed_deltas));sd=float(np.std(seed_deltas,ddof=1))
                sign=np.sign(mean)
                directional_counts=[sum(np.sign(d)==sign for d in p.values()) for p in paired]
                threshold=protocol['scene_direction_minimum'][source]
                stable=bool(sign!=0 and all(np.sign(d)==sign for d in seed_deltas)
                            and abs(mean)>sd and min(directional_counts)>=threshold)
                comparisons.append(dict(control=control,source=source,metric=metric,
                    definition='Chromatic minus named control; positive favours Chromatic only for PSNR/SSIM',
                    seeds=seeds,seed_means=seed_deltas,mean=mean,seed_sd=sd,
                    scene_differences=paired,scene_direction_counts=directional_counts,
                    stability_passed=stable,chromatic_favoured=bool((mean>0)==(metric in HIGHER)),
                    interpretation='descriptive frozen stability rule, not a significance test'))
    return dict(per_model_source=summaries,paired_comparisons=comparisons,
                unit='seed n=3 for stability; original scenes nested within seed, not 42 independent trials')


def run(args):
    protocol=read_protocol(args.protocol,args.protocol_sha256,
                           allow_test=args.allow_frozen_test or args.mode=='benchmark')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bundle=load_bundle(args.bundle,args.physics,protocol)
    split='val' if args.mode=='benchmark' else 'test'
    records=check_inputs(args.prepared,protocol,split)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    ledger=ROOT/protocol['access_ledger']
    if args.mode=='formal':
        # Exclusive durable marker precedes the only test-array load. Never resume
        # automatically or restart into a fresh output folder after any failure.
        with ledger.open('x',encoding='utf-8') as f:
            json.dump(dict(status='started',started_utc=datetime.now(timezone.utc).isoformat(),
                           code_commit=commit,protocol_sha256=args.protocol_sha256),f,indent=2)
    with np.load(args.prepared/(split+'.npz'),allow_pickle=False) as archive:
        cubes=archive['cube']
    if cubes.shape!=(len(records),21,512,512) or not np.isfinite(cubes).all():
        raise ValueError('invalid frozen cube shape or values')
    selected_models=bundle['models'][:1] if args.mode=='benchmark' else bundle['models']
    selected_records=records[:1] if args.mode=='benchmark' else records
    started=time.perf_counter();rows=[]
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for item in selected_models:
            meta=item['metadata']
            model=restore(item,args.device,bundle['original_physics'][meta['arm']])
            for r in selected_records:
                x=torch.from_numpy(np.array(cubes[r['array_index']:r['array_index']+1],copy=True)).to(args.device)
                y=model.physics(x)+torch.from_numpy(shared_noise((512,512),r['array_index'],protocol)).to(args.device)
                out=model(y)
                metric=reconstruction_metrics(out['spectral_cube_hat'][0].cpu().numpy(),
                    x[0].cpu().numpy(),y[0].cpu().numpy(),out['reprojected_measurement'][0].cpu().numpy())
                rows.append(dict(scene_id=r['scene_id'],dataset_id=r['dataset_id'],
                    run_dir=meta['run_dir'],arm=meta['arm'],seed=meta['seed'],metrics=metric))
                print(json.dumps(dict(completed=len(rows),elapsed_seconds=time.perf_counter()-started)),flush=True)
            del model
    result=dict(status='benchmark_only' if args.mode=='benchmark' else 'completed',split=split,
        protocol_sha256=args.protocol_sha256,code_commit=commit,models=protocol['models'],
        input_sha256=protocol['input_sha256'],absolute_sigma=protocol['absolute_sigma'],
        noise_seed=protocol['noise_seed'],inference_bundle_sha256=protocol['inference_bundle_sha256'],
        original_physics_sha256=protocol['original_physics_sha256'],
        elapsed_seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        runtime=dict(torch=torch.__version__,numpy=np.__version__,device=args.device),
        scope='frozen reconstruction holdout; simulation only; no task or physical-information claim')
    if args.mode=='formal':
        result.update(rows=rows,summary=summarize(rows,protocol))
    # Benchmark never publishes scores; it cannot serve checkpoint selection.
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    if args.mode=='formal':
        record=json.loads(ledger.read_bytes());record.update(status='completed',
            result_sha256=file_digest(args.output),completed_utc=datetime.now(timezone.utc).isoformat())
        ledger.write_text(json.dumps(record,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    for key in ('prepared','bundle','physics','protocol','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--mode',choices=['benchmark','formal'],required=True)
    p.add_argument('--allow-frozen-test',action='store_true')
    p.add_argument('--device',default='cuda')
    run(p.parse_args())


if __name__=='__main__':main()
