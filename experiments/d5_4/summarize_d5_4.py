"""Stratify without pooling directions, masks, scales, noise, or model seeds."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from experiments.d5_4.evidence_audit import digest


def flatten(value,prefix=''):
    result={}
    for k,v in value.items():
        name=prefix+k
        if isinstance(v,dict): result.update(flatten(v,name+'.'))
        elif isinstance(v,(int,float)) and not isinstance(v,bool): result[name]=v
    return result


def summarise(path):
    document=json.loads(path.read_text())
    groups=collections.defaultdict(list)
    for row in document['rows']:
        model=row.get('model',{})
        key=(row['direction'],row['mask'],row['level'],row.get('noise_fraction'),
             row.get('arm',model.get('arm')),model.get('seed'),row['dataset_id'])
        groups[key].append(row)
    strata=[]
    for key,rows in groups.items():
        # Exactly one value per original scene in every reported stratum.
        if len({r['scene_id'] for r in rows})!=len(rows):
            raise ValueError('duplicate scene within stratum')
        fields=collections.defaultdict(list)
        for row in rows:
            payload={k:row[k] for k in ('snr','single_oracle','paired_oracle','inverse','response','noise_only') if k in row}
            for name,value in flatten(payload).items():
                fields[name].append((row['scene_id'],value))
        stats={name:dict(mean=float(np.mean([v for _,v in values])),
            min=float(min(v for _,v in values)),max=float(max(v for _,v in values)),
            per_scene=dict(values)) for name,values in fields.items()}
        strata.append(dict(direction=key[0],mask=key[1],level=key[2],noise_fraction=key[3],
            arm=key[4],model_seed=key[5],dataset=key[6],scene_count=len(rows),metrics=stats,
            converged=sum(r.get('inverse',{}).get('convergence',{}).get('converged',False) for r in rows)))
    return dict(status=document['status'],source_sha256=digest(path),source_bytes=path.stat().st_size,
        provenance=document.get('provenance'),elapsed_seconds=document.get('elapsed_seconds'),
        peak_cuda_bytes=document.get('peak_cuda_bytes'),strata=strata,
        unit='original scene; model seeds remain separate; no pooled significance test')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x',encoding='utf-8') as f: json.dump(summarise(a.input),f,indent=2,allow_nan=False)


if __name__=='__main__': main()
