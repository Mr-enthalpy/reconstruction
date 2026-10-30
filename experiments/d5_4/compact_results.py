"""Lossless scalar case tables for the D5.4 report, without repeated metadata."""
import argparse
import json
from pathlib import Path
from experiments.d5_4.evidence_audit import digest


def scalars(value,prefix=''):
    out={}
    for k,v in value.items():
        key=prefix+k
        if isinstance(v,dict): out.update(scalars(v,key+'.'))
        elif isinstance(v,(float,int,bool)) or v is None: out[key]=v
        elif isinstance(v,str) and k in ('status','reason'): out[key]=v
    return out


def compact(source,kind):
    doc=json.loads(source.read_text())
    allowed={'operator':('snr',), 'oracle':('single_oracle','paired_oracle'),
             'inverse':('inverse',), 'frozen':('response','noise_only')}[kind]
    records=[]; models={}
    for row in doc['rows']:
        model=row.get('model',{})
        record={k:row[k] for k in ('scene_id','dataset_id','direction','mask','level','noise_fraction')}
        record.update(arm=row.get('arm',model.get('arm')),model_seed=model.get('seed'))
        if model: models[model['run_dir']]=model
        if kind=='operator':
            record.update({k:row[k] for k in ('sigma','delta_measurement_norm',
                'forward_difference_identity_error','relative_measurement_difference')})
        payload={k:row[k] for k in allowed}
        record.update(scalars(payload))
        records.append(record)
    columns=sorted(set().union(*(r.keys() for r in records)))
    return dict(status=doc['status'],kind=kind,source_sha256=digest(source),
        provenance=doc['provenance'],elapsed_seconds=doc['elapsed_seconds'],
        peak_cuda_bytes=doc['peak_cuda_bytes'],models=models,
        unit='Each row is one condition in an original scene; scenes and model seeds are not pooled replicates.',
        omitted='Repeated construction metadata is in natural_counterfactual_summary.json. Full CG histories remain in the hashed source.',
        columns=columns,rows=[[r.get(k) for k in columns] for r in records])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['operator','oracle','inverse','frozen'],required=True)
    a=p.parse_args()
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(compact(a.input,a.kind),f,separators=(',',':'),allow_nan=False)


if __name__=='__main__': main()
