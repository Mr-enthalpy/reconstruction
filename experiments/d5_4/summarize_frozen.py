"""Scene-balanced summaries; keep model seeds, directions and noise separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from experiments.d5_4.decision_summary import group
from experiments.d5_4.evidence_audit import digest


def summarize(source):
    doc=json.loads(source.read_text())
    if doc['status']!='completed' or len(doc['rows'])!=6561:
        raise ValueError('complete nine-model sweep required')
    groups=defaultdict(list)
    for row in doc['rows']:
        m=row['model']
        for dataset in ('all',row['dataset_id']):
            groups[(m['arm'],m['seed'],row['noise_fraction'],dataset)].append(row)
    aggregate=[]
    for (arm,seed,noise,dataset),rows in sorted(groups.items()):
        scenes=defaultdict(list)
        for row in rows: scenes[row['scene_id']].append(row)
        if any(len(v)!=27 for v in scenes.values()):
            raise ValueError('missing condition within scene')
        summary={}
        for category in ('response','noise_only'):
            summary[category]={}
            for metric in ('relative_l2','cosine','response_fraction','projection_gain',
                           'correct_direction_fraction','local_relative_l2'):
                means=[]
                for values in scenes.values():
                    numbers=[r[category][metric] for r in values if r[category][metric] is not None]
                    if numbers: means.append(float(np.mean(numbers)))
                summary[category][metric]=dict(scene_mean=float(np.mean(means)) if means else None,
                    minimum_scene_mean=min(means) if means else None,
                    maximum_scene_mean=max(means) if means else None)
        worst=max(rows,key=lambda r:r['response']['relative_l2'])
        aggregate.append(dict(arm=arm,seed=seed,noise_fraction=noise,dataset_id=dataset,
            original_scenes=len(scenes),conditions_per_scene=27,metrics=summary,
            worst_case={k:worst[k] for k in ('scene_id','direction','mask','level','response')}))
    return dict(status='completed',source_sha256=digest(source),provenance=doc['provenance'],
        aggregates=aggregate,strata=group(doc['rows'],'frozen'),
        interpretation='Scene-balanced descriptive means across 27 fixed conditions; no pooled scene-times-seed significance. Every seed and complete stratum retained.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=summarize(a.input)
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,separators=(',',':'),allow_nan=False)


if __name__=='__main__': main()
