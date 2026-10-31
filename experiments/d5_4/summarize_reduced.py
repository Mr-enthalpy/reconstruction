"""Validate the complete fixed baseline-substitution sweep and retain all failures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from experiments.d5_4.evidence_audit import digest
from experiments.d5_4.decision_summary import recovered
from experiments.d5_4.reduced_oracle import passing_strata


def merge(paths,inverse_path):
    inverse=json.loads(inverse_path.read_text()); selected=passing_strata(inverse)
    original=[dict(zip(inverse['columns'],r)) for r in inverse['rows']]
    fields=('scene_id','direction','mask','level','noise_fraction')
    expected={tuple(r[k] for k in fields) for r in original if r['arm']=='chromatic'
        and (r['direction'],r['mask'],r['level'],r['noise_fraction']) in selected}
    docs=[json.loads(p.read_text()) for p in paths]; count=len(docs)
    if {d['shard']['index'] for d in docs}!=set(range(count)):
        raise ValueError('missing/duplicate reduced partition')
    if any(d['status']!='completed_shard' or d['shard']['count']!=count for d in docs):
        raise ValueError('incomplete reduced partition')
    if any(d['provenance']!=docs[0]['provenance'] for d in docs):
        raise ValueError('reduced partition provenance differs')
    if docs[0]['provenance']['inverse_sha256']!=digest(inverse_path):
        raise ValueError('inverse source bytes differ')
    rows=[r for d in docs for r in d['rows']]
    keys=[tuple(r[k] for k in fields) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('reduced cases differ from the complete predefined gate')
    groups=defaultdict(list)
    for row in rows:
        row['convergence'].pop('normal_residual_history',None)
        info=row['convergence'].get('cg_info')
        row['budget_censored']=bool(info is not None and info>0)
        row['recovery_passed']=bool(row['convergence']['converged'] and recovered(row['correction_metrics']))
        groups[tuple(row[k] for k in fields[1:])].append(row)
    strata=[]
    for key,items in sorted(groups.items()):
        if len(items)!=9: raise ValueError('incomplete scene stratum')
        passed=sum(r['recovery_passed'] for r in items)
        strata.append(dict(direction=key[0],mask=key[1],level=key[2],noise_fraction=key[3],
            passed_scenes=passed,gate_passed=passed>=8))
    aggregate=[]
    for dataset in ['all',*sorted({r['dataset_id'] for r in rows})]:
        subset=[r for r in rows if dataset=='all' or r['dataset_id']==dataset]
        scenes=defaultdict(list)
        for r in subset: scenes[r['scene_id']].append(r)
        means={k:float(np.mean([np.mean([r['correction_metrics'][k] for r in values])
            for values in scenes.values()])) for k in ('relative_l2','cosine','correlation','projection_gain','response_fraction')}
        ratios=[r['baseline_measurement_bias_norm']/r['true_perturbation_measurement_norm'] for r in subset]
        worst=max(subset,key=lambda r:r['correction_metrics']['relative_l2'])
        aggregate.append(dict(dataset_id=dataset,original_scenes=len(scenes),cases=len(subset),
            scene_balanced_means=means,baseline_bias_to_signal_ratio_quantiles=np.quantile(ratios,[0,.25,.5,.75,1]).tolist(),
            worst_case={k:worst[k] for k in (*fields,'correction_metrics')}))
    return dict(status='completed',provenance=docs[0]['provenance'],
        source_sha256=[digest(p) for p in paths],rows=rows,strata=strata,aggregates=aggregate,
        converged_cases=sum(r['convergence']['converged'] for r in rows),
        budget_censored_cases=sum(r['budget_censored'] for r in rows),
        elapsed_worker_seconds=sum(d['elapsed_seconds'] for d in docs),
        peak_cuda_bytes=max(d['peak_cuda_bytes'] for d in docs),
        prior=docs[0]['prior'],observation_count=2,reference_noise_fraction=0,
        limitation=docs[0]['limitation']+' Direct clean-reference subtraction would reproduce the ideal residual; this is avoidable baseline-substitution bias, not a fundamental reduced-prior limit.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--inverse',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=merge(a.inputs,a.inverse)
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,separators=(',',':'),allow_nan=False)


if __name__=='__main__': main()
