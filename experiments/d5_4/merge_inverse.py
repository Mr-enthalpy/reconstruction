"""Validate and concatenate fixed case partitions; never reduce numerical budgets."""
import argparse
import json
from pathlib import Path
from experiments.d5_4.evidence_audit import digest


def merge(paths):
    docs=[json.loads(p.read_text()) for p in paths]
    count=len(docs)
    if {d['shard']['index'] for d in docs}!=set(range(count)):
        raise ValueError('missing or duplicate shards')
    if any(d['status']!='completed_shard' or d['shard']['count']!=count for d in docs):
        raise ValueError('incomplete partition')
    if any(d['provenance']!=docs[0]['provenance'] for d in docs):
        raise ValueError('partition provenance differs')
    rows=[r for d in docs for r in d['rows']]
    fields=('scene_id','direction','mask','level','arm','noise_fraction')
    keys={tuple(r[f] for f in fields) for r in rows}
    if len(keys)!=2187 or len(rows)!=2187:
        raise ValueError('expected unique 243 cases x 3 arms x 3 noise levels')
    result=dict(docs[0]); result.update(status='completed',rows=rows,
        shard_sources=[dict(sha256=digest(p),index=d['shard']['index'],seconds=d['elapsed_seconds']) for p,d in zip(paths,docs)],
        elapsed_seconds=sum(d['elapsed_seconds'] for d in docs),
        peak_cuda_bytes=max(d['peak_cuda_bytes'] or 0 for d in docs))
    result.pop('shard')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=merge(a.inputs)
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,separators=(',',':'),allow_nan=False)


if __name__=='__main__': main()
