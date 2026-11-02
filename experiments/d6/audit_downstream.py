"""Metadata-only audit; no images, labels or scores enter N1."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def audit(task_path, n1_path):
    task=[r for r in json.loads(task_path.read_bytes()) if r['augmentation'] is None]
    n1=json.loads(n1_path.read_bytes())
    # D3 group IDs have a CAVE_ prefix; D5 also merges related scene families.
    # Join by source scene, then inherit the reconstruction family membership.
    lookup={(r['dataset_id'],r['scene_id']):r for r in n1}
    training_groups={(r['dataset_id'],r['source_group_id']) for r in n1 if r['split']=='train'}
    overlap=[];unmatched=collections.Counter()
    for row in task:
        match=lookup.get((row['dataset_id'],row['source_scene_or_acquisition_id']))
        if match is None:
            unmatched[row['dataset_id']]+=1
        elif (match['dataset_id'],match['source_group_id']) in training_groups:
            overlap.append(row)
    return dict(task_manifest_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
        n1_manifest_sha256=hashlib.sha256(n1_path.read_bytes()).hexdigest(),
        original_counts=[dict(dataset=d,split=s,label=l,count=c) for (d,s,l),c in
            sorted(collections.Counter((r['dataset_id'],r['split'],r['class_label']) for r in task).items())],
        n1_training_overlap_by_task_split=dict(collections.Counter(r['split'] for r in overlap)),
        unmatched_task_records_by_source=dict(unmatched),
        overlap_key='dataset + source scene, expanded to N1 source_group_id family; raw group strings differ between manifests',
        limitations=['Legacy task test scenes were evaluated historically; not a fresh integration holdout',
            'CAVE validation/test have no positive ROIs in this manifest',
            'UWA and CAVE have different radiometry and acquisition distributions',
            'D3 32x32 cropped observations cannot substitute for full-field 512x512 A6 inference',
            'UWA absent from N1 manifest; unmatched IDs alone do not establish all historical exposure'],
        decision='Do not claim independent RGB-versus-RGB+N1 gain from the legacy manifest. Freeze full-field candidates, matched capacity, source-balanced reporting and an untouched final holdout before a task experiment.')


def main():
    p=argparse.ArgumentParser()
    for key in ('task','n1','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x',encoding='utf-8') as f:json.dump(audit(a.task,a.n1),f,indent=2)


if __name__=='__main__':main()
