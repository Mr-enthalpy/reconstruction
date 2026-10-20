"""Explicit-path D2 preparation; never alter D1 manifests or measurements."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from lcd_reconstruction.datasets.controlled_views import common_scene,make_view

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--uwa',type=Path,required=True);ap.add_argument('--cave',type=Path,required=True);ap.add_argument('--psf',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader((a.uwa/'dataset_manifest.csv').open()))
    faces={r['acquisition_id']:r for r in rows if r['status']=='accepted_face' and r['augmentation']=='none'}
    negatives={r['acquisition_id']:r for r in rows if r['status']=='accepted_non_face' and r['augmentation']=='none'}
    pairs=sorted(faces.keys()&negatives.keys());records=[];cubes=[]
    def add(cube,meta,bits):
        x,factor=common_scene(cube,bits);idx=len(records)
        meta.update(index=idx,normalization_factor=factor,normalization_policy='bit range then cube grand mean 0.2; dimensionless equal-energy flat-illumination assumption',wavelengths_nm=list(range(450,651,10)),bit_depth=bits,augmentation=None)
        records.append(meta);cubes.append(x)
    for sid in pairs:
        for r in [faces[sid],negatives[sid]]:
            f=a.uwa/r['path'];assert sha(f)==r['sample_sha256']
            z=np.load(f);m=json.loads(str(z['metadata_json']))
            add(z['cube'],dict(sample_id=r['sample_id'],parent_id=sid,subject_id=r['subject_id'],split=r['split'],label=r['label'],source_dataset='UWA',source_type='uncalibrated_processed_intensity',source_sha256=sha(f),crop_xywh=m.get('crop_xywh'),label_review_status='D1 accepted; Agent visual review'),8)
    inventory=json.loads((a.cave/'cave_inventory.json').read_text())
    review=json.loads((a.cave/'cave_review_decisions.json').read_text())
    accepted=[r for r in inventory if review.get(r['scene_id'],{}).get('decision')=='accepted_non_face']
    # Deterministic scene split before model fitting; no patch-level random split.
    ordered=sorted(accepted,key=lambda r:hashlib.sha256(('D2-scene-'+r['scene_id']).encode()).hexdigest())
    n=len(ordered);train=int(.6*n);val=int(.2*n)
    if n<5:raise ValueError('need at least five reviewed scenes for train/val/test; preserve rejected data')
    for i,r in enumerate(ordered):
        split='train' if i<train else 'val' if i<train+val else 'test'
        add(np.load(a.cave/r['cube_path'])['cube'],dict(sample_id='CAVE_'+r['scene_id'],parent_id=r['scene_id'],scene_id=r['scene_id'],parent_scene_id=r['scene_id'],subject_id=None,split=split,label='non_face',source_dataset='CAVE',source_type='approximate_reflectance',source_sha256=r['source_sha256'],crop_xywh=[0,0,512,512],label_review_status=review[r['scene_id']]),16)
    original_count=len(records)
    for i in range(original_count):
        r=records[i]
        if r['split']!='train':continue
        seed=int(hashlib.sha256(('D2-'+r['source_dataset']+r['parent_id']).encode()).hexdigest()[:8],16)
        view,params=make_view(cubes[i],seed);replay,_=make_view(cubes[i],seed)
        assert np.array_equal(view,replay)
        records.append(dict(r,index=len(records),sample_id=r['sample_id']+'_aug',augmentation=params));cubes.append(view)
    np.savez_compressed(a.output/'scenes.npz',cube=np.stack(cubes),global_spectrum=np.stack(cubes).mean((-2,-1)),wavelengths_nm=np.arange(450,651,10))
    (a.output/'manifest.json').write_text(json.dumps(records,indent=2))
    protocols={name:[r['index'] for r in records if (name=='B' or (name=='A' and r['source_dataset']=='UWA') or (name=='C' and (r['label']=='face' or r['source_dataset']=='CAVE')))] for name in ['A','B','C']}
    (a.output/'protocols.json').write_text(json.dumps(protocols,indent=2))
    kernels={512:[],1024:[]};audit=[]
    for nm in range(450,651,10):
        f=a.psf/f'roi_extent_review/roi1024_{nm}.npz';z=np.load(f);m=json.loads(str(z['metadata_json']))
        assert m['wavelength_nm']==nm and m['roi_xywh']==[866,554,1024,1024] and m['normalization']=='none'
        cmd=m['camera_settings']['requested_exposure_us'];probe={201.6:1,403.2:2,1612.8:3}[cmd]
        dark=sum(np.load(a.psf/f'received/R21Dark{phase}01/probe_{probe:03d}_mean.npy')[554:1578,866:1890].astype(float) for phase in ['Pre','Post'])/2
        corrected=z['psf'].astype(float)-dark;mask=np.ones((1024,1024),bool);mask[40:-40,40:-40]=False;corrected-=corrected[mask].mean()
        audit.append(dict(wavelength_nm=nm,source_sha256=sha(f),signed_sum=float(corrected.sum()),negative_fraction=float((corrected<0).mean()),camera_settings=m['camera_settings']))
        for size in kernels:
            k=corrected if size==1024 else corrected[256:768,256:768]
            k=k/k.sum();k=k.reshape(size//16,16,size//16,16).sum((1,3))
            kernels[size].append(k)
    np.savez_compressed(a.output/'kernels.npz',k512=np.stack(kernels[512]).astype('float32'),k1024=np.stack(kernels[1024]).astype('float32'),weights=np.ones(21),sensor_pixels_per_pixel=16)
    (a.output/'psf_audit.json').write_text(json.dumps(audit,indent=2))
    summary=dict(original_rois=original_count,derived_training_views=len(records)-original_count,uwa_paired_acquisitions=len(pairs),cave_scenes=n,group_checks=True,augmentation_replay=True,original_manifest_sha256=sha(a.uwa/'dataset_manifest.csv'),frozen_splits_sha256=sha(a.uwa/'splits.json'))
    # All original and derived examples of a group stay in one split.
    groups={}
    for r in records:
        key=(r['source_dataset'],r['subject_id'] or r['parent_id'])
        assert groups.setdefault(key,r['split'])==r['split']
    (a.output/'preparation_checks.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
