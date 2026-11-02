"""Run the D6 wiring check on one validation scene; never opens test cubes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.d6.g4 import load_bundle, check_inputs, shared_noise
from experiments.d6.inference_bundle import restore
from experiments.d6.pipeline import run_pipeline


def main():
    p=argparse.ArgumentParser()
    for key in ('prepared','bundle','physics','protocol','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    protocol=json.loads(a.protocol.read_bytes())
    bundle=load_bundle(a.bundle,a.physics,protocol)
    records=check_inputs(a.prepared,protocol,'val')
    item=bundle['models'][0]  # Predetermined first seed, no score-based selection.
    model=restore(item,a.device,bundle['original_physics'][item['metadata']['arm']])
    r=records[0]
    with np.load(a.prepared/'val.npz',allow_pickle=False) as z:
        x=torch.from_numpy(z['cube'][r['array_index']:r['array_index']+1].copy()).to(a.device)
    with torch.no_grad():
        y=model.physics(x)+torch.from_numpy(shared_noise((512,512),r['array_index'],protocol)).to(a.device)
    # A fixed deterministic head exercises the C interface, with no task claim.
    def wiring_head(f):
        return torch.stack((f[:,:3].mean(1),f[:,3:].mean(1)),dim=1)
    out=run_pipeline(y,model,[[0,0,512,512],[128,128,256,256]],
        observation_id=r['dataset_id']+':'+r['scene_id'],
        model_sha256=item['metadata']['checkpoint_sha256'],
        candidate_source='two fixed geometry boxes; no detection or ground truth',
        classifier=wiring_head,classifier_id='deterministic wiring fixture; untrained; no class semantics')
    np.savez_compressed(a.output/'private_arrays.npz',**{k:v for k,v in out.items() if k!='metadata'})
    receipt=dict(out['metadata'],split='val',candidates=len(out['boxes']),
                 cube_shape=list(out['cube'].shape),feature_shape=list(out['features'].shape),
                 score_shape=list(out['scores'].shape),
                 acceptance='data flow only; no classifier accuracy, detector or physical validation')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
