"""Finite cropped-operator probes, not circulant singular-value claims."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.d5_4.restricted_inverse_operator import RestrictedOperator
from experiments.d5_4.evidence_audit import digest
from lcd_reconstruction.operators.psf_provenance import array_digest


def main():
    p=argparse.ArgumentParser()
    for name in ('kernel-file','basis','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cpu'); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(4)
    k=np.load(a.kernel_file)['kernels']; basis=json.loads(a.basis.read_text())
    protocol=json.loads(a.protocol.read_text())
    if array_digest(k)!=protocol['arms']['s1c']['kernel_array_sha256']:
        raise ValueError('kernel mismatch')
    yy,xx=np.mgrid[:512,:512]
    probes={'constant':np.ones((512,512)), 'local_smooth':np.exp(-((xx-256)**2+(yy-256)**2)/(2*64**2))}
    for f in (.01,.05,.2,.45): probes[f'cosine_x_{f}']=np.cos(2*np.pi*f*xx)
    rows=[]
    for arm,kernels in dict(chromatic=k,achromatic_mean=np.repeat(k.mean(0,keepdims=True),21,0),
                            achromatic_550=np.repeat(k[10:11],21,0)).items():
        for index,v in enumerate(basis['directions']):
            b=RestrictedOperator(kernels,v,(512,512),a.device)
            for name,z in probes.items():
                y=b.forward(z); total=float(np.linalg.norm(y)**2)
                rows.append(dict(arm=arm,direction=index,probe=name,
                    amplitude_gain=float(np.linalg.norm(y)/np.linalg.norm(z)),
                    interior_energy_fraction=float(np.linalg.norm(y[64:-64,64:-64])**2/total) if total>1e-25 else None))
    result=dict(status='completed_descriptive_probes',rows=rows,kernel_sha256=digest(a.kernel_file),
        basis_sha256=digest(a.basis),boundary='existing full zero-padded convolution and centre crop',
        limitation='Probe gains are not singular values or a condition-number estimate; no circulant inverse used.')
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__': main()
