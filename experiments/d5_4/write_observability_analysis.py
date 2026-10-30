"""Describe every direction/mask/amplitude stratum of the operator sweep."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from experiments.d5_4.evidence_audit import digest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    doc=json.loads(a.input.read_text()); groups=defaultdict(list)
    for r in doc['rows']:
        if r['arm']=='chromatic' and r['noise_fraction']:
            groups[(r['noise_fraction'],r['mask'],r['level'],r['direction'])].append(r)
    rows=[]
    for key,items in sorted(groups.items()):
        values=[r['single_oracle']['expected_amplitude_snr'] for r in items]
        rows.append(dict(noise=key[0],mask=key[1],level=key[2],direction=key[3],
            mean_single_template_snr=float(np.mean(values)),
            scenes_snr_ge_3=sum(v>=3 for v in values),
            scene_values={r['scene_id']:v for r,v in zip(items,values)},
            mean_paired_pixel_rms_snr=float(np.mean([r['snr']['per_pixel_snr'] for r in items]))))
    result=dict(source_sha256=digest(a.input),strata=rows,
        max_forward_difference_identity_error=max(r['forward_difference_identity_error'] for r in doc['rows']),
        max_achromatic_difference_norm=max(r['delta_measurement_norm'] for r in doc['rows'] if r['arm']!='chromatic'),
        interpretation='Each row fixes noise, mask, amplitude quantile and spectral direction; means use nine scenes. '
            'The known-template single-observation SNR uses the half-pair template. Paired pixel SNR uses the full pair and sqrt(2) noise.')
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='strata'}))
    for fraction in (.001,.005):
        s=[r for r in rows if r['noise']==fraction]
        print(json.dumps(dict(noise=fraction,configurations=len(s),
            configurations_with_8_of_9_detectable=sum(r['scenes_snr_ge_3']>=8 for r in s),
            maximum_scene_mean_paired_pixel_snr=max(r['mean_paired_pixel_rms_snr'] for r in s))))


if __name__=='__main__': main()
