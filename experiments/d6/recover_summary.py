"""Recover fully written G4 rows after summary JSON serialization failure.

Never imports a data loader or runs inference. Retains the partial source bytes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.d6.g4 import summarize


def recover(source, protocol):
    # The top-level summary is written after the complete rows array.
    prefix, separator, _ = source.replace('\r\n','\n').partition(',\n  "summary":')
    if not separator:
        raise ValueError('no complete rows prefix preceding summary')
    result=json.loads(prefix+'\n}')
    if result['status']!='completed' or result['split']!='test':
        raise ValueError('not a completed inference record')
    if result['models']!=protocol['models'] or len(result['rows'])!=126:
        raise ValueError('unexpected frozen coverage')
    result['summary']=summarize(result['rows'],protocol)
    # NumPy counts must become native JSON numbers, without changing any value.
    def native(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    return json.loads(json.dumps(result,allow_nan=False,default=native))


def main():
    p=argparse.ArgumentParser()
    for key in ('source','protocol','output','ledger','audit'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.audit.exists():raise FileExistsError('recovery exists')
    raw=a.source.read_bytes();protocol=json.loads(a.protocol.read_bytes())
    result=recover(raw.decode('utf-8'),protocol)
    ledger=json.loads(a.ledger.read_bytes())
    assert ledger['status']=='started'
    assert ledger['protocol_sha256']==result['protocol_sha256']==hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    assert ledger['code_commit']==result['code_commit']
    audit=dict(reason='NumPy int64 scene_direction_counts failed JSON serialization after all 126 rows were written',
        source_sha256=hashlib.sha256(raw).hexdigest(),source_file=a.source.name,
        original_ledger=dict(ledger),rows=126,test_cubes_reopened=False,inference_repeated=False,
        procedure='Parse complete top-level prefix before summary; recompute frozen summary from saved rows; convert NumPy scalar counts to native JSON numbers',
        recovered_utc=datetime.now(timezone.utc).isoformat())
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    audit['result_sha256']=hashlib.sha256(a.output.read_bytes()).hexdigest()
    with a.audit.open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2)
    ledger.update(status='completed_after_summary_serialization_recovery',
        result_file=a.output.name,result_sha256=audit['result_sha256'],
        recovery_audit=a.audit.name,completed_utc=audit['recovered_utc'])
    a.ledger.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':main()
