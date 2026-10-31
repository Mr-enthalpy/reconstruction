"""Prove only LF/CRLF transport equivalence; never ignore numerical changes."""
import hashlib
import json


def verify_basis(path, recorded_hashes):
    raw=path.read_bytes(); lf=raw.replace(b'\r\n',b'\n')
    variants={hashlib.sha256(value).hexdigest() for value in (lf,lf.replace(b'\n',b'\r\n'))}
    if not set(recorded_hashes)<=variants:
        raise ValueError('basis differs beyond proved LF/CRLF transport')
    canonical=json.dumps(json.loads(raw),sort_keys=True,separators=(',',':')).encode()
    return dict(accepted_byte_sha256=sorted(variants),canonical_json_sha256=hashlib.sha256(canonical).hexdigest(),
        explanation='Exactly the supplied basis bytes with LF or CRLF. No numerical tolerance or unverified alias.')
