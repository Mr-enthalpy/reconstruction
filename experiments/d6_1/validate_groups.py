"""Fail-closed metadata guards. This module never reads image arrays."""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_roi(roi, shape):
    if not isinstance(roi, (list, tuple)) or len(roi) != 4 or not shape or len(shape) != 2:
        raise ValueError('native ROI coordinates and native spatial shape required')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or int(x) != x for x in roi):
        raise ValueError('finite integer xywh required')
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > shape[1] or y+h > shape[0]:
        raise ValueError('ROI outside native full field')


def native_to_working(roi, shape):
    validate_roi(roi, shape)
    x, y, w, h = roi
    # Fixed coverage convention, no label- or score-dependent box adjustment.
    x0, y0 = math.floor(x*512/shape[1]), math.floor(y*512/shape[0])
    x1, y1 = math.ceil((x+w)*512/shape[1]), math.ceil((y+h)*512/shape[0])
    return [x0, y0, x1-x0, y1-y0]


def full_field_eligible(row):
    shape = row.get('native_spatial_shape')
    return bool(row.get('available_full_field') and row.get('full_field_verified')
                and row.get('full_field_kind') in ('source_band_archive', 'native_full_cube')
                and shape and len(shape) == 2 and min(shape) >= 512
                and row.get('source_file_sha256_verified'))


def label_status(row):
    if row.get('presentation_type') == 'printed_photo' and row.get('recognizable_face_confirmed'):
        if row.get('label') != 'face':
            raise ValueError('recognizable printed face must remain positive')
    if row.get('semantic_ambiguity') or row.get('label_review_status') in ('ambiguous', 'unreviewed'):
        return 'ambiguous_review_required'
    if row.get('label') not in ('face', 'non_face'):
        return 'ambiguous_review_required'
    return 'accepted_legacy_semantics'


def final_reasons(row):
    reasons = []
    if not all(row.get(k) for k in ('original_source_id', 'source_group_id', 'source_file_sha256')):
        reasons.append('missing_source_identity')
    if row.get('independence_status') != 'verified_independent':
        reasons.append('independence_not_verified')
    if row.get('n1_exposure'):
        reasons.append('known_n1_exposure')
    if row.get('historical_task_exposure') or row.get('other_exposure'):
        reasons.append('historical_exposure')
    if row.get('g4_consumed'):
        reasons.append('g4_consumed')
    if not full_field_eligible(row):
        reasons.append('full_field_ineligible')
    if label_status(row) != 'accepted_legacy_semantics':
        reasons.append('label_review_required')
    if not row.get('license_local_research_recorded'):
        reasons.append('license_evidence_missing')
    return reasons


def validate_groups(rows):
    groups, samples = {}, set()
    for r in rows:
        if r['sample_id'] in samples:
            raise ValueError('duplicate sample identity')
        samples.add(r['sample_id'])
        ds, role = r['dataset_id'], r['protocol_role']
        # Review-only records do not enter a training/evaluation partition.
        if role not in ('development_train', 'development_val', 'development_test', 'final_test'):
            continue
        keys = [(ds, 'family', r.get('source_group_id')),
                (ds, 'source', r.get('original_source_id')),
                (ds, 'acquisition', r.get('source_scene_or_acquisition_id'))]
        if ds == 'UWA':
            if not r.get('subject_id'):
                raise ValueError('UWA subject identity required')
            keys += [(ds, 'subject', r['subject_id'])]
        if r.get('acquisition_family_id'):
            keys += [(ds, 'acquisition_family', r['acquisition_family_id'])]
        if r.get('session_family_id'):
            keys += [(ds, 'session_family', r['session_family_id'])]
        if not all(r.get(k) for k in ('original_source_id', 'source_group_id', 'source_scene_or_acquisition_id')):
            raise ValueError('missing source identity')
        for key in keys:
            if groups.setdefault(key, role) != role:
                raise ValueError('source group crosses partitions')
        if r.get('augmentation') is not None and role != 'development_train':
            raise ValueError('augmentation outside training')
        if role == 'final_test' and final_reasons(r):
            raise ValueError('ineligible final holdout')
        if not full_field_eligible(r):
            raise ValueError('full_field_ineligible')
        validate_roi(r.get('roi_xywh'), r.get('native_spatial_shape'))
        if label_status(r) != 'accepted_legacy_semantics':
            raise ValueError('unreviewed labels in active partition')
    return dict(status='passed', active_records=sum(r['protocol_role'].startswith('development_') or
                r['protocol_role']=='final_test' for r in rows), group_keys_checked=len(groups))


def assign_roles(rows, admitted_ids):
    result = []
    for row in sorted(rows, key=lambda r: r['sample_id']):
        r = dict(row)
        r['protocol_role'] = 'review_only'
        if (r['sample_id'] in admitted_ids and r.get('original_split') in ('train', 'val', 'test')
                and full_field_eligible(r) and label_status(r) == 'accepted_legacy_semantics'):
            r['protocol_role'] = 'development_' + r['original_split']
        r['final_holdout_exclusion_reasons'] = final_reasons(r)
        result.append(r)
    validate_groups(result)
    return result


def metric_eligibility(labels):
    both = set(labels) == {'face', 'non_face'}
    return dict(binary_metrics_allowed=both,
                balanced_accuracy=None if not both else 'eligible',
                macro_f1=None if not both else 'eligible',
                auroc=None if not both else 'eligible',
                auprc=None if not both else 'eligible',
                report='both class recalls' if both else 'present-class recall/error only; absent-class metrics undefined')


def load_development(manifest, lock, role, *, expected_lock_sha256):
    """Return frozen metadata only; final-test roles are denied before any file IO."""
    if role not in ('development_train', 'development_val', 'development_test'):
        raise PermissionError('development loader cannot access final holdout')
    if digest(lock) != expected_lock_sha256:
        raise ValueError('protocol lock changed')
    frozen = json.loads(Path(lock).read_bytes())
    if digest(manifest) != frozen['private_manifest_sha256']:
        raise ValueError('frozen sample manifest changed')
    rows = json.loads(Path(manifest).read_bytes())
    validate_groups(rows)
    return [r for r in rows if r['protocol_role'] == role]
