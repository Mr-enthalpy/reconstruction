"""Read-only source audit. Pixels and G4 cube archives are never decoded."""
import argparse
import collections
import csv
import json
from pathlib import Path
import struct
import subprocess
import zipfile
import numpy as np
from experiments.d6.audit_downstream import audit as legacy_audit
from experiments.d6_1.validate_groups import digest, assign_roles, native_to_working
from experiments.d6_1.freeze_classification_protocol import publish


def read(path):
    return json.loads(Path(path).read_bytes())


def npz_header(path):
    with zipfile.ZipFile(path) as z, z.open('cube.npy') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('unsupported NPY header')
    return list(shape), str(dtype)


def inspect_uwa(archive, subject, session):
    members = [f'UWA HSFD V1.1/HyperSpec_Face_Session{session}/{subject}/F_{band:02d}.png'
               for band in range(6, 27)]
    for member in members:
        if member not in archive.namelist():
            return False, None, members
        with archive.open(member) as f:
            header = f.read(29)
        if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
            return False, None, members
        w, h, depth, colour = struct.unpack('>IIBB', header[16:26])
        if (h, w, depth, colour) != (1024, 1024, 8, 0):
            return False, None, members
    return True, [1024, 1024], members


def build(a):
    task, d2, n1 = read(a.task), read(a.d2), read(a.n1)
    d1 = list(csv.DictReader(a.d1_manifest.open(encoding='utf-8')))
    splits, d1_review, cave_review = read(a.d1_splits), read(a.d1_review), read(a.cave_review)
    d4 = read(a.d4_manifest)
    d4_uwa = read(a.d4_uwa) if a.d4_uwa.is_file() else []
    oracle = read(a.d1_oracle) if a.d1_oracle.is_file() else {}
    crop_rejections = read(a.d1_crop_rejections) if a.d1_crop_rejections.is_file() else {}
    uwa_samples = read(a.uwa_samples)
    registry = read(a.g4_protocol)
    g4_access = read(a.g4_access)
    if not g4_access['status'].startswith('completed'):
        raise ValueError('G4 exposure ledger must record actual completion')
    if digest(a.n1) != registry['input_sha256']['manifest.json']:
        raise ValueError('N1 manifest differs from the frozen G4 source')
    if digest(a.task) != read(a.d6_audit)['task_manifest_sha256']:
        raise ValueError('D3 manifest differs from prior D6 audit')
    existing = legacy_audit(a.task, a.n1)
    if existing['n1_training_overlap_by_task_split'] != {'train': 18, 'test': 6}:
        raise ValueError('expected D6 overlap evidence changed; audit before versioning')
    archive_sha = digest(a.uwa_archive) if a.uwa_archive.is_file() else None
    expected_archive = read(a.uwa_receipt)['sha256']
    archive_verified = archive_sha == expected_archive
    if archive_sha and not archive_verified:
        raise ValueError('UWA archive differs from acquisition receipt')
    source_inputs = {name: path for name, path in vars(a).items()
                     if isinstance(path, Path) and name not in ('output', 'private_output', 'uwa_root', 'cave_root')}
    receipts = {name: dict(available=path.is_file(), sha256=digest(path) if path.is_file() else None,
                           bytes=path.stat().st_size if path.is_file() else None)
                for name, path in source_inputs.items()}
    task_originals = {r['sample_id']: r for r in task if r['augmentation'] is None}
    d2_originals = {r['sample_id']: r for r in d2 if r['augmentation'] is None}
    lookup = {(r['dataset_id'], r['scene_id']): r for r in n1}
    d4_lookup = {r['scene_id']: r for r in d4}
    d4_subjects = {r['subject_id'] for r in d4_uwa}
    oracle_ids = set(oracle.get('sample_ids', []))
    family_history = collections.defaultdict(set)
    family_n1 = collections.defaultdict(set)
    for stage, index in [('D2', d2_originals), ('D3', task_originals)]:
        for r in index.values():
            if r.get('source_dataset',r.get('dataset_id'))=='UWA':
                family_history[r['subject_id']].add(stage+'_task_'+r['split']+'_subject_family')
                family_n1[r['subject_id']].add(stage+'_ROI_reconstructor_'+r['split']+'_subject_family')
    rows = []
    source_files = []
    archive = zipfile.ZipFile(a.uwa_archive) if archive_verified else None
    field_cache = {}
    try:
        for r in d1:
            if r['augmentation'] != 'none':
                continue
            sid, subject, acquisition = r['sample_id'], r['subject_id'], r['acquisition_id']
            if subject not in splits['subject_to_split'] or acquisition not in splits['acquisition_to_split']:
                raise ValueError('D1 source missing from original split map')
            if r['split'] != splits['subject_to_split'][subject] or r['split'] != splits['acquisition_to_split'][acquisition]:
                raise ValueError('D1 original split disagreement')
            source = uwa_samples.get(acquisition)
            if source is None or str(source['subject']) != subject or int(source['session']) != int(r['session_id']):
                raise ValueError('UWA acquisition identity not supported by source inventory')
            path = a.uwa_root/r['path']
            metadata = {}
            file_verified = False
            if path.is_file():
                actual = digest(path)
                if actual != r['sample_sha256']:
                    raise ValueError('D1 derivative file hash mismatch')
                with np.load(path, allow_pickle=False) as z:
                    metadata = json.loads(str(z['metadata_json']))  # No cube/pixel read.
                file_verified = True
                if metadata['subject_id'] != subject or metadata['acquisition_id'] != acquisition:
                    raise ValueError('D1 derivative identity mismatch')
                source_files.append(dict(kind='UWA_D1_derivative', sha256=actual))
            t = task_originals.get(sid)
            if t and (t['input_file_sha256'] != r['sample_sha256'] or t['crop_xywh'] != metadata.get('crop_xywh')):
                raise ValueError('D3 to D1 hash/coordinate link differs')
            if acquisition not in field_cache:
                field_cache[acquisition] = inspect_uwa(archive, subject, r['session_id']) if archive else (False, None, [])
            full, shape, members = field_cache[acquisition]
            history = sorted(family_history[subject])
            n1_exposure = sorted(family_n1[subject])
            for stage, index in [('D2', d2_originals), ('D3', task_originals)]:
                if sid in index:
                    history.append(stage+'_task_'+index[sid]['split'])
                    n1_exposure.append(stage+'_ROI_reconstructor_'+index[sid]['split'])
            if sid in oracle_ids:
                history.append('D1_oracle_'+r['split'])
            other = ['D1_source_and_quality_review']
            if subject in d4_subjects:
                other.append('D4_UWA_whole_field_diagnostic_subject')
            review = (d1_review['samples'].get(sid) if r['label']=='face'
                      else d1_review['negative_samples'].get(sid))
            accepted = r['status'].startswith('accepted_') and bool(review)
            rows.append(dict(sample_id=sid, dataset_id='UWA', original_source_id=acquisition,
                source_scene_or_acquisition_id=acquisition, subject_id=subject, session_id=r['session_id'],
                source_group_id='UWA_subject_'+subject, acquisition_family_id=acquisition,
                session_family_id=acquisition, source_file_sha256=archive_sha,
                source_file_sha256_verified=archive_verified,
                derivative_sha256=r['sample_sha256'] or None, derivative_verified=file_verified,
                original_split=r['split'], historical_task_exposure=history, n1_exposure=n1_exposure,
                a6_exposure='absent_from_frozen_A6_source_manifest_not_global_independence_proof',
                other_exposure=other, g4_consumed=False,
                independence_status='historically_exposed' if history else 'independence_unverified',
                identity_limits='subject IDs inherited; cross-ID person identity and shared background independence unverified',
                label=r['label'], label_review_status='legacy_reviewed' if accepted else 'unreviewed',
                legacy_review_text=review, legacy_reviewer='Agent; not human adjudication',
                recognizable_face_confirmed=bool(accepted and r['label']=='face'),
                presentation_type='physical_person' if accepted and r['label']=='face' else 'not_applicable' if r['label']=='non_face' else 'unverified',
                roi_xywh=metadata.get('crop_xywh',crop_rejections.get(sid,{}).get('proposed_crop_xywh')),
                roi_status='rejected_out_of_bounds_proposal' if sid in crop_rejections else 'legacy_native_coordinates',
                native_spatial_shape=shape,
                available_full_field=full, full_field_verified=full and archive_verified,
                full_field_kind='source_band_archive', full_field_members=members,
                radiometric_type='uncalibrated processed uint8 DN; divide by 255 for full-field simulation',
                quality_flags=[x for x in r['quality_flags'].split(';') if x],
                legacy_status=r['status'], semantic_ambiguity=not accepted,
                hard_negative_type=None, hardness_review_status='unassessed' if r['label']=='non_face' else 'not_applicable',
                license_local_research_recorded=a.license_evidence.is_file(), augmentation=None,
                private_source_path=str(a.uwa_archive), missing_derivative=not file_verified))
    finally:
        if archive: archive.close()
    cave_inventory = read(a.cave_inventory)
    cave_task = [r for r in task_originals.values() if r['dataset_id']=='CAVE']
    represented = {r['source_scene_or_acquisition_id'] for r in cave_task}
    # Non-admitted CAVE scenes remain explicit review records, not invented labels.
    candidates = cave_task + [dict(sample_id='CAVE_review_'+r['scene_id'],
        source_scene_or_acquisition_id=r['scene_id'], split=None, class_label='ambiguous',
        crop_xywh=None, presentation_type='unverified') for r in cave_inventory if r['scene_id'] not in represented]
    inv = {r['scene_id']: r for r in cave_inventory}
    for r in candidates:
        scene = r['source_scene_or_acquisition_id']
        source = inv[scene]
        match = lookup.get(('CAVE', scene))
        consumed = bool(match and match['split']=='test')
        family = match['source_group_id'] if match else scene
        path = a.cave_root/source['cube_path'] if source.get('cube_path') else None
        present = bool(path and path.is_file())
        verified = False; shape = None; actual = None
        # Never open any consumed G4 source cube, even for header verification.
        if present and not consumed:
            actual = digest(path)
            if match and actual != match['source_sha256']:
                raise ValueError('CAVE full-field file differs from N1 provenance')
            if r.get('input_file_sha256') and actual != r['input_file_sha256']:
                raise ValueError('CAVE full-field file differs from D3 provenance')
            header_shape, dtype = npz_header(path)
            verified = header_shape == [21,512,512] and dtype=='uint16'
            shape = header_shape[-2:]
            source_files.append(dict(kind='CAVE_full_field', sha256=actual))
        elif present:
            shape = source.get('shape', [None,None])[-2:]
        history = []; n1_exp = []
        for stage, index in [('D2', d2_originals), ('D3', task_originals)]:
            if r['sample_id'] in index:
                history.append(stage+'_task_'+index[r['sample_id']]['split'])
                n1_exp.append(stage+'_ROI_reconstructor_'+index[r['sample_id']]['split'])
        if match:
            n1_exp.append({'train':'A6_train_family','val':'A6_model_selection_val','test':'G4_consumed_test'}[match['split']])
        if scene in d4_lookup:
            n1_exp.append('D4_'+d4_lookup[scene]['split'])
        accepted = r['sample_id'] in task_originals
        rows.append(dict(sample_id=r['sample_id'],dataset_id='CAVE',original_source_id=scene,
            source_scene_or_acquisition_id=scene,subject_id=None,session_id=None,
            source_group_id='CAVE_'+family,acquisition_family_id='CAVE_'+family,session_family_id=None,
            source_file_sha256=actual or (match or {}).get('source_sha256'),
            source_file_sha256_verified=verified,original_archive_sha256=source.get('source_sha256'),
            original_split=r['split'],historical_task_exposure=history,n1_exposure=n1_exp,
            a6_exposure=(match or {}).get('split','unverified'),g4_consumed=consumed,
            other_exposure=['D2_agent_content_review'] + (['D5_D5.4_train_val_diagnostics'] if match and match['split']!='test' else []),
            independence_status='known_overlap' if match and match['split'] in ('train','val') else 'historically_exposed' if consumed else 'independence_unverified',
            label=r['class_label'],label_review_status='legacy_reviewed' if accepted else 'ambiguous',
            legacy_review_text=r.get('label_review_status',cave_review.get(scene)),
            legacy_reviewer='Agent; not human adjudication',
            presentation_type=r['presentation_type'],recognizable_face_confirmed=r['class_label']=='face',
            roi_xywh=r.get('crop_xywh'),native_spatial_shape=shape,
            available_full_field=present,full_field_verified=verified,full_field_kind='native_full_cube',
            radiometric_type='approximate reflectance; uint16 / 65535',
            quality_flags=r.get('quality_flags',[]),semantic_ambiguity=not accepted,
            hard_negative_type=None,hardness_review_status='unassessed' if r['class_label']=='non_face' else 'pending_semantic_review' if not accepted else 'not_applicable',
            license_local_research_recorded=a.license_evidence.is_file(),augmentation=None,
            private_source_path=str(path) if path else None,
            verification_note='G4 source not opened' if consumed else 'NPY header and whole-file hash only'))
    rows = assign_roles(rows, set(task_originals))
    for r in rows:
        r['working_roi_xywh'] = native_to_working(r['roi_xywh'], r['native_spatial_shape']) if r['protocol_role'].startswith('development_') else None
    # Validate all historical derivatives inherit parent split and source identity.
    derivative_count = 0
    lineage=[]
    for stage,history in (('D2',d2),('D3',task)):
        originals = {r['sample_id']:r for r in history if r['augmentation'] is None}
        for r in history:
            if r['augmentation'] is None:continue
            parent=originals.get(r['sample_id'].removesuffix('_aug'))
            if parent is None or r['split']!='train' or parent['split']!='train' or r['parent_id']!=parent['parent_id']:
                raise ValueError('historical augmentation ancestry or split violation')
            if r['label']!=parent['label'] or r['source_sha256']!=parent['source_sha256']:
                raise ValueError('historical augmentation label/source changed')
            lineage.append(dict(stage=stage,sample_id=r['sample_id'],parent_sample_id=parent['sample_id'],
                original_source_id=parent['parent_id'],subject_id=parent.get('subject_id'),
                original_split=r['split'],augmentation=r['augmentation']))
            derivative_count += 1
    d1_parents={(r['acquisition_id'],r['label']):r for r in d1 if r['augmentation']=='none'}
    for r in d1:
        if r['augmentation']!='none':
            if r['split']!='train' or splits['subject_to_split'][r['subject_id']]!='train':
                raise ValueError('D1 augmentation crosses subject split')
            parent=d1_parents.get((r['acquisition_id'],r['label']))
            if parent is None or parent['subject_id']!=r['subject_id'] or parent['split']!=r['split']:
                raise ValueError('D1 derivative parent missing or inconsistent')
            lineage.append(dict(stage='D1',sample_id=r['sample_id'],parent_sample_id=parent['sample_id'],
                original_source_id=r['acquisition_id'],subject_id=r['subject_id'],
                original_split=r['split'],augmentation=r['augmentation']))
            derivative_count += 1
    meta = dict(start_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_receipts=receipts, source_files_verified=source_files,
        historical_derivatives_checked=derivative_count,derivative_lineage=lineage,
        d3_nonaugmented_records=len(task_originals), d1_original_records=sum(r['augmentation']=='none' for r in d1),
        uwa_acquisitions_in_source_inventory=len(uwa_samples),
        uwa_subjects_in_original_split_map=len(splits['subject_to_split']),
        g4_scene_count=sum(r['split']=='test' for r in n1),
        g4_arrays_opened=False, pixel_arrays_decoded=False, classifier_training=False,
        cave_review=cave_review, d6_audit=existing,
        private_paths={k:str(v) for k,v in vars(a).items() if isinstance(v,Path)})
    return rows, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    names = ('task','d2','n1','d1-manifest','d1-splits','d1-review','d1-oracle','d1-crop-rejections','uwa-root',
             'uwa-samples','uwa-archive','uwa-receipt','cave-root','cave-inventory','cave-review',
             'd4-manifest','d4-uwa','g4-protocol','g4-access','d6-audit','license-evidence',
             'output','private-output')
    for name in names:p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists() or a.private_output.exists():
        raise FileExistsError('outputs must be new directories')
    rows, meta = build(a)
    publish(rows, meta, a.output, a.private_output)


if __name__=='__main__':main()
