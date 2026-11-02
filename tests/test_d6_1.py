"""Synthetic metadata only: no research images, GPU or external data required."""
import copy
import json
import pytest
from experiments.d6_1.validate_groups import (
    validate_groups, assign_roles, label_status, final_reasons, full_field_eligible,
    native_to_working, validate_roi, metric_eligibility, load_development, digest)


def row(**kw):
    r=dict(sample_id='r1',dataset_id='UWA',original_source_id='acq1',
        source_scene_or_acquisition_id='acq1',subject_id='person1',session_id='1',
        source_group_id='person1',acquisition_family_id='acq1',session_family_id='session1',
        source_file_sha256='verified-fixture-hash',source_file_sha256_verified=True,
        original_split='train',protocol_role='development_train',historical_task_exposure=[],
        n1_exposure=[],other_exposure=[],g4_consumed=False,independence_status='verified_independent',
        label='face',label_review_status='legacy_reviewed',presentation_type='physical_person',
        recognizable_face_confirmed=True,semantic_ambiguity=False,roi_xywh=[0,0,100,100],
        native_spatial_shape=[1024,1024],available_full_field=True,full_field_verified=True,
        full_field_kind='source_band_archive',license_local_research_recorded=True,augmentation=None)
    r.update(kw)
    return r


def separate(**kw):
    return row(sample_id='r2',subject_id='person2',source_group_id='person2',
               original_source_id='acq2',source_scene_or_acquisition_id='acq2',
               acquisition_family_id='acq2',session_family_id='session2',**kw)


def test_same_uwa_subject_never_crosses_splits():
    b=separate(protocol_role='development_val');b['subject_id']='person1'
    with pytest.raises(ValueError,match='crosses'):validate_groups([row(),b])


@pytest.mark.parametrize('field',['acquisition_family_id','session_family_id'])
def test_acquisition_and_session_family_never_cross(field):
    b=separate(protocol_role='development_val');b[field]=row()[field]
    with pytest.raises(ValueError,match='crosses'):validate_groups([row(),b])


def test_same_cave_scene_multiple_rois_stay_together():
    a=row(dataset_id='CAVE');b=copy.deepcopy(a)
    b.update(sample_id='r2',source_group_id='different_alias',protocol_role='development_test',roi_xywh=[100,100,50,50])
    with pytest.raises(ValueError,match='crosses'):validate_groups([a,b])


def test_conservative_cave_positive_family_cannot_split():
    a=row(dataset_id='CAVE',original_source_id='face',source_group_id='face_family')
    b=separate(dataset_id='CAVE',protocol_role='development_val');b['source_group_id']='face_family'
    with pytest.raises(ValueError,match='crosses'):validate_groups([a,b])


def test_augmented_positive_cannot_enter_eval():
    r=row(protocol_role='development_val',augmentation={'flip':True})
    with pytest.raises(ValueError,match='augmentation'):validate_groups([r])


@pytest.mark.parametrize('exposure',['A6_train_family','A6_model_selection_val','D3_ROI_reconstructor_train_subject_family'])
def test_n1_exposure_disqualifies_formal_test(exposure):
    with pytest.raises(ValueError,match='ineligible'):
        validate_groups([row(protocol_role='final_test',n1_exposure=[exposure])])


def test_g4_consumed_cannot_be_fresh():
    with pytest.raises(ValueError,match='ineligible'):
        validate_groups([row(protocol_role='final_test',g4_consumed=True)])


def test_missing_source_cannot_become_independent():
    r=row(original_source_id=None,independence_status='independence_unverified')
    assert 'missing_source_identity' in final_reasons(r)
    assert 'independence_not_verified' in final_reasons(r)


def test_clear_printed_face_remains_positive():
    assert label_status(row(presentation_type='printed_photo'))=='accepted_legacy_semantics'
    with pytest.raises(ValueError,match='printed'):
        label_status(row(presentation_type='printed_photo',label='non_face'))


def test_ambiguous_face_like_content_stays_review_only():
    r=row(semantic_ambiguity=True,label_review_status='ambiguous')
    assert assign_roles([r],{'r1'})[0]['protocol_role']=='review_only'
    assert label_status(r)=='ambiguous_review_required'


def test_single_class_metrics_are_undefined():
    r=metric_eligibility(['non_face']*5)
    assert not r['binary_metrics_allowed']
    assert all(r[k] is None for k in ('balanced_accuracy','macro_f1','auroc','auprc'))


@pytest.mark.parametrize('roi',[[-1,0,10,10],[0,0,1025,10],[0,0,0,3],[.5,0,5,5],[0,float('nan'),5,5]])
def test_roi_bounds_native_frame(roi):
    with pytest.raises(ValueError):validate_roi(roi,[1024,1024])


def test_mapping_is_declared_full_field_geometry():
    assert native_to_working([147,49,605,605],[1024,1024])==[73,24,303,303]


@pytest.mark.parametrize('changes',[{'native_spatial_shape':[32,32]},
    {'full_field_kind':'upsampled_roi'},{'full_field_verified':False},
    {'source_file_sha256_verified':False}])
def test_legacy_roi_cannot_impersonate_full_field(changes):
    assert not full_field_eligible(row(**changes))


def test_fixed_assignment_deterministic_and_preserves_history():
    rows=[row(),separate(original_split='val')]
    a=assign_roles(rows,{'r1','r2'});b=assign_roles(list(reversed(rows)),{'r2','r1'})
    assert a==b
    assert [r['protocol_role'] for r in a]==['development_train','development_val']
    assert rows[1]['original_split']=='val'


def test_unknown_independence_not_upgraded_by_assignment():
    r=assign_roles([row(independence_status='independence_unverified')],{'r1'})[0]
    assert r['independence_status']=='independence_unverified'
    assert r['protocol_role']=='development_train'
    assert r['final_holdout_exclusion_reasons']


def test_frozen_manifest_and_lock_tampering_rejected(tmp_path):
    manifest=tmp_path/'manifest.json';lock=tmp_path/'lock.json'
    manifest.write_text(json.dumps([row()]))
    lock.write_text(json.dumps({'private_manifest_sha256':digest(manifest)}))
    sha=digest(lock)
    assert len(load_development(manifest,lock,'development_train',expected_lock_sha256=sha))==1
    manifest.write_text(json.dumps([row(label='non_face')]))
    with pytest.raises(ValueError,match='manifest changed'):
        load_development(manifest,lock,'development_train',expected_lock_sha256=sha)
    lock.write_text(json.dumps({'private_manifest_sha256':digest(manifest)}))
    with pytest.raises(ValueError,match='lock changed'):
        load_development(manifest,lock,'development_train',expected_lock_sha256=sha)


def test_final_access_denied_before_any_file_read(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('file access attempted')
    monkeypatch.setattr('pathlib.Path.open',forbidden)
    with pytest.raises(PermissionError,match='final'):
        load_development(tmp_path/'final.json',tmp_path/'lock.json','final_test',expected_lock_sha256='x')


def test_legacy_test_exposure_blocks_final_even_without_n1():
    with pytest.raises(ValueError,match='ineligible'):
        validate_groups([row(protocol_role='final_test',historical_task_exposure=['D3_task_test'])])
