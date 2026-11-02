"""Generate aggregate-only reports and a hash-bound private sample protocol."""
import collections
import json
from pathlib import Path
from experiments.d6_1.validate_groups import digest, validate_groups, metric_eligibility


def write_json(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def counts(rows, fields):
    c=collections.Counter(tuple(r.get(k) for k in fields) for r in rows)
    return [dict(zip(fields,key),count=value) for key,value in sorted(c.items(),key=lambda kv:str(kv[0]))]


def publish(rows, meta, public, private):
    public.mkdir(parents=True,exist_ok=False)
    private.mkdir(parents=True,exist_ok=False)
    write_json(private/'downstream_manifest_v1.json',rows)
    write_json(private/'source_exposure_audit.json',meta)
    write_json(private/'derivative_lineage.json',meta['derivative_lineage'])
    manifest_sha=digest(private/'downstream_manifest_v1.json')
    active=[r for r in rows if r['protocol_role'].startswith('development_')]
    training=[r for r in active if r['protocol_role']=='development_train']
    validation=[r for r in active if r['protocol_role']=='development_val']
    ready=all(set(r['label'] for r in part)=={'face','non_face'} for part in (training,validation))
    status='READY_FOR_DEVELOPMENT' if ready else 'BLOCKED_BY_DATA'
    source_stats=[]
    for ds in sorted({r['dataset_id'] for r in rows}):
        part=[r for r in rows if r['dataset_id']==ds]
        admitted=[r for r in active if r['dataset_id']==ds]
        source_stats.append(dict(dataset_id=ds,original_candidate_records=len(part),
            original_sources=len({r['original_source_id'] for r in part}),
            conservative_groups=len({r['source_group_id'] for r in part}),
            active_development_rois=len(admitted),
            active_development_sources=len({r['original_source_id'] for r in admitted}),
            active_development_groups=len({r['source_group_id'] for r in admitted}),
            candidate_labels=dict(collections.Counter(r['label'] for r in part)),
            admitted_labels=dict(collections.Counter(r['label'] for r in admitted)),
            independence_status=dict(collections.Counter(r['independence_status'] for r in part)),
            full_fields_verified=sum(r['full_field_verified'] for r in part),
            source_records_without_verified_full_field=sum(not r['full_field_verified'] for r in part),
            missing_derivative_records=sum(r.get('missing_derivative',False) for r in part),
            formal_test_ineligible=len(part)))
    known=collections.Counter(tag for r in rows for tag in r['n1_exposure'])
    history=collections.Counter(tag for r in rows for tag in r['historical_task_exposure'])
    reasons=collections.Counter(reason for r in rows for reason in r['final_holdout_exclusion_reasons'])
    source_audit=dict(scope='All non-augmented D1 candidate records plus D3 CAVE ROIs and remaining CAVE inventory review records',
        counting='ROI/candidate counts and source/family counts are distinct; counts of overlapping reasons are nonexclusive',
        by_source=source_stats,n1_exposure_counts=dict(known),historical_task_exposure_counts=dict(history),
        other_exposure_counts=dict(collections.Counter(tag for r in rows for tag in r['other_exposure'])),
        final_exclusion_counts=dict(reasons),private_manifest_sha256=manifest_sha,
        verified_independent_records=sum(r['independence_status']=='verified_independent' for r in rows),
        a6_absent_uwa_is_not_proof=True,d6_baseline_audit=meta['d6_audit'])
    write_json(public/'source_exposure_audit.json',source_audit)
    write_json(public/'input_inventory.json',dict(start_commit=meta['start_commit'],
        source_receipts=meta['source_receipts'],source_files_verified=meta['source_files_verified'],
        private_manifest_sha256=manifest_sha,local_availability_audited=True,
        missing_inputs=[k for k,v in meta['source_receipts'].items() if not v['available']],
        checks={k:meta[k] for k in ('historical_derivatives_checked','d3_nonaugmented_records','d1_original_records',
            'uwa_acquisitions_in_source_inventory','uwa_subjects_in_original_split_map','g4_scene_count',
            'g4_arrays_opened','pixel_arrays_decoded','classifier_training')}))
    policy=dict(version='D6.1-label-v1',positive='recognizable human face in candidate region, including a clear photographed face',
        negative='no recognizable human face under the same definition',
        excluded_tasks=['liveness','identity','real-versus-photo','authenticity','body detection'],
        positive_presentations=['physical_person','printed_photo','screen_displayed_face','other_recognizable_human_face'],
        allowed_review_states=['legacy_reviewed','agent_reviewed','human_reviewed','human_adjudicated','unreviewed','ambiguous'],
        review_mapping='Prior Agent evidence becomes legacy_reviewed with original reviewer retained; never human_reviewed',
        ambiguous='unresolved statue/toy/mask/painting semantics require content review or human adjudication; never infer labels from names',
        hard_negative_rule='confirmed non_face AND explicit content-supported hard_negative_type; no photo-as-negative shortcut',
        hard_negative_types=['nonhuman_face_structure','pareidolia_texture','head_or_eye_like_nonhuman_object',
                             'detector_false_positive_background','confirmed_negative_low_visibility'],
        cave_positive_preservation=dict(rois=3,source_scenes=2,conservative_groups=1,
            physical_person=2,printed_photo=1,coordinates_unchanged=True,train_only=True),
        newly_reviewed_images=0,new_human_adjudications=0)
    write_json(public/'label_policy_v1.json',policy)
    annotation=dict(review_states=counts(rows,['dataset_id','label_review_status']),
        presentation_counts=counts(rows,['dataset_id','presentation_type']),
        active_development_labels=counts(active,['dataset_id','label']),
        cave_positive_policy=policy['cave_positive_preservation'],
        raw_historical_labels_unchanged=True,all_original_coordinates_preserved=True,
        review_evidence='D1 review_decisions, D2 CAVE decisions and D3 actual-image review records; no new visual inspection')
    write_json(public/'annotation_audit.json',annotation)
    review_rows=[dict(sample_id=r['sample_id'],dataset_id=r['dataset_id'],
                     original_source_id=r['original_source_id'],roi_xywh=r['roi_xywh'],
                     original_label=r['label'],review_status=r['label_review_status'],
                     reason='semantic/quality review' if r['semantic_ambiguity'] else 'hardness unassessed',
                     legacy_evidence=r['legacy_review_text'])
                 for r in rows if r['semantic_ambiguity'] or r['label']=='non_face']
    write_json(private/'review_queue.json',review_rows)
    write_json(public/'review_queue.json',dict(private_items=len(review_rows),
        by_source_and_reason=counts(review_rows,['dataset_id','reason']),
        identities_and_coordinates='private review_queue.json only',
        no_automatic_adjudication=True))
    negatives=[r for r in rows if r['label']=='non_face']
    hard=dict(confirmed_hard_negatives=0,
        unassessed_negative_counts=counts(negatives,['dataset_id','protocol_role']),
        confirmed_types=[],evidence='Legacy negative acceptance establishes semantic absence, not measured difficulty',
        ambiguous_cave_source_records=sum(r['dataset_id']=='CAVE' and r['semantic_ambiguity'] for r in rows),
        sources_requiring_human_semantic_review=3,format_or_label_review_sources=1,
        mining='training groups only; val and final candidates frozen; no error-based test-to-train recycling',
        missing_type_metrics='hard-negative false-positive rate is null with reason no confirmed denominator')
    write_json(public/'hard_negative_inventory.json',hard)
    write_json(private/'hard_negative_inventory.json',[dict(sample_id=r['sample_id'],
        original_source_id=r['original_source_id'],source_group_id=r['source_group_id'],
        roi_xywh=r['roi_xywh'],label=r['label'],hard_negative_type=None,
        review_status=r['hardness_review_status']) for r in negatives])
    balance=[]
    for ds in sorted({r['dataset_id'] for r in active}):
        for role in ('development_train','development_val','development_test'):
            part=[r for r in active if r['dataset_id']==ds and r['protocol_role']==role]
            balance.append(dict(dataset_id=ds,role=role,rois=len(part),
                original_sources=len({r['original_source_id'] for r in part}),
                groups=len({r['source_group_id'] for r in part}),
                labels=dict(collections.Counter(r['label'] for r in part)),
                groups_by_label={label:len({r['source_group_id'] for r in part if r['label']==label})
                                 for label in ('face','non_face')},
                confirmed_hard_negatives=0,metrics=metric_eligibility([r['label'] for r in part])))
    write_json(public/'source_class_balance.json',balance)
    split=dict(version='D6.1-split-v1',assignment='sorted sample ID; D3 original train/val/test preserved as development roles; all other candidates review_only',
        role_counts=counts(rows,['protocol_role']),private_manifest_sha256=manifest_sha,
        uwa_group='subject_id includes all sessions, exposures, ROIs, derivatives; merge known aliases before version freeze',
        cave_group='N1 source family; face and photo_and_face together; toy/chart family together',
        additional_sources='KAIST not admitted as labelled task data',
        family_limits='UWA cross-ID identities/shared backgrounds unverified; no global independence assertion',
        historical_test='development_test only; cannot select this version thresholds or checkpoint from it',
        augmentation='no augmentation in v1; all historical augmented examples excluded, ancestry validated',
        final_holdout_status='not_available',new_random_roi_split=False)
    write_json(public/'split_protocol_v1.json',split)
    integrity=validate_groups(rows)
    integrity.update(original_manifests_unchanged=True,historical_derivatives_checked=meta['historical_derivatives_checked'],
        original_cave_positive_coordinates_preserved=True,all_native_to_512_coordinates_deterministic=True,
        limits='partition integrity is not proof of unknown cross-ID identity or background independence')
    write_json(public/'group_integrity_audit.json',integrity)
    holdout=dict(final_holdout_status='not_available',eligible_records=0,
        excluded_candidates=len(rows),exclusion_counts=dict(reasons),
        consumed_g4_scenes=14,all_historical_task_tests_ineligible=True,
        unreviewed_unmatched_data_not_promoted=True,
        additional_uwa_candidates='D1 content/quality exposed; cross-session subjects overlap; not fresh final data')
    write_json(public/'holdout_eligibility.json',holdout)
    # Constants below are selected before any classifier training or scores.
    protocol=dict(version='D6.1-classification-v1',status='frozen_development_protocol_final_test_disabled',
        scope='simulation-only fixed candidate Face/Non-face feasibility; no physical spectral or real UDC claim',
        label_policy='label_policy_v1.json',split_protocol='split_protocol_v1.json',
        label_encoding=dict(non_face=0,face=1),
        private_manifest_sha256=manifest_sha,source_class_counts=balance,
        final_holdout_status='not_available',classification_training_performed=False,
        n1=dict(arm='s1c',seed=20261021,checkpoint_sha256='b20d2b136316aeda2484295106ee0610c18c0ece6821f7c6ef3502f49b349f9f',
            selection='first pre-existing seed, fixed before classifier outcomes; not selected from G4 scores',
            frozen=True,output_shape=[1,21,512,512],labels_and_classification_gradients_prohibited=True,
            inference_bundle_sha256='1e3cd50cfd4faf52cf2356fa7c3404ca7624f5a81ee9fbe5af846e52aa797635',
            original_physics_sha256='dec5b2d71595e38128f184e8f2a5d464ca1bd07906613c9926f62d23124d7050'),
        full_field=dict(uwa='F_06..F_26 uint8 original full 1024x1024 bands /255, area-resize whole field to512; no per-ROI enlargement',
            cave='native 21x512x512 uint16 /65535; no per-scene energy normalization',
            roi='native xywh; floor mapped left/top, ceil right/bottom, fixed half-open512 bounds; reconstruct before ROI',
            normalization='no clipping or per-observation fitted scaling; feature means/SD fitted on training only',
            physical_limits='fixed signed shape-only operator; UWA sequential motion and unequal native radiometry remain',
            noise=dict(absolute_sigma=0.017329356670379638,seed=20261102,
                rule='UTF8(dataset_id+":"+original_source_id) SHA256 first8 bytes little-endian integer; NumPy default_rng(SeedSequence([20261102,integer])); normal(0,sigma,[1,1,512,512]) cast float32; one cached realization per source shared across arms and seeds')),
        candidates=dict(source='unchanged D3 candidate coordinates verified against D1/source evidence; fixed proposals are privileged, not detector-derived',
            detector='not used; no detector recall claim; future detector protocol must retain misses and false positives',
            paired_across_all_arms=True),
        arms=[dict(id='B0',input='same scalar observation only',features='global mean plus row-major3x7 integer-partition spatial cell means',dimensions=22,parameters=23),
              dict(id='B1',input='B0 features + frozen N1',features='B0 22 +21 band means in identical ROI',dimensions=43,parameters=44),
              dict(id='B0_capacity',input='same scalar observation only',features='global mean plus row-major6x7 spatial cell means',dimensions=43,parameters=44)],
        pooling='For HxW ROI, row cell bounds floor(i*H/rows)..floor((i+1)*H/rows), likewise columns; reject empty cells; mean signed values, no interpolation',
        capacity_control='B0_capacity mandatory; report23 versus44 parameters; B1 advantage over B0 alone does not isolate feature content from capacity',
        training=dict(head='one affine logit with sigmoid; bias included; no hidden layers',device='cpu',
            classifier_precision='float64; features pooled in float64 from float32 scalar/N1 outputs; deterministic algorithms',
            seeds=[20261102,20261103,20261104],optimizer='AdamW',learning_rate=0.001,weight_decay=0.01,
            betas=[0.9,0.999],epsilon=1e-8,weight_decay_bias=False,max_epochs=300,patience=30,batch='full training partition',
            initialization='all weights normal(0,0.01), bias0 under each fixed seed',
            loss='binary cross entropy; each training source group equal total weight then inverse weighted class frequency; weights normalized to mean1',
            validation_weights='inverse count of ROIs in each validation group, normalized to mean1; no class balancing fitted to validation',
            standardization='training-weighted per-feature mean and population SD; SD<1e-8 replaced by1; reuse for val/test',
            checkpoint='minimum group-weighted validation BCE; ties within1e-12 choose earliest epoch; improvement must exceed1e-8 to reset patience',
            threshold=0.5,augmentation='none',threshold_search=False,budget_extension=False),
        metrics=['confusion counts','positive recall','negative recall','balanced accuracy','macro F1','AUROC','AUPRC',
                 'hard-negative false-positive rate','per-independent-group errors'],
        reporting=['pooled','UWA within-source','CAVE within-source where both classes available',
                   'source-by-class cross table','group-weighted and ROI-weighted results',
                   'three seed descriptive means and spread; groups nested; no pooled ROI significance'],
        single_class_policy='only present-class recall/error; balanced accuracy, macro F1, AUROC and AUPRC null with reason',
        ranking_metrics='AUROC average-rank ties; AUPRC average precision with recall-step weighting, not trapezoidal PR area; report ROI- and group-weighted variants',
        source_confounding=dict(source_only='predict class from training source-specific class prevalence, threshold0.5, same group weights; unseen source uses train pooled prior',
            source_probe='separate23-parameter affine source classifier using B0 features, UWA=1 CAVE=0, identical budget/group-weight procedure/threshold, no effect on task selection',
            interpretation='pooled improvements not sufficient; inspect within-source gains, spatial-scale and background shortcuts'),
        hard_negatives=hard,
        access=dict(development_loader='load_development requires externally recorded expected lock SHA256 and matching private manifest digest; refuses final_test before any file IO',
            future_final='disabled in v1; new version must freeze independent groups and record hashes, code, exact checkpoints and exclusive started/completed/failed ledger before first access',
            reuse='any final results used for mining/development consume that holdout and require a new version'),
        forbidden_claims=['real RGB from repeated scalar channels','live-versus-spoof classification','identity recognition',
                          'fresh test from D3/G4 reuse','physical extra spectral information','Axon20 or universal UDC performance'])
    write_json(public/'classification_protocol_v1.json',protocol)
    decision=dict(status=status,development_ready=ready,final_test_ready=False,final_holdout_status='not_available',
        can_start_C_in_next_stage=ready,current_task_trains_C=False,
        development_scope='historically exposed, fixed-candidate, full-field scalar simulation; no detector or fresh final claim',
        execution_gates=['implement full-field loader and exact-state frozen N1 validation',
            'verify hash-bound private manifest and protocol lock before loading',
            'verify declared feature pooling/standardization/capacity counts using CPU fixtures'],
        limitations=['no confirmed hard-negative inventory','CAVE val/test have no positive source',
            'six CAVE historical task test ROIs overlap A6 training','UWA cross-subject shared background is not independent',
            'final independent acquisition set absent'])
    write_json(public/'readiness_decision.json',decision)
    table='\n'.join(f"| {r['dataset_id']} | {r['original_candidate_records']} | {r['original_sources']} | {r['conservative_groups']} | {r['active_development_rois']} |" for r in source_stats)
    bal='\n'.join(f"| {r['dataset_id']} | {r['role']} | {r['labels'].get('face',0)} | {r['labels'].get('non_face',0)} | {r['groups']} |" for r in balance)
    documents={
        'source_exposure_analysis.md':f'''# Source and exposure audit

All D1 non-augmented candidates were audited, not just the116 paired D3 UWA ROIs.
CAVE includes29 D3 ROIs plus four source-level review entries. Candidate counts
are not independent observations. No new identity inference was performed.

| Source | Original candidates | Acquisition/scenes | Conservative groups | Admitted development ROIs |
| --- | ---: | ---: | ---: | ---: |
{table}

All116 D3 UWA rows map back through sample hash and native ROI to D1 acquisitions,
the original subject split and the verified UWA archive. Their absence from the
A6 manifest is explicitly scoped; D2/D3 reconstructor and task exposures remain.
D4's16 subject-level whole-field diagnostic records are joined by subject,
including all sessions conservatively. Cross-ID identity aliases and common
background independence remain unverified; exact file hashes cannot establish them.

CAVE family joins reproduce18 task-train and6 task-test ROIs in A6 training
families, plus5 in A6 model-selection validation. The four additional CAVE
source records belong to consumed G4 test scenes; their source cubes were not
opened. All14 G4 scenes, including KAIST, remain consumed; KAIST has no admitted
task labels. Source files are hashed, UWA PNG headers and CAVE NPY headers checked,
and D1 metadata read without decoding pixels. No G4 tensor archive was opened.

Machine-readable counts and nonexclusive exclusion reasons are in
`source_exposure_audit.json`; input availability/digests are in `input_inventory.json`.
Full identities, ancestry, source paths and per-sample audit stay in the private
output. Zero candidates have verified fresh final independence. Unmatched or
additional D1 candidates are review-only, never promoted by absence of evidence.
''',
        'hard_negative_coverage.md':'''# Hard-negative coverage

There are zero confirmed, typed hard negatives. Existing semantic non-face
acceptance does not establish difficulty. Counts by source and development role
are in `hard_negative_inventory.json`; per-source/ROI details are private.

The legacy CAVE review explicitly excluded three sources for humanoid toy,
sculpture or animal/cartoon face ambiguity. Preserve these as semantic review
items; do not classify them from scene names. The fourth non-admitted source
has a format/annotation gap. These are not verified negatives and have no newly
reviewed ROI. Clear printed human faces stay positive. No new image review or
human adjudication occurred.

UWA's reviewed negatives are clothing/background with possible scale, common
backdrop and illumination shortcuts. They share acquisitions/subjects with
positive ROIs and inherit the same partitions. No difficulty type was supplied
by historical review. CAVE ordinary negatives also need explicit content-based
hardness review. Neither source supports a hard-negative FPR denominator yet.

Mining is permitted on development-train only. Candidate geometry is shared
across B0/B1/capacity control. Validation/final prediction-driven mining requires
a new version and consumes any previously independent test. Future detection
must record missed positives and false proposals, not only successful boxes.
''',
        'data_gap_report.md':'''# Data gaps and minimum next evidence

Development can use the legacy admitted groups; this does not create a new
independent final set. CAVE validation and development-test each lack any
positive group, and the only CAVE positive family is train-only. At least one
new independent positive family per validation and final partition is needed
before either can support CAVE within-source two-class assessment; never move
the existing face family to fill this gap.

Fresh final data: zero eligible positive or negative groups. A bare engineering
floor is at least two new independent source groups supporting each class, with
at least two independently reviewed hard-negative groups. Groups may contain
both classes, but individual ROIs never count as independent groups. This is
a minimum for non-singleton reporting, not a powered scientific sample size;
set a justified larger acquisition budget in a new preregistration. Cross-source
hard-negative claims need confirmed examples from at least two sources; currently
both UWA and CAVE have zero confirmed types. Augmentation cannot fill these gaps.

All admitted D3 ROIs have verified native full-field sources. Legacy32x32
arrays are ineligible inputs. Review-only records with missing/invalid ROI or
unverified full-field compatibility are counted in the audit; unavailable
derivatives are explicitly recorded privately, not reconstructed or invented.
Cross-ID UWA aliases and shared-background independence are unverified. D1/D2/D3
task tests, D4 diagnostics and all14 G4 scenes cannot be a new final reserve.

For future Axon20 data, record device/sensor/calibration version, acquisition
and session ID, subject/family aliases privately, original file SHA256, timestamp,
display state/timing, exposure/gain, native resolution/orientation/CFA, black and
white levels, radiometric units, wavelength/PSF/noise calibration provenance,
candidate coordinates and transforms, label semantics/reviewer/version,
presentation and confirmed difficulty type, consent/license/use restrictions,
all prior training/selection/test access, and a locked source-group split.
Keep all sessions/derivatives of the same subject and related scenes in one
partition. Separate calibration/development/final acquisitions; hold final
truth and arrays behind a separate access ledger. No Axon20 data was acquired
or assumed available here, and no phone specification is inferred.
''',
        'classification_protocol_v1.md':'''# Frozen D6.1 classification protocol

`classification_protocol_v1.json` is the exact executable specification; the
protocol lock binds it, source audits, code and private sample manifest hashes.
This version permits development experiments only; final testing is disabled.
No classifier was trained while preparing this protocol.

The task is recognizable Face/Non-face, including printed faces. R and N1 use
the same scalar full-field observation. N1 is the pre-existing fixed first A6
seed, with original state hashes, frozen evaluation mode and no label input.
Process the full512 field before native-to-working ROI selection. UWA uses
original1024 bands downsampled as a full field, never enlarged32x32 ROI data.

B0 uses22 intensity features; B1 adds21 reconstructed band means. Their affine
heads have23 and44 parameters. A mandatory44-parameter scalar spatial-feature
control addresses capacity; merely beating B0 is insufficient to attribute a
benefit to the N1 representation. Train-only weighted normalization, CPU optimizer,
three seeds, budgets, patience, checkpoint rule and threshold0.5 are frozen in
JSON. No tuning against historical development-test scores within this version.

Report pooled and within-source metrics, group and ROI weights, source/class
cross tables, source-only prevalence baseline and separate source-identifiability
probe. CAVE single-class validation/test permits only negative recall/error;
balanced accuracy, macro-F1, AUROC and AUPRC are undefined there. Hard-negative
FPR is undefined until typed reviewed negatives exist. Candidate boxes are
privileged fixed annotations, not evidence of detector recall. Repeated scalar
channels cannot be called RGB. Any gain is development representation evidence,
not physically identified spectral information, liveness or real UDC performance.

Before a future final test, create a new protocol version with independent
unused sources and a once-only access ledger. Never relabel D3/G4 consumed data
as fresh. Every failure, reuse or error-driven development must be recorded.
''',
        'README.md':f'''# D6.1 downstream data and protocol — 2026-11-02

Decision: **{status}**. Development ready: **{str(ready).lower()}**.
Final test ready: **false**; `final_holdout_status = not_available`.
Started from `{meta['start_commit']}` on the dedicated D6.1 branch.
No classifier training, GPU training, image review or G4 cube access occurred.

| Source | Role | Face ROIs | Non-face ROIs | Conservative groups |
| --- | --- | ---: | ---: | ---: |
{bal}

These are existing historically exposed development groups, not verified fresh
independent groups. Native full-field files are available for all admitted ROIs;
the complete inventory also retains excluded/review-only D1 and CAVE candidates.
See [source/exposure audit](source_exposure_analysis.md) for distinct ROI,
acquisition and subject/family counts and missing/unknown evidence.

Face semantics remain consistent: recognizable photographed faces are positive.
The three original CAVE positive coordinates and historical Agent reviewer
status are preserved privately, with both scenes in one train-only family.
There are **zero confirmed hard negatives**; legacy semantic acceptance is not
difficulty annotation. Ambiguous source records remain queued, not relabelled.

All old D1/D2/D3 task tests and all14 consumed G4 scenes are ineligible for a new
final set. Six CAVE legacy task-test ROIs overlap A6 training; CAVE validation
and task-test have no positive group. UWA is mapped to original subject/session
evidence, but common background and cross-ID person independence remain unknown.

The next agent may implement C development training under the frozen
[classification protocol](classification_protocol_v1.md), after the declared
loader/feature/hash checks. Its result must be labelled historical development
feasibility; final-test, detector and hard-negative performance are not ready.
No new external data is assumed. [Data gaps](data_gap_report.md) specify minimum
new sources and the acquisition/identity/calibration/isolation fields required
for future Axon20 data. The device has not been used or validated in this phase.

Full per-sample manifest, annotations, review queue and local paths remain in
the private output directory. Published JSON contains aggregate statistics,
protocols and digests only. Inputs and all D2–D6 reports are unchanged; Meta
submodule pins are unchanged. See `input_inventory.json` and `protocol_lock.json`.

## Reproduce without opening images

Use the reconstruction research environment, `PYTHONPATH=src` and CPU only.
`python -m experiments.d6_1.audit_sources --help` lists required explicit inputs.
Provide D1 manifest/split/review/oracle/crop-rejection records, the UWA source
inventory/archive/receipt, CAVE inventory/review/full-field directory, D2/D3
manifests, D4 manifest/UWA diagnostic metadata, the N1 manifest, G4 protocol/access
ledger and D6 audit. Both output directories must be new. Point `--output` at
a staging report directory and `--private-output` at a Git-ignored private path.
Do not supply any G4 array file. No local data root is hardcoded in the modules.

For a frozen run, use `python -m experiments.d6_1.verify_frozen --report REPORT
--manifest PRIVATE_MANIFEST --lock-sha256 RECORDED_LOCK_HASH`. Record the lock
hash from the committed report; do not recompute it to accept a changed lock.
The checker verifies all public protocol bytes, implementation source hashes
and private sample bytes. A changed sample/protocol requires a new explicit
version; rerunning the builder cannot overwrite an existing output directory.

Verification:29 new synthetic metadata tests cover all15 requested failure modes;
the complete CPU/no-hardware suite passed173 tests. Native-file and manifest
hash checks, deterministic replay and publication privacy scans are recorded
in the accompanying validation receipt. No research classifier was trained.
'''}
    for name, text in documents.items():
        with (public/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    code_dir=Path(__file__).parent
    lock=dict(version='D6.1-lock-v1',private_manifest_sha256=manifest_sha,
        input_sha256={k:v['sha256'] for k,v in meta['source_receipts'].items()},
        public_file_sha256={p.name:digest(p) for p in sorted(public.iterdir())},
        implementation_sha256_lf={p.name:__import__('hashlib').sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                  for p in sorted(code_dir.glob('*.py'))},
        policy='Any changed samples, protocol or implementation requires an explicit new version; no silent overwrite')
    write_json(public/'protocol_lock.json',lock)
    print(json.dumps(dict(status=status,rows=len(rows),development_rois=len(active),
        final_holdout_status='not_available',private_manifest_sha256=manifest_sha)))
