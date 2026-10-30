"""Preparation and approval gates only; no workers, GPU or network."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_chamfer_variants_campaign import (ARMS, CAP_SECONDS, build_jobs,
                                                  estimate_campaign, prepare, verify_approval,
                                                  verify_calibration)


def plan(root=Path('/content/studies/chamfer-guidance-variants')):
    return dict(schema_version=1, session='tta-chamfer-guidance-variants-a100-20261030-r1',
                cap_seconds=18000, source_root=str(root / 'assets/source_snapshot'),
                asset_root=str(root / 'assets'), python='/usr/local/envs/3dd_tta_env/bin/python',
                research_root=str(root), drive_root='/content/drive/MyDrive/thesis/result/chamfer_guidance_variants_20261030',
                asset_manifest=dict(path=str(root / 'assets/manifest.json'), bytes=1, sha256='a'*64),
                source_receipt=dict(path=str(root / 'source_receipt.json'), bytes=1, sha256='b'*64),
                calibration_report=str(root / 'calibration.json'),
                datasets={dataset: dict(dataset_root='/data/' + dataset, label_path='/data/' + dataset + '/label.npy',
                                        pointmae_config='cfgs/' + dataset + '.yaml', pointmae_checkpoint='checkpoints/' + dataset + '.pth')
                          for dataset in ('modelnet40_c', 'scanobjectnn_c')},
                selections={stage: {dataset: dict(path=str(root / 'assets/ids' / (stage + '_' + dataset + '.json')),
                                                   bytes=1, sha256='c'*64)
                                     for dataset in ('modelnet40_c', 'scanobjectnn_c')}
                            for stage in ('smoke', 'development', 'confirmation')})


@pytest.mark.parametrize('stage,count', [('calibration', 2), ('smoke', 14), ('development', 10), ('confirmation', 30), ('timing', 30)])
def test_exact_independent_campaign_scope(stage, count):
    jobs = build_jobs(plan(), stage)
    assert len(jobs) == count
    assert all(job['command'][job['command'].index('--batch-size')+1] == '70' for job in jobs)
    for job in jobs:
        command = job['command']
        start = command.index('--backbones') + 1
        assert command[start:start+(1 if stage == 'calibration' else 4)] == (['pointmae'] if stage == 'calibration' else ['dgcnn', 'curvenet', 'pointnext', 'pointmae'])
    for job in jobs:
        command = job['command']
        assert command[command.index('--component-arm')+1] == 'chamfer'
        assert command[command.index('--spectral-weight')+1] == '0'
        assert '--spectral-loss-kind' not in command
        if stage in ('smoke', 'timing'):
            assert command[command.index('--corruptions')+1:command.index('--corruptions')+3] == ['gaussian', 'background']
        else:
            assert command[-2:] == ['--corruptions', 'all']
    assert {job['seed'] for job in jobs} == ({0, 1, 2} if stage == 'confirmation' else {0})


def test_paired_cache_namespaces_and_fresh_timing_rotation():
    p = plan()
    development = build_jobs(p, 'development')
    paths = {job['command'][job['command'].index('--paired-state-root')+1] for job in development}
    assert len(paths) == 1  # Dataset is a subdirectory inside the immutable cache.
    assert set(job['arm'] for job in development) == set(ARMS)
    confirmation = build_jobs(p, 'confirmation')
    assert len({job['command'][job['command'].index('--paired-state-root')+1] for job in confirmation}) == 3
    timing = build_jobs(p, 'timing')
    assert all('--paired-state-root' not in job['command'] for job in timing)
    assert [job['arm'] for job in timing[:5]] == list(ARMS)
    assert [job['arm'] for job in timing[10:15]] == list(ARMS[1:] + ARMS[:1])
    assert len({job['result_root'] for job in timing}) == 30


def test_smoke_contains_two_uncalibrated_zero_weight_shams_per_dataset():
    jobs = build_jobs(plan(), 'smoke')
    shams = [job for job in jobs if job['role'] == 'technical_sham']
    assert len(shams) == 4
    for job in shams:
        command = job['command']
        assert command[command.index('--semantic-weight')+1] == '0'
        assert '--calibration-report' not in command
        assert job['executable']


def test_prepare_writes_fresh_review_artifact_without_remote_actions(tmp_path):
    output = tmp_path / 'prepared'
    report = prepare(plan(), output)
    assert report['status'] == 'prepared_not_executed'
    assert report['inputs']['status'] == 'not_verified'
    assert sum(len(jobs) for jobs in report['stages'].values()) == 86
    assert any(not job['executable'] for job in report['stages']['smoke'])
    assert (output / 'campaign_commands.json').is_file()
    with pytest.raises(FileExistsError):
        prepare(plan(), output)


def test_fixed_calibration_resolves_both_rates_and_separate_semantic_coefficients():
    coefficients = dict(semantic_kl=.2, semantic_h=.3, weighted_scale=.5)
    jobs = build_jobs(plan(), 'development', coefficients, calibration_sha256='d'*64)
    for job in jobs:
        command = job['command']
        if job['arm'].startswith('semantic_'):
            expected = coefficients[job['arm']]
            assert command[command.index('--semantic-weight')+1] == str(expected)
        if job['arm'] == 'weighted':
            assert command[command.index('--eta')+1] == '0.005'
            assert command[command.index('--gamma')+1] == '0.005'
            assert command[command.index('--density-k')+1] == '10'
        if job['arm'] not in ('baseline', 'sparse'):
            assert command[command.index('--calibration-report-sha256')+1] == 'd'*64


def test_whole_campaign_estimate_and_hard_cap():
    estimate = estimate_campaign()
    assert estimate['nominal_seconds'] == 13780
    assert estimate['projected_seconds_with_reserve'] == 16958
    assert estimate['projected_seconds_with_reserve'] < CAP_SECONDS
    assert estimate['development_jobs'] == 10 and estimate['confirmation_jobs'] == 30


def test_approval_is_exact_plan_session_cap_and_hashed_explicit_reply(tmp_path):
    path = tmp_path / 'approval.json'
    value = dict(status='approved', plan_sha256='a'*64, session=plan()['session'], cap_seconds=CAP_SECONDS,
                 user_reply='Approve this entire 5-hour campaign')
    path.write_text(json.dumps(value))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_approval(path, digest, 'a'*64, plan())['status'] == 'approved'
    for bad_digest, bad_plan, mutation in [('b'*64, 'a'*64, {}), (digest, 'b'*64, {}),
                                           (None, 'a'*64, dict(user_reply='')),
                                           (None, 'a'*64, dict(cap_seconds=20000)),
                                           (None, 'a'*64, dict(session='peer-runtime'))]:
        path.write_text(json.dumps(dict(value, **mutation)))
        digest_now = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError):
            verify_approval(path, bad_digest or digest_now, bad_plan, plan())


def test_only_actual_label_free_frozen_rule_calibration_accepted(tmp_path):
    path = tmp_path / 'calibration.json'
    report = dict(schema_version=1, purpose='label_free_chamfer_calibration', status='verified_label_free_gpu',
                  loss_reduction='sum for Chamfer, weighted Chamfer, and per-object semantic losses',
                  target_semantic_update_ratio=.1, coefficients=dict(semantic_kl=.2, semantic_h=.3, weighted_scale=.5),
                  rule='semantic=0.1/median(candidate/C combined update norm); weighted_scale=1/median(weighted/C)',
                  ratios={candidate: dict(count=4200, missing=0, min=ratio, median=ratio, p90=ratio, max=ratio)
                          for candidate, ratio in (('semantic_kl', .5), ('semantic_h', 1/3), ('weighted', 2.))},
                  calibration_bundles={'modelnet40_c': {'sha256': 'a'*64}, 'scanobjectnn_c': {'sha256': 'b'*64}})
    path.write_text(json.dumps(report))
    assert verify_calibration(path)['coefficients'] == report['coefficients']
    report['status'] = 'synthetic_only'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_calibration(path)


def test_input_identity_preflight_accepts_cpu_only_source_seal_without_claiming_bulk_validation(tmp_path):
    from scripts.run_chamfer_variants_campaign import identity, verify_inputs
    value = plan(tmp_path)
    asset_root = tmp_path / 'assets'
    (asset_root / 'ids').mkdir(parents=True)
    files = {}
    for stage, datasets in value['selections'].items():
        for dataset in datasets:
            indices = list(range(2 if stage == 'smoke' else 70)) if stage != 'confirmation' else list(range(70, 210))
            path = Path(datasets[dataset]['path'])
            path.write_text(json.dumps(dict(dataset=dataset, n_examples=2468 if dataset == 'modelnet40_c' else 581,
                                           stage='heldout' if stage == 'confirmation' else stage, indices=indices)))
            datasets[dataset] = dict(path=str(path), **identity(path))
            files[path.relative_to(asset_root).as_posix()] = identity(path)
    manifest = Path(value['asset_manifest']['path'])
    manifest.write_text(json.dumps(dict(files=files)))
    value['asset_manifest'] = dict(path=str(manifest), **identity(manifest))
    receipt = Path(value['source_receipt']['path'])
    receipt.write_text(json.dumps(dict(status='sealed_CPU_only', asset_layout='overlay_only_reuses_exact_base_bulk')))
    value['source_receipt'] = dict(path=str(receipt), **identity(receipt))
    result = verify_inputs(value)
    assert result['status'] == 'verified_cpu_inputs'
    assert result['bulk_assets_verified'] is False
