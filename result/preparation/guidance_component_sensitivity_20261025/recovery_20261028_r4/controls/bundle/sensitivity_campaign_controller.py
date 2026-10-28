"""Plan-bound Agent B sensitivity stages; one approved campaign ledger."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path('/content/studies/guidance-components')
DRIVE = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
ASSETS = ROOT / 'prepared_sensitivity_v2/assets'
PYTHON = '/usr/local/envs/3dd_tta_env/bin/python'
SESSION = 'tta-guidance-components-a100-20261028-r4'
CAP = 19800
LEDGER_NAME = 'sensitivity_campaign_r4_ledger.json'
STAGES = ('calibrate', 'smoke', 'development', 'confirmation')
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')
KINDS = ('legacy_l2', 'huber', 'power')
GSD_METHOD = 'gsd_inspired_latent_spectral_guidance'
OLD_MANIFEST = '7012c30ad88222bbc5435d8508f781fc33385ed3976951c45f48da67f625ed75'
OLD_HELPER_SHA = '1b7410a33189481978df630c1de12278a6564f7d3a665ddae6658b48d02491ac'
OLD_DEPLOY_SHA = '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(path, expected):
    result = identity(path)
    require(result == {key: expected[key] for key in ('bytes', 'sha256')}, 'File identity mismatch: ' + str(path))
    return result


def owned(path, root=ROOT):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and not path.is_symlink()
            and not any(parent.is_symlink() for parent in path.parents), 'Unsafe owned path: ' + str(path))
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError('Path outside owned root: ' + str(path))
    return path


def read_reference(reference):
    path = owned(reference['path'])
    verify(path, reference)
    return json.loads(path.read_text())


def approval_verified(plan_sha, path=None):
    path = ROOT / 'sensitivity_gpu_approval.json' if path is None else Path(path)
    approval = json.loads(path.read_text())
    require(approval.get('status') == 'approved' and approval.get('phase_plan_sha256') == plan_sha
            and approval.get('session') == SESSION and approval.get('cap_seconds') == CAP
            and isinstance(approval.get('user_reply'), str) and bool(approval['user_reply'].strip()),
            'Explicit whole-campaign GPU approval for this exact plan is required')
    return approval


def accepted_stage(path, plan_sha):
    accepted = json.loads(Path(path).read_text())
    require(accepted.get('status') == 'accepted' and accepted.get('phase_plan_sha256') == plan_sha,
            'Independent accepted stage proof for this exact plan is required')
    archive = accepted['archive']
    require(str(archive['drive_path']).startswith(str(DRIVE) + '/'), 'Accepted evidence outside owned Drive')
    verify(archive['drive_path'], archive)
    return accepted


def cpu_preflight_verified(plan):
    parent = Path(plan['asset_root']).parent
    require((parent / '.complete').read_text().strip() == plan['manifest_sha256'], 'New16-head CPU preflight marker missing')
    receipt = json.loads((parent / 'accepted_cpu_preflight_receipt.json').read_text())
    state = receipt['state']
    require(state['status'] == 'complete' and state['strict_loads'] == 16 and state['cuda_initialized'] is False
            and state['manifest_sha256'] == plan['manifest_sha256'] and state['session'] == SESSION,
            'New original/GSD CPU source and checkpoint loads are incomplete')
    archive = receipt['archive']
    require(str(archive['drive_path']).startswith(str(DRIVE) + '/'), 'CPU proof outside owned Drive')
    verify(archive['local_path'], archive); verify(archive['drive_path'], archive)
    verify(parent / 'cpu_preflight.json', receipt['report_identity'])
    return receipt


def registry_verified(registry):
    trials = registry['trials']
    require(registry.get('schema_version') == 1 and len(trials) == 19
            and len({trial['id'] for trial in trials}) == len(trials), 'Expected18 family configurations plus original3DD')
    for family in ('chamfer', 'combined', 'spectral'):
        group = [trial for trial in trials if trial['family'] == family]
        require(len(group) == 6 and sum(trial['reuse'] is True for trial in group) == 1,
                'Each family requires six trials including one accepted baseline')
    require({trial['id'] for trial in trials if trial['reuse']} == {'c-base', 'cs-base', 's-base'}, 'Unregistered baseline reuse')
    for trial in trials:
        require(type(trial['reuse']) is bool and trial['id'] and Path(trial['id']).name == trial['id'], 'Invalid trial identity')
        require(trial['family'] in ('chamfer', 'combined', 'spectral', 'original3dd'), 'Unknown family')
        original = trial['family'] == 'original3dd'
        require(trial['method'] == ('original3dd' if original else 'gsd'), 'Method/family mismatch')
        for key in ('eta', 'gamma', 'p', 'chamfer_weight'):
            require(type(trial[key]) in (int, float) and math.isfinite(trial[key]) and trial[key] >= 0, 'Invalid trial scalar')
        require(0 < trial['p'] <= 1, 'Invalid retained Chamfer fraction')
        require(trial['chamfer_weight'] == (0. if trial['family'] == 'spectral' else 1.), 'Active Chamfer family changed')
        weight = trial['spectral_weight']
        if isinstance(weight, dict):
            require(set(weight) == {'loss_kind', 'rho'} and weight['loss_kind'] in KINDS
                    and weight['loss_kind'] == trial['spectral_loss_kind'] and weight['rho'] in (.1, 1.), 'Invalid frozen spectral calibration rule')
        else:
            require(type(weight) in (int, float) and math.isfinite(weight) and weight >= 0
                    and (weight > 0) == (trial['family'] in ('combined', 'spectral')), 'Invalid active spectral family')
        require(trial['spectral_loss_kind'] in (None, *KINDS), 'Unknown spectral loss')
        require(trial['family'] in ('combined', 'spectral') or trial['spectral_loss_kind'] is None, 'Inactive spectral objective')
        if original:
            require(trial['id'] == 'original3dd' and not trial['reuse'] and trial['eta'] == trial['gamma'] == .01
                    and trial['p'] == .95 and weight == 0., 'Original3DD fixed protocol changed')
    require(sum(trial['family'] == 'original3dd' for trial in trials) == 1, 'Missing original control')
    return {trial['id']: trial for trial in trials}


def calibration_verified(path, plan):
    verify(path, {'bytes': 99029, 'sha256': '82ebc1fdb2784a0663968a20d46909e3020e127310f237971382c78c40e211e0'})
    report = json.loads(Path(path).read_text())
    require(report.get('status') == 'verified_label_free_gpu' and report.get('manifest_sha256') == plan['manifest_sha256']
            and report.get('source_snapshot_sha256') == plan['source_snapshot_sha256']
            and report.get('legacy_inline_parity') is True and len(report.get('cache_records', [])) == 30,
            'Actual label-free GPU calibration and full legacy parity are required')
    require(set(report['coefficients']) == set(KINDS), 'Missing loss calibration')
    for value in report['coefficients'].values():
        require(type(value) in (int, float) and math.isfinite(value) and 1e-12 <= value <= 1e12, 'Invalid calibrated coefficient')
    return report


def spectral_weight(trial, calibration):
    value = trial['spectral_weight']
    if isinstance(value, dict):
        require(calibration is not None and calibration.get('status') == 'verified_label_free_gpu', 'Cannot resolve GPU weights from synthetic calibration')
        value = calibration['coefficients'][value['loss_kind']] * value['rho']
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid resolved spectral weight')
    return float(value)


def expected_protocol(trial, calibration):
    if trial['method'] == 'original3dd':
        return dict(gamma=.01, eta=.01, p=.95, total=100, scd_normalize=False)
    result = dict(M_max=800, beta=2., gamma=trial['gamma'], eta=trial['eta'], p=trial['p'], total=100,
                  loss_weights=dict(chamfer=trial['chamfer_weight'], spectral=spectral_weight(trial, calibration), invariant=0.),
                  use_static_style=False, dynamic_graph=False, graph_update_interval=3, skip_inactive_components=True)
    if trial['spectral_loss_kind'] is not None:
        result['spectral_loss_kind'] = trial['spectral_loss_kind']
    return result


def selection_path(plan, stage, dataset):
    key = 'confirmation' if stage == 'confirmation' else 'smoke' if stage == 'smoke' else 'development'
    return plan['id_manifests'][key][dataset]['path']


def cache_path(stage, seed):
    require(type(seed) is int and seed >= 0, 'Invalid seed')
    if stage == 'development':
        require(seed == 0, 'Development uses seed0')
        return ROOT / 'paired_state_v1_seed0'
    if stage == 'smoke':
        require(seed == 0, 'Smoke uses seed0')
        return ROOT / 'paired_sensitivity_smoke_v2_seed0'
    require(stage == 'confirmation' and seed in (0, 1, 2), 'Invalid confirmation seed')
    return ROOT / ('paired_sensitivity_confirmation_v2_seed%d' % seed)


def command_for(plan, trial, dataset, stage, seed, result_root, calibration):
    require(dataset in DATASETS and stage in ('smoke', 'development', 'confirmation'), 'Unknown command scope')
    require(type(seed) is int and (seed in (0, 1, 2) if stage == 'confirmation' else seed == 0), 'Unregistered command seed')
    data = Path('/content/3DD-TTA/data/modelnet40_c') if dataset == 'modelnet40_c' else Path(plan['asset_root']) / 'datasets/scanobjectnn_c'
    original = trial['method'] == 'original3dd'
    command = [PYTHON, '-u', str(Path(plan['asset_root']) / 'source_snapshot/scripts/eval_multibackbone_guidance.py'),
               '--dataset', dataset, '--dataset-root', str(data), '--label-path', str(data / 'label.npy'),
               '--asset-root', plan['asset_root'], '--asset-manifest-sha256', plan['manifest_sha256'],
               '--result-root', str(result_root), '--drive-root', str(DRIVE), '--seed', str(seed), '--method', trial['method'],
               '--backbones', *HEADS, '--batch-size', '32' if original else '70',
               '--pointmae-config', 'source_snapshot/cfgs/' + ('tta_modelnet.yaml' if dataset == 'modelnet40_c' else 'tta_scanobj.yaml'),
               '--pointmae-checkpoint', 'checkpoints/' + ('modelnet_jt.pth' if dataset == 'modelnet40_c' else 'scanobject_jt.pth'),
               '--index-manifest', selection_path(plan, stage, dataset),
               '--corruptions', *(['gaussian', 'background'] if stage == 'smoke' else ['all'])]
    if not original:
        command.extend(['--trajectory-observe', '--component-arm', trial['family'], '--paired-state-root', str(cache_path(stage, seed))])
        for key, default in (('eta', .01), ('gamma', .01), ('p', .95)):
            if trial[key] != default:
                command.extend(['--' + key, str(trial[key])])
        default_weight = 1.17 if trial['family'] in ('combined', 'spectral') else 0.
        weight = spectral_weight(trial, calibration)
        if weight != default_weight:
            command.extend(['--spectral-weight', str(weight)])
        if trial['spectral_loss_kind'] is not None:
            command.extend(['--spectral-loss-kind', trial['spectral_loss_kind']])
    return command


def protocol_key(trial, calibration):
    return trial['method'], trial['family'], json.dumps(expected_protocol(trial, calibration), sort_keys=True)


def frozen_conditions(plan, registry, calibration, plan_sha):
    freeze_path = owned(plan['confirmation_freeze_path'])
    accepted_path = owned(plan['accepted_development_path'])
    freeze = json.loads(freeze_path.read_text())
    accepted = accepted_stage(accepted_path, plan_sha)
    require(freeze.get('status') == 'frozen' and accepted.get('status') == 'accepted'
            and freeze.get('phase_plan_sha256') == accepted.get('phase_plan_sha256') == plan_sha
            and freeze['development_acceptance'] == identity(accepted_path), 'Independent development acceptance/freeze missing')
    require(accepted['selection_policy'] == 'probability_mean', 'Unregistered policy selection')
    spectral_id, chamfer_id = freeze['selected_spectral_trial_id'], freeze['selected_chamfer_trial_id']
    require(spectral_id == accepted['selected_spectral_trial_id'] and chamfer_id == accepted['selected_chamfer_trial_id'], 'Frozen winner differs from accepted ranking')
    spectral, chamfer = registry[spectral_id], registry[chamfer_id]
    require(spectral['family'] in ('combined', 'spectral') and chamfer['family'] == 'chamfer', 'Wrong frozen families')
    matched = copy.deepcopy(registry['c-base'])
    matched.update(id='matched-chamfer', reuse=False, eta=spectral['eta'], gamma=spectral['gamma'], p=spectral['p'])
    result = [dict(id='spectral-candidate', role='spectral_candidate', trial=spectral),
              dict(id='matched-chamfer', role='matched_chamfer', trial=matched)]
    if protocol_key(matched, calibration) != protocol_key(chamfer, calibration):
        result.append(dict(id='tuned-chamfer', role='tuned_chamfer', trial=chamfer))
    result.append(dict(id='original3dd', role='original3dd', trial=registry[plan['original_trial_id']]))
    expected = [dict(id=row['id'], role=row['role'], trial_id=row['trial']['id']) for row in result]
    require(freeze['conditions'] == expected and 3 <= len(result) <= 4, 'Frozen condition list differs from registered construction')
    return result


def jobs_for(plan, registry, stage, output, calibration, plan_sha):
    if stage == 'smoke':
        conditions = [dict(id=name, role='smoke', trial=registry[name]) for name in plan['smoke_trial_ids']]
    elif stage == 'development':
        conditions = [dict(id=name, role='development', trial=trial) for name, trial in registry.items() if not trial['reuse']]
    else:
        conditions = frozen_conditions(plan, registry, calibration, plan_sha)
    jobs = []
    for seed in ((0, 1, 2) if stage == 'confirmation' else (0,)):
        for condition in conditions:
            for dataset in DATASETS:
                trial = condition['trial']
                result_root = Path(output) / ('seed%d' % seed) / condition['id'] / dataset
                jobs.append(dict(condition_id=condition['id'], role=condition['role'], trial=trial, dataset=dataset, seed=seed,
                                 result_root=str(result_root), command=command_for(plan, trial, dataset, stage, seed, result_root, calibration)))
    expected = 2 * len(plan['smoke_trial_ids']) if stage == 'smoke' else 32 if stage == 'development' else 6 * len(conditions)
    require(len(jobs) == expected, 'Stage job count mismatch')
    return jobs


def ledger_valid(ledger, plan_sha):
    require(ledger.get('schema_version') == 1 and ledger.get('session') == SESSION and ledger.get('phase_plan_sha256') == plan_sha
            and ledger.get('cap_seconds') == CAP and type(ledger.get('spent_seconds')) in (int, float)
            and math.isfinite(ledger['spent_seconds']) and 0 <= ledger['spent_seconds'] <= CAP
            and ledger.get('active') is None, 'Campaign ledger conflict, unresolved worker or exhausted budget')

    require(ledger.get('spent_seconds', -1) >= 155.86660684299568 and ledger.get('recovery', {}).get('original_plan_sha256') == '4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
            and ledger['recovery'].get('original_ledger_sha256') == '6c0e3cb8cebdf4a43bc3383f1599b782f4a1d4129690eb2de1a7de387a5915b3', 'Recovered campaign floor/provenance missing')


def load_ledger(plan_sha, local=None, remote=None):
    if local is None and remote is None:
        require((ROOT / LEDGER_NAME).exists() or (DRIVE / LEDGER_NAME).exists(), 'Recovery ledger must be installed before GPU work')
    local = ROOT / LEDGER_NAME if local is None else Path(local)
    remote = DRIVE / LEDGER_NAME if remote is None else Path(remote)
    if remote.exists():
        ledger = json.loads(remote.read_text()); ledger_valid(ledger, plan_sha)
        if local.exists():
            verify(local, identity(remote))
        else:
            local.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(remote, local)
            verify(local, identity(remote))
        return ledger
    require(not local.exists(), 'Local campaign ledger exists without matching Drive receipt; reconcile before launch')
    return dict(schema_version=1, session=SESSION, phase_plan_sha256=plan_sha, cap_seconds=CAP,
                spent_seconds=155.86660684299568, active=None, attempts=[], recovery={'original_plan_sha256': '4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1', 'original_ledger_sha256': '6c0e3cb8cebdf4a43bc3383f1599b782f4a1d4129690eb2de1a7de387a5915b3'})


def save_ledger(ledger, local=None, remote=None):
    local = ROOT / LEDGER_NAME if local is None else Path(local)
    remote = DRIVE / LEDGER_NAME if remote is None else Path(remote)
    raw = json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + '\n'
    for path in (local, remote):
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix('.pending.json'); pending.write_text(raw); pending.replace(path)
    verify(remote, identity(local))


def campaign_budget_verified(plan, stage, spent):
    estimates = plan['stage_expected_seconds']
    require(set(estimates) == set(STAGES) and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                                                 for value in estimates.values()), 'Missing/invalid whole-campaign timing basis')
    require(plan['packing_retry_reserve_seconds'] == 1200 and stage in STAGES
            and type(spent) in (int, float) and math.isfinite(spent) and spent >= 0, 'Invalid campaign reserve/spent time')
    remaining = sum(estimates[name] for name in STAGES[STAGES.index(stage):])
    projected = spent + 1.2 * remaining + plan['packing_retry_reserve_seconds']
    require(projected <= CAP, 'Updated whole campaign cannot fit the approved cap; stop before further GPU work')
    return projected


def remaining_seconds(spent, elapsed, reserve=300):
    remaining = CAP - spent - elapsed - reserve
    require(math.isfinite(remaining) and remaining > 0, 'Approved whole-campaign cap/packing reserve exhausted')
    return remaining


def load_runtime_tools():
    phases = ROOT / 'phases'
    verify(phases / 'gpu_components_template.py', dict(bytes=29577, sha256=OLD_HELPER_SHA))
    require(identity(phases / 'deploy_assets_template.py')['sha256'] == OLD_DEPLOY_SHA, 'Old evidence helper changed')
    sys.path.insert(0, str(phases))
    spec = importlib.util.spec_from_file_location('agent_b_pinned_stream_helper', phases / 'gpu_components_template.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import deploy_assets_template as evidence
    return module, evidence


def plan_verified(path, expected_sha):
    path = owned(path)
    require(identity(path)['sha256'] == expected_sha, 'Campaign plan SHA mismatch')
    plan = json.loads(path.read_text())
    require(plan['schema_version'] == 1 and plan['session'] == SESSION and plan['cap_seconds'] == CAP
            and plan['asset_root'] == str(ASSETS), 'Campaign ownership/budget/source root changed')
    require(len(plan['manifest_sha256']) == len(plan['source_snapshot_sha256']) == 64, 'Missing sealed assets/source')
    here = Path(__file__).resolve().parent
    require(Path(__file__).name in plan['phase_sources'], 'Executing controller is not bound by plan')
    for name, expected in plan['phase_sources'].items():
        require(Path(name).name == name, 'Unsafe phase source name')
        verify(here / name, expected)
    registry = registry_verified(read_reference(plan['trial_registry']))
    require(plan['original_trial_id'] == 'original3dd' and len(set(plan['smoke_trial_ids'])) == 7
            and set(plan['smoke_trial_ids']) <= set(registry), 'Unregistered smoke/original scope')
    selections = {}
    for stage in ('smoke', 'development', 'confirmation'):
        for dataset in DATASETS:
            selected = read_reference(plan['id_manifests'][stage][dataset])
            n = 140 if stage == 'confirmation' else 70
            total = 2468 if dataset == 'modelnet40_c' else 581
            require(selected['dataset'] == dataset and selected['n_examples'] == total
                    and selected['stage'] == ('heldout' if stage == 'confirmation' else stage)
                    and len(selected['indices']) == n and selected['indices'] == sorted(set(selected['indices']))
                    and all(type(value) is int and 0 <= value < total for value in selected['indices']), 'Invalid registered IDs')
            selections[(stage, dataset)] = selected
    for dataset in DATASETS:
        require(selections[('smoke', dataset)]['indices'] == selections[('development', dataset)]['indices'], 'Smoke selection changed')
        require(not set(selections[('development', dataset)]['indices']) & set(selections[('confirmation', dataset)]['indices']), 'Development/confirmation overlap')
        old_assets = ROOT / 'prepared_v1/assets'
        require(identity(old_assets / 'manifest.json')['sha256'] == OLD_MANIFEST, 'Historical ID exclusion source changed')
        old_manifest = json.loads((old_assets / 'manifest.json').read_text())
        old_relative = 'study/' + dataset + '_heldout_ids.json'
        verify(old_assets / old_relative, old_manifest['files'][old_relative])
        previous_heldout = json.loads((old_assets / old_relative).read_text())
        require(not set(previous_heldout['indices']) & set(selections[('confirmation', dataset)]['indices']), 'Previously exposed heldout objects reused for new confirmation')
    return plan, registry, selections


def verify_common_config(config, job, plan, manifest, selected, calibration):
    dataset, trial, seed = job['dataset'], job['trial'], job['seed']
    original = trial['method'] == 'original3dd'
    require(type(config['seed']) is int and config['seed'] == seed and config['dataset'] == dataset
            and config['status'] == 'partial' and config['severity'] == 5 and config['selected_original_ids'] == selected['indices']
            and config['selection_manifest'] == selected and config['asset_manifest_sha256'] == plan['manifest_sha256']
            and config['asset_manifest'] == manifest and config['batch_size'] == (32 if original else 70), 'Raw config/selection/source drift')
    require(config['num_classes'] == (40 if dataset == 'modelnet40_c' else 15)
            and config['split'] == ('official_modelnet40_test' if dataset == 'modelnet40_c' else 'main_split'), 'Dataset/class contract drift')
    require(config['method'] == ('3dd_original' if original else GSD_METHOD)
            and config['guidance_protocol'] == expected_protocol(trial, calibration) and config['lion_ema'] is False,
            'Unregistered reconstruction objective or parameters')
    require(config['classifier_state_before_sha256'] == config['classifier_state_after_sha256']
            and config['lion_state_before'] == config['lion_state_after'] and config['initialization_rng_guard']['rng_unchanged'], 'Frozen state/initialization guard failed')
    source_files = ['scripts/eval_multibackbone_guidance.py', 'scripts/eval_multibackbone_source.py', 'classifier_backbones.py', 'spectral_guidance_losses.py']
    source_files += ['tta.py'] if original else ['tta_gsd_v2.py', 'shared_trajectory.py']
    for name in source_files:
        require(config['executed_source_files'].get('assets/source_snapshot/' + name) == manifest['files']['source_snapshot/' + name], 'Unsealed executed source: ' + name)
    load = config['checkpoint_load']; checkpoint = config['pointmae_checkpoint'] if config['backbone'] == 'pointmae' else config['checkpoint']
    require(load['strict'] and not load['missing_keys'] and not load['unexpected_keys']
            and load['checkpoint_sha256'] == manifest['files'][checkpoint]['sha256'], 'Classifier checkpoint binding failed')
    require(config['native_bootstrap']['no_build'] is True, 'Native build unexpectedly allowed')
    audit_path = Path(plan['asset_root']) / manifest['guidance']['native_audit']
    verify(audit_path, manifest['files'][manifest['guidance']['native_audit']])
    audit = json.loads(audit_path.read_text())
    require(set(config['native_bootstrap']['bindings']) == set(audit['backends']), 'Native backend set changed')
    for name, entries in audit['backends'].items():
        entry, = entries
        require(config['native_bootstrap']['bindings'][name] == {key: entry[key] for key in ('path', 'bytes', 'sha256')}, 'Pinned native binding changed')
    if original:
        require(config['decoder_style'] == 'original' and config['scheduler_final_alpha_to_one'] is True
                and config['graph'] is None and not config.get('trajectory_observation')
                and (config['normal_reverse_steps'], config['background_reverse_steps']) == (5, 35), 'Original3DD policy drift')
        if config['backbone'] == 'pointmae':
            require(config['matched_source']['status'] == 'unavailable', 'Unvalidated Point-MAE source row fabricated')
    else:
        require(config['decoder_style'] == 'updated' and config['scheduler_final_alpha_to_one'] is False
                and config['component_study'] is True and config['component_arm'] == trial['family']
                and config['graph'] == dict(k=10, delta=.1, gamma=.6, use_4d_gft=False, isolate_policy='legacy'), 'GSD host/graph drift')


def validate_job(job, stage, plan, manifest, selected, calibration, evidence):
    import numpy as np
    from scripts.analyze_scan_trajectory import analyze_bundle, Bundle
    from research_artifacts import CORRUPTIONS
    original = job['trial']['method'] == 'original3dd'
    corruptions = ['gaussian', 'background'] if stage == 'smoke' else list(CORRUPTIONS)
    method = '3dd_original' if original else GSD_METHOD
    configs = list((Path(job['result_root']) / job['dataset'] / method).glob('*/config.json'))
    require(len(configs) == 4, 'Require four final frozen-head bundles')
    data_root = Path('/content/3DD-TTA/data/modelnet40_c') if job['dataset'] == 'modelnet40_c' else Path(plan['asset_root']) / 'datasets/scanobjectnn_c'
    label_file = data_root / 'label.npy'; verify(label_file, manifest['datasets'][job['dataset']]['files']['label.npy'])
    ids = selected['indices']; labels = np.load(label_file, allow_pickle=False).reshape(-1)
    heads, traces, archives, reports, cost, contracts = set(), {}, [], [], {}, {}
    lion_state = None
    for config_path in configs:
        archive_path = Path(str(config_path.parent) + '.zip')
        with zipfile.ZipFile(archive_path) as archive:
            evidence.checked_members(archive)
        bundle = Bundle(archive_path)
        try:
            require({'command.txt', 'config.json', 'environment.txt', 'stdout.log', 'summary.csv', 'per_corruption.csv', 'notes.md'} <= bundle.names, 'Required raw run files missing')
            config = bundle.json('config.json'); head = config['backbone']; heads.add(head)
            verify_common_config(config, job, plan, manifest, selected, calibration)
            selection_bytes = bundle.read('selection_manifest.json')
            require(json.loads(selection_bytes) == selected and config['selection_manifest_identity']['bytes'] == len(selection_bytes)
                    and config['selection_manifest_identity']['sha256'] == hashlib.sha256(selection_bytes).hexdigest(), 'Archived selection manifest differs')
            contracts[head] = dict(state_sha256=config['classifier_state_before_sha256'], checkpoint_sha256=config['checkpoint_load']['checkpoint_sha256'],
                                   preprocessing=config['source_classifier_preprocessing'], resolved_model_config=config['resolved_model_config'])
            if lion_state is None:
                lion_state = config['lion_state_before']
            require(config['lion_state_before'] == lion_state, 'Heads loaded different LION states')
            require(set(config['corruptions']) == set(corruptions), 'Corruption scope changed')
            require(archive_path.name == config['run_id'] + '.zip', 'Run archive/name mismatch')
            drive_run = DRIVE / job['dataset'] / method / config['run_id']
            receipt = json.loads((drive_run / 'final.receipt.json').read_text())
            require(receipt['status'] == 'complete' and receipt['result_status'] == 'partial', 'Final Drive receipt incomplete')
            verify(archive_path, receipt['archive']); verify(drive_run / 'final.zip', receipt['archive'])
            rows = []
            batch_size = 32 if original else 70
            for corruption in corruptions:
                ordered = sorted(name for name in bundle.names if name.startswith('batches/' + corruption + '/') and name.endswith('/predictions.npz'))
                require(len(ordered) == (len(ids) + batch_size - 1) // batch_size, 'Incomplete multi-batch coverage')
                seen, correct, parts = [], 0, []
                for number, name in enumerate(ordered):
                    prefix = 'batches/' + corruption + '/batch%04d/' % number
                    require(name == prefix + 'predictions.npz', 'Batch order/path mismatch')
                    with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as archive:
                        data = {key: archive[key].copy() for key in archive.files}
                    expected_ids = np.asarray(ids[number * batch_size:(number + 1) * batch_size], dtype=np.int64)
                    n = len(expected_ids)
                    require(np.array_equal(data['index'], expected_ids) and np.array_equal(data['label'], labels[expected_ids])
                            and data['prediction'].shape == data['label'].shape == (n,)
                            and data['logits'].shape == (n, config['num_classes']) and np.isfinite(data['logits']).all()
                            and np.array_equal(data['prediction'], data['logits'].argmax(-1)), 'Indexed labels/logits/count mismatch')
                    diagnostic = bundle.json(prefix + 'diagnostics.json')
                    require(diagnostic['seed'] == job['seed'] and diagnostic['rng_unchanged'] is True
                            and diagnostic['geometry_shape'] == [n, 2048, 3], 'Batch geometry/RNG guard failed')
                    trace = dict(geometry_sha256=diagnostic['geometry_sha256'])
                    if original:
                        scheduler = diagnostic['scheduler']; horizon = 35 if corruption == 'background' else 5
                        expected_scheduler = dict(beta_end=.02, beta_start=.0001, beta_schedule='linear', clip_sample=False, num_train_timesteps=1000, prediction_type='epsilon', set_alpha_to_one=True)
                        require(all(scheduler['config'].get(key) == value for key, value in expected_scheduler.items())
                                and scheduler == config['scheduler_records'][corruption]
                                and scheduler['timesteps'] == list(range(990, -1, -10))
                                and scheduler['active_timesteps'] == list(range(990, -1, -10))[-horizon:]
                                and scheduler['final_alpha_cumprod'] == 1.
                                and scheduler['config']['set_alpha_to_one'] is True, 'Original scheduler execution mismatch')
                        require('trajectory_logits' not in data and 'trajectory' not in diagnostic and not diagnostic['guidance_metrics'], 'Original terminal path changed')
                    else:
                        horizon = 30 if corruption == 'background' else 10
                        weights = config['guidance_protocol']['loss_weights']; metrics = diagnostic['guidance_metrics']
                        require(all(math.isfinite(value) and value >= 0 for value in metrics.values())
                                and metrics['graph_calls'] == int(weights['spectral'] > 0)
                                and metrics['chamfer_calls'] == 2 * horizon * int(weights['chamfer'] > 0)
                                and metrics['guidance_backward_calls'] == horizon, 'Active/inactive component execution mismatch')
                        cached = diagnostic['paired_state']; path = owned(cached['path'])
                        require(path.parent.parent.parent == cache_path(stage, job['seed'])
                                and path.parent.parent.name == job['dataset'] and path.parent.name == corruption
                                and path.name == hashlib.sha256(expected_ids.tobytes()).hexdigest() + '.npz', 'Wrong seed/dataset/corruption/cache IDs')
                        verify(path, cached)
                        trace.update(initial_hashes=diagnostic['trajectory']['initial_hashes'], paired_state=cached)
                    key = corruption + '/batch%04d' % number
                    if key in traces:
                        require(traces[key] == trace, 'Frozen heads observed different shared reconstruction')
                    else:
                        traces[key] = trace
                        cost[key] = dict(shared_reconstruction_seconds=diagnostic['shared_reconstruction_seconds'],
                                         whole_process_peak_gpu_memory_mb=diagnostic['whole_process_peak_gpu_memory_mb'],
                                         preparation_mode=diagnostic['preparation_mode'])
                    batch_receipt_path = drive_run / prefix / 'receipt.json'
                    batch_receipt = json.loads(batch_receipt_path.read_text())
                    require(batch_receipt['status'] == 'complete' and set(batch_receipt['files']) == {'predictions.npz', 'diagnostics.json'}, 'Batch Drive receipt mismatch')
                    for filename, expected in batch_receipt['files'].items():
                        verify(config_path.parent / prefix / filename, expected); verify(batch_receipt_path.parent / filename, expected)
                    seen.extend(data['index'].tolist()); correct += int((data['prediction'] == data['label']).sum())
                    parts.append(data)
                require(seen == ids, 'Corruption object coverage mismatch')
                with np.load(io.BytesIO(bundle.read(corruption + '_predictions.npz')), allow_pickle=False) as aggregate:
                    for key in ('index', 'label', 'prediction', 'logits'):
                        require(np.array_equal(aggregate[key], np.concatenate([part[key] for part in parts])), 'Aggregated predictions differ from batch shards')
                corruption_receipt = json.loads((drive_run / ('corruption-' + corruption + '.receipt.json')).read_text())
                require(corruption_receipt['status'] == 'complete', 'Corruption Drive receipt incomplete')
                verify(corruption_receipt['archive']['path'], corruption_receipt['archive'])
                rows.append(dict(corruption=corruption, n=len(ids), correct=correct, accuracy_pct=100. * correct / len(ids)))
            if original:
                csv_rows = list(csv.DictReader(io.StringIO(bundle.read('per_corruption.csv').decode())))
                require(len(csv_rows) == len(corruptions) and {row['corruption'] for row in csv_rows} == set(corruptions), 'Original CSV scope mismatch')
                for row in rows:
                    saved, = [value for value in csv_rows if value['corruption'] == row['corruption']]
                    require(int(saved['n_examples']) == row['n'] and int(saved['n_correct']) == row['correct']
                            and abs(float(saved['accuracy']) * 100. - row['accuracy_pct']) <= 1e-8, 'Original CSV/prediction disagreement')
                summary_rows = list(csv.DictReader(io.StringIO(bundle.read('summary.csv').decode())))
                require(len(summary_rows) == 1, 'Original summary row count mismatch')
                saved_summary = summary_rows[0]
                total_correct = sum(row['correct'] for row in rows)
                require(saved_summary['run_id'] == config['run_id'] and saved_summary['status'] == 'partial'
                        and int(saved_summary['n_corruptions']) == len(corruptions)
                        and int(saved_summary['total_examples']) == len(ids) * len(corruptions)
                        and int(saved_summary['total_correct']) == total_correct
                        and abs(float(saved_summary['macro_accuracy']) * 100. - sum(row['accuracy_pct'] for row in rows) / len(rows)) <= 1e-8, 'Original summary/count disagreement')
                report = dict(dataset=job['dataset'], backbone=head, method=method, per_corruption=rows,
                              macro=[dict(dataset=job['dataset'], backbone=head, policy='terminal', n=len(ids) * len(rows),
                                          accuracy_pct=sum(row['accuracy_pct'] for row in rows) / len(rows))])
            else:
                report = analyze_bundle(archive_path, allow_partial=True)
                require(report['status'] == 'partial' and report['selection_complete']
                        and len(report['per_corruption']) == 3 * len(corruptions)
                        and all(row['n'] == len(ids) for row in report['per_corruption']), 'Invalid GSD subset trajectory')
            reports.append(report)
            archives.append(dict(backbone=head, local_path=str(archive_path), drive_path=str(drive_run / 'final.zip'), **identity(archive_path)))
        finally:
            bundle.close()
    require(heads == set(HEADS), 'Missing/duplicate frozen heads')
    return dict(reports=reports, raw_archives=archives, traces=traces, timings=cost, classifier_contracts=contracts, lion_state=lion_state)


def pairing_verified(records):
    groups = {}
    for job in records:
        if job['trial']['method'] != 'gsd':
            continue
        key = job['dataset'], job['seed']
        for batch, value in job['traces'].items():
            reference = dict(initial_hashes=value['initial_hashes'], paired_state=value['paired_state'])
            group = groups.setdefault(key, {})
            require(batch not in group or group[batch] == reference, 'GSD trials did not share exact initial state/noise/cache')
            group[batch] = reference


def persist_caches(records, evidence):
    paths = {value['paired_state']['path'] for record in records for value in record.get('traces', {}).values() if 'paired_state' in value}
    result = []
    for name in sorted(paths):
        path = owned(name); relative = path.relative_to(ROOT); target = DRIVE / relative
        expected = identity(path)
        if target.exists():
            verify(target, expected)
        else:
            evidence.copy_checked(path, target, expected)
        result.append(dict(local_path=str(path), drive_path=str(target), **expected))
    return result


def execute(args):
    plan, registry, selections = plan_verified(args.phase_plan, args.phase_plan_sha256)
    approval_verified(args.phase_plan_sha256)
    cpu_preflight_verified(plan)
    gpu, evidence = load_runtime_tools()
    signal.signal(signal.SIGTERM, gpu.raise_on_terminate)
    ledger = load_ledger(args.phase_plan_sha256)
    campaign_budget_verified(plan, args.stage, ledger['spent_seconds'])
    require(not any(attempt['stage'] == args.stage for attempt in ledger['attempts']), 'Stage already attempted; inspect immutable evidence before reviewed recovery')
    if args.stage != 'calibrate':
        previous = 'calibrate' if args.stage == 'smoke' else 'smoke' if args.stage == 'development' else 'development'
        status = json.loads((ROOT / ('sensitivity_' + previous + '_validation.json')).read_text())
        require(status['status'] == 'validated' and status['phase_plan_sha256'] == args.phase_plan_sha256, 'Prior stage is not accepted')
    verify(ASSETS / 'manifest.json', dict(bytes=(ASSETS / 'manifest.json').stat().st_size, sha256=plan['manifest_sha256']))
    verify(ASSETS / 'source_snapshot.zip', dict(bytes=(ASSETS / 'source_snapshot.zip').stat().st_size, sha256=plan['source_snapshot_sha256']))
    sys.path.insert(0, str(ASSETS / 'source_snapshot'))
    from scripts import eval_multibackbone_source as source
    require(source.REPO == ASSETS / 'source_snapshot', 'Unsealed project import')
    manifest = source.read_manifest(ASSETS, plan['manifest_sha256'])
    for entry in plan['id_manifests'].values():
        for reference in entry.values():
            relative = Path(reference['path']).relative_to(ASSETS)
            require({key: reference[key] for key in ('bytes', 'sha256')} == manifest['files'][str(relative)], 'Unsealed selection manifest')
    if args.stage == 'development':
        accepted_stage(ROOT / 'accepted_sensitivity_smoke.json', args.phase_plan_sha256)
        parity = read_reference(plan['baseline_equivalence'])
        require(parity.get('status') == 'passed' and parity.get('default_configurations_exact') is True
                and parity.get('legacy_losses_gradients_exact') is True and parity.get('old_manifest_sha256') == OLD_MANIFEST,
                'Baseline source/numerical equivalence proof missing')
    output = ROOT / ('sensitivity_' + args.stage + '_' + evidence.utc()); output.mkdir()
    drive_output = DRIVE / 'phase-evidence' / output.name; drive_output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', stage=args.stage, session=SESSION, phase_plan_sha256=args.phase_plan_sha256,
                 manifest_sha256=plan['manifest_sha256'], source_snapshot_sha256=plan['source_snapshot_sha256'], jobs=[], full_benchmark=False)
    started, previous_spent = time.monotonic(), ledger['spent_seconds']
    ledger['active'] = dict(stage=args.stage, output=str(output), controller_pid=__import__('os').getpid())
    ledger['attempts'].append(dict(stage=args.stage, output=str(output), status='running', utc=evidence.utc()))
    try:
        save_ledger(ledger)
        shutil.copyfile(args.phase_plan, output / 'campaign_plan.json')
        shutil.copyfile(plan['trial_registry']['path'], output / 'trial_registry.json')
        shutil.copyfile(ROOT / 'sensitivity_gpu_approval.json', output / 'gpu_approval.json')
        shutil.copyfile(ASSETS.parent / 'accepted_cpu_preflight_receipt.json', output / 'accepted_cpu_preflight_receipt.json')
        for name in plan['phase_sources']:
            shutil.copyfile(Path(__file__).resolve().parent / name, output / name)
        def run(command, log_path, extra):
            def on_start(pid):
                ledger['active'].update(worker_group_pid=pid, command=command, **extra); save_ledger(ledger)
            gpu.stream_job(command, log_path, remaining_seconds(previous_spent, time.monotonic() - started, plan['packing_retry_reserve_seconds']), on_start)
            ledger['active'].pop('worker_group_pid', None)
        if args.stage == 'calibrate':
            command = plan['calibration']['command']
            require(command[:2] == [PYTHON, '-u'] and Path(command[2]).parent == Path(__file__).resolve().parent
                    and Path(command[2]).name in plan['phase_sources'], 'Calibration entrypoint is not sealed')
            run(command, output / 'calibration.log', dict(job='calibration'))
            report = calibration_verified(plan['calibration']['report_path'], plan)
            directory = owned(Path(plan['calibration']['report_path']).parent)
            require(directory != ROOT, 'Calibration evidence needs a dedicated owned directory')
            shutil.copytree(directory, output / 'calibration')
            state.update(calibration=report, status='complete')
        else:
            calibration = calibration_verified(plan['calibration']['report_path'], plan)
            jobs = jobs_for(plan, registry, args.stage, output, calibration, args.phase_plan_sha256)
            (output / 'commands.json').write_text(json.dumps(jobs, indent=2, sort_keys=True) + '\n')
            if args.stage == 'confirmation':
                for name in ('confirmation_freeze_path', 'accepted_development_path'):
                    shutil.copyfile(plan[name], output / Path(plan[name]).name)
                for seed in (0, 1, 2):
                    require(not cache_path(args.stage, seed).exists() and not (DRIVE / cache_path(args.stage, seed).name).exists(), 'Confirmation cache already exists; inspect previous attempt')
            elif args.stage == 'smoke':
                require(not cache_path(args.stage, 0).exists() and not (DRIVE / cache_path(args.stage, 0).name).exists(), 'Smoke cache already exists; inspect previous attempt')
            for number, job in enumerate(jobs):
                record = dict(job, status='running'); state['jobs'].append(record)
                before = time.monotonic()
                run(job['command'], output / ('job_%03d.log' % number), dict(job=number, condition_id=job['condition_id'], seed=job['seed']))
                key = 'confirmation' if args.stage == 'confirmation' else args.stage
                record.update(validate_job(job, args.stage, plan, manifest, selections[(key, job['dataset'])], calibration, evidence),
                              status='complete', elapsed_seconds=time.monotonic() - before)
                record['cache_records'] = persist_caches([record], evidence)
                for head, contract in record['classifier_contracts'].items():
                    key = job['dataset'] + '/' + head
                    frozen = ledger.setdefault('classifier_contracts', {})
                    require(key not in frozen or frozen[key] == contract, 'Classifier state/checkpoint/preprocessing changed across campaign jobs')
                    frozen[key] = contract
                require('lion_state' not in ledger or ledger['lion_state'] == record['lion_state'], 'LION state changed across campaign jobs')
                ledger['lion_state'] = record['lion_state']
                (output / ('job_%03d_validation.json' % number)).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
                ledger['spent_seconds'] = previous_spent + time.monotonic() - started; save_ledger(ledger)
            pairing_verified(state['jobs'])
            state['cache_records'] = persist_caches(state['jobs'], evidence)
            state['status'] = 'complete'
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        if args.stage == 'calibrate':
            directory = owned(Path(plan['calibration']['report_path']).parent)
            require(directory != ROOT, 'Calibration evidence needs a dedicated owned directory')
            if directory.exists() and not (output / 'calibration').exists():
                shutil.copytree(directory, output / 'calibration')
        state['elapsed_seconds'] = time.monotonic() - started
        ledger['spent_seconds'] = previous_spent + state['elapsed_seconds']; ledger['active'] = None
        ledger['attempts'][-1].update(status=state['status'], elapsed_seconds=state['elapsed_seconds'])
        save_ledger(ledger)
        (output / 'state.json').write_text(json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + '\n')
        (output / 'ledger_snapshot.json').write_text(json.dumps(ledger, indent=2) + '\n')
        receipt = dict(state=state, archive=evidence.seal(output, drive_output / ('sensitivity_' + args.stage + '_evidence.zip')))
        ledger['spent_seconds'] = previous_spent + time.monotonic() - started; save_ledger(ledger)
        name = 'sensitivity_' + args.stage + '_receipt.json'
        (ROOT / name).write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        (drive_output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        print('AGENT_B_SENSITIVITY_STAGE_RECEIPT', json.dumps(receipt['archive']), flush=True)
    require(ledger['spent_seconds'] <= CAP, 'Whole campaign cap exceeded while preserving evidence')
    validation = dict(status='validated', stage=args.stage, phase_plan_sha256=args.phase_plan_sha256,
                      manifest_sha256=plan['manifest_sha256'], source_snapshot_sha256=plan['source_snapshot_sha256'],
                      spent_seconds=ledger['spent_seconds'], output=str(output), evidence=receipt['archive'], independent_local_validation_pending=True)
    (ROOT / ('sensitivity_' + args.stage + '_validation.json')).write_text(json.dumps(validation, indent=2) + '\n')
    print('AGENT_B_SENSITIVITY_' + args.stage.upper() + '_VALIDATED', flush=True)


def self_check():
    with tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent)) as temporary:
        root = Path(temporary); old = root / 'initial_gpu_phase_ledger.json'; old.write_text('old ledger preserved\n'); old_hash = identity(old)
        local, remote = root / 'local.json', root / 'drive.json'
        ledger = load_ledger('a' * 64, local, remote); ledger['spent_seconds'] = 278.8666068429957
        save_ledger(ledger, local, remote); local.unlink()
        require(load_ledger('a' * 64, local, remote)['spent_seconds'] == 278.8666068429957 and identity(old) == old_hash, 'Lost/rewritten campaign budget')
        wrong = dict(ledger, phase_plan_sha256='b' * 64)
        save_ledger(wrong, local, remote)
        try:
            load_ledger('a' * 64, local, remote)
        except ValueError:
            pass
        else:
            raise AssertionError('Foreign plan ledger accepted')
        approval = root / 'approval.json'
        for value in ({}, dict(status='approved', phase_plan_sha256='b' * 64, session=SESSION, cap_seconds=CAP, user_reply='yes')):
            approval.write_text(json.dumps(value))
            try:
                approval_verified('a' * 64, approval)
            except ValueError:
                pass
            else:
                raise AssertionError('Missing/foreign approval accepted')
        approval.write_text(json.dumps(dict(status='approved', phase_plan_sha256='a' * 64, session=SESSION, cap_seconds=CAP, user_reply='explicit whole phase approval')))
        approval_verified('a' * 64, approval)
    trial = dict(id='cs-test', family='combined', method='gsd', reuse=False, eta=.003, gamma=.01, p=.95,
                 spectral_loss_kind='power', spectral_weight=dict(loss_kind='power', rho=.1), chamfer_weight=1.)
    calibration = dict(status='verified_label_free_gpu', coefficients=dict(legacy_l2=10., huber=20., power=30.))
    require(spectral_weight(trial, calibration) == 3. and expected_protocol(trial, calibration)['spectral_loss_kind'] == 'power', 'Calibrated coefficient resolution mismatch')
    try:
        spectral_weight(trial, dict(status='synthetic_only', coefficients=calibration['coefficients']))
    except ValueError:
        pass
    else:
        raise AssertionError('Synthetic calibration accepted for execution')
    plan = dict(asset_root=str(ASSETS), manifest_sha256='a' * 64,
                id_manifests={stage: {dataset: dict(path=str(ASSETS / 'study' / (dataset + '_' + stage + '.json'))) for dataset in DATASETS}
                              for stage in ('smoke', 'development', 'confirmation')})
    for stage, seeds in [('smoke', (0,)), ('development', (0,)), ('confirmation', (0, 1, 2))]:
        for seed in seeds:
            for dataset in DATASETS:
                command = command_for(plan, trial, dataset, stage, seed, ROOT / 'selfcheck-result', calibration)
                require(command[command.index('--seed') + 1] == str(seed) and command[command.index('--spectral-weight') + 1] == '3.0'
                        and command[command.index('--paired-state-root') + 1] == str(cache_path(stage, seed)), 'Command seed/weight/cache mismatch')
    require(cache_path('smoke', 0) != cache_path('development', 0) and len({cache_path('confirmation', seed) for seed in (0, 1, 2)}) == 3, 'Cache namespaces overlap')
    require(remaining_seconds(123, 456) == CAP - 123 - 456 - 300, 'Remaining time excludes prior phase or packing')
    for values in [(CAP, 0), (0, CAP), (float('nan'), 0)]:
        try:
            remaining_seconds(*values)
        except ValueError:
            pass
        else:
            raise AssertionError('Exhausted/nonfinite budget accepted')
    timing = dict(stage_expected_seconds=dict(calibrate=300., smoke=840., development=6714., confirmation=7224.), packing_retry_reserve_seconds=1200)
    require(campaign_budget_verified(timing, 'calibrate', 0.) <= CAP, 'Registered complete campaign should fit cap')
    require(campaign_budget_verified(timing, 'confirmation', 9000.) == 9000. + 1.2 * 7224. + 1200., 'Future campaign projection dropped spent time')
    try:
        campaign_budget_verified(timing, 'development', 5000.)
    except ValueError:
        pass
    else:
        raise AssertionError('Overspent future campaign accepted')
    print('AGENT_B_SENSITIVITY_CONTROLLER_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--phase-plan', type=Path, required=True)
        parser.add_argument('--phase-plan-sha256', required=True)
        parser.add_argument('--stage', choices=STAGES, required=True)
        execute(parser.parse_args())
