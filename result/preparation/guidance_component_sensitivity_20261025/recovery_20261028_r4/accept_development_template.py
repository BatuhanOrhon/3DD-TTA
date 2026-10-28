"""CPU-only r4 development acceptance and frozen confirmation installation."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

PACKAGE_IDENTITY = None  # Bind only after genuine completed local development analysis.
ROOT = Path('/content/studies/guidance-components')
PLAN_SHA = '1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
MANIFEST_SHA = '599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
SOURCE_SHA = '73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8'
OLD_MANIFEST = '7012c30ad88222bbc5435d8508f781fc33385ed3976951c45f48da67f625ed75'
FILES = ('accepted_sensitivity_development.json', 'validation.json', 'sensitivity_confirmation_freeze.json',
         'post_development_budget_projection.json', 'raw_archives.json', 'cache_validation.json', 'calibration_cache_validation.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bytes_identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def validate_acceptance(content, registry, plan, ledger, prior):
    accepted, validation, freeze, projection, raw, caches, calibration_caches = [json.loads(content[name]) for name in FILES]
    for row in (accepted, validation):
        require(row['status'] == 'accepted' and row['stage'] == 'development' and row['phase_plan_sha256'] == PLAN_SHA
                and row['manifest_sha256'] == MANIFEST_SHA and row['source_snapshot_sha256'] == SOURCE_SHA
                and row['physical_cache_complete'] is True and row['selection_policy'] == 'probability_mean', 'Development acceptance/source/policy incomplete')
    require(accepted['local_validation'] == bytes_identity(content['validation.json'])
            and accepted['archive'] == validation['phase_archive'], 'Accepted validation/archive identity differs')
    require(validation['calibration_physical_cache_complete'] is True and validation['cache_files'] == validation['physically_verified_cache_files'] == 30
            and validation['jobs'] == 32 and validation['raw_archives'] == 152 and validation['registered_trials'] == 19, 'Incomplete32-job/152-archive/30-cache analysis')
    cache_sets = []
    for rows in (caches, calibration_caches):
        require(len(rows) == 30 and len({row['remote_path'] for row in rows}) == 30
                and all(row['local_verified'] is True and row['local_paths'] for row in rows), 'Thirty physical caches are required')
        cache_sets.append({row['remote_path']: {key: row[key] for key in ('bytes','sha256')} for row in rows})
    require(cache_sets[0] == cache_sets[1], 'Development/calibration preparation cache identities differ')
    expected_cases = {(name,dataset,head) for name in registry for dataset in ('modelnet40_c','scanobjectnn_c') for head in ('dgcnn','curvenet','pointnext','pointmae')}
    require(len(raw) == 152 and {(row['trial_id'],row['dataset'],row['backbone']) for row in raw} == expected_cases
            and sum(row['reused'] is True for row in raw) == 24 and sum(row['reused'] is False for row in raw) == 128, 'Expected128 new and24 historical raw archives')
    for row in raw:
        require(row['reused'] is registry[row['trial_id']]['reuse'] and row['source_manifest_sha256'] == (OLD_MANIFEST if row['reused'] else MANIFEST_SHA), 'Raw source/history relabeled')
    scores = validation['balanced_scores']
    require(len(scores) == 19 and [row['trial_id'] for row in scores] == list(registry), 'Registered score order/tie rule differs')
    for row in scores:
        trial = registry[row['trial_id']]
        require(row['family'] == trial['family'] and row['n_cases'] == 8 and row['policy'] == ('terminal' if trial['method'] == 'original3dd' else 'probability_mean')
                and type(row['accuracy_pct']) in (int,float) and math.isfinite(row['accuracy_pct']) and 0 <= row['accuracy_pct'] <= 100, 'Invalid balanced trial score')
    spectral = max((row for row in scores if row['family'] in ('combined','spectral')),key=lambda row:row['accuracy_pct'])['trial_id']
    chamfer = max((row for row in scores if row['family'] == 'chamfer'),key=lambda row:row['accuracy_pct'])['trial_id']
    require(freeze['status'] == 'frozen' and freeze['phase_plan_sha256'] == PLAN_SHA
            and freeze['development_acceptance'] == bytes_identity(content['accepted_sensitivity_development.json'])
            and freeze['selected_spectral_trial_id'] == accepted['selected_spectral_trial_id'] == spectral
            and freeze['selected_chamfer_trial_id'] == accepted['selected_chamfer_trial_id'] == chamfer, 'Frozen winner/acceptance identity differs')
    for key in ('spent_seconds','remaining_point_seconds','conservative_max_ratio','upper_seconds'):
        require(type(projection[key]) in (int,float) and math.isfinite(projection[key]) and projection[key] >= 0, 'Invalid post-development budget value')
    require(projection['status'] == 'whole_campaign_reestimate_fits_approved_cap' and projection['phase_plan_sha256'] == PLAN_SHA
            and projection['spent_seconds'] == ledger['spent_seconds'] and projection['cap_seconds'] == ledger['cap_seconds'] == 19800
            and projection['remaining_basis'] == 'registered_maximum_24_jobs' and projection['margin_multiplier'] == 1.2
            and projection['packing_retry_reserve_seconds'] == 1200, 'Post-development budget is stale/incomplete')
    require(projection['conservative_max_ratio'] >= max(1.025455024723712,prior['conservative_max_ratio'])
            and projection['original_future_budget_seconds']['confirmation_per_dataset_seed'] == 300
            and prior['original_future_budget_seconds']['confirmation_per_dataset_seed'] == 300
            and projection['original_allowance_increase_seconds'] == 180, 'Original conservative future allowances were reduced')
    lower = plan['stage_expected_seconds']['confirmation'] * projection['conservative_max_ratio'] + 180
    calculated = ledger['spent_seconds'] + 1.2 * projection['remaining_point_seconds'] + 1200
    require(projection['remaining_point_seconds'] >= lower and abs(projection['upper_seconds'] - calculated) <= 1e-6
            and projection['upper_seconds'] <= 19800, 'Conservative whole campaign no longer fits approved cap')
    return accepted, validation, freeze, projection


def main():
    package = Path('/content/agent_b_r4_development_local_acceptance.zip')
    require(isinstance(PACKAGE_IDENTITY,dict) and set(PACKAGE_IDENTITY) == {'bytes','sha256'}, 'Bind actual completed acceptance ZIP identity before execution')
    require(bytes_identity(package.read_bytes()) == PACKAGE_IDENTITY, 'Local acceptance package identity mismatch')
    path = ROOT / 'sensitivity_phases/sensitivity_campaign_controller.py'
    require(bytes_identity(path.read_bytes())['sha256'] == '0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc', 'Current controller changed')
    spec = importlib.util.spec_from_file_location('development_acceptance_control',path)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    plan,registry,_ = c.plan_verified(ROOT / 'sensitivity_campaign_plan.json',PLAN_SHA)
    ledger = c.load_ledger(PLAN_SHA)
    require(ledger['active'] is None, 'GPU worker is still active')
    attempts = [row for row in ledger['attempts'] if row['stage'] == 'development']
    require(len(attempts) == 1 and attempts[0]['status'] == 'complete', 'Development worker is not complete')
    helper = ROOT / 'phases/deploy_assets_template.py'
    require(bytes_identity(helper.read_bytes())['sha256'] == '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0', 'Pinned evidence helper changed')
    sys.path.insert(0,str(helper.parent))
    from deploy_assets_template import checked_members,copy_checked,identity
    with zipfile.ZipFile(package) as archive:
        checked_members(archive)
        content = {name:archive.read(name) for name in FILES}
    prior = json.loads((ROOT / 'post_smoke_budget_projection.json').read_text())
    accepted,validation,freeze,projection = validate_acceptance(content,registry,plan,ledger,prior)
    remote_validation = json.loads((ROOT / 'sensitivity_development_validation.json').read_text())
    require(remote_validation['status'] == 'validated' and remote_validation['stage'] == 'development'
            and remote_validation['phase_plan_sha256'] == PLAN_SHA and remote_validation['manifest_sha256'] == MANIFEST_SHA
            and remote_validation['source_snapshot_sha256'] == SOURCE_SHA and remote_validation['output'] == attempts[0]['output']
            and remote_validation['evidence'] == accepted['archive'], 'Remote completion and independent acceptance disagree')
    c.owned(accepted['archive']['drive_path'],c.DRIVE)
    c.campaign_budget_verified(plan,'confirmation',ledger['spent_seconds'])
    for row in json.loads(content['cache_validation.json']):
        local = c.owned(row['remote_path']);relative = local.relative_to(ROOT)
        require(relative.parts[0] == 'paired_state_v1_seed0' and local.suffix == '.npz', 'Foreign preparation cache namespace')
        c.verify(local,row);c.verify(c.owned(c.DRIVE / relative,c.DRIVE),row)
    calibration = c.calibration_verified(plan['calibration']['report_path'],plan)
    with tempfile.TemporaryDirectory(prefix='review_development_acceptance_',dir=str(ROOT)) as temporary:
        directory = Path(temporary)
        (directory / 'accepted.json').write_bytes(content['accepted_sensitivity_development.json'])
        (directory / 'freeze.json').write_bytes(content['sensitivity_confirmation_freeze.json'])
        candidate_plan = dict(plan,accepted_development_path=str(directory/'accepted.json'),confirmation_freeze_path=str(directory/'freeze.json'))
        c.frozen_conditions(candidate_plan,registry,calibration,PLAN_SHA)
    targets = {'accepted_sensitivity_development.json':'accepted_sensitivity_development.json','validation.json':'accepted_sensitivity_development_validation.json',
               'sensitivity_confirmation_freeze.json':'sensitivity_confirmation_freeze.json','post_development_budget_projection.json':'post_development_budget_projection.json'}
    require(all(not c.owned(ROOT / name).exists() for name in targets.values()) and not (ROOT / 'sensitivity_development_local_acceptance_receipt.json').exists(), 'Acceptance/freeze already exists; inspect before retry')
    drive = c.DRIVE / 'phase-evidence' / ('sensitivity_development_local_acceptance_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    drive.mkdir(parents=True,exist_ok=False)
    copy_checked(package,drive/package.name,PACKAGE_IDENTITY)
    for source,target in targets.items():
        (ROOT/target).write_bytes(content[source])
    c.frozen_conditions(plan,registry,calibration,PLAN_SHA)
    receipt = dict(status='accepted',confirmation_frozen=True,phase_plan_sha256=PLAN_SHA,archive=dict(PACKAGE_IDENTITY,drive_path=str(drive/package.name)),
                   raw_phase=accepted['archive'],accepted_development=bytes_identity(content['accepted_sensitivity_development.json']),
                   freeze=bytes_identity(content['sensitivity_confirmation_freeze.json']),local_validation=bytes_identity(content['validation.json']),
                   budget_projection=bytes_identity(content['post_development_budget_projection.json']),upper_campaign_seconds=projection['upper_seconds'])
    (ROOT/'sensitivity_development_local_acceptance_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (drive/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('AGENT_B_SENSITIVITY_DEVELOPMENT_LOCAL_ACCEPTANCE',json.dumps(receipt),flush=True)
    print('AGENT_B_SENSITIVITY_DEVELOPMENT_ACCEPTANCE_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
