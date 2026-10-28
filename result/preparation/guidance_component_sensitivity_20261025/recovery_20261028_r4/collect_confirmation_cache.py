"""CPU-only immutable collection of completed r4 confirmation preparation caches."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import zipfile

ROOT = Path('/content/studies/guidance-components')
DRIVE = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
SESSION = 'tta-guidance-components-a100-20261028-r4'
PLAN_SHA = '1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
CONTROLLER_SHA = '0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc'
MANIFEST_SHA = '599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
SOURCE_SHA = '73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8'
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
CORRUPTIONS = ('uniform', 'gaussian', 'background', 'impulse', 'upsampling', 'distortion_rbf',
               'distortion_rbf_inv', 'density', 'density_inc', 'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def owned(path, root):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path != root and root in path.parents
            and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents), 'Unsafe cache/evidence path: ' + str(path))
    return path


def expected_cache_paths(ids, root):
    result = {}
    require(set(ids) == set(DATASETS), 'Both confirmation datasets required')
    for dataset in DATASETS:
        selected = ids[dataset]
        require(len(selected) == 140 and selected == sorted(set(selected))
                and all(type(index) is int and 0 <= index < (2468 if dataset == DATASETS[0] else 581) for index in selected), 'Expected140 registered original IDs')
        for seed in (0, 1, 2):
            for corruption in CORRUPTIONS:
                for batch in (0, 1):
                    digest = hashlib.sha256(struct.pack('<70q', *selected[70*batch:70*(batch+1)])).hexdigest()
                    path = root / ('paired_sensitivity_confirmation_v2_seed%d' % seed) / dataset / corruption / (digest + '.npz')
                    result[path] = dict(seed=seed, dataset=dataset, corruption=corruption, batch=batch)
    require(len(result) == 180, 'Expected180 unique confirmation NPZ paths')
    return result


def validated_records(state, freeze, ids, root=ROOT, drive=DRIVE):
    require(state.get('status') == 'complete' and state.get('stage') == 'confirmation' and state.get('session') == SESSION
            and state.get('phase_plan_sha256') == PLAN_SHA and state.get('manifest_sha256') == MANIFEST_SHA
            and state.get('source_snapshot_sha256') == SOURCE_SHA, 'Incomplete/foreign confirmation state')
    require(freeze.get('status') == 'frozen' and freeze.get('phase_plan_sha256') == PLAN_SHA
            and len(freeze['conditions']) in (3, 4), 'Missing frozen conditions')
    expected = expected_cache_paths(ids, root)
    rows = state['cache_records']
    require(len(rows) == 180 and len({row['local_path'] for row in rows}) == 180, 'Require exactly180 unique cache records')
    records = {row['local_path']: row for row in rows}
    require({Path(name) for name in records} == set(expected), 'Cache seed/dataset/corruption/two-batch coverage mismatch')
    counts = {seed: 0 for seed in (0, 1, 2)}
    for row in rows:
        path = owned(row['local_path'], root); target = owned(row['drive_path'], drive)
        require(target.relative_to(drive) == path.relative_to(root) and path.suffix == '.npz'
                and type(row['bytes']) is int and row['bytes'] > 0
                and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None, 'Cache identity/Drive namespace mismatch')
        counts[expected[path]['seed']] += 1
    require(set(counts.values()) == {60}, 'Each seed requires60 caches')
    conditions = {row['id'] for row in freeze['conditions']}
    require(len(conditions) == len(freeze['conditions']) and 'original3dd' in conditions, 'Duplicate/missing frozen condition')
    expected_jobs = {(name, dataset, seed) for name in conditions for dataset in DATASETS for seed in (0, 1, 2)}
    jobs = state['jobs']
    require(len(jobs) == len(expected_jobs) and {(job['condition_id'], job['dataset'], job['seed']) for job in jobs} == expected_jobs,
            'Incomplete frozen condition/dataset/seed jobs')
    for job in jobs:
        require(job['status'] == 'complete' and type(job['seed']) is int, 'Incomplete confirmation worker')
        subset = job['cache_records']
        if job['trial']['method'] == 'original3dd':
            require(job['condition_id'] == 'original3dd' and not subset, 'Original3DD unexpectedly used paired cache')
            continue
        require(job['trial']['method'] == 'gsd' and len(subset) == 30 and len({row['local_path'] for row in subset}) == 30,
                'Each GSD dataset job needs30 cache NPZs')
        paths = {Path(row['local_path']) for row in subset}
        wanted = {path for path, item in expected.items() if item['seed'] == job['seed'] and item['dataset'] == job['dataset']}
        require(paths == wanted and all(row == records[row['local_path']] for row in subset), 'Job cache differs from final records')
        require(set(job['traces']) == {item['corruption'] + '/batch%04d' % item['batch'] for path, item in expected.items() if path in wanted},
                'Incomplete two-batch GSD traces')
        for path in wanted:
            item = expected[path]; trace = job['traces'][item['corruption'] + '/batch%04d' % item['batch']]['paired_state']
            row = records[str(path)]
            require(trace == dict(path=str(path), bytes=row['bytes'], sha256=row['sha256']), 'Trace/cache identity disagreement')
    return [records[key] for key in sorted(records)], counts


def main():
    helper = ROOT / 'phases/deploy_assets_template.py'
    require(hashlib.sha256(helper.read_bytes()).hexdigest() == '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0', 'Pinned CPU evidence helper changed')
    sys.path.insert(0, str(helper.parent))
    from deploy_assets_template import verify, copy_checked, seal, utc, identity, checked_members
    controller = ROOT / 'sensitivity_phases/sensitivity_campaign_controller.py'
    require(identity(controller)['sha256'] == CONTROLLER_SHA, 'Migrated controller changed')
    spec = importlib.util.spec_from_file_location('confirmation_cache_controller', controller)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    plan_path = ROOT / 'sensitivity_campaign_plan.json'
    plan, registry, selections = c.plan_verified(plan_path, PLAN_SHA)
    receipt_path = ROOT / 'sensitivity_confirmation_receipt.json'
    validation_path = ROOT / 'sensitivity_confirmation_validation.json'
    ledger_path = ROOT / 'sensitivity_campaign_r4_ledger.json'
    receipt = json.loads(receipt_path.read_text()); validation = json.loads(validation_path.read_text())
    ledger = json.loads(ledger_path.read_text()); c.ledger_valid(ledger, PLAN_SHA)
    require(ledger['active'] is None and ledger['spent_seconds'] <= 19800
            and validation['status'] == 'validated' and validation['stage'] == 'confirmation'
            and validation['phase_plan_sha256'] == PLAN_SHA and validation['manifest_sha256'] == MANIFEST_SHA
            and validation['source_snapshot_sha256'] == SOURCE_SHA, 'Confirmation is not complete/validated/inactive')
    attempts = [row for row in ledger['attempts'] if row['stage'] == 'confirmation']
    require(len(attempts) == 1 and attempts[0]['status'] == 'complete' and attempts[0]['output'] == validation['output'], 'Unresolved confirmation attempt')
    require(ledger['spent_seconds'] >= validation['spent_seconds'], 'Campaign spent time moved backwards')
    phase_archive = receipt['archive']
    require(validation['evidence'] == phase_archive and phase_archive['local_path'] == validation['output'] + '.zip', 'Phase receipt/archive mismatch')
    for key, base in (('local_path', ROOT), ('drive_path', DRIVE)):
        verify(owned(phase_archive[key], base), phase_archive)
    with zipfile.ZipFile(phase_archive['local_path']) as archive:
        checked_members(archive)
        require(json.loads(archive.read('state.json')) == receipt['state'], 'Physical phase state differs from receipt')
        require(archive.read('campaign_plan.json') == plan_path.read_bytes(), 'Physical phase plan changed')
        freeze_path = owned(plan['confirmation_freeze_path'], ROOT)
        require(archive.read(freeze_path.name) == freeze_path.read_bytes(), 'Frozen conditions changed after confirmation')
    freeze = json.loads(freeze_path.read_text())
    records, counts = validated_records(receipt['state'], freeze,
                                       {dataset: selections[('confirmation', dataset)]['indices'] for dataset in DATASETS})
    c.pairing_verified(receipt['state']['jobs'])
    output = ROOT / ('sensitivity_confirmation_cache_evidence_' + utc()); output.mkdir(exist_ok=False)
    for number, row in enumerate(records, 1):
        path = owned(row['local_path'], ROOT); relative = path.relative_to(ROOT)
        verify(path, row); verify(owned(row['drive_path'], DRIVE), row)
        copy_checked(path, output / relative, row)
        if number % 30 == 0:
            print('AGENT_B_SENSITIVITY_CONFIRMATION_CACHE_PROGRESS', number, 180, flush=True)
    for path in (receipt_path, validation_path, ledger_path, freeze_path, plan_path, controller):
        shutil.copyfile(path, output / path.name)
    (output / 'cache_records.json').write_text(json.dumps(records, indent=2) + '\n')
    archive = seal(output, DRIVE / 'phase-evidence' / output.name / 'confirmation_cache_evidence.zip')
    result = dict(status='verified', session=SESSION, phase_plan_sha256=PLAN_SHA, manifest_sha256=MANIFEST_SHA,
                  source_snapshot_sha256=SOURCE_SHA, cache_count=180, per_seed_cache_count=counts,
                  confirmation_phase_archive=phase_archive, inactive_ledger_identity=identity(ledger_path), archive=archive)
    (ROOT / 'sensitivity_confirmation_cache_evidence_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    (Path(archive['drive_path']).parent / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print('AGENT_B_SENSITIVITY_CONFIRMATION_CACHE_EVIDENCE', json.dumps(archive), flush=True)
    print('AGENT_B_SENSITIVITY_CONFIRMATION_CACHE_COLLECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
