"""Prepare the fixed Chamfer campaign and enforce its explicit whole-GPU gate.

Preparation is CPU-only. This module intentionally has no worker-launch mode:
the reviewed command ledger and approval validation are the handoff boundary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chamfer_calibration_report import validate_calibration_report

ARMS = ('baseline', 'sparse', 'semantic_kl', 'semantic_h', 'weighted')
SHAM_ARMS = ('sham_kl', 'sham_h')
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')
STAGES = ('calibration', 'smoke', 'development', 'confirmation', 'timing')
SESSION = 'tta-chamfer-guidance-variants-a100-20261030-r1'
CAP_SECONDS = 18000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def verify_reference(reference: dict) -> Path:
    path = Path(reference['path'])
    require(path.is_absolute() and '..' not in path.parts and not path.is_symlink(), 'Unsafe input reference')
    require(identity(path) == {key: reference[key] for key in ('bytes', 'sha256')}, 'Input SHA/size mismatch: ' + str(path))
    return path


def verify_approval(path: Path, approval_sha256: str, plan_sha256: str, plan: dict) -> dict:
    """Verify the retained explicit reply before a separate GPU launcher runs."""
    require(identity(path)['sha256'] == approval_sha256, 'Approval file SHA mismatch')
    approval = json.loads(Path(path).read_text())
    require(approval.get('status') == 'approved' and approval.get('plan_sha256') == plan_sha256
            and approval.get('session') == plan.get('session') == SESSION
            and approval.get('cap_seconds') == plan.get('cap_seconds') == CAP_SECONDS
            and isinstance(approval.get('user_reply'), str) and bool(approval['user_reply'].strip()),
            'Explicit approval of this exact whole campaign/session/5h cap is required')
    return approval


def verify_calibration(path: Path) -> dict:
    """Accept one GPU label-free calibration, never synthetic or accuracy-fit weights."""
    return validate_calibration_report(json.loads(Path(path).read_text()))


def validate_plan(plan: dict) -> None:
    require(plan.get('schema_version') == 1 and plan.get('session') == SESSION
            and plan.get('cap_seconds') == CAP_SECONDS, 'Unregistered campaign/session/cap')
    for key in ('research_root', 'drive_root', 'source_root', 'asset_root', 'python', 'calibration_report'):
        value = Path(plan[key])
        require(value.is_absolute() and '..' not in value.parts, 'Absolute safe path required: ' + key)
    require(Path(plan['research_root']) != Path(plan['drive_root']), 'Local and Drive roots must differ')
    for key in ('asset_manifest', 'source_receipt'):
        require(re.fullmatch('[0-9a-f]{64}', plan[key]['sha256']), 'Missing pinned ' + key)
    require(set(plan['datasets']) == set(DATASETS), 'Both datasets required')
    require(set(plan['selections']) == {'smoke', 'development', 'confirmation'}, 'Three fixed selection stages required')
    for values in plan['selections'].values():
        require(set(values) == set(DATASETS), 'Selection coverage changed')


def verify_inputs(plan: dict) -> dict:
    """CPU read-only preflight; checkpoint/data coverage stays in the asset manifest."""
    validate_plan(plan)
    manifest_path = verify_reference(plan['asset_manifest'])
    receipt_path = verify_reference(plan['source_receipt'])
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('status') in ('complete', 'validated', 'sealed_CPU_only'), 'Source preparation receipt is not accepted')
    selected = {}
    for stage, datasets in plan['selections'].items():
        for dataset, reference in datasets.items():
            path = verify_reference(reference)
            values = json.loads(path.read_text())
            ids = values.get('indices', [])
            total = 2468 if dataset == 'modelnet40_c' else 581
            count = {'smoke': 2, 'development': 70, 'confirmation': 140}[stage]
            require(values.get('dataset') == dataset and values.get('n_examples') == total
                    and values.get('stage') == ('heldout' if stage == 'confirmation' else stage)
                    and len(ids) == count and ids == sorted(set(ids))
                    and all(type(i) is int and 0 <= i < total for i in ids), 'Fixed selection mismatch')
            selected[(stage, dataset)] = set(ids)
            relative = path.relative_to(Path(plan['asset_root'])).as_posix()
            require(manifest.get('files', {}).get(relative) == {key: reference[key] for key in ('bytes', 'sha256')},
                    'Selection is not sealed by asset manifest')
    for dataset in DATASETS:
        require(selected[('smoke', dataset)] <= selected[('development', dataset)]
                and not selected[('development', dataset)] & selected[('confirmation', dataset)],
                'Smoke/dev/heldout object split changed or overlaps')
    return dict(status='verified_cpu_inputs', bulk_assets_verified=False,
                asset_manifest=plan['asset_manifest'], source_receipt=plan['source_receipt'])


def _common_command(plan, dataset, stage, seed, result_root, drive_root):
    values = plan['datasets'][dataset]
    selection_stage = 'development' if stage in ('calibration', 'timing') else stage
    command = [plan['python'], '-u', str(Path(plan['source_root']) / 'scripts/eval_multibackbone_guidance.py'),
               '--dataset', dataset, '--dataset-root', values['dataset_root'], '--label-path', values['label_path'],
               '--asset-root', plan['asset_root'], '--asset-manifest-sha256', plan['asset_manifest']['sha256'],
               '--result-root', str(result_root), '--drive-root', str(drive_root), '--seed', str(seed),
               '--batch-size', '70', '--backbones', *HEADS, '--method', 'gsd', '--component-arm', 'chamfer',
               '--trajectory-observe', '--eta', '0.01', '--gamma', '0.01', '--p', '0.95', '--spectral-weight', '0',
               '--chamfer-weight', '1', '--index-manifest', plan['selections'][selection_stage][dataset]['path'],
               '--pointmae-config', values['pointmae_config'], '--pointmae-checkpoint', values['pointmae_checkpoint']]
    if stage not in ('timing', 'calibration'):
        command += ['--paired-state-root', str(Path(plan['research_root']) / 'paired_states' / stage / ('seed%d' % seed))]
    return command


def build_jobs(plan: dict, stage: str, coefficients: dict = None, calibration_sha256: str = None) -> list:
    """Build fixed argv; unresolved calibration tokens are explicitly non-executable."""
    validate_plan(plan)
    require(stage in STAGES, 'Unknown campaign stage')
    if coefficients is not None:
        require(set(coefficients) == {'semantic_kl', 'semantic_h', 'weighted_scale'}
                and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in coefficients.values())
                and isinstance(calibration_sha256, str) and re.fullmatch('[0-9a-f]{64}', calibration_sha256),
                'Resolved commands require positive coefficients and pinned calibration report')
    jobs = []
    for repeat in (range(3) if stage == 'timing' else (0,)):
        arms = ARMS[repeat:] + ARMS[:repeat] if stage == 'timing' else ARMS + SHAM_ARMS if stage == 'smoke' else ARMS
        for seed in ((0, 1, 2) if stage == 'confirmation' else (0,)):
            for dataset in DATASETS:
                for arm in (('calibration',) if stage == 'calibration' else arms):
                    relative = Path(stage) / ('repeat%d' % repeat) / ('seed%d' % seed) / dataset / arm
                    result_root = Path(plan['research_root']) / 'results' / relative
                    drive_root = Path(plan['drive_root']) / 'results' / relative
                    command = _common_command(plan, dataset, stage, seed, result_root, drive_root)
                    if stage == 'calibration':
                        command[2:3] = [str(Path(plan['source_root']) / 'scripts/calibrate_chamfer_variants.py'), 'run']
                        position = command.index('--backbones')
                        command[position+1:position+5] = ['pointmae']
                    elif arm != 'baseline':
                        variant = 'semantic' if arm.startswith(('semantic_', 'sham_')) else arm
                        command[2] = str(Path(plan['source_root']) / ('scripts/eval_chamfer_' + variant + '.py'))
                        if arm == 'sparse':
                            command += ['--sparse-interval', '3']
                        elif arm in SHAM_ARMS:
                            command += ['--semantic-kind', 'kl' if arm == 'sham_kl' else 'marginal_entropy', '--semantic-weight', '0']
                        else:
                            command += ['--calibration-report', plan['calibration_report'], '--calibration-report-sha256',
                                        calibration_sha256 or '@calibration:report_sha256']
                            if arm == 'weighted':
                                command += ['--density-k', '10']
                                rate = str(.01 * coefficients['weighted_scale']) if coefficients else '@calibration:0.01*weighted_scale'
                                command[command.index('--eta')+1] = rate
                                command[command.index('--gamma')+1] = rate
                            else:
                                command += ['--semantic-kind', 'kl' if arm == 'semantic_kl' else 'marginal_entropy',
                                            '--semantic-weight', str(coefficients[arm]) if coefficients else '@calibration:' + arm]
                    command += ['--corruptions', 'gaussian', 'background'] if stage in ('smoke', 'timing') else ['--corruptions', 'all']
                    jobs.append(dict(stage=stage, arm=arm, dataset=dataset, seed=seed, repeat=repeat,
                                     role='technical_sham' if arm in SHAM_ARMS else 'label_free_calibration' if stage == 'calibration' else 'independent_arm',
                                     result_root=str(result_root), drive_root=str(drive_root), command=command,
                                     executable=not any(value.startswith('@calibration:') for value in command)))
    return jobs


def estimate_campaign() -> dict:
    stages = dict(calibration=900, smoke=840, development=10*178, confirmation=30*300, timing=30*42)
    nominal = sum(stages.values())
    return dict(stage_seconds=stages, nominal_seconds=nominal, planning_margin_fraction=.1,
                packing_retry_reserve_seconds=1800, projected_seconds_with_reserve=round(1.1*nominal+1800),
                hard_cap_seconds=CAP_SECONDS, development_jobs=10, confirmation_jobs=30, timing_jobs=30,
                technical_smoke_jobs=14,
                basis='Prior all15 C: dev70 about178s; heldout140 about300s per dataset/job. '
                      'Timing Gaussian+background70 about42s. Calibration/smoke remain uncertain. '
                      'Re-estimate the entire remaining phase after smoke; stop if the5h cap cannot hold.')


def prepare(plan: dict, output: Path, verify_input_files: bool = False) -> dict:
    """Write a fresh review artifact; never allocate a session or invoke a worker."""
    validate_plan(plan)
    preflight = verify_inputs(plan) if verify_input_files else dict(status='not_verified', reason='preapproval argv preparation only')
    calibration_path = Path(plan['calibration_report'])
    calibration = verify_calibration(calibration_path) if calibration_path.exists() else None
    coefficient_sha = identity(calibration_path)['sha256'] if calibration else None
    jobs = {stage: build_jobs(plan, stage, calibration['coefficients'] if calibration else None, coefficient_sha) for stage in STAGES}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='prepared_not_executed', session=SESSION, full_benchmark=False,
                  plan=plan, inputs=preflight, estimate=estimate_campaign(), stages=jobs,
                  calibration_summary_command=[plan['python'], '-u', str(Path(plan['source_root']) / 'scripts/calibrate_chamfer_variants.py'),
                                               'summarize', '@completed:modelnet_pointmae_bundle', '@completed:scan_pointmae_bundle',
                                               '--output', plan['calibration_report']],
                  calibration_rule='semantic_kl=.1/median_KL_to_C_actual_update; semantic_h=.1/median_H_to_C_actual_update; '
                                   'weighted_scale=1/median_weighted_to_C_actual_update; no label/accuracy selection',
                  calibration_heads=['pointmae'], evaluation_heads=list(HEADS),
                  smoke_acceptance='Before accuracy: real CUDA/native KL/H gradients must be finite/nonzero for local and style; '
                                   'semantic weight0 KL and H shams must match baseline decoded geometry and all4head logits exactly '
                                   'on the same cached2IDs, Gaussian/background noise and frozen model states.',
                  execution_gate='Before any GPU worker: verify exact plan SHA, explicit approval file SHA/user reply/session/5h cap, '
                                 'source/asset/selection references and fresh result/Drive directories. '
                                 'Freeze actual calibration report SHA before smoke. This preparer launches no workers.',
                  stopping='Stop on source/native/data/frozen-state mismatch, nonfinite gradient/loss, nonzero graph/spectral calls, '
                           'missing/shrunken expectedIDs, broken cache pairing, artifact-copy failure or whole-phase cap projection. '
                           'No automatic retries, directory reuse, combined arms or accuracy HPO.')
    (output / 'campaign_commands.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-input-files', action='store_true')
    parser.add_argument('--approval-file', type=Path)
    parser.add_argument('--approval-file-sha256')
    args = parser.parse_args(argv)
    require(identity(args.plan)['sha256'] == args.plan_sha256, 'Campaign plan SHA mismatch')
    plan = json.loads(args.plan.read_text())
    if args.approval_file is not None:
        verify_approval(args.approval_file, args.approval_file_sha256, args.plan_sha256, plan)
    report = prepare(plan, args.output, args.verify_input_files)
    print(json.dumps(dict(status=report['status'], jobs=sum(len(v) for v in report['stages'].values()), estimate=report['estimate'])))


if __name__ == '__main__':
    main()
