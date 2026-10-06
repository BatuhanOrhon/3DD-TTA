"""CPU-only preparation, audit and packaging for the approved P_PC all15 run.

Inference remains exclusively in run_gsd_composition.py. Colab browser downloads
belong in the notebook kernel, never in this conda subprocess.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys
import uuid
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gsd_composition_protocol import (
    build_phase_plan, canonical_sha256, derive_draw_seeds, make_selection_manifest,
    validate_all15_reference, validate_arm_bundle, validate_selection_manifest,
)
from scripts.run_gsd_composition import build_plan

RESULT_ROOT = REPO / 'result/modelnet40_c/gsd_guidance_composition_v1'
REFERENCE_ZIP_SHA = '028d31bec666bf0bd0e4427e7a64799ade78f2b3f3eb2e283b0285a6ed1e3a29'
REFERENCE_MANIFEST_SHA = 'd2fc29a4c9bb7054a4c3d3339eb61452c3edb8de8e9f97954ecfa47c5cd54fd6'
RATIONALE = (
    'Phase 6 authorized by the user on 2026-10-06 after validated '
    'replicate/attempt-0001 passed every registered Phase 5 gate. '
    'Freeze canonical P_PC and compare with matched C_SCD on all15, '
    '2468 examples per corruption, seeds 0/1/2. Preserve the original '
    'inference settings. Full-test-set development performance comparison; '
    'not independent confirmation or a full-scope mechanism ablation.'
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict, *, preserve: bool = False) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if preserve and path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == value,
                'Existing preparation differs; inspect before replacing: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if preserve else 'w', encoding='utf-8') as out:
        out.write(text)


def selection_manifest() -> dict:
    return make_selection_manifest('replicate', 'P_PC', rationale=RATIONALE,
                                   relevant_comparators=['C_SCD'])


def extract_reference(archive: Path, destination: Path) -> Path:
    """Recover this specific accepted ZIP without replacing existing raw bytes."""
    require(file_sha(archive) == REFERENCE_ZIP_SHA, 'Unexpected Phase 5 ZIP hash')
    with ZipFile(archive) as z:
        require(z.testzip() is None, 'Reference ZIP CRC failure')
        names = z.namelist()
        require(len(names) == len({n.casefold() for n in names}), 'Duplicate ZIP paths')
        for info in z.infolist():
            name = PurePosixPath(info.filename)
            require(not name.is_absolute() and '..' not in name.parts and
                    '\\' not in info.filename and ':' not in info.filename and
                    (info.external_attr >> 16) & 0o170000 != 0o120000,
                    'Unsafe ZIP path: ' + info.filename)
            target = destination / info.filename
            require(not target.is_symlink(), 'Symlink in reference destination')
            target.resolve().relative_to(destination.resolve())
            data = z.read(info)
            if target.exists():
                require(target.read_bytes() == data, 'Existing reference differs: ' + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as out:
                    out.write(data)
    return destination


def prepare(reference: Path, result_root: Path) -> dict:
    """Validate the accepted reference and save a pure all15 plan preview."""
    prep = result_root / 'all15/preparation'
    if reference.is_file():
        reference = extract_reference(reference, prep / 'reference_raw')
    manifest_path = reference / 'phase_manifest.json'
    require(file_sha(manifest_path) == REFERENCE_MANIFEST_SHA,
            'Reference is not the accepted Phase 5 phase_manifest.json')
    raw = json.loads(manifest_path.read_text(encoding='utf-8'))
    # Relocate storage paths in a separate derivative; raw manifests stay immutable.
    relocated = copy.deepcopy(raw)
    for block in relocated['paired_blocks']:
        for row in block['arms']:
            local_zip = reference / 'arms' / PurePosixPath(row['zip_path']).name
            require(local_zip.is_file(), 'Missing reference arm ZIP: ' + str(local_zip))
            row['bundle_path'] = str(local_zip.resolve())
            row['zip_path'] = str(local_zip.resolve())
    selection = selection_manifest()
    validate_all15_reference(selection, relocated)
    # Retain the inference code used by the accepted replication.
    for name, identity in raw['resolved_ref']['source_manifest'].items():
        data = (REPO / name).read_bytes().replace(b'\r\n', b'\n')
        require(len(data) == identity['bytes'] and hashlib.sha256(data).hexdigest() == identity['sha256'],
                'Inference source differs from accepted Phase 5: ' + name)
    reference_copy = prep / 'reference_manifest.json'
    selection_path = result_root / 'all15/selection_manifest.json'
    write_json(reference_copy, relocated, preserve=True)
    write_json(selection_path, selection, preserve=True)
    argv = ['--phase', 'all15', '--result-root', str(result_root.resolve()),
            '--reference-manifest', str(reference_copy.resolve()),
            '--selection-manifest', str(selection_path.resolve())]
    plan = build_plan(argv)  # CPU bundle validation only; no CLI attempt creation.
    require([a['arm_id'] for a in plan['arms']] == ['C_SCD', 'P_PC'], 'Unexpected arms')
    require(plan['indices'] == list(range(2468)) and plan['seeds'] == [0, 1, 2], 'Unexpected scope')
    require(len(plan['corruptions']) == 15 and len(plan['blocks']) == 45, 'Unexpected block count')
    require(all(len(b['batches']) == 78 and len(b['batches'][-1]['indices']) == 4
                for b in plan['blocks']), 'Unexpected batch partition')
    preview = dict(phase='all15', arms=['C_SCD', 'P_PC'], corruptions=plan['corruptions'],
                   examples_per_corruption=2468, seeds=[0, 1, 2], paired_blocks=45,
                   arm_archives=90, classifications=222120, config_sha256=plan['config_sha256'],
                   resolved_ref=plan['resolved_ref'], selection_sha256=selection['selection_sha256'],
                   reference_original_sha256=REFERENCE_MANIFEST_SHA, runner_argv=argv,
                   inference_executed=False)
    write_json(prep / 'preview.json', preview)
    print(json.dumps(preview, indent=2))
    return preview


def audit_attempt(attempt: Path) -> dict:
    """Audit complete all15 artifacts. Reject partial output as complete accuracy."""
    m = json.loads((attempt / 'phase_manifest.json').read_text(encoding='utf-8'))
    require(m['phase'] == 'all15' and m['status'] == m['execution_status'] == 'complete',
            'Phase incomplete or failed; complete accuracy is unavailable')
    require(not m.get('arm_failures') and m['completed_blocks'] == 45, 'Incomplete blocks')
    selection = m['selection_manifest']
    validate_selection_manifest(selection)
    require(selection == selection_manifest(), 'Frozen Phase 6 selection differs')
    plan = build_phase_plan('all15', selection)
    require(m['indices'] == list(plan.indices) and m['seeds'] == list(plan.seeds)
            and m['corruptions'] == list(plan.corruptions), 'Scope differs')
    from dataclasses import asdict
    configs = {a.arm_id: asdict(a.config) for a in plan.arms}
    require({a['arm_id']: a['config'] for a in m['arms']} == configs, 'Arm config differs')
    cfg = m['config']
    require((cfg['batch_size'], cfg['ddim_total_steps'], cfg['normal_reverse_steps'],
             cfg['background_reverse_steps'], cfg['gamma'], cfg['eta'], cfg['lambda']) ==
            (32, 100, 5, 35, .01, .01, .95), 'Inference config differs')
    require(cfg['graph'] == dict(profile='hard-v1', k=10, delta=.1, graph_gamma=.6, modes=100),
            'Graph config differs')
    require(cfg['dataset'] == 'modelnet40_c' and cfg['severity'] == 5 and
            cfg['method'] == m['method'] == 'gsd_guidance_composition_v1', 'Method/data scope differs')
    require(cfg['model'] == dict(classifier='frozen Point-MAE', lion='raw frozen/eval, EMA off',
                                 global_prior='unused', decoder_style='original encoded shape latent'),
            'Model/decode config differs')
    ref = m['resolved_ref']
    # Colab uses a detached Phase6 worktree to preserve its pre-existing dirty
    # notebook checkout. The resolved commit and per-file source hashes below
    # carry the executable identity; an empty branch name is expected there.
    require(ref['branch'] in ('', 'gsd-smooth-spectrum'), 'Unexpected run branch')
    require(len(ref['source_manifest']) == 9, 'Incomplete inference source inventory')
    for name, identity in ref['source_manifest'].items():
        data = subprocess.check_output(['git', 'show', ref['commit'] + ':' + name], cwd=REPO)
        require(len(data) == identity['bytes'] and hashlib.sha256(data).hexdigest() == identity['sha256'],
                'Run source hash does not match its commit: ' + name)
        require((REPO / name).read_bytes().replace(b'\r\n', b'\n') == data,
                'Working inference source changed since run: ' + name)
    shared = dict(method=m['method'], dataset=cfg['dataset'], severity=5,
                  source_manifest=ref['source_manifest'], git_commit=ref['commit'],
                  runtime=cfg, draw_derivation_version=1)
    require(canonical_sha256(shared) == m['pairing_config_sha256'], 'Pairing fingerprint differs')
    phase_config = dict(shared, phase='all15', arms=[dict(arm_id=a['arm_id'], config=a['config']) for a in m['arms']],
                        corruptions=m['corruptions'], indices=m['indices'], seeds=m['seeds'],
                        selection_sha256=selection['selection_sha256'])
    require(canonical_sha256(phase_config) == m['config_sha256'] == m['experiment_fingerprint'],
            'Phase fingerprint differs')
    require(canonical_sha256(dict(planned_runtime_fingerprint=m['pairing_config_sha256'],
                                  runtime_identity=m['runtime_identity'])) == m['runtime_fingerprint'],
            'Runtime fingerprint differs')
    require(m['runtime_identity']['asset_manifest'] == m['asset_manifest'] and
            m['runtime_identity']['native_extensions'] == m['native_extension_manifest'], 'Runtime inventory differs')
    require(m['asset_manifest'] and m['native_extension_manifest'], 'Missing assets/native identities')
    require(set(m['asset_manifest']) == {'classifier_checkpoint', 'classifier_config', 'lion_checkpoint',
                                        'lion_config', 'labels'} | {'data_' + c for c in plan.corruptions},
            'Asset inventory differs')
    for item in list(m['asset_manifest'].values()) + [v['file'] for v in m['native_extension_manifest'].values()]:
        require(item['bytes'] > 0 and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid asset/native hash')
    blocks = m['paired_blocks']
    require(len(blocks) == 45 and {(b['corruption'], b['seed']) for b in blocks} ==
            {(c, s) for c in plan.corruptions for s in plan.seeds}, 'Block inventory differs')
    rows, seen = [], set()
    component_order = ['input_points', 'shape_latent', 'local_latent', 'style_conditioning', 'local_noise']
    for block in blocks:
        identity = block['identity']
        require(block['status'] == 'complete' and block['indices'] == identity['indices'] == m['indices'], 'Block incomplete')
        require(len(block['arms']) == 2 and {r['arm_id'] for r in block['arms']} == set(configs), 'Arm inventory differs')
        require(identity['runtime_fingerprint'] == identity['fingerprint'] == m['runtime_fingerprint'] and
                identity['locked_config_fingerprint'] == m['pairing_config_sha256'], 'Block fingerprint differs')
        require(block['block_id'] == canonical_sha256(dict(corruption=block['corruption'], seed=block['seed'],
                                                          indices=m['indices'], runtime_fingerprint=m['pairing_config_sha256'])),
                'Block ID differs')
        batches = identity['prepared_components']
        require(len(batches) == 78, 'Prepared batch count differs')
        input_hashes = []
        for number, batch in enumerate(batches):
            ix = m['indices'][number * 32:(number + 1) * 32]
            require(batch['batch_index'] == number and batch['indices'] == ix, 'Prepared indices differ')
            draw = derive_draw_seeds(cfg['dataset'], block['corruption'], block['seed'], ix)
            require(identity['preparation_keys'][number] == draw['preparation_key'] and
                    identity['classification_keys'][number] == draw['classification_key'], 'Draw keys differ')
            comp = batch['components']
            require(set(comp) == set(component_order + ['timesteps', 'alpha_bar']), 'Component inventory differs')
            require(all(re.fullmatch('[0-9a-f]{64}', value) for value in
                        [*comp.values(), batch['scheduler_config_sha256']]), 'Malformed component hashes')
            input_hashes.append(hashlib.sha256(''.join(comp[k] for k in component_order).encode('ascii')).hexdigest())
        require(canonical_sha256(input_hashes) == identity['input_sha256'], 'Prepared input fingerprint differs')
        labels = None
        for record in block['arms']:
            path = attempt / 'arms' / PurePosixPath(record['zip_path']).name
            require(path not in seen, 'Duplicate arm ZIP')
            seen.add(path)
            with ZipFile(path) as z:
                require(z.testzip() is None, 'Arm ZIP CRC failure')
            bundle = validate_arm_bundle(path)
            arm = bundle['manifest']
            require(arm['status'] == record['status'] == 'complete' and arm['phase'] == 'all15', 'Arm incomplete')
            require((arm['arm_id'], arm['corruption'], arm['seed'], arm['block_id']) ==
                    (record['arm_id'], block['corruption'], block['seed'], block['block_id']), 'Arm identity differs')
            require(arm['identity'] == record['identity'] == identity, 'Prepared identity differs between arms')
            require(arm['original_indices'] == bundle['indices'].tolist() == m['indices'] and
                    bundle['n_examples'] == 2468 and arm['config'] == configs[arm['arm_id']], 'Arm scope/config differs')
            for key in ('runtime_identity', 'runtime_fingerprint', 'asset_manifest',
                        'native_extension_manifest', 'experiment_fingerprint'):
                require(arm[key] == m[key], 'Arm provenance differs: ' + key)
            require(arm['input_sha256'] == identity['input_sha256'] and
                    arm['locked_config_fingerprint'] == m['pairing_config_sha256'], 'Arm hashes differ')
            if labels is None:
                labels = bundle['labels'].tolist()
            require(labels == bundle['labels'].tolist(), 'Paired labels differ')
            steps = 35 if block['corruption'] == 'background' else 5
            step_rows = [r for r in bundle['diagnostics'] if r['kind'] == 'step']
            require(len(step_rows) == 2468 * steps and
                    {(r['original_index'], r['step_index'], r['timestep']) for r in step_rows} ==
                    {(i, s, (steps - s - 1) * 10) for i in m['indices'] for s in range(steps)},
                    'Sample-step coverage differs')
            rows.append(dict(arm_id=arm['arm_id'], corruption=arm['corruption'], seed=arm['seed'],
                             n_correct=bundle['n_correct'], n_examples=bundle['n_examples'],
                             total_runtime_seconds=arm['total_runtime_seconds'],
                             peak_gpu_memory_mb=arm['peak_gpu_memory_mb']))
            del bundle, step_rows  # Do not retain all per-step diagnostics during this audit.
        print('Validated paired block:', block['corruption'], block['seed'], flush=True)
    require(seen == set((attempt / 'arms').glob('*.zip')) and len(seen) == 90, 'Unexpected ZIP inventory')
    require(sum(r['n_examples'] for r in rows) == 222120, 'Classification count differs')
    return dict(validated_complete=True, paired_blocks=45, arm_archives=90,
                classifications=222120, arm_rows=rows,
                provenance_limit='Recorded component/assets/native hashes agree; original tensors/binaries are not bundled.')


def accuracy_summary(rows: list) -> dict:
    scopes = dict(macro15=lambda c: True, macro14_without_background=lambda c: c != 'background',
                  background=lambda c: c == 'background')
    result = {}
    for name, include in scopes.items():
        values = {arm: [statistics.mean(100 * r['n_correct'] / r['n_examples'] for r in rows
                                       if r['arm_id'] == arm and r['seed'] == seed and include(r['corruption']))
                        for seed in (0, 1, 2)] for arm in ('C_SCD', 'P_PC')}
        delta = [p - c for p, c in zip(values['P_PC'], values['C_SCD'])]
        result[name] = dict(seed_accuracy_percent=values, seed_delta_pp=delta,
                            mean_delta_pp=statistics.mean(delta), sample_sd_delta_pp=statistics.stdev(delta))
    return result


def package_attempt(attempt: Path, archive: Path, extra_files: dict) -> str:
    """Package every raw file; omit only byte-identical directory copies of ZIP members."""
    require(attempt.is_dir(), 'Attempt directory does not exist')
    require(not archive.exists(), 'Refusing to overwrite existing package')
    # Build exact duplicate file identities. A divergent/unsealed raw file is retained.
    duplicates = {}
    for arm_zip in sorted((attempt / 'arms').glob('*.zip')):
        try:
            with ZipFile(arm_zip) as z:
                require(z.testzip() is None, 'CRC failure')
                for info in z.infolist():
                    name = PurePosixPath(info.filename)
                    require(not name.is_absolute() and '..' not in name.parts and
                            '\\' not in info.filename and ':' not in info.filename, 'Unsafe arm ZIP path')
                    target = attempt / 'arms' / info.filename
                    if target.is_file() and target.stat().st_size == info.file_size:
                        digest = hashlib.sha256(z.read(info)).hexdigest()
                        if file_sha(target) == digest:
                            duplicates[target] = True
        except (OSError, ValueError, RuntimeError, EOFError, BadZipFile) as error:
            print('Retaining raw files for unreadable arm ZIP:', arm_zip.name, str(error), flush=True)
    pending = archive.with_name(archive.name + '.partial')
    expected = []
    with ZipFile(pending, 'x', allowZip64=True) as z:
        for item in sorted(attempt.rglob('*')):
            if item.is_file() and item not in duplicates:
                name = item.relative_to(attempt).as_posix()
                z.write(item, arcname=name, compress_type=ZIP_STORED if item.suffix in ('.zip', '.gz') else ZIP_DEFLATED)
                expected.append(name)
        for name, source in extra_files.items():
            require(name not in expected, 'Package extra collides with raw file')
            z.write(source, arcname=name, compress_type=ZIP_DEFLATED)
            expected.append(name)
    with ZipFile(pending) as z:
        require(z.namelist() == expected and z.testzip() is None, 'Output package verification failed')
    pending.rename(archive)
    digest = file_sha(archive)
    archive.with_suffix('.zip.sha256').write_text(digest + '  ' + archive.name + '\n', encoding='ascii')
    return digest


def finalize(attempt: Path, receipt: Path, log: Path = None) -> dict:
    """Package raw output even when auditing or offline analysis fails."""
    require(attempt.is_dir(), 'No attempt exists to package: ' + str(attempt))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    postrun = attempt / ('postrun-' + stamp)
    postrun.mkdir()
    status = dict(validated_complete=False, analysis_complete=False, errors=[])
    try:
        audit = audit_attempt(attempt)
        write_json(postrun / 'audit.json', audit)
        write_json(postrun / 'accuracy_summary.json', accuracy_summary(audit['arm_rows']))
        status['validated_complete'] = True
        # Conda subprocess inherits its own Python. No google.colab import here.
        code = subprocess.call([sys.executable, str(REPO / 'scripts/analyze_gsd_composition.py'),
                                '--result-root', str(attempt), '--output', str(postrun / 'analysis')], cwd=REPO)
        require(code == 0 and (postrun / 'analysis/analysis.json').is_file(),
                'Analyzer failed with exit code ' + str(code))
        status['analysis_complete'] = True
    except Exception as error:
        status['errors'].append(type(error).__name__ + ': ' + str(error))
        print('Complete result not certified; packaging raw output:', status['errors'][-1], flush=True)
    write_json(postrun / 'status.json', status)
    extras = {'_handoff/gsd_all15_colab.py': Path(__file__).resolve()}
    if log is not None and log.is_file():
        extras['_handoff/execution.log'] = log
    for name in ('selection_manifest.json', 'preparation/preview.json', 'preparation/reference_manifest.json'):
        source = attempt.parent / name
        if source.is_file():
            extras['_handoff/' + name] = source
    label = 'validated' if status['validated_complete'] and status['analysis_complete'] else 'UNVALIDATED'
    archive = attempt.parent / (attempt.name + '_all15_' + label + '_' + stamp + '.zip')
    print('Packaging all available output:', archive, flush=True)
    digest = package_attempt(attempt, archive, extras)
    result = dict(status, archive=str(archive.resolve()), sha256=digest, bytes=archive.stat().st_size)
    write_json(receipt, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--reference', type=Path, required=True, help='Accepted Phase 5 directory or outer ZIP')
    prep.add_argument('--result-root', type=Path, default=RESULT_ROOT)
    done = commands.add_parser('finalize')
    done.add_argument('--attempt', type=Path, required=True)
    done.add_argument('--receipt', type=Path, required=True)
    done.add_argument('--log', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.reference.resolve(), args.result_root.resolve())
    else:
        finalize(args.attempt.resolve(), args.receipt.resolve(), args.log)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
