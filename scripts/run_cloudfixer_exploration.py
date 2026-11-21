#!/usr/bin/env python3
"""Bounded, resumable N6..12 and dense N8 diagnostics on the fixed development IDs."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import Budget, MountStore, canonical, commit_unit, load_units, record_setup, sha, unit_key
from scripts import run_cloudfixer_transfer as transfer

PRIOR_ARCHIVE_SHA = '28baa27f2b80ae32c42fb5b56d569f8bb3dfac4e94fb6c055a2e89d1dabf50e7'


def registration() -> dict:
    previous = transfer.registration()
    return dict(schema=1, dataset=previous['dataset'], severity=5,
        selection_seed=previous['selection_seed'], development_ids=previous['development_ids'],
        corruptions=previous['corruptions'], batch_size=32, seeds=[0], retained=.95,
        steps=list(range(6, 13)), start=80, tau=.1, budget_seconds=28800,
        repeat_policy='strict same input/RNG, logits allclose atol1e-4 rtol1e-5 and identical argmax',
        exposure='Exploratory reuse of accepted development IDs; not confirmation or full benchmark')


def units(campaign: str, reg: dict, role: str) -> list:
    if role not in ('nstep', 'entropy'):
        raise ValueError('Unknown exploration role')
    result = []
    for corruption in reg['corruptions']:
        for start in range(0, len(reg['development_ids']), reg['batch_size']):
            for steps in (reg['steps'] if role == 'nstep' else [8]):
                config = dict(method='proximal', steps=steps, start=reg['start'], tau=reg['tau'], retained=reg['retained'])
                result.append(dict(campaign=campaign, phase='exploration', role=role,
                    config=config, config_id='proximal_'+sha(canonical(config))[:12],
                    corruption=corruption, seed=0, ids=reg['development_ids'][start:start+reg['batch_size']]))
    return result


def probability_stats(logits: np.ndarray) -> dict:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values-values.max(-1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = exponentials.sum(-1, keepdims=True)
    logp = shifted-np.log(denominator)
    probabilities = exponentials/denominator
    entropy = -(probabilities*logp).sum(-1)
    return dict(probabilities=probabilities, entropy=entropy,
        normalized_entropy=entropy/np.log(40), max_probability=probabilities.max(-1))


def unpack(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if (len(archive.namelist()) != len(set(archive.namelist()))
                or sum(item.file_size for item in archive.infolist()) > 100_000_000):
            raise ValueError('Unsafe prediction archive')
    with np.load(io.BytesIO(data), allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def pack(values: dict) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def validate_provenance(payload: dict, manifest: dict, environment_bytes: bytes) -> None:
    if (str(payload['source_sha256']) != sha(canonical(manifest))
            or str(payload['environment_sha256']) != sha(environment_bytes)):
        raise ValueError('Source manifest or immutable environment receipt hash mismatch')
    environment = json.loads(environment_bytes)
    if (payload['model_hashes'].tolist() != environment.get('model_hashes')
            or environment.get('gamma_verified') is not True or environment.get('lion_loaded') is not False):
        raise ValueError('Frozen-model identities or native-model environment mismatch')


def compare_prior(current: dict, previous: dict) -> dict:
    for field in ('ids', 'labels', 'model_hashes', 'input_shape'):
        if not np.array_equal(current[field], previous[field]):
            raise ValueError('Old N8 parity identity mismatch: '+field)
    a, b = (json.loads(str(p['sampler'])) for p in (current, previous))
    if (not re.fullmatch('[a-f0-9]{64}', a.get('input_rng_sha256', ''))
            or a['input_rng_sha256'] != b.get('input_rng_sha256')):
        raise ValueError('Old N8 parity input/RNG mismatch')
    logits = current['logits']
    if logits.shape[1] == 9:
        logits = logits[:, [1, 3, 5, 8]]
    baseline = previous['logits']
    if (logits.shape != baseline.shape or logits.shape[1:] != (4, 40)
            or not np.isfinite(logits).all() or not np.isfinite(baseline).all()):
        raise ValueError('Old N8 parity shape/nonfinite mismatch')
    changes = int((logits.argmax(-1) != baseline.argmax(-1)).sum())
    difference = float(np.abs(logits.astype(np.float64)-baseline).max())
    if changes or not np.allclose(logits, baseline, atol=1e-4, rtol=1e-5):
        raise ValueError('Old N8 canonical-four logit parity failed')
    current_pm = probability_stats(logits)['probabilities'].mean(1)
    previous_pm = probability_stats(baseline)['probabilities'].mean(1)
    pm_changes = int((current_pm.argmax(-1) != previous_pm.argmax(-1)).sum())
    if pm_changes:
        raise ValueError('Old N8 canonical-four PM prediction parity failed')
    return dict(status='passed', max_abs=difference, argmax_changes=changes,
        pm_argmax_changes=pm_changes, pm_max_probability_abs=float(np.abs(current_pm-previous_pm).max()))


class PriorArchive:
    """Read canonical old units by their recomputed campaign and receipt key."""
    def __init__(self, path: Path, manifest: dict, *, expected_sha: str = PRIOR_ARCHIVE_SHA):
        if transfer.file_hash(path) != expected_sha:
            raise ValueError('Pinned prior archive SHA256 mismatch')
        self.archive = zipfile.ZipFile(path)
        names = self.archive.namelist()
        configs = [name for name in names if name.endswith('/config.json')]
        if len(names) != len(set(names)) or len(configs) != 1:
            raise ValueError('Ambiguous prior archive identity')
        config = json.loads(self.archive.read(configs[0]))
        self.manifest = config['manifest']
        self.campaign = sha(canonical(dict(manifest=self.manifest, phase='development', selected=None)))
        if (config['campaign'] != self.campaign or config['phase'] != 'development'
                or config['execution_status'] != 'complete' or self.manifest['registration'] != transfer.registration()
                or manifest.get('prior_manifest_sha256') != sha(canonical(self.manifest))):
            raise ValueError('Prior campaign or registered manifest mismatch')
        for field in ('official_files', 'dataset_files', 'classifier_sha256', 'diffusion_sha256', 'author_commit'):
            if manifest[field] != self.manifest[field]:
                raise ValueError('Old/new asset identity differs: '+field)
        self.prefix = configs[0].rsplit('/', 1)[0]+'/'
        self.digest = expected_sha

    def check(self, current: dict, unit: dict) -> dict:
        old_unit = {key: value for key, value in unit.items() if key != 'role'}
        old_unit.update(campaign=self.campaign, phase='development')
        name = self.prefix+'pending_'+unit_key(old_unit)+'.npz'
        previous = transfer.validate_payload(self.archive.read(name), old_unit)
        result = compare_prior(current, previous)
        return dict(result, old_archive_sha256=self.digest, old_unit_key=unit_key(old_unit))


def validate_payload(data: bytes, unit: dict, labels=None, *, prior=None, source_sha256=None) -> dict:
    try:
        p = unpack(data)
        n, steps = len(unit['ids']), unit['config']['steps']
        loops = list(range(1, steps+1)) if unit['role'] == 'entropy' else [steps//4, steps//2, 3*steps//4]
        grid = [round(unit['config']['start']*(steps-i)/steps) for i in range(steps+1)]
        metadata = json.loads(str(p['sampler']))
        if (json.loads(str(p['unit'])) != unit or not np.array_equal(p['ids'], unit['ids'])
                or p['logits'].shape != (n, len(loops)+1, 40) or p['logits'].dtype != np.float32
                or not np.isfinite(p['logits']).all() or p['labels'].shape != (n,)
                or p['labels'].dtype.kind not in 'iu' or not ((p['labels'] >= 0) & (p['labels'] < 40)).all()
                or int(p['denoiser_calls']) != steps+1 or metadata['denoiser_calls'] != steps+1
                or metadata['indices'] != grid or metadata['candidate_loops'] != loops+[steps]
                or metadata['candidate_indices'] != [grid[i-1] for i in loops]+[0]
                or not math.isfinite(float(p['seconds'])) or float(p['seconds']) <= 0
                or not math.isfinite(float(p['peak_mb'])) or float(p['peak_mb']) < 0):
            raise ValueError('Prediction identity, logits, trajectory, calls or timing mismatch')
        shape, counts = p['input_shape'].tolist(), metadata['valid_reference_counts']
        if (len(shape) != 3 or shape[0] != n or shape[2] != 3 or shape[1] < 1
                or len(counts) != n or any(type(c) is not int or not 1 <= c <= shape[1] for c in counts)
                or metadata['retained_forward_count'] != int(shape[1]*unit['config']['retained'])
                or metadata['retained_reference_counts'] != [int(c*unit['config']['retained']) for c in counts]):
            raise ValueError('Invalid observed mask or retained-count evidence')
        if labels is not None and not np.array_equal(p['labels'], labels[unit['ids']]):
            raise ValueError('Pinned original-ID labels differ')
        hashes = [str(p['source_sha256']), str(p['environment_sha256']), metadata['input_rng_sha256'], *p['model_hashes'].tolist()]
        if len(p['model_hashes']) != 2 or any(not re.fullmatch('[a-f0-9]{64}', h) for h in hashes):
            raise ValueError('Missing source/environment/model/input identity')
        if source_sha256 is not None and str(p['source_sha256']) != source_sha256:
            raise ValueError('Restored payload belongs to another source manifest')
        for field, expected in probability_stats(p['logits']).items():
            if (p[field].dtype != np.float64 or p[field].shape != expected.shape
                    or not np.isfinite(p[field]).all() or not np.allclose(p[field], expected, atol=1e-12, rtol=1e-12)):
                raise ValueError('Saved probability/entropy diagnostics differ from logits')
        parity = json.loads(str(p['parity']))
        if steps == 8:
            if (parity.get('status') != 'passed' or parity.get('old_archive_sha256') != PRIOR_ARCHIVE_SHA
                    or parity.get('argmax_changes') != 0 or parity.get('pm_argmax_changes') != 0
                    or not re.fullmatch(r'unit_[a-f0-9]{64}\.json', parity.get('old_unit_key', ''))
                    or not math.isfinite(parity.get('max_abs', math.nan))
                    or not math.isfinite(parity.get('pm_max_probability_abs', math.nan))):
                raise ValueError('Missing accepted old N8 parity')
            if prior is not None and prior.check(p, unit) != parity:
                raise ValueError('Restored old N8 parity receipt differs')
        return p
    except (KeyError, IndexError, TypeError, zipfile.BadZipFile) as error:
        raise ValueError('Malformed exploration payload') from error


def execute_pending(store, campaign: str, wanted: list, infer, *, cap: float, labels=None,
                    progress=None, budget=None, validate=None) -> list:
    budget = budget or Budget(store, cap=cap)
    validate = validate or (lambda data, unit: validate_payload(data, unit, labels))
    with budget.operation(1800):
        accepted = load_units(store, campaign)
        expected = {unit_key(u): u for u in wanted}
        if len(expected) != len(wanted) or not set(accepted) <= set(expected):
            raise ValueError('Incompatible registered coverage')
        for key, record in accepted.items():
            validate(record['data'], expected[key])
    for unit in wanted:
        key = unit_key(unit)
        if key in accepted:
            continue
        with budget.operation(300):
            data = infer(unit)
            validate(data, unit)
            commit_unit(store, unit, data)
            accepted[key] = dict(unit=unit, data=data)
        if progress:
            progress(dict(completed=len(accepted), total=len(wanted), last=unit,
                spent_seconds=cap-budget.remaining(), cap_seconds=cap))
    return list(accepted.values())


def _check_repeat(first: dict, repeated: dict) -> dict:
    for field in ('ids', 'labels', 'model_hashes', 'source_sha256', 'input_shape'):
        if not np.array_equal(first[field], repeated[field]):
            raise ValueError('Strict repeat identity mismatch: '+field)
    a, b = (json.loads(str(p['sampler'])) for p in (first, repeated))
    if a['input_rng_sha256'] != b['input_rng_sha256']:
        raise ValueError('Strict repeat input/RNG mismatch')
    x, y = first['logits'], repeated['logits']
    if x.shape != y.shape or not np.allclose(x, y, atol=1e-4, rtol=1e-5) or np.any(x.argmax(-1) != y.argmax(-1)):
        raise ValueError('Strict repeated logits failed')
    return dict(max_abs=float(np.abs(x.astype(np.float64)-y).max()), argmax_changes=0, atol=1e-4, rtol=1e-5)


def source_repeat_gate(store, unit: dict, infer, verify, *, budget, source_sha256: str,
                       environment_bytes: bytes, labels=None) -> dict:
    config = dict(method='source')
    source = dict(unit, config=config, config_id='source_'+sha(canonical(config))[:12])
    with budget.operation(300):
        if 'source_repeat_gate.json' in store.keys('source_repeat_gate.json'):
            gate = json.loads(store.get('source_repeat_gate.json'))
            if gate.get('status') != 'passed' or gate.get('unit_key') != unit_key(source):
                raise ValueError('Source-repeat gate belongs to another unit')
            data = [store.get(gate[field]) for field in ('first_payload', 'repeated_payload')]
            if any(gate[field] != 'source_smoke_'+sha(value)+'.npz'
                    for field, value in zip(('first_payload', 'repeated_payload'), data)):
                raise ValueError('Source-repeat evidence hash mismatch')
        else:
            data = []
            for _ in range(2):
                value = infer(source)
                store.put('source_smoke_'+sha(value)+'.npz', value)
                data.append(value)
            gate = dict(status='passed', unit_key=unit_key(source), source_sha256=source_sha256,
                first_payload='source_smoke_'+sha(data[0])+'.npz', repeated_payload='source_smoke_'+sha(data[1])+'.npz')
        payloads = [transfer.validate_payload(value, source, labels) for value in data]
        for payload in payloads:
            payload['source_sha256'] = np.array(source_sha256)
            if payload['model_hashes'].tolist() != json.loads(environment_bytes)['model_hashes']:
                raise ValueError('Source-repeat frozen models differ from inference environment')
        repeat = _check_repeat(*payloads)
        if 'repeat' in gate and gate['repeat'] != repeat:
            raise ValueError('Restored source-repeat evidence differs')
        if gate.get('source_sha256') != source_sha256:
            raise ValueError('Source-repeat source identity differs')
        gate['repeat'] = repeat
        verify()
        store.put('source_repeat_gate.json', canonical(gate))
        return gate


def smoke_gate(store, reopen, campaign: str, wanted: list, infer, verify, *, budget,
               validate=validate_payload) -> dict:
    first = next(u for u in wanted if u['config']['steps'] == 8)
    key = unit_key(first)
    with budget.operation(300):
        restored = load_units(store, campaign)
        expected = {unit_key(u) for u in wanted}
        if not set(restored) <= expected:
            raise ValueError('Smoke recovered incompatible registered coverage')
        if 'smoke_gate.json' in store.keys('smoke_gate.json'):
            gate = json.loads(store.get('smoke_gate.json'))
            if gate.get('status') != 'passed' or gate.get('unit_key') != key:
                raise ValueError('Smoke gate identity differs')
            data = [store.get(gate[field]) for field in ('first_payload', 'repeated_payload')]
            if (any(gate[field] != 'smoke_'+sha(value)+'.npz'
                    for field, value in zip(('first_payload', 'repeated_payload'), data))
                    or key not in restored or data[0] != restored[key]['data']):
                raise ValueError('Restored smoke payload hashes or committed first unit differ')
            a, b = (validate(value, first) for value in data)
            if _check_repeat(a, b) != gate['repeat'] or key not in restored:
                raise ValueError('Restored smoke evidence differs')
            verify()
        else:
            a_data = restored[key]['data'] if key in restored else infer(first)
            a = validate(a_data, first)
            store.put('smoke_'+sha(a_data)+'.npz', a_data)
            b_data = infer(first)
            b = validate(b_data, first)
            store.put('smoke_'+sha(b_data)+'.npz', b_data)
            repeat = _check_repeat(a, b)
            verify()
            if key not in restored:
                commit_unit(store, first, a_data)
            gate = dict(status='passed', unit_key=key, repeat=repeat,
                first_payload='smoke_'+sha(a_data)+'.npz', repeated_payload='smoke_'+sha(b_data)+'.npz')
        fresh = load_units(reopen(), campaign)
        if not set(fresh) <= expected or key not in fresh:
            raise ValueError('Fresh mounted smoke recovery failed')
        for record in fresh.values():
            validate(record['data'], record['unit'])
        store.put('smoke_gate.json', canonical(gate))
    return gate


def write_reports(output: Path, analysis: dict, campaign: str, role: str) -> None:
    (output/'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False)+'\n')
    (output/'notes.md').write_text(
        'Exploratory reused development subset: 128 original IDs, all15 corruptions, severity5, seed0. '
        'Frozen author XYZ diffusion and Point-MAE; no LION. No confirmation or automatic winner promotion. '
        'CSV records canonical four-candidate PM and terminal; full clean-probe entropy diagnostics are in analysis.json. '
        'Timing includes four classifier calls for nstep and nine for entropy; not terminal-only latency. '
        'N changes discretization, cumulative per-step correction, and actual floor-snapshot times. '
        'Dense all9 PM changes time weighting and is descriptive. Entropy association is not calibration. '
        'DriveFS fsync/rename/readback does not independently prove remote synchronization. Role: '+role+'.\n')
    summary_fields = ['run_id', 'dataset', 'severity', 'method', 'seed', 'n_corruptions', 'macro_accuracy',
        'total_examples', 'total_correct', 'micro_accuracy', 'total_runtime_seconds', 'status']
    corruption_fields = ['run_id', 'dataset', 'severity', 'method', 'seed', 'corruption', 'n_examples',
        'n_correct', 'accuracy', 'runtime_seconds', 'peak_gpu_memory_mb', 'status']
    with (output/'summary.csv').open('w', newline='') as summary_file, (output/'per_corruption.csv').open('w', newline='') as corruption_file:
        summary = csv.DictWriter(summary_file, fieldnames=summary_fields)
        corruption = csv.DictWriter(corruption_file, fieldnames=corruption_fields)
        summary.writeheader(); corruption.writeheader()
        if analysis['status'] != 'complete':
            return
        for group in analysis['configurations']:
            for readout, suffix in (('old4_pm', 'pm'), ('terminal', 'terminal')):
                values = group['readouts'][readout]
                identity = dict(run_id=campaign, dataset='modelnet40_c', severity=5,
                    method=group['config_id']+'_'+suffix, seed=0, status='complete')
                summary.writerow(dict(identity, n_corruptions=len(values['per_corruption']),
                    macro_accuracy=values['macro_accuracy'], total_examples=values['count'],
                    total_correct=values['n_correct'], micro_accuracy=values['accuracy'],
                    total_runtime_seconds=group['runtime_seconds']))
                for name, row in values['per_corruption'].items():
                    corruption.writerow(dict(identity, corruption=name, n_examples=row['count'],
                        n_correct=row['n_correct'], accuracy=row['accuracy'],
                        runtime_seconds=group['corruption_runtime_seconds'][name],
                        peak_gpu_memory_mb=group['peak_gpu_memory_mb']))


def main(argv=None) -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--register', type=Path)
    parser.add_argument('--role', choices=('nstep', 'entropy'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prior-archive', type=Path)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--drive-folder', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--setup-receipt', type=Path)
    parser.add_argument('--official-root', default='/content/CloudFixer')
    parser.add_argument('--dataset-root', default='/content/3DD-TTA/data/modelnet40_c')
    parser.add_argument('--classifier', default='/content/CloudFixer/checkpoints/pointMAE_modelnet40.pth')
    parser.add_argument('--diffusion', default='/content/CloudFixer/checkpoints/diffusion_model_transformer_modelnet40.npy')
    args = parser.parse_args(argv)
    if args.register:
        args.register.write_bytes(canonical(registration())); return
    if not args.manifest or not args.role:
        parser.error('--manifest and --role are required')
    manifest = json.loads(args.manifest.read_text())
    reg = manifest['registration']
    required = {'scripts/run_cloudfixer_exploration.py', 'scripts/analyze_cloudfixer_exploration.py'}
    if reg != registration() or not required <= set(manifest['overlay_files']):
        raise ValueError('Exploration registration or source manifest differs')
    identity = dict(manifest=manifest, role=args.role)
    campaign = sha(canonical(identity))
    wanted = units(campaign, reg, args.role)
    if args.preview:
        print(json.dumps(dict(campaign=campaign, role=args.role, n_units=len(wanted), registration=reg), indent=2)); return
    if not args.output or not args.drive_folder or not args.prior_archive:
        parser.error('Execution/report requires --output, --drive-folder and --prior-archive')
    if not args.report_only and not args.setup_receipt:
        parser.error('Execution requires --setup-receipt with canonical owned-VM attempts')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    folder_root = args.drive_folder.resolve()
    if (not Path('/content/drive').is_mount()
            or Path('/content/drive/MyDrive/thesis/result').resolve() not in folder_root.parents):
        raise ValueError('Expected mounted role folder under MyDrive/thesis/result')
    budget_store = MountStore(folder_root)
    budget_store.put('task.json', canonical(dict(manifest=manifest, role=args.role)))
    if not args.report_only:
        record_setup(budget_store, json.loads(args.setup_receipt.read_text()), cap=reg['budget_seconds'])
    budget = Budget(budget_store, cap=reg['budget_seconds'], started=started)
    folder = folder_root/(args.role+'_'+campaign[:12])
    store = MountStore(folder)
    store.put('campaign.json', canonical(identity))
    source_sha = sha(canonical(manifest))
    (args.output/'config.json').write_bytes(canonical(dict(identity, campaign=campaign, execution_status='running')))
    (args.output/'command.txt').write_text(' '.join(sys.argv)+'\n')
    try:
        with budget.operation(1800):
            prior = PriorArchive(args.prior_archive.resolve(), manifest)
            labels_path = Path(args.dataset_root)/'label.npy'
            if transfer.file_hash(labels_path) != manifest['dataset_files']['label.npy']:
                raise ValueError('Pinned original label array mismatch')
            labels = np.load(labels_path, allow_pickle=False).reshape(-1)
            def validate(data, unit):
                payload = validate_payload(data, unit, labels, prior=prior, source_sha256=source_sha)
                environment_bytes = store.get('environment_'+str(payload['environment_sha256'])+'.json')
                validate_provenance(payload, manifest, environment_bytes)
                return payload
        if not args.report_only:
            with budget.operation(1800):
                raw_infer, verify, environment = transfer.runtime(args, manifest, trace_all=args.role == 'entropy')
                environment_bytes = canonical(environment)
                environment_sha = sha(environment_bytes)
                store.put('environment_'+environment_sha+'.json', environment_bytes)
                (args.output/'environment.txt').write_bytes(environment_bytes)
            def source_infer(unit):
                value = raw_infer(unit)
                (args.output/('pending_'+unit_key(unit)+'.npz')).unlink()
                return value
            source_gate = source_repeat_gate(store, next(u for u in wanted if u['config']['steps'] == 8),
                source_infer, verify, budget=budget, source_sha256=source_sha,
                environment_bytes=environment_bytes, labels=labels)
            (args.output/'source_repeat_gate.json').write_bytes(canonical(source_gate))
            for field in ('first_payload', 'repeated_payload'):
                (args.output/source_gate[field]).write_bytes(store.get(source_gate[field]))
            def infer(unit):
                values = unpack(raw_infer(unit))
                values.update(probability_stats(values['logits']))
                parity = prior.check(values, unit) if unit['config']['steps'] == 8 else {}
                values.update(source_sha256=source_sha, environment_sha256=environment_sha, parity=json.dumps(parity, sort_keys=True))
                data = pack(values)
                validate(data, unit)
                (args.output/('pending_'+unit_key(unit)+'.npz')).write_bytes(data)
                return data
            gate = smoke_gate(store, lambda: MountStore(folder), campaign, wanted, infer, verify,
                budget=budget, validate=validate)
            (args.output/'smoke_gate.json').write_bytes(canonical(gate))
            for field in ('first_payload', 'repeated_payload'):
                (args.output/gate[field]).write_bytes(store.get(gate[field]))
            def progress(value):
                print('CF_EXPLORATION_PROGRESS', json.dumps(value), flush=True)
                (args.output/'progress.json').write_bytes(canonical(value))
            if not args.smoke_only:
                execute_pending(store, campaign, wanted, infer, cap=reg['budget_seconds'], budget=budget,
                    validate=validate, progress=progress)
        with budget.operation(300):
            restored = load_units(MountStore(folder), campaign)
            for gate_name in ('smoke_gate.json', 'source_repeat_gate.json'):
                if gate_name in store.keys(gate_name):
                    gate_bytes = store.get(gate_name)
                    gate = json.loads(gate_bytes)
                    (args.output/gate_name).write_bytes(gate_bytes)
                    for field in ('first_payload', 'repeated_payload'):
                        value = store.get(gate[field])
                        prefix = 'source_smoke_' if gate_name == 'source_repeat_gate.json' else 'smoke_'
                        if gate[field] != prefix+sha(value)+'.npz':
                            raise ValueError('Archived smoke payload hash mismatch')
                        (args.output/gate[field]).write_bytes(value)
            expected = {unit_key(u): u for u in wanted}
            if not set(restored) <= set(expected):
                raise ValueError('Report coverage differs from registration')
            for key, record in restored.items():
                payload = validate(record['data'], expected[key])
                environment_name = 'environment_'+str(payload['environment_sha256'])+'.json'
                (args.output/environment_name).write_bytes(store.get(environment_name))
                # Include restored payloads in the attempt archive as well as newly run units.
                (args.output/('pending_'+key+'.npz')).write_bytes(record['data'])
            if args.report_only:
                environments = {p.name: json.loads(p.read_bytes()) for p in args.output.glob('environment_*.json')}
                (args.output/'environment.txt').write_bytes(canonical(dict(restored_inference_environments=environments)))
            if not args.report_only:
                verify()
            complete = len(restored) == len(wanted)
            coverage = dict(completed_units=len(restored), expected_units=len(wanted),
                status='complete' if complete else 'partial', parity_units=sum(r['unit']['config']['steps'] == 8 for r in restored.values()))
            if complete:
                from scripts.analyze_cloudfixer_exploration import analyze
                analysis = analyze(list(restored.values()), wanted)
            else:
                analysis = dict(coverage, configurations=[], claim='Incomplete coverage; no aggregate or ranking')
            write_reports(args.output, analysis, campaign, args.role)
            (args.output/'coverage.json').write_bytes(canonical(coverage))
            (args.output/'config.json').write_bytes(canonical(dict(identity, campaign=campaign, execution_status=coverage['status'])))
            for name in ('analysis.json', 'coverage.json', 'config.json', 'command.txt', 'notes.md', 'summary.csv', 'per_corruption.csv'):
                store.put('report_'+args.output.name+'_'+name, (args.output/name).read_bytes())
        print('CF_EXPLORATION_DONE', args.role, coverage['status'], flush=True)
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
