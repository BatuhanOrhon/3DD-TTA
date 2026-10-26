"""Independent CPU validation of the registered sensitivity calibration archive."""
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

PLAN_SHA = '4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
ROOT = PurePosixPath('/content/studies/guidance-components')
DRIVE = PurePosixPath('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
KINDS = ('legacy_l2', 'huber', 'power')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def verify_bytes(raw, expected, name):
    require(digest(raw) == {key: expected[key] for key in ('bytes', 'sha256')}, 'SHA/size mismatch: ' + str(name))


def validated_zip(archive):
    entries = archive.infolist(); names = [item.filename for item in entries]
    require(len(names) == len(set(names)) and len(names) <= 20000, 'Duplicate/excessive ZIP members')
    for item in entries:
        name = PurePosixPath(item.filename)
        require(name.parts and not name.is_absolute() and '..' not in name.parts and '\\' not in item.filename
                and ':' not in item.filename and name.as_posix() == item.filename.rstrip('/')
                and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR), 'Unsafe ZIP member')
    require(archive.testzip() is None, 'ZIP CRC failure')
    return {item.filename for item in entries if not item.is_dir()}


def load_helper(prep, plan):
    path = prep / 'calibrate_spectral_losses.py'
    verify_bytes(path.read_bytes(), plan['phase_sources'][path.name], path)
    spec = importlib.util.spec_from_file_location('verified_calibration_helper', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    require('torch' not in sys.modules, 'Local validator imported Torch')
    return helper


def validate_rows(rows, ids, corruptions, helper):
    expected = {(dataset, name, index) for dataset in DATASETS for name in corruptions for index in ids[dataset]}
    require(len(rows) == len(expected) == 2100 and {(row['dataset'], row['corruption'], row['index']) for row in rows} == expected,
            'Wrong original-ID/corruption coverage')
    for row in rows:
        horizon = 30 if row['corruption'] == 'background' else 10
        require(type(row['index']) is int and row['horizon'] == horizon and row['timestep'] == (290 if horizon == 30 else 90), 'Wrong first-step horizon/timestep')
        require(set(row['ratios']) == set(KINDS), 'Missing recorded ratios')
        for kind in KINDS:
            c, spectral = row['norms']['chamfer']['local'], row['norms'][kind]['local']
            ratio = c / spectral if min(c, spectral) > helper.TINY else None
            require(row['ratios'][kind] == ratio, 'Recorded ratio differs from actual norms')
    return helper.coefficient_report(rows)


def validate(prep, receipt_path, archive_path, output, old_prep):
    import numpy as np
    plan_raw = (prep / 'campaign_plan.json').read_bytes()
    require(digest(plan_raw)['sha256'] == PLAN_SHA, 'Unregistered campaign plan')
    plan = json.loads(plan_raw); helper = load_helper(prep, plan)
    receipt = json.loads(receipt_path.read_text()); archive_bytes = archive_path.read_bytes()
    verify_bytes(archive_bytes, receipt['archive'], archive_path)
    require(PurePosixPath(receipt['archive']['drive_path']).is_relative_to(DRIVE), 'Foreign Drive archive')
    assets = prep / 'assets_v2'
    manifest_raw = (assets / 'manifest.json').read_bytes()
    require(digest(manifest_raw)['sha256'] == plan['manifest_sha256'], 'Asset manifest differs from approved plan')
    manifest = json.loads(manifest_raw)
    old = json.loads((old_prep / 'development_phase_receipt.json').read_text())
    require({key: old['archive'][key] for key in helper.PHASE} == helper.PHASE, 'Unrecognized accepted development phase')
    old_zip = old_prep / 'development_phase_evidence_drive_roundtrip.zip'
    verify_bytes(old_zip.read_bytes(), helper.PHASE, old_zip)
    with zipfile.ZipFile(old_zip) as archive:
        require(json.loads(archive.read('state.json')) == old['state'], 'Old cache receipt is not phase-bound')
    expected_cache, expected_traces = {}, {}
    for job in old['state']['jobs']:
        for cached in job['cache']:
            key = cached['local_path']
            require(key not in expected_cache or expected_cache[key] == cached, 'Old cache identity conflict')
            expected_cache[key] = cached
        for name, trace in job['traces'].items():
            value = dict(initial_hashes=trace['initial_hashes'], paired_state=trace['paired_state'])
            key = job['dataset'], name
            require(key not in expected_traces or expected_traces[key] == value, 'Old preparation mismatch')
            expected_traces[key] = value
    require(len(expected_cache) == len(expected_traces) == 30, 'Wrong old cache scope')
    corruptions = list(old['state']['jobs'][0]['traces'])
    ids = {}
    for dataset in DATASETS:
        path = 'study/' + dataset + '_development_ids.json'
        raw = (assets / path).read_bytes(); verify_bytes(raw, manifest['files'][path], path)
        selection = json.loads(raw)
        require(selection['dataset'] == dataset and selection['stage'] == 'development' and len(selection['indices']) == 70,
                'Unexpected calibration selection')
        ids[dataset] = selection['indices']
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = validated_zip(archive)
        sealed = json.loads(archive.read('evidence_manifest.json'))
        require(set(sealed) == names - {'evidence_manifest.json'}, 'Evidence manifest coverage mismatch')
        for name, expected in sealed.items():
            verify_bytes(archive.read(name), expected, name)
        require(archive.read('campaign_plan.json') == plan_raw, 'Archived plan differs from approved local plan')
        for name, expected in plan['phase_sources'].items():
            verify_bytes(archive.read(name), expected, name)
        verify_bytes(archive.read('calibration/calibrate_spectral_losses.py'), plan['phase_sources']['calibrate_spectral_losses.py'], 'executed calibration source')
        nested = archive.read('calibration/source_snapshot.zip')
        require(digest(nested)['sha256'] == plan['source_snapshot_sha256'], 'Executed source snapshot changed')
        with zipfile.ZipFile(io.BytesIO(nested)) as source_zip:
            validated_zip(source_zip)
        state = json.loads(archive.read('state.json'))
        require(state == receipt['state'] and state['status'] == 'complete' and state['stage'] == 'calibrate'
                and state['phase_plan_sha256'] == PLAN_SHA and state['session'] == plan['session'] and not state['jobs']
                and state['full_benchmark'] is False, 'Calibration controller failed or changed scope')
        approval = json.loads(archive.read('gpu_approval.json'))
        require(approval['status'] == 'approved' and approval['phase_plan_sha256'] == PLAN_SHA and approval['cap_seconds'] == 19800
                and approval['session'] == plan['session'], 'Missing registered comprehensive GPU approval')
        ledger = json.loads(archive.read('ledger_snapshot.json'))
        require(ledger['phase_plan_sha256'] == PLAN_SHA and ledger['cap_seconds'] == 19800 and ledger['active'] is None
                and 0 <= ledger['spent_seconds'] <= 19800 and ledger['attempts'][-1]['stage'] == 'calibrate'
                and ledger['attempts'][-1]['status'] == 'complete', 'Campaign ledger is unresolved/over budget')
        report_bytes = archive.read('calibration/calibration_report.json')
        report = json.loads(report_bytes)
        require(state['calibration'] == report, 'Controller/scalar report mismatch')
        require(report['status'] == 'verified_label_free_gpu' and report['manifest_sha256'] == plan['manifest_sha256']
                and report['source_snapshot_sha256'] == plan['source_snapshot_sha256'] and report['session'] == plan['session']
                and type(report['seed']) is int and report['seed'] == 0 and report['labels_used'] is False
                and report['classifier_forwards'] == 0 and report['completed_batches'] == 30 and report['n_input_events'] == 2100
                and report['legacy_inline_parity'] is True and report['lion_state_before'] == report['lion_state_after'],
                'Calibration failed label-free/frozen-state/scope checks')
        final = json.loads(archive.read('calibration/state.json'))
        require({k:v for k,v in final.items() if k != 'elapsed_seconds'} == {k:v for k,v in report.items() if k != 'elapsed_seconds'},
                'Final calibration state contradicts scalar report')
        require(report['tiny_norm_threshold'] == 1e-12 and report['minimum_valid_fraction'] == .9
                and report['scalar_bounds'] == [1e-12, 1e12], 'Calibration numerical rules changed')
        native_audit = json.loads((assets / manifest['guidance']['native_audit']).read_text())
        bindings = {name: {k: value[0][k] for k in ('path','bytes','sha256')} for name,value in native_audit['backends'].items()}
        require(report['native']['no_build'] is True and report['native']['bindings'] == bindings, 'Native runtime is not pinned')
        for name, expected in report['executed_source_files'].items():
            require(name.startswith('assets/') and expected == manifest['files'].get(name[len('assets/'):]), 'Unsealed executed source')
        raw_norms = archive.read('calibration/per_object_norms.jsonl'); verify_bytes(raw_norms, report['per_object_norms'], 'per-object norms')
        rows = [json.loads(line) for line in raw_norms.splitlines()]
    diagnostics = validate_rows(rows, ids, corruptions, helper)
    require(report['coefficient_diagnostics'] == diagnostics and report['coefficients'] == {kind: diagnostics[kind]['coefficient'] for kind in KINDS},
            'Global coefficients/coverage differ from independent recomputation')
    caches = report['cache_records']; parity = report['per_batch_parity']
    require(len(caches) == len(parity) == 30 and len({row['local_path'] for row in caches}) == 30
            and {(row['dataset'], row['corruption']) for row in parity} == set(expected_traces), 'Wrong cache/parity coverage')
    parity_by_key = {(row['dataset'],row['corruption']):row for row in parity}
    maxima = {block: dict(helper=0., repeat=0.) for block in ('local','style')}
    for cached in caches:
        path = cached['local_path']; require(path in expected_cache, 'Unregistered cached state')
        expected = expected_cache[path]
        require(all(cached[key] == expected[key] for key in expected), 'Cache physical receipt changed')
        relative = PurePosixPath(path).relative_to(ROOT / 'paired_state_v1_seed0')
        require(len(relative.parts) == 3 and '..' not in relative.parts, 'Unsafe cache reference')
        dataset,name,_ = relative.parts
        physical = old_prep / 'paired_cache_extracted/paired_state_v1_seed0' / Path(*relative.parts)
        verify_bytes(physical.read_bytes(), cached, physical)
        with np.load(physical, allow_pickle=False) as cache:
            metadata = json.loads(str(cache['metadata'].item()))
            require(metadata['schema_version'] == 1 and metadata['cuda_devices'] == 1, 'Cache metadata changed')
            for field in ('input_sha256','rng_before_sha256','rng_after_sha256','tensor_hashes'):
                require(cached[field] == metadata[field], 'Calibration/cache metadata mismatch: ' + field)
            initial = expected_traces[(dataset,name)]['initial_hashes']
            require(metadata['input_sha256'] == initial['input_points'], 'Accepted input tensor mismatch')
            for field in ('shape_latent','local_latent','noise'):
                array = cache[field]
                require(array.dtype == np.float32 and array.shape[0] == 70 and array.ndim == 4 and np.isfinite(array).all(), 'Invalid cached tensor')
                sha = hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()
                require(sha == metadata['tensor_hashes'][field] == initial[field], 'Physical latent/noise tensor hash mismatch')
        value = parity_by_key[(dataset,name)]
        require(value['horizon'] == (30 if name == 'background' else 10) and value['scalar_exact'] is True
                and value['direct_coefficient_gradient_exact'] is True and value['absolute_relative_l2_cap'] == 1e-3
                and value['minimum_relative_l2_tolerance'] == 1e-5 and value['null_floor_multiplier'] == 3
                and value['rng_before_sha256'] == value['rng_after_sha256'] == metadata['rng_after_sha256'], 'First-step parity/RNG contract failed')
        require(set(value['native_gradient_differences']) == {'local','style'}, 'Missing native parity block')
        for block, metrics in value['native_gradient_differences'].items():
            difference, floor = metrics['helper_vs_inline_relative_l2'], metrics['inline_repeat_relative_l2']
            helper.validate_native_parity(difference, floor)
            require(metrics['allowed_relative_l2'] == min(1e-3,max(1e-5,3*floor)), 'Parity tolerance was enlarged')
            maxima[block]['helper'] = max(maxima[block]['helper'],difference); maxima[block]['repeat'] = max(maxima[block]['repeat'],floor)
    result = dict(status='accepted_local_label_free_calibration', phase_plan_sha256=PLAN_SHA,
        archive=receipt['archive'], receipt_identity=digest(receipt_path.read_bytes()), calibration_report_identity=digest(report_bytes),
        coefficients=report['coefficients'], coefficient_diagnostics=diagnostics, n_input_events=2100, physical_cache_npz=30,
        parity_batches=30, scalar_and_direct_gradient_exact=True, native_relative_l2_maxima=maxima,
        frozen_state_unchanged=True, labels_used=False, classifier_forwards=0, campaign_spent_snapshot_seconds=ledger['spent_seconds'],
        limits='No GPU work repeated. Input tensor identity joins accepted preparation metadata; latent/noise NPZ bytes are rehashed. Native prior gradients satisfy the preregistered repeatability-floor rule, not bitwise identity.')
    require(not output.exists(), 'Local validation output already exists; inspect instead of overwriting')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2))
    return result


def self_check(helper):
    names = ['background'] + ['c%d' % value for value in range(14)]
    ids = {dataset: list(range(70)) for dataset in DATASETS}
    rows = [dict(dataset=dataset, corruption=name, index=index, horizon=30 if name == 'background' else 10,
                 timestep=290 if name == 'background' else 90,
                 norms={kind: dict(local=8. if kind == 'chamfer' else 2., style=1.) for kind in ('chamfer', *KINDS)},
                 ratios={kind: 4. for kind in KINDS}) for dataset in DATASETS for name in names for index in range(70)]
    validate_rows(rows, ids, names, helper)
    rows[0]['ratios']['power'] = 5.
    try:
        validate_rows(rows, ids, names, helper)
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect recorded ratio was accepted')
    print('LOCAL_CALIBRATION_VALIDATOR_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prep', type=Path, required=True)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--old-prep', type=Path)
    args = parser.parse_args()
    plan_raw = (args.prep / 'campaign_plan.json').read_bytes()
    require(digest(plan_raw)['sha256'] == PLAN_SHA, 'Unregistered campaign plan')
    if args.self_check:
        self_check(load_helper(args.prep, json.loads(plan_raw)))
    else:
        require(all(value is not None for value in (args.receipt,args.archive,args.output,args.old_prep)), 'Require receipt/archive/output/old-prep paths')
        validate(args.prep,args.receipt,args.archive,args.output,args.old_prep)
