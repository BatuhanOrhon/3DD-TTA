#!/usr/bin/env python3
"""CPU acceptance and posthoc geometry/readout analysis of frozen shared probes."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_scan_trajectory import Bundle
from scripts import eval_multibackbone_source as source
from scripts import scanobjectnn_severity5_colab as archive_checks


def helper(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = helper('shared_probe_accepted_oracle', 'knowledge/evidence/decoded_physical_followup_20261104/analyze_oracle.py')
IDS = [459, 545, 1229, 1435]
CORRUPTIONS = ['gaussian', 'background', 'shear', 'lidar']
ARMS = ['zero', 'C', 'S_lat', 'S_phys', 'P100_C', 'CS_lat', 'CS_phys']
FRACTIONS = [.5, 1.]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def checked_extract(path, expected_sha, output, manifest_name):
    observed = source.file_identity(path)
    if observed['sha256'] != expected_sha:
        raise ValueError('Pinned ZIP SHA differs: '+str(path))
    with zipfile.ZipFile(path) as archive:
        archive_checks.checked_members(archive)
        entries = archive.infolist()
        if len(entries) > 20000 or sum(e.file_size for e in entries) > 4*1024**3:
            raise ValueError('Excessive archive size/member count')
        if any(stat.S_IFMT(e.external_attr >> 16) not in (0, stat.S_IFREG, stat.S_IFDIR)
               or e.file_size > 2*1024**3 or PurePosixPath(e.filename).as_posix() != e.filename.rstrip('/') for e in entries):
            raise ValueError('Unsafe archive type/path/size')
        output.mkdir(parents=True, exist_ok=False)
        for entry in entries:
            target = output/entry.filename
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as incoming, target.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    if manifest_name == 'bundle_manifest.json':
        config_paths = list(output.rglob('config.json'))
        roots = [p.parent for p in config_paths if (p.parent/manifest_name).is_file()]
        if len(roots) != 1:
            raise ValueError('Require one sealed capture root')
        root = roots[0]
        bundle = Bundle(root)
        bundle.close()
    else:
        root = output
        manifest = json.loads(source.asset_path(root, manifest_name).read_text())
        files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if set(manifest) != files-{manifest_name}:
            raise ValueError('Outer evidence manifest coverage differs')
        for name, identity in manifest.items():
            source.verify_identity(source.asset_path(root, name), identity)
    return root, observed


def cluster_summary(rows, field):
    grouped = {}
    for row in rows:
        if row.get(field) is not None:
            grouped.setdefault(row['object_id'], []).append(float(row[field]))
    per_object = [dict(object_id=key, n_dependent_rows=len(values), mean=float(np.mean(values)))
                  for key, values in sorted(grouped.items())]
    return dict(n_original_objects=len(per_object), per_object=per_object,
                equal_object_mean=float(np.mean([r['mean'] for r in per_object])) if per_object else None,
                positive_objects=sum(r['mean'] > 0 for r in per_object),
                negative_objects=sum(r['mean'] < 0 for r in per_object),
                inference='Descriptive means across four exposed object clusters; no independent-row or population claim')


def summarize_raw(mechanism, capture, report, config):
    artifacts = report['artifacts']
    for name, expected in artifacts.items():
        source.verify_identity(source.asset_path(mechanism, name), expected)
    if report['capture_config'] != source.file_identity(capture/'config.json'):
        raise ValueError('Captured config differs from mechanism input')
    if (config['selected_original_ids'] != IDS or config['corruptions'] != CORRUPTIONS or config['count'] != 4
            or config['capture_shared_states'] is not True or not config['diagnostic_complete']
            or config['lion_state_before'] != config['lion_state_after']
            or config['classifier_state_before_sha256'] != config['classifier_state_after_sha256']):
        raise ValueError('Capture protocol/model state differs')
    for name, expected in config['executed_source_files'].items():
        if name.startswith('repo/'):
            source.verify_identity(source.asset_path(capture, 'source_snapshot/'+name[5:]), expected)
    cases = {(row['corruption'], row['batch']): row for row in report['cases']}
    if len(cases) != 8 or set(cases) != {(c, b) for c in CORRUPTIONS for b in range(2)}:
        raise ValueError('Mechanism case coverage differs')
    states, coverage, jobs, gradients = [], [], [], []
    for corruption in CORRUPTIONS:
        for batch in range(2):
            prefix = '%s/batch%04d' % (corruption, batch)
            ids = IDS[batch*2:batch*2+2]
            raw_case = capture/'batches'/prefix
            shared = json.loads(source.asset_path(raw_case, 'shared_states.json').read_text())
            diagnostics = json.loads(source.asset_path(raw_case, 'diagnostics.json').read_text())
            source.verify_identity(raw_case/'shared_states.json', diagnostics['shared_states'])
            if not shared['complete'] or shared['original_ids'] != ids:
                raise ValueError('Raw shared snapshot IDs/completion differ')
            snapshots = {(r['loop_count'], r['stage']): r for r in shared['snapshots']}
            loops = [9, 27] if corruption == 'background' else [2, 6]
            if len(shared['snapshots']) != 7 or set(snapshots) != {(0, 'frame')} | {(s, t) for s in loops for t in ('pre_prior', 'pre_decoder', 'decoded')}:
                raise ValueError('Raw snapshot stage coverage differs')
            for snapshot in snapshots.values():
                source.verify_identity(source.asset_path(raw_case, snapshot['file']), snapshot['identity'])
            frame = arrays(raw_case/snapshots[(0, 'frame')]['file'])
            if frame['index'].tolist() != ids:
                raise ValueError('Captured frame IDs differ')
            for name in ('normalization_center', 'normalization_scale'):
                if oracle_tensor_hash(frame[name]) != snapshots[(0, 'frame')]['tensor_sha256'][name]:
                    raise ValueError('Captured frame tensor hash differs')
            input_name = prefix+'/input.json'
            if input_name in artifacts:
                inputs = json.loads((mechanism/input_name).read_text())
                if inputs['original_ids'] != ids:
                    raise ValueError('Mechanism input original IDs differ')
                for name, expected in inputs['identities'].items():
                    source.verify_identity(source.asset_path(raw_case, name), expected)
            for loop in loops:
                state_prefix = prefix+'/loop%02d' % loop
                state = dict(corruption=corruption, batch=batch, loop_count=loop, original_ids=ids)
                for name in ('state_complete', 'h2', 'h3', 'directions'):
                    relative = state_prefix+'/'+name+'.json'
                    state[name] = json.loads((mechanism/relative).read_text()) if relative in artifacts else None
                states.append(state)
                directions = state['directions']
                entries = directions.get('branches', []) if directions else []
                keys = [(r['name'], r['fraction']) for r in entries]
                if len(keys) != len(set(keys)) or any(a not in ARMS or f not in FRACTIONS for a, f in keys):
                    raise ValueError('Unexpected/duplicate branch keys')
                for arm in ARMS:
                    for fraction in FRACTIONS:
                        stem = state_prefix+'/branch_%s_%s' % (arm, str(fraction).replace('.', 'p'))
                        branch = next((r for r in entries if (r['name'], r['fraction']) == (arm, fraction)), None)
                        row = dict(corruption=corruption, batch=batch, loop_count=loop, arm=arm, fraction=fraction,
                                   original_ids=ids, key=stem, status='unavailable')
                        if branch and branch['status'] == 'complete':
                            if input_name not in artifacts or stem+'_state.npz' not in artifacts or stem+'_prediction.npz' not in artifacts:
                                raise ValueError('Complete branch lacks sealed state/cloud lineage')
                            cloud = arrays(mechanism/(stem+'_prediction.npz'))['decoded_xyz']
                            if cloud.shape != (2, 2048, 3) or cloud.dtype != np.float32 or not np.isfinite(cloud).all():
                                raise ValueError('Invalid branch physical XYZ')
                            row.update(status='available', cloud_file=stem+'_prediction.npz',
                                       cloud_sha256=artifacts[stem+'_prediction.npz'])
                            jobs.append(dict(row=row, frame=frame))
                        else:
                            row['reason'] = branch.get('error', branch['status']) if branch else directions.get('unavailable', {}).get(arm, directions.get('status')) if directions else 'No completed direction receipt'
                        coverage.append(row)
                gradient_paths = {kind: state_prefix+'/gradient_'+kind+'.npz' for kind in ('C', 'S_lat', 'S_phys', 'P100_C')}
                if gradient_paths['C'] in artifacts:
                    c = arrays(mechanism/gradient_paths['C'])
                    for kind in ('S_lat', 'S_phys', 'P100_C'):
                        if gradient_paths[kind] not in artifacts:
                            continue
                        other = arrays(mechanism/gradient_paths[kind])
                        for i, object_id in enumerate(ids):
                            left = np.concatenate([.03*c['local_gradient'][i].reshape(-1), .003*c['style_gradient'][i].reshape(-1)]).astype(np.float64)
                            right = np.concatenate([.03*other['local_gradient'][i].reshape(-1), .003*other['style_gradient'][i].reshape(-1)]).astype(np.float64)
                            denom = np.linalg.norm(left)*np.linalg.norm(right)
                            gradients.append(dict(corruption=corruption, loop_count=loop, object_id=object_id, kind=kind,
                                cosine_vs_C=float(np.dot(left, right)/denom) if denom else None,
                                C_joint_norm=float(np.linalg.norm(left)), other_joint_norm=float(np.linalg.norm(right)),
                                technically_available=bool(directions and kind not in directions.get('unavailable', {})),
                                scope='Raw recorded gradients; failed FD/native evidence does not become an applied direction'))
    if len(states) != 16 or len(coverage) != 224:
        raise ValueError('Frozen expected state/branch coverage differs')
    summary = dict(expected_states=16, recorded_state_receipts=sum(s['state_complete'] is not None for s in states),
        state_statuses=dict(Counter(s['state_complete']['status'] if s['state_complete'] else 'missing' for s in states)),
        h3_statuses=dict(Counter(s['h3']['status'] if s['h3'] else 'missing' for s in states)),
        directions_statuses=dict(Counter(s['directions']['status'] if s['directions'] else 'missing' for s in states)),
        expected_branch_clouds=224, available_branch_clouds=len(jobs),
        unavailable_by_arm=dict(Counter(r['arm'] for r in coverage if r['status'] != 'available')),
        h2_objects=[dict(corruption=s['corruption'], loop_count=s['loop_count'], object_id=i, **row)
            for s in states if s['h2'] for i, row in zip(s['original_ids'], s['h2']['objects'])],
        h3_perturbations=[dict(corruption=s['corruption'], loop_count=s['loop_count'], **r)
            for s in states if s['h3'] for r in s['h3'].get('perturbations', [])],
        forward_and_fd=[dict(corruption=s['corruption'], loop_count=s['loop_count'], original_ids=s['original_ids'],
            objectives=s['directions'].get('objectives'), unavailable=s['directions'].get('unavailable'),
            prior_forward=s['directions'].get('live_prior_forward_gate'), decoder_forward=s['directions'].get('live_decoder_forward_gate'))
            for s in states if s['directions']],
        interpretation='Descriptive fixed-state diagnostics. Failed native/FD gates remain unavailable; no applied accuracy or speed claim')
    return summary, states, coverage, jobs, gradients


def oracle_tensor_hash(value):
    value = np.asarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes(order='C')).hexdigest()


def analyze_clean(mechanism, jobs, clean_subset, clean_receipt):
    clean, proof = oracle.load_oracle(clean_subset, clean_receipt)
    rows = []
    for job in jobs:
        metadata, frame = job['row'], job['frame']
        cloud = arrays(mechanism/metadata['cloud_file'])['decoded_xyz']
        for i, object_id in enumerate(metadata['original_ids']):
            target = oracle.physical_clean(clean[oracle.IDS.index(object_id)], frame['normalization_center'][i], frame['normalization_scale'][i])
            distance = oracle.squared_chamfer(cloud[i], target)
            permutation = abs(distance-oracle.squared_chamfer(cloud[i, ::-1], target))
            if permutation > 1e-12*(1+distance):
                raise ValueError('Clean set distance changed under point permutation')
            rows.append(dict(corruption=metadata['corruption'], loop_count=metadata['loop_count'], arm=metadata['arm'],
                fraction=metadata['fraction'], object_id=object_id, clean_cd=distance, permutation_error=permutation))
    lookup = {(r['corruption'], r['loop_count'], r['fraction'], r['arm'], r['object_id']): r for r in rows}
    for row in rows:
        key = (row['corruption'], row['loop_count'], row['fraction'])
        zero, c = (lookup.get((*key, arm, row['object_id'])) for arm in ('zero', 'C'))
        row['delta_vs_zero'] = row['clean_cd']-zero['clean_cd'] if zero else None
        row['delta_vs_C'] = row['clean_cd']-c['clean_cd'] if c else None
        z1, z2 = (lookup.get((row['corruption'], row['loop_count'], f, 'zero', row['object_id'])) for f in FRACTIONS)
        row['observed_zero_repeat_cd_delta'] = abs(z1['clean_cd']-z2['clean_cd']) if z1 and z2 else None
        row['exceeds_observed_zero_repeat'] = (abs(row['delta_vs_zero']) > row['observed_zero_repeat_cd_delta']
            if row['delta_vs_zero'] is not None and row['observed_zero_repeat_cd_delta'] is not None else None)
    groups = {arm+'_'+str(f): {field: cluster_summary([r for r in rows if r['arm'] == arm and r['fraction'] == f], field)
        for field in ('delta_vs_zero', 'delta_vs_C')} for arm in ARMS for f in FRACTIONS}
    return dict(rows=rows, summaries=groups, clean_receipt=proof,
        frame='Accepted oracle.physical_clean: corrupted center/scale, factor3.3885, rotate(-z,y,x); set NN distance, no slot-clean assumption',
        scope='Frozen actual one-next-sampler-prediction clouds; no direction/weight/scale selection or full trajectory claim')


def analyze_readouts(readouts, mechanism_report_identity, capture, coverage):
    report = json.loads((readouts/'complete.json').read_text())
    if (report['inputs']['mechanism_report'] != mechanism_report_identity
            or report['inputs']['capture_config'] != source.file_identity(capture/'config.json')):
        raise ValueError('Readouts used another mechanism report')
    if report['status'] not in ('complete', 'complete_with_unavailable') or any(
            not report[key] for key in ('model_unchanged', 'sources_unchanged', 'native_files_unchanged', 'rng_unchanged') if key in report):
        raise ValueError('Readout final state/source/native/RNG boundary failed')
    for name, expected in report['artifacts'].items():
        source.verify_identity(source.asset_path(readouts, name), expected)
    labels = {}
    for corruption in CORRUPTIONS:
        for batch in range(2):
            payload = arrays(capture/'batches'/corruption/('batch%04d' % batch)/'predictions.npz')
            for object_id, label in zip(payload['index'], payload['label']):
                if int(object_id) in labels and labels[int(object_id)] != int(label):
                    raise ValueError('Captured official labels disagree across corruption rows')
                labels[int(object_id)] = int(label)
    rows = []
    available = {row['key']: row for row in coverage if row['status'] == 'available'}
    seen = set()
    for receipt in report['readouts']:
        if receipt['status'] != 'complete':
            continue
        payload = arrays(readouts/receipt['readout_file'])
        key = receipt['key']
        if key in seen or key not in available or receipt['cloud_identity'] != available[key]['cloud_sha256']:
            raise ValueError('Readout cloud lineage/key differs from certified branch')
        seen.add(key)
        for field, value in [('corruption', receipt['corruption']), ('loop_count', receipt['loop_count']),
                             ('branch', receipt['arm']), ('fraction', receipt['fraction']), ('next_timestep', receipt['next_timestep'])]:
            if payload[field].shape != (2,) or not bool((payload[field] == value).all()):
                raise ValueError('Readout payload/receipt key differs: '+field)
        logits = payload['logits'].astype(np.float64)
        if logits.shape != (2, 40) or not np.isfinite(logits).all() or payload['index'].tolist() != receipt['original_ids']:
            raise ValueError('Readout logit/key shape differs')
        if not np.array_equal(logits.argmax(1), payload['prediction']):
            raise ValueError('Stored readout prediction differs from raw logits')
        for i, object_id in enumerate(payload['index']):
            label = labels[int(object_id)]
            other = logits[i].copy(); other[label] = -np.inf
            rows.append(dict(corruption=receipt['corruption'], loop_count=receipt['loop_count'], arm=receipt['arm'],
                fraction=receipt['fraction'], object_id=int(object_id), true_label=label,
                correct=int(payload['prediction'][i] == label), true_margin=float(logits[i, label]-other.max())))
    lookup = {(r['corruption'], r['loop_count'], r['fraction'], r['arm'], r['object_id']): r for r in rows}
    for row in rows:
        for arm in ('zero', 'C'):
            other = lookup.get((row['corruption'], row['loop_count'], row['fraction'], arm, row['object_id']))
            row['margin_delta_vs_'+arm] = row['true_margin']-other['true_margin'] if other else None
            row['correct_delta_vs_'+arm] = row['correct']-other['correct'] if other else None
            row['paired_'+arm+'_correct'] = other['correct'] if other else None
    summaries = {arm+'_'+str(f): dict(correct=sum(r['correct'] for r in rows if r['arm'] == arm and r['fraction'] == f),
        n=sum(r['arm'] == arm and r['fraction'] == f for r in rows),
        paired_C_correct=sum(r['paired_C_correct'] for r in rows if r['arm'] == arm and r['fraction'] == f and r['paired_C_correct'] is not None),
        paired_C_n=sum(r['arm'] == arm and r['fraction'] == f and r['paired_C_correct'] is not None for r in rows),
        clusters={field: cluster_summary([r for r in rows if r['arm'] == arm and r['fraction'] == f], field)
                  for field in ('true_margin', 'margin_delta_vs_zero', 'margin_delta_vs_C', 'correct_delta_vs_C')})
        for arm in ARMS for f in FRACTIONS}
    return dict(rows=rows, summaries=summaries, readout_status=report['status'],
        label_origin='Verified original shared capture predictions.npz official label field, opened only after cloud/readout freeze',
        scope='One next prediction correct/n and margins, dependent rows/four exposed object clusters; no PM/full-benchmark accuracy')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mechanism-zip', 'mechanism-sha256', 'shared-capture', 'shared-capture-sha256',
                 'clean-subset', 'clean-receipt', 'operator-receipt', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--readouts-zip'); parser.add_argument('--readouts-sha256')
    args = parser.parse_args(argv)
    if bool(args.readouts_zip) != bool(args.readouts_sha256):
        raise ValueError('Readout archive and SHA must be supplied together')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    try:
        mechanism, mechanism_id = checked_extract(Path(args.mechanism_zip), args.mechanism_sha256, output/'raw_mechanism', 'evidence_manifest.json')
        capture, capture_id = checked_extract(Path(args.shared_capture), args.shared_capture_sha256, output/'raw_capture', 'bundle_manifest.json')
        report, config = json.loads((mechanism/'complete.json').read_text()), json.loads((capture/'config.json').read_text())
        operator_receipt = json.loads(Path(args.operator_receipt).read_text())
        expected_operator, = [v for k, v in operator_receipt['files'].items() if k.endswith('probe_shared_states_validation_v2.py')]
        if any(report['executed_operator'].get(k) != expected_operator[k] for k in ('bytes', 'sha256')):
            raise ValueError('Actual mechanism operator differs from reviewed amendment')
        if report['status'] not in ('complete', 'complete_with_unavailable') or not report['model_unchanged'] or not report['entry_rng_restored']:
            raise ValueError('Mechanism phase failed its final model/RNG boundary')
        raw, states, coverage, jobs, gradients = summarize_raw(mechanism, capture, report, config)
        write(output/'raw_summary.json', raw); write(output/'raw_states.json', states)
        write(output/'branch_coverage.json', coverage); write(output/'gradient_conflicts.json', gradients)
        write(output/'inputs.json', dict(mechanism_archive=mechanism_id, capture_archive=capture_id,
            mechanism_report=source.file_identity(mechanism/'complete.json'), operator_receipt=source.file_identity(Path(args.operator_receipt)),
            analyzer=source.file_identity(Path(__file__)), helpers={name: source.file_identity(Path(module.__file__)) for name, module in [('oracle', oracle), ('archive', archive_checks)]}))
        # All raw coverage/gates are persisted before any clean or label lookup.
        clean_result = analyze_clean(mechanism, jobs, Path(args.clean_subset), Path(args.clean_receipt))
        write(output/'clean_geometry.json', clean_result)
        if args.readouts_zip:
            readouts, readouts_id = checked_extract(Path(args.readouts_zip), args.readouts_sha256, output/'raw_readouts', 'evidence_manifest.json')
            readout_result = analyze_readouts(readouts, source.file_identity(mechanism/'complete.json'), capture, coverage)
            readout_result['archive_identity'] = readouts_id
            write(output/'classification.json', readout_result)
        write(output/'complete.json', dict(status='complete', raw_summary=source.file_identity(output/'raw_summary.json'),
            clean_geometry=source.file_identity(output/'clean_geometry.json'), classification_available=bool(args.readouts_zip),
            artifacts={p.name: source.file_identity(p) for p in sorted(output.glob('*.json'))}))
    except Exception as error:
        write(output/'failure.json', dict(status='failed_analysis', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
