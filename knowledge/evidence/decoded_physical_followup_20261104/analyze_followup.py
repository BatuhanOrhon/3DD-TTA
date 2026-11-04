"""Strict CPU analysis of sealed P0/P1 diagnostic bundles; no model forward."""
import hashlib
import io
import itertools
import json
from pathlib import Path
import stat
import sys
import zipfile

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent/'scripts/analyze_scan_trajectory.py').is_file():
        sys.path.insert(0, str(parent))
        break
from scripts.analyze_scan_trajectory import Bundle


def arrays(bundle, name):
    with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def open_bundle(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        assert len(members) < 20000 and sum(m.file_size for m in members) < 3*1024**3
        assert all(stat.S_IFMT(m.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR)
                   and m.file_size < 256*1024**2 for m in members)
    bundle = Bundle(path)
    config = bundle.json('config.json')
    assert config['diagnostic_complete'] and config['status'] == 'partial'
    assert config['arm'] == 'diagnostic' and config['weight'] == 0
    assert config['lion_state_before'] == config['lion_state_after']
    assert config['classifier_state_before_sha256'] == config['classifier_state_after_sha256']
    assert all(name in bundle.names for name in
               ('command.txt', 'config.json', 'environment.txt', 'stdout.log',
                'summary.csv', 'per_corruption.csv', 'notes.md'))
    return bundle, config


def difference(left, right):
    assert left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    delta = left.astype(np.float64)-right.astype(np.float64)
    return dict(max_absolute=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))))


def summarize_p0(paths):
    assert len(paths) == 6
    groups, runs = {'legacy': [], 'controlled': []}, []
    for path in paths:
        bundle, config = open_bundle(path)
        try:
            assert config['selected_original_ids'] == [44, 91] and config['corruptions'] == ['gaussian']
            case = 'batches/gaussian/batch0000/'
            pred = arrays(bundle, case+'predictions.npz')
            reference = arrays(bundle, case+'reference.npz')
            first = arrays(bundle, case+'decoded_first_checkpoint.npz')['decoded_xyz']
            diag = bundle.json(case+'diagnostics.json')
            records = bundle.json(case+'probe_records.json')
            assert len(records) == 3 and pred['trajectory_logits'].shape == (2, 4, 40)
            identity = dict(source=config['executed_source_files'], native=config['native_extensions'],
                            bootstrap=config['native_bootstrap'], guidance=config['guidance_protocol'],
                            scheduler=diag['scheduler'], initial=diag['trajectory']['initial_hashes'],
                            initial_rng=diag['trajectory']['initial_rng_sha256'],
                            ids=pred['index'].tolist(), candidate_counts=pred['candidate_loop_counts'].tolist(),
                            lion=config['lion_state_before'], classifier=config['classifier_state_before_sha256'],
                            paired_sha256=diag['paired_state']['sha256'], prepared=diag['prepared_x_sha256'],
                            input_rng=diag['input_rng_before_sha256'], output_rng=diag['reconstruction_rng_after_sha256'])
            if runs:
                assert identity == runs[0]['identity'], 'Repeat control identity changed'
                assert all(np.array_equal(reference[k], runs[0]['reference'][k]) for k in reference)
            witness = [dict(loop_count=r['loop_count'],
                            full_repeat_max=r['full_repeat_decoder_max_absolute_difference'],
                            full_repeat_rms=r['full_repeat_decoder_rms_difference'],
                            detached_max=r['detached_decoder_max_absolute_difference'],
                            detached_rms=r['detached_decoder_rms_difference'],
                            full_repeat_requires_grad=r['full_repeat_decoder_output_requires_grad']) for r in records]
            assert all(r['full_repeat_requires_grad'] for r in witness)
            row = dict(path=str(path), archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                       backend=config['backend_policy'], runtime=config['runtime'], witness=witness,
                       identity=identity, reference=reference, first=first, logits=pred['trajectory_logits'])
            groups[row['backend']].append(row)
            runs.append(row)
        finally:
            bundle.close()
    results = {}
    for policy, group in groups.items():
        assert len(group) == 3
        pairs = []
        for a, b in itertools.combinations(group, 2):
            logit_rows = [dict(object_id=object_id, checkpoint=c,
                               **difference(a['logits'][i, c], b['logits'][i, c]))
                          for i, object_id in enumerate((44, 91)) for c in range(4)]
            decoded_rows = [dict(object_id=object_id, **difference(a['first'][i], b['first'][i]))
                            for i, object_id in enumerate((44, 91))]
            passed = (all(r['max_absolute'] <= .05 and r['rms'] <= .01 for r in logit_rows)
                      and all(r['rms'] <= 1e-4 for r in decoded_rows))
            pairs.append(dict(archives=[a['archive_sha256'], b['archive_sha256']],
                              logits=logit_rows, first_decoded=decoded_rows, passed=passed))
        witnesses = [r for run in group for r in run['witness']]
        results[policy] = dict(pairs=pairs, cross_process_passed=all(r['passed'] for r in pairs),
                              same_state_full_repeat_passed=all(r['full_repeat_rms'] <= 1e-5 and r['full_repeat_max'] <= 5e-4 for r in witnesses),
                              same_state_detached_passed=all(r['detached_rms'] <= 1e-5 and r['detached_max'] <= 5e-4 for r in witnesses),
                              runs=[{k:v for k,v in run.items() if k not in ('identity', 'reference', 'first', 'logits')} for run in group])
    return dict(stage='P0', policies=results, identical_controls_verified=True,
                thresholds=dict(logit_max=.05, logit_rms=.01, decoded_rms=1e-4,
                                same_state_max=5e-4, same_state_rms=1e-5),
                limitation='Backend protocol diagnostic; custom native atomics remain; no accuracy or speed attribution')


def summarize_p1(path):
    bundle, config = open_bundle(path)
    try:
        assert config['backend_policy'] == 'controlled' and config['count'] == 8
        assert config['selected_original_ids'] == [459, 545, 1229, 1435, 1441, 1557, 1881, 2095]
        selection = bundle.read('selection_manifest.json')
        assert hashlib.sha256(selection).hexdigest() == config['index_manifest_sha256']
        assert json.loads(selection) == config['selection_manifest']
        assert config['selection_manifest']['indices'] == config['selected_original_ids']
        assert set(config['corruptions']) == {'gaussian', 'background', 'shear', 'lidar'}
        result = {}
        for corruption in config['corruptions']:
            rows = []
            for batch in range(4):
                case = 'batches/%s/batch%04d/' % (corruption, batch)
                records = bundle.json(case+'probe_records.json')
                assert len(records) == 3
                assert case+'terminal_geometry.npz' in bundle.names
                for r in records:
                    cs, ratios = r['residual_freeze_update_cosine_per_object'], r['residual_freeze_over_full_update_norm_ratio_per_object']
                    assert len(cs) == len(ratios) == 2
                    for j, (cosine, ratio) in enumerate(zip(cs, ratios)):
                        resolved = cosine is not None and ratio is not None and np.isfinite(cosine) and np.isfinite(ratio)
                        rows.append(dict(object_id=config['selected_original_ids'][2*batch+j], loop_count=r['loop_count'],
                                         cosine=cosine, H8_over_full_norm_ratio=ratio,
                                         passed=bool(resolved and cosine >= .99 and .9 <= ratio <= 1.1)))
            assert len(rows) == 24
            available = [r for r in rows if r['cosine'] is not None and r['H8_over_full_norm_ratio'] is not None]
            result[corruption] = dict(rows=rows, passed=sum(r['passed'] for r in rows), total=len(rows),
                unresolved=len(rows)-len(available), min_cosine=min((r['cosine'] for r in available), default=None),
                min_ratio=min((r['H8_over_full_norm_ratio'] for r in available), default=None),
                max_ratio=max((r['H8_over_full_norm_ratio'] for r in available), default=None))
        assert sum(r['total'] for r in result.values()) == 96
        return dict(stage='P1', archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    indices=config['selected_original_ids'], corruptions=result,
                    passed=sum(r['passed'] for r in result.values()), total=96,
                    thresholds=dict(min_cosine=.99, H8_over_full_norm_ratio=[.9, 1.1]),
                    limitation='Zero-guidance same-state derivatives on stage-disjoint mechanism cohort; not unseen confirmation or applied accuracy')
    finally:
        bundle.close()


if __name__ == '__main__':
    mode, *paths = sys.argv[1:]
    result = summarize_p0(paths) if mode == 'P0' else summarize_p1(paths[0])
    print(json.dumps(result, indent=2, allow_nan=False))
