"""CPU fixtures for paired complete-shard aggregation; no remote inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_analysis', HERE/'analyze_full.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def rejects(action):
    try:
        action()
    except (AssertionError, ValueError, FileNotFoundError):
        return
    raise AssertionError('Invalid evidence was accepted')


with tempfile.TemporaryDirectory(prefix='full-analysis-check-') as temporary:
    root = Path(temporary)
    (root/'operator').mkdir()
    (root/'accepted').mkdir()
    plan = dict(n_examples=2468, asset_manifest_sha256='a'*64, frozen_alpha100=4433.612924010353,
        corruptions=list(analysis.CORRUPTIONS), prior_M100_ids=[0, 3, *range(100, 118)],
        transfer_gate=dict(minimum_delta_pp=1., paired_object_cluster_ci95_lower_strictly_above=0.), shards=[])
    (root/'common_identity.json').write_text(json.dumps(dict(asset_manifest_sha256='a'*64, native='same')))
    for number in range(2):
        shard_label = 'shard%03d' % number
        ids = [number*2, number*2+1]
        selection = dict(schema_version=1, dataset='modelnet40_c', indices=ids)
        (root/'operator'/(shard_label+'.json')).write_text(json.dumps(selection))
        plan['shards'].append(dict(number=number, label=shard_label, count=2, selection_file=shard_label+'.json',
                                  arms=['control', 'half0', 'M100']))
        for arm in ('cache', 'control', 'half0', 'M100'):
            label = shard_label+'_'+arm
            archive = root/(label+'.zip')
            cases, counts = [], {}
            with zipfile.ZipFile(archive, 'x') as zipped:
                for corruption in analysis.CORRUPTIONS:
                    truth = [i % 40 for i in ids]
                    predictions = truth if arm in ('cache', 'M100') else [(i+1) % 40 for i in truth]
                    cases.append(dict(case=corruption+'/0', ids=ids, labels=truth, pm=predictions,
                        terminal=truth, paired_identity={'prepared': 'same'},
                        filter_sha256=None if arm == 'control' else ['b'*64]*2,
                        rank=None if arm == 'control' else [100, 101]))
                    counts[corruption] = dict(n=2, pm_correct=2 if arm in ('cache', 'M100') else 0,
                        terminal_correct=2, reconstruction_seconds=1., peak_gpu_memory_mb=128.)
                    records = [] if arm == 'control' else [dict(decoder_and_hash_seconds=.1, spectral_loss_seconds=.01)]*3
                    diagnostic = dict(original_ids=ids, timing={'graph_build_seconds': 0. if arm == 'control' else .2},
                                      trajectory=dict(decoded_spectral=dict(records=records)))
                    zipped.writestr('bundle/batches/'+corruption+'/batch0000/diagnostics.json', json.dumps(diagnostic))
                zipped.writestr('bundle/config.json', '{}')
            identity = analysis.identity(archive)
            details = dict(cache_count=15, selection_ids=ids) if arm == 'cache' else dict(counts=counts, cases=cases,
                worker_wall_seconds=20., basis_checks=0 if arm == 'control' else 15, basis_passed=True)
            receipt = dict(status='accepted and cleanup ack uploaded', label=label,
                kind='cache' if arm == 'cache' else 'inference', archive_identity=identity,
                local_archive=str(archive), ack_uploaded=True, details=details,
                ack=dict(status='local_and_drive_verified', label=label, kind='cache' if arm == 'cache' else 'inference',
                         local_archive=str(archive), **identity))
            (root/'accepted'/(label+'.json')).write_text(json.dumps(receipt))
    (root/'operator/plan.json').write_text(json.dumps(plan))
    result = analysis.aggregate(root)
    assert result['status'] == 'partial' and result['external_gate'] is None
    assert result['coverage']['objects'] == 4 and result['coverage']['predictions_per_arm'] == 60
    assert result['strata']['prior_M100_20']['objects'] == 2
    assert result['strata']['outside_prior_M100_2448']['objects'] == 2
    metrics = result['comparisons']['M100_vs_control']['pm']
    assert metrics['delta_pp'] == 100. and metrics['corrected'] == 60 and metrics['harmed'] == 0
    assert result['per_corruption']['control']['gaussian']['pm_correct'] == 0
    assert result['ranks']['M100']['count'] == 60 and result['ranks']['M100']['max'] == 101
    assert result['timing']['component_pending_until_final'] is True
    rejects(lambda: analysis.aggregate(root, final=True))
    components = analysis.final_components(root/'shard000_M100.zip',
        json.loads((root/'accepted/shard000_M100.json').read_text())['details']['cases'])
    assert abs(components['graph_build_seconds']-3.) < 1e-12
    assert abs(components['decoder_and_hash_seconds']-4.5) < 1e-12
    path = root/'accepted/shard000_M100.json'
    good = json.loads(path.read_text())
    for changed in ('labels', 'paired_identity', 'filter_sha256', 'pm', 'duplicate', 'missing', 'count'):
        bad = copy.deepcopy(good)
        if changed == 'duplicate':
            bad['details']['cases'].append(copy.deepcopy(bad['details']['cases'][0]))
        elif changed == 'missing':
            bad['details']['cases'].pop()
        elif changed == 'count':
            bad['details']['counts']['gaussian']['pm_correct'] = 0
        elif changed == 'paired_identity':
            bad['details']['cases'][0][changed] = {'prepared': 'wrong'}
        elif changed == 'filter_sha256':
            bad['details']['cases'][0][changed][0] = 'c'*64
        else:
            bad['details']['cases'][0][changed][0] = 39
        path.write_text(json.dumps(bad))
        rejects(lambda: analysis.aggregate(root))
    path.write_text(json.dumps(good))
    cache = root/'accepted/shard001_cache.json'
    held = cache.read_text()
    cache.unlink()
    partial = analysis.aggregate(root)
    assert partial['coverage']['objects'] == 2 and partial['incomplete_shards'] == ['shard001']
    assert partial['external_gate'] is None
    cache.write_text(held)

    # Full count/protocol fixture; statistical and timing functions were exercised above.
    full = root/'full'
    (full/'operator').mkdir(parents=True)
    (full/'accepted').mkdir()
    full_plan = copy.deepcopy(plan)
    full_plan['shards'] = []
    (full/'common_identity.json').write_bytes((root/'common_identity.json').read_bytes())
    for ordinal in range(39):
        shard_label = 'shard%03d' % ordinal
        ids = list(range(ordinal*64, min((ordinal+1)*64, 2468)))
        selection = dict(schema_version=1, dataset='modelnet40_c', indices=ids)
        (full/'operator'/(shard_label+'.json')).write_text(json.dumps(selection))
        full_plan['shards'].append(dict(number=ordinal, label=shard_label, count=len(ids),
            selection_file=shard_label+'.json', arms=['control', 'half0', 'M100']))
        for arm in ('cache', *analysis.ARMS):
            label = shard_label+'_'+arm
            archive = full/(label+'.zip')
            archive.write_bytes(b'Full-count fixture only; no inference claim')
            archive_identity = analysis.identity(archive)
            cases, counts = [], {}
            for corruption in analysis.CORRUPTIONS:
                for start in range(0, len(ids), 2):
                    truth = [index % 40 for index in ids[start:start+2]]
                    cases.append(dict(case=corruption+'/'+str(start//2), ids=ids[start:start+2], labels=truth,
                        pm=truth if arm == 'M100' else [(value+1) % 40 for value in truth], terminal=truth,
                        paired_identity={'same': 'prepared'}, filter_sha256=None if arm == 'control' else ['b'*64]*2,
                        rank=None if arm == 'control' else [100, 101]))
                counts[corruption] = dict(n=len(ids), pm_correct=len(ids) if arm == 'M100' else 0,
                    terminal_correct=len(ids), reconstruction_seconds=1., peak_gpu_memory_mb=128.)
            details = dict(cache_count=15*len(ids)//2, selection_ids=ids) if arm == 'cache' else dict(
                cases=cases, counts=counts, worker_wall_seconds=20., basis_passed=True,
                basis_checks=0 if arm == 'control' else 15*len(ids)//2)
            kind = 'cache' if arm == 'cache' else 'inference'
            receipt = dict(status='accepted and cleanup ack uploaded', label=label, kind=kind,
                archive_identity=archive_identity, local_archive=str(archive), ack_uploaded=True, details=details,
                ack=dict(status='local_and_drive_verified', label=label, kind=kind, local_archive=str(archive), **archive_identity))
            (full/'accepted'/(label+'.json')).write_text(json.dumps(receipt))
    (full/'operator/plan.json').write_text(json.dumps(full_plan))
    analysis.final_components = lambda *args: dict(graph_build_seconds=1., decoder_and_hash_seconds=2., spectral_loss_seconds=.1)
    def exact_metrics(winner, baseline):
        delta = float((winner.astype(float)-baseline).mean()*100)
        return dict(delta_pp=delta, object_cluster_ci95_pp=[delta, delta])
    analysis.accuracy.paired_metrics = exact_metrics
    accepted_full = analysis.aggregate(full, final=True)
    assert accepted_full['status'] == 'complete' and accepted_full['external_gate']['advance_to_external'] is True
    assert accepted_full['coverage']['objects'] == 2468 and accepted_full['coverage']['predictions_per_arm'] == 37020
    assert accepted_full['coverage']['inference_jobs'] == 117 and accepted_full['coverage']['cache_jobs'] == 39
    assert accepted_full['strata']['prior_M100_20']['objects'] == 20
    assert accepted_full['strata']['outside_prior_M100_2448']['objects'] == 2448
    assert accepted_full['provenance']['raw_zip_sha_verified'] is True
    analysis.accuracy.paired_metrics = lambda *args: dict(delta_pp=1., object_cluster_ci95_pp=[0., 2.])
    assert analysis.aggregate(full, final=True)['external_gate']['advance_to_external'] is False
    analysis.accuracy.paired_metrics = lambda *args: dict(delta_pp=1., object_cluster_ci95_pp=[.001, 2.])
    assert analysis.aggregate(full, final=True)['external_gate']['advance_to_external'] is True
    analysis.accuracy.paired_metrics = lambda *args: dict(delta_pp=.99, object_cluster_ci95_pp=[.001, 2.])
    assert analysis.aggregate(full, final=True)['external_gate']['advance_to_external'] is False
    altered = full/'shard000_control.zip'
    altered.write_bytes(b'Altered raw bytes must be rejected by the final hash')
    rejects(lambda: analysis.aggregate(full, final=True))

print('FULL_AGGREGATION_CPU_SELFCHECK_COMPLETE')
