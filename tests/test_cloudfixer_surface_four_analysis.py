import numpy as np
import pytest

from scripts.analyze_cloudfixer_surface_four import pm_and_terminal, record_predictions, accuracy_row


def test_float64_probability_mean_and_true_terminal():
    logits = np.full((1, 3, 40), -100., dtype=np.float32)
    logits[0, :, 0] = [2., 2., -20.]
    logits[0, :, 1] = 0
    pm, terminal = pm_and_terminal(logits)
    assert pm.tolist() == [0] and terminal.tolist() == [1]
    assert logits.mean(axis=1).argmax(axis=-1).tolist() == [1]
    logits[0, 1, 1] = np.nan
    with pytest.raises(ValueError):
        pm_and_terminal(logits)


def test_recount_rejects_repeated_and_out_of_range_ids():
    predictions = np.full(4, -1, dtype=np.int16)
    record_predictions(predictions, np.array([0, 2]), np.array([3, 4]))
    for bad in ([2], [1, 1], [-1], [4]):
        with pytest.raises(ValueError):
            record_predictions(predictions, np.array(bad), np.ones(len(bad), dtype=int))


def test_partial_full_reference_stays_missing():
    row = accuracy_row('pointmae', 'all15', np.array([0, 1]), np.array([0, 0]),
                       np.array([0, 1]), np.array([True, False]),
                       {'cf': np.array([0, -1])})
    assert row['n'] == 2 and row['pm_correct'] == 1 and row['terminal_correct'] == 2
    assert row['cf_correct'] is None and row['pm_minus_cf_pp'] is None
    assert row['accepted'] == 1


def test_end_to_end_archive_report_matches_ids_and_keeps_partial_reference(tmp_path, monkeypatch):
    import csv
    import json
    from types import SimpleNamespace
    from scripts import analyze_cloudfixer_surface_four as analysis
    from scripts import run_cloudfixer_surface_four as runner

    labels = np.array([0, 1, 2, 3])
    lineage = dict(author_commit='a', dataset_files={}, checkpoint_sha256={m:m for m in analysis.MODELS},
                   diffusion_sha256='prior', frozen_states={m:m for m in analysis.MODELS})
    class_names = json.loads((analysis.ROOT/'cfgs/cloudfixer_point2vec_assets.json').read_text())['class_order']

    class Runtime:
        identity = dict(author_commit='a', dataset_files={}, checkpoint_sha256=dict(lineage['checkpoint_sha256'], prior='prior'),
                        frozen_states=dict(lineage['frozen_states'], prior=analysis.PRIOR_STATE), heads={m:'adapter' for m in analysis.MODELS},
                        class_order=class_names, hardware={'device':'CPU fixture'}, timing='test', rng='test')

        def calibrate(self, corr):
            return dict(selected_batch_size=2, trials=[dict(size=1, equivalent=True), dict(size=2, equivalent=True)],
                        independent_b1={'0': {'heads': {m:{'exact':True} for m in analysis.MODELS}}})

        def predict(self, corr, ids):
            n = len(ids)
            logits = np.zeros((n,4,3,40), dtype=np.float32)
            logits[np.arange(n), :, :, np.asarray(ids)] = 3
            return dict(logits=logits, pm=np.repeat(np.asarray(ids)[:,None], 4, axis=1),
                        terminal=np.repeat(np.asarray(ids)[:,None], 4, axis=1), pose=np.tile(np.eye(3), (n,1,1)), reason=np.repeat('unstable',n),
                        angle_degrees=np.zeros(n), denoise_seconds=.1, head_seconds=np.ones(4)*.1,
                        wall_seconds=.5, peak_mib=50., input_sha256=['a'*64]*n, trajectory_sha256=['b'*64]*n)

        def verify_frozen(self):
            return self.identity['frozen_states']

    monkeypatch.setattr(runner, 'N_OBJECTS', len(labels))
    monkeypatch.setattr(analysis, 'N', len(labels))
    runtime = Runtime()
    runtime.labels = labels
    raw = tmp_path/'raw'
    runner.run(runtime, raw)
    archive = runner.finalize(raw)
    references = tmp_path/'references'
    references.mkdir()
    scores = np.zeros((15,len(labels),4), dtype=np.int16)
    scores[:,:,3] = -1
    scores[2,:,3] = 0
    pilot = np.tile(labels, (3,15,1))
    pilot[:,:,1:] = -1
    source = references/'references.npz'
    np.savez_compressed(source, cf=scores, baseline_pm=scores, baseline_terminal=scores,
                        pilot=pilot, labels=labels)
    metadata = dict(lineage=lineage, historical_timing=[], protocol={}, recovery=[],
                    predictions_sha256=analysis.file_sha(source))
    (references/'references.json').write_text(json.dumps(metadata))
    args = SimpleNamespace(archive=archive, references=references, sha256=analysis.file_sha(archive), out=tmp_path/'report')
    analysis.report(args)
    with (args.out/'per_corruption.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (args.out/'summary.csv').open() as f:
        summary = list(csv.DictReader(f))
    assert len(rows) == 60 and all(r['pm_correct'] == '4' for r in rows)
    assert next(r for r in rows if r['model'] == 'pointmae' and r['group'] == 'background')['cf_correct'] == '1'
    pointmae, = [r for r in summary if r['model'] == 'pointmae' and r['group'] == 'all15']
    assert pointmae['n'] == '60' and pointmae['cf_correct'] == '' and pointmae['pm_minus_cf_pp'] == ''
    with (args.out/'pointmae_paired128.csv').open() as f:
        pilot_total, = [r for r in csv.DictReader(f) if r['group'] == 'all15']
    assert pilot_total['unique_id_corruption_pairs'] == '15' and pilot_total['repeat2_correct'] == '15'
    with (args.out/'runtime.csv').open() as f:
        timed, = [r for r in csv.DictReader(f) if r['group'] == 'all15' and r['model'] == 'pointmae']
    assert float(timed['pipeline_all_heads_s']) == 15 and float(timed['pipeline_objects_per_s']) == 4
    runtime.identity['checkpoint_sha256']['pointmae'] = 'foreign'
    with pytest.raises(ValueError, match='checkpoint'):
        analysis.check_lineage(runtime.identity, lineage)
