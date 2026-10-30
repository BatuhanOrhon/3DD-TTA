"""CPU fixtures for local evidence trust boundaries, pairing and frozen selection."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import zipfile

import numpy as np

import analyze_sensitivity_local as analysis

PREP = None


def archive_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def make_original_fixture(root, context, bad_ids=False):
    """140 held-out IDs deliberately require five original B32 shards/corruption."""
    assets = root / 'assets'; assets.mkdir()
    audit = dict(backends=dict(fixture=[dict(path='/fixture/native.so', bytes=4, sha256='a' * 64)]))
    (assets / 'native.json').write_text(json.dumps(audit))
    manifest = dict(guidance=dict(native_audit='native.json'), files={
        'native.json': analysis.identity(assets / 'native.json'), 'checkpoint.pth': dict(bytes=1, sha256='b' * 64)})
    ids = list(range(1, 281, 2)); dataset = 'modelnet40_c'
    selected = dict(dataset=dataset, indices=ids, n_examples=2468, stage='heldout')
    selection_bytes = json.dumps(selected).encode()
    scheduler = dict(config=dict(beta_end=.02, beta_start=.0001, beta_schedule='linear', clip_sample=False,
        num_train_timesteps=1000, prediction_type='epsilon', set_alpha_to_one=True),
        final_alpha_cumprod=1., timesteps=list(range(990, -1, -10)))
    schedulers = {c: dict(scheduler, active_timesteps=scheduler['timesteps'][-(35 if c == 'background' else 5):])
                  for c in ('gaussian', 'background')}
    config = dict(backbone='pointmae', dataset=dataset, seed=0, selected_original_ids=ids, status='partial',
        num_classes=40, n_examples=2468, severity=5, batch_size=32, corruptions=list(schedulers), run_id='fixture-original',
        selection_manifest=selected, selection_manifest_identity=dict(bytes=len(selection_bytes), sha256=hashlib.sha256(selection_bytes).hexdigest()),
        classifier_state_before_sha256='c' * 64, classifier_state_after_sha256='c' * 64,
        lion_state_before=dict(vae='d', priors='e'), lion_state_after=dict(vae='d', priors='e'),
        initialization_rng_guard=dict(rng_unchanged=True), pointmae_checkpoint='checkpoint.pth',
        checkpoint_load=dict(strict=True, missing_keys=[], unexpected_keys=[], checkpoint_sha256='b' * 64),
        native_bootstrap=dict(no_build=True, bindings=dict(fixture=audit['backends']['fixture'][0])),
        source_classifier_preprocessing=dict(fixture=True), resolved_model_config={}, scheduler_records=schedulers)
    files = {name: b'fixture' for name in ('command.txt', 'environment.txt', 'stdout.log', 'notes.md')}
    files.update({'config.json': json.dumps(config).encode(), 'selection_manifest.json': selection_bytes})
    for corruption in schedulers:
        parts = []
        for number, start in enumerate(range(0, len(ids), 32)):
            index = np.array(ids[start:start + 32], dtype=np.int64)
            labels = index % 3
            logits = np.zeros((len(index), 40), dtype=np.float32)
            logits[np.arange(len(index)), labels] = 3.
            data = dict(index=index.copy(), label=labels, prediction=labels.copy(), logits=logits)
            if bad_ids and number == 4:
                data['index'][0] += 1
            parts.append(data)
            prefix = 'batches/' + corruption + '/batch%04d/' % number
            files[prefix + 'predictions.npz'] = archive_bytes(data)
            files[prefix + 'diagnostics.json'] = json.dumps(dict(seed=0, rng_unchanged=True,
                geometry_shape=[len(index), 2048, 3], geometry_sha256='geometry-%d' % number,
                scheduler=schedulers[corruption], guidance_metrics={}, shared_reconstruction_seconds=1.,
                head_seconds=.2, whole_process_peak_gpu_memory_mb=7., preparation_mode='fresh_encode')).encode()
        files[corruption + '_predictions.npz'] = archive_bytes({k: np.concatenate([part[k] for part in parts]) for k in parts[0]})
    files['per_corruption.csv'] = b'corruption,n_examples,n_correct,accuracy\ngaussian,140,140,1\nbackground,140,140,1\n'
    files['summary.csv'] = b'run_id,status,n_corruptions,total_examples,total_correct,macro_accuracy\nfixture-original,partial,2,280,280,1\n'
    files['bundle_manifest.json'] = json.dumps({name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in files.items()}).encode()
    path = root / 'fixture-original.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    fake_controller = types.SimpleNamespace(verify_common_config=lambda *args: None)
    local = list(context)
    local[1], local[3], local[4], local[6], local[-1] = fake_controller, assets, manifest, {('development', dataset): selected}, ('gaussian', 'background')
    job = dict(dataset=dataset, trial=dict(id='original3dd', method='original3dd', family='original3dd'))
    return path, job, tuple(local)


class LocalAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = analysis.load_context(PREP)

    def test_original_pointmae_140_ids_five_b32_shards_and_terminal_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path, job, context = make_original_fixture(Path(directory), self.context)
            config, cases, rows, traces, costs = analysis.inspect_raw(path, job, 'development', context, {}, False, {}, {}, {}, [])
            self.assertEqual(config['backbone'], 'pointmae')
            self.assertEqual(len(traces), 10)
            self.assertEqual(sum(row['n'] for row in rows), 280)
            self.assertEqual({row['policy'] for row in rows}, {'terminal'})
            self.assertEqual(cases[0]['accuracy_pct'], 100.)
            self.assertEqual(len(costs), 10)

    def test_original_late_remainder_wrong_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, job, context = make_original_fixture(Path(directory), self.context, bad_ids=True)
            with self.assertRaisesRegex(ValueError, 'indexed logits/labels'):
                analysis.inspect_raw(path, job, 'development', context, {}, False, {}, {}, {}, [])

    def test_physical_cache_missing_is_pending_and_corrupt_bytes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'namespace/modelnet40_c/gaussian/a.npz'; file.parent.mkdir(parents=True)
            np.savez(file, z=np.array([1.], dtype=np.float32))
            record = dict(local_path=str(analysis.REMOTE / file.relative_to(root).as_posix()),
                          drive_path=analysis.DRIVE + 'cache.npz', **analysis.identity(file))
            self.assertFalse(analysis.physical_caches([record], [root / 'missing'])[0]['local_verified'])
            self.assertTrue(analysis.physical_caches([record], [root])[0]['local_verified'])
            file.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                analysis.physical_caches([record], [root])

    def test_freeze_matches_registered_controller_three_or_four_conditions(self):
        controller, registry = self.context[1], self.context[5]
        calibration = dict(status='verified_label_free_gpu', coefficients=dict(legacy_l2=10., huber=20., power=.3))
        for spectral, chamfer, n in [('cs-huber-r010', 'c-base', 3), ('cs-l2-r010-step-low', 'c-p80', 4)]:
            scores = [dict(trial_id=name, family=trial['family'], accuracy_pct=99. if name in (spectral, chamfer) else 1.)
                      for name, trial in registry.items()]
            freeze = analysis.freeze_selection(scores, registry, dict(bytes=1, sha256='f' * 64), controller, calibration)
            self.assertEqual(freeze['selected_spectral_trial_id'], spectral)
            self.assertEqual(freeze['selected_chamfer_trial_id'], chamfer)
            self.assertEqual(len(freeze['conditions']), n)
            self.assertEqual(freeze['conditions'][1], dict(id='matched-chamfer', role='matched_chamfer', trial_id='matched-chamfer'))

    def test_manifest_tampering_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, context = make_original_fixture(root, self.context)
            with zipfile.ZipFile(path) as source:
                files = {name: source.read(name) for name in source.namelist()}
            files['notes.md'] = b'changed after sealing'
            with zipfile.ZipFile(path, 'w') as target:
                for name, data in files.items():
                    target.writestr(name, data)
            with self.assertRaisesRegex(ValueError, 'Sealed member SHA/size'):
                context[7](path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prep', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    PREP = args.prep.resolve()
    unittest.main(argv=[__file__] + remaining)
