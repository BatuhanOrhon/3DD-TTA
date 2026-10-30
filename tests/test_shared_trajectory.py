"""CPU checks; no native kernels, real weights or GPU inference."""
import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from legacy_trajectory import model_buffer_hashes, rng_hash
from shared_trajectory import SharedTrajectoryObserver
from scripts.analyze_legacy_trajectory import policies
from scripts.analyze_scan_trajectory import analyze_bundle


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('counter', torch.zeros(1))
        self.calls = 0

    def decoder(self, unused, beta, context, style):
        self.calls += 1
        self.counter.add_(1)
        return context.reshape(-1, 2, 3) + torch.rand(len(context), 2, 3)


def collect(count=21, horizon=10, num_classes=15):
    vae = VAE().eval()
    heads = {name: torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, num_classes)).eval()
             for name in ('curvenet', 'pointnext', 'dgcnn', 'pointmae')}
    for model in [vae, *heads.values()]:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    def classify(points):
        return {name: head(points.mean(1)) for name, head in heads.items()}
    observer = SharedTrajectoryObserver(vae, heads, classify, horizon, num_classes=num_classes)
    latent = torch.randn(count, 6, 1, 1, requires_grad=True)
    style = torch.randn(count, 3, 1, 1, requires_grad=True)
    observer.initial(input_points=latent, style=style)
    return observer, latent, style, vae, heads


def finish(observer, latent, style):
    for i in range(observer.horizon):
        observer.clean(latent, style, i, (observer.horizon - i - 1) * 10)
    observer.final_state(latent, style, 0)
    terminal = {name: torch.zeros(len(latent), observer.num_classes) for name in observer.heads}
    observer.terminal(terminal)


class SharedObserverTests(unittest.TestCase):
    def test_shared_stochastic_probes_restore_rng_buffers_and_live_autograd(self):
        for horizon in (10, 30):
            observer, latent, style, vae, heads = collect(horizon=horizon)
            # BN running buffers are saved by the live graph; unchanged copies break backward.
            loss = sum(head(latent[:, :3, 0, 0]).sum() for head in heads.values())
            before_rng = rng_hash()
            before_buffers = model_buffer_hashes(dict(vae=vae, **heads))
            finish(observer, latent, style)
            loss.backward()
            self.assertIsNotNone(latent.grad)
            self.assertEqual(before_rng, rng_hash())
            self.assertEqual(before_buffers, model_buffer_hashes(dict(vae=vae, **heads)))
            self.assertEqual(vae.calls, 3)
            for name in heads:
                arrays = observer.arrays(name)
                self.assertEqual(arrays['trajectory_logits'].shape, (21, 4, 15))
                self.assertEqual(arrays['candidate_loop_counts'].tolist(), observer.steps + [horizon])
            self.assertTrue(observer.diagnostics()['complete'])
            json.dumps(observer.diagnostics(), allow_nan=False)

    def test_missing_checkpoint_and_bad_logits_fail_without_advancing_rng(self):
        observer, latent, style, _, _ = collect()
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            observer.clean(latent, style, 4, 50)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            observer.arrays('dgcnn')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            observer.terminal({})
        observer.classify_decoded = lambda points: {'dgcnn': torch.zeros(len(points), 40)}
        before = rng_hash()
        with self.assertRaisesRegex(ValueError, 'heads'):
            observer.clean(latent, style, 2, 70)
        self.assertEqual(before, rng_hash())

    def test_entropy_terminal_ties_and_probability_mean_are_class_agnostic(self):
        logits = np.zeros((2, 4, 15))
        logits[1, 0, 2] = 20
        logits[1, 3, 4] = 2
        result = policies(logits, 3)
        self.assertEqual(result['selected_candidate'].tolist(), [3, 0])
        self.assertEqual(result['minimum_entropy'].tolist(), [0, 2])
        self.assertEqual(result['probability_mean'].tolist(), [0, 2])

    def test_modelnet_class_contract_remains_strict(self):
        observer, latent, style, _, _ = collect(num_classes=40)
        finish(observer, latent, style)
        self.assertEqual(observer.arrays('dgcnn')['trajectory_logits'].shape, (21, 4, 40))
        for classes in (15, 40):
            observer, _, _, _, _ = collect(num_classes=classes)
            with self.assertRaisesRegex(ValueError, 'logits'):
                observer._logits({name: torch.zeros(21, 40 if classes == 15 else 15) for name in observer.heads})
        with self.assertRaisesRegex(ValueError, 'classes'):
            collect(num_classes=16)

    def test_modelnet_40_class_probes_preserve_rng_buffers_and_gradient(self):
        for horizon in (10, 30):
            observer, latent, style, vae, heads = collect(count=18, horizon=horizon, num_classes=40)
            loss = sum(head(latent[:, :3, 0, 0]).sum() for head in heads.values())
            before = rng_hash(), model_buffer_hashes(dict(vae=vae, **heads))
            finish(observer, latent, style)
            loss.backward()
            self.assertIsNotNone(latent.grad)
            self.assertEqual(before, (rng_hash(), model_buffer_hashes(dict(vae=vae, **heads))))
            self.assertEqual(vae.calls, 3)
            for name in heads:
                self.assertEqual(observer.arrays(name)['trajectory_logits'].shape, (18, 4, 40))

    def test_observer_rejects_class_count_mismatch_and_nonfinite_logits(self):
        for classes, wrong in ((15, 40), (40, 15)):
            observer, latent, style, _, heads = collect(num_classes=classes)
            for bad in (torch.zeros(21, wrong), torch.full((21, classes), float('nan'))):
                observer.classify_decoded = lambda points: {name: bad for name in heads}
                before = rng_hash()
                with self.assertRaisesRegex(ValueError, 'logits'):
                    observer.clean(latent, style, 2, 70)
                self.assertEqual(before, rng_hash())
        for invalid in (0, 16, 39, 41):
            with self.assertRaises(ValueError):
                SharedTrajectoryObserver(observer.vae, heads, observer.classify_decoded, 10,
                                         num_classes=invalid)


class AnalyzerTests(unittest.TestCase):
    def seal(self, root):
        manifest = {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in root.rglob('*') if p.is_file() and p.name != 'bundle_manifest.json'}
        (root / 'bundle_manifest.json').write_text(json.dumps(manifest))

    def fixture(self, root, full=False, dataset='scanobjectnn_c', indices=None):
        from research_artifacts import CORRUPTIONS
        corruptions = list(CORRUPTIONS) if full else ['gaussian']
        total, classes, split = (581, 15, 'main_split') if dataset == 'scanobjectnn_c' else (2468, 40, 'official_modelnet40_test')
        config = dict(trajectory_observation=True, dataset=dataset, n_examples=total,
                      split=split, num_classes=classes, batch_size=70, method='gsd_inspired_latent_spectral_guidance',
                      severity=5, seed=0, status='complete' if full else 'partial',
                      classifier_state_before_sha256='abc', classifier_state_after_sha256='abc',
                      lion_state_before={'vae': 'x'}, lion_state_after={'vae': 'x'},
                      corruptions=corruptions, backbone='dgcnn')
        if indices is not None:
            selection = dict(dataset=dataset, n_examples=total, indices=indices, stage='development')
            raw = json.dumps(selection).encode()
            (root / 'selection_manifest.json').write_bytes(raw)
            config.update(selected_original_ids=indices, evaluation_n_examples=len(indices), selection_manifest=selection,
                          selection_manifest_identity=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        (root / 'config.json').write_text(json.dumps(config))
        count = len(indices) if indices is not None else total if full else 70
        for corruption in corruptions:
            for batch, start in enumerate(range(0, count, 70)):
                end = min(start + 70, count)
                observer, latent, style, _, _ = collect(end - start, 30 if corruption == 'background' else 10, num_classes=classes)
                finish(observer, latent, style)
                directory = root / 'batches' / corruption / ('batch%04d' % batch)
                directory.mkdir(parents=True)
                data = observer.arrays('dgcnn')
                data.update(index=np.asarray(indices[start:end]) if indices is not None else np.arange(start, end), label=np.zeros(end - start, dtype=np.int64),
                            prediction=np.zeros(end - start, dtype=np.int64), logits=data['trajectory_logits'][:, 3])
                np.savez_compressed(directory / 'predictions.npz', **data)
                (directory / 'diagnostics.json').write_text(json.dumps(dict(trajectory=observer.diagnostics(), rng_unchanged=True)))
        self.seal(root)
        return directory

    def test_modelnet_nonprefix_selection_and_split_class_mismatches(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root, dataset='modelnet40_c', indices=list(range(7, 149, 2)))
            report = analyze_bundle(root, True)
            self.assertEqual(report['status'], 'partial')
            self.assertTrue(report['selection_complete'])
            self.assertEqual(report['macro'][0]['n'], 71)
            with self.assertRaisesRegex(ValueError, 'all15'):
                analyze_bundle(root)
            config_path = root / 'config.json'
            config = json.loads(config_path.read_text())
            for key, wrong in [('num_classes', 15), ('split', 'main_split'), ('n_examples', 581)]:
                modified = dict(config, **{key: wrong})
                config_path.write_text(json.dumps(modified))
                self.seal(root)
                with self.assertRaisesRegex(ValueError, 'dataset'):
                    analyze_bundle(root, True)

    def test_full_modelnet_all15_class_and_original_id_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root, full=True, dataset='modelnet40_c')
            report = analyze_bundle(root)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['macro'][0]['n'], 37020)
            self.assertEqual(len(report['per_corruption']), 45)
            self.assertEqual(report['macro'][0]['accuracy_pct'], 100.)
            with np.load(root / 'batches' / 'background' / 'batch0035' / 'predictions.npz',
                         allow_pickle=False) as values:
                self.assertEqual(values['trajectory_logits'].shape, (18, 4, 40))

    def test_full_all15_analysis_and_guarded_partial_prefix(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root, full=True)
            result = analyze_bundle(root)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['per_corruption']), 45)
            self.assertEqual(result['macro'][0]['n'], 8715)
            self.assertEqual(result['macro'][0]['accuracy_pct'], 100.)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directory = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'all15'):
                analyze_bundle(root)
            self.assertEqual(analyze_bundle(root, True)['status'], 'partial')
            with np.load(directory / 'predictions.npz') as values:
                arrays = {k: values[k].copy() for k in values.files}
            arrays['index'][-1] = 0
            np.savez_compressed(directory / 'predictions.npz', **arrays)
            self.seal(root)
            with self.assertRaisesRegex(ValueError, 'Original IDs'):
                analyze_bundle(root, True)

    def test_terminal_or_probe_guard_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directory = self.fixture(root)
            path = directory / 'diagnostics.json'
            diagnostic = json.loads(path.read_text())
            diagnostic['trajectory']['probes'][0]['rng_after_sha256'] = 'changed'
            path.write_text(json.dumps(diagnostic))
            self.seal(root)
            with self.assertRaisesRegex(ValueError, 'purity'):
                analyze_bundle(root, True)

    def test_realistic_sealed_zip_root_and_member_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / 'run-id'
            root.mkdir()
            self.fixture(root)
            archive = root.parent / 'run.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                for path in root.rglob('*'):
                    if path.is_file():
                        stream.write(path, 'run-id/' + path.relative_to(root).as_posix())
            self.assertEqual(analyze_bundle(archive, True)['status'], 'partial')
            (root / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA/size'):
                analyze_bundle(root, True)

    def test_modelnet_schema_and_terminal_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directory = self.fixture(root, dataset='modelnet40_c')
            with self.assertRaisesRegex(ValueError, 'all15'):
                analyze_bundle(root)
            self.assertEqual(analyze_bundle(root, True)['status'], 'partial')
            config_path = root / 'config.json'
            original = json.loads(config_path.read_text())
            for key, value in (('num_classes', 15), ('n_examples', 581), ('split', 'main_split')):
                config_path.write_text(json.dumps(dict(original, **{key: value})))
                self.seal(root)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    analyze_bundle(root, True)
            config_path.write_text(json.dumps(original))
            path = directory / 'predictions.npz'
            with np.load(path) as values:
                original_arrays = {key: values[key].copy() for key in values.files}
            for key, value in (('label', np.full(70, 40)),
                               ('trajectory_logits', np.zeros((70, 4, 15))),
                               ('logits', np.ones((70, 40))),
                               ('index', np.arange(1, 71))):
                np.savez_compressed(path, **dict(original_arrays, **{key: value}))
                self.seal(root)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    analyze_bundle(root, True)


if __name__ == '__main__':
    unittest.main()
