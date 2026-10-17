"""CPU contracts for sampling views; native CUDA parity remains a Colab gate."""
import json
import hashlib
import random
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

import numpy as np
import torch

from trajectory_sampling import alternate_fps, collect_views, view_policies, view_starts


def cpu_sampler(points, count):
    """Deterministic index0-start FPS reference on small test point clouds."""
    batch, n, _ = points.shape
    indices = torch.zeros(batch, count, dtype=torch.long)
    distances = torch.full((batch, n), float('inf'))
    for k in range(1, count):
        last = points[torch.arange(batch), indices[:, k - 1]]
        distances = torch.minimum(distances, (points - last[:, None]).square().sum(-1))
        indices[:, k] = distances.argmax(1)
    return points.gather(1, indices[..., None].expand(-1, -1, 3)), indices


class SamplingTests(unittest.TestCase):
    def test_analyzer_requires_baseline_model_data_and_spec_identities(self):
        import copy
        from pathlib import Path
        from eval_gsd_tta_v2 import parse_arguments
        from scripts.analyze_trajectory_sampling import validate_assets
        name = 'cfgs/legacy_trajectory_provenance.json'
        content = Path(name).read_bytes().replace(b'\r\n', b'\n')
        spec = json.loads(content)
        config = dict(cli=vars(parse_arguments([])), corruptions=['gaussian'],
            assets=copy.deepcopy(spec['baseline_assets']), source_delivery=dict(files={
                name: dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())}))
        validate_assets(config)
        # Relocated assets are accepted only when their content identity matches.
        original = 'pointnet_ckpts/modelnet_jt.pth'
        config['cli']['pointmae_ckpt'] = '/alternate/model.pth'
        config['assets']['/alternate/model.pth'] = config['assets'].pop(original)
        validate_assets(config)
        for asset in ('/alternate/model.pth', 'data/modelnet40_c/data_gaussian_5.npy',
                      'data/modelnet40_c/label.npy'):
            for field, bad_value in [('sha256', 'wrong'), ('bytes', 0)]:
                broken = copy.deepcopy(config)
                broken['assets'][asset][field] = bad_value
                with self.assertRaisesRegex(ValueError, 'Baseline asset identity mismatch'):
                    validate_assets(broken)
            broken = copy.deepcopy(config)
            del broken['assets'][asset]
            with self.assertRaisesRegex(ValueError, 'Baseline asset identity mismatch'):
                validate_assets(broken)
        config['source_delivery']['files'][name]['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'specification differs'):
            validate_assets(config)

    def test_partial_packaging_is_backed_up_and_modified_selection_blocks_launch(self):
        import zipfile
        from tests.test_legacy_gsd_restore import workspace_temp
        from scripts.trajectory_sampling_colab import package, file_sha, run
        with workspace_temp() as root:
            session = root / 'session'; session.mkdir()
            (session / 'smoke_indices.json').write_text('{}')
            (session / 'session.json').write_text(json.dumps(dict(delivery={},
                index_manifest_sha256={'smoke': 'intentionally incorrect'})))
            partial = session / 'smoke_seed0'; partial.mkdir()
            (partial / 'stdout.log').write_text('partial synthetic trace')
            receipt = package(session, root / 'drive')
            from pathlib import Path
            self.assertEqual(file_sha(Path(receipt['path'])), file_sha(Path(receipt['drive_path'])))
            with zipfile.ZipFile(receipt['path']) as archive:
                self.assertIn('smoke_seed0/stdout.log', archive.namelist())
            with patch('scripts.trajectory_sampling_colab.verify_delivery', return_value={}), \
                    patch('scripts.trajectory_sampling_colab.subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'original IDs changed'):
                    run(session, 'smoke', root / 'drive')
                launch.assert_not_called()

    def test_delivery_rejects_source_drift_and_preview_preserves_stage_scope(self):
        from pathlib import Path
        from tests.test_legacy_gsd_restore import workspace_temp
        from scripts.trajectory_sampling_colab import BASE, BASE_ZIP_SHA, MANIFEST, OVERRIDES, verify_delivery, command, CONDA
        from eval_gsd_tta_v2 import parse_arguments
        from legacy_gsd_artifacts import protocol
        with workspace_temp() as root:
            files = {}
            for name in OVERRIDES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic\n')
                files[name] = dict(bytes=10, sha256=hashlib.sha256(b'synthetic\n').hexdigest())
            (root / MANIFEST).write_text(json.dumps(dict(schema_version=1, base_commit=BASE,
                baseline_archive_sha256=BASE_ZIP_SHA, files=files, source_delivery='CPU fixture')))
            self.assertEqual(verify_delivery(root)['base_commit'], BASE)
            (root / OVERRIDES[0]).write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_delivery(root)
            for stage, names, count in [('smoke', ['gaussian'], 140),
                                         ('pilot', ['gaussian'], 280)]:
                # Command construction never launches; full macro14 scope is
                # checked by parsing a manifest containing all14 corruption IDs.
                from scripts.trajectory_sampling_colab import MACRO14
                selected = names if stage == 'smoke' else MACRO14
                (root / (stage + '_indices.json')).write_text(json.dumps(dict(
                    schema_version=1, dataset='modelnet40_c',
                    indices={c: list(range(count)) for c in selected})))
                cmd = command(root, stage)
                self.assertEqual(cmd[:len(CONDA)], CONDA)
                parsed = parse_arguments(cmd[len(CONDA) + 1:])
                cfg = protocol(parsed)
                self.assertEqual(cfg['expected_classifications'], count * len(selected))
                self.assertNotIn('background', cfg['corruptions'])

    def test_saved_paired_trace_preserves_a_logits_rng_and_passes_cpu_analysis(self):
        from tests.test_legacy_gsd_restore import workspace_temp
        from eval_gsd_tta_v2 import parse_arguments
        from legacy_gsd_artifacts import protocol
        from legacy_trajectory import TrajectoryObserver, rng_hash
        from scripts.analyze_trajectory_sampling import analyze, load_sampling
        from scripts.analyze_legacy_trajectory import Bundle
        from pathlib import Path

        def prefix_sampler(points, count=1024):
            indices = torch.arange(count).expand(len(points), count).clone()
            return points[:, :count].contiguous(), indices

        class VAE(torch.nn.Module):
            def decoder(self, unused, beta, context, style):
                return context.reshape(-1, 2048, 3)

        class Classifier(torch.nn.Module):
            def classification_only(self, points, only_unmasked):
                random.random(); np.random.rand()
                return points.mean((1, 2))[:, None] * torch.arange(40)[None] + torch.rand(len(points), 40)

        with workspace_temp() as root, patch('trajectory_sampling.native_fps', prefix_sampler):
            manifest = root / 'indices.json'
            manifest.write_text(json.dumps(dict(schema_version=1, dataset='modelnet40_c',
                                                 indices={'gaussian': [7, 11]})))
            outputs, rngs = [], []
            for mode in ('off', 'paired'):
                random.seed(3); np.random.seed(3); torch.manual_seed(3)
                args = parse_arguments(['--trajectory_mode', 'observe', '--seed', '0',
                    '--corruption', 'gaussian', '--corruption_scope', 'macro14',
                    '--trajectory_sampling', mode, '--M_max', '800',
                    '--indices_manifest', str(manifest)])
                config = protocol(args)
                directory = root / mode
                vae, classifier = VAE().eval(), Classifier().eval()
                observer = TrajectoryObserver(directory, 'gaussian', 0, [7, 11], 10, config['trajectory'])
                observer.begin(2, torch.device('cpu'))
                observer.bind(vae, classifier, lambda p: prefix_sampler(p)[0],
                              prepare_sampling_cloud=lambda p: p)
                cloud = torch.arange(2 * 2048 * 3, dtype=torch.float32).reshape(2, 2048, 3) / 10000
                for step in (3, 5, 8):
                    observer.clean(cloud.reshape(2, -1, 1, 1), torch.zeros(2, 3, 1, 1),
                                   step - 1, (10 - step) * 10)
                observer.final_state(cloud, torch.zeros(2, 3), 0)
                sampled, indices_a = prefix_sampler(cloud)
                logits = classifier.classification_only(sampled, False)
                targets = torch.tensor([3, 39])
                observer.terminal(logits, targets, sampling_context=(cloud, indices_a))
                rngs.append(rng_hash())
                outputs.append([np.load(directory / 'trajectory/gaussian' / ('0000_%s.npz' % name))['logits']
                                for name in ('clean_03', 'clean_05', 'clean_08', 'terminal')])
                if mode == 'paired':
                    (directory / 'predictions').mkdir()
                    np.savez_compressed(directory / 'predictions/gaussian_0000.npz',
                        indices=np.array([7, 11]), logits=logits.numpy(),
                        predictions=logits.argmax(1).numpy(), targets=targets.numpy())
                    files = {}
                    for name in ('eval_gsd_tta_v2.py', 'legacy_gsd_artifacts.py', 'legacy_trajectory.py', 'trajectory_sampling.py', 'cfgs/legacy_trajectory_provenance.json'):
                        content = Path(name).read_bytes().replace(b'\r\n', b'\n')
                        files[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                    config.update(execution_status='complete', completed_examples=2,
                        assets=json.loads(Path('cfgs/legacy_trajectory_provenance.json').read_text())['baseline_assets'],
                        source_manifest=files, model_state_unchanged=True,
                        model_state_before={}, model_state_after={},
                        source_delivery=dict(base_commit='a351e66ec58f08f3a2104e46cab2cb8b20cdf961',
                            manifest_sha256='synthetic CPU fixture', files=files),
                        trajectory_artifact_manifest={p.relative_to(directory).as_posix():
                            dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in directory.rglob('*') if p.is_file()})
                    (directory / 'config.json').write_text(json.dumps(config))
                    report = analyze([directory], root / 'analysis')
                    self.assertEqual(report['summary'][0]['n'], 2)
                    self.assertEqual(len(report['diagnostics'][0]['selected_entropy_histogram']), 4)
                    with self.assertRaises(FileExistsError):
                        analyze([directory], root / 'analysis')
                    # Metadata drift cannot pass even when original artifact bytes are intact.
                    config['source_manifest']['trajectory_sampling.py']['sha256'] = 'bad'
                    config['source_delivery']['files'] = {k: dict(v) for k, v in files.items()}
                    config['source_delivery']['files']['trajectory_sampling.py']['sha256'] = 'expected'
                    (directory / 'config.json').write_text(json.dumps(config))
                    bundle = Bundle(directory)
                    try:
                        with self.assertRaisesRegex(ValueError, 'Runtime source differs'):
                            load_sampling(bundle)
                    finally:
                        bundle.close()
            self.assertEqual(rngs[0], rngs[1])
            for a, b in zip(*outputs):
                np.testing.assert_array_equal(a, b)

    def test_starts_are_batch_and_view_count_independent_without_rng_changes(self):
        from legacy_trajectory import rng_hash
        before = rng_hash()
        all_starts = view_starts(0, [7, 11], 25, 7)
        np.testing.assert_array_equal(all_starts[:, :1], view_starts(0, [7, 11], 25, 1))
        np.testing.assert_array_equal(all_starts[1:], view_starts(0, [11], 25, 7))
        self.assertTrue(all(len(set(row)) == 7 and min(row) > 0 for row in all_starts))
        self.assertEqual(before, rng_hash())

    def test_alternate_fps_maps_to_original_coordinates_and_validates_native_start(self):
        points = torch.arange(24, dtype=torch.float32).view(1, 8, 3)
        sampled, indices = alternate_fps(points, [3], 4, cpu_sampler)
        self.assertEqual(indices[0, 0], 3)
        torch.testing.assert_close(sampled, points[:, indices[0]], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'start at input index0'):
            alternate_fps(points, [3], 4, lambda p, n: (p[:, :n], torch.ones(1, n).long()))

    def test_extra_views_restore_rng_and_buffers_even_if_classifier_raises(self):
        from legacy_trajectory import rng_hash
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('state', torch.zeros(1))
                self.fail = False

            def classification_only(self, points, only_unmasked):
                random.random(); np.random.rand(); torch.rand(1)
                self.state.add_(1)
                if self.fail:
                    raise RuntimeError('test failure')
                return torch.zeros(len(points), 40)

        model = Classifier().eval()
        points = torch.arange(24, dtype=torch.float32).view(1, 8, 3)
        _, a = cpu_sampler(points, 4)
        original_logits = torch.ones(1, 40)
        before = rng_hash()
        arrays, hashes = collect_views(points, a, original_logits, model, [11], 0,
                                        2, lambda event: None, cpu_sampler)
        np.testing.assert_array_equal(arrays['sampling_logits'][:, 0], original_logits.numpy())
        self.assertEqual(before, rng_hash())
        self.assertEqual(model.state.item(), 0)
        self.assertEqual(hashes['rng_before'], hashes['rng_after'])
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            collect_views(points, a, original_logits, model, [11], 0,
                          2, lambda event: None, cpu_sampler)
        self.assertEqual(before, rng_hash())
        self.assertEqual(model.state.item(), 0)

    def test_disagreement_score_and_terminal_tie_are_label_free(self):
        logits = np.zeros((2, 4, 2, 40))
        logits[0, :, :, 0] = 5
        logits[0, 0, 0, 0] = 50
        logits[0, 0, 1] = 0
        logits[0, 0, 1, 1] = 50
        votes = np.repeat(logits[:, -1, :1], 8, axis=1)
        result = view_policies(logits, votes)
        self.assertGreater(result['js'][0, 0], .99)
        np.testing.assert_allclose(result['js'][:, 1:], 0, atol=1e-12)
        self.assertEqual(result['selected_entropy_js'].tolist(), [3, 3])
        # A confident two-class conflict can have lower entropy than a softer
        # 40-class prediction; the separate JS penalty changes this choice.
        self.assertEqual(result['selected_entropy'].tolist(), [0, 3])
        self.assertEqual(result['eight_view_mean'].tolist(), [0, 0])
        bad = votes.copy(); bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            view_policies(logits, bad)

    def test_preview_is_opt_in_and_background_is_rejected(self):
        from eval_gsd_tta_v2 import parse_arguments
        from legacy_gsd_artifacts import protocol
        base = ['--trajectory_sampling', 'paired', '--trajectory_mode', 'observe',
                '--corruption_scope', 'macro14', '--seed', '0', '--M_max', '800']
        config = protocol(parse_arguments(base))
        self.assertEqual(config['expected_classifications'], 14 * 2468)
        self.assertNotIn('background', config['corruptions'])
        self.assertEqual(config['trajectory']['sampling']['terminal_views'], 8)
        self.assertNotIn('sampling', protocol(parse_arguments([]))['trajectory'])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_arguments(base + ['--corruption', 'background'])

    def test_sampling_analyzer_rejects_indices_and_a_logit_tampering(self):
        from scripts.analyze_trajectory_sampling import validate_sampling_arrays
        logits = np.zeros((1, 40))
        starts = view_starts(0, [7], 2048, 1)
        indices = np.stack((np.arange(1024), np.arange(1024)), axis=0)[None]
        indices[0, 1, 0] = starts[0, 0]
        values = dict(sampling_logits=np.zeros((1, 2, 40)), sampling_indices=indices,
                      sampling_starts=np.array([[0, starts[0, 0]]]),
                      sampling_cloud_size=np.array(2048))
        validate_sampling_arrays(values, logits, [7], 0, 2)
        values['sampling_logits'][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'original A'):
            validate_sampling_arrays(values, logits, [7], 0, 2)
        values['sampling_logits'][0, 0, 0] = 0
        values['sampling_indices'][0, 1, 1] = 2048
        with self.assertRaisesRegex(ValueError, 'indices'):
            validate_sampling_arrays(values, logits, [7], 0, 2)


if __name__ == '__main__':
    unittest.main()
