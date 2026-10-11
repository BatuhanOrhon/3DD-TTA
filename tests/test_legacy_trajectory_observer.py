"""Synthetic CPU checks only: no checkpoints, native kernels or GPU inference."""
import random
import json
import hashlib
from contextlib import redirect_stderr
from io import StringIO
import unittest
from pathlib import Path
import numpy as np
import torch

from tests.test_legacy_gsd_restore import workspace_temp


class ObserverTests(unittest.TestCase):
    def test_smoke_validation_keys_are_json_serializable(self):
        from scripts.legacy_trajectory_colab import smoke_arm_key
        arms = {smoke_arm_key(corruption, mode): {'mode': mode}
                for corruption in ('gaussian', 'background')
                for mode in ('off', 'audit', 'observe')}
        encoded = json.dumps(arms)
        self.assertEqual(set(json.loads(encoded)), {
            'gaussian_off', 'gaussian_audit', 'gaussian_observe',
            'background_off', 'background_audit', 'background_observe'})

    def test_git_sources_reject_edits_but_allow_unrelated_build_outputs(self):
        from unittest.mock import patch
        from scripts.legacy_trajectory_colab import verify_sources, BASE_SHA, PROVENANCE_PATH
        with workspace_temp() as root:
            spec = dict(schema_version=1, branch='gsd-smooth-spectrum',
                        baseline_archive_sha256=BASE_SHA, source_paths=['model.py'], baseline_assets={})
            blobs = {PROVENANCE_PATH: json.dumps(spec).encode(), 'model.py': b'x = 1\n'}
            for name, data in blobs.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data.replace(b'\n', b'\r\n'))
            (root / 'unrelated_build.py').write_text('generated')

            def git(cmd, cwd):
                self.assertEqual(cwd, root)
                if cmd[1:] == ['rev-parse', 'HEAD']:
                    return b'abc123\n'
                if cmd[1:] == ['branch', '--show-current']:
                    return b'gsd-smooth-spectrum\n'
                self.assertEqual(cmd[1], 'show')
                self.assertTrue(cmd[2].startswith('abc123:'))
                return blobs[cmd[2].split(':', 1)[1]]

            with patch('scripts.legacy_trajectory_colab.REPO', root), \
                    patch('scripts.legacy_trajectory_colab.subprocess.check_output', side_effect=git):
                manifest = verify_sources()
                self.assertEqual(manifest['resolved_commit'], 'abc123')
                self.assertEqual(set(manifest['files']), set(blobs))
                (root / 'model.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'differs from Git commit: model.py'):
                    verify_sources()
                (root / 'model.py').write_bytes(blobs['model.py'])
                (root / PROVENANCE_PATH).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'differs from Git commit: cfgs/'):
                    verify_sources()

    def test_saved_trace_counts_policy_analysis_and_corruption_rejected(self):
        from legacy_trajectory import TrajectoryObserver, trajectory_protocol
        from eval_gsd_tta_v2 import parse_arguments
        from legacy_gsd_artifacts import protocol
        from scripts.analyze_legacy_trajectory import Bundle, analyze, load_trace
        class VAE(torch.nn.Module):
            def decoder(self, unused, beta, context, style):
                return context.view(2, 2, 3)
        class Classifier(torch.nn.Module):
            def classification_only(self, points, only_unmasked):
                logits = torch.zeros(2, 40)
                logits[0, 1] = 10
                logits[1, 2] = 5
                return logits
        with workspace_temp() as root:
            run = root / 'trace'; run.mkdir()
            args = parse_arguments(['--trajectory_mode', 'observe', '--seed', '0', '--corruption', 'gaussian',
                                    '--batch_size', '2', '--max_batches', '1', '--M_max', '800'])
            config = protocol(args)
            config.update(execution_status='complete', completed_examples=2, source_manifest={}, assets={},
                          model_state_before={}, model_state_after={}, model_state_unchanged=True,
                          native_extensions={}, external_python_sources={}, model_modes={})
            observer = TrajectoryObserver(run, 'gaussian', 0, [0, 1], 10, trajectory_protocol(args))
            observer.begin(2, torch.device('cpu'))
            observer.bind(VAE().eval(), Classifier().eval(), lambda p: p)
            for count in (3, 5, 8):
                observer.clean(torch.zeros(2, 6, 1, 1), torch.zeros(2, 3, 1, 1), count-1, (10-count)*10)
            observer.final_state(torch.ones(2, 6, 1, 1), torch.ones(2, 3, 1, 1), 0)
            terminal = torch.zeros(2, 40); terminal[0, 0] = 2; terminal[1, 2] = 2
            observer.terminal(terminal, torch.tensor([0, 2]))
            (run / 'predictions').mkdir()
            np.savez_compressed(run / 'predictions/gaussian_0000.npz', logits=terminal.numpy(),
                                predictions=np.array([0, 2]), targets=np.array([0, 2]), indices=np.array([0, 1]))
            config['trajectory_artifact_manifest'] = {
                p.relative_to(run).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in run.rglob('*') if p.is_file()}
            (run / 'config.json').write_text(json.dumps(config))
            (run / 'environment.txt').write_text('{}')
            report = analyze([run], root / 'analysis')
            self.assertTrue(all(row['sample_seed_sd'] is None for row in report['summary']))
            diag = report['diagnostics'][0]
            self.assertEqual(diag['n'], 2)
            self.assertEqual(diag['minimum_entropy_wrong_count'], 1)
            self.assertEqual(diag['label_using_oracle_any_correct'], 1.)
            self.assertEqual(diag['any_temporal_disagreement'], .5)
            self.assertEqual(diag['selected_histogram']['clean_03'], 2)
            # The completed flag cannot conceal missing checkpoints.
            (run / 'trajectory/gaussian/0000_clean_05.npz').rename(run / 'trajectory/gaussian/missing.npz')
            bundle = Bundle(run)
            with self.assertRaises((ValueError, FileNotFoundError)):
                load_trace(bundle)
            bundle.close()

    def test_manifest_original_indices_and_preview_do_not_launch_gpu(self):
        from scripts.legacy_trajectory_colab import commands, initialize, CONDA
        from unittest.mock import patch
        from eval_gsd_tta_v2 import parse_arguments
        from legacy_gsd_artifacts import protocol
        from research_artifacts import CORRUPTIONS
        with workspace_temp() as root:
            with patch('scripts.legacy_trajectory_colab.verify_sources', return_value={}):
                initialize(root / 'session')
            session = json.loads((root / 'session/session.json').read_text())
            self.assertEqual(session['pilot_seeds'], [0])
            self.assertEqual(session['deferred_seeds'], [1, 2])
            actual_ids = json.loads((root / 'session/pilot_indices.json').read_text())['indices']
            self.assertEqual(sum(map(len, actual_ids.values())), 4200)
            manifest = root / 'pilot_indices.json'
            manifest.write_text(json.dumps(dict(schema_version=1, dataset='modelnet40_c',
                                               indices={c: [7, 3, 99] for c in CORRUPTIONS})))
            args = parse_arguments(['--indices_manifest', str(manifest), '--M_max', '800'])
            config = protocol(args)
            self.assertEqual(config['expected_classifications'], 45)
            self.assertEqual(config['selected_indices']['gaussian'], [7, 3, 99])
            smoke, pilot = list(commands(root, 'smoke')), list(commands(root, 'pilot'))
            self.assertEqual(len(smoke), 6); self.assertEqual(len(pilot), 1)
            for command in smoke + pilot:
                self.assertEqual(command[:len(CONDA)], CONDA)
                self.assertEqual(command[command.index('--seed') + 1], '0')
                parsed = parse_arguments(command[len(CONDA)+1:] + ['--preview']) if command in pilot else None
                if parsed:
                    self.assertEqual(parsed.M_max, 800)
                    self.assertEqual(parsed.trajectory_mode, 'observe')

    def test_real_legacy_loop_preserves_guidance_and_pre_update_style(self):
        from types import SimpleNamespace, ModuleType
        from tests.test_legacy_gsd_numerics import load_source, Scheduler, VAE, Prior, chamfer
        from legacy_trajectory import TrajectoryObserver, rng_hash
        native = ModuleType('third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D')
        native.chamfer_3DDist = lambda: chamfer
        diffusers, utilities = ModuleType('diffusers'), ModuleType('utilities_3dd_tta')
        diffusers.DDIMScheduler = Scheduler
        utilities.grad_freeze = lambda m: m.requires_grad_(False)
        module = load_source('tta_gsd_v2', {native.__name__: native, 'diffusers': diffusers,
                                          'utilities_3dd_tta': utilities})
        class RecordingVAE(VAE):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.register_buffer('constant', torch.ones(1))
            def decoder(self, unused, beta, context, style):
                self.calls.append((context.detach().clone(), style.detach().clone(), torch.is_grad_enabled()))
                return super().decoder(unused, beta, context, style)
        class Classifier(torch.nn.Module):
            def classification_only(self, points, only_unmasked):
                self.assertion = not only_unmasked
                random.random(); np.random.rand(); torch.rand(5)
                return points.mean((1, 2))[:, None].expand(-1, 40)
        class RecordingPrior(Prior):
            def __init__(self):
                super().__init__(); self.styles = []
            def forward(self, x, t, condition_input, clip_feat):
                self.styles.append(condition_input.detach().clone())
                return super().forward(x, t, condition_input, clip_feat)
        class Graph:
            use_4d_gft = False
            def __call__(self, local):
                return local[:, :, :3].detach(), torch.eye(2048)[None], torch.linspace(0, 4, 2048)[None]
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with workspace_temp() as root:
                results = []
                for mode in ('off', 'audit', 'observe'):
                    random.seed(42); np.random.seed(42); torch.manual_seed(42)
                    vae, prior, classifier = RecordingVAE().eval(), RecordingPrior().eval(), Classifier().eval()
                    observer = None
                    if mode != 'off':
                        observer = TrajectoryObserver(root / mode, 'gaussian', 42, [13], 10,
                            dict(mode=mode, checkpoints={'10': [3, 5, 8]}, geometry_samples_per_batch=1))
                        observer.begin(1, torch.device('cpu'))
                        observer.bind(vae, classifier, lambda x: x)
                    points, metrics = module.tta_gsd_reconstruct(torch.zeros(1, 2048, 3),
                        SimpleNamespace(vae=vae, priors=[None, prior]), Graph(), 10, .01, .01, .95,
                        loss_weights={'spectral': 1.17, 'chamfer': 1.}, M_max=800, trajectory_observer=observer)
                    results.append((points.detach().clone(), metrics, rng_hash(), prior.styles))
                    if mode == 'observe':
                        self.assertEqual(len(vae.calls), 4)
                        for call, step in zip(vae.calls[:-1], (3, 5, 8)):
                            torch.testing.assert_close(call[1], prior.styles[step-1].squeeze(3).squeeze(2), rtol=0, atol=0)
                            self.assertFalse(call[2])
                        self.assertFalse(torch.equal(vae.calls[-1][1], prior.styles[-1].squeeze(3).squeeze(2)))
                        self.assertTrue(classifier.assertion)
                    if observer:
                        observer.terminal(torch.zeros(1, 40), torch.tensor([2]))
                        if mode == 'audit':
                            record = json.loads((root / mode / 'trajectory' / 'gaussian' / '0000.json').read_text())
                            self.assertTrue(record['encode_replay']['exact_latent_match'])
                            self.assertTrue(record['encode_replay']['post_rng_restored'])
                            self.assertTrue(record['encode_replay']['buffers_unchanged'])
                        if mode == 'observe':
                            record = json.loads((root / mode / 'trajectory' / 'gaussian' / '0000.json').read_text())
                            for candidate in record['candidates']:
                                if candidate['kind'] == 'predicted_clean':
                                    self.assertEqual(candidate['hashes']['probe_rng_before'],
                                                     candidate['hashes']['probe_rng_after'])
                                    self.assertEqual(candidate['hashes']['probe_buffers_before'],
                                                     candidate['hashes']['probe_buffers_after'])
                for result in results[1:]:
                    torch.testing.assert_close(result[0], results[0][0], rtol=0, atol=0)
                    self.assertEqual(result[1:3], results[0][1:3])
                    for a, b in zip(result[3], results[0][3]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_probe_restores_rng_buffers_and_gradients_even_on_exception(self):
        from legacy_trajectory import isolated_probe, rng_hash
        model = torch.nn.BatchNorm1d(3).eval()
        model.weight.grad = torch.ones_like(model.weight)
        before = rng_hash()
        version = model.running_mean._version
        with isolated_probe([model]):
            random.random(); np.random.random(); torch.rand(2)
        self.assertEqual(before, rng_hash())
        self.assertEqual(version, model.running_mean._version)
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with isolated_probe([model]):
                model.running_mean.add_(1)
                torch.rand(3)
                raise RuntimeError('test failure')
        self.assertEqual(before, rng_hash())
        torch.testing.assert_close(model.running_mean, torch.zeros(3))
        torch.testing.assert_close(model.weight.grad, torch.ones(3))

    def test_probabilities_not_logits_are_averaged_and_ties_prefer_terminal(self):
        from scripts.analyze_legacy_trajectory import policies
        # One highly confident wrong probe, two mildly correct candidates.
        logits = np.array([[[0., 10.], [1., 0.], [1., 0.]],
                           [[2., 0.], [0., 2.], [2., 0.]]])
        result = policies(logits, terminal=2)
        self.assertEqual(result['terminal'].tolist(), [0, 0])
        self.assertEqual(result['minimum_entropy'].tolist(), [1, 0])
        self.assertEqual(result['probability_mean'].tolist(), [1, 0])
        self.assertEqual(result['selected_candidate'].tolist(), [0, 2])
        with self.assertRaises(ValueError):
            policies(np.full((1, 3, 2), np.nan), terminal=2)

    def test_preview_rejects_ambiguous_schedule_and_keeps_default_off(self):
        from eval_gsd_tta_v2 import parse_arguments
        from legacy_gsd_artifacts import protocol
        self.assertEqual(parse_arguments([]).trajectory_mode, 'off')
        args = parse_arguments(['--trajectory_mode', 'observe', '--seed', '0', '--M_max', '800'])
        self.assertEqual(protocol(args)['trajectory']['checkpoints'], {'10': [3, 5, 8], '30': [8, 15, 23]})
        for flags in (['--denoising_step', '20'], ['--isolate_policy', 'offdiag'],
                      ['--trajectory_steps_normal', '3,3,8']):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_arguments(['--trajectory_mode', 'observe', '--seed', '0', *flags])


if __name__ == '__main__':
    unittest.main()
