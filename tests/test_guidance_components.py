"""Same-state component controls; fake native kernels, no model/GPU evidence."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy_trajectory import _rng_state, rng_hash
from scripts import eval_multibackbone_guidance as runner
from tests.test_legacy_gsd_numerics import load_source, chamfer, Scheduler, VAE, Prior


class ComponentTests(unittest.TestCase):
    def arguments(self, *extra):
        return ['--preview', '--dataset', 'modelnet40_c', '--dataset-root', '/data',
                '--label-path', '/data/labels.npy', '--asset-root', '/assets', '--result-root', '/results', *extra]

    def test_component_run_id_identifies_owner_and_arm_before_launch(self):
        for extra, expected in [(('--component-arm', 'chamfer', '--trajectory-observe'), 'guidance-components-chamfer-curvenet-seed0'),
                                ((), 'curvenet-seed0')]:
            arguments = self.arguments(*extra)[1:] + ['--asset-manifest-sha256', 'a' * 64]
            with patch.object(runner.source.RunBundle, 'create', side_effect=RuntimeError('creation captured')) as create:
                with self.assertRaisesRegex(RuntimeError, 'creation captured'):
                    runner.main(arguments)
            self.assertEqual(create.call_args.args[1], expected)

    def test_explicit_arms_and_selection_do_not_mutate_default_protocol(self):
        original = json.dumps(runner.PROTOCOL, sort_keys=True)
        for arm, (chamfer_weight, spectral_weight) in runner.COMPONENT_WEIGHTS.items():
            configs = runner.configurations(self.arguments('--component-arm', arm, '--trajectory-observe'))
            self.assertEqual(len(configs), 2)
            for config in configs:
                weights = config['guidance_protocol']['loss_weights']
                self.assertEqual((weights['chamfer'], weights['spectral'], weights['invariant']), (chamfer_weight, spectral_weight, 0.))
                self.assertTrue(config['guidance_protocol']['skip_inactive_components'])
                self.assertEqual(config['num_classes'], 40)
        self.assertEqual(json.dumps(runner.PROTOCOL, sort_keys=True), original)
        self.assertNotIn('skip_inactive_components', runner.configurations(self.arguments())[0]['guidance_protocol'])
        for extra in [('--component-arm', 'combined'), ('--eta', '.02'),
                      ('--component-arm', 'chamfer', '--trajectory-observe', '--spectral-weight', '1'),
                      ('--component-arm', 'combined', '--trajectory-observe', '--p', 'nan')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                runner.configurations(self.arguments(*extra))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'ids.json'
            selection = dict(dataset='modelnet40_c', n_examples=2468, indices=[3, 90, 1000], stage='development')
            path.write_text(json.dumps(selection))
            configs = runner.configurations(self.arguments('--component-arm', 'combined', '--trajectory-observe', '--index-manifest', str(path), '--eta', '.02'))
            self.assertEqual(configs[0]['selected_original_ids'], [3, 90, 1000])
            self.assertEqual(configs[0]['guidance_protocol']['eta'], .02)
            self.assertEqual(configs[0]['selection_manifest_identity'], runner.source.file_identity(path))
            for ids in ([3, 3], [90, 3], [True], [2468], []):
                path.write_text(json.dumps(dict(selection, indices=ids)))
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    runner.configurations(self.arguments('--index-manifest', str(path)))

    def reconstruction(self):
        chamfer_module = types.ModuleType('third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D')
        chamfer_module.chamfer_3DDist = lambda: chamfer
        utilities = types.ModuleType('utilities_3dd_tta')
        utilities.grad_freeze = lambda model: model.requires_grad_(False)
        return load_source('tta_gsd_v2', {chamfer_module.__name__: chamfer_module,
                           'diffusers': types.SimpleNamespace(DDIMScheduler=Scheduler), 'utilities_3dd_tta': utilities})

    def test_inactive_skip_matches_reference_and_calls_only_active_components(self):
        module = self.reconstruction()
        basis = torch.eye(2048).unsqueeze(0)
        eigenvalues = torch.linspace(0, 4, 2048).unsqueeze(0)
        class Graph:
            use_4d_gft = False
            calls = 0
            def __call__(self, local):
                self.calls += 1
                return local[:, :, :3].detach(), basis, eigenvalues
        for arm, (cw, sw) in runner.COMPONENT_WEIGHTS.items():
            results = []
            for skip in (False, True):
                vae, prior = VAE(), Prior()
                graph = Graph()
                torch.manual_seed(12)
                with patch.object(module, 'chamfer_grad', wraps=module.chamfer_grad) as constructor:
                    points, metrics = module.tta_gsd_reconstruct(torch.zeros(1, 2048, 3), types.SimpleNamespace(vae=vae, priors=[None, prior]),
                        graph, 1, .03, .02, .95, loss_weights=dict(spectral=sw, chamfer=cw), M_max=800,
                        skip_inactive_components=skip)
                self.assertEqual(graph.calls, int(sw > 0 or not skip))
                self.assertEqual(constructor.call_count, int(cw > 0))
                results.append((points.detach(), vae.final_style, rng_hash()))
            for left, right in zip(results[0][:2], results[1][:2]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(results[0][2], results[1][2])
            self.assertEqual(metrics['graph_calls'], int(sw > 0))
            self.assertEqual(metrics['chamfer_calls'], 2 * int(cw > 0))
            self.assertEqual(metrics['guidance_backward_calls'], int(cw > 0 or sw > 0))
            if arm == 'unguided':
                self.assertEqual(metrics['mean_local_update_norm'], 0.)
                self.assertEqual(metrics['mean_style_update_norm'], 0.)
            else:
                self.assertGreater(metrics['mean_local_update_norm'], 0.)

    def test_cached_preparation_reuses_encode_noise_and_post_prepare_rng(self):
        module = self.reconstruction()
        class CountingVAE(VAE):
            calls = 0
            def encode(self, points):
                self.calls += 1
                torch.rand(3)
                return super().encode(points)
        prepared = torch.zeros(1, 2048, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / 'cache.npz'
            results = []
            for iteration in range(2):
                vae, prior = CountingVAE(), Prior()
                torch.manual_seed(74)
                state, recorder = runner.paired_preparation(path, prepared)
                points, metrics = module.tta_gsd_reconstruct(prepared, types.SimpleNamespace(vae=vae, priors=[None, prior]), None,
                    1, .01, .01, .95, loss_weights=dict(spectral=0., chamfer=0.),
                    skip_inactive_components=True, prepared_state=state, state_observer=recorder)
                self.assertEqual(vae.calls, 1 if iteration == 0 else 0)
                results.append((points.detach(), rng_hash(), runner.source.file_identity(path)))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            self.assertEqual(results[0][1:], results[1][1:])
            torch.manual_seed(74)
            with self.assertRaisesRegex(ValueError, 'identity'):
                runner.paired_preparation(path, prepared + 1)
            torch.manual_seed(75)
            with self.assertRaisesRegex(ValueError, 'identity'):
                runner.paired_preparation(path, prepared)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays['local_latent'][0, 0] += 1
            np.savez_compressed(path, **arrays)
            torch.manual_seed(74)
            with self.assertRaisesRegex(ValueError, 'tensor hash'):
                runner.paired_preparation(path, prepared)


if __name__ == '__main__':
    unittest.main()
