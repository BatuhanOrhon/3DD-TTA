"""Synthetic CPU math checks. No native CUDA kernels or trained models run."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, replacements):
    spec = importlib.util.spec_from_file_location('_test_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


def chamfer(left, right):
    distances = (left-right).square().sum(-1)
    return distances, distances, None, None


class Scheduler:
    def __init__(self, **options):
        self.options = options

    def set_timesteps(self, total, device):
        self.timesteps = torch.arange(total-1, -1, -1, device=device)
        self.alphas_cumprod = torch.full((total,), .8, device=device)

    def step(self, noise, t, latent):
        return types.SimpleNamespace(pred_original_sample=latent-noise, prev_sample=latent-.1*noise)


class VAE(torch.nn.Module):
    def encode(self, points):
        return None, None, [[torch.ones(len(points), 8)],
                            [torch.linspace(-.2, .2, 8192).repeat(len(points), 1)]]

    def global2style(self, style):
        return style*2

    def decoder(self, unused, beta, context, style):
        self.final_context = context.detach().clone()
        self.final_style = style.detach().clone()
        return context.view(-1, 2048, 4)[:, :, :3]


class Prior(torch.nn.Module):
    def forward(self, x, t, condition_input, clip_feat):
        self.local = x.detach().clone()
        self.style = condition_input.detach().clone()
        return .1*x + .01*condition_input.mean(dim=1, keepdim=True)


class NumericsTests(unittest.TestCase):
    def test_fixed_source_changes_only_isolation_mask(self):
        original = (ROOT / 'graph_spectral_v2.py').read_text(encoding='utf-8')
        fixed = (ROOT / 'graph_spectral_v2_offdiag.py').read_text(encoding='utf-8')
        start = fixed.index('        # Minimal isolate correction:')
        end = fixed.index('\n', fixed.index('        isolated_nodes =', start))
        old_line = next(line for line in original.splitlines() if 'isolated_nodes =' in line)
        self.assertEqual(fixed[:start] + old_line + fixed[end:], original)

    def test_nonisolated_graph_is_unchanged(self):
        knn_module = types.ModuleType('knn_cuda')
        knn_module.KNN = lambda k, **kwargs: lambda a, b: torch.cdist(a, b).topk(k, dim=-1, largest=False)
        old = load_source('graph_spectral_v2', {'knn_cuda': knn_module})
        fixed = load_source('graph_spectral_v2_offdiag', {'knn_cuda': knn_module})
        points = torch.tensor([[[0., 0., 0., 0.], [.01, 0., 0., 0.]]])
        before = old.GraphSpectralDNA(k=2)(points)
        after = fixed.GraphSpectralDNA(k=2)(points)
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_offdiag_fix_penalizes_self_only_vertices(self):
        knn_module = types.ModuleType('knn_cuda')
        knn_module.KNN = lambda k, **kwargs: lambda a, b: torch.cdist(a, b).topk(k, dim=-1, largest=False)
        old = load_source('graph_spectral_v2', {'knn_cuda': knn_module})
        fixed = load_source('graph_spectral_v2_offdiag', {'knn_cuda': knn_module})
        points = torch.tensor([[[0.,0.,0.,0.],[.1,0.,0.,0.],[10.,0.,0.,0.],[20.,0.,0.,0.]]])
        old_values = old.GraphSpectralDNA(k=2)(points)[2]
        new_values = fixed.GraphSpectralDNA(k=2)(points)[2]
        self.assertEqual(int((old_values > 999).sum()), 0)
        self.assertEqual(int((new_values > 999).sum()), 2)

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_graph_uses_symmetric_mean_cutoff_penalty_and_raw_spectrum(self):
        distances = torch.tensor([[[0., .1], [0., .1], [0., 2.], [0., 2.]]])
        indices = torch.tensor([[[0, 1], [1, 0], [2, 3], [3, 2]]])
        knn_module = types.ModuleType('knn_cuda')
        knn_module.KNN = lambda **kwargs: lambda left, right: (distances, indices)
        module = load_source('graph_spectral_v2', {'knn_cuda': knn_module})
        graph = module.GraphSpectralDNA(k=2, delta=.1, gamma=1., device='cpu')
        points = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        gft, basis, eigenvalues = graph(points)
        # Self-query includes self edges; first pair survives, second pair is filtered.
        w = torch.exp(torch.tensor(-.5))
        expected = torch.tensor([[w, -w, 0, 0], [-w, w, 0, 0],
                                 [0, 0, 1000, 0], [0, 0, 0, 1000]]) + torch.eye(4)*1e-5
        recovered = basis @ torch.diag_embed(eigenvalues) @ basis.transpose(1, 2)
        torch.testing.assert_close(recovered[0], expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(gft, basis.transpose(1, 2) @ points[:, :, :3])
        self.assertGreater(eigenvalues.max().item(), 999)
        self.assertFalse(basis.requires_grad)

    def test_one_step_matches_explicit_loss_and_both_update_rates(self):
        chamfer_module = types.ModuleType('third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D')
        chamfer_module.chamfer_3DDist = lambda: chamfer
        diffusers = types.ModuleType('diffusers')
        schedulers = []
        def factory(**kwargs):
            scheduler = Scheduler(**kwargs)
            schedulers.append(scheduler)
            return scheduler
        diffusers.DDIMScheduler = factory
        utilities = types.ModuleType('utilities_3dd_tta')
        utilities.grad_freeze = lambda module: module.requires_grad_(False)
        module = load_source('tta_gsd_v2', {chamfer_module.__name__: chamfer_module,
                                          'diffusers': diffusers, 'utilities_3dd_tta': utilities})
        vae, prior = VAE(), Prior()
        lion = types.SimpleNamespace(vae=vae, priors=[None, prior])
        basis = torch.eye(2048).unsqueeze(0)
        eigenvalues = torch.linspace(0, 4, 2048).unsqueeze(0)
        graph = types.SimpleNamespace(use_4d_gft=False)
        class Graph:
            use_4d_gft = False
            def __call__(self, local):
                self.original = local.detach().clone()
                return local[:, :, :3].detach(), basis, eigenvalues
        graph = Graph()
        torch.manual_seed(12)
        _, metrics = module.tta_gsd_reconstruct(
            torch.zeros(1, 2048, 3), lion, graph, 1, .03, .02, .95,
            loss_weights={'spectral': 1.17, 'chamfer': 1.}, beta=2., M_max=800)
        local = prior.local.clone().requires_grad_(True)
        style = prior.style.clone().requires_grad_(True)
        noise = .1*local + .01*style.mean(dim=1, keepdim=True)
        predicted = (local-noise).view(1, 2048, 4)[:, :, :3]
        weights = torch.exp(-2*eigenvalues)
        weights[:, 800:] = 0
        residual = predicted-graph.original[:, :, :3]
        spectral = (weights.unsqueeze(-1)*residual.square()).sum()/(3*2048)
        distances = residual.square().sum(-1).sort(dim=1).values[:, :int(.95*2048)]
        scd = 2*distances.sum()
        gl, gs = torch.autograd.grad(1.17*spectral+scd, (local, style))
        expected_local = local-.1*noise-.02*gl  # eta controls local in legacy
        expected_style = style-.03*gs  # gamma controls style in legacy
        torch.testing.assert_close(vae.final_context, expected_local.squeeze(3).squeeze(2))
        torch.testing.assert_close(vae.final_style, expected_style.squeeze(3).squeeze(2))
        self.assertAlmostEqual(metrics['mean_raw_spectral'], spectral.item(), places=6)
        self.assertAlmostEqual(metrics['mean_raw_chamfer'], scd.item(), places=3)
        self.assertFalse(schedulers[0].options['set_alpha_to_one'])
        self.assertGreater(float(gs.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
