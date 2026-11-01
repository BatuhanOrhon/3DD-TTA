"""CPU mathematical witnesses and descriptive replay of immutable latent caches."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def math_checks(loss):
    generator = torch.Generator().manual_seed(20261101)
    a = torch.rand(9, 9, dtype=torch.float64, generator=generator)
    a = (a + a.T) / 2
    a.fill_diagonal_(0)
    laplacian = torch.diag(a.sum(1)) - a
    eigenvalues, u = torch.linalg.eigh(laplacian)
    q = torch.randn(1, 9, 3, dtype=torch.float64, generator=generator)
    y = torch.randn(1, 9, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    eig = eigenvalues.unsqueeze(0)
    target = u.T @ q
    actual = loss(u.T @ y, target, eig, 2., 5)
    w = torch.exp(-2 * eigenvalues)
    w[5:] = 0
    operator = (u * w) @ u.T
    expected = ((y - q) * (operator @ (y - q))).sum() / 27
    gradient, = torch.autograd.grad(actual, y)
    expected_gradient = 2 * (operator @ (y - q)) / 27
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(lambda value: loss(u.T @ value, target, eig, 2., 5), (y,))
    torch.testing.assert_close(u @ target, q, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(target.square().sum(), q.square().sum())
    repeated = y.detach().repeat(2, 1, 1).requires_grad_()
    repeated_gradient, = torch.autograd.grad(loss(u.T @ repeated, target.repeat(2, 1, 1), eig.repeat(2, 1), 2., 5), repeated)
    torch.testing.assert_close(repeated_gradient[0], gradient[0])
    signs = torch.tensor([1., -1.] * 4 + [1.], dtype=torch.float64)
    signed = u * signs
    torch.testing.assert_close(loss(signed.T @ y, signed.T @ q, eig, 2., 5), actual)
    permutation = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8])
    permuted = q[:, permutation]
    distance = torch.cdist(permuted, q).square()
    chamfer = distance.min(-1).values.sum() + distance.min(-2).values.sum()
    permutation_loss = loss(u.T @ permuted, target, eig, 2., 9)
    assert float(chamfer) < 1e-12 and float(permutation_loss) > 0
    clean = q + .3
    clean_loss = loss(u.T @ clean, target, eig, 2., 9)
    translation_expected = float(torch.tensor(.3, dtype=torch.float64).square() * torch.exp(-2 * eigenvalues[0]))
    assert abs(float(clean_loss) - translation_expected) < 1e-12
    residual = torch.tensor([[[1., 0., 0.], [0., 0., 0.]]], dtype=torch.float64)
    zero = torch.zeros_like(residual)
    tied = torch.zeros(1, 2, dtype=torch.float64)
    cutoff_before = loss(residual, zero, tied, 2., 1)
    cutoff_after = loss(residual.flip(1), zero, tied, 2., 1)
    assert float(cutoff_before) > 0 and float(cutoff_after) == 0
    return dict(roundtrip_max_error=float((u @ target - q).abs().max()),
                gradient_max_error=float((gradient - expected_gradient).abs().max().detach()),
                finite_difference=True, batch_per_sample_gradient_invariance=True,
                shared_basis_sign_invariance=True, permutation_chamfer=float(chamfer),
                permutation_spectral=float(permutation_loss), translation_spectral=float(clean_loss),
                tied_eigenspace_cutoff_losses=[float(cutoff_before), float(cutoff_after)],
                meaning='Correct GFT/loss can penalize set-equivalent point permutations and coherent translations; no accuracy causality.')


def outcome_summaries(archive_root):
    prep = archive_root / 'result/preparation'
    phases = [('fixed', prep / 'guidance_component_study_20261025/local_frozen_confirmation_analysis/case_policy.csv', 'combined', 'chamfer'),
              ('calibrated', prep / 'guidance_component_sensitivity_20261025/recovery_20261028_r4/local_partial_confirmation_analysis_v1/case_policy.csv', 'spectral-candidate', 'matched-chamfer')]
    results = []
    for phase, path, cs, c in phases:
        rows = list(csv.DictReader(path.open()))
        for dataset in ('modelnet40_c', 'scanobjectnn_c'):
            for backbone in ('all_four', 'pointmae', 'dgcnn', 'curvenet', 'pointnext'):
                def mean(arm):
                    selected = [r for r in rows if r['dataset'] == dataset and r['policy'] == 'probability_mean'
                                and r.get('condition_id', r.get('component_arm')) == arm
                                and (backbone == 'all_four' or r['backbone'] == backbone)]
                    assert len(selected) == (12 if backbone == 'all_four' else 3)
                    return sum(float(r['accuracy_pct']) for r in selected) / len(selected)
                results.append(dict(phase=phase, dataset=dataset, backbone=backbone,
                                    c_percent=mean(c), cs_percent=mean(cs), cs_minus_c_pp=mean(cs) - mean(c),
                                    csv_path=str(path.relative_to(archive_root)), csv_sha256=sha(path)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    helper = load(args.repo / 'spectral_guidance_losses.py', 'audited_spectral_loss')
    captured = {}

    class CpuKNN:
        def __init__(self, k, **kwargs):
            self.k = k

        def __call__(self, left, right):
            distances = torch.cdist(left, right, compute_mode='donot_use_mm_for_euclid_dist')
            values, indices = distances.topk(self.k, largest=False)
            captured.update(values=values, indices=indices)
            return values, indices

    knn = types.ModuleType('knn_cuda')
    knn.KNN = CpuKNN
    graph_path = args.repo / 'graph_spectral_v2.py'
    assert sha(graph_path) == '1cadd63642dc58bb0aa22969c08e00e4c9dd597696cf53107dbe081d3c68807e'
    with patch.dict(sys.modules, {'knn_cuda': knn}):
        graph = load(graph_path, 'audited_legacy_graph')
    report = dict(scope=__doc__, torch=torch.__version__, numpy=np.__version__,
                  graph_sha256=sha(graph_path), helper_sha256=sha(args.repo / 'spectral_guidance_losses.py'),
                  mathematical_witnesses=math_checks(helper.spectral_guidance_loss),
                  outcome_summaries=outcome_summaries(args.archive_root), graph_probes=[])
    for row in report['outcome_summaries']:
        assert sha(args.repo / row['csv_path']) == row['csv_sha256']
    object_ids = {}
    development = args.archive_root / 'result/preparation/guidance_component_study_20261025/development_extracted'
    for config_path in development.rglob('config.json'):
        config = json.loads(config_path.read_text())
        if config.get('component_study') and config.get('selected_original_ids'):
            dataset, ids = config['dataset'], config['selected_original_ids']
            assert len(ids) == 70 and (dataset not in object_ids or object_ids[dataset] == ids)
            object_ids[dataset] = ids
    assert set(object_ids) == {'modelnet40_c', 'scanobjectnn_c'}
    cache_root = args.archive_root / 'result/preparation/guidance_component_study_20261025/paired_cache_extracted/paired_state_v1_seed0'
    for dataset in ('modelnet40_c', 'scanobjectnn_c'):
        for directory in sorted((cache_root / dataset).iterdir()):
            path, = directory.glob('*.npz')
            with np.load(path, allow_pickle=False) as cache:
                h = torch.from_numpy(cache['local_latent'][:1].copy()).view(1, 2048, 4)
            for scale in ([1., .5, 2.] if directory.name == 'gaussian' else [1.]):
                with torch.no_grad():
                    coefficients, basis, eigenvalues = graph.GraphSpectralDNA(device='cpu')(h * scale)
                adjacency = torch.zeros(1, 2048, 2048)
                adjacency.scatter_(2, captured['indices'], torch.exp(-captured['values'].square() / .02))
                adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
                degree = adjacency.sum(-1)
                threshold = .6 * degree.mean(-1, keepdim=True)
                retained = degree > threshold
                adjacency *= retained.unsqueeze(1) * retained.unsqueeze(2)
                degree = adjacency.sum(-1)
                nonself_edges = adjacency.count_nonzero(dim=-1) - (adjacency.diagonal(dim1=1, dim2=2) != 0).long()
                missed = (degree > 0) & (nonself_edges == 0)
                weights = torch.exp(-2 * eigenvalues)
                weights[:, 800:] = 0
                reconstructed = basis @ coefficients
                roundtrip = float((reconstructed - (h * scale)[:, :, :3]).abs().max())
                assert roundtrip < 2e-4 and bool(torch.isfinite(coefficients).all())
                report['graph_probes'].append(dict(dataset=dataset, corruption=directory.name,
                    position=0, original_id=object_ids[dataset][0], coordinate_scale=scale, cache_path=str(path.relative_to(args.archive_root)),
                    cache_sha256=sha(path), spectral_weight_mass=float(weights.sum()),
                    kept800_energy_fraction=float(coefficients[:, :800].square().sum() / coefficients.square().sum()),
                    missed_self_only_nodes=int(missed.sum()), penalized_zero_row_nodes=int((degree == 0).sum()),
                    eigenvalue_799=float(eigenvalues[0, 799]), eigenvalue_800=float(eigenvalues[0, 800]),
                    roundtrip_max_error=roundtrip, threshold=float(threshold),
                    latent_xyz_rms=float((h * scale)[:, :, :3].square().mean().sqrt())))
                print(dataset, directory.name, scale, roundtrip, float(weights.sum()), flush=True)
    assert len(report['graph_probes']) == 34 and not torch.cuda.is_initialized()
    report['limitations'] = ('One original development object per dataset, repeated across15 corruptions:2 objects,30 cells;4 additional coordinate-scale counterfactuals. '
        'CPU Euclidean kNN substitutes for CUDA; no native tie/numerical parity, prevalence, clean-reference or accuracy-causal claim. '
        'Outcome summaries reuse accepted CSVs; raw prediction acceptance remains the existing17-ZIP recount.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
