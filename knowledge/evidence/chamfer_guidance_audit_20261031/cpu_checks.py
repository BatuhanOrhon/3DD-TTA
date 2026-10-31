"""Audit historical math on CPU; no model/native/GPU imports or production edits."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HISTORY = ROOT / 'knowledge/evidence/chamfer_guidance_executed_code_20261031'


def functions(path, names):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    scope = {'torch': torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def main():
    torch.set_num_threads(2)
    records = json.loads((HISTORY / 'manifest.json').read_text())['files']
    for record in records:
        raw = (HISTORY / record['path']).read_bytes()
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    results = {'manifest_files_verified': len(records), 'torch': torch.__version__, 'device': 'cpu'}
    for revision in ('original_0fd2581', 'curvenet_95a4997'):
        scope = functions(HISTORY / revision / 'semantic_trajectory.py', ['semantic_loss'])
        loss = scope['semantic_loss']
        p = torch.tensor([[1., -.5, .1]], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[-.2, .4, .3]], dtype=torch.float64)
        for kind in ('kl', 'marginal_entropy'):
            value = loss(p, q, kind)
            gradient, = torch.autograd.grad(value, p)
            assert value.item() > 0 and torch.isfinite(gradient).all() and gradient.norm() > 0
            assert torch.autograd.gradcheck(lambda x: loss(x, q, kind), (p,))
            assert torch.allclose(loss(p.repeat(2, 1), q.repeat(2, 1), kind), 2 * value)
            duplicate = p.detach().repeat(2, 1).requires_grad_()
            dg, = torch.autograd.grad(loss(duplicate, q.repeat(2, 1), kind), duplicate)
            assert torch.allclose(dg, gradient.repeat(2, 1))
            assert loss(p.detach() - 1e-3 * gradient, q, kind) < value
        # A confident wrong fixed-X reference can reduce the true class probability.
        wrong_q = torch.tensor([[5., -5.]], dtype=torch.float64)
        decoded = torch.tensor([[-.2, .2]], dtype=torch.float64, requires_grad=True)
        gradient, = torch.autograd.grad(loss(decoded, wrong_q, 'marginal_entropy'), decoded)
        before = decoded.softmax(-1)[0, 1].item()
        after = (decoded - .1 * gradient).softmax(-1)[0, 1].item()
        assert after < before
        results[revision] = {'semantic_gradcheck': ['kl', 'marginal_entropy'],
                             'sum_loss_and_per_object_gradient_batch_scaling': 'passed',
                             'wrong_anchor_true_class_probability': [before, after]}
    math = functions(HISTORY / 'original_0fd2581/chamfer_guidance_variants.py',
                     ['guidance_loop_counts', 'reference_density_weights', 'weighted_selective_chamfer'])
    assert math['guidance_loop_counts'](10, 3) == (1, 4, 7, 10)
    assert math['guidance_loop_counts'](30, 3) == (*range(1, 31, 3), 30)
    # Unequal point counts and nonidentity NN mapping; trim remains distance-based.
    d1 = torch.tensor([[9., 1., 4.]], requires_grad=True)
    d2 = torch.tensor([[2., 6., 3., 8.]], requires_grad=True)
    nn = torch.tensor([[2, 0, 3]], dtype=torch.int32)
    weights = torch.tensor([[2., 4., 6., 8.]])
    weighted = math['weighted_selective_chamfer'](d1, d2, nn, weights, .75)
    assert weighted.item() == 2 + 32 + 4 + 24 + 18
    g1, g2 = torch.autograd.grad(weighted, (d1, d2))
    assert torch.equal(g1, torch.tensor([[0., 2., 8.]]))
    assert torch.equal(g2, torch.tensor([[2., 4., 6., 0.]]))
    equal = math['weighted_selective_chamfer'](d1, d2, nn, torch.ones_like(weights), .75)
    assert equal == d1.sort().values[:, :2].sum() + d2.sort().values[:, :3].sum()
    points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]]], requires_grad=True)
    density = math['reference_density_weights'](points, 1)
    assert not density.requires_grad and torch.isclose(density.mean(), torch.tensor(1.))
    assert density[0, 0] > density[0, 2]
    report = json.loads((ROOT / 'knowledge/evidence/chamfer_author_code_audit_20261030/frozen_calibration_report_20261030.json').read_text())
    results['calibrated_ratio_tail'] = {
        name: {stat: report['ratios'][name][stat] * report['coefficients'][name] for stat in ('median', 'p90', 'max')}
        for name in ('semantic_kl', 'semantic_h')}
    results['weighted_NN_trim_gradient_and_unit_weight_control'] = 'passed'
    results['sparse_forward_schedule'] = {'10': 4, '30': 11}
    results['limitations'] = 'Algebra/toy gradients only; no pretrained inference, native CUDA or accuracy causality.'
    output = Path(sys.argv[1])
    with output.open('x') as stream:
        json.dump(results, stream, indent=2)
        stream.write('\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
