"""CPU algebra and static historical contracts; no trained model execution."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(20261031)
    zc = torch.randn(3, 128, generator=generator, dtype=torch.float64)
    proposal = torch.randn(3, 128, generator=generator, dtype=torch.float64)
    lam = 3.
    z = ((proposal + lam * zc) / (1 + lam)).requires_grad_()
    objective = .5 * (z - proposal).square().sum() + .5 * lam * (z - zc).square().sum()
    gradient, = torch.autograd.grad(objective, z)
    assert gradient.norm() < 1e-12
    assert torch.allclose(z - zc, (proposal - zc) / (1 + lam))
    radius = .1
    delta = z.detach() - zc
    bounded = zc + delta * (radius / delta.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    assert torch.all((bounded - zc).norm(dim=1) <= radius + 1e-12)
    assert torch.equal(zc + delta * 0, zc)
    # A raw noisy state is not the clean z that the decoder/local prior expect.
    noise = torch.randn(zc.shape, generator=generator, dtype=zc.dtype)
    alpha = .7
    noisy = alpha ** .5 * zc + (1 - alpha) ** .5 * noise
    predicted_clean = (noisy - (1 - alpha) ** .5 * noise) / alpha ** .5
    assert torch.allclose(predicted_clean, zc)
    assert (noisy - zc).square().sum() > 0
    # Hierarchical pair consistency counterexample; not a model of LION accuracy.
    x = torch.tensor([2.], dtype=torch.float64)
    old_z, new_z = torch.tensor([.5]), torch.tensor([1.])
    old_h = x - old_z
    stale_pair_output = old_h + new_z
    repaired_pair_output = (x - new_z) + new_z
    assert stale_pair_output != x and repaired_pair_output == x
    # Isolating extra global noise must preserve the existing local RNG stream.
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    torch.randn((3, 128), generator=generator)
    assert torch.equal(before, torch.get_rng_state())
    source = subprocess.check_output(['git', 'show', 'a458cd4:eval_gsd_tta_dual.py'], text=True)
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id in ('tta_gsd_reconstruct_seq', 'tta_gsd_reconstruct_sync')]
    assert len(calls) == 2
    assert all(next(k.value for k in call.keywords if k.arg == 'p').value == .5 for call in calls)
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'lambdaa' for n in ast.walk(tree))
    # The actual locked 1000-beta /100-step grid; alpha is independent of B.
    alpha_bar = np.cumprod(1 - np.linspace(.0001, .02, 1000))
    rows = []
    for count in (1, 3, 5, 10, 30):
        timestep = 10 * (count - 1)
        a = float(alpha_bar[timestep])
        rows.append(dict(reverse_steps=count, timestep=timestep, alpha_bar=a,
                         forward_noise_std=float(np.sqrt(1 - a)),
                         expected_rms_change_if_unit_variance=float(np.sqrt(2 * (1 - np.sqrt(a))))))
    result = dict(status='passed', device='cpu', tests=['quadratic_anchor_stationarity', 'anchor_shrinkage',
                  'per_object_radius_bound', 'zero_radius_identity', 'clean_vs_noisy_anchor',
                  'hierarchical_pair_counterexample', 'separate_global_noise_generator',
                  'historical_launcher_ignores_lambdaa_and_passes_p_half'], schedule=rows,
                  limits='Algebra/static source only. No assertion of LION denoiser quality, runtime, CUDA derivatives, or old run provenance.')
    with Path(sys.argv[1]).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
