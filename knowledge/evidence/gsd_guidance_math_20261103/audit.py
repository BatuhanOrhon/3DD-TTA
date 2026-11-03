#!/usr/bin/env python3
"""Asserted CPU counterexamples and sparse heat-filter equivalence; no LION run."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch


def run():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261103)
    dtype = torch.float64
    n = 64
    # Two path components and one isolate; all operators are explicitly symmetric.
    adjacency = torch.zeros(n, n, dtype=dtype)
    for i in range(n - 2):
        if i != 30:
            adjacency[i, i + 1] = adjacency[i + 1, i] = .5 + torch.rand((), generator=generator, dtype=dtype)
    degree = adjacency.sum(1)
    active = degree > 0
    laplacian = (torch.diag(degree) - adjacency) / degree[active].mean()
    eigenvalues, basis = torch.linalg.eigh(laplacian)
    reference = torch.randn(n, 3, generator=generator, dtype=dtype)
    predicted = torch.randn(n, 3, generator=generator, dtype=dtype, requires_grad=True)
    residual = (predicted - reference) * active[:, None]
    beta = 2.
    weights = torch.exp(-beta * eigenvalues)
    spectral = (weights[:, None] * (basis.T @ residual).square()).sum() / (3 * n)
    heat = torch.matrix_exp(-beta * laplacian)
    spatial = (residual * (heat @ residual)).sum() / (3 * n)
    exact_gradient, = torch.autograd.grad(spatial, predicted, retain_graph=True)
    formula_gradient = 2 * active[:, None] * (heat @ residual) / (3 * n)
    torch.testing.assert_close(spectral, spatial, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(exact_gradient, formula_gradient, atol=1e-12, rtol=1e-12)
    roundtrip_error = (basis @ (basis.T @ predicted) - predicted).abs().max().item()
    assert roundtrip_error < 1e-12

    # Smooth square-root filter avoids a possibly indefinite polynomial metric.
    # Spectral bounds use Gershgorin (2*dmax), without computing eigenvalues.
    upper = 2 * laplacian.diag().max().item()
    scaled_sparse = (2 * laplacian / upper - torch.eye(n, dtype=dtype)).to_sparse_coo()
    quadrature = 256
    theta = torch.pi * (torch.arange(quadrature, dtype=dtype) + .5) / quadrature
    samples = torch.exp(-beta * upper * (torch.cos(theta) + 1) / 4)
    approximation = []
    for order in (8, 16, 32):
        coefficients = 2 / quadrature * (torch.cos(torch.arange(order + 1, dtype=dtype)[:, None] * theta) @ samples)
        coefficients[0] /= 2
        previous = residual
        current = torch.sparse.mm(scaled_sparse, residual)
        filtered = coefficients[0] * previous + coefficients[1] * current
        # Independently evaluate the scalar polynomial on the exact spectrum.
        scalar_x = 2 * eigenvalues / upper - 1
        scalar_previous = torch.ones_like(scalar_x)
        scalar_current = scalar_x
        scalar_result = coefficients[0] * scalar_previous + coefficients[1] * scalar_current
        for j in range(2, order + 1):
            following = 2 * torch.sparse.mm(scaled_sparse, current) - previous
            filtered = filtered + coefficients[j] * following
            previous, current = current, following
            scalar_following = 2 * scalar_x * scalar_current - scalar_previous
            scalar_result += coefficients[j] * scalar_following
            scalar_previous, scalar_current = scalar_current, scalar_following
        polynomial_loss = (filtered * active[:, None]).square().sum() / (3 * n)
        gradient, = torch.autograd.grad(polynomial_loss, predicted, retain_graph=True)
        epsilon = (scalar_result - torch.exp(-beta * eigenvalues / 2)).abs().max().item()
        metric_bound = epsilon * (2 + epsilon)
        loss_bound = metric_bound * residual.square().sum().item() / (3 * n)
        gradient_bound = 2 * metric_bound * residual.norm().item() / (3 * n)
        loss_error = abs((polynomial_loss - spatial).item())
        gradient_error = (gradient - exact_gradient).norm().item()
        assert loss_error <= loss_bound + 2e-14
        assert gradient_error <= gradient_bound + 2e-14
        torch.testing.assert_close(gradient[-1], torch.zeros(3, dtype=dtype), atol=0, rtol=0)
        approximation.append(dict(order=order, loss_error=loss_error, gradient_l2_error=gradient_error,
                                  spectral_pointwise_error=epsilon, loss_bound=loss_bound,
                                  gradient_bound=gradient_bound))
    assert approximation[1]['gradient_l2_error'] < 1e-10

    # This differentiable toy graph isolates the graph-Jacobian term. It has no
    # kNN changes; production dynamic_graph intentionally detaches this term.
    q = torch.tensor([[-.8], [.1], [.9]], dtype=dtype)
    y = torch.tensor([[-.4], [.5], [1.2]], dtype=dtype, requires_grad=True)
    distances = (y - y.T).square()
    affinity = torch.exp(-distances / 2) * (1 - torch.eye(3, dtype=dtype))
    dynamic_laplacian = torch.diag(affinity.sum(1)) - affinity
    dynamic_filter = torch.matrix_exp(-2 * dynamic_laplacian)
    e = y - q
    full_loss = (e * (dynamic_filter @ e)).sum() / 3
    detached_loss = (e * (dynamic_filter.detach() @ e)).sum() / 3
    full_gradient, = torch.autograd.grad(full_loss, y, retain_graph=True)
    detached_gradient, = torch.autograd.grad(detached_loss, y)
    jacobian_gap = (full_gradient - detached_gradient).norm().item()
    assert jacobian_gap > 1e-4
    direction = torch.tensor([[.2], [-.7], [.4]], dtype=dtype)
    epsilon_fd = 1e-5
    values = []
    for sign in (-1, 1):
        point = y.detach() + sign * epsilon_fd * direction
        affinity_fd = torch.exp(-(point - point.T).square() / 2) * (1 - torch.eye(3, dtype=dtype))
        laplacian_fd = torch.diag(affinity_fd.sum(1)) - affinity_fd
        error_fd = point - q
        values.append((error_fd * (torch.matrix_exp(-2 * laplacian_fd) @ error_fd)).sum() / 3)
    fd = (values[1] - values[0]) / (2 * epsilon_fd)
    dynamic_fd_error = abs((fd - (full_gradient * direction).sum()).item())
    assert dynamic_fd_error < 1e-9
    filter_change = torch.linalg.matrix_norm(torch.matrix_exp(-beta * 1.05 * laplacian) - heat, ord=2).item()
    filter_bound = beta * torch.linalg.matrix_norm(.05 * laplacian, ord=2).item()
    assert filter_change <= filter_bound + 1e-12

    # Low latent frequency need not stay low after a residual decoder Jacobian.
    low, high = basis[:, 2], basis[:, -1]
    decoder_jacobian = torch.eye(n, dtype=dtype) + .6 * high[:, None] @ low[None, :]
    output_shift = decoder_jacobian @ low
    high_fraction = ((high @ output_shift).square() / output_shift.square().sum()).item()
    assert abs(high_fraction - .36 / 1.36) < 1e-12

    # Closed-form modewise MSE for a prior estimate and a corrupted anchor.
    prior_variance, anchor_variance, covariance = .04, 1., 0.
    optimal_t = (prior_variance - covariance) / (prior_variance + anchor_variance - 2 * covariance)
    def risk(t):
        return (1 - t)**2 * prior_variance + t**2 * anchor_variance + 2 * t * (1 - t) * covariance
    assert risk(optimal_t) < risk(0) < risk(.5) < risk(1)
    # The exact clean estimate can still have positive fidelity gradient.
    clean = torch.tensor([[-1., 0., 0.], [1., 0., 0.]], dtype=dtype, requires_grad=True)
    noisy_anchor = clean.detach() + torch.tensor([.3, 0., 0.], dtype=dtype)
    small_laplacian = torch.tensor([[1., -1.], [-1., 1.]], dtype=dtype)
    clean_residual = clean - noisy_anchor
    clean_loss = (clean_residual * (torch.matrix_exp(-2 * small_laplacian) @ clean_residual)).sum() / 6
    clean_gradient, = torch.autograd.grad(clean_loss, clean)
    updated = clean.detach() - clean_gradient
    clean_error_after_step = (updated - clean.detach()).square().sum().item()
    assert clean_error_after_step > 0

    return dict(evidence='CPU synthetic algebra only; not LION/accuracy/deployment timing',
                environment=dict(python=platform.python_version(), torch=torch.__version__, device='cpu', dtype='float64', seed=20261103),
                fixed_basis=dict(updated_signal_roundtrip_max_error=roundtrip_error,
                                 coefficient_operator_loss_error=abs((spectral - spatial).item()),
                                 analytic_gradient_max_error=(exact_gradient - formula_gradient).abs().max().item()),
                polynomial=dict(vertices=n, active_vertices=int(active.sum()), laplacian_nnz=int(laplacian.count_nonzero()),
                                upper_bound=upper, beta=beta, results=approximation,
                                bound_scope='max error on this graph spectrum, not a certified whole-interval bound'),
                dynamic_graph=dict(full_vs_detached_gradient_l2=jacobian_gap,
                                   directional_finite_difference_error=dynamic_fd_error,
                                   heat_operator_change=filter_change, heat_change_bound=filter_bound,
                                   full_gradient=full_gradient.detach().flatten().tolist(),
                                   detached_gradient=detached_gradient.flatten().tolist()),
                decoder_counterexample=dict(high_mode_energy_fraction=high_fraction, is_actual_lion=False),
                anchoring=dict(prior_variance=prior_variance, anchor_variance=anchor_variance, covariance=covariance,
                               optimal_reference_fraction=optimal_t,
                               risks={str(t): risk(t) for t in (0., optimal_t, .5, 1.)},
                               exactly_clean_fidelity=clean_loss.item(),
                               clean_squared_error_after_fidelity_step=clean_error_after_step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run()
    report['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as destination:
            destination.write(serialized)
    else:
        print(serialized, end='')
