"""CPU algebra witnesses for a proposed method; no LION or accuracy experiment."""
import json
import platform
from pathlib import Path

import numpy as np


def cg(matvec, rhs, iterations=100):
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    squared = float(residual @ residual)
    initial = squared
    if initial == 0:
        return x, 0
    for step in range(iterations):
        product = matvec(direction)
        denominator = float(direction @ product)
        assert denominator > 0
        scale = squared / denominator
        x += scale * direction
        residual -= scale * product
        following = float(residual @ residual)
        if following <= 1e-22 * initial:
            return x, step + 1
        direction = residual + (following / squared) * direction
        squared = following
    raise AssertionError('Synthetic SPD solve did not converge')


def main():
    rng = np.random.default_rng(20261104)
    n, m = 15, 23
    # One isolated vertex verifies that a large artificial diagonal is unnecessary.
    edges = np.array([(i, (i + 1) % (n - 1)) for i in range(n - 1)])
    edge_weights = rng.uniform(.3, 1., len(edges))
    adjacency = np.zeros((n, n))
    adjacency[edges[:, 0], edges[:, 1]] = edge_weights
    adjacency += adjacency.T
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    node = rng.integers(0, n, m)
    normals = rng.normal(size=(m, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    observation = np.zeros((m, 3 * n))
    for j in range(m):
        observation[j, 3 * node[j]:3 * node[j] + 3] = normals[j]
    confidence = rng.uniform(.1, 1., m)
    confidence[::5] = 0
    residual = rng.normal(size=m)
    sigma, huber_c = .6, 1.345
    weights = confidence / sigma**2 * np.minimum(1., huber_c * sigma / np.maximum(abs(residual), 1e-15))
    kappa, strength = 1.7, .8
    precision = kappa * np.eye(3 * n) + strength * np.kron(laplacian, np.eye(3))
    hessian = precision + observation.T @ (weights[:, None] * observation)
    rhs = observation.T @ (weights * residual)

    def matvec(vector):
        points = vector.reshape(n, 3)
        output = kappa * points.copy()
        flux = strength * edge_weights[:, None] * (points[edges[:, 0]] - points[edges[:, 1]])
        np.add.at(output, edges[:, 0], flux)
        np.add.at(output, edges[:, 1], -flux)
        projected = np.sum(normals * points[node], axis=1)
        np.add.at(output, node, (weights * projected)[:, None] * normals)
        return output.ravel()

    probe = rng.normal(size=3 * n)
    action_error = float(np.max(abs(matvec(probe) - hessian @ probe)))
    assert action_error < 1e-12
    delta, iterations = cg(matvec, rhs)
    exact = np.linalg.solve(hessian, rhs)
    relative_error = float(np.linalg.norm(delta - exact) / np.linalg.norm(exact))
    assert relative_error < 1e-9
    minimum_eigenvalue = float(np.linalg.eigvalsh(hessian).min())
    assert minimum_eigenvalue >= kappa - 1e-12

    def quadratic(value):
        difference = observation @ value - residual
        return .5 * (value @ precision @ value + np.sum(weights * difference**2))

    actual_change = float(quadratic(exact) - quadratic(np.zeros_like(exact)))
    predicted_change = float(-.5 * rhs @ exact)
    assert abs(actual_change - predicted_change) < 1e-12 and actual_change < 0
    direction = rng.normal(size=3 * n)
    epsilon = 1e-5
    finite_difference = (quadratic(probe + epsilon * direction) - quadratic(probe - epsilon * direction)) / (2 * epsilon)
    analytic = float(direction @ (hessian @ probe - rhs))
    assert abs(finite_difference - analytic) < 1e-7
    scaled = .2 * delta
    assert quadratic(scaled) < quadratic(np.zeros_like(delta))

    def robust(value):
        z = abs((observation @ value - residual) / sigma)
        rho = np.where(z <= huber_c, .5 * z**2, huber_c * z - .5 * huber_c**2)
        return .5 * value @ precision @ value + confidence @ rho

    robust_change = float(robust(delta) - robust(np.zeros_like(delta)))
    assert robust_change < 0  # Fixed correspondences/confidences: one IRLS majorizer.
    permutation = rng.permutation(n)
    columns = (3 * permutation[:, None] + np.arange(3)).ravel()
    rows = rng.permutation(m)
    permuted = observation[rows][:, columns]
    h_permuted = precision[np.ix_(columns, columns)] + permuted.T @ (weights[rows, None] * permuted)
    b_permuted = permuted.T @ (weights[rows] * residual[rows])
    permutation_error = float(np.max(abs(np.linalg.solve(h_permuted, b_permuted) - exact[columns])))
    assert permutation_error < 1e-12
    flips = rng.choice([-1., 1.], m)
    flipped = flips[:, None] * observation
    np.testing.assert_allclose(flipped.T @ (weights * flips * residual), rhs, atol=1e-12)
    np.testing.assert_allclose(flipped.T @ (weights[:, None] * flipped), observation.T @ (weights[:, None] * observation), atol=1e-12)
    zero, zero_iterations = cg(lambda x: precision @ x, np.zeros(3 * n))
    assert not zero.any() and zero_iterations == 0

    frequencies, basis = np.linalg.eigh(laplacian)
    scalar_residual = rng.normal(size=n)
    weight = .9
    response = weight / (kappa + weight + strength * frequencies)
    spectral_delta = basis @ (response * (basis.T @ scalar_residual))
    spatial_delta = np.linalg.solve((kappa + weight) * np.eye(n) + strength * laplacian, weight * scalar_residual)
    spectral_error = float(np.max(abs(spectral_delta - spatial_delta)))
    assert spectral_error < 1e-12
    noncommutation = np.linalg.norm(laplacian @ np.diag(np.arange(n)) - np.diag(np.arange(n)) @ laplacian)
    assert noncommutation > 0  # Nonuniform weights cannot generally be diagonalized in U.

    a_t, a_s = .75, .9
    sigma_t, sigma_s = np.sqrt(1 - a_t**2), np.sqrt(1 - a_s**2)
    clean, noise = rng.normal(size=(2, n, 4))
    lift = np.zeros_like(clean)
    lift[:, :3] = delta.reshape(n, 3)
    noisy = a_t * clean + sigma_t * noise
    corrected_clean = clean + lift
    corrected_noise = (noisy - a_t * corrected_clean) / sigma_t
    baseline = a_s * clean + sigma_s * noise
    coefficient = a_s - sigma_s * a_t / sigma_t
    corrected = a_s * corrected_clean + sigma_s * corrected_noise
    ddim_error = float(np.max(abs(corrected - baseline - coefficient * lift)))
    assert ddim_error < 1e-12 and coefficient > 0
    assert np.max(abs((corrected - baseline)[:, 3])) < 1e-12
    frozen_noise_difference = float(np.linalg.norm((a_s - coefficient) * lift))
    assert frozen_noise_difference > 1e-4

    # Residual lift is only approximately an inverse of a residual decoder.
    matrix = rng.normal(size=(3 * n, 4 * n)) / (20 * np.sqrt(n))
    decoder = lambda h: h.reshape(n, 4)[:, :3].ravel() + .01 * np.tanh(matrix @ h.ravel())
    lift_error = np.linalg.norm(decoder(clean + lift) - decoder(clean) - delta)
    lift_bound = .01 * np.linalg.norm(matrix, 2) * np.linalg.norm(lift)
    assert lift_error <= lift_bound + 1e-12

    # A clean planar surface with tangential resampling needs no normal correction.
    grid = np.array([(i, j, 0.) for i in range(4) for j in range(4)])
    observed = grid[rng.permutation(len(grid))] + [.1, .1, 0.]
    distance = ((observed[:, None] - grid[None])**2).sum(-1)
    nearest = distance.argmin(1)
    normal_residual = (observed - grid[nearest])[:, 2]
    assert not normal_residual.any()
    indexed_error = float(np.mean((observed - grid)**2))
    assert indexed_error > 0

    # A coherent wrong observation survives any graph smoothing of its DC mode.
    wrong = np.full(n, .4)
    harmful = np.linalg.solve((kappa + weight) * np.eye(n) + strength * laplacian, weight * wrong)
    np.testing.assert_allclose(harmful, weight / (kappa + weight) * wrong, atol=1e-12)
    wrong_cost = lambda d: .5 * (kappa * (d @ d) + strength * (d @ laplacian @ d) + weight * np.sum((d - wrong)**2))
    assert wrong_cost(harmful) < wrong_cost(np.zeros(n))
    clean_mse_increase = float(np.mean(harmful**2))
    assert clean_mse_increase > 0
    result = dict(status='passed',scope='synthetic CPU algebra only; no LION, CUDA, classification or timing evidence',
                  python=platform.python_version(),numpy=np.__version__,seed=20261104,
                  matrix_free_action_max_error=action_error,cg_iterations=iterations,cg_relative_solution_error=relative_error,
                  minimum_hessian_eigenvalue=minimum_eigenvalue,quadratic_change=actual_change,quadratic_change_formula=predicted_change,
                  finite_difference_error=abs(finite_difference-analytic),huber_objective_change=robust_change,
                  permutation_max_error=permutation_error,spectral_solution_max_error=spectral_error,
                  ddim_max_error=ddim_error,ddim_coefficient=coefficient,frozen_noise_vs_recomputed_difference=frozen_noise_difference,
                  toy_decoder_lift_error=float(lift_error),toy_decoder_lipschitz_bound=float(lift_bound),
                  planar_normal_residual_max=float(abs(normal_residual).max()),planar_indexed_mse=indexed_error,
                  coherent_wrong_anchor_clean_mse_increase=clean_mse_increase)
    target=Path(__file__).with_name('audit.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
