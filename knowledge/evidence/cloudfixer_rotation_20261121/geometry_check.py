"""CPU geometry sanity checks for the rotation note, no learned model or test data."""
import numpy as np


def best_rotation(source, target, weights):
    covariance = (source * weights[:, None]).T @ target
    u, _, vt = np.linalg.svd(covariance)
    sign = np.linalg.det(u @ vt)
    return (u * [1, 1, np.sign(sign)]) @ vt


def angle_deg(rotation):
    cosine = (np.trace(rotation) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def trimmed_pairs(a, b, keep=0.95):
    distance = ((a[:, None] - b[None, :]) ** 2).sum(axis=2)
    forward = distance.argmin(axis=1)
    reverse = distance.argmin(axis=0)
    i = np.argsort(distance[np.arange(len(a)), forward])[:int(len(a) * keep)]
    j = np.argsort(distance[reverse, np.arange(len(b))])[:int(len(b) * keep)]
    return np.r_[a[i], a[reverse[j]]], np.r_[b[forward[i]], b[j]]


def trimmed_error(a, b):
    p, q = trimmed_pairs(a, b)
    return np.mean(np.sum((p - q) ** 2, axis=1))


def main():
    rng = np.random.default_rng(93)
    clean = rng.normal(size=(128, 3)) * [1.4, .7, .3]
    clean -= clean.mean(axis=0)
    theta = np.deg2rad(25.)
    c, s = np.cos(theta), np.sin(theta)
    applied = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    x = clean @ applied
    fit = best_rotation(x, clean, np.ones(len(x)))
    assert np.allclose(fit, applied.T) and np.isclose(np.linalg.det(fit), 1)
    assert trimmed_error(x @ fit, clean) < 1e-20
    assert trimmed_error(clean @ applied, x) < 1e-20 < trimmed_error(clean, x)
    # Freeze actual two-way trimmed nearest neighbors for exactly one pose step.
    p, q = trimmed_pairs(x, clean)
    incremental = best_rotation(p, q, np.ones(len(p)))
    remaining = angle_deg(applied @ incremental)
    assert 0 < remaining < 25
    # Same ideal clean target throughout. Only the nearest neighbors change.
    pose = incremental.copy()
    errors = [remaining]
    for _ in range(7):
        a, b = trimmed_pairs(x @ pose, clean)
        pose = pose @ best_rotation(a, b, np.ones(len(a)))
        errors.append(angle_deg(applied @ pose))
    assert np.allclose(pose.T @ pose, np.eye(3)) and np.isclose(np.linalg.det(pose), 1)
    assert np.allclose([errors[i] for i in (0, 2, 7)], [24.3876, 22.7932, 7.1371], atol=.0001)
    p, q = trimmed_pairs(x, x)
    identity = best_rotation(p, q, np.ones(len(p)))
    assert np.allclose(identity, np.eye(3))
    symmetric = np.r_[np.eye(3), -np.eye(3)]
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert trimmed_error(symmetric @ quarter_turn, symmetric) == 0
    # Moving anchors or changing coordinates around the same origin is equivalent for the local solve.
    def proximal(points, anchors):
        p, q = trimmed_pairs(points, anchors)
        index = {tuple(row): i for i, row in enumerate(points)}
        sums, counts = np.zeros_like(points), np.zeros((len(points), 1))
        for row, mate in zip(p, q):
            i = index[tuple(row)]
            sums[i] += mate
            counts[i] += 1
        return (points + .1 * sums) / (1 + .1 * counts)
    assert np.allclose(proximal(clean, x @ incremental),
                       proximal(clean @ incremental.T, x) @ incremental)
    print(f"Known correspondences: residual angle {angle_deg(applied @ fit):.5f} deg")
    print(f"One trimmed nearest-neighbor step: {25:.2f} -> {remaining:.4f} deg")
    print(f"Fixed clean target after 1, 3, 8 fits: {errors[0]:.4f}, {errors[2]:.4f}, {errors[7]:.4f} deg")
    print("Input copied by target: identity. Octahedron: 90 deg has zero set error.")
    print("Local correction in the two coordinate frames agrees.")


if __name__ == "__main__":
    main()
