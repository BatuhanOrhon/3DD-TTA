"""Model-independent frozen-neighbor Chamfer quadratic, shared by diffusion samplers."""
import math


def proximal_chamfer(q, reference, forward, reverse, keep_forward, keep_reverse, tau):
    """Solve the frozen, bidirectional trimmed Chamfer quadratic exactly."""
    if not math.isfinite(tau) or tau < 0:
        raise ValueError('tau must be finite and nonnegative')
    forward, reverse = forward.long(), reverse.long()
    targets = reference.gather(1, forward[..., None].expand(-1, -1, 3))
    counts = keep_forward.to(q.dtype)[..., None]
    sums = targets * counts
    reverse_weight = keep_reverse.to(q.dtype)[..., None]
    sums = sums.scatter_add(1, reverse[..., None].expand(-1, -1, 3), reference * reverse_weight)
    counts = counts.scatter_add(1, reverse[..., None], reverse_weight)
    return (q + tau * sums) / (1 + tau * counts)
