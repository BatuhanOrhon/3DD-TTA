"""Small mathematical controls for isolated Chamfer-only experiments."""
import math
from numbers import Real

import torch


def validate_chamfer_experiment(experiment, *, skip_inactive, chamfer_weight,
                                spectral_weight, invariant_weight, graph, dynamic_graph):
    """Validate the opt-in boundary before any model or graph calculation."""
    if experiment is None:
        return None
    if (not skip_inactive or chamfer_weight <= 0 or spectral_weight != 0
            or invariant_weight != 0 or graph is not None or dynamic_graph):
        raise ValueError('Chamfer experiments require Chamfer-only, inactive skips and graph=None')
    if not isinstance(experiment, dict):
        raise ValueError('Chamfer experiment must be a mapping')
    variant = experiment.get('variant')
    options = {'sparse': 'interval', 'semantic': 'semantic_weight', 'weighted': 'density_k'}
    if not isinstance(variant, str) or variant not in options or set(experiment) - {'variant', options[variant]}:
        raise ValueError('Unknown or mixed Chamfer experiment options')
    result = dict(experiment)
    if variant == 'semantic':
        weight = result.get('semantic_weight')
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Semantic guidance requires an explicit finite nonnegative weight')
    else:
        key = options[variant]
        result.setdefault(key, 3 if variant == 'sparse' else 10)
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    return result


def guidance_loop_counts(horizon: int, interval: int) -> tuple:
    """One-based guidance counts, retaining the ordinary final update."""
    if type(horizon) is not int or horizon < 1 or type(interval) is not int or interval < 1:
        raise ValueError('Horizon and interval must be positive integers')
    return tuple(sorted(set(range(1, horizon + 1, interval)) | {horizon}))


def reference_density_weights(reference: torch.Tensor, k: int) -> torch.Tensor:
    """Detached inverse mean squared nonself-kNN distances, with mean weight1."""
    if (reference.ndim != 3 or reference.shape[-1] != 3 or not reference.is_floating_point()
            or reference.shape[0] == 0 or not torch.isfinite(reference).all()
            or type(k) is not int or not 0 < k < reference.shape[1]):
        raise ValueError('Expected finite nonempty BxNx3 reference and 0 < density_k < N')
    with torch.no_grad():
        points = reference.detach()
        neighborhoods = []
        for start in range(0, points.shape[1], 128):
            end = min(start + 128, points.shape[1])
            distances = torch.cdist(points[:, start:end], points).square()
            rows = torch.arange(end - start, device=points.device)
            distances[:, rows, rows + start] = float('inf')
            neighborhoods.append(distances.topk(k, dim=-1, largest=False).values.mean(-1))
        weights = (torch.cat(neighborhoods, dim=1) + 1e-6).reciprocal()
        weights = weights / weights.mean(dim=1, keepdim=True)
        if not torch.isfinite(weights).all():
            raise ValueError('Nonfinite reference-density weights')
        return weights


def weighted_selective_chamfer(pred_distances: torch.Tensor, reference_distances: torch.Tensor,
                               nearest_reference: torch.Tensor, reference_weights: torch.Tensor,
                               retained_fraction: float) -> torch.Tensor:
    """Keep the original distance trim and weight both directed match sets."""
    if (pred_distances.ndim != 2 or reference_distances.ndim != 2
            or reference_weights.shape != reference_distances.shape
            or nearest_reference is None or nearest_reference.shape != pred_distances.shape
            or pred_distances.shape[0] != reference_distances.shape[0]
            or nearest_reference.dtype not in (torch.int32, torch.int64)
            or not 0 < retained_fraction <= 1 or not torch.isfinite(reference_weights).all()
            or torch.any(reference_weights < 0)
            or torch.any(nearest_reference < 0)
            or torch.any(nearest_reference >= reference_weights.shape[1])):
        raise ValueError('Invalid weighted selective-Chamfer inputs')
    pred_count = int(pred_distances.shape[1] * retained_fraction)
    reference_count = int(reference_distances.shape[1] * retained_fraction)
    if not pred_count or not reference_count:
        raise ValueError('Selective Chamfer must retain at least one match per direction')
    pred_order = pred_distances.sort(dim=1).indices[:, :pred_count]
    reference_order = reference_distances.sort(dim=1).indices[:, :reference_count]
    pred_weights = reference_weights.gather(1, nearest_reference.long())
    return ((pred_distances.gather(1, pred_order) * pred_weights.gather(1, pred_order)).sum()
            + (reference_distances.gather(1, reference_order)
               * reference_weights.gather(1, reference_order)).sum())
