"""[Code] Static latent XYZ targets for GSD-inspired spectral guidance.

[Inference] This fidelity objective is not a reproduction of GSDTTA's learned
point shifts or model adaptation. Graph construction never consumes RNG state.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, List, Tuple

import torch

__all__ = ["SpectralConfig", "SpectralTarget", "build_spectral_target"]


@dataclass(frozen=True)
class SpectralConfig:
    k: int = 10
    delta: float = 0.1
    graph_gamma: float = 0.6
    modes: int = 100
    eigenspace_rtol: float = 1e-5
    eigenspace_atol: float = 1e-7

    def __post_init__(self) -> None:
        for name in ("k", "modes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("delta", "graph_gamma", "eigenspace_rtol", "eigenspace_atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value) or value < 0 or (name == "delta" and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'delta' else 'nonnegative'}")


def _validate_xyz(xyz: torch.Tensor, name: str) -> None:
    if not isinstance(xyz, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[0] < 1 or xyz.shape[1] < 2:
        raise ValueError(f"{name} must have shape (B, N, 3) with B >= 1 and N >= 2")
    if xyz.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError(f"{name} contains nonfinite coordinates")


@dataclass(frozen=True)
class SpectralTarget:
    """[Code] Detached reference and per-sample orthonormal selected bases."""

    reference_xyz: torch.Tensor
    bases: Tuple[torch.Tensor, ...]
    diagnostics: List[dict]

    def loss(self, pred_xyz: torch.Tensor) -> torch.Tensor:
        """[Code] Sum sample losses; preserve gradients only through prediction."""
        _validate_xyz(pred_xyz, "pred_xyz")
        if pred_xyz.shape != self.reference_xyz.shape:
            raise ValueError("prediction and reference shapes must match")
        if pred_xyz.device != self.reference_xyz.device or pred_xyz.dtype != self.reference_xyz.dtype:
            raise ValueError("prediction and reference device/dtype must match")
        # This connected zero also supports an entirely isolated batch.
        loss = pred_xyz.reshape(-1)[0] * 0.0
        for prediction, reference, basis in zip(pred_xyz, self.reference_xyz, self.bases):
            rank = basis.shape[1]
            if rank:
                coefficients = basis.transpose(0, 1) @ (prediction - reference)
                loss = loss + coefficients.square().sum() / (3 * rank)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("spectral loss is nonfinite; prediction/reference scale exceeds the working dtype")
        return loss


def _sample_target(reference: torch.Tensor, config: SpectralConfig) -> Tuple[torch.Tensor, dict]:
    started = time.perf_counter()
    count = reference.shape[0]
    # Direct Euclidean differences avoid cancellation in the matrix-product
    # cdist implementation, including at duplicate coordinates.
    distances_squared = torch.cdist(
        reference, reference, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    ).square()
    if not bool(torch.isfinite(distances_squared).all()):
        raise ValueError("reference pairwise squared distances exceed the working dtype")
    distances_squared.fill_diagonal_(float("inf"))
    # Stable ordering resolves exact distance ties by input vertex index.
    neighbors = distances_squared.argsort(dim=1, stable=True)[:, :config.k]
    weights = torch.exp(-distances_squared.gather(1, neighbors) / (2 * config.delta * config.delta))
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("nonfinite spectral graph weights")
    directed = torch.zeros_like(distances_squared).scatter_(1, neighbors, weights)
    del distances_squared, neighbors, weights
    directed_degree = directed.sum(dim=1)
    threshold = config.graph_gamma * directed.sum() / (count * config.k)
    if not bool(torch.isfinite(threshold)):
        raise ValueError("graph threshold exceeds the working dtype")
    retained = directed_degree > threshold
    adjacency = torch.maximum(directed, directed.transpose(0, 1))
    del directed
    adjacency *= retained[:, None] & retained[None, :]
    degree = adjacency.sum(dim=1)
    active = degree > 0
    active_count = int(active.sum().item())
    eigenvalues = reference.new_empty(0)
    rank = 0
    zero_mode_count = 0
    roundoff_floor = 0.0
    effective_boundary_atol = config.eigenspace_atol
    effective_zero_atol = config.eigenspace_atol
    eigendecomposition_seconds = 0.0
    eigengap = None
    basis = reference.new_zeros((count, 0))
    if active_count:
        active_adjacency = adjacency[active][:, active]
        laplacian = torch.diag(active_adjacency.sum(dim=1)) - active_adjacency
        if not bool(torch.isfinite(laplacian).all()):
            raise FloatingPointError("nonfinite spectral graph Laplacian")
        # [Inference] A dtype/size/norm allowance prevents numerically split
        # eigenspaces (especially disconnected zero modes) from being truncated.
        # Pairwise comparison allows error at both eigenvalues. This may retain
        # nearby modes whose separation is unresolved at the working precision.
        laplacian_infinity_norm = float(laplacian.abs().sum(dim=1).max().item())
        roundoff_floor = torch.finfo(reference.dtype).eps * active_count * laplacian_infinity_norm
        effective_boundary_atol = max(config.eigenspace_atol, 2 * roundoff_floor)
        effective_zero_atol = max(config.eigenspace_atol, roundoff_floor)
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        eigen_started = time.perf_counter()
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        if reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        eigendecomposition_seconds = time.perf_counter() - eigen_started
        if not bool(torch.isfinite(eigenvalues).all()) or not bool(torch.isfinite(eigenvectors).all()):
            raise FloatingPointError("spectral eigendecomposition returned nonfinite values")
        zero_mode_count = int((eigenvalues.abs() <= effective_zero_atol).sum().item())
        rank = min(config.modes, active_count)
        # Anchor tolerance to the requested boundary; do not chain together
        # successively close but collectively distinct eigenvalues.
        boundary = eigenvalues[rank - 1]
        boundary_tolerance = effective_boundary_atol + config.eigenspace_rtol * boundary.abs()
        while rank < active_count and bool((eigenvalues[rank] - boundary).abs() <= boundary_tolerance):
            rank += 1
        if rank < active_count:
            eigengap = float((eigenvalues[rank] - eigenvalues[rank - 1]).item())
        basis = reference.new_zeros((count, rank))
        basis[active] = eigenvectors[:, :rank]
    # Scale cancels in the fraction. Normalization avoids diagnostic overflow
    # for finite large coordinates, and float64 reduces projector roundoff.
    reference_scale = reference.abs().max()
    scaled_reference = reference.to(torch.float64) / reference_scale if reference_scale > 0 else reference.to(torch.float64)
    total_energy = float(scaled_reference.square().sum().item())
    selected_energy = float((basis.to(torch.float64).transpose(0, 1) @ scaled_reference).square().sum().item())
    low_energy_fraction = min(1.0, max(0.0, selected_energy / total_energy)) if total_energy else 0.0
    diagnostics: dict[str, Any] = {
        "vertices": count,
        "k": config.k,
        "requested_rank": config.modes,
        "actual_rank": rank,
        "zero_mode_count": zero_mode_count,
        "roundoff_floor": roundoff_floor,
        "effective_boundary_atol": effective_boundary_atol,
        "effective_zero_atol": effective_zero_atol,
        "active_vertices": active_count,
        "isolation_count": count - active_count,
        "threshold_rejected_count": int((~retained).sum().item()),
        "threshold": float(threshold.item()),
        "directed_degree_min": float(directed_degree.min().item()),
        "directed_degree_mean": float(directed_degree.mean().item()),
        "directed_degree_max": float(directed_degree.max().item()),
        "degree_min": float(degree.min().item()),
        "degree_mean": float(degree.mean().item()),
        "degree_max": float(degree.max().item()),
        "eigenvalue_min": float(eigenvalues[0].item()) if active_count else None,
        "selected_eigenvalue_max": float(eigenvalues[rank - 1].item()) if rank else None,
        "eigenvalue_max": float(eigenvalues[-1].item()) if active_count else None,
        "eigengap": eigengap,
        "low_energy_fraction": low_energy_fraction,
        # The complement includes unselected active modes AND isolated rows;
        # both fractions are zero when the reference has zero total energy.
        "remaining_energy_fraction": 1.0 - low_energy_fraction if total_energy else 0.0,
        "eigendecomposition_seconds": eigendecomposition_seconds,
        "runtime_seconds": time.perf_counter() - started,
    }
    return basis.detach(), diagnostics


@torch.no_grad()
def build_spectral_target(reference_xyz: torch.Tensor, config: SpectralConfig) -> SpectralTarget:
    """[Code] Build each dense graph sequentially, retaining only N x rank bases.

    [Code] k counts nonself neighbors, including distinct vertices at identical
    coordinates. Directed RBF degrees set the threshold before max symmetry.
    Both endpoints must survive it; final isolates are excluded from eigh.
    """
    _validate_xyz(reference_xyz, "reference_xyz")
    if not isinstance(config, SpectralConfig):
        raise TypeError("config must be SpectralConfig")
    if config.k >= reference_xyz.shape[1]:
        raise ValueError("k must be smaller than the number of vertices")
    denominator = reference_xyz.new_tensor(2 * config.delta * config.delta)
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("2 * delta squared must be finite and positive in the reference dtype")
    reference = reference_xyz.detach().clone()
    bases, diagnostics = [], []
    for sample in reference:
        basis, sample_diagnostics = _sample_target(sample, config)
        bases.append(basis)
        diagnostics.append(sample_diagnostics)
    return SpectralTarget(reference, tuple(bases), diagnostics)
