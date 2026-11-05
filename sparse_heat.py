"""Standalone H4 diagnostic: dense reference graph setup, eigenfree heat filtering.

No production guidance path imports this module. The graph recipe deliberately
matches graph_spectral._sample_target; parity tests guard this duplicated setup.
Only the frozen normalized Laplacian and its active indices survive setup.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Tuple

import torch

from graph_spectral import SpectralConfig, _validate_xyz


@dataclass(frozen=True)
class SparseHeatGraph:
    active_indices: torch.Tensor
    laplacian: torch.Tensor  # active L / mean_active_degree, coalesced COO
    mean_active_degree: float
    upper_bound: float  # Gershgorin: 2*dmax/mean_active_degree, no eigh


@dataclass(frozen=True)
class SparseHeatTarget:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    graphs: Tuple[SparseHeatGraph, ...]

    def prepare(self, *, order: int, beta: float = 2.) -> SparseHeatFilter:
        """CPU-scalar Chebyshev coefficients for exp(-beta*Lbar/2)."""
        if isinstance(order, bool) or not isinstance(order, int) or order not in (8, 16, 32):
            raise ValueError("diagnostic order must be 8, 16 or 32")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        coefficients = []
        for graph in self.graphs:
            # Deterministic cosine quadrature; no device transfers or RNG use.
            # Increase resolution for unusually wide graph spectral intervals.
            scale = float(beta) * graph.upper_bound / 4
            if not math.isfinite(scale):
                raise ValueError("beta times graph bound must be finite")
            nodes = max(256, int(math.ceil(math.sqrt(scale) * 16)))
            theta = [math.pi * (j + .5) / nodes for j in range(nodes)]
            samples = [math.exp(-scale * (math.cos(t) + 1)) for t in theta]
            row = [2 / nodes * math.fsum(math.cos(k * t) * y for t, y in zip(theta, samples))
                   for k in range(order + 1)]
            row[0] *= .5
            coefficients.append(tuple(row))
        return SparseHeatFilter(self, tuple(coefficients), order, float(beta))


@dataclass(frozen=True)
class SparseHeatFilter:
    target: SparseHeatTarget
    coefficients: Tuple[Tuple[float, ...], ...]
    order: int
    beta: float

    def loss(self, residual_xyz: torch.Tensor) -> torch.Tensor:
        """Batch sum ||p_K(Lbar) E_active||²/(3*N), with a fixed graph."""
        _validate_xyz(residual_xyz, "residual_xyz")
        target = self.target
        if (residual_xyz.shape != target.shape or residual_xyz.dtype != target.dtype or
                residual_xyz.device != target.device):
            raise ValueError("residual shape, dtype and device must match the graph reference")
        loss = residual_xyz.reshape(-1)[0] * 0.
        for residual, graph, coefficients in zip(residual_xyz, target.graphs, self.coefficients):
            if graph.active_indices.numel() == 0:
                continue
            previous = residual[graph.active_indices]
            current = 2 / graph.upper_bound * torch.sparse.mm(graph.laplacian, previous) - previous
            filtered = coefficients[0] * previous + coefficients[1] * current
            for coefficient in coefficients[2:]:
                following = 2 * (2 / graph.upper_bound * torch.sparse.mm(graph.laplacian, current) - current) - previous
                filtered = filtered + coefficient * following
                previous, current = current, following
            loss = loss + filtered.square().sum() / (3 * target.shape[1])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("sparse heat loss is nonfinite")
        return loss


@torch.no_grad()
def build_sparse_heat_target(reference_xyz: torch.Tensor, config: SpectralConfig) -> SparseHeatTarget:
    """Match the existing dense cdist/kNN/RBF/prune/max-sym graph, without eigh.

    Setup remains quadratic in memory; this is not a sparse kNN implementation.
    Final isolates are excluded rather than assigned heat weight one.
    """
    _validate_xyz(reference_xyz, "reference_xyz")
    if not isinstance(config, SpectralConfig):
        raise TypeError("config must be SpectralConfig")
    count = reference_xyz.shape[1]
    if config.k >= count:
        raise ValueError("k must be smaller than the number of vertices")
    denominator = reference_xyz.new_tensor(2 * config.delta * config.delta)
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("2 * delta squared must be finite and positive in the reference dtype")
    graphs = []
    for reference in reference_xyz.detach():
        distances_squared = torch.cdist(reference, reference, p=2,
                                       compute_mode="donot_use_mm_for_euclid_dist").square()
        if not bool(torch.isfinite(distances_squared).all()):
            raise ValueError("reference pairwise squared distances exceed the working dtype")
        distances_squared.fill_diagonal_(float("inf"))
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
        active = adjacency.sum(dim=1) > 0
        indices = active.nonzero(as_tuple=False).flatten()
        active_adjacency = adjacency[active][:, active]
        laplacian = torch.diag(active_adjacency.sum(dim=1)) - active_adjacency
        if not bool(torch.isfinite(laplacian).all()):
            raise FloatingPointError("nonfinite spectral graph Laplacian")
        mean_degree, upper = 0., 0.
        if indices.numel():
            mean_degree = float((laplacian.diagonal().sum() / indices.numel()).item())
            if not math.isfinite(mean_degree) or mean_degree <= 0:
                raise FloatingPointError("active graph has invalid mean degree")
            upper = 2 * float(laplacian.diagonal().max().item()) / mean_degree
            laplacian = laplacian / mean_degree
        graphs.append(SparseHeatGraph(indices, laplacian.to_sparse_coo().coalesce(), mean_degree, upper))
    return SparseHeatTarget(reference_xyz.shape, reference_xyz.dtype, reference_xyz.device, tuple(graphs))


def measure_sparse_heat(reference_xyz: torch.Tensor, residual_xyz: torch.Tensor,
                        config: SpectralConfig) -> dict:
    """One synchronized cold diagnostic per K; no warmup or speedup claim.

    Autograd computes only a residual VJP on a detached clone. CUDA peak counters
    are reset for setup and each order; peaks are absolute allocated bytes, with
    an additional increase over that scope's starting allocated bytes.
    """
    device = reference_xyz.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def start_peak():
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            return torch.cuda.memory_allocated(device)
        return None

    def peak(baseline):
        if baseline is None:
            return {"peak_allocated_bytes": None, "peak_increase_bytes": None}
        maximum = torch.cuda.max_memory_allocated(device)
        return {"peak_allocated_bytes": maximum, "peak_increase_bytes": maximum - baseline}

    baseline = start_peak()
    started = time.perf_counter()
    target = build_sparse_heat_target(reference_xyz, config)
    sync()
    report = {"setup_seconds": time.perf_counter() - started, "setup_memory": peak(baseline), "orders": []}
    for order in (8, 16, 32):
        baseline = start_peak()
        started = time.perf_counter()
        prepared = target.prepare(order=order)
        coefficient_seconds = time.perf_counter() - started
        residual = residual_xyz.detach().clone().requires_grad_()
        sync()
        started = time.perf_counter()
        loss = prepared.loss(residual)
        sync()
        loss_seconds = time.perf_counter() - started
        started = time.perf_counter()
        gradient, = torch.autograd.grad(loss, residual)
        sync()
        backward_seconds = time.perf_counter() - started
        row = dict(order=order, beta=2., coefficient_seconds=coefficient_seconds,
                   loss_seconds=loss_seconds, backward_seconds=backward_seconds,
                   loss=float(loss.detach()), gradient_l2=float(gradient.norm()), **peak(baseline))
        report["orders"].append(row)
        del residual, loss, gradient, prepared
    return report
