"""Pure per-example composition for local and style guidance gradients."""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Dict, List, Optional, Tuple

import torch


COMPOSITION_SCHEMA_VERSION = 1
LOCAL_MODES = frozenset(("off", "scd", "spectral", "sum"))
STYLE_MODES = frozenset((
    "off", "scd", "spectral", "sum", "pcgrad", "scd_priority",
    "sum_norm_pcgrad",
))


@dataclass(frozen=True)
class CompositionConfig:
    """Serialized routing choices and explicit loss weights for each block."""

    local_mode: str = "off"
    style_mode: str = "off"
    local_scd_weight: float = 1.0
    local_spectral_weight: float = 1.0
    style_scd_weight: float = 1.0
    style_spectral_weight: float = 1.0
    norm_floor: float = 1e-12
    schema_version: int = COMPOSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.local_mode not in LOCAL_MODES:
            raise ValueError("local_mode must be one of: off, scd, spectral, sum")
        if self.style_mode not in STYLE_MODES:
            raise ValueError("unsupported style_mode: {}".format(self.style_mode))
        for name in (
            "local_scd_weight", "local_spectral_weight",
            "style_scd_weight", "style_spectral_weight",
        ):
            _validate_nonnegative_finite(getattr(self, name), name)
        _validate_positive_finite(self.norm_floor, "norm_floor")
        if self.schema_version != COMPOSITION_SCHEMA_VERSION:
            raise ValueError("unsupported composition schema_version")


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("{} must be a finite nonnegative number".format(name))
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("{} must be a finite nonnegative number".format(name))


def _validate_positive_finite(value: float, name: str) -> None:
    _validate_nonnegative_finite(value, name)
    if value <= 0:
        raise ValueError("{} must be positive".format(name))


def compose_block(
    scd_grad: Optional[torch.Tensor],
    spectral_grad: Optional[torch.Tensor],
    *,
    scd_weight: float,
    spectral_weight: float,
    mode: str,
    norm_floor: float = 1e-12,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    """Compose two guidance gradients independently for every batch example.

    A missing gradient is treated as a shape-correct zero when the other
    gradient supplies the shape. Diagnostics retain its presence separately.
    Projection arithmetic accumulates in float64 and the returned direction
    uses the present input's dtype and device.
    """
    if mode not in STYLE_MODES:
        raise ValueError("unsupported composition mode: {}".format(mode))
    _validate_nonnegative_finite(scd_weight, "scd_weight")
    _validate_nonnegative_finite(spectral_weight, "spectral_weight")
    _validate_positive_finite(norm_floor, "norm_floor")
    if scd_grad is None and spectral_grad is None:
        raise ValueError("at least one gradient tensor is required to define shape")

    template = scd_grad if scd_grad is not None else spectral_grad
    assert template is not None
    if not torch.is_tensor(template) or not template.is_floating_point():
        raise ValueError("gradients must be floating-point tensors")
    if template.ndim < 1 or template.shape[0] == 0:
        raise ValueError("gradients must have a nonempty batch dimension")

    for name, gradient in (("scd_grad", scd_grad), ("spectral_grad", spectral_grad)):
        if gradient is None:
            continue
        if not torch.is_tensor(gradient) or not gradient.is_floating_point():
            raise ValueError("{} must be a floating-point tensor".format(name))
        if gradient.shape != template.shape:
            raise ValueError("gradient tensors must have identical shape")
        if gradient.device != template.device or gradient.dtype != template.dtype:
            raise ValueError("gradient tensors must have identical dtype and device")
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError("gradient tensors must contain only finite values")

    scd_present = scd_grad is not None
    spectral_present = spectral_grad is not None
    if scd_grad is None:
        scd_grad = torch.zeros_like(template)
    if spectral_grad is None:
        spectral_grad = torch.zeros_like(template)

    # Promote before weighting and flatten only nonbatch dimensions. This
    # gives stable per-example reductions without allowing examples to mix.
    c = (scd_grad.to(dtype=torch.float64) * float(scd_weight)).reshape(template.shape[0], -1)
    q = (spectral_grad.to(dtype=torch.float64) * float(spectral_weight)).reshape(template.shape[0], -1)
    if not bool(torch.isfinite(c).all().item()) or not bool(torch.isfinite(q).all().item()):
        raise ValueError("weighted gradients must remain finite")
    c_norm = torch.linalg.vector_norm(c, dim=1)
    q_norm = torch.linalg.vector_norm(q, dim=1)
    dot = torch.sum(c * q, dim=1)
    valid_cosine = (c_norm > norm_floor) & (q_norm > norm_floor)
    cosine = torch.full_like(dot, float("nan"))
    cosine[valid_cosine] = (dot[valid_cosine] / (c_norm[valid_cosine] * q_norm[valid_cosine])).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))

    summed = c + q
    sum_norm = torch.linalg.vector_norm(summed, dim=1)
    raw_projected = summed.clone()
    projection_skipped = torch.zeros(template.shape[0], dtype=torch.bool, device=template.device)
    cap_factor = torch.ones_like(sum_norm)
    applied_scale = torch.ones_like(sum_norm)

    if mode == "off":
        result = torch.zeros_like(summed)
    elif mode == "scd":
        result = c
    elif mode == "spectral":
        result = q
    elif mode == "sum":
        result = summed
    elif mode in ("pcgrad", "sum_norm_pcgrad"):
        conflict = dot < 0
        valid_projection = (c_norm > norm_floor) & (q_norm > norm_floor)
        project = conflict & valid_projection
        projection_skipped = conflict & ~valid_projection
        projected_c = c.clone()
        projected_q = q.clone()
        projected_c[project] = c[project] - (
            dot[project] / q_norm[project].square()
        ).unsqueeze(1) * q[project]
        projected_q[project] = q[project] - (
            dot[project] / c_norm[project].square()
        ).unsqueeze(1) * c[project]
        raw_projected = projected_c + projected_q
        raw_norm = torch.linalg.vector_norm(raw_projected, dim=1)
        cap_factor = torch.ones_like(raw_norm)
        cap_rows = raw_norm > norm_floor
        cap_factor[cap_rows] = torch.minimum(
            torch.ones_like(raw_norm[cap_rows]), sum_norm[cap_rows] / raw_norm[cap_rows]
        )
        capped = raw_projected * cap_factor.unsqueeze(1)
        if mode == "pcgrad":
            result = capped
            applied_scale = cap_factor
        else:
            # Norm-match the ordinary sum to the capped symmetric projection
            # at this same state, without increasing its magnitude.
            matched = sum_norm > norm_floor
            applied_scale[matched] = torch.minimum(
                torch.ones_like(sum_norm[matched]),
                torch.linalg.vector_norm(capped, dim=1)[matched] / sum_norm[matched],
            )
            result = summed * applied_scale.unsqueeze(1)
    else:  # scd_priority
        result_q = q.clone()
        project = (dot < 0) & (c_norm > norm_floor)
        projection_skipped = (dot < 0) & (c_norm <= norm_floor)
        result_q[project] = q[project] - (
            dot[project] / c_norm[project].square()
        ).unsqueeze(1) * c[project]
        raw_projected = c + result_q
        result = raw_projected

    result_tensor = result.reshape(template.shape).to(dtype=template.dtype)
    if not bool(torch.isfinite(result_tensor).all().item()):
        raise ValueError("composed direction is not finite in the input dtype")
    raw_norm = torch.linalg.vector_norm(raw_projected, dim=1)
    applied_norm = torch.linalg.vector_norm(result, dim=1)
    diagnostics_values = {
        "mode": [mode] * template.shape[0],
        "scd_present": [scd_present] * template.shape[0],
        "spectral_present": [spectral_present] * template.shape[0],
        "scd_norm": c_norm,
        "spectral_norm": q_norm,
        "dot": dot,
        "cosine": cosine,
        "angle_degrees": angle,
        "sum_norm": sum_norm,
        "raw_projected_norm": raw_norm if mode in ("pcgrad", "sum_norm_pcgrad") else [None] * template.shape[0],
        "capped_projected_norm": raw_norm * cap_factor if mode in ("pcgrad", "sum_norm_pcgrad") else [None] * template.shape[0],
        "cap_factor": cap_factor if mode in ("pcgrad", "sum_norm_pcgrad") else [None] * template.shape[0],
        "applied_scale": applied_scale if mode in ("pcgrad", "sum_norm_pcgrad") else [None] * template.shape[0],
        "applied_norm": applied_norm,
        "scd_direction_dot": torch.sum(result * c, dim=1),
        "spectral_direction_dot": torch.sum(result * q, dim=1),
        "projection_skipped": [bool(value) for value in projection_skipped.tolist()],
    }
    rows: List[Dict[str, object]] = []
    for index in range(template.shape[0]):
        row: Dict[str, object] = {}
        for key, values in diagnostics_values.items():
            value = values[index] if isinstance(values, list) else values[index].item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            row[key] = value
        rows.append(row)
    return result_tensor, rows
