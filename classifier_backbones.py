"""Frozen classifier logits; checkpoint loading, sampling and normalization stay outside."""

from typing import Literal, Optional

import torch
from torch import nn


class FrozenClassifier(nn.Module):
    """Adapt an audited model call layout while keeping gradients through XYZ.

    Pass the underlying Point-MAE module, as existing callers do. The explicit
    layout belongs to the pinned implementation, not to the backbone name.
    PointNeXt features are BCN; deriving height or other features stays external.
    """

    def __init__(self, model: nn.Module,
                 backbone: Literal["pointmae", "dgcnn", "curvenet", "pointnext"],
                 num_classes: int, *, input_layout: Literal["bnc", "bcn"]) -> None:
        super().__init__()
        if backbone not in ("pointmae", "dgcnn", "curvenet", "pointnext"):
            raise ValueError(f"Unsupported classifier backbone: {backbone}")
        if input_layout not in ("bnc", "bcn"):
            raise ValueError("input_layout must be 'bnc' or 'bcn'")
        if backbone in ("pointmae", "pointnext") and input_layout != "bnc":
            raise ValueError(f"{backbone} requires BxNx3 position input")
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        self.model = model
        self.backbone = backbone
        self.num_classes = num_classes
        self.input_layout = input_layout
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenClassifier":
        """Keep dropout and BatchNorm frozen even when a parent enters train mode."""
        super().train(False)
        return self

    def forward(self, points: torch.Tensor, *,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(points, torch.Tensor) or not points.is_floating_point():
            raise TypeError("Classifier input must be a floating-point XYZ tensor")
        if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
            raise ValueError("Classifier input must have nonempty BxNx3 shape")
        if not torch.isfinite(points).all():
            raise ValueError("Classifier input contains nonfinite XYZ coordinates")
        if features is not None:
            if self.backbone != "pointnext":
                raise ValueError("Explicit features are supported only for PointNeXt")
            if not isinstance(features, torch.Tensor) or not features.is_floating_point():
                raise TypeError("PointNeXt features must be a floating-point tensor")
            if (features.ndim != 3 or features.shape[0] != points.shape[0]
                    or features.shape[1] < 1 or features.shape[2] != points.shape[1]):
                raise ValueError("PointNeXt features must have matching BxCxN shape")
            if features.device != points.device or features.dtype != points.dtype:
                raise ValueError("PointNeXt features must match XYZ device and dtype")
            if not torch.isfinite(features).all():
                raise ValueError("PointNeXt features contain nonfinite values")
        if self.backbone == "pointmae":
            logits = self.model.classification_only(points, only_unmasked=False)
        elif self.backbone == "pointnext":
            if features is None:
                features = points.transpose(1, 2).contiguous()
            logits = self.model({"pos": points, "x": features})
        else:
            inputs = points if self.input_layout == "bnc" else points.transpose(1, 2).contiguous()
            logits = self.model(inputs)
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise TypeError("Classifier must return a floating-point logits tensor")
        if logits.shape != (points.shape[0], self.num_classes):
            raise ValueError(f"Classifier logits must have shape Bx{self.num_classes}")
        if not torch.isfinite(logits).all():
            raise ValueError("Classifier returned nonfinite logits")
        return logits
