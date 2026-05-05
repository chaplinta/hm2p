"""Weighted heatmap target generator for DLC 3.x.

Subclasses the default Gaussian heatmap generator to upweight specific
keypoints (e.g. ears) during training. DLC discovers this via its
target-generator registry when ``type: WeightedHeatmapGaussianGenerator``
is set in ``pytorch_config.yaml``.

Usage (in pytorch_config.yaml)::

    model:
      heads:
        bodypart:
          target_generator:
            type: WeightedHeatmapGaussianGenerator
            keypoint_weights:
              1: 3.0   # left_ear
              2: 3.0   # right_ear

Requires ``scripts/`` on ``PYTHONPATH`` so DLC can import the module at
training time. The EC2 user-data script sets this automatically.
"""

from __future__ import annotations

from typing import Any

import torch


def _get_base_class() -> type:
    """Import the base heatmap generator from DLC, handling API changes."""
    try:
        from deeplabcut.pose_estimation_pytorch.models.target_generators import (
            HeatmapGaussianGenerator,
        )

        return HeatmapGaussianGenerator
    except ImportError:
        # Fallback for older DLC 3.x layout
        from deeplabcut.pose_estimation_pytorch.data.target_generators import (
            HeatmapGaussianGenerator,
        )

        return HeatmapGaussianGenerator


# Defer class creation to import time — the base class must be importable.
_Base = _get_base_class()


class WeightedHeatmapGaussianGenerator(_Base):  # type: ignore[misc]
    """Gaussian heatmap generator with per-keypoint loss weighting.

    Parameters
    ----------
    keypoint_weights : dict[int, float]
        Mapping from keypoint index (0-based) to weight multiplier.
        Keypoints not listed default to weight 1.0.
    **kwargs
        Forwarded to the base ``HeatmapGaussianGenerator``.
    """

    def __init__(self, keypoint_weights: dict[int, float] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.keypoint_weights: dict[int, float] = keypoint_weights or {}

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Generate heatmaps, then multiply weights for specified keypoints."""
        result = super().__call__(*args, **kwargs)
        if "weights" in result and self.keypoint_weights:
            w = result["weights"]
            for kp_idx, multiplier in self.keypoint_weights.items():
                if kp_idx < w.shape[-1]:
                    w[..., kp_idx] *= multiplier
            result["weights"] = w
        elif "heatmaps" in result and self.keypoint_weights:
            # If the base generator returns heatmaps without a separate
            # weights tensor, create one and apply the multipliers.
            heatmaps = result["heatmaps"]
            w = torch.ones(heatmaps.shape[0], dtype=heatmaps.dtype)
            for kp_idx, multiplier in self.keypoint_weights.items():
                if kp_idx < w.shape[0]:
                    w[kp_idx] = multiplier
            result["weights"] = w
        return result
