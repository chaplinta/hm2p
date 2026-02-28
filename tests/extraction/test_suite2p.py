"""Tests for extraction/suite2p.py — Suite2p extractor and ROI classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def make_synthetic_stat(n_rois: int, rng: np.random.Generator) -> list[dict]:  # type: ignore[type-arg]
    """Create synthetic Suite2p stat list with shape statistics."""
    stats = []
    for i in range(n_rois):
        # Alternate compact (soma) and elongated (dend) shapes
        if i % 3 == 0:
            aspect_ratio = rng.uniform(3.0, 8.0)  # elongated → dend
            radius = rng.uniform(2.0, 5.0)
        else:
            aspect_ratio = rng.uniform(1.0, 1.8)  # compact → soma
            radius = rng.uniform(4.0, 12.0)
        stats.append(
            {
                "aspect_ratio": aspect_ratio,
                "radius": radius,
                "compact": 1.0 / aspect_ratio,
            }
        )
    return stats


def test_classify_roi_types_output_length(tmp_path: Path, rng: np.random.Generator) -> None:
    """classify_roi_types returns one label per ROI."""
    pytest.skip("Requires synthetic sklearn classifiers — implement alongside suite2p.py")


def test_classify_roi_types_valid_labels(tmp_path: Path, rng: np.random.Generator) -> None:
    """All labels are one of: soma, dend, artefact."""
    pytest.skip("Requires synthetic sklearn classifiers — implement alongside suite2p.py")
