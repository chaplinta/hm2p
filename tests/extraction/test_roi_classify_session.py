"""Tests for classify_session's early-exit branches (no model needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.extraction.roi_classify import classify_session


def test_classify_session_requires_fps(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        classify_session(tmp_path, fps=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        classify_session(tmp_path, fps=0.0)


def test_classify_session_empty_rois(tmp_path: Path) -> None:
    """Zero ROIs → empty labels written and returned without loading a model."""
    plane = tmp_path / "plane0"
    plane.mkdir()
    np.save(plane / "stat.npy", np.array([], dtype=object))
    np.save(plane / "F.npy", np.zeros((0, 100), dtype=np.float32))
    np.save(plane / "Fneu.npy", np.zeros((0, 100), dtype=np.float32))
    # ops["fs"] is ignored: the imaging rate must be passed explicitly
    np.save(plane / "ops.npy", np.array({"fs": 29.97}, dtype=object))

    result = classify_session(plane, fps=9.6)

    assert result["n_soma"] == 0
    assert result["n_dend"] == 0
    assert result["n_artefact"] == 0
    assert result["labels"].shape == (0,)
    assert result["probs"].shape == (0, 3)
    # Outputs were written to disk by the early-exit path.
    assert (plane / "roi_class.npy").exists()
