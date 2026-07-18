"""Tests for classify_session's early-exit branches (no model needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.extraction.roi_classify import classify_session


def test_classify_session_empty_rois(tmp_path: Path) -> None:
    """Zero ROIs → empty labels written and returned without loading a model.

    Also exercises the fps-from-ops path (fps=None reads ops.npy["fs"]).
    """
    plane = tmp_path / "plane0"
    plane.mkdir()
    np.save(plane / "stat.npy", np.array([], dtype=object))
    np.save(plane / "F.npy", np.zeros((0, 100), dtype=np.float32))
    np.save(plane / "Fneu.npy", np.zeros((0, 100), dtype=np.float32))
    np.save(plane / "ops.npy", np.array({"fs": 9.6}, dtype=object))

    result = classify_session(plane, fps=None)

    assert result["n_soma"] == 0
    assert result["n_dend"] == 0
    assert result["n_artefact"] == 0
    assert result["labels"].shape == (0,)
    assert result["probs"].shape == (0, 3)
    # Outputs were written to disk by the early-exit path.
    assert (plane / "roi_class.npy").exists()
