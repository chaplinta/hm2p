"""Tests for hm2p.extraction.roi_classify."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.extraction.roi_classify import LABEL_NAMES, _write_outputs


class TestWriteOutputs:
    def test_writes_all_files(self, tmp_path: Path) -> None:
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int8)
        probs = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.05, 0.9, 0.05],
            [0.85, 0.1, 0.05],
        ], dtype=np.float32)

        _write_outputs(tmp_path, labels, probs)

        assert (tmp_path / "roi_class.npy").exists()
        assert (tmp_path / "roi_class_prob.npy").exists()
        assert (tmp_path / "iscell.npy").exists()

    def test_roi_class_values(self, tmp_path: Path) -> None:
        labels = np.array([0, 1, 2], dtype=np.int8)
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
        _write_outputs(tmp_path, labels, probs)

        loaded = np.load(tmp_path / "roi_class.npy")
        np.testing.assert_array_equal(loaded, labels)

    def test_iscell_soma_only(self, tmp_path: Path) -> None:
        """iscell marks only soma (label=1) as cells."""
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int8)
        probs = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.05, 0.9, 0.05],
            [0.85, 0.1, 0.05],
        ], dtype=np.float32)
        _write_outputs(tmp_path, labels, probs)

        iscell = np.load(tmp_path / "iscell.npy")
        assert iscell.shape == (5, 2)
        # Only soma (indices 1, 3) should be marked as cell
        assert iscell[0, 0] == 0.0  # artefact
        assert iscell[1, 0] == 1.0  # soma
        assert iscell[2, 0] == 0.0  # dendrite
        assert iscell[3, 0] == 1.0  # soma
        assert iscell[4, 0] == 0.0  # artefact
        # Column 1 is P(soma)
        assert iscell[1, 1] == pytest.approx(0.8)
        assert iscell[3, 1] == pytest.approx(0.9)

    def test_empty_rois(self, tmp_path: Path) -> None:
        labels = np.array([], dtype=np.int8)
        probs = np.zeros((0, 3), dtype=np.float32)
        _write_outputs(tmp_path, labels, probs)

        iscell = np.load(tmp_path / "iscell.npy")
        assert iscell.shape == (0, 2)
