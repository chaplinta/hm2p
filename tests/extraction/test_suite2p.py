"""Tests for extraction/suite2p.py — Suite2p extractor and ROI classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.extraction.suite2p import (
    Suite2pExtractor,
    _classify_heuristic,
    classify_roi_types,
)

_suite2p_available = False
try:
    import suite2p  # noqa: F401
    _suite2p_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Helpers: write synthetic Suite2p plane0 files
# ---------------------------------------------------------------------------


def _write_plane0(
    suite2p_dir: Path,
    n_rois: int = 10,
    n_cells: int = 6,
    n_frames: int = 200,
    include_stat: bool = False,
    include_ops: bool = False,
    rng: np.random.Generator | None = None,
) -> None:
    """Write minimal synthetic Suite2p plane0 numpy files."""
    if rng is None:
        rng = np.random.default_rng(42)
    plane = suite2p_dir / "plane0"
    plane.mkdir(parents=True)

    F = rng.uniform(100, 500, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (n_rois, n_frames)).astype(np.float32)
    iscell = np.zeros((n_rois, 2), dtype=np.float32)
    iscell[:n_cells, 0] = 1.0

    np.save(plane / "F.npy", F)
    np.save(plane / "Fneu.npy", Fneu)
    np.save(plane / "iscell.npy", iscell)

    if include_stat:
        stat = []
        for i in range(n_rois):
            ar = rng.uniform(3.0, 8.0) if i % 3 == 0 else rng.uniform(1.0, 1.8)
            stat.append(
                {
                    "aspect_ratio": ar,
                    "radius": rng.uniform(3.0, 12.0),
                    "compact": 1.0 / ar,
                    "skew": rng.uniform(0.5, 5.0),
                    "npix": int(rng.integers(50, 500)),
                    "npix_norm": rng.uniform(0.5, 3.0),
                    "ypix": np.array([0, 1, 2], dtype=int),
                    "xpix": np.array([0, 1, 2], dtype=int),
                }
            )
        np.save(plane / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)

    if include_ops:
        ops = {"fs": 29.97, "Ly": 64, "Lx": 64}
        np.save(plane / "ops.npy", ops)


# ---------------------------------------------------------------------------
# Suite2pExtractor — constructor
# ---------------------------------------------------------------------------


class TestSuite2pExtractorInit:
    def test_basic_load(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, n_rois=10, n_cells=6, n_frames=200)
        ext = Suite2pExtractor(s2p)
        assert ext.n_rois == 10
        assert ext.n_frames == 200

    def test_missing_plane0_raises(self, tmp_path: Path) -> None:
        (tmp_path / "suite2p").mkdir()
        with pytest.raises(FileNotFoundError, match="plane0"):
            Suite2pExtractor(tmp_path / "suite2p")

    def test_missing_npy_raises(self, tmp_path: Path) -> None:
        plane = tmp_path / "suite2p" / "plane0"
        plane.mkdir(parents=True)
        np.save(plane / "F.npy", np.zeros((5, 100)))
        with pytest.raises(FileNotFoundError, match="Fneu.npy"):
            Suite2pExtractor(tmp_path / "suite2p")

    def test_from_path(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p)
        ext = Suite2pExtractor.from_path(s2p)
        assert isinstance(ext, Suite2pExtractor)


# ---------------------------------------------------------------------------
# Suite2pExtractor — trace methods
# ---------------------------------------------------------------------------


class TestSuite2pTraces:
    def test_raw_traces_shape(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, n_rois=10, n_cells=6, n_frames=200)
        ext = Suite2pExtractor(s2p)
        F = ext.get_raw_traces()
        assert F.shape == (6, 200)
        assert F.dtype == np.float32

    def test_neuropil_traces_shape(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, n_rois=10, n_cells=6, n_frames=200)
        ext = Suite2pExtractor(s2p)
        Fneu = ext.get_neuropil_traces()
        assert Fneu is not None
        assert Fneu.shape == (6, 200)

    def test_accepted_roi_ids(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, n_rois=10, n_cells=6)
        ext = Suite2pExtractor(s2p)
        ids = ext.get_accepted_roi_ids()
        assert len(ids) == 6
        assert ids == list(range(6))


# ---------------------------------------------------------------------------
# Suite2pExtractor — optional stat/ops methods
# ---------------------------------------------------------------------------


class TestSuite2pOptionalMethods:
    def test_get_roi_masks_requires_stat(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, include_stat=False, include_ops=False)
        ext = Suite2pExtractor(s2p)
        with pytest.raises(RuntimeError, match="stat.npy"):
            ext.get_roi_masks()

    def test_get_roi_masks_with_stat(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, n_rois=6, n_cells=4, include_stat=True, include_ops=True)
        ext = Suite2pExtractor(s2p)
        masks = ext.get_roi_masks()
        assert masks.shape == (4, 64, 64)
        assert masks.dtype == bool

    def test_get_sampling_frequency_with_ops(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, include_ops=True)
        ext = Suite2pExtractor(s2p)
        assert pytest.approx(ext.get_sampling_frequency()) == 29.97

    def test_get_sampling_frequency_no_ops_raises(self, tmp_path: Path) -> None:
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, include_ops=False)
        ext = Suite2pExtractor(s2p)
        with pytest.raises(RuntimeError, match="ops.npy"):
            ext.get_sampling_frequency()

    def test_get_roi_types_without_stat_raises(self, tmp_path: Path) -> None:
        """Without stat.npy, raises RuntimeError."""
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, n_rois=8, n_cells=5)
        ext = Suite2pExtractor(s2p)
        with pytest.raises(RuntimeError, match="stat.npy"):
            ext.get_roi_types()

    @pytest.mark.skipif(not _suite2p_available, reason="suite2p not installed")
    def test_get_roi_types_with_stat(self, tmp_path: Path) -> None:
        """With stat.npy, ROIs classified as soma/dend/artefact."""
        s2p = tmp_path / "suite2p"
        _write_plane0(s2p, n_rois=9, n_cells=6, include_stat=True)
        ext = Suite2pExtractor(s2p)
        types = ext.get_roi_types()
        assert len(types) == 6
        assert all(t in ("soma", "dend", "artefact") for t in types)


# ---------------------------------------------------------------------------
# classify_roi_types (standalone function)
# ---------------------------------------------------------------------------


class TestClassifyHeuristic:
    """Tests for the heuristic fallback classifier."""

    def test_output_length(self) -> None:
        stat = [
            {"aspect_ratio": 1.2, "radius": 5.0, "compact": 0.8},
            {"aspect_ratio": 4.0, "radius": 3.0, "compact": 0.25},
            {"aspect_ratio": 1.5, "radius": 7.0, "compact": 0.67},
        ]
        labels = _classify_heuristic(stat)
        assert len(labels) == 3

    def test_soma_classification(self) -> None:
        stat = [{"aspect_ratio": 1.2, "radius": 5.0, "compact": 0.8}]
        labels = _classify_heuristic(stat)
        assert labels[0] == "soma"

    def test_dend_classification(self) -> None:
        stat = [{"aspect_ratio": 4.0, "radius": 4.0, "compact": 0.25}]
        labels = _classify_heuristic(stat)
        assert labels[0] == "dend"

    def test_artefact_classification_small_radius(self) -> None:
        stat = [{"aspect_ratio": 1.0, "radius": 1.0, "compact": 1.0}]
        labels = _classify_heuristic(stat)
        assert labels[0] == "artefact"

    def test_artefact_classification_low_compact(self) -> None:
        stat = [{"aspect_ratio": 1.0, "radius": 5.0, "compact": 0.05}]
        labels = _classify_heuristic(stat)
        assert labels[0] == "artefact"

    def test_valid_labels_only(self) -> None:
        rng = np.random.default_rng(7)
        stat = []
        for _ in range(20):
            stat.append(
                {
                    "aspect_ratio": rng.uniform(0.5, 8.0),
                    "radius": rng.uniform(1.0, 15.0),
                    "compact": rng.uniform(0.05, 1.0),
                }
            )
        labels = _classify_heuristic(stat)
        assert all(label in ("soma", "dend", "artefact") for label in labels)

    def test_missing_keys_use_defaults(self) -> None:
        """Missing stat keys fall back to defaults (soma)."""
        stat = [{}]
        labels = _classify_heuristic(stat)
        assert labels[0] == "soma"


@pytest.mark.skipif(not _suite2p_available, reason="suite2p not installed")
class TestClassifyRoiTypes:
    """Tests for classify_roi_types using Suite2p trained classifiers."""

    def test_uses_suite2p_classifiers(self) -> None:
        """Classifiers produce valid labels for all ROIs."""
        stat = [{
            "aspect_ratio": 1.2, "radius": 5.0, "compact": 0.8,
            "skew": 3.0, "npix_norm": 1.5, "npix": 200,
        }]
        labels = classify_roi_types(stat)
        assert len(labels) == 1
        assert labels[0] in ("soma", "dend", "artefact")

    def test_valid_labels_only(self) -> None:
        """All labels must be one of the three valid types."""
        rng = np.random.default_rng(7)
        stat = []
        for _ in range(20):
            stat.append({
                "aspect_ratio": rng.uniform(0.5, 8.0),
                "radius": rng.uniform(1.0, 15.0),
                "compact": rng.uniform(0.05, 1.0),
                "skew": rng.uniform(0.1, 5.0),
                "npix_norm": rng.uniform(0.3, 4.0),
                "npix": int(rng.integers(20, 500)),
            })
        labels = classify_roi_types(stat)
        assert all(label in ("soma", "dend", "artefact") for label in labels)

    def test_missing_classifier_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when classifiers are missing."""
        stat = [{"aspect_ratio": 1.0, "radius": 5.0, "compact": 0.8}]
        with pytest.raises(FileNotFoundError, match="classifier_soma"):
            classify_roi_types(
                stat,
                classifier_soma_path=tmp_path / "classifier_soma.npy",
                classifier_dend_path=tmp_path / "classifier_dend.npy",
            )

    def test_dend_only_if_not_soma(self) -> None:
        """ROI classified as dend only if dend=True AND soma=False."""
        from unittest.mock import patch

        # Mock classify to return soma=True, dend=True → should be soma
        def mock_classify(stat_arr, classfile):
            n = len(stat_arr)
            return np.column_stack([np.ones(n), np.ones(n)])

        with patch(
            "suite2p.classification.classify.classify", mock_classify
        ):
            stat = [{"skew": 2.0, "compact": 0.5, "npix_norm": 1.0}]
            labels = classify_roi_types(stat)
            assert labels[0] == "soma"
