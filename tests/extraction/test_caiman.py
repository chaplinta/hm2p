"""Tests for extraction/caiman.py — CaImAn extractor."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from hm2p.extraction.caiman import CaimanExtractor

# ---------------------------------------------------------------------------
# Helpers: write synthetic CaImAn HDF5
# ---------------------------------------------------------------------------


def _write_caiman_h5(
    path: Path,
    n_rois: int = 15,
    n_accepted: int = 10,
    n_frames: int = 300,
    include_A: bool = False,
    dims: tuple[int, int] = (64, 64),
    rng: np.random.Generator | None = None,
) -> None:
    """Write a synthetic CaImAn-style HDF5 results file."""
    if rng is None:
        rng = np.random.default_rng(42)

    with h5py.File(path, "w") as f:
        C = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        f.create_dataset("C", data=C)
        f.create_dataset("idx_components", data=np.arange(n_accepted, dtype=int))
        f.create_dataset("dims", data=np.array(dims, dtype=int))

        if include_A:
            A = rng.uniform(0, 1, (dims[0] * dims[1], n_rois)).astype(np.float32)
            f.create_dataset("A", data=A)


# ---------------------------------------------------------------------------
# CaimanExtractor — constructor
# ---------------------------------------------------------------------------


class TestCaimanExtractorInit:
    def test_basic_load(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p, n_rois=15, n_accepted=10, n_frames=300)
        ext = CaimanExtractor(p)
        assert ext.n_rois == 15
        assert ext.n_frames == 300

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="CaImAn"):
            CaimanExtractor(tmp_path / "nonexistent.hdf5")

    def test_missing_c_matrix_key_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.hdf5"
        with h5py.File(p, "w") as f:
            f.create_dataset("wrong_key", data=np.zeros((5, 100)))
        with pytest.raises(KeyError, match="C"):
            CaimanExtractor(p)

    def test_from_path(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p)
        ext = CaimanExtractor.from_path(p)
        assert isinstance(ext, CaimanExtractor)


# ---------------------------------------------------------------------------
# CaimanExtractor — interface methods
# ---------------------------------------------------------------------------


class TestCaimanTraces:
    def test_raw_traces_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p, n_rois=15, n_accepted=10, n_frames=300)
        ext = CaimanExtractor(p)
        C = ext.get_raw_traces()
        assert C.shape == (10, 300)
        assert C.dtype == np.float32

    def test_neuropil_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p)
        ext = CaimanExtractor(p)
        assert ext.get_neuropil_traces() is None

    def test_accepted_ids(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p, n_rois=15, n_accepted=10)
        ext = CaimanExtractor(p)
        ids = ext.get_accepted_roi_ids()
        assert len(ids) == 10
        assert ids == list(range(10))

    def test_roi_types_all_soma(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p, n_accepted=8)
        ext = CaimanExtractor(p)
        types = ext.get_roi_types()
        assert len(types) == 8
        assert all(t == "soma" for t in types)

    def test_sampling_frequency_default(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p)
        ext = CaimanExtractor(p)
        assert ext.get_sampling_frequency() == 30.0


# ---------------------------------------------------------------------------
# CaimanExtractor — spatial masks
# ---------------------------------------------------------------------------


class TestCaimanMasks:
    def test_no_spatial_components_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p, include_A=False)
        ext = CaimanExtractor(p)
        with pytest.raises(RuntimeError, match="Spatial components"):
            ext.get_roi_masks()

    def test_with_spatial_components(self, tmp_path: Path) -> None:
        p = tmp_path / "caiman.hdf5"
        _write_caiman_h5(p, n_rois=10, n_accepted=6, include_A=True, dims=(32, 32))
        ext = CaimanExtractor(p)
        masks = ext.get_roi_masks()
        assert masks.shape == (6, 32, 32)
        assert masks.dtype == bool


# ---------------------------------------------------------------------------
# Edge case: no idx_components → all ROIs accepted
# ---------------------------------------------------------------------------


def test_no_idx_components_all_accepted(tmp_path: Path) -> None:
    """When idx_components is absent, all ROIs are treated as accepted."""
    p = tmp_path / "minimal.hdf5"
    with h5py.File(p, "w") as f:
        f.create_dataset("C", data=np.zeros((5, 100), dtype=np.float32))
    ext = CaimanExtractor(p)
    assert len(ext.get_accepted_roi_ids()) == 5
    assert ext.get_raw_traces().shape == (5, 100)
