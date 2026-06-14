"""Unit tests for the registered-movie / FISSA-from-movie helpers in
hm2p.calcium.neuropil.

These cover the parts that do NOT require the optional ``fissa`` package:
the binary loader and the input-validation guards. Synthetic arrays only.
"""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.calcium.neuropil import (
    load_registered_movie,
    subtract_fissa_from_movie,
)


def _write_bin(path, arr):
    """Write an int16 array to a flat little-endian binary like Suite2p."""
    arr.astype("<i2").tofile(path)


# ---------------------------------------------------------------------------
# load_registered_movie
# ---------------------------------------------------------------------------


class TestLoadRegisteredMovie:
    def test_roundtrip_infers_frames(self, tmp_path):
        movie = (np.arange(3 * 4 * 5).reshape(3, 4, 5)).astype(np.int16)
        p = tmp_path / "data.bin"
        _write_bin(p, movie)
        out = load_registered_movie(p, crop_ly=4, crop_lx=5)
        assert out.shape == (3, 4, 5)
        assert out.dtype == np.int16
        np.testing.assert_array_equal(np.asarray(out), movie)

    def test_mmap_false_reads_into_ram(self, tmp_path):
        movie = np.random.default_rng(0).integers(-100, 100, (2, 3, 3)).astype(np.int16)
        p = tmp_path / "data.bin"
        _write_bin(p, movie)
        out = load_registered_movie(p, 3, 3, mmap=False)
        assert not isinstance(out, np.memmap)
        np.testing.assert_array_equal(out, movie)

    def test_explicit_n_frames_subset(self, tmp_path):
        movie = np.zeros((5, 2, 2), dtype=np.int16)
        movie[1] = 7
        p = tmp_path / "data.bin"
        _write_bin(p, movie)
        out = load_registered_movie(p, 2, 2, n_frames=2)
        assert out.shape == (2, 2, 2)
        assert out[1, 0, 0] == 7

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="registered binary not found"):
            load_registered_movie(tmp_path / "absent.bin", 4, 4)

    @pytest.mark.parametrize("ly,lx", [(0, 4), (4, 0), (-1, 4)])
    def test_nonpositive_crop_raises(self, tmp_path, ly, lx):
        p = tmp_path / "data.bin"
        _write_bin(p, np.zeros((1, 4, 4), dtype=np.int16))
        with pytest.raises(ValueError, match="must be positive"):
            load_registered_movie(p, ly, lx)

    def test_size_not_multiple_raises(self, tmp_path):
        # 3*4*5 int16 values, but claim a 4x4 frame → size not a multiple.
        p = tmp_path / "data.bin"
        _write_bin(p, np.zeros((3, 4, 5), dtype=np.int16))
        with pytest.raises(ValueError, match="not a multiple"):
            load_registered_movie(p, 4, 4)

    def test_requested_frames_exceeds_file_raises(self, tmp_path):
        p = tmp_path / "data.bin"
        _write_bin(p, np.zeros((2, 2, 2), dtype=np.int16))
        with pytest.raises(ValueError, match="only holds"):
            load_registered_movie(p, 2, 2, n_frames=5)


# ---------------------------------------------------------------------------
# subtract_fissa_from_movie — validation guards (no fissa package needed)
# ---------------------------------------------------------------------------


class TestSubtractFissaFromMovieValidation:
    def test_non_3d_movie_raises(self, tmp_path):
        with pytest.raises(ValueError, match="must be 3-D"):
            subtract_fissa_from_movie(
                np.zeros((4, 5)), [np.ones((4, 5), bool)], tmp_path
            )

    def test_mask_shape_mismatch_raises(self, tmp_path):
        movie = np.zeros((3, 4, 5), dtype=np.int16)
        masks = [np.ones((4, 5), bool), np.ones((6, 6), bool)]
        with pytest.raises(ValueError, match=r"roi_masks\[1\] shape"):
            subtract_fissa_from_movie(movie, masks, tmp_path)


class TestRunFissaRoiFormat:
    """The ROI masks must reach fissa.Experiment as a single roiset.

    FISSA indexes ``rois`` per trial (``self.rois[trial]``). With one movie
    (one trial) the roiset must be the whole list of N masks wrapped once:
    ``[[mask0, ..., maskN-1]]``. A flat ``[[mask0], [mask1], ...]`` is read as
    N trials and collapses to a single ROI — the bug this guards against.
    """

    def _install_fake_fissa(self, monkeypatch, captured, n_frames):
        import sys
        import types

        class FakeExperiment:
            def __init__(self, images, rois, folder, nRegions):
                captured["images"] = images
                captured["rois"] = rois
                captured["nRegions"] = nRegions
                # FISSA reads the per-trial roiset from rois[0]; replicate its
                # own ``len(rois) == 1`` semantics so n_roi matches reality.
                roiset = rois[0] if len(rois) == 1 else rois
                self._n_roi = len(roiset)

            def separate(self):
                self.result = [
                    [np.zeros((1, n_frames), dtype=np.float32)]
                    for _ in range(self._n_roi)
                ]

        fake = types.ModuleType("fissa")
        fake.Experiment = FakeExperiment
        monkeypatch.setitem(sys.modules, "fissa", fake)

    def test_single_roiset_with_all_masks(self, tmp_path, monkeypatch):
        n_rois, n_frames = 5, 40
        movie = np.zeros((n_frames, 6, 6), dtype=np.int16)
        masks = [np.zeros((6, 6), bool) for _ in range(n_rois)]
        for i, m in enumerate(masks):
            m[i % 6, i % 6] = True

        captured: dict = {}
        self._install_fake_fissa(monkeypatch, captured, n_frames)

        out = subtract_fissa_from_movie(movie, masks, tmp_path)

        # One outer element (single roiset), listing all N masks.
        assert len(captured["rois"]) == 1
        assert len(captured["rois"][0]) == n_rois
        assert len(captured["images"]) == 1  # one trial (one movie)
        assert out.shape == (n_rois, n_frames)
