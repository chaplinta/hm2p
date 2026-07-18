"""Unit tests for hm2p.calcium.fissa_masks.

Synthetic small arrays only — no real data files (per project test policy).
"""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.calcium.fissa_masks import (
    build_roi_mask,
    build_roi_masks_from_plane,
    build_roi_masks_from_stat,
    crop_masks_to_window,
)

# ---------------------------------------------------------------------------
# build_roi_mask
# ---------------------------------------------------------------------------


class TestBuildRoiMask:
    def test_marks_only_roi_pixels(self):
        ypix = np.array([1, 2, 2])
        xpix = np.array([0, 3, 4])
        mask = build_roi_mask(ypix, xpix, Ly=5, Lx=6)
        assert mask.shape == (5, 6)
        assert mask.dtype == bool
        assert mask.sum() == 3
        assert mask[1, 0] and mask[2, 3] and mask[2, 4]
        # Everything else is False.
        mask[1, 0] = mask[2, 3] = mask[2, 4] = False
        assert not mask.any()

    def test_empty_pixels_gives_all_false(self):
        mask = build_roi_mask(np.array([], dtype=int), np.array([], dtype=int), 4, 4)
        assert mask.shape == (4, 4)
        assert not mask.any()

    def test_duplicate_pixels_idempotent(self):
        mask = build_roi_mask(np.array([1, 1]), np.array([2, 2]), 3, 3)
        assert mask.sum() == 1
        assert mask[1, 2]

    @pytest.mark.parametrize("Ly,Lx", [(0, 5), (5, 0), (-1, 5), (5, -2)])
    def test_nonpositive_dims_raise(self, Ly, Lx):
        with pytest.raises(ValueError, match="must be positive"):
            build_roi_mask(np.array([0]), np.array([0]), Ly, Lx)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            build_roi_mask(np.array([0, 1]), np.array([0]), 5, 5)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            build_roi_mask(np.zeros((2, 2), int), np.zeros((2, 2), int), 5, 5)

    @pytest.mark.parametrize(
        "ypix,xpix",
        [
            (np.array([5]), np.array([0])),  # y == Ly
            (np.array([0]), np.array([6])),  # x == Lx
            (np.array([-1]), np.array([0])),  # y < 0
            (np.array([0]), np.array([-1])),  # x < 0
        ],
    )
    def test_out_of_bounds_raise(self, ypix, xpix):
        with pytest.raises(ValueError, match="out of bounds"):
            build_roi_mask(ypix, xpix, Ly=5, Lx=6)


# ---------------------------------------------------------------------------
# build_roi_masks_from_stat
# ---------------------------------------------------------------------------


def _toy_stat():
    return [
        {"ypix": np.array([0, 1]), "xpix": np.array([0, 0])},
        {"ypix": np.array([3]), "xpix": np.array([4])},
        {"ypix": np.array([2, 2]), "xpix": np.array([2, 3])},
    ]


class TestBuildRoiMasksFromStat:
    def test_all_rois_in_order(self):
        stat = _toy_stat()
        masks = build_roi_masks_from_stat(stat, Ly=5, Lx=6)
        assert len(masks) == 3
        assert all(m.shape == (5, 6) for m in masks)
        assert [int(m.sum()) for m in masks] == [2, 1, 2]
        assert masks[1][3, 4]

    def test_roi_indices_subset_and_order(self):
        stat = _toy_stat()
        masks = build_roi_masks_from_stat(stat, 5, 6, roi_indices=[2, 0])
        assert len(masks) == 2
        # First requested mask is ROI 2 (2 pixels), second is ROI 0 (2 pixels).
        assert masks[0][2, 2] and masks[0][2, 3]
        assert masks[1][0, 0] and masks[1][1, 0]

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            build_roi_masks_from_stat(_toy_stat(), 5, 6, roi_indices=[3])

    def test_missing_pixels_key_raises(self):
        stat = [{"ypix": np.array([0])}]  # no xpix
        with pytest.raises(ValueError, match="missing ypix/xpix"):
            build_roi_masks_from_stat(stat, 5, 6)

    def test_nonpositive_dims_raise(self):
        with pytest.raises(ValueError, match="must be positive"):
            build_roi_masks_from_stat(_toy_stat(), 0, 6)

    def test_empty_stat_gives_empty_list(self):
        assert build_roi_masks_from_stat([], 5, 6) == []


# ---------------------------------------------------------------------------
# crop_masks_to_window
# ---------------------------------------------------------------------------


class TestCropMasksToWindow:
    def test_crop_dimensions_and_content(self):
        m = np.zeros((10, 12), dtype=bool)
        m[5, 6] = True  # inside window
        m[0, 0] = True  # outside window (edge)
        cropped = crop_masks_to_window([m], yrange=(2, 8), xrange=(3, 9))
        assert len(cropped) == 1
        assert cropped[0].shape == (6, 6)
        # (5,6) maps to (5-2, 6-3) = (3, 3); (0,0) is dropped.
        assert cropped[0][3, 3]
        assert cropped[0].sum() == 1

    def test_returns_copy_not_view(self):
        m = np.ones((6, 6), dtype=bool)
        cropped = crop_masks_to_window([m], (0, 3), (0, 3))[0]
        cropped[0, 0] = False
        assert m[0, 0]  # original untouched

    def test_degenerate_window_raises(self):
        m = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="degenerate"):
            crop_masks_to_window([m], (3, 3), (0, 4))

    def test_window_exceeds_bounds_raises(self):
        m = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="exceeds mask"):
            crop_masks_to_window([m], (0, 6), (0, 4))

    def test_edge_roi_emptied_warns(self, caplog):
        m = np.zeros((10, 10), dtype=bool)
        m[0, 0] = True  # only pixel is outside the crop window
        import logging

        with caplog.at_level(logging.WARNING):
            cropped = crop_masks_to_window([m], (5, 9), (5, 9))
        assert not cropped[0].any()
        assert "no pixels inside" in caplog.text


# ---------------------------------------------------------------------------
# build_roi_masks_from_plane
# ---------------------------------------------------------------------------


class TestBuildRoiMasksFromPlane:
    def test_loads_stat_and_ops(self, tmp_path):
        stat = np.array(_toy_stat(), dtype=object)
        ops = {"Ly": 5, "Lx": 6}
        np.save(tmp_path / "stat.npy", stat)
        np.save(tmp_path / "ops.npy", np.array(ops, dtype=object))
        masks, n_rois = build_roi_masks_from_plane(tmp_path)
        assert n_rois == 3
        assert len(masks) == 3
        assert all(m.shape == (5, 6) for m in masks)

    def test_missing_stat_raises(self, tmp_path):
        np.save(tmp_path / "ops.npy", np.array({"Ly": 5, "Lx": 6}, dtype=object))
        with pytest.raises(FileNotFoundError, match="stat.npy"):
            build_roi_masks_from_plane(tmp_path)

    def test_missing_ops_raises(self, tmp_path):
        np.save(tmp_path / "stat.npy", np.array(_toy_stat(), dtype=object))
        with pytest.raises(FileNotFoundError, match="ops.npy"):
            build_roi_masks_from_plane(tmp_path)
