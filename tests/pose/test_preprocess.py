"""Tests for pose/preprocess.py — video pre-processing."""

from __future__ import annotations

import numpy as np

from hm2p.pose.preprocess import crop_to_maze_roi


def test_crop_to_maze_roi_shape() -> None:
    """crop_to_maze_roi returns correct output shape."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = (100, 50, 200, 150)  # x, y, w, h
    cropped = crop_to_maze_roi(frame, roi)
    assert cropped.shape == (150, 200, 3)


def test_crop_to_maze_roi_content() -> None:
    """Cropped region matches the corresponding pixels in the input frame."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    roi = (100, 50, 200, 150)
    cropped = crop_to_maze_roi(frame, roi)
    expected = frame[50:200, 100:300]
    np.testing.assert_array_equal(cropped, expected)
