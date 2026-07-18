"""Tests for ``scripts/run_stage4_fissa.py`` — FISSA reprocessing helpers.

Covers the pure alignment-validation function that gates the FISSA reprocessing
path: it must report a high rank correlation when the regenerated movie is
aligned to the existing ROI masks, and a low one when the movie is spatially
shifted relative to the masks (the failure mode the gate must catch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_stage4_fissa as rsf  # noqa: E402


def _synthetic_movie_and_masks(
    n_frames: int = 60,
    ly: int = 16,
    lx: int = 16,
    n_rois: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Build a movie whose ROI pixels follow distinct per-ROI time courses.

    Each ROI is a small square with its own random temporal signal added on top
    of a shared background, so the mean over an ROI's mask tracks that ROI's
    signal. ``stored_F`` is computed as the mask-mean over the same movie, i.e.
    the ideal aligned re-extraction.
    """
    rng = np.random.default_rng(seed)
    background = rng.normal(100.0, 1.0, size=(n_frames, ly, lx))
    masks: list[np.ndarray] = []
    movie = background.copy()
    # Place ROIs on a grid so they do not overlap.
    centres = [(4, 4), (4, 11), (11, 4), (11, 11)][:n_rois]
    for cy, cx in centres:
        mask = np.zeros((ly, lx), dtype=bool)
        mask[cy - 1 : cy + 2, cx - 1 : cx + 2] = True
        signal = rng.normal(0.0, 20.0, size=n_frames)
        movie[:, mask] += signal[:, None]
        masks.append(mask)

    stored_F = np.stack([movie[:, m].mean(axis=1) for m in masks], axis=0)
    return movie, masks, stored_F


def test_aligned_movie_reports_high_correlation():
    movie, masks, stored_F = _synthetic_movie_and_masks()
    report = rsf.validate_movie_alignment(movie, masks, stored_F)
    assert report["n_rois"] == len(masks)
    assert report["n_evaluated"] == len(masks)
    # Mask-mean re-extraction of the same movie reproduces stored_F exactly.
    assert report["median_spearman"] > 0.99
    assert report["min_spearman"] > 0.99


def test_shifted_movie_reports_low_correlation():
    movie, masks, stored_F = _synthetic_movie_and_masks()
    # Roll the movie spatially so masks no longer cover their ROI pixels.
    shifted = np.roll(movie, shift=5, axis=2)
    report = rsf.validate_movie_alignment(shifted, masks, stored_F)
    # The gate (threshold 0.9) must reject this.
    assert report["median_spearman"] < 0.9


def test_empty_mask_is_skipped_not_evaluated():
    movie, masks, stored_F = _synthetic_movie_and_masks()
    masks[0] = np.zeros_like(masks[0])  # ROI with no pixels
    report = rsf.validate_movie_alignment(movie, masks, stored_F)
    assert report["n_rois"] == len(masks)
    assert report["n_evaluated"] == len(masks) - 1


def test_constant_trace_is_skipped():
    movie, masks, stored_F = _synthetic_movie_and_masks()
    # Make ROI 1's stored trace constant -> spearman undefined -> skipped.
    stored_F[1, :] = 7.0
    report = rsf.validate_movie_alignment(movie, masks, stored_F)
    assert report["n_evaluated"] == len(masks) - 1


def test_all_masks_empty_returns_nan_report():
    movie, masks, stored_F = _synthetic_movie_and_masks()
    masks = [np.zeros_like(m) for m in masks]
    report = rsf.validate_movie_alignment(movie, masks, stored_F)
    assert report["n_evaluated"] == 0
    assert np.isnan(report["median_spearman"])
    assert np.isnan(report["min_spearman"])


def test_roi_count_mismatch_raises():
    movie, masks, stored_F = _synthetic_movie_and_masks()
    with pytest.raises(ValueError, match="ROIs"):
        rsf.validate_movie_alignment(movie, masks[:-1], stored_F)


def test_frame_count_mismatch_raises():
    movie, masks, stored_F = _synthetic_movie_and_masks()
    with pytest.raises(ValueError, match="frames"):
        rsf.validate_movie_alignment(movie[:-1], masks, stored_F)
