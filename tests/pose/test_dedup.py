"""Tests for hm2p.pose.dedup — image-based duplicate frame detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hm2p.pose.dedup import (
    MIN_CHANGED_PCT,
    PIXEL_NOISE,
    filter_duplicates_against_existing,
    is_duplicate,
    load_frame_gray,
)

# ── is_duplicate ────────────────────────────────────────────────────


def test_is_duplicate_identical_frames() -> None:
    """Two identical frames are duplicates with 0% changed."""
    frame = np.full((20, 20), 128, dtype=np.uint8)
    is_dup, pct = is_duplicate(frame, frame.copy())
    assert is_dup is True
    assert pct == 0.0


def test_is_duplicate_completely_different() -> None:
    """A black vs white frame is not a duplicate; 100% pixels changed."""
    black = np.zeros((20, 20), dtype=np.uint8)
    white = np.full((20, 20), 255, dtype=np.uint8)
    is_dup, pct = is_duplicate(black, white)
    assert is_dup is False
    assert pct == pytest.approx(100.0)


def test_is_duplicate_shape_mismatch() -> None:
    """Mismatched shapes return (False, 100.0) without comparing pixels."""
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    is_dup, pct = is_duplicate(a, b)
    assert is_dup is False
    assert pct == 100.0


def test_is_duplicate_noise_below_threshold() -> None:
    """Small per-pixel noise under PIXEL_NOISE is absorbed → duplicate."""
    a = np.full((50, 50), 100, dtype=np.uint8)
    b = a.copy()
    # Change every pixel by 10 (< PIXEL_NOISE=15) → not counted as changed.
    b[:] = 110
    is_dup, pct = is_duplicate(a, b)
    assert pct == 0.0
    assert is_dup is True


def test_is_duplicate_just_above_min_changed_pct() -> None:
    """When >= min_changed_pct pixels differ strongly, not a duplicate."""
    a = np.zeros((10, 10), dtype=np.uint8)  # 100 pixels
    b = a.copy()
    # Change 5 pixels (5%) by a large amount, well above 1% threshold.
    b.flat[:5] = 255
    is_dup, pct = is_duplicate(a, b)
    assert pct == pytest.approx(5.0)
    assert is_dup is False


def test_is_duplicate_custom_thresholds() -> None:
    """Custom thresholds change the classification boundary."""
    a = np.zeros((10, 10), dtype=np.uint8)
    b = a.copy()
    b.flat[:5] = 255  # 5% changed
    # With min_changed_pct=10, 5% < 10% → duplicate.
    is_dup, pct = is_duplicate(a, b, pixel_noise=15, min_changed_pct=10.0)
    assert is_dup is True
    assert pct == pytest.approx(5.0)


def test_module_constants() -> None:
    """Calibrated constants have expected defaults."""
    assert PIXEL_NOISE == 15
    assert MIN_CHANGED_PCT == 1.0


# ── load_frame_gray ─────────────────────────────────────────────────


def test_load_frame_gray_reads_png(tmp_path: Path) -> None:
    """A written PNG is read back as a 2-D grayscale array."""
    img = np.arange(400, dtype=np.uint8).reshape(20, 20)
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), img)
    loaded = load_frame_gray(p)
    assert loaded is not None
    assert loaded.ndim == 2
    assert loaded.shape == (20, 20)


def test_load_frame_gray_missing_file(tmp_path: Path) -> None:
    """A non-existent path returns None (cv2.imread failure)."""
    result = load_frame_gray(tmp_path / "does_not_exist.png")
    assert result is None


def test_load_frame_gray_follows_symlink(tmp_path: Path) -> None:
    """A symlink is resolved and the target image is loaded."""
    img = np.full((15, 15), 77, dtype=np.uint8)
    real = tmp_path / "real.png"
    cv2.imwrite(str(real), img)
    link = tmp_path / "link.png"
    link.symlink_to(real)
    loaded = load_frame_gray(link)
    assert loaded is not None
    assert loaded.shape == (15, 15)


# ── filter_duplicates_against_existing ──────────────────────────────


def _write_video(path: Path, frames: list[np.ndarray]) -> bool:
    """Write grayscale frames to a lossless AVI. Returns success."""
    h, w = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def test_filter_duplicates_unopenable_video(tmp_path: Path) -> None:
    """If the video cannot be opened, candidates are returned unchanged."""
    existing = tmp_path / "existing"
    existing.mkdir()
    candidates = [0, 5, 10]
    result = filter_duplicates_against_existing(
        tmp_path / "no_such_video.mp4", candidates, existing
    )
    assert result == candidates


def test_filter_duplicates_removes_batch_dups(tmp_path: Path) -> None:
    """Consecutive identical candidate frames are deduplicated."""
    black = np.zeros((32, 32), dtype=np.uint8)
    white = np.full((32, 32), 255, dtype=np.uint8)
    video = tmp_path / "vid.avi"
    if not _write_video(video, [black, black.copy(), white]):
        pytest.skip("No usable lossless video codec available")

    existing = tmp_path / "existing"
    existing.mkdir()  # empty — no disk frames to compare against

    kept = filter_duplicates_against_existing(video, [0, 1, 2], existing)
    # Frame 1 duplicates frame 0 → dropped; frame 2 is distinct → kept.
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


def test_filter_duplicates_against_disk_frame(tmp_path: Path) -> None:
    """A candidate matching an existing on-disk PNG is removed."""
    black = np.zeros((32, 32), dtype=np.uint8)
    white = np.full((32, 32), 255, dtype=np.uint8)
    video = tmp_path / "vid.avi"
    if not _write_video(video, [black, white]):
        pytest.skip("No usable lossless video codec available")

    existing = tmp_path / "existing"
    existing.mkdir()
    # Existing PNG identical to frame 0 (black) → frame 0 should be dropped.
    cv2.imwrite(str(existing / "prev.png"), black)

    kept = filter_duplicates_against_existing(video, [0, 1], existing)
    assert 0 not in kept
    assert 1 in kept


def test_filter_duplicates_skips_unreadable_index(tmp_path: Path) -> None:
    """Out-of-range frame indices are skipped without error."""
    black = np.zeros((32, 32), dtype=np.uint8)
    video = tmp_path / "vid.avi"
    if not _write_video(video, [black, black.copy()]):
        pytest.skip("No usable lossless video codec available")

    existing = tmp_path / "existing"
    existing.mkdir()
    # Index 999 is past the end → read fails and is skipped.
    kept = filter_duplicates_against_existing(video, [0, 999], existing)
    assert 0 in kept
    assert 999 not in kept
