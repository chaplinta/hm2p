"""Tests for helper functions in scripts/render_exemplar_clips.py.

Covers:
  - find_bouts: contiguous bout detection
  - select_exemplar_bouts: diversity + median-proximity selection
  - add_bout_border: pixel-level border rendering
  - extract_clip_with_context: boundary arithmetic (cv2 mocked)
  - load_exemplar_summary: S3 JSON loading (boto3 mocked)

All tests use synthetic numpy arrays only — no real data files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts directory to path so the module can be imported without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from render_exemplar_clips import (
    BORDER_COLOR_BGR,
    BORDER_PX,
    CONTEXT_FRAMES,
    MAX_CLIP_FRAMES,
    MIN_CLIP_FRAMES,
    add_bout_border,
    find_bouts,
    select_exemplar_bouts,
)


# ---------------------------------------------------------------------------
# find_bouts
# ---------------------------------------------------------------------------


class TestFindBouts:
    def test_empty_array_returns_empty(self):
        bouts = find_bouts(np.array([], dtype=int), target_syl=0)
        assert bouts == []

    def test_single_frame_match(self):
        bouts = find_bouts(np.array([5]), target_syl=5)
        assert bouts == [(0, 1)]

    def test_single_frame_no_match(self):
        bouts = find_bouts(np.array([3]), target_syl=5)
        assert bouts == []

    def test_all_same_syllable(self):
        ids = np.full(10, fill_value=2, dtype=int)
        bouts = find_bouts(ids, target_syl=2)
        assert bouts == [(0, 10)]

    def test_no_matching_syllable(self):
        ids = np.array([0, 1, 0, 1, 0], dtype=int)
        bouts = find_bouts(ids, target_syl=7)
        assert bouts == []

    def test_single_bout_in_middle(self):
        ids = np.array([0, 0, 3, 3, 3, 0, 0], dtype=int)
        bouts = find_bouts(ids, target_syl=3)
        assert bouts == [(2, 3)]

    def test_bout_at_start(self):
        ids = np.array([1, 1, 1, 0, 0], dtype=int)
        bouts = find_bouts(ids, target_syl=1)
        assert bouts == [(0, 3)]

    def test_bout_at_end(self):
        ids = np.array([0, 0, 2, 2, 2], dtype=int)
        bouts = find_bouts(ids, target_syl=2)
        assert bouts == [(2, 3)]

    def test_multiple_separate_bouts(self):
        ids = np.array([1, 0, 0, 1, 1, 0, 1], dtype=int)
        bouts = find_bouts(ids, target_syl=1)
        assert bouts == [(0, 1), (3, 2), (6, 1)]

    def test_alternating_single_frames(self):
        ids = np.array([5, 0, 5, 0, 5], dtype=int)
        bouts = find_bouts(ids, target_syl=5)
        assert bouts == [(0, 1), (2, 1), (4, 1)]

    def test_bout_duration_sums_to_match_count(self):
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 4, size=200)
        target = 2
        bouts = find_bouts(ids, target_syl=target)
        total_from_bouts = sum(d for _, d in bouts)
        total_actual = int(np.sum(ids == target))
        assert total_from_bouts == total_actual

    def test_start_frame_indices_correct(self):
        ids = np.array([0, 0, 7, 7, 0, 7], dtype=int)
        bouts = find_bouts(ids, target_syl=7)
        starts = [s for s, _ in bouts]
        assert starts == [2, 5]

    def test_duration_one_each(self):
        ids = np.array([1, 0, 1, 0], dtype=int)
        bouts = find_bouts(ids, target_syl=1)
        assert all(d == 1 for _, d in bouts)

    def test_two_bouts_adjacent_after_gap(self):
        # [target, other, target, target]
        ids = np.array([3, 0, 3, 3], dtype=int)
        bouts = find_bouts(ids, target_syl=3)
        assert bouts == [(0, 1), (2, 2)]

    def test_returns_list_of_tuples(self):
        ids = np.array([1, 1, 0], dtype=int)
        bouts = find_bouts(ids, target_syl=1)
        assert isinstance(bouts, list)
        assert all(isinstance(b, tuple) and len(b) == 2 for b in bouts)

    def test_long_array_single_bout(self):
        ids = np.zeros(1000, dtype=int)
        ids[200:400] = 9
        bouts = find_bouts(ids, target_syl=9)
        assert bouts == [(200, 200)]

    def test_target_zero(self):
        ids = np.array([0, 0, 1, 0], dtype=int)
        bouts = find_bouts(ids, target_syl=0)
        assert bouts == [(0, 2), (3, 1)]

    @given(
        st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=200),
        st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=200)
    def test_hypothesis_total_frames_consistent(self, id_list, target):
        ids = np.array(id_list, dtype=int)
        bouts = find_bouts(ids, target_syl=target)
        total_from_bouts = sum(d for _, d in bouts)
        total_actual = int(np.sum(ids == target))
        assert total_from_bouts == total_actual

    @given(
        st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=200),
        st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=200)
    def test_hypothesis_starts_in_bounds(self, id_list, target):
        ids = np.array(id_list, dtype=int)
        bouts = find_bouts(ids, target_syl=target)
        for start, dur in bouts:
            assert 0 <= start < len(ids)
            assert dur >= 1
            assert start + dur <= len(ids)

    @given(
        st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=200),
        st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=200)
    def test_hypothesis_bout_frames_all_match_target(self, id_list, target):
        ids = np.array(id_list, dtype=int)
        bouts = find_bouts(ids, target_syl=target)
        for start, dur in bouts:
            assert np.all(ids[start : start + dur] == target)


# ---------------------------------------------------------------------------
# select_exemplar_bouts
# ---------------------------------------------------------------------------


class TestSelectExemplarBouts:
    def test_empty_input_returns_empty(self):
        selected = select_exemplar_bouts([], n_exemplars=3, min_duration=3)
        assert selected == []

    def test_fewer_bouts_than_requested(self):
        bouts = [(0, 10, 5), (1, 20, 6)]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=1)
        assert len(selected) <= 2
        assert all(b in bouts for b in selected)

    def test_exactly_n_exemplars_when_enough(self):
        bouts = [(i, i * 10, 5 + i) for i in range(10)]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=1)
        assert len(selected) == 3

    def test_min_duration_filter_removes_short_bouts(self):
        bouts = [
            (0, 0, 1),   # too short
            (1, 10, 2),  # too short
            (2, 20, 5),  # valid
            (3, 30, 6),  # valid
            (4, 40, 7),  # valid
        ]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=3)
        for _, _, d in selected:
            assert d >= 3

    def test_all_bouts_below_min_duration_returns_raw_slice(self):
        # When no valid bouts exist, falls back to all_bouts[:n_exemplars].
        bouts = [(0, 0, 1), (1, 5, 2)]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=5)
        assert len(selected) <= len(bouts)

    def test_prefers_different_sessions(self):
        # Three bouts from session 0, one from session 1.
        # With n_exemplars=2 the selection should include session 1.
        bouts = [
            (0, 0, 5),
            (0, 10, 5),
            (0, 20, 5),
            (1, 30, 5),  # different session, same duration
        ]
        selected = select_exemplar_bouts(bouts, n_exemplars=2, min_duration=1)
        session_indices = [s for s, _, _ in selected]
        assert len(set(session_indices)) > 1

    def test_closest_to_median_duration_selected(self):
        # Durations: [10, 100, 50] → median = 50.
        # Scores: abs(10-50)=40, abs(100-50)=50, abs(50-50)=0.
        # Closest to median is duration 50 → should be selected first.
        bouts = [
            (0, 0, 10),   # score 40 from median
            (1, 10, 100), # score 50 from median
            (2, 20, 50),  # score  0 — exactly at median
        ]
        selected = select_exemplar_bouts(bouts, n_exemplars=1, min_duration=1)
        assert len(selected) == 1
        _, _, dur = selected[0]
        assert dur == 50

    def test_output_is_subset_of_valid_input(self):
        bouts = [(i % 3, i * 5, 3 + i) for i in range(12)]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=3)
        valid_bouts = [(s, f, d) for s, f, d in bouts if d >= 3]
        for b in selected:
            assert b in valid_bouts

    def test_no_duplicates_in_selection(self):
        bouts = [(i, i * 10, 5) for i in range(10)]
        selected = select_exemplar_bouts(bouts, n_exemplars=5, min_duration=1)
        assert len(selected) == len(set(selected))

    def test_n_exemplars_zero_returns_empty(self):
        bouts = [(0, 0, 5), (1, 10, 6)]
        selected = select_exemplar_bouts(bouts, n_exemplars=0, min_duration=1)
        assert selected == []

    def test_single_bout_single_session(self):
        bouts = [(0, 5, 10)]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=1)
        assert len(selected) == 1
        assert selected[0] == (0, 5, 10)

    def test_returns_list_of_triples(self):
        bouts = [(i, i * 10, 5 + i) for i in range(5)]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=1)
        assert isinstance(selected, list)
        assert all(isinstance(b, tuple) and len(b) == 3 for b in selected)

    def test_diversity_over_extra_sessions(self):
        # 5 sessions each with 1 bout at the median duration.
        bouts = [(i, i * 20, 8) for i in range(5)]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=1)
        sessions = [s for s, _, _ in selected]
        # All three should be from different sessions.
        assert len(set(sessions)) == 3

    def test_second_pass_fills_remaining_slots(self):
        # Only 2 sessions but need 3 exemplars — second pass must fill from same session.
        bouts = [
            (0, 0, 5),
            (0, 10, 6),
            (0, 20, 4),
            (1, 30, 5),
        ]
        selected = select_exemplar_bouts(bouts, n_exemplars=3, min_duration=1)
        assert len(selected) == 3


# ---------------------------------------------------------------------------
# add_bout_border
# ---------------------------------------------------------------------------


class TestAddBoutBorder:
    def _make_frames(self, n: int, h: int = 50, w: int = 60) -> np.ndarray:
        """Create black frames (n, h, w, 3) uint8."""
        return np.zeros((n, h, w, 3), dtype=np.uint8)

    def test_input_not_mutated(self):
        frames = self._make_frames(5)
        original = frames.copy()
        add_bout_border(frames, bout_offset=1, bout_duration=2)
        np.testing.assert_array_equal(frames, original)

    def _assert_strip_color(self, strip: np.ndarray) -> None:
        """Assert every pixel in strip equals BORDER_COLOR_BGR."""
        color = np.array(BORDER_COLOR_BGR, dtype=np.uint8)
        assert np.all(strip == color), (
            f"Expected all pixels to be {BORDER_COLOR_BGR}, got unique values: "
            f"{np.unique(strip.reshape(-1, 3), axis=0)}"
        )

    def test_top_border_pixels_set(self):
        frames = self._make_frames(5)
        result = add_bout_border(frames, bout_offset=1, bout_duration=2)
        b = BORDER_PX
        # Frames 1 and 2 should have the top border set to BORDER_COLOR_BGR.
        for i in [1, 2]:
            self._assert_strip_color(result[i, :b, :])

    def test_bottom_border_pixels_set(self):
        frames = self._make_frames(5, h=50, w=60)
        result = add_bout_border(frames, bout_offset=0, bout_duration=3)
        b = BORDER_PX
        for i in range(3):
            self._assert_strip_color(result[i, -b:, :])

    def test_left_border_pixels_set(self):
        frames = self._make_frames(5)
        result = add_bout_border(frames, bout_offset=0, bout_duration=3)
        b = BORDER_PX
        for i in range(3):
            self._assert_strip_color(result[i, :, :b])

    def test_right_border_pixels_set(self):
        frames = self._make_frames(5)
        result = add_bout_border(frames, bout_offset=0, bout_duration=3)
        b = BORDER_PX
        for i in range(3):
            self._assert_strip_color(result[i, :, -b:])

    def test_non_bout_frames_unchanged(self):
        frames = self._make_frames(6)
        result = add_bout_border(frames, bout_offset=2, bout_duration=2)
        # Frames 0, 1, 4, 5 are outside the bout.
        for i in [0, 1, 4, 5]:
            np.testing.assert_array_equal(result[i], frames[i])

    def test_interior_pixels_unchanged(self):
        # Interior pixels (not within BORDER_PX of any edge) must stay zero.
        frames = self._make_frames(3, h=50, w=60)
        result = add_bout_border(frames, bout_offset=0, bout_duration=3)
        b = BORDER_PX
        for i in range(3):
            interior = result[i, b:-b, b:-b]
            np.testing.assert_array_equal(interior, 0)

    def test_bout_offset_zero_first_frames_have_border(self):
        frames = self._make_frames(4)
        result = add_bout_border(frames, bout_offset=0, bout_duration=2)
        b = BORDER_PX
        self._assert_strip_color(result[0, :b, :])
        self._assert_strip_color(result[1, :b, :])

    def test_bout_duration_longer_than_frames_clips_gracefully(self):
        frames = self._make_frames(3)
        # bout_offset=1, bout_duration=10 extends past end — should not raise.
        result = add_bout_border(frames, bout_offset=1, bout_duration=10)
        # Frames 1 and 2 should have border; frame 0 should not.
        b = BORDER_PX
        self._assert_strip_color(result[1, :b, :])
        np.testing.assert_array_equal(result[0], frames[0])

    def test_border_color_correct(self):
        frames = self._make_frames(2)
        result = add_bout_border(frames, bout_offset=0, bout_duration=2)
        b = BORDER_PX
        # Green border: BGR = (0, 200, 0)
        for i in range(2):
            top_row = result[i, 0, 10]  # arbitrary interior column
            assert top_row[0] == BORDER_COLOR_BGR[0]
            assert top_row[1] == BORDER_COLOR_BGR[1]
            assert top_row[2] == BORDER_COLOR_BGR[2]

    def test_output_dtype_preserved(self):
        frames = self._make_frames(3).astype(np.uint8)
        result = add_bout_border(frames, bout_offset=0, bout_duration=3)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self):
        frames = self._make_frames(5, h=48, w=64)
        result = add_bout_border(frames, bout_offset=1, bout_duration=2)
        assert result.shape == frames.shape

    def test_empty_frames_returns_empty(self):
        frames = np.zeros((0, 50, 60, 3), dtype=np.uint8)
        result = add_bout_border(frames, bout_offset=0, bout_duration=0)
        assert result.shape == (0, 50, 60, 3)

    def test_zero_bout_duration_no_frames_changed(self):
        frames = self._make_frames(4)
        result = add_bout_border(frames, bout_offset=0, bout_duration=0)
        np.testing.assert_array_equal(result, frames)

    def test_single_frame_bout(self):
        frames = self._make_frames(3)
        result = add_bout_border(frames, bout_offset=1, bout_duration=1)
        b = BORDER_PX
        self._assert_strip_color(result[1, :b, :])
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[2], frames[2])

    def test_non_black_interior_preserved(self):
        frames = np.full((3, 50, 60, 3), 128, dtype=np.uint8)
        result = add_bout_border(frames, bout_offset=0, bout_duration=3)
        b = BORDER_PX
        # Interior must remain 128.
        for i in range(3):
            interior = result[i, b:-b, b:-b]
            np.testing.assert_array_equal(interior, 128)


# ---------------------------------------------------------------------------
# extract_clip_with_context — boundary arithmetic (cv2 mocked)
# ---------------------------------------------------------------------------


class TestExtractClipWithContextArithmetic:
    """Test the clip boundary calculations in extract_clip_with_context.

    cv2.VideoCapture is mocked to return a fixed set of synthetic frames,
    allowing tests to focus on boundary arithmetic without real video files.
    """

    def _make_mock_cap(self, n_frames: int, h: int = 20, w: int = 30):
        """Build a mock cv2.VideoCapture that yields `n_frames` black frames."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = float(n_frames)
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, blank.copy())] * n_frames + [(False, None)] * 10
        return cap

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_normal_bout_returns_frames(self, mock_vc):
        mock_vc.return_value = self._make_mock_cap(n_frames=500)
        from render_exemplar_clips import extract_clip_with_context

        frames, clip_start, bout_offset, bout_dur = extract_clip_with_context(
            "dummy.mp4", bout_start=100, bout_duration=10, total_video_frames=500
        )
        assert frames is not None
        assert frames.ndim == 4
        assert frames.shape[3] == 3

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_bout_offset_equals_pre_context(self, mock_vc):
        """bout_offset reflects how many context frames precede the bout."""
        mock_vc.return_value = self._make_mock_cap(n_frames=500)
        from render_exemplar_clips import extract_clip_with_context

        _, clip_start, bout_offset, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=100, bout_duration=5, total_video_frames=500
        )
        expected_pre = 100 - clip_start
        assert bout_offset == expected_pre

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_clip_start_clamped_to_zero_for_early_bout(self, mock_vc):
        """When the bout is near frame 0, clip_start must not go negative."""
        mock_vc.return_value = self._make_mock_cap(n_frames=500)
        from render_exemplar_clips import extract_clip_with_context

        _, clip_start, _, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=2, bout_duration=5, total_video_frames=500
        )
        assert clip_start == 0

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_bout_at_start_clip_start_zero(self, mock_vc):
        """Bout starting at frame 0 — clip_start must be 0, bout_offset must be 0."""
        mock_vc.return_value = self._make_mock_cap(n_frames=200)
        from render_exemplar_clips import extract_clip_with_context

        _, clip_start, bout_offset, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=0, bout_duration=8, total_video_frames=200
        )
        assert clip_start == 0
        assert bout_offset == 0

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_min_clip_length_enforced(self, mock_vc):
        """Clips shorter than MIN_CLIP_FRAMES are padded to MIN_CLIP_FRAMES."""
        # Very short bout (1 frame) at frame 50 in a 500-frame video.
        n = 500
        mock_vc.return_value = self._make_mock_cap(n_frames=n)
        from render_exemplar_clips import extract_clip_with_context

        frames, _, _, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=50, bout_duration=1, total_video_frames=n
        )
        assert frames is not None
        assert len(frames) >= MIN_CLIP_FRAMES

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_max_clip_length_enforced(self, mock_vc):
        """Clips longer than MAX_CLIP_FRAMES are capped at MAX_CLIP_FRAMES."""
        n = 5000
        mock_vc.return_value = self._make_mock_cap(n_frames=n)
        from render_exemplar_clips import extract_clip_with_context

        frames, _, _, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=200, bout_duration=MAX_CLIP_FRAMES + 50, total_video_frames=n
        )
        assert frames is not None
        assert len(frames) <= MAX_CLIP_FRAMES

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_returns_none_when_cap_fails_to_open(self, mock_vc):
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_vc.return_value = cap
        from render_exemplar_clips import extract_clip_with_context

        frames, clip_start, bout_offset, bout_dur = extract_clip_with_context(
            "nonexistent.mp4", bout_start=0, bout_duration=5
        )
        assert frames is None
        assert clip_start == 0
        assert bout_offset == 0
        assert bout_dur == 0

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_clip_clamped_at_video_end(self, mock_vc):
        """When bout_start + context would exceed total frames, clip is trimmed."""
        n = 50
        mock_vc.return_value = self._make_mock_cap(n_frames=n)
        from render_exemplar_clips import extract_clip_with_context

        frames, clip_start, _, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=45, bout_duration=3, total_video_frames=n
        )
        if frames is not None:
            assert clip_start + len(frames) <= n

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_bout_start_beyond_video_returns_none(self, mock_vc):
        """Bout start beyond total video frames should return None."""
        n = 30
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = float(n)
        cap.read.return_value = (False, None)
        mock_vc.return_value = cap
        from render_exemplar_clips import extract_clip_with_context

        frames, _, _, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=n + 10, bout_duration=5, total_video_frames=n
        )
        assert frames is None

    @patch("render_exemplar_clips.cv2.VideoCapture")
    def test_total_video_frames_none_reads_from_cap(self, mock_vc):
        """When total_video_frames is None, it is read from cv2.CAP_PROP_FRAME_COUNT."""
        cap = self._make_mock_cap(n_frames=100)
        mock_vc.return_value = cap
        from render_exemplar_clips import extract_clip_with_context

        frames, _, _, _ = extract_clip_with_context(
            "dummy.mp4", bout_start=30, bout_duration=5, total_video_frames=None
        )
        # cap.get should have been called to read frame count.
        cap.get.assert_called()
        assert frames is not None


# ---------------------------------------------------------------------------
# Boundary arithmetic unit tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestClipBoundaryArithmetic:
    """Verify clip boundary calculations in isolation without exercising cv2."""

    def _compute_boundaries(self, bout_start: int, bout_duration: int, total_frames: int):
        """Replicate the boundary arithmetic from extract_clip_with_context."""
        pre_context = CONTEXT_FRAMES
        post_context = CONTEXT_FRAMES
        clip_start = max(0, bout_start - pre_context)
        actual_pre = bout_start - clip_start
        clip_length = actual_pre + bout_duration + post_context
        clip_length = max(clip_length, MIN_CLIP_FRAMES)
        clip_length = min(clip_length, MAX_CLIP_FRAMES)
        if clip_start + clip_length > total_frames:
            clip_length = total_frames - clip_start
        return clip_start, actual_pre, clip_length

    def test_clip_start_never_negative(self):
        for bout_start in range(0, 20):
            clip_start, _, _ = self._compute_boundaries(bout_start, 5, 500)
            assert clip_start >= 0

    def test_clip_end_within_video(self):
        for total in [30, 100, 500]:
            clip_start, _, clip_length = self._compute_boundaries(
                bout_start=10, bout_duration=5, total_frames=total
            )
            assert clip_start + clip_length <= total

    def test_actual_pre_matches_context_when_no_clamping(self):
        bout_start = 200
        clip_start, actual_pre, _ = self._compute_boundaries(
            bout_start=bout_start, bout_duration=5, total_frames=1000
        )
        assert actual_pre == bout_start - clip_start
        assert actual_pre == CONTEXT_FRAMES

    def test_actual_pre_less_than_context_near_start(self):
        clip_start, actual_pre, _ = self._compute_boundaries(
            bout_start=5, bout_duration=5, total_frames=500
        )
        assert actual_pre < CONTEXT_FRAMES
        assert actual_pre == 5  # only 5 frames before bout

    def test_clip_length_at_least_min(self):
        # Very short bout deep in the video.
        _, _, clip_length = self._compute_boundaries(
            bout_start=100, bout_duration=1, total_frames=1000
        )
        assert clip_length >= MIN_CLIP_FRAMES

    def test_clip_length_at_most_max(self):
        _, _, clip_length = self._compute_boundaries(
            bout_start=200, bout_duration=MAX_CLIP_FRAMES, total_frames=10000
        )
        assert clip_length <= MAX_CLIP_FRAMES

    @given(
        st.integers(min_value=0, max_value=500),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=50, max_value=2000),
    )
    @settings(max_examples=300)
    def test_hypothesis_boundaries_always_valid(self, bout_start, bout_duration, total_frames):
        if bout_start >= total_frames:
            return  # skip degenerate case handled by cv2 path
        clip_start, actual_pre, clip_length = self._compute_boundaries(
            bout_start, bout_duration, total_frames
        )
        assert clip_start >= 0
        assert clip_start + clip_length <= total_frames
        assert clip_length >= 0
        assert actual_pre == bout_start - clip_start
        assert actual_pre >= 0


# ---------------------------------------------------------------------------
# load_exemplar_summary (frontend/data.py)
# ---------------------------------------------------------------------------


class TestLoadExemplarSummary:
    """Tests for load_exemplar_summary in frontend/data.py.

    boto3 and streamlit cache are mocked to avoid real S3 calls.
    """

    def _make_summary(self) -> dict:
        return {
            "n_syllables": 5,
            "n_exemplars_per_syllable": 3,
            "n_sessions": 10,
            "total_clips_rendered": 15,
            "syllables": [
                {
                    "syllable_id": i,
                    "total_frames": 100 + i * 10,
                    "total_bouts": 5 + i,
                    "median_duration_frames": 8,
                    "median_duration_sec": 0.27,
                    "exemplars": [],
                }
                for i in range(5)
            ],
        }

    def _import_load_fn(self):
        """Import load_exemplar_summary bypassing st.cache_data."""
        import importlib
        import sys

        # Provide stub streamlit module so the import does not need a real Streamlit.
        if "streamlit" not in sys.modules:
            st_stub = MagicMock()
            st_stub.cache_data = lambda *a, **kw: (lambda f: f)
            sys.modules["streamlit"] = st_stub
        else:
            # Patch cache_data on the already-imported stub.
            sys.modules["streamlit"].cache_data = lambda *a, **kw: (lambda f: f)

        frontend_path = str(Path(__file__).resolve().parent.parent.parent / "frontend")
        if frontend_path not in sys.path:
            sys.path.insert(0, frontend_path)

        if "data" in sys.modules:
            del sys.modules["data"]

        import data as frontend_data

        return frontend_data.load_exemplar_summary

    def test_valid_json_returns_dict(self):
        summary = self._make_summary()
        load_fn = self._import_load_fn()

        with patch("data.download_s3_bytes", return_value=json.dumps(summary).encode()):
            result = load_fn()

        assert isinstance(result, dict)
        assert result["n_syllables"] == 5
        assert result["total_clips_rendered"] == 15

    def test_missing_data_returns_none(self):
        load_fn = self._import_load_fn()

        with patch("data.download_s3_bytes", return_value=None):
            result = load_fn()

        assert result is None

    def test_invalid_json_returns_none(self):
        load_fn = self._import_load_fn()

        with patch("data.download_s3_bytes", return_value=b"not-valid-json{{{"):
            result = load_fn()

        assert result is None

    def test_syllables_list_preserved(self):
        summary = self._make_summary()
        load_fn = self._import_load_fn()

        with patch("data.download_s3_bytes", return_value=json.dumps(summary).encode()):
            result = load_fn()

        assert isinstance(result["syllables"], list)
        assert len(result["syllables"]) == 5

    def test_empty_json_object_returned_as_dict(self):
        load_fn = self._import_load_fn()

        with patch("data.download_s3_bytes", return_value=b"{}"):
            result = load_fn()

        assert result == {}

    def test_empty_bytes_returns_none(self):
        load_fn = self._import_load_fn()

        with patch("data.download_s3_bytes", return_value=b""):
            result = load_fn()

        assert result is None
