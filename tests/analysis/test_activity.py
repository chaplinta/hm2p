"""Tests for hm2p.analysis.activity — condition-split activity analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hm2p.analysis.activity import (
    compute_batch_activity,
    compute_cell_activity,
    condition_event_rate,
    condition_mean_amplitude,
    condition_mean_signal,
    modulation_index,
    split_conditions,
)


# ---------------------------------------------------------------------------
# split_conditions
# ---------------------------------------------------------------------------


class TestSplitConditions:
    """Tests for split_conditions."""

    def test_basic_split(self):
        """Known speed/light arrays produce correct boolean masks."""
        # 8 frames: speed pattern high/low, light pattern on/off
        speed = np.array([3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0])
        light_on = np.array([True, True, False, False, True, True, False, False])
        active = np.ones(8, dtype=bool)

        conds = split_conditions(speed, light_on, active, speed_threshold=2.5)

        np.testing.assert_array_equal(
            conds["moving_light"],
            [True, False, False, False, True, False, False, False],
        )
        np.testing.assert_array_equal(
            conds["moving_dark"],
            [False, False, True, False, False, False, True, False],
        )
        np.testing.assert_array_equal(
            conds["stationary_light"],
            [False, True, False, False, False, True, False, False],
        )
        np.testing.assert_array_equal(
            conds["stationary_dark"],
            [False, False, False, True, False, False, False, True],
        )

    def test_active_mask_excludes_frames(self):
        """Frames where active_mask is False are excluded from all conditions."""
        speed = np.array([3.0, 3.0, 1.0, 1.0])
        light_on = np.array([True, True, True, True])
        active = np.array([True, False, True, False])

        conds = split_conditions(speed, light_on, active)

        # Only frames 0 and 2 are active.
        assert conds["moving_light"].sum() == 1  # frame 0
        assert conds["stationary_light"].sum() == 1  # frame 2
        assert conds["moving_dark"].sum() == 0
        assert conds["stationary_dark"].sum() == 0

    def test_all_conditions_mutually_exclusive(self):
        """No frame belongs to more than one condition."""
        rng = np.random.default_rng(42)
        n = 100
        speed = rng.uniform(0, 5, n)
        light_on = rng.choice([True, False], n)
        active = rng.choice([True, False], n)

        conds = split_conditions(speed, light_on, active)
        total = sum(c.astype(int) for c in conds.values())
        assert np.all(total <= 1)

    def test_conditions_cover_active_frames(self):
        """Active frames are fully partitioned across conditions."""
        rng = np.random.default_rng(7)
        n = 50
        speed = rng.uniform(0, 5, n)
        light_on = rng.choice([True, False], n)
        active = rng.choice([True, False], n)

        conds = split_conditions(speed, light_on, active)
        total = sum(c.astype(int) for c in conds.values())
        np.testing.assert_array_equal(total, active.astype(int))

    def test_threshold_boundary(self):
        """Speed exactly at threshold counts as moving."""
        speed = np.array([2.5])
        light_on = np.array([True])
        active = np.array([True])

        conds = split_conditions(speed, light_on, active, speed_threshold=2.5)
        assert conds["moving_light"][0]


# ---------------------------------------------------------------------------
# condition_event_rate
# ---------------------------------------------------------------------------


class TestConditionEventRate:
    """Tests for condition_event_rate."""

    def test_known_pattern(self):
        """Two onsets in 10 frames at 10 fps = 2 events / 1 s = 2.0 Hz."""
        # Onsets at frame 2 and frame 7.
        event_mask = np.array(
            [False, False, True, True, False, False, False, True, True, False]
        )
        condition_mask = np.ones(10, dtype=bool)
        rate = condition_event_rate(event_mask, condition_mask, fps=10.0)
        assert rate == pytest.approx(2.0)

    def test_no_condition_frames(self):
        """Returns 0.0 when condition_mask is all False."""
        event_mask = np.array([True, True, False])
        condition_mask = np.zeros(3, dtype=bool)
        assert condition_event_rate(event_mask, condition_mask, fps=10.0) == 0.0

    def test_onset_at_first_frame(self):
        """An event starting at frame 0 counts as one onset."""
        event_mask = np.array([True, False, False, False, False])
        condition_mask = np.ones(5, dtype=bool)
        rate = condition_event_rate(event_mask, condition_mask, fps=5.0)
        assert rate == pytest.approx(1.0)

    def test_onset_outside_condition_not_counted(self):
        """Onsets at frames outside condition_mask are ignored."""
        event_mask = np.array([False, True, True, False, False, True, False])
        # Only include frames 3-6 (onset at 5 is counted, onset at 1 is not).
        condition_mask = np.array([False, False, False, True, True, True, True])
        rate = condition_event_rate(event_mask, condition_mask, fps=4.0)
        # 1 onset in 4 frames at 4 fps = 1 event / 1 s = 1.0
        assert rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# condition_mean_signal
# ---------------------------------------------------------------------------


class TestConditionMeanSignal:
    def test_basic_mean(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        assert condition_mean_signal(signal, mask) == pytest.approx(2.0)

    def test_no_frames_returns_nan(self):
        signal = np.array([1.0, 2.0])
        mask = np.array([False, False])
        assert math.isnan(condition_mean_signal(signal, mask))


# ---------------------------------------------------------------------------
# condition_mean_amplitude
# ---------------------------------------------------------------------------


class TestConditionMeanAmplitude:
    def test_basic(self):
        signal = np.array([0.0, 5.0, 10.0, 0.0])
        event_mask = np.array([False, True, True, False])
        cond_mask = np.array([True, True, False, True])
        # Only frame 1 qualifies (event & condition).
        assert condition_mean_amplitude(signal, event_mask, cond_mask) == pytest.approx(
            5.0
        )

    def test_no_overlap_returns_nan(self):
        signal = np.array([1.0, 2.0])
        event_mask = np.array([True, False])
        cond_mask = np.array([False, True])
        assert math.isnan(condition_mean_amplitude(signal, event_mask, cond_mask))


# ---------------------------------------------------------------------------
# modulation_index
# ---------------------------------------------------------------------------


class TestModulationIndex:
    def test_both_zero(self):
        assert modulation_index(0.0, 0.0) == 0.0

    def test_one_zero(self):
        assert modulation_index(5.0, 0.0) == pytest.approx(1.0)
        assert modulation_index(0.0, 5.0) == pytest.approx(-1.0)

    def test_equal_rates(self):
        assert modulation_index(3.0, 3.0) == pytest.approx(0.0)

    def test_known_values(self):
        # (4 - 1) / (4 + 1) = 0.6
        assert modulation_index(4.0, 1.0) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# compute_cell_activity
# ---------------------------------------------------------------------------


class TestComputeCellActivity:
    def test_returns_expected_keys(self):
        """Output dict contains all expected keys."""
        n = 20
        signal = np.random.default_rng(0).standard_normal(n)
        event_mask = np.zeros(n, dtype=bool)
        event_mask[3:6] = True
        speed = np.full(n, 3.0)
        light_on = np.ones(n, dtype=bool)
        active = np.ones(n, dtype=bool)

        result = compute_cell_activity(signal, event_mask, speed, light_on, active, 10.0)

        expected_keys = set()
        for cond in ("moving_light", "moving_dark", "stationary_light", "stationary_dark"):
            for metric in ("event_rate", "mean_signal", "mean_amplitude"):
                expected_keys.add(f"{cond}_{metric}")
        expected_keys.add("movement_modulation")
        expected_keys.add("light_modulation")

        assert set(result.keys()) == expected_keys

    def test_all_values_are_float(self):
        n = 10
        signal = np.ones(n)
        event_mask = np.zeros(n, dtype=bool)
        speed = np.full(n, 3.0)
        light_on = np.ones(n, dtype=bool)
        active = np.ones(n, dtype=bool)

        result = compute_cell_activity(signal, event_mask, speed, light_on, active, 10.0)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# compute_batch_activity
# ---------------------------------------------------------------------------


class TestComputeBatchActivity:
    def test_returns_list_of_correct_length(self):
        n_rois, n_frames = 3, 20
        rng = np.random.default_rng(1)
        signals = rng.standard_normal((n_rois, n_frames))
        event_masks = rng.choice([True, False], (n_rois, n_frames))
        speed = rng.uniform(0, 5, n_frames)
        light_on = rng.choice([True, False], n_frames)
        active = np.ones(n_frames, dtype=bool)

        results = compute_batch_activity(signals, event_masks, speed, light_on, active, 10.0)

        assert len(results) == n_rois
        assert all(isinstance(r, dict) for r in results)
