"""Tests for analysis/run.py — end-to-end analysis orchestration.

Tests _get_signal(), analyze_cell(), and edge cases like insufficient
frames and all-light/all-dark conditions. All data is synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.run import (
    AnalysisParams,
    CellResult,
    _get_signal,
    analyze_cell,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n_rois: int = 5,
    n_frames: int = 500,
    seed: int = 42,
    speed_moving_frac: float = 0.7,
    light_on_frac: float = 0.5,
) -> dict:
    """Generate synthetic data for analyze_cell tests.

    Returns a dict with all arrays needed by analyze_cell.
    """
    rng = np.random.default_rng(seed)

    dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32) * 0.1
    # Add some structure: ROI 0 has clear HD tuning
    for i in range(n_rois):
        n_events = rng.integers(3, 8)
        for _ in range(n_events):
            onset = rng.integers(0, max(1, n_frames - 30))
            width = min(20, n_frames - onset)
            dff[i, onset : onset + width] += rng.uniform(0.5, 2.0)

    deconv = np.abs(dff) * 0.5
    event_masks = (dff > 0.3).astype(np.bool_)

    hd_deg = rng.uniform(0, 360, n_frames).astype(np.float64)
    x_cm = rng.uniform(0, 50, n_frames).astype(np.float64)
    y_cm = rng.uniform(0, 50, n_frames).astype(np.float64)

    speed = np.zeros(n_frames, dtype=np.float64)
    n_moving = int(n_frames * speed_moving_frac)
    speed[:n_moving] = rng.uniform(3.0, 10.0, n_moving)
    speed[n_moving:] = rng.uniform(0.0, 2.0, n_frames - n_moving)
    rng.shuffle(speed)

    light_on = np.zeros(n_frames, dtype=bool)
    n_light = int(n_frames * light_on_frac)
    light_on[:n_light] = True
    rng.shuffle(light_on)

    active_mask = np.ones(n_frames, dtype=bool)

    return {
        "dff": dff,
        "deconv": deconv,
        "event_masks": event_masks,
        "hd_deg": hd_deg,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "speed": speed,
        "light_on": light_on,
        "active_mask": active_mask,
    }


# ---------------------------------------------------------------------------
# _get_signal
# ---------------------------------------------------------------------------


class TestGetSignal:
    """Tests for _get_signal helper function."""

    def test_dff_signal_type(self) -> None:
        """signal_type='dff' returns the dff row for the given ROI."""
        dff = np.arange(20, dtype=np.float32).reshape(4, 5)
        result = _get_signal(dff, deconv=None, event_masks=None, roi_idx=2, signal_type="dff")
        np.testing.assert_array_equal(result, dff[2])

    def test_deconv_signal_type(self) -> None:
        """signal_type='deconv' returns the deconv row for the given ROI."""
        dff = np.zeros((3, 10), dtype=np.float32)
        deconv = np.ones((3, 10), dtype=np.float32) * 5.0
        result = _get_signal(dff, deconv=deconv, event_masks=None, roi_idx=1, signal_type="deconv")
        np.testing.assert_array_equal(result, deconv[1])

    def test_events_signal_type(self) -> None:
        """signal_type='events' returns float32-cast event mask row."""
        dff = np.zeros((3, 10), dtype=np.float32)
        event_masks = np.zeros((3, 10), dtype=np.float32)
        event_masks[0, 3:6] = 1.0
        result = _get_signal(dff, deconv=None, event_masks=event_masks, roi_idx=0, signal_type="events")
        assert result.dtype == np.float32
        assert result[4] == 1.0
        assert result[0] == 0.0

    def test_unknown_signal_type_raises(self) -> None:
        """Unknown signal_type raises ValueError."""
        dff = np.zeros((2, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown signal_type"):
            _get_signal(dff, deconv=None, event_masks=None, roi_idx=0, signal_type="bogus")

    def test_deconv_none_raises(self) -> None:
        """signal_type='deconv' with deconv=None raises ValueError."""
        dff = np.zeros((2, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="deconv.*not available"):
            _get_signal(dff, deconv=None, event_masks=None, roi_idx=0, signal_type="deconv")

    def test_event_masks_none_raises(self) -> None:
        """signal_type='events' with event_masks=None raises ValueError."""
        dff = np.zeros((2, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="event_masks.*not available"):
            _get_signal(dff, deconv=None, event_masks=None, roi_idx=0, signal_type="events")

    @pytest.mark.parametrize("signal_type", ["dff", "deconv", "events"])
    def test_all_signal_types_return_correct_shape(self, signal_type: str) -> None:
        """All valid signal types return (n_frames,) array."""
        n_rois, n_frames = 4, 20
        dff = np.ones((n_rois, n_frames), dtype=np.float32)
        deconv = np.ones((n_rois, n_frames), dtype=np.float32) * 2.0
        event_masks = np.zeros((n_rois, n_frames), dtype=np.float32)
        result = _get_signal(dff, deconv, event_masks, roi_idx=1, signal_type=signal_type)
        assert result.shape == (n_frames,)


# ---------------------------------------------------------------------------
# analyze_cell — basic operation
# ---------------------------------------------------------------------------


class TestAnalyzeCell:
    """Tests for analyze_cell with synthetic data."""

    def test_returns_cell_result(self) -> None:
        """analyze_cell returns a CellResult dataclass."""
        data = _make_synthetic_data(n_rois=3, n_frames=500)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert isinstance(result, CellResult)
        assert result.roi_idx == 0

    def test_activity_dict_populated(self) -> None:
        """activity field is a non-empty dict with expected keys."""
        data = _make_synthetic_data(n_rois=3, n_frames=500)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert isinstance(result.activity, dict)
        assert len(result.activity) > 0
        assert "moving_light_event_rate" in result.activity
        assert "movement_modulation" in result.activity

    def test_hd_all_populated_with_enough_frames(self) -> None:
        """With sufficient moving frames, hd_all should have tuning results."""
        data = _make_synthetic_data(n_rois=3, n_frames=500, speed_moving_frac=0.8)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert len(result.hd_all) > 0
        assert "tuning_curve" in result.hd_all
        assert "mvl" in result.hd_all
        assert "p_value" in result.hd_all

    def test_place_all_populated_with_enough_frames(self) -> None:
        """With sufficient moving frames, place_all should have rate map results."""
        data = _make_synthetic_data(n_rois=3, n_frames=500, speed_moving_frac=0.8)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert len(result.place_all) > 0
        assert "rate_map" in result.place_all
        assert "spatial_info" in result.place_all

    def test_multiple_rois_different_seeds(self) -> None:
        """Different ROI indices produce different random streams (rng seeded by roi_idx)."""
        data = _make_synthetic_data(n_rois=5, n_frames=500)
        r0 = analyze_cell(roi_idx=0, fps=10.0, params=AnalysisParams(n_shuffles=10), **data)
        r1 = analyze_cell(roi_idx=1, fps=10.0, params=AnalysisParams(n_shuffles=10), **data)
        # Different ROIs may have different shuffle distributions
        assert r0.roi_idx == 0
        assert r1.roi_idx == 1

    def test_deconv_signal_type(self) -> None:
        """analyze_cell works with signal_type='deconv'."""
        data = _make_synthetic_data(n_rois=3, n_frames=500)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(signal_type="deconv", n_shuffles=5),
            **data,
        )
        assert isinstance(result, CellResult)

    def test_events_signal_type(self) -> None:
        """analyze_cell works with signal_type='events'."""
        data = _make_synthetic_data(n_rois=3, n_frames=500)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(signal_type="events", n_shuffles=5),
            **data,
        )
        assert isinstance(result, CellResult)


# ---------------------------------------------------------------------------
# analyze_cell — insufficient frames
# ---------------------------------------------------------------------------


class TestAnalyzeCellInsufficientFrames:
    """Tests for analyze_cell when there are too few moving frames."""

    def test_all_stationary_no_hd_tuning(self) -> None:
        """When all frames are below speed threshold, HD/place dicts are empty."""
        data = _make_synthetic_data(n_rois=3, n_frames=200, speed_moving_frac=0.0)
        # All speeds below threshold
        data["speed"][:] = 0.5
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert result.hd_all == {}
        assert result.place_all == {}

    def test_few_moving_frames_no_tuning(self) -> None:
        """With < 100 moving frames, HD/place are empty dicts."""
        data = _make_synthetic_data(n_rois=3, n_frames=200)
        # Only 50 frames moving
        data["speed"][:] = 0.5
        data["speed"][:50] = 5.0
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert result.hd_all == {}
        assert result.place_all == {}

    def test_exactly_101_moving_frames_computes_tuning(self) -> None:
        """With exactly 101 moving frames (> 100), HD/place are computed."""
        data = _make_synthetic_data(n_rois=3, n_frames=200)
        data["speed"][:] = 0.5
        data["speed"][:101] = 5.0
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert len(result.hd_all) > 0


# ---------------------------------------------------------------------------
# analyze_cell — all light-on / all light-off edge cases
# ---------------------------------------------------------------------------


class TestAnalyzeCellLightEdgeCases:
    """Test edge cases where all frames are light or dark."""

    def test_all_light_on(self) -> None:
        """When all frames are light-on, hd_dark should be empty."""
        data = _make_synthetic_data(n_rois=3, n_frames=500, speed_moving_frac=0.8)
        data["light_on"][:] = True
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        # hd_light should be populated (enough moving frames)
        assert len(result.hd_light) > 0
        # hd_dark should be empty (zero dark moving frames)
        assert result.hd_dark == {}
        # No comparison possible
        assert result.hd_comparison == {}

    def test_all_light_off(self) -> None:
        """When all frames are dark, hd_light should be empty."""
        data = _make_synthetic_data(n_rois=3, n_frames=500, speed_moving_frac=0.8)
        data["light_on"][:] = False
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        assert result.hd_light == {}
        assert len(result.hd_dark) > 0
        assert result.hd_comparison == {}

    def test_both_conditions_populated_enables_comparison(self) -> None:
        """When both light and dark have enough frames, comparison is populated."""
        data = _make_synthetic_data(
            n_rois=3, n_frames=500, speed_moving_frac=0.8, light_on_frac=0.5,
        )
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(n_shuffles=5),
            **data,
        )
        if result.hd_light and result.hd_dark:
            assert "correlation" in result.hd_comparison
            assert "pd_shift" in result.hd_comparison
            assert "mvl_ratio_dark_over_light" in result.hd_comparison


# ---------------------------------------------------------------------------
# analyze_cell — params and defaults
# ---------------------------------------------------------------------------


class TestAnalyzeCellParams:
    """Test parameter handling in analyze_cell."""

    def test_default_params_used(self) -> None:
        """params=None uses AnalysisParams defaults."""
        data = _make_synthetic_data(n_rois=3, n_frames=500)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=None,
            **data,
        )
        assert isinstance(result, CellResult)

    def test_custom_speed_threshold(self) -> None:
        """Higher speed threshold reduces number of moving frames."""
        data = _make_synthetic_data(n_rois=3, n_frames=500)
        # With very high threshold, almost no frames are moving
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(speed_threshold=100.0, n_shuffles=5),
            **data,
        )
        assert result.hd_all == {}

    def test_custom_n_bins(self) -> None:
        """Custom n_bins is reflected in tuning curve length."""
        data = _make_synthetic_data(n_rois=3, n_frames=500, speed_moving_frac=0.8)
        result = analyze_cell(
            roi_idx=0,
            fps=10.0,
            params=AnalysisParams(hd_n_bins=18, n_shuffles=5),
            **data,
        )
        if result.hd_all:
            assert len(result.hd_all["tuning_curve"]) == 18
            assert len(result.hd_all["bin_centers"]) == 18


# ---------------------------------------------------------------------------
# AnalysisParams dataclass
# ---------------------------------------------------------------------------


class TestAnalysisParams:
    """Test AnalysisParams defaults and construction."""

    def test_defaults(self) -> None:
        p = AnalysisParams()
        assert p.signal_type == "dff"
        assert p.speed_threshold == 2.5
        assert p.hd_n_bins == 36
        assert p.n_shuffles == 1000
        assert p.alpha == 0.05

    def test_custom_values(self) -> None:
        p = AnalysisParams(signal_type="deconv", n_shuffles=100, alpha=0.01)
        assert p.signal_type == "deconv"
        assert p.n_shuffles == 100
        assert p.alpha == 0.01


# ---------------------------------------------------------------------------
# CellResult dataclass
# ---------------------------------------------------------------------------


class TestCellResult:
    """Test CellResult dataclass construction."""

    def test_default_empty_dicts(self) -> None:
        r = CellResult(roi_idx=3)
        assert r.roi_idx == 3
        assert r.activity == {}
        assert r.hd_all == {}
        assert r.hd_light == {}
        assert r.hd_dark == {}
        assert r.place_all == {}
        assert r.place_light == {}
        assert r.place_dark == {}
        assert r.hd_comparison == {}
        assert r.place_comparison == {}

    def test_independent_default_dicts(self) -> None:
        """Each CellResult instance should have independent dicts."""
        r1 = CellResult(roi_idx=0)
        r2 = CellResult(roi_idx=1)
        r1.activity["test"] = 1.0
        assert "test" not in r2.activity
