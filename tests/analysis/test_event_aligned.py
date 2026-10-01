"""Tests for hm2p.analysis.event_aligned — event-aligned responses and shuffle tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis.event_aligned import (
    _as_events,
    _binom_greater,
    _delta_matrix,
    _draw_shifts,
    _drop_in_mask,
    _nanmean,
    _seconds_to_frames,
    _shuffle_stats,
    _window_frames,
    cell_event_table,
    junction_events,
    light_events,
    movement_events,
    peri_event_response,
    population_peri_event,
    session_summary,
    shuffle_test,
    turn_events,
)

FPS = 10.0
N = 3000


def _events_and_signals(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two cells: cell 0 fires after every event, cell 1 is noise."""
    rng = np.random.default_rng(seed)
    events = np.sort(rng.choice(np.arange(100, N - 100), size=25, replace=False))
    sig = rng.random((2, N)) * 0.2
    for e in events:
        sig[0, e : e + 10] += 1.0
    return events, sig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_nanmean_all_nan(self) -> None:
        assert np.isnan(_nanmean(np.array([np.nan, np.nan])))

    def test_seconds_to_frames(self) -> None:
        assert _seconds_to_frames(1.0, 9.6, "x") == 10
        with pytest.raises(ValueError):
            _seconds_to_frames(1.0, 0.0, "x")
        with pytest.raises(ValueError):
            _seconds_to_frames(-1.0, 10.0, "x")

    def test_as_events_sorted_unique(self) -> None:
        np.testing.assert_array_equal(_as_events([5, 1, 5, 3]), [1, 3, 5])

    def test_drop_in_mask(self) -> None:
        ev = np.array([1, 3, 20])
        ex = np.zeros(10, dtype=bool)
        ex[3] = True
        np.testing.assert_array_equal(_drop_in_mask(ev, ex), [1, 20])
        np.testing.assert_array_equal(_drop_in_mask(ev, None), ev)

    def test_window_frames_validation(self) -> None:
        assert _window_frames(10.0, 2.0, 2.0, 0.5) == (20, 20, 5)
        with pytest.raises(ValueError):
            _window_frames(10.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            _window_frames(10.0, 1.0, 0.0, 0.0)

    def test_delta_matrix_values_and_edges(self) -> None:
        sig = np.zeros((1, 50))
        sig[0, 20:25] = 2.0
        d = _delta_matrix(sig, np.array([2, 20, 48]), pre=5, post=5, gap=0)
        assert d.shape == (1, 1)  # events at 2 and 48 leave the recording
        assert d[0, 0] == pytest.approx(2.0)
        assert _delta_matrix(sig, np.array([0]), 5, 5, 0).shape == (1, 0)

    def test_delta_matrix_gap_excludes_frames(self) -> None:
        sig = np.zeros((1, 30))
        sig[0, 8:10] = 5.0  # inside the gap
        d = _delta_matrix(sig, np.array([10]), pre=5, post=5, gap=2)
        assert d[0, 0] == pytest.approx(0.0)

    def test_draw_shifts_respects_min_shift(self) -> None:
        rng = np.random.default_rng(0)
        s = _draw_shifts(200, 2000, 30, rng)
        assert s.min() >= 30
        assert s.max() <= 170
        with pytest.raises(ValueError):
            _draw_shifts(50, 10, 30, rng)

    def test_binom_greater(self) -> None:
        assert np.isnan(_binom_greater(0, 0))
        assert _binom_greater(10, 20) < 1e-6
        assert _binom_greater(0, 20) == pytest.approx(1.0)

    def test_shuffle_stats_too_few_events(self) -> None:
        sig = np.random.default_rng(0).random((2, 500))
        res = _shuffle_stats(
            sig, np.array([100, 200]), FPS, 20, 5.0, np.random.default_rng(0), 1.0, 1.0, 0.0
        )
        assert np.all(np.isnan(res["p"]))
        assert np.all(np.isnan(res["observed"]))
        assert list(res["n_events"]) == [2, 2]


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


class TestLightEvents:
    def test_transitions(self) -> None:
        light = np.array([1, 1, 0, 0, 1, 1, 0], dtype=bool)
        ev = light_events(light)
        np.testing.assert_array_equal(ev["light_off"], [2, 6])
        np.testing.assert_array_equal(ev["light_on"], [4])

    def test_bad_behav_drops(self) -> None:
        light = np.array([1, 1, 0, 0, 1, 1, 0], dtype=bool)
        bad = np.zeros(7, dtype=bool)
        bad[6] = True
        ev = light_events(light, bad)
        np.testing.assert_array_equal(ev["light_off"], [2])

    def test_constant_light(self) -> None:
        ev = light_events(np.ones(20, dtype=bool))
        assert ev["light_on"].size == 0
        assert ev["light_off"].size == 0


class TestMovementEvents:
    def test_onsets_and_offsets(self) -> None:
        # still 20, move 20, still 20, move 5 (too short), still 20
        act = np.concatenate(
            [np.zeros(20), np.ones(20), np.zeros(20), np.ones(5), np.zeros(20)]
        ).astype(bool)
        ev = movement_events(act, FPS, min_still_s=1.0, min_move_s=1.0)
        np.testing.assert_array_equal(ev["move_onset"], [20])
        np.testing.assert_array_equal(ev["move_offset"], [40])

    def test_short_still_rejected(self) -> None:
        act = np.concatenate([np.ones(20), np.zeros(3), np.ones(20)]).astype(bool)
        ev = movement_events(act, FPS)
        assert ev["move_onset"].size == 0
        assert ev["move_offset"].size == 0

    def test_bad_behav_in_bout_drops(self) -> None:
        act = np.concatenate([np.zeros(20), np.ones(20), np.zeros(20)]).astype(bool)
        bad = np.zeros(act.size, dtype=bool)
        bad[15] = True  # inside the still bout before the onset
        ev = movement_events(act, FPS, bad_behav=bad)
        assert ev["move_onset"].size == 0
        np.testing.assert_array_equal(ev["move_offset"], [40])

    def test_empty(self) -> None:
        ev = movement_events(np.zeros(0, dtype=bool), FPS)
        assert ev["move_onset"].size == 0


class TestTurnEvents:
    def test_left_and_right(self) -> None:
        ahv = np.zeros(100)
        ahv[30:33] = 200.0  # CW -> right
        ahv[60:63] = -150.0  # CCW -> left
        ev = turn_events(ahv, FPS, threshold_deg_s=90.0, quiet_s=1.0)
        np.testing.assert_array_equal(ev["turn_right"], [30])
        np.testing.assert_array_equal(ev["turn_left"], [60])

    def test_quiet_period_required(self) -> None:
        ahv = np.zeros(100)
        ahv[30:33] = 200.0
        ahv[36:39] = 200.0  # only 3 quiet frames before
        ev = turn_events(ahv, FPS, quiet_s=1.0)
        np.testing.assert_array_equal(ev["turn_right"], [30])

    def test_nan_breaks_quiet(self) -> None:
        ahv = np.zeros(100)
        ahv[25] = np.nan
        ahv[30] = 200.0
        ev = turn_events(ahv, FPS, quiet_s=1.0)
        assert ev["turn_right"].size == 0

    def test_nan_onset_not_a_turn(self) -> None:
        ahv = np.full(50, np.nan)
        ev = turn_events(ahv, FPS)
        assert ev["turn_left"].size == 0 and ev["turn_right"].size == 0

    def test_mask(self) -> None:
        ahv = np.zeros(100)
        ahv[30] = 200.0
        mask = np.ones(100, dtype=bool)
        mask[30] = False
        assert turn_events(ahv, FPS, mask=mask)["turn_right"].size == 0

    def test_too_early(self) -> None:
        ahv = np.zeros(50)
        ahv[3] = 200.0
        assert turn_events(ahv, FPS, quiet_s=1.0)["turn_right"].size == 0

    def test_bad_threshold(self) -> None:
        with pytest.raises(ValueError):
            turn_events(np.zeros(10), FPS, threshold_deg_s=0.0)


class TestJunctionEvents:
    def _maze(self):
        from hm2p.maze.topology import build_rose_maze

        return build_rose_maze()

    def _path_xy(self, maze, path, frames_per_cell: int = 5) -> tuple[np.ndarray, np.ndarray]:
        xy = np.repeat(np.array(path, dtype=float) + 0.5, frames_per_cell, axis=0)
        return xy[:, 0], xy[:, 1]

    def test_junction_and_dead_end_entries(self) -> None:
        maze = self._maze()
        # walk (2,0)->(1,0)->(0,0): (1,0) is a T-junction, (0,0) a dead end
        assert maze.node_types[(1, 0)] in ("t_junction", "crossroads")
        assert maze.node_types[(0, 0)] == "dead_end"
        path = [(2, 0), (1, 0), (0, 0), (1, 0), (1, 1)]
        x, y = self._path_xy(maze, path)
        ev = junction_events(x, y, np.ones(x.size, dtype=bool), maze)
        np.testing.assert_array_equal(ev["dead_end_entry"], [10])
        np.testing.assert_array_equal(ev["junction_entry"], [5, 15])

    def test_mask_blocks_entry(self) -> None:
        maze = self._maze()
        path = [(2, 0), (1, 0), (0, 0)]
        x, y = self._path_xy(maze, path)
        mask = np.ones(x.size, dtype=bool)
        mask[5:10] = False  # junction frames invalid
        ev = junction_events(x, y, mask, maze)
        assert ev["dead_end_entry"].size == 0
        assert ev["junction_entry"].size == 0

    def test_default_maze_and_empty(self) -> None:
        ev = junction_events(np.zeros(0), np.zeros(0), None)
        assert ev["junction_entry"].size == 0
        assert ev["dead_end_entry"].size == 0


# ---------------------------------------------------------------------------
# Responses and shuffle test
# ---------------------------------------------------------------------------


class TestPeriEventResponse:
    def test_step_response(self) -> None:
        sig = np.zeros(200)
        for e in (50, 100, 150):
            sig[e : e + 20] = 1.0
        res = peri_event_response(sig, [50, 100, 150], FPS, pre_s=2.0, post_s=2.0)
        assert res["n_events"] == 3
        assert res["mean_delta"] == pytest.approx(1.0)
        assert res["time_s"][0] == pytest.approx(-2.0)
        assert res["mean_trace"].shape == (40,)

    def test_edge_events_dropped(self) -> None:
        sig = np.ones(100)
        res = peri_event_response(sig, [5, 50, 95], FPS, pre_s=1.0, post_s=1.0)
        assert res["n_events"] == 1

    def test_no_events(self) -> None:
        res = peri_event_response(np.ones(100), [], FPS)
        assert res["n_events"] == 0
        assert np.isnan(res["mean_delta"])

    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError):
            peri_event_response(np.ones((2, 10)), [5], FPS)

    def test_matches_vectorised(self) -> None:
        events, sig = _events_and_signals()
        res = peri_event_response(sig[0], events, FPS, gap_s=0.3)
        d = _delta_matrix(sig, events, 20, 20, 3)
        np.testing.assert_allclose(res["delta"], d[0])

    @settings(max_examples=30, deadline=None)
    @given(st.floats(min_value=-5, max_value=5, allow_nan=False))
    def test_constant_offset_invariant(self, c: float) -> None:
        events, sig = _events_and_signals()
        a = peri_event_response(sig[0], events, FPS)["mean_delta"]
        b = peri_event_response(sig[0] + c, events, FPS)["mean_delta"]
        assert a == pytest.approx(b, abs=1e-9)


class TestShuffleTest:
    def test_responsive_cell_significant(self) -> None:
        events, sig = _events_and_signals()
        res = shuffle_test(sig[0], events, FPS, n_shuffles=200, rng=np.random.default_rng(1))
        assert res["p"] < 0.05
        assert res["z"] > 3
        assert res["n_events"] == 25

    def test_noise_cell_not_significant(self) -> None:
        events, sig = _events_and_signals()
        res = shuffle_test(sig[1], events, FPS, n_shuffles=200, rng=np.random.default_rng(1))
        assert res["p"] > 0.05

    def test_inhibited_cell(self) -> None:
        events, sig = _events_and_signals()
        res = shuffle_test(-sig[0], events, FPS, n_shuffles=200, rng=np.random.default_rng(1))
        assert res["p"] < 0.05
        assert res["z"] < -3

    def test_few_events_nan(self) -> None:
        res = shuffle_test(np.ones(500), [100, 200], FPS, n_shuffles=10)
        assert np.isnan(res["z"]) and np.isnan(res["p"])
        assert res["n_events"] == 2

    def test_too_short_recording_raises(self) -> None:
        with pytest.raises(ValueError):
            shuffle_test(np.ones(150), [30, 60, 90], FPS, n_shuffles=10, min_shift_s=10.0)

    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError):
            shuffle_test(np.ones((2, 10)), [5], FPS)

    def test_p_bounds(self) -> None:
        events, sig = _events_and_signals()
        res = shuffle_test(sig[0], events, FPS, n_shuffles=50, rng=np.random.default_rng(0))
        assert 1 / 51 <= res["p"] <= 1.0

    def test_zero_shuffles_gives_observed_only(self) -> None:
        events, sig = _events_and_signals()
        res = shuffle_test(sig[0], events, FPS, n_shuffles=0)
        assert np.isfinite(res["observed"])
        assert np.isnan(res["p"])

    def test_periodic_events_conservative(self) -> None:
        # events every 100 frames: shifts near multiples of 100 re-align the train,
        # so the null contains the true response and p is inflated
        n = 3000
        events = np.arange(100, n - 100, 100)
        sig = np.random.default_rng(0).random(n) * 0.2
        for e in events:
            sig[e : e + 10] += 1.0
        periodic = shuffle_test(sig, events, FPS, n_shuffles=300, rng=np.random.default_rng(0))
        jitter = np.random.default_rng(1).integers(-40, 40, events.size)
        irregular = events + jitter
        sig2 = np.random.default_rng(0).random(n) * 0.2
        for e in irregular:
            sig2[e : e + 10] += 1.0
        irr = shuffle_test(sig2, irregular, FPS, n_shuffles=300, rng=np.random.default_rng(0))
        assert periodic["p"] > irr["p"]
        assert periodic["z"] < irr["z"]

    def test_default_rng(self) -> None:
        events, sig = _events_and_signals()
        res = shuffle_test(sig[0], events, FPS, n_shuffles=20)
        assert np.isfinite(res["p"])


class TestCellEventTable:
    def test_table(self) -> None:
        events, sig = _events_and_signals()
        tab = cell_event_table(
            sig,
            {"a": events, "b": np.array([500])},
            FPS,
            n_shuffles=100,
            rng=np.random.default_rng(0),
        )
        assert len(tab) == 4
        a = tab[tab["event_type"] == "a"].set_index("roi")
        assert bool(a.loc[0, "responsive"]) and a.loc[0, "direction"] == 1
        assert not bool(a.loc[1, "responsive"])
        b = tab[tab["event_type"] == "b"]
        assert not b["responsive"].any()
        assert (b["direction"] == 0).all()

    def test_one_d_and_empty(self) -> None:
        events, sig = _events_and_signals()
        tab = cell_event_table(sig[0], {"a": events}, FPS, n_shuffles=20)
        assert len(tab) == 1
        empty = cell_event_table(sig, {}, FPS)
        assert empty.empty and "responsive" in empty.columns

    def test_rejects_3d(self) -> None:
        with pytest.raises(ValueError):
            cell_event_table(np.ones((1, 2, 3)), {"a": [1]}, FPS)


class TestPopulationPeriEvent:
    def test_mean_over_cells(self) -> None:
        sig = np.zeros((2, 200))
        sig[0, 100:110] = 2.0
        res = population_peri_event(sig, [100], FPS, pre_s=1.0, post_s=1.0)
        assert res["mean"].shape == (20,)
        assert res["mean"][10] == pytest.approx(1.0)
        assert res["n_cells"] == 2 and res["n_events"] == 1
        assert np.isfinite(res["sem"][10])

    def test_single_cell_sem_nan(self) -> None:
        res = population_peri_event(np.ones(100), [50], FPS, pre_s=1.0, post_s=1.0)
        assert np.all(np.isnan(res["sem"]))

    def test_no_events(self) -> None:
        res = population_peri_event(np.ones((2, 100)), [], FPS, pre_s=1.0, post_s=1.0)
        assert res["n_events"] == 0
        assert np.all(np.isnan(res["mean"]))


class TestSessionSummary:
    def test_values(self) -> None:
        tab = pd.DataFrame(
            {
                "roi": [0, 1, 2, 3, 0],
                "event_type": ["a", "a", "a", "a", "b"],
                "observed": [1.0, -1.0, 0.0, np.nan, 0.5],
                "z": [3.0, -3.0, 0.1, np.nan, 1.0],
                "p": [0.01, 0.02, 0.9, np.nan, 0.3],
                "n_events": [10, 10, 10, 1, 4],
                "responsive": [True, True, False, False, False],
                "direction": [1, -1, 1, 0, 1],
            }
        )
        out = session_summary(tab).set_index("event_type")
        assert out.loc["a", "n_cells"] == 3
        assert out.loc["a", "n_responsive"] == 2
        assert out.loc["a", "frac_responsive"] == pytest.approx(2 / 3)
        assert out.loc["a", "frac_excited"] == pytest.approx(1 / 3)
        assert out.loc["a", "frac_inhibited"] == pytest.approx(1 / 3)
        assert out.loc["a", "median_z"] == pytest.approx(0.1)
        assert out.loc["a", "median_observed"] == pytest.approx(0.0)
        assert out.loc["a", "binom_p"] < 0.01
        assert out.loc["b", "frac_responsive"] == 0.0

    def test_all_untested(self) -> None:
        tab = pd.DataFrame(
            {
                "roi": [0],
                "event_type": ["a"],
                "observed": [np.nan],
                "z": [np.nan],
                "p": [np.nan],
                "n_events": [1],
                "responsive": [False],
                "direction": [0],
            }
        )
        out = session_summary(tab)
        assert out.loc[0, "n_cells"] == 0
        assert np.isnan(out.loc[0, "frac_responsive"])
        assert np.isnan(out.loc[0, "binom_p"])

    def test_empty(self) -> None:
        out = session_summary(pd.DataFrame())
        assert out.empty and "frac_responsive" in out.columns
