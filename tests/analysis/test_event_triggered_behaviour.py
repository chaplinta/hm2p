"""Tests for ``hm2p.analysis.event_triggered_behaviour`` (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import event_triggered_behaviour as etb

FPS = 9.6


def _speed_drop_session(
    seed: int = 0, n_frames: int = 6000, n_events: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Speed trace with brief drops and a dF/F trace with a transient at each drop."""
    rng = np.random.default_rng(seed)
    speed = np.clip(rng.normal(10.0, 1.0, n_frames), 0.0, None)
    drops = np.sort(rng.choice(np.arange(100, n_frames - 100, 60), n_events, replace=False))
    for d in drops:
        speed[d : d + 20] = 0.5
    dff = rng.normal(0.0, 0.03, n_frames)
    kern = np.exp(-np.arange(40) / 15.0)
    for d in drops:
        dff[d : d + 40] += kern
    return speed, dff, drops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_nanmean_all_nan(self) -> None:
        assert np.isnan(etb._nanmean(np.array([np.nan, np.nan])))

    def test_null_summary(self) -> None:
        null = np.array([-1.0, 0.0, 1.0, 0.0])
        mu, z, p = etb._null_summary(5.0, null)
        assert mu == 0.0
        assert z > 0
        assert p == pytest.approx(1 / 5)

    def test_null_summary_empty_and_constant(self) -> None:
        assert all(np.isnan(v) for v in etb._null_summary(1.0, np.array([np.nan])))
        mu, z, p = etb._null_summary(1.0, np.array([2.0]))
        assert mu == 2.0 and np.isnan(z) and p == 0.5

    @given(st.floats(-1e4, 1e4, allow_nan=False))
    @settings(max_examples=50, deadline=None)
    def test_wrap_deg_range(self, a: float) -> None:
        w = float(etb.wrap_deg(a))
        assert -180.0 <= w < 180.0
        assert np.isclose(np.cos(np.deg2rad(w)), np.cos(np.deg2rad(a)), atol=1e-6)

    def test_window_to_frames(self) -> None:
        assert etb._window_to_frames((0.0, 2.0), 10.0, "w") == (0, 20)
        with pytest.raises(ValueError):
            etb._window_to_frames((2.0, 1.0), 10.0, "w")
        with pytest.raises(ValueError):
            etb._window_to_frames((0.0, 0.01), 10.0, "w")

    def test_window_means(self) -> None:
        x = np.array([1.0, np.nan, 3.0, 5.0])
        fin = np.isfinite(x)
        cs = np.concatenate([[0.0], np.cumsum(np.where(fin, x, 0))])
        cc = np.concatenate([[0], np.cumsum(fin)])
        out = etb._window_means(cs, cc, np.array([0, 1]), np.array([3, 2]))
        assert out[0] == pytest.approx(2.0)
        assert np.isnan(out[1])

    def test_circular_correlation_matches_roll(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=37)
        b = (rng.random(37) > 0.7).astype(float)
        cc = etb._circular_correlation(a, b)
        for s in (0, 3, 20):
            assert cc[s] == pytest.approx(np.sum(a * np.roll(b, s)))


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class TestHeadBody:
    def test_bearing_matches_kinematics_convention(self) -> None:
        from hm2p.kinematics.compute import _vector_angle_deg

        rng = np.random.default_rng(0)
        xb, yb, xh, yh = rng.normal(size=(4, 20))
        ours = etb.body_head_bearing_deg(xb, yb, xh, yh)
        ref = np.mod(_vector_angle_deg(xb, yb, xh, yh), 360.0)
        np.testing.assert_allclose(ours, ref)

    def test_bearing_axes(self) -> None:
        # +y body->head is 90 deg, +x is 180 deg in the pipeline convention
        assert etb.body_head_bearing_deg(0, 0, 0, 1) == pytest.approx(90.0)
        assert etb.body_head_bearing_deg(0, 0, 1, 0) == pytest.approx(180.0)

    def test_sign_convention(self) -> None:
        n = 4
        arrays = _base_arrays(n)
        arrays.update(
            x_body_mm=np.zeros(n),
            y_body_mm=np.zeros(n),
            x_head_mm=np.zeros(n),
            y_head_mm=np.ones(n),  # bearing 90
            hd_deg=np.array([90.0, 120.0, 60.0, 450.0 - 200.0]),
        )
        ch = etb.behaviour_channels(arrays)
        np.testing.assert_allclose(ch["head_body_angle_deg"], [0.0, 30.0, -30.0, 160.0])
        np.testing.assert_allclose(ch["abs_head_body_angle_deg"], [0.0, 30.0, 30.0, 160.0])

    def test_absent_positions(self) -> None:
        ch = etb.behaviour_channels(_base_arrays(10))
        assert "head_body_angle_deg" not in ch


def _base_arrays(n: int) -> dict:
    return {
        "speed_cm_s": np.arange(n, dtype=float),
        "ahv_deg_s": -np.arange(n, dtype=float),
        "active": np.ones(n, dtype=bool),
        "light_on": np.zeros(n, dtype=bool),
        "fps": 1.0,
    }


class TestTimeInEpoch:
    def test_basic(self) -> None:
        lo = np.array([1, 1, 0, 0, 0, 1, 1], dtype=bool)
        out = etb.time_in_light_epoch_s(lo, fps=2.0)
        assert np.isnan(out[:2]).all()
        np.testing.assert_allclose(out[2:], [0.0, 0.5, 1.0, 0.0, 0.5])

    def test_no_transitions_and_short(self) -> None:
        assert np.isnan(etb.time_in_light_epoch_s(np.ones(5, bool), 1.0)).all()
        assert etb.time_in_light_epoch_s(np.ones(1, bool), 1.0).size == 1

    def test_bad_fps(self) -> None:
        with pytest.raises(ValueError):
            etb.time_in_light_epoch_s(np.ones(5, bool), 0.0)


class TestNodeTypes:
    def test_one_hot(self) -> None:
        from hm2p.maze.topology import build_rose_maze

        maze = build_rose_maze()
        picks = {}
        for cell, t in maze.node_types.items():
            picks.setdefault(etb.NODE_GROUPS[t], cell)
        assert set(picks) == {"junction", "corridor", "dead_end"}
        order = ["junction", "corridor", "dead_end"]
        x = np.array([picks[g][0] + 0.5 for g in order] + [np.nan])
        y = np.array([picks[g][1] + 0.5 for g in order] + [np.nan])
        ch = etb.node_type_channels(x, y, maze)
        np.testing.assert_array_equal(ch["node_junction"][:3], [1, 0, 0])
        np.testing.assert_array_equal(ch["node_corridor"][:3], [0, 1, 0])
        np.testing.assert_array_equal(ch["node_dead_end"][:3], [0, 0, 1])
        for v in ch.values():
            assert np.isnan(v[3])
        total = ch["node_junction"][:3] + ch["node_corridor"][:3] + ch["node_dead_end"][:3]
        np.testing.assert_array_equal(total, 1.0)

    def test_crossroads_grouped_as_junction(self) -> None:
        assert etb.NODE_GROUPS["crossroads"] == etb.NODE_GROUPS["t_junction"] == "junction"

    def test_default_maze(self) -> None:
        ch = etb.node_type_channels(np.array([0.5]), np.array([0.5]))
        assert sum(float(v[0]) for v in ch.values()) == 1.0


class TestBehaviourChannels:
    def test_core_channels_and_bad_behav(self) -> None:
        n = 6
        arr = _base_arrays(n)
        arr["dff"] = np.zeros((2, n - 1))  # frame count from dff
        arr["bad_behav"] = np.array([0, 1, 0, 0, 0, 0], dtype=bool)
        arr["x_maze"] = np.full(n, 0.5)
        arr["y_maze"] = np.full(n, 0.5)
        ch = etb.behaviour_channels(arr)
        assert {"speed_cm_s", "abs_ahv", "active", "light_on", "time_in_light_epoch_s"} <= set(ch)
        assert {"node_junction", "node_corridor", "node_dead_end"} <= set(ch)
        assert "syllable_id" not in ch
        for v in ch.values():
            assert v.size == n - 1
            assert np.isnan(v[1])
        np.testing.assert_allclose(ch["abs_ahv"][[0, 2]], [0.0, 2.0])

    def test_input_not_modified(self) -> None:
        arr = _base_arrays(4)
        arr["bad_behav"] = np.array([1, 0, 0, 0], dtype=bool)
        etb.behaviour_channels(arr)
        assert arr["speed_cm_s"][0] == 0.0


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_detects_planted(self) -> None:
        _, dff, drops = _speed_drop_session()
        on, dur = etb.event_onsets(dff, FPS)
        assert on.size == drops.size
        assert np.all(np.abs(on - drops) <= 3)
        assert np.all(dur > 1.0)

    def test_min_duration_filters(self) -> None:
        _, dff, _ = _speed_drop_session()
        on, _ = etb.event_onsets(dff, FPS, min_duration_s=100.0)
        assert on.size == 0

    def test_bounds_and_mask(self) -> None:
        mask = etb.event_mask_from_bounds([1, 8], [3, 20], 10)
        np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2, 8, 9])
        assert not etb.event_mask_from_bounds([], [], 5).any()


class TestETA:
    def test_shape_and_edges(self) -> None:
        x = np.arange(100, dtype=float)
        res = etb.event_triggered_average(x, [2, 50, 98], fps=1.0, pre_s=5, post_s=5)
        assert res["n"] == 1
        assert res["time_s"][0] == -5.0 and res["time_s"].size == 10
        np.testing.assert_allclose(res["mean"], np.arange(45, 55))

    def test_nan_aware(self) -> None:
        x = np.ones(100)
        x[48] = np.nan
        res = etb.event_triggered_average(x, [50, 60], fps=1.0, pre_s=5, post_s=5)
        assert np.all(res["mean"] == 1.0)

    def test_no_events(self) -> None:
        res = etb.event_triggered_average(np.ones(10), [], fps=1.0)
        assert res["n"] == 0 and np.isnan(res["mean"]).all()


# ---------------------------------------------------------------------------
# Shuffle tests
# ---------------------------------------------------------------------------


class TestOnsetShuffle:
    def test_speed_drop_negative_significant(self) -> None:
        speed, dff, _ = _speed_drop_session()
        on, _ = etb.event_onsets(dff, FPS)
        res = etb.onset_vs_shuffle(speed, on, FPS, n_shuffles=200, rng=np.random.default_rng(0))
        assert res["observed"] < 0
        assert res["z"] < -2
        assert res["p"] < 0.05
        assert res["n"] == on.size

    def test_random_events_not_significant(self) -> None:
        speed, _, _ = _speed_drop_session()
        rng = np.random.default_rng(3)
        onsets = np.sort(rng.choice(speed.size, 40, replace=False))
        res = etb.onset_vs_shuffle(speed, onsets, FPS, n_shuffles=200, rng=rng)
        assert res["p"] > 0.05

    def test_occupancy_mode_is_window_mean(self) -> None:
        x = np.zeros(1000)
        on = np.arange(100, 900, 100)
        for o in on:
            x[o : o + 10] = 1.0
        res = etb.onset_vs_shuffle(
            x, on, fps=10.0, window=(0.0, 1.0), baseline=None, n_shuffles=50,
            rng=np.random.default_rng(0),
        )  # fmt: skip
        assert res["observed"] == pytest.approx(1.0)
        assert res["p"] < 0.05

    def test_edge_events_dropped_and_too_few(self) -> None:
        x = np.ones(500)
        res = etb.onset_vs_shuffle(x, [0, 1, 499, 250], fps=10.0, n_shuffles=10)
        assert res["n"] == 1
        assert np.isnan(res["observed"]) and np.isnan(res["p"])

    def test_zero_shuffles(self) -> None:
        speed, dff, drops = _speed_drop_session()
        res = etb.onset_vs_shuffle(speed, drops, FPS, n_shuffles=0)
        assert np.isfinite(res["observed"]) and np.isnan(res["p"])

    def test_short_recording_raises(self) -> None:
        with pytest.raises(ValueError):
            etb.onset_vs_shuffle(np.ones(100), [60, 70, 80], fps=10.0, n_shuffles=5, baseline=None)

    def test_tiny_recording_all_events_outside(self) -> None:
        res = etb.onset_vs_shuffle(np.ones(20), [5, 10, 15], fps=10.0, n_shuffles=5)
        assert res["n"] == 0


class TestDuringShuffle:
    def test_speed_lower_during_events(self) -> None:
        speed, dff, _ = _speed_drop_session()
        on, off = etb.event_bounds(dff, FPS)
        res = etb.during_event_shuffle(
            speed, on, off, FPS, n_shuffles=200, rng=np.random.default_rng(0)
        )
        assert res["during"] < res["outside"]
        assert res["observed"] == pytest.approx(res["during"] - res["outside"])
        assert res["z"] < -2 and res["p"] < 0.05
        assert etb.during_event_mean(speed, on, off) == pytest.approx(res["during"])

    def test_null_matches_explicit_roll(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=300)
        x[10] = np.nan
        on, off = np.array([20, 100, 200]), np.array([30, 120, 205])
        res = etb.during_event_shuffle(x, on, off, fps=1.0, n_shuffles=1,
                                       min_shift_s=50, rng=np.random.default_rng(7))  # fmt: skip
        s = int(np.random.default_rng(7).integers(50, 251, size=1)[0])
        mask = np.roll(etb.event_mask_from_bounds(on, off, x.size), s)
        expected = np.nanmean(x[mask]) - np.nanmean(x[~mask])
        assert res["null_mean"] == pytest.approx(expected)

    def test_random_mask_not_significant(self) -> None:
        speed, _, _ = _speed_drop_session()
        rng = np.random.default_rng(5)
        on = np.sort(rng.choice(np.arange(0, speed.size - 30, 50), 40, replace=False))
        res = etb.during_event_shuffle(speed, on, on + 25, FPS, n_shuffles=200, rng=rng)
        assert res["p"] > 0.05

    def test_too_few_events_and_zero_shuffles(self) -> None:
        res = etb.during_event_shuffle(np.ones(100), [1], [3], fps=1.0)
        assert res["n"] == 1 and np.isnan(res["observed"])
        res = etb.during_event_shuffle(np.arange(100.0), [1, 20, 40], [5, 25, 45], 1.0, 0)
        assert np.isfinite(res["observed"]) and np.isnan(res["p"])

    def test_during_mean_empty(self) -> None:
        assert np.isnan(etb.during_event_mean(np.ones(10), [], []))


class TestSyllables:
    def test_planted_syllable_enriched(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        syl = rng.integers(0, 5, n)
        on = np.sort(rng.choice(np.arange(n), 60, replace=False))
        syl[on] = 3
        syl[:5] = -1
        tab = etb.syllable_enrichment(syl, on, n_shuffles=200, rng=rng, fps=FPS)
        assert set(tab["syllable"]) == {0, 1, 2, 3, 4}
        row = tab.set_index("syllable").loc[3]
        assert row["observed_frac"] > 0.9
        assert row["z"] > 3 and row["p"] < 0.05
        assert tab["observed_frac"].sum() == pytest.approx(1.0)
        others = tab[tab["syllable"] != 3]
        assert (others["z"] < 0).all()

    def test_empty_and_missing(self) -> None:
        assert etb.syllable_enrichment(np.full(100, -1), [1, 2, 3], 10).empty
        assert etb.syllable_enrichment(np.zeros(100), [1], 10).empty
        nan_syl = np.full(2000, np.nan)
        nan_syl[:1000] = 1
        tab = etb.syllable_enrichment(nan_syl, [1, 2, 3, 1500], 0)
        assert tab["n_onsets"].tolist() == [3]
        assert np.isnan(tab["p"]).all()


class TestDurationCorrelation:
    def test_long_events_during_stillness(self) -> None:
        n = 3000
        speed = np.full(n, 10.0)
        on = np.arange(100, 2900, 200)
        dur = np.linspace(0.5, 5.0, on.size)
        for o, d in zip(on, dur, strict=True):
            speed[o : o + int(d * FPS)] = 10.0 / (1 + d)
        res = etb.duration_behaviour_correlation(dur, on, speed, FPS)
        assert res["rho"] < -0.9 and res["p"] < 0.05 and res["n"] == on.size

    def test_degenerate(self) -> None:
        res = etb.duration_behaviour_correlation([1.0, 1.0], [1, 5], np.ones(20), 1.0)
        assert np.isnan(res["rho"])
        res = etb.duration_behaviour_correlation(np.ones(6), np.arange(6), np.arange(20.0), 1.0)
        assert np.isnan(res["rho"])
        with pytest.raises(ValueError):
            etb.duration_behaviour_correlation([1.0], [1, 2], np.ones(5), 1.0)


# ---------------------------------------------------------------------------
# Cell table
# ---------------------------------------------------------------------------


class TestCellTable:
    def test_rows_and_signs(self) -> None:
        speed, dff, _ = _speed_drop_session(n_frames=3000, n_events=25)
        noise = np.random.default_rng(9).normal(0, 0.03, speed.size)
        light = (np.arange(speed.size) // 576) % 2 == 0
        beh = {"speed_cm_s": speed, "light_on": light.astype(float)}
        tab = etb.cell_etb_table(
            np.vstack([dff, noise]), beh, FPS, n_shuffles=100, rng=np.random.default_rng(0)
        )
        assert list(tab.columns) == etb._CELL_COLUMNS
        assert len(tab) == 4
        r = tab[(tab["roi"] == 0) & (tab["channel"] == "speed_cm_s")].iloc[0]
        assert r["onset_z"] < 0 and r["onset_p"] < 0.05
        assert r["during_z"] < 0 and r["during_p"] < 0.05
        assert r["median_duration_s"] > 1.0
        lr = tab[(tab["roi"] == 0) & (tab["channel"] == "light_on")].iloc[0]
        assert 0.0 <= lr["onset_observed"] <= 1.0  # occupancy: fraction in light

    def test_precomputed_events_and_1d(self) -> None:
        x = np.arange(500, dtype=float)
        ev = [(np.array([100, 200, 300]), np.array([110, 210, 310]))]
        tab = etb.cell_etb_table(np.zeros(500), {"c": x}, 10.0, n_shuffles=5, events=ev)
        assert tab["n_events"].iloc[0] == 3
        assert tab["median_duration_s"].iloc[0] == pytest.approx(1.0)
        with pytest.raises(ValueError):
            etb.cell_etb_table(np.zeros((2, 500)), {"c": x}, 10.0, events=ev)

    def test_bad_shape_and_empty(self) -> None:
        with pytest.raises(ValueError):
            etb.cell_etb_table(np.zeros((1, 2, 3)), {}, FPS)
        out = etb.cell_etb_table(np.zeros((0, 100)), {"c": np.ones(100)}, FPS)
        assert out.empty and list(out.columns) == etb._CELL_COLUMNS
        assert isinstance(out, pd.DataFrame)
