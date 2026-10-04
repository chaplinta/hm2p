"""Tests for hm2p.analysis.movement_state (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.analysis import movement_state as ms
from hm2p.analysis.state_dynamics import syllable_information

FPS = 10.0


def _behaviour(n: int = 6000, seed: int = 0, block: int = 20):
    """Blocks of still (0), straight (1), turn (2), runturn (3)."""
    rng = np.random.default_rng(seed)
    blk = np.repeat(rng.integers(0, 4, n // block + 1), block)[:n]
    speed = np.where(blk % 2 == 1, rng.uniform(6, 20, n), rng.uniform(0, 0.4, n))
    sign = np.repeat(rng.choice([-1.0, 1.0], n // block + 1), block)[:n]
    ahv = np.where(blk >= 2, sign * rng.uniform(150, 250, n), rng.normal(0, 5, n))
    return blk, speed, ahv


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def test_nan_smooth_basic_and_nan() -> None:
    x = np.array([0.0, 3.0, np.nan, 3.0, 0.0])
    out = ms.nan_smooth(x, 3)
    assert out[1] == pytest.approx(1.5) and out[2] == pytest.approx(3.0)
    assert np.isnan(ms.nan_smooth(np.full(4, np.nan), 3)).all()
    np.testing.assert_array_equal(ms.nan_smooth(x, 1), x)
    assert ms.nan_smooth(np.zeros(0), 5).size == 0
    # even window is centred by widening to odd
    np.testing.assert_allclose(ms.nan_smooth(np.arange(7.0), 2)[1:-1], np.arange(1.0, 6.0))


@given(
    arrays(np.float64, st.integers(1, 60), elements=st.floats(-1e3, 1e3)),
    st.integers(1, 9),
)
@settings(max_examples=60, deadline=None)
def test_nan_smooth_bounded_by_data(x: np.ndarray, w: int) -> None:
    out = ms.nan_smooth(x, w)
    assert out.shape == x.shape
    assert np.all(out <= x.max() + 1e-9) and np.all(out >= x.min() - 1e-9)


def test_smooth_kinematics_rectifies_after_smoothing() -> None:
    n = 50
    ahv = np.where(np.arange(n) % 2 == 0, 100.0, -100.0)  # alternating jitter
    s, a, v = ms.smooth_kinematics(np.ones(n), ahv, np.ones(n, bool), FPS, smooth_s=0.5)
    assert v.all() and np.median(a) < 30  # cancels, not 100
    bad = np.ones(n, bool)
    bad[10:20] = False
    s, a, v = ms.smooth_kinematics(np.ones(n), ahv, bad, FPS)
    assert not v[10:20].any() and np.isnan(s[10:20]).all()
    with pytest.raises(ValueError):
        ms.smooth_kinematics(np.ones(3), np.ones(4), np.ones(3, bool), FPS)


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------


def test_classify_states_recovers_blocks() -> None:
    blk, speed, ahv = _behaviour()
    code = ms.classify_states(speed, np.abs(ahv), np.ones(blk.size, bool), FPS)
    assigned = code >= 0
    assert assigned.mean() > 0.95
    assert np.mean(code[assigned] == blk[assigned]) > 0.99


def test_classify_states_min_bout_and_dead_zone() -> None:
    speed = np.r_[np.zeros(2), np.full(10, 10.0), np.full(10, 2.0)]
    a = np.zeros(speed.size)
    code = ms.classify_states(speed, a, np.ones(speed.size, bool), FPS, min_bout_s=0.3)
    assert (code[:2] == -1).all()  # still bout of 2 frames < 3
    assert (code[2:12] == 1).all() and (code[12:] == -1).all()  # 2 cm/s is in dead zone
    with pytest.raises(ValueError):
        ms.classify_states(speed, a, np.ones(speed.size, bool), FPS, speed_lo=5, speed_hi=1)
    with pytest.raises(ValueError):
        ms.classify_states(speed, a, np.ones(speed.size, bool), FPS, ahv_lo=90, ahv_hi=30)


def test_classify_states_respects_invalid() -> None:
    speed = np.full(20, 10.0)
    valid = np.ones(20, bool)
    valid[5:15] = False
    code = ms.classify_states(speed, np.zeros(20), valid, FPS)
    assert (code[5:15] == -1).all() and (code[:5] == 1).all()


def test_state_summary_counts_and_rare() -> None:
    blk, speed, ahv = _behaviour(n=2000)
    v = np.ones(blk.size, bool)
    code = ms.classify_states(speed, np.abs(ahv), v, FPS)
    light = np.arange(blk.size) < 1000
    out = ms.state_summary(
        code, speed, np.abs(ahv), v, FPS, light_on=light, abs_ahv_raw=np.abs(ahv)
    )
    for name in ms.STATE_NAMES:
        assert out[f"n_{name}"] == out[f"n_{name}_light"] + out[f"n_{name}_dark"]
        assert out[f"s_{name}"] == pytest.approx(out[f"n_{name}"] / FPS)
    assert out["rare_states"] == ""
    assert np.isfinite(out["abs_ahv_raw_p50"]) and np.isfinite(out["rho_speed_ahv_running"])
    code2 = np.where(code == 2, -1, code)
    assert "turn" in ms.state_summary(code2, speed, np.abs(ahv), v, FPS)["rare_states"]


def test_state_summary_no_valid_frames() -> None:
    n = 10
    out = ms.state_summary(np.full(n, -1), np.zeros(n), np.zeros(n), np.zeros(n, bool), FPS)
    assert np.isnan(out["frac_unassigned"]) and np.isnan(out["speed_p50"])
    assert np.isnan(out["median_bout_s_still"])


# ---------------------------------------------------------------------------
# syllables
# ---------------------------------------------------------------------------


def test_syllable_table_classes() -> None:
    n_per = 60
    syl = np.repeat(np.arange(6), n_per)
    speed = np.repeat(np.arange(6, dtype=float), n_per)
    ahv = np.repeat(np.arange(6, dtype=float) * 10, n_per)
    syl = np.r_[syl, np.full(10, 9), np.full(5, -1)]  # 9 too rare, -1 missing
    speed = np.r_[speed, np.zeros(15)]
    ahv = np.r_[ahv, np.zeros(15)]
    df = ms.syllable_table(syl, speed, ahv, np.ones(syl.size, bool), min_frames=50)
    assert list(df["syllable"]) == [0, 1, 2, 3, 4, 5, 9]
    inc = df[df["included"]]
    assert list(inc["speed_class"]) == [0, 0, 1, 1, 2, 2]
    assert df.loc[df["syllable"] == 9, "speed_class"].item() == -1
    assert inc["mean_abs_ahv"].tolist() == [0, 10, 20, 30, 40, 50]


def test_syllable_table_empty_and_few() -> None:
    assert ms.syllable_table(np.full(5, -1), np.zeros(5), np.zeros(5), np.ones(5, bool)).empty
    syl = np.repeat([0, 1], 60)
    df = ms.syllable_table(syl, np.arange(120.0), np.zeros(120), np.ones(120, bool))
    assert (df["speed_class"] == -1).all()
    nan_syl = np.r_[np.full(60, np.nan), np.zeros(60)]
    df = ms.syllable_table(nan_syl, np.zeros(120), np.zeros(120), np.ones(120, bool))
    assert list(df["syllable"]) == [0]


# ---------------------------------------------------------------------------
# information
# ---------------------------------------------------------------------------


def test_quantile_codes_ties_and_nan() -> None:
    x = np.r_[np.zeros(50), np.arange(1.0, 51.0), np.nan]
    c = ms.quantile_codes(x, 10)
    assert c[-1] == -1 and len(np.unique(c[:50])) == 1
    assert (ms.quantile_codes(np.full(5, np.nan)) == -1).all()
    assert (ms.quantile_codes(np.ones(5)) == 0).all()
    with pytest.raises(ValueError):
        ms.quantile_codes(x, 1)


def test_mi_codes_matches_state_dynamics() -> None:
    rng = np.random.default_rng(1)
    syl = rng.integers(0, 6, 3000)
    x = rng.normal(size=3000) + syl * 0.5
    mine = ms.mutual_information_codes(ms.quantile_codes(x, 10), syl)
    ref = syllable_information(x, syl, np.ones(3000, bool), n_signal_bins=10)
    assert mine == pytest.approx(ref, rel=1e-9)


def test_mi_codes_edge_cases() -> None:
    assert ms.mutual_information_codes(np.array([0, -1]), np.array([0, 1])) == 0.0
    with pytest.raises(ValueError):
        ms.mutual_information_codes(np.zeros(3, int), np.zeros(4, int))
    x = np.array([0, 1] * 50)
    assert ms.mutual_information_codes(x, x) == pytest.approx(1.0)


@given(
    st.integers(2, 6),
    st.integers(2, 8),
    st.integers(0, 2**31 - 1),
)
@settings(max_examples=50, deadline=None)
def test_mi_bounds_and_data_processing(nx: int, ny: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, nx, 400)
    y = rng.integers(0, ny, 400)
    mi = ms.mutual_information_codes(x, y, nx, ny)
    assert 0.0 <= mi <= min(np.log2(nx), np.log2(ny)) + 1e-9
    # collapsing labels never increases plug-in information
    assert ms.mutual_information_codes(x, y // 2, nx, ny) <= mi + 1e-9
    # relabelling the categories leaves it unchanged
    perm = rng.permutation(ny)
    assert ms.mutual_information_codes(x, perm[y], nx, ny) == pytest.approx(mi, abs=1e-9)


# ---------------------------------------------------------------------------
# design + cell statistics
# ---------------------------------------------------------------------------


def _session(n: int = 6000, seed: int = 0):
    blk, speed, ahv = _behaviour(n, seed)
    rng = np.random.default_rng(seed + 10)
    # syllables nested in states: 2 syllables per state
    syl = blk * 2 + np.repeat(rng.integers(0, 2, n // 10 + 1), 10)[:n]
    design, summary, sdf = ms.build_design(
        speed, ahv, np.ones(n, bool), FPS, syllable_id=syl, light_on=np.arange(n) % 2 == 0
    )
    return blk, syl, speed, ahv, design, summary, sdf


def test_build_design_contents() -> None:
    blk, syl, speed, ahv, d, summary, sdf = _session()
    assert d.n == blk.size and d.n_syl == 8 and d.n_fast >= 2
    assert summary["n_syllables_included"] == 8
    assert summary["frac_frames_in_included_syllables"] > 0.9
    assert set(np.unique(d.class_code[d.class_code >= 0])) == {0, 1, 2}
    assert d.run_idx.size == d.run_stratum.size == d.run_ahv_bin.size
    assert d.run_ahv_bin.max() <= d.n_ahv_bins - 1
    assert "n_still_light" in summary


def test_build_design_without_syllables_or_running() -> None:
    n = 500
    d, summary, sdf = ms.build_design(np.zeros(n), np.zeros(n), np.ones(n, bool), FPS)
    assert d.n_syl == 0 and sdf.empty and d.run_idx.size == 0
    out = ms.cell_statistics(np.random.default_rng(0).random(n), d, n_shuffles=20)
    assert np.isnan(out["syl_mi"]) and np.isnan(out["ahv_rho_matched"])
    assert np.isnan(out["class_frac_of_syl_mi"])


def test_run_cell_is_translation_biased() -> None:
    blk, syl, speed, ahv, d, _, _ = _session()
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 0.05, blk.size) + (blk == 1) * 1.0
    out = ms.cell_statistics(x, d, n_shuffles=100, rng=rng)
    assert out["run_index"] > 0.5 and out["run_index_p"] < 0.05 and out["run_index_z"] > 0
    assert out["run_vs_turn_bias"] > 0.5 and out["run_vs_turn_bias_p"] < 0.05
    assert out["interaction_index"] < 0
    assert out["syl_speed_rho"] > 0 and out["syl_mi_debiased"] > 0 and out["syl_mi_p"] < 0.05


def test_turn_cell_is_rotation_biased() -> None:
    blk, syl, speed, ahv, d, _, _ = _session()
    rng = np.random.default_rng(4)
    x = rng.gamma(2.0, 0.05, blk.size) + (blk >= 2) * 1.0
    out = ms.cell_statistics(x, d, n_shuffles=100, rng=rng)
    assert out["turn_index"] > 0.5 and out["turn_index_p"] < 0.05
    assert out["run_vs_turn_bias"] < -0.5 and out["run_vs_turn_bias_z"] < 0
    assert out["syl_ahv_rho"] > 0


def test_state_cell_information_is_in_class_not_syllable() -> None:
    """A speed-state cell: syllable information is carried by speed class."""
    n = 9000
    rng = np.random.default_rng(5)
    blk = np.repeat(rng.integers(0, 3, n // 20 + 1), 20)[:n]  # still / slow / fast
    base = np.array([0.2, 5.0, 15.0])[blk]
    sub = np.repeat(rng.integers(0, 2, n // 10 + 1), 10)[:n]
    syl = blk * 2 + sub
    speed = base + rng.uniform(0, 0.1, n) + sub * 0.5
    d, _, _ = ms.build_design(speed, np.zeros(n), np.ones(n, bool), FPS, syllable_id=syl)
    x = rng.gamma(2.0, 0.05, n) + blk * 0.5
    out = ms.cell_statistics(x, d, n_shuffles=100, rng=rng)
    assert out["syl_mi_debiased"] > 0.1
    assert out["class_frac_of_syl_mi"] > 0.9
    assert out["within_fast_mi_p"] > 0.01
    assert out["cond_syl_mi_given_class"] < 0.1 * out["syl_mi_debiased"]


def test_ahv_rho_matched_graded() -> None:
    n = 6000
    rng = np.random.default_rng(6)
    speed = rng.uniform(5, 20, n)
    ahv = rng.uniform(0, 300, n) * rng.choice([-1, 1], n)
    d, _, _ = ms.build_design(speed, ahv, np.ones(n, bool), FPS, smooth_s=0.0)
    x = np.abs(ahv) / 300 + rng.normal(0, 0.05, n)
    out = ms.cell_statistics(x, d, n_shuffles=50, rng=rng)
    assert out["ahv_rho_matched"] > 0.9 and out["ahv_rho_matched_p"] < 0.05


def test_observed_statistics_without_precomputed_codes() -> None:
    _, _, _, _, d, _, _ = _session(n=2000)
    x = np.random.default_rng(0).random(2000)
    a = ms.observed_statistics(x, d)
    b = ms.observed_statistics(x, d, ms.quantile_codes(x, d.n_signal_bins))
    assert a["syl_mi"] == b["syl_mi"]


def test_cell_statistics_short_session_no_null() -> None:
    n = 100
    blk, speed, ahv = _behaviour(n)
    d, _, _ = ms.build_design(speed, ahv, np.ones(n, bool), FPS, min_state_frames=5)
    out = ms.cell_statistics(np.random.default_rng(0).random(n), d, n_shuffles=50)
    assert np.isnan(out["run_index_p"]) and np.isnan(out["syl_mi_debiased"])


def test_cell_statistics_constant_signal() -> None:
    _, _, _, _, d, _, _ = _session(n=2000)
    out = ms.cell_statistics(np.ones(2000), d, n_shuffles=20)
    assert out["run_index"] == 0.0 and np.isnan(out["run_d"])
    assert out["syl_mi"] == 0.0


def test_helpers() -> None:
    assert np.isnan(ms._norm_index(1.0, -1.0)) and np.isnan(ms._norm_index(np.nan, 1.0))
    assert ms._norm_index(3.0, 1.0) == pytest.approx(0.5)
    assert np.isnan(ms._spearman([1, 2], [1, 2]))
    assert ms._spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    m = ms._group_means(np.array([1.0, 2.0, np.nan, 4.0]), np.array([0, 0, 1, -1]), 2, 1)
    assert m[0] == 1.5 and np.isnan(m[1])
    mu, z, p = ms._null_summary(5.0, np.arange(20.0), two_sided=False)
    assert mu == pytest.approx(9.5) and z < 0 and p > 0.5
    assert np.isnan(ms._null_summary(1.0, np.ones(5), True)[2])
    assert np.isnan(ms._null_summary(1.0, np.ones(20), True)[1])
