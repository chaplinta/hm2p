"""Tests for occupancy/kinematics-matched MVL and shuffle debiasing.

Uses small synthetic arrays only — no real data files.
"""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.matched_tuning import (
    match_indices_1d,
    match_indices_2d,
    matched_condition_mvl,
    occupancy_histogram,
    shuffle_debiased_mvl,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length


def _tuned_signal(hd_deg, pd_deg=90.0, kappa=4.0, baseline=1.0, gain=2.0):
    """Von-Mises-like HD-tuned signal."""
    return baseline + gain * np.exp(kappa * np.cos(np.deg2rad(hd_deg - pd_deg)))


# ---------------------------------------------------------------------------
# occupancy_histogram
# ---------------------------------------------------------------------------


def test_occupancy_histogram_sums_to_one():
    rng = np.random.default_rng(0)
    hd = rng.uniform(0, 360, 1000)
    h = occupancy_histogram(hd, n_bins=36)
    assert h.shape == (36,)
    assert np.isclose(h.sum(), 1.0)


def test_occupancy_histogram_empty():
    h = occupancy_histogram(np.array([]), n_bins=12)
    assert h.shape == (12,)
    assert h.sum() == 0.0


def test_occupancy_histogram_uniform_is_flat():
    hd = np.linspace(0, 360, 3601, endpoint=False)
    h = occupancy_histogram(hd, n_bins=36)
    assert np.allclose(h, 1.0 / 36, atol=0.01)


# ---------------------------------------------------------------------------
# match_indices_1d
# ---------------------------------------------------------------------------


def test_match_1d_equalises_distribution():
    rng = np.random.default_rng(1)
    # A: uniform. B: concentrated near 0.
    a = rng.uniform(0, 360, 5000)
    b = np.concatenate([rng.uniform(0, 90, 4000), rng.uniform(0, 360, 1000)])
    ia, ib = match_indices_1d(a, b, n_bins=36, circular=True, rng=rng)
    ha = occupancy_histogram(a[ia], 36)
    hb = occupancy_histogram(b[ib], 36)
    # Matched histograms should be nearly identical.
    assert np.max(np.abs(ha - hb)) < 1e-9


def test_match_1d_keeps_min_per_bin():
    rng = np.random.default_rng(2)
    a = np.array([10.0] * 10 + [200.0] * 3)
    b = np.array([10.0] * 4 + [200.0] * 9)
    ia, ib = match_indices_1d(a, b, n_bins=36, circular=True, rng=rng)
    # bin(10): min(10,4)=4 ; bin(200): min(3,9)=3 -> 7 each
    assert len(ia) == 7
    assert len(ib) == 7


def test_match_1d_linear_range():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 10, 2000)
    b = rng.uniform(0, 5, 2000)
    ia, ib = match_indices_1d(a, b, n_bins=10, circular=False, value_range=(0, 10), rng=rng)
    # b has no samples above 5, so matched sets should not exceed 5.
    assert a[ia].max() <= 5.5
    assert b[ib].max() <= 5.5


# ---------------------------------------------------------------------------
# match_indices_2d
# ---------------------------------------------------------------------------


def test_match_2d_equalises_joint_distribution():
    rng = np.random.default_rng(4)
    xa = rng.uniform(0, 10, 4000)
    ya = rng.uniform(0, 10, 4000)
    xb = rng.uniform(0, 5, 4000)
    yb = rng.uniform(0, 5, 4000)
    ia, ib = match_indices_2d(
        xa, ya, xb, yb, n_bins=(5, 5), x_range=(0, 10), y_range=(0, 10), rng=rng
    )

    # Joint counts per cell must match exactly.
    def grid(x, y):
        xe = np.linspace(0, 10, 6)
        ye = np.linspace(0, 10, 6)
        bx = np.clip(np.digitize(x, xe) - 1, 0, 4)
        by = np.clip(np.digitize(y, ye) - 1, 0, 4)
        return bx * 5 + by

    ca = np.bincount(grid(xa[ia], ya[ia]), minlength=25)
    cb = np.bincount(grid(xb[ib], yb[ib]), minlength=25)
    assert np.array_equal(ca, cb)


# ---------------------------------------------------------------------------
# shuffle_debiased_mvl
# ---------------------------------------------------------------------------


def test_debias_reduces_mvl_for_random_signal():
    rng = np.random.default_rng(5)
    n = 2000
    hd = rng.uniform(0, 360, n)
    signal = rng.normal(0, 1, n)  # no HD tuning
    r = shuffle_debiased_mvl(signal, hd, n_shuffles=100, rng=rng)
    # Untuned: debiased MVL should be near zero, well below raw.
    assert abs(r["mvl_debiased"]) < r["mvl_raw"] + 1e-9
    assert abs(r["mvl_debiased"]) < 0.15


def test_debias_preserves_strong_tuning():
    rng = np.random.default_rng(6)
    n = 3000
    hd = rng.uniform(0, 360, n)
    signal = _tuned_signal(hd, kappa=6.0)
    r = shuffle_debiased_mvl(signal, hd, n_shuffles=100, rng=rng)
    # Strongly tuned: debiased MVL stays high and positive.
    assert r["mvl_debiased"] > 0.5
    assert r["mvl_bias"] < r["mvl_raw"]


def test_debias_keys_and_shapes():
    rng = np.random.default_rng(7)
    hd = rng.uniform(0, 360, 800)
    sig = _tuned_signal(hd)
    r = shuffle_debiased_mvl(sig, hd, n_shuffles=50, rng=rng)
    assert set(r) >= {"mvl_raw", "mvl_bias", "mvl_debiased", "shuffle_dist"}
    assert r["shuffle_dist"].shape == (50,)


# ---------------------------------------------------------------------------
# matched_condition_mvl
# ---------------------------------------------------------------------------


def test_matched_none_reproduces_raw_mvl():
    """match='none', debias=False must reproduce the plain tuning-curve MVL."""
    rng = np.random.default_rng(8)
    n = 1500
    hd_a = rng.uniform(0, 360, n)
    hd_b = rng.uniform(0, 360, n)
    sig_a = _tuned_signal(hd_a, pd_deg=45)
    sig_b = _tuned_signal(hd_b, pd_deg=200)

    out = matched_condition_mvl(sig_a, hd_a, sig_b, hd_b, match="none", debias=False, rng=rng)

    tca, bca = compute_hd_tuning_curve(sig_a, hd_a, np.ones(n, bool))
    tcb, bcb = compute_hd_tuning_curve(sig_b, hd_b, np.ones(n, bool))
    assert np.isclose(out["mvl_a"], mean_vector_length(tca, bca), atol=1e-9)
    assert np.isclose(out["mvl_b"], mean_vector_length(tcb, bcb), atol=1e-9)


def test_matched_occupancy_equalises_then_compares():
    """When both conditions share the SAME tuning, matched MVLs should be close."""
    rng = np.random.default_rng(9)
    n = 4000
    # A samples HD uniformly; B oversamples near the PD (would inflate raw MVL).
    hd_a = rng.uniform(0, 360, n)
    hd_b = np.concatenate([rng.uniform(60, 120, 3000), rng.uniform(0, 360, 1000)])
    sig_a = _tuned_signal(hd_a, pd_deg=90, kappa=3.0)
    sig_b = _tuned_signal(hd_b, pd_deg=90, kappa=3.0)
    out = matched_condition_mvl(
        sig_a, hd_a, sig_b, hd_b, match="occupancy", n_boot=10, n_shuffles=30, debias=True, rng=rng
    )
    # Same underlying code -> matched debiased MVLs should be similar.
    assert abs(out["mvl_a"] - out["mvl_b"]) < 0.2
    assert out["n_boot"] > 0


def test_matched_kinematics_requires_vars():
    rng = np.random.default_rng(10)
    hd = rng.uniform(0, 360, 200)
    sig = _tuned_signal(hd)
    with pytest.raises(ValueError):
        matched_condition_mvl(sig, hd, sig, hd, match="kinematics", rng=rng)


def test_matched_kinematics_runs():
    rng = np.random.default_rng(11)
    n = 3000
    hd_a = rng.uniform(0, 360, n)
    hd_b = rng.uniform(0, 360, n)
    sig_a = _tuned_signal(hd_a)
    sig_b = _tuned_signal(hd_b)
    spd_a = rng.uniform(0, 20, n)
    ahv_a = rng.uniform(0, 100, n)
    spd_b = rng.uniform(0, 10, n)
    ahv_b = rng.uniform(0, 50, n)
    out = matched_condition_mvl(
        sig_a,
        hd_a,
        sig_b,
        hd_b,
        match_vars_a=(spd_a, ahv_a),
        match_vars_b=(spd_b, ahv_b),
        match="kinematics",
        n_boot=5,
        n_shuffles=20,
        debias=True,
        rng=rng,
    )
    assert out["n_boot"] > 0
    assert np.isfinite(out["mvl_a"])


def test_matched_invalid_mode():
    rng = np.random.default_rng(12)
    hd = rng.uniform(0, 360, 100)
    sig = _tuned_signal(hd)
    with pytest.raises(ValueError):
        matched_condition_mvl(sig, hd, sig, hd, match="bogus", rng=rng)
