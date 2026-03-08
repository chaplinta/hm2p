"""Property-based tests for analysis modules using hypothesis.

Tests numerical invariants of HD tuning analysis functions with
auto-generated adversarial inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.analysis.information import (
    mutual_information_binned,
    skaggs_info_rate,
)
from hm2p.analysis.gain import gain_modulation_index
from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
)


# --- Strategy helpers ---

def hd_signal_strategy(min_frames=100, max_frames=500):
    """Strategy for (signal, hd_deg, mask) tuples."""
    return st.integers(min_value=min_frames, max_value=max_frames).flatmap(
        lambda n: st.tuples(
            arrays(np.float64, n, elements=st.floats(0, 10, allow_nan=False, allow_infinity=False)),
            arrays(np.float64, n, elements=st.floats(0, 360, allow_nan=False, allow_infinity=False, exclude_max=True)),
            st.just(np.ones(n, dtype=bool)),
        )
    )


class TestMVLInvariants:
    """Property-based tests for mean vector length."""

    @given(hd_signal_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_mvl_bounded(self, data):
        """MVL should always be in [0, 1]."""
        signal, hd, mask = data
        tc, bc = compute_hd_tuning_curve(signal, hd, mask, n_bins=18)
        mvl = mean_vector_length(tc, bc)
        assert 0 <= mvl <= 1 + 1e-9

    def test_uniform_tuning_low_mvl_deterministic(self):
        """Constant signal with good HD coverage should give low MVL."""
        rng = np.random.default_rng(42)
        hd = rng.uniform(0, 360, 500)
        constant_signal = np.ones(500) * 5.0
        mask = np.ones(500, dtype=bool)
        tc, bc = compute_hd_tuning_curve(constant_signal, hd, mask, n_bins=18)
        mvl = mean_vector_length(tc, bc)
        assert mvl < 0.1  # Truly flat tuning → very low MVL


class TestMIInvariants:
    """Property-based tests for mutual information."""

    @given(hd_signal_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_mi_non_negative(self, data):
        """MI should always be >= 0."""
        signal, hd, mask = data
        mi = mutual_information_binned(signal, hd, mask, n_hd_bins=18, n_signal_bins=5)
        assert mi >= -1e-9  # Allow tiny floating point error

    @given(hd_signal_strategy())
    @settings(max_examples=20, deadline=5000)
    def test_constant_signal_low_mi(self, data):
        """Constant signal has near-zero MI with any HD."""
        _, hd, mask = data
        constant_signal = np.ones_like(hd) * 3.0
        mi = mutual_information_binned(constant_signal, hd, mask, n_hd_bins=18, n_signal_bins=5)
        assert mi < 0.5  # Should be very low


class TestSkaggsInvariants:
    """Property-based tests for Skaggs information rate."""

    @given(
        arrays(np.float64, 36, elements=st.floats(0.001, 100, allow_nan=False, allow_infinity=False)),
        arrays(np.float64, 36, elements=st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=50, deadline=5000)
    def test_si_non_negative(self, tuning_curve, occupancy):
        """Skaggs information rate should be non-negative."""
        si = skaggs_info_rate(tuning_curve, occupancy)
        assert si >= -1e-9

    @given(arrays(np.float64, 36, elements=st.floats(1, 1000, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=30, deadline=5000)
    def test_uniform_tuning_zero_si(self, occupancy):
        """Flat tuning curve should give SI = 0."""
        flat_tc = np.ones(36) * 5.0
        si = skaggs_info_rate(flat_tc, occupancy)
        assert abs(si) < 1e-9


class TestGainInvariants:
    """Property-based tests for gain modulation."""

    @given(hd_signal_strategy(min_frames=200, max_frames=400))
    @settings(max_examples=20, deadline=5000)
    def test_gain_index_bounded(self, data):
        """Gain modulation index should be in [-1, 1]."""
        signal, hd, mask = data
        light_on = np.zeros_like(mask)
        light_on[:len(mask)//2] = True
        result = gain_modulation_index(signal, hd, mask, light_on, n_bins=18)
        assert -1 <= result["gain_index"] <= 1

    @given(hd_signal_strategy(min_frames=200, max_frames=400))
    @settings(max_examples=20, deadline=5000)
    def test_peaks_non_negative(self, data):
        """Peak values should be non-negative."""
        signal, hd, mask = data
        light_on = np.zeros_like(mask)
        light_on[:len(mask)//2] = True
        result = gain_modulation_index(signal, hd, mask, light_on, n_bins=18)
        assert result["peak_light"] >= 0
        assert result["peak_dark"] >= 0
