"""Tests for light cycle computation logic used in light analysis pages."""

from __future__ import annotations

import numpy as np
import pytest


class TestLightOnMask:
    """Test the light_on mask construction from timestamps."""

    @staticmethod
    def build_light_on(
        frame_times: np.ndarray,
        light_on_times: np.ndarray,
        light_off_times: np.ndarray,
    ) -> np.ndarray:
        """Reproduce the light_on mask logic from light_page.py."""
        n_frames = len(frame_times)
        light_on = np.zeros(n_frames, dtype=bool)
        for on_t in light_on_times:
            off_after = light_off_times[light_off_times > on_t]
            off_t = off_after[0] if len(off_after) > 0 else frame_times[-1] + 1
            mask = (frame_times >= on_t) & (frame_times < off_t)
            light_on[mask] = True
        return light_on

    def test_simple_on_off(self):
        """One light-on epoch from t=10 to t=20."""
        frame_times = np.arange(0, 30, 0.1)
        light_on = self.build_light_on(
            frame_times,
            np.array([10.0]),
            np.array([20.0]),
        )
        # Frames at t<10 should be dark, 10<=t<20 light, t>=20 dark
        assert not light_on[0]
        assert light_on[100]  # t=10
        assert light_on[150]  # t=15
        assert not light_on[200]  # t=20

    def test_multiple_cycles(self):
        """Two light-on epochs."""
        frame_times = np.arange(0, 60, 0.1)
        light_on = self.build_light_on(
            frame_times,
            np.array([10.0, 30.0]),
            np.array([20.0, 40.0]),
        )
        assert not light_on[0]      # t=0 dark
        assert light_on[100]         # t=10 light
        assert not light_on[250]     # t=25 dark
        assert light_on[350]         # t=35 light
        assert not light_on[450]     # t=45 dark

    def test_light_on_at_start(self):
        """Light is on from the very beginning."""
        frame_times = np.arange(0, 30, 0.1)
        light_on = self.build_light_on(
            frame_times,
            np.array([0.0]),
            np.array([15.0]),
        )
        assert light_on[0]
        assert light_on[50]
        assert not light_on[200]

    def test_light_stays_on_till_end(self):
        """No off time after last on time."""
        frame_times = np.arange(0, 30, 0.1)
        light_on = self.build_light_on(
            frame_times,
            np.array([10.0]),
            np.array([]),  # No off times
        )
        # Should stay on from t=10 onwards
        assert not light_on[0]
        assert light_on[100]
        assert light_on[250]

    def test_empty_cycle(self):
        """No light on/off times."""
        frame_times = np.arange(0, 30, 0.1)
        light_on = self.build_light_on(
            frame_times,
            np.array([]),
            np.array([]),
        )
        assert not light_on.any()


class TestLightModulationIndex:
    """Test light modulation index computation."""

    def test_equal_activity_gives_zero(self):
        lmi = (5.0 - 5.0) / (5.0 + 5.0 + 1e-10)
        assert abs(lmi) < 1e-8

    def test_more_light_gives_positive(self):
        lmi = (10.0 - 5.0) / (10.0 + 5.0 + 1e-10)
        assert lmi > 0

    def test_more_dark_gives_negative(self):
        lmi = (3.0 - 7.0) / (3.0 + 7.0 + 1e-10)
        assert lmi < 0

    def test_bounded(self):
        """LMI should be bounded between -1 and 1."""
        for light, dark in [(0, 10), (10, 0), (5, 5), (0.1, 100), (100, 0.1)]:
            lmi = (light - dark) / (light + dark + 1e-10)
            assert -1 <= lmi <= 1

    def test_zero_activity(self):
        """Both zero gives ~0 (protected by epsilon)."""
        lmi = (0.0 - 0.0) / (0.0 + 0.0 + 1e-10)
        assert abs(lmi) < 1e-5
