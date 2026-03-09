"""Tests for sync/align.py — neural-behavioural synchronisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.sync.align import (
    _BOOL_KEYS,
    resample_bool_to_imaging_rate,
    resample_to_imaging_rate,
)


# ---------------------------------------------------------------------------
# resample_to_imaging_rate — linear (default)
# ---------------------------------------------------------------------------


class TestResampleLinear:
    def test_constant_signal(self):
        src = np.linspace(0, 10, 1000)
        dst = np.linspace(0, 10, 300)
        vals = np.full(1000, 42.0)
        result = resample_to_imaging_rate(vals, src, dst)
        np.testing.assert_allclose(result, 42.0, rtol=1e-5)

    def test_linear_ramp_preserves_values(self):
        src = np.linspace(0.0, 1.0, 1000)
        dst = np.linspace(0.0, 1.0, 100)
        result = resample_to_imaging_rate(src.copy(), src, dst)
        np.testing.assert_allclose(result, dst, atol=1e-3)

    def test_output_shape(self):
        src = np.linspace(0, 60, 6000)
        dst = np.linspace(0, 60, 1800)
        vals = np.random.default_rng(0).standard_normal(6000)
        result = resample_to_imaging_rate(vals, src, dst)
        assert result.shape == (1800,)

    def test_single_source_point(self):
        src = np.array([5.0])
        dst = np.array([3.0, 5.0, 7.0])
        vals = np.array([99.0])
        result = resample_to_imaging_rate(vals, src, dst)
        # np.interp clamps to boundary values for out-of-range
        np.testing.assert_allclose(result, 99.0)

    def test_single_destination_point(self):
        src = np.linspace(0, 10, 100)
        dst = np.array([5.0])
        vals = src * 2.0
        result = resample_to_imaging_rate(vals, src, dst)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 10.0, atol=0.2)

    def test_dst_outside_src_range_clamps(self):
        src = np.array([1.0, 2.0, 3.0])
        dst = np.array([0.0, 4.0])
        vals = np.array([10.0, 20.0, 30.0])
        result = resample_to_imaging_rate(vals, src, dst)
        # np.interp clamps: before range → first val, after → last val
        np.testing.assert_allclose(result, [10.0, 30.0])

    def test_sinusoidal_signal(self):
        src = np.linspace(0, 2 * np.pi, 10000)
        dst = np.linspace(0, 2 * np.pi, 500)
        vals = np.sin(src)
        result = resample_to_imaging_rate(vals, src, dst)
        expected = np.sin(dst)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_identical_src_dst(self):
        times = np.linspace(0, 5, 200)
        vals = np.arange(200, dtype=float)
        result = resample_to_imaging_rate(vals, times, times)
        np.testing.assert_allclose(result, vals, atol=1e-10)

    def test_upsampling(self):
        src = np.array([0.0, 1.0])
        dst = np.linspace(0, 1, 11)
        vals = np.array([0.0, 10.0])
        result = resample_to_imaging_rate(vals, src, dst)
        np.testing.assert_allclose(result, np.linspace(0, 10, 11), atol=1e-10)

    def test_output_dtype_is_float(self):
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([0.5, 1.5])
        vals = np.array([1, 2, 3], dtype=np.int32)
        result = resample_to_imaging_rate(vals, src, dst)
        assert np.issubdtype(result.dtype, np.floating)

    def test_nan_in_values(self):
        src = np.array([0.0, 1.0, 2.0, 3.0])
        dst = np.array([0.5, 1.5, 2.5])
        vals = np.array([1.0, np.nan, 3.0, 4.0])
        result = resample_to_imaging_rate(vals, src, dst)
        # np.interp interpolates through NaN — result should contain NaN
        assert result.shape == (3,)
        assert np.isnan(result[0])  # interp between 1.0 and NaN


# ---------------------------------------------------------------------------
# resample_to_imaging_rate — nearest
# ---------------------------------------------------------------------------


class TestResampleNearest:
    def test_basic_nearest(self):
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([0.4, 0.6, 1.4])
        vals = np.array([10.0, 20.0, 30.0])
        result = resample_to_imaging_rate(vals, src, dst, method="nearest")
        # searchsorted left: 0.4→idx1→20, 0.6→idx1→20, 1.4→idx2→30
        np.testing.assert_array_equal(result, [20.0, 20.0, 30.0])

    def test_nearest_exact_match(self):
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([0.0, 1.0, 2.0])
        vals = np.array([10.0, 20.0, 30.0])
        result = resample_to_imaging_rate(vals, src, dst, method="nearest")
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_nearest_beyond_range(self):
        src = np.array([1.0, 2.0, 3.0])
        dst = np.array([0.0, 4.0])
        vals = np.array([10.0, 20.0, 30.0])
        result = resample_to_imaging_rate(vals, src, dst, method="nearest")
        # 0.0 → searchsorted idx=0 → clipped to 0 → 10.0
        # 4.0 → searchsorted idx=3 → clipped to 2 → 30.0
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_nearest_returns_float(self):
        src = np.array([0.0, 1.0])
        dst = np.array([0.5])
        vals = np.array([5, 10], dtype=np.int32)
        result = resample_to_imaging_rate(vals, src, dst, method="nearest")
        assert np.issubdtype(result.dtype, np.floating)

    def test_nearest_single_source(self):
        src = np.array([5.0])
        dst = np.array([0.0, 5.0, 10.0])
        vals = np.array([42.0])
        result = resample_to_imaging_rate(vals, src, dst, method="nearest")
        np.testing.assert_array_equal(result, [42.0, 42.0, 42.0])


# ---------------------------------------------------------------------------
# resample_bool_to_imaging_rate
# ---------------------------------------------------------------------------


class TestResampleBool:
    def test_preserves_bool_dtype(self):
        src = np.linspace(0, 60, 6000)
        dst = np.linspace(0, 60, 1800)
        mask = np.zeros(6000, dtype=bool)
        mask[2000:4000] = True
        result = resample_bool_to_imaging_rate(mask, src, dst)
        assert result.dtype == bool
        assert result.shape == (1800,)

    def test_all_true(self):
        src = np.linspace(0, 10, 100)
        dst = np.linspace(0, 10, 30)
        mask = np.ones(100, dtype=bool)
        result = resample_bool_to_imaging_rate(mask, src, dst)
        assert np.all(result)

    def test_all_false(self):
        src = np.linspace(0, 10, 100)
        dst = np.linspace(0, 10, 30)
        mask = np.zeros(100, dtype=bool)
        result = resample_bool_to_imaging_rate(mask, src, dst)
        assert not np.any(result)

    def test_alternating_pattern(self):
        src = np.arange(10, dtype=float)
        dst = np.arange(10, dtype=float)  # same timestamps
        mask = np.array([True, False] * 5)
        result = resample_bool_to_imaging_rate(mask, src, dst)
        np.testing.assert_array_equal(result, mask)

    def test_single_source_frame(self):
        src = np.array([1.0])
        dst = np.array([0.0, 1.0, 2.0])
        mask = np.array([True])
        result = resample_bool_to_imaging_rate(mask, src, dst)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, [True, True, True])

    def test_dst_beyond_src_clips(self):
        src = np.array([0.0, 1.0, 2.0])
        dst = np.array([3.0, 4.0])  # beyond src range
        mask = np.array([True, False, True])
        result = resample_bool_to_imaging_rate(mask, src, dst)
        # idx clips to 2 → True
        np.testing.assert_array_equal(result, [True, True])

    def test_transition_boundary(self):
        # Check that the transition from False to True is preserved
        src = np.arange(6, dtype=float)
        dst = np.arange(6, dtype=float)
        mask = np.array([False, False, False, True, True, True])
        result = resample_bool_to_imaging_rate(mask, src, dst)
        np.testing.assert_array_equal(result, mask)


# ---------------------------------------------------------------------------
# _BOOL_KEYS constant
# ---------------------------------------------------------------------------


def test_bool_keys_contains_expected():
    assert "light_on" in _BOOL_KEYS
    assert "bad_behav" in _BOOL_KEYS
    assert "active" in _BOOL_KEYS
    assert len(_BOOL_KEYS) == 3


def test_bool_keys_is_frozenset():
    assert isinstance(_BOOL_KEYS, frozenset)


# ---------------------------------------------------------------------------
# run() — full Stage 5 pipeline integration tests
# ---------------------------------------------------------------------------


def _write_synthetic_kinematics(path: Path, n: int = 600) -> None:
    from hm2p.io.hdf5 import write_h5

    frame_times = np.linspace(0, 6.0, n, dtype=np.float64)
    write_h5(
        path,
        arrays={
            "frame_times": frame_times,
            "hd_deg": np.sin(frame_times).astype(np.float32),
            "x_mm": np.linspace(0, 50, n, dtype=np.float32),
            "y_mm": np.linspace(0, 30, n, dtype=np.float32),
            "speed_cm_s": np.ones(n, dtype=np.float32) * 5.0,
            "ahv_deg_s": np.zeros(n, dtype=np.float32),
            "active": np.ones(n, dtype=bool),
            "light_on": np.tile([True, False], n // 2).astype(bool),
            "bad_behav": np.zeros(n, dtype=bool),
        },
        attrs={"session_id": "test", "fps_camera": 100.0},
    )


def _write_synthetic_ca(path: Path, t: int = 180, n_rois: int = 10) -> None:
    from hm2p.io.hdf5 import write_h5

    frame_times = np.linspace(0, 6.0, t, dtype=np.float64)
    write_h5(
        path,
        arrays={
            "frame_times": frame_times,
            "dff": np.random.default_rng(5).standard_normal((n_rois, t)).astype(
                np.float32
            ),
        },
        attrs={"session_id": "test", "fps_imaging": 30.0, "extractor": "suite2p"},
    )


class TestRunPipeline:
    def test_creates_file(self, tmp_path):
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test_ses", output_path=out_h5)
        assert out_h5.exists()

    def test_frame_times_match_ca(self, tmp_path):
        from hm2p.io.hdf5 import read_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test_ses", output_path=out_h5)
        sync = read_h5(out_h5)
        ca = read_h5(ca_h5)
        np.testing.assert_array_equal(sync["frame_times"], ca["frame_times"])

    def test_kinematics_resampled_length(self, tmp_path):
        from hm2p.io.hdf5 import read_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test_ses", output_path=out_h5)
        sync = read_h5(out_h5)
        ca = read_h5(ca_h5)
        T = len(ca["frame_times"])
        assert sync["hd_deg"].shape == (T,)
        assert sync["speed_cm_s"].shape == (T,)
        assert sync["x_mm"].shape == (T,)
        assert sync["y_mm"].shape == (T,)
        assert sync["ahv_deg_s"].shape == (T,)

    def test_bool_signals_preserved(self, tmp_path):
        from hm2p.io.hdf5 import read_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test_ses", output_path=out_h5)
        sync = read_h5(out_h5)
        assert sync["light_on"].dtype == bool
        assert sync["bad_behav"].dtype == bool
        assert sync["active"].dtype == bool

    def test_ca_arrays_present(self, tmp_path):
        from hm2p.io.hdf5 import read_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test_ses", output_path=out_h5)
        sync = read_h5(out_h5)
        assert "dff" in sync

    def test_session_id_attr(self, tmp_path):
        from hm2p.io.hdf5 import read_attrs
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="20220804_13_52_02_1117646", output_path=out_h5)
        attrs = read_attrs(out_h5)
        assert attrs["session_id"] == "20220804_13_52_02_1117646"

    def test_inherits_ca_attrs(self, tmp_path):
        from hm2p.io.hdf5 import read_attrs
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test", output_path=out_h5)
        attrs = read_attrs(out_h5)
        assert attrs["extractor"] == "suite2p"
        assert attrs["fps_imaging"] == 30.0

    def test_float32_kinematics_in_sync(self, tmp_path):
        from hm2p.io.hdf5 import read_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test", output_path=out_h5)
        sync = read_h5(out_h5)
        for key in ("hd_deg", "x_mm", "y_mm", "speed_cm_s", "ahv_deg_s"):
            assert sync[key].dtype == np.float32, f"{key} should be float32"

    def test_all_kinematics_keys_present(self, tmp_path):
        from hm2p.io.hdf5 import read_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5)
        _write_synthetic_ca(ca_h5)
        run(kin_h5, ca_h5, session_id="test", output_path=out_h5)
        sync = read_h5(out_h5)
        expected_keys = {
            "frame_times",
            "hd_deg",
            "x_mm",
            "y_mm",
            "speed_cm_s",
            "ahv_deg_s",
            "active",
            "light_on",
            "bad_behav",
            "dff",
        }
        assert expected_keys.issubset(set(sync.keys()))

    def test_different_rates(self, tmp_path):
        """Camera at 200 frames, imaging at 50 frames — verify resampling."""
        from hm2p.io.hdf5 import read_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5, n=200)
        _write_synthetic_ca(ca_h5, t=50, n_rois=5)
        run(kin_h5, ca_h5, session_id="test", output_path=out_h5)
        sync = read_h5(out_h5)
        assert sync["hd_deg"].shape == (50,)
        assert sync["dff"].shape == (5, 50)

    def test_off_by_one_frame_times_trimmed(self, tmp_path):
        """Suite2p often has N+1 frame_times for N dF/F frames; sync should trim."""
        from hm2p.io.hdf5 import read_h5, write_h5
        from hm2p.sync.align import run

        kin_h5 = tmp_path / "kinematics.h5"
        ca_h5 = tmp_path / "ca.h5"
        out_h5 = tmp_path / "sync.h5"
        _write_synthetic_kinematics(kin_h5, n=600)

        # Write ca.h5 with N+1 frame_times for N dF/F columns
        n_rois, n_frames = 8, 180
        write_h5(
            ca_h5,
            arrays={
                "frame_times": np.linspace(0, 6.0, n_frames + 1, dtype=np.float64),
                "dff": np.random.default_rng(7).standard_normal(
                    (n_rois, n_frames)
                ).astype(np.float32),
            },
            attrs={"session_id": "test", "fps_imaging": 30.0, "extractor": "suite2p"},
        )

        run(kin_h5, ca_h5, session_id="test", output_path=out_h5)
        sync = read_h5(out_h5)
        # Resampled kinematics should match dff columns, not frame_times length
        assert sync["hd_deg"].shape == (n_frames,)
        assert sync["dff"].shape == (n_rois, n_frames)
