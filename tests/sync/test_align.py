"""Tests for sync/align.py — neural–behavioural synchronisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.sync.align import resample_bool_to_imaging_rate, resample_to_imaging_rate


def test_resample_constant_signal() -> None:
    """Resampling a constant signal returns the same constant."""
    src_times = np.linspace(0, 10, 1000)
    dst_times = np.linspace(0, 10, 300)
    values = np.full(1000, 42.0)
    result = resample_to_imaging_rate(values, src_times, dst_times)
    np.testing.assert_allclose(result, 42.0, rtol=1e-5)


def test_resample_linear_signal() -> None:
    """Resampling a linear ramp preserves values at interpolation points."""
    src_times = np.linspace(0.0, 1.0, 1000)
    dst_times = np.linspace(0.0, 1.0, 100)
    values = src_times.copy()  # identity ramp
    result = resample_to_imaging_rate(values, src_times, dst_times)
    np.testing.assert_allclose(result, dst_times, atol=1e-3)


def test_resample_output_shape() -> None:
    """Output length equals len(dst_times)."""
    src_times = np.linspace(0, 60, 6000)
    dst_times = np.linspace(0, 60, 1800)
    values = np.random.default_rng(0).standard_normal(6000)
    result = resample_to_imaging_rate(values, src_times, dst_times)
    assert result.shape == (1800,)


def test_resample_bool_preserves_dtype() -> None:
    """resample_bool_to_imaging_rate returns bool array."""
    src_times = np.linspace(0, 60, 6000)
    dst_times = np.linspace(0, 60, 1800)
    mask = np.zeros(6000, dtype=bool)
    mask[2000:4000] = True
    result = resample_bool_to_imaging_rate(mask, src_times, dst_times)
    assert result.dtype == bool
    assert result.shape == (1800,)


def test_resample_nearest_method() -> None:
    """nearest method uses searchsorted, not linear interpolation."""
    src_times = np.array([0.0, 1.0, 2.0])
    dst_times = np.array([0.4, 0.6, 1.4])
    values = np.array([10.0, 20.0, 30.0])
    result = resample_to_imaging_rate(values, src_times, dst_times, method="nearest")
    # searchsorted left: 0.4→idx=1→20, 0.6→idx=1→20, 1.4→idx=2→30
    np.testing.assert_array_equal(result, [20.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# run() — full Stage 5 pipeline
# ---------------------------------------------------------------------------


def _write_synthetic_kinematics(path: Path) -> None:
    """Write a minimal kinematics.h5 at camera rate."""
    from hm2p.io.hdf5 import write_h5

    n = 600  # camera frames
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


def _write_synthetic_ca(path: Path) -> None:
    """Write a minimal ca.h5 at imaging rate."""
    from hm2p.io.hdf5 import write_h5

    t = 180  # imaging frames
    n_rois = 10
    frame_times = np.linspace(0, 6.0, t, dtype=np.float64)
    write_h5(
        path,
        arrays={
            "frame_times": frame_times,
            "dff": np.random.default_rng(5).standard_normal((n_rois, t)).astype(np.float32),
        },
        attrs={"session_id": "test", "fps_imaging": 30.0, "extractor": "suite2p"},
    )


def test_sync_run_creates_file(tmp_path: Path) -> None:
    """run() creates the output sync.h5 file."""
    from hm2p.sync.align import run

    kin_h5 = tmp_path / "kinematics.h5"
    ca_h5 = tmp_path / "ca.h5"
    out_h5 = tmp_path / "sync.h5"

    _write_synthetic_kinematics(kin_h5)
    _write_synthetic_ca(ca_h5)
    run(kin_h5, ca_h5, session_id="test_ses", output_path=out_h5)
    assert out_h5.exists()


def test_sync_run_frame_times_at_imaging_rate(tmp_path: Path) -> None:
    """sync.h5 frame_times match ca.h5 imaging frame times."""
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


def test_sync_run_kinematics_resampled_length(tmp_path: Path) -> None:
    """Resampled kinematics signals have imaging-rate length."""
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


def test_sync_run_bool_signals_preserved(tmp_path: Path) -> None:
    """Boolean signals (light_on, bad_behav, active) are bool dtype in sync.h5."""
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


def test_sync_run_ca_arrays_present(tmp_path: Path) -> None:
    """Calcium arrays (dff) from ca.h5 are present in sync.h5."""
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


def test_sync_run_session_id_attr(tmp_path: Path) -> None:
    """session_id attribute in sync.h5 matches the argument passed to run()."""
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
