"""End-to-end pandera schema validation tests for pipeline stage outputs.

These integration tests verify that the pipeline's save functions produce HDF5
files that conform to their pandera schemas. Each test builds minimal synthetic
data, writes it through the actual pipeline save path, reads it back, and
validates against the schema.

Stages covered:
  - timestamps.h5  (Stage 0 — DAQ parse)
  - ca.h5          (Stage 4 — calcium processing)
  - sync.h5        (Stage 5 — neural-behavioural sync)
  - analysis.h5    (Stage 6 — analysis results)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.io.hdf5 import (
    read_h5,
    validate_ca_h5,
    validate_sync_h5,
    validate_timestamps_h5,
    write_h5,
)

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _build_timestamps_arrays(n_cam: int = 3000, n_img: int = 900) -> dict[str, np.ndarray]:
    """Build a valid timestamps.h5 dict matching Stage 0 output."""
    return {
        "frame_times_camera": np.linspace(0.0, 30.0, n_cam, dtype=np.float64),
        "frame_times_imaging": np.linspace(0.0, 30.0, n_img, dtype=np.float64),
        "light_on_times": np.array([0.0, 120.0], dtype=np.float64),
        "light_off_times": np.array([60.0, 180.0], dtype=np.float64),
    }


def _build_ca_arrays(
    n_rois: int = 8, T: int = 900, rng: np.random.Generator | None = None
) -> dict[str, np.ndarray]:
    """Build a valid ca.h5 dict matching Stage 4 output."""
    if rng is None:
        rng = np.random.default_rng(0)
    frame_times = np.linspace(0.0, 30.0, T, dtype=np.float64)
    dff = rng.standard_normal((n_rois, T)).astype(np.float32)
    event_masks = rng.integers(0, 2, (n_rois, T)).astype(np.float32)
    roi_types = np.zeros(n_rois, dtype=np.uint8)  # all soma
    return {
        "frame_times": frame_times,
        "dff": dff,
        "event_masks": event_masks,
        "noise_probs": rng.random((n_rois, T)).astype(np.float32),
        "roi_types": roi_types,
    }


def _build_kinematics_arrays(T: int = 3000) -> dict[str, np.ndarray]:
    """Build a valid kinematics.h5 dict matching Stage 3 output."""
    rng = np.random.default_rng(1)
    return {
        "frame_times": np.linspace(0.0, 30.0, T, dtype=np.float64),
        "hd_deg": rng.uniform(0, 360, T).astype(np.float32),
        "x_mm": rng.uniform(0, 500, T).astype(np.float32),
        "y_mm": rng.uniform(0, 500, T).astype(np.float32),
        "speed_cm_s": np.abs(rng.standard_normal(T)).astype(np.float32),
        "ahv_deg_s": rng.standard_normal(T).astype(np.float32),
        "active": rng.integers(0, 2, T).astype(bool),
        "light_on": rng.integers(0, 2, T).astype(bool),
        "bad_behav": np.zeros(T, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Test: timestamps.h5 round-trip through write_h5 + schema validation
# ---------------------------------------------------------------------------


class TestTimestampsE2E:
    """Write timestamps.h5 via write_h5, read back, validate schema."""

    def test_timestamps_write_read_validate(self, tmp_path: Path) -> None:
        arrays = _build_timestamps_arrays()
        path = tmp_path / "timestamps.h5"

        write_h5(path, arrays, attrs={"session_id": "20220804_13_52_02_1117646"})
        loaded = read_h5(path)

        # Schema validation must pass without raising SchemaError
        validate_timestamps_h5(loaded)

        # Verify array shapes survived round-trip
        assert loaded["frame_times_camera"].shape == (3000,)
        assert loaded["frame_times_imaging"].shape == (900,)

    def test_timestamps_corrupt_after_write_detected(self, tmp_path: Path) -> None:
        """Corrupt a key after writing and confirm schema catches it."""
        from pandera.errors import SchemaError

        arrays = _build_timestamps_arrays()
        path = tmp_path / "timestamps.h5"
        write_h5(path, arrays)
        loaded = read_h5(path)

        # Corrupt: make camera frame_times non-monotonic
        loaded["frame_times_camera"][50] = loaded["frame_times_camera"][0]
        with pytest.raises(SchemaError, match="strictly increasing"):
            validate_timestamps_h5(loaded)


# ---------------------------------------------------------------------------
# Test: ca.h5 round-trip
# ---------------------------------------------------------------------------


class TestCaH5E2E:
    """Write ca.h5 via write_h5, read back, validate schema."""

    def test_ca_write_read_validate(self, tmp_path: Path) -> None:
        n_rois, T = 8, 900
        arrays = _build_ca_arrays(n_rois=n_rois, T=T)
        path = tmp_path / "ca.h5"

        write_h5(path, arrays, attrs={"session_id": "test_session", "fps_imaging": 9.6})
        loaded = read_h5(path)

        # The validator only checks frame_times + dff (+ optional spikes).
        # Extra keys (event_masks, roi_types) are ignored by the schema.
        validate_ca_h5(loaded)

        assert loaded["dff"].shape == (n_rois, T)
        assert loaded["frame_times"].dtype == np.float64

    def test_ca_with_spikes_validates(self, tmp_path: Path) -> None:
        """ca.h5 with optional spikes array passes validation."""
        n_rois, T = 5, 500
        arrays = _build_ca_arrays(n_rois=n_rois, T=T)
        arrays["spikes"] = np.abs(np.random.default_rng(2).standard_normal((n_rois, T))).astype(
            np.float32
        )
        path = tmp_path / "ca.h5"

        write_h5(path, arrays)
        loaded = read_h5(path)
        validate_ca_h5(loaded)


# ---------------------------------------------------------------------------
# Test: sync.h5 end-to-end via sync/align.py run()
# ---------------------------------------------------------------------------


class TestSyncH5E2E:
    """Run the Stage 5 sync pipeline on synthetic files, validate output."""

    def test_sync_run_produces_valid_sync_h5(self, tmp_path: Path) -> None:
        from hm2p.sync.align import run as sync_run

        # n_cam @ 100 Hz must span same duration as n_img @ 9.6 Hz so the
        # sync diagnostics classifier doesn't reject for temporal-overlap.
        # 9375 / 100 ≈ 93.75 s ≈ 900 / 9.6.
        n_cam, n_img, n_rois = 9375, 900, 8

        # Write synthetic kinematics.h5
        kin_arrays = _build_kinematics_arrays(T=n_cam)
        kin_path = tmp_path / "kinematics.h5"
        write_h5(kin_path, kin_arrays, attrs={"session_id": "test"})

        # Write synthetic ca.h5
        ca_arrays = _build_ca_arrays(n_rois=n_rois, T=n_img)
        ca_path = tmp_path / "ca.h5"
        write_h5(ca_path, ca_arrays, attrs={"session_id": "test", "fps_imaging": 9.6})

        # Write synthetic timestamps.h5 so Stage 5 can compute sync_status
        # and produce a full (non-stub) sync.h5 for the happy-path test.
        ts_path = tmp_path / "timestamps.h5"
        cam_dt = 1.0 / 100.0
        img_dt = 1.0 / 9.6
        ts_arrays = {
            "frame_times_camera": np.arange(n_cam, dtype=np.float64) * cam_dt,
            "frame_times_imaging": np.arange(n_img, dtype=np.float64) * img_dt,
            "line_clock_times": np.arange(n_img * 162, dtype=np.float64) * (img_dt / 162),
            "light_on_times": np.array([0.0, 120.0], dtype=np.float64),
            "light_off_times": np.array([60.0, 180.0], dtype=np.float64),
        }
        write_h5(
            ts_path,
            ts_arrays,
            attrs={
                "session_id": "test",
                "fps_camera": 100.0,
                "fps_imaging": 9.6,
            },
        )

        # Run the sync pipeline
        sync_path = tmp_path / "sync.h5"
        sync_run(
            kinematics_h5=kin_path,
            ca_h5=ca_path,
            session_id="20220804_13_52_02_1117646",
            output_path=sync_path,
            timestamps_h5=ts_path,
        )

        # Read back and validate. The new validator branches on sync_status,
        # so we must pass attrs.
        loaded = read_h5(sync_path)
        from hm2p.io.hdf5 import read_attrs

        attrs = read_attrs(sync_path)
        validate_sync_h5(loaded, attrs=attrs)

        # All arrays should be at imaging rate (n_img frames)
        assert loaded["frame_times"].shape == (n_img,)
        assert loaded["dff"].shape == (n_rois, n_img)
        assert loaded["hd_deg"].shape == (n_img,)
        assert loaded["speed_cm_s"].shape == (n_img,)
        assert loaded["light_on"].dtype == bool
        assert loaded["bad_behav"].dtype == bool


# ---------------------------------------------------------------------------
# Test: analysis.h5 round-trip via save/load
# ---------------------------------------------------------------------------


class TestAnalysisH5E2E:
    """Write analysis.h5 via save_analysis_results, load back, verify structure."""

    def test_analysis_save_load_roundtrip(self, tmp_path: Path) -> None:
        from hm2p.analysis.run import AnalysisParams, CellResult
        from hm2p.analysis.save import load_analysis_results, save_analysis_results

        n_rois = 5
        n_bins = 36
        rng = np.random.default_rng(99)
        params = AnalysisParams(
            signal_type="dff",
            hd_n_bins=n_bins,
            n_shuffles=100,
        )

        bin_centers = np.linspace(5.0, 355.0, n_bins, dtype=np.float32)

        # Build synthetic CellResult objects
        results: list[CellResult] = []
        for i in range(n_rois):
            tc = np.abs(rng.standard_normal(n_bins)).astype(np.float32)
            mvl = float(rng.uniform(0, 1))
            pd = float(rng.uniform(0, 360))
            r = CellResult(
                roi_idx=i,
                activity={"mean_rate": float(rng.uniform(0, 5))},
                hd_all={
                    "tuning_curve": tc,
                    "mvl": mvl,
                    "preferred_direction": pd,
                    "tuning_width": float(rng.uniform(20, 90)),
                    "p_value": float(rng.uniform(0, 1)),
                    "significant": rng.random() < 0.3,
                    "bin_centers": bin_centers,
                },
                hd_light={
                    "tuning_curve": tc * 0.9,
                    "mvl": mvl * 0.9,
                    "preferred_direction": pd,
                    "tuning_width": 60.0,
                    "p_value": 0.05,
                    "significant": False,
                },
                hd_dark={
                    "tuning_curve": tc * 0.8,
                    "mvl": mvl * 0.8,
                    "preferred_direction": pd + 10,
                    "tuning_width": 70.0,
                    "p_value": 0.01,
                    "significant": True,
                },
                hd_comparison={
                    "correlation": float(rng.uniform(0.5, 1.0)),
                    "pd_shift": float(rng.uniform(-30, 30)),
                    "mvl_ratio_dark_over_light": float(rng.uniform(0.5, 1.5)),
                },
                place_all={
                    "spatial_info": float(rng.uniform(0, 2)),
                    "spatial_coherence": float(rng.uniform(0, 1)),
                    "sparsity": float(rng.uniform(0, 1)),
                    "p_value": float(rng.uniform(0, 1)),
                    "significant": False,
                },
                place_light={},
                place_dark={},
                place_comparison={"correlation": float(rng.uniform(0, 1))},
            )
            results.append(r)

        path = tmp_path / "analysis.h5"
        save_analysis_results(
            output_path=path,
            results_by_signal={"dff": results},
            params=params,
            session_id="20220804_13_52_02_1117646",
            n_rois=n_rois,
            n_frames=900,
            fps=9.6,
            signal_types_available=["dff"],
        )

        # Load back
        data = load_analysis_results(path)

        # Verify metadata
        assert data["meta"]["session_id"] == "20220804_13_52_02_1117646"
        assert data["meta"]["n_rois"] == n_rois
        assert data["meta"]["fps"] == pytest.approx(9.6)

        # Verify params
        assert data["params"]["hd_n_bins"] == n_bins
        assert data["params"]["n_shuffles"] == 100

        # Verify signal type results structure
        assert "dff" in data
        dff_data = data["dff"]

        # Activity
        assert "activity" in dff_data
        assert "mean_rate" in dff_data["activity"]
        assert dff_data["activity"]["mean_rate"].shape == (n_rois,)

        # HD tuning — all conditions
        assert "hd" in dff_data
        for condition in ("all", "light", "dark"):
            assert condition in dff_data["hd"]
            cond = dff_data["hd"][condition]
            assert "tuning_curves" in cond
            assert cond["tuning_curves"].shape == (n_rois, n_bins)
            assert "mvl" in cond
            assert cond["mvl"].shape == (n_rois,)
            assert "significant" in cond
            assert cond["significant"].shape == (n_rois,)

        # HD comparison
        assert "comparison" in dff_data["hd"]
        comp = dff_data["hd"]["comparison"]
        assert comp["correlation"].shape == (n_rois,)
        assert comp["pd_shift"].shape == (n_rois,)

        # Place tuning
        assert "place" in dff_data
        assert "all" in dff_data["place"]
        assert dff_data["place"]["all"]["spatial_info"].shape == (n_rois,)

    def test_analysis_multiple_signal_types(self, tmp_path: Path) -> None:
        """analysis.h5 with multiple signal types saves/loads correctly."""
        from hm2p.analysis.run import AnalysisParams, CellResult
        from hm2p.analysis.save import load_analysis_results, save_analysis_results

        n_rois = 3
        n_bins = 36
        params = AnalysisParams(hd_n_bins=n_bins, n_shuffles=50)

        def _make_results() -> list[CellResult]:
            return [
                CellResult(
                    roi_idx=i,
                    activity={"mean_rate": float(i)},
                    hd_all={
                        "tuning_curve": np.ones(n_bins, dtype=np.float32),
                        "mvl": 0.5,
                        "preferred_direction": 180.0,
                        "tuning_width": 45.0,
                        "p_value": 0.01,
                        "significant": True,
                        "bin_centers": np.linspace(5, 355, n_bins, dtype=np.float32),
                    },
                    hd_light={},
                    hd_dark={},
                    hd_comparison={},
                    place_all={},
                    place_light={},
                    place_dark={},
                    place_comparison={},
                )
                for i in range(n_rois)
            ]

        path = tmp_path / "analysis.h5"
        save_analysis_results(
            output_path=path,
            results_by_signal={"dff": _make_results(), "deconv": _make_results()},
            params=params,
            session_id="test",
            n_rois=n_rois,
            n_frames=100,
            fps=9.6,
            signal_types_available=["dff", "deconv"],
        )

        data = load_analysis_results(path)
        assert "dff" in data
        assert "deconv" in data
        assert data["dff"]["activity"]["mean_rate"].shape == (n_rois,)
        assert data["deconv"]["activity"]["mean_rate"].shape == (n_rois,)
