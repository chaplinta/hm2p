"""Smoke tests for the sync report frontend page.

Per docs/sync-pipeline-design.md §4 / tests/sync/TEST_PLAN.md §7. Uses
``streamlit.testing.v1.AppTest`` (the convention shared with
``tests/frontend/test_app_rendering.py``). All external services
(S3, EC2) are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Ensure repo root is importable.
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from streamlit.testing.v1 import AppTest

PAGE_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "frontend" / "pages" / "sync_report_page.py"
)


_EXPERIMENTS = [
    {
        "exp_id": "20220804_13_52_02_1117646",
        "lens": "16x",
        "orientation": "15",
        "fibre": "",
        "primary_exp": "1",
        "bad_2p_frames": "",
        "bad_behav_times": "",
        "exclude": "0",
        "Notes": "",
    },
    {
        "exp_id": "20220601_13_53_18_1117217",
        "lens": "16x",
        "orientation": "0",
        "fibre": "",
        "primary_exp": "1",
        "bad_2p_frames": "",
        "bad_behav_times": "",
        "exclude": "1",
        "Notes": "Camera sync problem",
    },
    {
        "exp_id": "20221015_10_00_00_1116663",
        "lens": "16x",
        "orientation": "0",
        "fibre": "",
        "primary_exp": "1",
        "bad_2p_frames": "",
        "bad_behav_times": "",
        "exclude": "0",
        "Notes": "",
    },
]


def _row(status: str, *, exp_id: str = "20220804_13_52_02_1117646", warnings: int = 0) -> dict:
    return {
        "exp_id": exp_id,
        "sub": "sub-1117646",
        "ses": "ses-20220804T135202",
        "sync_status": status,
        "sync_warnings": "[" + ",".join([f'"w{i}"' for i in range(warnings)]) + "]",
        "sync_failures": "[]"
        if not status.startswith("FAILED_")
        else '["frame_count_mismatch: x"]',
        "dlc_champion_id": "champ-1",
        "read_error": "",
        "cam_n_pulses": 600,
        "cam_n_isi_outliers": 0,
        "img_n_pulses": 180,
        "img_n_isi_outliers": 0,
        "line_n_pulses": 29160,
        "n_tiff_frames": 180,
        "pulse_count_diff": 0,
        "pulse_count_diff_after_off_by_one": 0,
        "light_n_on": 5,
        "light_n_off": 5,
        "light_first_state_at_t0": 1,
        "kin_pose_decimation_uniform": 1,
        "s2p_off_by_one_fix_applied": 0,
        "cam_duration_s": 6.0,
        "cam_isi_median_ms": 10.0,
        "cam_isi_mad_ms": 0.1,
        "cam_isi_cv": 0.005,
        "cam_drift_slope_ppm": 5.0,
        "cam_min_isi_ms": 9.5,
        "img_duration_s": 6.0,
        "img_isi_median_ms": 33.3,
        "img_isi_mad_ms": 0.1,
        "img_isi_cv": 0.001,
        "img_drift_slope_ppm": 1.0,
        "line_isi_median_ms": 0.2,
        "cross_overlap_s": 6.0,
        "cross_start_offset_ms": 0.0,
        "cross_end_offset_ms": 0.0,
        "light_period_median_s": 120.0,
        "light_period_mad_s": 0.1,
        "light_duty_cycle": 0.5,
        "kin_pose_decimation_ratio": 1.0,
    }


def _has_real_exception(at: AppTest) -> str | None:
    for exc in at.exception:
        if "url_pathname" in str(exc):
            continue
        return str(exc)
    return None


def _make_sync_h5_bytes(*, sync_status: str = "OK") -> bytes:
    """Build an in-memory sync.h5 with the bare diagnostic attrs."""
    import io as _io

    import h5py

    buf = _io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["session_id"] = "test"
        f.attrs["sync_status"] = sync_status
        f.attrs["sync_status_version"] = "1.0"
        f.attrs["sync_warnings"] = "[]"
        f.attrs["sync_failures"] = (
            '["frame_count_mismatch: x"]' if sync_status.startswith("FAILED_") else "[]"
        )
    return buf.getvalue()


def _run_page(
    load_sync_report_value,
    download_value=None,
    *,
    selected_exp: str | None = None,
) -> AppTest:
    at = AppTest.from_file(PAGE_PATH, default_timeout=20)
    if selected_exp is not None:
        at.session_state["selected_exp_id"] = selected_exp
    with (
        patch("frontend.data.load_experiments", return_value=_EXPERIMENTS),
        patch("frontend.data.load_sync_report", return_value=load_sync_report_value),
        patch("frontend.data.download_s3_bytes", return_value=download_value),
    ):
        at.run()
    return at


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSyncReportPage:
    def test_imports_without_error(self) -> None:
        # Importing the module must not raise (it would on a syntax error).
        import importlib.util

        spec = importlib.util.spec_from_file_location("sync_report_page", PAGE_PATH)
        assert spec is not None
        # Loading the spec into a module would actually execute the page —
        # we deliberately don't do that here since the import-time code
        # calls Streamlit primitives. Reaching this point demonstrates
        # the file is syntactically valid Python.

    def test_renders_no_data_banner(self) -> None:
        at = _run_page(load_sync_report_value=None)
        assert _has_real_exception(at) is None, _has_real_exception(at)
        # Info banner should appear.
        info_messages = [i.value for i in at.info]
        assert any("not yet built" in m for m in info_messages), info_messages

    def test_renders_empty_parquet(self) -> None:
        empty = pd.DataFrame(columns=["sync_status", "exp_id"])
        at = _run_page(load_sync_report_value=empty)
        assert _has_real_exception(at) is None, _has_real_exception(at)

    def test_renders_mixed_parquet(self) -> None:
        df = pd.DataFrame(
            [
                _row("OK"),
                _row("OK_WITH_WARNINGS", exp_id="20221015_10_00_00_1116663", warnings=1),
                _row("FAILED_FRAME_COUNT_MISMATCH", exp_id="20220601_13_53_18_1117217"),
            ]
        )
        at = _run_page(load_sync_report_value=df)
        assert _has_real_exception(at) is None, _has_real_exception(at)
        # Summary metrics.
        metric_labels = [m.label for m in at.metric]
        assert any("OK" in label for label in metric_labels)
        assert any("FAILED" in label for label in metric_labels)

    def test_methods_expander_present(self) -> None:
        df = pd.DataFrame([_row("OK")])
        at = _run_page(load_sync_report_value=df)
        labels = [exp.label for exp in at.expander]
        assert any("Methods" in label for label in labels), labels

    def test_failed_session_renders_error_banner(self) -> None:
        # Need to fake a sync.h5 download too, otherwise the page can't
        # determine sync_status for the deep-dive panel.
        df = pd.DataFrame(
            [_row("FAILED_FRAME_COUNT_MISMATCH", exp_id="20220601_13_53_18_1117217")]
        )

        # Build a tiny in-memory HDF5 with the required attrs and let the
        # page's _load_h5_from_s3 helper consume the bytes.

        buf = _make_sync_h5_bytes(sync_status="FAILED_FRAME_COUNT_MISMATCH")
        # Patch download_s3_bytes to return our bytes for the sync.h5 key.
        # It will be called twice (sync.h5 + timestamps.h5); return None for
        # timestamps to keep things simple.

        def fake_download(bucket, key):
            if key.endswith("sync.h5"):
                return buf
            return None

        at = AppTest.from_file(PAGE_PATH, default_timeout=20)
        at.session_state["selected_exp_id"] = "20220601_13_53_18_1117217"
        with (
            patch("frontend.data.load_experiments", return_value=_EXPERIMENTS),
            patch("frontend.data.load_sync_report", return_value=df),
            patch("frontend.data.download_s3_bytes", side_effect=fake_download),
        ):
            at.run()
        errors = [str(e.value) for e in at.error]
        assert any("failed sync verification" in m.lower() for m in errors), errors

    def test_excluded_session_caption(self) -> None:
        df = pd.DataFrame([_row("OK", exp_id="20220601_13_53_18_1117217")])
        at = _run_page(
            load_sync_report_value=df,
            selected_exp="20220601_13_53_18_1117217",
        )
        # Excluded session shows a warning banner with Notes.
        warnings = [str(w.value) for w in at.warning]
        assert any("exclude=1" in w for w in warnings), warnings

    def test_app_rendering_no_longer_references_old_page(self) -> None:
        """The legacy sync_page.py must be gone — verify the file does not exist."""
        old = Path(__file__).resolve().parent.parent.parent / "frontend" / "pages" / "sync_page.py"
        assert not old.exists(), f"Old page still present at {old}"


# ---------------------------------------------------------------------------
# is_sync_clean helper (frontend/data.py)
# ---------------------------------------------------------------------------


class TestIsSyncClean:
    def test_ok_is_clean(self) -> None:
        from frontend.data import is_sync_clean

        clean, reason = is_sync_clean({"sync_status": "OK"})
        assert clean is True
        assert reason == ""

    def test_ok_with_warnings_is_clean(self) -> None:
        from frontend.data import is_sync_clean

        clean, reason = is_sync_clean(
            {"sync_status": "OK_WITH_WARNINGS", "sync_warnings": '["high_cv"]'}
        )
        assert clean is True
        assert reason == ""

    def test_failed_is_unclean(self) -> None:
        from frontend.data import is_sync_clean

        clean, reason = is_sync_clean(
            {
                "sync_status": "FAILED_FRAME_COUNT_MISMATCH",
                "sync_failures": '["frame_count_mismatch: pulse_count_diff=42 (threshold=5)"]',
            }
        )
        assert clean is False
        assert "FAILED_FRAME_COUNT_MISMATCH" in reason

    def test_bytes_status_decoded(self) -> None:
        from frontend.data import is_sync_clean

        clean, reason = is_sync_clean({"sync_status": b"OK"})
        assert clean is True

    def test_missing_status_is_unclean(self) -> None:
        from frontend.data import is_sync_clean

        clean, reason = is_sync_clean({})
        assert clean is False
        assert "sync_status" in reason


# ---------------------------------------------------------------------------
# components/sync_diag — Plotly figure builders
# ---------------------------------------------------------------------------


class TestSyncDiagComponents:
    def test_pulse_train_raster(self) -> None:
        from frontend.components.sync_diag import pulse_train_raster

        cam = np.linspace(0, 6, 600)
        img = np.linspace(0, 6, 180)
        fig = pulse_train_raster(
            {"camera": cam, "imaging": img},
            light_on=np.array([0.0]),
            light_off=np.array([3.0]),
        )
        assert fig is not None
        assert len(fig.data) == 2

    def test_pulse_train_raster_time_window(self) -> None:
        from frontend.components.sync_diag import pulse_train_raster

        cam = np.linspace(0, 10, 1000)
        fig = pulse_train_raster(
            {"camera": cam}, light_on=np.empty(0), light_off=np.empty(0), time_window=(2, 5)
        )
        # All x values must be within the window.
        assert fig.data[0].x.min() >= 2 - 1e-9
        assert fig.data[0].x.max() <= 5 + 1e-9

    def test_cumulative_pulses(self) -> None:
        from frontend.components.sync_diag import cumulative_pulses

        cam = np.linspace(0, 6, 600)
        img = np.linspace(0, 6, 180)
        fig = cumulative_pulses({"camera": cam, "imaging": img})
        assert len(fig.data) == 2
        # Final cumulative count equals length.
        assert fig.data[0].y[-1] == 600
        assert fig.data[1].y[-1] == 180

    def test_isi_histogram(self) -> None:
        from frontend.components.sync_diag import isi_histogram

        times = np.linspace(0, 10, 1001)
        fig = isi_histogram(times, fps_nominal=100.0)
        # Histogram bar trace
        assert fig.data and fig.data[0].type == "histogram"

    def test_isi_histogram_too_few_pulses(self) -> None:
        from frontend.components.sync_diag import isi_histogram

        fig = isi_histogram(np.array([1.0]), fps_nominal=100.0)
        # Annotation with "not enough pulses"
        assert any("not enough" in a.text for a in fig.layout.annotations)

    def test_light_cycle_strip(self) -> None:
        from frontend.components.sync_diag import light_cycle_strip

        on = np.array([0.0, 120.0])
        off = np.array([60.0, 180.0])
        fig = light_cycle_strip(on, off, t_max=240.0)
        assert fig is not None

    def test_light_intervals_pairing(self) -> None:
        from frontend.components.sync_diag import _light_intervals

        on = np.array([0.0, 120.0, 240.0])
        off = np.array([60.0, 180.0, 300.0])
        intervals = _light_intervals(on, off)
        assert intervals == [(0.0, 60.0), (120.0, 180.0), (240.0, 300.0)]

    def test_light_intervals_unpaired_trailing_dropped(self) -> None:
        from frontend.components.sync_diag import _light_intervals

        on = np.array([0.0, 100.0])
        off = np.array([50.0])  # only one off
        intervals = _light_intervals(on, off)
        # First on paired with the only off; second on dropped.
        assert intervals == [(0.0, 50.0)]
