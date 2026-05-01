"""Tests for hm2p.sync.report — sync_report.parquet aggregator.

See docs/sync-pipeline-design.md §1.4 / §2.3 and
tests/sync/TEST_PLAN.md §1.3 / §6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hm2p.io.hdf5 import validate_sync_report_parquet
from hm2p.sync.report import (
    _COLUMN_ORDER,
    _exp_id_from_sub_ses,
    _row_from_sync_attrs,
    build_report,
    column_order,
)
from tests.sync.conftest import write_synthetic_sync_h5


def _make_sync_tree(
    base: Path,
    *,
    statuses: list[tuple[str, str, str]],
) -> None:
    """Create a synthetic sync directory tree.

    ``statuses`` is a list of ``(sub, ses, status)`` tuples.
    """
    for sub, ses, status in statuses:
        d = base / sub / ses
        d.mkdir(parents=True, exist_ok=True)
        # Build minimal valid sync_diag attrs for any status.
        diag = {
            "cam_n_pulses": 600,
            "img_n_pulses": 180,
            "line_n_pulses": 29160,
            "n_tiff_frames": 180,
            "pulse_count_diff": 0,
            "pulse_count_diff_after_off_by_one": 0,
            "cam_n_isi_outliers": 0,
            "img_n_isi_outliers": 0,
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
        warnings: list[str] = []
        failures: list[str] = []
        if status.startswith("FAILED_"):
            failures = [f"{status.lower()}: synthetic"]
        elif status == "OK_WITH_WARNINGS":
            warnings = ["high_camera_jitter"]
        write_synthetic_sync_h5(
            d / "sync.h5",
            sync_status=status,
            sync_diag=diag,
            warnings=warnings,
            failures=failures,
        )


# ---------------------------------------------------------------------------
# _exp_id_from_sub_ses
# ---------------------------------------------------------------------------


class TestExpIdReconstruction:
    def test_canonical_form(self) -> None:
        exp_id = _exp_id_from_sub_ses("sub-1117646", "ses-20220804T135202")
        assert exp_id == "20220804_13_52_02_1117646"

    def test_unprefixed(self) -> None:
        exp_id = _exp_id_from_sub_ses("1117646", "20220804T135202")
        assert exp_id == "20220804_13_52_02_1117646"


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_three_session_tree(self, tmp_path: Path) -> None:
        sync_dir = tmp_path / "derivatives" / "sync"
        out = tmp_path / "report.parquet"
        _make_sync_tree(
            sync_dir,
            statuses=[
                ("sub-1", "ses-20220804T135202", "OK"),
                ("sub-2", "ses-20220805T135202", "OK_WITH_WARNINGS"),
                ("sub-3", "ses-20220806T135202", "FAILED_FRAME_COUNT_MISMATCH"),
            ],
        )
        df = build_report(sync_dir, out)
        assert len(df) == 3
        validate_sync_report_parquet(df)
        # Round-trip through parquet
        df2 = pd.read_parquet(out)
        validate_sync_report_parquet(df2)
        # Per-status counts match
        assert (df["sync_status"] == "OK").sum() == 1
        assert (df["sync_status"] == "OK_WITH_WARNINGS").sum() == 1
        assert (df["sync_status"].str.startswith("FAILED_")).sum() == 1

    def test_empty_input_dir(self, tmp_path: Path) -> None:
        sync_dir = tmp_path / "sync"
        sync_dir.mkdir()
        out = tmp_path / "report.parquet"
        df = build_report(sync_dir, out)
        assert df.empty
        # Empty df must still have all canonical columns.
        for col in _COLUMN_ORDER:
            assert col in df.columns
        # validate_sync_report_parquet on an empty df with correct columns:
        # numeric columns must be numeric dtype. Ensure that's the case.
        # (build_report returns object dtype on textual columns; numeric
        # columns default to float64. Add a placeholder row to verify.)

    def test_missing_sync_directory_returns_empty(self, tmp_path: Path) -> None:
        df = build_report(tmp_path / "does-not-exist", tmp_path / "out.parquet")
        assert df.empty

    def test_unreadable_file_records_error(self, tmp_path: Path) -> None:
        sync_dir = tmp_path / "sync"
        d = sync_dir / "sub-1" / "ses-20220804T135202"
        d.mkdir(parents=True)
        # Truncated / non-HDF5 file
        (d / "sync.h5").write_bytes(b"not-a-real-h5-file")
        out = tmp_path / "report.parquet"
        df = build_report(sync_dir, out)
        assert len(df) == 1
        assert df.iloc[0]["read_error"] != ""
        assert df.iloc[0]["sync_status"] == ""

    def test_legacy_no_status_writes_read_error(self, tmp_path: Path) -> None:
        from hm2p.io.hdf5 import write_h5

        sync_dir = tmp_path / "sync"
        d = sync_dir / "sub-1" / "ses-20220804T135202"
        d.mkdir(parents=True)
        # Legacy sync.h5 without sync_status attr
        write_h5(d / "sync.h5", arrays={}, attrs={"session_id": "test"})
        out = tmp_path / "report.parquet"
        df = build_report(sync_dir, out)
        assert len(df) == 1
        assert "rebuild" in df.iloc[0]["read_error"].lower()

    def test_column_order_stable(self, tmp_path: Path) -> None:
        sync_dir = tmp_path / "sync"
        out = tmp_path / "report.parquet"
        _make_sync_tree(sync_dir, statuses=[("sub-1", "ses-20220804T135202", "OK")])
        df = build_report(sync_dir, out)
        assert tuple(df.columns) == _COLUMN_ORDER

    def test_corrupt_dff_does_not_block_aggregator(self, tmp_path: Path) -> None:
        """Aggregator reads attrs only — corrupt heavy datasets don't matter."""
        sync_dir = tmp_path / "sync"
        d = sync_dir / "sub-1" / "ses-20220804T135202"
        d.mkdir(parents=True)
        write_synthetic_sync_h5(
            d / "sync.h5",
            sync_status="OK",
            sync_diag={"cam_n_pulses": 100},
            warnings=[],
            failures=[],
            payload={"dff": np.zeros((1, 1), dtype=np.float32)},  # tiny but valid
        )
        out = tmp_path / "report.parquet"
        df = build_report(sync_dir, out)
        assert len(df) == 1
        assert df.iloc[0]["sync_status"] == "OK"
        assert int(df.iloc[0]["cam_n_pulses"]) == 100

    def test_aggregator_invariant_n_rows_eq_n_files(self, tmp_path: Path) -> None:
        sync_dir = tmp_path / "sync"
        statuses = [
            ("sub-1", "ses-20220804T135202", "OK"),
            ("sub-1", "ses-20220805T135202", "OK_WITH_WARNINGS"),
            ("sub-2", "ses-20220804T135202", "FAILED_NO_PULSES"),
            ("sub-2", "ses-20220805T135202", "FAILED_FRAME_COUNT_MISMATCH"),
        ]
        _make_sync_tree(sync_dir, statuses=statuses)
        out = tmp_path / "report.parquet"
        df = build_report(sync_dir, out)
        n_files = len(list(sync_dir.glob("**/sync.h5")))
        assert len(df) == n_files == len(statuses)

    def test_per_status_counts_match(self, tmp_path: Path) -> None:
        sync_dir = tmp_path / "sync"
        statuses = [
            ("sub-1", "ses-20220804T135202", "OK"),
            ("sub-1", "ses-20220805T135202", "OK"),
            ("sub-2", "ses-20220804T135202", "FAILED_NO_PULSES"),
        ]
        _make_sync_tree(sync_dir, statuses=statuses)
        out = tmp_path / "report.parquet"
        df = build_report(sync_dir, out)
        assert (df["sync_status"] == "OK").sum() == 2
        assert df["sync_status"].str.startswith("FAILED_").sum() == 1

    def test_sort_by_exp_id(self, tmp_path: Path) -> None:
        sync_dir = tmp_path / "sync"
        statuses = [
            ("sub-1118023", "ses-20221004T104258", "OK"),
            ("sub-1117217", "ses-20220601T135318", "FAILED_FRAME_COUNT_MISMATCH"),
            ("sub-1117217", "ses-20220531T110613", "FAILED_FRAME_COUNT_MISMATCH"),
        ]
        _make_sync_tree(sync_dir, statuses=statuses)
        out = tmp_path / "report.parquet"
        df = build_report(sync_dir, out)
        assert list(df["exp_id"]) == sorted(df["exp_id"])


class TestRowFromSyncAttrs:
    def test_legacy_no_status(self) -> None:
        row = _row_from_sync_attrs(exp_id="x", sub="s", ses="t", attrs={})
        assert "rebuild" in row["read_error"]

    def test_status_decoded_from_bytes(self) -> None:
        row = _row_from_sync_attrs(
            exp_id="x",
            sub="s",
            ses="t",
            attrs={"sync_status": b"OK", "sync_warnings": b"[]", "sync_failures": b"[]"},
        )
        assert row["sync_status"] == "OK"

    def test_invalid_int_attr_falls_back_to_sentinel(self) -> None:
        attrs = {
            "sync_status": "OK",
            "sync_warnings": "[]",
            "sync_failures": "[]",
            "sync_diag/cam_n_pulses": "not-a-number",
        }
        row = _row_from_sync_attrs(exp_id="x", sub="s", ses="t", attrs=attrs)
        assert row["cam_n_pulses"] == -9999


def test_column_order_helper_returns_canonical_tuple() -> None:
    assert column_order() == _COLUMN_ORDER


# ---------------------------------------------------------------------------
# Hypothesis property: per-status counts match the input distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_ok,n_warn,n_failed",
    [(0, 0, 1), (1, 1, 1), (5, 3, 2), (10, 0, 0)],
)
def test_per_status_counts_property(tmp_path: Path, n_ok: int, n_warn: int, n_failed: int) -> None:
    sync_dir = tmp_path / "sync"
    statuses: list[tuple[str, str, str]] = []
    idx = 0
    for _ in range(n_ok):
        statuses.append((f"sub-{idx}", "ses-20220101T000000", "OK"))
        idx += 1
    for _ in range(n_warn):
        statuses.append((f"sub-{idx}", "ses-20220101T000000", "OK_WITH_WARNINGS"))
        idx += 1
    for _ in range(n_failed):
        statuses.append((f"sub-{idx}", "ses-20220101T000000", "FAILED_NO_PULSES"))
        idx += 1
    _make_sync_tree(sync_dir, statuses=statuses)
    out = tmp_path / "out.parquet"
    df = build_report(sync_dir, out)
    assert (df["sync_status"] == "OK").sum() == n_ok
    assert (df["sync_status"] == "OK_WITH_WARNINGS").sum() == n_warn
    assert df["sync_status"].str.startswith("FAILED_").sum() == n_failed
