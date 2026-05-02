"""Tests for the Stage 6 sync_status entry guard.

See docs/sync-pipeline-design.md §5.2 and tests/sync/TEST_PLAN.md §5.2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hm2p.analysis.run import check_sync_status, write_skipped_analysis_h5


def _write_sync(
    path: Path,
    *,
    sync_status: str,
    sync_warnings: list[str] | None = None,
    sync_failures: list[str] | None = None,
) -> None:
    from tests.sync.conftest import write_synthetic_sync_h5

    write_synthetic_sync_h5(
        path,
        sync_status=sync_status,
        warnings=sync_warnings or [],
        failures=sync_failures or [],
    )


class TestCheckSyncStatus:
    def test_ok_proceeds(self, tmp_path: Path) -> None:
        path = tmp_path / "sync.h5"
        _write_sync(path, sync_status="OK")
        proceed, status, reason = check_sync_status(path)
        assert proceed is True
        assert status == "OK"
        assert reason == ""

    def test_ok_with_warnings_proceeds(self, tmp_path: Path) -> None:
        path = tmp_path / "sync.h5"
        _write_sync(path, sync_status="OK_WITH_WARNINGS", sync_warnings=["high_camera_jitter"])
        proceed, status, _ = check_sync_status(path)
        assert proceed is True
        assert status == "OK_WITH_WARNINGS"

    def test_failed_blocks_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "sync.h5"
        _write_sync(
            path,
            sync_status="FAILED_FRAME_COUNT_MISMATCH",
            sync_failures=["frame_count_mismatch: pulse_count_diff=42 (threshold=5)"],
        )
        proceed, status, reason = check_sync_status(path)
        assert proceed is False
        assert status == "FAILED_FRAME_COUNT_MISMATCH"
        assert "frame_count" in reason.lower()

    def test_override_proceeds(self, tmp_path: Path) -> None:
        path = tmp_path / "sync.h5"
        _write_sync(path, sync_status="FAILED_NO_TIMESTAMPS")
        proceed, status, _ = check_sync_status(path, include_failed_sync=True)
        assert proceed is True
        assert status == "FAILED_NO_TIMESTAMPS"

    def test_missing_file(self, tmp_path: Path) -> None:
        proceed, status, _ = check_sync_status(tmp_path / "nope.h5")
        assert proceed is False
        assert status == "NO_SYNC_FILE"

    def test_legacy_no_status(self, tmp_path: Path) -> None:
        from hm2p.io.hdf5 import write_h5

        path = tmp_path / "sync.h5"
        write_h5(path, arrays={}, attrs={"session_id": "test"})
        proceed, status, reason = check_sync_status(path)
        assert proceed is False
        assert status == "NO_STATUS"
        assert (
            "rebuild" in reason.lower() or "rerun" in reason.lower() or "re-run" in reason.lower()
        )


class TestWriteSkippedAnalysisH5:
    def test_writes_sentinel_attrs(self, tmp_path: Path) -> None:
        from hm2p.io.hdf5 import read_attrs

        path = tmp_path / "analysis.h5"
        write_skipped_analysis_h5(
            path,
            session_id="test_session",
            skipped_reason="FAILED_FRAME_COUNT_MISMATCH: too many missing frames",
            sync_status="FAILED_FRAME_COUNT_MISMATCH",
        )
        attrs = read_attrs(path)
        assert attrs["session_id"] == "test_session"
        assert attrs["sync_status"] == "FAILED_FRAME_COUNT_MISMATCH"
        assert "FAILED_" in str(attrs["skipped_reason"])

    @pytest.mark.parametrize("status", ["FAILED_NO_TIMESTAMPS", "FAILED_NO_PULSES"])
    def test_sentinel_round_trips(self, tmp_path: Path, status: str) -> None:
        from hm2p.io.hdf5 import read_attrs, read_h5

        path = tmp_path / "analysis.h5"
        write_skipped_analysis_h5(
            path,
            session_id="test_session",
            skipped_reason="reason",
            sync_status=status,
        )
        # No analysis arrays present — only the sentinel attrs.
        arrays = read_h5(path)
        assert arrays == {}
        attrs = read_attrs(path)
        assert attrs["sync_status"] == status
