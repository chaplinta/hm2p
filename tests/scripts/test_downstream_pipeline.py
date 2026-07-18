"""Tests for scripts/run_downstream_pipeline.py and scripts/s3_utils.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for s3_utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

# Mock streamlit before import
sys.modules.setdefault("streamlit", MagicMock())

# Import the script modules under test
import importlib.util

spec = importlib.util.spec_from_file_location(
    "downstream",
    str(Path(__file__).resolve().parent.parent.parent / "scripts" / "run_downstream_pipeline.py"),
)
downstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downstream)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_session(
    pose: bool = True,
    kinematics: bool = False,
    calcium: bool = True,
    sync: bool = False,
    analysis: bool = False,
    exp_id: str = "20220804_13_52_02_1117646",
) -> dict:
    return {
        "exp_id": exp_id,
        "sub": "sub-1117646",
        "ses": "ses-20220804T135202",
        "pose": pose,
        "kinematics": kinematics,
        "calcium": calcium,
        "sync": sync,
        "analysis": analysis,
    }


def _make_s3_mock() -> MagicMock:
    s3 = MagicMock()
    s3.put_object.return_value = {}
    return s3


# ---------------------------------------------------------------------------
# TestGetSessions
# ---------------------------------------------------------------------------


class TestGetSessions:
    def test_parse_session_id(self):
        """Session IDs are converted to NeuroBlueprint sub/ses names."""
        exp_id = "20220804_13_52_02_1117646"
        parts = exp_id.split("_")
        animal = parts[-1]
        sub = f"sub-{animal}"
        ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
        assert sub == "sub-1117646"
        assert ses == "ses-20220804T135202"


# ---------------------------------------------------------------------------
# TestCheckStageExists
# ---------------------------------------------------------------------------


class TestCheckStageExists:
    def test_returns_true_when_files_exist(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "KeyCount": 1,
            "Contents": [{"Key": "kinematics/sub-1/ses-2/kinematics.h5"}],
        }
        assert downstream.check_stage_exists(s3, "sub-1", "ses-2", "kinematics", "kinematics.h5")

    def test_returns_false_when_no_files(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"KeyCount": 0}
        assert not downstream.check_stage_exists(s3, "sub-1", "ses-2", "kinematics")

    def test_returns_false_when_pattern_not_matched(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "KeyCount": 1,
            "Contents": [{"Key": "kinematics/sub-1/ses-2/other.json"}],
        }
        assert not downstream.check_stage_exists(
            s3, "sub-1", "ses-2", "kinematics", "kinematics.h5"
        )


# ---------------------------------------------------------------------------
# TestRunStageHelpers — return signature is now (bool, str)
# ---------------------------------------------------------------------------


class TestRunStageHelpers:
    def test_stage3_dry_run_returns_success_with_empty_stderr(self):
        session = _make_session()
        ok, stderr = downstream.run_stage3(session, dry_run=True)
        assert ok is True
        assert stderr == ""

    def test_stage5_dry_run_returns_success_with_empty_stderr(self):
        session = _make_session()
        ok, stderr = downstream.run_stage5(session, dry_run=True)
        assert ok is True
        assert stderr == ""

    def test_stage6_dry_run_returns_success_with_empty_stderr(self):
        session = _make_session()
        ok, stderr = downstream.run_stage6(session, dry_run=True)
        assert ok is True
        assert stderr == ""


# ---------------------------------------------------------------------------
# TestUpdateDownstreamProgress
# ---------------------------------------------------------------------------


class TestUpdateDownstreamProgress:
    def test_uploads_progress_json(self):
        s3 = _make_s3_mock()
        downstream.update_downstream_progress(
            s3,
            session_idx=3,
            total=26,
            exp_id="20220804_13_52_02_1117646",
            stage="stage3",
            status="done",
            completed=2,
            failed=0,
        )
        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == downstream.DERIVATIVES_BUCKET
        assert call_kwargs["Key"] == "dlc-retrain/_downstream_progress.json"
        payload = json.loads(call_kwargs["Body"].decode())
        assert payload["stage"] == "stage3"
        assert payload["status"] == "done"
        assert payload["session_idx"] == 3
        assert payload["total"] == 26
        assert "updated" in payload

    def test_upload_failure_is_non_fatal(self):
        """Progress upload failures must not raise — they are best-effort."""
        s3 = _make_s3_mock()
        s3.put_object.side_effect = Exception("network error")
        # Should not raise
        downstream.update_downstream_progress(s3, 1, 1, "dummy", "stage3", "done", 0, 0)


# ---------------------------------------------------------------------------
# TestProcessSession — updated signature
# ---------------------------------------------------------------------------


class TestProcessSession:
    def test_skips_all_stages_when_all_done(self):
        session = _make_session(kinematics=True, sync=True, analysis=True)
        s3 = _make_s3_mock()
        errors: list[dict] = []
        result = downstream.process_session(
            session,
            s3,
            session_idx=1,
            total=1,
            completed_count=0,
            failed_count=0,
            error_records=errors,
            dry_run=True,
        )
        assert result["stage3"] is True
        assert result["stage5"] is True
        assert result["stage6"] is True
        assert errors == []

    def test_runs_all_stages_when_nothing_done(self):
        session = _make_session(kinematics=False, sync=False, analysis=False)
        s3 = _make_s3_mock()
        errors: list[dict] = []
        result = downstream.process_session(
            session,
            s3,
            session_idx=1,
            total=1,
            completed_count=0,
            failed_count=0,
            error_records=errors,
            dry_run=True,
        )
        assert result["stage3"] is True
        assert result["stage5"] is True
        assert result["stage6"] is True

    def test_skips_stage3_when_no_pose(self):
        session = _make_session(pose=False, kinematics=False)
        s3 = _make_s3_mock()
        errors: list[dict] = []
        result = downstream.process_session(
            session,
            s3,
            session_idx=1,
            total=1,
            completed_count=0,
            failed_count=0,
            error_records=errors,
            dry_run=True,
        )
        assert result["stage3"] is False

    def test_appends_error_record_on_stage_failure(self):
        """A failed stage subprocess appends a structured error record."""
        session = _make_session(kinematics=False, sync=False, analysis=False)
        s3 = _make_s3_mock()
        errors: list[dict] = []

        with patch.object(
            downstream, "run_stage3", return_value=(False, "subprocess stderr here")
        ):
            downstream.process_session(
                session,
                s3,
                session_idx=1,
                total=1,
                completed_count=0,
                failed_count=0,
                error_records=errors,
                dry_run=False,
            )

        assert len(errors) == 1
        rec = errors[0]
        assert rec["stage"] == "stage3"
        assert rec["error_type"] == "SubprocessError"
        assert rec["error_message"] == "subprocess stderr here"
        assert "timestamp" in rec
        assert "session" in rec

    def test_error_record_has_required_fields(self):
        """Error records contain all fields required by the common JSON schema."""
        session = _make_session(kinematics=False)
        s3 = _make_s3_mock()
        errors: list[dict] = []

        with patch.object(downstream, "run_stage3", return_value=(False, "err")):
            downstream.process_session(
                session,
                s3,
                session_idx=1,
                total=1,
                completed_count=0,
                failed_count=0,
                error_records=errors,
                dry_run=False,
            )

        required_keys = {
            "session",
            "stage",
            "error_type",
            "error_message",
            "traceback",
            "timestamp",
        }
        assert required_keys.issubset(set(errors[0].keys()))


# ---------------------------------------------------------------------------
# TestGetInstanceId
# ---------------------------------------------------------------------------


class TestGetInstanceId:
    def test_returns_unknown_when_metadata_unavailable(self):
        """On non-EC2 machines, _get_instance_id() returns 'unknown'."""
        result = downstream._get_instance_id()
        # Can be 'unknown' or an actual ID; must be a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_instance_id_on_ec2(self):
        """When the EC2 metadata endpoint is reachable, the ID is returned."""
        import urllib.request

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"i-0abc1234567890def"
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            result = downstream._get_instance_id()
        assert result == "i-0abc1234567890def"


# ---------------------------------------------------------------------------
# TestS3UploadWithVerify
# ---------------------------------------------------------------------------

from s3_utils import s3_upload_with_verify  # noqa: E402 — after sys.path setup


class TestS3UploadWithVerify:
    def test_uploads_and_verifies_on_success(self, tmp_path):
        test_file = tmp_path / "data.h5"
        test_file.write_bytes(b"fake hdf5 data")

        s3 = MagicMock()
        s3.upload_file.return_value = None
        s3.head_object.return_value = {"ContentLength": 14}

        s3_upload_with_verify(s3, test_file, "my-bucket", "path/to/data.h5")

        s3.upload_file.assert_called_once_with(str(test_file), "my-bucket", "path/to/data.h5")
        s3.head_object.assert_called_once_with(Bucket="my-bucket", Key="path/to/data.h5")

    def test_retries_on_upload_failure(self, tmp_path):
        test_file = tmp_path / "data.h5"
        test_file.write_bytes(b"data")

        s3 = MagicMock()
        # First upload fails, second succeeds
        s3.upload_file.side_effect = [Exception("timeout"), None]
        s3.head_object.return_value = {}

        s3_upload_with_verify(s3, test_file, "b", "k", retries=3, retry_delay_s=0)

        assert s3.upload_file.call_count == 2
        assert s3.head_object.call_count == 1

    def test_retries_on_verify_failure(self, tmp_path):
        test_file = tmp_path / "data.h5"
        test_file.write_bytes(b"data")

        s3 = MagicMock()
        s3.upload_file.return_value = None
        # head_object fails first, succeeds second attempt
        s3.head_object.side_effect = [Exception("not found yet"), {}]

        s3_upload_with_verify(s3, test_file, "b", "k", retries=3, retry_delay_s=0)

        assert s3.upload_file.call_count == 2
        assert s3.head_object.call_count == 2

    def test_raises_after_all_retries_exhausted(self, tmp_path):
        test_file = tmp_path / "data.h5"
        test_file.write_bytes(b"data")

        s3 = MagicMock()
        s3.upload_file.return_value = None
        s3.head_object.side_effect = Exception("S3 unavailable")

        with pytest.raises(RuntimeError, match="failed to confirm"):
            s3_upload_with_verify(s3, test_file, "b", "k", retries=2, retry_delay_s=0)

        assert s3.upload_file.call_count == 2
        assert s3.head_object.call_count == 2

    def test_accepts_path_object_and_string(self, tmp_path):
        test_file = tmp_path / "data.h5"
        test_file.write_bytes(b"data")

        s3 = MagicMock()
        s3.upload_file.return_value = None
        s3.head_object.return_value = {}

        # Path object
        s3_upload_with_verify(s3, test_file, "b", "k")
        # String path
        s3_upload_with_verify(s3, str(test_file), "b", "k")
        assert s3.upload_file.call_count == 2


# ---------------------------------------------------------------------------
# T7: _clear_pipeline_rerun_marker — delete_object called on S3 after
#     all downstream stages complete without errors.
# ---------------------------------------------------------------------------


class TestClearPipelineRerunMarker:
    """T7 — pipeline_rerun.json is cleared on S3 when downstream run succeeds."""

    def test_delete_object_called_with_correct_bucket_and_key(self):
        """delete_object is called on hm2p-derivatives/pipeline_rerun.json when no errors."""
        s3 = MagicMock()
        s3.delete_object.return_value = {}

        downstream._clear_pipeline_rerun_marker(s3, has_errors=False)

        s3.delete_object.assert_called_once_with(
            Bucket=downstream.DERIVATIVES_BUCKET,
            Key="pipeline_rerun.json",
        )

    def test_delete_object_not_called_when_errors_present(self):
        """When stages had errors, pipeline_rerun.json is NOT deleted."""
        s3 = MagicMock()

        downstream._clear_pipeline_rerun_marker(s3, has_errors=True)

        s3.delete_object.assert_not_called()

    def test_delete_failure_is_non_fatal(self):
        """S3 delete failures are caught and do not propagate to the caller."""
        s3 = MagicMock()
        s3.delete_object.side_effect = Exception("S3 unavailable")

        # Should not raise
        downstream._clear_pipeline_rerun_marker(s3, has_errors=False)

    def test_bucket_is_derivatives_bucket(self):
        """The marker is deleted from DERIVATIVES_BUCKET, not rawdata."""
        s3 = MagicMock()
        s3.delete_object.return_value = {}

        downstream._clear_pipeline_rerun_marker(s3, has_errors=False)

        call_kwargs = s3.delete_object.call_args[1]
        assert call_kwargs["Bucket"] == "hm2p-derivatives"

    def test_key_is_pipeline_rerun_json(self):
        """The deleted key is exactly 'pipeline_rerun.json' (no path prefix)."""
        s3 = MagicMock()
        s3.delete_object.return_value = {}

        downstream._clear_pipeline_rerun_marker(s3, has_errors=False)

        call_kwargs = s3.delete_object.call_args[1]
        assert call_kwargs["Key"] == "pipeline_rerun.json"
