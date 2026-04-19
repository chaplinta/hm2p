"""Tests for frontend.data module."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock streamlit before importing the module under test.
# st.cache_data must act as a passthrough decorator (with optional kwargs).
# ---------------------------------------------------------------------------


def _passthrough_decorator(*args, **kwargs):
    """Mimic st.cache_data — return the function unchanged."""
    if args and callable(args[0]):
        # Called as @st.cache_data (no parens)
        return args[0]
    # Called as @st.cache_data(ttl=300) — return identity decorator
    def wrapper(fn):
        return fn
    return wrapper


_st_mock = MagicMock()
_st_mock.cache_data = _passthrough_decorator

# Save the original streamlit module (if any) so we can restore it after import.
_orig_st = sys.modules.get("streamlit")
_orig_data = sys.modules.get("frontend.data")

# Force mock streamlit before (re)importing frontend.data so
# @st.cache_data acts as a passthrough even when the full test suite has
# already imported the real streamlit module.
sys.modules["streamlit"] = _st_mock
if "frontend.data" in sys.modules:
    del sys.modules["frontend.data"]

# Now it is safe to import the module under test.
from frontend.data import (  # noqa: E402
    DERIVATIVES_BUCKET,
    DOWNSTREAM_DEPS,
    RAWDATA_BUCKET,
    REGION,
    STAGE_PREFIXES,
    check_stale_data_warning,
    download_s3_bytes,
    download_s3_numpy,
    get_ec2_instances,
    get_pipeline_status,
    get_progress,
    get_s3_client,
    list_s3_session_files,
    load_all_sync_data,
    load_animals,
    load_experiments,
    parse_session_id,
)

# Restore original streamlit module so other test files aren't affected.
if _orig_st is not None:
    sys.modules["streamlit"] = _orig_st
else:
    sys.modules.pop("streamlit", None)


# ===================================================================
# parse_session_id
# ===================================================================


class TestParseSessionId:
    """Tests for parse_session_id()."""

    def test_standard_format(self):
        sub, ses = parse_session_id("20220804_13_52_02_1117646")
        assert sub == "sub-1117646"
        assert ses == "ses-20220804T135202"

    def test_different_animal_id(self):
        sub, ses = parse_session_id("20230115_09_01_30_9999999")
        assert sub == "sub-9999999"
        assert ses == "ses-20230115T090130"

    def test_midnight_time(self):
        sub, ses = parse_session_id("20221231_00_00_00_1234567")
        assert sub == "sub-1234567"
        assert ses == "ses-20221231T000000"

    def test_end_of_day_time(self):
        sub, ses = parse_session_id("20220101_23_59_59_42")
        assert sub == "sub-42"
        assert ses == "ses-20220101T235959"

    def test_short_animal_id(self):
        sub, ses = parse_session_id("20220101_10_20_30_1")
        assert sub == "sub-1"
        assert ses == "ses-20220101T102030"

    def test_sub_prefix(self):
        """sub- prefix is always added."""
        sub, _ = parse_session_id("20220101_10_20_30_ABC")
        assert sub == "sub-ABC"

    def test_ses_prefix_and_T_separator(self):
        """ses- prefix uses T separator between date and time."""
        _, ses = parse_session_id("20220804_13_52_02_1117646")
        assert ses.startswith("ses-")
        assert "T" in ses

    def test_raises_on_too_few_parts(self):
        """Fewer than 5 underscore-delimited parts should raise."""
        with pytest.raises((IndexError, ValueError)):
            parse_session_id("20220804_13_52")

    def test_raises_on_empty(self):
        with pytest.raises((IndexError, ValueError)):
            parse_session_id("")


# ===================================================================
# STAGE_PREFIXES
# ===================================================================


class TestStagePrefixes:
    """Verify the STAGE_PREFIXES constant."""

    def test_has_seven_stages(self):
        assert len(STAGE_PREFIXES) == 7

    def test_expected_keys(self):
        assert set(STAGE_PREFIXES.keys()) == {
            "ca_extraction",
            "dlc_training",
            "pose",
            "kinematics",
            "calcium",
            "sync",
            "analysis",
        }

    def test_ca_extraction_label(self):
        assert STAGE_PREFIXES["ca_extraction"] == "Stage 1 — Suite2p"

    def test_dlc_training_label(self):
        assert STAGE_PREFIXES["dlc_training"] == "Stage 2a — DLC Training"

    def test_pose_label(self):
        assert STAGE_PREFIXES["pose"] == "Stage 2b — DLC Inference"

    def test_kinematics_label(self):
        assert STAGE_PREFIXES["kinematics"] == "Stage 3 — Kinematics"

    def test_calcium_label(self):
        assert STAGE_PREFIXES["calcium"] == "Stage 4 — Calcium"

    def test_sync_label(self):
        assert STAGE_PREFIXES["sync"] == "Stage 5 — Sync"

    def test_dlc_training_is_inference_dependency(self):
        from frontend.data import DOWNSTREAM_DEPS
        assert "pose" in DOWNSTREAM_DEPS["dlc_training"]

    def test_pose_label_says_inference(self):
        assert "Inference" in STAGE_PREFIXES["pose"]

    def test_dlc_training_label_says_training(self):
        assert "Training" in STAGE_PREFIXES["dlc_training"]


# ===================================================================
# Constants
# ===================================================================


class TestConstants:
    def test_region(self):
        assert REGION == "ap-southeast-2"

    def test_rawdata_bucket(self):
        assert RAWDATA_BUCKET == "hm2p-rawdata"

    def test_derivatives_bucket(self):
        assert DERIVATIVES_BUCKET == "hm2p-derivatives"


# ===================================================================
# load_experiments / load_animals
# ===================================================================


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Helper — write a list of dicts as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestLoadExperiments:
    def test_loads_csv_rows(self, tmp_path):
        csv_path = tmp_path / "experiments.csv"
        rows = [
            {"exp_id": "20220804_13_52_02_1117646", "notes": "ok"},
            {"exp_id": "20221018_10_56_17_1117788", "notes": "good"},
        ]
        _write_csv(csv_path, rows)

        with patch("frontend.data.METADATA_DIR", tmp_path):
            result = load_experiments()

        assert len(result) == 2
        assert result[0]["exp_id"] == "20220804_13_52_02_1117646"
        assert result[1]["notes"] == "good"

    def test_returns_list_of_dicts(self, tmp_path):
        _write_csv(tmp_path / "experiments.csv", [{"col": "val"}])
        with patch("frontend.data.METADATA_DIR", tmp_path):
            result = load_experiments()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_empty_csv(self, tmp_path):
        # Header only, no data rows
        (tmp_path / "experiments.csv").write_text("exp_id,notes\n")
        with patch("frontend.data.METADATA_DIR", tmp_path):
            result = load_experiments()
        assert result == []

    def test_missing_file_raises(self, tmp_path):
        with patch("frontend.data.METADATA_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_experiments()


class TestLoadAnimals:
    def test_loads_csv_rows(self, tmp_path):
        csv_path = tmp_path / "animals.csv"
        rows = [
            {"animal_id": "1117646", "celltype": "penk"},
            {"animal_id": "1117788", "celltype": "nonpenk"},
        ]
        _write_csv(csv_path, rows)

        with patch("frontend.data.METADATA_DIR", tmp_path):
            result = load_animals()

        assert len(result) == 2
        assert result[0]["animal_id"] == "1117646"
        assert result[1]["celltype"] == "nonpenk"

    def test_missing_file_raises(self, tmp_path):
        with patch("frontend.data.METADATA_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_animals()


# ===================================================================
# get_s3_client
# ===================================================================


class TestGetS3Client:
    @patch("frontend.data.boto3")
    def test_creates_client_with_region(self, mock_boto3):
        get_s3_client()
        mock_boto3.client.assert_called_once_with("s3", region_name="ap-southeast-2")


# ===================================================================
# get_pipeline_status
# ===================================================================


class TestGetPipelineStatus:
    @patch("frontend.data.get_s3_client")
    @patch("frontend.data.load_experiments")
    def test_returns_status_dict(self, mock_load, mock_s3):
        mock_load.return_value = [
            {"exp_id": "20220804_13_52_02_1117646"},
        ]
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"KeyCount": 1}
        mock_s3.return_value = mock_client

        result = get_pipeline_status()

        assert "20220804_13_52_02_1117646" in result
        session_status = result["20220804_13_52_02_1117646"]
        # All stages should be True (KeyCount=1)
        for prefix in STAGE_PREFIXES:
            assert session_status[prefix] is True

    @patch("frontend.data.get_s3_client")
    @patch("frontend.data.load_experiments")
    def test_missing_stage_returns_false(self, mock_load, mock_s3):
        mock_load.return_value = [
            {"exp_id": "20220804_13_52_02_1117646"},
        ]
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"KeyCount": 0}
        mock_s3.return_value = mock_client

        result = get_pipeline_status()
        for prefix in STAGE_PREFIXES:
            assert result["20220804_13_52_02_1117646"][prefix] is False

    @patch("frontend.data.get_s3_client")
    @patch("frontend.data.load_experiments")
    def test_s3_error_returns_false(self, mock_load, mock_s3):
        mock_load.return_value = [
            {"exp_id": "20220804_13_52_02_1117646"},
        ]
        mock_client = MagicMock()
        mock_client.list_objects_v2.side_effect = Exception("Network error")
        mock_s3.return_value = mock_client

        result = get_pipeline_status()
        for prefix in STAGE_PREFIXES:
            assert result["20220804_13_52_02_1117646"][prefix] is False

    @patch("frontend.data.get_s3_client")
    @patch("frontend.data.load_experiments")
    def test_correct_s3_prefix_construction(self, mock_load, mock_s3):
        mock_load.return_value = [
            {"exp_id": "20220804_13_52_02_1117646"},
        ]
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"KeyCount": 0}
        mock_s3.return_value = mock_client

        get_pipeline_status()

        # Verify S3 was queried with correct prefixes
        calls = mock_client.list_objects_v2.call_args_list
        prefixes_queried = [c.kwargs["Prefix"] for c in calls]
        assert "ca_extraction/sub-1117646/ses-20220804T135202/" in prefixes_queried
        assert "sync/sub-1117646/ses-20220804T135202/" in prefixes_queried


# ===================================================================
# get_progress
# ===================================================================


class TestGetProgress:
    @patch("frontend.data.get_s3_client")
    def test_returns_json_data(self, mock_s3):
        progress_data = {"status": "running", "pct": 50}
        body_mock = MagicMock()
        body_mock.read.return_value = json.dumps(progress_data).encode()
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": body_mock}
        mock_s3.return_value = mock_client

        result = get_progress("ca_extraction")

        assert result == progress_data
        mock_client.get_object.assert_called_once_with(
            Bucket=DERIVATIVES_BUCKET, Key="ca_extraction/_progress.json"
        )

    @patch("frontend.data.get_s3_client")
    def test_returns_none_on_no_such_key(self, mock_s3):
        mock_client = MagicMock()
        # Simulate NoSuchKey exception
        exc_cls = type("NoSuchKey", (Exception,), {})
        mock_client.exceptions.NoSuchKey = exc_cls
        mock_client.get_object.side_effect = exc_cls("Not found")
        mock_s3.return_value = mock_client

        result = get_progress("nonexistent_stage")
        assert result is None

    @patch("frontend.data.get_s3_client")
    def test_returns_none_on_generic_error(self, mock_s3):
        mock_client = MagicMock()
        mock_client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        mock_client.get_object.side_effect = RuntimeError("boom")
        mock_s3.return_value = mock_client

        result = get_progress("ca_extraction")
        assert result is None


# ===================================================================
# get_ec2_instances
# ===================================================================


class TestGetEC2Instances:
    @patch("frontend.data.boto3")
    def test_returns_instance_list(self, mock_boto3):
        mock_ec2 = MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-abc123",
                            "InstanceType": "g4dn.xlarge",
                            "State": {"Name": "running"},
                            "PublicIpAddress": "1.2.3.4",
                            "LaunchTime": "2026-03-07T10:00:00Z",
                            "Tags": [{"Key": "Project", "Value": "hm2p-suite2p"}],
                        }
                    ]
                }
            ]
        }

        result = get_ec2_instances()

        assert len(result) == 1
        assert result[0]["id"] == "i-abc123"
        assert result[0]["type"] == "g4dn.xlarge"
        assert result[0]["state"] == "running"
        assert result[0]["ip"] == "1.2.3.4"
        assert result[0]["project"] == "hm2p-suite2p"

    @patch("frontend.data.boto3")
    def test_missing_tags_and_ip(self, mock_boto3):
        mock_ec2 = MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-xyz",
                            "InstanceType": "c5.large",
                            "State": {"Name": "pending"},
                        }
                    ]
                }
            ]
        }

        result = get_ec2_instances()
        assert result[0]["ip"] == ""
        assert result[0]["project"] == ""

    @patch("frontend.data.boto3")
    def test_returns_empty_on_error(self, mock_boto3):
        mock_ec2 = MagicMock()
        mock_boto3.client.return_value = mock_ec2
        mock_ec2.describe_instances.side_effect = Exception("API error")

        result = get_ec2_instances()
        assert result == []


# ===================================================================
# list_s3_session_files
# ===================================================================


class TestListS3SessionFiles:
    @patch("frontend.data.get_s3_client")
    def test_lists_files(self, mock_s3):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "ca_extraction/sub-1/ses-2/file.npy",
                        "Size": 1_000_000,
                        "LastModified": "2026-03-07",
                    }
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        mock_s3.return_value = mock_client

        result = list_s3_session_files("hm2p-derivatives", "ca_extraction/sub-1/ses-2/")

        assert len(result) == 1
        assert result[0]["key"] == "ca_extraction/sub-1/ses-2/file.npy"
        assert result[0]["size_mb"] == 1.0

    @patch("frontend.data.get_s3_client")
    def test_empty_prefix(self, mock_s3):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_client.get_paginator.return_value = mock_paginator
        mock_s3.return_value = mock_client

        result = list_s3_session_files("bucket", "nonexistent/")
        assert result == []

    @patch("frontend.data.get_s3_client")
    def test_no_contents_key(self, mock_s3):
        """S3 returns no 'Contents' key when prefix has no objects."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{}]
        mock_client.get_paginator.return_value = mock_paginator
        mock_s3.return_value = mock_client

        result = list_s3_session_files("bucket", "empty/")
        assert result == []

    @patch("frontend.data.get_s3_client")
    def test_returns_empty_on_error(self, mock_s3):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = Exception("S3 error")
        mock_s3.return_value = mock_client

        result = list_s3_session_files("bucket", "prefix/")
        assert result == []


# ===================================================================
# download_s3_bytes
# ===================================================================


class TestDownloadS3Bytes:
    @patch("frontend.data.get_s3_client")
    def test_returns_bytes(self, mock_s3):
        body_mock = MagicMock()
        body_mock.read.return_value = b"hello world"
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": body_mock}
        mock_s3.return_value = mock_client

        result = download_s3_bytes("bucket", "key.bin")

        assert result == b"hello world"
        mock_client.get_object.assert_called_once_with(Bucket="bucket", Key="key.bin")

    @patch("frontend.data.get_s3_client")
    def test_returns_none_on_error(self, mock_s3):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = Exception("Not found")
        mock_s3.return_value = mock_client

        result = download_s3_bytes("bucket", "missing.bin")
        assert result is None


# ===================================================================
# download_s3_numpy
# ===================================================================


class TestDownloadS3Numpy:
    @patch("frontend.data.download_s3_bytes")
    def test_loads_npy(self, mock_download):
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0])
        buf = io.BytesIO()
        np.save(buf, arr)
        mock_download.return_value = buf.getvalue()

        result = download_s3_numpy("bucket", "data.npy")

        assert result is not None
        np.testing.assert_array_equal(result, arr)

    @patch("frontend.data.download_s3_bytes")
    def test_returns_none_when_download_fails(self, mock_download):
        mock_download.return_value = None

        result = download_s3_numpy("bucket", "missing.npy")
        assert result is None


# ===================================================================
# T1: _count_cascade_outputs — spikes key absent/present
# ===================================================================


def _make_ca_h5_bytes(include_spikes: bool) -> bytes:
    """Build an in-memory ca.h5 with dff and deconv; optionally add spikes."""
    import io as _io

    import h5py
    import numpy as np

    n_rois, n_frames = 5, 100
    buf = _io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("dff", data=np.zeros((n_rois, n_frames), dtype=np.float32))
        f.create_dataset("deconv", data=np.zeros((n_rois, n_frames), dtype=np.float32))
        if include_spikes:
            f.create_dataset("spikes", data=np.zeros((n_rois, n_frames), dtype=np.float32))
    buf.seek(0)
    return buf.read()


class TestCountCascadeOutputs:
    """T1 — _count_cascade_outputs returns 0/26 based on spikes key."""

    @patch("frontend.data.download_s3_bytes")
    def test_no_spikes_key_returns_zero(self, mock_dl):
        """CASCADE count is 0 when spikes key is absent from ca.h5."""
        from frontend.data import _count_cascade_outputs

        mock_dl.return_value = _make_ca_h5_bytes(include_spikes=False)
        result = _count_cascade_outputs()
        assert result == 0

    @patch("frontend.data.download_s3_bytes")
    def test_with_spikes_key_returns_26(self, mock_dl):
        """CASCADE count is 26 when spikes key is present."""
        from frontend.data import _count_cascade_outputs

        mock_dl.return_value = _make_ca_h5_bytes(include_spikes=True)
        result = _count_cascade_outputs()
        assert result == 26

    @patch("frontend.data.download_s3_bytes")
    def test_download_returns_none_gives_zero(self, mock_dl):
        """If S3 download fails (returns None), count is 0."""
        from frontend.data import _count_cascade_outputs

        mock_dl.return_value = None
        result = _count_cascade_outputs()
        assert result == 0

    @patch("frontend.data.download_s3_bytes")
    def test_corrupt_bytes_returns_zero(self, mock_dl):
        """Corrupt HDF5 bytes are handled gracefully — count returns 0."""
        from frontend.data import _count_cascade_outputs

        mock_dl.return_value = b"not valid hdf5 bytes"
        result = _count_cascade_outputs()
        assert result == 0


# ===================================================================
# T3: check_stale_data_warning — blocks or warns based on rerun status
# ===================================================================


class TestCheckStaleDataWarning:
    """T3 — check_stale_data_warning calls st.stop when block=True."""

    @patch("frontend.data._get_rerun_status")
    def test_blocks_when_sync_stage_is_rerunning(self, mock_rerun):
        """check_stale_data_warning calls st.stop when upstream is re-running."""
        mock_rerun.return_value = {
            "rerunning": ["pose"],
            "reason": "DLC inference running",
        }
        # pose is upstream of sync via DOWNSTREAM_DEPS["pose"]
        mock_st = MagicMock()
        with patch("frontend.data.st", mock_st):
            check_stale_data_warning(stages=["sync"], block=True)

        mock_st.stop.assert_called_once()
        mock_st.error.assert_called_once()

    @patch("frontend.data._get_rerun_status")
    def test_does_not_block_when_nothing_is_rerunning(self, mock_rerun):
        """check_stale_data_warning does not call st.stop when pipeline is idle."""
        mock_rerun.return_value = {}
        mock_st = MagicMock()
        with patch("frontend.data.st", mock_st):
            result = check_stale_data_warning(stages=["sync"], block=True)

        mock_st.stop.assert_not_called()
        assert result is False

    @patch("frontend.data._get_rerun_status")
    def test_warns_not_stops_when_block_false(self, mock_rerun):
        """When block=False, a warning is shown but st.stop is not called."""
        mock_rerun.return_value = {
            "rerunning": ["pose"],
            "reason": "DLC running",
        }
        mock_st = MagicMock()
        with patch("frontend.data.st", mock_st):
            result = check_stale_data_warning(stages=["sync"], block=False)

        mock_st.stop.assert_not_called()
        mock_st.warning.assert_called_once()
        assert result is True

    @patch("frontend.data._get_rerun_status")
    def test_returns_false_when_rerunning_stage_does_not_affect_checked_stages(
        self, mock_rerun
    ):
        """A calcium re-run does not affect kinematics pages."""
        mock_rerun.return_value = {
            "rerunning": ["calcium"],
            "reason": "Calcium re-running",
        }
        mock_st = MagicMock()
        with patch("frontend.data.st", mock_st):
            # kinematics is not downstream of calcium
            result = check_stale_data_warning(stages=["kinematics"], block=True)

        mock_st.stop.assert_not_called()
        assert result is False

    @patch("frontend.data._get_rerun_status")
    def test_default_stages_cover_sync_and_analysis(self, mock_rerun):
        """Default stages parameter covers sync and analysis."""
        mock_rerun.return_value = {
            "rerunning": ["pose"],
            "reason": "DLC running",
        }
        mock_st = MagicMock()
        with patch("frontend.data.st", mock_st):
            # Call without explicit stages — should default to ["sync","analysis"]
            check_stale_data_warning(block=True)

        # pose is upstream of sync and analysis — should block
        mock_st.stop.assert_called_once()


# ===================================================================
# T4: load_all_sync_data calls check_stale_data_warning first
# ===================================================================


class TestLoadAllSyncDataStalenessCheck:
    """T4 — load_all_sync_data invokes check_stale_data_warning before returning."""

    @patch("frontend.data._fetch_all_sync_data")
    @patch("frontend.data.check_stale_data_warning")
    def test_staleness_check_is_called_before_fetch(
        self, mock_staleness, mock_fetch
    ):
        """load_all_sync_data calls check_stale_data_warning(stages=['sync'], block=True)."""
        mock_staleness.return_value = False
        mock_fetch.return_value = {"sessions": [], "n_sessions": 0, "n_total_rois": 0}

        mock_st = MagicMock()
        mock_st.session_state = {}
        with patch("frontend.data.st", mock_st):
            load_all_sync_data()

        mock_staleness.assert_called_once_with(stages=["sync"], block=True)

    @patch("frontend.data._fetch_all_sync_data")
    @patch("frontend.data.check_stale_data_warning")
    def test_staleness_check_called_even_when_cached(
        self, mock_staleness, mock_fetch
    ):
        """Staleness check runs even if data is already in session_state cache."""
        mock_staleness.return_value = False

        cached_data = {"sessions": [{"exp_id": "dummy"}], "n_sessions": 1, "n_total_rois": 5}

        mock_st = MagicMock()
        mock_st.session_state = {"_hm2p_cache_sync_data": cached_data}
        with patch("frontend.data.st", mock_st):
            result = load_all_sync_data()

        mock_staleness.assert_called_once_with(stages=["sync"], block=True)
        # Data served from cache when available
        assert result == cached_data
        mock_fetch.assert_not_called()


# ===================================================================
# T5: sync.h5 key set matches frontend read contract
# ===================================================================


def _build_sync_h5(tmp_path: "Path", include_optional: bool = True) -> "Path":
    """Write a minimal synthetic sync.h5 matching the Stage 5 write contract."""
    import numpy as np

    from hm2p.io.hdf5 import write_h5

    n_rois, n_frames = 4, 50
    datasets: dict = {
        "dff": np.zeros((n_rois, n_frames), dtype=np.float32),
        "hd_deg": np.zeros(n_frames, dtype=np.float32),
        "speed_cm_s": np.zeros(n_frames, dtype=np.float32),
        "light_on": np.ones(n_frames, dtype=bool),
        "active": np.ones(n_frames, dtype=bool),
        "bad_behav": np.zeros(n_frames, dtype=bool),
        "frame_times": np.linspace(0, 5, n_frames).astype(np.float64),
        "roi_types": np.zeros(n_rois, dtype=np.uint8),
    }
    if include_optional:
        datasets["deconv"] = np.zeros((n_rois, n_frames), dtype=np.float32)
        datasets["spikes"] = np.zeros((n_rois, n_frames), dtype=np.float32)
        datasets["event_masks"] = np.zeros((n_rois, n_frames), dtype=bool)
        datasets["event_masks_sd"] = np.zeros((n_rois, n_frames), dtype=bool)
        datasets["x_mm"] = np.zeros(n_frames, dtype=np.float32)
        datasets["y_mm"] = np.zeros(n_frames, dtype=np.float32)
        datasets["ahv_deg_s"] = np.zeros(n_frames, dtype=np.float32)

    out = tmp_path / "sync.h5"
    write_h5(out, datasets, attrs={"session_id": "test"})
    return out


def _read_sync_h5_like_frontend(path: "Path") -> dict:
    """Simulate _fetch_all_sync_data's HDF5 reading logic on a local file."""
    import io as _io

    import h5py
    import numpy as np

    with open(path, "rb") as fh:
        data_bytes = fh.read()

    result: dict = {}
    buf = _io.BytesIO(data_bytes)
    with h5py.File(buf, "r") as f:
        result["dff"] = f["dff"][:]
        result["hd_deg"] = f["hd_deg"][:]
        n = len(result["hd_deg"])
        result["speed_cm_s"] = f["speed_cm_s"][:] if "speed_cm_s" in f else np.zeros(n)
        result["light_on"] = f["light_on"][:] if "light_on" in f else np.ones(n, dtype=bool)
        result["active"] = f["active"][:] if "active" in f else np.ones(n, dtype=bool)
        result["bad_behav"] = f["bad_behav"][:] if "bad_behav" in f else np.zeros(n, dtype=bool)
        result["frame_times"] = f["frame_times"][:] if "frame_times" in f else np.arange(n, dtype=float)
        result["roi_types"] = f["roi_types"][:] if "roi_types" in f else np.zeros(result["dff"].shape[0], dtype=np.uint8)
        result["deconv"] = f["deconv"][:] if "deconv" in f else None
        result["spikes"] = f["spikes"][:] if "spikes" in f else None
        result["event_masks"] = f["event_masks"][:] if "event_masks" in f else None
        result["event_masks_sd"] = f["event_masks_sd"][:] if "event_masks_sd" in f else None
        result["x_mm"] = f["x_mm"][:] if "x_mm" in f else None
        result["y_mm"] = f["y_mm"][:] if "y_mm" in f else None
        result["ahv_deg_s"] = f["ahv_deg_s"][:] if "ahv_deg_s" in f else None
    return result


class TestSyncH5KeyContract:
    """T5 — sync.h5 structure matches the frontend read contract."""

    REQUIRED_KEYS = (
        "dff", "hd_deg", "speed_cm_s", "light_on", "active",
        "bad_behav", "frame_times", "roi_types",
    )
    OPTIONAL_KEYS = (
        "deconv", "spikes", "event_masks", "event_masks_sd",
        "x_mm", "y_mm", "ahv_deg_s",
    )

    def test_required_keys_all_present_and_non_none(self, tmp_path):
        """All required keys must be present and non-None after reading sync.h5."""
        path = _build_sync_h5(tmp_path, include_optional=False)
        session = _read_sync_h5_like_frontend(path)
        for key in self.REQUIRED_KEYS:
            assert key in session, f"Required key missing: {key}"
            assert session[key] is not None, f"Required key is None: {key}"

    def test_optional_keys_present_when_written(self, tmp_path):
        """Optional keys are non-None when explicitly written to sync.h5."""
        path = _build_sync_h5(tmp_path, include_optional=True)
        session = _read_sync_h5_like_frontend(path)
        for key in self.OPTIONAL_KEYS:
            assert key in session, f"Optional key missing from result: {key}"
            assert session[key] is not None, f"Optional key is None despite being written: {key}"

    def test_optional_keys_are_none_when_absent(self, tmp_path):
        """Optional keys default to None when not present in sync.h5."""
        path = _build_sync_h5(tmp_path, include_optional=False)
        session = _read_sync_h5_like_frontend(path)
        for key in self.OPTIONAL_KEYS:
            assert session[key] is None, f"Expected None for absent key: {key}"

    def test_dff_shape_is_rois_by_frames(self, tmp_path):
        """dff array is (n_rois, n_frames) — rows are ROIs, columns are frames."""
        path = _build_sync_h5(tmp_path)
        session = _read_sync_h5_like_frontend(path)
        assert session["dff"].ndim == 2
        n_rois, n_frames = session["dff"].shape
        assert n_rois == 4
        assert n_frames == 50

    def test_behavioural_arrays_have_same_length(self, tmp_path):
        """hd_deg, speed, light_on, active, bad_behav all have the same length."""
        path = _build_sync_h5(tmp_path)
        session = _read_sync_h5_like_frontend(path)
        n = len(session["hd_deg"])
        for key in ("speed_cm_s", "light_on", "active", "bad_behav", "frame_times"):
            assert len(session[key]) == n, (
                f"Length mismatch for {key}: {len(session[key])} != {n}"
            )

    def test_roi_types_length_matches_dff_rows(self, tmp_path):
        """roi_types has one entry per ROI (matches dff.shape[0])."""
        path = _build_sync_h5(tmp_path)
        session = _read_sync_h5_like_frontend(path)
        assert len(session["roi_types"]) == session["dff"].shape[0]

    def test_light_on_is_boolean(self, tmp_path):
        """light_on values are boolean."""
        path = _build_sync_h5(tmp_path)
        session = _read_sync_h5_like_frontend(path)
        assert session["light_on"].dtype == bool

    def test_bad_behav_is_boolean(self, tmp_path):
        """bad_behav values are boolean."""
        path = _build_sync_h5(tmp_path)
        session = _read_sync_h5_like_frontend(path)
        assert session["bad_behav"].dtype == bool


# ===================================================================
# T6: DOWNSTREAM_DEPS cascade is transitively complete
# ===================================================================


class TestDownstreamDeps:
    """T6 — DOWNSTREAM_DEPS dict captures the full invalidation cascade."""

    def test_dlc_training_invalidates_pose_through_analysis(self):
        """DLC training re-run invalidates pose, kinematics, kpms, sync, analysis."""
        deps = DOWNSTREAM_DEPS["dlc_training"]
        for stage in ("pose", "kinematics", "kpms", "sync", "analysis"):
            assert stage in deps, f"Expected '{stage}' in dlc_training downstream"

    def test_dlc_training_does_not_invalidate_calcium(self):
        """DLC training re-run must NOT invalidate calcium (Stage 4 is independent)."""
        deps = DOWNSTREAM_DEPS["dlc_training"]
        assert "calcium" not in deps
        assert "ca_extraction" not in deps

    def test_pose_invalidates_kinematics_through_analysis(self):
        """Pose re-run invalidates kinematics, kpms, sync, analysis."""
        deps = DOWNSTREAM_DEPS["pose"]
        for stage in ("kinematics", "kpms", "sync", "analysis"):
            assert stage in deps, f"Expected '{stage}' in pose downstream"

    def test_pose_does_not_invalidate_calcium_or_ca_extraction(self):
        """Pose re-run must NOT invalidate calcium or ca_extraction."""
        deps = DOWNSTREAM_DEPS["pose"]
        assert "calcium" not in deps
        assert "ca_extraction" not in deps

    def test_kinematics_invalidates_sync_and_analysis(self):
        """Kinematics re-run invalidates sync and analysis."""
        deps = DOWNSTREAM_DEPS["kinematics"]
        assert "sync" in deps
        assert "analysis" in deps

    def test_kinematics_does_not_invalidate_calcium_or_pose(self):
        """Kinematics re-run does not affect calcium or pose stages."""
        deps = DOWNSTREAM_DEPS["kinematics"]
        assert "calcium" not in deps
        assert "pose" not in deps

    def test_calcium_invalidates_sync_and_analysis(self):
        """Calcium re-run invalidates sync and analysis (both use ca.h5)."""
        deps = DOWNSTREAM_DEPS["calcium"]
        assert "sync" in deps
        assert "analysis" in deps

    def test_calcium_is_independent_of_pose_stages(self):
        """Calcium processing is independent of pose stages."""
        deps = DOWNSTREAM_DEPS["calcium"]
        assert "pose" not in deps
        assert "kinematics" not in deps
        assert "dlc_training" not in deps

    def test_cascade_does_not_invalidate_downstream(self):
        """CASCADE (Stage 4b) adds spikes to ca.h5 without invalidating downstream."""
        deps = DOWNSTREAM_DEPS["cascade"]
        assert deps == [], f"Expected cascade downstream=[], got {deps}"

    def test_analysis_has_no_downstream(self):
        """Analysis is the terminal stage — nothing downstream of it."""
        deps = DOWNSTREAM_DEPS["analysis"]
        assert deps == [], f"Expected analysis downstream=[], got {deps}"

    def test_sync_invalidates_only_analysis(self):
        """Sync re-run invalidates only analysis."""
        deps = DOWNSTREAM_DEPS["sync"]
        assert "analysis" in deps
        assert "kinematics" not in deps
        assert "calcium" not in deps

    def test_all_expected_keys_present(self):
        """DOWNSTREAM_DEPS has an entry for every pipeline stage."""
        expected_keys = {
            "dlc_training", "pose", "ca_extraction", "kinematics",
            "kpms", "calcium", "cascade", "sync", "analysis",
        }
        assert set(DOWNSTREAM_DEPS.keys()) == expected_keys
