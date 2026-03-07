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
sys.modules.setdefault("streamlit", _st_mock)

# Now it is safe to import the module under test.
from frontend.data import (  # noqa: E402
    DERIVATIVES_BUCKET,
    RAWDATA_BUCKET,
    REGION,
    STAGE_PREFIXES,
    download_s3_bytes,
    download_s3_numpy,
    get_ec2_instances,
    get_pipeline_status,
    get_progress,
    get_s3_client,
    list_s3_session_files,
    load_animals,
    load_experiments,
    parse_session_id,
)


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

    def test_has_five_stages(self):
        assert len(STAGE_PREFIXES) == 5

    def test_expected_keys(self):
        assert set(STAGE_PREFIXES.keys()) == {
            "ca_extraction",
            "pose",
            "kinematics",
            "calcium",
            "sync",
        }

    def test_ca_extraction_label(self):
        assert STAGE_PREFIXES["ca_extraction"] == "Stage 1 — Suite2p"

    def test_pose_label(self):
        assert STAGE_PREFIXES["pose"] == "Stage 2 — DLC"

    def test_kinematics_label(self):
        assert STAGE_PREFIXES["kinematics"] == "Stage 3 — Kinematics"

    def test_calcium_label(self):
        assert STAGE_PREFIXES["calcium"] == "Stage 4 — Calcium"

    def test_sync_label(self):
        assert STAGE_PREFIXES["sync"] == "Stage 5 — Sync"

    def test_labels_contain_stage_number(self):
        for i, key in enumerate(STAGE_PREFIXES, start=1):
            assert f"Stage {i}" in STAGE_PREFIXES[key]


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
