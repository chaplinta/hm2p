"""Tests for data-loading helpers in ``frontend/pages/dlc_training_page.py``.

All S3 calls are mocked. Streamlit is mocked to prevent page-level rendering
side effects. Tests cover:

- ``_load_champion_info`` — valid JSON / missing key
- ``_load_eval_results`` — valid JSON / missing key
- ``_load_per_bodypart_eval`` — valid JSON with per_frame / missing key
- ``_load_retrain_progress`` — valid JSON / missing key
- ``_load_gpu_monitor`` — valid CSV parsing / empty CSV / malformed rows
- ``_check_model_exists`` — model found / not found
- ``_parse_training_curves`` — learning_stats.csv / run log / empty data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Mock streamlit before importing any frontend code ──────────────────
_st_mock = MagicMock()
_st_mock.cache_data = lambda *a, **kw: a[0] if a and callable(a[0]) else (lambda fn: fn)
_st_mock.title = MagicMock()
_st_mock.header = MagicMock()
_st_mock.caption = MagicMock()
_st_mock.markdown = MagicMock()
_st_mock.info = MagicMock()
_st_mock.success = MagicMock()
_st_mock.warning = MagicMock()
_st_mock.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()])
_st_mock.expander = MagicMock()
_st_mock.metric = MagicMock()
_st_mock.plotly_chart = MagicMock()
_st_mock.json = MagicMock()
_st_mock.number_input = MagicMock(return_value=20)
_st_mock.dataframe = MagicMock()
# Save originals so we can restore them after loading the page module.
_orig_st_dtrain = sys.modules.get("streamlit")
_orig_frontend_dtrain = sys.modules.get("frontend")
_orig_fdata_dtrain = sys.modules.get("frontend.data")

sys.modules["streamlit"] = _st_mock


# ── Mock frontend.data before importing the page module ────────────────
_data_mock = MagicMock()
_data_mock.DERIVATIVES_BUCKET = "hm2p-derivatives"
_data_mock.download_s3_bytes = MagicMock(return_value=None)
_data_mock.get_ec2_instances = MagicMock(return_value=[])
_data_mock.get_s3_client = MagicMock()
_data_mock.sanitize_error = MagicMock(side_effect=lambda x: x)
sys.modules["frontend"] = MagicMock()
sys.modules["frontend.data"] = _data_mock

# Now import the individual helper functions.
# We do NOT import the page module directly (it has top-level st calls).
# Instead, we import the functions by loading the module carefully.

import importlib
import types

_page_path = (
    Path(__file__).resolve().parent.parent.parent / "frontend" / "pages" / "dlc_training_page.py"
)


def _load_page_functions() -> types.ModuleType:
    """Load dlc_training_page.py as a module, extracting its functions."""
    spec = importlib.util.spec_from_file_location("dlc_training_page", str(_page_path))
    mod = importlib.util.module_from_spec(spec)
    # Patch globals so the top-level page rendering doesn't crash
    mod.st = _st_mock
    spec.loader.exec_module(mod)
    return mod


# Load the module once and grab references to functions.
_page_mod = _load_page_functions()

# Restore sys.modules so other test files aren't contaminated.
if _orig_st_dtrain is not None:
    sys.modules["streamlit"] = _orig_st_dtrain
else:
    sys.modules.pop("streamlit", None)

if _orig_frontend_dtrain is not None:
    sys.modules["frontend"] = _orig_frontend_dtrain
else:
    sys.modules.pop("frontend", None)

if _orig_fdata_dtrain is not None:
    sys.modules["frontend.data"] = _orig_fdata_dtrain
else:
    sys.modules.pop("frontend.data", None)


# ═══════════════════════════════════════════════════════════════════════
# _load_champion_info
# ═══════════════════════════════════════════════════════════════════════


class TestLoadChampionInfo:
    def test_valid_json(self):
        """Returns parsed dict when S3 returns valid champion JSON."""
        champion_data = {
            "champion_id": "champ_abc",
            "architecture": "hrnet_w32",
            "snapshot": "best-200",
            "training_date": "2024-03-01",
        }
        _data_mock.download_s3_bytes.return_value = json.dumps(champion_data).encode()
        result = _page_mod._load_champion_info()
        assert result is not None
        assert result["champion_id"] == "champ_abc"
        assert result["architecture"] == "hrnet_w32"

    def test_missing_key_returns_none(self):
        """Returns None when the S3 key does not exist."""
        _data_mock.download_s3_bytes.return_value = None
        result = _page_mod._load_champion_info()
        assert result is None

    def test_invalid_json_returns_none(self):
        """Returns None when S3 returns non-JSON data."""
        _data_mock.download_s3_bytes.return_value = b"not valid json {{"
        result = _page_mod._load_champion_info()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# _load_eval_results
# ═══════════════════════════════════════════════════════════════════════


class TestLoadEvalResults:
    def test_valid_json(self):
        """Returns parsed dict with train/test metrics."""
        eval_data = {
            "champion_id": "champ_123",
            "train": {"rmse": 3.5, "mAP": 85.0, "mAR": 80.0},
            "test": {"rmse": 5.2, "mAP": 72.0, "mAR": 68.0},
            "training_fraction": 0.8,
            "best_epoch": 100,
            "total_epochs": 400,
            "n_labeled_frames": 184,
        }
        _data_mock.download_s3_bytes.return_value = json.dumps(eval_data).encode()
        result = _page_mod._load_eval_results()
        assert result is not None
        assert result["train"]["rmse"] == pytest.approx(3.5)
        assert result["test"]["mAP"] == pytest.approx(72.0)
        assert result["n_labeled_frames"] == 184

    def test_missing_key_returns_none(self):
        """Returns None when eval results are not on S3."""
        _data_mock.download_s3_bytes.return_value = None
        result = _page_mod._load_eval_results()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# _load_per_bodypart_eval
# ═══════════════════════════════════════════════════════════════════════


class TestLoadPerBodypartEval:
    def test_valid_json_with_per_frame(self):
        """Returns dict with bodyparts and per_frame arrays."""
        bp_data = {
            "bodyparts": {
                "nose_tip": {"rmse": 4.2, "n": 100, "pck_10": 95.0},
                "left_ear": {"rmse": 3.8, "n": 100, "pck_10": 97.0},
            },
            "n_total_matched": 200,
            "per_frame": [
                {
                    "frame_id": "frame_001.png",
                    "split": "train",
                    "errors": {"nose_tip": 3.5, "left_ear": 2.1},
                    "gt": {"nose_tip": [100.0, 200.0], "left_ear": [150.0, 250.0]},
                    "pred": {"nose_tip": [103.0, 201.5], "left_ear": [151.0, 252.0]},
                }
            ],
        }
        _data_mock.download_s3_bytes.return_value = json.dumps(bp_data).encode()
        result = _page_mod._load_per_bodypart_eval()
        assert result is not None
        assert "bodyparts" in result
        assert "per_frame" in result
        assert result["bodyparts"]["nose_tip"]["rmse"] == pytest.approx(4.2)
        assert len(result["per_frame"]) == 1
        assert result["per_frame"][0]["split"] == "train"

    def test_missing_key_returns_none(self):
        """Returns None when per-bodypart eval is not on S3."""
        _data_mock.download_s3_bytes.return_value = None
        result = _page_mod._load_per_bodypart_eval()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# _load_retrain_progress
# ═══════════════════════════════════════════════════════════════════════


class TestLoadRetrainProgress:
    def test_valid_progress(self):
        """Returns progress dict with status and timestamp."""
        progress = {
            "status": "Training (SA): 120 epochs",
            "updated": "2024-03-01T12:00:00Z",
            "completed": 5,
            "total": 26,
        }
        _data_mock.download_s3_bytes.return_value = json.dumps(progress).encode()
        result = _page_mod._load_retrain_progress()
        assert result is not None
        assert result["status"] == "Training (SA): 120 epochs"
        assert result["total"] == 26

    def test_missing_key_returns_none(self):
        _data_mock.download_s3_bytes.return_value = None
        result = _page_mod._load_retrain_progress()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# _load_gpu_monitor
# ═══════════════════════════════════════════════════════════════════════


class TestLoadGpuMonitor:
    def test_valid_csv_parsing(self):
        """Parses GPU monitor CSV into list of dicts with correct fields."""
        csv_content = (
            "timestamp, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
            "2024/03/01 12:00:00.000, 85 %, 4096 MiB, 16384 MiB\n"
            "2024/03/01 12:01:00.000, 92 %, 4200 MiB, 16384 MiB\n"
        )
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._load_gpu_monitor()
        assert result is not None
        assert len(result) == 2
        assert result[0]["gpu_util_pct"] == 85
        assert result[0]["mem_used_mb"] == 4096
        assert result[0]["mem_total_mb"] == 16384
        assert result[1]["gpu_util_pct"] == 92

    def test_empty_csv_returns_none(self):
        """CSV with only a header line returns None."""
        csv_content = "timestamp, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._load_gpu_monitor()
        assert result is None

    def test_no_data_returns_none(self):
        """Returns None when S3 key is missing."""
        _data_mock.download_s3_bytes.return_value = None
        result = _page_mod._load_gpu_monitor()
        assert result is None

    def test_malformed_rows_skipped(self):
        """Rows with non-numeric GPU values are silently skipped."""
        csv_content = (
            "timestamp, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
            "2024/03/01 12:00:00.000, 85 %, 4096 MiB, 16384 MiB\n"
            "2024/03/01 12:01:00.000, N/A, N/A, N/A\n"
            "2024/03/01 12:02:00.000, 90 %, 4100 MiB, 16384 MiB\n"
        )
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._load_gpu_monitor()
        assert result is not None
        assert len(result) == 2  # middle row skipped
        assert result[0]["gpu_util_pct"] == 85
        assert result[1]["gpu_util_pct"] == 90

    def test_no_percent_suffix_handled(self):
        """Values without ' %' suffix are parsed correctly."""
        csv_content = (
            "timestamp, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
            "2024/03/01 12:00:00.000, 75, 3000, 16384\n"
        )
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._load_gpu_monitor()
        assert result is not None
        assert result[0]["gpu_util_pct"] == 75
        assert result[0]["mem_used_mb"] == 3000

    def test_timestamp_preserved(self):
        """Timestamp string is preserved as-is in the output."""
        csv_content = (
            "timestamp, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
            "2024/03/01 14:30:00.500, 50 %, 2000 MiB, 16384 MiB\n"
        )
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._load_gpu_monitor()
        assert result[0]["timestamp"] == "2024/03/01 14:30:00.500"


# ═══════════════════════════════════════════════════════════════════════
# _check_model_exists
# ═══════════════════════════════════════════════════════════════════════


class TestCheckModelExists:
    def test_model_found(self):
        """Returns True when S3 lists a .pt file under the model prefix."""
        s3_mock = MagicMock()
        s3_mock.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "dlc-retrain/models/snapshot-best-200.pt"},
            ]
        }
        _data_mock.get_s3_client.return_value = s3_mock

        result = _page_mod._check_model_exists()
        assert result is True

    def test_no_model_files(self):
        """Returns False when no model weight files are found.

        Only non-model files are present (e.g. .yaml, .csv, .txt).
        Note: .json IS treated as a model suffix by the function.
        """
        s3_mock = MagicMock()
        s3_mock.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "dlc-retrain/models/config.yaml"},
                {"Key": "dlc-retrain/models/learning_stats.csv"},
                {"Key": "dlc-retrain/models/notes.txt"},
            ]
        }
        _data_mock.get_s3_client.return_value = s3_mock

        result = _page_mod._check_model_exists()
        assert result is False

    def test_empty_bucket(self):
        """Returns False when the model prefix has no objects."""
        s3_mock = MagicMock()
        s3_mock.list_objects_v2.return_value = {}
        _data_mock.get_s3_client.return_value = s3_mock

        result = _page_mod._check_model_exists()
        assert result is False

    def test_s3_error_returns_false(self):
        """Returns False when S3 call raises an exception."""
        _data_mock.get_s3_client.side_effect = Exception("ConnectionError")

        result = _page_mod._check_model_exists()
        assert result is False

        # Restore for subsequent tests
        _data_mock.get_s3_client.side_effect = None

    def test_pth_extension_detected(self):
        """Returns True for .pth model files."""
        s3_mock = MagicMock()
        s3_mock.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "dlc-retrain/models/model_weights.pth"},
            ]
        }
        _data_mock.get_s3_client.return_value = s3_mock

        result = _page_mod._check_model_exists()
        assert result is True

    def test_pkl_extension_detected(self):
        """Returns True for .pkl model files."""
        s3_mock = MagicMock()
        s3_mock.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "dlc-retrain/models/pose_cfg.pkl"},
            ]
        }
        _data_mock.get_s3_client.return_value = s3_mock

        result = _page_mod._check_model_exists()
        assert result is True


# ═══════════════════════════════════════════════════════════════════════
# _parse_training_curves
# ═══════════════════════════════════════════════════════════════════════


class TestParseTrainingCurves:
    def test_learning_stats_csv(self):
        """Parses learning_stats.csv into list of dicts with epoch/loss/rmse."""
        csv_content = (
            "step,losses/train.total_loss,losses/eval.total_loss,"
            "metrics/test.rmse,metrics/test.rmse_pcutoff,metrics/test.mAP\n"
            "10,0.0050,0.0080,15.3,12.1,45.0\n"
            "20,0.0035,0.0060,10.2,8.5,65.0\n"
            "30,0.0025,0.0045,7.8,6.2,78.0\n"
        )
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._parse_training_curves()

        assert result is not None
        assert len(result) == 3

        # First row
        assert result[0]["epoch"] == 10
        assert result[0]["train_loss"] == pytest.approx(0.005)
        assert result[0]["rmse_px"] == pytest.approx(15.3)
        assert result[0]["mAP"] == pytest.approx(45.0)

        # Last row
        assert result[2]["epoch"] == 30
        assert result[2]["total_epochs"] == 30
        assert result[2]["rmse_pcutoff_px"] == pytest.approx(6.2)

    def test_run_log_fallback(self):
        """Falls back to parsing _run_log.txt when learning_stats.csv is missing."""
        # learning_stats.csv returns None for all shuffle candidates
        call_count = [0]

        def mock_download(bucket, key):
            call_count[0] += 1
            if "learning_stats.csv" in key:
                return None
            if "_run_log.txt" in key:
                return (
                    b"Epoch 1/100 (lr=0.001), train loss 0.01500\n"
                    b"Epoch 2/100 (lr=0.001), train loss 0.01200, valid loss 0.01400\n"
                    b"Epoch 3/100 (lr=0.0005), train loss 0.01000\n"
                )
            return None

        _data_mock.download_s3_bytes.side_effect = mock_download
        result = _page_mod._parse_training_curves()
        _data_mock.download_s3_bytes.side_effect = None  # Reset

        assert result is not None
        assert len(result) == 3

        assert result[0]["epoch"] == 1
        assert result[0]["total_epochs"] == 100
        assert result[0]["train_loss"] == pytest.approx(0.015)
        assert result[0]["valid_loss"] is None
        assert result[0]["rmse_px"] is None  # Not available from log

        assert result[1]["valid_loss"] == pytest.approx(0.014)
        assert result[2]["lr"] == pytest.approx(0.0005)

    def test_empty_data_returns_none(self):
        """Returns None when neither learning_stats.csv nor run log are available."""
        _data_mock.download_s3_bytes.return_value = None
        result = _page_mod._parse_training_curves()
        assert result is None

    def test_learning_stats_with_nan_values(self):
        """Handles NaN values in learning_stats.csv gracefully."""
        csv_content = (
            "step,losses/train.total_loss,losses/eval.total_loss,"
            "metrics/test.rmse,metrics/test.rmse_pcutoff,metrics/test.mAP\n"
            "10,0.0050,,,,\n"
            "20,0.0035,0.0060,10.2,8.5,65.0\n"
        )
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._parse_training_curves()

        assert result is not None
        assert len(result) == 2
        # First row has no eval/rmse/mAP
        assert result[0]["valid_loss"] is None
        assert result[0]["rmse_px"] is None
        assert result[0]["mAP"] is None
        # Second row has all values
        assert result[1]["rmse_px"] == pytest.approx(10.2)

    def test_total_epochs_is_max_step(self):
        """total_epochs for each row is set to the max step in the data."""
        csv_content = (
            "step,losses/train.total_loss,losses/eval.total_loss,"
            "metrics/test.rmse,metrics/test.rmse_pcutoff,metrics/test.mAP\n"
            "100,0.001,0.002,5.0,4.0,90.0\n"
            "200,0.0008,0.0015,4.0,3.5,92.0\n"
            "400,0.0005,0.001,3.0,2.5,95.0\n"
        )
        _data_mock.download_s3_bytes.return_value = csv_content.encode()
        result = _page_mod._parse_training_curves()

        assert result is not None
        # All rows should have total_epochs = 400 (max step)
        for row in result:
            assert row["total_epochs"] == 400
