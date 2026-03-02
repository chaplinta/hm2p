"""Tests for pose/run.py — tracker dispatch and DLC/SLEAP/LP runners."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hm2p.pose.run import run_tracker
from hm2p.session import Session


def _session(tracker: str = "dlc") -> Session:
    return Session(
        session_id="20220804_13_52_02_1117646",
        animal_id="1117646",
        celltype="penk",
        gcamp="GCaMP7f",
        virus_id="ADD3",
        tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Dispatch logic
# ---------------------------------------------------------------------------


class TestRunTrackerDispatch:
    def test_unknown_tracker_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown tracker"):
            run_tracker(
                session=_session("dlc"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
                tracker="nonexistent",
            )

    def test_tracker_override(self, tmp_path):
        """tracker param overrides session.tracker — dispatches to sleap."""
        with (
            patch("hm2p.pose.run._run_sleap", side_effect=ImportError("sleap")),
            pytest.raises(ImportError, match="sleap"),
        ):
            run_tracker(
                session=_session("dlc"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
                tracker="sleap",
            )

    def test_session_tracker_used_when_no_override(self, tmp_path):
        """When tracker=None, session.tracker is used."""
        with (
            patch("hm2p.pose.run._run_lp", side_effect=ImportError("lp")),
            pytest.raises(ImportError, match="lp"),
        ):
            run_tracker(
                session=_session("lp"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )


# ---------------------------------------------------------------------------
# DLC runner
# ---------------------------------------------------------------------------


class TestRunDlc:
    def test_missing_video_raises(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.yaml").write_text("dummy")

        with pytest.raises(FileNotFoundError, match="Video file"):
            run_tracker(
                session=_session("dlc"),
                video_path=tmp_path / "missing.mp4",
                model_dir=model_dir,
                output_dir=tmp_path / "output",
            )

    def test_missing_config_raises(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # No config.yaml

        with pytest.raises(FileNotFoundError, match="config.yaml"):
            run_tracker(
                session=_session("dlc"),
                video_path=video,
                model_dir=model_dir,
                output_dir=tmp_path / "output",
            )

    def test_importerror_without_dlc(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.yaml").write_text("dummy")

        with (
            patch.dict("sys.modules", {"deeplabcut": None}),
            pytest.raises(ImportError, match="deeplabcut"),
        ):
            run_tracker(
                session=_session("dlc"),
                video_path=video,
                model_dir=model_dir,
                output_dir=tmp_path / "output",
            )

    def test_successful_run_with_mock(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.yaml").write_text("dummy")
        output_dir = tmp_path / "output"

        mock_dlc = MagicMock()

        def fake_analyze(config, videos, destfolder, save_as_csv):
            """Simulate DLC writing an output .h5 file."""
            out = Path(destfolder)
            out.mkdir(parents=True, exist_ok=True)
            (out / "videoDLC_resnet50_modelFeb1shuffle1_100000.h5").write_bytes(b"\x00")

        mock_dlc.analyze_videos = fake_analyze

        with patch.dict("sys.modules", {"deeplabcut": mock_dlc}):
            result = run_tracker(
                session=_session("dlc"),
                video_path=video,
                model_dir=model_dir,
                output_dir=output_dir,
            )

        assert result.exists()
        assert "DLC" in result.name

    def test_no_output_h5_raises(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.yaml").write_text("dummy")
        output_dir = tmp_path / "output"

        mock_dlc = MagicMock()
        mock_dlc.analyze_videos = MagicMock()  # doesn't create output

        with (
            patch.dict("sys.modules", {"deeplabcut": mock_dlc}),
            pytest.raises(RuntimeError, match="no output"),
        ):
            run_tracker(
                session=_session("dlc"),
                video_path=video,
                model_dir=model_dir,
                output_dir=output_dir,
            )


# ---------------------------------------------------------------------------
# SLEAP runner
# ---------------------------------------------------------------------------


class TestRunSleap:
    def test_importerror_without_sleap(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        with patch.dict("sys.modules", {"sleap": None}), pytest.raises(ImportError, match="sleap"):
            run_tracker(
                session=_session("sleap"),
                video_path=video,
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )

    def test_missing_video_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Video file"):
            run_tracker(
                session=_session("sleap"),
                video_path=tmp_path / "missing.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )


# ---------------------------------------------------------------------------
# LightningPose runner
# ---------------------------------------------------------------------------


class TestRunLp:
    def test_importerror_without_lp(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")

        with patch.dict("sys.modules", {
            "lightning_pose": None,
            "lightning_pose.utils": None,
            "lightning_pose.utils.predictions": None,
        }), pytest.raises(ImportError, match="lightning_pose"):
            run_tracker(
                session=_session("lp"),
                video_path=video,
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )

    def test_missing_video_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Video file"):
            run_tracker(
                session=_session("lp"),
                video_path=tmp_path / "missing.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )
