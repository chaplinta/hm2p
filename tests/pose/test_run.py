"""Tests for pose/run.py — tracker dispatch."""

from __future__ import annotations

from pathlib import Path

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


class TestRunTracker:
    def test_dlc_dispatch_raises_not_implemented(self, tmp_path: Path) -> None:
        """DLC backend is stubbed — should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            run_tracker(
                session=_session("dlc"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )

    def test_sleap_dispatch_raises_not_implemented(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            run_tracker(
                session=_session("sleap"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )

    def test_lp_dispatch_raises_not_implemented(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            run_tracker(
                session=_session("lp"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )

    def test_unknown_tracker_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown tracker"):
            run_tracker(
                session=_session("dlc"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
                tracker="nonexistent",
            )

    def test_tracker_override(self, tmp_path: Path) -> None:
        """tracker param overrides session.tracker."""
        with pytest.raises(NotImplementedError):
            run_tracker(
                session=_session("dlc"),
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
                tracker="sleap",
            )

    def test_session_tracker_used_when_no_override(self, tmp_path: Path) -> None:
        """When tracker=None, session.tracker is used."""
        ses = _session("lp")
        with pytest.raises(NotImplementedError):
            run_tracker(
                session=ses,
                video_path=tmp_path / "video.mp4",
                model_dir=tmp_path / "model",
                output_dir=tmp_path / "output",
            )
