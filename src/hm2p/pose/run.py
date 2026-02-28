"""Stage 2 — dispatch pose estimation to tracker backend.

The tracker is selected from session.tracker (set in experiments.csv).
All trackers produce a native pose file in derivatives/pose/<sub>/<ses>/.
Stage 3 then loads the result via movement regardless of tracker used.

Current backends:
    dlc   — DeepLabCut 3.x (default)
    sleap — SLEAP 1.4+
    lp    — LightningPose
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from hm2p.session import Session


def run_tracker(
    session: Session,
    video_path: Path,
    model_dir: Path,
    output_dir: Path,
    tracker: Literal["dlc", "sleap", "lp"] | None = None,
) -> Path:
    """Run pose estimation for one session using the session's configured tracker.

    Args:
        session: Session metadata (tracker field used if tracker arg is None).
        video_path: Path to the pre-processed overhead .mp4.
        model_dir: Directory containing the tracker model / config.
        output_dir: Where to write the tracker-native output file.
        tracker: Override session.tracker if provided.

    Returns:
        Path to the output pose file (DLC .h5, SLEAP .h5, LP .csv).
    """
    _tracker = tracker or session.tracker
    if _tracker == "dlc":
        return _run_dlc(video_path, model_dir, output_dir)
    elif _tracker == "sleap":
        return _run_sleap(video_path, model_dir, output_dir)
    elif _tracker == "lp":
        return _run_lp(video_path, model_dir, output_dir)
    else:
        raise ValueError(f"Unknown tracker: {_tracker!r}")


def _run_dlc(video_path: Path, model_dir: Path, output_dir: Path) -> Path:
    raise NotImplementedError


def _run_sleap(video_path: Path, model_dir: Path, output_dir: Path) -> Path:
    raise NotImplementedError


def _run_lp(video_path: Path, model_dir: Path, output_dir: Path) -> Path:
    raise NotImplementedError
