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

import logging
from pathlib import Path
from typing import Literal

from hm2p.session import Session

log = logging.getLogger(__name__)


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
    """Run DeepLabCut inference on a single video.

    Uses ``deeplabcut.analyze_videos()`` with a pre-trained model. The model
    directory should contain the DLC project ``config.yaml`` and trained
    snapshot files.

    Args:
        video_path: Path to the overhead .mp4 video.
        model_dir: Path to the DLC project directory (contains config.yaml).
        output_dir: Where to write the DLC output .h5 file.

    Returns:
        Path to the output .h5 pose file.

    Raises:
        ImportError: If deeplabcut is not installed.
        FileNotFoundError: If video or model config is missing.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"DLC config.yaml not found in {model_dir}")

    try:
        import deeplabcut
    except ImportError as exc:
        raise ImportError(
            "deeplabcut is not installed. "
            "Install via: pip install 'deeplabcut[tf]>=3.0'\n"
            "See: https://deeplabcut.github.io/DeepLabCut/"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running DeepLabCut on %s", video_path.name)
    deeplabcut.analyze_videos(
        str(config_path),
        [str(video_path)],
        destfolder=str(output_dir),
        save_as_csv=False,
    )

    # DLC writes: <video_stem>DLC_<model>_<shuffle>.h5
    h5_files = sorted(output_dir.glob("*DLC*.h5"))
    if not h5_files:
        raise RuntimeError(
            f"DeepLabCut completed but no output .h5 found in {output_dir}. "
            "Check DLC logs for errors."
        )

    result_path = h5_files[0]
    log.info("DLC complete. Output: %s", result_path)
    return result_path


def _run_sleap(video_path: Path, model_dir: Path, output_dir: Path) -> Path:
    """Run SLEAP inference on a single video.

    Args:
        video_path: Path to the overhead .mp4 video.
        model_dir: Path to the SLEAP model directory.
        output_dir: Where to write the SLEAP output .h5 file.

    Returns:
        Path to the output .h5 pose file.

    Raises:
        ImportError: If sleap is not installed.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        import sleap
    except ImportError as exc:
        raise ImportError(
            "sleap is not installed. "
            "Install via: pip install sleap\n"
            "See: https://sleap.ai/"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running SLEAP on %s", video_path.name)

    # Find model files
    model_paths = sorted(model_dir.glob("*.json")) + sorted(model_dir.glob("*model*"))
    if not model_paths:
        raise FileNotFoundError(f"No SLEAP model files found in {model_dir}")

    # Run SLEAP inference
    labels = sleap.load_model(str(model_paths[0]))
    video = sleap.load_video(str(video_path))
    predictions = labels.predict(video)

    # Export to .h5
    output_path = output_dir / f"{video_path.stem}_sleap.h5"
    predictions.export(str(output_path))

    log.info("SLEAP complete. Output: %s", output_path)
    return output_path


def _run_lp(video_path: Path, model_dir: Path, output_dir: Path) -> Path:
    """Run LightningPose inference on a single video.

    Args:
        video_path: Path to the overhead .mp4 video.
        model_dir: Path to the LP model checkpoint directory.
        output_dir: Where to write the LP output .csv file.

    Returns:
        Path to the output .csv pose file.

    Raises:
        ImportError: If lightning_pose is not installed.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        from lightning_pose.utils.predictions import predict_single_video
    except ImportError as exc:
        raise ImportError(
            "lightning_pose is not installed. "
            "See: https://github.com/danbider/lightning-pose"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoint
    ckpt_files = sorted(model_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {model_dir}")

    output_path = output_dir / f"{video_path.stem}_lp.csv"

    log.info("Running LightningPose on %s", video_path.name)
    predict_single_video(
        video_file=str(video_path),
        ckpt_file=str(ckpt_files[0]),
        save_file=str(output_path),
    )

    log.info("LightningPose complete. Output: %s", output_path)
    return output_path
