#!/usr/bin/env python3
"""Select outlier frames for DLC retraining using DLC's own extraction.

Wraps ``deeplabcut.extract_outlier_frames`` which identifies frames with
large tracking jumps and low confidence, with built-in diversity
selection. Two passes per session: 'jump' then 'uncertain'.

Safety: existing CollectedData_*.csv/.h5 files are never modified.
Extracted frames appear as PNGs in the DLC labeled-data directory.
The user then labels them with ``deeplabcut.refine_labels`` or
``scripts/interactive_label.py``.

Usage:
    # Scan sessions to see labeling status:
    uv run python scripts/select_hard_frames.py --scan

    # Extract outlier frames for all sessions (DLC defaults):
    uv run python scripts/select_hard_frames.py

    # Limit to 8 frames per session:
    uv run python scripts/select_hard_frames.py --per-session 8

    # Limit to 200 frames total:
    uv run python scripts/select_hard_frames.py --total 200

    # One session only:
    uv run python scripts/select_hard_frames.py --session 20220804_11_21

    # Adjust thresholds:
    uv run python scripts/select_hard_frames.py --jump-threshold 15 --p-bound 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"
LABELED_DIR = (
    REPO_ROOT
    / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
)
DLC_CONFIG = (
    REPO_ROOT
    / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/config.yaml"
)

sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def get_sessions() -> list[dict]:
    """Load session list from experiments.csv."""
    sessions = []
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sessions.append({
                "exp_id": eid,
                "sub": f"sub-{parts[-1]}",
                "ses": f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}",
                "primary": row.get("primary_exp", "").lower() != "false",
                "exclude": row.get("exclude", "").lower() == "true",
            })
    return sessions


def _count_labeled(session_dir: Path) -> tuple[int, int]:
    """Return (n_pngs, n_labeled) for a labeled-data session dir."""
    import pandas as pd

    pngs = len(list(session_dir.glob("frame_*.png")))
    n_labeled = 0
    for h5 in session_dir.glob("CollectedData_*.h5"):
        try:
            df = pd.read_hdf(h5)
            n_labeled = int((~df.isna().all(axis=1)).sum())
        except Exception:
            pass
        break
    return pngs, n_labeled


def find_labeled_data_dir(sub: str, ses: str) -> Path | None:
    """Find the labeled-data directory for a session."""
    if not LABELED_DIR.exists():
        return None
    ses_date = ses.replace("ses-", "").split("T")[0]
    animal = sub.replace("sub-", "")
    for ld in LABELED_DIR.iterdir():
        if ld.is_dir() and ses_date in ld.name and animal in ld.name:
            return ld
    return None


def find_video_local(sub: str, ses: str) -> Path | None:
    """Find overhead video on local disk (rawdata or retrain_frames)."""
    # Check rawdata
    rawdata = REPO_ROOT / "rawdata" / sub / ses / "behav"
    if rawdata.exists():
        for mp4 in rawdata.glob("*.mp4"):
            if "side" not in mp4.name.lower():
                return mp4
    return None


def download_video_from_s3(s3: Any, sub: str, ses: str, dest: Path) -> Path | None:
    """Download overhead .mp4 from S3."""
    prefix = f"rawdata/{sub}/{ses}/behav/"
    resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = key.split("/")[-1]
        if not fname.endswith(".mp4") or "side" in fname.lower():
            continue
        local = dest / fname
        if not local.exists():
            s3.download_file(RAWDATA_BUCKET, key, str(local))
        return local
    return None


# ---------------------------------------------------------------------------
# Scan mode
# ---------------------------------------------------------------------------


def scan_sessions() -> None:
    """Print labeling status for all sessions."""
    import pandas as pd

    sessions = get_sessions()

    print(f"\n{'Session':<25s}  {'PNGs':>5s}  {'Lbl':>5s}  {'Status':<8s}  {'Flags'}")
    print("-" * 65)

    total_pngs = 0
    total_labeled = 0
    for ses_info in sessions:
        ld = find_labeled_data_dir(ses_info["sub"], ses_info["ses"])
        if ld is None:
            pngs, labeled = 0, 0
        else:
            pngs, labeled = _count_labeled(ld)

        total_pngs += pngs
        total_labeled += labeled

        if pngs == 0 and labeled == 0:
            status = "empty"
        elif pngs == labeled:
            status = "done"
        else:
            status = "partial"

        flags = []
        if ses_info["primary"]:
            flags.append("primary")
        if ses_info["exclude"]:
            flags.append("excl")

        print(f"{ses_info['exp_id'][:25]:<25s}  {pngs:>5d}  {labeled:>5d}  "
              f"{status:<8s}  {' '.join(flags)}")

    print(f"\nTotal: {total_pngs} PNGs, {total_labeled} labeled")


# ---------------------------------------------------------------------------
# Extract outlier frames
# ---------------------------------------------------------------------------


def extract_outliers_for_session(
    s3: Any,
    ses_info: dict,
    per_session: int | None,
    min_per_session: int | None,
    jump_threshold: float,
    p_bound: float,
    dry_run: bool,
) -> int:
    """Run DLC extract_outlier_frames on one session.

    Returns the number of new frames extracted.
    """
    import deeplabcut

    sub, ses = ses_info["sub"], ses_info["ses"]
    exp_id = ses_info["exp_id"]

    # Check if session already has enough frames
    ld = find_labeled_data_dir(sub, ses)
    pngs_before = len(list(ld.glob("frame_*.png"))) if ld else 0

    if min_per_session is not None and pngs_before >= min_per_session:
        log.info("  %s: already has %d frames (>= min %d), skipping",
                 exp_id[:25], pngs_before, min_per_session)
        return 0

    # How many frames to extract
    n_to_extract = per_session
    if min_per_session is not None:
        need = min_per_session - pngs_before
        if n_to_extract is None:
            n_to_extract = need
        else:
            n_to_extract = min(n_to_extract, need)
        if n_to_extract <= 0:
            return 0

    # Find or download video
    video_path = find_video_local(sub, ses)
    tmp_dir = None
    if video_path is None:
        tmp_dir = tempfile.mkdtemp(prefix=f"hm2p-outlier-{exp_id[:20]}-")
        video_path = download_video_from_s3(s3, sub, ses, Path(tmp_dir))
        if video_path is None:
            log.warning("  No video for %s", exp_id)
            return 0

    log.info("  Video: %s  (have %d, extracting up to %s)",
             video_path.name, pngs_before,
             str(n_to_extract) if n_to_extract else "DLC default")

    if dry_run:
        log.info("  [DRY RUN] Would run extract_outlier_frames on %s", exp_id)
        return 0

    # Build kwargs for numframes2pick
    extract_kwargs: dict[str, Any] = {}
    if n_to_extract is not None:
        # DLC splits between the two algorithms, so give half to each
        extract_kwargs["numframes2pick"] = max(1, n_to_extract // 2)

    # Pass 1: jump-based outliers
    try:
        log.info("  Pass 1: jump outliers (threshold=%d px)", jump_threshold)
        deeplabcut.extract_outlier_frames(
            config=str(DLC_CONFIG),
            videos=[str(video_path)],
            outlieralgorithm="jump",
            epsilon=jump_threshold,
            automatic=True,
            **extract_kwargs,
        )
    except Exception as e:
        log.warning("  Jump extraction failed: %s", e)

    # Pass 2: uncertainty-based outliers
    try:
        log.info("  Pass 2: uncertain outliers (p_bound=%.3f)", p_bound)
        deeplabcut.extract_outlier_frames(
            config=str(DLC_CONFIG),
            videos=[str(video_path)],
            outlieralgorithm="uncertain",
            p_bound=p_bound,
            automatic=True,
            **extract_kwargs,
        )
    except Exception as e:
        log.warning("  Uncertain extraction failed: %s", e)

    # Count PNGs after
    ld = find_labeled_data_dir(sub, ses)
    pngs_after = len(list(ld.glob("frame_*.png"))) if ld else 0
    n_new = pngs_after - pngs_before

    # Clean up temp video
    if tmp_dir is not None:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info("  Extracted %d new frames (total PNGs: %d)", n_new, pngs_after)
    return n_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract outlier frames for DLC retraining using DLC's "
                    "own extract_outlier_frames (jump + uncertain)."
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Show labeling status for all sessions.",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Process only this session (exp_id or partial match).",
    )
    parser.add_argument(
        "--per-session", type=int, default=None,
        help="Max new frames per session (split between jump + uncertain).",
    )
    parser.add_argument(
        "--total", type=int, default=None,
        help="Max total new frames across all sessions. Splits evenly.",
    )
    parser.add_argument(
        "--min-per-session", type=int, default=None,
        help="Ensure each session has at least this many frames. "
             "Only extracts for sessions below the minimum.",
    )
    parser.add_argument(
        "--jump-threshold", type=float, default=20,
        help="Jump outlier threshold in pixels (default 20).",
    )
    parser.add_argument(
        "--p-bound", type=float, default=0.01,
        help="Likelihood threshold for uncertain outliers (default 0.01).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without extracting.",
    )
    args = parser.parse_args()

    if args.scan:
        scan_sessions()
        return

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.session:
        sessions = [s for s in sessions if args.session in s["exp_id"]]
        if not sessions:
            print(f"No session matching '{args.session}'")
            sys.exit(1)

    # Compute per-session limit
    per_session = args.per_session
    if args.total is not None and per_session is None:
        per_session = max(1, args.total // len(sessions))

    total_new = 0
    for ses_info in sessions:
        log.info("\n=== %s ===", ses_info["exp_id"])
        n = extract_outliers_for_session(
            s3, ses_info, per_session, args.min_per_session,
            args.jump_threshold, args.p_bound, args.dry_run,
        )
        total_new += n
        if args.total is not None and total_new >= args.total:
            log.info("Reached total limit of %d frames", args.total)
            break

    print(f"\nTotal new frames extracted: {total_new}")
    if total_new > 0 and not args.dry_run:
        print(
            "\nNext steps:\n"
            "  1. Label:   uv run python scripts/interactive_label.py\n"
            "     or:      uv run deeplabcut refine-labels --config "
            f"{DLC_CONFIG.relative_to(REPO_ROOT)}\n"
            "  2. Upload:  uv run python scripts/upload_dlc_labels.py\n"
            "  3. Retrain: uv run python scripts/launch_dlc_finetune_ec2.py --sa-finetune"
        )


if __name__ == "__main__":
    main()
