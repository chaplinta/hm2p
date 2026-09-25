#!/usr/bin/env python3
"""Run Stage 5 (sync) for all sessions.

Downloads kinematics.h5 + ca.h5 from S3, resamples kinematics to imaging rate,
merges with calcium data, and uploads sync.h5 to S3 derivatives.

Usage:
    python scripts/run_stage5_sync.py              # all sessions
    python scripts/run_stage5_sync.py --session 0   # first session only
    python scripts/run_stage5_sync.py --dry-run     # show what would be done
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import boto3

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from s3_utils import s3_upload_with_verify

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"


def get_sessions() -> list[dict]:
    """Read session list from metadata/experiments.csv."""
    csv_path = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    sessions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row["exp_id"]
            parts = exp_id.split("_")
            animal = parts[-1]
            sub = f"sub-{animal}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({"exp_id": exp_id, "sub": sub, "ses": ses})
    return sessions


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    """Check whether an S3 key exists."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def attach_syllables(s3, sub: str, ses: str, kin_local: Path) -> bool:
    """Append ``syllable_id`` from ``syllables.npz`` on S3 to a local kinematics.h5.

    Stage 3b (keypoint-MoSeq) writes ``kinematics/<sub>/<ses>/syllables.npz``
    but did not append it to kinematics.h5, so syllables never reached
    sync.h5. Returns True when syllables were attached, False when the file is
    absent or its length does not match the kinematics frame count (logged,
    not raised).
    """
    import io

    import numpy as np

    from hm2p.kinematics.syllables import append_syllables_to_h5

    key = f"kinematics/{sub}/{ses}/syllables.npz"
    try:
        body = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001 - absent file is the normal case
        print("  no syllables.npz — sync.h5 will not carry syllable_id")
        return False
    with np.load(io.BytesIO(body)) as z:
        if "syllable_id" not in z.files:
            print("  syllables.npz has no syllable_id — skipping")
            return False
        syllable_id = np.asarray(z["syllable_id"], dtype=np.int16)
    try:
        append_syllables_to_h5(kin_local, syllable_id)
    except ValueError as exc:
        print(f"  syllables not attached: {exc}")
        return False
    print(f"  attached syllable_id ({len(syllable_id)} frames)")
    return True


def run_session(
    s3,
    sub: str,
    ses: str,
    exp_id: str,
    work_dir: Path,
    dry_run: bool = False,
    force: bool = False,
    champion_id: str = "",
) -> str:
    """Run Stage 5 for a single session. Returns status string.

    Parameters
    ----------
    champion_id:
        The current champion id string from the manifest. Used to verify
        that kinematics.h5 was produced by the current champion before
        running sync. If empty, champion enforcement is skipped (should
        not happen in normal pipeline operation).
    """
    print(f"\n--- {sub}/{ses} ({exp_id}) ---")

    kin_key = f"kinematics/{sub}/{ses}/kinematics.h5"
    ca_key = f"calcium/{sub}/{ses}/ca.h5"
    sync_key = f"sync/{sub}/{ses}/sync.h5"

    # Check if sync.h5 already exists
    if not force and s3_key_exists(s3, DERIVATIVES_BUCKET, sync_key):
        print("  SKIP: sync.h5 already exists (use --force to re-run)")
        return "skip_exists"

    # Check for kinematics.h5
    if not s3_key_exists(s3, DERIVATIVES_BUCKET, kin_key):
        print(f"  SKIP: no kinematics.h5 at {kin_key}")
        return "skip_no_kinematics"

    # Check for ca.h5
    if not s3_key_exists(s3, DERIVATIVES_BUCKET, ca_key):
        print(f"  SKIP: no ca.h5 at {ca_key}")
        return "skip_no_ca"

    if dry_run:
        print("  DRY RUN: would process and upload sync.h5")
        return "dry_run"

    session_dir = work_dir / sub / ses
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download kinematics.h5
        kin_local = session_dir / "kinematics.h5"
        print("  Downloading kinematics.h5...")
        s3.download_file(DERIVATIVES_BUCKET, kin_key, str(kin_local))

        # --- Champion enforcement: verify kinematics.h5 was produced by
        # the current champion model. If the dlc_champion_id attribute
        # does not match, refuse to run. ---
        import h5py

        with h5py.File(kin_local, "r") as f:
            kin_champion_id = f.attrs.get("dlc_champion_id", "unknown")
            if isinstance(kin_champion_id, bytes):
                kin_champion_id = kin_champion_id.decode("utf-8")

        if champion_id and kin_champion_id != champion_id:
            print(
                f"  ERROR: kinematics.h5 was produced by champion "
                f"{kin_champion_id!r}, but the current champion is "
                f"{champion_id!r}. Re-run Stage 3 first."
            )
            return (
                f"error: champion mismatch — kinematics.h5 has "
                f"dlc_champion_id={kin_champion_id!r}, expected {champion_id!r}"
            )
        print(f"  Champion check passed: {kin_champion_id}")

        # Download ca.h5
        ca_local = session_dir / "ca.h5"
        print("  Downloading ca.h5...")
        s3.download_file(DERIVATIVES_BUCKET, ca_key, str(ca_local))

        # Download timestamps.h5
        ts_key = f"movement/{sub}/{ses}/timestamps.h5"
        ts_local = session_dir / "timestamps.h5"
        if s3_key_exists(s3, DERIVATIVES_BUCKET, ts_key):
            print("  Downloading timestamps.h5...")
            s3.download_file(DERIVATIVES_BUCKET, ts_key, str(ts_local))
        else:
            ts_local = None
            print(f"  WARNING: no timestamps.h5 at {ts_key}")

        # Report input stats
        with h5py.File(kin_local, "r") as f:
            kin_frames = f["frame_times"].shape[0]
            print(f"  Kinematics frames: {kin_frames}")

        with h5py.File(ca_local, "r") as f:
            ca_frame_times = f["frame_times"].shape[0]
            dff_shape = f["dff"].shape
            n_rois = dff_shape[0]
            n_imaging_frames = dff_shape[1]
            print(f"  Ca frame_times: {ca_frame_times}, dff shape: {dff_shape}")

        # Warn about frame_times / dff mismatch
        if ca_frame_times != n_imaging_frames:
            print(
                f"  WARNING: ca.h5 frame_times ({ca_frame_times}) != "
                f"dff columns ({n_imaging_frames}) — off by "
                f"{ca_frame_times - n_imaging_frames}"
            )

        # Run sync pipeline
        print("  Running sync pipeline...")
        # Fold keypoint-MoSeq syllables (Stage 3b output) into the local
        # kinematics copy so sync.h5 carries syllable_id at imaging rate.
        attach_syllables(s3, sub, ses, kin_local)

        from hm2p.sync.align import run

        output_path = session_dir / "sync.h5"
        session_id = f"{sub}/{ses}"
        run(
            kinematics_h5=kin_local,
            ca_h5=ca_local,
            session_id=session_id,
            output_path=output_path,
            timestamps_h5=ts_local,
        )

        # Verify that sync.h5 carries dlc_champion_id (it should, via
        # _read_kin_provenance in align.py). If not, stamp it explicitly.
        with h5py.File(output_path, "a") as f:
            existing = f.attrs.get("dlc_champion_id", None)
            if existing is None and champion_id:
                f.attrs["dlc_champion_id"] = champion_id
                print(f"  Stamped dlc_champion_id={champion_id} on sync.h5")

        # Report output stats
        with h5py.File(output_path, "r") as f:
            keys = list(f.keys())
            sync_status = f.attrs.get("sync_status", "unknown")
            sync_champion = f.attrs.get("dlc_champion_id", "missing")
            if isinstance(sync_champion, bytes):
                sync_champion = sync_champion.decode("utf-8")
            print(
                f"  sync.h5 keys: {len(keys)} datasets, "
                f"sync_status={sync_status}, "
                f"dlc_champion_id={sync_champion}"
            )

            if "frame_times" not in f or "dff" not in f:
                # Stub output — sync classified as FAILED
                print(f"  WARNING: sync.h5 is a stub (sync_status={sync_status})")
                print(f"  Available keys: {keys[:10]}")
            else:
                ft_len = f["frame_times"].shape[0]
                dff_out = f["dff"].shape
                print(f"  ROIs: {dff_out[0]}, imaging frames: {dff_out[1]}")
                print(f"  frame_times length: {ft_len}")
                for k in ("hd_deg", "x_mm", "y_mm", "speed_cm_s"):
                    if k in f:
                        print(f"  {k} length: {f[k].shape[0]} (resampled)")

        # Upload to S3 with verify
        print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{sync_key}")
        s3_upload_with_verify(s3, output_path, DERIVATIVES_BUCKET, sync_key)

        print("  DONE")

        return "ok"

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return f"error: {e}"

    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 5 sync")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Process only this session (exp_id string or 0-based index)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if sync.h5 already exists on S3",
    )
    args = parser.parse_args()

    sessions = get_sessions()
    print(f"Found {len(sessions)} sessions")

    s3 = boto3.client("s3", region_name=REGION)
    work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage5-"))
    print(f"Work dir: {work_dir}")

    # Load champion manifest at startup — required for champion enforcement.
    from hm2p.pose.select import ChampionMismatchError, load_champion_manifest

    try:
        champion_manifest = load_champion_manifest(s3, DERIVATIVES_BUCKET)
    except ChampionMismatchError as e:
        print(f"ERROR: {e}")
        print("Declare a champion with scripts/declare_dlc_champion.py first.")
        sys.exit(1)
    champion_id = str(champion_manifest.get("champion_id", ""))
    print(f"Champion: {champion_id}")

    if args.session is not None:
        if args.session.isdigit():
            sessions = [sessions[int(args.session)]]
        else:
            sessions = [s for s in sessions if s["exp_id"] == args.session]
            if not sessions:
                print(f"Session {args.session} not found")
                sys.exit(1)

    results = {}
    try:
        for i, ses in enumerate(sessions):
            status = run_session(
                s3,
                ses["sub"],
                ses["ses"],
                ses["exp_id"],
                work_dir,
                dry_run=args.dry_run,
                force=args.force,
                champion_id=champion_id,
            )
            results[ses["exp_id"]] = status
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Summary
    print(f"\n{'=' * 60}")
    print("Stage 5 Summary:")
    ok = sum(1 for v in results.values() if v == "ok")
    skip = sum(1 for v in results.values() if v.startswith("skip"))
    err = sum(1 for v in results.values() if v.startswith("error"))
    dry = sum(1 for v in results.values() if v == "dry_run")
    print(f"  OK: {ok}, Skipped: {skip}, Errors: {err}, Dry run: {dry}")

    if err > 0:
        print("\nFailed sessions:")
        for exp_id, status in results.items():
            if status.startswith("error"):
                print(f"  {exp_id}: {status}")

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
