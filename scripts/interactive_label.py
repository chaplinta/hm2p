#!/usr/bin/env python3
"""Interactive DLC labelling — pick sessions from a menu, label in napari.

Scans labeled-data/ for sessions with frames. Shows an interactive menu.
Pick a session → DLC's napari labelling GUI opens with all frames + existing labels.
Close napari → back to menu. Quit when done.

Uses deeplabcut.label_frames() for the native DLC labelling interface
(napari-deeplabcut plugin with bodypart dropdown, save widget, etc.).

DLC rc13 bug workaround: label_frames() only shows the first folder under
labeled-data/. We temporarily stash other session folders so only the
selected session is visible.

Usage:
    uv run python scripts/interactive_label.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

DLC_PROJECT = Path("sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20")
LABELED_DIR = DLC_PROJECT / "labeled-data"
CONFIG_PATH = DLC_PROJECT / "config.yaml"
STASH_DIR = Path("/tmp/dlc-label-stash")
RETRAIN_FRAMES_DIR = Path("retrain_frames")

BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
]


def _clip_to_retrain_dir(clip_name: str) -> Path | None:
    """Map a labeled-data clip dir name to its retrain_frames/ directory.

    Matches on date + animal ID (the video timestamp differs from the
    session timestamp).
    """
    parts = clip_name.split("_")
    if len(parts) < 5:
        return None
    date = parts[0]
    animal = parts[4].split("-")[0]
    for d in RETRAIN_FRAMES_DIR.iterdir():
        if not d.is_dir():
            continue
        # e.g. sub-1114353_ses-20210823T165950
        fp = d.name.split("_")
        if len(fp) < 2:
            continue
        f_animal = fp[0].replace("sub-", "")
        f_date = fp[1].replace("ses-", "")[:8]
        if f_animal == animal and f_date == date:
            return d
    return None


def _ensure_pngs(labeled_dir: Path) -> int:
    """Symlink PNGs from retrain_frames/ into labeled-data/ if missing.

    Uses relative symlinks so they work across machines (Mac + devcontainer).
    Returns the number of PNGs available after linking.
    """
    existing = sorted(labeled_dir.glob("*.png"))
    if existing:
        # Check first symlink isn't broken
        if existing[0].is_symlink() and not existing[0].exists():
            # Broken symlinks — remove and re-link
            for p in existing:
                if p.is_symlink():
                    p.unlink()
            existing = []
        else:
            return len(existing)

    retrain_dir = _clip_to_retrain_dir(labeled_dir.name)
    if retrain_dir is None or not retrain_dir.exists():
        return 0

    retrain_pngs = sorted(retrain_dir.glob("*.png"))
    for png in retrain_pngs:
        dest = labeled_dir / png.name
        if not dest.exists():
            # Relative symlink: from labeled-data/clip_dir/ to retrain_frames/sub_ses/
            import os
            rel = os.path.relpath(png.resolve(), labeled_dir.resolve())
            dest.symlink_to(rel)

    return len(list(labeled_dir.glob("*.png")))


def scan_sessions() -> list[dict]:
    """Find all sessions in labeled-data/ that have PNGs or labels."""
    if not LABELED_DIR.exists():
        return []

    sessions = []
    for d in sorted(LABELED_DIR.iterdir()):
        if not d.is_dir():
            continue

        # Ensure PNGs are linked from retrain_frames/ if missing
        n_pngs = _ensure_pngs(d)

        # Count existing labels
        n_labelled = 0
        for h5_f in d.glob("CollectedData_*.h5"):
            try:
                df = pd.read_hdf(h5_f)
                n_labelled = int((~df.isna().all(axis=1)).sum())
                break
            except Exception:
                pass
        if n_labelled == 0:
            for csv_f in d.glob("CollectedData_*.csv"):
                try:
                    df = pd.read_csv(csv_f, header=[0, 1, 2], index_col=[0, 1, 2])
                    n_labelled = int((~df.isna().all(axis=1)).sum())
                    break
                except Exception:
                    pass

        if n_pngs == 0 and n_labelled == 0:
            continue

        sessions.append({
            "dir": d,
            "name": d.name,
            "n_frames": n_pngs,
            "n_labelled": n_labelled,
            "needs_pngs": n_pngs == 0,
        })

    return sessions


def _validate_h5(session_dir: Path) -> None:
    """Check CollectedData H5 is DLC-compatible; remove if corrupt.

    napari_deeplabcut crashes on H5 files that have:
    - All-NaN data (no actual labels, just frame placeholders)
    - Flat string index instead of 3-level MultiIndex
    - Wrong HDF5 key name (must be readable by pd.read_hdf)

    If the H5 is corrupt or empty, it is deleted so DLC creates a fresh
    one when napari opens. The CSV is kept as a fallback reference.
    """
    h5_path = session_dir / "CollectedData_tristan.h5"
    if not h5_path.exists():
        return

    try:
        df = pd.read_hdf(h5_path)
    except Exception:
        print(f"  Removing unreadable H5 (DLC will recreate)")
        h5_path.unlink()
        return

    if len(df) == 0:
        return  # Empty is fine — DLC handles it

    # Check for all-NaN (frame placeholders with no labels)
    if not df.notna().any().any():
        print(f"  Removing all-NaN H5 (DLC will recreate with fresh index)")
        h5_path.unlink()
        return

    # Check 3-level MultiIndex on rows
    if df.index.nlevels != 3:
        print(f"  Removing H5 with flat index (DLC will recreate)")
        h5_path.unlink()
        return

    # Check HDF5 key structure matches what napari_deeplabcut expects
    import h5py

    with h5py.File(h5_path, "r") as f:
        main_key = list(f.keys())[0]
        has_axis1_levels = f"{main_key}/axis1_level0" in f
    if not has_axis1_levels:
        print(f"  Removing H5 with missing axis1 levels (DLC will recreate)")
        h5_path.unlink()


def label_session(session_dir: Path):
    """Open DLC's napari labelling GUI for a single session.

    Uses the deeplabcut.label_frames() function which provides the full
    napari-deeplabcut interface: bodypart dropdown, point placement,
    save button, navigation controls.

    DLC rc13 workaround: stash other session folders so only the
    selected session is visible in the GUI.
    """
    import deeplabcut

    if not CONFIG_PATH.exists():
        print(f"  ERROR: config.yaml not found at {CONFIG_PATH}")
        return

    # Pre-flight: ensure H5 is DLC-compatible (remove corrupt files)
    _validate_h5(session_dir)

    labeled_base = DLC_PROJECT / "labeled-data"
    session_name = session_dir.name
    STASH_DIR.mkdir(parents=True, exist_ok=True)

    # Stash other sessions (DLC rc13 only shows first folder)
    stashed = []
    for other_dir in labeled_base.iterdir():
        if other_dir.is_dir() and other_dir.name != session_name:
            dest = STASH_DIR / other_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(other_dir), str(dest))
            stashed.append(other_dir.name)

    if stashed:
        print(f"  Stashed {len(stashed)} other session(s) during labelling")

    try:
        print(f"  Opening DLC labelling GUI...")
        print(f"  - Select bodypart from the dropdown")
        print(f"  - Click to place labels")
        print(f"  - Use the slider or arrow keys to navigate frames")
        print(f"  - Save with Ctrl+S or the save button")
        print(f"  - Close the napari window when done")

        deeplabcut.label_frames(str(CONFIG_PATH))

        try:
            import napari
            napari.run()
        except (ImportError, RuntimeError):
            pass

    finally:
        # Always restore stashed sessions
        for name in stashed:
            src = STASH_DIR / name
            dest = labeled_base / name
            if src.exists():
                shutil.move(str(src), str(dest))
        if stashed:
            print(f"  Restored {len(stashed)} stashed session(s)")

        # Auto-commit labels after each session so work is never lost
        import subprocess

        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(session_dir)],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            subprocess.run(
                ["git", "add",
                 str(session_dir / "CollectedData_tristan.csv"),
                 str(session_dir / "CollectedData_tristan.h5")],
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m",
                 f"Auto-save labels: {session_dir.name[:50]}"],
                capture_output=True,
            )
            print(f"  Labels auto-committed to git.")


def main():
    print("=" * 60)
    print("  hm2p Interactive DLC Labeller")
    print("=" * 60)

    # Verify DLC is available
    try:
        import deeplabcut  # noqa: F401
    except ImportError:
        print("\nERROR: deeplabcut not installed.")
        print("Install with: uv pip install deeplabcut")
        return

    if not CONFIG_PATH.exists():
        print(f"\nERROR: DLC project config not found at {CONFIG_PATH}")
        return

    while True:
        sessions = scan_sessions()

        if not sessions:
            print("\nNo sessions with frames found in labeled-data/.")
            print("Run scripts/select_labelling_frames.py first to extract frames.")
            break

        print(f"\n  {'#':<4} {'Session':<50} {'PNGs':>6} {'Labels':>7} {'Status':>10}")
        print("  " + "-" * 80)
        for i, s in enumerate(sessions):
            if s["needs_pngs"]:
                status = "no PNGs"
            elif s["n_labelled"] >= s["n_frames"]:
                status = "done"
            elif s["n_labelled"] > 0:
                status = "partial"
            else:
                status = "todo"
            print(f"  {i:<4} {s['name'][:50]:<50} {s['n_frames']:>6} {s['n_labelled']:>7} {status:>10}")

        n_with_pngs = sum(1 for s in sessions if not s["needs_pngs"])
        n_needs_pngs = sum(1 for s in sessions if s["needs_pngs"])
        total_labelled = sum(s["n_labelled"] for s in sessions)
        print(f"\n  {len(sessions)} sessions ({n_with_pngs} with PNGs, {n_needs_pngs} need extraction)")
        print(f"  {total_labelled} frames labelled")
        if n_needs_pngs:
            print(f"  Sessions marked 'no PNGs' need frame extraction first:")
            print(f"    uv run python scripts/prepare_retrain_frames.py <sub/ses> <frame_indices...>")

        print(f"\n  Enter session number (0-{len(sessions)-1}), 'a' for all, or 'q' to quit:")
        choice = input("  > ").strip()

        if choice.lower() == "q":
            break
        elif choice.lower() == "a":
            for i, s in enumerate(sessions):
                print(f"\n[{i+1}/{len(sessions)}] {s['name']}")
                label_session(s["dir"])
        elif choice.isdigit() and 0 <= int(choice) < len(sessions):
            s = sessions[int(choice)]
            if s["needs_pngs"]:
                print(f"\n  Session {s['name'][:50]} has no PNGs.")
                print(f"  Extract frames first with prepare_retrain_frames.py")
                continue
            print(f"\nOpening {s['name']} ({s['n_frames']} frames)...")
            label_session(s["dir"])
        else:
            print("  Invalid choice.")

    # Auto-commit labels to git so they're never lost
    import subprocess

    result = subprocess.run(
        ["git", "status", "--porcelain", "--", str(LABELED_DIR)],
        capture_output=True, text=True,
    )
    changed = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if changed:
        print(f"\n  {len(changed)} label file(s) changed. Committing to git...")
        subprocess.run(
            ["git", "add", str(LABELED_DIR / "*/CollectedData_*")],
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Update DLC labels from interactive labelling session"],
            capture_output=True,
        )
        print("  Labels committed to git.")
    else:
        print("\n  No label changes to commit.")

    print("\nDone. Next steps:")
    print("  git push origin main")
    print("  uv run python scripts/upload_dlc_labels.py")
    print("  uv run python scripts/launch_dlc_finetune_ec2.py")


if __name__ == "__main__":
    main()
