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

BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
]


def scan_sessions() -> list[dict]:
    """Find all sessions with frames in labeled-data/."""
    if not LABELED_DIR.exists():
        return []

    sessions = []
    for d in sorted(LABELED_DIR.iterdir()):
        if not d.is_dir():
            continue
        pngs = sorted(d.glob("*.png"))
        if not pngs:
            continue

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

        sessions.append({
            "dir": d,
            "name": d.name,
            "n_frames": len(pngs),
            "n_labelled": n_labelled,
        })

    return sessions


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

        print(f"\n  {'#':<4} {'Session':<65} {'Frames':>6} {'Labelled':>9}")
        print("  " + "-" * 86)
        for i, s in enumerate(sessions):
            status = "✓" if s["n_labelled"] >= s["n_frames"] else " "
            print(f"  {i:<4} {s['name']:<65} {s['n_frames']:>6} {s['n_labelled']:>8} {status}")

        total_frames = sum(s["n_frames"] for s in sessions)
        total_labelled = sum(s["n_labelled"] for s in sessions)
        print(f"\n  Total: {total_frames} frames, {total_labelled} labelled "
              f"({total_labelled/total_frames*100:.0f}%)")

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
            print(f"\nOpening {s['name']} ({s['n_frames']} frames)...")
            label_session(s["dir"])
        else:
            print("  Invalid choice.")

    print("\nDone. Next steps:")
    print("  uv run python scripts/upload_dlc_labels.py")
    print("  uv run python scripts/launch_dlc_finetune_ec2.py")


if __name__ == "__main__":
    main()
