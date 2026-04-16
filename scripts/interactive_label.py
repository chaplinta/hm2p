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
EXPERIMENTS_CSV = Path("metadata/experiments.csv")

BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
]


def _clip_to_retrain_dir(clip_name: str) -> Path | None:
    """Map a labeled-data clip dir name to its retrain_frames/ directory.

    Matches on date + animal ID, then picks the closest time match
    (handles multiple sessions per day for the same animal).
    """
    parts = clip_name.split("_")
    if len(parts) < 5:
        return None
    date = parts[0]
    clip_time = int(parts[1] + parts[2] + parts[3])  # e.g. 112203
    animal = parts[4].split("-")[0]

    candidates = []
    for d in RETRAIN_FRAMES_DIR.iterdir():
        if not d.is_dir():
            continue
        fp = d.name.split("_")
        if len(fp) < 2:
            continue
        f_animal = fp[0].replace("sub-", "")
        f_ses = fp[1].replace("ses-", "")  # e.g. 20220804T112159
        f_date = f_ses[:8]
        if f_animal == animal and f_date == date:
            f_time = int(f_ses[9:])  # e.g. 112159
            candidates.append((abs(f_time - clip_time), d))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


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


def _load_experiment_flags() -> dict[tuple[str, str], dict]:
    """Load exclude/primary_exp flags from experiments.csv.

    Returns mapping from (date, animal_id) to {"exclude": bool, "primary": bool}.
    """
    import csv

    flags: dict[tuple[str, str], dict] = {}
    if not EXPERIMENTS_CSV.exists():
        return flags
    with open(EXPERIMENTS_CSV) as f:
        for row in csv.DictReader(f):
            parts = row["exp_id"].split("_")
            date = parts[0]
            animal = parts[-1]
            flags[(date, animal)] = {
                "exclude": str(row.get("exclude", "0")).strip() == "1",
                "primary": str(row.get("primary_exp", "1")).strip() == "1",
            }
    return flags


def scan_sessions() -> list[dict]:
    """Find all sessions in labeled-data/ that have PNGs or labels."""
    if not LABELED_DIR.exists():
        return []

    exp_flags = _load_experiment_flags()
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

        # Match to experiments.csv for exclude/primary flags
        parts = d.name.split("_")
        date = parts[0] if len(parts) >= 5 else ""
        animal = parts[4].split("-")[0] if len(parts) >= 5 else ""
        flags = exp_flags.get((date, animal), {"exclude": False, "primary": True})

        sessions.append({
            "dir": d,
            "name": d.name,
            "n_frames": n_pngs,
            "n_labelled": n_labelled,
            "needs_pngs": n_pngs == 0,
            "exclude": flags["exclude"],
            "primary": flags["primary"],
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

    DLC rc13 workaround: label_frames() only reads the first folder
    under labeled-data/. Instead of moving directories (which risks
    data loss on crash), we create a temporary DLC project directory
    with a labeled-data/ containing only the target session (via
    symlink), plus a copy of config.yaml pointing to it.
    """
    import deeplabcut
    import tempfile
    import yaml

    if not CONFIG_PATH.exists():
        print(f"  ERROR: config.yaml not found at {CONFIG_PATH}")
        return

    # Pre-flight: ensure H5 is DLC-compatible (remove corrupt files)
    _validate_h5(session_dir)

    # Create a temporary DLC project with only this session visible.
    # This avoids moving/stashing real directories (crash-safe).
    with tempfile.TemporaryDirectory(prefix="dlc-label-") as tmp:
        tmp_path = Path(tmp)
        tmp_labeled = tmp_path / "labeled-data" / session_dir.name
        tmp_labeled.mkdir(parents=True)

        # Symlink all files from the real session dir into the temp one
        for f in session_dir.iterdir():
            (tmp_labeled / f.name).symlink_to(f.resolve())

        # Copy config.yaml, updating the project_path
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        config["project_path"] = str(tmp_path)
        tmp_config = tmp_path / "config.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(config, f)

        print(f"  Opening DLC labelling GUI...")
        print(f"  - Select bodypart from the dropdown")
        print(f"  - Click to place labels")
        print(f"  - Use the slider or arrow keys to navigate frames")
        print(f"  - Save with Ctrl+S or the save button")
        print(f"  - Close the napari window when done")

        try:
            deeplabcut.label_frames(str(tmp_config))
            try:
                import napari
                napari.run()
            except (ImportError, RuntimeError):
                pass
        except Exception as exc:
            print(f"  ERROR in napari: {exc}")

        # Copy ALL CollectedData files from temp back to real dir.
        # DLC/napari may: (a) write through symlinks, (b) replace
        # symlinks with new files, or (c) create new files alongside.
        # We unconditionally copy any non-symlink CollectedData file
        # and also re-read the real dir to check for in-place updates.
        copied = 0
        for f in tmp_labeled.iterdir():
            if not f.name.startswith("CollectedData"):
                continue
            real_dest = session_dir / f.name
            if f.is_symlink():
                # DLC wrote through the symlink — real file already updated
                pass
            else:
                # DLC created a new file (replaced the symlink) — copy back
                shutil.copy2(str(f), str(real_dest))
                copied += 1
        if copied:
            print(f"  Copied {copied} label file(s) back to labeled-data/")

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

        print(f"\n  {'#':<4} {'Session':<50} {'PNGs':>5} {'Lbl':>5} {'Status':>8} {'Flags':>10}")
        print("  " + "-" * 86)
        for i, s in enumerate(sessions):
            if s["needs_pngs"]:
                status = "no PNGs"
            elif s["n_labelled"] >= s["n_frames"]:
                status = "done"
            elif s["n_labelled"] > 0:
                status = "partial"
            else:
                status = "todo"
            flags = ""
            if s["exclude"]:
                flags += "excl"
            if not s["primary"]:
                flags += " 2nd" if flags else "2nd"
            if not flags:
                flags = "primary"
            print(f"  {i:<4} {s['name'][:50]:<50} {s['n_frames']:>5} {s['n_labelled']:>5} {status:>8} {flags:>10}")

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
    import glob as _glob

    # Find all changed CollectedData files across all sessions
    csv_files = _glob.glob(str(LABELED_DIR / "*/CollectedData_*.csv"))
    h5_files = _glob.glob(str(LABELED_DIR / "*/CollectedData_*.h5"))
    label_files = csv_files + h5_files

    if label_files:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--"] + label_files,
            capture_output=True, text=True,
        )
        changed = [l for l in result.stdout.strip().split("\n") if l.strip()]
    else:
        changed = []

    if changed:
        print(f"\n  {len(changed)} label file(s) changed. Committing to git...")
        subprocess.run(["git", "add"] + label_files, capture_output=True)
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
