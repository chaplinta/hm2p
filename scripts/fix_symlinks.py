#!/usr/bin/env python3
"""Fix all labeled-data symlinks to point to the correct retrain_frames session.

Usage:
    uv run python scripts/fix_symlinks.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interactive_label import _clip_to_retrain_dir

LABELED_DIR = Path("sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data")

fixed = 0
for d in sorted(LABELED_DIR.iterdir()):
    if not d.is_dir():
        continue
    retrain_dir = _clip_to_retrain_dir(d.name)
    if retrain_dir is None:
        continue
    expected = str(retrain_dir.resolve())
    for p in sorted(d.glob("*.png")):
        if not p.is_symlink():
            continue
        if not p.exists() or expected not in str(p.resolve()):
            correct = retrain_dir / p.name
            if correct.exists():
                p.unlink()
                rel = os.path.relpath(correct.resolve(), d.resolve())
                p.symlink_to(rel)
                fixed += 1

print(f"Fixed {fixed} symlinks" if fixed else "All symlinks correct")
