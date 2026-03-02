"""Tests for kinematics/syllables.py — behavioural syllable analysis (Stage 3b)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_syllables_module_importable() -> None:
    """syllables module can be imported without optional dependencies."""
    from hm2p.kinematics import syllables  # noqa: F401


def test_keypoint_moseq_not_required_for_import() -> None:
    """keypoint_moseq import failure does not break the module import."""
    from hm2p.kinematics.syllables import append_syllables_to_h5  # noqa: F401


def test_run_keypoint_moseq_not_implemented() -> None:
    """run_keypoint_moseq raises NotImplementedError (deferred)."""
    from hm2p.kinematics.syllables import run_keypoint_moseq

    with pytest.raises(NotImplementedError):
        run_keypoint_moseq(dlc_files=[], project_dir=Path("."), output_dir=Path("."))


def test_run_vame_not_implemented() -> None:
    """run_vame raises NotImplementedError (deferred)."""
    from hm2p.kinematics.syllables import run_vame

    with pytest.raises(NotImplementedError):
        run_vame(pose_datasets=[], session_ids=[], project_dir=Path("."), output_dir=Path("."))


def test_append_syllables_not_implemented() -> None:
    """append_syllables_to_h5 raises NotImplementedError (deferred)."""
    from hm2p.kinematics.syllables import append_syllables_to_h5

    with pytest.raises(NotImplementedError):
        append_syllables_to_h5(
            kinematics_h5=Path("fake.h5"),
            syllable_ids=np.array([0, 1, 2], dtype=np.int16),
        )
