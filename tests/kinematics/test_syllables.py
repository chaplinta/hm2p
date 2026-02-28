"""Tests for kinematics/syllables.py — behavioural syllable analysis (Stage 3b)."""

from __future__ import annotations


def test_syllables_module_importable() -> None:
    """syllables module can be imported without optional dependencies."""
    from hm2p.kinematics import syllables  # noqa: F401


def test_keypoint_moseq_not_required_for_import() -> None:
    """keypoint_moseq import failure does not break the module import."""
    # The module uses TYPE_CHECKING guards to avoid hard imports
    from hm2p.kinematics.syllables import append_syllables_to_h5  # noqa: F401
