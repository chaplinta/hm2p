"""Tests for io/nwb.py — NWB export."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_nwb_module_importable() -> None:
    """nwb module can be imported without neuroconv installed."""
    from hm2p.io import nwb  # noqa: F401


def test_export_session_to_nwb_not_implemented() -> None:
    """export_session_to_nwb raises NotImplementedError (deferred)."""
    from hm2p.io.nwb import export_session_to_nwb

    with pytest.raises(NotImplementedError):
        export_session_to_nwb(
            ca_h5=Path("ca.h5"),
            kinematics_h5=Path("kinematics.h5"),
            sync_h5=Path("sync.h5"),
            session_id="test",
            output_path=Path("output.nwb"),
        )
