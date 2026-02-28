"""Tests for io/nwb.py — NWB export."""

from __future__ import annotations


def test_nwb_module_importable() -> None:
    """nwb module can be imported without neuroconv installed."""
    from hm2p.io import nwb  # noqa: F401
