"""Tests for calcium/events.py — Voigts & Harnett event detection."""

from __future__ import annotations


def test_events_module_importable() -> None:
    """events module can be imported without CASCADE."""
    from hm2p.calcium import events  # noqa: F401
