"""Tests for extraction/caiman.py — CaImAn extractor."""

from __future__ import annotations

import pytest


def test_caiman_neuropil_returns_none() -> None:
    """CaimanExtractor.get_neuropil_traces() always returns None."""
    pytest.skip("Requires synthetic CaImAn HDF5 — implement alongside caiman.py")
