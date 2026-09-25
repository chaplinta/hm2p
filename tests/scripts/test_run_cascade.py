"""Tests for the pure helpers of ``scripts/run_cascade.py`` (no S3, no TensorFlow)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_cascade as rc  # noqa: E402


class _FakeCascade:
    """Stands in for cascade2p.cascade: returns the block scaled by 2."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def predict(self, model_name, traces, model_folder=None, padding=0, verbosity=1):
        self.calls.append(traces.shape)
        assert padding == 0 and verbosity == 0
        return traces * 2.0


def test_predict_in_chunks_concatenates_and_chunks() -> None:
    rng = np.random.default_rng(0)
    dff = rng.random((10, 50)).astype(np.float32)
    fake = _FakeCascade()
    out = rc.predict_in_chunks(fake, "m", dff, "folder", chunk=4)
    assert out.shape == dff.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, dff * 2.0, rtol=1e-6)
    assert fake.calls == [(4, 50), (4, 50), (2, 50)]


def test_predict_in_chunks_single_chunk() -> None:
    dff = np.ones((3, 20), dtype=np.float32)
    fake = _FakeCascade()
    out = rc.predict_in_chunks(fake, "m", dff, "folder", chunk=100)
    assert fake.calls == [(3, 20)] and float(out.sum()) == 120.0


def test_get_sessions_reads_registry() -> None:
    sessions = rc.get_sessions()
    assert len(sessions) >= 1
    assert sessions[0]["sub"].startswith("sub-") and sessions[0]["ses"].startswith("ses-")
