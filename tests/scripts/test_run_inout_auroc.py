"""Tests for ``scripts/run_inout_auroc.py`` (synthetic random walk)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_inout_auroc as ra  # noqa: E402
from test_run_popdec_light_dark import _walk  # noqa: E402


def test_session_auroc_runs_on_random_walk() -> None:
    t, nulls = ra.session_auroc(_walk(20_000), n_shuffles=20)
    assert len(t) > 0 and set(t.condition) <= set(ra.CONDITIONS)
    assert t.auc.between(0, 1).all() and (t.animal_id == "3").all()
    assert all(k in nulls[c] for c in nulls for k in ("raw", "hd"))
