"""Tests for ``scripts/run_behaviour_mi.py`` (synthetic random walk)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_behaviour_mi as rb  # noqa: E402
from test_run_popdec_light_dark import _walk  # noqa: E402


def test_session_labels_and_mi() -> None:
    d = _walk(20_000)
    labels = rb.session_labels(d, 20_000)
    expected = {"inout", "position", "light", "running", "hd"}
    expected |= {f"{b}@{c}" for b in ("position", "hd") for c in ("light", "dark")}
    assert set(labels) == expected
    assert set(labels["light"][labels["light"] >= 0]) == {0, 1}
    t = rb.session_mi(d, n_shuffles=5)
    assert {"light", "running"} <= set(t.behaviour) and (t.animal_id == "3").all()
    assert set(t.variant) <= {
        "plain",
        "given_hd",
        "given_position",
        "given_speed",
        "given_hd_speed",
        "given_position_speed",
    }
    assert set(t[t.variant == "given_position"].behaviour) <= {"hd", "hd@light", "hd@dark"}
