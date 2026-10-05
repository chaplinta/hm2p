# ruff: noqa: E501
"""Tests for ``scripts/make_rsp_maze_report.py`` (synthetic tables only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_rsp_maze_report as mr  # noqa: E402


def _entries(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(6):
        for _ in range(40):
            rec = float(np.exp(rng.uniform(1, 5)))
            light = float(rng.random() > 0.5)
            pop = (0.1 * np.log(rec) if light == 0 else 0.0) + rng.normal(0, 0.05)
            rows.append(
                {
                    "exp_id": f"s{s}",
                    "animal_id": str(s % 4),
                    "recency_s": rec,
                    "light_frac": light,
                    "pop_response": pop,
                    "speed_pre": 5 + 0.5 * np.log(rec) + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_binned_population_rises_in_dark_only() -> None:
    b = mr.binned_population(_entries())
    assert (
        b["dark"]["median"][2] > b["dark"]["median"][0] and b["dark"]["long_minus_short_p"] < 0.05
    )
    assert abs(b["light"]["median"][2] - b["light"]["median"][0]) < 0.1
    assert b["dark"]["median_recency_s"][0] < b["dark"]["median_recency_s"][2]


def test_approach_behaviour() -> None:
    a = mr.approach_behaviour(_entries())
    assert a["dark"]["median_rho"] > 0.5 and a["light"]["n_animals"] == 4


def test_per_animal_and_variants(tmp_path: Path) -> None:
    cells = pd.DataFrame(
        {
            "exp_id": ["s"] * 4,
            "roi": [0, 0, 1, 1],
            "animal_id": ["a"] * 4,
            "cond": ["dark", "light"] * 2,
            "z": [1.0, -1.0, 3.0, 0.0],
        }
    )
    pa = mr.per_animal(cells)
    assert pa == [{"animal_id": "a", "dark": 2.0, "light": -0.5, "n_cells": 2}]
    d = tmp_path / "dead_end_recency" / "spikes_w1.5"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "contrast": ["dark", "dark_minus_light"],
            "median_z": [0.3, 0.4],
            "animal_p": [0.02, 0.01],
            "session_frac_pos": [0.7, 0.8],
            "n_animals": [15, 15],
        }
    ).to_csv(d / "summary.csv", index=False)
    v = mr.recency_variants(tmp_path)
    assert v[0]["key"] == "spikes_w1.5" and v[0]["dark"]["median_z"] == 0.3 and "light" not in v[0]


def test_build_empty_and_clean(tmp_path: Path) -> None:
    out = mr.build(tmp_path)
    assert out["variants"] == [] and out["trips"] == [] and out["decoding"] == {}
    assert out["inout_sessions"] == [] and len(out["leads"]) == 5 and "headline" in out
    assert mr._clean({"a": [float("nan"), np.bool_(True)]}) == {"a": [None, True]}


def test_inout_tables(tmp_path: Path) -> None:
    d = tmp_path / "popdec_light_dark"
    d.mkdir()
    rows = []
    for a in range(5):
        for cond in ("light", "dark"):
            for f in ("neural", "behaviour"):
                rows.append(
                    {
                        "decoder": "inout",
                        "metric": "balanced_accuracy",
                        "condition": cond,
                        "feature_set": f,
                        "exp_id": f"s{a}",
                        "animal_id": a,
                        "observed": 0.6,
                        "null_mean": 0.5,
                        "excess": 0.1 + a / 100,
                    }
                )
    pd.DataFrame(rows).to_csv(d / "sessions_spikes.csv", index=False)
    sessions, summary = mr.inout_tables(tmp_path)
    assert (
        len(sessions) == 20 and summary[0]["condition"] == "light" and summary[0]["n_animals"] == 5
    )
    assert summary[0]["animal_p"] < 0.1
