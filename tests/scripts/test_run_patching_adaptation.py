"""Tests for helpers in ``scripts/run_patching_adaptation.py`` (synthetic inputs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_patching_adaptation as rpa  # noqa: E402


def test_cell_table_joins_date(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "cell_index": [1, 2, 3],
            "animal_id": ["A", "A", "B"],
            "cell_type": ["penkpos", "penkpos", "penkneg"],
            "ephys_id": ["SW0001", "", "SW0001"],
        }
    ).to_csv(tmp_path / "cells.csv", index=False)
    pd.DataFrame({"animal_id": ["A", "B"], "date_slice": ["211217", "220121"]}).to_csv(
        tmp_path / "animals.csv", index=False
    )
    t = rpa.cell_table(tmp_path)
    assert t["cell_index"].tolist() == [1, 3]
    assert t["date_slice"].tolist() == ["211217", "220121"]


def test_find_iv_file(tmp_path: Path) -> None:
    (tmp_path / "211217" / "SW0001" / "IV").mkdir(parents=True)
    (tmp_path / "211217" / "SW0001" / "IV" / "IV2_0001.h5").write_text("x")
    assert rpa.find_iv_file(tmp_path, "211217", "SW0001").name == "IV2_0001.h5"
    (tmp_path / "211217" / "SW0001" / "IV3_0001.h5").write_text("x")
    assert rpa.find_iv_file(tmp_path, "211217", "SW0001").name == "IV3_0001.h5"
    assert rpa.find_iv_file(tmp_path, "211217", "SW0009") is None


def test_cliffs_delta() -> None:
    assert rpa.cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert rpa.cliffs_delta(np.array([1.0]), np.array([1.0])) == 0.0
    assert np.isnan(rpa.cliffs_delta(np.array([np.nan]), np.array([1.0])))


def test_compare_shapes() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    rows = []
    plan = {"A": "penkpos", "B": "penkpos", "C": "penkpos", "D": "penkneg", "E": "penkneg"}
    for a, ct in plan.items():
        for _ in range(4):
            rows.append(
                {"animal_id": a, "cell_type": ct, **{m: rng.normal() for m in rpa.METRICS}}
            )
    rows.append({"animal_id": "E", "cell_type": "penkpos", **{m: 0.0 for m in rpa.METRICS}})
    out = rpa.compare(pd.DataFrame(rows), n_perms=50, seed=0)
    assert len(out) == len(rpa.METRICS)
    assert {"cliffs_delta", "animal_perm_p", "within_E_delta"} <= set(out.columns)


def test_parser() -> None:
    args = rpa.build_parser().parse_args(["--dry-run", "--target-spikes", "8"])
    assert args.dry_run and args.target_spikes == 8
