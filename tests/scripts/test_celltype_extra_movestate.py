"""Tests for ``scripts/celltype_extra_movestate.py`` (synthetic sessions, no S3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_movestate as mv  # noqa: E402

FPS = 9.6


def _arrays(seed: int, n: int = 3000, n_rois: int = 3, syllables: bool = True) -> dict[str, Any]:
    """Blocks of still / straight / turn / runturn; ROI 0 fires during straight running."""
    rng = np.random.default_rng(seed)
    blk = np.repeat(rng.integers(0, 4, n // 20 + 1), 20)[:n]
    speed = np.where(blk % 2 == 1, rng.uniform(6, 20, n), rng.uniform(0, 0.4, n))
    ahv = np.where(blk >= 2, rng.uniform(150, 250, n), rng.normal(0, 5, n))
    sig = rng.gamma(2.0, 0.05, (n_rois, n))
    sig[0] += (blk == 1) * 1.0
    out = {
        "dff": sig,
        "speed_cm_s": speed,
        "ahv_deg_s": ahv,
        "light_on": (np.arange(n) // 576) % 2 == 0,
        "bad_behav": np.zeros(n, bool),
        "fps": FPS,
        "roi_idx": np.arange(n_rois),
    }
    if syllables:
        out["syllable_id"] = blk * 2 + np.repeat(rng.integers(0, 2, n // 10 + 1), 10)[:n]
    return out


def _row(exp: str, animal: str, ct: str) -> pd.Series:
    return pd.Series(
        {
            "exp_id": exp,
            "animal_id": animal,
            "celltype": ct,
            "fibre": "SFB",
            "lens": "f4mm",
            "virus_id": "x",
            "gcamp": "7f",
        }
    )


def _args(**kw: Any) -> argparse.Namespace:
    base = {"signal": "dff", "n_perms": 50, "seed": 0, "n_shuffles_evt": 30}
    base.update(kw)
    return argparse.Namespace(**base)


def test_design_kwargs_defaults_and_overrides() -> None:
    from hm2p.analysis import movement_state as ms

    kw = mv.design_kwargs(_args())
    assert kw["speed_lo"] == ms.DEFAULT_SPEED_LO and kw["ahv_hi"] == ms.DEFAULT_AHV_HI
    kw = mv.design_kwargs(_args(ms_ahv_lo=45.0, ms_smooth_s=None))
    assert kw["ahv_lo"] == 45.0 and kw["smooth_s"] == ms.DEFAULT_SMOOTH_S


def test_session_tables() -> None:
    cells, summary, syl = mv.session_tables(_arrays(0), _args())
    assert len(cells) == 3 and len(summary) == 1
    for k in mv.TESTED:
        assert f"{k}_p" in cells.columns
    assert {mv.VALUE_COLUMN[k] for k in mv.VALUE_COLUMN} <= set(cells.columns)
    assert cells.loc[0, "run_index"] > 0.5 and cells.loc[0, "run_vs_turn_bias"] > 0.5
    assert summary.loc[0, "has_syllables"] and summary.loc[0, "n_syllables_included"] == 8
    assert "n_turn_dark" in summary.columns and len(syl) == 8


def test_session_tables_without_syllables_or_bad_behav() -> None:
    arr = _arrays(1, syllables=False)
    arr.pop("bad_behav")
    arr.pop("light_on")
    cells, summary, syl = mv.session_tables(arr, _args())
    assert syl.empty and not summary.loc[0, "has_syllables"]
    assert cells["syl_mi"].isna().all()


def test_session_tables_rare_state_warns(caplog: Any) -> None:
    arr = _arrays(2)
    arr["ahv_deg_s"] = np.zeros_like(arr["ahv_deg_s"])  # no turning at all
    with caplog.at_level("WARNING", logger="celltype_programme"):
        _, summary, _ = mv.session_tables(arr, _args())
    assert "turn" in summary.loc[0, "rare_states"]
    assert "rare states" in caplog.text


def test_within_group() -> None:
    vals = [0.2, 0.3, 0.1, 0.4, 0.2, 0.5]
    cols: dict[str, Any] = {"celltype": ["penk"] * 6, "animal_id": list("aabbcc")}
    for k in mv.TESTED + mv.DESCRIPTIVE:
        cols[mv.VALUE_COLUMN.get(k, k)] = vals
    for k in mv.TESTED:
        cols[f"{k}_p"] = [0.01, 0.2, 0.03, 0.5, 0.01, 0.9]
        cols[f"{k}_z"] = [2.5, 1.0, -2.2, 0.1, 3.0, 0.0]
    wg = mv.within_group(pd.DataFrame(cols))
    assert len(wg) == len(mv.TESTED) + len(mv.DESCRIPTIVE)
    row = wg[wg.measure == "run_index"].iloc[0]
    assert row["n_sig"] == 3 and row["n_sig_pos"] == 2 and row["n_sig_neg"] == 1
    assert np.isfinite(row["animal_wilcoxon_p"]) and row["n_animals"] == 3
    assert "syl_mi_debiased" in set(wg.measure)
    assert np.isnan(wg[wg.measure == "run_d"].iloc[0].get("n_sig", np.nan))


def test_report_metrics() -> None:
    cells = pd.DataFrame({"run_index": [0.1], "syl_mi_debiased": [np.nan], "bias_d": [1.0]})
    assert mv.report_metrics(cells) == ["run_index", "bias_d"]


def test_run_movestate(tmp_path: Path) -> None:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "penk"), ("n1", "nonpenk"), ("n2", "nonpenk")]
    sessions = [(_row(f"e{i}", a, ct), _arrays(i)) for i, (a, ct) in enumerate(specs)]
    res = mv.run_movestate(_args(), sessions, tmp_path)
    assert res["n_sessions"] == 5 and res["n_cells"] == 15
    for name in (
        "cells",
        "state_summary",
        "syllable_summary",
        "within_group",
        "between_group_report",
    ):
        assert (tmp_path / f"{name}.csv").exists()
    report = pd.read_csv(tmp_path / "between_group_report.csv")
    assert "run_index" in set(report["metric"])


def test_run_movestate_empty(tmp_path: Path) -> None:
    assert mv.run_movestate(_args(), [], tmp_path)["n_sessions"] == 0


def test_registration() -> None:
    import run_celltype_programme as rcp

    assert mv.RUNNER_KEY == "movestate" and mv.RUNNER is mv.run_movestate
    assert mv._get(argparse.Namespace(x=None), "x", 2.0) == 2.0
    assert "movestate" in rcp.HYPOTHESES
    assert rcp.load_extra_runners()["movestate"] is mv.run_movestate
