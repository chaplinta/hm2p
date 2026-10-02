"""Tests for ``scripts/celltype_extra_tctx.py`` with synthetic sessions (no S3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_tctx as tctx  # noqa: E402

FPS = 2.0
EPOCH_FRAMES = 120  # 60 s epochs at 2 Hz
N_EPOCHS = 8


def _arrays(seed: int, kind: str, n_rois: int = 9) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = EPOCH_FRAMES * N_EPOCHS
    light_on = (np.arange(n) // EPOCH_FRAMES) % 2 == 0
    dff = rng.standard_normal((n_rois, n)) * 0.3
    if kind == "drift":
        t = np.linspace(0.0, 1.0, n)
        dff += rng.choice([-1.0, 1.0], n_rois)[:, None] * 2.0 * t[None, :]
    bad = np.zeros(n, dtype=bool)
    bad[10:20] = True
    return {
        "dff": dff,
        "light_on": light_on,
        "bad_behav": bad,
        "fps": FPS,
        "roi_idx": np.arange(n_rois) + 100,
    }


def _meta(exp_id: str, animal: str, celltype: str) -> pd.Series:
    return pd.Series(
        {
            "exp_id": exp_id,
            "animal_id": animal,
            "celltype": celltype,
            "fibre": "SFB",
            "lens": "f4mm",
            "virus_id": "ADD3" if celltype == "penk" else "344",
            "gcamp": "7f",
        }
    )


@pytest.fixture
def args() -> argparse.Namespace:
    return argparse.Namespace(
        signal="dff",
        n_perms=20,
        seed=0,
        n_perms_tctx_decode=10,
        n_shuffles_tctx=20,
        n_draws_tctx=3,
        n_cells_matched=4,
    )


@pytest.fixture
def sessions() -> list[tuple[pd.Series, dict[str, Any]]]:
    specs = [
        ("a1", "penk", "drift", 9),
        ("a2", "penk", "drift", 9),
        ("a3", "penk", "drift", 3),  # fewer cells than matched N
        ("b1", "nonpenk", "noise", 9),
        ("b2", "nonpenk", "noise", 9),
        ("b3", "nonpenk", "noise", 9),
    ]
    return [
        (_meta(f"2022010{i}_10_00_00_{a}", a, ct), _arrays(i, kind, n))
        for i, (a, ct, kind, n) in enumerate(specs)
    ]


def test_module_exports() -> None:
    assert tctx.RUNNER_KEY == "tctx"
    assert tctx.RUNNER is tctx.run_tctx
    assert tctx.EXTRA_SYNC_KEYS == []


def test_params_defaults() -> None:
    p = tctx._params(argparse.Namespace())
    assert p["bin_s"] == 10.0 and p["bin_s_sel"] == 1.0 and p["n_matched"] == 8


def test_frac_sig() -> None:
    assert tctx._frac_sig(pd.Series([0.01, 0.2, np.nan, 0.04])) == pytest.approx(2 / 3)
    assert np.isnan(tctx._frac_sig(pd.Series([np.nan])))


def test_matched_epoch_accuracy() -> None:
    rng = np.random.default_rng(0)
    idx = np.repeat(np.arange(6), 5)
    light = idx % 2 == 0
    x = rng.standard_normal((5, 6))[:, idx] * 3 + 0.1 * rng.standard_normal((5, idx.size))
    val = tctx.matched_epoch_accuracy(x, idx, light, 3, 4, rng)
    assert val > 0.3
    assert np.isnan(tctx.matched_epoch_accuracy(x, idx, light, 10, 4, rng))


def test_session_tctx_drift_vs_noise(args: argparse.Namespace) -> None:
    params = tctx._params(args)
    rng = np.random.default_rng(0)
    drift = _arrays(0, "drift")
    cells, rec = tctx.session_tctx(drift["dff"], drift, params, rng)
    assert len(cells) == 9
    assert rec["time_rho_full"] > 0.8
    assert rec["time_rho_matched"] > 0.5
    assert np.all(cells["drift_abs_rho"] > 0.5)
    noise = _arrays(1, "noise")
    _, rec_n = tctx.session_tctx(noise["dff"], noise, params, rng)
    assert rec_n["time_rho_full"] < rec["time_rho_full"]
    assert rec_n["time_mae_full_s"] > rec["time_mae_full_s"]
    assert rec["n_dark_epochs"] == N_EPOCHS // 2


def test_run_tctx_outputs(args: argparse.Namespace, sessions: list, tmp_path: Path) -> None:
    out = tctx.run_tctx(args, iter(sessions), tmp_path)
    assert out["n_sessions"] == 6
    for name in ("cells", "sessions", "within_group", "between_group_report"):
        assert (tmp_path / f"{name}.csv").exists()
    cells = pd.read_csv(tmp_path / "cells.csv")
    sess = pd.read_csv(tmp_path / "sessions.csv")
    assert len(cells) == 9 * 5 + 3
    assert set(cells["roi_idx"]) >= {100, 108}
    assert {"drift_rho", "epoch_sel_eps2", "dark_first_minus_later"} <= set(cells.columns)
    small = sess[sess["animal_id"] == "a3"].iloc[0]
    assert np.isnan(small["time_mae_matched_s"])
    penk = sess[sess["celltype"] == "penk"]["time_rho_full"].median()
    nonpenk = sess[sess["celltype"] == "nonpenk"]["time_rho_full"].median()
    assert penk > nonpenk
    within = out["within"]
    assert {"cell_frac", "session_frac", "animal_wilcoxon"} == set(within["test"])
    report = out["report"]
    assert {"tctx_session", "tctx_cell"} <= set(report["family"])


def test_run_tctx_skips_empty_signal(
    args: argparse.Namespace, sessions: list, tmp_path: Path
) -> None:
    row, arrays = sessions[0]
    empty = dict(arrays, dff=np.zeros((0, arrays["dff"].shape[1])))
    out = tctx.run_tctx(args, iter([(row, empty)]), tmp_path)
    assert out == {"n_sessions": 0}
    assert (tmp_path / "summary.json").exists()


def test_run_tctx_no_sessions(args: argparse.Namespace, tmp_path: Path) -> None:
    assert tctx.run_tctx(args, iter([]), tmp_path) == {"n_sessions": 0}


def test_within_group_wilcoxon_needs_two_animals() -> None:
    cells = pd.DataFrame(
        {"celltype": ["penk"] * 3, "drift_p": [0.01, 0.2, 0.03], "epoch_sel_p": [0.5] * 3}
    )
    sess = pd.DataFrame(
        {
            "celltype": ["penk"],
            "animal_id": ["a"],
            "time_p_full": [0.01],
            "epoch_p": [0.5],
            "tdist_p": [np.nan],
            **{c: [0.3] for c in tctx.SESSION_WILCOXON_COLS},
        }
    )
    w = tctx.tctx_within_group(cells, sess)
    frac = w[(w["test"] == "cell_frac") & (w["metric"] == "drift_p")].iloc[0]
    assert frac["k_significant"] == 2 and frac["n"] == 3
    wil = w[w["test"] == "animal_wilcoxon"]
    assert wil["p_value"].isna().all()
    tdist = w[(w["test"] == "session_frac") & (w["metric"] == "tdist_p")].iloc[0]
    assert tdist["n"] == 0 and np.isnan(tdist["fraction"])


def test_params_treats_none_as_default() -> None:
    import argparse

    import celltype_extra_tctx as ext

    p = ext._params(argparse.Namespace(n_cells_matched=None, bin_s_ctx=None))
    assert p["n_matched"] == 8 and p["bin_s"] == 10.0
    assert ext._params(argparse.Namespace(n_cells_matched=5))["n_matched"] == 5
