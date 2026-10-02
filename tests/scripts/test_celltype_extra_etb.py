"""Tests for ``scripts/celltype_extra_etb.py`` with synthetic sessions (no S3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_etb as etbr  # noqa: E402

FPS = 9.6
N_FRAMES = 1200
N_ROIS = 3


def _synthetic_arrays(
    seed: int, n_rois: int = N_ROIS, n_frames: int = N_FRAMES, head_body: bool = True
) -> dict[str, Any]:
    """Small session: cell 0 has transients at speed drops, others are noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    light_on = (t // 300) % 2 == 0
    speed = np.clip(rng.normal(10.0, 1.0, n_frames), 0.0, None)
    drops = np.arange(80, n_frames - 80, 70)
    for d in drops:
        speed[d : d + 15] = 0.3
    active = speed > 2.0
    hd = np.mod(np.cumsum(rng.normal(0, 8, n_frames)), 360.0)
    ahv = np.gradient(np.unwrap(np.deg2rad(hd))) * FPS * 180 / np.pi
    dff = rng.normal(0.0, 0.03, (n_rois, n_frames))
    kern = np.exp(-np.arange(30) / 10.0)
    for d in drops:
        dff[0, d : d + 30] += kern
    syl = rng.integers(0, 4, n_frames).astype(np.int16)
    syl[drops] = 2
    arrays: dict[str, Any] = {
        "dff": dff,
        "hd_deg": hd,
        "ahv_deg_s": ahv,
        "speed_cm_s": speed,
        "light_on": light_on,
        "active": active,
        "bad_behav": np.zeros(n_frames, dtype=bool),
        "mask": active.copy(),
        "x_maze": rng.uniform(0, 7, n_frames),
        "y_maze": rng.uniform(0, 5, n_frames),
        "syllable_id": syl,
        "roi_idx": np.arange(n_rois),
        "fps": FPS,
    }
    if head_body:
        arrays["x_body_mm"] = rng.uniform(0, 500, n_frames)
        arrays["y_body_mm"] = rng.uniform(0, 500, n_frames)
        arrays["x_head_mm"] = arrays["x_body_mm"] + rng.normal(0, 5, n_frames)
        arrays["y_head_mm"] = arrays["y_body_mm"] + rng.normal(0, 5, n_frames)
    return arrays


def _meta_row(exp_id: str, animal: str, celltype: str) -> pd.Series:
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
def sessions() -> list[tuple[pd.Series, dict[str, Any]]]:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "penk"), ("n1", "nonpenk"), ("n2", "nonpenk")]
    return [
        (_meta_row(f"2022010{i}_10_00_00_{a}", a, ct), _synthetic_arrays(i))
        for i, (a, ct) in enumerate(specs)
    ]


@pytest.fixture
def args() -> argparse.Namespace:
    return argparse.Namespace(signal="dff", n_perms=20, seed=0, n_shuffles_evt=30)


class TestRegistration:
    def test_module_exports(self) -> None:
        assert etbr.RUNNER_KEY == "etb"
        assert etbr.RUNNER is etbr.run_etb
        assert etbr.EXTRA_SYNC_KEYS == ["x_head_mm", "y_head_mm", "x_body_mm", "y_body_mm"]

    def test_grid(self) -> None:
        assert etbr.ETB_GRID_S[0] == -5.0 and etbr.ETB_GRID_S[-1] == 5.0


class TestRunner:
    def test_runs_and_writes(self, sessions, args, tmp_path: Path) -> None:
        res = etbr.run_etb(args, sessions, tmp_path)
        assert res["n_sessions"] == 5
        assert res["n_cells"] == 5 * N_ROIS
        for name in (
            "cells.csv",
            "sessions.csv",
            "event_triggered_timecourses.json",
            "within_group.csv",
            "between_group_report.csv",
        ):
            assert (tmp_path / name).exists(), name
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert {"head_body_angle_deg", "node_junction", "speed_cm_s"} <= set(cells["channel"])
        assert {"syllable_max_z", "syllable_enriched", "roi_idx", "celltype"} <= set(cells.columns)
        # planted cell: speed falls after onset and during events
        planted = cells[(cells["roi_idx"] == 0) & (cells["channel"] == "speed_cm_s")]
        assert (planted["onset_z"] < 0).all()
        assert (planted["during_z"] < 0).all()
        assert (planted["onset_p"] < 0.1).mean() >= 0.8
        sess = pd.read_csv(tmp_path / "sessions.csv")
        assert {"frac_onset_sig", "frac_during_sig", "frac_cells_syllable_enriched"} <= set(
            sess.columns
        )
        rep = res["report"]
        assert {"etb_onset", "etb_during"} <= set(rep["family"])
        within = res["within"]
        assert set(within["test"]) == {"onset", "during"}
        assert set(within["celltype"]) == {"penk", "nonpenk"}

    def test_timecourses(self, sessions, args, tmp_path: Path) -> None:
        etbr.run_etb(args, sessions[:2], tmp_path)
        tc = json.loads((tmp_path / "event_triggered_timecourses.json").read_text())
        assert len(tc["time_s"]) == len(etbr.ETB_GRID_S)
        assert tc["speed_cm_s"]["penk"]["n_sessions"] == 2
        assert len(tc["speed_cm_s"]["penk"]["mean"]) == len(etbr.ETB_GRID_S)

    def test_missing_optional_keys(self, args, tmp_path: Path) -> None:
        arr = _synthetic_arrays(0, n_rois=2, head_body=False)
        for k in ("syllable_id", "x_maze", "y_maze"):
            arr.pop(k)
        args.signal = "spikes"  # ignored: events are defined on dF/F
        res = etbr.run_etb(args, [(_meta_row("e", "a", "penk"), arr)], tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert "head_body_angle_deg" not in set(cells["channel"])
        assert "syllable_max_z" not in cells.columns
        assert res["report"].empty or "skipped" in res["report"].columns

    def test_default_shuffles(self, tmp_path: Path) -> None:
        args = argparse.Namespace(signal="dff", n_perms=10, seed=0)
        arr = _synthetic_arrays(0, n_rois=1, n_frames=600)
        res = etbr.run_etb(args, [(_meta_row("e", "a", "penk"), arr)], tmp_path)
        assert res["n_sessions"] == 1

    def test_empty(self, args, tmp_path: Path) -> None:
        assert etbr.run_etb(args, [], tmp_path)["n_sessions"] == 0
        assert (tmp_path / "summary.json").exists()


class TestPieces:
    def test_syllable_cell_summary(self) -> None:
        rng = np.random.default_rng(0)
        syl = rng.integers(0, 4, 2000)
        on = np.arange(100, 1900, 50)
        syl[on] = 1
        events = [(on, on + 5), (np.array([], dtype=int), np.array([], dtype=int))]
        bad = np.zeros(2000, dtype=bool)
        bad[:10] = True
        out = etbr.syllable_cell_summary(syl, events, FPS, 50, rng, bad)
        assert out.loc[0, "syllable_top"] == 1
        assert bool(out.loc[0, "syllable_enriched"])
        assert not bool(out.loc[1, "syllable_enriched"])
        assert np.isnan(out.loc[1, "syllable_max_z"])

    def test_session_summary(self) -> None:
        tab = pd.DataFrame(
            {
                "channel": ["a", "a", "b"],
                "n_events": [3, 5, 4],
                "onset_p": [0.01, 0.5, np.nan],
                "onset_z": [-3.0, 0.1, np.nan],
                "during_p": [0.01, 0.02, 0.9],
                "during_z": [2.0, 3.0, 0.0],
            }
        )
        out = etbr.etb_session_summary(tab).set_index("channel")
        assert out.loc["a", "frac_onset_sig"] == 0.5
        assert out.loc["a", "frac_during_sig"] == 1.0
        assert out.loc["b", "n_onset_tested"] == 0
        assert np.isnan(out.loc["b", "frac_onset_sig"])
        assert np.isnan(out.loc["b", "median_onset_z"])

    def test_within_group(self) -> None:
        assert etbr.etb_within_group(pd.DataFrame(), pd.DataFrame()).empty
        sess = pd.DataFrame(
            {
                "channel": ["s"] * 3,
                "celltype": ["penk"] * 3,
                "animal_id": ["a", "b", "c"],
                "exp_id": ["e1", "e2", "e3"],
                "median_onset_z": [-2.0, -3.0, -1.0],
                "median_during_z": [0.0, 0.0, 0.0],
                "n_onset_tested": [5, 5, 5],
                "n_onset_sig": [3, 4, 2],
                "n_during_tested": [5, 5, 5],
                "n_during_sig": [0, 0, 0],
            }
        )
        cells = pd.DataFrame(
            {"channel": ["s"], "celltype": ["penk"], "onset_z": [-2.0], "during_z": [np.nan]}
        )
        out = etbr.etb_within_group(cells, sess).set_index("test")
        assert out.loc["onset", "wilcoxon_p"] < 0.5
        assert out.loc["onset", "pooled_binom_p"] < 0.01
        assert np.isnan(out.loc["during", "wilcoxon_p"])  # all-zero medians
        assert np.isnan(out.loc["during", "median_cell_z"])

    def test_wide(self) -> None:
        cells = pd.DataFrame(
            {
                "exp_id": ["e", "e"],
                "animal_id": ["a", "a"],
                "celltype": ["penk", "penk"],
                "fibre": [np.nan, np.nan],
                "roi_idx": [0, 0],
                "channel": ["x", "y"],
                "onset_z": [1.0, 2.0],
            }
        )
        wide = etbr.etb_wide(cells, "onset")
        assert len(wide) == 1
        assert {"onset_z_x", "onset_z_y"} <= set(wide.columns)

    def test_session_etas_skip_no_events(self) -> None:
        out = etbr._session_etas(
            {"c": np.ones(200)}, [(np.array([], dtype=int), np.array([], dtype=int))], FPS
        )
        assert out == {}
