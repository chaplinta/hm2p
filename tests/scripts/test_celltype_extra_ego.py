"""Tests for ``scripts/celltype_extra_ego.py`` with synthetic sessions (no S3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_ego as cee  # noqa: E402

from hm2p.analysis import egocentric as ego  # noqa: E402
from hm2p.maze.topology import build_rose_maze  # noqa: E402

FPS = 9.6
N_FRAMES = 1200
N_ROIS = 3
MM_PER_UNIT = 100.0


def _maze_walk(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positions moving between adjacent q-rose maze cells (maze units)."""
    maze = build_rose_maze()
    cell = maze.cell_list[0]
    xs, ys = np.empty(n), np.empty(n)
    for i in range(n):
        if i % 5 == 0:
            cell = maze.adj[cell][rng.integers(len(maze.adj[cell]))]
        xs[i] = cell[0] + rng.uniform(0.15, 0.85)
        ys[i] = cell[1] + rng.uniform(0.15, 0.85)
    return xs, ys


def _synthetic_arrays(seed: int, n_frames: int = N_FRAMES, positions: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    hd = np.mod(np.cumsum(rng.normal(0, 25, n_frames)), 360.0)
    heading = ego.hd_to_frame_heading(hd)
    active = (t // 40) % 3 != 0
    dff = rng.normal(0, 0.1, (N_ROIS, n_frames))
    out: dict[str, Any] = {
        "dff": dff,
        "hd_deg": hd,
        "ahv_deg_s": np.gradient(np.unwrap(np.deg2rad(hd))) * FPS * 180 / np.pi,
        "speed_cm_s": rng.uniform(2, 10, n_frames),
        "light_on": (t // 300) % 2 == 0,
        "active": active,
        "bad_behav": np.zeros(n_frames, dtype=bool),
        "mask": active.copy(),
        "roi_idx": np.arange(N_ROIS) * 2,
        "fps": FPS,
    }
    if not positions:
        return out
    x, y = _maze_walk(n_frames, rng)
    walls = ego.maze_wall_segments()
    d90 = ego.ray_wall_distances(x, y, heading, walls, [90.0])[:, 0]
    dff[0] += (d90 < 0.5).astype(float)  # cell 0: egocentric boundary at +90 deg
    hba = rng.uniform(-60, 60, n_frames)
    body_dir = heading - hba
    dff[1] += hba / 60.0  # cell 1: head-body angle
    out.update(
        {
            "x_maze": x,
            "y_maze": y,
            "x_body_mm": x * MM_PER_UNIT,
            "y_body_mm": y * MM_PER_UNIT,
            "x_head_mm": x * MM_PER_UNIT + 30.0 * np.cos(np.deg2rad(body_dir)),
            "y_head_mm": y * MM_PER_UNIT + 30.0 * np.sin(np.deg2rad(body_dir)),
        }
    )
    return out


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


def _args(**kw: Any) -> argparse.Namespace:
    base = {"signal": "dff", "seed": 0, "n_perms": 50, "n_shuffles_evt": 20}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def sessions() -> list[tuple[pd.Series, dict[str, Any]]]:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "penk"), ("n1", "nonpenk"), ("n2", "nonpenk")]
    specs += [("n3", "nonpenk")]
    return [
        (_meta_row(f"2022010{i}_10_00_00_{a}", a, ct), _synthetic_arrays(i))
        for i, (a, ct) in enumerate(specs)
    ]


def test_module_constants() -> None:
    assert cee.RUNNER_KEY == "ego"
    assert cee.RUNNER is cee.run_ego
    assert cee.EXTRA_SYNC_KEYS == ["x_head_mm", "y_head_mm", "x_body_mm", "y_body_mm"]


class TestRunEgo:
    def test_outputs_and_recovery(self, sessions: list, tmp_path: Path) -> None:
        res = cee.run_ego(_args(), sessions, tmp_path)
        assert res["n_sessions"] == 6 and res["n_cells"] == 6 * N_ROIS
        for name in ("cells", "sessions", "within_group", "between_group_report"):
            assert (tmp_path / f"{name}.csv").exists()
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert set(cells["roi_idx"]) == {0, 2, 4}
        ebc = cells[cells["roi_idx"] == 0]
        assert (ebc["ebc_p"] < 0.05).mean() >= 0.8
        dev = np.abs(ego.wrap_deg(ebc["ebc_pref_angle"].to_numpy() - 90.0))
        assert np.median(dev) < 30.0
        hba = cells[cells["roi_idx"] == 2]
        assert (hba["hba_p"] < 0.05).all() and (hba["hba_slope_sign"] == 1).all()
        noise = cells[cells["roi_idx"] == 4]
        assert (noise["ebc_p"] < 0.05).mean() <= 0.5
        assert cells["ebc_mrl_light"].notna().all() and cells["ebc_mrl_dark"].notna().all()
        np.testing.assert_allclose(
            cells["ebc_mrl_dark_minus_light"], cells["ebc_mrl_dark"] - cells["ebc_mrl_light"]
        )
        assert set(res["report"]["family"]) == {"ego"}
        within = res["within"]
        assert set(within["measure"]) == {"ebc", "hba", "wall", "ebc_dark_minus_light"}
        assert set(within["celltype"]) == {"penk", "nonpenk"}
        sess = pd.read_csv(tmp_path / "sessions.csv")
        assert {"frac_ebc_sig", "n_hba_tested", "n_valid_frames"} <= set(sess.columns)

    def test_missing_positions(self, tmp_path: Path) -> None:
        arr = _synthetic_arrays(0, positions=False)
        res = cee.run_ego(_args(), [(_meta_row("e1", "a1", "penk"), arr)], tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert res["n_cells"] == N_ROIS
        assert cells[["ebc_mrl", "hba_modulation", "wall_rho"]].isna().all().all()

    def test_spikes_signal_falls_back_and_default_shuffles(self, tmp_path: Path) -> None:
        arr = _synthetic_arrays(1, n_frames=500)
        args = argparse.Namespace(signal="spikes", seed=0, n_perms=10)
        res = cee.run_ego(args, [(_meta_row("e1", "a1", "penk"), arr)], tmp_path)
        assert res["n_cells"] == N_ROIS

    def test_empty(self, tmp_path: Path) -> None:
        assert cee.run_ego(_args(), [], tmp_path)["n_sessions"] == 0
        assert (tmp_path / "summary.json").exists()


class TestHelpers:
    def test_get(self) -> None:
        assert cee._get({}, "x", 3) is None
        assert cee._get({"x": np.arange(2)}, "x", 3) is None
        np.testing.assert_array_equal(cee._get({"x": np.arange(5)}, "x", 3), [0, 1, 2])

    def test_session_geometry_truncates(self) -> None:
        arr = _synthetic_arrays(2, n_frames=300)
        arr["hd_deg"] = np.concatenate([arr["hd_deg"], [0.0, 0.0]])
        geo = cee.session_geometry(arr, ego.maze_wall_segments())
        assert geo["n_frames"] == 300 and geo["ray_dist"].shape == (300, cee.N_ANGLE_BINS)
        assert geo["hba"].shape == (300,)

    def test_short_conditions_skip_light_dark(self) -> None:
        arr = _synthetic_arrays(3, n_frames=400)
        arr["light_on"] = np.ones(400, dtype=bool)
        geo = cee.session_geometry(arr, ego.maze_wall_segments())
        rec = cee.cell_ego_measures(
            arr["dff"][0], geo, ego.maze_wall_segments(), FPS, 5, np.random.default_rng(0)
        )
        assert np.isfinite(rec["ebc_mrl_light"]) and np.isnan(rec["ebc_mrl_dark"])
        assert np.isnan(rec["ebc_mrl_dark_minus_light"])
        assert isinstance(rec["ebc_sig"], bool)

    def test_session_summary(self) -> None:
        cells = pd.DataFrame(
            {"ebc_p": [0.01, 0.5, np.nan], "hba_p": [0.01, 0.02, 0.9], "wall_p": np.nan}
        )
        s = cee.session_summary(cells)
        assert s["n_ebc_tested"] == 2 and s["n_ebc_sig"] == 1 and s["frac_ebc_sig"] == 0.5
        assert s["frac_hba_sig"] == pytest.approx(2 / 3)
        assert s["n_wall_tested"] == 0 and np.isnan(s["frac_wall_sig"])
        assert np.isnan(cee.session_summary(pd.DataFrame())["frac_ebc_sig"])

    def test_within_group(self) -> None:
        rng = np.random.default_rng(0)
        n = 24
        cells = pd.DataFrame(
            {
                "celltype": ["penk"] * 12 + ["nonpenk"] * 12,
                "animal_id": np.repeat(["a", "b", "c", "d", "e", "f"], 4),
                "ebc_p": np.r_[np.full(12, 0.001), rng.uniform(0.2, 1, 12)],
                "hba_p": np.nan,
                "wall_p": rng.uniform(0, 1, n),
                "ebc_mrl_dark_minus_light": rng.normal(0, 0.1, n),
            }
        )
        out = cee.ego_within_group(cells)
        penk = out[(out["celltype"] == "penk") & (out["measure"] == "ebc")].iloc[0]
        assert penk["pooled_frac_sig"] == 1.0 and penk["pooled_binom_p"] < 1e-6
        assert penk["n_animals"] == 3 and np.isfinite(penk["wilcoxon_p"])
        hba = out[out["measure"] == "hba"]
        assert (hba["n_cells"] == 0).all() and hba["wilcoxon_p"].isna().all()
        dl = out[out["measure"] == "ebc_dark_minus_light"]
        assert len(dl) == 2 and dl["wilcoxon_p"].notna().all()

    def test_within_group_empty(self) -> None:
        assert cee.ego_within_group(pd.DataFrame()).empty
