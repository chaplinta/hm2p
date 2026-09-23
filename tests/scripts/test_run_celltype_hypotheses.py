"""Tests for ``scripts/run_celltype_hypotheses.py`` with synthetic sessions (no S3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_celltype_hypotheses as rch  # noqa: E402

FPS = 9.6
N_FRAMES = 1200
N_ROIS = 5


def _synthetic_arrays(seed: int, n_rois: int = N_ROIS, n_frames: int = N_FRAMES) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    light_on = (t // 300) % 2 == 0  # 300-frame light/dark blocks -> 3 transitions
    active = (t // 40) % 3 != 0  # movement bouts of 80 frames, pauses of 40
    hd = np.mod(np.cumsum(rng.normal(0, 8, n_frames)), 360.0)
    ahv = np.gradient(np.unwrap(np.deg2rad(hd))) * FPS * 180 / np.pi
    speed = np.where(active, rng.uniform(2, 15, n_frames), rng.uniform(0, 0.4, n_frames))
    dff = rng.random((n_rois, n_frames)) * 0.1
    # cell 0: HD tuned, cell 1: movement onset transients, others noise
    dff[0] += np.exp(np.cos(np.deg2rad(hd - 90)) * 2) / 5
    onsets = np.where(np.diff(active.astype(int)) == 1)[0] + 1
    for o in onsets:
        dff[1, o : o + 5] += 1.0
    return {
        "dff": dff,
        "hd_deg": hd,
        "ahv_deg_s": ahv,
        "speed_cm_s": speed,
        "light_on": light_on,
        "active": active,
        "bad_behav": np.zeros(n_frames, dtype=bool),
        "mask": active.copy(),
        "x_mm": rng.uniform(0, 500, n_frames),
        "y_mm": rng.uniform(0, 500, n_frames),
        "syllable_id": rng.integers(0, 4, n_frames).astype(np.int16),
        "roi_types": np.zeros(n_rois, dtype=np.uint8),
        "roi_idx": np.arange(n_rois),
        "fps": FPS,
    }


def _meta_row(exp_id: str, animal: str, celltype: str, fibre: str = "SFB") -> pd.Series:
    return pd.Series(
        {
            "exp_id": exp_id,
            "animal_id": animal,
            "celltype": celltype,
            "fibre": fibre,
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
    return argparse.Namespace(
        signal="dff", n_perms=30, seed=0, features=None, gbm=False, glm_selection=False
    )


class TestHelpers:
    def test_signal_matrix_fallback(self) -> None:
        arr = _synthetic_arrays(0)
        assert rch._signal_matrix(arr, "spikes") is not None
        np.testing.assert_array_equal(rch._signal_matrix(arr, "spikes"), arr["dff"])
        arr["spikes"] = np.ones_like(arr["dff"])
        assert rch._signal_matrix(arr, "spikes").sum() == arr["dff"].size

    def test_per_animal(self) -> None:
        df = pd.DataFrame(
            {"animal_id": ["a", "a", "b"], "celltype": ["penk", "penk", "nonpenk"], "m": [1, 3, 5]}
        )
        out = rch._per_animal(df, ["m"])
        assert out.set_index("animal_id").loc["a", "m"] == 2

    def test_empty_result(self, tmp_path: Path) -> None:
        res = rch._empty_result("h4", tmp_path)
        assert res["n_sessions"] == 0
        assert (tmp_path / "summary.json").exists()

    def test_report_families_skips_absent(self, args: argparse.Namespace, tmp_path: Path) -> None:
        df = pd.DataFrame({"animal_id": ["a"], "celltype": ["penk"], "m": [1.0]})
        rep = rch._report_families(df, {"f": ["missing"]}, tmp_path, args)
        assert rep.empty
        assert (tmp_path / "between_group_report.csv").exists()


class TestH4:
    def test_runs_and_reports(self, sessions, args: argparse.Namespace, tmp_path: Path) -> None:
        res = rch.run_h4(args, sessions, tmp_path)
        assert res["n_sessions"] == 5
        assert res["n_cells"] == 5 * N_ROIS
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert {"onset_transient_index", "syllable_information_bits", "celltype"} <= set(cells)
        # cell 1 has onset transients in every session -> larger transient index than cell 0
        c1 = cells.loc[cells["roi_idx"] == 1, "onset_transient_index"]
        c0 = cells.loc[cells["roi_idx"] == 0, "onset_transient_index"]
        assert np.nanmedian(c1) > np.nanmedian(c0)
        rep = res["report"]
        assert "onset_transient_index" in set(rep["metric"])
        assert (tmp_path / "between_group_report.csv").exists()

    def test_empty(self, args: argparse.Namespace, tmp_path: Path) -> None:
        assert rch.run_h4(args, [], tmp_path)["n_sessions"] == 0


class TestH5:
    def test_runs_and_reports(self, sessions, args: argparse.Namespace, tmp_path: Path) -> None:
        res = rch.run_h5(args, sessions, tmp_path)
        assert res["n_sessions"] == 5
        cells = pd.read_csv(tmp_path / "cells.csv")
        sess = pd.read_csv(tmp_path / "sessions.csv")
        assert {"ltd_early_amplitude", "dtl_early_amplitude", "mvl_recovery_time_s"} <= set(cells)
        assert len(sess) == 5
        assert (sess["ltd_n_transitions"] >= 1).all()
        assert (tmp_path / "population_timecourses.json").exists()
        rep = res["report"]
        assert "ltd_early_amplitude" in set(rep["metric"])

    def test_empty(self, args: argparse.Namespace, tmp_path: Path) -> None:
        assert rch.run_h5(args, [], tmp_path)["n_sessions"] == 0


def test_registry_keys() -> None:
    assert {"h4", "h5"} <= set(rch.RUNNERS)


# ---------------------------------------------------------------------------
# H2 / H3 / H6 / H8 / H10
# ---------------------------------------------------------------------------


def _with_maze(arrays: dict[str, Any]) -> dict[str, Any]:
    """Add maze coordinates that walk along corridors of the Rosenberg maze."""
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    cells = np.array(maze.cell_list, dtype=float)
    n = arrays["dff"].shape[1]
    # random walk over accessible cells with small jitter
    rng = np.random.default_rng(1)
    idx = np.cumsum(rng.integers(0, 2, n)) % len(cells)
    xy = cells[idx] + rng.uniform(0.1, 0.9, (n, 2))
    arrays = dict(arrays)
    arrays["x_maze"] = xy[:, 0]
    arrays["y_maze"] = xy[:, 1]
    return arrays


class TestH2:
    def test_feature_table_and_report(self, sessions, args, tmp_path: Path) -> None:
        res = rch.run_h2(args, sessions, tmp_path)
        assert res["n_sessions"] == 5 and res["n_cells"] == 5 * N_ROIS
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert {"ev_event_rate", "tr_skewness", "tun_mvl", "celltype", "virus_id"} <= set(cells)
        assert (tmp_path / "virus_variant_kruskal.csv").exists()
        assert (tmp_path / "snr_confounds.csv").exists()
        rep = res["report"]
        assert "ev_event_rate" in set(rep["metric"])

    def test_virus_variant_check(self) -> None:
        df = pd.DataFrame(
            {
                "celltype": ["penk"] * 6 + ["nonpenk"] * 2,
                "animal_id": ["a", "a", "b", "b", "c", "c", "n", "n"],
                "virus_id": ["ADD3", "ADD3", "ADD3", "ADD3", "A122", "A122", "344", "344"],
                "m": [1, 2, 1, 2, 5, 6, 0, 0],
                "single": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        out = rch.virus_variant_check(df, ["m", "single", "absent"])
        assert set(out["metric"]) == {"m", "single"}
        assert out.set_index("metric").loc["m", "n_variants"] == 2

    def test_snr_confounds(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"ev_snr": rng.random(20), "m": rng.random(20)})
        out = rch.snr_confounds(df, ["m", "ev_snr", "absent"])
        assert len(out) == 1 and out.iloc[0]["metric"] == "m"

    def test_empty(self, args, tmp_path: Path) -> None:
        assert rch.run_h2(args, [], tmp_path)["n_sessions"] == 0


class TestH3:
    def test_runs(self, sessions, args, tmp_path: Path) -> None:
        res = rch.run_h3(args, sessions, tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert res["n_cells"] == len(cells)
        assert {"ahv_modulation_depth", "ahv_dark_minus_light_matched", "atd_best_lag_ms"} <= set(
            cells
        )
        assert cells["ahv_modulation_depth"].notna().any()
        assert "ahv_modulation_depth" in set(res["report"]["metric"])

    def test_ahv_depth_short_mask(self) -> None:
        arr = _synthetic_arrays(0)
        m = np.zeros(N_FRAMES, dtype=bool)
        assert np.isnan(rch._ahv_depth(arr["dff"][0], arr["ahv_deg_s"], m))

    def test_empty(self, args, tmp_path: Path) -> None:
        assert rch.run_h3(args, [], tmp_path)["n_sessions"] == 0


class TestH6:
    def test_from_features_file(self, sessions, args, tmp_path: Path) -> None:
        rch.run_h2(args, sessions, tmp_path / "h2")
        args.features = tmp_path / "h2" / "cells.csv"
        args.n_perms = 30
        res = rch.run_h6(args, [], tmp_path / "h6")
        assert res["n_cells"] > 0
        summ = res["summary"]
        for key in ("dispersion", "energy_distance", "gmm", "penk_like", "classifier_logistic"):
            assert key in summ
        assert 0.0 <= summ["classifier_logistic"]["balanced_accuracy"] <= 1.0
        assert (tmp_path / "h6" / "summary.json").exists()
        assert (tmp_path / "h6" / "importance_logistic.csv").exists()

    def test_builds_when_no_features(self, sessions, args, tmp_path: Path) -> None:
        args.n_perms = 10
        res = rch.run_h6(args, sessions[:4], tmp_path)
        assert res["n_cells"] > 0
        assert "classifier_gbm" not in res["summary"]

    def test_gbm_branch(self, sessions, args, tmp_path: Path) -> None:
        rch.run_h2(args, sessions[:4], tmp_path / "h2")
        args.features = tmp_path / "h2" / "cells.csv"
        args.n_perms = 5
        args.gbm = True
        res = rch.run_h6(args, [], tmp_path / "h6")
        assert "classifier_gbm" in res["summary"]
        assert (tmp_path / "h6" / "importance_gbm.csv").exists()

    def test_empty(self, args, tmp_path: Path) -> None:
        assert rch.run_h6(args, [], tmp_path)["n_sessions"] == 0


class TestH8:
    def test_runs(self, sessions, args, tmp_path: Path) -> None:
        res = rch.run_h8(args, sessions[:3], tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert {"full_dev_expl", "part_hd", "part_speed", "dominant"} <= set(cells)
        assert len(cells) == 3 * N_ROIS
        assert (tmp_path / "dominant_by_celltype.csv").exists()
        assert res["n_sessions"] == 3

    def test_selection_branch(self, sessions, args, tmp_path: Path) -> None:
        args.glm_selection = True
        small = [
            (row, {**arr, "dff": arr["dff"][:2], "roi_idx": arr["roi_idx"][:2]})
            for row, arr in sessions[:1]
        ]
        res = rch.run_h8(args, small, tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert res["n_cells"] == 2 and "selected" in cells.columns

    def test_glm_response_selection(self) -> None:
        arr = _synthetic_arrays(0)
        assert (rch._glm_response(arr, "dff") >= 0).all()
        arr["event_masks"] = np.zeros_like(arr["dff"], dtype=bool)
        assert rch._glm_response(arr, "dff").sum() == 0
        arr["spikes"] = -np.ones_like(arr["dff"])
        assert rch._glm_response(arr, "spikes").sum() == 0

    def test_empty(self, args, tmp_path: Path) -> None:
        assert rch.run_h8(args, [], tmp_path)["n_sessions"] == 0


class TestH10:
    def test_runs_with_maze(self, sessions, args, tmp_path: Path) -> None:
        maze_sessions = [(row, _with_maze(arr)) for row, arr in sessions]
        res = rch.run_h10(args, maze_sessions, tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        sess = pd.read_csv(tmp_path / "sessions.csv")
        assert {"place_skaggs_info", "syllable_information_bits", "familiarity_spearman_r"} <= set(
            cells
        )
        assert len(sess) == 5
        assert res["n_cells"] == len(cells)
        assert (tmp_path / "between_group_report.csv").exists()

    def test_runs_without_maze(self, sessions, args, tmp_path: Path) -> None:
        res = rch.run_h10(args, sessions[:3], tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert "familiarity_spearman_r" not in cells.columns
        assert res["n_sessions"] == 3

    def test_empty(self, args, tmp_path: Path) -> None:
        assert rch.run_h10(args, [], tmp_path)["n_sessions"] == 0


def test_registry_complete_so_far() -> None:
    assert {"h2", "h3", "h4", "h5", "h6", "h8", "h10"} <= set(rch.RUNNERS)


class TestH7:
    def test_runs(self, sessions, args, tmp_path: Path) -> None:
        args.n_cells_matched = 3
        res = rch.run_h7(args, sessions, tmp_path)
        sess = pd.read_csv(tmp_path / "sessions.csv")
        assert len(sess) == 5 and res["n_sessions"] == 5
        assert {
            "ring_angle_hd_corr_all",
            "persistence_ring_score",
            "matched_decode_error_median",
        } <= set(sess)
        assert sess["matched_decode_error_median"].notna().all()
        assert "ring_angle_hd_corr_all" in set(res["report"]["metric"])

    def test_skips_tiny_sessions(self, args, tmp_path: Path) -> None:
        small = [(_meta_row("x", "p1", "penk"), _synthetic_arrays(0, n_rois=2))]
        assert rch.run_h7(args, small, tmp_path)["n_sessions"] == 0


class TestH9:
    def test_runs(self, sessions, args, tmp_path: Path) -> None:
        res = rch.run_h9(args, sessions, tmp_path)
        cells = pd.read_csv(tmp_path / "cells.csv")
        assert len(cells) == 5 * N_ROIS and res["n_cells"] == len(cells)
        assert {"pop_coupling_all", "xcorr_asymmetry", "roi_idx"} <= set(cells)
        assert "pop_coupling_all" in set(res["report"]["metric"])

    def test_empty(self, args, tmp_path: Path) -> None:
        assert rch.run_h9(args, [], tmp_path)["n_sessions"] == 0


def test_registry_complete() -> None:
    assert set(rch.RUNNERS) == {f"h{i}" for i in range(2, 11)}
