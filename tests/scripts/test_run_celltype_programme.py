"""Tests for ``scripts/run_celltype_programme.py`` shared helpers (no S3)."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_celltype_programme as rcp  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic metadata
# ---------------------------------------------------------------------------


def _write_metadata(base: Path) -> None:
    animals = pd.DataFrame(
        {
            "animal_id": ["1001", "1002", "2001", "2002"],
            "celltype": ["penk", "penk", "nonpenk", "nonpenk"],
            "virus_id": ["ADD3", "ADD3.1", "344", "344"],
            "gcamp": ["7f", "7f", "7f", "7f"],
        }
    )
    exps = pd.DataFrame(
        {
            "exp_id": [
                "20220101_10_00_00_1001",
                "20220102_10_00_00_1002",
                "20220103_10_00_00_2001",
                "20220104_10_00_00_2002",
                "20220105_10_00_00_2002",
            ],
            "fibre": ["SFB", "TFB", "TFB", "SFB", "SFB"],
            "lens": ["f4mm", "f6mm", "f6mm", "f4mm", "f4mm"],
            "exclude": [0, 0, 0, 0, 1],
            "primary_exp": [1, 1, 1, 1, 0],
        }
    )
    animals.to_csv(base / "animals.csv", index=False)
    exps.to_csv(base / "experiments.csv", index=False)


@pytest.fixture
def meta(tmp_path: Path) -> pd.DataFrame:
    _write_metadata(tmp_path)
    return rcp.load_metadata(tmp_path)


class TestMetadata:
    def test_join_and_columns(self, meta: pd.DataFrame) -> None:
        assert len(meta) == 5
        assert set(meta.columns) >= {"exp_id", "animal_id", "celltype", "fibre", "lens"}
        assert meta.loc[meta["exp_id"].str.endswith("1001"), "celltype"].item() == "penk"
        assert meta["exclude"].dtype.kind == "i"

    def test_select_sessions_filters(self, meta: pd.DataFrame) -> None:
        sel = rcp.select_sessions(meta)
        assert len(sel) == 4
        sel_all = rcp.select_sessions(meta, include_excluded=True)
        assert len(sel_all) == 5
        prim = rcp.select_sessions(meta, primary_only=True)
        assert (prim["primary_exp"] == 1).all()

    def test_select_sessions_limit_keeps_both_types(self, meta: pd.DataFrame) -> None:
        sel = rcp.select_sessions(meta, limit=2)
        assert len(sel) == 2
        assert set(sel["celltype"]) == {"penk", "nonpenk"}

    def test_sync_key(self) -> None:
        key = rcp.sync_key("20220804_13_52_02_1117646", "1117646")
        assert key == "sync/sub-1117646/ses-20220804T135202/sync.h5"


# ---------------------------------------------------------------------------
# Session arrays
# ---------------------------------------------------------------------------


def _write_sync(path: Path, n_rois: int = 4, n_frames: int = 200, status: str = "OK") -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("dff", data=rng.random((n_rois, n_frames)).astype(np.float32))
        f.create_dataset("hd_deg", data=rng.uniform(0, 360, n_frames + 3).astype(np.float32))
        f.create_dataset("speed_cm_s", data=rng.random(n_frames).astype(np.float32))
        f.create_dataset("light_on", data=(np.arange(n_frames) % 100 < 50).astype(np.uint8))
        f.create_dataset("active", data=np.ones(n_frames, dtype=np.uint8))
        f.create_dataset("bad_behav", data=np.zeros(n_frames, dtype=np.uint8))
        f.create_dataset("roi_types", data=np.array([0, 0, 1, 2], dtype=np.uint8)[:n_rois])
        f.create_dataset("frame_times", data=np.arange(n_frames) / 9.6)
        f.attrs["sync_status"] = status
        f.attrs["fps_imaging"] = 9.6


class TestReadSessionArrays:
    def test_soma_only_and_truncation(self, tmp_path: Path) -> None:
        p = tmp_path / "sync.h5"
        _write_sync(p)
        with h5py.File(p, "r") as f:
            arr = rcp.read_session_arrays(f, soma_only=True)
        assert arr is not None
        assert arr["dff"].shape == (2, 200)
        assert arr["hd_deg"].shape == (200,)
        assert arr["ahv_deg_s"].shape == (200,)
        assert arr["mask"].dtype == bool and arr["mask"].all()
        assert arr["fps"] == pytest.approx(9.6)
        assert arr["roi_idx"].tolist() == [0, 1]

    def test_all_rois(self, tmp_path: Path) -> None:
        p = tmp_path / "sync.h5"
        _write_sync(p)
        with h5py.File(p, "r") as f:
            arr = rcp.read_session_arrays(f, soma_only=False)
        assert arr is not None and arr["dff"].shape[0] == 4

    def test_failed_sync_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "sync.h5"
        _write_sync(p, status="FAILED_NO_PULSES")
        with h5py.File(p, "r") as f:
            assert rcp.read_session_arrays(f) is None

    def test_missing_dataset_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "sync.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("dff", data=np.zeros((2, 10)))
        with h5py.File(p, "r") as f:
            assert rcp.read_session_arrays(f) is None

    def test_no_soma_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "sync.h5"
        _write_sync(p, n_rois=2)
        with h5py.File(p, "r+") as f:
            f["roi_types"][:] = 1
        with h5py.File(p, "r") as f:
            assert rcp.read_session_arrays(f, soma_only=True) is None


def test_attach_session_meta(meta: pd.DataFrame) -> None:
    df = pd.DataFrame({"roi_idx": [0, 1], "x": [1.0, 2.0]})
    out = rcp.attach_session_meta(df, meta.iloc[0])
    assert out["celltype"].tolist() == ["penk", "penk"]
    assert out["fibre"].tolist() == ["SFB", "SFB"]
    assert "x" in out.columns and len(out) == 2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cells(effect: float, noise: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    animals = {"penk": ["p1", "p2", "p3", "p4"], "nonpenk": ["n1", "n2", "n3"]}
    for ct, ids in animals.items():
        for a in ids:
            fibre, lens = ("SFB", "f4mm") if a in ("p1", "p2", "n1") else ("TFB", "f6mm")
            for _ in range(6):
                base = effect if ct == "penk" else 0.0
                rows.append(
                    {
                        "animal_id": a,
                        "celltype": ct,
                        "fibre": fibre,
                        "lens": lens,
                        "m1": base + rng.normal(0, noise),
                        "m2": rng.normal(0, noise),
                    }
                )
    return pd.DataFrame(rows)


class TestReporting:
    def test_naive_cell_level(self) -> None:
        df = _cells(1.0)
        res = rcp.naive_cell_level(df, "m1")
        assert res["naive_p_value"] < 0.01
        assert res["naive_n"] == len(df)

    def test_naive_single_group(self) -> None:
        df = _cells(1.0)
        res = rcp.naive_cell_level(df[df["celltype"] == "penk"], "m1")
        assert np.isnan(res["naive_p_value"])

    def test_between_group_report_columns(self) -> None:
        df = _cells(1.0)
        rep = rcp.between_group_report(df, ["m1", "m2"], family="test", n_perms=300)
        assert len(rep) == 2
        for col in (
            "summary_p_value",
            "perm_p_value",
            "naive_p_value",
            "lmm_available",
            "loao_direction_stable",
            "matched_summary_p",
            "p_fdr",
            "family",
        ):
            assert col in rep.columns, col
        m1 = rep.set_index("metric").loc["m1"]
        # 4 vs 3 animals: the two-sided Mann-Whitney floor is 2/35 = 0.057
        assert m1["summary_p_value"] == pytest.approx(2 / 35)
        assert bool(m1["loao_direction_stable"]) is True

    def test_between_group_report_skips_insufficient(self) -> None:
        df = _cells(1.0)
        df = df[df["animal_id"].isin(["p1", "n1"])]
        rep = rcp.between_group_report(df, ["m1"], family="test", n_perms=50)
        assert rep.loc[0, "skipped"] == "insufficient groups"

    def test_write_outputs(self, tmp_path: Path) -> None:
        p_csv = rcp.write_outputs(tmp_path / "x", "table", pd.DataFrame({"a": [1]}))
        p_json = rcp.write_outputs(
            tmp_path / "x",
            "summary",
            {"arr": np.arange(3), "f": np.float64(1.5), "b": np.bool_(True), "p": Path("q")},
        )
        assert p_csv.exists() and p_json.exists()
        assert '"arr": [\n    0,' in p_json.read_text() or '"arr": [0, 1, 2]' in p_json.read_text()

    def test_json_default(self) -> None:
        assert rcp._json_default(np.int64(3)) == 3
        assert rcp._json_default(np.bool_(False)) is False
        assert rcp._json_default(Path("a")) == "a"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_parser_defaults(self) -> None:
        args = rcp._build_arg_parser().parse_args(["h2"])
        assert args.hypothesis == "h2"
        assert args.sessions is None
        assert args.dry_run is False
        assert args.n_perms == 10000

    def test_parser_rejects_unknown(self) -> None:
        with pytest.raises(SystemExit):
            rcp._build_arg_parser().parse_args(["h99"])

    def test_dry_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_metadata(tmp_path)
        original = rcp.load_metadata
        monkeypatch.setattr(rcp, "load_metadata", lambda base=None: original(tmp_path))
        args = rcp._build_arg_parser().parse_args(["h5", "--dry-run", "--sessions", "2"])
        summary = rcp.dry_run(args)
        assert summary["hypothesis"] == "h5"
        assert summary["n_sessions"] == 2
        assert set(summary["sessions_by_celltype"]) == {"penk", "nonpenk"}
        assert not (tmp_path / "results").exists()

    def test_hypotheses_registry(self) -> None:
        assert set(rcp.HYPOTHESES) == {f"h{i}" for i in range(2, 11)}
