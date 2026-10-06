"""Tests for hm2p.extraction.roi_training_data (synthetic legacy session folders)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.extraction.roi_training_data import load_all_sessions, load_session_data


def _disk(cy: int, cx: int, r: int = 3) -> dict:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    m = yy**2 + xx**2 <= r**2
    n = int(m.sum())
    return {
        "ypix": yy[m] + cy,
        "xpix": xx[m] + cx,
        "med": [cy, cx],
        "npix": n,
        "lam": np.ones(n),
        "radius": float(r),
        "compact": 1.0,
        "aspect_ratio": 1.0,
        "skew": 1.0,
        "std": 1.0,
        "npix_norm": 1.0,
        "overlap": np.zeros(n, bool),
        "soma_crop": np.ones(n, bool),
    }


def _session(root: Path, name: str, soma: list[int], dend: list[int], n: int = 4) -> None:
    rng = np.random.default_rng(0)
    for run, acc in (("suite2p_soma", soma), ("suite2p_dend", dend)):
        d = root / name / run / "plane0"
        d.mkdir(parents=True)
        ic = np.zeros((n, 2))
        ic[acc, 0] = 1
        np.save(d / "iscell.npy", ic)
        np.save(d / "stat.npy", np.array([_disk(10 + 12 * i, 20) for i in range(n)], dtype=object))
        F = (100 + rng.normal(0, 5, (n, 400))).astype(np.float32)
        np.save(d / "F.npy", F)
        np.save(d / "Fneu.npy", F * 0.5)
        np.save(d / "ops.npy", np.array({"fs": 9.7}, dtype=object))


def test_labels_and_features(tmp_path):
    _session(tmp_path, "s1", soma=[0], dend=[2])
    d = load_session_data(tmp_path / "s1")
    assert d["labels"].tolist() == [1, 0, 2, 0]
    assert d["features"].shape == (4, 26)
    assert (d["n_soma"], d["n_dend"], d["n_artefact"]) == (1, 1, 2)


def test_unusable_sessions_skipped(tmp_path):
    _session(tmp_path, "overlap", soma=[0], dend=[0])
    _session(tmp_path, "nolabels", soma=[], dend=[])
    (tmp_path / "incomplete").mkdir()
    assert load_session_data(tmp_path / "overlap") is None
    assert load_session_data(tmp_path / "nolabels") is None
    assert load_session_data(tmp_path / "incomplete") is None
    with pytest.raises(FileNotFoundError):
        load_all_sessions(tmp_path)


def test_load_all_sessions_groups(tmp_path):
    _session(tmp_path, "a", soma=[0], dend=[1])
    _session(tmp_path, "b", soma=[3], dend=[])
    X, y, g = load_all_sessions(tmp_path)
    assert X.shape == (8, 26) and y.size == 8
    assert sorted(set(g)) == ["a", "b"]
