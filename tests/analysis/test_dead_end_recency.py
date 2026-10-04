"""Tests for ``hm2p.analysis.dead_end_recency`` (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import dead_end_recency as r


def test_maze_visits_drops_short_and_merges() -> None:
    seq = np.array([0, 0, 0, 1, 0, 0, 0, 2, 2, 2, -1, -1, -1])
    assert r.maze_visits(seq, min_frames=3) == [(0, 0, 7), (2, 7, 10)]
    assert r.maze_visits(np.array([], dtype=int)) == []


def test_dead_end_entries_recency_and_prev_light() -> None:
    visits = [(5, 0, 10), (1, 10, 20), (5, 20, 30), (6, 30, 40), (5, 40, 50)]
    light = np.r_[np.ones(15), np.zeros(35)]
    e = r.dead_end_entries(visits, {5, 6}, light, fps=10.0)
    assert e["frame"].tolist() == [20, 40]
    assert e["recency_s"].tolist() == [1.0, 1.0]
    assert e["prev_light"].tolist() == [True, False]
    assert r.dead_end_entries([], {5}, light, 10.0).empty


def test_partial_spearman_removes_covariate() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = z + 0.1 * rng.normal(size=300)
    y = z + 0.1 * rng.normal(size=300)
    assert r.partial_spearman(y, x, np.zeros((300, 0))) > 0.9
    assert abs(r.partial_spearman(y, x, z[:, None])) < 0.2
    assert np.isnan(r.partial_spearman(np.ones(10), np.arange(10.0), np.zeros((10, 0))))


@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=0, max_value=10_000))
def test_rank_residual_orthogonal_to_covariates(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=40)
    X = np.column_stack([rng.normal(size=40), (rng.random(40) > 0.5).astype(float)])
    res = r.rank_residual(y, X)
    assert abs(res.sum()) < 1e-8


def _session(effect: float, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    fps, n = 10.0, 30_000
    frames = np.sort(rng.choice(np.arange(100, n - 100), 160, replace=False))
    rec = np.exp(rng.uniform(1, 5, frames.size))
    ent = pd.DataFrame(
        {
            "frame": frames,
            "dead_end": rng.integers(0, 3, frames.size),
            "recency_s": rec,
            "prev_light": rng.random(frames.size) > 0.5,
        }
    )
    light = (np.arange(n) // 600) % 2 == 0  # 60 s epochs
    sig = rng.poisson(0.2, (4, n)).astype(float)
    w = 10
    for f, rc in zip(frames, rec, strict=True):
        if not light[f]:
            sig[0, f : f + w] += effect * np.log(rc)
    cov = r.entry_covariates(
        ent, rng.normal(5, 1, n), rng.normal(0, 50, n), light.astype(float), fps, w
    )
    return sig, ent, cov, w


def test_recency_cell_table_detects_dark_effect() -> None:
    sig, ent, cov, w = _session(effect=1.0)
    t = r.recency_cell_table(sig, ent, cov, w, n_shift=60, min_shift=200)
    dark0 = t[(t.cond == "dark") & (t.roi == 0)].iloc[0]
    light0 = t[(t.cond == "light") & (t.roi == 0)].iloc[0]
    assert dark0.rho > 0.4 and dark0.p < 0.05 and dark0.z > 2
    assert abs(light0.rho) < 0.3
    assert set(t.cond) <= set(r.CONDITIONS)


def test_entry_covariates_and_condition_masks() -> None:
    ent = pd.DataFrame(
        {
            "frame": [20, 50],
            "dead_end": [1, 2],
            "recency_s": [5.0, 9.0],
            "prev_light": [True, False],
        }
    )
    light = np.r_[np.ones(40), np.zeros(40)]
    cov = r.entry_covariates(ent, np.arange(80.0), np.full(80, -30.0), light, fps=10.0, w=5)
    assert cov.loc[0, "speed_pre"] == pytest.approx(17.0) and cov.loc[
        1, "speed_post"
    ] == pytest.approx(52.0)
    assert cov.loc[0, "abs_ahv_post"] == 30.0 and cov.loc[1, "since_switch_s"] == pytest.approx(
        1.0
    )
    m = r.condition_masks(cov.light_frac.to_numpy(), ent.prev_light.to_numpy())
    assert m["light"].tolist() == [True, False] and m["dark_prev_dark"].tolist() == [False, True]


def test_summarise_contrasts() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for a in range(6):
        for roi in range(10):
            for cond, mu in (("dark", 0.5), ("light", -0.2)):
                z = rng.normal(mu, 0.5)
                rows.append(
                    {
                        "cond": cond,
                        "roi": roi,
                        "z": z,
                        "rho": z / 10,
                        "p": 0.5,
                        "exp_id": f"s{a}",
                        "animal_id": str(a),
                    }
                )
    s = r.summarise(pd.DataFrame(rows)).set_index("contrast")
    assert s.loc["dark", "median_z"] > 0.3 and s.loc["dark", "cell_p"] < 0.001
    assert (
        s.loc["dark_minus_light", "animal_p"] < 0.05
        and s.loc["dark_minus_light", "n_animals"] == 6
    )
