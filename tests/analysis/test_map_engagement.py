"""Tests for map-engagement (population-vector consistency). Synthetic only."""

from __future__ import annotations

import numpy as np

from hm2p.analysis.map_engagement import (
    consistency_debiased,
    extract_visit_vectors,
    mean_pairwise_corr,
)

# ---------------------------------------------------------------------------
# extract_visit_vectors
# ---------------------------------------------------------------------------


def test_extract_collapses_contiguous_runs():
    # 2 cells, frames: cell 0 (x2), cell 1 (x3), cell 0 (x1)
    cell_idx = np.array([0, 0, 1, 1, 1, 0])
    signal = np.array([[1.0, 3.0, 10.0, 10.0, 10.0, 5.0]])  # 1 cell
    valid = np.ones(6, bool)
    cells, vecs = extract_visit_vectors(signal, cell_idx, valid)
    assert list(cells) == [0, 1, 0]
    assert np.allclose(vecs[:, 0], [2.0, 10.0, 5.0])


def test_extract_drops_invalid_frames():
    cell_idx = np.array([0, 0, 1, 1])
    signal = np.array([[1.0, 9.0, 4.0, 6.0]])
    valid = np.array([True, False, True, True])
    cells, vecs = extract_visit_vectors(signal, cell_idx, valid)
    # frame 1 invalid -> visit 0 is just frame 0
    assert list(cells) == [0, 1]
    assert np.isclose(vecs[0, 0], 1.0)
    assert np.isclose(vecs[1, 0], 5.0)


def test_extract_empty():
    cells, vecs = extract_visit_vectors(
        np.zeros((3, 0)), np.array([], dtype=int), np.array([], dtype=bool)
    )
    assert cells.size == 0
    assert vecs.shape == (0, 3)


# ---------------------------------------------------------------------------
# mean_pairwise_corr
# ---------------------------------------------------------------------------


def test_pairwise_identical_is_one():
    v = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (4, 1))
    assert np.isclose(mean_pairwise_corr(v), 1.0)


def test_pairwise_single_row_nan():
    assert np.isnan(mean_pairwise_corr(np.array([[1.0, 2.0, 3.0]])))


# ---------------------------------------------------------------------------
# consistency_debiased
# ---------------------------------------------------------------------------


def _build_visits(n_cells_pop, n_maze_cells, visits_per_cell, place_strength, rng):
    """Each maze cell has a fixed population template; visits = template + noise.

    place_strength scales the template vs noise. 0 -> activity independent of cell.
    """
    cells, vecs = [], []
    templates = rng.normal(size=(n_maze_cells, n_cells_pop))
    for c in range(n_maze_cells):
        for _ in range(visits_per_cell):
            v = place_strength * templates[c] + rng.normal(size=n_cells_pop)
            cells.append(c)
            vecs.append(v)
    return np.array(cells), np.vstack(vecs)


def test_consistency_positive_when_place_locked():
    rng = np.random.default_rng(0)
    cells, vecs = _build_visits(20, 8, 5, place_strength=3.0, rng=rng)
    out = consistency_debiased(cells, vecs, k_visits=3, n_boot=30, rng=rng)
    # same cell -> same template -> within >> across
    assert out["debiased"] > 0.2
    assert out["within"] > out["across"]


def test_consistency_near_zero_when_independent():
    rng = np.random.default_rng(1)
    cells, vecs = _build_visits(20, 8, 5, place_strength=0.0, rng=rng)
    out = consistency_debiased(cells, vecs, k_visits=3, n_boot=30, rng=rng)
    # activity independent of cell -> debiased near zero
    assert abs(out["debiased"]) < 0.15


def test_consistency_respects_cell_cap_and_eligibility():
    rng = np.random.default_rng(2)
    cells, vecs = _build_visits(10, 6, 4, place_strength=2.0, rng=rng)
    out = consistency_debiased(cells, vecs, k_visits=3, n_cells_cap=3, n_boot=10, rng=rng)
    assert out["n_eligible_cells"] == 6
    assert out["n_used_cells"] == 3


def test_consistency_too_few_cells_returns_nan():
    rng = np.random.default_rng(3)
    # only 1 cell has >= k visits
    cells = np.array([0, 0, 0, 1])
    vecs = rng.normal(size=(4, 5))
    out = consistency_debiased(cells, vecs, k_visits=3, rng=rng)
    assert np.isnan(out["debiased"])
    assert out["n_used_cells"] < 2
