"""Tests for analysis cache (DuckDB)."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.cache import (
    cache_session_results,
    clear_cache,
    get_celltype_breakdown,
    get_connection,
    get_summary_stats,
    is_session_cached,
    load_all_cells,
    load_all_sessions,
    load_tuning_curve,
)


@pytest.fixture
def db(tmp_path):
    """Create an in-memory DuckDB connection with schema."""
    conn = get_connection(tmp_path / "test.duckdb")
    yield conn
    conn.close()


def _sample_cell_results(n=5, n_hd=3):
    cells = []
    for i in range(n):
        cells.append({
            "cell": i,
            "is_hd": i < n_hd,
            "grade": "A" if i < n_hd else "D",
            "mvl": 0.4 + 0.1 * i if i < n_hd else 0.05,
            "p_value": 0.001 if i < n_hd else 0.5,
            "reliability": 0.8 if i < n_hd else 0.2,
            "mi": 0.5 if i < n_hd else 0.05,
            "preferred_direction": 60.0 * i,
            "gain_index": 0.1,
            "smi": 0.05,
        })
    return cells


def _sample_summary(n_rois=5, n_hd=3, n_frames=3000):
    return {
        "n_rois": n_rois,
        "n_frames": n_frames,
        "n_hd": n_hd,
        "n_non_hd": n_rois - n_hd,
        "fraction_hd": n_hd / n_rois,
        "mean_mvl": 0.3,
        "mean_gain_index": 0.1,
        "mean_smi": 0.05,
    }


class TestCacheBasics:
    def test_cache_and_retrieve(self, db):
        cells = _sample_cell_results()
        summary = _sample_summary()
        cache_session_results(db, "ses1", "animal1", "penk", cells, summary)

        assert is_session_cached(db, "ses1")
        assert not is_session_cached(db, "ses2")

    def test_load_all_cells(self, db):
        cells = _sample_cell_results(5, 3)
        cache_session_results(db, "ses1", "a1", "penk", cells, _sample_summary())
        cache_session_results(db, "ses2", "a2", "nonpenk",
                              _sample_cell_results(3, 1),
                              _sample_summary(3, 1))

        all_cells = load_all_cells(db)
        assert len(all_cells) == 8  # 5 + 3

    def test_filter_by_celltype(self, db):
        cache_session_results(db, "ses1", "a1", "penk",
                              _sample_cell_results(4, 2), _sample_summary(4, 2))
        cache_session_results(db, "ses2", "a2", "nonpenk",
                              _sample_cell_results(3, 1), _sample_summary(3, 1))

        penk = load_all_cells(db, celltype="penk")
        assert len(penk) == 4
        nonpenk = load_all_cells(db, celltype="nonpenk")
        assert len(nonpenk) == 3

    def test_filter_by_animal(self, db):
        cache_session_results(db, "ses1", "a1", "penk",
                              _sample_cell_results(4, 2), _sample_summary(4, 2))
        cache_session_results(db, "ses2", "a2", "penk",
                              _sample_cell_results(3, 1), _sample_summary(3, 1))

        a1_cells = load_all_cells(db, animal_id="a1")
        assert len(a1_cells) == 4

    def test_load_all_sessions(self, db):
        cache_session_results(db, "ses1", "a1", "penk",
                              _sample_cell_results(), _sample_summary())
        cache_session_results(db, "ses2", "a2", "nonpenk",
                              _sample_cell_results(3, 1), _sample_summary(3, 1))

        sessions = load_all_sessions(db)
        assert len(sessions) == 2


class TestTuningCurves:
    def test_store_and_load(self, db):
        cells = _sample_cell_results(2, 1)
        summary = _sample_summary(2, 1)
        tc_data = [
            {
                "cell_idx": 0,
                "bin_centers": np.linspace(5, 355, 36),
                "tuning_curve": np.random.rand(36),
            },
        ]
        cache_session_results(db, "ses1", "a1", "penk", cells, summary,
                              tuning_data=tc_data)

        tc = load_tuning_curve(db, "ses1", 0)
        assert tc is not None
        assert len(tc["bin_centers"]) == 36
        assert len(tc["tuning_curve"]) == 36

    def test_missing_tuning_curve(self, db):
        cells = _sample_cell_results(2, 1)
        cache_session_results(db, "ses1", "a1", "penk", cells, _sample_summary(2, 1))

        tc = load_tuning_curve(db, "ses1", 0)
        assert tc is None


class TestSummaryStats:
    def test_aggregate_stats(self, db):
        cache_session_results(db, "ses1", "a1", "penk",
                              _sample_cell_results(5, 3), _sample_summary(5, 3))
        cache_session_results(db, "ses2", "a2", "nonpenk",
                              _sample_cell_results(4, 2), _sample_summary(4, 2))

        stats = get_summary_stats(db)
        assert stats["n_cells"] == 9
        assert stats["n_sessions"] == 2
        assert stats["n_hd"] == 5

    def test_celltype_breakdown(self, db):
        cache_session_results(db, "ses1", "a1", "penk",
                              _sample_cell_results(5, 3), _sample_summary(5, 3))
        cache_session_results(db, "ses2", "a2", "nonpenk",
                              _sample_cell_results(4, 2), _sample_summary(4, 2))

        breakdown = get_celltype_breakdown(db)
        assert len(breakdown) == 2
        penk = next(b for b in breakdown if b["celltype"] == "penk")
        assert penk["n_cells"] == 5
        assert penk["n_hd"] == 3


class TestCacheClear:
    def test_clear(self, db):
        cache_session_results(db, "ses1", "a1", "penk",
                              _sample_cell_results(), _sample_summary())
        assert is_session_cached(db, "ses1")

        clear_cache(db)
        assert not is_session_cached(db, "ses1")
        assert len(load_all_cells(db)) == 0


class TestUpsert:
    def test_recompute_replaces(self, db):
        """Re-caching a session should replace, not duplicate."""
        cells1 = _sample_cell_results(5, 3)
        cache_session_results(db, "ses1", "a1", "penk", cells1, _sample_summary(5, 3))
        assert len(load_all_cells(db)) == 5

        cells2 = _sample_cell_results(4, 2)
        cache_session_results(db, "ses1", "a1", "penk", cells2, _sample_summary(4, 2))
        assert len(load_all_cells(db)) == 4  # Replaced, not 9
