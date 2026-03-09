"""Pre-computed analysis cache using DuckDB.

Stores per-cell classification results, tuning metrics, and population
statistics in a local DuckDB database. The frontend reads from this
cache instead of re-running analysis on every page load.

The cache is a single file (analysis_cache.duckdb) that can be
regenerated at any time from the sync.h5 files on S3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "analysis_cache.duckdb"


def get_connection(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection, creating the schema if needed."""
    db_path = str(path or DEFAULT_CACHE_PATH)
    conn = duckdb.connect(db_path)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cells (
            exp_id VARCHAR,
            animal_id VARCHAR,
            celltype VARCHAR,
            cell_idx INTEGER,
            is_hd BOOLEAN,
            grade VARCHAR,
            mvl DOUBLE,
            p_value DOUBLE,
            reliability DOUBLE,
            mi DOUBLE,
            preferred_direction DOUBLE,
            gain_index DOUBLE,
            speed_modulation_index DOUBLE,
            n_frames INTEGER,
            PRIMARY KEY (exp_id, cell_idx)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            exp_id VARCHAR PRIMARY KEY,
            animal_id VARCHAR,
            celltype VARCHAR,
            n_rois INTEGER,
            n_frames INTEGER,
            n_hd INTEGER,
            n_non_hd INTEGER,
            fraction_hd DOUBLE,
            mean_mvl DOUBLE,
            mean_gain_index DOUBLE,
            mean_smi DOUBLE,
            computed_at TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tuning_curves (
            exp_id VARCHAR,
            cell_idx INTEGER,
            bin_centers DOUBLE[],
            tuning_curve DOUBLE[],
            PRIMARY KEY (exp_id, cell_idx)
        )
    """)


def is_session_cached(conn: duckdb.DuckDBPyConnection, exp_id: str) -> bool:
    """Check if a session's analysis results are already cached."""
    result = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE exp_id = ?", [exp_id]
    ).fetchone()
    return result[0] > 0


def cache_session_results(
    conn: duckdb.DuckDBPyConnection,
    exp_id: str,
    animal_id: str,
    celltype: str,
    cell_results: list[dict],
    session_summary: dict,
    tuning_data: list[dict] | None = None,
) -> None:
    """Store analysis results for one session.

    Parameters
    ----------
    conn : DuckDB connection
    exp_id : str
        Session identifier.
    animal_id : str
    celltype : str
    cell_results : list of dict
        Per-cell dicts with keys: cell, is_hd, grade, mvl, p_value,
        reliability, mi, preferred_direction, gain_index, smi.
    session_summary : dict
        Session-level summary with n_rois, n_frames, n_hd, etc.
    tuning_data : list of dict or None
        Per-cell tuning curves: [{cell_idx, bin_centers, tuning_curve}].
    """
    # Delete existing data for this session (upsert)
    conn.execute("DELETE FROM cells WHERE exp_id = ?", [exp_id])
    conn.execute("DELETE FROM sessions WHERE exp_id = ?", [exp_id])
    conn.execute("DELETE FROM tuning_curves WHERE exp_id = ?", [exp_id])

    # Insert cells
    for cell in cell_results:
        conn.execute("""
            INSERT INTO cells (exp_id, animal_id, celltype, cell_idx,
                is_hd, grade, mvl, p_value, reliability, mi,
                preferred_direction, gain_index, speed_modulation_index, n_frames)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            exp_id, animal_id, celltype, cell.get("cell", 0),
            cell.get("is_hd", False), cell.get("grade", "D"),
            cell.get("mvl", 0.0), cell.get("p_value", 1.0),
            cell.get("reliability", 0.0), cell.get("mi", 0.0),
            cell.get("preferred_direction", 0.0),
            cell.get("gain_index", 0.0),
            cell.get("smi", 0.0),
            session_summary.get("n_frames", 0),
        ])

    # Insert session summary
    conn.execute("""
        INSERT INTO sessions (exp_id, animal_id, celltype, n_rois, n_frames,
            n_hd, n_non_hd, fraction_hd, mean_mvl, mean_gain_index, mean_smi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        exp_id, animal_id, celltype,
        session_summary.get("n_rois", 0),
        session_summary.get("n_frames", 0),
        session_summary.get("n_hd", 0),
        session_summary.get("n_non_hd", 0),
        session_summary.get("fraction_hd", 0.0),
        session_summary.get("mean_mvl", 0.0),
        session_summary.get("mean_gain_index", 0.0),
        session_summary.get("mean_smi", 0.0),
    ])

    # Insert tuning curves
    if tuning_data:
        for tc_row in tuning_data:
            conn.execute("""
                INSERT INTO tuning_curves (exp_id, cell_idx, bin_centers, tuning_curve)
                VALUES (?, ?, ?, ?)
            """, [
                exp_id, tc_row["cell_idx"],
                tc_row["bin_centers"].tolist(),
                tc_row["tuning_curve"].tolist(),
            ])

    conn.commit()
    log.info("Cached %d cells for %s", len(cell_results), exp_id)


def load_all_cells(
    conn: duckdb.DuckDBPyConnection,
    celltype: str | None = None,
    animal_id: str | None = None,
) -> list[dict]:
    """Load all cached cell results, optionally filtered.

    Parameters
    ----------
    conn : DuckDB connection
    celltype : str or None
        Filter by celltype (e.g. "penk", "nonpenk").
    animal_id : str or None
        Filter by animal.

    Returns
    -------
    list of dict
        Per-cell result dicts.
    """
    query = "SELECT * FROM cells WHERE 1=1"
    params = []
    if celltype:
        query += " AND celltype = ?"
        params.append(celltype)
    if animal_id:
        query += " AND animal_id = ?"
        params.append(animal_id)
    query += " ORDER BY exp_id, cell_idx"

    result = conn.execute(query, params).fetchdf()
    return result.to_dict("records")


def load_all_sessions(
    conn: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """Load all cached session summaries."""
    result = conn.execute(
        "SELECT * FROM sessions ORDER BY exp_id"
    ).fetchdf()
    return result.to_dict("records")


def load_tuning_curve(
    conn: duckdb.DuckDBPyConnection,
    exp_id: str,
    cell_idx: int,
) -> dict | None:
    """Load a single cell's tuning curve."""
    result = conn.execute(
        "SELECT bin_centers, tuning_curve FROM tuning_curves "
        "WHERE exp_id = ? AND cell_idx = ?",
        [exp_id, cell_idx],
    ).fetchone()
    if result is None:
        return None
    return {
        "bin_centers": np.array(result[0]),
        "tuning_curve": np.array(result[1]),
    }


def get_summary_stats(conn: duckdb.DuckDBPyConnection) -> dict:
    """Get aggregate statistics across all cached data."""
    result = conn.execute("""
        SELECT
            COUNT(*) as n_cells,
            COUNT(DISTINCT exp_id) as n_sessions,
            SUM(CASE WHEN is_hd THEN 1 ELSE 0 END) as n_hd,
            AVG(mvl) as mean_mvl,
            AVG(CASE WHEN is_hd THEN mvl ELSE NULL END) as mean_hd_mvl,
            AVG(gain_index) as mean_gain_index,
            AVG(speed_modulation_index) as mean_smi
        FROM cells
    """).fetchone()

    return {
        "n_cells": result[0],
        "n_sessions": result[1],
        "n_hd": result[2],
        "n_non_hd": result[0] - result[2],
        "mean_mvl": float(result[3]) if result[3] else 0.0,
        "mean_hd_mvl": float(result[4]) if result[4] else 0.0,
        "mean_gain_index": float(result[5]) if result[5] else 0.0,
        "mean_smi": float(result[6]) if result[6] else 0.0,
    }


def get_celltype_breakdown(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Get per-celltype statistics."""
    result = conn.execute("""
        SELECT
            celltype,
            COUNT(*) as n_cells,
            SUM(CASE WHEN is_hd THEN 1 ELSE 0 END) as n_hd,
            AVG(mvl) as mean_mvl,
            AVG(CASE WHEN is_hd THEN mvl ELSE NULL END) as mean_hd_mvl,
            COUNT(DISTINCT exp_id) as n_sessions
        FROM cells
        GROUP BY celltype
        ORDER BY celltype
    """).fetchdf()
    return result.to_dict("records")


def clear_cache(conn: duckdb.DuckDBPyConnection) -> None:
    """Delete all cached data."""
    conn.execute("DELETE FROM tuning_curves")
    conn.execute("DELETE FROM cells")
    conn.execute("DELETE FROM sessions")
    conn.commit()
    log.info("Cache cleared")
