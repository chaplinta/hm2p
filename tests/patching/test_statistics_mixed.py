"""Tests for hm2p.patching.statistics.mixed_model_comparison.

The LMM is a supplementary (non-primary) check for ICC reporting only;
the primary cell-type comparison remains Mann-Whitney U elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hm2p.patching.statistics import mixed_model_comparison


def _make_two_group_df(seed: int = 0, n_per_animal: int = 6) -> pd.DataFrame:
    """Synthetic table: 4 animals across 2 cell types, one metric."""
    rng = np.random.default_rng(seed)
    rows = []
    animals = {
        "penk": ["a1", "a2"],
        "nonpenk": ["a3", "a4"],
    }
    for cell_type, animal_ids in animals.items():
        offset = 0.0 if cell_type == "penk" else 2.0
        for animal in animal_ids:
            animal_effect = rng.normal(0, 0.3)
            for _ in range(n_per_animal):
                rows.append(
                    {
                        "cell_type": cell_type,
                        "animal_id": animal,
                        "vm": offset + animal_effect + rng.normal(0, 0.5),
                    }
                )
    return pd.DataFrame(rows)


def test_mixed_model_basic_columns() -> None:
    """Output has the documented columns and one row per metric."""
    df = _make_two_group_df()
    res = mixed_model_comparison(df, ["vm"])
    assert len(res) == 1
    for col in (
        "metric",
        "beta",
        "se",
        "z",
        "lmm_p_supplementary",
        "lmm_p_fdr",
        "lmm_significant",
        "icc",
        "n_groups",
        "converged",
    ):
        assert col in res.columns
    assert res.loc[0, "metric"] == "vm"


def test_mixed_model_fits_and_reports_icc() -> None:
    """A well-formed dataset yields a finite ICC in [0, 1] and n_groups=4."""
    df = _make_two_group_df()
    res = mixed_model_comparison(df, ["vm"])
    row = res.iloc[0]
    assert row["n_groups"] == 4
    if row["converged"]:
        assert np.isfinite(row["icc"])
        assert 0.0 <= row["icc"] <= 1.0
        assert np.isfinite(row["beta"])


def test_mixed_model_detects_group_difference() -> None:
    """With a strong cell-type offset, the fixed effect is non-trivial."""
    df = _make_two_group_df(seed=3, n_per_animal=10)
    res = mixed_model_comparison(df, ["vm"])
    row = res.iloc[0]
    if row["converged"] and np.isfinite(row["beta"]):
        assert abs(row["beta"]) > 0.5


def test_mixed_model_insufficient_animals() -> None:
    """Fewer than 2 animals → not fitted, NaN stats, converged=False."""
    df = pd.DataFrame(
        {
            "cell_type": ["penk"] * 3 + ["nonpenk"] * 3,
            "animal_id": ["only"] * 6,
            "vm": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    res = mixed_model_comparison(df, ["vm"])
    row = res.iloc[0]
    assert not row["converged"]
    assert np.isnan(row["beta"])
    assert np.isnan(row["lmm_p_supplementary"])
    assert row["n_groups"] == 1


def test_mixed_model_too_few_rows() -> None:
    """Fewer than 5 valid rows → skipped without fitting."""
    df = pd.DataFrame(
        {
            "cell_type": ["penk", "penk", "nonpenk", "nonpenk"],
            "animal_id": ["a1", "a2", "a3", "a4"],
            "vm": [1.0, 2.0, 3.0, 4.0],
        }
    )
    res = mixed_model_comparison(df, ["vm"])
    assert not res.iloc[0]["converged"]


def test_mixed_model_single_cell_type_skipped() -> None:
    """Only one cell type present → cannot fit fixed effect, skipped."""
    df = pd.DataFrame(
        {
            "cell_type": ["penk"] * 8,
            "animal_id": ["a1", "a1", "a2", "a2", "a3", "a3", "a4", "a4"],
            "vm": np.linspace(1, 8, 8),
        }
    )
    res = mixed_model_comparison(df, ["vm"])
    assert not res.iloc[0]["converged"]
    assert np.isnan(res.iloc[0]["beta"])


def test_mixed_model_nan_dropped() -> None:
    """NaN metric values are dropped before fitting."""
    df = _make_two_group_df(seed=5)
    df.loc[0:3, "vm"] = np.nan
    res = mixed_model_comparison(df, ["vm"])
    # Still fits from the remaining rows without raising.
    assert len(res) == 1


def test_mixed_model_multiple_metrics_fdr() -> None:
    """Multiple metrics get FDR-corrected p-values (>= raw where valid)."""
    df = _make_two_group_df(seed=7)
    df["cap"] = np.random.default_rng(7).normal(0, 1, size=len(df))
    res = mixed_model_comparison(df, ["vm", "cap"])
    assert len(res) == 2
    for _, row in res.iterrows():
        if np.isfinite(row["lmm_p_supplementary"]) and np.isfinite(row["lmm_p_fdr"]):
            assert row["lmm_p_fdr"] >= row["lmm_p_supplementary"] - 1e-9
