#!/usr/bin/env python3
"""Rank-composite running-coupling and light-coupling scores per animal.

Implements the composites fixed in docs/plan-penk-vs-nonpenk.md ("Composite
scores", 2026-10-02): for each measure the animal median, oriented so that
higher means more coupled, is ranked across animals; each animal's composite
is the mean rank over the measures available for it. The composite is tested
with a one-sided Mann-Whitney U in the stated direction and with an exact
permutation over every assignment of the cell-type labels to animals, plus a
leave-one-animal-out direction check and per-measure direction agreement.

Inputs are the result tables written by run_celltype_programme.py; the
script reads local CSVs only (no S3).

Usage
-----
    python scripts/run_celltype_composites.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = RES / "celltype_programme_composite"

# (name, csv path, column, sign, optional row filter (column, value))
RUNNING: list[tuple[str, str, str, int, tuple[str, str] | None]] = [
    (
        "R1_movement_modulation",
        "celltype_programme/h2/cells.csv",
        "act_movement_modulation",
        1,
        None,
    ),
    (
        "R2_fast_vs_slow_corridor",
        "celltype_programme_locrun/locrun/cells.csv",
        "run_index_corr",
        1,
        None,
    ),
    (
        "R3_speed_during_events",
        "celltype_programme_etb/etb/cells.csv",
        "during_z",
        1,
        ("channel", "speed_cm_s"),
    ),
    (
        "R4_moving_during_events",
        "celltype_programme_etb/etb/cells.csv",
        "during_z",
        1,
        ("channel", "active"),
    ),
    (
        "R5_run_vs_still_step",
        "celltype_programme_runshape/runshape/cells.csv",
        "step_index",
        1,
        None,
    ),
]
LIGHT: list[tuple[str, str, str, int, tuple[str, str] | None]] = [
    (
        "L1_events_at_light_on",
        "celltype_programme_etb/etb/cells.csv",
        "onset_z",
        1,
        ("channel", "light_on"),
    ),
    ("L2_light_modulation", "celltype_programme/h2/cells.csv", "act_light_modulation", 1, None),
    (
        "L3_lights_on_sustained",
        "celltype_programme/h5/sessions.csv",
        "dtl_late_amplitude",
        1,
        None,
    ),
    ("L4_lights_off_drop", "celltype_programme/h5/sessions.csv", "ltd_early_amplitude", -1, None),
    (
        "L5_ahv_loss_in_dark",
        "celltype_programme/h3/cells.csv",
        "ahv_dark_minus_light_matched",
        -1,
        None,
    ),
    ("L6_glm_light_share", "celltype_programme/h8/cells.csv", "part_light", 1, None),
]


def animal_measure(
    df: pd.DataFrame, column: str, sign: int = 1, row_filter: tuple[str, str] | None = None
) -> pd.DataFrame:
    """Animal median of *column* (times *sign*) with the animal's cell type."""
    d = df
    if row_filter is not None:
        d = d[d[row_filter[0]] == row_filter[1]]
    d = d.dropna(subset=[column])
    g = d.groupby("animal_id").agg(value=(column, "median"), celltype=("celltype", "first"))
    g["value"] = g["value"] * sign
    g.index = g.index.astype(str)
    return g


def composite_scores(measures: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-animal ranks for each measure and their mean (the composite)."""
    ranks = {}
    celltype: dict[str, str] = {}
    for name, m in measures.items():
        ranks[name] = m["value"].rank(method="average")
        celltype.update(m["celltype"].to_dict())
    table = pd.DataFrame(ranks)
    table["composite"] = table.mean(axis=1, skipna=True)
    table["n_measures"] = table[list(ranks)].notna().sum(axis=1)
    table["celltype"] = pd.Series(celltype)
    return table


def exact_permutation_p(
    values: pd.Series, labels: pd.Series, higher_group: str
) -> tuple[float, float, int]:
    """One-sided exact permutation p for mean(higher_group) - mean(rest).

    Every assignment of ``len(higher_group)`` labels to the animals is
    enumerated. Returns (observed difference, p, number of assignments).
    """
    v = values.to_numpy(dtype=np.float64)
    is_g = (labels == higher_group).to_numpy()
    k = int(is_g.sum())
    n = v.size
    if k == 0 or k == n:
        return float("nan"), float("nan"), 0
    obs = float(v[is_g].mean() - v[~is_g].mean())
    count = 0
    total = 0
    for combo in itertools.combinations(range(n), k):
        sel = np.zeros(n, dtype=bool)
        sel[list(combo)] = True
        diff = v[sel].mean() - v[~sel].mean()
        count += int(diff >= obs - 1e-12)
        total += 1
    return obs, count / total, total


def test_composite(table: pd.DataFrame, higher_group: str) -> dict[str, Any]:
    """MWU (one-sided), exact permutation, CLES, LOAO stability for the composite."""
    t = table.dropna(subset=["composite", "celltype"])
    hi = t.loc[t["celltype"] == higher_group, "composite"]
    lo = t.loc[t["celltype"] != higher_group, "composite"]
    res: dict[str, Any] = {
        "higher_group": higher_group,
        "n_higher": int(hi.size),
        "n_lower": int(lo.size),
        "median_higher": float(hi.median()) if hi.size else np.nan,
        "median_lower": float(lo.median()) if lo.size else np.nan,
    }
    if hi.size and lo.size:
        u = stats.mannwhitneyu(hi, lo, alternative="greater")
        res["mwu_p_one_sided"] = float(u.pvalue)
        res["cles"] = float(u.statistic / (hi.size * lo.size))
    else:
        res["mwu_p_one_sided"] = res["cles"] = np.nan
    obs, p, n_assign = exact_permutation_p(t["composite"], t["celltype"], higher_group)
    res.update({"perm_diff": obs, "perm_p_one_sided": p, "n_assignments": n_assign})
    signs = []
    for a in t.index:
        sub = t.drop(index=a)
        h = sub.loc[sub["celltype"] == higher_group, "composite"]
        lo_ = sub.loc[sub["celltype"] != higher_group, "composite"]
        if h.size and lo_.size:
            signs.append(np.sign(h.mean() - lo_.mean()))
    res["loao_direction_stable"] = bool(signs) and all(s > 0 for s in signs)
    return res


def measure_agreement(table: pd.DataFrame, measures: list[str], higher_group: str) -> pd.DataFrame:
    """Per measure: mean rank by group and whether it leans the hypothesised way."""
    rows = []
    for m in measures:
        hi = table.loc[table["celltype"] == higher_group, m].dropna()
        lo = table.loc[table["celltype"] != higher_group, m].dropna()
        rows.append(
            {
                "measure": m,
                "mean_rank_higher_group": float(hi.mean()) if hi.size else np.nan,
                "mean_rank_other": float(lo.mean()) if lo.size else np.nan,
                "agrees": bool(hi.size and lo.size and hi.mean() > lo.mean()),
            }
        )
    return pd.DataFrame(rows)


def load_measures(
    spec: list[tuple[str, str, str, int, tuple[str, str] | None]], root: Path = RES
) -> dict[str, pd.DataFrame]:
    """Read each measure's table and reduce it to animal medians; missing files skipped."""
    out = {}
    for name, rel, col, sign, filt in spec:
        path = root / rel
        if not path.exists():
            print(f"missing input for {name}: {path}")
            continue
        df = pd.read_csv(path, dtype={"animal_id": str})
        if col not in df.columns:
            print(f"missing column {col} in {path}")
            continue
        out[name] = animal_measure(df, col, sign, filt)
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - reads result files
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, spec, higher in (("running", RUNNING, "penk"), ("light", LIGHT, "nonpenk")):
        meas = load_measures(spec)
        table = composite_scores(meas)
        table.to_csv(args.out / f"{label}_animals.csv")
        agree = measure_agreement(table, list(meas), higher)
        agree.to_csv(args.out / f"{label}_measures.csv", index=False)
        res = test_composite(table, higher)
        res["measures_used"] = list(meas)
        res["measures_agreeing"] = int(agree["agrees"].sum())
        summary[label] = res
        print(f"== {label}: {json.dumps(res, default=str)}")
        print(agree.round(2).to_string(index=False))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
