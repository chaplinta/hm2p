#!/usr/bin/env python3
"""Penk+ vs Penk-CamKII+ between-group comparison on the behaviour and
map-engagement measures.

Reads the per-session results already on disk (docs/manuscripts/
behaviour-results.json and results/map_engagement/map_engagement.csv), reduces
to one value per animal (median over that animal's non-excluded sessions), and
runs a Mann-Whitney U between cell-type groups. Per-animal unit avoids
pseudoreplication; the design is underpowered (≈11 vs 4 animals), so this is a
control / hypothesis-generating check, not a confirmatory test.

For each light/dark measure it tests two things:
  - the light→dark DELTA (light minus dark) — does the dark-induced change
    differ by group?
  - the LIGHT baseline — are the groups behaviourally matched to begin with?
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
BEHAV = ROOT / "docs" / "manuscripts" / "behaviour-results.json"
MAPENG = ROOT / "results" / "map_engagement" / "map_engagement.csv"
OUT = ROOT / "results" / "celltype"

# (label, light_key, dark_key)
BEHAV_MEASURES = [
    ("speed (cm/s)", "median_speed_light", "median_speed_dark"),
    ("coverage", "mean_epoch_coverage_light", "mean_epoch_coverage_dark"),
    ("distance/epoch (m)", "mean_epoch_distance_light_m", "mean_epoch_distance_dark_m"),
    ("cells per metre", "mean_epoch_cells_per_m_light", "mean_epoch_cells_per_m_dark"),
    ("revisitation", "mean_epoch_revisit_light", "mean_epoch_revisit_dark"),
    ("occupancy entropy", "mean_epoch_entropy_light", "mean_epoch_entropy_dark"),
    ("normalised LZ", "mean_epoch_lz_light", "mean_epoch_lz_dark"),
    ("coverage vs null (z)", "mean_epoch_zcov_light", "mean_epoch_zcov_dark"),
]


def cliffs_delta(a, b):
    """Cliff's delta effect size for two groups (non-parametric)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (a.size * b.size)


def mann_whitney(penk, nonpenk):
    penk = [v for v in penk if v is not None and np.isfinite(v)]
    nonpenk = [v for v in nonpenk if v is not None and np.isfinite(v)]
    if len(penk) < 2 or len(nonpenk) < 2:
        return {"p": np.nan, "delta": np.nan, "n_penk": len(penk), "n_nonpenk": len(nonpenk),
                "med_penk": np.nan, "med_nonpenk": np.nan}
    u, p = stats.mannwhitneyu(penk, nonpenk, alternative="two-sided")
    return {"p": float(p), "delta": float(cliffs_delta(penk, nonpenk)),
            "n_penk": len(penk), "n_nonpenk": len(nonpenk),
            "med_penk": float(np.median(penk)), "med_nonpenk": float(np.median(nonpenk))}


def per_animal(sessions, value_fn):
    """Median per animal of value_fn(session); returns (penk_values, nonpenk_values)."""
    by_animal = defaultdict(list)
    ctype = {}
    for s in sessions:
        v = value_fn(s)
        if v is None or not np.isfinite(v):
            continue
        by_animal[s["animal_id"]].append(v)
        ctype[s["animal_id"]] = s.get("celltype", "")
    penk, nonpenk = [], []
    for a, vals in by_animal.items():
        m = float(np.median(vals))
        (penk if ctype[a] == "penk" else nonpenk).append(m)
    return penk, nonpenk


def fmt(label, what, res):
    star = " *" if (np.isfinite(res["p"]) and res["p"] < 0.05) else ""
    return (
        f"| {label} ({what}) | {res['med_penk']:.3f} | {res['med_nonpenk']:.3f} | "
        f"p={res['p']:.3f}{star} | δ={res['delta']:.2f} | "
        f"{res['n_penk']}v{res['n_nonpenk']} |"
    )


def main():
    sessions = [s for s in json.load(open(BEHAV))["per_session"]
                if not s.get("exclude", False)]
    lines = ["# Penk+ vs Penk-CamKII+ — between-group (per-animal Mann-Whitney)", ""]
    lines.append(f"Non-excluded sessions: {len(sessions)} "
                 f"({sum(s.get('celltype')=='penk' for s in sessions)} penk / "
                 f"{sum(s.get('celltype')=='nonpenk' for s in sessions)} nonpenk sessions). "
                 "Unit = animal (median over sessions). Underpowered control.")
    lines += ["", "| Measure | Penk+ | CamKII+ | p | Cliff δ | N |",
              "| --- | --- | --- | --- | --- | --- |"]

    n_sig = 0
    for label, kl, kd in BEHAV_MEASURES:
        # light → dark delta (light - dark): the dark-induced change
        dp, dn = per_animal(sessions, lambda s, kl=kl, kd=kd:
                            (s.get(kl) - s.get(kd)) if (s.get(kl) is not None and s.get(kd) is not None) else None)
        rd = mann_whitney(dp, dn)
        # light baseline
        lp, ln = per_animal(sessions, lambda s, kl=kl: s.get(kl))
        rl = mann_whitney(lp, ln)
        lines.append(fmt(label, "Δ light-dark", rd))
        lines.append(fmt(label, "light baseline", rl))
        n_sig += int(np.isfinite(rd["p"]) and rd["p"] < 0.05)
        n_sig += int(np.isfinite(rl["p"]) and rl["p"] < 0.05)

    # Map engagement
    if MAPENG.exists():
        df = pd.read_csv(MAPENG)
        msess = [
            {"animal_id": str(r["animal_id"]), "celltype": r["celltype"],
             "dl": r["debiased_light"], "dd": r["debiased_dark"]}
            for _, r in df.iterrows()
        ]
        dp, dn = per_animal(msess, lambda s: s["dl"] - s["dd"])
        rd = mann_whitney(dp, dn)
        lp, ln = per_animal(msess, lambda s: s["dl"])
        rl = mann_whitney(lp, ln)
        lines.append(fmt("map engagement", "Δ light-dark", rd))
        lines.append(fmt("map engagement", "light baseline", rl))
        n_sig += int(np.isfinite(rd["p"]) and rd["p"] < 0.05)
        n_sig += int(np.isfinite(rl["p"]) and rl["p"] < 0.05)

    lines += ["", f"**Significant at uncorrected p<0.05: {n_sig} of {2*len(BEHAV_MEASURES)+2} "
              "tests** (no multiple-comparison correction applied; treat as generating)."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
