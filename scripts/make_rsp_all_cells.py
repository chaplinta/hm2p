#!/usr/bin/env python3
"""What do RSP neurons respond to in the maze? All cells pooled, no cell-type split.

Collects every per-cell test already run in the programme (each cell against its
own shuffle null) and summarises, for all soma ROIs together:

1. Fraction of cells individually significant for each behavioural variable,
   tested against the 5 % expected by chance at the animal level (Wilcoxon
   signed-rank of each animal's fraction minus 0.05) and, descriptively, at the
   cell level (binomial).
2. Co-occurrence: for each pair of variables, the log odds ratio of a cell
   being significant for both (Fisher exact test), within the cells tested
   for both.
3. Mixed selectivity: the number of variables per cell against a null that
   permutes each variable's significance labels across cells within each
   session (keeps every session's rate per variable, breaks the pairing).
4. Correlation between cells' z scores for each pair of variables (Spearman).

Population decoding (``popdec``) and maze-running (``mazerun``) results are
summarised pooled over cells.

Writes ``docs/figures/rsp_all/data/all_cells.json``.

References
----------
Rigotti M et al. 2013. "The importance of mixed selectivity in complex
cognitive tasks." Nature 497:585-590. doi:10.1038/nature12160
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "docs" / "figures" / "rsp_all" / "data" / "all_cells.json"
KEYS = ["exp_id", "roi_idx"]
ALPHA = 0.05


@dataclass(frozen=True)
class Var:
    """One per-cell significance test."""

    key: str
    label: str
    group: str
    path: str
    p_col: str | None = None  # p value column (sig = p < 0.05)
    sig_col: str | None = None  # boolean column, used instead of p_col
    z_col: str | None = None  # signed statistic for the sign requirement / correlations
    sign: int = 0  # +1 requires z > 0, -1 requires z < 0
    row_filter: tuple[str, str] | None = None
    rename: dict[str, str] | None = None  # column renames to KEYS / animal_id


GROUP_LABELS = {
    "locomotion": "Locomotion",
    "turning": "Turning and posture",
    "direction": "Direction and egocentric",
    "maze": "Maze events and routes",
    "light": "Light",
    "time": "Time in session",
}

VARS: tuple[Var, ...] = (
    Var(
        "run_state",
        "Running vs still (step)",
        "locomotion",
        "celltype_programme_runshape/runshape/cells.csv",
        p_col="step_index_p",
        z_col="step_index_z",
        sign=1,
    ),
    Var(
        "run_coupling",
        "Coupled to running (cross-corr.)",
        "locomotion",
        "celltype_programme_mazerun_ec2/spikes/mazerun/cells.csv",
        p_col="xc_peak_r_p",
        z_col="xc_peak_r_z",
        sign=1,
    ),
    Var(
        "speed_events",
        "Faster during events",
        "locomotion",
        "celltype_programme_etb/etb/cells.csv",
        p_col="during_p",
        z_col="during_z",
        sign=1,
        row_filter=("channel", "speed_cm_s"),
    ),
    Var(
        "graded_speed",
        "Graded speed while running",
        "locomotion",
        "celltype_programme_runshape/runshape/cells.csv",
        p_col="graded_rho_p",
        z_col="graded_rho_z",
    ),
    Var(
        "move_onset",
        "Movement onset response",
        "locomotion",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "move_onset"),
    ),
    Var(
        "move_offset",
        "Movement offset response",
        "locomotion",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "move_offset"),
    ),
    Var(
        "ahv_events",
        "Head turning during events",
        "turning",
        "celltype_programme_etb/etb/cells.csv",
        p_col="during_p",
        z_col="during_z",
        sign=1,
        row_filter=("channel", "abs_ahv"),
    ),
    Var(
        "turn_left",
        "Left turn response",
        "turning",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "turn_left"),
    ),
    Var(
        "turn_right",
        "Right turn response",
        "turning",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "turn_right"),
    ),
    Var(
        "head_body",
        "Head-body angle",
        "turning",
        "celltype_programme_ego/ego/cells.csv",
        sig_col="hba_sig",
        z_col="hba_z",
    ),
    Var(
        "hd",
        "Head direction",
        "direction",
        "headline_controls/cells_allne.csv",
        sig_col="hd_sig_all",
        rename={"roi": "roi_idx", "animal": "animal_id"},
    ),
    Var(
        "ebc",
        "Egocentric boundary",
        "direction",
        "celltype_programme_ego/ego/cells.csv",
        sig_col="ebc_sig",
        z_col="ebc_z",
    ),
    Var(
        "wall",
        "Distance to wall",
        "direction",
        "celltype_programme_ego/ego/cells.csv",
        sig_col="wall_sig",
    ),
    Var(
        "junction",
        "Junction entry response",
        "maze",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "junction_entry"),
    ),
    Var(
        "dead_end",
        "Dead-end entry response",
        "maze",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "dead_end_entry"),
    ),
    Var(
        "in_out",
        "Into vs out of dead ends",
        "maze",
        "celltype_programme_mazerun_ec2/spikes/mazerun/cells.csv",
        p_col="approach_return_p",
        z_col="approach_return_z",
    ),
    Var(
        "route",
        "Route specificity",
        "maze",
        "celltype_programme_mazerun_ec2/spikes/mazerun/cells.csv",
        p_col="route_reliability_p",
        z_col="route_reliability_z",
        sign=1,
    ),
    Var(
        "novelty",
        "New vs visited cells",
        "maze",
        "celltype_programme_mazerun_ec2/spikes/mazerun/cells.csv",
        p_col="novel_repeat_p",
        z_col="novel_repeat_z",
    ),
    Var(
        "light_on",
        "Lights-on response",
        "light",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "light_on"),
    ),
    Var(
        "light_off",
        "Lights-off response",
        "light",
        "celltype_programme_evt/evt/cells.csv",
        sig_col="responsive",
        z_col="z",
        row_filter=("event_type", "light_off"),
    ),
    Var(
        "light_events",
        "Lights on during events",
        "light",
        "celltype_programme_etb/etb/cells.csv",
        p_col="during_p",
        z_col="during_z",
        sign=1,
        row_filter=("channel", "light_on"),
    ),
    Var(
        "drift",
        "Slow drift over session",
        "time",
        "celltype_programme_tctx/tctx/cells.csv",
        p_col="drift_p",
        z_col="drift_rho",
    ),
    Var(
        "epoch",
        "Light-epoch identity",
        "time",
        "celltype_programme_tctx/tctx/cells.csv",
        p_col="epoch_sel_p",
        z_col="epoch_sel_eps2",
    ),
)


def load_var(v: Var, root: Path = RES) -> pd.DataFrame | None:
    """Per-cell significance (and z, if any) for one variable; None if unavailable."""
    path = root / v.path
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if v.rename:
        df = df.rename(columns=v.rename)
    if v.row_filter is not None:
        df = df[df[v.row_filter[0]] == v.row_filter[1]]
    need = set(KEYS) | {"animal_id"}
    if not need <= set(df.columns):
        return None
    if v.sig_col is not None:
        if v.sig_col not in df.columns:
            return None
        sig = df[v.sig_col].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0"))
        sig = sig.astype(float).where(df[v.sig_col].notna())
    elif v.p_col is not None and v.p_col in df.columns:
        p = pd.to_numeric(df[v.p_col], errors="coerce")
        sig = (p < ALPHA).astype(float).where(p.notna())
    else:
        return None
    z = pd.to_numeric(df[v.z_col], errors="coerce") if v.z_col and v.z_col in df.columns else None
    if v.sign and z is not None:
        sig = sig.where(~((sig == 1) & ~((z * v.sign) > 0)), 0.0)
    out = df[KEYS].copy()
    out["animal_id"] = df["animal_id"].astype(str)
    out["sig"] = sig.to_numpy()
    out["z"] = z.to_numpy() if z is not None else np.nan
    return out.dropna(subset=["sig"]).drop_duplicates(KEYS).reset_index(drop=True)


def fraction_summary(d: pd.DataFrame) -> dict[str, Any]:
    """Fraction significant: cell-level binomial and animal-level Wilcoxon vs 0.05."""
    k, n = int(d.sig.sum()), int(len(d))
    per_animal = d.groupby("animal_id").sig.mean()
    excess = per_animal.to_numpy() - ALPHA
    nz = excess[excess != 0]
    return {
        "n_cells": n,
        "n_sig": k,
        "frac": k / n if n else float("nan"),
        "binom_p": float(stats.binomtest(k, n, ALPHA).pvalue) if n else float("nan"),
        "n_animals": int(per_animal.size),
        "animal_fracs": [round(float(x), 4) for x in per_animal],
        "animal_median_frac": float(per_animal.median()) if per_animal.size else float("nan"),
        "animal_p": float(stats.wilcoxon(nz).pvalue) if nz.size >= 3 else float("nan"),
    }


def cooccurrence(table: pd.DataFrame, keys: list[str], min_n: int = 30) -> list[dict[str, Any]]:
    """Log odds ratio of joint significance for every pair (Fisher exact, Haldane-corrected OR)."""
    rows = []
    for a, b in combinations(keys, 2):
        d = table[[a, b]].dropna()
        if len(d) < min_n:
            continue
        x, y = d[a].astype(bool), d[b].astype(bool)
        t = np.array([[np.sum(x & y), np.sum(x & ~y)], [np.sum(~x & y), np.sum(~x & ~y)]])
        lor = float(
            np.log(((t[0, 0] + 0.5) * (t[1, 1] + 0.5)) / ((t[0, 1] + 0.5) * (t[1, 0] + 0.5)))
        )
        rows.append(
            {
                "a": a,
                "b": b,
                "n": int(len(d)),
                "n_both": int(t[0, 0]),
                "log_or": lor,
                "fisher_p": float(stats.fisher_exact(t).pvalue),
            }
        )
    return rows


def mixed_selectivity(
    table: pd.DataFrame, keys: list[str], n_perm: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Number of significant variables per cell vs a within-session permutation null.

    Only cells with every variable defined are used. The null permutes each
    variable's labels across cells within each session independently.
    """
    d = table.dropna(subset=keys)
    if d.empty:
        return {}
    obs_counts = d[keys].sum(axis=1).astype(int).to_numpy()
    kmax = len(keys)
    obs_hist = np.bincount(obs_counts, minlength=kmax + 1)
    rng = np.random.default_rng(seed)
    null_hist = np.zeros((n_perm, kmax + 1))
    sessions = d["exp_id"].to_numpy()
    mat = d[keys].to_numpy(dtype=float)
    groups = [np.flatnonzero(sessions == s) for s in np.unique(sessions)]
    for i in range(n_perm):
        perm = mat.copy()
        for idx in groups:
            for j in range(kmax):
                perm[idx, j] = mat[rng.permutation(idx), j]
        null_hist[i] = np.bincount(perm.sum(axis=1).astype(int), minlength=kmax + 1)
    obs_none = obs_hist[0]
    obs_multi = obs_hist[2:].sum()
    return {
        "keys": keys,
        "n_cells": int(len(d)),
        "observed": obs_hist.tolist(),
        "null_mean": null_hist.mean(axis=0).tolist(),
        "null_lo": np.percentile(null_hist, 2.5, axis=0).tolist(),
        "null_hi": np.percentile(null_hist, 97.5, axis=0).tolist(),
        "frac_none": float(obs_none / len(d)),
        "frac_multi": float(obs_multi / len(d)),
        "p_more_none": float((1 + np.sum(null_hist[:, 0] >= obs_none)) / (1 + n_perm)),
        "p_more_multi": float(
            (1 + np.sum(null_hist[:, 2:].sum(axis=1) >= obs_multi)) / (1 + n_perm)
        ),
    }


def z_correlations(zt: pd.DataFrame, keys: list[str], min_n: int = 30) -> list[dict[str, Any]]:
    """Spearman correlation across cells between z scores of each pair of variables."""
    rows = []
    for a, b in combinations(keys, 2):
        d = zt[[a, b]].dropna()
        if len(d) < min_n:
            continue
        r = stats.spearmanr(d[a], d[b])
        rows.append(
            {"a": a, "b": b, "n": int(len(d)), "rho": float(r.statistic), "p": float(r.pvalue)}
        )
    return rows


def celltype_specificity(
    loaded: dict[str, tuple[Var, pd.DataFrame]],
    cells: pd.DataFrame,
    n_iter: int = 200,
    n_bins: int = 10,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Responsive-cell fractions per variable, Penk+ vs Penk-CamKII+, raw and brightness-matched.

    ``cells`` needs exp_id, roi_idx, celltype and f_baseline. Brightness
    matching draws, within deciles of log baseline fluorescence, equal numbers
    of cells of each type (``n_iter`` draws); the interval is the 2.5-97.5
    percentile over draws and reflects cell resampling only, not animals.
    Raw comparisons: Fisher exact on cells; Mann-Whitney U on per-animal fractions.
    """
    c = cells.dropna(subset=["f_baseline"]).copy()
    c = c[c.celltype.isin(["penk", "nonpenk"])]
    c["bin"] = pd.qcut(
        np.log(c.f_baseline.clip(lower=1e-6)), n_bins, labels=False, duplicates="drop"
    )
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_iter):
        pick = []
        for _, g in c.groupby("bin"):
            p, n = g[g.celltype == "penk"], g[g.celltype == "nonpenk"]
            m = min(len(p), len(n))
            if m:
                pick.append(p.iloc[rng.choice(len(p), m, replace=False)])
                pick.append(n.iloc[rng.choice(len(n), m, replace=False)])
        draws.append(
            pd.concat(pick)[KEYS + ["celltype"]] if pick else c.iloc[:0][KEYS + ["celltype"]]
        )
    rows = []
    for k, (v, d) in loaded.items():
        dd = d.merge(c[KEYS + ["celltype"]], on=KEYS)
        a, b = dd[dd.celltype == "penk"], dd[dd.celltype == "nonpenk"]
        if a.empty or b.empty:
            continue
        t = [[a.sig.sum(), len(a) - a.sig.sum()], [b.sig.sum(), len(b) - b.sig.sum()]]
        am = dd.groupby(["animal_id", "celltype"]).sig.mean().reset_index()
        pa = am[am.celltype == "penk"].sig
        pb = am[am.celltype == "nonpenk"].sig
        diffs, fa_m, fb_m = [], [], []
        for sel in draws:
            m = d.merge(sel, on=KEYS)
            x, y = m[m.celltype == "penk"].sig, m[m.celltype == "nonpenk"].sig
            if len(x) and len(y):
                fa_m.append(x.mean())
                fb_m.append(y.mean())
                diffs.append(x.mean() - y.mean())
        diffs_a = np.asarray(diffs)
        rows.append(
            {
                "key": k,
                "label": v.label,
                "group": v.group,
                "frac_penk": float(a.sig.mean()),
                "frac_nonpenk": float(b.sig.mean()),
                "n_penk": int(len(a)),
                "n_nonpenk": int(len(b)),
                "fisher_p": float(stats.fisher_exact(t).pvalue),
                "animal_mwu_p": float(stats.mannwhitneyu(pa, pb).pvalue)
                if len(pa) and len(pb)
                else float("nan"),
                "matched_frac_penk": float(np.mean(fa_m)) if fa_m else float("nan"),
                "matched_frac_nonpenk": float(np.mean(fb_m)) if fb_m else float("nan"),
                "matched_diff": float(diffs_a.mean()) if diffs_a.size else float("nan"),
                "matched_lo": float(np.percentile(diffs_a, 2.5)) if diffs_a.size else float("nan"),
                "matched_hi": float(np.percentile(diffs_a, 97.5))
                if diffs_a.size
                else float("nan"),
            }
        )
    return rows


MOVESTATE_METRICS = (
    ("run_d", "Running straight vs still"),
    ("turn_d", "Turning in place vs still"),
    ("bias_d", "Running vs turning in place"),
    ("interaction_contrast_d", "Turning added to running"),
    ("ahv_rho_matched", "Graded turning at matched speed"),
    ("syl_speed_rho", "Syllable preference follows syllable speed"),
    ("syl_mi", "Syllable information"),
    ("within_fast_mi", "Syllable information within fast syllables"),
)


def movestate_summary(root: Path, signal: str = "spikes") -> dict[str, Any]:
    """Pooled and per-cell-type movestate summaries (hm2p.analysis.movement_state)."""
    base = root / "celltype_programme_movestate_ec2" / signal / "movestate"
    if not (base / "cells.csv").exists():
        return {}
    c = pd.read_csv(base / "cells.csv")
    c["animal_id"] = c["animal_id"].astype(str)
    st = pd.read_csv(base / "state_summary.csv") if (base / "state_summary.csv").exists() else None
    metrics = []
    for key, label in MOVESTATE_METRICS:
        val = f"{key}_debiased" if f"{key}_debiased" in c.columns else key
        if val not in c.columns:
            continue
        row: dict[str, Any] = {"key": key, "label": label, "groups": {}}
        for g in ("all", "penk", "nonpenk"):
            d = c if g == "all" else c[c.celltype == g]
            am = d.groupby("animal_id")[val].median().dropna()
            nz = am[am != 0]
            p, z = d.get(f"{key}_p"), d.get(f"{key}_z")
            row["groups"][g] = {
                "median": float(d[val].median()),
                "n_cells": int(d[val].notna().sum()),
                "n_animals": int(am.size),
                "animal_p": float(stats.wilcoxon(nz).pvalue) if nz.size >= 4 else float("nan"),
                "frac_sig_pos": float(((p < ALPHA) & (z > 0)).mean())
                if p is not None
                else float("nan"),
                "frac_sig_neg": float(((p < ALPHA) & (z < 0)).mean())
                if p is not None
                else float("nan"),
            }
        a = c.loc[c.celltype == "penk", val].dropna()
        b = c.loc[c.celltype == "nonpenk", val].dropna()
        am = c.groupby(["animal_id", "celltype"])[val].median().reset_index()
        row["between_cell_p"] = (
            float(stats.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else float("nan")
        )
        pa, pb = (
            am[am.celltype == "penk"][val].dropna(),
            am[am.celltype == "nonpenk"][val].dropna(),
        )
        row["between_animal_p"] = (
            float(stats.mannwhitneyu(pa, pb).pvalue) if len(pa) and len(pb) else float("nan")
        )
        metrics.append(row)
    frac = {}
    if "class_frac_of_syl_mi" in c.columns:
        ok = c["syl_mi_p"] < ALPHA if "syl_mi_p" in c.columns else pd.Series(True, index=c.index)
        for g in ("all", "penk", "nonpenk"):
            d = c[ok] if g == "all" else c[ok & (c.celltype == g)]
            frac[g] = float(d["class_frac_of_syl_mi"].clip(upper=1.5).median())
    seconds = {}
    if st is not None:
        for k in ("s_still", "s_straight", "s_turn", "s_runturn"):
            if k in st.columns:
                seconds[k] = [float(x) for x in st[k].quantile([0, 0.5, 1.0])]
    cells = {
        g: {
            "run_d": c.loc[c.celltype == g, "run_d"].round(4).tolist(),
            "turn_d": c.loc[c.celltype == g, "turn_d"].round(4).tolist(),
            "animal_id": c.loc[c.celltype == g, "animal_id"].tolist(),
        }
        for g in ("penk", "nonpenk")
    }
    return {
        "signal": signal,
        "metrics": metrics,
        "speed_class_share": frac,
        "state_seconds": seconds,
        "cells": cells,
    }


def popdec_summary(root: Path) -> list[dict[str, Any]]:
    """Pooled-cell population decoding headline rows (CASCADE spikes)."""
    p = root / "celltype_programme_popdec_ec2" / "spikes" / "popdec" / "summary.csv"
    if not p.exists():
        return []
    s = pd.read_csv(p)
    s = s[(s["group"] == "all") & (s["headline"])]
    cols = [
        "decoder",
        "metric",
        "feature_set",
        "n_sessions",
        "median_observed",
        "median_null_mean",
        "median_excess",
        "frac_sessions_sig",
        "session_wilcoxon_p",
        "n_animals",
        "animal_wilcoxon_p",
    ]
    return s[cols].to_dict(orient="records")


def _specificity(root: Path, loaded: dict[str, tuple[Var, pd.DataFrame]]) -> list[dict[str, Any]]:
    p = root / "celltype_programme_ctl" / "ctl" / "cells.csv"
    if not p.exists():
        return []
    cells = pd.read_csv(p)[KEYS + ["celltype", "f_baseline"]]
    return celltype_specificity(loaded, cells)


def build(root: Path = RES, n_perm: int = 1000) -> dict[str, Any]:
    """Assemble all pooled-cell results."""
    loaded = {v.key: (v, load_var(v, root)) for v in VARS}
    loaded = {k: (v, d) for k, (v, d) in loaded.items() if d is not None and len(d)}
    variables = []
    for k, (v, d) in loaded.items():
        variables.append({"key": k, "label": v.label, "group": v.group, **fraction_summary(d)})
    sig = None
    zt = None
    for k, (_, d) in loaded.items():
        s = d[KEYS + ["animal_id", "sig"]].rename(columns={"sig": k})
        z = d[KEYS + ["z"]].rename(columns={"z": k})
        if sig is None:
            sig, zt = s, z
        else:
            sig = sig.merge(s.drop(columns="animal_id"), on=KEYS, how="outer")
            zt = zt.merge(z, on=KEYS, how="outer")
    keys = list(loaded)
    ms_keys = [k for k in keys if sig[k].notna().mean() > 0.8]
    return {
        "groups": GROUP_LABELS,
        "n_cells_total": int(len(sig)) if sig is not None else 0,
        "variables": variables,
        "cooccurrence": cooccurrence(sig, keys) if sig is not None else [],
        "z_correlation": z_correlations(zt, [k for k in keys if zt[k].notna().any()])
        if zt is not None
        else [],
        "mixed": mixed_selectivity(sig, ms_keys, n_perm=n_perm) if sig is not None else {},
        "popdec": popdec_summary(root),
        "specificity": _specificity(root, loaded),
        "movestate": movestate_summary(root),
    }


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.generic,)):
        return _clean(obj.item())
    return obj


def main() -> None:  # pragma: no cover - file I/O entry point
    data = _clean(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":  # pragma: no cover
    main()
