#!/usr/bin/env python3
"""Within-epoch dynamics of the RSP population maze code (four questions).

Uses :mod:`hm2p.analysis.maze_dynamics` on a local cache of pickled session
arrays (as written by ``scripts/run_dead_end_recency.py --write-cache``):

1. decoding error vs time since the last light switch, lights-on correction;
2. decoded travel direction around mid-maze reversals;
3. decoding confidence before repeat vs new junction choices;
4. per-epoch decoding quality vs exploration efficiency.

Per-session tables and across-animal Wilcoxon tests are written to
``--out``. All cells of a session are pooled; cell type is attached for
descriptive splits only.

Usage
-----
    python scripts/run_maze_dynamics.py --cache results/session_cache
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hm2p.analysis import maze_dynamics as md
from hm2p.maze.topology import build_rose_maze

REPO = Path(__file__).resolve().parent.parent
SWITCH_METRICS = (
    "neural_pos_err",
    "neural_pos_excess",
    "neural_pos_ptrue",
    "neural_dir_ptrue",
    "pv_r",
    "behaviour_pos_excess",
    "behaviour_dir_ptrue",
)


def session_tables(
    d: dict[str, Any], signal: str, p: md.DynParams, n_null: int
) -> dict[str, pd.DataFrame]:
    """All per-session tables for one signal."""
    maze = build_rose_maze()
    sig = np.nan_to_num(np.asarray(d[signal], dtype=float))
    n = sig.shape[1]

    def g(k: str) -> np.ndarray:
        return np.asarray(d[k], dtype=float)[:n]

    valid = ~np.asarray(d["bad_behav"], dtype=bool)[:n]
    light = np.asarray(d["light_on"], dtype=bool)[:n]
    fps = float(d["fps"])
    args = (
        sig,
        g("x_maze"),
        g("y_maze"),
        g("speed_cm_s"),
        g("hd_deg"),
        g("ahv_deg_s"),
        light,
        valid,
        fps,
    )
    meta = {
        "exp_id": d["meta"]["exp_id"],
        "animal_id": str(d["meta"]["animal_id"]),
        "celltype": d["meta"]["celltype"],
        "signal": signal,
        "n_cells": sig.shape[0],
    }
    out: dict[str, pd.DataFrame] = {}
    df = md.frame_table(*args, params=p, maze=maze)
    run = df["running"].to_numpy(bool)

    # Session decodability (circular-shift null of the mean posterior of the true cell).
    rng = np.random.default_rng(p.seed)
    min_shift = int(round(60 * fps))
    shifts = np.r_[0, rng.integers(min_shift, n - min_shift, n_null)]
    cells = df["cell"].to_numpy()
    counts = np.bincount(cells[run], minlength=maze.n_cells)
    tr = run & np.isin(cells, np.flatnonzero(counts >= p.min_class_frames))
    nv = md.position_shift_null(
        sig,
        cells,
        tr,
        shifts,
        max(1, int(round(p.smooth_s * fps))),
        p.n_blocks,
        int(round(p.gap_s * fps)),
        p.C,
        p.max_iter,
    )
    dec = {
        **meta,
        "n_frames": n,
        "n_running": int(run.sum()),
        "ptrue_obs": nv[0],
        "ptrue_null_mean": float(np.mean(nv[1:])),
        "ptrue_null_sd": float(np.std(nv[1:])),
        "ptrue_p": float((1 + np.sum(nv[1:] >= nv[0])) / (1 + n_null)),
    }
    for cond, lv in (("light", True), ("dark", False)):
        r = df[df["running"] & (df["light"] == lv)]
        for m in (
            "neural_pos_err",
            "neural_pos_ptrue",
            "behaviour_pos_err",
            "neural_dir_ptrue",
            "behaviour_dir_ptrue",
            "pv_r",
            "pv_spec",
        ):
            dec[f"{m}_{cond}"] = float(r[m].mean())
        dec[f"speed_{cond}"] = float(r["speed"].mean())
    out["decodability"] = pd.DataFrame([dec])

    # Q1
    out["slopes"] = md.drift_slopes(df).assign(**meta)
    out["profiles"] = md.epoch_profile(
        df, md.DRIFT_METRICS + ("speed",), p.epoch_bin_s, p.epoch_len_s
    ).assign(**meta)
    sw = md.switch_table(df, SWITCH_METRICS, fps, p.edge_window_s, p.min_window_frames)
    out["switches"] = md.switch_summary(sw, SWITCH_METRICS).assign(**meta)

    # Q2
    rv = md.reversal_session(*args, params=p, maze=maze)
    out["rev_summary"] = pd.DataFrame([{**meta, **rv["summary"]}])
    out["rev_curves"] = rv["curves"].assign(**meta)
    out["rev_events"] = rv["events"].assign(**meta)

    # Q3
    dep = md.junction_departures(cells, maze, fps, p.dep_n_back)
    feat = md.departure_features(dep, df, fps, p.dep_window_s)
    out["departures"] = md.departure_effects(feat, p.min_events).assign(**meta)

    # Q4
    ep = md.epoch_exploration(
        df,
        maze,
        fps,
        np.random.default_rng(p.seed),
        p.n_rw_sims,
        p.min_epoch_steps,
        p.min_epoch_run_frames,
    )
    out["epochs"] = ep.assign(**meta)
    out["epoch_corr"] = md.epoch_correlations(ep).assign(**meta)
    return out


# ---------------------------------------------------------------------------
# Across-animal tests
# ---------------------------------------------------------------------------


def _test_row(df: pd.DataFrame, value: str, **keys: Any) -> dict[str, Any]:
    return {**keys, "value": value, **md.animal_test(df, value)}


def _paired(
    df: pd.DataFrame, index: list[str], col: str, a: str, b: str, value: str
) -> pd.DataFrame:
    w = df.pivot_table(index=index, columns=col, values=value, aggfunc="first").reset_index()
    if a not in w or b not in w:
        return pd.DataFrame()
    w["diff"] = w[a] - w[b]
    return w


def summarise(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Across-animal Wilcoxon tests for every question (per signal)."""
    rows: list[dict[str, Any]] = []
    idx = ["exp_id", "animal_id", "signal"]
    for sigl in t["slopes"]["signal"].unique():
        # Q1 slopes
        sl = t["slopes"][t["slopes"]["signal"] == sigl]
        for m, g in sl.groupby("metric"):
            for cond, gg in g.groupby("condition"):
                rows.append(
                    _test_row(
                        gg, "prho", question="Q1_slope", signal=sigl, metric=m, condition=cond
                    )
                )
            w = _paired(g, idx, "condition", "dark", "light", "prho")
            if len(w):
                rows.append(
                    _test_row(
                        w,
                        "diff",
                        question="Q1_slope",
                        signal=sigl,
                        metric=m,
                        condition="dark-light",
                    )
                )
        # Q1 switches
        sw = t["switches"][t["switches"]["signal"] == sigl]
        for (m, s), g in sw.groupby(["metric", "switch"]):
            rows.append(
                _test_row(
                    g,
                    "median_correction",
                    question="Q1_switch",
                    signal=sigl,
                    metric=m,
                    condition=s,
                )
            )
            rows.append(
                _test_row(
                    g,
                    "rho_drift_correction",
                    question="Q1_switch_scaling",
                    signal=sigl,
                    metric=m,
                    condition=s,
                )
            )
        for m, g in sw.groupby("metric"):
            w = _paired(g, idx, "switch", "dark_to_light", "light_to_dark", "median_correction")
            if len(w):
                rows.append(
                    _test_row(
                        w, "diff", question="Q1_switch", signal=sigl, metric=m, condition="d2l-l2d"
                    )
                )
        # Q2
        rs = t["rev_summary"][t["rev_summary"]["signal"] == sigl].copy()
        rs["anticip_neural_minus_behaviour"] = rs["neural_anticip"] - rs["behaviour_anticip"]
        for v in (
            "neural_anticip",
            "behaviour_anticip",
            "anticip_neural_minus_behaviour",
            "neural_t_half_headturn",
            "behaviour_t_half_headturn",
            "lead_vs_behaviour_headturn",
            "neural_t_half",
            "behaviour_t_half",
            "lead_vs_behaviour",
            "lag_ref_s",
        ):
            rows.append(_test_row(rs, v, question="Q2", signal=sigl, metric=v, condition="all"))
        # Q3
        dp = t["departures"][t["departures"]["signal"] == sigl]
        for (m, c), g in dp.groupby(["metric", "condition"]):
            for v in ("prho", "diff"):
                rows.append(_test_row(g, v, question="Q3", signal=sigl, metric=m, condition=c))
        for m, g in dp.groupby("metric"):
            w = _paired(g, idx, "condition", "dark", "light", "prho")
            if len(w):
                rows.append(
                    _test_row(
                        w, "diff", question="Q3", signal=sigl, metric=m, condition="dark-light"
                    )
                )
        # Q4
        ec = t["epoch_corr"][t["epoch_corr"]["signal"] == sigl]
        for (m, c), g in ec.groupby(["metric", "condition"]):
            rows.append(_test_row(g, "prho", question="Q4", signal=sigl, metric=m, condition=c))
    out = pd.DataFrame(rows)
    out["animal_p_fdr"] = np.nan
    for (_, _), g in out.groupby(["question", "signal"]):
        out.loc[g.index, "animal_p_fdr"] = md.bh_fdr(g["animal_p"])
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=REPO / "results" / "session_cache")
    ap.add_argument("--signals", nargs="+", default=["spikes", "dff"])
    ap.add_argument("--n-null", type=int, default=9)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "maze_dynamics")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    p = md.DynParams()
    files = sorted(args.cache.glob("*.pkl"))
    if args.limit:
        files = files[: args.limit]

    def one(f: Path) -> list[dict[str, pd.DataFrame]]:
        t0 = time.time()
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        res = [session_tables(d, s, p, args.n_null) for s in args.signals]
        print("done", f.stem, f"{time.time() - t0:.0f}s", flush=True)
        return res

    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=args.n_jobs)(delayed(one)(f) for f in files)
    args.out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    for key in parts[0][0]:
        tables[key] = pd.concat([r[key] for sess in parts for r in sess], ignore_index=True)
        tables[key].to_csv(args.out / f"{key}.csv", index=False)
    summarise(tables).to_csv(args.out / "tests.csv", index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
