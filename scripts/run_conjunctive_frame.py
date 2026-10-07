#!/usr/bin/env python3
"""Controls and reference frame of direction-dependent place coding (conjunctive_frame).

Per session (local cache of pickled session arrays), all cells, CASCADE spikes:

1. ``mi``: shift-debiased MI with maze position (running frames) conditional on
   head direction (8 sectors), acceleration tertiles, speed tertiles, travel
   direction (4 maze axes) and their joints; plus head direction given place
   and travel direction, and travel direction given place and head direction.
   All variants use the same frames (running, valid, defined head direction,
   travel direction and acceleration); conditions all / light / dark.
2. ``frames``: per-cell reference-frame scores (allo, root, deadend, ego_side
   on (cell, axis) locations; ego_turn on directed states), raw activity and
   activity with speed x acceleration tertile means removed; all / light / dark.
3. ``stability``: directed-state (cell x travel direction) maps, light vs dark
   Spearman with a circular-shift null, split-half (alternate epochs)
   within-condition vs cross-condition correlations, full maps and maps with
   each maze cell's mean over directions removed.

Animal-level summaries (Wilcoxon signed-rank on per-animal medians of cells)
are written to ``summary.csv``.

Usage
-----
    python scripts/run_conjunctive_frame.py --cache results/session_cache
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from hm2p.analysis import conjunctive_frame as cf
from hm2p.analysis.behaviour_mi import hd_sectors, plugin_mi
from hm2p.analysis.population_maze import draw_shifts, session_cells
from hm2p.maze.topology import build_rose_maze

REPO = Path(__file__).resolve().parent.parent
RUN_CM_S = 2.5
MIN_SHIFT_S = 30.0

# (name, label, conditioning variables)
MI_VARIANTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("P", "pos", ()),
    ("P|H", "pos", ("hd",)),
    ("P|acc", "pos", ("acc",)),
    ("P|H,acc", "pos", ("hd", "acc")),
    ("P|spd", "pos", ("spd",)),
    ("P|H,spd", "pos", ("hd", "spd")),
    ("P|spd,acc", "pos", ("spd", "acc")),
    ("P|H,spd,acc", "pos", ("hd", "spd", "acc")),
    ("P|TD", "pos", ("td",)),
    ("P|H,TD", "pos", ("hd", "td")),
    ("P|H4", "pos", ("hd4",)),
    ("H|P", "hd", ("pos",)),
    ("H|P,TD", "hd", ("pos", "td")),
    ("TD|P", "td", ("pos",)),
    ("TD|P,H", "td", ("pos", "hd")),
)


def session_arrays(d: dict[str, Any]) -> dict[str, Any]:
    """Per-frame behavioural variables for one session."""
    sig = np.nan_to_num(np.asarray(d["spikes"], dtype=float))
    n = sig.shape[1]
    fps = float(d["fps"])

    def g(k: str) -> np.ndarray:
        return np.asarray(d[k], dtype=float)[:n]

    maze = build_rose_maze()
    valid = ~np.asarray(d["bad_behav"], dtype=bool)[:n]
    light = np.asarray(d["light_on"], dtype=bool)[:n]
    x, y, spd = g("x_maze"), g("y_maze"), g("speed_cm_s")
    cells = session_cells(x, y, valid, maze)
    run = valid & (cells >= 0) & (spd >= RUN_CM_S)
    td = cf.travel_direction(x, y, cells, maze, fps)
    acc = cf.acceleration(spd, fps)
    return {
        "sig": sig,
        "n": n,
        "fps": fps,
        "maze": maze,
        "light": light,
        "cells": cells,
        "run": run,
        "td": np.where(run, td, -1),
        "hd": np.where(run, hd_sectors(g("hd_deg")), -1),
        "hd4": np.where(run, hd_sectors(g("hd_deg"), 4), -1),
        "hd_deg": g("hd_deg"),
        "x": x,
        "y": y,
        "spd": cf.tertiles(spd, run),
        "acc": cf.tertiles(acc, run),
        "acc_raw": acc,
    }


def session_info(a: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Descriptive behaviour numbers (sampling, head-direction / travel-direction overlap)."""
    m = a["run"] & (a["hd"] >= 0) & (a["td"] >= 0)
    td_ang = np.radians(a["td"][m] * 90.0)
    hd = np.radians(a["hd_deg"][m])
    r_same = float(np.abs(np.mean(np.exp(1j * (hd - td_ang))))) if m.any() else np.nan
    r_mirror = float(np.abs(np.mean(np.exp(1j * (hd + td_ang))))) if m.any() else np.nan
    acc = a["acc_raw"]
    run_td = a["run"] & (a["td"] >= 0)
    sl = cf.state_labels(a["maze"])
    inbound = np.zeros(a["n"], dtype=bool)
    inbound[run_td] = sl.toward_deadend[a["cells"][run_td], a["td"][run_td]] == 1
    outbound = np.zeros(a["n"], dtype=bool)
    outbound[run_td] = sl.toward_deadend[a["cells"][run_td], a["td"][run_td]] == 0
    return {
        "exp_id": meta["exp_id"],
        "animal_id": str(meta["animal_id"]),
        "celltype": meta.get("celltype", ""),
        "n_cells": int(a["sig"].shape[0]),
        "n_frames": int(a["n"]),
        "n_running": int(a["run"].sum()),
        "n_mi_frames": int((m & (a["acc"] >= 0)).sum()),
        "frac_running_td_defined": float(run_td.sum() / max(a["run"].sum(), 1)),
        "hd_td_R_same": r_same,
        "hd_td_R_mirror": r_mirror,
        "mi_hd8_td_bits": float(plugin_mi(a["hd"][m], a["td"][m])) if m.any() else np.nan,
        "acc_median_inbound": float(np.nanmedian(acc[inbound])) if inbound.any() else np.nan,
        "acc_median_outbound": float(np.nanmedian(acc[outbound])) if outbound.any() else np.nan,
    }


def mi_rows(a: dict[str, Any], shifts: np.ndarray) -> pd.DataFrame:
    """Per-cell MI variants (all / light / dark) on a common frame set."""
    common = a["run"] & (a["hd"] >= 0) & (a["td"] >= 0) & (a["acc"] >= 0) & (a["spd"] >= 0)
    var = {"pos": a["cells"], "hd": a["hd"], "hd4": a["hd4"], "td": a["td"]}
    var.update(spd=a["spd"], acc=a["acc"])
    rows = []
    for cond, cm in (
        ("all", common),
        ("light", common & a["light"]),
        ("dark", common & ~a["light"]),
    ):
        if cm.sum() < 200:
            continue
        vv = {k: v[cm] for k, v in var.items()}
        for j in range(a["sig"].shape[0]):
            bins = cf.shifted_bins(a["sig"][j], cm, shifts)
            for name, lab, conds in MI_VARIANTS:
                c = cf.joint(*(vv[k] for k in conds)) if conds else None
                r = cf.mi_with_null(bins, vv[lab], c)
                rows.append(
                    {"roi": j, "condition": cond, "variant": name, "n_frames": int(cm.sum()), **r}
                )
    return pd.DataFrame(rows)


def frame_rows(a: dict[str, Any], n_shuffles: int, rng: np.random.Generator) -> pd.DataFrame:
    """Per-cell reference-frame scores, raw and speed x acceleration residualised."""
    maze = a["maze"]
    signs = cf.frame_signs(maze)
    turn = cf.upcoming_turn(a["cells"], maze)
    strata = np.where(a["run"] & (a["spd"] >= 0) & (a["acc"] >= 0), a["spd"] * 3 + a["acc"], -1)
    sigs = {"raw": a["sig"], "spd_acc_removed": cf.residualise(a["sig"], strata)}
    nl = 4 * maze.n_cells
    turn_frames = {"ego_turn": (np.ones(nl), np.zeros(nl, dtype=np.int64))}
    out = []
    base = a["run"] & (a["td"] >= 0)
    for cond, m in (("all", base), ("light", base & a["light"]), ("dark", base & ~a["light"])):
        # residualised signal: only frames with a speed x acceleration stratum
        for sname, s in sigs.items():
            mm = m & (strata >= 0) if sname != "raw" else m
            loc, lab = cf.axis_locations(a["cells"], a["td"], mm)
            t = cf.frame_scores(s, loc, lab, signs, n_shuffles, rng)
            tl, tb = cf.turn_locations(a["cells"], a["td"], turn, mm)
            t2 = cf.frame_scores(s, tl, tb, turn_frames, n_shuffles, rng)
            t = pd.concat([t, t2], ignore_index=True)
            t["condition"], t["signal"] = cond, sname
            out.append(t)
    return pd.concat(out, ignore_index=True)


def stability_rows(
    a: dict[str, Any], shifts: np.ndarray, min_frames: int = 10, min_frames_half: int = 5
) -> pd.DataFrame:
    """Light vs dark directed-state map correlations with shift null and split-half reference."""
    n_states = 4 * a["maze"].n_cells
    state = np.where(a["run"] & (a["td"] >= 0), a["cells"] * 4 + a["td"], -1)
    lt = a["light"]
    par = cf.epoch_parity(lt)
    masks = {
        "L": lt,
        "D": ~lt,
        "L0": lt & (par == 0),
        "L1": lt & (par == 1),
        "D0": ~lt & (par == 0),
        "D1": ~lt & (par == 1),
    }

    def maps(sig: np.ndarray) -> dict[str, np.ndarray]:
        return {
            k: cf.state_map(
                sig, state, v, n_states, min_frames if len(k) == 1 else min_frames_half
            )
            for k, v in masks.items()
        }

    def corr(mp: dict[str, np.ndarray], resid: bool) -> dict[str, np.ndarray]:
        f = cf.direction_residual if resid else (lambda z: z)
        q = {k: f(v) for k, v in mp.items()}
        mn = 4 if resid else 5
        r = {"LD": cf.rowwise_spearman(q["L"], q["D"], mn)}
        r["within"] = np.nanmean(
            np.stack(
                [
                    cf.rowwise_spearman(q["L0"], q["L1"], mn),
                    cf.rowwise_spearman(q["D0"], q["D1"], mn),
                ]
            ),
            axis=0,
        )
        r["within_light"] = cf.rowwise_spearman(q["L0"], q["L1"], mn)
        r["within_dark"] = cf.rowwise_spearman(q["D0"], q["D1"], mn)
        r["across"] = np.nanmean(
            np.stack(
                [cf.rowwise_spearman(q[x], q[y], mn) for x in ("L0", "L1") for y in ("D0", "D1")]
            ),
            axis=0,
        )
        return r

    sig = a["sig"]
    out = []
    obs = {False: corr(maps(sig), False), True: corr(maps(sig), True)}
    nulls: dict[bool, list[np.ndarray]] = {False: [], True: []}
    for s in shifts:
        mp = maps(np.roll(sig, int(s), axis=1))
        for resid in (False, True):
            nulls[resid].append(corr(mp, resid)["LD"])
    for resid in (False, True):
        nl = np.stack(nulls[resid])
        o = obs[resid]
        with np.errstate(invalid="ignore"):
            nm, sd = np.nanmean(nl, axis=0), np.nanstd(nl, axis=0)
            z = np.where(sd > 0, (o["LD"] - nm) / sd, np.nan)
        p = (1 + np.sum(nl >= o["LD"][None, :], axis=0)) / (1 + nl.shape[0])
        out.append(
            pd.DataFrame(
                {
                    "roi": np.arange(sig.shape[0]),
                    "map": "direction_residual" if resid else "full",
                    "rho_LD": o["LD"],
                    "rho_LD_null_mean": nm,
                    "rho_LD_z": z,
                    "rho_LD_p": np.where(np.isfinite(o["LD"]), p, np.nan),
                    "rho_within": o["within"],
                    "rho_within_light": o["within_light"],
                    "rho_within_dark": o["within_dark"],
                    "rho_across": o["across"],
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def run_session(path: Path, n_shuffles: int, n_frame_shuffles: int, n_map_shuffles: int) -> dict:
    """All analyses for one cached session."""
    t0 = time.time()
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    meta = d["meta"]
    a = session_arrays(d)
    rng = np.random.default_rng(0)
    shifts = draw_shifts(a["n"], n_shuffles, int(round(MIN_SHIFT_S * a["fps"])), rng)[1:]
    mshifts = shifts[:n_map_shuffles]
    res = {
        "info": pd.DataFrame([session_info(a, meta)]),
        "mi": mi_rows(a, shifts),
        "frames": frame_rows(a, n_frame_shuffles, rng),
        "stability": stability_rows(a, mshifts),
    }
    for k in ("mi", "frames", "stability"):
        res[k]["exp_id"] = meta["exp_id"]
        res[k]["animal_id"] = str(meta["animal_id"])
        res[k]["celltype"] = meta.get("celltype", "")
    print(f"done {path.stem} {a['sig'].shape[0]} cells {time.time() - t0:.0f}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _row(test: str, cond: str, df: pd.DataFrame, col: str, **extra: Any) -> dict[str, Any]:
    r = cf.animal_test(df, col)
    return {"test": test, "condition": cond, **extra, **r}


def summarise(mi: pd.DataFrame, frames: pd.DataFrame, stab: pd.DataFrame) -> pd.DataFrame:
    """Animal-level tests for every comparison (rows), BH-FDR within each family."""
    rows: list[dict[str, Any]] = []
    # --- task 1/2: MI differences
    w = mi.pivot_table(
        index=["exp_id", "animal_id", "celltype", "roi", "condition"],
        columns="variant",
        values="mi_debiased",
    ).reset_index()
    diffs = {
        "gain_plain: P|H - P": ("P|H", "P"),
        "gain_spd: P|H,spd - P|spd": ("P|H,spd", "P|spd"),
        "gain_acc: P|H,acc - P|acc": ("P|H,acc", "P|acc"),
        "gain_spdacc: P|H,spd,acc - P|spd,acc": ("P|H,spd,acc", "P|spd,acc"),
        "gain_td: P|TD - P": ("P|TD", "P"),
        "P|TD - P|H": ("P|TD", "P|H"),
        "P|TD - P|H4": ("P|TD", "P|H4"),
        "P|H,TD - P|TD": ("P|H,TD", "P|TD"),
        "P|H,TD - P|H": ("P|H,TD", "P|H"),
    }
    for cond in ("all", "light", "dark"):
        sub = w[w.condition == cond].copy()
        for name, (x, y) in diffs.items():
            sub["_d"] = sub[x] - sub[y]
            rows.append(_row(name, cond, sub, "_d", family="mi_diff", a=x, b=y))
        for v in (
            "P",
            "P|H",
            "P|acc",
            "P|H,acc",
            "P|spd,acc",
            "P|H,spd,acc",
            "P|TD",
            "H|P",
            "H|P,TD",
            "TD|P",
            "TD|P,H",
        ):
            sub["_v"] = sub[v]
            rows.append(_row(f"MI {v} > 0", cond, sub, "_v", family="mi_level", a=v))
    # light vs dark of the gains
    ld = w[w.condition.isin(["light", "dark"])].copy()
    for name, (x, y) in diffs.items():
        ld["_d"] = ld[x] - ld[y]
        p = ld.pivot_table(
            index=["animal_id", "exp_id", "roi"], columns="condition", values="_d"
        ).dropna()
        p = p.reset_index()
        p["_ld"] = p["light"] - p["dark"]
        rows.append(
            _row(f"light - dark of [{name}]", "light-dark", p, "_ld", family="mi_lightdark")
        )
    # --- task 3: frames
    fr = frames.copy()
    for sname in ("raw", "spd_acc_removed"):
        for cond in ("all", "light", "dark"):
            sub = fr[(fr.signal == sname) & (fr.condition == cond)]
            for f in ("allo", "root", "deadend", "ego_turn"):
                s = sub[sub.frame == f].copy()
                s["_x"] = s["score"] - s["score_null_mean"]
                rows.append(
                    _row(f"frame score - null [{f}]", cond, s, "_x", family=f"frame_{sname}", a=f)
                )
                rows.append(
                    _row(f"consistency z [{f}]", cond, s, "cons_z", family=f"frame_{sname}", a=f)
                )
                rows.append(
                    _row(
                        f"mean signed pref [{f}]",
                        cond,
                        s,
                        "mean_signed",
                        family=f"frame_sign_{sname}",
                        a=f,
                    )
                )
            sc = sub.pivot_table(
                index=["animal_id", "exp_id", "roi"], columns="frame", values="score"
            ).reset_index()
            for x, y in (("root", "allo"), ("deadend", "allo"), ("root", "deadend")):
                if x in sc and y in sc:
                    sc["_d"] = sc[x] - sc[y]
                    rows.append(
                        _row(
                            f"score {x} - {y}",
                            cond,
                            sc,
                            "_d",
                            family=f"frame_cmp_{sname}",
                            a=x,
                            b=y,
                        )
                    )
        sub = fr[fr.signal == sname]
        p = (
            sub[sub.condition.isin(["light", "dark"])]
            .pivot_table(
                index=["animal_id", "exp_id", "roi", "frame"], columns="condition", values="score"
            )
            .dropna()
            .reset_index()
        )
        for f in ("allo", "root", "deadend", "ego_turn"):
            s = p[p.frame == f].copy()
            s["_ld"] = s["light"] - s["dark"]
            rows.append(
                _row(
                    f"score light - dark [{f}]",
                    "light-dark",
                    s,
                    "_ld",
                    family=f"frame_ld_{sname}",
                    a=f,
                )
            )
    # --- task 4: stability
    for mp in ("full", "direction_residual"):
        s = stab[stab["map"] == mp].copy()
        s["_x"] = s["rho_LD"] - s["rho_LD_null_mean"]
        rows.append(_row("rho light-dark - shift null", mp, s, "_x", family="stability"))
        rows.append(_row("rho light-dark", mp, s, "rho_LD", family="stability_level"))
        rows.append(_row("rho within (split-half)", mp, s, "rho_within", family="stability_level"))
        rows.append(_row("rho across (half maps)", mp, s, "rho_across", family="stability_level"))
        s["_d"] = s["rho_within"] - s["rho_across"]
        rows.append(_row("within - across", mp, s, "_d", family="stability"))
        s["_wl"] = s["rho_within_light"] - s["rho_within_dark"]
        rows.append(_row("within light - within dark", mp, s, "_wl", family="stability"))
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    for _fam, g in out.groupby("family"):
        ok = g["p"].notna()
        if ok.any():
            out.loc[g.index[ok], "p_fdr"] = stats.false_discovery_control(
                g.loc[ok, "p"].to_numpy()
            )
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n-shuffles", type=int, default=200, help="MI circular shifts")
    ap.add_argument("--n-frame-shuffles", type=int, default=300)
    ap.add_argument("--n-map-shuffles", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "conjunctive_frame")
    args = ap.parse_args(argv)
    files = sorted(args.cache.glob("*.pkl"))
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(run_session)(f, args.n_shuffles, args.n_frame_shuffles, args.n_map_shuffles)
        for f in files
    )
    args.out.mkdir(parents=True, exist_ok=True)
    cat = {k: pd.concat([r[k] for r in res], ignore_index=True) for k in res[0]}
    for k, v in cat.items():
        v.to_csv(args.out / f"{k}.csv", index=False)
    summarise(cat["mi"], cat["frames"], cat["stability"]).to_csv(
        args.out / "summary.csv", index=False
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
