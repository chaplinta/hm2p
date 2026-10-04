"""Dead-end recency coding: does a cell's response on entering a dead end depend on
how long ago the animal last visited that same dead end?

For every entry into a dead end that has been visited before in the session,
the entry response of each cell is the mean activity in a window after entry
minus the mean in an equal window before it. Recency is log time since the
previous visit to the same dead end ended. Per cell, the association is a
partial Spearman correlation (rank residuals) controlling for:

- time in session (slow drift),
- speed before entry and after entry, and |AHV| after entry (behaviour),
- time since the last light switch (light epochs alternate every minute),
- dead-end identity (dummy variables; removes place tuning), and
- the pre-entry activity of the cell.

Significance uses a circular shift of the activity relative to behaviour, with
the entries and covariates held fixed. Entries are analysed in the light and
in the dark separately, and in the dark also restricted to entries whose
previous visit to the same dead end was itself in the dark.

Maze topology: Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a
labyrinth show rapid learning, sudden insight, and efficient exploration."
eLife 10:e66175. doi:10.7554/eLife.66175

Partial rank correlation via rank residuals: Conover WJ, Iman RL. 1981. "Rank
transformations as a bridge between parametric and nonparametric statistics."
The American Statistician 35:124-129. doi:10.1080/00031305.1981.10479327
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

CONDITIONS = ("all", "light", "dark", "dark_prev_dark", "light_prev_light")


def maze_visits(cell_idx: npt.ArrayLike, min_frames: int = 3) -> list[tuple[int, int, int]]:
    """Contiguous stays ``(cell, start, end)`` in a per-frame maze-cell sequence.

    Stays shorter than ``min_frames`` and invalid frames (cell < 0) are
    dropped; neighbouring stays in the same cell left after dropping are merged.
    """
    c = np.asarray(cell_idx)
    raw: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, c.size + 1):
        if i == c.size or c[i] != c[start]:
            if c[start] >= 0 and i - start >= min_frames:
                raw.append((int(c[start]), start, i))
            start = i
    merged: list[tuple[int, int, int]] = []
    for v in raw:
        if merged and merged[-1][0] == v[0]:
            merged[-1] = (v[0], merged[-1][1], v[2])
        else:
            merged.append(v)
    return merged


def dead_end_entries(
    visits: Sequence[tuple[int, int, int]],
    dead_ends: set[int],
    light_on: npt.ArrayLike,
    fps: float,
) -> pd.DataFrame:
    """Entries into previously visited dead ends with their recency.

    Columns: ``frame`` (entry frame), ``dead_end``, ``recency_s`` (entry minus
    end of the previous visit to the same dead end), ``prev_light`` (previous
    visit mostly in light).
    """
    lo = np.asarray(light_on, dtype=float)
    last: dict[int, tuple[int, bool]] = {}
    rows = []
    for c, s, e in visits:
        if c not in dead_ends:
            continue
        if c in last:
            rows.append(
                {
                    "frame": s,
                    "dead_end": c,
                    "recency_s": (s - last[c][0]) / fps,
                    "prev_light": last[c][1],
                }
            )
        last[c] = (e, bool(np.nanmean(lo[s:e]) > 0.5))
    return pd.DataFrame(rows, columns=["frame", "dead_end", "recency_s", "prev_light"])


def rank_residual(y: npt.ArrayLike, covariates: npt.ArrayLike) -> np.ndarray:
    """Ranks of ``y`` minus their least-squares fit on the covariates.

    Continuous covariates enter as ranks; 0/1 columns (dummy variables) enter
    unchanged. An intercept is always included.
    """
    yv = np.asarray(y, dtype=float)
    X = np.asarray(covariates, dtype=float).reshape(yv.size, -1)
    cols = [np.ones(yv.size)]
    for x in X.T:
        cols.append(x if np.isin(x, (0.0, 1.0)).all() else stats.rankdata(x))
    R = np.column_stack(cols)
    ry = stats.rankdata(yv)
    beta, *_ = np.linalg.lstsq(R, ry, rcond=None)
    return ry - R @ beta


def partial_spearman(y: npt.ArrayLike, x: npt.ArrayLike, covariates: npt.ArrayLike) -> float:
    """Partial Spearman correlation of ``y`` and ``x`` given ``covariates`` (NaN if degenerate)."""
    a = rank_residual(y, covariates)
    b = rank_residual(x, covariates)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _window_mean(x: np.ndarray, starts: np.ndarray, w: int, before: bool) -> np.ndarray:
    """Mean of ``x`` (cells x frames or frames) in windows before or after each start."""
    idx = (
        (starts[:, None] - w + np.arange(w)[None, :])
        if before
        else (starts[:, None] + np.arange(w)[None, :])
    )
    return np.nanmean(x[..., idx], axis=-1)


def entry_covariates(
    entries: pd.DataFrame,
    speed: npt.ArrayLike,
    ahv: npt.ArrayLike,
    light_on: npt.ArrayLike,
    fps: float,
    w: int,
) -> pd.DataFrame:
    """Per-entry behavioural covariates and light condition."""
    on = entries["frame"].to_numpy(dtype=int)
    spd = np.asarray(speed, dtype=float)
    a = np.abs(np.asarray(ahv, dtype=float))
    lo = np.asarray(light_on, dtype=float)
    out = pd.DataFrame(index=entries.index)
    out["t_session_s"] = on / fps
    for name, arr, before in (
        ("speed_pre", spd, True),
        ("speed_post", spd, False),
        ("abs_ahv_post", a, False),
    ):
        v = _window_mean(arr, on, w, before)
        out[name] = np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0.0)
    out["light_frac"] = _window_mean(lo, on, w, before=False)
    switches = np.flatnonzero(np.diff((lo > 0.5).astype(int)) != 0) + 1
    out["since_switch_s"] = [
        (s - (switches[switches <= s].max() if np.any(switches <= s) else 0)) / fps for s in on
    ]
    return out


def condition_masks(light_frac: np.ndarray, prev_light: np.ndarray) -> dict[str, np.ndarray]:
    """Entry selections for each analysis condition."""
    light = light_frac > 0.9
    dark = light_frac < 0.1
    prev_light = prev_light.astype(bool)
    return {
        "all": np.ones(light_frac.size, dtype=bool),
        "light": light,
        "dark": dark,
        "dark_prev_dark": dark & ~prev_light,
        "light_prev_light": light & prev_light,
    }


def recency_cell_table(
    signal: npt.ArrayLike,
    entries: pd.DataFrame,
    covariates: pd.DataFrame,
    w: int,
    n_shift: int = 200,
    min_shift: int = 300,
    min_entries: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-cell partial Spearman of entry response with log recency per condition, shift null.

    ``signal`` is cells x frames (NaN treated as 0). Rows: cond, roi, rho,
    null_mean, null_sd, z, p (two-sided), n_entries.
    """
    sig = np.nan_to_num(np.asarray(signal, dtype=float))
    n_frames = sig.shape[1]
    on = entries["frame"].to_numpy(dtype=int)
    keep = (on - w >= 0) & (on + w < n_frames)
    on = on[keep]
    ent = entries[keep].reset_index(drop=True)
    cov = covariates[keep].reset_index(drop=True)
    rec = np.log(ent["recency_s"].to_numpy(dtype=float))
    ids = ent["dead_end"].to_numpy()
    rng = np.random.default_rng(seed)
    hi = max(min_shift + 1, n_frames - min_shift)
    shifts = rng.integers(min_shift, hi, n_shift)

    def responses(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _window_mean(S, on, w, before=False), _window_mean(S, on, w, before=True)

    post, pre = responses(sig)
    nulls = [responses(np.roll(sig, int(k), axis=1)) for k in shifts]
    masks = condition_masks(cov["light_frac"].to_numpy(), ent["prev_light"].to_numpy())
    base_cols = ["t_session_s", "speed_pre", "since_switch_s", "speed_post", "abs_ahv_post"]
    rows = []
    for cond in CONDITIONS:
        sel = masks[cond]
        if sel.sum() < min_entries:
            continue
        uniq = np.unique(ids[sel])
        dummies = [(ids[sel] == u).astype(float) for u in uniq[1:]]
        Z = np.column_stack([cov.loc[sel, c].to_numpy() for c in base_cols] + dummies)
        for j in range(sig.shape[0]):
            yv = post[j, sel] - pre[j, sel]
            if np.nanstd(yv) == 0:
                continue
            obs = partial_spearman(yv, rec[sel], np.column_stack([Z, pre[j, sel]]))
            nul = np.array(
                [
                    partial_spearman(
                        p_[j, sel] - q_[j, sel], rec[sel], np.column_stack([Z, q_[j, sel]])
                    )
                    for p_, q_ in nulls
                ]
            )
            nul = nul[np.isfinite(nul)]
            sd = float(nul.std()) if nul.size else float("nan")
            rows.append(
                {
                    "cond": cond,
                    "roi": j,
                    "rho": obs,
                    "null_mean": float(nul.mean()) if nul.size else float("nan"),
                    "null_sd": sd,
                    "z": (obs - float(nul.mean())) / sd if nul.size and sd > 0 else float("nan"),
                    "p": float((1 + np.sum(np.abs(nul) >= abs(obs))) / (1 + nul.size))
                    if nul.size
                    else float("nan"),
                    "n_entries": int(sel.sum()),
                }
            )
    return pd.DataFrame(
        rows, columns=["cond", "roi", "rho", "null_mean", "null_sd", "z", "p", "n_entries"]
    )


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    """Across cells, sessions and animals: z against 0 per condition, and paired dark - light.

    Expects columns cond, z, roi, exp_id, animal_id. Wilcoxon signed-rank tests.
    """
    rows: list[dict[str, Any]] = []

    def _w(x: pd.Series) -> float:
        x = x.dropna()
        x = x[x != 0]
        return float(stats.wilcoxon(x).pvalue) if len(x) >= 4 else float("nan")

    for cond, g in cells.groupby("cond"):
        ses = g.groupby("exp_id").z.median()
        ani = g.groupby("animal_id").z.median()
        rows.append(
            {
                "contrast": cond,
                "n_cells": int(g.z.notna().sum()),
                "median_z": float(g.z.median()),
                "cell_p": _w(g.z),
                "n_sessions": int(ses.size),
                "session_frac_pos": float((ses > 0).mean()),
                "session_p": _w(ses),
                "n_animals": int(ani.size),
                "animal_p": _w(ani),
                "frac_sig_pos": float(((g.p < 0.05) & (g.rho > 0)).mean()),
                "frac_sig_neg": float(((g.p < 0.05) & (g.rho < 0)).mean()),
            }
        )
    w = cells.pivot_table(
        index=["exp_id", "roi", "animal_id"], columns="cond", values="z"
    ).reset_index()
    if {"dark", "light"} <= set(w.columns):
        w["dl"] = w["dark"] - w["light"]
        ses = w.groupby("exp_id").dl.median()
        ani = w.groupby("animal_id").dl.median()
        rows.append(
            {
                "contrast": "dark_minus_light",
                "n_cells": int(w.dl.notna().sum()),
                "median_z": float(w.dl.median()),
                "cell_p": _w(w.dl),
                "n_sessions": int(ses.size),
                "session_frac_pos": float((ses > 0).mean()),
                "session_p": _w(ses),
                "n_animals": int(ani.size),
                "animal_p": _w(ani),
                "frac_sig_pos": float("nan"),
                "frac_sig_neg": float("nan"),
            }
        )
    return pd.DataFrame(rows)
