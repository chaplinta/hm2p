"""Single-cell selectivity for runs into vs out of dead ends (AUROC).

For each cell, the area under the ROC curve separating its mean activity on
runs into dead ends (positive class) from runs out of dead ends, using the
duration- and speed-matched runs of :func:`hm2p.analysis.population_maze.inout_samples`.
AUROC = 0.5 means no preference; > 0.5 prefers inbound runs, < 0.5 outbound.

A head-direction-removed variant regresses each cell's per-run activity on
the sine and cosine of the mean head direction during the run (across runs,
with an intercept) and computes the AUROC of the residuals.

Significance: circular shifts of the activity relative to behaviour (runs and
labels fixed), two-sided on ``|AUROC - 0.5|``. The same shifts are used for
every cell, so the pooled null distribution across cells is the reference for
the whole-population histogram.

AUROC as a Mann-Whitney U statistic: Hanley JA, McNeil BJ. 1982. "The meaning
and use of the area under a receiver operating characteristic (ROC) curve."
Radiology 143:29-36. doi:10.1148/radiology.143.1.7063747
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from hm2p.analysis.population_maze import WindowMeans


def auroc(x: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """AUROC of ``x`` for label 1 vs label 0 (ties count half; NaN if a class is empty)."""
    v = np.asarray(x, dtype=float)
    lab = np.asarray(labels).astype(bool)
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(v)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_matrix(x: np.ndarray, labels: npt.ArrayLike) -> np.ndarray:
    """AUROC per column of ``x`` (samples x cells)."""
    lab = np.asarray(labels).astype(bool)
    n1, n0 = int(lab.sum()), int((~lab).sum())
    out = np.full(x.shape[1], np.nan)
    if n1 == 0 or n0 == 0:
        return out
    for j in range(x.shape[1]):
        out[j] = auroc(x[:, j], lab)
    return out


def remove_hd(act: np.ndarray, hd_sin: npt.ArrayLike, hd_cos: npt.ArrayLike) -> np.ndarray:
    """Residuals of each column of ``act`` (samples x cells) regressed on [1, sin HD, cos HD]."""
    X = np.column_stack(
        [np.ones(act.shape[0]), np.asarray(hd_sin, float), np.asarray(hd_cos, float)]
    )
    a = np.where(np.isfinite(act), act, np.nanmean(act, axis=0, keepdims=True))
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    return a - X @ beta


def cell_inout_auroc(
    signal: npt.ArrayLike,
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    labels: npt.ArrayLike,
    hd_deg: npt.ArrayLike,
    shifts: npt.ArrayLike,
    n_null_keep: int = 20,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-cell AUROC (raw and HD-removed) with a circular-shift null.

    Returns a table with columns roi, auc, auc_null_mean, auc_null_sd, auc_z,
    auc_p, auc_hd, auc_hd_null_mean, auc_hd_null_sd, auc_hd_z, auc_hd_p, and a
    dict of null AUROC samples (``raw``, ``hd``; the first ``n_null_keep``
    shifts for every cell) for pooled plotting.
    """
    sig = np.atleast_2d(np.asarray(signal, dtype=float))
    wm = WindowMeans(sig)
    hd = np.deg2rad(np.asarray(hd_deg, dtype=float))
    beh = WindowMeans(np.vstack([np.sin(hd), np.cos(hd)]))
    st, en = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    lab = np.asarray(labels).astype(int)
    hd_win = beh.means(st, en)  # behaviour stays aligned with the runs
    hs = np.where(np.isfinite(hd_win[:, 0]), hd_win[:, 0], 0.0)
    hc = np.where(np.isfinite(hd_win[:, 1]), hd_win[:, 1], 0.0)

    def both(shift: int) -> tuple[np.ndarray, np.ndarray]:
        act = wm.means(st, en, shift)
        return auroc_matrix(act, lab), auroc_matrix(remove_hd(act, hs, hc), lab)

    obs, obs_hd = both(0)
    sh = np.asarray(shifts, dtype=np.int64)
    null = np.array([both(int(s)) for s in sh]) if sh.size else np.zeros((0, 2, sig.shape[0]))
    rows: list[dict[str, Any]] = []
    for j in range(sig.shape[0]):
        rec: dict[str, Any] = {"roi": j}
        for name, o, k in (("auc", obs, 0), ("auc_hd", obs_hd, 1)):
            nv = null[:, k, j] if null.size else np.zeros(0)
            nv = nv[np.isfinite(nv)]
            sd = float(nv.std()) if nv.size else float("nan")
            rec[name] = float(o[j])
            rec[f"{name}_null_mean"] = float(nv.mean()) if nv.size else float("nan")
            rec[f"{name}_null_sd"] = sd
            rec[f"{name}_z"] = (
                (float(o[j]) - float(nv.mean())) / sd if nv.size and sd > 0 else float("nan")
            )
            rec[f"{name}_p"] = (
                float((1 + np.sum(np.abs(nv - 0.5) >= abs(o[j] - 0.5))) / (1 + nv.size))
                if nv.size and np.isfinite(o[j])
                else float("nan")
            )
        rows.append(rec)
    keep = {
        "raw": null[:n_null_keep, 0, :].ravel() if null.size else np.zeros(0),
        "hd": null[:n_null_keep, 1, :].ravel() if null.size else np.zeros(0),
    }
    return pd.DataFrame(rows), keep
