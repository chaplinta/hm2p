"""Maze-graph neural analyses — activity, tuning, and decoding at maze nodes.

Combines discretized maze position (from ``hm2p.maze.discretize``) with calcium
signals to examine how neural activity relates to maze topology.  Four analysis
groups:

1. **Light/dark graph annotation** — occupancy-normalised activity per cell,
   compared between light-on and light-off conditions.
2. **Decision-point HD tuning** — HD tuning curves split by location type
   (junction vs corridor vs dead-end).
3. **Path familiarity** — activity change with repeated corridor traversals.
4. **Junction choice prediction** — cross-validated logistic decoding of
   turn choice from pre-junction population vectors.

All functions are pure numpy — no I/O, no logging side-effects, no HDF5.
Insufficient data returns NaN (never raises).

Inspired by:
    Koren Iton A, Iton E, Michaelson DM, Blinder P. 2025. "NaviGraph: A
    graph-based framework for multimodal analysis of spatial
    decision-making." bioRxiv. doi:10.1101/2025.05.18.654725
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length
from hm2p.maze.analysis import classify_turn
from hm2p.maze.topology import RoseMaze

# =========================================================================
# Analysis 4 — Light/dark graph annotation
# =========================================================================


def node_activity_map(
    signals: np.ndarray,
    cell_indices: np.ndarray,
    condition_mask: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy-normalised mean activity per ROI per maze cell.

    Parameters
    ----------
    signals : (n_rois, n_frames) or (n_frames,) float
        Calcium signal matrix.  1-D is promoted to ``(1, n_frames)``.
    cell_indices : (n_frames,) int
        Maze cell index per frame (-1 = invalid).
    condition_mask : (n_frames,) bool
        Frames to include.
    n_cells : int
        Number of maze cells.

    Returns
    -------
    activity_map : (n_rois, n_cells) float
        Mean activity per cell.  NaN where occupancy is zero.
    occupancy : (n_cells,) int64
        Frame count per cell.

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph: A graph-based framework for
    multimodal analysis of spatial decision-making." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    signals = np.asarray(signals, dtype=np.float64)
    cell_indices = np.asarray(cell_indices, dtype=np.intp)
    condition_mask = np.asarray(condition_mask, dtype=bool)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    n_rois = signals.shape[0]
    activity = np.full((n_rois, n_cells), np.nan, dtype=np.float64)
    occ = np.zeros(n_cells, dtype=np.int64)
    if signals.size == 0 or cell_indices.size == 0:
        return activity, occ
    valid = condition_mask & (cell_indices >= 0) & (cell_indices < n_cells)
    if not valid.any():
        return activity, occ
    ci_v = cell_indices[valid]
    sig_v = signals[:, valid]
    np.add.at(occ, ci_v, 1)
    sig_sum = np.zeros((n_rois, n_cells), dtype=np.float64)
    for roi in range(n_rois):
        np.add.at(sig_sum[roi], ci_v, sig_v[roi])
    occupied = occ > 0
    for roi in range(n_rois):
        activity[roi, occupied] = sig_sum[roi, occupied] / occ[occupied]
    return activity, occ


def light_dark_node_contrast(
    signals: np.ndarray,
    cell_indices: np.ndarray,
    light_on: np.ndarray,
    base_mask: np.ndarray,
    n_cells: int,
    min_frames: int = 20,
) -> dict[str, np.ndarray]:
    """Compare per-node activity between light and dark conditions.

    Parameters
    ----------
    signals : (n_rois, n_frames) or (n_frames,) float
    cell_indices : (n_frames,) int
    light_on : (n_frames,) bool
    base_mask : (n_frames,) bool
    n_cells : int
    min_frames : int

    Returns
    -------
    dict
        "activity_light" : (n_rois, n_cells) float
        "activity_dark" : (n_rois, n_cells) float
        "delta" : (n_rois, n_cells) float — dark - light.
        "valid_cells" : (n_cells,) bool

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    signals = np.asarray(signals, dtype=np.float64)
    cell_indices = np.asarray(cell_indices, dtype=np.intp)
    light_on = np.asarray(light_on, dtype=bool)
    base_mask = np.asarray(base_mask, dtype=bool)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    lm, occ_l = node_activity_map(signals, cell_indices, base_mask & light_on, n_cells)
    dm, occ_d = node_activity_map(signals, cell_indices, base_mask & ~light_on, n_cells)
    valid_cells = (occ_l >= min_frames) & (occ_d >= min_frames)
    delta = dm - lm
    n_rois = signals.shape[0]
    for roi in range(n_rois):
        delta[roi, ~valid_cells] = np.nan
        lm[roi, ~valid_cells] = np.nan
        dm[roi, ~valid_cells] = np.nan
    return {
        "activity_light": lm,
        "activity_dark": dm,
        "delta": delta,
        "valid_cells": valid_cells,
    }


def light_modulation_by_node_type(
    delta: np.ndarray,
    valid_cells: np.ndarray,
    maze: RoseMaze,
    celltype_labels: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate light-dark contrast by node type.

    Parameters
    ----------
    delta : (n_rois, n_cells) float
    valid_cells : (n_cells,) bool
    maze : RoseMaze
    celltype_labels : (n_rois,) str, optional

    Returns
    -------
    dict[str, dict] with keys "junction", "corridor", "dead_end".

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    delta = np.asarray(delta, dtype=np.float64)
    valid_cells = np.asarray(valid_cells, dtype=bool)
    if delta.ndim == 1:
        delta = delta[np.newaxis, :]
    ntmap = {maze.cell_to_idx[c]: t for c, t in maze.node_types.items()}
    groups: dict[str, list[int]] = {"junction": [], "corridor": [], "dead_end": []}
    for idx in range(delta.shape[1]):
        if idx >= len(valid_cells) or not valid_cells[idx]:
            continue
        nt = ntmap.get(idx)
        if nt in ("t_junction", "crossroads"):
            groups["junction"].append(idx)
        elif nt == "corridor":
            groups["corridor"].append(idx)
        elif nt == "dead_end":
            groups["dead_end"].append(idx)
    n_rois = delta.shape[0]
    result: dict[str, dict[str, Any]] = {}
    for ntype, idxs in groups.items():
        e: dict[str, Any] = {}
        if not idxs:
            e["mean_delta"] = np.full(n_rois, np.nan)
            e["wilcoxon_p"] = np.nan
        else:
            md = np.nanmean(delta[:, idxs], axis=1)
            e["mean_delta"] = md
            fin = np.isfinite(md)
            if int(fin.sum()) >= 6:
                try:
                    _, p = sp_stats.wilcoxon(md[fin], alternative="two-sided")
                    e["wilcoxon_p"] = float(p)
                except ValueError:
                    e["wilcoxon_p"] = np.nan
            else:
                e["wilcoxon_p"] = np.nan
        if celltype_labels is not None:
            ct = np.asarray(celltype_labels)
            fin = np.isfinite(e["mean_delta"])
            ut = np.unique(ct[fin]) if fin.any() else np.array([])
            if len(ut) == 2:
                va = e["mean_delta"][(ct == ut[0]) & fin]
                vb = e["mean_delta"][(ct == ut[1]) & fin]
                if len(va) >= 3 and len(vb) >= 3:
                    try:
                        u, p = sp_stats.mannwhitneyu(va, vb, alternative="two-sided")
                        e["mannwhitney_U"] = float(u)
                        e["mannwhitney_p"] = float(p)
                    except ValueError:
                        e["mannwhitney_U"] = np.nan
                        e["mannwhitney_p"] = np.nan
                else:
                    e["mannwhitney_U"] = np.nan
                    e["mannwhitney_p"] = np.nan
            else:
                e["mannwhitney_U"] = np.nan
                e["mannwhitney_p"] = np.nan
        result[ntype] = e
    return result


# =========================================================================
# Analysis 1 — Decision-point HD tuning
# =========================================================================


def classify_frames_by_node_type(
    cell_indices: np.ndarray,
    maze: RoseMaze,
) -> dict[str, np.ndarray]:
    """Classify each frame by maze-node type.

    Returns dict with bool masks: "junction", "corridor", "dead_end", "invalid".

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    ci = np.asarray(cell_indices, dtype=np.intp)
    n = len(ci)
    masks: dict[str, np.ndarray] = {
        k: np.zeros(n, dtype=bool) for k in ("junction", "corridor", "dead_end", "invalid")
    }
    lut: dict[int, str] = {}
    for cell, nt in maze.node_types.items():
        idx = maze.cell_to_idx[cell]
        if nt in ("t_junction", "crossroads"):
            lut[idx] = "junction"
        elif nt == "corridor":
            lut[idx] = "corridor"
        elif nt == "dead_end":
            lut[idx] = "dead_end"
    for i in range(n):
        c = int(ci[i])
        if c < 0 or c >= maze.n_cells:
            masks["invalid"][i] = True
        else:
            masks[lut.get(c, "invalid")][i] = True
    return masks


def hd_tuning_by_location_type(
    signal: np.ndarray,
    hd_deg: np.ndarray,
    location_masks: dict[str, np.ndarray],
    condition_mask: np.ndarray,
    n_bins: int = 36,
    min_frames: int = 50,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """HD tuning per location type.

    Returns dict mapping location type to ``(tuning_curve, bin_centers, mvl)``.

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    signal = np.asarray(signal, dtype=np.float64)
    hd_deg = np.asarray(hd_deg, dtype=np.float64)
    condition_mask = np.asarray(condition_mask, dtype=bool)
    result: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for lt in ("junction", "corridor", "dead_end"):
        lm = location_masks.get(lt)
        if lm is None:
            continue
        comb = condition_mask & lm
        if comb.sum() < min_frames:
            continue
        tc, bins = compute_hd_tuning_curve(
            signal,
            hd_deg,
            comb,
            n_bins=n_bins,
            smoothing_sigma_deg=6.0,
        )
        mvl = mean_vector_length(tc, bins)
        mvl = float(np.clip(mvl, 0.0, 1.0))
        result[lt] = (tc, bins, mvl)
    return result


def junction_vs_corridor_mvl(
    signals: np.ndarray,
    hd_deg: np.ndarray,
    cell_indices: np.ndarray,
    maze: RoseMaze,
    condition_mask: np.ndarray,
    n_bins: int = 36,
    min_frames: int = 50,
) -> dict[str, Any]:
    """Compare MVL at junctions vs corridors.

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    n_rois = signals.shape[0]
    lm = classify_frames_by_node_type(cell_indices, maze)
    cond = np.asarray(condition_mask, dtype=bool)
    jm = lm["junction"] & cond
    cm = lm["corridor"] & cond
    nj, nc = int(jm.sum()), int(cm.sum())
    j_mvl = np.full(n_rois, np.nan)
    c_mvl = np.full(n_rois, np.nan)
    hd = np.asarray(hd_deg, dtype=np.float64)
    for r in range(n_rois):
        if nj >= min_frames:
            tc, b = compute_hd_tuning_curve(
                signals[r],
                hd,
                jm,
                n_bins=n_bins,
                smoothing_sigma_deg=6.0,
            )
            j_mvl[r] = mean_vector_length(tc, b)
        if nc >= min_frames:
            tc, b = compute_hd_tuning_curve(
                signals[r],
                hd,
                cm,
                n_bins=n_bins,
                smoothing_sigma_deg=6.0,
            )
            c_mvl[r] = mean_vector_length(tc, b)
    both = np.isfinite(j_mvl) & np.isfinite(c_mvl)
    if int(both.sum()) >= 6:
        try:
            s, p = sp_stats.wilcoxon(
                j_mvl[both],
                c_mvl[both],
                alternative="two-sided",
            )
            ws, wp = float(s), float(p)
        except ValueError:
            ws, wp = np.nan, np.nan
    else:
        ws, wp = np.nan, np.nan
    return {
        "junction_mvl": j_mvl,
        "corridor_mvl": c_mvl,
        "wilcoxon_stat": ws,
        "wilcoxon_p": wp,
    }


# =========================================================================
# Analysis 2 — Path familiarity
# =========================================================================


def count_corridor_traversals(
    cell_indices: np.ndarray,
    maze: RoseMaze,
) -> np.ndarray:
    """Per-cell corridor traversal counter.

    Returns (n_frames,) int32: 0 for non-corridor / invalid frames.

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    ci = np.asarray(cell_indices, dtype=np.intp)
    n = len(ci)
    out = np.zeros(n, dtype=np.int32)
    corr = {maze.cell_to_idx[c] for c in maze.corridors}
    cnt: dict[int, int] = {c: 0 for c in corr}
    prev = -2
    for i in range(n):
        c = int(ci[i])
        if c in corr:
            if c != prev:
                cnt[c] += 1
            out[i] = cnt[c]
        prev = c
    return out


def activity_by_traversal_number(
    signal: np.ndarray,
    cell_indices: np.ndarray,
    traversal_number: np.ndarray,
    condition_mask: np.ndarray,
    max_traversal: int = 10,
) -> dict[str, Any]:
    """Mean activity vs traversal number with Spearman correlation.

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    sig = np.asarray(signal, dtype=np.float64)
    tn = np.asarray(traversal_number, dtype=np.intp)
    mask = np.asarray(condition_mask, dtype=bool)
    ok = mask & (tn > 0) & (tn <= max_traversal)
    ts: list[int] = []
    ms: list[float] = []
    ns: list[int] = []
    for t in range(1, max_traversal + 1):
        m = ok & (tn == t)
        nt = int(m.sum())
        if nt > 0:
            ts.append(t)
            ms.append(float(np.nanmean(sig[m])))
            ns.append(nt)
    ta = np.array(ts, dtype=np.int64)
    ma = np.array(ms, dtype=np.float64)
    na = np.array(ns, dtype=np.int64)
    if len(ta) >= 3:
        r, p = sp_stats.spearmanr(ta, ma)
        sr, sp_ = float(r), float(p)
    else:
        sr, sp_ = np.nan, np.nan
    return {
        "traversal_nums": ta,
        "mean_activity": ma,
        "n_frames": na,
        "spearman_r": sr,
        "spearman_p": sp_,
    }


def familiarity_effect_by_cell_type(
    signals: np.ndarray,
    cell_indices: np.ndarray,
    traversal_number: np.ndarray,
    condition_mask: np.ndarray,
    celltype_labels: np.ndarray,
    max_traversal: int = 10,
) -> dict[str, Any]:
    """Compare familiarity Spearman r between cell types.

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    signals = np.asarray(signals, dtype=np.float64)
    ct = np.asarray(celltype_labels)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    nr = signals.shape[0]
    ar = np.full(nr, np.nan)
    ap = np.full(nr, np.nan)
    for roi in range(nr):
        res = activity_by_traversal_number(
            signals[roi],
            cell_indices,
            traversal_number,
            condition_mask,
            max_traversal,
        )
        ar[roi] = res["spearman_r"]
        ap[roi] = res["spearman_p"]
    ut = np.unique(ct)
    if len(ut) == 2:
        fin = np.isfinite(ar)
        a = ar[(ct == ut[0]) & fin]
        b = ar[(ct == ut[1]) & fin]
        if len(a) >= 3 and len(b) >= 3:
            try:
                u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                mu, mp = float(u), float(p)
            except ValueError:
                mu, mp = np.nan, np.nan
        else:
            mu, mp = np.nan, np.nan
    else:
        mu, mp = np.nan, np.nan
    return {
        "spearman_r": ar,
        "spearman_p": ap,
        "mannwhitney_U": mu,
        "mannwhitney_p": mp,
    }


# =========================================================================
# Analysis 3 — Junction choice prediction
# =========================================================================


@dataclass
class JunctionEvent:
    """A single junction approach-departure event (for internal use).

    Attributes
    ----------
    junction_frame : int
    junction_cell_idx : int
    prev_cell_idx : int
    next_cell_idx : int
    turn : str
    """

    junction_frame: int
    junction_cell_idx: int
    prev_cell_idx: int
    next_cell_idx: int
    turn: str


def extract_junction_events(
    cell_indices: np.ndarray,
    maze: RoseMaze,
    min_pre_frames: int = 2,
) -> list[dict[str, Any]]:
    """Extract junction approach-departure events.

    Parameters
    ----------
    cell_indices : (n_frames,) int
    maze : RoseMaze
    min_pre_frames : int

    Returns
    -------
    list[dict]
        Keys: "junction", "junction_frame", "prev_cell", "next_cell", "turn".

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    ci = np.asarray(cell_indices, dtype=np.intp)
    n = len(ci)
    jset = {maze.cell_to_idx[c] for c in maze.junctions}
    cl = maze.cell_list
    events: list[dict[str, Any]] = []
    if n == 0:
        return events
    runs: list[tuple[int, int, int]] = []
    rs = 0
    for i in range(1, n):
        if ci[i] != ci[rs]:
            runs.append((int(ci[rs]), rs, i - rs))
            rs = i
    runs.append((int(ci[rs]), rs, n - rs))
    for ri in range(1, len(runs) - 1):
        cv, fs, _ = runs[ri]
        pv, _, pl = runs[ri - 1]
        nv, _, _ = runs[ri + 1]
        if cv < 0 or cv not in jset:
            continue
        if pv < 0 or pv >= maze.n_cells or pl < min_pre_frames:
            continue
        if nv < 0 or nv >= maze.n_cells:
            continue
        turn = classify_turn(cl[pv], cl[cv], cl[nv])
        events.append(
            {
                "junction": cv,
                "junction_frame": fs,
                "prev_cell": pv,
                "next_cell": nv,
                "turn": turn,
            }
        )
    return events


def pre_junction_population_vectors(
    signals: np.ndarray,
    junction_events: list[dict[str, Any]],
    pre_window_frames: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Population vectors from pre-junction frames.

    All events are included. y contains turn label strings.

    Parameters
    ----------
    signals : (n_rois, n_frames) or (n_frames,) float
    junction_events : list[dict]
    pre_window_frames : int

    Returns
    -------
    X : (n_events, n_rois) float
    y : (n_events,) object — turn label strings

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]
    nr, nf = sig.shape
    xs: list[np.ndarray] = []
    ys: list[str] = []
    for ev in junction_events:
        frame = ev["junction_frame"]
        s = frame - pre_window_frames
        e = frame
        if s < 0 or e > nf:
            continue
        xs.append(np.mean(sig[:, s:e], axis=1))
        ys.append(ev["turn"])
    if not xs:
        return np.empty((0, nr), dtype=np.float64), np.array([], dtype=object)
    return np.stack(xs), np.array(ys, dtype=object)


def decode_junction_choice(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    n_folds: int = 5,
    min_events: int = 15,
) -> dict[str, Any]:
    """Cross-validated logistic decoding of junction choice.

    Parameters
    ----------
    X : (n_events, n_features) float
    y : (n_events,) — class labels (int or str)
    n_folds : int
    min_events : int

    Returns
    -------
    dict with keys: accuracy, chance_level, p_value, n_events,
    n_classes, fold_accuracies

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    X = np.asarray(X, dtype=np.float64)  # noqa: N806
    y = np.asarray(y)
    ne = len(y)
    uc = np.unique(y)
    nc = len(uc)
    chance = 1.0 / nc if nc > 0 else np.nan
    insuf: dict[str, Any] = {
        "accuracy": np.nan,
        "chance_level": float(chance) if np.isfinite(chance) else np.nan,
        "p_value": np.nan,
        "n_events": ne,
        "n_classes": nc,
        "fold_accuracies": [],
    }
    if ne < min_events or nc < 2:
        return insuf
    cc = {c: int((y == c).sum()) for c in uc}
    min_class = min(cc.values())
    if min_class < 2:
        return insuf

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    af = min(n_folds, min_class)
    if af < 2:
        return insuf
    skf = StratifiedKFold(n_splits=af, shuffle=True, random_state=42)
    fa: list[float] = []
    nc_tot = nt_tot = 0
    for tri, tei in skf.split(X, y_enc):
        sc = StandardScaler()
        x_tr = sc.fit_transform(X[tri])
        x_te = sc.transform(X[tei])
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(x_tr, y_enc[tri])
        pred = clf.predict(x_te)
        correct = int((pred == y_enc[tei]).sum())
        nc_tot += correct
        nt_tot += len(tei)
        fa.append(correct / len(tei))
    acc = nc_tot / nt_tot if nt_tot > 0 else np.nan
    if nt_tot > 0:
        bp = float(
            sp_stats.binomtest(
                nc_tot,
                nt_tot,
                chance,
                alternative="greater",
            ).pvalue
        )
    else:
        bp = np.nan
    return {
        "accuracy": float(acc),
        "chance_level": float(chance),
        "p_value": bp,
        "n_events": ne,
        "n_classes": nc,
        "fold_accuracies": fa,
    }
