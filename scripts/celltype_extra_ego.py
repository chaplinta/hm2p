#!/usr/bin/env python3
"""EGO runner — egocentric coding in the maze, Penk+ vs Penk-CamKII+.

Plugs into ``run_celltype_programme.py`` with the same ``(args, sessions,
out_dir)`` signature as the runners in ``run_celltype_hypotheses.py``.
Per cell (pure functions in :mod:`hm2p.analysis.egocentric`):

- egocentric boundary vector (EBC) ratemap, mean resultant length (MRL),
  preferred egocentric angle and distance, circular-shift p-value; MRL is
  also computed separately in light and dark frames;
- head-body angle tuning: modulation index with a circular-shift z and p;
- nearest-wall distance tuning: Spearman rho with a circular-shift p.

Outputs: ``cells.csv``, ``sessions.csv``, ``within_group.csv`` (fraction of
significant cells per measure vs 0.05: pooled exact binomial, descriptive;
animal-level Wilcoxon signed-rank, inferential) and
``between_group_report.csv`` (family ``ego``). No S3 code; tested with
synthetic sessions.

Units: ``x_maze``/``y_maze`` are maze grid units (one unit = one corridor
cell) and are used for the EBC ratemaps and wall distance; ``x_head_mm`` etc.
are mm and are used only for the head-body angle. Each measure uses a single
coordinate frame internally, so the two unit systems are never mixed. Both
frames share the handedness of the camera image, so pipeline ``hd_deg`` is
converted to a frame heading with
:func:`hm2p.analysis.egocentric.hd_to_frame_heading` for both.

References
----------
Alexander AS et al. 2020. "Egocentric boundary vector tuning of the
retrosplenial cortex." Science Advances 6:eaaz2322.
doi:10.1126/sciadv.aaz2322
Hinman JR et al. 2019. "Neuronal representation of environmental boundaries
in egocentric coordinates." Nature Communications 10:2772.
doi:10.1038/s41467-019-10722-y
Muller RU et al. 1987. "The effects of changes in the environment on the
spatial firing of hippocampal complex-spike cells." J Neurosci 7:1951-1968.
doi:10.1523/JNEUROSCI.07-07-01951.1987
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_celltype_hypotheses import (  # noqa: E402
    SessionIter,
    _empty_result,
    _report_families,
    _signal_matrix,
)
from run_celltype_programme import attach_session_meta, write_outputs  # noqa: E402

from hm2p.analysis import egocentric as ego  # noqa: E402

log = logging.getLogger("celltype_programme")

RUNNER_KEY = "ego"
# x_head_mm/y_head_mm/x_body_mm/y_body_mm are in mm (camera frame), whereas
# x_maze/y_maze are maze grid units: the head-body angle uses the mm keys, the
# EBC ratemaps and wall distance use maze units.
EXTRA_SYNC_KEYS: list[str] = ["x_head_mm", "y_head_mm", "x_body_mm", "y_body_mm"]

ALPHA = 0.05
MIN_ANIMALS_WILCOXON = 2
MIN_SHIFT_S = 10.0
MIN_FRAMES = 200
"""Minimum valid frames for a measure (and per light condition for MRL)."""

N_ANGLE_BINS = 24
N_HBA_BINS = 18
N_WALL_BINS = 10

MEASURES = {"ebc": "ebc_p", "hba": "hba_p", "wall": "wall_p"}

EGO_FAMILIES = {
    "ego": [
        "ebc_mrl",
        "ebc_z",
        "ebc_mrl_light",
        "ebc_mrl_dark",
        "ebc_mrl_dark_minus_light",
        "hba_modulation",
        "hba_z",
        "wall_rho",
        "wall_abs_rho",
    ]
}

_CELL_COLS = [
    "ebc_mrl",
    "ebc_p",
    "ebc_z",
    "ebc_null_95",
    "ebc_pref_angle",
    "ebc_pref_dist",
    "ebc_mrl_light",
    "ebc_mrl_dark",
    "ebc_mrl_dark_minus_light",
    "hba_modulation",
    "hba_p",
    "hba_z",
    "hba_slope_sign",
    "wall_rho",
    "wall_abs_rho",
    "wall_p",
]


def _get(arrays: dict[str, Any], key: str, n: int) -> np.ndarray | None:
    """Float copy of ``arrays[key]`` truncated to *n* frames, or None if absent/short."""
    v = arrays.get(key)
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size < n:
        log.warning("%s has %d frames, need %d; skipping", key, a.size, n)
        return None
    return a[:n]


def session_geometry(arrays: dict[str, Any], walls: np.ndarray) -> dict[str, Any]:
    """Cell-independent per-session inputs for the egocentric measures.

    Returns the frame heading, valid-frame masks (overall, light, dark), the
    EBC ray distances (computed once per session), the nearest-wall distance
    and the head-body angle. Entries are None when the required positions are
    absent.
    """
    n = int(np.asarray(arrays["dff"]).shape[1])
    hd = np.asarray(arrays["hd_deg"], dtype=np.float64)[:n]
    heading = ego.hd_to_frame_heading(hd)
    mask = np.asarray(arrays["mask"], dtype=bool)[:n] & np.isfinite(heading)
    light = np.asarray(arrays["light_on"], dtype=bool)[:n]
    out: dict[str, Any] = {
        "n_frames": n,
        "heading": heading,
        "mask": mask,
        "ebc_mask": None,
        "ray_dist": None,
        "wall_d": None,
        "hba": None,
        "hba_mask": None,
    }
    xm, ym = _get(arrays, "x_maze", n), _get(arrays, "y_maze", n)
    if xm is not None and ym is not None:
        ebc_mask = mask & np.isfinite(xm) & np.isfinite(ym)
        out.update(
            {
                "x": xm,
                "y": ym,
                "ebc_mask": ebc_mask,
                "light_mask": ebc_mask & light,
                "dark_mask": ebc_mask & ~light,
                "ray_dist": ego.ray_wall_distances(
                    xm, ym, heading, walls, ego.ebc_angle_centres(N_ANGLE_BINS)
                ),
                "wall_d": ego.wall_distance(xm, ym, walls),
            }
        )
    keys = ("x_head_mm", "y_head_mm", "x_body_mm", "y_body_mm")
    pos = [_get(arrays, k, n) for k in keys]
    if all(p is not None for p in pos):
        xh, yh, xb, yb = pos
        hba = ego.head_body_angle(heading, xh, yh, xb, yb)
        out["hba"] = hba
        out["hba_mask"] = mask & np.isfinite(hba)
    return out


def cell_ego_measures(
    signal: np.ndarray,
    geo: dict[str, Any],
    walls: np.ndarray,
    fps: float,
    n_shuffles: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """All egocentric measures for one cell (NaN when inputs are missing)."""
    rec: dict[str, Any] = dict.fromkeys(_CELL_COLS, np.nan)
    shuffle_kw = {"n_shuffles": n_shuffles, "min_shift_s": MIN_SHIFT_S, "fps": fps, "rng": rng}
    if geo["ebc_mask"] is not None and geo["ebc_mask"].sum() >= MIN_FRAMES:
        x, y, heading, rd = geo["x"], geo["y"], geo["heading"], geo["ray_dist"]
        sig = ego.ebc_significance(
            signal,
            x,
            y,
            heading,
            walls,
            geo["ebc_mask"],
            n_angle_bins=N_ANGLE_BINS,
            ray_dist=rd,
            **shuffle_kw,
        )
        rec.update(
            {
                "ebc_mrl": sig["observed"],
                "ebc_p": sig["p"],
                "ebc_z": sig["z"],
                "ebc_null_95": sig["null_95"],
                "ebc_pref_angle": sig["preferred_angle"],
                "ebc_pref_dist": sig["preferred_distance"],
            }
        )
        for cond in ("light", "dark"):
            cmask = geo[f"{cond}_mask"]
            if cmask.sum() < MIN_FRAMES:
                continue
            rm = ego.egocentric_boundary_ratemap(
                signal, x, y, heading, walls, cmask, n_angle_bins=N_ANGLE_BINS, ray_dist=rd
            )
            score = ego.ebc_score(rm["ratemap"], rm["angle_centres"], rm["dist_centres"])
            rec[f"ebc_mrl_{cond}"] = score["mrl"]
        rec["ebc_mrl_dark_minus_light"] = rec["ebc_mrl_dark"] - rec["ebc_mrl_light"]
    if geo["wall_d"] is not None and geo["ebc_mask"].sum() >= MIN_FRAMES:
        wt = ego.distance_tuning(
            signal, geo["wall_d"], geo["ebc_mask"], n_bins=N_WALL_BINS, **shuffle_kw
        )
        rec.update({"wall_rho": wt["rho"], "wall_abs_rho": abs(wt["rho"]), "wall_p": wt["p"]})
    if geo["hba"] is not None and geo["hba_mask"].sum() >= MIN_FRAMES:
        at = ego.angle_tuning_significance(
            signal, geo["hba"], geo["hba_mask"], n_bins=N_HBA_BINS, **shuffle_kw
        )
        rec.update(
            {
                "hba_modulation": at["observed"],
                "hba_p": at["p"],
                "hba_z": at["z"],
                "hba_slope_sign": at["slope_sign"],
            }
        )
    for name, pcol in MEASURES.items():
        p = rec[pcol]
        rec[f"{name}_sig"] = bool(p < ALPHA) if np.isfinite(p) else np.nan
    return rec


def session_summary(cells: pd.DataFrame) -> dict[str, Any]:
    """Per-session count and fraction of significant cells for each measure."""
    rec: dict[str, Any] = {}
    for name, pcol in MEASURES.items():
        p = cells[pcol].to_numpy(dtype=float) if pcol in cells else np.array([])
        p = p[np.isfinite(p)]
        rec[f"n_{name}_tested"] = int(p.size)
        rec[f"n_{name}_sig"] = int(np.sum(p < ALPHA))
        rec[f"frac_{name}_sig"] = float(np.mean(p < ALPHA)) if p.size else np.nan
    return rec


def ego_within_group(cells: pd.DataFrame) -> pd.DataFrame:
    """Within-group tests that each population contains egocentric cells.

    Per measure and cell type: the fraction of cells with p < 0.05 pooled
    over sessions against 0.05 (exact one-sided binomial; treats cells as
    independent and is descriptive) and a one-sided Wilcoxon signed-rank test
    of the per-animal fraction minus 0.05 (inferential). A row
    ``ebc_dark_minus_light`` tests the per-animal median MRL difference
    (dark - light) against 0 (two-sided Wilcoxon).
    """
    from scipy import stats

    rows: list[dict[str, Any]] = []
    if cells.empty or "celltype" not in cells:
        return pd.DataFrame(rows)
    for ct, g in cells.groupby("celltype", sort=True):
        for name, pcol in MEASURES.items():
            sub = g[np.isfinite(g[pcol].astype(float))]
            n, k = int(len(sub)), int((sub[pcol] < ALPHA).sum())
            rec: dict[str, Any] = {
                "measure": name,
                "celltype": ct,
                "n_cells": n,
                "n_sig": k,
                "pooled_frac_sig": k / n if n else np.nan,
                "pooled_binom_p": (
                    float(stats.binomtest(k, n, ALPHA, alternative="greater").pvalue)
                    if n
                    else np.nan
                ),
                "n_animals": 0,
                "median_animal_value": np.nan,
                "wilcoxon_p": np.nan,
            }
            if n:
                frac = sub.groupby("animal_id")[pcol].apply(lambda s: float(np.mean(s < ALPHA)))
                diff = frac.to_numpy() - ALPHA
                rec["n_animals"] = int(frac.size)
                rec["median_animal_value"] = float(frac.median())
                if frac.size >= MIN_ANIMALS_WILCOXON and np.any(diff != 0):
                    rec["wilcoxon_p"] = float(stats.wilcoxon(diff, alternative="greater").pvalue)
            rows.append(rec)
        dl = g.groupby("animal_id")["ebc_mrl_dark_minus_light"].median().dropna()
        rec = {
            "measure": "ebc_dark_minus_light",
            "celltype": ct,
            "n_cells": int(g["ebc_mrl_dark_minus_light"].notna().sum()),
            "n_animals": int(dl.size),
            "median_animal_value": float(dl.median()) if dl.size else np.nan,
            "wilcoxon_p": np.nan,
        }
        if dl.size >= MIN_ANIMALS_WILCOXON and np.any(dl.to_numpy() != 0):
            rec["wilcoxon_p"] = float(stats.wilcoxon(dl.to_numpy()).pvalue)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_ego(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    """Egocentric boundary, head-body angle and wall-distance coding per cell.

    Signal from ``--signal`` (dF/F by default). Frames are those in
    ``arrays["mask"]`` with finite heading and, per measure, finite positions.
    The number of circular shifts is ``args.n_shuffles_evt`` (default 200).
    """
    n_shuffles = int(getattr(args, "n_shuffles_evt", 200))
    rng = np.random.default_rng(args.seed)
    walls = ego.maze_wall_segments()
    cell_rows: list[pd.DataFrame] = []
    sess_rows: list[pd.DataFrame] = []
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        fps = float(arrays["fps"])
        geo = session_geometry(arrays, walls)
        if geo["ebc_mask"] is None:
            log.warning("%s: no x_maze/y_maze, EBC and wall distance skipped", row["exp_id"])
        if geo["hba"] is None:
            log.warning("%s: no head/body positions, head-body angle skipped", row["exp_id"])
        recs = []
        for i in range(sig.shape[0]):
            rec = cell_ego_measures(sig[i, : geo["n_frames"]], geo, walls, fps, n_shuffles, rng)
            rec["roi_idx"] = int(np.asarray(arrays["roi_idx"])[i])
            recs.append(rec)
        cells_s = pd.DataFrame(recs)
        cell_rows.append(attach_session_meta(cells_s, row))
        summ = session_summary(cells_s)
        summ["n_cells"] = int(len(cells_s))
        summ["n_valid_frames"] = int(geo["mask"].sum())
        sess_rows.append(attach_session_meta(pd.DataFrame([summ]), row))
    if not cell_rows:
        return _empty_result("ego", out_dir)
    cells = pd.concat(cell_rows, ignore_index=True)
    sess = pd.concat(sess_rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    write_outputs(out_dir, "sessions", sess)
    within = ego_within_group(cells)
    write_outputs(out_dir, "within_group", within)
    report = _report_families(cells, EGO_FAMILIES, out_dir, args)
    return {
        "n_sessions": len(cell_rows),
        "n_cells": int(len(cells)),
        "within": within,
        "report": report,
    }


RUNNER = run_ego
