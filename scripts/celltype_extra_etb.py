#!/usr/bin/env python3
"""ETB runner — event-triggered behaviour for the Penk+ vs Penk-CamKII+ programme.

Reverses the usual direction of tuning analysis: calcium events are detected
in each cell's dF/F trace and behaviour is summarised around and during those
events (``hm2p.analysis.event_triggered_behaviour``). Answers "what is the
animal doing when this cell has an event?", which is relevant because Penk+
events are longer and rarer than Penk-CamKII+ events.

The runner follows the ``(args, sessions, out_dir) -> dict`` contract of
``run_celltype_hypotheses`` and contains no S3 code. Module-level
``RUNNER_KEY``, ``RUNNER`` and ``EXTRA_SYNC_KEYS`` let the programme
coordinator register it and read the extra head/body position datasets.

Outputs (under *out_dir*): ``cells.csv`` (one row per cell x channel),
``sessions.csv`` (one row per session x channel),
``event_triggered_timecourses.json`` (per channel and cell type, mean
event-triggered average on a common -5..+5 s grid), ``within_group.csv`` and
``between_group_report.csv`` (families ``etb_onset`` and ``etb_during``).

References
----------
Voigts J, Harnett MT. 2020. "Somatic and dendritic encoding of spatial
variables in retrosplenial cortex differs during 2D navigation." Neuron
105(2):237-245. doi:10.1016/j.neuron.2019.10.016

Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior by
linking point tracking to pose dynamics." Nature Methods 21:1329-1339.
doi:10.1038/s41592-024-02318-2
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_celltype_hypotheses import (  # noqa: E402
    SessionIter,
    _empty_result,
    _resample_timecourse,
    _signal_matrix,
)
from run_celltype_programme import (  # noqa: E402
    attach_session_meta,
    between_group_report,
    write_outputs,
)

log = logging.getLogger("celltype_programme")

RUNNER_KEY = "etb"
EXTRA_SYNC_KEYS = ["x_head_mm", "y_head_mm", "x_body_mm", "y_body_mm"]

ETB_GRID_S = np.round(np.arange(-5.0, 5.0 + 1e-9, 0.1), 3)
"""Common time grid (s) for event-triggered averages across sessions."""

ETB_ALPHA = 0.05
ETB_MIN_ANIMALS_WILCOXON = 2
META_COLS = ["exp_id", "animal_id", "celltype", "fibre", "lens", "virus_id", "gcamp"]
TESTS = ("onset", "during")


# ---------------------------------------------------------------------------
# Per-session pieces
# ---------------------------------------------------------------------------


def syllable_cell_summary(
    syllable_id: np.ndarray,
    events: list[tuple[np.ndarray, np.ndarray]],
    fps: float,
    n_shuffles: int,
    rng: np.random.Generator,
    bad_behav: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-cell syllable enrichment at event onsets.

    Returns one row per cell with ``syllable_max_z`` (largest enrichment z
    over syllables), ``syllable_top`` (its syllable id) and
    ``syllable_enriched`` (any syllable with z above
    ``SYLLABLE_ENRICHED_Z``). ``bad_behav`` frames are treated as missing
    syllables.
    """
    from hm2p.analysis.event_triggered_behaviour import (
        SYLLABLE_ENRICHED_Z,
        syllable_enrichment,
    )

    syl = np.asarray(syllable_id, dtype=np.float64).copy()
    if bad_behav is not None:
        bad = np.asarray(bad_behav, dtype=bool)[: syl.size]
        syl[: bad.size][bad] = -1
    rows = []
    for roi, (on, _off) in enumerate(events):
        tab = syllable_enrichment(syl, on, n_shuffles=n_shuffles, rng=rng, fps=fps)
        z = tab["z"].to_numpy(dtype=float) if not tab.empty else np.array([])
        if z.size and np.isfinite(z).any():
            j = int(np.nanargmax(z))
            rec = {
                "roi": roi,
                "syllable_max_z": float(z[j]),
                "syllable_top": int(tab["syllable"].iloc[j]),
                "syllable_enriched": bool(z[j] > SYLLABLE_ENRICHED_Z),
            }
        else:
            rec = {
                "roi": roi,
                "syllable_max_z": np.nan,
                "syllable_top": np.nan,
                "syllable_enriched": False,
            }
        rows.append(rec)
    return pd.DataFrame(
        rows, columns=["roi", "syllable_max_z", "syllable_top", "syllable_enriched"]
    )


def etb_session_summary(tab: pd.DataFrame) -> pd.DataFrame:
    """Per-channel summary of a session's cell table.

    For each test (onset, during): number of tested cells (finite p), number
    and fraction with ``p < 0.05``, and the median z.
    """
    rows = []
    for ch, g in tab.groupby("channel", sort=False):
        rec: dict[str, Any] = {
            "channel": ch,
            "median_n_events": float(np.median(g["n_events"])) if len(g) else np.nan,
        }
        for t in TESTS:
            p = g[f"{t}_p"].to_numpy(dtype=float)
            z = g[f"{t}_z"].to_numpy(dtype=float)
            tested = np.isfinite(p)
            n = int(tested.sum())
            k = int(np.sum(p[tested] < ETB_ALPHA))
            rec[f"n_{t}_tested"] = n
            rec[f"n_{t}_sig"] = k
            rec[f"frac_{t}_sig"] = k / n if n else np.nan
            rec[f"median_{t}_z"] = float(np.nanmedian(z)) if np.isfinite(z).any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _session_etas(
    channels: dict[str, np.ndarray],
    events: list[tuple[np.ndarray, np.ndarray]],
    fps: float,
) -> dict[str, np.ndarray]:
    """Mean over cells of each cell's event-triggered average, on ``ETB_GRID_S``."""
    from hm2p.analysis.event_triggered_behaviour import event_triggered_average

    out: dict[str, np.ndarray] = {}
    for name, x in channels.items():
        per_cell = []
        time_s = None
        for on, _off in events:
            eta = event_triggered_average(x, on, fps, pre_s=5.0, post_s=5.0)
            time_s = eta["time_s"]
            if eta["n"] > 0:
                per_cell.append(eta["mean"])
        if not per_cell or time_s is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(np.vstack(per_cell), axis=0)
        out[name] = _resample_timecourse(time_s, mean, ETB_GRID_S)
    return out


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------


def etb_within_group(cells: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    """Within-group tests per channel, cell type and test (onset / during).

    Animal level (inferential): Wilcoxon signed-rank of the animal medians of
    per-session median z against 0. Pooled cells (descriptive): one-sided
    exact binomial test of the fraction of cells with ``p < 0.05`` against
    0.05.
    """
    from scipy import stats

    from hm2p.analysis.event_aligned import _binom_greater

    rows: list[dict[str, Any]] = []
    if sess.empty:
        return pd.DataFrame(rows)
    for (ch, ct), g in sess.groupby(["channel", "celltype"], sort=True):
        sub = cells[(cells["channel"] == ch) & (cells["celltype"] == ct)]
        for t in TESTS:
            animal_z = g.groupby("animal_id")[f"median_{t}_z"].median().dropna()
            rec: dict[str, Any] = {
                "channel": ch,
                "celltype": ct,
                "test": t,
                "n_animals": int(animal_z.size),
                "n_sessions": int(g["exp_id"].nunique()),
                "median_animal_z": float(animal_z.median()) if animal_z.size else np.nan,
                "wilcoxon_p": np.nan,
            }
            vals = animal_z.to_numpy()
            if vals.size >= ETB_MIN_ANIMALS_WILCOXON and np.any(vals != 0):
                rec["wilcoxon_p"] = float(stats.wilcoxon(vals).pvalue)
            n = int(g[f"n_{t}_tested"].sum())
            k = int(g[f"n_{t}_sig"].sum())
            z = sub[f"{t}_z"]
            rec.update(
                {
                    "n_cells": n,
                    "n_sig": k,
                    "pooled_frac_sig": k / n if n else np.nan,
                    "pooled_binom_p": _binom_greater(k, n, ETB_ALPHA),
                    "median_cell_z": float(np.nanmedian(z)) if z.notna().any() else np.nan,
                }
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def etb_wide(cells: pd.DataFrame, test: str) -> pd.DataFrame:
    """Pivot per-cell z to one column per channel: ``{test}_z_{channel}``."""
    index = [c for c in META_COLS + ["roi_idx"] if c in cells.columns]
    wide = (
        cells.groupby(index + ["channel"], dropna=False, sort=False)[f"{test}_z"]
        .first()
        .unstack("channel")
    )
    wide.columns = [f"{test}_z_{c}" for c in wide.columns]
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_etb(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    """Event-triggered behaviour per cell, summarised per session and group.

    Always uses dF/F, because events are defined on dF/F (``--signal`` is
    ignored). Shuffle count is ``args.n_shuffles_evt`` (default 300).
    """
    from hm2p.analysis.event_triggered_behaviour import (
        behaviour_channels,
        cell_etb_table,
        event_bounds,
    )

    if getattr(args, "signal", "dff") != "dff":
        log.info("etb: events are defined on dF/F; ignoring --signal %s", args.signal)
    n_shuffles = int(getattr(args, "n_shuffles_evt", 300))
    rng = np.random.default_rng(args.seed)
    cell_rows: list[pd.DataFrame] = []
    session_rows: list[pd.DataFrame] = []
    traces: dict[str, dict[str, list[np.ndarray]]] = {}
    for row, arrays in sessions:
        dff = _signal_matrix(arrays, "dff")
        fps = float(arrays["fps"])
        channels = behaviour_channels(arrays)
        events = [event_bounds(dff[i], fps) for i in range(dff.shape[0])]
        tab = cell_etb_table(dff, channels, fps, n_shuffles=n_shuffles, rng=rng, events=events)
        frac_syl = np.nan
        if arrays.get("syllable_id") is not None:
            syl = syllable_cell_summary(
                arrays["syllable_id"], events, fps, n_shuffles, rng, arrays.get("bad_behav")
            )
            tab = tab.merge(syl, on="roi", how="left")
            frac_syl = float(syl["syllable_enriched"].mean()) if len(syl) else np.nan
        tab["roi_idx"] = np.asarray(arrays["roi_idx"])[tab["roi"].to_numpy(dtype=int)]
        cell_rows.append(attach_session_meta(tab.drop(columns=["roi"]), row))
        summ = etb_session_summary(tab)
        summ["n_cells"] = int(dff.shape[0])
        summ["frac_cells_syllable_enriched"] = frac_syl
        session_rows.append(attach_session_meta(summ, row))
        for name, tc in _session_etas(channels, events, fps).items():
            traces.setdefault(name, {}).setdefault(str(row["celltype"]), []).append(tc)
    if not cell_rows:
        return _empty_result("etb", out_dir)
    cells = pd.concat(cell_rows, ignore_index=True)
    sess = pd.concat(session_rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    write_outputs(out_dir, "sessions", sess)

    timecourses: dict[str, Any] = {"time_s": ETB_GRID_S}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, by_ct in traces.items():
            timecourses[name] = {
                ct: {"mean": np.nanmean(np.vstack(v), axis=0), "n_sessions": len(v)}
                for ct, v in by_ct.items()
            }
    write_outputs(out_dir, "event_triggered_timecourses", timecourses)

    within = etb_within_group(cells, sess)
    write_outputs(out_dir, "within_group", within)

    reports = []
    for t in TESTS:
        wide = etb_wide(cells, t)
        metrics = [c for c in wide.columns if c.startswith(f"{t}_z_")]
        if metrics:
            reports.append(
                between_group_report(
                    wide, metrics, family=f"etb_{t}", n_perms=args.n_perms, seed=args.seed
                )
            )
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    write_outputs(out_dir, "between_group_report", report)
    return {
        "n_sessions": len(cell_rows),
        "n_cells": int(cells.groupby(["exp_id", "roi_idx"]).ngroups),
        "within": within,
        "report": report,
    }


RUNNER = run_etb
