"""Sync Diagnostics Report — single-page scrolling layout.

Replaces the legacy 4-tab ``sync_page.py`` per
``docs/sync-pipeline-design.md`` §4. Three sections separated by
``st.divider()``:

A. Summary table (one row per session, colour-coded sync_status chip).
B. Aggregate panels (six histograms / scatter plots over the parquet).
C. Per-session deep-dive (pulse-train raster, cumulative count,
   ISI histograms, light cycle strip, scalar table, methods expander).

No sidebar filters, no synthetic fallback data. Sessions with
``exclude=1`` in ``experiments.csv`` remain visible per CLAUDE.md
("Process ALL sessions"); the deep-dive renders the ``Notes`` field.
Sessions whose ``sync_status`` starts with ``FAILED_`` are still
diagnosable here even though Stage 6 refuses to consume them.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure src is on path for any hm2p imports.
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from frontend.components.sync_diag import (  # noqa: E402
    COLOR_FAIL,
    COLOR_OK,
    COLOR_WARN,
    cumulative_pulses,
    isi_histogram,
    light_cycle_strip,
    pulse_train_raster,
)
from frontend.data import (  # noqa: E402
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_experiments,
    load_sync_report,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.sync_report")

st.title("Sync Diagnostics Report")
st.caption(
    "Per-session synchronisation classification across the cohort. "
    "See the methodology expander at the bottom for the algorithm."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def _load_h5_from_s3(bucket: str, key: str) -> dict | None:
    """Download an HDF5 file from S3 and return datasets + attrs as a dict."""
    import h5py

    data = download_s3_bytes(bucket, key)
    if data is None:
        return None
    try:
        f = h5py.File(io.BytesIO(data), "r")
        result: dict = {}
        for k in f:
            try:
                result[k] = f[k][:]
            except Exception:
                result[k] = None
        for k, v in f.attrs.items():
            result[f"_attr_{k}"] = v
        f.close()
        return result
    except Exception:
        log.exception("Error reading HDF5 from s3://%s/%s", bucket, key)
        return None


def _decode(val: object) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _status_color(status: str) -> str:
    if status == "OK":
        return COLOR_OK
    if status == "OK_WITH_WARNINGS":
        return COLOR_WARN
    if status.startswith("FAILED_"):
        return COLOR_FAIL
    return "#6b7280"


def _render_status_chip(status: str) -> str:
    color = _status_color(status)
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;">{status}</span>'
    )


# ---------------------------------------------------------------------------
# Section A — Summary table
# ---------------------------------------------------------------------------

st.subheader("Summary")

report_df = load_sync_report()
experiments = load_experiments()
exclude_map = {e["exp_id"]: str(e.get("exclude", "0")) for e in experiments}
notes_map = {e["exp_id"]: e.get("Notes", "") for e in experiments}

if report_df is None or report_df.empty:
    st.info(
        "Sync report not yet built. Run `snakemake sync_report` (Stage 5b) "
        "to populate `sync_report.parquet` on S3."
    )
else:
    # Append exclude flag from experiments.csv.
    df = report_df.copy()
    df["excluded"] = df["exp_id"].map(exclude_map).fillna("0")
    # Ordered column subset for display.
    display_cols = [
        "exp_id",
        "sync_status",
        "cam_n_pulses",
        "img_n_pulses",
        "n_tiff_frames",
        "pulse_count_diff_after_off_by_one",
        "cam_isi_cv",
        "img_isi_cv",
        "cam_drift_slope_ppm",
        "light_period_median_s",
        "excluded",
    ]
    # Map legacy alias for cleaner column names.
    df_display = df[display_cols].copy()
    df_display = df_display.rename(
        columns={
            "cam_n_pulses": "cam_n",
            "img_n_pulses": "img_n",
            "n_tiff_frames": "tiff_n",
            "pulse_count_diff_after_off_by_one": "Δ frames",
            "cam_isi_cv": "cam ISI CV",
            "img_isi_cv": "img ISI CV",
            "cam_drift_slope_ppm": "drift cam (ppm)",
            "light_period_median_s": "light period (s)",
        }
    )
    df_display["# warnings"] = (
        df["sync_warnings"].apply(lambda x: len(json.loads(x)) if x else 0).astype(int)
    )
    # Sort by failures first, then warnings.
    df_display["__sort_status"] = df_display["sync_status"].apply(
        lambda s: 0 if s.startswith("FAILED_") else (1 if s == "OK_WITH_WARNINGS" else 2)
    )
    df_display = (
        df_display.sort_values(
            ["__sort_status", "# warnings", "exp_id"], ascending=[True, False, True]
        )
        .drop(columns="__sort_status")
        .reset_index(drop=True)
    )

    n_total = len(df_display)
    n_ok = (df_display["sync_status"] == "OK").sum()
    n_warn = (df_display["sync_status"] == "OK_WITH_WARNINGS").sum()
    n_fail = (df_display["sync_status"].str.startswith("FAILED_")).sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", n_total)
    col2.metric("OK", int(n_ok))
    col3.metric("OK_WITH_WARNINGS", int(n_warn))
    col4.metric("FAILED_*", int(n_fail))

    st.dataframe(df_display, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Section B — Aggregate panels
# ---------------------------------------------------------------------------

st.subheader("Aggregate distributions")

if report_df is None or report_df.empty:
    st.caption("No data — populate `sync_report.parquet` first.")
else:
    import plotly.express as px

    df = report_df.copy()
    # Stacked bar of sync_status counts.
    status_counts = df["sync_status"].value_counts().reset_index()
    status_counts.columns = ["sync_status", "count"]
    fig_status = px.bar(
        status_counts,
        x="sync_status",
        y="count",
        color="sync_status",
        color_discrete_map={
            "OK": COLOR_OK,
            "OK_WITH_WARNINGS": COLOR_WARN,
        },
    )
    fig_status.update_layout(height=280, showlegend=False, margin=dict(l=40, r=20, t=20, b=40))

    fig_diff = px.histogram(
        df,
        x="pulse_count_diff_after_off_by_one",
        nbins=21,
        range_x=[-10, 10],
        title=None,
    )
    fig_diff.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40))

    fig_cv_cam = px.histogram(df, x="cam_isi_cv", nbins=40, log_y=True)
    fig_cv_cam.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40))

    fig_cv_img = px.histogram(df, x="img_isi_cv", nbins=40, log_y=True)
    fig_cv_img.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40))

    fig_drift = px.scatter(
        df,
        x="cam_drift_slope_ppm",
        y="img_drift_slope_ppm",
        color="sync_status",
        color_discrete_map={
            "OK": COLOR_OK,
            "OK_WITH_WARNINGS": COLOR_WARN,
        },
        hover_data=["exp_id"],
    )
    fig_drift.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40))

    fig_light = px.histogram(df, x="light_period_median_s", nbins=30)
    fig_light.add_vrect(x0=100, x1=140, fillcolor="#fde68a", opacity=0.3, line_width=0)
    fig_light.update_layout(height=280, margin=dict(l=40, r=20, t=20, b=40))

    a1, a2 = st.columns(2)
    a1.plotly_chart(fig_status, use_container_width=True)
    a2.plotly_chart(fig_diff, use_container_width=True)
    a3, a4 = st.columns(2)
    a3.plotly_chart(fig_cv_cam, use_container_width=True)
    a4.plotly_chart(fig_cv_img, use_container_width=True)
    a5, a6 = st.columns(2)
    a5.plotly_chart(fig_drift, use_container_width=True)
    a6.plotly_chart(fig_light, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section C — Per-session deep-dive
# ---------------------------------------------------------------------------

st.subheader("Per-session deep-dive")

# Session selector — inline at top of section, not in sidebar.
if experiments:
    exp_ids = [e["exp_id"] for e in experiments]
else:
    exp_ids = []

default_idx = 0
if "selected_exp_id" in st.session_state:
    sel = st.session_state["selected_exp_id"]
    if sel in exp_ids:
        default_idx = exp_ids.index(sel)
selected_exp = st.selectbox(
    "Session", exp_ids, index=default_idx if exp_ids else 0, key="sync_report_exp"
)

if selected_exp:
    sub, ses = parse_session_id(selected_exp)
    sync_key = f"sync/{sub}/{ses}/sync.h5"
    ts_key = f"timestamps/{sub}/{ses}/timestamps.h5"
    sync = _load_h5_from_s3(DERIVATIVES_BUCKET, sync_key)
    ts = _load_h5_from_s3(DERIVATIVES_BUCKET, ts_key)

    if sync is None:
        st.warning(
            "No `sync.h5` available for this session. "
            "Stage 5 has not yet produced an output (or the file is unreachable)."
        )

    # --- Header / verdict block ---
    excluded = exclude_map.get(selected_exp, "0") == "1"
    notes = notes_map.get(selected_exp, "")
    sync_status = (
        _decode(sync["_attr_sync_status"]) if (sync and "_attr_sync_status" in sync) else "UNKNOWN"
    )
    sync_warnings_raw = (
        _decode(sync["_attr_sync_warnings"]) if (sync and "_attr_sync_warnings" in sync) else "[]"
    )
    sync_failures_raw = (
        _decode(sync["_attr_sync_failures"]) if (sync and "_attr_sync_failures" in sync) else "[]"
    )
    try:
        sync_warnings = json.loads(sync_warnings_raw)
    except Exception:
        sync_warnings = []
    try:
        sync_failures = json.loads(sync_failures_raw)
    except Exception:
        sync_failures = []

    chip = _render_status_chip(sync_status)
    st.markdown(
        f"### {selected_exp} {chip}",
        unsafe_allow_html=True,
    )
    st.caption(f"`{sub}/{ses}`")
    if excluded:
        st.warning(f"This session is marked `exclude=1` in experiments.csv. Notes: {notes}")
    elif notes:
        st.caption(f"Notes: {notes}")

    if sync_status.startswith("FAILED_"):
        st.error(
            "This session failed sync verification — data is not used by Stage 6 unless an "
            "override is set. Pulse-train and cumulative-count plots remain available "
            "below for diagnosis. Re-run Stage 5 once the underlying issue is resolved."
        )

    # Verdict bullet list.
    if sync_failures or sync_warnings:
        if sync_failures:
            st.markdown("**Failures**")
            for f in sync_failures:
                st.markdown(f"- `{f}`")
        if sync_warnings:
            st.markdown("**Warnings**")
            for w in sync_warnings:
                st.markdown(f"- `{w}`")

    # --- Pulse-train raster + cumulative count ---
    if ts is None:
        st.info(
            "No `timestamps.h5` for this session. Stage 0 has not yet produced one — "
            "pulse-train plots are unavailable."
        )
    else:
        cam = ts.get("frame_times_camera")
        img = ts.get("frame_times_imaging")
        line = ts.get("line_clock_times")
        light_on = ts.get("light_on_times")
        light_off = ts.get("light_off_times")
        channels: dict[str, np.ndarray] = {}
        if cam is not None:
            channels["camera"] = cam
        if img is not None:
            channels["imaging"] = img
        if line is not None:
            channels["line clock"] = line
        if channels:
            st.markdown("**Pulse-train raster**")
            fig_raster = pulse_train_raster(
                channels,
                light_on=light_on,
                light_off=light_off,
            )
            st.plotly_chart(fig_raster, use_container_width=True)

            st.markdown("**Cumulative pulse count**")
            fig_cum = cumulative_pulses(channels)
            st.plotly_chart(fig_cum, use_container_width=True)

            st.markdown("**ISI histograms**")
            fps_cam = float(ts.get("_attr_fps_camera", 100.0))
            fps_img = float(ts.get("_attr_fps_imaging", 30.0))
            i1, i2, i3 = st.columns(3)
            if cam is not None:
                with i1:
                    st.plotly_chart(isi_histogram(cam, fps_cam), use_container_width=True)
            if img is not None:
                with i2:
                    st.plotly_chart(isi_histogram(img, fps_img), use_container_width=True)
            if line is not None:
                with i3:
                    st.plotly_chart(isi_histogram(line, fps_img * 162), use_container_width=True)

            if light_on is not None and light_off is not None and cam is not None:
                st.markdown("**Light cycle strip**")
                t_max = float(cam.max()) if cam.size else 0.0
                fig_light = light_cycle_strip(light_on, light_off, t_max)
                st.plotly_chart(fig_light, use_container_width=True)

    # --- Diagnostic scalars table ---
    if sync is not None:
        diag_rows = []
        for k, v in sync.items():
            if not isinstance(k, str) or not k.startswith("_attr_sync_diag/"):
                continue
            field = k.replace("_attr_sync_diag/", "")
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            diag_rows.append((field, v))
        if diag_rows:
            st.markdown("**Diagnostic scalars**")
            import pandas as pd

            diag_df = pd.DataFrame(diag_rows, columns=["scalar", "value"])
            st.dataframe(diag_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Methods & References — verbatim from design §4.5
# ---------------------------------------------------------------------------

with st.expander("Methods & references"):
    st.markdown(  # noqa: E501
        """
**What this page checks.** Each session has two timing streams produced by
National Instruments DAQ: a camera-trigger pulse train (~100 Hz) and the
SciScan two-photon line-clock pulse train (~9.6 Hz × y_pix lines per frame).
Light-state edges and TIFF frame counts are also recorded. We compute
non-parametric scalar statistics on each stream — median inter-pulse interval
(ISI), median absolute deviation (MAD), coefficient of variation
(CV = MAD / median), linear drift slope (least-squares regression of pulse
index → pulse time, expressed as parts per million away from nominal), and
cross-stream metrics (start-offset, end-offset, overlap fraction).

**Classification.** Sessions are classified into one of seven `sync_status`
tiers. The first matching predicate wins; thresholds are configured in
`config/sync.yaml`. A `FAILED_*` status means the session is excluded from
Stage 6 analysis by default; an `OK_WITH_WARNINGS` status means the data
is usable but a non-blocking warning was raised.

**Why non-parametric.** Pulse-train artefacts (single dropped frames,
duplicate pulses, transient jitter) produce heavy-tailed ISI distributions.
Median + MAD are insensitive to such outliers, whereas mean + SD are not.
This is the same rationale documented in `docs/stats-strategy.md`.

**Frame-count sanity check.** Suite2p's `ops.npy` records the TIFF frame
count post-extraction. A 1-frame mismatch (`|img_n_pulses − tiff_n| == 1`)
is a known SciScan edge case (the line clock occasionally records one
extra final-frame pulse) and is corrected by `align.run` before
resampling; larger mismatches indicate genuine pulse loss.

**Light protocol.** The room lights follow a 60 s on / 60 s off cycle
(Lyons & Foster 2024 chronobiology protocol; period 120 s). The expected
phase at t = 0 (lights on or lights off) is recorded in
`config/sync.yaml`; sessions whose first observed edge does not match
emit a `light_phase_unknown` warning rather than failing.

**References.**

- Pnevmatikakis et al. 2017. *Neuron* 89(2):285. doi:10.1016/j.neuron.2015.11.037 — frame-count alignment in two-photon pipelines.
- Tukey, J. W. 1977. *Exploratory Data Analysis* — MAD and median for outlier-robust dispersion.
- The `nptdms` library — National Instruments TDMS file format
  documentation. https://github.com/adamreeve/npTDMS.
"""
    )
