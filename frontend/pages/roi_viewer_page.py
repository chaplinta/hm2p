"""ROI Viewer — fast single-ROI inspection across all sessions.

Browse ALL ROIs across ALL sessions with arrow-key navigation.
A flat global index cycles through every ROI in every session.
Data is cached per session so switching ROIs is instant.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_all_ca_data,
    load_all_suite2p_spatial,
)

log = logging.getLogger("hm2p.frontend.roi_viewer")

ROI_TYPE_NAMES = {0: "Soma", 1: "Dendrite", 2: "Non-cell"}
ROI_TYPE_COLORS = {0: "turquoise", 1: "darkorchid", 2: "gray"}

st.title("ROI Viewer")
st.caption("Browse all ROIs across all sessions. Click the index and use arrow keys.")


# ── Cached loaders ───────────────────────────────────────────────────────


@st.cache_data(ttl=1800, show_spinner="Loading calcium data...")
def _load_ca():
    return load_all_ca_data()


@st.cache_data(ttl=1800, show_spinner="Loading Suite2p spatial data...")
def _load_spatial():
    return load_all_suite2p_spatial()


@st.cache_data(ttl=1800, show_spinner="Loading raw fluorescence...")
def _load_raw_f(sub: str, ses: str) -> dict | None:
    prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0"
    f_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/F.npy")
    fneu_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/Fneu.npy")
    iscell_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/iscell.npy")
    if f_data is None:
        return None
    F_all = np.load(io.BytesIO(f_data))
    Fneu_all = np.load(io.BytesIO(fneu_data)) if fneu_data else None
    if iscell_data is not None:
        iscell = np.load(io.BytesIO(iscell_data))
        mask = iscell[:, 0].astype(bool)
        F_all = F_all[mask]
        Fneu_all = Fneu_all[mask] if Fneu_all is not None else None
    return {"F": F_all, "Fneu": Fneu_all}


ca_sessions = _load_ca()
spatial_data = _load_spatial()

if not ca_sessions:
    st.warning("No calcium data available. Run Stage 4 first.")
    st.stop()


# ── Sidebar filters ──────────────────────────────────────────────────────

fc1, fc2 = st.columns(2)
with fc1:
    roi_filter = st.radio(
        "ROI type",
        ["Soma only", "Non-soma only", "All"],
        index=0,
        key="rv_type",
        horizontal=True,
    )
with fc2:
    session_filter = st.radio(
        "Sessions",
        ["Primary (non-excluded)", "All sessions"],
        index=0,
        key="rv_ses_filter",
        horizontal=True,
    )


# ── Build flat global ROI list ───────────────────────────────────────────
# Each entry: (session_index, roi_index_within_session)

roi_list: list[tuple[int, int]] = []

for si, ses in enumerate(ca_sessions):
    # Session filter
    if session_filter == "Primary (non-excluded)":
        if str(ses.get("exclude", "0")).strip() == "1":
            continue

    roi_types = ses.get("roi_types", np.zeros(ses["n_rois"], dtype=np.uint8))

    for ri in range(ses["n_rois"]):
        rt = int(roi_types[ri]) if ri < len(roi_types) else 0
        if roi_filter == "Soma only" and rt != 0:
            continue
        if roi_filter == "Non-soma only" and rt == 0:
            continue
        roi_list.append((si, ri))

if not roi_list:
    st.warning("No ROIs match the current filters.")
    st.stop()

n_total = len(roi_list)


# ── Global ROI selector ──────────────────────────────────────────────────

roi_pos = st.number_input(
    f"ROI index (0–{n_total - 1}, {n_total} total)",
    min_value=0,
    max_value=n_total - 1,
    value=0,
    step=1,
    key="rv_global",
    help="Click here then use ↑↓ arrow keys to browse all ROIs across all sessions",
)

ses_idx, roi_idx = roi_list[roi_pos]
ses = ca_sessions[ses_idx]
roi_types = ses.get("roi_types", np.zeros(ses["n_rois"], dtype=np.uint8))
roi_type = int(roi_types[roi_idx]) if roi_idx < len(roi_types) else 0
dff = ses["dff"][roi_idx]
frame_times = ses.get("frame_times")
t = (frame_times - frame_times[0]) if frame_times is not None else np.arange(len(dff)) / 9.8
duration_s = float(t[-1] - t[0]) if len(t) > 1 else 1.0

event_masks = ses.get("event_masks")
n_events = (
    int(event_masks[roi_idx].sum())
    if event_masks is not None and roi_idx < event_masks.shape[0]
    else 0
)
event_rate = n_events / duration_s if duration_s > 0 else 0

baseline_vals = dff[dff < np.nanpercentile(dff, 25)] if len(dff) > 10 else dff
baseline_std = float(np.nanstd(baseline_vals)) if len(baseline_vals) > 0 else 0
peak_dff = float(np.nanmax(dff)) if len(dff) > 0 else 0
snr = peak_dff / baseline_std if baseline_std > 0 else 0
celltype_label = "Penk+" if ses["celltype"] == "penk" else "Penk\u207bCamKII+"


# ── Metrics bar (always visible at top) ──────────────────────────────────

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Global #", f"{roi_pos}")
c2.metric("Session", ses["exp_id"].split("_")[-1])
c3.metric("ROI", roi_idx)
c4.metric("Type", ROI_TYPE_NAMES.get(roi_type, "?"))
c5.metric("Celltype", celltype_label)
c6.metric("SNR", f"{snr:.1f}")
c7.metric("Events", f"{n_events}")
c8.metric("Rate", f"{event_rate:.2f}/s")

st.markdown(
    f"<small style='color:gray'>{ses['exp_id']} &mdash; "
    f"peak dF/F₀ = {peak_dff:.2f}, baseline σ = {baseline_std:.4f}, "
    f"duration = {duration_s:.0f}s, {ses['n_rois']} ROIs in session</small>",
    unsafe_allow_html=True,
)


# ── Spatial images ───────────────────────────────────────────────────────

spatial = spatial_data.get(ses["exp_id"], {})
mean_img = spatial.get("mean_img")
max_img = spatial.get("max_img")
shape_features = spatial.get("shape_features", [])

_img_choice = st.radio(
    "Background image",
    ["Max projection", "Mean"],
    horizontal=True,
    key="rv_img_type",
)
bg_img = max_img if _img_choice == "Max projection" and max_img is not None else mean_img

col_mean, col_roi = st.columns(2)

if bg_img is not None:
    with col_mean:
        fig = go.Figure(
            data=go.Heatmap(
                z=bg_img,
                colorscale="gray",
                showscale=False,
            )
        )
        fig.update_layout(
            height=300,
            title=_img_choice,
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            margin=dict(t=30, b=5, l=5, r=5),
        )
        st.plotly_chart(fig, use_container_width=True, key="rv_mean")

    with col_roi:
        fig = go.Figure(
            data=go.Heatmap(
                z=bg_img,
                colorscale="gray",
                showscale=False,
            )
        )
        if roi_idx < len(shape_features) and shape_features[roi_idx] is not None:
            sf = shape_features[roi_idx]
            ypix = sf.get("ypix", [])
            xpix = sf.get("xpix", [])
            if len(xpix) > 0:
                color = ROI_TYPE_COLORS.get(roi_type, "yellow")
                fig.add_trace(
                    go.Scatter(
                        x=xpix,
                        y=ypix,
                        mode="markers",
                        marker=dict(size=3, color=color, opacity=0.8),
                        name=f"ROI {roi_idx}",
                    )
                )
        img_h, img_w = bg_img.shape[:2]
        fig.update_layout(
            height=300,
            title=f"ROI {roi_idx} ({ROI_TYPE_NAMES.get(roi_type, '?')})",
            xaxis=dict(range=[0, img_w], constrain="domain"),
            yaxis=dict(range=[img_h, 0], scaleanchor="x", constrain="domain"),
            margin=dict(t=30, b=5, l=5, r=5),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="rv_roi_img")
else:
    st.info("No spatial data for this session.")


# ── Time series ──────────────────────────────────────────────────────────

raw = _load_raw_f(ses["sub"], ses["ses"])

n_panels = 2
has_raw = raw is not None and raw["F"] is not None and roi_idx < raw["F"].shape[0]
has_spikes = (
    ses.get("spikes") is not None and roi_idx < (ses.get("spikes", np.empty((0,))).shape[0])
)
if has_raw:
    n_panels = 4
if has_spikes:
    n_panels += 1

fig = make_subplots(
    rows=n_panels,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[1] * n_panels,
)

row = 1

if has_raw:
    F_trace = raw["F"][roi_idx]
    Fneu_trace = (
        raw["Fneu"][roi_idx]
        if raw["Fneu"] is not None and roi_idx < raw["Fneu"].shape[0]
        else None
    )

    # Panel 1: Raw F + neuropil
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=F_trace,
            mode="lines",
            line=dict(color="gray", width=1),
            name="Raw F",
        ),
        row=row,
        col=1,
    )
    if Fneu_trace is not None:
        fig.add_trace(
            go.Scattergl(
                x=t,
                y=Fneu_trace,
                mode="lines",
                line=dict(color="green", width=1, dash="dot"),
                name="Neuropil",
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text="F (a.u.)", row=row, col=1)
    row += 1

    # Panel 2: F corrected + F0
    neuropil_coeff = 0.7
    F_corr = F_trace - neuropil_coeff * Fneu_trace if Fneu_trace is not None else F_trace
    safe_dff = np.where(np.abs(dff) > 0.01, dff, 0.01)
    F0_approx = F_corr / (safe_dff + 1.0)
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=F_corr,
            mode="lines",
            line=dict(color="steelblue", width=1),
            name="F corrected",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=F0_approx,
            mode="lines",
            line=dict(color="red", width=1.5),
            name="F₀ baseline",
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="F / F₀", row=row, col=1)
    row += 1

# Panel 3: dF/F₀ + events
fig.add_trace(
    go.Scattergl(
        x=t,
        y=dff,
        mode="lines",
        line=dict(color="royalblue", width=1),
        name="dF/F₀",
    ),
    row=row,
    col=1,
)
if event_masks is not None and roi_idx < event_masks.shape[0]:
    events = event_masks[roi_idx].astype(bool)
    if events.any():
        fig.add_trace(
            go.Scattergl(
                x=t[events],
                y=dff[events],
                mode="markers",
                marker=dict(size=3, color="red", symbol="circle"),
                name="Events",
            ),
            row=row,
            col=1,
        )
fig.update_yaxes(title_text="dF/F₀", row=row, col=1)
row += 1

# Panel 4: Deconvolved (normalized)
deconv_norm = ses.get("deconv_norm")
if deconv_norm is not None and roi_idx < deconv_norm.shape[0]:
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=deconv_norm[roi_idx],
            mode="lines",
            line=dict(color="darkorange", width=1),
            name="Deconv (norm)",
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="Deconv", row=row, col=1)
else:
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=np.zeros_like(dff),
            mode="lines",
            line=dict(color="lightgray", width=0.5),
            name="(no deconv)",
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="(no deconv)", row=row, col=1)

# Panel: CASCADE spikes (if available)
row += 1
spikes = ses.get("spikes")
if has_spikes and spikes is not None and roi_idx < spikes.shape[0]:
    spike_trace = np.nan_to_num(spikes[roi_idx])
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=spike_trace,
            mode="lines",
            line=dict(color="#d62728", width=1),
            name="CASCADE spikes",
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(title_text="Spk/s", row=row, col=1)

fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
fig.update_layout(
    height=130 * n_panels + 40,
    margin=dict(t=10, b=35, l=55, r=15),
    showlegend=True,
    legend=dict(orientation="h", y=-0.06),
)
st.plotly_chart(fig, use_container_width=True, key="rv_traces")


# ── Per-ROI QC table and histograms ──────────────────────────────────────

st.subheader("ROI Quality Control")
st.caption(
    "Metrics computed by Stage 4. Thresholds are recommendations only — "
    "no ROIs are excluded automatically."
)

roi_qc_ses = ses.get("roi_qc")  # dict of 1D arrays keyed by short name, or None

if roi_qc_ses is None:
    st.info("No QC data available for this session. Run Stage 4 to generate roi_qc metrics.")
else:
    import pandas as pd

    from hm2p.calcium.qc import (
        ACTIVE_FRAC_MIN,
        BLEACH_MAX_LOSS,
        FNEU_CORR_MAX,
        SNR_MIN,
        TAU_MAX_S,
        TAU_MIN_S,
    )

    # Build DataFrame for this session's QC metrics.
    # ``p_soma`` / ``p_dend`` / ``p_artefact`` are written by the soma
    # classifier framework (hm2p.extraction.soma_classifier).  When they
    # are absent (e.g. older ca.h5 files), the column is filled with NaN
    # and rendered as "—" in the table.
    n_rois_ses = ses["n_rois"]
    qc_df = pd.DataFrame(
        {
            "roi_index": roi_qc_ses.get("roi_index", np.arange(n_rois_ses, dtype=np.int32)),
            "snr_event": roi_qc_ses.get(
                "snr_event", np.full(n_rois_ses, np.nan, dtype=np.float32)
            ),
            "decay_tau_s": roi_qc_ses.get(
                "decay_tau_s", np.full(n_rois_ses, np.nan, dtype=np.float32)
            ),
            "fneu_dff_corr": roi_qc_ses.get(
                "fneu_dff_corr", np.full(n_rois_ses, np.nan, dtype=np.float32)
            ),
            "bleach_slope": roi_qc_ses.get(
                "bleach_slope", np.full(n_rois_ses, np.nan, dtype=np.float32)
            ),
            "active_fraction": roi_qc_ses.get(
                "active_fraction", np.full(n_rois_ses, np.nan, dtype=np.float32)
            ),
            "p_soma": roi_qc_ses.get("p_soma", np.full(n_rois_ses, np.nan, dtype=np.float32)),
            "p_dend": roi_qc_ses.get("p_dend", np.full(n_rois_ses, np.nan, dtype=np.float32)),
            "p_artefact": roi_qc_ses.get(
                "p_artefact", np.full(n_rois_ses, np.nan, dtype=np.float32)
            ),
        }
    )

    # Flag criteria (True = fails threshold; NaN = unknown, not flagged)
    def _flag_col(series, bad_fn):
        return np.where(series.notna(), bad_fn(series.fillna(0.0)), False)

    qc_df["flag_snr"] = _flag_col(qc_df["snr_event"], lambda x: x < SNR_MIN)
    qc_df["flag_tau"] = _flag_col(
        qc_df["decay_tau_s"], lambda x: (x < TAU_MIN_S) | (x > TAU_MAX_S)
    )
    qc_df["flag_fneu"] = _flag_col(qc_df["fneu_dff_corr"], lambda x: x > FNEU_CORR_MAX)
    qc_df["flag_bleach"] = _flag_col(qc_df["bleach_slope"], lambda x: x < BLEACH_MAX_LOSS)
    qc_df["flag_active"] = _flag_col(qc_df["active_fraction"], lambda x: x < ACTIVE_FRAC_MIN)
    qc_df["n_flags"] = (
        qc_df["flag_snr"].astype(int)
        + qc_df["flag_tau"].astype(int)
        + qc_df["flag_fneu"].astype(int)
        + qc_df["flag_bleach"].astype(int)
        + qc_df["flag_active"].astype(int)
    )
    qc_df["flagged"] = qc_df["n_flags"] > 0

    # ── Soma classifier probabilities for the current ROI ────────────────
    if roi_idx < len(qc_df):
        row_classifier = qc_df.iloc[roi_idx]
        p_soma_val = row_classifier.get("p_soma", float("nan"))
        p_dend_val = row_classifier.get("p_dend", float("nan"))
        p_art_val = row_classifier.get("p_artefact", float("nan"))
        if any(np.isfinite([p_soma_val, p_dend_val, p_art_val])):
            cps1, cps2, cps3 = st.columns(3)
            cps1.metric(
                "p_soma",
                "—" if not np.isfinite(p_soma_val) else f"{p_soma_val:.3f}",
            )
            cps2.metric(
                "p_dend",
                "—" if not np.isfinite(p_dend_val) else f"{p_dend_val:.3f}",
            )
            cps3.metric(
                "p_artefact",
                "—" if not np.isfinite(p_art_val) else f"{p_art_val:.3f}",
            )

    # ── Callout for the currently viewed ROI ─────────────────────────────
    if roi_idx < len(qc_df):
        row_qc = qc_df.iloc[roi_idx]
        if row_qc["flagged"]:
            failed = []
            if row_qc["flag_snr"]:
                failed.append(f"SNR {row_qc['snr_event']:.1f} < {SNR_MIN:.1f}")
            if row_qc["flag_tau"]:
                failed.append(
                    f"tau {row_qc['decay_tau_s']:.2f} s outside [{TAU_MIN_S}, {TAU_MAX_S}] s"
                )
            if row_qc["flag_fneu"]:
                failed.append(f"Fneu corr {row_qc['fneu_dff_corr']:.2f} > {FNEU_CORR_MAX:.2f}")
            if row_qc["flag_bleach"]:
                failed.append(f"bleach slope {row_qc['bleach_slope']:.2f} < {BLEACH_MAX_LOSS:.2f}")
            if row_qc["flag_active"]:
                failed.append(
                    f"active fraction {row_qc['active_fraction']:.3f} < {ACTIVE_FRAC_MIN:.3f}"
                )
            st.warning(
                f"ROI {roi_idx} fails {row_qc['n_flags']} QC threshold(s): "
                + "; ".join(failed)
                + ". This ROI is not excluded — review manually."
            )
        else:
            st.success(f"ROI {roi_idx} passes all QC thresholds.")

    # ── Session-level summary ─────────────────────────────────────────────
    n_flagged_ses = int(qc_df["flagged"].sum())
    st.markdown(f"**Session:** {n_flagged_ses} / {n_rois_ses} ROIs fail at least one threshold.")

    # ── Per-metric histograms ─────────────────────────────────────────────
    with st.expander("Per-metric histograms (session)", expanded=False):
        metrics_hist = [
            ("snr_event", "SNR (event)", "Events-based SNR", SNR_MIN, None),
            ("decay_tau_s", "Decay tau (s)", "GCaMP decay time constant", TAU_MIN_S, TAU_MAX_S),
            (
                "fneu_dff_corr",
                "Fneu–dF/F corr",
                "Spearman r (neuropil contamination)",
                None,
                FNEU_CORR_MAX,
            ),
            (
                "bleach_slope",
                "Bleach slope",
                "Fractional fluorescence change",
                BLEACH_MAX_LOSS,
                None,
            ),
            (
                "active_fraction",
                "Active fraction",
                "Fraction of frames with detected events",
                ACTIVE_FRAC_MIN,
                None,
            ),
        ]

        _ncols = 3
        cols_hist = st.columns(_ncols)
        for _idx, (col_name, label, hover_title, thresh_lo, thresh_hi) in enumerate(metrics_hist):
            vals = qc_df[col_name].dropna().values
            with cols_hist[_idx % _ncols]:
                fig_h = go.Figure()
                fig_h.add_trace(
                    go.Histogram(
                        x=vals,
                        nbinsx=30,
                        marker_color="steelblue",
                        opacity=0.75,
                        name=label,
                    )
                )
                # Mark the currently viewed ROI
                if roi_idx < len(qc_df):
                    roi_val = float(qc_df.iloc[roi_idx][col_name])
                    if np.isfinite(roi_val):
                        fig_h.add_vline(
                            x=roi_val,
                            line_color="royalblue",
                            line_dash="solid",
                            line_width=2,
                            annotation_text=f"ROI {roi_idx}",
                            annotation_position="top right",
                        )
                # Threshold lines
                if thresh_lo is not None:
                    fig_h.add_vline(
                        x=thresh_lo,
                        line_color="red",
                        line_dash="dash",
                        line_width=1.5,
                        annotation_text="threshold",
                        annotation_position="top left",
                    )
                if thresh_hi is not None:
                    fig_h.add_vline(
                        x=thresh_hi,
                        line_color="red",
                        line_dash="dash",
                        line_width=1.5,
                        annotation_text="threshold",
                        annotation_position="top right",
                    )
                fig_h.update_layout(
                    title=dict(text=label, font_size=12),
                    xaxis_title=hover_title,
                    yaxis_title="# ROIs",
                    height=220,
                    margin=dict(t=30, b=30, l=30, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_h, use_container_width=True, key=f"rv_hist_{col_name}")

    # ── QC table (all ROIs in session) ────────────────────────────────────
    with st.expander("Full QC table (all ROIs in session)", expanded=False):
        st.caption(
            "Sort by **p_soma** ascending to surface ambiguous ROIs that the "
            "soma classifier is uncertain about — these are the candidates "
            "for manual curation. The classifier framework lives in "
            "`hm2p.extraction.soma_classifier`; see `docs/soma-classifier.md`."
        )
        display_cols = [
            "roi_index",
            "p_soma",
            "p_dend",
            "p_artefact",
            "snr_event",
            "decay_tau_s",
            "fneu_dff_corr",
            "bleach_slope",
            "active_fraction",
            "n_flags",
        ]
        display_df = qc_df[display_cols].copy()
        display_df = display_df.rename(
            columns={
                "roi_index": "ROI",
                "p_soma": "p_soma",
                "p_dend": "p_dend",
                "p_artefact": "p_artefact",
                "snr_event": "SNR",
                "decay_tau_s": "Tau (s)",
                "fneu_dff_corr": "Fneu corr",
                "bleach_slope": "Bleach slope",
                "active_fraction": "Active frac",
                "n_flags": "# flags",
            }
        )

        # Highlight flagged rows in the table
        def _highlight_flags(row):
            color = "background-color: #fff3cd" if row["# flags"] > 0 else ""
            return [color] * len(row)

        st.dataframe(
            display_df.style.apply(_highlight_flags, axis=1).format(
                {
                    "p_soma": "{:.3f}",
                    "p_dend": "{:.3f}",
                    "p_artefact": "{:.3f}",
                    "SNR": "{:.2f}",
                    "Tau (s)": "{:.3f}",
                    "Fneu corr": "{:.3f}",
                    "Bleach slope": "{:.3f}",
                    "Active frac": "{:.3f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Threshold criteria: SNR ≥ {SNR_MIN}, tau in [{TAU_MIN_S}, {TAU_MAX_S}] s, "
            f"Fneu corr ≤ {FNEU_CORR_MAX}, bleach slope ≥ {BLEACH_MAX_LOSS}, "
            f"active fraction ≥ {ACTIVE_FRAC_MIN}. "
            "Yellow rows fail at least one criterion."
        )


# ── Methods & References ─────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
**ROI quality control metrics** — computed by Stage 4 (`hm2p.calcium.qc`):

- **SNR (event-based):** mean amplitude of detected calcium events divided by the
  standard deviation of dF/F outside events.
  Pnevmatikakis et al. 2016. "Simultaneous Denoising, Deconvolution, and Demixing
  of Calcium Imaging Data." *Neuron* 89(2):285–299. doi:10.1016/j.neuron.2015.11.037

- **Decay tau:** median exponential time constant (A·e^(−t/τ) + C) fitted to the
  post-peak decay of each detected calcium event. NaN if fewer than 3 events.

- **Fneu correlation:** Spearman rank correlation between the ROI's dF/F trace and
  the mean neuropil (Fneu) signal across all ROIs. High values indicate residual
  neuropil contamination. Non-parametric (Spearman) per project statistics policy.

- **Bleach slope:** fractional change in mean raw fluorescence from the first to the
  last 10 % of frames: (F_end − F_start) / F_start. Negative = photobleaching loss.

- **Active fraction:** fraction of imaging frames where the V&H event mask is 1.
  Falls back to fraction of frames where dF/F > 3·MAD if events are unavailable.
  Voigts & Harnett 2020. "Somatic and dendritic encoding of spatial variables in
  retrosplenial cortex differs during 2D navigation." *Neuron* 105(2):237–245.
  doi:10.1016/j.neuron.2019.10.016

**Suite2p ROI detection** with Cellpose 3 anatomical prior (anatomical_only=2):
Stringer & Pachitariu 2025. "Cellpose3: one-click image restoration for improved
cellular segmentation." *Nature Methods*. doi:10.1038/s41592-025-02595-5.
GitHub: https://github.com/MouseLand/cellpose

**Soma / dendrite / artefact probabilities** (`p_soma`, `p_dend`, `p_artefact`):
classifier framework in `hm2p.extraction.soma_classifier`. The current default
is a *provisional rule-based scorer* whose argmax exactly reproduces the legacy
shape-only thresholds — switching to it does not relabel any existing ROI. Once
~200 manual labels have been curated in Suite2p's GUI, a logistic-regression
classifier can be trained via `scripts/train_soma_classifier.py` and dropped in
at `sourcedata/trackers/suite2p/soma_classifier.pkl`. See `docs/soma-classifier.md`
for the full workflow.
Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python." *JMLR*
12:2825–2830. https://scikit-learn.org
""")


# ── Footer ───────────────────────────────────────────────────────────────

st.caption(
    "Click the ROI index input and use **↑↓ arrow keys** to cycle through all ROIs "
    "across all sessions. Session data is cached — switching is instant."
)
