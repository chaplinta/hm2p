"""Animal Summary — per-animal overview grouping sessions by mouse."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)
from hm2p.constants import CELLTYPE_HEX

log = logging.getLogger("hm2p.frontend.animals")

st.title("Animal Summary")
st.caption("Per-animal overview — sessions, ROI counts, and quality grouped by mouse.")

experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

# Build per-animal session count
animal_sessions = {}
for exp in experiments:
    animal_id = exp["exp_id"].split("_")[-1]
    if animal_id not in animal_sessions:
        animal_sessions[animal_id] = []
    animal_sessions[animal_id].append(exp)


@st.cache_data(ttl=600)
def load_animal_roi_counts() -> dict[str, dict]:
    """Load ROI counts per session from ca.h5 files."""
    import h5py

    counts = {}
    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if data is None:
            counts[exp_id] = {"n_rois": 0, "median_snr": 0, "mean_event_rate": 0}
            continue
        try:
            f = h5py.File(io.BytesIO(data), "r")
            dff = f["dff"][:]
            n_rois, n_frames = dff.shape
            fps = float(f.attrs.get("fps_imaging", 9.8))

            snrs = []
            event_rates = []
            for i in range(n_rois):
                trace = dff[i]
                baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
                peak = np.percentile(trace, 95)
                snrs.append(peak / baseline_std if baseline_std > 0 else 0)

                if "event_masks" in f:
                    em = f["event_masks"][i].astype(bool)
                    onsets = np.flatnonzero(em[1:] & ~em[:-1])
                    n_events = len(onsets) + (1 if em[0] else 0)
                    event_rates.append(n_events / (n_frames / fps / 60))

            counts[exp_id] = {
                "n_rois": n_rois,
                "median_snr": float(np.median(snrs)) if snrs else 0,
                "mean_event_rate": float(np.mean(event_rates)) if event_rates else 0,
            }
            f.close()
        except Exception:
            counts[exp_id] = {"n_rois": 0, "median_snr": 0, "mean_event_rate": 0}

    return counts


with st.spinner("Loading ROI data..."):
    roi_counts = load_animal_roi_counts()

# Build animal summary table
rows = []
for animal_id in sorted(animal_sessions.keys()):
    animal = animal_map.get(animal_id, {})
    sessions = animal_sessions[animal_id]

    total_rois = sum(roi_counts.get(e["exp_id"], {}).get("n_rois", 0) for e in sessions)
    session_snrs = [roi_counts.get(e["exp_id"], {}).get("median_snr", 0) for e in sessions]
    session_rates = [roi_counts.get(e["exp_id"], {}).get("mean_event_rate", 0) for e in sessions]

    rows.append({
        "animal_id": animal_id,
        "celltype": animal.get("celltype", "?"),
        "strain": animal.get("strain", "?"),
        "gcamp": animal.get("gcamp", "?"),
        "hemisphere": animal.get("hemisphere", "?"),
        "sex": animal.get("sex", "?"),
        "n_sessions": len(sessions),
        "total_rois": total_rois,
        "mean_session_snr": float(np.mean(session_snrs)) if session_snrs else 0,
        "mean_event_rate": float(np.mean(session_rates)) if session_rates else 0,
    })

df = pd.DataFrame(rows)

# --- Summary ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Animals", len(df))
col2.metric("Penk+", len(df[df["celltype"] == "penk"]))
col3.metric("Non-Penk", len(df[df["celltype"] == "nonpenk"]))
col4.metric("Total ROIs", df["total_rois"].sum())

# --- Animal table ---
st.subheader("Animal Table")

def style_celltype(val):
    if val == "penk":
        return "background-color: #d4edda"
    elif val == "nonpenk":
        return "background-color: #cce5ff"
    return ""

styled = df.style.map(style_celltype, subset=["celltype"])
st.dataframe(styled, use_container_width=True, height=400)

# --- Per-animal visualizations ---
st.subheader("Per-Animal Comparison")

import plotly.express as px
import plotly.graph_objects as go

tab_rois, tab_quality, tab_sessions = st.tabs(["ROI Counts", "Quality", "Session List"])

with tab_rois:
    fig = px.bar(
        df.sort_values("celltype"),
        x="animal_id",
        y="total_rois",
        color="celltype",
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        title="Total ROIs per Animal",
        hover_data=["n_sessions", "mean_session_snr"],
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ROIs per session
    fig2 = px.bar(
        df.sort_values("celltype"),
        x="animal_id",
        y=df["total_rois"] / df["n_sessions"].replace(0, 1),
        color="celltype",
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        title="Mean ROIs per Session",
    )
    fig2.update_layout(height=350, yaxis_title="ROIs / session")
    st.plotly_chart(fig2, use_container_width=True)

with tab_quality:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df.sort_values("mean_session_snr", ascending=False),
            x="animal_id",
            y="mean_session_snr",
            color="celltype",
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="Mean Session SNR per Animal",
        )
        fig.add_hline(y=3, line_dash="dash", line_color="red", opacity=0.5)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df.sort_values("mean_event_rate", ascending=False),
            x="animal_id",
            y="mean_event_rate",
            color="celltype",
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="Mean Event Rate per Animal",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Penk vs non-Penk comparison
    st.subheader("Penk vs Non-Penk (per animal)")
    penk_rois = df[df["celltype"] == "penk"]["total_rois"]
    nonpenk_rois = df[df["celltype"] == "nonpenk"]["total_rois"]

    if len(penk_rois) > 0 and len(nonpenk_rois) > 0:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(
            df[df["celltype"] == "penk"]["mean_session_snr"],
            df[df["celltype"] == "nonpenk"]["mean_session_snr"],
            alternative="two-sided",
        )
        st.markdown(
            f"**SNR comparison:** Penk mean = {df[df['celltype']=='penk']['mean_session_snr'].mean():.2f}, "
            f"Non-Penk mean = {df[df['celltype']=='nonpenk']['mean_session_snr'].mean():.2f}, "
            f"Mann-Whitney p = {pval:.4f}"
        )

with tab_sessions:
    st.subheader("Sessions per Animal")

    selected_animal = st.selectbox(
        "Animal",
        df["animal_id"].tolist(),
        format_func=lambda x: f"{x} ({animal_map.get(x, {}).get('celltype', '?')})",
        key="animal_detail",
    )

    if selected_animal:
        sessions = animal_sessions.get(selected_animal, [])
        session_rows = []
        for exp in sessions:
            exp_id = exp["exp_id"]
            rc = roi_counts.get(exp_id, {})
            session_rows.append({
                "exp_id": exp_id,
                "date": exp_id[:8],
                "n_rois": rc.get("n_rois", 0),
                "median_snr": round(rc.get("median_snr", 0), 2),
                "event_rate": round(rc.get("mean_event_rate", 0), 1),
                "exclude": exp.get("exclude", "0"),
                "notes": exp.get("Notes", ""),
            })
        st.dataframe(pd.DataFrame(session_rows), use_container_width=True)

        # Animal metadata
        animal = animal_map.get(selected_animal, {})
        st.markdown("**Animal metadata:**")
        for key in ["celltype", "strain", "gcamp", "virus_id", "hemisphere", "sex"]:
            st.text(f"  {key}: {animal.get(key, '')}")
