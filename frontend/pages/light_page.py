"""Light Analysis — compare neural activity between light-on and light-off epochs."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.light")

st.title("Light/Dark Analysis")
st.caption("Compare neural activity between light-on and light-off epochs. No kinematics required.")

# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox(
    "Session",
    exp_ids,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="light_session",
)

if not selected:
    st.stop()

sub, ses = parse_session_id(selected)
animal_id = selected.split("_")[-1]
celltype = animal_map.get(animal_id, {}).get("celltype", "?")


@st.cache_data(ttl=300)
def load_light_data(sub: str, ses: str) -> dict | None:
    """Load ca.h5 and timestamps.h5 for light analysis."""
    import h5py

    # Load calcium data
    ca_data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if ca_data is None:
        return None

    f = h5py.File(io.BytesIO(ca_data), "r")
    result = {
        "dff": f["dff"][:],
        "fps": float(f.attrs.get("fps_imaging", 9.8)),
    }
    if "event_masks" in f:
        result["event_masks"] = f["event_masks"][:]
    if "spks" in f:
        result["spks"] = f["spks"][:]
    f.close()

    # Load timestamps for light cycle info
    ts_data = download_s3_bytes(DERIVATIVES_BUCKET, f"movement/{sub}/{ses}/timestamps.h5")
    if ts_data is None:
        return result  # Return calcium data without light info

    f = h5py.File(io.BytesIO(ts_data), "r")
    if "light_on_times" in f and "light_off_times" in f and "frame_times_imaging" in f:
        light_on_times = f["light_on_times"][:]
        light_off_times = f["light_off_times"][:]
        frame_times = f["frame_times_imaging"][:]

        # Trim frame_times to match dff shape (off-by-one fix)
        n_frames = result["dff"].shape[1]
        frame_times = frame_times[:n_frames]

        # Build per-frame light_on boolean
        light_on = np.zeros(n_frames, dtype=bool)
        for on_t in light_on_times:
            # Find next off time after this on time
            off_times_after = light_off_times[light_off_times > on_t]
            if len(off_times_after) > 0:
                off_t = off_times_after[0]
            else:
                off_t = frame_times[-1] + 1  # Light stays on till end

            mask = (frame_times >= on_t) & (frame_times < off_t)
            light_on[mask] = True

        result["light_on"] = light_on
        result["frame_times"] = frame_times
        result["light_on_times"] = light_on_times
        result["light_off_times"] = light_off_times

    f.close()
    return result


with st.spinner("Loading session data..."):
    data = load_light_data(sub, ses)

if data is None:
    st.warning("No calcium data found.")
    st.stop()

dff = data["dff"]
n_rois, n_frames = dff.shape
fps = data["fps"]
time_s = np.arange(n_frames) / fps

if "light_on" not in data:
    st.warning("No light cycle data available for this session (timestamps.h5 missing).")
    st.stop()

light_on = data["light_on"]
n_light = light_on.sum()
n_dark = (~light_on).sum()

st.markdown(f"**{sub} / {ses}** — {celltype} — {n_rois} ROIs")

# --- Summary ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Light ON frames", f"{n_light:,} ({n_light/n_frames*100:.1f}%)")
col2.metric("Light OFF frames", f"{n_dark:,} ({n_dark/n_frames*100:.1f}%)")
col3.metric("Light transitions", len(data.get("light_on_times", [])) + len(data.get("light_off_times", [])))
col4.metric("Cycle period", f"~{2 * n_frames / fps / max(1, len(data.get('light_on_times', []))):.0f}s")

# --- Tabs ---
tab_overview, tab_per_roi, tab_modulation, tab_population = st.tabs([
    "Overview", "Per-ROI", "Modulation Index", "Population",
])

import plotly.graph_objects as go
import plotly.express as px

# Compute per-ROI light/dark metrics
light_means = []
dark_means = []
light_event_rates = []
dark_event_rates = []
snrs = []

for i in range(n_rois):
    trace = dff[i]
    light_means.append(float(np.nanmean(trace[light_on])))
    dark_means.append(float(np.nanmean(trace[~light_on])))

    baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
    peak = np.percentile(trace, 95)
    snrs.append(peak / baseline_std if baseline_std > 0 else 0)

    if "event_masks" in data:
        em = data["event_masks"][i].astype(bool)
        light_frames = light_on
        dark_frames = ~light_on

        light_events = em & light_frames
        dark_events = em & dark_frames

        # Count events (onsets)
        light_onsets = np.flatnonzero(light_events[1:] & ~light_events[:-1])
        dark_onsets = np.flatnonzero(dark_events[1:] & ~dark_events[:-1])

        light_dur_s = n_light / fps
        dark_dur_s = n_dark / fps

        light_event_rates.append(len(light_onsets) / (light_dur_s / 60) if light_dur_s > 0 else 0)
        dark_event_rates.append(len(dark_onsets) / (dark_dur_s / 60) if dark_dur_s > 0 else 0)

light_means = np.array(light_means)
dark_means = np.array(dark_means)
snrs = np.array(snrs)

# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Light vs Dark Activity")

    # Mean dF/F comparison (paired scatter — same ROIs in both conditions)
    from hm2p.plotting import paired_condition_scatter, format_pvalue

    col1, col2 = st.columns(2)
    with col1:
        fig_dff, stat_dff = paired_condition_scatter(
            light_means, dark_means,
            "Light", "Dark", "Mean dF/F",
            height=400, width=400,
        )
        st.plotly_chart(fig_dff, use_container_width=True)
        st.markdown(
            f"Wilcoxon signed-rank: {format_pvalue(stat_dff['p'])}, n={stat_dff['n']}"
        )

    with col2:
        if light_event_rates:
            fig_ev, stat_ev = paired_condition_scatter(
                light_event_rates, dark_event_rates,
                "Light", "Dark", "Event Rate (events/min)",
                height=400, width=400,
            )
            st.plotly_chart(fig_ev, use_container_width=True)
            st.markdown(
                f"Wilcoxon signed-rank: {format_pvalue(stat_ev['p'])}, n={stat_ev['n']}"
            )

    # Light cycle timeline
    st.subheader("Light Cycle Timeline")
    ds = max(1, n_frames // 2000)
    fig = go.Figure()

    # Mean population trace
    mean_trace = dff.mean(axis=0)
    kernel = np.ones(max(1, int(fps))) / max(1, int(fps))
    mean_smooth = np.convolve(mean_trace, kernel, mode="same")

    fig.add_trace(go.Scatter(
        x=time_s[::ds], y=mean_smooth[::ds],
        mode="lines", name="Mean dF/F",
        line=dict(color="black", width=1),
    ))

    # Shade dark periods
    dark_starts = []
    dark_ends = []
    in_dark = not light_on[0]
    if in_dark:
        dark_starts.append(time_s[0])
    for i in range(1, n_frames):
        if not light_on[i] and light_on[i - 1]:
            dark_starts.append(time_s[i])
        elif light_on[i] and not light_on[i - 1]:
            dark_ends.append(time_s[i])
    if len(dark_starts) > len(dark_ends):
        dark_ends.append(time_s[-1])

    for ds_t, de_t in zip(dark_starts, dark_ends):
        fig.add_vrect(x0=ds_t, x1=de_t, fillcolor="rgba(50,50,50,0.15)", line_width=0, layer="below")

    fig.update_layout(
        height=300, title="Population Mean dF/F with Light Cycles",
        xaxis_title="Time (s)", yaxis_title="Mean dF/F",
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 2: Per-ROI ---
with tab_per_roi:
    st.subheader("Per-ROI Light vs Dark")

    fig = px.scatter(
        x=light_means, y=dark_means,
        labels={"x": "Mean dF/F (Light ON)", "y": "Mean dF/F (Light OFF)"},
        title="Per-ROI: Light vs Dark Mean dF/F",
        opacity=0.6,
    )
    # Add unity line
    max_val = max(np.max(light_means), np.max(dark_means))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    if light_event_rates:
        fig = px.scatter(
            x=light_event_rates, y=dark_event_rates,
            labels={"x": "Event Rate (Light ON, /min)", "y": "Event Rate (Light OFF, /min)"},
            title="Per-ROI: Light vs Dark Event Rate",
            opacity=0.6,
        )
        max_rate = max(max(light_event_rates), max(dark_event_rates)) if light_event_rates else 1
        fig.add_trace(go.Scatter(
            x=[0, max_rate], y=[0, max_rate],
            mode="lines", line=dict(dash="dash", color="gray"),
            showlegend=False,
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Modulation Index ---
with tab_modulation:
    st.subheader("Light Modulation Index")
    st.markdown(
        "**LMI = (light - dark) / (light + dark)**. "
        "Positive = more active in light, Negative = more active in dark."
    )

    # Compute modulation index
    lmi_dff = (light_means - dark_means) / (light_means + dark_means + 1e-10)

    fig = px.histogram(
        x=lmi_dff, nbins=30,
        title="Light Modulation Index (dF/F)",
        labels={"x": "LMI", "y": "Count"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=np.median(lmi_dff), line_dash="dash", line_color="red",
                  annotation_text=f"median={np.median(lmi_dff):.3f}")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean LMI", f"{np.mean(lmi_dff):.3f}")
    col2.metric("Median LMI", f"{np.median(lmi_dff):.3f}")
    col3.metric("% prefer dark", f"{(lmi_dff < 0).mean()*100:.0f}%")

    if light_event_rates:
        lmi_events = np.array([
            (lr - dr) / (lr + dr + 1e-10)
            for lr, dr in zip(light_event_rates, dark_event_rates)
        ])

        fig = px.histogram(
            x=lmi_events, nbins=30,
            title="Light Modulation Index (Event Rate)",
            labels={"x": "LMI", "y": "Count"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=np.median(lmi_events), line_dash="dash", line_color="red",
                      annotation_text=f"median={np.median(lmi_events):.3f}")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # LMI vs SNR
    fig = px.scatter(
        x=snrs, y=lmi_dff,
        labels={"x": "SNR", "y": "Light Modulation Index"},
        title="LMI vs SNR (high-SNR cells more reliable)",
        opacity=0.6,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 4: Population ---
with tab_population:
    st.subheader("Population Light/Dark Response")

    # Population event rate in each epoch
    if "event_masks" in data:
        em = data["event_masks"]
        pop_rate = em.mean(axis=0)

        # Split by light condition
        light_pop_rate = pop_rate[light_on].mean()
        dark_pop_rate = pop_rate[~light_on].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Pop. rate (Light)", f"{light_pop_rate:.4f}")
        col2.metric("Pop. rate (Dark)", f"{dark_pop_rate:.4f}")
        col3.metric("Dark/Light ratio", f"{dark_pop_rate/light_pop_rate:.2f}" if light_pop_rate > 0 else "N/A")

    # Per-cycle analysis
    if "light_on_times" in data and "light_off_times" in data:
        st.subheader("Per-Cycle Analysis")

        frame_times = data["frame_times"]
        light_on_times = data["light_on_times"]
        light_off_times = data["light_off_times"]

        cycle_data = []
        for i, on_t in enumerate(light_on_times):
            off_times_after = light_off_times[light_off_times > on_t]
            if len(off_times_after) == 0:
                continue
            off_t = off_times_after[0]

            # Light epoch
            light_mask = (frame_times >= on_t) & (frame_times < off_t)
            light_dur = off_t - on_t

            # Dark epoch (from off to next on)
            next_on_times = light_on_times[light_on_times > off_t]
            if len(next_on_times) > 0:
                next_on = next_on_times[0]
            else:
                next_on = frame_times[-1]
            dark_mask = (frame_times >= off_t) & (frame_times < next_on)
            dark_dur = next_on - off_t

            light_mean = float(dff[:, light_mask].mean()) if light_mask.any() else 0
            dark_mean = float(dff[:, dark_mask].mean()) if dark_mask.any() else 0

            cycle_data.append({
                "cycle": i + 1,
                "light_start": on_t,
                "light_dur": light_dur,
                "dark_dur": dark_dur,
                "light_mean_dff": light_mean,
                "dark_mean_dff": dark_mean,
            })

        if cycle_data:
            import pandas as pd
            cycle_df = pd.DataFrame(cycle_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cycle_df["cycle"], y=cycle_df["light_mean_dff"],
                mode="lines+markers", name="Light ON",
                marker=dict(color="gold"),
            ))
            fig.add_trace(go.Scatter(
                x=cycle_df["cycle"], y=cycle_df["dark_mean_dff"],
                mode="lines+markers", name="Light OFF",
                marker=dict(color="gray"),
            ))
            fig.update_layout(
                height=300, title="Mean Population dF/F per Light Cycle",
                xaxis_title="Cycle", yaxis_title="Mean dF/F",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(cycle_df.round(4), use_container_width=True)
