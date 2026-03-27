"""Event Browser — browse individual calcium events, view waveforms, compare properties."""

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
    load_all_sync_data,
    load_animals,
    load_experiments,
    parse_session_id,
    session_filter_sidebar,
)

log = logging.getLogger("hm2p.frontend.events")

st.title("Event Browser")
st.caption("Browse individual calcium transients, view waveforms, and compare event properties.")

# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox(
    "Session",
    exp_ids,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="events_session",
)

if not selected:
    st.stop()

sub, ses = parse_session_id(selected)


@st.cache_data(ttl=300)
def load_event_data(sub: str, ses: str) -> dict | None:
    """Load ca.h5 for event analysis."""
    import h5py

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if data is None:
        return None

    f = h5py.File(io.BytesIO(data), "r")
    result = {
        "dff": f["dff"][:],
        "fps": float(f.attrs.get("fps_imaging", 9.8)),
    }
    for key in ("event_masks", "event_masks_sd", "noise_probs"):
        if key in f:
            result[key] = f[key][:]
    f.close()
    return result


with st.spinner("Loading calcium data..."):
    data = load_event_data(sub, ses)

if data is None:
    st.warning("No calcium data found.")
    st.stop()

if "event_masks" not in data and "event_masks_sd" not in data:
    st.warning("No event detection data in this session's ca.h5.")
    st.stop()

dff = data["dff"]
n_rois, n_frames = dff.shape
fps = data["fps"]
time_s = np.arange(n_frames) / fps

# Event method selector
_methods = []
if "event_masks" in data:
    _methods.append("V&H (percentile)")
if "event_masks_sd" in data:
    _methods.append("SD threshold")

event_method = st.radio(
    "Event detection method", _methods, horizontal=True, key="ev_method",
    help="V&H: Voigts & Harnett 2020 percentile-based. "
         "SD: 2×SD of noise floor (Zong et al. 2022).",
)
_use_sd = event_method == "SD threshold"

if _use_sd:
    event_masks = data["event_masks_sd"]
else:
    event_masks = data["event_masks"]


def extract_events(roi_idx: int) -> list[dict]:
    """Extract individual events for a ROI."""
    trace = dff[roi_idx]
    mask = event_masks[roi_idx].astype(bool)

    events = []
    in_event = False
    start = 0

    for i in range(n_frames):
        if mask[i] and not in_event:
            start = i
            in_event = True
        elif not mask[i] and in_event:
            # Event ended
            dur = i - start
            segment = trace[start:i]
            events.append({
                "onset": start,
                "offset": i,
                "duration_frames": dur,
                "duration_s": dur / fps,
                "peak_dff": float(np.max(segment)),
                "mean_dff": float(np.mean(segment)),
                "auc": float(np.sum(segment) / fps),
                "onset_time_s": start / fps,
            })
            in_event = False

    # Handle event at end
    if in_event:
        dur = n_frames - start
        segment = trace[start:]
        events.append({
            "onset": start,
            "offset": n_frames,
            "duration_frames": dur,
            "duration_s": dur / fps,
            "peak_dff": float(np.max(segment)),
            "mean_dff": float(np.mean(segment)),
            "auc": float(np.sum(segment) / fps),
            "onset_time_s": start / fps,
        })

    return events


# Compute SNR for all ROIs
snrs = []
for i in range(n_rois):
    trace = dff[i]
    baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
    peak = np.percentile(trace, 95)
    snrs.append(peak / baseline_std if baseline_std > 0 else 0)
snrs = np.array(snrs)

# --- ROI selector ---
col1, col2 = st.columns([1, 3])
with col1:
    min_snr = st.slider("Min SNR", 0.0, 15.0, 0.0, 0.5, key="ev_snr")
    good_rois = np.where(snrs >= min_snr)[0]
    roi_idx = st.selectbox(
        "ROI",
        good_rois.tolist(),
        format_func=lambda i: f"ROI {i} (SNR={snrs[i]:.1f})",
        key="ev_roi",
    )

events = extract_events(roi_idx)

with col2:
    st.markdown(f"**ROI {roi_idx}** — SNR: {snrs[roi_idx]:.1f} — **{len(events)} events** in {n_frames/fps:.0f}s ({len(events)/(n_frames/fps/60):.1f} events/min)")

# --- Tabs ---
tab_browse, tab_waveforms, tab_stats, tab_population, tab_cross = st.tabs([
    "Browse Events", "Waveform Gallery", "Event Statistics", "Population Events", "Cross-Session",
])

import plotly.graph_objects as go
import plotly.express as px

# --- Tab 1: Browse individual events ---
with tab_browse:
    if not events:
        st.info("No events detected for this ROI.")
    else:
        context_s = st.slider("Context window (s)", 1.0, 10.0, 3.0, 0.5, key="ev_context")
        context_frames = int(context_s * fps)

        event_idx = st.slider("Event #", 0, len(events) - 1, 0, key="ev_idx")
        ev = events[event_idx]

        # Show event in context
        start = max(0, ev["onset"] - context_frames)
        end = min(n_frames, ev["offset"] + context_frames)
        t_window = time_s[start:end]
        trace_window = dff[roi_idx, start:end]

        fig = go.Figure()

        # Event highlight
        fig.add_vrect(
            x0=time_s[ev["onset"]],
            x1=time_s[min(ev["offset"], n_frames - 1)],
            fillcolor="rgba(255,0,0,0.1)",
            layer="below",
            line_width=0,
        )

        # Full trace
        fig.add_trace(go.Scatter(
            x=t_window,
            y=trace_window,
            mode="lines",
            name="dF/F\u2080",
            line=dict(color="black", width=1),
        ))

        # Event portion
        ev_start = ev["onset"] - start
        ev_end = ev["offset"] - start
        fig.add_trace(go.Scatter(
            x=time_s[ev["onset"]:ev["offset"]],
            y=dff[roi_idx, ev["onset"]:ev["offset"]],
            mode="lines",
            name="Event",
            line=dict(color="red", width=2),
        ))

        # Plot detection threshold
        if _use_sd:
            # SD method: 2×SD of noise floor (estimated from negative dF/F values)
            trace_full = dff[roi_idx]
            neg_vals = trace_full[trace_full < 0]
            if len(neg_vals) > 10:
                noise_sd = np.std(neg_vals) / np.sqrt(1.0 - 2.0 / np.pi)
            else:
                noise_sd = np.median(np.abs(trace_full - np.median(trace_full))) * 1.4826
            threshold = 2.0 * noise_sd
            fig.add_hline(
                y=threshold, line_dash="dot", line_color="blue", opacity=0.6,
                annotation_text=f"2×SD = {threshold:.3f}",
                annotation_position="top right",
            )
        elif "noise_probs" in data:
            # V&H: plot noise probability on secondary y-axis
            noise_prob_window = data["noise_probs"][roi_idx, start:end]
            fig.add_trace(go.Scatter(
                x=t_window,
                y=noise_prob_window,
                mode="lines",
                name="P(noise)",
                line=dict(color="blue", width=0.8, dash="dot"),
                yaxis="y2",
                opacity=0.5,
            ))
            fig.update_layout(
                yaxis2=dict(
                    title="P(noise)", overlaying="y", side="right",
                    range=[0, 1], showgrid=False,
                ),
            )
            # Onset threshold line on secondary axis
            fig.add_hline(
                y=0.2, line_dash="dot", line_color="blue", opacity=0.3,
                annotation_text="onset threshold",
                annotation_position="top right",
                yref="y2",
            )

        fig.update_layout(
            height=350,
            title=f"Event {event_idx + 1}/{len(events)} — onset: {ev['onset_time_s']:.1f}s, dur: {ev['duration_s']:.2f}s, peak: {ev['peak_dff']:.3f}",
            xaxis_title="Time (s)",
            yaxis_title="dF/F\u2080",
            margin=dict(l=50, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Event details
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Duration", f"{ev['duration_s']:.2f}s")
        col2.metric("Peak dF/F\u2080", f"{ev['peak_dff']:.3f}")
        col3.metric("Mean dF/F\u2080", f"{ev['mean_dff']:.3f}")
        col4.metric("AUC", f"{ev['auc']:.3f}")
        col5.metric("Onset", f"{ev['onset_time_s']:.1f}s")

        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if event_idx > 0:
                st.button("Previous", key="ev_prev", on_click=lambda: st.session_state.update(ev_idx=event_idx - 1))
        with col3:
            if event_idx < len(events) - 1:
                st.button("Next", key="ev_next", on_click=lambda: st.session_state.update(ev_idx=event_idx + 1))


# --- Tab 2: Waveform gallery ---
with tab_waveforms:
    if not events:
        st.info("No events detected for this ROI.")
    else:
        st.subheader("All Event Waveforms (aligned to onset)")

        # Align all events to onset
        pre_frames = st.slider("Pre-onset frames", 5, 50, 15, key="ev_pre")
        post_frames = st.slider("Post-offset frames", 5, 50, 15, key="ev_post")

        max_dur = max(ev["duration_frames"] for ev in events) + pre_frames + post_frames
        time_aligned = (np.arange(max_dur) - pre_frames) / fps

        fig = go.Figure()
        all_waveforms = []

        for i, ev in enumerate(events):
            start = max(0, ev["onset"] - pre_frames)
            end = min(n_frames, ev["offset"] + post_frames)
            waveform = dff[roi_idx, start:end]

            # Pad if needed
            pad_left = max(0, pre_frames - ev["onset"])
            if pad_left > 0:
                waveform = np.concatenate([np.full(pad_left, np.nan), waveform])

            t = (np.arange(len(waveform)) - pre_frames - pad_left) / fps

            fig.add_trace(go.Scatter(
                x=t,
                y=waveform,
                mode="lines",
                line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Store for mean computation
            padded = np.full(max_dur, np.nan)
            pad_start = pre_frames - (ev["onset"] - start)
            padded[pad_start:pad_start + len(waveform) - pad_left] = waveform[pad_left:]
            all_waveforms.append(padded)

        # Mean waveform
        if all_waveforms:
            stacked = np.array(all_waveforms)
            mean_wf = np.nanmean(stacked, axis=0)
            fig.add_trace(go.Scatter(
                x=time_aligned,
                y=mean_wf,
                mode="lines",
                name="Mean",
                line=dict(color="red", width=2.5),
            ))

        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            height=400,
            title=f"Event Waveforms (n={len(events)}, aligned to onset)",
            xaxis_title="Time from onset (s)",
            yaxis_title="dF/F\u2080",
            margin=dict(l=50, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Event statistics ---
with tab_stats:
    if not events:
        st.info("No events detected for this ROI.")
    else:
        import pandas as pd

        ev_df = pd.DataFrame(events)
        st.subheader(f"Event Statistics — ROI {roi_idx}")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.histogram(ev_df, x="duration_s", nbins=30, title="Event Duration Distribution")
            fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(ev_df, x="peak_dff", nbins=30, title="Peak dF/F\u2080 Distribution")
            fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(ev_df, x="auc", nbins=30, title="AUC Distribution")
            fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        # Inter-event intervals
        if len(events) > 1:
            ieis = np.diff([ev["onset_time_s"] for ev in events])
            fig = px.histogram(
                x=ieis, nbins=30,
                title="Inter-Event Intervals",
                labels={"x": "IEI (s)", "y": "Count"},
            )
            fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        # Summary stats table
        st.subheader("Summary")
        summary = {
            "N events": len(events),
            "Event rate (events/min)": f"{len(events) / (n_frames/fps/60):.1f}",
            "Mean duration (s)": f"{ev_df['duration_s'].mean():.3f}",
            "Median duration (s)": f"{ev_df['duration_s'].median():.3f}",
            "Mean peak dF/F\u2080": f"{ev_df['peak_dff'].mean():.4f}",
            "Median peak dF/F\u2080": f"{ev_df['peak_dff'].median():.4f}",
            "Mean AUC": f"{ev_df['auc'].mean():.4f}",
            "Active fraction": f"{event_masks[roi_idx].astype(bool).mean():.3f}",
        }
        if len(events) > 1:
            summary["Mean IEI (s)"] = f"{np.mean(ieis):.2f}"
            summary["CV of IEI"] = f"{np.std(ieis)/np.mean(ieis):.2f}" if np.mean(ieis) > 0 else "N/A"

        st.dataframe(
            pd.DataFrame(summary.items(), columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
        )


# --- Tab 4: Population event view ---
with tab_population:
    st.subheader("Population Event Overview")

    # Compute event stats for all ROIs
    pop_stats = []
    for i in range(n_rois):
        mask = event_masks[i].astype(bool)
        onsets = np.flatnonzero(mask[1:] & ~mask[:-1])
        n_events = len(onsets) + (1 if mask[0] else 0)
        active_frac = float(mask.mean())
        pop_stats.append({
            "roi": i,
            "snr": snrs[i],
            "n_events": n_events,
            "event_rate": n_events / (n_frames / fps / 60),
            "active_frac": active_frac,
        })

    import pandas as pd
    pop_df = pd.DataFrame(pop_stats)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            pop_df, x="snr", y="event_rate",
            hover_data=["roi"],
            title="SNR vs Event Rate (all ROIs)",
            opacity=0.6,
        )
        # Highlight selected ROI
        fig.add_trace(go.Scatter(
            x=[snrs[roi_idx]],
            y=[pop_df.loc[pop_df["roi"] == roi_idx, "event_rate"].values[0]],
            mode="markers",
            marker=dict(color="red", size=12, symbol="star"),
            name=f"ROI {roi_idx}",
        ))
        fig.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(pop_df, x="event_rate", nbins=30, title="Event Rate Distribution")
        # Add line for selected ROI
        roi_rate = pop_df.loc[pop_df["roi"] == roi_idx, "event_rate"].values[0]
        fig.add_vline(x=roi_rate, line_dash="dash", line_color="red")
        fig.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Population raster (event onsets)
    st.subheader("Event Raster")
    max_rois_raster = st.slider("Max ROIs in raster", 5, n_rois, min(30, n_rois), key="ev_raster_max")

    # Sort by event rate and take top N
    top_rois = pop_df.nlargest(max_rois_raster, "event_rate")["roi"].values

    fig = go.Figure()
    for plot_idx, ri in enumerate(top_rois):
        mask = event_masks[ri].astype(bool)
        onset_frames = np.flatnonzero(mask[1:] & ~mask[:-1])
        if mask[0]:
            onset_frames = np.concatenate([[0], onset_frames])
        onset_times = onset_frames / fps

        fig.add_trace(go.Scatter(
            x=onset_times,
            y=np.full_like(onset_times, plot_idx),
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=8,
                line=dict(width=1, color="black"),
            ),
            name=f"ROI {ri}",
            showlegend=False,
            hovertemplate=f"ROI {ri}<br>Time: %{{x:.1f}}s<extra></extra>",
        ))

    fig.update_layout(
        height=max(200, max_rois_raster * 12),
        title=f"Event Raster (top {max_rois_raster} ROIs by event rate)",
        xaxis_title="Time (s)",
        yaxis=dict(
            tickvals=list(range(len(top_rois))),
            ticktext=[f"ROI {r}" for r in top_rois],
            title="",
        ),
        margin=dict(l=80, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 5: Cross-session event comparison ---
with tab_cross:
    st.subheader("Cross-Session Event Statistics")
    st.caption(
        "Event properties pooled across all sessions, split by cell type. "
        "Uses soma ROIs only."
    )

    @st.cache_data(ttl=600)
    def _load_all_event_stats(method_key):
        """Load event stats from all ca.h5 files on S3."""
        import pandas as pd
        import h5py as _h5py

        exps = load_experiments()
        anim_map = {a["animal_id"]: a.get("celltype", "unknown") for a in load_animals()}
        rows = []

        for exp in exps:
            eid = exp["exp_id"]
            parts = eid.split("_")
            animal_id = parts[-1]
            sub_s, ses_s = parse_session_id(eid)
            ct = anim_map.get(animal_id, "unknown")

            ca_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub_s}/{ses_s}/ca.h5")
            if ca_bytes is None:
                continue
            try:
                with _h5py.File(io.BytesIO(ca_bytes), "r") as f:
                    dff_arr = f["dff"][:]
                    _fps = float(f.attrs.get("fps_imaging", 9.8))
                    roi_types_arr = f["roi_types"][:] if "roi_types" in f else np.zeros(dff_arr.shape[0], dtype=np.uint8)
                    em = f[method_key][:] if method_key in f else None
            except Exception:
                continue

            if em is None:
                continue

            nr, nf = dff_arr.shape
            dur_min = nf / _fps / 60.0

            for ri in range(nr):
                if roi_types_arr[ri] != 0:
                    continue
                mask_i = em[ri].astype(bool)
                onsets_i = np.flatnonzero(mask_i[1:] & ~mask_i[:-1])
                n_ev = len(onsets_i) + (1 if mask_i[0] else 0)
                trace_i = dff_arr[ri]

                amps = []
                durs = []
                in_ev = False
                ev_start = 0
                for j in range(nf):
                    if mask_i[j] and not in_ev:
                        ev_start = j
                        in_ev = True
                    elif not mask_i[j] and in_ev:
                        amps.append(float(np.max(trace_i[ev_start:j])))
                        durs.append((j - ev_start) / _fps)
                        in_ev = False

                rows.append({
                    "session": eid,
                    "animal_id": animal_id,
                    "celltype": ct,
                    "roi": ri,
                    "n_events": n_ev,
                    "event_rate": n_ev / dur_min if dur_min > 0 else 0,
                    "active_frac": float(mask_i.mean()),
                    "mean_amp": float(np.mean(amps)) if amps else np.nan,
                    "median_amp": float(np.median(amps)) if amps else np.nan,
                    "mean_dur_s": float(np.mean(durs)) if durs else np.nan,
                    "mean_dff": float(np.nanmean(trace_i)),
                })

        return pd.DataFrame(rows)

    import pandas as pd

    _em_key = "event_masks_sd" if _use_sd else "event_masks"
    with st.spinner("Loading event stats from all sessions..."):
        all_ev = _load_all_event_stats(_em_key)

    if all_ev.empty:
        st.warning("No event data available across sessions.")
    else:
        penk_ev = all_ev[all_ev["celltype"] == "penk"]
        nonpenk_ev = all_ev[all_ev["celltype"] == "nonpenk"]

        st.markdown(
            f"**{len(all_ev)} soma ROIs** across {all_ev['session'].nunique()} sessions | "
            f"Penk+: {len(penk_ev)} | Penk\u207bCamKII+: {len(nonpenk_ev)}"
        )

        from scipy.stats import mannwhitneyu

        metrics = [
            ("event_rate", "Rate (ev/min)"),
            ("active_frac", "Active frac"),
            ("mean_amp", "Mean amplitude"),
            ("mean_dur_s", "Mean dur (s)"),
            ("mean_dff", "Mean dF/F"),
        ]

        cols = st.columns(len(metrics))
        for i, (key, label) in enumerate(metrics):
            pv = penk_ev[key].dropna()
            nv = nonpenk_ev[key].dropna()
            with cols[i]:
                if len(pv) >= 2 and len(nv) >= 2:
                    _, p = mannwhitneyu(pv, nv, alternative="two-sided")
                    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                    sig = " *" if p < 0.05 else ""
                    st.metric(label, f"P:{pv.median():.2f} N:{nv.median():.2f}")
                    st.caption(f"{p_str}{sig}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                all_ev, x="event_rate", color="celltype", nbins=40,
                barmode="overlay", opacity=0.6,
                title="Event Rate by Cell Type",
                color_discrete_map={"penk": "#1f77b4", "nonpenk": "#ff7f0e"},
            )
            fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True, key="cross_rate_hist")

        with c2:
            fig = px.histogram(
                all_ev, x="mean_amp", color="celltype", nbins=40,
                barmode="overlay", opacity=0.6,
                title="Event Amplitude by Cell Type",
                color_discrete_map={"penk": "#1f77b4", "nonpenk": "#ff7f0e"},
            )
            fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True, key="cross_amp_hist")

        with st.expander("Per-session breakdown"):
            session_summary = all_ev.groupby(["session", "celltype"]).agg(
                n_rois=("roi", "count"),
                median_rate=("event_rate", "median"),
                median_amp=("mean_amp", "median"),
                median_active=("active_frac", "median"),
            ).reset_index()
            st.dataframe(session_summary, use_container_width=True, hide_index=True)
