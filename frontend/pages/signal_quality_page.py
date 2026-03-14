"""Signal Quality — GCaMP signal quality assessment per session.

Analyzes photobleaching trends, temporal autocorrelation, noise floor,
and signal-to-noise characteristics for quality control.
"""

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

log = logging.getLogger("hm2p.frontend.signal_quality")

st.title("Signal Quality")
st.caption("GCaMP signal quality assessment — photobleaching, noise, and temporal structure.")


@st.cache_data(ttl=600)
def load_session_ca(exp_id: str):
    """Load calcium data for a session."""
    import h5py

    sub, ses = parse_session_id(exp_id)
    data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if data is None:
        return None

    f = h5py.File(io.BytesIO(data), "r")
    result = {
        "dff": f["dff"][:],
        "fps": float(f.attrs.get("fps_imaging", 9.8)),
    }
    if "event_masks" in f:
        result["event_masks"] = f["event_masks"][:].astype(bool)
    f.close()
    return result


# Session selector
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox("Session", exp_ids, key="sq_session")

if not selected:
    st.stop()

with st.spinner("Loading calcium data..."):
    ca = load_session_ca(selected)

if ca is None:
    st.warning("No calcium data available for this session.")
    st.stop()

dff = ca["dff"]
fps = ca["fps"]
n_rois, n_frames = dff.shape
duration_s = n_frames / fps

animal_id = selected.split("_")[-1]
celltype = animal_map.get(animal_id, {}).get("celltype", "?")

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROIs", n_rois)
col2.metric("Frames", n_frames)
col3.metric("Duration", f"{duration_s / 60:.1f} min")
col4.metric("FPS", f"{fps:.1f} Hz")

# --- Tabs ---
import plotly.express as px
import plotly.graph_objects as go

tab_bleach, tab_f0, tab_noise, tab_autocorr, tab_summary = st.tabs([
    "Photobleaching", "F0 Baseline", "Noise Floor", "Autocorrelation", "ROI Summary",
])

with tab_bleach:
    st.subheader("Photobleaching Assessment")
    st.markdown(
        "Checks for slow drift in baseline fluorescence. "
        "Ideal: flat baseline. Photobleaching: declining trend."
    )

    # Compute baseline trend (sliding window mean)
    window_s = st.slider("Smoothing window (s)", 10, 120, 60, 10, key="bleach_window")
    window_frames = max(1, int(window_s * fps))

    # Population mean dF/F0 baseline
    mean_trace = np.nanmean(dff, axis=0)

    # Sliding window smooth
    kernel = np.ones(window_frames) / window_frames
    smoothed = np.convolve(mean_trace, kernel, mode="same")
    time_s = np.arange(n_frames) / fps

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_s, y=mean_trace, mode="lines",
        name="Population mean dF/F0", opacity=0.3, line=dict(width=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=time_s, y=smoothed, mode="lines",
        name=f"Smoothed ({window_s}s)", line=dict(width=2, color="red"),
    ))
    fig.update_layout(
        height=350,
        title="Population Mean dF/F0 Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Mean dF/F0",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-ROI baseline drift
    st.markdown("**Per-ROI baseline drift:**")
    # Compare first 10% vs last 10% of recording
    n10 = max(1, n_frames // 10)
    first_mean = np.nanmean(dff[:, :n10], axis=1)
    last_mean = np.nanmean(dff[:, -n10:], axis=1)
    drift = last_mean - first_mean
    drift_frac = drift / (np.abs(first_mean) + 1e-10)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            drift, nbins=30,
            title="Baseline Drift (last 10% - first 10%)",
            labels={"value": "dF/F0 drift"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            drift_frac * 100, nbins=30,
            title="Relative Drift (%)",
            labels={"value": "Drift (%)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Drift warning
    n_drifting = np.sum(np.abs(drift_frac) > 0.5)
    if n_drifting > 0:
        st.warning(f"{n_drifting}/{n_rois} ROIs show >50% baseline drift.")
    else:
        st.success("No ROIs show excessive baseline drift.")


with tab_f0:
    st.subheader("F0 Baseline Estimation")
    st.markdown(
        "Rolling baseline F0 estimated from dF/F0 using 3-step filter: "
        "Gaussian smooth (\u03c3=10s) \u2192 rolling min (60s) \u2192 rolling max (60s). "
        "Pachitariu et al. 2017, doi:10.1101/061507."
    )

    from hm2p.calcium.dff import compute_baseline

    # Reconstruct a proxy of F0 shape from dff
    # F0 \u221d baseline of (1 + dff), since dff = (F - F0)/F0 \u2192 F = F0*(1+dff)
    # The baseline of (1+dff) tracks F0/mean(F), i.e. the normalized baseline
    f_proxy = 1.0 + dff  # proxy for F/mean(F)
    f0_proxy = compute_baseline(f_proxy.astype(np.float32), fps)

    time_s = np.arange(n_frames) / fps

    # Plot individual F0 traces
    st.markdown("**Per-cell F0 baseline:**")
    fig = go.Figure()
    for i in range(n_rois):
        fig.add_trace(go.Scatter(
            x=time_s, y=f0_proxy[i],
            mode="lines", name=f"ROI {i}",
            opacity=0.5, line=dict(width=1),
        ))
    fig.update_layout(
        height=400,
        title="F0 Baseline (all cells)",
        xaxis_title="Time (s)",
        yaxis_title="F0 (normalised)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Average normalised F0: each F0 normalised by its mean
    st.markdown("**Average normalised F0** (each cell's F0 divided by its mean):")
    f0_normed = np.zeros_like(f0_proxy)
    for i in range(n_rois):
        mean_f0 = np.nanmean(f0_proxy[i])
        if mean_f0 > 0:
            f0_normed[i] = f0_proxy[i] / mean_f0
        else:
            f0_normed[i] = f0_proxy[i]

    mean_normed = np.nanmean(f0_normed, axis=0)
    std_normed = np.nanstd(f0_normed, axis=0)

    fig = go.Figure()
    # Shaded +/-1 SD
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_s, time_s[::-1]]),
        y=np.concatenate([mean_normed + std_normed, (mean_normed - std_normed)[::-1]]),
        fill="toself", fillcolor="rgba(65, 105, 225, 0.2)",
        line=dict(width=0), name="\u00b11 SD",
    ))
    fig.add_trace(go.Scatter(
        x=time_s, y=mean_normed,
        mode="lines", name="Mean normalised F0",
        line=dict(color="royalblue", width=2),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=350,
        title="Mean Normalised F0 (each cell F0 / mean(F0))",
        xaxis_title="Time (s)",
        yaxis_title="Normalised F0",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    overall_drift = (np.nanmean(mean_normed[-int(fps*30):]) - np.nanmean(mean_normed[:int(fps*30)]))
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean F0 drift", f"{overall_drift*100:.1f}%")
    col2.metric("F0 CV", f"{np.nanstd(mean_normed)/np.nanmean(mean_normed)*100:.1f}%")
    col3.metric("ROIs plotted", n_rois)


with tab_noise:
    st.subheader("Noise Floor Analysis")
    st.markdown(
        "Estimates noise level from the baseline (sub-median) portion of each trace."
    )

    # Per-ROI noise estimation
    noise_stds = []
    snrs = []
    for roi in range(n_rois):
        trace = dff[roi]
        valid = trace[np.isfinite(trace)]
        if len(valid) == 0:
            noise_stds.append(np.nan)
            snrs.append(np.nan)
            continue
        baseline = valid[valid < np.percentile(valid, 50)]
        noise_std = np.std(baseline) if len(baseline) > 0 else np.nan
        peak = np.percentile(valid, 95)
        snr = peak / noise_std if noise_std > 0 else 0
        noise_stds.append(noise_std)
        snrs.append(snr)

    noise_stds = np.array(noise_stds)
    snrs = np.array(snrs)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            noise_stds[np.isfinite(noise_stds)], nbins=30,
            title="Baseline Noise (std of sub-median dF/F0)",
            labels={"value": "Noise σ"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            snrs[np.isfinite(snrs)], nbins=30,
            title="Signal-to-Noise Ratio (95th / noise σ)",
            labels={"value": "SNR"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # SNR quality thresholds
    n_good = np.sum(snrs >= 5)
    n_fair = np.sum((snrs >= 2) & (snrs < 5))
    n_poor = np.sum(snrs < 2)
    st.markdown(
        f"**Quality:** {n_good} good (SNR≥5), {n_fair} fair (2≤SNR<5), "
        f"{n_poor} poor (SNR<2)"
    )

    # Noise vs ROI index (check for systematic trends)
    fig = px.scatter(
        x=np.arange(n_rois), y=noise_stds,
        title="Noise Level vs ROI Index",
        labels={"x": "ROI #", "y": "Noise σ"},
        opacity=0.6,
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


with tab_autocorr:
    st.subheader("Temporal Autocorrelation")
    st.markdown(
        "Autocorrelation decay reveals GCaMP dynamics. "
        "Fast decay = noise-dominated. Slow decay = sustained activity or drift."
    )

    n_rois_show = st.slider("ROIs to analyze", 1, min(n_rois, 20), min(5, n_rois), key="ac_nrois")
    max_lag_s = st.slider("Max lag (s)", 1.0, 30.0, 5.0, 0.5, key="ac_maxlag")
    max_lag = int(max_lag_s * fps)

    # Compute autocorrelation for selected ROIs
    fig = go.Figure()
    lags_s = np.arange(max_lag) / fps

    # Sort ROIs by SNR and take top N
    roi_order = np.argsort(snrs)[::-1][:n_rois_show]

    for roi_idx in roi_order:
        trace = dff[roi_idx]
        valid = np.isfinite(trace)
        if valid.sum() < max_lag * 2:
            continue
        # Subtract mean
        t = trace.copy()
        t[~valid] = 0
        t -= np.nanmean(t)
        # Autocorrelation via FFT
        n = len(t)
        fft = np.fft.fft(t, n=2 * n)
        acf = np.fft.ifft(fft * np.conj(fft)).real[:n]
        acf /= acf[0] if acf[0] != 0 else 1
        fig.add_trace(go.Scatter(
            x=lags_s, y=acf[:max_lag],
            mode="lines", name=f"ROI {roi_idx} (SNR={snrs[roi_idx]:.1f})",
            opacity=0.7,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=400,
        title="Temporal Autocorrelation (top SNR ROIs)",
        xaxis_title="Lag (s)",
        yaxis_title="Autocorrelation",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Population mean autocorrelation
    st.markdown("**Population mean autocorrelation:**")
    all_acf = []
    for roi_idx in range(n_rois):
        trace = dff[roi_idx]
        valid = np.isfinite(trace)
        if valid.sum() < max_lag * 2:
            continue
        t = trace.copy()
        t[~valid] = 0
        t -= np.nanmean(t)
        n = len(t)
        fft = np.fft.fft(t, n=2 * n)
        acf = np.fft.ifft(fft * np.conj(fft)).real[:n]
        acf /= acf[0] if acf[0] != 0 else 1
        all_acf.append(acf[:max_lag])

    if all_acf:
        mean_acf = np.mean(all_acf, axis=0)
        std_acf = np.std(all_acf, axis=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lags_s, y=mean_acf,
            mode="lines", name="Mean", line=dict(width=2),
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([lags_s, lags_s[::-1]]),
            y=np.concatenate([mean_acf + std_acf, (mean_acf - std_acf)[::-1]]),
            fill="toself", fillcolor="rgba(65, 105, 225, 0.2)",
            line=dict(width=0), name="±1 SD",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # Find half-decay time
        half_idx = np.argmax(mean_acf < 0.5)
        if half_idx > 0:
            half_time = half_idx / fps
            fig.add_vline(x=half_time, line_dash="dot", line_color="orange",
                          annotation_text=f"τ½={half_time:.2f}s")

        fig.update_layout(
            height=350,
            title="Population Mean Autocorrelation",
            xaxis_title="Lag (s)",
            yaxis_title="Autocorrelation",
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_summary:
    st.subheader("ROI Quality Summary")

    # Build summary table
    summary_data = []
    for roi in range(n_rois):
        trace = dff[roi]
        valid = trace[np.isfinite(trace)]
        if len(valid) == 0:
            continue

        baseline = valid[valid < np.percentile(valid, 50)]
        noise_std = np.std(baseline) if len(baseline) > 0 else np.nan
        peak = np.percentile(valid, 95)
        snr = peak / noise_std if noise_std > 0 else 0

        # Skewness
        skew = float(np.mean(((valid - np.mean(valid)) / (np.std(valid) + 1e-10)) ** 3))

        # Max dF/F
        max_dff = float(np.nanmax(valid))

        # Event rate
        n_events = 0
        event_rate = 0
        if "event_masks" in ca:
            em = ca["event_masks"][roi]
            onsets = np.flatnonzero(em[1:] & ~em[:-1])
            n_events = len(onsets) + (1 if em[0] else 0)
            event_rate = n_events / (duration_s / 60)

        # Active fraction
        active_frac = float(em.mean()) if "event_masks" in ca else 0

        # Baseline drift
        n10 = max(1, n_frames // 10)
        first_m = np.nanmean(trace[:n10])
        last_m = np.nanmean(trace[-n10:])
        drift_pct = (last_m - first_m) / (abs(first_m) + 1e-10) * 100

        # Quality grade
        grade = "A"
        if snr < 2 or abs(drift_pct) > 50:
            grade = "D"
        elif snr < 3 or abs(drift_pct) > 30:
            grade = "C"
        elif snr < 5 or abs(drift_pct) > 15:
            grade = "B"

        summary_data.append({
            "ROI": roi,
            "SNR": round(snr, 2),
            "Noise σ": round(noise_std, 4),
            "Skewness": round(skew, 2),
            "Max dF/F0": round(max_dff, 3),
            "Events": n_events,
            "Event rate (/min)": round(event_rate, 1),
            "Active %": round(active_frac * 100, 1),
            "Drift %": round(drift_pct, 1),
            "Grade": grade,
        })

    df = pd.DataFrame(summary_data)
    if not df.empty:
        # Grade distribution
        grade_counts = df["Grade"].value_counts().sort_index()
        col1, col2, col3, col4 = st.columns(4)
        for i, (grade, color) in enumerate([("A", "green"), ("B", "blue"), ("C", "orange"), ("D", "red")]):
            count = grade_counts.get(grade, 0)
            [col1, col2, col3, col4][i].metric(f"Grade {grade}", count)

        # Scatter: SNR vs Skewness
        fig = px.scatter(
            df, x="SNR", y="Skewness", color="Grade",
            color_discrete_map={"A": "green", "B": "blue", "C": "orange", "D": "red"},
            hover_data=["ROI", "Event rate (/min)", "Drift %"],
            title="SNR vs Skewness (colored by quality grade)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("Download ROI quality (CSV)", csv, "roi_quality.csv", "text/csv")
