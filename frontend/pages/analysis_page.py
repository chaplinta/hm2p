"""Analysis page — multi-signal comparison, HD tuning, place tuning, robustness.

Loads ca.h5 + kinematics data from S3, runs analysis with configurable
parameters across all signal types (dF/F, deconv, events), and compares
whether conclusions hold across calcium measures.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def _list_sessions_with_calcium() -> list[str]:
    """List sessions that have ca.h5 on S3."""
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")
    sessions = []
    for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix="calcium/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            sub_prefix = cp["Prefix"]
            for page2 in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=sub_prefix, Delimiter="/"):
                for cp2 in page2.get("CommonPrefixes", []):
                    ses_prefix = cp2["Prefix"]
                    resp = s3.list_objects_v2(
                        Bucket=DERIVATIVES_BUCKET,
                        Prefix=ses_prefix + "ca.h5",
                        MaxKeys=1,
                    )
                    if resp.get("KeyCount", 0) > 0:
                        parts = ses_prefix.rstrip("/").split("/")
                        sessions.append(f"{parts[-2]}/{parts[-1]}")
    return sorted(sessions)


@st.cache_data(ttl=600)
def _download_h5(bucket: str, key: str) -> dict:
    """Download an HDF5 file from S3 and return contents as dict."""
    import tempfile

    import boto3
    import h5py

    s3 = boto3.client("s3", region_name=REGION)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_file(bucket, key, tmp.name)
        with h5py.File(tmp.name, "r") as f:
            data = {}
            for k in f.keys():
                data[k] = f[k][:]
            for k, v in f.attrs.items():
                data[k] = v
            return data


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis — Multi-Signal HD & Place Tuning")

# Sidebar
st.sidebar.header("Session")
sessions = _list_sessions_with_calcium()
if not sessions:
    st.warning("No sessions with ca.h5 found on S3.")
    st.stop()

selected = st.sidebar.selectbox("Session", sessions)
sub, ses = selected.split("/")

st.sidebar.header("Parameters")
speed_threshold = st.sidebar.slider("Speed threshold (cm/s)", 0.0, 10.0, 2.5, 0.5)

st.sidebar.subheader("HD Tuning")
hd_n_bins = st.sidebar.select_slider("HD bins", [12, 18, 24, 36, 72], value=36)
hd_sigma = st.sidebar.slider("HD smoothing (deg)", 0.0, 20.0, 6.0, 1.0)

st.sidebar.subheader("Place Tuning")
place_bin = st.sidebar.slider("Place bin size (cm)", 1.0, 10.0, 2.5, 0.5)
place_sigma = st.sidebar.slider("Place smoothing (cm)", 0.0, 10.0, 3.0, 0.5)

n_shuffles = st.sidebar.number_input("Bootstrap shuffles", 100, 10000, 500, 100)

# Load data
try:
    ca = _download_h5(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
except Exception as e:
    st.error(f"Failed to load ca.h5: {e}")
    st.stop()

dff = ca.get("dff")
if dff is None:
    st.error("ca.h5 has no 'dff' dataset")
    st.stop()

n_rois, n_frames = dff.shape
fps = float(ca.get("fps_imaging", 9.8))
deconv = ca.get("spks")
event_masks = ca.get("event_masks")

# Available signal types
available_signals = ["dff"]
if deconv is not None:
    available_signals.append("deconv")
if event_masks is not None:
    available_signals.append("events")

st.sidebar.markdown(f"**ROIs:** {n_rois} | **Frames:** {n_frames} | **FPS:** {fps:.1f}")
st.sidebar.markdown(f"**Signals:** {', '.join(available_signals)}")

# Try to load kinematics
has_kinematics = False
try:
    ts = _download_h5(DERIVATIVES_BUCKET, f"movement/{sub}/{ses}/timestamps.h5")
    kin = _download_h5(DERIVATIVES_BUCKET, f"movement/{sub}/{ses}/kinematics.h5")
    has_kinematics = True
except Exception:
    ts = None
    kin = None


def _get_signal_array(signal_type: str) -> np.ndarray:
    """Get signal array for a given type."""
    if signal_type == "dff":
        return dff[:, :n_frames]
    elif signal_type == "deconv" and deconv is not None:
        return deconv[:, :n_frames]
    elif signal_type == "events" and event_masks is not None:
        return event_masks[:, :n_frames].astype(np.float32)
    return dff[:, :n_frames]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_compare, tab_activity, tab_hd, tab_place, tab_robust, tab_population = st.tabs([
    "Signal Comparison",
    "Activity by Condition",
    "HD Tuning",
    "Place Tuning",
    "Robustness",
    "Population Summary",
])

# ---- Tab 1: Multi-signal comparison ----
with tab_compare:
    st.subheader("Cross-Signal Comparison")

    if not has_kinematics:
        st.info("Kinematics not available. Run Stages 2-3 first.")

        # Basic comparison without kinematics
        st.markdown("### Signal Statistics (no behavioural data)")
        import plotly.graph_objects as go

        for sig_name in available_signals:
            sig = _get_signal_array(sig_name)
            mean_vals = np.nanmean(sig, axis=1)
            std_vals = np.nanstd(sig, axis=1)

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(n_rois)), y=mean_vals))
                fig.update_layout(title=f"Mean {sig_name} per ROI", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(n_rois)), y=std_vals))
                fig.update_layout(title=f"Std {sig_name} per ROI", height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Cross-signal correlation
        if len(available_signals) > 1:
            st.markdown("### Cross-Signal Correlation")
            for i in range(n_rois):
                pass  # Will show per-ROI correlation between signal types

            # Aggregate: mean correlation across ROIs
            pairs = []
            for i, s1 in enumerate(available_signals):
                for s2 in available_signals[i+1:]:
                    sig1 = _get_signal_array(s1)
                    sig2 = _get_signal_array(s2)
                    corrs = []
                    for roi in range(n_rois):
                        r = np.corrcoef(sig1[roi], sig2[roi])[0, 1]
                        if np.isfinite(r):
                            corrs.append(r)
                    if corrs:
                        pairs.append({"pair": f"{s1} vs {s2}", "mean_r": np.mean(corrs), "std_r": np.std(corrs)})

            if pairs:
                import pandas as pd
                st.dataframe(pd.DataFrame(pairs), use_container_width=True)

    else:
        from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length
        from hm2p.sync.align import resample_to_imaging_rate

        cam_times = ts["frame_times_camera"]
        img_times = ts["frame_times_imaging"]

        hd_deg = resample_to_imaging_rate(kin["hd_deg"], cam_times, img_times)[:n_frames]
        speed = resample_to_imaging_rate(kin["speed_cm_s"], cam_times, img_times)[:n_frames]
        light_on = resample_to_imaging_rate(
            kin["light_on"].astype(np.float64), cam_times, img_times,
        )[:n_frames] > 0.5
        bad = resample_to_imaging_rate(
            kin["bad_behav"].astype(np.float64), cam_times, img_times,
        )[:n_frames] > 0.5
        active_mask = ~bad
        moving_mask = (speed >= speed_threshold) & active_mask

        st.markdown(
            "Compare HD tuning metrics computed with different calcium signals. "
            "If the same cells are significant across all signal types, "
            "conclusions are robust to the choice of calcium measure."
        )

        # Compute MVL for each signal type
        import plotly.graph_objects as go

        mvl_data = {}
        for sig_name in available_signals:
            sig = _get_signal_array(sig_name)
            mvls = []
            for roi in range(n_rois):
                if moving_mask.sum() > 50:
                    tc, centers = compute_hd_tuning_curve(
                        sig[roi], hd_deg, moving_mask,
                        n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                    )
                    mvls.append(mean_vector_length(tc, centers))
                else:
                    mvls.append(np.nan)
            mvl_data[sig_name] = np.array(mvls)

        # MVL comparison scatter
        if len(available_signals) >= 2:
            st.subheader("MVL Cross-Signal Scatter")
            pairs = [(available_signals[i], available_signals[j])
                     for i in range(len(available_signals))
                     for j in range(i+1, len(available_signals))]

            cols = st.columns(len(pairs))
            for col, (s1, s2) in zip(cols, pairs):
                with col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=mvl_data[s1], y=mvl_data[s2],
                        mode="markers",
                        text=[f"ROI {i}" for i in range(n_rois)],
                        marker=dict(size=6),
                    ))
                    # Add identity line
                    max_val = max(mvl_data[s1].max(), mvl_data[s2].max())
                    fig.add_trace(go.Scatter(
                        x=[0, max_val], y=[0, max_val],
                        mode="lines", line=dict(dash="dash", color="gray"),
                        showlegend=False,
                    ))
                    valid = np.isfinite(mvl_data[s1]) & np.isfinite(mvl_data[s2])
                    if valid.sum() > 2:
                        r = np.corrcoef(mvl_data[s1][valid], mvl_data[s2][valid])[0, 1]
                        title = f"MVL: {s1} vs {s2} (r={r:.3f})"
                    else:
                        title = f"MVL: {s1} vs {s2}"
                    fig.update_layout(
                        title=title,
                        xaxis_title=f"MVL ({s1})",
                        yaxis_title=f"MVL ({s2})",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # MVL distribution overlay
        st.subheader("MVL Distribution by Signal Type")
        fig = go.Figure()
        for sig_name in available_signals:
            vals = mvl_data[sig_name]
            vals = vals[np.isfinite(vals)]
            fig.add_trace(go.Histogram(
                x=vals, name=sig_name, opacity=0.6, nbinsx=20,
            ))
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Mean Vector Length",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Agreement table
        st.subheader("Significance Agreement")
        st.caption(
            "Run significance testing across signal types and compare "
            "which ROIs are significant. High agreement = robust conclusions."
        )

        if st.button("Run significance comparison", key="sig_compare"):
            from hm2p.analysis.significance import hd_tuning_significance

            sig_results = {}
            progress = st.progress(0)
            total_steps = len(available_signals) * n_rois

            for si, sig_name in enumerate(available_signals):
                sig = _get_signal_array(sig_name)
                significant = np.zeros(n_rois, dtype=bool)
                p_values = np.full(n_rois, np.nan)

                for roi in range(n_rois):
                    if moving_mask.sum() > 50:
                        res = hd_tuning_significance(
                            sig[roi], hd_deg, moving_mask,
                            n_shuffles=n_shuffles,
                            n_bins=hd_n_bins,
                            smoothing_sigma_deg=hd_sigma,
                            rng=np.random.default_rng(roi),
                        )
                        p_values[roi] = res["p_value"]
                        significant[roi] = res["p_value"] < 0.05
                    step = si * n_rois + roi + 1
                    progress.progress(step / total_steps)

                sig_results[sig_name] = {
                    "significant": significant,
                    "p_values": p_values,
                    "n_significant": int(significant.sum()),
                }

            progress.empty()

            # Summary
            import pandas as pd

            summary_rows = []
            for sig_name, res in sig_results.items():
                summary_rows.append({
                    "Signal": sig_name,
                    "N significant": res["n_significant"],
                    "Fraction": f"{res['n_significant']/n_rois:.1%}",
                    "Mean p-value": f"{np.nanmean(res['p_values']):.4f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # Agreement matrix
            if len(available_signals) >= 2:
                st.markdown("### Pairwise Agreement")
                for s1, s2 in pairs:
                    both = sig_results[s1]["significant"] & sig_results[s2]["significant"]
                    either = sig_results[s1]["significant"] | sig_results[s2]["significant"]
                    jaccard = both.sum() / either.sum() if either.sum() > 0 else 0
                    st.text(
                        f"  {s1} vs {s2}: "
                        f"Both={both.sum()}, Either={either.sum()}, "
                        f"Jaccard={jaccard:.2f}"
                    )

                # Per-ROI table
                roi_rows = []
                for roi in range(n_rois):
                    row = {"ROI": roi}
                    for sig_name, res in sig_results.items():
                        row[f"{sig_name}_sig"] = "Y" if res["significant"][roi] else "-"
                        row[f"{sig_name}_p"] = f"{res['p_values'][roi]:.4f}"
                        row[f"{sig_name}_mvl"] = f"{mvl_data[sig_name][roi]:.4f}"
                    roi_rows.append(row)
                st.dataframe(pd.DataFrame(roi_rows), use_container_width=True)


# ---- Tab 2: Activity by condition ----
with tab_activity:
    st.subheader("Activity by Condition (2x2: Movement x Light)")

    if not has_kinematics:
        st.info("Kinematics data not yet available. Run Stage 3 first.")
        st.markdown("### Basic dF/F Statistics")

        import plotly.graph_objects as go

        mean_dff = np.nanmean(dff, axis=1)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(n_rois)), y=mean_dff, name="Mean dF/F"))
        fig.update_layout(xaxis_title="ROI", yaxis_title="Mean dF/F", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        signal_type_act = st.selectbox("Signal type", available_signals, key="act_sig")
        signals = _get_signal_array(signal_type_act)
        evts = event_masks[:, :n_frames] if event_masks is not None else np.zeros_like(signals, dtype=bool)

        from hm2p.analysis.activity import compute_batch_activity

        results = compute_batch_activity(
            signals, evts, speed, light_on, active_mask, fps,
            speed_threshold=speed_threshold,
        )

        import plotly.graph_objects as go

        conditions = ["moving_light", "moving_dark", "stationary_light", "stationary_dark"]

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for cond in conditions:
                vals = [r[f"{cond}_event_rate"] for r in results]
                fig.add_trace(go.Box(y=vals, name=cond.replace("_", " ").title(), boxmean=True))
            fig.update_layout(
                title=f"Event Rate by Condition ({signal_type_act})",
                yaxis_title="Events/s", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            mov_mod = [r["movement_modulation"] for r in results]
            light_mod = [r["light_modulation"] for r in results]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mov_mod, y=light_mod, mode="markers",
                text=[f"ROI {i}" for i in range(n_rois)],
                marker=dict(size=8),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Modulation Indices",
                xaxis_title="Movement Modulation",
                yaxis_title="Light Modulation",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Mean signal by condition
        st.subheader("Mean Signal by Condition")
        fig_mean = go.Figure()
        for cond in conditions:
            vals = [r[f"{cond}_mean_signal"] for r in results]
            fig_mean.add_trace(go.Box(y=vals, name=cond.replace("_", " ").title(), boxmean=True))
        fig_mean.update_layout(
            title=f"Mean Signal by Condition ({signal_type_act})",
            yaxis_title="Mean signal", height=350,
        )
        st.plotly_chart(fig_mean, use_container_width=True)

        # Summary table
        st.markdown("### Per-ROI Summary")
        import pandas as pd
        rows = []
        for i, r in enumerate(results):
            rows.append({
                "ROI": i,
                "Move+Light": f"{r['moving_light_event_rate']:.3f}",
                "Move+Dark": f"{r['moving_dark_event_rate']:.3f}",
                "Still+Light": f"{r['stationary_light_event_rate']:.3f}",
                "Still+Dark": f"{r['stationary_dark_event_rate']:.3f}",
                "Move MI": f"{r['movement_modulation']:.3f}",
                "Light MI": f"{r['light_modulation']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---- Tab 3: HD Tuning ----
with tab_hd:
    st.subheader("Head Direction Tuning")

    if not has_kinematics:
        st.info("Kinematics data not yet available.")
    else:
        from hm2p.analysis.tuning import (
            compute_hd_tuning_curve,
            mean_vector_length,
            preferred_direction,
        )

        signal_type_hd = st.selectbox("Signal type", available_signals, key="hd_sig")
        signals_hd = _get_signal_array(signal_type_hd)

        roi_select = st.selectbox("ROI", list(range(n_rois)), key="hd_roi")
        sig = signals_hd[roi_select]

        moving_light_mask = moving_mask & light_on
        moving_dark_mask = moving_mask & ~light_on

        import plotly.graph_objects as go

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### All moving frames")
            if moving_mask.sum() > 50:
                tc, centers = compute_hd_tuning_curve(
                    sig, hd_deg, moving_mask,
                    n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                )
                mvl = mean_vector_length(tc, centers)
                pd_val = preferred_direction(tc, centers)

                tc_closed = np.append(tc, tc[0])
                centers_closed = np.append(centers, centers[0] + 360)
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=tc_closed, theta=centers_closed,
                    mode="lines", fill="toself", name="All",
                ))
                fig.update_layout(
                    title=f"ROI {roi_select} ({signal_type_hd}) -- MVL={mvl:.3f}, PD={pd_val:.0f}",
                    polar=dict(radialaxis=dict(visible=True)),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Light vs Dark")
            if moving_light_mask.sum() > 50 and moving_dark_mask.sum() > 50:
                tc_l, centers = compute_hd_tuning_curve(
                    sig, hd_deg, moving_light_mask,
                    n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                )
                tc_d, _ = compute_hd_tuning_curve(
                    sig, hd_deg, moving_dark_mask,
                    n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                )
                mvl_l = mean_vector_length(tc_l, centers)
                mvl_d = mean_vector_length(tc_d, centers)

                tc_l_c = np.append(tc_l, tc_l[0])
                tc_d_c = np.append(tc_d, tc_d[0])
                centers_c = np.append(centers, centers[0] + 360)

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=tc_l_c, theta=centers_c, mode="lines",
                    name=f"Light (MVL={mvl_l:.3f})", line=dict(color="gold"),
                ))
                fig.add_trace(go.Scatterpolar(
                    r=tc_d_c, theta=centers_c, mode="lines",
                    name=f"Dark (MVL={mvl_d:.3f})", line=dict(color="navy"),
                ))
                fig.update_layout(
                    title=f"ROI {roi_select} -- Light vs Dark",
                    polar=dict(radialaxis=dict(visible=True)),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                from hm2p.analysis.comparison import preferred_direction_shift, tuning_curve_correlation
                corr = tuning_curve_correlation(tc_l, tc_d)
                pd_shift = preferred_direction_shift(tc_l, tc_d, centers)
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("TC correlation", f"{corr:.3f}")
                col_m2.metric("PD shift (deg)", f"{pd_shift:.1f}")

        # Population HD summary
        st.markdown("### Population HD Summary")
        mvls = []
        pds = []
        for i in range(n_rois):
            if moving_mask.sum() > 50:
                tc_i, c_i = compute_hd_tuning_curve(
                    signals_hd[i], hd_deg, moving_mask,
                    n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                )
                mvls.append(mean_vector_length(tc_i, c_i))
                pds.append(preferred_direction(tc_i, c_i))

        if mvls:
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=mvls, nbinsx=20))
                fig.update_layout(
                    title="MVL Distribution",
                    xaxis_title="MVL", yaxis_title="Count", height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Polar plot of preferred directions weighted by MVL
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=mvls, theta=pds,
                    mode="markers",
                    marker=dict(size=6),
                    text=[f"ROI {i}" for i in range(len(mvls))],
                ))
                fig.update_layout(
                    title="Preferred Directions (radius = MVL)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)


# ---- Tab 4: Place Tuning ----
with tab_place:
    st.subheader("Place Tuning")

    if not has_kinematics:
        st.info("Kinematics data not yet available.")
    else:
        from hm2p.analysis.tuning import compute_place_rate_map, spatial_information

        signal_type_pl = st.selectbox("Signal type", available_signals, key="pl_sig")
        signals_pl = _get_signal_array(signal_type_pl)

        x_cm = resample_to_imaging_rate(kin["x_mm"], cam_times, img_times)[:n_frames] / 10.0
        y_cm = resample_to_imaging_rate(kin["y_mm"], cam_times, img_times)[:n_frames] / 10.0

        roi_select_pl = st.selectbox("ROI", list(range(n_rois)), key="place_roi")
        sig_pl = signals_pl[roi_select_pl]

        import plotly.graph_objects as go

        col1, col2, col3 = st.columns(3)
        for col, (label, mask) in zip(
            [col1, col2, col3],
            [("All moving", moving_mask), ("Light", moving_mask & light_on), ("Dark", moving_mask & ~light_on)],
        ):
            with col:
                st.markdown(f"#### {label}")
                if mask.sum() > 50:
                    rm, om, bx, by = compute_place_rate_map(
                        sig_pl, x_cm, y_cm, mask,
                        bin_size=place_bin, smoothing_sigma=place_sigma,
                        min_occupancy_s=0.5, fps=fps,
                    )
                    si = spatial_information(rm, om)
                    fig = go.Figure(data=go.Heatmap(
                        z=rm.T, colorscale="Hot", colorbar=dict(title="Rate"),
                    ))
                    fig.update_layout(
                        title=f"SI={si:.3f} bits",
                        height=350, width=350,
                        yaxis=dict(scaleanchor="x"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Occupancy map
        st.subheader("Occupancy Map")
        if moving_mask.sum() > 50:
            _, occ, _, _ = compute_place_rate_map(
                sig_pl, x_cm, y_cm, moving_mask,
                bin_size=place_bin, smoothing_sigma=0,
                min_occupancy_s=0.0, fps=fps,
            )
            fig_occ = go.Figure(data=go.Heatmap(
                z=occ.T, colorscale="Blues",
                colorbar=dict(title="Seconds"),
            ))
            fig_occ.update_layout(
                title="Occupancy (all moving frames)",
                height=350,
                yaxis=dict(scaleanchor="x"),
            )
            st.plotly_chart(fig_occ, use_container_width=True)


# ---- Tab 5: Robustness ----
with tab_robust:
    st.subheader("Robustness -- Parameter Sensitivity")

    if not has_kinematics:
        st.info("Kinematics data not yet available.")
    else:
        st.markdown("### Parameter Grid (HD Tuning)")
        st.markdown(
            "Test how the fraction of significant cells changes across "
            "signal types, bin sizes, and smoothing levels."
        )

        sig_types = st.multiselect("Signal types", available_signals, default=available_signals, key="rob_sig")
        bin_options = st.multiselect("HD bins", [12, 18, 24, 36, 72], default=[18, 36], key="rob_bins")
        sigma_options = st.multiselect("Smoothing (deg)", [0, 3, 6, 9, 12], default=[3, 6], key="rob_sigma")
        grid_shuffles = st.number_input("Shuffles per cell", 50, 1000, 100, 50, key="grid_shuf")

        if st.button("Run Parameter Grid", key="run_grid"):
            from hm2p.analysis.significance import hd_tuning_significance

            progress = st.progress(0)
            total = len(sig_types) * len(bin_options) * len(sigma_options)
            grid_results = []
            step = 0

            for st_name in sig_types:
                sigs = _get_signal_array(st_name)

                for nb in bin_options:
                    for sg in sigma_options:
                        n_sig = 0
                        for roi in range(n_rois):
                            if moving_mask.sum() > 50:
                                res = hd_tuning_significance(
                                    sigs[roi], hd_deg, moving_mask,
                                    n_shuffles=grid_shuffles,
                                    n_bins=nb, smoothing_sigma_deg=float(sg),
                                    rng=np.random.default_rng(roi),
                                )
                                if res["p_value"] < 0.05:
                                    n_sig += 1
                        frac = n_sig / n_rois if n_rois > 0 else 0
                        grid_results.append({
                            "signal": st_name,
                            "bins": nb,
                            "sigma": sg,
                            "n_significant": n_sig,
                            "frac_significant": frac,
                        })
                        step += 1
                        progress.progress(step / total)

            progress.empty()

            import pandas as pd
            df = pd.DataFrame(grid_results)
            st.dataframe(df, use_container_width=True)

            if len(df) > 1:
                import plotly.express as px
                fig = px.scatter(
                    df, x="bins", y="sigma", size="frac_significant",
                    color="signal", hover_data=["n_significant"],
                    title="Fraction of HD-tuned cells across parameters",
                )
                st.plotly_chart(fig, use_container_width=True)


# ---- Tab 6: Population Summary ----
with tab_population:
    st.subheader("Population Summary")

    if not has_kinematics:
        st.info("Kinematics data not yet available.")
    else:
        st.markdown(
            "Overview of all ROIs across all available signal types."
        )

        import pandas as pd
        import plotly.graph_objects as go
        from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length, preferred_direction

        # Build population table
        pop_rows = []
        for roi in range(n_rois):
            row = {"ROI": roi}
            for sig_name in available_signals:
                sig = _get_signal_array(sig_name)
                row[f"{sig_name}_mean"] = f"{np.nanmean(sig[roi]):.4f}"
                row[f"{sig_name}_max"] = f"{np.nanmax(sig[roi]):.3f}"

                if moving_mask.sum() > 50:
                    tc, centers = compute_hd_tuning_curve(
                        sig[roi], hd_deg, moving_mask,
                        n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                    )
                    row[f"{sig_name}_mvl"] = f"{mean_vector_length(tc, centers):.4f}"
                    row[f"{sig_name}_pd"] = f"{preferred_direction(tc, centers):.0f}"
            pop_rows.append(row)

        df_pop = pd.DataFrame(pop_rows)
        st.dataframe(df_pop, use_container_width=True, height=400)

        # Download as CSV
        csv_data = df_pop.to_csv(index=False)
        st.download_button(
            "Download as CSV",
            csv_data,
            f"population_summary_{sub}_{ses}.csv",
            "text/csv",
        )
