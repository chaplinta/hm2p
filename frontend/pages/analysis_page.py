"""Analysis page — condition-dependent activity, HD tuning, place tuning.

Loads ca.h5 + kinematics data from S3, runs analysis with configurable
parameters, and displays results with interactive plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis — Activity, HD & Place Tuning")

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def _list_sessions_with_calcium() -> list[str]:
    """List sessions that have both ca.h5 and timestamps.h5 on S3."""
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
                    # Check for ca.h5
                    resp = s3.list_objects_v2(
                        Bucket=DERIVATIVES_BUCKET,
                        Prefix=ses_prefix + "ca.h5",
                        MaxKeys=1,
                    )
                    if resp.get("KeyCount", 0) > 0:
                        # Extract sub/ses from prefix
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
# Sidebar — session and parameter selection
# ---------------------------------------------------------------------------

st.sidebar.header("Session")
sessions = _list_sessions_with_calcium()
if not sessions:
    st.warning("No sessions with ca.h5 found on S3.")
    st.stop()

selected = st.sidebar.selectbox("Session", sessions)
sub, ses = selected.split("/")

st.sidebar.header("Parameters")
signal_type = st.sidebar.selectbox("Signal type", ["dff", "deconv", "events"], index=0)
speed_threshold = st.sidebar.slider("Speed threshold (cm/s)", 0.0, 10.0, 2.5, 0.5)

st.sidebar.subheader("HD Tuning")
hd_n_bins = st.sidebar.select_slider("HD bins", [12, 18, 24, 36, 72], value=36)
hd_sigma = st.sidebar.slider("HD smoothing (deg)", 0.0, 20.0, 6.0, 1.0)

st.sidebar.subheader("Place Tuning")
place_bin = st.sidebar.slider("Place bin size (cm)", 1.0, 10.0, 2.5, 0.5)
place_sigma = st.sidebar.slider("Place smoothing (cm)", 0.0, 10.0, 3.0, 0.5)

n_shuffles = st.sidebar.number_input("Bootstrap shuffles", 100, 10000, 500, 100)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

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

st.sidebar.markdown(f"**ROIs:** {n_rois} | **Frames:** {n_frames} | **FPS:** {fps:.1f}")

# Try to load kinematics (resampled to imaging rate)
has_kinematics = False
try:
    ts = _download_h5(DERIVATIVES_BUCKET, f"movement/{sub}/{ses}/timestamps.h5")
    # For now, check if kinematics.h5 exists
    try:
        kin = _download_h5(DERIVATIVES_BUCKET, f"movement/{sub}/{ses}/kinematics.h5")
        has_kinematics = True
    except Exception:
        pass
except Exception:
    ts = None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_activity, tab_hd, tab_place, tab_robust = st.tabs([
    "Activity by Condition",
    "HD Tuning",
    "Place Tuning",
    "Robustness",
])

# ---- Tab 1: Activity by condition ----
with tab_activity:
    st.subheader("Activity by Condition (2x2: Movement x Light)")

    if not has_kinematics:
        st.info(
            "Kinematics data not yet available for this session. "
            "Run Stage 3 (kinematics) first to enable condition-split analysis."
        )
        # Show basic dF/F stats without condition splitting
        st.markdown("### Basic dF/F Statistics (no condition split)")
        try:
            import plotly.graph_objects as go

            mean_dff = np.nanmean(dff, axis=1)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(n_rois)), y=mean_dff, name="Mean dF/F"))
            fig.update_layout(
                xaxis_title="ROI", yaxis_title="Mean dF/F",
                title="Mean dF/F per ROI", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.write("Install plotly for interactive charts")
    else:
        from hm2p.analysis.activity import compute_batch_activity, split_conditions
        from hm2p.sync.align import resample_to_imaging_rate

        cam_times = ts["frame_times_camera"]
        img_times = ts["frame_times_imaging"]

        speed = resample_to_imaging_rate(kin["speed_cm_s"], cam_times, img_times)[:n_frames]
        light_on = resample_to_imaging_rate(
            kin["light_on"].astype(np.float64), cam_times, img_times,
        )[:n_frames] > 0.5
        bad = resample_to_imaging_rate(
            kin["bad_behav"].astype(np.float64), cam_times, img_times,
        )[:n_frames] > 0.5
        active_mask = ~bad

        # Select signal
        if signal_type == "dff":
            signals = dff[:, :n_frames]
        elif signal_type == "deconv" and deconv is not None:
            signals = deconv[:, :n_frames]
        elif signal_type == "events" and event_masks is not None:
            signals = event_masks[:, :n_frames].astype(np.float32)
        else:
            signals = dff[:, :n_frames]

        evts = event_masks[:, :n_frames] if event_masks is not None else np.zeros_like(signals, dtype=bool)

        results = compute_batch_activity(
            signals, evts, speed, light_on, active_mask, fps,
            speed_threshold=speed_threshold,
        )

        # Build dataframe-like structure for plotting
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        conditions = ["moving_light", "moving_dark", "stationary_light", "stationary_dark"]
        metric = "event_rate"

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for cond in conditions:
                vals = [r[f"{cond}_{metric}"] for r in results]
                fig.add_trace(go.Box(y=vals, name=cond.replace("_", " ").title(), boxmean=True))
            fig.update_layout(
                title=f"Event Rate by Condition ({signal_type})",
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

        # Summary table
        st.markdown("### Per-ROI Summary")
        import pandas as pd
        rows = []
        for i, r in enumerate(results):
            rows.append({
                "ROI": i,
                "Move+Light rate": f"{r['moving_light_event_rate']:.3f}",
                "Move+Dark rate": f"{r['moving_dark_event_rate']:.3f}",
                "Still+Light rate": f"{r['stationary_light_event_rate']:.3f}",
                "Still+Dark rate": f"{r['stationary_dark_event_rate']:.3f}",
                "Move MI": f"{r['movement_modulation']:.3f}",
                "Light MI": f"{r['light_modulation']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---- Tab 2: HD Tuning ----
with tab_hd:
    st.subheader("Head Direction Tuning")

    if not has_kinematics:
        st.info("Kinematics data not yet available. Run Stage 3 first.")
    else:
        from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length, preferred_direction

        hd_deg = resample_to_imaging_rate(kin["hd_deg"], cam_times, img_times)[:n_frames]
        moving_mask = (speed >= speed_threshold) & active_mask

        if signal_type == "dff":
            signals = dff[:, :n_frames]
        elif signal_type == "deconv" and deconv is not None:
            signals = deconv[:, :n_frames]
        elif signal_type == "events" and event_masks is not None:
            signals = event_masks[:, :n_frames].astype(np.float32)
        else:
            signals = dff[:, :n_frames]

        # Select ROI
        roi_select = st.selectbox("ROI", list(range(n_rois)), key="hd_roi")
        sig = signals[roi_select]

        moving_light_mask = moving_mask & light_on
        moving_dark_mask = moving_mask & ~light_on

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

                import plotly.graph_objects as go
                # Close the polar plot by appending first value
                tc_closed = np.append(tc, tc[0])
                centers_closed = np.append(centers, centers[0] + 360)
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=tc_closed, theta=centers_closed,
                    mode="lines", fill="toself", name="All",
                ))
                fig.update_layout(
                    title=f"ROI {roi_select} — MVL={mvl:.3f}, PD={pd_val:.0f} deg",
                    polar=dict(radialaxis=dict(visible=True)),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough moving frames")

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
                    title=f"ROI {roi_select} — Light vs Dark",
                    polar=dict(radialaxis=dict(visible=True)),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Comparison metrics
                from hm2p.analysis.comparison import preferred_direction_shift, tuning_curve_correlation
                corr = tuning_curve_correlation(tc_l, tc_d)
                pd_shift = preferred_direction_shift(tc_l, tc_d, centers)
                st.metric("Tuning curve correlation", f"{corr:.3f}")
                st.metric("PD shift (deg)", f"{pd_shift:.1f}")
            else:
                st.warning("Not enough frames in light or dark condition")

        # Population summary
        st.markdown("### Population HD Summary")
        mvls = []
        pds = []
        for i in range(n_rois):
            if moving_mask.sum() > 50:
                tc_i, c_i = compute_hd_tuning_curve(
                    signals[i], hd_deg, moving_mask,
                    n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                )
                mvls.append(mean_vector_length(tc_i, c_i))
                pds.append(preferred_direction(tc_i, c_i))

        if mvls:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=mvls, nbinsx=20, name="MVL"))
            fig.update_layout(
                title="MVL Distribution (all ROIs, moving frames)",
                xaxis_title="Mean Vector Length", yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 3: Place Tuning ----
with tab_place:
    st.subheader("Place Tuning")

    if not has_kinematics:
        st.info("Kinematics data not yet available. Run Stage 3 first.")
    else:
        from hm2p.analysis.tuning import compute_place_rate_map, spatial_information

        x_cm = resample_to_imaging_rate(kin["x_mm"], cam_times, img_times)[:n_frames] / 10.0
        y_cm = resample_to_imaging_rate(kin["y_mm"], cam_times, img_times)[:n_frames] / 10.0

        if signal_type == "dff":
            signals = dff[:, :n_frames]
        elif signal_type == "deconv" and deconv is not None:
            signals = deconv[:, :n_frames]
        elif signal_type == "events" and event_masks is not None:
            signals = event_masks[:, :n_frames].astype(np.float32)
        else:
            signals = dff[:, :n_frames]

        roi_select_place = st.selectbox("ROI", list(range(n_rois)), key="place_roi")
        sig = signals[roi_select_place]

        col1, col2, col3 = st.columns(3)

        for col, (label, mask) in zip(
            [col1, col2, col3],
            [("All moving", moving_mask), ("Light", moving_light_mask), ("Dark", moving_dark_mask)],
        ):
            with col:
                st.markdown(f"#### {label}")
                if mask.sum() > 50:
                    rm, om, bx, by = compute_place_rate_map(
                        sig, x_cm, y_cm, mask,
                        bin_size=place_bin, smoothing_sigma=place_sigma,
                        min_occupancy_s=0.5, fps=fps,
                    )
                    si = spatial_information(rm, om)

                    import plotly.graph_objects as go
                    fig = go.Figure(data=go.Heatmap(
                        z=rm.T, colorscale="Hot", colorbar=dict(title="Rate"),
                    ))
                    fig.update_layout(
                        title=f"SI={si:.3f} bits",
                        height=350, width=350,
                        yaxis=dict(scaleanchor="x"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough frames")


# ---- Tab 4: Robustness ----
with tab_robust:
    st.subheader("Robustness — Parameter Sensitivity")
    st.info(
        "Run the analysis across multiple parameter combinations to verify "
        "conclusions are robust. This tab will show how the fraction of significant "
        "cells changes across signal types, bin sizes, and smoothing levels."
    )

    if not has_kinematics:
        st.info("Kinematics data not yet available.")
    else:
        st.markdown("### Parameter Grid (HD Tuning)")
        st.markdown(
            "Select parameter ranges to test. The analysis computes the fraction of "
            "significantly HD-tuned cells for each combination."
        )

        sig_types = st.multiselect("Signal types", ["dff", "deconv", "events"], default=["dff"])
        bin_options = st.multiselect("HD bins", [12, 18, 24, 36, 72], default=[18, 36])
        sigma_options = st.multiselect("Smoothing (deg)", [0, 3, 6, 9, 12], default=[3, 6])
        grid_shuffles = st.number_input("Shuffles per cell", 50, 1000, 100, 50, key="grid_shuf")

        if st.button("Run Parameter Grid"):
            from hm2p.analysis.significance import hd_tuning_significance

            progress = st.progress(0)
            total = len(sig_types) * len(bin_options) * len(sigma_options)
            grid_results = []
            step = 0

            for st_name in sig_types:
                if st_name == "dff":
                    sigs = dff[:, :n_frames]
                elif st_name == "deconv" and deconv is not None:
                    sigs = deconv[:, :n_frames]
                elif st_name == "events" and event_masks is not None:
                    sigs = event_masks[:, :n_frames].astype(np.float32)
                else:
                    continue

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
