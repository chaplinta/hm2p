"""DLC monitoring page — live tracking progress, pose quality, session viewer."""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    REGION,
    download_s3_bytes,
    get_progress,
    get_s3_client,
    list_s3_session_files,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.dlc")

st.title("DLC Pose Estimation")

# --- Progress overview ---
st.header("Processing Progress")

dlc_progress = get_progress("pose")
experiments = load_experiments()
total = len(experiments)

if dlc_progress:
    completed = dlc_progress.get("completed", 0)
    failed = dlc_progress.get("failed", 0)
    skipped = dlc_progress.get("skipped", 0)
    status = dlc_progress.get("status", "")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Completed", f"{completed}/{total}")
    col2.metric("Failed", failed)
    col3.metric("Skipped", skipped)
    col4.metric("Remaining", total - completed - failed - skipped)
    col5.metric("Updated", dlc_progress.get("updated", "?")[:19])

    if total > 0:
        st.progress(
            (completed + skipped) / total,
            text=f"{status}",
        )

    # Estimated time remaining
    if completed > 0 and completed < total:
        remaining = total - completed - skipped
        # ~3h per session on T4
        est_hours = remaining * 3
        st.info(
            f"Estimated time remaining: ~{est_hours}h "
            f"({remaining} sessions x ~3h each on g4dn.xlarge)"
        )

    # Completed sessions list
    completed_sessions = dlc_progress.get("completed_sessions", [])
    if completed_sessions:
        with st.expander(f"Completed sessions ({len(completed_sessions)})"):
            for s in completed_sessions:
                st.text(f"  {s}")

    # Failed sessions
    failed_sessions = dlc_progress.get("failed_sessions", [])
    failed_errors = dlc_progress.get("failed_errors", {})
    if failed_sessions:
        st.error(f"{len(failed_sessions)} failed sessions")
        for s in failed_sessions:
            err = failed_errors.get(s, "No error message")
            st.text(f"  {s}: {err}")
else:
    st.info("No progress data available. DLC may not have started yet.")


# --- EC2 Instance Status ---
st.header("EC2 Instance")

try:
    import boto3
    ec2 = boto3.client("ec2", region_name=REGION)
    resp = ec2.describe_instances(
        Filters=[{"Name": "tag:Project", "Values": ["hm2p-dlc"]}],
    )
    for res in resp["Reservations"]:
        for inst in res["Instances"]:
            state = inst["State"]["Name"]
            ip = inst.get("PublicIpAddress", "-")
            itype = inst["InstanceType"]
            launch = str(inst.get("LaunchTime", ""))

            color = "green" if state == "running" else "red"
            st.markdown(f":{color}[**{state}**] | `{itype}` | IP: `{ip}` | Launched: {launch[:19]}")

except Exception as e:
    log.exception("Could not check EC2 instance")
    st.warning("Could not check EC2 instance. Check server logs for details.")

st.markdown("---")

# --- Per-session pose data viewer ---
st.header("Pose Data Viewer")


@st.cache_data(ttl=120)
def list_pose_sessions() -> list[str]:
    """List sessions with DLC output on S3."""
    s3 = get_s3_client()
    sessions = []
    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        prefix = f"pose/{sub}/{ses}/"
        try:
            resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=1)
            if resp.get("KeyCount", 0) > 0:
                sessions.append(f"{sub}/{ses}")
        except Exception:
            pass
    return sessions


pose_sessions = list_pose_sessions()

if not pose_sessions:
    st.info("No pose data available yet.")
else:
    selected = st.selectbox("Session with pose data", pose_sessions, key="dlc_session")
    sub, ses = selected.split("/")

    # List files
    files = list_s3_session_files(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/")
    st.caption(f"Found {len(files)} files")
    for f_info in files:
        name = f_info["key"].split("/")[-1]
        st.text(f"  {name} ({f_info['size_mb']:.1f} MB)")

    # Load DLC meta
    meta_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/dlc_meta.json")
    if meta_bytes:
        meta = json.loads(meta_bytes)
        st.markdown(
            f"**Tracking FPS:** {meta.get('tracking_fps', '?')} | "
            f"**Original FPS:** {meta.get('original_fps', '?')} | "
            f"**Model:** {meta.get('model', '?')}"
        )

    # Try to load the DLC .h5 file and show pose quality
    h5_files = [f_info for f_info in files if f_info["key"].endswith(".h5")]
    if h5_files:
        h5_key = h5_files[0]["key"]
        st.subheader("Pose Quality")
        st.caption(f"Loading from `{h5_key.split('/')[-1]}`...")

        try:
            import pandas as pd

            h5_data = download_s3_bytes(DERIVATIVES_BUCKET, h5_key)
            if h5_data:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
                    tmp.write(h5_data)
                    tmp.flush()
                    df = pd.read_hdf(tmp.name)

                # DLC multi-index: scorer -> bodyparts -> coords
                if isinstance(df.columns, pd.MultiIndex):
                    scorer = df.columns.get_level_values(0)[0]
                    bodyparts = df.columns.get_level_values(1).unique().tolist()

                    st.markdown(f"**Scorer:** `{scorer}`")
                    st.markdown(f"**Body parts:** {', '.join(bodyparts)}")
                    st.markdown(f"**Frames:** {len(df)}")

                    # Likelihood statistics per bodypart
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    for bp in bodyparts:
                        likelihood = df[(scorer, bp, "likelihood")].values
                        fig.add_trace(go.Box(y=likelihood, name=bp, boxmean=True))

                    fig.update_layout(
                        title="Tracking Confidence per Body Part",
                        yaxis_title="Likelihood",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Trajectory plot
                    st.subheader("Trajectories")
                    bp_select = st.selectbox("Body part", bodyparts, key="traj_bp")

                    x = df[(scorer, bp_select, "x")].values
                    y = df[(scorer, bp_select, "y")].values
                    likelihood = df[(scorer, bp_select, "likelihood")].values

                    # Filter by likelihood threshold
                    thresh = st.slider("Likelihood threshold", 0.0, 1.0, 0.5, 0.05, key="lik_thresh")
                    good = likelihood >= thresh
                    n_good = good.sum()
                    st.caption(f"{n_good}/{len(good)} frames above threshold ({n_good/len(good)*100:.1f}%)")

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_traj = go.Figure()
                        fig_traj.add_trace(go.Scatter(
                            x=x[good], y=y[good],
                            mode="markers",
                            marker=dict(
                                size=1,
                                color=np.arange(len(x))[good],
                                colorscale="Viridis",
                                colorbar=dict(title="Frame"),
                            ),
                        ))
                        fig_traj.update_layout(
                            title=f"{bp_select} trajectory",
                            xaxis_title="x (px)", yaxis_title="y (px)",
                            height=400,
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)

                    with col2:
                        # Speed (pixel/frame)
                        dx = np.diff(x)
                        dy = np.diff(y)
                        speed_px = np.sqrt(dx**2 + dy**2)
                        speed_px[~good[1:]] = np.nan

                        # Downsample for display
                        ds = max(1, len(speed_px) // 2000)
                        fig_speed = go.Figure()
                        fig_speed.add_trace(go.Scatter(
                            y=speed_px[::ds],
                            mode="lines",
                            line=dict(width=0.5),
                        ))
                        fig_speed.update_layout(
                            title="Speed (px/frame)",
                            yaxis_title="Speed", xaxis_title="Frame",
                            height=400,
                        )
                        st.plotly_chart(fig_speed, use_container_width=True)

                    # Frame-by-frame likelihood
                    st.subheader("Likelihood Over Time")
                    ds = max(1, len(likelihood) // 3000)
                    fig_lik = go.Figure()
                    for bp in bodyparts[:5]:  # Show first 5 body parts
                        lik = df[(scorer, bp, "likelihood")].values[::ds]
                        fig_lik.add_trace(go.Scatter(
                            y=lik, mode="lines",
                            line=dict(width=0.5), name=bp,
                        ))
                    fig_lik.add_hline(y=thresh, line_dash="dash", line_color="red")
                    fig_lik.update_layout(
                        title="Likelihood over time",
                        yaxis_title="Likelihood", xaxis_title="Frame",
                        height=300,
                    )
                    st.plotly_chart(fig_lik, use_container_width=True)

        except Exception as e:
            st.error("Could not load DLC data. Check server logs for details.")
            log.exception("Error loading DLC h5")
