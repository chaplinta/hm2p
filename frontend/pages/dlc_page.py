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
    get_dlc_champion,
    get_mm_per_pix,
    get_progress,
    get_s3_client,
    is_session_current,
    list_s3_session_files,
    load_experiments,
    parse_session_id,
    render_champion_staleness_warning,
    sanitize_error,
)

log = logging.getLogger("hm2p.frontend.dlc")

st.title("DLC Inference (Stage 2b)")

# --- Progress overview (from actual S3 file counts) ---
st.header("Processing Progress")

experiments = load_experiments()
total = len(experiments)

@st.cache_data(ttl=120)
def _count_pose_sessions():
    """Count sessions with .h5 pose files on S3."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    sessions_done = set()
    for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix="pose/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".h5"):
                parts = obj["Key"].split("/")
                if len(parts) >= 3:
                    sessions_done.add(f"{parts[1]}/{parts[2]}")
    return sessions_done

with st.spinner("Checking S3..."):
    done_sessions = _count_pose_sessions()

completed = len(done_sessions)
remaining = total - completed

col1, col2, col3 = st.columns(3)
col1.metric("Completed", f"{completed}/{total}")
col2.metric("Remaining", remaining)
if total > 0:
    col3.metric("Progress", f"{completed/total*100:.0f}%")

if total > 0 and completed > 0:
    st.progress(min(completed / total, 1.0))

if completed == total:
    st.success("All sessions have DLC pose data.")
elif remaining > 0:
    st.info(f"{remaining} sessions remaining (~{remaining * 3}h on g4dn.xlarge)")

# Show which sessions are missing
all_session_keys = set()
for exp in experiments:
    eid = exp["exp_id"]
    parts = eid.split("_")
    all_session_keys.add(f"sub-{parts[-1]}/ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}")

missing = all_session_keys - done_sessions
if missing:
    with st.expander(f"Missing sessions ({len(missing)})"):
        for s in sorted(missing):
            st.text(f"  {s}")


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
        except Exception as e:
            st.warning(f"Could not check pose data for {exp_id}: {e}")
    return sessions


pose_sessions = list_pose_sessions()

if not pose_sessions:
    st.info("No pose data available yet.")
else:
    selected = st.selectbox("Session with pose data", pose_sessions, key="dlc_session")
    sub, ses = selected.split("/")
    mm_per_pix = get_mm_per_pix(sub, ses)

    # Champion staleness check — pose output is derived from the DLC model
    # declared as champion. Staleness is assessed via dlc_champion_id stored
    # in sync.h5 by Stage 5. If sync.h5 is absent the check is skipped.
    @st.cache_data(ttl=300)
    def _read_sync_champion_id_dlc(sub: str, ses: str) -> str:
        import h5py
        data = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5")
        if data is None:
            return "unknown"
        try:
            with h5py.File(io.BytesIO(data), "r") as f:
                val = f.attrs.get("dlc_champion_id", "unknown")
                return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
        except Exception:
            return "unknown"

    _champion = get_dlc_champion()
    _session_cid = _read_sync_champion_id_dlc(sub, ses)
    _is_current, _stale_reason = is_session_current({"dlc_champion_id": _session_cid}, _champion)
    if not _is_current:
        render_champion_staleness_warning(_stale_reason)

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
        # Pick the most recently modified .h5 file.
        # Skip files > 200 MB (old multi-animal runs are huge and slow to download).
        reasonable = [f for f in h5_files if f.get("size_mb", 0) < 200]
        if not reasonable:
            reasonable = h5_files
        reasonable.sort(key=lambda f: f.get("modified", ""), reverse=True)
        h5_key = reasonable[0]["key"]
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

                # DLC multi-index: scorer -> bodyparts -> coords (single animal)
                # or scorer -> individuals -> bodyparts -> coords (multi-animal)
                if isinstance(df.columns, pd.MultiIndex):
                    n_levels = df.columns.nlevels

                    if n_levels == 4:
                        # Multi-animal DLC format — collapse to single animal
                        scorer = df.columns.get_level_values(0)[0]
                        individuals = df.columns.get_level_values(1).unique().tolist()
                        bodyparts = df.columns.get_level_values(2).unique().tolist()

                        # With max_individuals=1, just drop the individuals level
                        ind = individuals[0]
                        df = df[scorer][ind]
                        df.columns = pd.MultiIndex.from_tuples(
                            [(scorer, bp, coord) for bp, coord in df.columns]
                        )
                        n_levels = 3

                    scorer = df.columns.get_level_values(0)[0]
                    bodyparts = df.columns.get_level_values(1).unique().tolist()

                    st.markdown(f"**Scorer:** `{scorer}`")
                    st.markdown(f"**Body parts:** {', '.join(bodyparts)}")
                    st.markdown(f"**Frames:** {len(df)}")

                    # Likelihood statistics per bodypart
                    import plotly.graph_objects as go

                    coords = df.columns.get_level_values(2).unique().tolist()
                    lik_col = "likelihood" if "likelihood" in coords else None

                    fig = go.Figure()
                    for bp in bodyparts:
                        if lik_col:
                            likelihood = df[(scorer, bp, lik_col)].values
                        else:
                            likelihood = np.ones(len(df))
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
                    if lik_col:
                        likelihood = df[(scorer, bp_select, lik_col)].values
                    else:
                        likelihood = np.ones(len(df))

                    # Filter by likelihood threshold
                    thresh = st.slider("Likelihood threshold", 0.0, 1.0, 0.5, 0.05, key="lik_thresh")
                    good = likelihood >= thresh
                    n_good = good.sum()
                    st.caption(f"{n_good}/{len(good)} frames above threshold ({n_good/len(good)*100:.1f}%)")

                    col1, col2 = st.columns(2)

                    with col1:
                        if mm_per_pix is not None:
                            x_plot = x * mm_per_pix
                            y_plot = y * mm_per_pix
                            traj_x_label = "x (mm)"
                            traj_y_label = "y (mm)"
                        else:
                            x_plot = x
                            y_plot = y
                            traj_x_label = "x (px)"
                            traj_y_label = "y (px)"

                        fig_traj = go.Figure()
                        fig_traj.add_trace(go.Scatter(
                            x=x_plot[good], y=y_plot[good],
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
                            xaxis_title=traj_x_label, yaxis_title=traj_y_label,
                            height=400,
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)

                    with col2:
                        # Speed (pixel/frame → cm/s when scale available)
                        dx = np.diff(x)
                        dy = np.diff(y)
                        speed_px = np.sqrt(dx**2 + dy**2)
                        speed_px[~good[1:]] = np.nan

                        fps_val = meta.get("tracking_fps", 30) if meta_bytes else 30
                        if mm_per_pix is not None:
                            speed_display = speed_px * fps_val * mm_per_pix / 10.0
                            speed_title = "Speed (cm/s)"
                            speed_y_label = "Speed (cm/s)"
                        else:
                            speed_display = speed_px
                            speed_title = "Speed (px/frame)"
                            speed_y_label = "Speed (px/frame)"

                        # Downsample for display
                        ds = max(1, len(speed_display) // 2000)
                        fig_speed = go.Figure()
                        fig_speed.add_trace(go.Scatter(
                            y=speed_display[::ds],
                            mode="lines",
                            line=dict(width=0.5),
                        ))
                        fig_speed.update_layout(
                            title=speed_title,
                            yaxis_title=speed_y_label, xaxis_title="Frame",
                            height=400,
                        )
                        st.plotly_chart(fig_speed, use_container_width=True)

                    # Frame-by-frame likelihood
                    st.subheader("Likelihood Over Time")
                    ds = max(1, len(likelihood) // 3000)
                    fig_lik = go.Figure()
                    for bp in bodyparts[:5]:  # Show first 5 body parts
                        lik = (df[(scorer, bp, lik_col)].values[::ds] if lik_col
                               else np.ones(len(df))[::ds])
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
