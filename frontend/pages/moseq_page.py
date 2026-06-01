"""keypoint-MoSeq — Behavioural syllable discovery pipeline status.

Shows completion status, summary statistics, and parameters for the
keypoint-MoSeq pipeline across all 26 sessions.

Reference:
    Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point
    tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

st.title("keypoint-MoSeq")
st.caption(
    "Zero-label behavioural syllable discovery from DLC pose data using "
    "autoregressive HMMs (Weinreb et al. 2024, Nature Methods)."
)

# ── Imports ──────────────────────────────────────────────────────────────

try:
    from frontend.data import (
        DERIVATIVES_BUCKET,
        PIPELINE_STAGES,
        REGION,
        list_syllable_sessions,
        load_all_syllable_data,
        load_animals,
        load_experiments,
        load_kpms_summary,
        sanitize_error,
    )
except ImportError as _imp_err:
    st.error(f"Frontend data module not available: {_imp_err}")
    st.stop()

TOTAL_SESSIONS = PIPELINE_STAGES["kpms"]["expected"]

if st.button("Refresh", key="refresh_moseq"):
    st.cache_data.clear()

# ── Section 1: Completion Status ─────────────────────────────────────────

st.header("Completion Status")

syllable_sessions = list_syllable_sessions()
n_done = len(syllable_sessions)

col1, col2, col3 = st.columns(3)
col1.metric("Sessions with syllables.npz", f"{n_done} / {TOTAL_SESSIONS}")
pct = n_done / TOTAL_SESSIONS * 100 if TOTAL_SESSIONS > 0 else 0
col2.metric("Progress", f"{pct:.0f}%")

if n_done >= TOTAL_SESSIONS:
    st.success(f"All {TOTAL_SESSIONS} sessions complete.")
elif n_done > 0:
    st.info(f"{n_done}/{TOTAL_SESSIONS} sessions have syllable data.")
else:
    st.warning("No syllable outputs found on S3 yet.")

# Show per-session table
if syllable_sessions:
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}
    file_info = []
    for ss in sorted(syllable_sessions, key=lambda x: x["key"]):
        animal_id = ss["sub"].replace("sub-", "")
        celltype = animal_map.get(animal_id, {}).get("celltype", "?")
        file_info.append(
            {
                "Subject": ss["sub"],
                "Session": ss["ses"],
                "Cell type": celltype,
                "Size (KB)": f"{ss['size'] / 1024:.0f}",
            }
        )

    with st.expander(f"Per-session files ({n_done} sessions)"):
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)

# ── Section 2: Summary from kpms_summary.json ───────────────────────────

st.header("Summary Statistics")

summary = load_kpms_summary()

if summary is not None:
    # kpms_summary.json structure: per-session frame counts & syllable counts
    # plus global stats
    sessions_info = summary.get("sessions", [])
    global_info = summary.get("global", {})

    if global_info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total unique syllables", global_info.get("n_syllables", "?"))
        col2.metric("Total frames", f"{global_info.get('total_frames', 0):,}")
        col3.metric(
            "Mean syllables/session", f"{global_info.get('mean_syllables_per_session', 0):.1f}"
        )
        col4.metric("Median bout length", f"{global_info.get('median_bout_frames', '?')}")

    if sessions_info:
        sum_df = pd.DataFrame(sessions_info)
        display_cols = [
            c for c in ["session", "sub", "ses", "n_frames", "n_syllables"] if c in sum_df.columns
        ]
        if display_cols:
            with st.expander("Per-session summary (from kpms_summary.json)"):
                st.dataframe(sum_df[display_cols], use_container_width=True)
else:
    # Fall back to computing stats from the actual npz files
    st.info("kpms_summary.json not found on S3. Computing stats from syllables.npz files...")

    syl_data = load_all_syllable_data()
    syl_sessions = syl_data["sessions"]

    if syl_sessions:
        all_unique = set()
        total_frames = 0
        stats_rows = []
        for s in syl_sessions:
            unique_ids = np.unique(s["syllable_id"])
            all_unique.update(unique_ids.tolist())
            total_frames += s["n_frames"]
            stats_rows.append(
                {
                    "Subject": s["sub"],
                    "Session": s["ses"],
                    "Cell type": s["celltype"],
                    "Frames": s["n_frames"],
                    "Syllable types": s["n_syllables"],
                }
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total unique syllables", len(all_unique))
        col2.metric("Total frames", f"{total_frames:,}")
        mean_per_ses = np.mean([s["n_syllables"] for s in syl_sessions])
        col3.metric("Mean syllables/session", f"{mean_per_ses:.1f}")

        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

        # Pooled syllable usage histogram
        pooled_counts: dict[int, int] = {}
        for s in syl_sessions:
            unique, counts = np.unique(s["syllable_id"], return_counts=True)
            for sid, cnt in zip(unique, counts):
                pooled_counts[int(sid)] = pooled_counts.get(int(sid), 0) + int(cnt)

        if pooled_counts:
            sorted_ids = sorted(pooled_counts.keys(), key=lambda x: pooled_counts[x], reverse=True)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[str(s) for s in sorted_ids],
                        y=[pooled_counts[s] for s in sorted_ids],
                        marker_color="steelblue",
                    )
                ]
            )
            fig.update_layout(
                title="Syllable Usage (pooled across all sessions)",
                xaxis_title="Syllable ID",
                yaxis_title="Frame count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load syllable data from S3.")

# ── Section 3: EC2 Instance Status ──────────────────────────────────────

with st.expander("EC2 Instance Status"):
    try:
        import boto3

        ec2 = boto3.client("ec2", region_name=REGION)
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": ["hm2p-kpms"]},
            ]
        )
        instances = []
        for res in resp["Reservations"]:
            for inst in res["Instances"]:
                state = inst["State"]["Name"]
                if state == "terminated":
                    continue
                instances.append(
                    {
                        "Instance ID": inst["InstanceId"],
                        "State": state,
                        "Type": inst.get("InstanceType", "---"),
                        "IP": inst.get("PublicIpAddress", "---"),
                        "Launch": str(inst.get("LaunchTime", "---"))[:19],
                    }
                )

        if instances:
            for inst in instances:
                state = inst["State"]
                if state == "running":
                    st.success(
                        f"Instance **{inst['Instance ID']}** is **running** at `{inst['IP']}`"
                    )
                elif state == "stopped":
                    st.warning(f"Instance **{inst['Instance ID']}** is **stopped**")
                else:
                    st.info(f"Instance **{inst['Instance ID']}** is **{state}**")
            st.dataframe(pd.DataFrame(instances), use_container_width=True)
        else:
            st.info("No kpms EC2 instances found (may have self-terminated after completion).")

    except Exception as e:
        st.warning(f"Could not check EC2: {sanitize_error(e)}")

# ── Section 4: Configuration ────────────────────────────────────────────

with st.expander("Pipeline Parameters"):
    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | **AR-HMM kappa** | Selected via sweep (1e3, 1e4, 1e5, 1e6); target 400 ms median bout |
    | **Num PCs** | 10 |
    | **Num iterations** | 100 (sweep: 25 per kappa) |
    | **Bodyparts** | nose, left_ear, right_ear, head_midpoint, neck, mid_back, mouse_center, tail_base (8 keypoints) |
    | **Input** | DLC `.h5` files from S3 `pose/` (30 fps subsampled) |
    | **Output** | `syllables.npz` + `syllables.provenance.json` per session in S3 `kinematics/{sub}/{ses}/` |
    | **Instance type** | c5.4xlarge (16 vCPU, 32 GB, CPU-only) |
    | **Docker image** | `hm2p-kpms` (isolated numpy<1.27 environment) |
    """)

# ── Methods & References ────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
    **keypoint-MoSeq** discovers behavioural syllables --- brief, reused motifs
    of movement --- from pose tracking data without any manual labeling.
    It fits an autoregressive hidden Markov model (AR-HMM) to keypoint
    trajectories, segmenting continuous behaviour into discrete states.

    The model runs on DLC pose output (8 bodyparts at 30 fps) in an
    isolated Docker container (due to numpy version conflicts). Results
    are `syllable_id` (int16) per frame and `syllable_prob` (float32)
    posterior probabilities.

    **References:**

    Weinreb, C., Osman, A., Datta, S.R., & Mathis, A. (2024).
    "Keypoint-MoSeq: parsing behavior by linking point tracking to pose
    dynamics." *Nature Methods*, 21(9), 1329-1339.
    [doi:10.1038/s41592-024-02318-2](https://doi.org/10.1038/s41592-024-02318-2).
    https://github.com/dattalab/keypoint-moseq
    """)
