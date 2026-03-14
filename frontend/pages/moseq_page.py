"""keypoint-MoSeq — Behavioural syllable discovery pipeline status.

Shows the status of the keypoint-MoSeq EC2 job, syllable outputs on S3,
and syllable statistics when available.

Reference:
    Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point
    tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import json
import logging

import numpy as np
import pandas as pd
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
        REGION,
        download_s3_bytes,
        get_s3_client,
        load_experiments,
        sanitize_error,
    )
except ImportError as _imp_err:
    st.error(f"Frontend data module not available: {_imp_err}")
    st.stop()

TOTAL_SESSIONS = 26

if st.button("Refresh", key="refresh_moseq"):
    st.cache_data.clear()

# ── Section 1: EC2 Instance Status ──────────────────────────────────────

st.header("EC2 Instance")

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
            instances.append({
                "Instance ID": inst["InstanceId"],
                "State": state,
                "Type": inst.get("InstanceType", "—"),
                "IP": inst.get("PublicIpAddress", "—"),
                "Launch": str(inst.get("LaunchTime", "—"))[:19],
            })

    if instances:
        for inst in instances:
            state = inst["State"]
            if state == "running":
                st.success(f"Instance **{inst['Instance ID']}** is **running** at `{inst['IP']}`")
                st.code(
                    f"ssh -i ~/.ssh/hm2p-suite2p.pem ubuntu@{inst['IP']} "
                    "'tail -f /var/log/kpms-setup.log'",
                    language="bash",
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

# ── Section 2: Job Completion Status ────────────────────────────────────

st.header("Job Status")

try:
    s3 = get_s3_client()

    # Check completion marker
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="kinematics/kpms_status.json")
        status_data = json.loads(obj["Body"].read().decode())
        if status_data.get("status") == "complete":
            st.success("keypoint-MoSeq job completed successfully.")
        else:
            st.info(f"Job status: {status_data.get('status', 'unknown')}")
    except s3.exceptions.NoSuchKey:
        st.warning("Job not yet complete (no status marker on S3).")
    except Exception:
        st.warning("Job not yet complete (no status marker on S3).")

    # Count syllable outputs
    resp = s3.list_objects_v2(
        Bucket=DERIVATIVES_BUCKET,
        Prefix="kinematics/",
    )
    all_keys = [obj["Key"] for obj in resp.get("Contents", [])]
    npz_files = [k for k in all_keys if k.endswith("syllables.npz")]

    col1, col2, col3 = st.columns(3)
    col1.metric("Sessions complete", f"{len(npz_files)} / {TOTAL_SESSIONS}")
    col2.metric("Progress", f"{len(npz_files) / TOTAL_SESSIONS * 100:.0f}%")

    # Check for summary JSON
    summary_keys = [k for k in all_keys if k.endswith("kpms_summary.json")]
    col3.metric("Summary files", len(summary_keys))

    if npz_files:
        st.subheader("Syllable Outputs on S3")

        file_info = []
        for key in sorted(npz_files):
            parts = key.split("/")
            # kinematics/sub-XXXX/ses-XXXX/syllables.npz
            sub = parts[1] if len(parts) > 1 else "—"
            ses = parts[2] if len(parts) > 2 else "—"
            file_info.append({"Subject": sub, "Session": ses, "S3 Key": key})

        st.dataframe(pd.DataFrame(file_info), use_container_width=True)

        # ── Section 3: Syllable Statistics ──────────────────────────────
        st.header("Syllable Statistics")

        @st.cache_data(ttl=600)
        def _load_syllable_stats(key: str) -> dict | None:
            """Load a syllable .npz and compute basic stats."""
            try:
                data = download_s3_bytes(DERIVATIVES_BUCKET, key)
                if data is None:
                    return None
                import io
                npz = np.load(io.BytesIO(data))
                syl_ids = npz.get("syllable_id", npz.get("syllable_ids", None))
                if syl_ids is None:
                    return None
                unique, counts = np.unique(syl_ids, return_counts=True)
                return {
                    "n_frames": len(syl_ids),
                    "n_syllables": len(unique),
                    "syllable_ids": unique.tolist(),
                    "syllable_counts": counts.tolist(),
                    "most_common": int(unique[np.argmax(counts)]),
                    "most_common_frac": float(counts.max() / counts.sum()),
                }
            except Exception as e:
                log.warning("Failed to load syllable stats: %s", e)
                return None

        # Load stats for all available sessions
        all_stats = []
        for fi in file_info:
            stats = _load_syllable_stats(fi["S3 Key"])
            if stats:
                all_stats.append({**fi, **stats})

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            display_cols = ["Subject", "Session", "n_frames", "n_syllables",
                            "most_common", "most_common_frac"]
            display_cols = [c for c in display_cols if c in stats_df.columns]
            st.dataframe(
                stats_df[display_cols].style.format(
                    {"most_common_frac": "{:.1%}"}, na_rep="—"
                ),
                use_container_width=True,
            )

            # Aggregate stats
            total_syllables = stats_df["n_syllables"].max()
            total_frames = stats_df["n_frames"].sum()

            col1, col2 = st.columns(2)
            col1.metric("Total syllable types", int(total_syllables))
            col2.metric("Total frames analysed", f"{total_frames:,}")

            # Syllable usage distribution (pooled)
            import plotly.graph_objects as go

            pooled_counts: dict[int, int] = {}
            for row in all_stats:
                for sid, cnt in zip(row["syllable_ids"], row["syllable_counts"]):
                    pooled_counts[sid] = pooled_counts.get(sid, 0) + cnt

            if pooled_counts:
                sorted_ids = sorted(pooled_counts.keys(),
                                     key=lambda x: pooled_counts[x], reverse=True)
                fig = go.Figure(data=[go.Bar(
                    x=[str(s) for s in sorted_ids],
                    y=[pooled_counts[s] for s in sorted_ids],
                    marker_color="steelblue",
                )])
                fig.update_layout(
                    title="Syllable Usage (pooled across sessions)",
                    xaxis_title="Syllable ID",
                    yaxis_title="Frame count",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not load syllable statistics from .npz files.")
    else:
        st.info("No syllable outputs on S3 yet. Job may still be running.")

except Exception as e:
    st.warning(f"Could not check S3: {sanitize_error(e)}")

# ── Section 4: Configuration ────────────────────────────────────────────

with st.expander("Pipeline Configuration"):
    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | Instance type | c5.2xlarge (8 vCPU, 16 GB, CPU-only) |
    | Input | DLC `.h5` files from S3 `pose/` |
    | Output | `syllables.npz` → S3 `kinematics/{sub}/{ses}/` |
    | AR-HMM kappa | 1,000,000 |
    | Num PCs | 10 |
    | Num iterations | 50 |
    | Docker | `hm2p-kpms` (isolated numpy<1.27 environment) |
    | Bodyparts | left_ear, right_ear, mid_back, mouse_center, tail_base |
    """)

# ── Methods & References ────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
    **keypoint-MoSeq** discovers behavioural syllables — brief, reused motifs
    of movement — from pose tracking data without any manual labeling.
    It fits an autoregressive hidden Markov model (AR-HMM) to keypoint
    trajectories, segmenting continuous behaviour into discrete states.

    The model runs on DLC pose output (5 bodyparts at 30 fps) in an
    isolated Docker container (due to numpy version conflicts). Results
    are `syllable_id` (int16) per frame and `syllable_prob` (float32)
    posterior probabilities.

    **References:**

    Weinreb, C., Osman, A., Datta, S.R., & Mathis, A. (2024).
    "Keypoint-MoSeq: parsing behavior by linking point tracking to pose
    dynamics." *Nature Methods*, 21(9), 1329-1339.
    doi:10.1038/s41592-024-02318-2.
    https://github.com/dattalab/keypoint-moseq
    """)
