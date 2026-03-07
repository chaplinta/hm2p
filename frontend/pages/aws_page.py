"""AWS job status and logs page — EC2 instances, S3 progress, job logs."""

from __future__ import annotations

import json
import logging

import streamlit as st

from frontend.data import (
    DERIVATIVES_BUCKET,
    REGION,
    STAGE_PREFIXES,
    download_s3_bytes,
    get_pipeline_status,
    get_progress,
    get_s3_client,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend")

TOTAL_SESSIONS = 26

st.title("AWS Job Status & Logs")

# ── Section 1: EC2 Instances ─────────────────────────────────────────────────

st.header("EC2 Instances")

if st.button("Refresh", key="refresh_ec2"):
    st.cache_data.clear()

try:
    import boto3

    ec2 = boto3.client("ec2", region_name=REGION)
    resp = ec2.describe_instances(
        Filters=[
            {
                "Name": "tag:Name",
                "Values": ["hm2p-*"],
            },
        ]
    )
    instances = []
    for res in resp["Reservations"]:
        for inst in res["Instances"]:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            state = inst["State"]["Name"]
            instances.append(
                {
                    "Name": tags.get("Name", ""),
                    "Instance ID": inst["InstanceId"],
                    "State": state,
                    "Type": inst["InstanceType"],
                    "Launch Time": str(inst.get("LaunchTime", "")),
                    "Public IP": inst.get("PublicIpAddress", "-"),
                }
            )

    if instances:
        # Build colored state display
        state_colors = {
            "running": "green",
            "stopped": "orange",
            "terminated": "red",
            "pending": "blue",
            "stopping": "orange",
            "shutting-down": "red",
        }
        for inst in instances:
            state = inst["State"]
            color = state_colors.get(state, "gray")
            inst["State"] = f":{color}[{state}]"

        st.dataframe(
            instances,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No EC2 instances found with Name tag starting with 'hm2p-'.")

except Exception as e:
    st.error(f"Could not connect to AWS EC2: {e}")
    log.exception("EC2 describe_instances failed")


# ── Section 2: S3 Progress ───────────────────────────────────────────────────

st.header("S3 Progress")


@st.cache_data(ttl=60)
def _list_completed_sessions(stage_prefix: str) -> list[str]:
    """List session identifiers that have output in a given S3 stage prefix."""
    s3 = get_s3_client()
    experiments = load_experiments()
    completed = []
    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        s3_prefix = f"{stage_prefix}/{sub}/{ses}/"
        try:
            resp = s3.list_objects_v2(
                Bucket=DERIVATIVES_BUCKET, Prefix=s3_prefix, MaxKeys=1
            )
            if resp.get("KeyCount", 0) > 0:
                completed.append(f"{sub}/{ses}")
        except Exception:
            pass
    return completed


# Suite2p sub-section
st.subheader("Suite2p (ca_extraction)")
try:
    s2p_completed = _list_completed_sessions("ca_extraction")
    n_s2p = len(s2p_completed)
    st.progress(
        n_s2p / TOTAL_SESSIONS,
        text=f"{n_s2p}/{TOTAL_SESSIONS} sessions completed",
    )
    if s2p_completed:
        with st.expander(f"Completed sessions ({n_s2p})"):
            for s in s2p_completed:
                st.text(s)
    else:
        st.info("No Suite2p outputs found yet.")
except Exception as e:
    st.error(f"Could not check Suite2p progress: {e}")

# DLC sub-section
st.subheader("DLC (pose)")
try:
    dlc_completed = _list_completed_sessions("pose")
    n_dlc = len(dlc_completed)
    st.progress(
        n_dlc / TOTAL_SESSIONS,
        text=f"{n_dlc}/{TOTAL_SESSIONS} sessions completed",
    )
    if dlc_completed:
        with st.expander(f"Completed sessions ({n_dlc})"):
            for s in dlc_completed:
                st.text(s)
    else:
        st.info("No DLC outputs found yet.")
except Exception as e:
    st.error(f"Could not check DLC progress: {e}")


# ── Section 3: Job Logs ──────────────────────────────────────────────────────

st.header("Job Logs")

log_sources = {
    "Suite2p log": "ca_extraction/_suite2p_log.txt",
    "DLC log": "pose/_dlc_log.txt",
}

selected_log = st.selectbox("Log source", list(log_sources.keys()))
log_key = log_sources[selected_log]

try:
    log_bytes = download_s3_bytes(DERIVATIVES_BUCKET, log_key)
    if log_bytes is not None:
        log_text = log_bytes.decode("utf-8", errors="replace")
        lines = log_text.splitlines()
        last_200 = lines[-200:] if len(lines) > 200 else lines
        st.code("\n".join(last_200), language="text")
        st.caption(
            f"Showing last {len(last_200)} of {len(lines)} lines "
            f"from s3://{DERIVATIVES_BUCKET}/{log_key}"
        )
    else:
        st.info(f"No log file found at s3://{DERIVATIVES_BUCKET}/{log_key}")
except Exception as e:
    st.error(f"Could not fetch log: {e}")

# Check _progress.json for the selected stage
stage_prefix = log_key.split("/")[0]  # "ca_extraction" or "pose"
progress_data = get_progress(stage_prefix)
if progress_data:
    st.subheader(f"Progress ({stage_prefix})")
    cols = st.columns(4)
    cols[0].metric("Status", progress_data.get("status", "?"))
    cols[1].metric("Completed", progress_data.get("completed", 0))
    cols[2].metric("Failed", progress_data.get("failed", 0))
    cols[3].metric("Skipped", progress_data.get("skipped", 0))

    if progress_data.get("total"):
        st.progress(
            (progress_data.get("completed", 0) + progress_data.get("skipped", 0))
            / progress_data["total"],
            text=f"{progress_data.get('completed', 0)}/{progress_data['total']} done",
        )


# ── Section 4: DLC Progress Detail ───────────────────────────────────────────

st.header("DLC Progress Detail")

dlc_progress = get_progress("pose")
if dlc_progress:
    st.subheader("Overview")
    cols = st.columns(4)
    cols[0].metric("Status", dlc_progress.get("status", "?"))
    cols[1].metric("Completed", dlc_progress.get("completed", 0))
    cols[2].metric("Failed", dlc_progress.get("failed", 0))
    cols[3].metric("Skipped", dlc_progress.get("skipped", 0))

    # Show failed sessions with error messages
    failed_sessions = dlc_progress.get("failed_sessions", [])
    if failed_sessions:
        st.subheader("Failed Sessions")
        for entry in failed_sessions:
            if isinstance(entry, dict):
                session = entry.get("session", "unknown")
                error = entry.get("error", "no error message")
                st.error(f"**{session}** -- {error}")
            else:
                st.error(str(entry))
    elif dlc_progress.get("failed", 0) > 0:
        st.warning(
            "Failed sessions exist but no detailed error messages are available "
            "in _progress.json."
        )
    else:
        st.success("No failed sessions.")
else:
    st.info(
        "No _progress.json found for DLC (pose). "
        "DLC processing may not have started yet."
    )
