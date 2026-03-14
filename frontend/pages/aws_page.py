"""AWS job status and logs page — EC2 instances, S3 progress, job logs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import logging

import streamlit as st

from frontend.data import (
    DERIVATIVES_BUCKET,
    PIPELINE_STAGES,
    REGION,
    download_s3_bytes,
    get_pipeline_status,
    get_progress,
    get_s3_client,
    load_experiments,
    parse_session_id,
    sanitize_error,
)

log = logging.getLogger("hm2p.frontend")

TOTAL_SESSIONS = PIPELINE_STAGES["pose"]["expected"]  # 26

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
    log.exception("EC2 describe_instances failed")
    st.error("Could not connect to AWS EC2. Check server logs for details.")


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


for _stage_key, _stage_info in PIPELINE_STAGES.items():
    _s3_prefix = _stage_info.get("s3_prefix")
    if _s3_prefix is None:
        continue  # skip ingest (rawdata bucket)

    _label = _stage_info["label"]
    _expected = _stage_info["expected"]

    st.subheader(_label)
    try:
        completed = _list_completed_sessions(_s3_prefix)
        n_done = len(completed)
        st.progress(
            n_done / _expected,
            text=f"{n_done}/{_expected} sessions completed",
        )
        if completed:
            with st.expander(f"Completed sessions ({n_done})"):
                for s in completed:
                    st.text(s)
        else:
            st.info(f"No {_stage_info['short']} outputs found yet.")
    except Exception as e:
        log.exception("Could not check %s progress", _label)
        st.error(f"Could not check {_stage_info['short']} progress. Check server logs.")


# ── Section 3: Job Logs ──────────────────────────────────────────────────────

st.header("Job Logs")

log_sources = {}
for _key, _info in PIPELINE_STAGES.items():
    _pfx = _info.get("s3_prefix")
    if _pfx is None:
        continue
    log_sources[f"{_info['short']} progress"] = f"{_pfx}/_progress.json"
# Keep the original text log files too
log_sources["Suite2p log (text)"] = "ca_extraction/_suite2p_log.txt"
log_sources["DLC log (text)"] = "pose/_dlc_log.txt"

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
    log.exception("Could not fetch log")
    st.error("Could not fetch log. Check server logs for details.")

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

