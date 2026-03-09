"""Pipeline monitoring page — active jobs, progress, stage completion matrix."""

from __future__ import annotations

import logging

import streamlit as st

from frontend.data import (
    DERIVATIVES_BUCKET,
    STAGE_PREFIXES,
    get_ec2_instances,
    get_pipeline_status,
    get_progress,
    get_s3_client,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend")

st.title("Pipeline Status")

# ── Stage overview with hardcoded status context ─────────────────────────
STAGE_STATUS = {
    "ingest": {
        "label": "Stage 0 -- Ingest",
        "status": "Complete",
        "detail": "26/26 sessions uploaded to S3 (hm2p-rawdata)",
    },
    "ca_extraction": {
        "label": "Stage 1 -- Suite2p",
        "status": "Complete",
        "detail": "26/26 sessions processed; outputs in ca_extraction/",
    },
    "pose": {
        "label": "Stage 2 -- DLC Pose",
        "status": "In progress",
        "detail": "Running on EC2 g4dn.xlarge (SuperAnimal TopViewMouse, DLC 3.0rc13, ~3h/session)",
    },
    "movement": {
        "label": "Stage 3 -- Kinematics",
        "status": "Blocked",
        "detail": "Blocked on Stage 2 (DLC pose estimation); script ready at scripts/run_stage3_kinematics.py",
    },
    "calcium": {
        "label": "Stage 4 -- Calcium processing",
        "status": "Complete",
        "detail": "26/26 ca.h5 on S3 (dF/F, events, noise_probs; 391 ROIs total)",
    },
    "sync": {
        "label": "Stage 5 -- Sync",
        "status": "Blocked",
        "detail": "Blocked on Stage 3; script ready at scripts/run_stage5_sync.py",
    },
    "analysis": {
        "label": "Stage 6 -- Analysis",
        "status": "Ready",
        "detail": "Multi-signal analysis ready (dff/deconv/events); run scripts/run_stage6_analysis.py",
    },
}

STATUS_COLOURS = {
    "Complete": "green",
    "In progress": "orange",
    "Blocked": "red",
}

st.subheader("Pipeline Overview")
for key, info in STAGE_STATUS.items():
    colour = STATUS_COLOURS.get(info["status"], "gray")
    st.markdown(
        f":{colour}[**{info['label']}**] -- {info['status']}  \n"
        f"&nbsp;&nbsp;&nbsp;&nbsp;{info['detail']}"
    )

st.markdown("---")

# ── S3-based session counts per stage ─────────────────────────────────────
st.subheader("S3 Completion Counts")
st.caption(
    "Live count of sessions with output in hm2p-derivatives, "
    "checked by listing S3 prefixes."
)

S3_STAGE_PREFIXES = {
    "ca_extraction": "Stage 1 -- Suite2p (ca_extraction/)",
    "pose": "Stage 2 -- DLC (pose/)",
    "movement": "Stage 3 -- Kinematics (movement/)",
    "calcium": "Stage 4 -- Calcium (calcium/)",
    "sync": "Stage 5 -- Sync (sync/)",
}


@st.cache_data(ttl=120)
def count_s3_sessions_per_stage() -> dict[str, int]:
    """Count unique sub/ses combos with output in each stage prefix on S3."""
    s3 = get_s3_client()
    experiments = load_experiments()
    counts: dict[str, int] = {}

    for prefix in S3_STAGE_PREFIXES:
        n = 0
        for exp in experiments:
            exp_id = exp["exp_id"]
            sub, ses = parse_session_id(exp_id)
            s3_prefix = f"{prefix}/{sub}/{ses}/"
            try:
                resp = s3.list_objects_v2(
                    Bucket=DERIVATIVES_BUCKET, Prefix=s3_prefix, MaxKeys=1
                )
                if resp.get("KeyCount", 0) > 0:
                    n += 1
            except Exception:
                pass
        counts[prefix] = n
    return counts


if st.button("Refresh S3 counts"):
    count_s3_sessions_per_stage.clear()

try:
    s3_counts = count_s3_sessions_per_stage()
    total_sessions = len(load_experiments())

    cols = st.columns(len(S3_STAGE_PREFIXES))
    for col, (prefix, label) in zip(cols, S3_STAGE_PREFIXES.items()):
        n = s3_counts.get(prefix, 0)
        col.metric(label.split("(")[0].strip(), f"{n}/{total_sessions}")
        if total_sessions > 0:
            col.progress(n / total_sessions)
except Exception as e:
    log.exception("Error counting S3 sessions")
    st.warning("Could not query S3. Check server logs for details.")

st.markdown("---")

# ── Active EC2 instances ──────────────────────────────────────────────────
st.subheader("Active EC2 Instances")
instances = get_ec2_instances()
if instances:
    for inst in instances:
        st.markdown(
            f"**{inst['project']}** | `{inst['id']}` | "
            f"{inst['type']} | {inst['state']} | "
            f"IP: {inst['ip']} | Since: {inst['launch_time']}"
        )
else:
    st.info("No running hm2p instances.")

# ── Progress JSON per stage ───────────────────────────────────────────────
st.subheader("Stage Progress (from _progress.json)")
for prefix, label in STAGE_PREFIXES.items():
    progress = get_progress(prefix)
    if progress:
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        skipped = progress.get("skipped", 0)
        status = progress.get("status", "")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{label}** -- {status}")
            if total > 0:
                st.progress(
                    (completed + skipped) / total,
                    text=f"{completed}/{total} done, {failed} failed, {skipped} skipped",
                )
        with col2:
            st.text(f"Updated: {progress.get('updated', '?')}")

        if failed > 0:
            errors = progress.get("failed_errors", {})
            with st.expander(f"Failed sessions ({failed})"):
                for sess in progress.get("failed_sessions", []):
                    err = errors.get(sess, "")
                    st.text(f"  {sess}: {err}" if err else f"  {sess}")
    else:
        st.text(f"{label}: no progress data")

# ── Stage completion matrix ───────────────────────────────────────────────
st.subheader("Stage Completion Matrix")
if st.button("Refresh status"):
    st.cache_data.clear()

pipeline_status = get_pipeline_status()
experiments = load_experiments()

header = "| Session | Animal | " + " | ".join(STAGE_PREFIXES.values()) + " |"
separator = "|---|---|" + "|".join(["---"] * len(STAGE_PREFIXES)) + "|"
rows = [header, separator]

for exp in experiments:
    exp_id = exp["exp_id"]
    animal = exp_id.split("_")[-1]
    status = pipeline_status.get(exp_id, {})
    cells = []
    for prefix in STAGE_PREFIXES:
        done = status.get(prefix, False)
        cells.append("Y" if done else "-")
    rows.append(f"| {exp_id[:8]} | {animal} | " + " | ".join(cells) + " |")

st.markdown("\n".join(rows))
