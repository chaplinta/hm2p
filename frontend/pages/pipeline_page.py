"""Pipeline monitoring page — active jobs, progress, stage completion matrix."""

from __future__ import annotations

import streamlit as st

from frontend.data import (
    STAGE_PREFIXES,
    get_ec2_instances,
    get_pipeline_status,
    get_progress,
    load_experiments,
)

st.title("Pipeline Status")

# Active EC2 instances
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

# Progress for each stage
st.subheader("Stage Progress")
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
            st.markdown(f"**{label}** — {status}")
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

# Stage completion matrix
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
