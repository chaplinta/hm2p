"""Home — project dashboard with key metrics and quick navigation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    get_pipeline_status,
    get_progress,
    load_animals,
    load_experiments,
    STAGE_PREFIXES,
)

log = logging.getLogger("hm2p.frontend.home")

st.title("hm2p Dashboard")
st.caption("Head direction tuning in Penk+ vs CamKII+ RSP neurons — pipeline and analysis dashboard.")

# --- Pipeline overview ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

n_sessions = len(experiments)
n_penk = sum(1 for e in experiments if animal_map.get(e["exp_id"].split("_")[-1], {}).get("celltype") == "penk")
n_nonpenk = sum(1 for e in experiments if animal_map.get(e["exp_id"].split("_")[-1], {}).get("celltype") == "nonpenk")
n_animals = len(set(e["exp_id"].split("_")[-1] for e in experiments))

st.subheader("Project Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sessions", n_sessions)
col2.metric("Animals", n_animals)
col3.metric("Penk+ sessions", n_penk)
col4.metric("Non-Penk sessions", n_nonpenk)

# --- Pipeline status ---
st.subheader("Pipeline Status")

with st.spinner("Checking pipeline status..."):
    pipeline_status = get_pipeline_status()

stage_done = {}
for prefix, label in STAGE_PREFIXES.items():
    done = sum(1 for s in pipeline_status.values() if s.get(prefix, False))
    stage_done[label] = done

cols = st.columns(len(STAGE_PREFIXES))
for i, (label, done) in enumerate(stage_done.items()):
    with cols[i]:
        short_label = label.split(" — ")[1]
        pct = done / n_sessions * 100 if n_sessions > 0 else 0
        st.metric(short_label, f"{done}/{n_sessions}")
        if pct >= 100:
            st.markdown(":green_circle: Complete")
        elif pct > 0:
            st.progress(pct / 100)
        else:
            st.markdown(":red_circle: Not started")

# --- DLC progress ---
dlc_progress = get_progress("pose")
if dlc_progress:
    completed = dlc_progress.get("completed", 0)
    status = dlc_progress.get("status", "")
    updated = dlc_progress.get("updated", "")[:19]

    if completed < n_sessions and "DONE" not in status:
        st.info(f"DLC processing: {status} (updated {updated})")

# --- Quick links ---
st.subheader("Quick Links")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Explore Data**")
    st.page_link("pages/timeline_page.py", label="Session Timeline")
    st.page_link("pages/gallery_page.py", label="ROI Gallery")
    st.page_link("pages/events_page.py", label="Event Browser")
    st.page_link("pages/trace_compare_page.py", label="Trace Comparison")
    st.page_link("pages/correlations_page.py", label="Correlations")

with col2:
    st.markdown("**Analysis**")
    st.page_link("pages/analysis_page.py", label="HD/Place Analysis")
    st.page_link("pages/compare_page.py", label="Cross-Session Compare")
    st.page_link("pages/population_page.py", label="Population Overview")
    st.page_link("pages/qc_report_page.py", label="QC Report")
    st.page_link("pages/animals_page.py", label="Animal Summary")

with col3:
    st.markdown("**Pipeline**")
    st.page_link("pages/sessions_page.py", label="Session Table")
    st.page_link("pages/suite2p_page.py", label="Suite2p Viewer")
    st.page_link("pages/calcium_page.py", label="Calcium Viewer")
    st.page_link("pages/dlc_page.py", label="DLC Progress")
    st.page_link("pages/batch_page.py", label="Batch Overview")

# --- Recent changelog ---
st.subheader("Recent Updates")
# Import changelog entries
try:
    from frontend.pages.changelog_page import CHANGELOG, CATEGORY_COLORS
    recent = CHANGELOG[:5]
    for date, time, cat, desc in recent:
        color = CATEGORY_COLORS.get(cat, "gray")
        st.markdown(f"**{date} {time}** — :{color}[{cat}] — {desc}")
    st.page_link("pages/changelog_page.py", label="View full changelog")
except ImportError:
    st.info("Changelog not available.")

# --- Key numbers ---
st.markdown("---")
st.caption(
    "hm2p v2 — ground-up redesign of the head direction tuning pipeline. "
    "RSP Penk+ (Cre-ON) vs CamKII+ (Cre-OFF) neurons. "
    "26 sessions, ~391 ROIs, freely-moving rose maze with 1-min light on/off cycles."
)
