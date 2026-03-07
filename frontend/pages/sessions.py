"""Sessions page — filterable table of all sessions with pipeline status."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.data import (
    STAGE_PREFIXES,
    get_pipeline_status,
    load_animals,
    load_experiments,
    parse_session_id,
)


def render():
    st.title("Sessions")

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}

    # Build pipeline status (cached, TTL=60s)
    with st.spinner("Checking pipeline status on S3..."):
        pipeline_status = get_pipeline_status()

    # Build dataframe
    rows = []
    for exp in experiments:
        exp_id = exp["exp_id"]
        animal_id = exp_id.split("_")[-1]
        animal = animal_map.get(animal_id, {})
        sub, ses = parse_session_id(exp_id)

        row = {
            "exp_id": exp_id,
            "sub": sub,
            "ses": ses,
            "animal": animal_id,
            "celltype": animal.get("celltype", ""),
            "date": exp_id[:8],
            "lens": exp.get("lens", ""),
            "orientation": exp.get("orientation", ""),
            "primary": exp.get("primary_exp", ""),
            "exclude": exp.get("exclude", "0"),
            "notes": exp.get("Notes", ""),
        }

        # Pipeline status columns
        status = pipeline_status.get(exp_id, {})
        for prefix, label in STAGE_PREFIXES.items():
            row[label] = "Done" if status.get(prefix, False) else "-"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        celltype_filter = st.multiselect(
            "Cell type", options=sorted(df["celltype"].unique()), default=[]
        )
    with col2:
        animal_filter = st.multiselect(
            "Animal", options=sorted(df["animal"].unique()), default=[]
        )
    with col3:
        exclude_filter = st.checkbox("Show excluded", value=True)

    filtered = df.copy()
    if celltype_filter:
        filtered = filtered[filtered["celltype"].isin(celltype_filter)]
    if animal_filter:
        filtered = filtered[filtered["animal"].isin(animal_filter)]
    if not exclude_filter:
        filtered = filtered[filtered["exclude"] != "1"]

    # Summary
    n_penk = len(filtered[filtered["celltype"] == "penk"])
    n_nonpenk = len(filtered[filtered["celltype"] == "nonpenk"])
    st.markdown(
        f"**{len(filtered)}** sessions ({n_penk} penk, {n_nonpenk} nonpenk)"
    )

    # Color-code pipeline status
    def style_status(val):
        if val == "Done":
            return "background-color: #d4edda"
        return ""

    stage_cols = list(STAGE_PREFIXES.values())
    styled = filtered.style.applymap(style_status, subset=stage_cols)
    st.dataframe(styled, use_container_width=True, height=600)

    # Session detail selector
    st.markdown("---")
    st.subheader("Session Detail")
    selected = st.selectbox(
        "Select session",
        options=filtered["exp_id"].tolist(),
        format_func=lambda x: f"{x} ({filtered[filtered['exp_id']==x]['celltype'].values[0]})",
    )

    if selected:
        exp = next(e for e in experiments if e["exp_id"] == selected)
        sub, ses = parse_session_id(selected)

        st.markdown(f"**Subject:** `{sub}` | **Session:** `{ses}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Experiment metadata:**")
            for key in [
                "lens",
                "orientation",
                "fibre",
                "primary_exp",
                "bad_2p_frames",
                "bad_behav_times",
                "exclude",
            ]:
                st.text(f"  {key}: {exp.get(key, '')}")
        with col2:
            animal_id = selected.split("_")[-1]
            animal = animal_map.get(animal_id, {})
            st.markdown("**Animal metadata:**")
            for key in [
                "celltype",
                "strain",
                "gcamp",
                "virus_id",
                "hemisphere",
                "sex",
            ]:
                st.text(f"  {key}: {animal.get(key, '')}")

        # Store selected session in session state for other pages
        st.session_state["selected_sub"] = sub
        st.session_state["selected_ses"] = ses
        st.session_state["selected_exp_id"] = selected
