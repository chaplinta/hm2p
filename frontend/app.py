"""hm2p Dashboard — Streamlit frontend for pipeline monitoring and data viewing.

Run:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="hm2p Dashboard", layout="wide")

# --- Navigation ---
pages = {
    "Sessions": "pages.sessions",
    "Pipeline": "pages.pipeline",
    "Suite2p": "pages.suite2p",
}

page = st.sidebar.radio("Navigation", list(pages.keys()))

if page == "Sessions":
    from frontend.pages import sessions

    sessions.render()
elif page == "Pipeline":
    from frontend.pages import pipeline

    pipeline.render()
elif page == "Suite2p":
    from frontend.pages import suite2p

    suite2p.render()
