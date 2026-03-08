"""hm2p Dashboard — Streamlit frontend for pipeline monitoring and data viewing.

Run from repo root:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `frontend.*` imports work
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Configure logging — writes to stderr (visible in terminal where streamlit runs)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hm2p.frontend")

import streamlit as st

st.set_page_config(page_title="hm2p Dashboard", layout="wide")
log.info("Page loaded: rendering app")

# --- Multipage navigation using Streamlit's native system ---
_app_dir = Path(__file__).resolve().parent
sessions_page = st.Page(str(_app_dir / "pages/sessions_page.py"), title="Sessions", icon=":material/table:")
pipeline_page = st.Page(str(_app_dir / "pages/pipeline_page.py"), title="Pipeline", icon=":material/monitoring:")
suite2p_page = st.Page(str(_app_dir / "pages/suite2p_page.py"), title="Suite2p", icon=":material/neurology:")
calcium_page = st.Page(str(_app_dir / "pages/calcium_page.py"), title="Calcium", icon=":material/science:")
dlc_page = st.Page(str(_app_dir / "pages/dlc_page.py"), title="DLC Pose", icon=":material/pets:")
sync_page = st.Page(str(_app_dir / "pages/sync_page.py"), title="Sync", icon=":material/sync:")
analysis_page = st.Page(str(_app_dir / "pages/analysis_page.py"), title="Analysis", icon=":material/analytics:")
compare_page = st.Page(str(_app_dir / "pages/compare_page.py"), title="Compare", icon=":material/compare:")
population_page = st.Page(str(_app_dir / "pages/population_page.py"), title="Population", icon=":material/groups:")
explorer_page = st.Page(str(_app_dir / "pages/explorer_page.py"), title="Explorer", icon=":material/explore:")
timeline_page = st.Page(str(_app_dir / "pages/timeline_page.py"), title="Timeline", icon=":material/timeline:")
gallery_page = st.Page(str(_app_dir / "pages/gallery_page.py"), title="ROI Gallery", icon=":material/grid_view:")
events_page = st.Page(str(_app_dir / "pages/events_page.py"), title="Events", icon=":material/electric_bolt:")
correlations_page = st.Page(str(_app_dir / "pages/correlations_page.py"), title="Correlations", icon=":material/hub:")
batch_page = st.Page(str(_app_dir / "pages/batch_page.py"), title="Batch", icon=":material/dashboard:")
changelog_page = st.Page(str(_app_dir / "pages/changelog_page.py"), title="Changelog", icon=":material/history:")
aws_page = st.Page(str(_app_dir / "pages/aws_page.py"), title="AWS", icon=":material/cloud:")

pg = st.navigation([
    sessions_page, pipeline_page, suite2p_page, calcium_page,
    dlc_page, sync_page, analysis_page, compare_page, population_page,
    explorer_page, timeline_page, gallery_page, events_page, correlations_page, batch_page, changelog_page, aws_page,
])
pg.run()
