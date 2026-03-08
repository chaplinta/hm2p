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
home_page = st.Page(str(_app_dir / "pages/home_page.py"), title="Home", icon=":material/home:", default=True)
sessions_page = st.Page(str(_app_dir / "pages/sessions_page.py"), title="Sessions", icon=":material/table:")
animals_page = st.Page(str(_app_dir / "pages/animals_page.py"), title="Animals", icon=":material/pets:")
pipeline_page = st.Page(str(_app_dir / "pages/pipeline_page.py"), title="Pipeline", icon=":material/monitoring:")
suite2p_page = st.Page(str(_app_dir / "pages/suite2p_page.py"), title="Suite2p", icon=":material/neurology:")
calcium_page = st.Page(str(_app_dir / "pages/calcium_page.py"), title="Calcium", icon=":material/science:")
dlc_page = st.Page(str(_app_dir / "pages/dlc_page.py"), title="DLC Pose", icon=":material/pets:")
sync_page = st.Page(str(_app_dir / "pages/sync_page.py"), title="Sync", icon=":material/sync:")
analysis_page = st.Page(str(_app_dir / "pages/analysis_page.py"), title="Analysis", icon=":material/analytics:")
compare_page = st.Page(str(_app_dir / "pages/compare_page.py"), title="Compare", icon=":material/compare:")
population_page = st.Page(str(_app_dir / "pages/population_page.py"), title="Population", icon=":material/groups:")
light_page = st.Page(str(_app_dir / "pages/light_page.py"), title="Light/Dark", icon=":material/light_mode:")
light_compare_page = st.Page(str(_app_dir / "pages/light_compare_page.py"), title="Light Compare", icon=":material/wb_twilight:")
stats_page = st.Page(str(_app_dir / "pages/stats_page.py"), title="Pub Stats", icon=":material/description:")
explorer_page = st.Page(str(_app_dir / "pages/explorer_page.py"), title="Explorer", icon=":material/explore:")
timeline_page = st.Page(str(_app_dir / "pages/timeline_page.py"), title="Timeline", icon=":material/timeline:")
gallery_page = st.Page(str(_app_dir / "pages/gallery_page.py"), title="ROI Gallery", icon=":material/grid_view:")
events_page = st.Page(str(_app_dir / "pages/events_page.py"), title="Events", icon=":material/electric_bolt:")
correlations_page = st.Page(str(_app_dir / "pages/correlations_page.py"), title="Correlations", icon=":material/hub:")
trace_compare_page = st.Page(str(_app_dir / "pages/trace_compare_page.py"), title="Trace Compare", icon=":material/compare_arrows:")
batch_page = st.Page(str(_app_dir / "pages/batch_page.py"), title="Batch", icon=":material/dashboard:")
qc_page = st.Page(str(_app_dir / "pages/qc_report_page.py"), title="QC Report", icon=":material/verified:")
maze_page = st.Page(str(_app_dir / "pages/maze_page.py"), title="Maze", icon=":material/map:")
signal_quality_page = st.Page(str(_app_dir / "pages/signal_quality_page.py"), title="Signal Quality", icon=":material/troubleshoot:")
hd_tuning_page = st.Page(str(_app_dir / "pages/hd_tuning_page.py"), title="HD Tuning", icon=":material/explore:")
decoder_page = st.Page(str(_app_dir / "pages/decoder_page.py"), title="Decoder", icon=":material/psychology:")
stability_page = st.Page(str(_app_dir / "pages/stability_page.py"), title="Stability", icon=":material/balance:")
pop_dynamics_page = st.Page(str(_app_dir / "pages/pop_dynamics_page.py"), title="Pop. Dynamics", icon=":material/scatter_plot:")
ahv_page = st.Page(str(_app_dir / "pages/ahv_page.py"), title="AHV", icon=":material/rotate_right:")
info_theory_page = st.Page(str(_app_dir / "pages/info_theory_page.py"), title="Info Theory", icon=":material/insights:")
classify_page = st.Page(str(_app_dir / "pages/classify_page.py"), title="Classify", icon=":material/category:")
drift_page = st.Page(str(_app_dir / "pages/drift_page.py"), title="Drift", icon=":material/moving:")
gain_page = st.Page(str(_app_dir / "pages/gain_page.py"), title="Gain", icon=":material/tune:")
anchoring_page = st.Page(str(_app_dir / "pages/anchoring_page.py"), title="Anchoring", icon=":material/anchor:")
speed_mod_page = st.Page(str(_app_dir / "pages/speed_page.py"), title="Speed", icon=":material/speed:")
changelog_page = st.Page(str(_app_dir / "pages/changelog_page.py"), title="Changelog", icon=":material/history:")
aws_page = st.Page(str(_app_dir / "pages/aws_page.py"), title="AWS", icon=":material/cloud:")

pg = st.navigation({
    "Overview": [home_page, sessions_page, animals_page, pipeline_page, batch_page],
    "Pipeline": [suite2p_page, calcium_page, dlc_page, sync_page],
    "Explore": [explorer_page, timeline_page, gallery_page, events_page, correlations_page, trace_compare_page],
    "Analysis": [analysis_page, compare_page, population_page, light_page, light_compare_page, stats_page, maze_page, hd_tuning_page, decoder_page, stability_page, drift_page, gain_page, anchoring_page, speed_mod_page, pop_dynamics_page, ahv_page, info_theory_page, classify_page, signal_quality_page, qc_page],
    "System": [aws_page, changelog_page],
})
pg.run()
