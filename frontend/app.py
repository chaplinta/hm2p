"""hm2p Dashboard — Streamlit frontend for pipeline monitoring and data viewing.

Run from repo root:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
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

# ---------------------------------------------------------------------------
# Google OAuth authentication
# ---------------------------------------------------------------------------
# Auth is enabled when GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars are
# set. When absent, auth is skipped entirely (local development mode).
# Only the email addresses listed in _ALLOWED_EMAILS may access the app.
# ---------------------------------------------------------------------------
_ALLOWED_EMAILS = ["tristan.chaplin@gmail.com"]

_google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
_google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
_auth_enabled = bool(_google_client_id and _google_client_secret)

if _auth_enabled:
    from streamlit_google_auth import Authenticate

    _redirect_uri = os.environ.get("STREAMLIT_REDIRECT_URI", "http://localhost:8501")

    # streamlit-google-auth requires a Google credentials JSON file.
    # We build one dynamically from environment variables so that secrets
    # are never committed to the repository.
    @st.cache_resource
    def _get_credentials_path() -> str:
        """Write a temporary Google OAuth client-secret JSON and return its path."""
        creds = {
            "web": {
                "client_id": _google_client_id,
                "client_secret": _google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [_redirect_uri],
            }
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="hm2p_oauth_"
        )
        json.dump(creds, tmp)
        tmp.close()
        return tmp.name

    _creds_path = _get_credentials_path()
    _auth = Authenticate(
        secret_credentials_path=_creds_path,
        redirect_uri=_redirect_uri,
        cookie_name="hm2p_auth",
        cookie_key=os.environ.get("STREAMLIT_COOKIE_KEY", os.urandom(32).hex()),
        cookie_expiry_days=30,
    )

    _auth.check_authentification()

    if not st.session_state.get("connected", False):
        st.title("hm2p Dashboard")
        st.info("Please sign in with your Google account to continue.")
        _auth.login()
        st.stop()

    # Verify the authenticated email is in the allowed list
    _user_email = st.session_state.get("user_info", {}).get("email", "")
    if _user_email not in _ALLOWED_EMAILS:
        log.warning("Unauthorised login attempt: %s", _user_email)
        st.error(f"Access denied. The account **{_user_email}** is not authorised.")
        _auth.logout()
        st.stop()

    # Show logout button in sidebar
    with st.sidebar:
        _user_name = st.session_state.get("user_info", {}).get("name", _user_email)
        st.caption(f"Signed in as **{_user_name}**")
        _auth.logout()
else:
    log.info("Auth disabled — GOOGLE_CLIENT_ID not set (local dev mode)")

# --- Multipage navigation using Streamlit's native system ---
_app_dir = Path(__file__).resolve().parent
home_page = st.Page(str(_app_dir / "pages/home_page.py"), title="Home", icon=":material/home:", default=True)
summary_page = st.Page(str(_app_dir / "pages/summary_page.py"), title="Cell Summary", icon=":material/summarize:")
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
event_dynamics_page = st.Page(str(_app_dir / "pages/event_dynamics_page.py"), title="Event Dynamics", icon=":material/show_chart:")
correlations_page = st.Page(str(_app_dir / "pages/correlations_page.py"), title="Correlations", icon=":material/hub:")
trace_compare_page = st.Page(str(_app_dir / "pages/trace_compare_page.py"), title="Trace Compare", icon=":material/compare_arrows:")
qc_page = st.Page(str(_app_dir / "pages/qc_report_page.py"), title="QC Report", icon=":material/verified:")
maze_page = st.Page(str(_app_dir / "pages/maze_page.py"), title="Maze", icon=":material/map:")
signal_quality_page = st.Page(str(_app_dir / "pages/signal_quality_page.py"), title="Signal Quality", icon=":material/troubleshoot:")
hd_tuning_page = st.Page(str(_app_dir / "pages/hd_tuning_page.py"), title="HD Tuning", icon=":material/explore:")
place_tuning_page = st.Page(str(_app_dir / "pages/place_tuning_page.py"), title="Place Tuning", icon=":material/place:")
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
tracking_quality_page = st.Page(str(_app_dir / "pages/tracking_quality_page.py"), title="Tracking QC", icon=":material/bug_report:")
zdrift_page = st.Page(str(_app_dir / "pages/zdrift_page.py"), title="Z-Drift", icon=":material/straighten:")
anatomy_page = st.Page(str(_app_dir / "pages/anatomy_page.py"), title="Anatomy", icon=":material/neurology:")
hypotheses_page = st.Page(str(_app_dir / "pages/hypotheses_page.py"), title="Hypotheses", icon=":material/science:")
patching_page = st.Page(str(_app_dir / "pages/patching_page.py"), title="Patching", icon=":material/electric_bolt:")
patching_traces_page = st.Page(str(_app_dir / "pages/patching_traces_page.py"), title="Patching Traces", icon=":material/show_chart:")
patching_morph_page = st.Page(str(_app_dir / "pages/patching_morph_page.py"), title="Morphology", icon=":material/account_tree:")
moseq_page = st.Page(str(_app_dir / "pages/moseq_page.py"), title="MoSeq", icon=":material/pets:")
moseq_explore_page = st.Page(str(_app_dir / "pages/moseq_explore_page.py"), title="MoSeq Explore", icon=":material/travel_explore:")
behaviour_page = st.Page(str(_app_dir / "pages/behaviour_page.py"), title="Behaviour", icon=":material/directions_run:")
changelog_page = st.Page(str(_app_dir / "pages/changelog_page.py"), title="Changelog", icon=":material/history:")
cost_page = st.Page(str(_app_dir / "pages/cost_page.py"), title="Costs", icon=":material/attach_money:")
aws_page = st.Page(str(_app_dir / "pages/aws_page.py"), title="AWS", icon=":material/cloud:")

pg = st.navigation({
    "Overview": [home_page, sessions_page, animals_page, pipeline_page, summary_page],
    "Pipeline": [suite2p_page, calcium_page, dlc_page, tracking_quality_page, sync_page, zdrift_page, anatomy_page, moseq_page],
    "Explore": [explorer_page, timeline_page, gallery_page, events_page, event_dynamics_page, correlations_page, trace_compare_page, moseq_explore_page, behaviour_page],
    "Analysis": [hypotheses_page, analysis_page, compare_page, population_page, light_page, light_compare_page, stats_page, maze_page, hd_tuning_page, place_tuning_page, decoder_page, stability_page, drift_page, gain_page, anchoring_page, speed_mod_page, pop_dynamics_page, ahv_page, info_theory_page, classify_page, signal_quality_page, qc_page, patching_page, patching_traces_page, patching_morph_page],
    "System": [aws_page, cost_page, changelog_page],
})
pg.run()
