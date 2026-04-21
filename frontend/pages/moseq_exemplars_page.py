"""MoSeq Exemplars — typical instances of each behavioural syllable.

Displays the 3 most typical video clips for each MoSeq syllable alongside
the crowd movie (averaged frames). Helps build intuition for what each
syllable corresponds to in the animal's behaviour.

"Most typical" = bouts whose duration is closest to the median for that
syllable, sampled from different sessions where possible.

Reference:
    Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point
    tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

log = logging.getLogger(__name__)

st.title("MoSeq Syllable Exemplars")
st.caption(
    "Typical video clips for each behavioural syllable. Green border marks "
    "the active bout; surrounding frames show context. Crowd movie (averaged "
    "across many bouts) shown for comparison."
)

# ── Imports ──────────────────────────────────────────────────────────────

try:
    from frontend.data import (
        DERIVATIVES_BUCKET,
        download_s3_bytes,
        load_exemplar_summary,
    )
except ImportError as _imp_err:
    st.error(f"Frontend data module not available: {_imp_err}")
    st.stop()

if st.button("Refresh", key="refresh_moseq_exemplars"):
    st.cache_data.clear()

# ── Load data ──────────────────────────────────────────────────────────────

with st.spinner("Loading exemplar clip data from S3..."):
    summary = load_exemplar_summary()

if summary is None:
    st.warning(
        "No exemplar clips found on S3 yet. "
        "Run `python scripts/render_exemplar_clips.py` to generate them."
    )
    st.info(
        "The script finds the 3 most typical bouts per syllable across all "
        "sessions, extracts video clips from the labelled pose videos, and "
        "uploads them to S3. It requires downloading session videos, so run "
        "it on a machine with good bandwidth to ap-southeast-2."
    )
    st.stop()

syllables = summary.get("syllables", [])
if not syllables:
    st.warning("Exemplar summary loaded but contains no syllable data.")
    st.stop()

# ── Overview metrics ───────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Syllables", summary.get("n_syllables", len(syllables)))
col2.metric("Clips per syllable", summary.get("n_exemplars_per_syllable", "?"))
col3.metric("Total clips", summary.get("total_clips_rendered", "?"))
col4.metric("Sessions used", summary.get("n_sessions", "?"))

st.divider()

# ── Display each syllable ──────────────────────────────────────────────────


def _load_crowd_movie_bytes(syl_id: int) -> bytes | None:
    """Load a crowd movie MP4 from S3."""
    key = f"kinematics/crowd_movies/syllable_{syl_id}.mp4"
    return download_s3_bytes(DERIVATIVES_BUCKET, key)


def _load_exemplar_bytes(s3_key: str) -> bytes | None:
    """Load an exemplar clip MP4 from S3."""
    return download_s3_bytes(DERIVATIVES_BUCKET, s3_key)


for syl_info in syllables:
    syl_id = syl_info["syllable_id"]
    total_frames = syl_info.get("total_frames", 0)
    total_bouts = syl_info.get("total_bouts", 0)
    median_dur_sec = syl_info.get("median_duration_sec", 0)
    median_dur_frames = syl_info.get("median_duration_frames", 0)
    exemplars = syl_info.get("exemplars", [])

    # Header
    st.subheader(f"Syllable {syl_id}")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total frames", f"{total_frames:,}")
    mc2.metric("Total bouts", f"{total_bouts:,}")
    mc3.metric("Median bout", f"{median_dur_sec}s ({median_dur_frames} frames)")

    # Video row: crowd movie + exemplars
    n_exemplars = len(exemplars)
    n_cols = 1 + n_exemplars  # crowd movie + exemplars
    cols = st.columns(n_cols)

    # Crowd movie in first column
    with cols[0]:
        st.markdown("**Crowd movie** (averaged)")
        crowd_bytes = _load_crowd_movie_bytes(syl_id)
        if crowd_bytes is not None:
            st.video(crowd_bytes, format="video/mp4", loop=True, autoplay=True)
        else:
            st.info("No crowd movie")

    # Exemplar clips
    for i, ex in enumerate(exemplars):
        with cols[1 + i]:
            sub = ex.get("sub", "?")
            ses = ex.get("ses", "?")
            bout_dur = ex.get("bout_duration_frames", 0)
            bout_sec = ex.get("bout_duration_sec", 0)
            st.markdown(f"**Exemplar {i + 1}**")

            s3_key = ex.get("s3_key", "")
            if s3_key:
                clip_bytes = _load_exemplar_bytes(s3_key)
                if clip_bytes is not None:
                    st.video(clip_bytes, format="video/mp4", loop=True, autoplay=True)
                else:
                    st.warning("Clip not found on S3")
            else:
                st.warning("No S3 key for this exemplar")

            st.caption(
                f"{sub}/{ses}\n\n"
                f"Bout: {bout_dur} frames ({bout_sec}s)"
            )

    st.divider()

# ── Methods & References ────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
    **Exemplar selection:** For each syllable, all bouts (contiguous runs of
    the same syllable ID) are collected across sessions. The median bout
    duration is computed, and the 3 bouts closest to the median duration are
    selected, preferring bouts from different sessions for diversity.

    **Video clips** are extracted from the labelled pose videos
    (`labelled_30fps.mp4`) with ~0.5s context before and after the bout.
    A green border marks the frames where the syllable is active.

    **Crowd movies** show the average of many aligned bouts of the same
    syllable --- they reveal the stereotyped component of the movement while
    individual exemplars show the natural variation.

    **Reference:**

    Weinreb, C., Osman, A., Datta, S.R., & Mathis, A. (2024).
    "Keypoint-MoSeq: parsing behavior by linking point tracking to pose
    dynamics." *Nature Methods*, 21(9), 1329-1339.
    [doi:10.1038/s41592-024-02318-2](https://doi.org/10.1038/s41592-024-02318-2).
    https://github.com/dattalab/keypoint-moseq
    """)
