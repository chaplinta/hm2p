"""Z-Drift — Visualize focal plane drift during imaging sessions.

Shows z-position over time (estimated from z-stack registration) for each
session that has a z-stack. Helps identify sessions with excessive z-drift
that may compromise ROI stability.
"""

from __future__ import annotations

import logging

import streamlit as st

log = logging.getLogger(__name__)

st.header("Z-Drift Analysis")

st.markdown("""
Estimates focal plane drift by registering each imaging frame against the
session's serial2p z-stack. Sessions without a z-stack are skipped.

**Method:** Phase-correlation registration of each frame against all z-planes,
followed by Gaussian smoothing and argmax to get z-position per frame.

> Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
> two-photon microscopy." *bioRxiv*. doi:10.1101/061507
""")

# ── Imports ──────────────────────────────────────────────────────────────
import io

import numpy as np
import pandas as pd

try:
    from frontend.data import (
        DERIVATIVES_BUCKET,
        download_s3_bytes,
        load_experiments,
        parse_session_id,
    )
except ImportError:
    st.error("Frontend data module not available.")
    st.stop()

# ── Helpers ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=1800)
def _load_zdrift(bucket: str, key: str) -> dict | None:
    """Download zdrift.h5 from S3 and parse it."""
    import h5py

    data = download_s3_bytes(bucket, key)
    if data is None:
        return None
    try:
        with h5py.File(io.BytesIO(data), "r") as f:
            result = {}
            for k in f.keys():
                result[k] = f[k][()]
            # Read attrs
            for attr_key in f.attrs:
                result[f"_attr_{attr_key}"] = f.attrs[attr_key]
            return result
    except Exception as e:
        log.warning("Failed to parse zdrift.h5: %s", e)
        return None


# ── Load experiments with z-stacks ───────────────────────────────────────

experiments = load_experiments()
zstack_sessions = [e for e in experiments if e.get("zstack_id", "").strip()]

if not zstack_sessions:
    st.warning("No sessions have z-stack IDs in experiments.csv.")
    st.stop()

st.info(f"{len(zstack_sessions)} sessions have z-stacks (out of {len(experiments)} total)")

# ── Session selector ─────────────────────────────────────────────────────

session_options = {e["exp_id"]: e for e in zstack_sessions}
selected_exp_id = st.selectbox(
    "Select session",
    options=list(session_options.keys()),
    format_func=lambda x: f"{x} (z-stack: {session_options[x]['zstack_id']})",
)

if not selected_exp_id:
    st.stop()

exp = session_options[selected_exp_id]
sub, ses = parse_session_id(selected_exp_id)
zstack_id = exp["zstack_id"]

st.markdown(f"**Z-stack:** `{zstack_id}` | **Animal:** `{sub}` | **Session:** `{ses}`")

# ── Load z-drift data ────────────────────────────────────────────────────

zdrift_key = f"derivatives/zdrift/{sub}/{ses}/zdrift.h5"
zdrift = _load_zdrift(DERIVATIVES_BUCKET, zdrift_key)

if zdrift is None:
    st.warning(
        f"No z-drift data found at `s3://{DERIVATIVES_BUCKET}/{zdrift_key}`. "
        "Z-drift computation has not been run for this session yet."
    )
    st.stop()

zpos = zdrift.get("zpos")
zcorr = zdrift.get("zcorr")
zpos_smooth = zdrift.get("zpos_smooth")
n_zplanes = int(zdrift.get("_attr_n_zplanes", zdrift.get("n_zplanes", 0)))

if zpos is None:
    st.error("zdrift.h5 is missing 'zpos' dataset.")
    st.stop()

n_frames = len(zpos)

# ── Summary metrics ──────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Frames", f"{n_frames:,}")
col2.metric("Z-planes", n_zplanes)
col3.metric("Z range", f"{int(np.min(zpos))}–{int(np.max(zpos))}")
drift_range = int(np.max(zpos)) - int(np.min(zpos))
col4.metric("Drift (planes)", drift_range)

# ── Z-position over time ─────────────────────────────────────────────────

st.subheader("Z-position over time")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(zpos, alpha=0.3, linewidth=0.5, label="Raw z-pos")
if zpos_smooth is not None:
    ax.plot(zpos_smooth, color="red", linewidth=1.0, label="Smoothed")
ax.set_xlabel("Frame")
ax.set_ylabel("Z-plane index")
ax.set_title(f"Z-drift: {selected_exp_id}")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, n_frames)
st.pyplot(fig)
plt.close(fig)

# ── Correlation heatmap ──────────────────────────────────────────────────

if zcorr is not None:
    st.subheader("Z-correlation heatmap")
    st.caption("Correlation of each frame with each z-plane. Bright = high correlation.")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    # zcorr shape: (n_frames, n_zplanes) — transpose for display
    aspect = max(1, n_frames // (n_zplanes * 20))
    ax2.imshow(
        zcorr.T,
        aspect="auto",
        cmap="viridis",
        interpolation="none",
    )
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Z-plane")
    ax2.set_title("Frame-to-Z correlation")
    st.pyplot(fig2)
    plt.close(fig2)

# ── Drift statistics ─────────────────────────────────────────────────────

st.subheader("Drift statistics")

# Split into quarters
quarter = n_frames // 4
quarters = []
for i in range(4):
    start = i * quarter
    end = (i + 1) * quarter if i < 3 else n_frames
    q = zpos[start:end]
    quarters.append({
        "Quarter": f"Q{i+1}",
        "Frames": f"{start:,}–{end:,}",
        "Mean z-pos": f"{np.mean(q):.1f}",
        "Std": f"{np.std(q):.2f}",
        "Min": int(np.min(q)),
        "Max": int(np.max(q)),
    })

st.dataframe(pd.DataFrame(quarters), use_container_width=True)

# ── All sessions overview ────────────────────────────────────────────────

st.subheader("All sessions: z-drift summary")

overview_rows = []
for exp_row in zstack_sessions:
    eid = exp_row["exp_id"]
    s, ss = parse_session_id(eid)
    key = f"derivatives/zdrift/{s}/{ss}/zdrift.h5"
    zd = _load_zdrift(DERIVATIVES_BUCKET, key)
    if zd is not None and "zpos" in zd:
        zp = zd["zpos"]
        overview_rows.append({
            "Session": eid,
            "Z-stack": exp_row["zstack_id"],
            "Frames": len(zp),
            "Mean z-pos": f"{np.mean(zp):.1f}",
            "Drift range": int(np.max(zp)) - int(np.min(zp)),
            "Std": f"{np.std(zp):.2f}",
        })
    else:
        overview_rows.append({
            "Session": eid,
            "Z-stack": exp_row["zstack_id"],
            "Frames": "—",
            "Mean z-pos": "—",
            "Drift range": "—",
            "Std": "not computed",
        })

if overview_rows:
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

# ── Methods ──────────────────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
    **Z-drift estimation** uses phase-correlation registration of each imaging
    frame against all planes of the session's serial2p z-stack. The maximum
    correlation across z-planes gives the estimated focal plane position for
    each frame. Gaussian smoothing (default sigma=2 frames) reduces noise.

    **References:**
    - Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
      two-photon microscopy." *bioRxiv*. doi:10.1101/061507
    """)
