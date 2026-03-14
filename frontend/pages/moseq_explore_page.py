"""MoSeq Explore — interactive syllable exploration per session.

Browse keypoint-MoSeq syllable assignments, usage distributions,
transition matrices, and syllable-behaviour relationships for
individual sessions.

Reference:
    Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point
    tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hm2p.constants import HEX_PENK, HEX_NONPENK

log = logging.getLogger(__name__)

st.title("MoSeq Explore")
st.caption(
    "Interactive exploration of keypoint-MoSeq syllable assignments. "
    "Select a session to view syllable usage, transitions, and ethograms."
)

# ── Imports ──────────────────────────────────────────────────────────────

try:
    from frontend.data import (
        DERIVATIVES_BUCKET,
        download_s3_bytes,
        get_s3_client,
        load_animals,
        load_experiments,
        parse_session_id,
        sanitize_error,
    )
except ImportError as _imp_err:
    st.error(f"Frontend data module not available: {_imp_err}")
    st.stop()

if st.button("Refresh", key="refresh_moseq_explore"):
    st.cache_data.clear()


# ── Data loading ─────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def _list_syllable_sessions() -> list[dict]:
    """List sessions that have syllables.npz on S3."""
    try:
        s3 = get_s3_client()
        resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix="kinematics/")
        results = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("syllables.npz"):
                parts = key.split("/")
                sub = parts[1] if len(parts) > 1 else "—"
                ses = parts[2] if len(parts) > 2 else "—"
                results.append({"sub": sub, "ses": ses, "key": key, "size": obj["Size"]})
        return results
    except Exception as e:
        log.warning("Failed to list syllable sessions: %s", e)
        return []


@st.cache_data(ttl=600)
def _load_syllable_data(s3_key: str) -> dict | None:
    """Load syllables.npz from S3 and return arrays."""
    try:
        data = download_s3_bytes(DERIVATIVES_BUCKET, s3_key)
        if data is None:
            return None
        npz = np.load(io.BytesIO(data))
        result = {}
        for k in npz.files:
            result[k] = npz[k]
        return result
    except Exception as e:
        log.warning("Failed to load syllable data from %s: %s", s3_key, e)
        return None


@st.cache_data(ttl=600)
def _load_sync_data(sub: str, ses: str) -> dict | None:
    """Load sync.h5 for a session (for behaviour overlay)."""
    import h5py

    key = f"sync/{sub}/{ses}/sync.h5"
    try:
        data = download_s3_bytes(DERIVATIVES_BUCKET, key)
        if data is None:
            return None
        f = h5py.File(io.BytesIO(data), "r")
        result = {}
        for k in f.keys():
            result[k] = f[k][:]
        f.close()
        return result
    except Exception as e:
        log.warning("Failed to load sync data: %s", e)
        return None


# ── Session selector ─────────────────────────────────────────────────────

syllable_sessions = _list_syllable_sessions()

if not syllable_sessions:
    st.warning(
        "No syllable outputs found on S3 yet. "
        "keypoint-MoSeq may still be running — check the MoSeq pipeline status page."
    )
    st.stop()

experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

# Build session labels
session_options = []
for ss in syllable_sessions:
    # Try to find matching experiment
    animal_id = ss["sub"].replace("sub-", "")
    celltype = animal_map.get(animal_id, {}).get("celltype", "?")
    label = f"{ss['sub']} / {ss['ses']} ({celltype})"
    session_options.append((label, ss))

selected_label = st.selectbox(
    "Select session",
    options=[opt[0] for opt in session_options],
    key="moseq_explore_session",
)

if not selected_label:
    st.stop()

selected = next(opt[1] for opt in session_options if opt[0] == selected_label)

# ── Load syllable data ───────────────────────────────────────────────────

st.header(f"{selected['sub']} / {selected['ses']}")

syl_data = _load_syllable_data(selected["key"])
if syl_data is None:
    st.error("Failed to load syllable data from S3.")
    st.stop()

# Extract syllable IDs
syl_ids = syl_data.get("syllable_id", syl_data.get("syllable_ids"))
if syl_ids is None:
    st.error("No syllable_id array found in .npz file.")
    st.stop()

syl_ids = syl_ids.astype(int)
n_frames = len(syl_ids)
unique_syls, syl_counts = np.unique(syl_ids, return_counts=True)
n_syllables = len(unique_syls)

# ── Overview metrics ─────────────────────────────────────────────────────

st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Frames", f"{n_frames:,}")
col2.metric("Syllable types", n_syllables)
col3.metric("Most common", int(unique_syls[np.argmax(syl_counts)]))
col4.metric("Dominance", f"{syl_counts.max() / syl_counts.sum():.1%}")

# ── Syllable usage distribution ──────────────────────────────────────────

st.subheader("Syllable Usage Distribution")

sort_order = np.argsort(-syl_counts)
fig_usage = go.Figure(data=[go.Bar(
    x=[str(unique_syls[i]) for i in sort_order],
    y=[syl_counts[i] for i in sort_order],
    marker_color="steelblue",
)])
fig_usage.update_layout(
    xaxis_title="Syllable ID",
    yaxis_title="Frame count",
    height=350,
)
st.plotly_chart(fig_usage, use_container_width=True)

# Usage table
usage_df = pd.DataFrame({
    "Syllable": unique_syls[sort_order],
    "Frames": syl_counts[sort_order],
    "Fraction": syl_counts[sort_order] / syl_counts.sum(),
}).reset_index(drop=True)
with st.expander(f"Usage table ({n_syllables} syllables)"):
    st.dataframe(
        usage_df.style.format({"Fraction": "{:.2%}"}),
        use_container_width=True,
    )

# ── Ethogram (syllable timeline) ─────────────────────────────────────────

st.subheader("Ethogram")
st.caption("Syllable identity over time. Each colour is a distinct syllable.")

# Downsample for display if too many frames
max_display = 5000
if n_frames > max_display:
    step = n_frames // max_display
    display_ids = syl_ids[::step]
    display_t = np.arange(len(display_ids)) * step
else:
    display_ids = syl_ids
    display_t = np.arange(n_frames)

fig_ethogram = go.Figure()
fig_ethogram.add_trace(go.Scattergl(
    x=display_t,
    y=display_ids,
    mode="markers",
    marker=dict(
        size=2,
        color=display_ids,
        colorscale="Turbo",
        showscale=True,
        colorbar=dict(title="Syllable"),
    ),
))
fig_ethogram.update_layout(
    xaxis_title="Frame",
    yaxis_title="Syllable ID",
    height=300,
)
st.plotly_chart(fig_ethogram, use_container_width=True)

# ── Transition matrix ────────────────────────────────────────────────────

st.subheader("Transition Matrix")
st.caption("Probability of transitioning from syllable i (row) to syllable j (column).")

# Build transition count matrix
syl_to_idx = {s: i for i, s in enumerate(unique_syls)}
trans_counts = np.zeros((n_syllables, n_syllables), dtype=int)
for i in range(len(syl_ids) - 1):
    from_idx = syl_to_idx[syl_ids[i]]
    to_idx = syl_to_idx[syl_ids[i + 1]]
    trans_counts[from_idx, to_idx] += 1

# Normalize to probabilities
row_sums = trans_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # avoid division by zero
trans_prob = trans_counts / row_sums

# Only show top N syllables for readability
max_show = st.slider("Max syllables to show", 5, min(n_syllables, 30), min(n_syllables, 15))
top_idx = sort_order[:max_show]
top_labels = [str(unique_syls[i]) for i in top_idx]

fig_trans = go.Figure(data=go.Heatmap(
    z=trans_prob[np.ix_(top_idx, top_idx)],
    x=top_labels,
    y=top_labels,
    colorscale="Blues",
    colorbar=dict(title="P(j|i)"),
))
fig_trans.update_layout(
    xaxis_title="To syllable",
    yaxis_title="From syllable",
    height=500,
    width=500,
)
st.plotly_chart(fig_trans)

# ── Syllable duration distribution ───────────────────────────────────────

st.subheader("Syllable Bout Durations")
st.caption("Distribution of consecutive bout lengths (in frames) for each syllable.")

# Compute bout durations
bout_durations: dict[int, list[int]] = {s: [] for s in unique_syls}
current_syl = syl_ids[0]
current_len = 1
for i in range(1, len(syl_ids)):
    if syl_ids[i] == current_syl:
        current_len += 1
    else:
        bout_durations[current_syl].append(current_len)
        current_syl = syl_ids[i]
        current_len = 1
bout_durations[current_syl].append(current_len)

# Summary stats
dur_stats = []
for syl in unique_syls[sort_order]:
    durs = bout_durations[syl]
    if durs:
        dur_stats.append({
            "Syllable": syl,
            "N bouts": len(durs),
            "Mean dur (frames)": np.mean(durs),
            "Median dur": np.median(durs),
            "Max dur": max(durs),
        })

dur_df = pd.DataFrame(dur_stats)
st.dataframe(
    dur_df.style.format({
        "Mean dur (frames)": "{:.1f}",
        "Median dur": "{:.0f}",
    }),
    use_container_width=True,
)

# Box plot of durations for top syllables
top_n_box = min(10, n_syllables)
fig_dur = go.Figure()
for syl in unique_syls[sort_order[:top_n_box]]:
    durs = bout_durations[syl]
    fig_dur.add_trace(go.Box(y=durs, name=str(syl)))
fig_dur.update_layout(
    xaxis_title="Syllable ID",
    yaxis_title="Bout duration (frames)",
    height=350,
    showlegend=False,
)
st.plotly_chart(fig_dur, use_container_width=True)

# ── Syllable-behaviour overlay (if sync data available) ──────────────────

st.subheader("Syllable × Behaviour")

sync_data = _load_sync_data(selected["sub"], selected["ses"])

if sync_data is not None and "hd" in sync_data:
    hd = sync_data["hd"]
    speed = sync_data.get("speed")
    ahv = sync_data.get("ahv")

    # Syllables are at camera rate, sync is at imaging rate — need to map
    # For now, show per-syllable HD distribution
    st.caption(
        "Mean head direction and speed per syllable bout. "
        "Syllables at camera rate; sync data at imaging rate."
    )

    # Resample syllable IDs to imaging rate (nearest-neighbour)
    n_imaging = len(hd)
    if n_frames > 0 and n_imaging > 0:
        ratio = n_frames / n_imaging
        imaging_syl = syl_ids[np.clip(
            (np.arange(n_imaging) * ratio).astype(int), 0, n_frames - 1
        )]

        # Per-syllable HD circular mean
        syl_hd_stats = []
        for syl in unique_syls[sort_order[:20]]:
            mask = imaging_syl == syl
            if mask.sum() < 5:
                continue
            hd_masked = hd[mask]
            # Circular mean
            hd_rad = np.deg2rad(hd_masked)
            cmean = np.rad2deg(np.arctan2(np.mean(np.sin(hd_rad)), np.mean(np.cos(hd_rad)))) % 360
            row = {"Syllable": syl, "Circular mean HD": cmean, "N frames": int(mask.sum())}
            if speed is not None:
                row["Mean speed"] = float(np.nanmean(speed[mask]))
            if ahv is not None:
                row["Mean |AHV|"] = float(np.nanmean(np.abs(ahv[mask])))
            syl_hd_stats.append(row)

        if syl_hd_stats:
            hd_df = pd.DataFrame(syl_hd_stats)
            st.dataframe(
                hd_df.style.format({
                    "Circular mean HD": "{:.1f}",
                    "Mean speed": "{:.1f}",
                    "Mean |AHV|": "{:.1f}",
                }),
                use_container_width=True,
            )

            # Polar plot: syllable HD preference
            if len(syl_hd_stats) > 1:
                fig_polar = go.Figure()
                for row in syl_hd_stats:
                    fig_polar.add_trace(go.Scatterpolar(
                        r=[row["N frames"]],
                        theta=[row["Circular mean HD"]],
                        mode="markers+text",
                        text=[str(row["Syllable"])],
                        textposition="top center",
                        marker=dict(size=10),
                        name=f"Syl {row['Syllable']}",
                        showlegend=False,
                    ))
                fig_polar.update_layout(
                    title="Syllable HD preference",
                    polar=dict(angularaxis=dict(direction="clockwise")),
                    height=400,
                )
                st.plotly_chart(fig_polar, use_container_width=True)

            # Speed by syllable
            if speed is not None and len(syl_hd_stats) > 1:
                fig_speed = go.Figure(data=[go.Bar(
                    x=[str(r["Syllable"]) for r in syl_hd_stats],
                    y=[r.get("Mean speed", 0) for r in syl_hd_stats],
                    marker_color="darkorange",
                )])
                fig_speed.update_layout(
                    xaxis_title="Syllable ID",
                    yaxis_title="Mean speed (cm/s)",
                    height=300,
                )
                st.plotly_chart(fig_speed, use_container_width=True)
        else:
            st.info("Not enough frames per syllable to compute behaviour stats.")
    else:
        st.info("Cannot map syllable to sync data (empty arrays).")
else:
    st.info(
        "No sync.h5 available for this session — "
        "syllable-behaviour overlay requires synced data."
    )

# ── Syllable probabilities (if available) ────────────────────────────────

syl_prob = syl_data.get("syllable_prob", syl_data.get("syllable_probs"))
if syl_prob is not None and syl_prob.ndim == 2:
    st.subheader("Syllable Posterior Probabilities")
    st.caption("Model confidence in syllable assignment over time.")

    # Show entropy over time as a confidence measure
    eps = 1e-10
    entropy = -np.sum(syl_prob * np.log2(syl_prob + eps), axis=1)

    fig_entropy = go.Figure()
    # Downsample
    if len(entropy) > max_display:
        step = len(entropy) // max_display
        fig_entropy.add_trace(go.Scattergl(
            x=np.arange(0, len(entropy), step),
            y=entropy[::step],
            mode="lines",
            line=dict(width=1, color="purple"),
        ))
    else:
        fig_entropy.add_trace(go.Scatter(
            x=np.arange(len(entropy)),
            y=entropy,
            mode="lines",
            line=dict(width=1, color="purple"),
        ))
    fig_entropy.update_layout(
        xaxis_title="Frame",
        yaxis_title="Entropy (bits)",
        height=250,
    )
    st.plotly_chart(fig_entropy, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Mean entropy", f"{entropy.mean():.2f} bits")
    col2.metric("Max possible", f"{np.log2(syl_prob.shape[1]):.2f} bits")

# ── Methods ──────────────────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
    **keypoint-MoSeq** discovers behavioural syllables — brief, reused motifs
    of movement — from pose tracking data without any manual labeling.
    It fits an autoregressive hidden Markov model (AR-HMM) to keypoint
    trajectories, segmenting continuous behaviour into discrete states.

    **Transition matrix** shows P(syllable_j | syllable_i), the probability
    of transitioning from one syllable to another. Strong diagonal = sticky
    syllables (long bouts). Off-diagonal structure reveals sequential motifs.

    **Syllable-behaviour overlay** resamples syllable IDs (camera rate) to
    imaging rate via nearest-neighbour interpolation, then computes per-syllable
    circular mean HD, mean speed, and mean |AHV|.

    **References:**

    Weinreb, C., Osman, A., Datta, S.R., & Mathis, A. (2024).
    "Keypoint-MoSeq: parsing behavior by linking point tracking to pose
    dynamics." *Nature Methods*, 21(9), 1329-1339.
    [doi:10.1038/s41592-024-02318-2](https://doi.org/10.1038/s41592-024-02318-2).
    https://github.com/dattalab/keypoint-moseq
    """)
