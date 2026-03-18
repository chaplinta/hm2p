"""MoSeq Explore — cross-session syllable analysis and celltype comparison.

Browse keypoint-MoSeq syllable usage, transitions, ethograms, and
compare syllable distributions between Penk+ and CamKII+ animals.

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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hm2p.constants import HEX_NONPENK, HEX_PENK

log = logging.getLogger(__name__)

st.title("MoSeq Explore")
st.caption(
    "Cross-session syllable analysis: usage distributions, transition matrices, "
    "ethograms, and Penk+ vs CamKII+ comparisons."
)

# ── Imports ──────────────────────────────────────────────────────────────

try:
    from frontend.data import (
        DERIVATIVES_BUCKET,
        download_s3_bytes,
        load_all_syllable_data,
        load_animals,
        load_syllable_npz,
        sanitize_error,
    )
except ImportError as _imp_err:
    st.error(f"Frontend data module not available: {_imp_err}")
    st.stop()

if st.button("Refresh", key="refresh_moseq_explore"):
    st.cache_data.clear()
    # Also clear session-state cached syllable data
    for k in list(st.session_state.keys()):
        if "syllable" in k:
            del st.session_state[k]

# ── Load all syllable data ──────────────────────────────────────────────

with st.spinner("Loading syllable data from S3..."):
    syl_data = load_all_syllable_data()

syl_sessions = syl_data["sessions"]

if not syl_sessions:
    st.warning(
        "No syllable outputs found on S3 yet. "
        "keypoint-MoSeq may still be running --- check the MoSeq pipeline status page."
    )
    st.stop()

st.info(f"Loaded syllable data for **{syl_data['n_sessions']}** sessions.")

# ── Compute global syllable set ─────────────────────────────────────────

all_syl_ids_global: set[int] = set()
for s in syl_sessions:
    all_syl_ids_global.update(np.unique(s["syllable_id"]).tolist())
all_syl_sorted = sorted(all_syl_ids_global)
n_total_syllables = len(all_syl_sorted)
syl_to_global_idx = {s: i for i, s in enumerate(all_syl_sorted)}

# ── Tab layout ──────────────────────────────────────────────────────────

tab_usage, tab_per_session, tab_transitions, tab_celltype, tab_ethogram = st.tabs([
    "Syllable Usage",
    "Per-Session Distribution",
    "Transition Matrix",
    "Celltype Comparison",
    "Ethogram",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: Pooled syllable usage histogram
# ════════════════════════════════════════════════════════════════════════

with tab_usage:
    st.subheader("Pooled Syllable Usage")
    st.caption("Total frame counts per syllable across all sessions.")

    pooled_counts: dict[int, int] = {}
    for s in syl_sessions:
        unique, counts = np.unique(s["syllable_id"], return_counts=True)
        for sid, cnt in zip(unique, counts):
            pooled_counts[int(sid)] = pooled_counts.get(int(sid), 0) + int(cnt)

    sorted_ids = sorted(pooled_counts.keys(),
                        key=lambda x: pooled_counts[x], reverse=True)
    total_frames = sum(pooled_counts.values())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total syllable types", n_total_syllables)
    col2.metric("Total frames", f"{total_frames:,}")
    col3.metric("Sessions", syl_data["n_sessions"])

    fig_usage = go.Figure(data=[go.Bar(
        x=[str(s) for s in sorted_ids],
        y=[pooled_counts[s] for s in sorted_ids],
        marker_color="steelblue",
        hovertemplate="Syllable %{x}<br>Frames: %{y:,}<br>Fraction: %{customdata:.1%}<extra></extra>",
        customdata=[pooled_counts[s] / total_frames for s in sorted_ids],
    )])
    fig_usage.update_layout(
        xaxis_title="Syllable ID (ranked by frequency)",
        yaxis_title="Frame count",
        height=400,
    )
    st.plotly_chart(fig_usage, use_container_width=True)

    # Usage table
    usage_rows = [
        {"Syllable": sid, "Frames": pooled_counts[sid],
         "Fraction": pooled_counts[sid] / total_frames}
        for sid in sorted_ids
    ]
    with st.expander(f"Usage table ({n_total_syllables} syllables)"):
        st.dataframe(
            pd.DataFrame(usage_rows).style.format({"Fraction": "{:.2%}"}),
            use_container_width=True,
        )

# ════════════════════════════════════════════════════════════════════════
# TAB 2: Per-session syllable distribution (heatmap)
# ════════════════════════════════════════════════════════════════════════

with tab_per_session:
    st.subheader("Per-Session Syllable Distribution")
    st.caption(
        "Fraction of time spent in each syllable per session. "
        "Rows = sessions, columns = syllable IDs (sorted by global frequency)."
    )

    # Build matrix: (n_sessions x n_syllables) as fractions
    session_labels = []
    frac_matrix = np.zeros((len(syl_sessions), n_total_syllables))

    for i, s in enumerate(syl_sessions):
        label = f"{s['sub']}/{s['ses']} ({s['celltype']})"
        session_labels.append(label)
        unique, counts = np.unique(s["syllable_id"], return_counts=True)
        total = counts.sum()
        for sid, cnt in zip(unique, counts):
            col_idx = syl_to_global_idx[int(sid)]
            frac_matrix[i, col_idx] = cnt / total

    # Order columns by global frequency
    col_order = [syl_to_global_idx[sid] for sid in sorted_ids]
    col_labels = [str(sid) for sid in sorted_ids]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=frac_matrix[:, col_order],
        x=col_labels,
        y=session_labels,
        colorscale="Viridis",
        colorbar=dict(title="Fraction"),
        hovertemplate="Session: %{y}<br>Syllable: %{x}<br>Fraction: %{z:.3f}<extra></extra>",
    ))
    fig_heatmap.update_layout(
        xaxis_title="Syllable ID (ranked by global frequency)",
        yaxis_title="Session",
        height=max(400, 25 * len(syl_sessions)),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Also show as stacked bar
    with st.expander("Stacked bar view"):
        # Show top N syllables in stacked bar
        top_n = min(15, n_total_syllables)
        fig_stacked = go.Figure()
        for rank in range(top_n):
            sid = sorted_ids[rank]
            col_idx = syl_to_global_idx[sid]
            fig_stacked.add_trace(go.Bar(
                name=f"Syl {sid}",
                x=session_labels,
                y=frac_matrix[:, col_idx],
            ))
        # Add "other" category
        other_cols = [syl_to_global_idx[sid] for sid in sorted_ids[top_n:]]
        if other_cols:
            other_frac = frac_matrix[:, other_cols].sum(axis=1)
            fig_stacked.add_trace(go.Bar(
                name="Other",
                x=session_labels,
                y=other_frac,
                marker_color="grey",
            ))
        fig_stacked.update_layout(
            barmode="stack",
            xaxis_title="Session",
            yaxis_title="Fraction",
            height=500,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 3: Transition matrix (pooled across sessions)
# ════════════════════════════════════════════════════════════════════════

with tab_transitions:
    st.subheader("Syllable Transition Matrix")
    st.caption(
        "Probability of transitioning from syllable i (row) to syllable j (column), "
        "pooled across all sessions."
    )

    # Build pooled transition count matrix
    trans_counts = np.zeros((n_total_syllables, n_total_syllables), dtype=np.int64)
    for s in syl_sessions:
        ids = s["syllable_id"]
        for t in range(len(ids) - 1):
            from_idx = syl_to_global_idx[int(ids[t])]
            to_idx = syl_to_global_idx[int(ids[t + 1])]
            trans_counts[from_idx, to_idx] += 1

    # Normalize to row probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans_counts / row_sums

    # Show top N syllables
    max_show = st.slider(
        "Max syllables to display", 5,
        min(n_total_syllables, 30),
        min(n_total_syllables, 15),
        key="trans_max_show",
    )
    top_global_idx = [syl_to_global_idx[sid] for sid in sorted_ids[:max_show]]
    top_labels = [str(sid) for sid in sorted_ids[:max_show]]

    fig_trans = go.Figure(data=go.Heatmap(
        z=trans_prob[np.ix_(top_global_idx, top_global_idx)],
        x=top_labels,
        y=top_labels,
        colorscale="Blues",
        colorbar=dict(title="P(j|i)"),
        hovertemplate="From %{y} -> To %{x}<br>P = %{z:.3f}<extra></extra>",
    ))
    fig_trans.update_layout(
        xaxis_title="To syllable",
        yaxis_title="From syllable",
        height=500,
        width=600,
    )
    st.plotly_chart(fig_trans)

    # Self-transition (stickiness) bar
    st.caption("Self-transition probability (diagonal) --- how 'sticky' each syllable is.")
    diag_probs = [trans_prob[syl_to_global_idx[sid], syl_to_global_idx[sid]]
                  for sid in sorted_ids[:max_show]]
    fig_diag = go.Figure(data=[go.Bar(
        x=top_labels,
        y=diag_probs,
        marker_color="darkcyan",
    )])
    fig_diag.update_layout(
        xaxis_title="Syllable ID",
        yaxis_title="P(stay)",
        height=300,
    )
    st.plotly_chart(fig_diag, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 4: Celltype comparison
# ════════════════════════════════════════════════════════════════════════

with tab_celltype:
    st.subheader("Penk+ vs CamKII+ Syllable Usage")
    st.caption(
        "Compare syllable usage fractions between cell types. "
        "Each data point is one animal's mean usage fraction for a syllable. "
        "Mann-Whitney U test per syllable."
    )

    # Group sessions by animal, then by celltype
    # Average syllable fractions per animal (some animals have multiple sessions)
    animal_fracs: dict[str, dict] = {}  # animal_id -> {celltype, fracs: np.array}
    for s in syl_sessions:
        aid = s["animal_id"]
        if aid not in animal_fracs:
            animal_fracs[aid] = {
                "celltype": s["celltype"],
                "frac_sum": np.zeros(n_total_syllables),
                "n_sessions": 0,
            }
        unique, counts = np.unique(s["syllable_id"], return_counts=True)
        total = counts.sum()
        for sid, cnt in zip(unique, counts):
            animal_fracs[aid]["frac_sum"][syl_to_global_idx[int(sid)]] += cnt / total
        animal_fracs[aid]["n_sessions"] += 1

    # Average across sessions per animal
    for aid in animal_fracs:
        n = animal_fracs[aid]["n_sessions"]
        animal_fracs[aid]["fracs"] = animal_fracs[aid]["frac_sum"] / n

    penk_animals = {aid: v for aid, v in animal_fracs.items() if v["celltype"] == "penk"}
    nonpenk_animals = {aid: v for aid, v in animal_fracs.items() if v["celltype"] == "nonpenk"}

    col1, col2 = st.columns(2)
    col1.metric("Penk+ animals", len(penk_animals))
    col2.metric("CamKII+ animals", len(nonpenk_animals))

    if len(penk_animals) >= 2 and len(nonpenk_animals) >= 2:
        # Build comparison for top syllables
        top_n_compare = min(15, n_total_syllables)
        comparison_rows = []

        penk_mat = np.array([v["fracs"] for v in penk_animals.values()])
        nonpenk_mat = np.array([v["fracs"] for v in nonpenk_animals.values()])

        for rank in range(top_n_compare):
            sid = sorted_ids[rank]
            col_idx = syl_to_global_idx[sid]
            penk_vals = penk_mat[:, col_idx]
            nonpenk_vals = nonpenk_mat[:, col_idx]

            # Mann-Whitney U test
            try:
                stat_u, p_val = sp_stats.mannwhitneyu(
                    penk_vals, nonpenk_vals, alternative="two-sided"
                )
            except ValueError:
                stat_u, p_val = np.nan, 1.0

            comparison_rows.append({
                "Syllable": sid,
                "Penk+ mean": np.mean(penk_vals),
                "CamKII+ mean": np.mean(nonpenk_vals),
                "U statistic": stat_u,
                "p-value": p_val,
            })

        comp_df = pd.DataFrame(comparison_rows)

        # Grouped bar chart
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Penk+",
            x=[str(r["Syllable"]) for r in comparison_rows],
            y=[r["Penk+ mean"] for r in comparison_rows],
            marker_color=HEX_PENK,
            error_y=dict(
                type="data",
                array=[np.std(penk_mat[:, syl_to_global_idx[r["Syllable"]]]) for r in comparison_rows],
                visible=True,
            ),
        ))
        fig_comp.add_trace(go.Bar(
            name="CamKII+",
            x=[str(r["Syllable"]) for r in comparison_rows],
            y=[r["CamKII+ mean"] for r in comparison_rows],
            marker_color=HEX_NONPENK,
            error_y=dict(
                type="data",
                array=[np.std(nonpenk_mat[:, syl_to_global_idx[r["Syllable"]]]) for r in comparison_rows],
                visible=True,
            ),
        ))
        fig_comp.update_layout(
            barmode="group",
            xaxis_title="Syllable ID",
            yaxis_title="Mean usage fraction",
            height=400,
            legend=dict(x=0.85, y=0.95),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Show stats table
        sig_rows = [r for r in comparison_rows if r["p-value"] < 0.05]
        if sig_rows:
            st.markdown(f"**{len(sig_rows)}** syllables with p < 0.05 (uncorrected):")
        st.dataframe(
            comp_df.style.format({
                "Penk+ mean": "{:.3f}",
                "CamKII+ mean": "{:.3f}",
                "U statistic": "{:.0f}",
                "p-value": "{:.4f}",
            }).apply(
                lambda row: ["background-color: #ffffcc" if row["p-value"] < 0.05 else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
        )

        # Overall distribution difference: KL divergence or cosine similarity
        penk_pooled = penk_mat.mean(axis=0)
        nonpenk_pooled = nonpenk_mat.mean(axis=0)
        # Normalize to proper distributions
        penk_dist = penk_pooled / (penk_pooled.sum() + 1e-10)
        nonpenk_dist = nonpenk_pooled / (nonpenk_pooled.sum() + 1e-10)

        # Jensen-Shannon divergence
        m = 0.5 * (penk_dist + nonpenk_dist)
        kl_pm = np.sum(penk_dist * np.log2((penk_dist + 1e-10) / (m + 1e-10)))
        kl_nm = np.sum(nonpenk_dist * np.log2((nonpenk_dist + 1e-10) / (m + 1e-10)))
        jsd = 0.5 * (kl_pm + kl_nm)

        # Cosine similarity
        cos_sim = np.dot(penk_dist, nonpenk_dist) / (
            np.linalg.norm(penk_dist) * np.linalg.norm(nonpenk_dist) + 1e-10
        )

        col1, col2 = st.columns(2)
        col1.metric("Jensen-Shannon divergence", f"{jsd:.4f}")
        col2.metric("Cosine similarity", f"{cos_sim:.4f}")

    else:
        st.warning(
            "Need at least 2 animals per cell type for comparison. "
            f"Found {len(penk_animals)} Penk+ and {len(nonpenk_animals)} CamKII+ animals."
        )

# ════════════════════════════════════════════════════════════════════════
# TAB 5: Ethogram (single-session time series)
# ════════════════════════════════════════════════════════════════════════

with tab_ethogram:
    st.subheader("Ethogram")
    st.caption("Syllable identity over time for a selected session (colour-coded).")

    # Session selector
    session_options = []
    for s in syl_sessions:
        label = f"{s['sub']} / {s['ses']} ({s['celltype']})"
        session_options.append(label)

    selected_label = st.selectbox(
        "Select session",
        options=session_options,
        key="moseq_ethogram_session",
    )

    if selected_label:
        sel_idx = session_options.index(selected_label)
        sel = syl_sessions[sel_idx]
        syl_ids = sel["syllable_id"]
        n_frames = len(syl_ids)

        # Metrics
        unique_syls, syl_counts = np.unique(syl_ids, return_counts=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frames", f"{n_frames:,}")
        col2.metric("Syllable types", len(unique_syls))
        col3.metric("Most common", int(unique_syls[np.argmax(syl_counts)]))
        col4.metric("Dominance", f"{syl_counts.max() / syl_counts.sum():.1%}")

        # Downsample for display
        max_display = 5000
        if n_frames > max_display:
            step = n_frames // max_display
            display_ids = syl_ids[::step]
            display_t = np.arange(len(display_ids)) * step
        else:
            display_ids = syl_ids
            display_t = np.arange(n_frames)

        # Ethogram as colour bar (heatmap with 1 row)
        fig_ethogram = go.Figure()

        # Scatter approach for ethogram (colour-coded points)
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

        # Per-session usage bar
        sort_order = np.argsort(-syl_counts)
        fig_ses_usage = go.Figure(data=[go.Bar(
            x=[str(unique_syls[i]) for i in sort_order],
            y=[syl_counts[i] for i in sort_order],
            marker_color="steelblue",
        )])
        fig_ses_usage.update_layout(
            xaxis_title="Syllable ID",
            yaxis_title="Frame count",
            height=300,
        )
        st.plotly_chart(fig_ses_usage, use_container_width=True)

        # Bout duration stats
        with st.expander("Bout durations"):
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

            st.dataframe(
                pd.DataFrame(dur_stats).style.format({
                    "Mean dur (frames)": "{:.1f}",
                    "Median dur": "{:.0f}",
                }),
                use_container_width=True,
            )

        # Syllable posterior entropy (if prob available)
        npz_data = load_syllable_npz(sel["key"])
        if npz_data is not None:
            syl_prob = npz_data.get("syllable_prob", npz_data.get("syllable_probs"))
            if syl_prob is not None and syl_prob.ndim == 2:
                with st.expander("Posterior entropy"):
                    eps = 1e-10
                    entropy = -np.sum(syl_prob * np.log2(syl_prob + eps), axis=1)
                    if len(entropy) > max_display:
                        step = len(entropy) // max_display
                        ent_x = np.arange(0, len(entropy), step)
                        ent_y = entropy[::step]
                    else:
                        ent_x = np.arange(len(entropy))
                        ent_y = entropy
                    fig_ent = go.Figure(data=go.Scattergl(
                        x=ent_x, y=ent_y,
                        mode="lines", line=dict(width=1, color="purple"),
                    ))
                    fig_ent.update_layout(
                        xaxis_title="Frame",
                        yaxis_title="Entropy (bits)",
                        height=250,
                    )
                    st.plotly_chart(fig_ent, use_container_width=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Mean entropy", f"{entropy.mean():.2f} bits")
                    c2.metric("Max possible", f"{np.log2(syl_prob.shape[1]):.2f} bits")

# ── Methods & References ────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
    **keypoint-MoSeq** discovers behavioural syllables --- brief, reused motifs
    of movement --- from pose tracking data without any manual labeling.
    It fits an autoregressive hidden Markov model (AR-HMM) to keypoint
    trajectories, segmenting continuous behaviour into discrete states.

    **Transition matrix** shows P(syllable_j | syllable_i), the probability
    of transitioning from one syllable to another, pooled across all sessions.
    Strong diagonal = sticky syllables (long bouts). Off-diagonal structure
    reveals sequential motifs.

    **Celltype comparison** averages syllable usage fractions per animal
    (across sessions), then compares Penk+ vs CamKII+ with Mann-Whitney U
    tests. Jensen-Shannon divergence and cosine similarity quantify overall
    distribution similarity.

    **References:**

    Weinreb, C., Osman, A., Datta, S.R., & Mathis, A. (2024).
    "Keypoint-MoSeq: parsing behavior by linking point tracking to pose
    dynamics." *Nature Methods*, 21(9), 1329-1339.
    [doi:10.1038/s41592-024-02318-2](https://doi.org/10.1038/s41592-024-02318-2).
    https://github.com/dattalab/keypoint-moseq
    """)
