"""ROI Gallery — grid view of all ROIs across all sessions with spatial images,
traces, classifier features, and filtering.

By default shows ALL sessions pooled. Users can filter by session, celltype,
animal, or ROI type in the sidebar.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    load_all_ca_data,
    load_all_suite2p_spatial,
)

log = logging.getLogger("hm2p.frontend.gallery")

st.title("ROI Gallery")
st.caption(
    "Browse all ROIs across all sessions. Shows spatial footprints on the mean image, "
    "dF/F traces, and classifier feature visualisation."
)


# ── Load all sessions ──────────────────────────────────────────────────────

with st.spinner("Loading ROI data from S3..."):
    ca_sessions = load_all_ca_data()
    spatial_data = load_all_suite2p_spatial()

    # Merge calcium data with Suite2p spatial metadata
    all_sessions = []
    for ses in ca_sessions:
        exp_id = ses["exp_id"]
        spatial = spatial_data.get(exp_id, {})
        shape_features = spatial.get("shape_features", [None] * ses["n_rois"])
        # Pad shape_features to match n_rois if needed
        if len(shape_features) < ses["n_rois"]:
            shape_features = list(shape_features) + [None] * (ses["n_rois"] - len(shape_features))
        all_sessions.append({
            **ses,
            "mean_img": spatial.get("mean_img"),
            "shape_features": shape_features,
        })

if not all_sessions:
    st.warning("No calcium data (ca.h5) available yet. This page will populate "
               "when Stage 4 (calcium processing) completes.")
    st.stop()


# ── Sidebar filters ────────────────────────────────────────────────────────

celltypes = sorted(set(s["celltype"] for s in all_sessions))
animal_ids = sorted(set(s["animal_id"] for s in all_sessions))
exp_ids = sorted(set(s["exp_id"] for s in all_sessions))

ROI_TYPE_NAMES = {0: "soma", 1: "dend", 2: "artefact"}

with st.sidebar:
    st.header("Filters")
    sel_celltypes = st.multiselect("Cell type", celltypes, default=celltypes, key="gal_ct")
    sel_animals = st.multiselect("Animal", animal_ids, default=animal_ids, key="gal_animal")
    sel_sessions = st.multiselect("Session", exp_ids, default=exp_ids, key="gal_session")
    roi_type_filter = st.radio(
        "ROI type", ["Soma only", "Dendrite only", "All ROIs"],
        index=0, key="gal_roi_type",
    )
    min_snr = st.slider("Min SNR", 0.0, 15.0, 0.0, 0.5, key="gal_snr")
    sort_by = st.selectbox(
        "Sort by", ["Session + ROI index", "SNR (high first)", "Event rate", "Max dF/F", "Aspect ratio"],
        key="gal_sort",
    )
    n_cols = st.selectbox("Grid columns", [3, 4, 5, 6], index=1, key="gal_cols")
    max_rois = st.selectbox("Max ROIs shown", [12, 24, 48, 96, 200], index=2, key="gal_max")

# Apply filters
sessions = [
    s for s in all_sessions
    if s["celltype"] in sel_celltypes
    and s["animal_id"] in sel_animals
    and s["exp_id"] in sel_sessions
]

if not sessions:
    st.warning("No sessions match the current filters.")
    st.stop()


# ── Build per-ROI DataFrame ───────────────────────────────────────────────

rows = []
for ses in sessions:
    dff = ses["dff"]
    n_rois = ses["n_rois"]
    n_frames = ses["n_frames"]
    fps = ses["fps"]

    for i in range(n_rois):
        roi_type_code = int(ses["roi_types"][i])
        roi_type_name = ROI_TYPE_NAMES.get(roi_type_code, "unknown")

        # Apply ROI type filter
        if roi_type_filter == "Soma only" and roi_type_code != 0:
            continue
        if roi_type_filter == "Dendrite only" and roi_type_code != 1:
            continue

        trace = dff[i]
        baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
        peak95 = np.percentile(trace, 95)
        snr = peak95 / baseline_std if baseline_std > 0 else 0.0

        if snr < min_snr:
            continue

        # Shape features
        sf = ses["shape_features"][i] if i < len(ses["shape_features"]) else None

        rows.append({
            "exp_id": ses["exp_id"],
            "animal_id": ses["animal_id"],
            "celltype": ses["celltype"],
            "roi_local": i,
            "roi_type": roi_type_name,
            "roi_type_code": roi_type_code,
            "snr": snr,
            "mean_dff": float(np.nanmean(trace)),
            "max_dff": float(np.nanmax(trace)),
            "skewness": float(((trace - trace.mean()) ** 3).mean() / max(trace.std() ** 3, 1e-12)),
            "aspect_ratio": sf["aspect_ratio"] if sf else np.nan,
            "radius": sf["radius"] if sf else np.nan,
            "compact": sf["compact"] if sf else np.nan,
            "npix": sf["npix"] if sf else 0,
            # Store references for plotting
            "_ses_idx": sessions.index(ses),
            "_roi_idx": i,
        })

if not rows:
    st.warning("No ROIs match the current filters.")
    st.stop()

roi_df = pd.DataFrame(rows)

# Sort
if sort_by == "SNR (high first)":
    roi_df = roi_df.sort_values("snr", ascending=False)
elif sort_by == "Max dF/F":
    roi_df = roi_df.sort_values("max_dff", ascending=False)
elif sort_by == "Aspect ratio":
    roi_df = roi_df.sort_values("aspect_ratio", ascending=False, na_position="last")
# Default: Session + ROI index (already in order)

roi_df = roi_df.head(max_rois).reset_index(drop=True)


# ── Top metrics ─────────────────────────────────────────────────────────────

n_total_all = sum(s["n_rois"] for s in sessions)
n_shown = len(roi_df)
n_soma = (roi_df["roi_type"] == "soma").sum()
n_dend = (roi_df["roi_type"] == "dend").sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sessions", len(sessions))
col2.metric("Total ROIs", n_total_all)
col3.metric("Shown", n_shown)
col4.metric("Soma", n_soma)
col5.metric("Dendrite", n_dend)

st.markdown("---")


# ── Tabs ────────────────────────────────────────────────────────────────────

tab_gallery, tab_classifier, tab_features = st.tabs([
    "Gallery", "Classifier Decisions", "Feature Distributions",
])


# ── Tab 1: Gallery grid ────────────────────────────────────────────────────

with tab_gallery:
    st.subheader("ROI Traces")

    for row_start in range(0, len(roi_df), n_cols):
        chunk = roi_df.iloc[row_start : row_start + n_cols]
        cols = st.columns(n_cols)

        for col_idx, (_, roi_row) in enumerate(chunk.iterrows()):
            with cols[col_idx]:
                ses = sessions[roi_row["_ses_idx"]]
                roi_i = roi_row["_roi_idx"]
                trace = ses["dff"][roi_i]
                n_frames = len(trace)
                fps = ses["fps"]

                # Downsample for performance
                ds = max(1, n_frames // 500)
                time_ds = np.arange(0, n_frames, ds) / fps
                trace_ds = trace[::ds]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_ds, y=trace_ds,
                    mode="lines",
                    line=dict(color="black", width=0.5),
                    showlegend=False,
                ))
                fig.update_layout(
                    height=120,
                    margin=dict(l=25, r=5, t=20, b=15),
                    title=dict(text=f"ROI {roi_i}", font=dict(size=10)),
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                )
                st.plotly_chart(fig, use_container_width=True,
                               key=f"gal_trace_{roi_row['exp_id']}_{roi_i}")

                # ROI spatial footprint on mean image (if available)
                sf = ses["shape_features"][roi_i] if roi_i < len(ses["shape_features"]) else None
                mean_img = ses["mean_img"]

                if mean_img is not None and sf is not None and "ypix" in sf and len(sf["ypix"]) > 0:
                    import matplotlib.pyplot as plt
                    from matplotlib.colors import Normalize

                    # Crop around ROI for a zoomed view
                    ypix, xpix = sf["ypix"], sf["xpix"]
                    pad = 15
                    y0 = max(0, int(ypix.min()) - pad)
                    y1 = min(mean_img.shape[0], int(ypix.max()) + pad)
                    x0 = max(0, int(xpix.min()) - pad)
                    x1 = min(mean_img.shape[1], int(xpix.max()) + pad)

                    crop = mean_img[y0:y1, x0:x1]
                    vmin, vmax = np.percentile(crop, [1, 99])

                    fig_img, ax = plt.subplots(figsize=(1.5, 1.5), dpi=80)
                    ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
                    # Overlay ROI pixels
                    color = {"soma": "cyan", "dend": "orange", "artefact": "red"}.get(
                        roi_row["roi_type"], "white"
                    )
                    y_local = ypix - y0
                    x_local = xpix - x0
                    ax.scatter(x_local, y_local, s=0.3, c=color, alpha=0.6)
                    ax.set_axis_off()
                    plt.tight_layout(pad=0)
                    st.pyplot(fig_img, use_container_width=True)
                    plt.close(fig_img)

                # Compact info line
                type_emoji = {"soma": "S", "dend": "D", "artefact": "X"}.get(roi_row["roi_type"], "?")
                ar_str = f"AR:{roi_row['aspect_ratio']:.1f}" if not np.isnan(roi_row["aspect_ratio"]) else ""
                st.markdown(
                    f"<div style='font-size:10px; line-height:1.2'>"
                    f"<b>{ses['exp_id'][:8]}</b> | {ses['celltype']} | [{type_emoji}] "
                    f"SNR:{roi_row['snr']:.1f} {ar_str}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ── Tab 2: Classifier decisions ────────────────────────────────────────────

with tab_classifier:
    st.subheader("Soma / Dendrite Classification")
    st.markdown(
        "ROIs are classified using Suite2p `stat.npy` shape features:\n\n"
        "- **Artefact**: `radius < 2.0` or `compact < 0.1`\n"
        "- **Dendrite**: `aspect_ratio > 2.5`\n"
        "- **Soma**: everything else\n\n"
        "The scatter plots below show how each ROI falls relative to the "
        "decision boundaries."
    )

    has_features = roi_df["aspect_ratio"].notna().any()
    if not has_features:
        st.warning("No shape features available (stat.npy not found on S3).")
    else:
        df_feat = roi_df[roi_df["aspect_ratio"].notna()].copy()

        # Main decision plot: aspect_ratio vs radius, colored by classification
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df_feat, x="radius", y="aspect_ratio",
                color="roi_type",
                color_discrete_map={"soma": "#1f77b4", "dend": "#ff7f0e", "artefact": "#d62728"},
                hover_data=["exp_id", "roi_local", "compact", "snr"],
                title="Aspect Ratio vs Radius",
                opacity=0.6,
            )
            # Decision boundaries
            fig.add_hline(y=2.5, line_dash="dash", line_color="orange",
                         annotation_text="AR=2.5 (dend threshold)")
            fig.add_vline(x=2.0, line_dash="dash", line_color="red",
                         annotation_text="r=2.0 (artefact)")
            fig.update_layout(height=400, margin=dict(l=50, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True, key="clf_ar_radius")

        with col2:
            fig = px.scatter(
                df_feat, x="compact", y="aspect_ratio",
                color="roi_type",
                color_discrete_map={"soma": "#1f77b4", "dend": "#ff7f0e", "artefact": "#d62728"},
                hover_data=["exp_id", "roi_local", "radius", "snr"],
                title="Aspect Ratio vs Compactness",
                opacity=0.6,
            )
            fig.add_hline(y=2.5, line_dash="dash", line_color="orange",
                         annotation_text="AR=2.5 (dend)")
            fig.add_vline(x=0.1, line_dash="dash", line_color="red",
                         annotation_text="compact=0.1 (artefact)")
            fig.update_layout(height=400, margin=dict(l=50, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True, key="clf_ar_compact")

        # Counts per session
        st.subheader("Classification Counts")
        counts = df_feat.groupby(["exp_id", "celltype", "roi_type"]).size().reset_index(name="count")
        pivot = counts.pivot_table(
            index=["exp_id", "celltype"], columns="roi_type",
            values="count", fill_value=0,
        ).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)

        # Mean image with all ROI footprints for a selected session
        st.subheader("ROI Map")
        map_sessions = [s for s in sessions if s["mean_img"] is not None]
        if map_sessions:
            map_exp_ids = [s["exp_id"] for s in map_sessions]
            sel_map = st.selectbox("Session for ROI map", map_exp_ids, key="gal_map_ses")
            ses_map = next(s for s in map_sessions if s["exp_id"] == sel_map)

            import matplotlib.pyplot as plt

            mean_img = ses_map["mean_img"]
            vmin, vmax = np.percentile(mean_img, [1, 99])

            fig_map, ax = plt.subplots(figsize=(8, 8), dpi=100)
            ax.imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax)

            colors = {"soma": "cyan", "dend": "orange", "artefact": "red"}
            type_names = {0: "soma", 1: "dend", 2: "artefact"}
            plotted_types = set()

            for i in range(ses_map["n_rois"]):
                sf = ses_map["shape_features"][i] if i < len(ses_map["shape_features"]) else None
                if sf is None or "ypix" not in sf or len(sf["ypix"]) == 0:
                    continue
                rt = type_names.get(int(ses_map["roi_types"][i]), "soma")
                c = colors.get(rt, "white")
                label = rt if rt not in plotted_types else None
                plotted_types.add(rt)
                ax.scatter(sf["xpix"], sf["ypix"], s=0.2, c=c, alpha=0.5, label=label)

            ax.legend(loc="upper right", fontsize=8, markerscale=10)
            ax.set_title(f"{sel_map} — {ses_map['celltype']}", fontsize=10)
            ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig_map, use_container_width=True)
            plt.close(fig_map)
        else:
            st.info("No mean images available (ops.npy not found on S3).")


# ── Tab 3: Feature distributions ───────────────────────────────────────────

with tab_features:
    st.subheader("Shape Feature Distributions")

    has_features = roi_df["aspect_ratio"].notna().any()
    if not has_features:
        st.warning("No shape features available (stat.npy not found).")
    else:
        df_feat = roi_df[roi_df["aspect_ratio"].notna()].copy()

        feature_metrics = [
            ("aspect_ratio", "Aspect Ratio"),
            ("radius", "Radius"),
            ("compact", "Compactness"),
            ("npix", "Number of Pixels"),
            ("snr", "SNR"),
            ("max_dff", "Max dF/F"),
        ]

        for row_start in range(0, len(feature_metrics), 3):
            cols = st.columns(3)
            for col_idx, (metric, label) in enumerate(feature_metrics[row_start:row_start + 3]):
                with cols[col_idx]:
                    fig = px.histogram(
                        df_feat, x=metric, color="roi_type", nbins=30,
                        color_discrete_map={"soma": "#1f77b4", "dend": "#ff7f0e", "artefact": "#d62728"},
                        title=label, barmode="overlay", opacity=0.7,
                    )
                    fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True, key=f"feat_{metric}")

        # Soma vs dendrite comparison
        st.subheader("Soma vs Dendrite Comparison")
        df_sd = df_feat[df_feat["roi_type"].isin(["soma", "dend"])]
        if len(df_sd) > 0 and df_sd["roi_type"].nunique() >= 2:
            comp_metrics = [
                ("snr", "SNR"),
                ("max_dff", "Max dF/F"),
                ("skewness", "Trace Skewness"),
                ("aspect_ratio", "Aspect Ratio"),
            ]
            cols = st.columns(len(comp_metrics))
            for col_idx, (metric, label) in enumerate(comp_metrics):
                with cols[col_idx]:
                    fig = px.box(
                        df_sd, x="roi_type", y=metric, color="roi_type",
                        color_discrete_map={"soma": "#1f77b4", "dend": "#ff7f0e"},
                        points="all", title=label,
                    )
                    fig.update_traces(marker=dict(size=3, opacity=0.4))
                    fig.update_layout(height=300, showlegend=False,
                                     margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True, key=f"comp_{metric}")

            # Summary table
            st.subheader("Summary by ROI Type")
            summary = df_sd.groupby("roi_type")[
                ["snr", "max_dff", "aspect_ratio", "radius", "compact", "npix"]
            ].agg(["count", "median", "mean", "std"])
            summary.columns = [f"{c}_{s}" for c, s in summary.columns]
            st.dataframe(summary.T, use_container_width=True)


# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("Methods"):
    st.markdown("""
**ROI extraction:** Suite2p (Pachitariu et al. 2017, doi:10.1101/061507).
[GitHub](https://github.com/MouseLand/suite2p)

**Soma/dendrite classification:** Heuristic from Suite2p `stat.npy` shape features:
- `radius < 2.0` or `compact < 0.1` → artefact
- `aspect_ratio > 2.5` → dendrite
- Otherwise → soma

Single imaging plane — soma and dendrite ROIs co-exist. Analysis defaults to
soma only.
""")
