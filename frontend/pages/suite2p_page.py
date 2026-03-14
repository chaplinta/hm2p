"""Suite2p viewer — ROI maps, traces, classification, and TIFF images."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import io
import logging

import numpy as np
import streamlit as st

import boto3

from frontend.data import (
    DERIVATIVES_BUCKET,
    RAWDATA_BUCKET,
    REGION,
    download_s3_bytes,
    download_s3_numpy,
    list_s3_session_files,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend")

st.title("Suite2p Results")

experiments = load_experiments()
exp_ids = [e["exp_id"] for e in experiments]


# ── S3 completion summary ─────────────────────────────────────────────────
@st.cache_data(ttl=120)
def get_suite2p_session_summary() -> list[dict]:
    """Check each session for Suite2p output on S3 and count ROIs."""
    import json

    s3 = boto3.client("s3", region_name=REGION)
    results = []
    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0/"
        info: dict = {"exp_id": exp_id, "sub": sub, "ses": ses, "done": False, "n_rois": 0, "n_cells": 0}
        try:
            resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=1)
            if resp.get("KeyCount", 0) > 0:
                info["done"] = True
                # Try to get ROI count from iscell.npy
                try:
                    import io as _io

                    obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=prefix + "iscell.npy")
                    iscell_data = np.load(_io.BytesIO(obj["Body"].read()), allow_pickle=False)
                    info["n_rois"] = len(iscell_data)
                    info["n_cells"] = int(iscell_data[:, 0].sum())
                except Exception:
                    pass
        except Exception:
            pass
        results.append(info)
    return results


with st.expander("Suite2p S3 Completion Summary", expanded=False):
    if st.button("Refresh Suite2p summary"):
        get_suite2p_session_summary.clear()
    try:
        summary = get_suite2p_session_summary()
        n_done = sum(1 for s in summary if s["done"])
        total_rois = sum(s["n_rois"] for s in summary)
        total_cells = sum(s["n_cells"] for s in summary)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sessions complete", f"{n_done}/{len(summary)}")
        col2.metric("Total ROIs", total_rois)
        col3.metric("Total cells", total_cells)
        col4.metric("Avg cells/session", f"{total_cells / max(n_done, 1):.0f}")

        if n_done > 0:
            st.progress(n_done / len(summary), text=f"{n_done}/{len(summary)} sessions processed")

            import pandas as pd

            df = pd.DataFrame(summary)
            df["status"] = df["done"].map({True: "Done", False: "Pending"})
            st.dataframe(
                df[["exp_id", "status", "n_rois", "n_cells"]].rename(
                    columns={"exp_id": "Session", "status": "Status", "n_rois": "ROIs", "n_cells": "Cells"}
                ),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as e:
        log.exception("Error fetching Suite2p summary")
        st.warning("Could not query S3. Check server logs for details.")

st.markdown("---")

# ── Per-session viewer ────────────────────────────────────────────────────

# Use selected session from Sessions page, or let user pick
default_idx = 0
if "selected_exp_id" in st.session_state:
    sel = st.session_state["selected_exp_id"]
    if sel in exp_ids:
        default_idx = exp_ids.index(sel)

selected = st.selectbox("Session", exp_ids, index=default_idx)
sub, ses = parse_session_id(selected)
s3_prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0/"

# Check what files exist
files = list_s3_session_files(DERIVATIVES_BUCKET, s3_prefix)
if not files:
    st.warning(f"No Suite2p output found at `{s3_prefix}`")
    st.stop()

file_names = [f["key"].split("/")[-1] for f in files]
st.caption(
    f"Found {len(files)} files: {', '.join(file_names[:15])}"
    + ("..." if len(files) > 15 else "")
)

# Load key arrays
with st.spinner("Loading Suite2p data from S3..."):
    ops = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "ops.npy", allow_pickle=True)
    stat = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "stat.npy", allow_pickle=True)
    iscell = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "iscell.npy")
    f_traces = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "F.npy")
    f_neu = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "Fneu.npy")
    spks = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "spks.npy")

if ops is None:
    st.error("Could not load ops.npy")
    st.stop()

ops_dict = ops.item() if isinstance(ops, np.ndarray) and ops.ndim == 0 else ops

# Summary stats
n_rois = len(iscell) if iscell is not None else 0
n_cells = int(iscell[:, 0].sum()) if iscell is not None else 0
n_frames = ops_dict.get("nframes", 0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROIs", n_rois)
col2.metric("Cells", n_cells)
col3.metric("Non-cells", n_rois - n_cells)
col4.metric("Frames", n_frames)

# Classifier info
classif = ops_dict.get("classification", {})
if classif:
    cpath = classif.get("classifier_path", "built-in")
    st.caption(f"Classifier: `{cpath}` | built-in: {classif.get('use_builtin_classifier', True)}")

# Compute calcium stats for cells
cell_mask = iscell[:, 0].astype(bool) if iscell is not None else np.ones(len(f_traces), dtype=bool)
if f_traces is not None and cell_mask.any():
    cell_f = f_traces[cell_mask]
    cell_neu = f_neu[cell_mask] if f_neu is not None else None

    # SNR: mean(dF/F0 peak) / std(baseline)
    snrs = []
    for i in range(len(cell_f)):
        trace = cell_f[i]
        f0 = np.percentile(trace, 10)
        if f0 > 0:
            dff = (trace - f0) / f0
            baseline_std = np.std(dff[dff < np.percentile(dff, 50)])
            peak = np.percentile(dff, 95)
            snrs.append(peak / baseline_std if baseline_std > 0 else 0)
        else:
            snrs.append(0)
    snrs = np.array(snrs)

    # Skewness of traces (high skew = bursty activity = likely real cell)
    from scipy.stats import skew as scipy_skew
    skews = np.array([scipy_skew(cell_f[i]) for i in range(len(cell_f))])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Median SNR", f"{np.median(snrs):.1f}")
    col2.metric("Mean skewness", f"{np.mean(skews):.2f}")
    col3.metric("Duration (s)", f"{n_frames / ops_dict.get('fs', 30):.0f}")
    col4.metric("Frame rate", f"{ops_dict.get('fs', 0):.1f} Hz")

# Tabs
tab_map, tab_traces, tab_soma, tab_class, tab_stats, tab_reg, tab_tiff = st.tabs(
    ["ROI Map", "Traces", "Soma / Dendrite", "Classification", "Cell Stats", "Registration", "TIFF Images"]
)


# ── ROI Map ──────────────────────────────────────────────────────────────
with tab_map:
    import matplotlib.pyplot as plt

    mean_img = ops_dict.get("meanImg")
    if mean_img is None:
        st.warning("No meanImg in ops")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        vmin, vmax = np.percentile(mean_img, [1, 99])
        axes[0].imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("Mean Image")
        axes[0].axis("off")

        axes[1].imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax)
        ly, lx = mean_img.shape
        roi_img = np.zeros((ly, lx, 4), dtype=np.float32)

        for i, s in enumerate(stat):
            is_cell = bool(iscell[i, 0]) if iscell is not None and i < len(iscell) else True
            ypix = s["ypix"]
            xpix = s["xpix"]
            mask = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
            ypix, xpix = ypix[mask], xpix[mask]
            if is_cell:
                roi_img[ypix, xpix] = [0, 1, 0, 0.3]
            else:
                roi_img[ypix, xpix] = [1, 0, 0, 0.15]

        axes[1].imshow(roi_img)
        axes[1].set_title(f"ROIs ({n_cells} cells / {n_rois} total)")
        axes[1].axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── Traces ───────────────────────────────────────────────────────────────
with tab_traces:
    import matplotlib.pyplot as plt

    if f_traces is None:
        st.warning("No F.npy found")
    else:
        cell_indices = np.where(iscell[:, 0] == 1)[0] if iscell is not None else np.arange(len(f_traces))
        nc = len(cell_indices)
        if nc == 0:
            st.info("No cells classified.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                n_show = st.slider("Cells to show", 1, min(20, nc), min(5, nc))
                offset = st.slider("Start from cell #", 0, max(0, nc - n_show), 0)
            with col2:
                show_neuropil = st.checkbox("Show neuropil", value=False)
                show_deconv = st.checkbox("Show deconvolved", value=False)

            sel_cells = cell_indices[offset: offset + n_show]
            fig, axes = plt.subplots(len(sel_cells), 1, figsize=(14, 2 * len(sel_cells)), sharex=True)
            if len(sel_cells) == 1:
                axes = [axes]

            for ax, idx in zip(axes, sel_cells):
                trace = f_traces[idx]
                f0 = np.percentile(trace, 10)
                dff = (trace - f0) / f0 if f0 > 0 else trace
                ax.plot(dff, linewidth=0.5, color="black", label=f"Cell {idx}")

                if show_neuropil and f_neu is not None:
                    neu = f_neu[idx]
                    f0n = np.percentile(neu, 10)
                    dff_neu = (neu - f0n) / f0n if f0n > 0 else neu
                    ax.plot(dff_neu, linewidth=0.3, color="blue", alpha=0.5)

                if show_deconv and spks is not None:
                    ax.plot(spks[idx] / max(spks[idx].max(), 1), linewidth=0.3, color="red", alpha=0.5)

                prob = iscell[idx, 1] if iscell is not None else 0
                ax.set_ylabel(f"Cell {idx}\np={prob:.2f}", fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)

            axes[-1].set_xlabel("Frame")
            fig.suptitle("dF/F0 Traces (cells only)", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ── Soma / Dendrite ──────────────────────────────────────────────────────
with tab_soma:
    import matplotlib.pyplot as plt

    if stat is None or iscell is None or f_traces is None:
        st.warning("stat.npy, iscell.npy, and F.npy are required for soma/dendrite classification.")
    else:
        try:
            from hm2p.extraction.suite2p import classify_roi_types
        except (ImportError, ModuleNotFoundError):
            st.warning("Suite2p is not installed in this environment. Soma/dendrite classification requires suite2p.")
            classify_roi_types = None  # type: ignore[assignment]

        if classify_roi_types is None:
            st.stop()

        # Classify all ROIs (accepted + rejected)
        try:
            all_types = classify_roi_types(list(stat))
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as e:
            st.warning(f"Could not classify ROIs: {e}")
            st.stop()
        all_types = np.array(all_types)

        # Filter to accepted cells only
        cell_idx = np.where(cell_mask)[0]
        cell_types = all_types[cell_mask]
        soma_mask = cell_types == "soma"
        dend_mask = cell_types == "dend"
        artefact_mask = cell_types == "artefact"

        n_soma = int(soma_mask.sum())
        n_dend = int(dend_mask.sum())
        n_artefact = int(artefact_mask.sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Soma", n_soma)
        col2.metric("Dendrite", n_dend)
        col3.metric("Artefact", n_artefact)
        col4.metric("Total cells", n_cells)

        st.caption(
            "Classification uses Suite2p stat.npy shape features: "
            "aspect_ratio > 2.5 → dendrite, radius < 2 or compact < 0.1 → artefact, else → soma."
        )

        # --- Shape feature distributions ---
        st.subheader("Shape Features by Type")

        aspect_ratios = np.array([s.get("aspect_ratio", 1.0) for s in stat[cell_mask]])
        radii = np.array([s.get("radius", 5.0) for s in stat[cell_mask]])
        compacts = np.array([s.get("compact", 1.0) for s in stat[cell_mask]])
        npix = np.array([s.get("npix", 0) for s in stat[cell_mask]])

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        colors = {"soma": "#2196F3", "dend": "#FF5722", "artefact": "#9E9E9E"}

        for label, mask in [("soma", soma_mask), ("dend", dend_mask), ("artefact", artefact_mask)]:
            if mask.any():
                axes[0].hist(aspect_ratios[mask], bins=20, alpha=0.6, color=colors[label], label=f"{label} ({mask.sum()})")
                axes[1].hist(radii[mask], bins=20, alpha=0.6, color=colors[label], label=label)
                axes[2].hist(compacts[mask], bins=20, alpha=0.6, color=colors[label], label=label)

        axes[0].axvline(2.5, color="red", linestyle="--", linewidth=1, label="threshold")
        axes[0].set_xlabel("Aspect Ratio")
        axes[0].set_title("Aspect Ratio")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("Radius (px)")
        axes[1].set_title("Radius")
        axes[2].set_xlabel("Compactness")
        axes[2].set_title("Compactness")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- ROI map colored by type ---
        st.subheader("ROI Map — Soma vs Dendrite")
        mean_img = ops_dict.get("meanImg")
        if mean_img is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            vmin, vmax = np.percentile(mean_img, [1, 99])
            ax.imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax)

            ly, lx = mean_img.shape
            roi_img = np.zeros((ly, lx, 4), dtype=np.float32)

            for i, roi_idx in enumerate(cell_idx):
                s = stat[roi_idx]
                ypix = s["ypix"]
                xpix = s["xpix"]
                valid = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
                yp, xp = ypix[valid], xpix[valid]

                if cell_types[i] == "soma":
                    roi_img[yp, xp] = [0.13, 0.59, 0.95, 0.4]  # blue
                elif cell_types[i] == "dend":
                    roi_img[yp, xp] = [1.0, 0.34, 0.13, 0.4]   # orange
                else:
                    roi_img[yp, xp] = [0.62, 0.62, 0.62, 0.3]   # grey

            ax.imshow(roi_img)
            # Legend
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor=colors["soma"], alpha=0.6, label=f"Soma ({n_soma})"),
                    Patch(facecolor=colors["dend"], alpha=0.6, label=f"Dendrite ({n_dend})"),
                    Patch(facecolor=colors["artefact"], alpha=0.6, label=f"Artefact ({n_artefact})"),
                ],
                loc="upper right", fontsize=9,
            )
            ax.set_title("ROI Classification: Soma (blue) / Dendrite (orange)")
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # --- Example traces by type ---
        st.subheader("Example Traces")

        fs = ops_dict.get("fs", 30.0)
        n_example = st.slider("Traces per type", 1, 10, 3, key="soma_n_traces")

        for label, mask, color in [("Soma", soma_mask, "#2196F3"), ("Dendrite", dend_mask, "#FF5722")]:
            idx_of_type = cell_idx[mask]
            if len(idx_of_type) == 0:
                continue

            st.markdown(f"**{label}** ({len(idx_of_type)} total)")
            show_idx = idx_of_type[:n_example]

            fig, axes = plt.subplots(len(show_idx), 1, figsize=(14, 1.8 * len(show_idx)), sharex=True)
            if len(show_idx) == 1:
                axes = [axes]

            for ax, roi in zip(axes, show_idx):
                trace = f_traces[roi]
                f0 = np.percentile(trace, 10)
                dff = (trace - f0) / f0 if f0 > 0 else trace
                time_s = np.arange(len(dff)) / fs

                ax.plot(time_s, dff, linewidth=0.5, color=color)
                ar = stat[roi].get("aspect_ratio", 0)
                rad = stat[roi].get("radius", 0)
                ax.set_ylabel(f"ROI {roi}\nAR={ar:.1f}", fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"{label} dF/F0 Traces", fontsize=12, y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # --- Soma vs Dendrite comparison stats ---
        st.subheader("Soma vs Dendrite Comparison")

        if n_soma > 0 and n_dend > 0:
            # Compute dF/F0 stats per ROI
            soma_idx = cell_idx[soma_mask]
            dend_idx = cell_idx[dend_mask]

            def _roi_stats(indices):
                peak_dff = []
                mean_dff = []
                for roi in indices:
                    trace = f_traces[roi]
                    f0 = np.percentile(trace, 10)
                    if f0 > 0:
                        dff = (trace - f0) / f0
                        peak_dff.append(np.percentile(dff, 95))
                        mean_dff.append(np.mean(dff))
                    else:
                        peak_dff.append(0)
                        mean_dff.append(0)
                return np.array(peak_dff), np.array(mean_dff)

            soma_peak, soma_mean = _roi_stats(soma_idx)
            dend_peak, dend_mean = _roi_stats(dend_idx)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            # Peak dF/F
            axes[0].hist(soma_peak, bins=15, alpha=0.6, color=colors["soma"], label="Soma")
            axes[0].hist(dend_peak, bins=15, alpha=0.6, color=colors["dend"], label="Dendrite")
            axes[0].set_xlabel("Peak dF/F0 (95th percentile)")
            axes[0].set_title("Peak Activity")
            axes[0].legend(fontsize=8)

            # Mean dF/F
            axes[1].hist(soma_mean, bins=15, alpha=0.6, color=colors["soma"], label="Soma")
            axes[1].hist(dend_mean, bins=15, alpha=0.6, color=colors["dend"], label="Dendrite")
            axes[1].set_xlabel("Mean dF/F0")
            axes[1].set_title("Mean Activity")

            # Size comparison
            axes[2].hist(npix[soma_mask], bins=15, alpha=0.6, color=colors["soma"], label="Soma")
            axes[2].hist(npix[dend_mask], bins=15, alpha=0.6, color=colors["dend"], label="Dendrite")
            axes[2].set_xlabel("ROI size (pixels)")
            axes[2].set_title("ROI Size")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Summary table
            import pandas as pd
            comparison = pd.DataFrame({
                "Metric": ["Count", "Mean peak dF/F0", "Mean dF/F0", "Median size (px)", "Median aspect ratio"],
                "Soma": [
                    n_soma,
                    f"{soma_peak.mean():.3f}",
                    f"{soma_mean.mean():.3f}",
                    f"{np.median(npix[soma_mask]):.0f}",
                    f"{np.median(aspect_ratios[soma_mask]):.2f}",
                ],
                "Dendrite": [
                    n_dend,
                    f"{dend_peak.mean():.3f}",
                    f"{dend_mean.mean():.3f}",
                    f"{np.median(npix[dend_mask]):.0f}",
                    f"{np.median(aspect_ratios[dend_mask]):.2f}",
                ],
            })
            st.dataframe(comparison, hide_index=True, use_container_width=True)
        elif n_soma > 0:
            st.info("No dendrites detected in this session — all accepted ROIs are classified as soma.")
        else:
            st.info("No soma detected — unusual, check classification thresholds.")


# ── Classification ───────────────────────────────────────────────────────
with tab_class:
    import matplotlib.pyplot as plt

    if iscell is None:
        st.warning("No iscell.npy found")
    else:
        probs = iscell[:, 1]
        labels = iscell[:, 0].astype(bool)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs[labels], bins=30, alpha=0.7, color="green", label=f"Cells ({labels.sum()})")
        ax.hist(probs[~labels], bins=30, alpha=0.7, color="red", label=f"Non-cells ({(~labels).sum()})")
        ax.set_xlabel("Classification probability")
        ax.set_ylabel("Count")
        ax.set_title("ROI Classification")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── Cell Stats ────────────────────────────────────────────────────────────
with tab_stats:
    import matplotlib.pyplot as plt

    if f_traces is not None and cell_mask.any():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # SNR distribution
        axes[0, 0].hist(snrs, bins=25, color="steelblue", edgecolor="white")
        axes[0, 0].axvline(np.median(snrs), color="red", linestyle="--", label=f"median={np.median(snrs):.1f}")
        axes[0, 0].set_xlabel("SNR")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Signal-to-Noise Ratio")
        axes[0, 0].legend()

        # Skewness distribution
        axes[0, 1].hist(skews, bins=25, color="darkorange", edgecolor="white")
        axes[0, 1].axvline(np.median(skews), color="red", linestyle="--", label=f"median={np.median(skews):.2f}")
        axes[0, 1].set_xlabel("Skewness")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Trace Skewness (higher = burstier)")
        axes[0, 1].legend()

        # ROI size distribution
        if stat is not None:
            sizes = np.array([s["npix"] for s in stat[cell_mask]])
            axes[1, 0].hist(sizes, bins=25, color="seagreen", edgecolor="white")
            axes[1, 0].set_xlabel("Pixels")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Cell ROI Size")

        # Compactness distribution
        if stat is not None:
            compacts = np.array([s.get("compact", 0) for s in stat[cell_mask]])
            axes[1, 1].hist(compacts, bins=25, color="mediumpurple", edgecolor="white")
            axes[1, 1].set_xlabel("Compactness")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("ROI Compactness (higher = rounder)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Per-cell stats table
        with st.expander("Per-cell statistics"):
            import pandas as pd
            cell_idx = np.where(cell_mask)[0]
            stats_df = pd.DataFrame({
                "ROI": cell_idx,
                "SNR": np.round(snrs, 2),
                "Skewness": np.round(skews, 2),
                "Size (px)": [stat[i]["npix"] for i in cell_idx] if stat is not None else 0,
                "Compact": [round(stat[i].get("compact", 0), 3) for i in cell_idx] if stat is not None else 0,
                "Prob": np.round(iscell[cell_idx, 1], 3) if iscell is not None else 0,
            })
            st.dataframe(stats_df, width="stretch")
    else:
        st.info("No cell traces available for stats.")


# ── Registration ─────────────────────────────────────────────────────────
with tab_reg:
    import matplotlib.pyplot as plt

    xoff = ops_dict.get("xoff")
    yoff = ops_dict.get("yoff")

    if xoff is not None and yoff is not None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        axes[0].plot(xoff, linewidth=0.5)
        axes[0].set_ylabel("X shift (px)")
        axes[0].set_title("Registration shifts")
        axes[1].plot(yoff, linewidth=0.5)
        axes[1].set_ylabel("Y shift (px)")
        axes[1].set_xlabel("Frame")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    ref_img = ops_dict.get("refImg")
    if ref_img is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(ref_img, cmap="gray")
        ax.set_title("Reference Image")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()


# ── TIFF Images ──────────────────────────────────────────────────────────
with tab_tiff:
    import matplotlib.pyplot as plt

    st.subheader("TIFF Summary Images")
    st.caption(
        "Suite2p stores mean, max-projection, enhanced-mean, and registered-binary images in ops.npy. "
        "These are shown below without downloading the raw TIFFs."
    )

    # Images available in ops.npy
    img_keys = {
        "meanImg": "Mean Image (registered)",
        "meanImg_chan2": "Mean Image Channel 2",
        "max_proj": "Max Projection",
        "meanImgE": "Enhanced Mean Image",
        "refImg": "Reference Image (registration target)",
    }

    available = {k: v for k, v in img_keys.items() if ops_dict.get(k) is not None}

    if not available:
        st.info("No summary images found in ops.npy")
    else:
        # Show all available images in a grid
        cols = st.columns(min(len(available), 3))
        for i, (key, title) in enumerate(available.items()):
            img = ops_dict[key]
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(5, 5))
                vmin, vmax = np.percentile(img, [1, 99])
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # Raw TIFF info from S3
    st.markdown("---")
    st.subheader("Raw TIFF on S3")
    raw_prefix = f"rawdata/{sub}/{ses}/funcimg/"
    raw_files = list_s3_session_files(RAWDATA_BUCKET, raw_prefix)
    tiff_files = [f for f in raw_files if f["key"].endswith((".tif", ".tiff"))]

    if tiff_files:
        for tf in tiff_files:
            fname = tf["key"].split("/")[-1]
            st.text(f"  {fname} — {tf['size_mb']:.1f} MB")

        st.caption(
            "Raw TIFFs are too large to stream in-browser. "
            "The summary images above are extracted from the registered data by Suite2p."
        )
    else:
        st.info(f"No TIFF files found at `s3://{RAWDATA_BUCKET}/{raw_prefix}`")

    # Registered binary (if available) — show a few frames
    st.markdown("---")
    st.subheader("Registered Frames")
    st.caption("Loading sample frames from the Suite2p registered binary (data.bin).")

    bin_key = s3_prefix + "data.bin"
    bin_files = [f for f in files if f["key"].endswith("data.bin")]

    if bin_files:
        bin_size_mb = bin_files[0]["size_mb"]
        ly = ops_dict.get("Ly", 0)
        lx = ops_dict.get("Lx", 0)
        nframes = ops_dict.get("nframes", 0)

        if ly > 0 and lx > 0 and nframes > 0:
            st.text(f"data.bin: {bin_size_mb:.1f} MB | {nframes} frames @ {ly}x{lx}")

            frame_idx = st.slider(
                "Frame to view",
                0, nframes - 1, nframes // 2,
                help="Select a frame from the registered binary",
            )

            # Download just the bytes for one frame using S3 range request
            bytes_per_frame = ly * lx * 2  # int16
            start = frame_idx * bytes_per_frame
            end = start + bytes_per_frame - 1

            with st.spinner(f"Loading frame {frame_idx}..."):
                try:
                    s3 = boto3.client("s3", region_name=REGION)
                    obj = s3.get_object(
                        Bucket=DERIVATIVES_BUCKET,
                        Key=bin_key,
                        Range=f"bytes={start}-{end}",
                    )
                    frame_bytes = obj["Body"].read()
                    frame = np.frombuffer(frame_bytes, dtype=np.int16).reshape(ly, lx)
                    log.info("Loaded frame %d from data.bin (%d bytes)", frame_idx, len(frame_bytes))

                    fig, ax = plt.subplots(figsize=(8, 8))
                    vmin, vmax = np.percentile(frame, [1, 99])
                    ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
                    ax.set_title(f"Registered Frame {frame_idx}/{nframes}")
                    ax.axis("off")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    log.exception("Error loading frame from data.bin")
                    st.error("Error loading frame. Check server logs for details.")
        else:
            st.warning("Could not determine frame dimensions from ops.npy")
    else:
        st.info("No data.bin found — registered binary not available on S3.")
