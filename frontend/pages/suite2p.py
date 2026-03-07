"""Suite2p viewer — ROI maps, traces, classification for processed sessions."""

from __future__ import annotations

import io

import numpy as np
import streamlit as st

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    download_s3_numpy,
    list_s3_session_files,
    load_experiments,
    parse_session_id,
)


def render():
    st.title("Suite2p Results")

    experiments = load_experiments()
    exp_ids = [e["exp_id"] for e in experiments]

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
        return

    file_names = [f["key"].split("/")[-1] for f in files]
    st.caption(
        f"Found {len(files)} files: {', '.join(file_names[:15])}"
        + ("..." if len(files) > 15 else "")
    )

    # Load key arrays
    with st.spinner("Loading Suite2p data from S3..."):
        ops = _load_npy(s3_prefix + "ops.npy")
        stat = _load_npy(s3_prefix + "stat.npy")
        iscell = _load_npy(s3_prefix + "iscell.npy")
        f_traces = _load_npy(s3_prefix + "F.npy")
        f_neu = _load_npy(s3_prefix + "Fneu.npy")
        spks = _load_npy(s3_prefix + "spks.npy")

    if ops is None:
        st.error("Could not load ops.npy")
        return

    # Summary stats
    n_rois = len(iscell) if iscell is not None else 0
    n_cells = int(iscell[:, 0].sum()) if iscell is not None else 0
    n_frames = ops.item().get("nframes", 0) if isinstance(ops, np.ndarray) else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROIs", n_rois)
    col2.metric("Cells", n_cells)
    col3.metric("Non-cells", n_rois - n_cells)
    col4.metric("Frames", n_frames)

    # Tabs
    tab_map, tab_traces, tab_class, tab_reg = st.tabs(
        ["ROI Map", "Traces", "Classification", "Registration"]
    )

    with tab_map:
        _render_roi_map(ops, stat, iscell)

    with tab_traces:
        _render_traces(f_traces, f_neu, iscell, spks)

    with tab_class:
        _render_classification(iscell)

    with tab_reg:
        _render_registration(ops)


def _load_npy(key: str):
    """Load a .npy file from S3."""
    return download_s3_numpy(DERIVATIVES_BUCKET, key)


def _render_roi_map(ops, stat, iscell):
    """Render mean image with ROI contours."""
    import matplotlib.pyplot as plt

    if ops is None or stat is None:
        st.warning("Missing ops.npy or stat.npy")
        return

    ops_dict = ops.item() if isinstance(ops, np.ndarray) and ops.ndim == 0 else ops

    # Mean image
    mean_img = ops_dict.get("meanImg")
    if mean_img is None:
        st.warning("No meanImg in ops")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: mean image
    axes[0].imshow(mean_img, cmap="gray", vmin=np.percentile(mean_img, 1),
                   vmax=np.percentile(mean_img, 99))
    axes[0].set_title("Mean Image")
    axes[0].axis("off")

    # Right: mean image + ROI contours
    axes[1].imshow(mean_img, cmap="gray", vmin=np.percentile(mean_img, 1),
                   vmax=np.percentile(mean_img, 99))

    ly, lx = mean_img.shape
    roi_img = np.zeros((ly, lx, 4), dtype=np.float32)

    for i, s in enumerate(stat):
        if iscell is not None and i < len(iscell):
            is_cell = bool(iscell[i, 0])
        else:
            is_cell = True

        ypix = s["ypix"]
        xpix = s["xpix"]
        mask = (ypix >= 0) & (ypix < ly) & (xpix >= 0) & (xpix < lx)
        ypix, xpix = ypix[mask], xpix[mask]

        if is_cell:
            roi_img[ypix, xpix] = [0, 1, 0, 0.3]  # green
        else:
            roi_img[ypix, xpix] = [1, 0, 0, 0.15]  # red, faint

    axes[1].imshow(roi_img)
    axes[1].set_title(f"ROIs ({int(iscell[:, 0].sum())} cells / {len(stat)} total)")
    axes[1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_traces(f_traces, f_neu, iscell, spks):
    """Render dF/F traces for selected cells."""
    import matplotlib.pyplot as plt

    if f_traces is None:
        st.warning("No F.npy found")
        return

    cell_indices = np.where(iscell[:, 0] == 1)[0] if iscell is not None else np.arange(len(f_traces))
    n_cells = len(cell_indices)

    if n_cells == 0:
        st.info("No cells classified.")
        return

    # Cell selector
    col1, col2 = st.columns([1, 3])
    with col1:
        n_show = st.slider("Cells to show", 1, min(20, n_cells), min(5, n_cells))
        offset = st.slider("Start from cell #", 0, max(0, n_cells - n_show), 0)
    with col2:
        show_neuropil = st.checkbox("Show neuropil", value=False)
        show_deconv = st.checkbox("Show deconvolved", value=False)

    selected_cells = cell_indices[offset : offset + n_show]

    fig, axes = plt.subplots(len(selected_cells), 1, figsize=(14, 2 * len(selected_cells)),
                              sharex=True)
    if len(selected_cells) == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected_cells):
        trace = f_traces[idx]
        # Simple dF/F: (F - F0) / F0 with rolling baseline
        f0 = np.percentile(trace, 10)
        if f0 > 0:
            dff = (trace - f0) / f0
        else:
            dff = trace

        ax.plot(dff, linewidth=0.5, color="black", label=f"Cell {idx}")

        if show_neuropil and f_neu is not None:
            neu = f_neu[idx]
            f0n = np.percentile(neu, 10)
            if f0n > 0:
                dff_neu = (neu - f0n) / f0n
            else:
                dff_neu = neu
            ax.plot(dff_neu, linewidth=0.3, color="blue", alpha=0.5, label="Neuropil")

        if show_deconv and spks is not None:
            ax.plot(spks[idx] / max(spks[idx].max(), 1), linewidth=0.3,
                    color="red", alpha=0.5, label="Deconv")

        prob = iscell[idx, 1] if iscell is not None else 0
        ax.set_ylabel(f"Cell {idx}\np={prob:.2f}", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("dF/F Traces (cells only)", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_classification(iscell):
    """Render classification probability histogram."""
    import matplotlib.pyplot as plt

    if iscell is None:
        st.warning("No iscell.npy found")
        return

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


def _render_registration(ops):
    """Render registration quality metrics."""
    import matplotlib.pyplot as plt

    if ops is None:
        return

    ops_dict = ops.item() if isinstance(ops, np.ndarray) and ops.ndim == 0 else ops

    # X/Y shifts
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

    # Reference image
    ref_img = ops_dict.get("refImg")
    if ref_img is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(ref_img, cmap="gray")
        ax.set_title("Reference Image")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()
