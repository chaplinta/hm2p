"""Suite2p viewer — ROI maps, traces, classification, and TIFF images."""

from __future__ import annotations

import io
import logging

import numpy as np
import streamlit as st

from frontend.data import (
    DERIVATIVES_BUCKET,
    RAWDATA_BUCKET,
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
    ops = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "ops.npy")
    stat = download_s3_numpy(DERIVATIVES_BUCKET, s3_prefix + "stat.npy")
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

# Tabs
tab_map, tab_traces, tab_class, tab_reg, tab_tiff = st.tabs(
    ["ROI Map", "Traces", "Classification", "Registration", "TIFF Images"]
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
            fig.suptitle("dF/F Traces (cells only)", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


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
                    s3 = __import__("boto3").client("s3", region_name="ap-southeast-2")
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
                    st.error(f"Error loading frame: {e}")
        else:
            st.warning("Could not determine frame dimensions from ops.npy")
    else:
        st.info("No data.bin found — registered binary not available on S3.")
