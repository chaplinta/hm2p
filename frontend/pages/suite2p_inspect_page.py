"""Suite2p ROI inspection and reclassification.

View ROI footprints overlaid on mean/max images, fluorescence traces,
and iscell classification. Allows toggling ROI classification and
uploading updated iscell.npy back to S3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

DATA_DIR = Path("data/suite2p")


def _get_sessions() -> list[str]:
    """List downloaded sessions."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and (d / "suite2p" / "plane0" / "stat.npy").exists()
    )


def _load_session(session_id: str) -> dict:
    """Load Suite2p outputs for a session."""
    plane_dir = DATA_DIR / session_id / "suite2p" / "plane0"
    data = {}

    ops_path = plane_dir / "ops.npy"
    if ops_path.exists():
        data["ops"] = np.load(ops_path, allow_pickle=True).item()

    stat_path = plane_dir / "stat.npy"
    if stat_path.exists():
        data["stat"] = np.load(stat_path, allow_pickle=True)

    iscell_path = plane_dir / "iscell.npy"
    if iscell_path.exists():
        data["iscell"] = np.load(iscell_path)

    f_path = plane_dir / "F.npy"
    if f_path.exists():
        data["F"] = np.load(f_path)

    fneu_path = plane_dir / "Fneu.npy"
    if fneu_path.exists():
        data["Fneu"] = np.load(fneu_path)

    return data


def _build_roi_image(
    ops: dict, stat: np.ndarray, iscell: np.ndarray, show_non_cells: bool = False
) -> np.ndarray:
    """Build RGB image with ROI footprints overlaid on mean image."""
    Ly, Lx = ops["Ly"], ops["Lx"]

    # Normalise mean image to 0-1
    mean_img = ops.get("meanImg", np.zeros((Ly, Lx)))
    mn, mx = np.percentile(mean_img, [1, 99])
    if mx > mn:
        img_norm = np.clip((mean_img - mn) / (mx - mn), 0, 1)
    else:
        img_norm = np.zeros_like(mean_img)

    # RGB: grey background
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

    for i, s in enumerate(stat):
        is_cell = iscell[i, 0] > 0.5
        if not is_cell and not show_non_cells:
            continue

        ypix = s["ypix"]
        xpix = s["xpix"]
        lam = s["lam"]
        lam_norm = lam / lam.max() if lam.max() > 0 else lam

        # Clip to image bounds
        valid = (ypix < Ly) & (xpix < Lx) & (ypix >= 0) & (xpix >= 0)
        ypix, xpix, lam_norm = ypix[valid], xpix[valid], lam_norm[valid]

        if is_cell:
            # Green for cells
            rgb[ypix, xpix, 1] = np.maximum(
                rgb[ypix, xpix, 1], lam_norm * 0.8
            )
            rgb[ypix, xpix, 0] *= 1 - lam_norm * 0.5
            rgb[ypix, xpix, 2] *= 1 - lam_norm * 0.5
        else:
            # Red for non-cells
            rgb[ypix, xpix, 0] = np.maximum(
                rgb[ypix, xpix, 0], lam_norm * 0.6
            )
            rgb[ypix, xpix, 1] *= 1 - lam_norm * 0.4
            rgb[ypix, xpix, 2] *= 1 - lam_norm * 0.4

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _build_single_roi_image(
    ops: dict, stat_entry: dict, padding: int = 20
) -> tuple[np.ndarray, int, int, int, int]:
    """Build a cropped image around a single ROI."""
    Ly, Lx = ops["Ly"], ops["Lx"]
    ypix = stat_entry["ypix"]
    xpix = stat_entry["xpix"]

    y0 = max(0, ypix.min() - padding)
    y1 = min(Ly, ypix.max() + padding)
    x0 = max(0, xpix.min() - padding)
    x1 = min(Lx, xpix.max() + padding)

    mean_img = ops.get("meanImg", np.zeros((Ly, Lx)))
    crop = mean_img[y0:y1, x0:x1]

    mn, mx = np.percentile(mean_img, [1, 99])
    if mx > mn:
        crop_norm = np.clip((crop - mn) / (mx - mn), 0, 1)
    else:
        crop_norm = np.zeros_like(crop, dtype=float)

    rgb = np.stack([crop_norm, crop_norm, crop_norm], axis=-1)

    # Overlay ROI
    lam = stat_entry["lam"]
    lam_norm = lam / lam.max() if lam.max() > 0 else lam
    yy = ypix - y0
    xx = xpix - x0
    valid = (yy >= 0) & (yy < rgb.shape[0]) & (xx >= 0) & (xx < rgb.shape[1])
    rgb[yy[valid], xx[valid], 1] = np.maximum(
        rgb[yy[valid], xx[valid], 1], lam_norm[valid] * 0.9
    )

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), y0, y1, x0, x1


def page():
    st.header("Suite2p ROI Inspection")

    sessions = _get_sessions()
    if not sessions:
        st.warning(
            "No Suite2p data downloaded. Run:\n\n"
            "```\npython scripts/download_suite2p.py --lightweight\n```"
        )
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        session_id = st.selectbox("Session", sessions)
    with col2:
        show_non_cells = st.checkbox("Show non-cells (red)", value=False)

    data = _load_session(session_id)
    if not data.get("stat") is not None:
        st.error("Could not load session data.")
        return

    ops = data["ops"]
    stat = data["stat"]
    iscell = data["iscell"]
    n_total = len(stat)
    n_cells = int(iscell[:, 0].sum())

    st.caption(
        f"{n_cells} cells / {n_total} ROIs | "
        f"Image: {ops['Ly']}×{ops['Lx']} | "
        f"Frames: {ops.get('nframes', '?')}"
    )

    # ── Summary images ──────────────────────────────────────────────────
    tab_overview, tab_images, tab_roi, tab_reclassify = st.tabs(
        ["ROI Overview", "Summary Images", "Single ROI", "Reclassify"]
    )

    with tab_overview:
        roi_img = _build_roi_image(ops, stat, iscell, show_non_cells)
        st.image(roi_img, caption="ROI footprints (green=cell, red=non-cell)", use_container_width=True)

    with tab_images:
        img_cols = st.columns(3)
        for col, key, label in zip(
            img_cols,
            ["meanImg", "meanImgE", "max_proj"],
            ["Mean Image", "Enhanced Mean", "Max Projection"],
        ):
            with col:
                if key in ops:
                    img = ops[key]
                    mn, mx = np.percentile(img, [1, 99])
                    if mx > mn:
                        img_disp = np.clip((img - mn) / (mx - mn), 0, 1)
                    else:
                        img_disp = np.zeros_like(img)
                    st.image(img_disp, caption=label, use_container_width=True)
                else:
                    st.caption(f"{label}: not available")

    with tab_roi:
        # ROI selector
        cell_indices = np.where(iscell[:, 0] > 0.5)[0]
        all_indices = list(range(n_total))
        show_only_cells = st.checkbox("Show only classified cells", value=True)
        idx_list = cell_indices.tolist() if show_only_cells else all_indices

        if not idx_list:
            st.info("No ROIs to show.")
        else:
            roi_idx = st.selectbox(
                "ROI index",
                idx_list,
                format_func=lambda i: (
                    f"ROI {i} ({'cell' if iscell[i, 0] > 0.5 else 'non-cell'}, "
                    f"p={iscell[i, 1]:.3f})"
                ),
            )

            col_img, col_trace = st.columns([1, 2])

            with col_img:
                roi_crop, *_ = _build_single_roi_image(ops, stat[roi_idx])
                st.image(roi_crop, caption=f"ROI {roi_idx}", width=250)

                # ROI stats
                s = stat[roi_idx]
                st.caption(
                    f"Pixels: {len(s['ypix'])} | "
                    f"Compact: {s.get('compact', '?'):.2f} | "
                    f"Skew: {s.get('skew', '?'):.2f} | "
                    f"p(cell): {iscell[roi_idx, 1]:.3f}"
                )

            with col_trace:
                if "F" in data:
                    F = data["F"]
                    Fneu = data.get("Fneu")

                    trace = F[roi_idx]
                    # Downsample for display if very long
                    if len(trace) > 5000:
                        bin_size = len(trace) // 5000
                        trace_ds = trace[: bin_size * 5000].reshape(-1, bin_size).mean(axis=1)
                        if Fneu is not None:
                            fneu_ds = Fneu[roi_idx][: bin_size * 5000].reshape(-1, bin_size).mean(axis=1)
                        t = np.arange(len(trace_ds)) * bin_size / ops.get("fs", 9.6)
                    else:
                        trace_ds = trace
                        fneu_ds = Fneu[roi_idx] if Fneu is not None else None
                        t = np.arange(len(trace_ds)) / ops.get("fs", 9.6)

                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 2.5))
                    ax.plot(t, trace_ds, "g", linewidth=0.5, alpha=0.8, label="F")
                    if fneu_ds is not None:
                        ax.plot(t, fneu_ds, "r", linewidth=0.3, alpha=0.5, label="Fneu")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Fluorescence")
                    ax.set_title(f"ROI {roi_idx}")
                    ax.legend(fontsize=7)
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # dF/F trace
                    if Fneu is not None:
                        Fc = F[roi_idx] - 0.7 * Fneu[roi_idx]
                        baseline = np.percentile(Fc, 10)
                        if baseline > 0:
                            dff = (Fc - baseline) / baseline
                            if len(dff) > 5000:
                                dff_ds = dff[: bin_size * 5000].reshape(-1, bin_size).mean(axis=1)
                            else:
                                dff_ds = dff

                            fig2, ax2 = plt.subplots(figsize=(10, 2))
                            ax2.plot(t, dff_ds, "k", linewidth=0.5)
                            ax2.set_xlabel("Time (s)")
                            ax2.set_ylabel("dF/F")
                            ax2.set_title(f"ROI {roi_idx} — dF/F (neuropil corrected)")
                            ax2.spines[["top", "right"]].set_visible(False)
                            fig2.tight_layout()
                            st.pyplot(fig2)
                            plt.close(fig2)
                else:
                    st.info("No trace data (F.npy). Download with full mode.")

    with tab_reclassify:
        st.subheader("Reclassify ROIs")
        st.caption(
            "Toggle ROI classification below. Changes are saved locally to "
            "data/suite2p/<session>/suite2p/plane0/iscell.npy. "
            "Use the upload button to push to S3."
        )

        # Initialize session state for iscell edits
        state_key = f"iscell_{session_id}"
        if state_key not in st.session_state:
            st.session_state[state_key] = iscell.copy()

        edited_iscell = st.session_state[state_key]
        n_edited_cells = int(edited_iscell[:, 0].sum())
        n_changed = int((edited_iscell[:, 0] != iscell[:, 0]).sum())

        st.caption(
            f"Current: {n_edited_cells} cells / {n_total} ROIs | "
            f"Changes: {n_changed}"
        )

        # Quick filter by probability
        prob_threshold = st.slider(
            "Classification threshold (p(cell))",
            0.0, 1.0, 0.5, 0.05,
            help="ROIs with p(cell) above this are classified as cells",
        )
        if st.button("Apply threshold"):
            edited_iscell[:, 0] = (edited_iscell[:, 1] >= prob_threshold).astype(float)
            st.session_state[state_key] = edited_iscell
            st.rerun()

        # Manual toggle for specific ROIs
        toggle_idx = st.number_input(
            "Toggle ROI index", min_value=0, max_value=n_total - 1, value=0
        )
        col_toggle, col_status = st.columns(2)
        with col_toggle:
            if st.button("Toggle cell/non-cell"):
                edited_iscell[toggle_idx, 0] = 1.0 - edited_iscell[toggle_idx, 0]
                st.session_state[state_key] = edited_iscell
                st.rerun()
        with col_status:
            status = "cell" if edited_iscell[toggle_idx, 0] > 0.5 else "non-cell"
            st.caption(f"ROI {toggle_idx}: {status} (p={edited_iscell[toggle_idx, 1]:.3f})")

        st.divider()

        # Save locally
        if st.button("Save iscell.npy locally"):
            save_path = DATA_DIR / session_id / "suite2p" / "plane0" / "iscell.npy"
            np.save(save_path, edited_iscell)
            st.success(f"Saved to {save_path}")

        # Upload to S3
        if st.button("Upload iscell.npy to S3"):
            import tempfile
            import boto3

            save_path = DATA_DIR / session_id / "suite2p" / "plane0" / "iscell.npy"
            np.save(save_path, edited_iscell)

            parts = session_id.split("_")
            animal = parts[-1]
            sub = f"sub-{animal}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            s3_key = f"ca_extraction/{sub}/{ses}/suite2p/plane0/iscell.npy"

            s3 = boto3.client("s3", region_name="ap-southeast-2")
            s3.upload_file(str(save_path), "hm2p-derivatives", s3_key)
            st.success(f"Uploaded to s3://hm2p-derivatives/{s3_key}")


page()
