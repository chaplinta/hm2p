#!/usr/bin/env python3
"""Visualise Suite2p extraction results for a single session.

Usage:
    python scripts/viz_suite2p.py data/derivatives/ca_extraction/sub-1117788/ses-20221018T105617/suite2p
    python scripts/viz_suite2p.py <suite2p_dir> --save output.png
    python scripts/viz_suite2p.py <suite2p_dir> --top 10   # show top 10 cells by SNR

Produces a multi-panel figure:
1. Mean image + max projection with ROI contours
2. Fluorescence traces for top cells (by probability / SNR)
3. ROI classification summary (cell vs non-cell)
4. dF/F heatmap across all cells
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def load_suite2p(suite2p_dir: Path) -> dict:
    """Load all Suite2p outputs from a plane0 directory."""
    plane0 = suite2p_dir / "plane0"
    if not plane0.exists():
        raise FileNotFoundError(f"plane0 not found in {suite2p_dir}")

    data = {}
    data["F"] = np.load(plane0 / "F.npy")
    data["Fneu"] = np.load(plane0 / "Fneu.npy")
    data["iscell"] = np.load(plane0 / "iscell.npy")
    data["stat"] = np.load(plane0 / "stat.npy", allow_pickle=True)
    data["ops"] = np.load(plane0 / "ops.npy", allow_pickle=True).item()
    if (plane0 / "spks.npy").exists():
        data["spks"] = np.load(plane0 / "spks.npy")
    return data


def compute_dff(F: np.ndarray, Fneu: np.ndarray, coeff: float = 0.7) -> np.ndarray:
    """Compute dF/F0 with neuropil subtraction."""
    Fc = F - coeff * Fneu
    # Sliding window baseline (simple: rolling percentile 8th)
    from scipy.ndimage import uniform_filter1d

    win = min(3000, Fc.shape[1] // 2)
    if win < 10:
        F0 = np.median(Fc, axis=1, keepdims=True)
    else:
        # Approximate baseline as smoothed minimum
        smoothed = uniform_filter1d(Fc, size=win, axis=1)
        F0 = np.minimum.accumulate(smoothed, axis=1)
        # Use percentile for more robust baseline
        F0 = np.percentile(Fc, 8, axis=1, keepdims=True)
    F0 = np.maximum(F0, 1.0)  # avoid division by zero
    return (Fc - F0) / F0


def build_roi_image(
    stat: np.ndarray, iscell: np.ndarray, ops: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Build cell and non-cell ROI masks from stat array."""
    Ly, Lx = ops["Ly"], ops["Lx"]
    cell_img = np.zeros((Ly, Lx), dtype=float)
    noncell_img = np.zeros((Ly, Lx), dtype=float)

    for i, s in enumerate(stat):
        ypix = s["ypix"]
        xpix = s["xpix"]
        lam = s["lam"]
        lam_norm = lam / lam.max() if lam.max() > 0 else lam
        if iscell[i, 0] > 0.5:
            cell_img[ypix, xpix] = np.maximum(cell_img[ypix, xpix], lam_norm)
        else:
            noncell_img[ypix, xpix] = np.maximum(noncell_img[ypix, xpix], lam_norm)

    return cell_img, noncell_img


def plot_suite2p(suite2p_dir: Path, n_top: int = 8, save_path: Path | None = None) -> None:
    """Generate multi-panel Suite2p visualization."""
    data = load_suite2p(suite2p_dir)
    F, Fneu, iscell, stat, ops = (
        data["F"], data["Fneu"], data["iscell"], data["stat"], data["ops"],
    )

    cell_idx = np.where(iscell[:, 0] > 0.5)[0]
    n_rois = F.shape[0]
    n_cells = len(cell_idx)
    n_frames = F.shape[1]
    fps = ops.get("fs", 29.97)
    t = np.arange(n_frames) / fps

    # Compute dF/F
    dff = compute_dff(F, Fneu)
    dff_cells = dff[cell_idx]

    # Sort cells by SNR (std of dF/F)
    snr = np.std(dff_cells, axis=1)
    sort_idx = np.argsort(snr)[::-1]
    top_n = min(n_top, len(cell_idx))

    # Build ROI images
    cell_img, noncell_img = build_roi_image(stat, iscell, ops)

    # Get mean/max images from ops
    mean_img = ops.get("meanImg", np.zeros((ops["Ly"], ops["Lx"])))

    # --- Figure ---
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Suite2p Results — {suite2p_dir.parent.name}/{suite2p_dir.parent.parent.name}\n"
        f"{n_rois} ROIs, {n_cells} cells, {n_frames} frames @ {fps:.1f} Hz",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Mean image + ROI contours
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(mean_img, cmap="gray", aspect="equal")
    ax1.imshow(cell_img, cmap="Greens", alpha=0.5, aspect="equal")
    ax1.imshow(noncell_img, cmap="Reds", alpha=0.3, aspect="equal")
    ax1.set_title(f"Mean image + ROIs\n(green=cell, red=non-cell)")
    ax1.axis("off")

    # Panel 2: Cell ROI map (cells only, colored by index)
    ax2 = fig.add_subplot(2, 3, 2)
    roi_color_img = np.zeros((ops["Ly"], ops["Lx"], 3))
    cmap = plt.cm.hsv
    for j, ci in enumerate(cell_idx):
        s = stat[ci]
        color = cmap(j / max(n_cells, 1))[:3]
        lam = s["lam"] / s["lam"].max() if s["lam"].max() > 0 else s["lam"]
        for c in range(3):
            roi_color_img[s["ypix"], s["xpix"], c] = np.maximum(
                roi_color_img[s["ypix"], s["xpix"], c], lam * color[c]
            )
    ax2.imshow(roi_color_img, aspect="equal")
    ax2.set_title(f"{n_cells} cells (colored by index)")
    ax2.axis("off")

    # Panel 3: Classification histogram
    ax3 = fig.add_subplot(2, 3, 3)
    probs = iscell[:, 1]
    ax3.hist(probs[iscell[:, 0] > 0.5], bins=20, alpha=0.7, color="green", label="Cell")
    ax3.hist(probs[iscell[:, 0] < 0.5], bins=20, alpha=0.7, color="red", label="Non-cell")
    ax3.set_xlabel("Classification probability")
    ax3.set_ylabel("Count")
    ax3.set_title("ROI classification")
    ax3.legend()

    # Panel 4: Top cell traces (dF/F)
    ax4 = fig.add_subplot(2, 1, 2)
    offset = 0
    offsets = []
    for j in range(top_n):
        idx = sort_idx[j]
        trace = dff_cells[idx]
        ax4.plot(t, trace + offset, linewidth=0.5, color=cmap(j / max(top_n, 1)))
        ax4.text(
            t[-1] + 1, offset, f"cell {cell_idx[idx]}", fontsize=7, va="center",
        )
        offsets.append(offset)
        offset += max(np.ptp(trace) * 1.2, 0.5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("dF/F (stacked)")
    ax4.set_title(f"Top {top_n} cells by SNR (neuropil-subtracted dF/F)")
    ax4.set_xlim(0, t[-1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        fig.savefig("/tmp/suite2p_viz.png", dpi=150, bbox_inches="tight")
        print("Saved to /tmp/suite2p_viz.png")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise Suite2p results")
    parser.add_argument("suite2p_dir", type=Path, help="Path to suite2p output directory")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to path")
    parser.add_argument("--top", type=int, default=8, help="Number of top cells to show")
    args = parser.parse_args()

    plot_suite2p(args.suite2p_dir, n_top=args.top, save_path=args.save)


if __name__ == "__main__":
    main()
