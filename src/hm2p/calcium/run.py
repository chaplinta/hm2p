"""Stage 4 — end-to-end calcium processing pipeline.

Reads Suite2p output (plane0/ numpy files), applies neuropil subtraction,
computes dF/F0, optionally runs CASCADE spike inference, and writes ca.h5.

Input:  ca_extraction/suite2p/   (from Stage 1 Suite2p run)
        timestamps.h5             (from Stage 0 DAQ parsing)
Output: calcium/ca.h5            (imaging rate, n_rois × n_frames)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_suite2p(suite2p_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw fluorescence arrays from a Suite2p plane0 directory.

    Args:
        suite2p_dir: Path to the suite2p/ output directory (contains plane0/).

    Returns:
        Tuple of (F, Fneu, cell_mask) where:
            F         (n_rois_all, n_frames) float32 — raw fluorescence
            Fneu      (n_rois_all, n_frames) float32 — neuropil traces
            cell_mask (n_rois_all,) bool — True for classified cells

    Raises:
        FileNotFoundError: If plane0/ or required .npy files are absent.
    """
    plane_dir = suite2p_dir / "plane0"
    if not plane_dir.exists():
        raise FileNotFoundError(f"Suite2p plane0 directory not found: {plane_dir}")

    for name in ("F.npy", "Fneu.npy", "iscell.npy"):
        if not (plane_dir / name).exists():
            raise FileNotFoundError(f"Required Suite2p file missing: {plane_dir / name}")

    F = np.load(plane_dir / "F.npy").astype(np.float32)
    Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)
    iscell = np.load(plane_dir / "iscell.npy")
    cell_mask = iscell[:, 0].astype(bool)

    return F, Fneu, cell_mask


def run(
    suite2p_dir: Path,
    timestamps_h5: Path,
    session_id: str,
    output_path: Path,
    neuropil_coefficient: float = 0.7,
    dff_baseline_window_s: float = 60.0,
    dff_gaussian_sigma_s: float = 10.0,
    run_cascade: bool = False,
    cascade_model: str = "Global_EXC_7.5Hz_smoothing200ms",
) -> None:
    """Stage 4 pipeline: Suite2p output → neuropil subtraction → dF/F0 → ca.h5.

    Reads F, Fneu, and iscell from plane0/, filters to classified cells, applies
    neuropil subtraction, computes sliding-window baseline and dF/F0, optionally
    runs CASCADE spike inference, and writes ca.h5 at imaging rate.

    Args:
        suite2p_dir: Path to suite2p/ extraction output directory (contains plane0/).
        timestamps_h5: Stage 0 timestamps file (provides frame_times_imaging).
        session_id: Canonical session identifier stored as HDF5 attribute.
        output_path: Destination ca.h5 file path (created or overwritten).
        neuropil_coefficient: Fixed neuropil subtraction coefficient (default 0.7).
        dff_baseline_window_s: Sliding window length for baseline F0 (seconds).
        dff_gaussian_sigma_s: Gaussian smoothing sigma for baseline (seconds).
        run_cascade: If True, run CASCADE spike inference and write 'spikes' array.
        cascade_model: CASCADE pre-trained model name. Ignored if run_cascade=False.
    """
    from hm2p.calcium.dff import compute_baseline, compute_dff
    from hm2p.calcium.neuropil import subtract_fixed_coefficient
    from hm2p.io.hdf5 import read_h5, write_h5

    # --- Load Suite2p arrays ---
    F_all, Fneu_all, cell_mask = load_suite2p(suite2p_dir)
    F = F_all[cell_mask]
    Fneu = Fneu_all[cell_mask]

    # --- Classify ROI types (soma / dend / artefact) ---
    from hm2p.extraction.suite2p import classify_roi_types

    plane_dir = suite2p_dir / "plane0"
    stat_path = plane_dir / "stat.npy"
    if stat_path.exists():
        stat = list(np.load(stat_path, allow_pickle=True))
        all_types = classify_roi_types(stat)
        roi_types = [all_types[i] for i in np.flatnonzero(cell_mask)]
    else:
        roi_types = ["soma"] * int(cell_mask.sum())

    # --- Load imaging frame times ---
    ts = read_h5(timestamps_h5)
    frame_times = ts["frame_times_imaging"].astype(np.float64)

    # Infer fps from frame times; fall back to median diff if only 1 frame
    if len(frame_times) > 1:
        fps = float(1.0 / np.median(np.diff(frame_times)))
    else:
        fps = 30.0  # fallback — should never be needed on real data

    # --- Neuropil subtraction ---
    F_corr = subtract_fixed_coefficient(F, Fneu, coefficient=neuropil_coefficient)
    F_corr = F_corr.astype(np.float32)

    # --- dF/F0 ---
    F0 = compute_baseline(
        F_corr,
        fps=fps,
        window_s=dff_baseline_window_s,
        gaussian_sigma_s=dff_gaussian_sigma_s,
    )
    dff = compute_dff(F_corr, F0)

    # --- Event detection (Voigts & Harnett) ---
    from hm2p.calcium.events import detect_events_batch

    batch_result = detect_events_batch(dff, fps=fps)

    # Encode roi_types as uint8 array: 0=soma, 1=dend, 2=artefact
    type_map = {"soma": 0, "dend": 1, "artefact": 2}
    roi_type_arr = np.array([type_map.get(t, 0) for t in roi_types], dtype=np.uint8)

    datasets: dict[str, np.ndarray] = {
        "frame_times": frame_times,
        "dff": dff,
        "event_masks": batch_result.event_masks,
        "noise_probs": batch_result.noise_probs,
        "roi_types": roi_type_arr,
    }

    # --- Optional CASCADE spike inference ---
    if run_cascade:
        from hm2p.calcium.spikes import predict_spike_rates

        spikes = predict_spike_rates(dff, model_name=cascade_model, fps=fps)
        datasets["spikes"] = spikes

    attrs: dict[str, object] = {
        "session_id": session_id,
        "fps_imaging": fps,
        "extractor": "suite2p",
        "neuropil_coefficient": neuropil_coefficient,
    }
    write_h5(output_path, datasets, attrs=attrs)
