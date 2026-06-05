"""Stage 4 — end-to-end calcium processing pipeline.

Reads Suite2p output (plane0/ numpy files), applies neuropil subtraction,
computes dF/F0, optionally runs CASCADE spike inference, and writes ca.h5.

Input:  ca_extraction/suite2p/   (from Stage 1 Suite2p run)
        timestamps.h5             (from Stage 0 DAQ parsing)
Output: calcium/ca.h5            (imaging rate, n_rois × n_frames)

Neuropil subtraction is dispatched via the ``neuropil_method`` parameter:
    fissa      — Spatial ICA; requires TIFFs and ROI masks (default)
    estimated  — Per-ROI coefficient estimated from lower-envelope regression
    fixed      — Fixed coefficient (default 0.7; backward-compatible)

Both F0_rolling and F0_percentile baselines are always stored in ca.h5.
The primary ``dff`` array is computed from the method chosen by ``f0_method``
(default ``rolling``).

Raw arrays F_raw and Fneu_raw are also stored alongside the corrected traces
so that neuropil subtraction can be re-derived without re-running extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def load_suite2p(suite2p_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw fluorescence arrays from a Suite2p plane0 directory.

    Parameters
    ----------
    suite2p_dir : Path
        Path to the suite2p/ output directory (contains plane0/).

    Returns
    -------
    F : np.ndarray
        (n_rois_all, n_frames) float32 — raw fluorescence
    Fneu : np.ndarray
        (n_rois_all, n_frames) float32 — neuropil traces
    cell_mask : np.ndarray
        (n_rois_all,) bool — True for classified cells

    Raises
    ------
    FileNotFoundError
        If plane0/ or required .npy files are absent.
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
    neuropil_method: str = "fissa",
    neuropil_coefficient: float = 0.7,
    fissa_tiff_paths: list[Path] | None = None,
    fissa_roi_masks: list[np.ndarray] | None = None,
    fissa_output_dir: Path | None = None,
    dff_baseline_window_s: float = 60.0,
    dff_gaussian_sigma_s: float = 10.0,
    f0_method: str = "rolling",
    run_cascade: bool = False,
    cascade_model: str = "Global_EXC_10Hz_smoothing200ms",
) -> None:
    """Stage 4 pipeline: Suite2p output → neuropil subtraction → dF/F0 → ca.h5.

    Reads F, Fneu, and iscell from plane0/, filters to classified cells, applies
    neuropil subtraction (fissa / estimated / fixed), computes both rolling and
    percentile baselines, and writes ca.h5 at imaging rate.

    Raw F and Fneu are stored in ca.h5 as F_raw and Fneu_raw so that subtraction
    can be re-derived without re-running extraction. Both F0_rolling and
    F0_percentile are stored. The primary ``dff`` dataset is computed from
    the baseline method selected by ``f0_method``.

    Parameters
    ----------
    suite2p_dir : Path
        Path to suite2p/ extraction output directory (contains plane0/).
    timestamps_h5 : Path
        Stage 0 timestamps file (provides frame_times_imaging).
    session_id : str
        Canonical session identifier stored as HDF5 attribute.
    output_path : Path
        Destination ca.h5 file path (created or overwritten).
    neuropil_method : str
        Neuropil subtraction method: "fissa", "estimated", or "fixed".
        Default "fissa". If "fissa" is requested but fails (e.g. TIFFs missing),
        falls back to "estimated" and logs a warning.
    neuropil_coefficient : float
        Fixed neuropil subtraction coefficient (used only when
        neuropil_method="fixed"; default 0.7).
    fissa_tiff_paths : list of Path or None
        Ordered TIFF paths required for FISSA. Must be provided when
        neuropil_method="fissa". If absent, FISSA is skipped and a warning
        logged; the pipeline falls back to "estimated".
    fissa_roi_masks : list of np.ndarray or None
        Per-ROI binary masks (height, width) required for FISSA. Must be
        provided together with fissa_tiff_paths.
    fissa_output_dir : Path or None
        Directory for FISSA intermediate files. Defaults to
        output_path.parent / "fissa_cache".
    dff_baseline_window_s : float
        Sliding window length for rolling baseline F0 (seconds, default 60 s).
    dff_gaussian_sigma_s : float
        Gaussian smoothing sigma for rolling baseline (seconds, default 10 s).
    f0_method : str
        Which baseline to use as the primary dff: "rolling" or "percentile".
        Default "rolling". The other is always stored as a sensitivity check.
    run_cascade : bool
        If True, run CASCADE spike inference and write 'spikes' array.
    cascade_model : str
        CASCADE pre-trained model name. Ignored if run_cascade=False.
    """
    from hm2p.calcium.dff import (
        compute_baseline,
        compute_baseline_percentile,
        compute_dff_with_clip_counts,
    )
    from hm2p.calcium.neuropil import (
        subtract_estimated_coefficient,
        subtract_fissa,
        subtract_fixed_coefficient,
    )
    from hm2p.io.hdf5 import read_h5, write_h5

    # --- Load Suite2p arrays ---
    # Process ALL ROIs (not just iscell=True) so dendrites and other
    # non-soma ROIs are available for analysis. The roi_types array
    # marks each ROI's classification; filtering happens at display time.
    F_all, Fneu_all, cell_mask = load_suite2p(suite2p_dir)
    F = F_all  # all ROIs, not filtered by iscell
    Fneu = Fneu_all

    # --- Load ops and extract bad imaging frames ---
    # Suite2p stores ops['badframes'] as an integer array of frame indices
    # that exceeded the th_badframes threshold during motion correction.
    # These frames have unreliable motion estimates and should be excluded
    # from downstream analyses.
    plane_dir = suite2p_dir / "plane0"
    ops_path = plane_dir / "ops.npy"
    bad_imaging_frames: np.ndarray | None = None
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        n_total_frames = F.shape[1]
        badframes_idx = ops.get("badframes", None)
        if badframes_idx is not None and len(badframes_idx) > 0:
            bad_mask = np.zeros(n_total_frames, dtype=bool)
            # ``badframes`` contains frame indices into the original
            # registration stream; clip to valid range. Indices beyond
            # ``n_total_frames`` indicate Suite2p's badframes catalogue
            # is out of step with the F.npy length — log a warning so
            # the inconsistency is auditable rather than silently
            # discarded (QA issue 1.10).
            beyond_mask = badframes_idx >= n_total_frames
            n_beyond = int(beyond_mask.sum())
            if n_beyond > 0:
                log.warning(
                    "Suite2p ops['badframes'] contains %d index/indices "
                    "beyond F.npy length (%d). Suite2p frame catalogue "
                    "is inconsistent with F traces — out-of-range "
                    "indices ignored, but please investigate ops.npy.",
                    n_beyond,
                    n_total_frames,
                )
            valid_idx = badframes_idx[~beyond_mask]
            bad_mask[valid_idx] = True
            bad_imaging_frames = bad_mask
        else:
            bad_imaging_frames = np.zeros(n_total_frames, dtype=bool)

    # --- Load ROI classification ---
    # The XGBoost 3-way classifier (soma/dend/artefact) runs at the end of
    # Stage 1 and writes roi_class.npy + roi_class_prob.npy alongside the
    # Suite2p outputs.  Read those here; fall back to iscell-based labels
    # if the classifier hasn't been run (e.g. legacy data).
    roi_class_path = plane_dir / "roi_class.npy"
    roi_class_prob_path = plane_dir / "roi_class_prob.npy"
    p_soma_arr: np.ndarray | None = None
    p_dend_arr: np.ndarray | None = None
    p_artefact_arr: np.ndarray | None = None

    label_map_inv = {0: "artefact", 1: "soma", 2: "dend"}

    if roi_class_path.exists():
        roi_class = np.load(roi_class_path)
        roi_types = [label_map_inv.get(int(c), "artefact") for c in roi_class]

        if roi_class_prob_path.exists():
            probs = np.load(roi_class_prob_path)
            # Column order: [P(artefact), P(soma), P(dendrite)]
            p_artefact_arr = probs[:, 0].astype(np.float32)
            p_soma_arr = probs[:, 1].astype(np.float32)
            p_dend_arr = probs[:, 2].astype(np.float32)

        log.info(
            "Loaded ROI classification from roi_class.npy: %d soma, %d dend, %d artefact",
            sum(1 for t in roi_types if t == "soma"),
            sum(1 for t in roi_types if t == "dend"),
            sum(1 for t in roi_types if t == "artefact"),
        )
    else:
        # Fallback: no XGBoost classifier output.  Use iscell as a binary
        # soma/artefact label (no dendrite distinction possible).
        log.warning(
            "roi_class.npy not found in %s — falling back to iscell.npy "
            "binary labels (soma vs artefact only, no dendrite class).",
            plane_dir,
        )
        roi_types = ["soma" if m else "artefact" for m in cell_mask]

    # --- Load imaging frame times ---
    ts = read_h5(timestamps_h5)
    frame_times = ts["frame_times_imaging"].astype(np.float64)

    # Infer fps from frame times; fall back to median diff if only 1 frame
    if len(frame_times) > 1:
        fps = float(1.0 / np.median(np.diff(frame_times)))
    else:
        fps = 30.0  # fallback — should never be needed on real data

    # --- Neuropil subtraction ---
    # Store raw traces before any correction so subtraction can be re-derived.
    F_raw = F.copy()
    Fneu_raw = Fneu.copy()

    neuropil_coeff_used: float | None = None

    if neuropil_method == "fissa":
        if fissa_tiff_paths is None or fissa_roi_masks is None:
            log.warning(
                "neuropil_method=fissa but fissa_tiff_paths/fissa_roi_masks not provided. "
                "Falling back to estimated-coefficient subtraction."
            )
            F_corr, coefficients = subtract_estimated_coefficient(F, Fneu)
            neuropil_coeff_used = float(np.median(coefficients))
        else:
            fissa_dir = fissa_output_dir or (output_path.parent / "fissa_cache")
            F_corr = subtract_fissa(
                tiff_paths=fissa_tiff_paths,
                roi_masks=fissa_roi_masks,
                output_dir=fissa_dir,
                F_fallback=F,
                Fneu_fallback=Fneu,
            )
            neuropil_coeff_used = None  # FISSA does not produce a scalar coefficient

    elif neuropil_method == "estimated":
        F_corr, coefficients = subtract_estimated_coefficient(F, Fneu)
        neuropil_coeff_used = float(np.median(coefficients))

    else:  # "fixed" or unrecognised — fall back to fixed with a warning
        if neuropil_method not in ("fixed",):
            log.warning(
                "Unrecognised neuropil_method %r; falling back to 'fixed' (coefficient=%.2f)",
                neuropil_method,
                neuropil_coefficient,
            )
        F_corr = subtract_fixed_coefficient(F, Fneu, coefficient=neuropil_coefficient)
        neuropil_coeff_used = neuropil_coefficient

    F_corr = F_corr.astype(np.float32)

    # --- Load Suite2p deconvolved spikes (spks.npy) ---
    # Persist the full population (n_rois_all rows) so the ROI axis matches
    # every other ROI-axis array in ca.h5 (F, Fneu, F_corr, dff, …). Filtering
    # by iscell here would silently misalign the ROI axis: consumers indexing
    # ``deconv[i]`` and ``dff[i]`` together would read different ROIs.
    spks_path = plane_dir / "spks.npy"
    if spks_path.exists():
        spks_all = np.load(spks_path).astype(np.float32)
        deconv = spks_all
        # Normalize per ROI by max (matching legacy pipeline: deconv / max)
        deconv_max = deconv.max(axis=1, keepdims=True)
        deconv_max[deconv_max == 0] = 1.0  # avoid division by zero
        deconv_norm = deconv / deconv_max
    else:
        deconv = None
        deconv_norm = None

    # --- Baselines: always compute both rolling and percentile ---
    F0_rolling = compute_baseline(
        F_corr,
        fps=fps,
        window_s=dff_baseline_window_s,
        gaussian_sigma_s=dff_gaussian_sigma_s,
    )
    F0_percentile = compute_baseline_percentile(
        F_corr,
        fps=fps,
        window_s=dff_baseline_window_s,
        percentile=8.0,
    )

    # Primary dff uses the method selected in config
    if f0_method == "percentile":
        F0_primary = F0_percentile
    else:
        if f0_method not in ("rolling",):
            log.warning("Unrecognised f0_method %r; using 'rolling'.", f0_method)
        F0_primary = F0_rolling

    dff, dff_n_clipped = compute_dff_with_clip_counts(F_corr, F0_primary)
    dff_percentile, _ = compute_dff_with_clip_counts(F_corr, F0_percentile)

    # --- Event detection ---
    from hm2p.calcium.events import detect_events_batch, detect_events_sd

    # V&H method (percentile-based noise model) with significance filtering.
    # prob_onset=0.2 and alpha=0.05 match the legacy pipeline and V&H paper.
    batch_result = detect_events_batch(dff, fps=fps, prob_onset=0.2, alpha=0.05)
    # SD-threshold method (more sensitive to small transients)
    event_masks_sd = detect_events_sd(dff, fps=fps, sd_threshold=2.0, min_duration_s=0.3)

    # --- Per-ROI QC metrics ---
    from hm2p.calcium.qc import compute_roi_qc

    qc_datasets = compute_roi_qc(
        dff=dff,
        F_raw=F_raw,
        Fneu_raw=Fneu_raw,
        event_results=batch_result.events,
        event_masks=batch_result.event_masks,
        fps=fps,
        bad_frames=bad_imaging_frames,
    )

    # Encode roi_types as uint8 per the documented spec:
    # 0=soma, 1=dendrite, 2=artefact. ``iscell`` is persisted separately as
    # a boolean so the Suite2p-rejected vs artefact distinction is preserved.
    type_map = {"soma": 0, "dend": 1, "artefact": 2}
    roi_type_arr = np.array([type_map.get(t, 2) for t in roi_types], dtype=np.uint8)
    iscell_arr = cell_mask.astype(bool)

    datasets: dict[str, np.ndarray] = {
        # Raw traces (pre-subtraction) — for re-derivation without re-extraction
        "F_raw": F_raw,
        "Fneu_raw": Fneu_raw,
        # Neuropil-corrected fluorescence (primary signal)
        "F_corr": F_corr,
        # Baselines
        "F0_rolling": F0_rolling,
        "F0_percentile": F0_percentile,
        # dF/F0 (primary method) and percentile sensitivity check
        "dff": dff,
        "dff_percentile": dff_percentile,
        # Event detection
        "frame_times": frame_times,
        "event_masks": batch_result.event_masks,
        "event_masks_sd": event_masks_sd,
        "noise_probs": batch_result.noise_probs,
        "roi_types": roi_type_arr,
        "iscell": iscell_arr,
    }

    # Persist Suite2p's bad-frame mask (frames that failed motion-correction
    # quality threshold during registration).  Length == number of imaging
    # frames.  Downstream (Stage 5 sync) ORs this with bad_behav to produce
    # the combined exclusion mask.
    if bad_imaging_frames is not None:
        # Trim/pad to match dff frame count when off by ±1 (a known
        # Suite2p edge case — see sync diagnostics off-by-one heuristic).
        # Larger mismatches indicate a real Suite2p frame-count
        # inconsistency; log a warning so it is auditable, and refuse to
        # silently absorb mismatches > 1 frame.
        n_dff = dff.shape[1]
        n_bad = len(bad_imaging_frames)
        delta = n_bad - n_dff
        if delta == 0:
            datasets["bad_imaging_frames"] = bad_imaging_frames
        elif abs(delta) <= 1:
            # Off-by-one: trim or pad with False, log info-level note.
            log.info(
                "bad_imaging_frames length %d ≠ dff frames %d (delta=%+d) — "
                "applying off-by-one trim/pad.",
                n_bad,
                n_dff,
                delta,
            )
            if delta > 0:
                datasets["bad_imaging_frames"] = bad_imaging_frames[:n_dff]
            else:
                pad = np.zeros(-delta, dtype=bool)
                datasets["bad_imaging_frames"] = np.concatenate([bad_imaging_frames, pad])
        else:
            # >1 frame mismatch is a real Suite2p inconsistency, not a
            # benign off-by-one. Warn loudly so the user investigates,
            # and still align to dff frame count (truncate / pad with
            # False) rather than crashing — analysis can proceed but the
            # warning is preserved in the log for QC.
            log.warning(
                "bad_imaging_frames length %d differs from dff frames %d "
                "by %+d — this is beyond the off-by-one tolerance and "
                "indicates a Suite2p frame-count inconsistency. "
                "Aligning to dff length (filling with False where short, "
                "truncating where long); investigate the underlying "
                "Suite2p output.",
                n_bad,
                n_dff,
                delta,
            )
            if delta > 0:
                datasets["bad_imaging_frames"] = bad_imaging_frames[:n_dff]
            else:
                pad = np.zeros(-delta, dtype=bool)
                datasets["bad_imaging_frames"] = np.concatenate([bad_imaging_frames, pad])

    # Suite2p deconvolved spikes (raw + max-normalized)
    if deconv is not None:
        datasets["deconv"] = deconv
        datasets["deconv_norm"] = deconv_norm

    # Per-ROI QC metrics (roi_qc/* keys)
    datasets.update(qc_datasets)

    # Persist per-ROI dff saturation counts for QC. Non-zero entries
    # signal F0 estimation issues for that ROI — the frontend ROI viewer
    # surfaces them alongside the other roi_qc/* fields.
    datasets["roi_qc/dff_n_clipped"] = dff_n_clipped.astype(np.int32)

    # Soma-classifier probabilities — surfaced in the ROI viewer to flag
    # ambiguous ROIs.  See docs/soma-classifier.md.
    if p_soma_arr is not None and p_dend_arr is not None and p_artefact_arr is not None:
        datasets["roi_qc/p_soma"] = p_soma_arr
        datasets["roi_qc/p_dend"] = p_dend_arr
        datasets["roi_qc/p_artefact"] = p_artefact_arr

    # --- Optional CASCADE spike inference ---
    if run_cascade:
        from hm2p.calcium.spikes import predict_spike_rates

        spikes = predict_spike_rates(dff, model_name=cascade_model, fps=fps)
        datasets["spikes"] = spikes

    attrs: dict[str, object] = {
        "session_id": session_id,
        "fps_imaging": fps,
        "extractor": "suite2p",
        "neuropil_method": neuropil_method,
        "f0_method": f0_method,
    }
    if neuropil_coeff_used is not None:
        attrs["neuropil_coefficient"] = neuropil_coeff_used

    # Persist the original TIFF frame count (pre-extraction) so Stage 5
    # can compare imaging-pulse count against the unmodified TIFF count
    # rather than the post-extraction dff column count. ``nframes_init``
    # is Suite2p's pre-trim count; ``nframes`` is the post-extraction
    # count. Either is preferred over ``dff.shape[1]`` because that
    # equals ``nframes`` and therefore cannot detect Suite2p's silent
    # frame trims.
    if ops_path.exists():
        ops_for_attrs = np.load(ops_path, allow_pickle=True).item()
        n_tiff_init = ops_for_attrs.get("nframes_init")
        if n_tiff_init is None:
            n_tiff_init = ops_for_attrs.get("nframes")
        if n_tiff_init is not None:
            attrs["n_tiff_frames"] = int(n_tiff_init)

    write_h5(output_path, datasets, attrs=attrs)
