"""Stage 4a — neuropil subtraction.

Three methods (configured in config/pipeline.yaml via neuropil_method):
    fissa      — Spatial ICA on ROI masks + raw movie (most accurate; default)
    estimated  — Per-ROI/per-session coefficient estimated from lower-envelope
                 regression (Dipoppa et al. 2018; more principled than fixed)
    fixed      — F_corr = F - 0.7 * Fneu  (backward-compatible fallback)

CaImAn handles neuropil internally — this module is a no-op for CaImAn sessions.

References:
    Keemink SW, Lowe SC, Pakan JMP, Dylda E, van Rossum MCW, Bhatt DH. 2018.
    "FISSA: A neuropil decontamination toolbox for calcium imaging signals."
    Scientific Reports 8:3493. doi:10.1038/s41598-018-21640-2
    https://github.com/rochefort-lab/fissa

    Dipoppa M, Ranson A, Krumin M, Pachitariu M, Carandini M, Harris KD. 2018.
    "Vision and locomotion shape the interactions between neuron types in mouse
    visual cortex." Neuron 98(3):602-615. doi:10.1016/j.neuron.2018.03.037

    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." doi:10.1101/061507
    https://github.com/MouseLand/suite2p

    Chen T-W, Wardill TJ, Sun Y, et al. 2013. "Ultrasensitive fluorescent
    proteins for imaging neuronal activity." Nature 499:295-300.
    doi:10.1038/nature12354
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def subtract_fixed_coefficient(
    F: np.ndarray,
    Fneu: np.ndarray,
    coefficient: float = 0.7,
) -> np.ndarray:
    """Apply fixed-coefficient neuropil subtraction.

    F_corr = F - coefficient * Fneu

    Parameters
    ----------
    F : np.ndarray
        (n_rois, n_frames) float32 — raw fluorescence traces.
    Fneu : np.ndarray
        (n_rois, n_frames) float32 — neuropil traces.
    coefficient : float
        Neuropil mixing coefficient (default 0.7, from Pachitariu et al. 2017).

    Returns
    -------
    np.ndarray
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
    """
    return F - coefficient * Fneu


def estimate_neuropil_coefficient(
    F: np.ndarray,
    Fneu: np.ndarray,
    percentile: float = 20.0,
) -> np.ndarray:
    """Estimate per-ROI neuropil contamination coefficient via lower-envelope regression.

    For each ROI, selects frames where fluorescence is in its lowest
    ``percentile`` (sparse-firing epochs) and regresses F on Fneu over those
    frames **with an intercept** (equivalent to mean-centring both signals
    before slope estimation). The slope of this regression is the
    estimated contamination coefficient for that ROI. This approach is
    adapted from Dipoppa et al. (2018), where they estimated the
    coefficient from the lower envelope of the F vs Fneu scatter plot
    using sparsely-firing cells.

    The intercept is required because resting somatic fluorescence has a
    DC offset independent of the neuropil signal: regressing through the
    origin would absorb that offset into the slope and bias the
    coefficient upward, especially for sparsely-firing somas with bright
    baselines. Mean-centring both vectors before computing the OLS slope
    is mathematically identical to fitting an intercept and is cheaper
    and numerically more stable than building a design matrix.

    Parameters
    ----------
    F : np.ndarray
        (n_rois, n_frames) float32 — raw fluorescence traces.
    Fneu : np.ndarray
        (n_rois, n_frames) float32 — neuropil traces.
    percentile : float
        Lower percentile threshold for selecting sparse-activity epochs
        (default 20.0). Frames where F is below this percentile of the ROI's
        own distribution are used for regression.

    Returns
    -------
    np.ndarray
        (n_rois,) float32 — per-ROI estimated neuropil coefficient. Values are
        clipped to [0.0, 1.0] since negative or >1 coefficients are physically
        implausible.

    References
    ----------
    Dipoppa M, Ranson A, Krumin M, Pachitariu M, Carandini M, Harris KD. 2018.
    "Vision and locomotion shape the interactions between neuron types in mouse
    visual cortex." Neuron 98(3):602-615. doi:10.1016/j.neuron.2018.03.037
    """
    if F.ndim != 2 or Fneu.ndim != 2:
        raise ValueError(
            f"F and Fneu must be 2-D (n_rois, n_frames); got {F.shape} and {Fneu.shape}"
        )
    if F.shape != Fneu.shape:
        raise ValueError(f"F shape {F.shape} != Fneu shape {Fneu.shape}")
    if not (0.0 < percentile < 100.0):
        raise ValueError(f"percentile must be in (0, 100); got {percentile}")

    n_rois = F.shape[0]
    coefficients = np.zeros(n_rois, dtype=np.float32)

    for i in range(n_rois):
        f_roi = F[i].astype(np.float64)
        fneu_roi = Fneu[i].astype(np.float64)

        # Select frames in the lower percentile (sparse-activity epochs)
        threshold = np.percentile(f_roi, percentile)
        mask = f_roi <= threshold

        if mask.sum() < 2:
            # Insufficient data — fall back to default coefficient
            log.warning(
                "ROI %d: fewer than 2 frames in bottom %g-th percentile; "
                "using default coefficient 0.7",
                i,
                percentile,
            )
            coefficients[i] = 0.7
            continue

        f_sel = f_roi[mask]
        fneu_sel = fneu_roi[mask]

        # OLS slope **with intercept** via mean-centring (Dipoppa et al.
        # 2018). Equivalent to ``F = a + alpha * Fneu``: the per-ROI DC
        # offset of F is absorbed into the intercept rather than biasing
        # the slope. Required because resting somatic fluorescence has a
        # baseline independent of neuropil — fitting through the origin
        # mixes that baseline into ``alpha`` and inflates the coefficient.
        f_centred = f_sel - f_sel.mean()
        fneu_centred = fneu_sel - fneu_sel.mean()
        fneu_var = float(np.dot(fneu_centred, fneu_centred))
        if fneu_var < 1e-12:
            # Near-zero neuropil variance in the selected window — cannot
            # estimate coefficient reliably.
            log.warning(
                "ROI %d: near-zero neuropil variance; using default coefficient 0.7",
                i,
            )
            coefficients[i] = 0.7
            continue

        alpha = float(np.dot(f_centred, fneu_centred) / fneu_var)
        # Clip to physically plausible range.
        coefficients[i] = float(np.clip(alpha, 0.0, 1.0))

    return coefficients


def subtract_estimated_coefficient(
    F: np.ndarray,
    Fneu: np.ndarray,
    percentile: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-ROI estimated neuropil subtraction.

    Estimates a neuropil coefficient for each ROI individually via
    lower-envelope regression on sparse-activity frames (Dipoppa et al. 2018),
    then applies it: F_corr[i] = F[i] - coefficients[i] * Fneu[i].

    Parameters
    ----------
    F : np.ndarray
        (n_rois, n_frames) float32 — raw fluorescence traces.
    Fneu : np.ndarray
        (n_rois, n_frames) float32 — neuropil traces.
    percentile : float
        Lower percentile used to select sparse-activity epochs for coefficient
        estimation (default 20.0).

    Returns
    -------
    F_corr : np.ndarray
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
    coefficients : np.ndarray
        (n_rois,) float32 — per-ROI estimated neuropil coefficients.

    References
    ----------
    Dipoppa M, Ranson A, Krumin M, Pachitariu M, Carandini M, Harris KD. 2018.
    "Vision and locomotion shape the interactions between neuron types in mouse
    visual cortex." Neuron 98(3):602-615. doi:10.1016/j.neuron.2018.03.037
    """
    coefficients = estimate_neuropil_coefficient(F, Fneu, percentile=percentile)
    F_corr = F - coefficients[:, np.newaxis] * Fneu
    return F_corr.astype(np.float32), coefficients


def load_registered_movie(
    data_bin: Path,
    crop_ly: int,
    crop_lx: int,
    n_frames: int | None = None,
    mmap: bool = True,
) -> np.ndarray:
    """Load a Suite2p registered binary (``data.bin``) as a movie array.

    Suite2p writes the motion-corrected movie to ``data.bin`` as a flat little-
    endian ``int16`` stream of shape ``(n_frames, crop_ly, crop_lx)``, where the
    spatial dimensions are the *cropped* registration window
    (``ops['yrange']`` × ``ops['xrange']``), not the full ``Ly × Lx`` field of
    view. This loader returns that movie so it can be passed to FISSA together
    with crop-aligned masks (see
    :func:`hm2p.calcium.fissa_masks.crop_masks_to_window`).

    Parameters
    ----------
    data_bin : Path
        Path to Suite2p's ``data.bin`` (registered movie).
    crop_ly : int
        Cropped image height (``ops['yrange'][1] - ops['yrange'][0]``).
    crop_lx : int
        Cropped image width (``ops['xrange'][1] - ops['xrange'][0]``).
    n_frames : int or None
        Number of frames. If ``None`` it is inferred from the file size
        (``size / (crop_ly * crop_lx * 2)``).
    mmap : bool
        If True (default) the file is memory-mapped read-only rather than read
        fully into RAM — a full session can be several GB.

    Returns
    -------
    np.ndarray
        ``(n_frames, crop_ly, crop_lx)`` ``int16`` movie (a memmap when
        ``mmap`` is True).

    Raises
    ------
    FileNotFoundError
        If ``data_bin`` does not exist.
    ValueError
        If ``crop_ly``/``crop_lx`` are not positive or the file size is not an
        integer multiple of the per-frame byte count.
    """
    data_bin = Path(data_bin)
    if not data_bin.exists():
        raise FileNotFoundError(f"registered binary not found: {data_bin}")
    if crop_ly <= 0 or crop_lx <= 0:
        raise ValueError(f"crop_ly/crop_lx must be positive; got {crop_ly}x{crop_lx}")

    bytes_per_frame = crop_ly * crop_lx * 2  # int16
    file_size = data_bin.stat().st_size
    if file_size % bytes_per_frame != 0:
        raise ValueError(
            f"data.bin size {file_size} is not a multiple of the per-frame size "
            f"{bytes_per_frame} (crop {crop_ly}x{crop_lx}, int16) — crop "
            "dimensions likely wrong."
        )
    inferred = file_size // bytes_per_frame
    if n_frames is None:
        n_frames = inferred
    elif n_frames > inferred:
        raise ValueError(
            f"requested {n_frames} frames but file only holds {inferred}"
        )

    if mmap:
        movie = np.memmap(
            data_bin, dtype=np.int16, mode="r", shape=(n_frames, crop_ly, crop_lx)
        )
    else:
        movie = np.fromfile(
            data_bin, dtype=np.int16, count=n_frames * crop_ly * crop_lx
        ).reshape(n_frames, crop_ly, crop_lx)
    return movie


def subtract_fissa_from_movie(
    movie: np.ndarray,
    roi_masks: list[np.ndarray],
    output_dir: Path,
    F_fallback: np.ndarray | None = None,
    Fneu_fallback: np.ndarray | None = None,
    n_components: int = 4,
) -> np.ndarray:
    """Run FISSA on an in-memory (or memmapped) movie array.

    Identical to :func:`subtract_fissa` but takes a single movie array of shape
    ``(n_frames, height, width)`` instead of TIFF paths. ``fissa.Experiment``
    accepts a list of numpy arrays as its ``images`` argument, so the Suite2p
    registered binary can be passed directly without first being written to a
    multi-GB temporary TIFF. The movie's spatial dimensions must match the
    ``roi_masks`` shape (use crop-aligned masks — see
    :func:`hm2p.calcium.fissa_masks.crop_masks_to_window`).

    Parameters
    ----------
    movie : np.ndarray
        ``(n_frames, height, width)`` array (e.g. the registered binary loaded
        by :func:`load_registered_movie`).
    roi_masks : list of np.ndarray
        Per-ROI ``(height, width)`` boolean masks aligned to ``movie``'s
        spatial dimensions.
    output_dir : Path
        Directory for FISSA's intermediate cache files. Created if absent.
    F_fallback, Fneu_fallback : np.ndarray or None
        Fallback traces used if FISSA fails (see :func:`subtract_fissa`).
    n_components : int
        Number of ICA components per ROI (default 4).

    Returns
    -------
    np.ndarray
        ``(n_rois, n_frames)`` float32 neuropil-corrected fluorescence.

    Raises
    ------
    ValueError
        If ``movie`` is not 3-D or its spatial dims do not match the masks.
    RuntimeError
        If FISSA fails and no fallback traces are available.
    ImportError
        If the ``fissa`` package is not installed.

    References
    ----------
    Keemink SW, Lowe SC, Pakan JMP, Dylda E, van Rossum MCW, Bhatt DH. 2018.
    "FISSA: A neuropil decontamination toolbox for calcium imaging signals."
    Scientific Reports 8:3493. doi:10.1038/s41598-018-21640-2
    https://github.com/rochefort-lab/fissa
    """
    movie = np.asarray(movie)
    if movie.ndim != 3:
        raise ValueError(f"movie must be 3-D (frames, H, W); got {movie.shape}")
    spatial = movie.shape[1:]
    for i, m in enumerate(roi_masks):
        if m.shape != spatial:
            raise ValueError(
                f"roi_masks[{i}] shape {m.shape} != movie spatial dims {spatial}"
            )

    return _run_fissa(
        images=[movie],
        roi_masks=roi_masks,
        output_dir=Path(output_dir),
        F_fallback=F_fallback,
        Fneu_fallback=Fneu_fallback,
        n_components=n_components,
    )


def subtract_fissa(
    tiff_paths: list[str | Path],
    roi_masks: list[np.ndarray],
    output_dir: Path,
    F_fallback: np.ndarray | None = None,
    Fneu_fallback: np.ndarray | None = None,
    n_components: int = 4,
) -> np.ndarray:
    """Apply FISSA spatial ICA neuropil subtraction.

    Decomposes each ROI's fluorescence into independent spatial sources using
    non-negative matrix factorisation, then returns the source component
    attributed to the ROI (the cell body signal free of neuropil contamination).
    This method is more accurate than fixed-coefficient subtraction in densely
    labelled tissue because it does not assume a fixed contamination fraction.

    FISSA requires the raw TIFF stack(s) and per-ROI spatial masks — it reads
    pixel time-series from concentric rings around each ROI and separates
    somatic from neuropil sources via ICA/NMF.

    Parameters
    ----------
    tiff_paths : list of str or Path
        Ordered list of TIFF file paths constituting the imaging session.
        Passed directly to ``fissa.Experiment`` as the ``images`` argument.
    roi_masks : list of np.ndarray
        Per-ROI binary masks. Each element is a 2-D bool array (height, width)
        marking the pixels belonging to that ROI. Passed to
        ``fissa.Experiment`` as the ``rois`` argument (list-of-lists format:
        ``[[mask_0], [mask_1], ...]``).
    output_dir : Path
        Directory for FISSA's intermediate cache files. Created if absent.
    F_fallback : np.ndarray or None
        (n_rois, n_frames) float32 — raw fluorescence traces to use if FISSA
        fails. Required when ``Fneu_fallback`` is also provided so the module
        can fall back to per-session estimated-coefficient subtraction.
    Fneu_fallback : np.ndarray or None
        (n_rois, n_frames) float32 — neuropil traces used for fallback
        subtraction. Must be provided together with ``F_fallback``.
    n_components : int
        Number of ICA components per ROI (default 4, matching FISSA defaults).

    Returns
    -------
    np.ndarray
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
        If FISSA fails and fallback traces are provided, returns the result of
        per-session estimated-coefficient subtraction instead. If no fallback
        is available, re-raises the FISSA exception.

    Raises
    ------
    RuntimeError
        If FISSA fails AND no fallback traces are available.
    ImportError
        If the ``fissa`` package is not installed.

    References
    ----------
    Keemink SW, Lowe SC, Pakan JMP, Dylda E, van Rossum MCW, Bhatt DH. 2018.
    "FISSA: A neuropil decontamination toolbox for calcium imaging signals."
    Scientific Reports 8:3493. doi:10.1038/s41598-018-21640-2
    https://github.com/rochefort-lab/fissa

    Chen T-W, Wardill TJ, Sun Y, et al. 2013. "Ultrasensitive fluorescent
    proteins for imaging neuronal activity." Nature 499:295-300.
    doi:10.1038/nature12354
    """
    return _run_fissa(
        images=[str(p) for p in tiff_paths],
        roi_masks=roi_masks,
        output_dir=Path(output_dir),
        F_fallback=F_fallback,
        Fneu_fallback=Fneu_fallback,
        n_components=n_components,
    )


def _run_fissa(
    images: list,
    roi_masks: list[np.ndarray],
    output_dir: Path,
    F_fallback: np.ndarray | None,
    Fneu_fallback: np.ndarray | None,
    n_components: int,
) -> np.ndarray:
    """Run ``fissa.Experiment`` on ``images`` and return separated ROI traces.

    Shared core for :func:`subtract_fissa` (TIFF paths) and
    :func:`subtract_fissa_from_movie` (movie array). ``images`` is whatever
    ``fissa.Experiment`` accepts as its ``images`` argument — a list of TIFF
    path strings or a list of ``(frames, H, W)`` numpy arrays.

    Parameters
    ----------
    images : list
        ``fissa.Experiment`` ``images`` argument (TIFF paths or array list).
    roi_masks : list of np.ndarray
        Per-ROI boolean masks.
    output_dir : Path
        FISSA cache directory (created if absent).
    F_fallback, Fneu_fallback : np.ndarray or None
        Fallback traces used if FISSA raises.
    n_components : int
        ICA components per ROI.

    Returns
    -------
    np.ndarray
        ``(n_rois, n_frames)`` float32 separated traces, or the estimated-
        coefficient fallback if FISSA fails and fallback traces are provided.

    Raises
    ------
    ImportError
        If ``fissa`` is not installed.
    RuntimeError
        If FISSA fails and no fallback traces are available.
    """
    try:
        import fissa  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "fissa is required for FISSA neuropil subtraction. "
            "Install it in a compatible environment: "
            "see docs/manual-installs.md (fissa pins scikit-learn<1.2)."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # FISSA indexes ``rois`` per trial: ``self.rois[trial]`` is the roiset (the
    # list of all ROI masks) for that trial. A single roiset is passed as a
    # one-element outer list — ``[[mask0, mask1, ...]]`` — which FISSA replicates
    # across trials (its ``len(rois) == 1`` branch). Passing a flat list of N
    # masks instead would be read as N trials and collapse to a single ROI.
    rois_fissa = [[np.asarray(mask, dtype=bool) for mask in roi_masks]]

    try:
        exp = fissa.Experiment(
            images=images,
            rois=rois_fissa,
            folder=str(output_dir),
            nRegions=n_components,
        )
        exp.separate()

        n_rois = len(roi_masks)
        # exp.result[i][0] is the separated somatic signal for ROI i
        # (cell index 0 — FISSA convention for the primary source)
        # Shape varies: (n_frames,) or (1, n_frames); normalise to 1-D then stack.
        traces = []
        for i in range(n_rois):
            raw = exp.result[i][0]
            if raw.ndim == 2:
                raw = raw[0]
            traces.append(raw.astype(np.float32))

        return np.stack(traces, axis=0)

    except Exception as exc:
        if F_fallback is not None and Fneu_fallback is not None:
            log.warning(
                "FISSA failed (%s). Falling back to per-session estimated-coefficient "
                "neuropil subtraction.",
                exc,
            )
            F_corr, _ = subtract_estimated_coefficient(F_fallback, Fneu_fallback)
            return F_corr
        raise RuntimeError(
            f"FISSA neuropil subtraction failed and no fallback traces were provided: {exc}"
        ) from exc
