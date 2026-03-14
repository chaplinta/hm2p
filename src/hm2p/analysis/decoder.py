"""Population decoders for head direction.

Implements multiple population decoding approaches for HD cells:

1. **Population Vector Average (PVA)** — the classic Georgopoulos (1986)
   approach. Weights each cell's preferred direction by its current activity.
   Works with as few as 3-5 HD cells.

2. **Template matching** — correlates the population activity vector at each
   time bin with the mean activity template at each HD bin. Simple,
   non-parametric, works with any N.

3. **K-fold cross-validated PVA** — builds tuning curves on training folds,
   decodes test folds.

References
----------
Georgopoulos, A. P., Schwartz, A. B. & Kettner, R. E. 1986. "Neuronal
    population coding of movement direction." Science.
    doi:10.1126/science.3749885

Peyrache, A., Lacber, M. M., Bhatt, D. & Bhatt D. 2015. "Internally
    organized mechanisms of the head direction sense." Nature Neuroscience.
    doi:10.1038/nn.3968

Wilson, M. A. & McNaughton, B. L. 1993. "Dynamics of the hippocampal
    ensemble code for space." Science. doi:10.1126/science.8351520
    (template matching / population vector approaches for spatial coding)

All functions are pure numpy — no I/O, no classes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import compute_hd_tuning_curve


def build_decoder(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Build tuning curves and preferred directions for PVA decoder.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
        Neural signals for each cell (dF/F, deconv, etc.).
    hd_deg : (n_frames,) float
        Head direction in degrees.
    mask : (n_frames,) bool
        Valid frames.
    n_bins : int
        Number of angular bins.
    smoothing_sigma_deg : float
        Gaussian smoothing sigma in degrees.

    Returns
    -------
    dict
        ``"tuning_curves"`` — (n_cells, n_bins) array.
        ``"bin_centers"`` — (n_bins,) array in degrees.
        ``"preferred_directions"`` — (n_cells,) PD in degrees [0, 360).
        ``"mvl"`` — (n_cells,) mean vector length from tuning curve.
        ``"n_cells"`` — number of cells.
        ``"n_bins"`` — number of bins.
    """
    n_cells = signals.shape[0]
    tuning_curves = np.empty((n_cells, n_bins), dtype=np.float64)
    preferred_directions = np.empty(n_cells, dtype=np.float64)
    mvl = np.empty(n_cells, dtype=np.float64)
    bin_centers = None

    for i in range(n_cells):
        tc, bc = compute_hd_tuning_curve(
            signals[i], hd_deg, mask, n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
        )
        tc = np.where(np.isnan(tc), 0.0, tc)
        tuning_curves[i] = tc
        if bin_centers is None:
            bin_centers = bc

        # Preferred direction: circular mean weighted by tuning curve
        theta_rad = np.deg2rad(bc)
        tc_pos = np.maximum(tc, 0.0)
        tc_sum = tc_pos.sum()
        if tc_sum > 0:
            C = np.sum(tc_pos * np.cos(theta_rad)) / tc_sum
            S = np.sum(tc_pos * np.sin(theta_rad)) / tc_sum
            preferred_directions[i] = np.rad2deg(np.arctan2(S, C)) % 360.0
            mvl[i] = np.sqrt(C**2 + S**2)
        else:
            preferred_directions[i] = 0.0
            mvl[i] = 0.0

    return {
        "tuning_curves": tuning_curves,
        "bin_centers": bin_centers,
        "preferred_directions": preferred_directions,
        "mvl": mvl,
        "n_cells": n_cells,
        "n_bins": n_bins,
    }


def decode_hd(
    signals: npt.NDArray[np.floating],
    decoder: dict,
    time_bins: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode HD using Population Vector Average (PVA).

    For each time point, computes the weighted circular mean of preferred
    directions, where weights are the z-scored activity multiplied by
    each cell's mean vector length (MVL).

    decoded_angle = atan2(sum(w_i * sin(PD_i)), sum(w_i * cos(PD_i)))

    Parameters
    ----------
    signals : (n_cells, n_frames) float
        Neural signals to decode.
    decoder : dict
        Output from :func:`build_decoder`.
    time_bins : int
        Number of frames to average per decode step. 1 = frame-by-frame.

    Returns
    -------
    decoded_deg : (n_steps,) float
        Decoded HD in degrees [0, 360).
    confidence : (n_steps,) float
        Resultant vector length per frame (0-1). Higher = more confident.
    """
    pds = decoder["preferred_directions"]  # (n_cells,)
    cell_mvl = decoder["mvl"]  # (n_cells,)
    n_cells = decoder["n_cells"]
    n_frames = signals.shape[1]

    # Bin signals if needed
    if time_bins > 1:
        n_steps = n_frames // time_bins
        binned = np.zeros((n_cells, n_steps), dtype=np.float64)
        for i in range(n_steps):
            binned[:, i] = np.mean(
                signals[:, i * time_bins:(i + 1) * time_bins], axis=1,
            )
    else:
        n_steps = n_frames
        binned = signals.astype(np.float64)

    # Z-score each cell's activity so all cells contribute proportionally
    cell_mean = np.mean(binned, axis=1, keepdims=True)
    cell_std = np.std(binned, axis=1, keepdims=True)
    cell_std = np.where(cell_std < 1e-10, 1.0, cell_std)
    z_activity = (binned - cell_mean) / cell_std  # (n_cells, n_steps)

    # Shift so minimum is 0 (PVA needs non-negative weights)
    z_min = z_activity.min(axis=1, keepdims=True)
    z_activity = z_activity - z_min

    # Pre-compute PD components
    pd_rad = np.deg2rad(pds)
    cos_pd = np.cos(pd_rad)  # (n_cells,)
    sin_pd = np.sin(pd_rad)  # (n_cells,)

    # Weights: activity * MVL for each cell
    weights = z_activity * cell_mvl[:, None]  # (n_cells, n_steps)

    # Vectorized PVA
    C = cos_pd @ weights  # (n_steps,)
    S = sin_pd @ weights  # (n_steps,)

    decoded_deg = np.rad2deg(np.arctan2(S, C)) % 360.0

    # Confidence = resultant vector length (normalize by sum of weights)
    R = np.sqrt(C**2 + S**2)
    w_sum = np.sum(weights, axis=0)
    w_sum = np.where(w_sum < 1e-10, 1.0, w_sum)
    confidence = R / w_sum
    confidence = np.clip(confidence, 0.0, 1.0)

    return decoded_deg, confidence


def pva_decode(
    signals: npt.NDArray[np.floating],
    preferred_directions: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_] | None = None,
    mvl_weights: npt.NDArray[np.floating] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standalone Population Vector Average decoder.

    Simplified PVA that takes preferred directions directly — no need to
    build tuning curves first. Works with as few as 3 HD cells.

    For each time bin:
        decoded_HD = atan2(sum(w_i * sin(PD_i)), sum(w_i * cos(PD_i)))

    where w_i is the (z-scored, shifted-positive) activity of cell i,
    optionally multiplied by per-cell MVL weights.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
        Neural signals (dF/F, deconv, etc.).
    preferred_directions : (n_cells,) float
        Preferred direction of each cell in degrees.
    mask : (n_frames,) bool or None
        Valid frames. If None, all frames are used. Masked-out frames
        get decoded_deg=NaN, confidence=0.
    mvl_weights : (n_cells,) float or None
        Per-cell weight (e.g. mean vector length). If None, all cells
        weighted equally.

    Returns
    -------
    decoded_deg : (n_frames,) float
        Decoded HD in degrees [0, 360). NaN for masked-out frames.
    confidence : (n_frames,) float
        Resultant vector length per frame (0-1). 0 for masked-out frames.

    References
    ----------
    Georgopoulos et al. 1986. "Neuronal population coding of movement
        direction." Science. doi:10.1126/science.3749885
    """
    signals = np.asarray(signals, dtype=np.float64)
    preferred_directions = np.asarray(preferred_directions, dtype=np.float64)
    n_cells, n_frames = signals.shape

    if preferred_directions.shape[0] != n_cells:
        raise ValueError(
            f"preferred_directions length ({preferred_directions.shape[0]}) "
            f"does not match n_cells ({n_cells})"
        )

    decoded_deg = np.full(n_frames, np.nan, dtype=np.float64)
    confidence = np.zeros(n_frames, dtype=np.float64)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid_idx = np.where(mask)[0]
    else:
        valid_idx = np.arange(n_frames)

    if len(valid_idx) == 0:
        return decoded_deg, confidence

    sig = signals[:, valid_idx]

    # Z-score each cell across valid frames
    cell_mean = np.mean(sig, axis=1, keepdims=True)
    cell_std = np.std(sig, axis=1, keepdims=True)
    cell_std = np.where(cell_std < 1e-10, 1.0, cell_std)
    z = (sig - cell_mean) / cell_std

    # Shift so minimum per cell is 0 (PVA needs non-negative weights)
    z = z - z.min(axis=1, keepdims=True)

    # Apply MVL weights if provided
    if mvl_weights is not None:
        mvl_weights = np.asarray(mvl_weights, dtype=np.float64)
        z = z * mvl_weights[:, None]

    # PVA computation
    pd_rad = np.deg2rad(preferred_directions)
    cos_pd = np.cos(pd_rad)
    sin_pd = np.sin(pd_rad)

    C = cos_pd @ z  # (n_valid,)
    S = sin_pd @ z

    dec = np.rad2deg(np.arctan2(S, C)) % 360.0
    R = np.sqrt(C**2 + S**2)
    w_sum = np.sum(z, axis=0)
    w_sum = np.where(w_sum < 1e-10, 1.0, w_sum)
    conf = np.clip(R / w_sum, 0.0, 1.0)

    decoded_deg[valid_idx] = dec
    confidence[valid_idx] = conf

    return decoded_deg, confidence


def template_decode(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Template matching decoder for head direction.

    Builds a population activity template (mean activity vector at each HD
    bin), then decodes each time point by finding the HD bin whose template
    has the highest correlation with the observed population activity.

    Works with any number of cells. No distributional assumptions.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
        Neural signals (dF/F, deconv, etc.).
    hd_deg : (n_frames,) float
        Head direction in degrees.
    mask : (n_frames,) bool
        Valid frames (used for both template building and decoding).
    n_bins : int
        Number of angular bins for the template.

    Returns
    -------
    decoded_deg : (n_frames,) float
        Decoded HD in degrees (bin center of best-matching template).
        NaN for masked-out frames.
    confidence : (n_frames,) float
        Peak correlation value per frame (higher = more confident).
        0 for masked-out frames.

    References
    ----------
    Wilson, M. A. & McNaughton, B. L. 1993. "Dynamics of the hippocampal
        ensemble code for space." Science. doi:10.1126/science.8351520
    """
    signals = np.asarray(signals, dtype=np.float64)
    hd_deg = np.asarray(hd_deg, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n_cells, n_frames = signals.shape

    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    valid_idx = np.where(mask)[0]
    hd_valid = hd_deg[valid_idx] % 360.0

    # Build template: mean activity of each cell at each HD bin
    template = np.zeros((n_cells, n_bins), dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    bin_assignments = np.digitize(hd_valid, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    for b in range(n_bins):
        in_bin = valid_idx[bin_assignments == b]
        if len(in_bin) > 0:
            template[:, b] = np.mean(signals[:, in_bin], axis=1)
            bin_counts[b] = len(in_bin)

    # Handle empty bins: interpolate from neighbors (circular)
    empty = bin_counts == 0
    if np.any(empty) and not np.all(empty):
        for b in np.where(empty)[0]:
            # Find nearest non-empty bins (circular)
            for offset in range(1, n_bins):
                left = (b - offset) % n_bins
                right = (b + offset) % n_bins
                neighbors = []
                if not empty[left]:
                    neighbors.append(template[:, left])
                if not empty[right]:
                    neighbors.append(template[:, right])
                if neighbors:
                    template[:, b] = np.mean(neighbors, axis=0)
                    break

    # Z-score the template across bins (per cell) so correlation is meaningful
    t_mean = np.mean(template, axis=1, keepdims=True)
    t_std = np.std(template, axis=1, keepdims=True)
    t_std = np.where(t_std < 1e-10, 1.0, t_std)
    template_z = (template - t_mean) / t_std  # (n_cells, n_bins)

    # Decode each valid frame
    decoded_deg = np.full(n_frames, np.nan, dtype=np.float64)
    confidence = np.zeros(n_frames, dtype=np.float64)

    if len(valid_idx) == 0:
        return decoded_deg, confidence

    # Z-score signals across valid frames (per cell)
    sig_valid = signals[:, valid_idx]
    s_mean = np.mean(sig_valid, axis=1, keepdims=True)
    s_std = np.std(sig_valid, axis=1, keepdims=True)
    s_std = np.where(s_std < 1e-10, 1.0, s_std)
    sig_z = (sig_valid - s_mean) / s_std  # (n_cells, n_valid)

    # Correlation of each frame with each template bin
    # corr = template_z.T @ sig_z / n_cells  →  (n_bins, n_valid)
    corr = template_z.T @ sig_z / n_cells

    best_bin = np.argmax(corr, axis=0)  # (n_valid,)
    best_corr = corr[best_bin, np.arange(len(valid_idx))]

    decoded_deg[valid_idx] = bin_centers[best_bin]
    confidence[valid_idx] = np.clip(best_corr, 0.0, 1.0)

    return decoded_deg, confidence


def template_decode_cv(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_folds: int = 5,
    n_bins: int = 36,
    rng: np.random.Generator | None = None,
) -> dict:
    """K-fold cross-validated template matching decoder.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    n_folds : int
    n_bins : int
    rng : Generator or None

    Returns
    -------
    dict
        Same structure as :func:`cross_validated_decode`.
    """
    if rng is None:
        rng = np.random.default_rng()

    valid_idx = np.where(mask)[0]
    n_valid = len(valid_idx)
    shuffled = rng.permutation(valid_idx)
    fold_size = n_valid // n_folds

    all_decoded = np.zeros(n_valid, dtype=np.float64)
    all_actual = np.zeros(n_valid, dtype=np.float64)
    all_confidence = np.zeros(n_valid, dtype=np.float64)

    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_valid
        test_idx = shuffled[test_start:test_end]
        train_idx = np.concatenate([shuffled[:test_start], shuffled[test_end:]])

        # Build template from training data
        train_mask = np.zeros_like(mask)
        train_mask[train_idx] = True

        # Use template_decode internals: build template on train, decode test
        # Build template
        n_cells = signals.shape[0]
        hd_train = hd_deg[train_idx] % 360.0
        template = np.zeros((n_cells, n_bins), dtype=np.float64)
        bin_assignments = np.clip(
            np.digitize(hd_train, bin_edges) - 1, 0, n_bins - 1,
        )
        bin_counts = np.zeros(n_bins, dtype=np.int64)
        for b in range(n_bins):
            in_bin = train_idx[bin_assignments == b]
            if len(in_bin) > 0:
                template[:, b] = np.mean(signals[:, in_bin], axis=1)
                bin_counts[b] = len(in_bin)

        # Fill empty bins
        empty = bin_counts == 0
        if np.any(empty) and not np.all(empty):
            for b in np.where(empty)[0]:
                for offset in range(1, n_bins):
                    left = (b - offset) % n_bins
                    right = (b + offset) % n_bins
                    neighbors = []
                    if not empty[left]:
                        neighbors.append(template[:, left])
                    if not empty[right]:
                        neighbors.append(template[:, right])
                    if neighbors:
                        template[:, b] = np.mean(neighbors, axis=0)
                        break

        # Z-score template
        t_mean = np.mean(template, axis=1, keepdims=True)
        t_std = np.std(template, axis=1, keepdims=True)
        t_std = np.where(t_std < 1e-10, 1.0, t_std)
        template_z = (template - t_mean) / t_std

        # Z-score test signals (using train statistics for proper CV)
        train_sig = signals[:, train_idx]
        s_mean = np.mean(train_sig, axis=1, keepdims=True)
        s_std = np.std(train_sig, axis=1, keepdims=True)
        s_std = np.where(s_std < 1e-10, 1.0, s_std)

        test_sig = (signals[:, test_idx] - s_mean) / s_std

        # Correlate
        corr = template_z.T @ test_sig / n_cells
        best_bin = np.argmax(corr, axis=0)
        best_corr = corr[best_bin, np.arange(len(test_idx))]

        all_decoded[test_start:test_end] = bin_centers[best_bin]
        all_actual[test_start:test_end] = hd_deg[test_idx] % 360.0
        all_confidence[test_start:test_end] = np.clip(best_corr, 0.0, 1.0)

    errors = decode_error(all_decoded, all_actual)

    return {
        "decoded_deg": all_decoded,
        "actual_deg": all_actual,
        "confidence": all_confidence,
        "errors": errors,
        "n_folds": n_folds,
    }


def decode_error(
    decoded_deg: npt.NDArray[np.floating],
    actual_deg: npt.NDArray[np.floating],
) -> dict:
    """Compute decoding error statistics.

    Parameters
    ----------
    decoded_deg : (n,) float
        Decoded HD in degrees.
    actual_deg : (n,) float
        Actual HD in degrees.

    Returns
    -------
    dict
        ``"errors_deg"`` — (n,) signed angular errors in [-180, 180].
        ``"abs_errors_deg"`` — (n,) absolute angular errors.
        ``"mean_abs_error"`` — mean absolute error in degrees.
        ``"median_abs_error"`` — median absolute error in degrees.
        ``"circular_mean_error"`` — circular mean of errors.
        ``"circular_std_error"`` — circular standard deviation of errors.
    """
    diff = decoded_deg - actual_deg
    errors = ((diff + 180.0) % 360.0) - 180.0
    abs_errors = np.abs(errors)

    # Circular statistics of error
    theta = np.deg2rad(errors)
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C**2 + S**2)
    circ_mean = float(np.rad2deg(np.arctan2(S, C)))
    R_clipped = min(max(R, 1e-10), 1.0)  # Clip to [eps, 1] for numerical safety
    circ_std = float(np.rad2deg(np.sqrt(-2 * np.log(R_clipped))))

    return {
        "errors_deg": errors,
        "abs_errors_deg": abs_errors,
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "circular_mean_error": circ_mean,
        "circular_std_error": circ_std,
    }


def cross_validated_decode(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_folds: int = 5,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """K-fold cross-validated PVA decoding.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    n_folds : int
    n_bins : int
    smoothing_sigma_deg : float
    rng : Generator or None

    Returns
    -------
    dict
        ``"decoded_deg"`` — (n_valid,) decoded HD.
        ``"actual_deg"`` — (n_valid,) actual HD.
        ``"confidence"`` — (n_valid,) PVA confidence per frame.
        ``"errors"`` — output of :func:`decode_error`.
        ``"n_folds"`` — number of folds.
    """
    if rng is None:
        rng = np.random.default_rng()

    valid_idx = np.where(mask)[0]
    n_valid = len(valid_idx)

    # Shuffle and split into folds
    shuffled = rng.permutation(valid_idx)
    fold_size = n_valid // n_folds

    all_decoded = np.zeros(n_valid, dtype=np.float64)
    all_actual = np.zeros(n_valid, dtype=np.float64)
    all_confidence = np.zeros(n_valid, dtype=np.float64)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_valid
        test_idx = shuffled[test_start:test_end]
        train_idx = np.concatenate([shuffled[:test_start], shuffled[test_end:]])

        # Build train mask
        train_mask = np.zeros_like(mask)
        train_mask[train_idx] = True

        # Build decoder on training data
        dec = build_decoder(
            signals, hd_deg, train_mask,
            n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg,
        )

        # Decode test data
        test_signals = signals[:, test_idx]
        decoded, confidence = decode_hd(test_signals, dec)

        all_decoded[test_start:test_end] = decoded
        all_actual[test_start:test_end] = hd_deg[test_idx] % 360.0
        all_confidence[test_start:test_end] = confidence

    errors = decode_error(all_decoded, all_actual)

    return {
        "decoded_deg": all_decoded,
        "actual_deg": all_actual,
        "confidence": all_confidence,
        "errors": errors,
        "n_folds": n_folds,
    }
