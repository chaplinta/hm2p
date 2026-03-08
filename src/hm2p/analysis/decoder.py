"""Bayesian population decoder for head direction.

Implements a maximum-likelihood Bayesian decoder that uses population
tuning curves to decode HD from single-trial neural activity. Standard
approach in HD cell research (Zhang, Sejnowski & Bhatt, 1998; Peyrache
et al., 2015).

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
    min_rate: float = 1e-6,
) -> dict:
    """Build tuning curves for all cells (decoder training).

    Parameters
    ----------
    signals : (n_cells, n_frames) float
        Neural signals for each cell.
    hd_deg : (n_frames,) float
        Head direction in degrees.
    mask : (n_frames,) bool
        Valid frames.
    n_bins : int
        Number of angular bins.
    smoothing_sigma_deg : float
        Gaussian smoothing sigma in degrees.
    min_rate : float
        Floor value to avoid log(0) in the decoder.

    Returns
    -------
    dict
        ``"tuning_curves"`` — (n_cells, n_bins) array.
        ``"bin_centers"`` — (n_bins,) array in degrees.
        ``"n_cells"`` — number of cells.
        ``"n_bins"`` — number of bins.
    """
    n_cells = signals.shape[0]
    tuning_curves = np.empty((n_cells, n_bins), dtype=np.float64)
    bin_centers = None

    for i in range(n_cells):
        tc, bc = compute_hd_tuning_curve(
            signals[i], hd_deg, mask, n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
        )
        tuning_curves[i] = np.where(np.isnan(tc), min_rate, np.maximum(tc, min_rate))
        if bin_centers is None:
            bin_centers = bc

    return {
        "tuning_curves": tuning_curves,
        "bin_centers": bin_centers,
        "n_cells": n_cells,
        "n_bins": n_bins,
    }


def decode_hd(
    signals: npt.NDArray[np.floating],
    decoder: dict,
    time_bins: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode HD from population activity using Bayesian maximum likelihood.

    For each time point (or time bin), computes the posterior probability
    over HD bins assuming Poisson-like firing with tuning curves as the
    rate model (flat prior). The decoded HD is the circular mean of the
    posterior.

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
    posterior : (n_steps, n_bins) float
        Posterior probability over HD bins for each step.
    """
    tc = decoder["tuning_curves"]  # (n_cells, n_bins)
    bin_centers = decoder["bin_centers"]  # (n_bins,)
    n_cells, n_bins = tc.shape
    n_frames = signals.shape[1]

    # Bin signals if needed
    if time_bins > 1:
        n_steps = n_frames // time_bins
        binned = np.zeros((n_cells, n_steps), dtype=np.float64)
        for i in range(n_steps):
            binned[:, i] = np.mean(signals[:, i * time_bins:(i + 1) * time_bins], axis=1)
    else:
        n_steps = n_frames
        binned = signals.astype(np.float64)

    # Log tuning curves for numerical stability
    log_tc = np.log(tc + 1e-30)  # (n_cells, n_bins)

    # Compute log-posterior for each time step
    # log P(θ|r) ∝ sum_i [r_i * log(f_i(θ)) - f_i(θ)]  (Poisson likelihood)
    # With flat prior, just sum over cells
    posterior = np.zeros((n_steps, n_bins), dtype=np.float64)

    for t in range(n_steps):
        r = binned[:, t]  # (n_cells,)
        # log-likelihood for each bin
        log_lik = np.sum(r[:, None] * log_tc - tc, axis=0)  # (n_bins,)
        # Convert to probability (softmax)
        log_lik -= np.max(log_lik)  # Numerical stability
        prob = np.exp(log_lik)
        prob /= np.sum(prob)
        posterior[t] = prob

    # Decoded direction: circular mean of posterior
    theta_rad = np.deg2rad(bin_centers)
    decoded_deg = np.zeros(n_steps, dtype=np.float64)
    for t in range(n_steps):
        C = np.sum(posterior[t] * np.cos(theta_rad))
        S = np.sum(posterior[t] * np.sin(theta_rad))
        decoded_deg[t] = np.rad2deg(np.arctan2(S, C)) % 360.0

    return decoded_deg, posterior


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
    """K-fold cross-validated decoding.

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
        decoded, _ = decode_hd(test_signals, dec)

        all_decoded[test_start:test_end] = decoded
        all_actual[test_start:test_end] = hd_deg[test_idx] % 360.0

    errors = decode_error(all_decoded, all_actual)

    return {
        "decoded_deg": all_decoded,
        "actual_deg": all_actual,
        "errors": errors,
        "n_folds": n_folds,
    }
