"""Information-theoretic analysis for HD cells.

Mutual information, Skaggs information rate, and redundancy analysis
for characterizing neural coding efficiency. All functions pure numpy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def skaggs_info_rate(
    tuning_curve: npt.NDArray[np.floating],
    occupancy: npt.NDArray[np.floating],
) -> float:
    """Skaggs et al. (1993) spatial/HD information rate (bits/spike).

    SI = sum_i p_i * (r_i / r_mean) * log2(r_i / r_mean)

    Parameters
    ----------
    tuning_curve : (n_bins,) float
        Mean signal per bin.
    occupancy : (n_bins,) float
        Time spent in each bin (frames or seconds).

    Returns
    -------
    float
        Information in bits/spike (or bits/event).
    """
    valid = ~np.isnan(tuning_curve) & (occupancy > 0)
    r = tuning_curve[valid]
    occ = occupancy[valid]

    total_occ = np.sum(occ)
    if total_occ == 0:
        return 0.0

    p = occ / total_occ
    r_mean = np.sum(p * r)
    if r_mean <= 0:
        return 0.0

    ratio = r / r_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p * ratio * np.log2(ratio)
    terms = np.where(np.isfinite(terms), terms, 0.0)
    return float(np.sum(terms))


def mutual_information_binned(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_hd_bins: int = 36,
    n_signal_bins: int = 10,
) -> float:
    """Estimate mutual information I(signal; HD) using binned method.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    n_hd_bins : int
    n_signal_bins : int

    Returns
    -------
    float
        Mutual information in bits.
    """
    sig = np.asarray(signal[mask], dtype=np.float64)
    hd = np.mod(np.asarray(hd_deg[mask], dtype=np.float64), 360.0)

    if len(sig) < 10:
        return 0.0

    # Bin signal using quantiles for equal-count bins
    sig_edges = np.quantile(sig, np.linspace(0, 1, n_signal_bins + 1))
    sig_edges[-1] += 1e-10  # Ensure last edge includes max
    sig_idx = np.clip(np.digitize(sig, sig_edges) - 1, 0, n_signal_bins - 1)

    hd_edges = np.linspace(0, 360, n_hd_bins + 1)
    hd_idx = np.clip(np.digitize(hd, hd_edges) - 1, 0, n_hd_bins - 1)

    # Joint probability
    joint = np.zeros((n_signal_bins, n_hd_bins), dtype=np.float64)
    np.add.at(joint, (sig_idx, hd_idx), 1.0)
    joint /= joint.sum()

    # Marginals
    p_sig = joint.sum(axis=1)
    p_hd = joint.sum(axis=0)

    # MI = sum p(x,y) log2(p(x,y) / (p(x) * p(y)))  — vectorized
    outer = p_sig[:, None] * p_hd[None, :]
    valid = (joint > 0) & (outer > 0)
    mi = np.sum(joint[valid] * np.log2(joint[valid] / outer[valid]))
    return float(mi)


def information_per_cell(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_hd_bins: int = 36,
    n_signal_bins: int = 10,
) -> np.ndarray:
    """Compute mutual information for each cell.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    n_hd_bins : int
    n_signal_bins : int

    Returns
    -------
    info : (n_cells,) float
        MI in bits per cell.
    """
    n_cells = signals.shape[0]
    info = np.zeros(n_cells, dtype=np.float64)
    for i in range(n_cells):
        info[i] = mutual_information_binned(
            signals[i], hd_deg, mask,
            n_hd_bins=n_hd_bins, n_signal_bins=n_signal_bins,
        )
    return info


def synergy_redundancy(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    cell_a: int,
    cell_b: int,
    n_hd_bins: int = 18,
    n_signal_bins: int = 5,
) -> dict:
    """Estimate synergy/redundancy between two cells.

    Uses the partial information decomposition approximation:
    Redundancy ≈ I(A;HD) + I(B;HD) - I(A,B;HD)

    If > 0: redundant. If < 0: synergistic.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    cell_a, cell_b : int
        Cell indices.
    n_hd_bins : int
    n_signal_bins : int

    Returns
    -------
    dict
        ``"info_a"`` — MI of cell A.
        ``"info_b"`` — MI of cell B.
        ``"info_joint"`` — MI of (A, B) jointly.
        ``"redundancy"`` — I(A) + I(B) - I(A,B). Positive = redundant.
    """
    info_a = mutual_information_binned(
        signals[cell_a], hd_deg, mask, n_hd_bins, n_signal_bins,
    )
    info_b = mutual_information_binned(
        signals[cell_b], hd_deg, mask, n_hd_bins, n_signal_bins,
    )

    # Joint signal: concatenate bins
    sig_a = signals[cell_a, mask]
    sig_b = signals[cell_b, mask]
    hd = np.mod(hd_deg[mask], 360.0)

    # 2D signal binning
    a_edges = np.quantile(sig_a, np.linspace(0, 1, n_signal_bins + 1))
    b_edges = np.quantile(sig_b, np.linspace(0, 1, n_signal_bins + 1))
    a_edges[-1] += 1e-10
    b_edges[-1] += 1e-10
    a_idx = np.clip(np.digitize(sig_a, a_edges) - 1, 0, n_signal_bins - 1)
    b_idx = np.clip(np.digitize(sig_b, b_edges) - 1, 0, n_signal_bins - 1)
    joint_idx = a_idx * n_signal_bins + b_idx

    hd_edges = np.linspace(0, 360, n_hd_bins + 1)
    hd_idx = np.clip(np.digitize(hd, hd_edges) - 1, 0, n_hd_bins - 1)

    n_joint = n_signal_bins * n_signal_bins
    joint = np.zeros((n_joint, n_hd_bins), dtype=np.float64)
    np.add.at(joint, (joint_idx, hd_idx), 1.0)
    joint /= joint.sum()

    p_sig = joint.sum(axis=1)
    p_hd = joint.sum(axis=0)

    outer = p_sig[:, None] * p_hd[None, :]
    valid = (joint > 0) & (outer > 0)
    info_joint = float(np.sum(joint[valid] * np.log2(joint[valid] / outer[valid])))

    redundancy = info_a + info_b - info_joint

    return {
        "info_a": info_a,
        "info_b": info_b,
        "info_joint": float(info_joint),
        "redundancy": float(redundancy),
    }
