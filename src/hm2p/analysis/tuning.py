"""HD and place tuning curve computation.

Pure numpy/scipy functions for computing head-direction tuning curves,
place rate maps, and associated statistics. No I/O — all functions take
numpy arrays as input and return numpy arrays or scalars.

References:
    Skaggs et al. 1996. "Theta phase precession in hippocampal neuronal
    populations and the compression of temporal sequences."
    Hippocampus 6(2):149-172. doi:10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K

    Taube et al. 1990. "Head-direction cells recorded from the
    postsubiculum in freely moving rats." J Neurosci 10(2):420-435.
    doi:10.1523/JNEUROSCI.10-02-00420.1990
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


# ---------------------------------------------------------------------------
# HD tuning
# ---------------------------------------------------------------------------


def compute_hd_tuning_curve(
    signal: np.ndarray,
    hd_deg: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an occupancy-normalised HD tuning curve.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal (dF/F, deconvolved rate, or binary events).
    hd_deg : (n_frames,) float
        Head direction in degrees (may be unwrapped; mod 360 applied internally).
    mask : (n_frames,) bool
        Valid frames to include (e.g. moving, not bad_behav).
    n_bins : int
        Number of angular bins spanning [0, 360).
    smoothing_sigma_deg : float
        Gaussian smoothing sigma in degrees.  0 disables smoothing.

    Returns
    -------
    tuning_curve : (n_bins,) float
        Mean signal per angular bin.  Bins with zero occupancy are NaN.
    bin_centers_deg : (n_bins,) float
        Centre of each angular bin in degrees.
    """
    signal = np.asarray(signal, dtype=np.float64)
    hd_deg = np.asarray(hd_deg, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    if signal.shape != hd_deg.shape or signal.shape != mask.shape:
        raise ValueError(
            "signal, hd_deg, and mask must have the same shape; "
            f"got {signal.shape}, {hd_deg.shape}, {mask.shape}"
        )

    hd_mod = np.mod(hd_deg, 360.0)

    bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
    bin_centers_deg = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Digitize into bins (1-indexed; clip to [1, n_bins])
    bin_idx = np.digitize(hd_mod[mask], bin_edges) - 1
    # Handle edge case: value == 360.0 maps to bin n_bins; wrap to 0
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    signal_sum = np.zeros(n_bins, dtype=np.float64)
    occupancy = np.zeros(n_bins, dtype=np.float64)

    np.add.at(signal_sum, bin_idx, signal[mask])
    np.add.at(occupancy, bin_idx, 1.0)

    tuning_curve = np.full(n_bins, np.nan, dtype=np.float64)
    occupied = occupancy > 0
    tuning_curve[occupied] = signal_sum[occupied] / occupancy[occupied]

    if smoothing_sigma_deg > 0:
        tuning_curve = _circular_smooth_1d(
            tuning_curve, smoothing_sigma_deg, bin_width_deg=360.0 / n_bins
        )

    return tuning_curve, bin_centers_deg


def _circular_smooth_1d(
    arr: np.ndarray,
    sigma_deg: float,
    bin_width_deg: float,
) -> np.ndarray:
    """Apply 1-D Gaussian smoothing with circular (wrap) boundary.

    NaN bins are temporarily replaced with 0 for smoothing, then restored.
    The smoothing is normalised so that NaN neighbours do not dilute the
    estimate — effectively a Nadaraya–Watson style correction.
    """
    sigma_bins = sigma_deg / bin_width_deg
    nan_mask = np.isnan(arr)
    arr_filled = np.where(nan_mask, 0.0, arr)
    weight = np.where(nan_mask, 0.0, 1.0)

    # Pad by wrapping for circular boundary
    pad = int(np.ceil(3 * sigma_bins))
    arr_padded = np.concatenate([arr_filled[-pad:], arr_filled, arr_filled[:pad]])
    wgt_padded = np.concatenate([weight[-pad:], weight, weight[:pad]])

    smoothed_arr = gaussian_filter1d(arr_padded, sigma=sigma_bins, mode="nearest")
    smoothed_wgt = gaussian_filter1d(wgt_padded, sigma=sigma_bins, mode="nearest")

    # Trim padding
    smoothed_arr = smoothed_arr[pad : pad + len(arr)]
    smoothed_wgt = smoothed_wgt[pad : pad + len(arr)]

    result = np.full_like(arr, np.nan)
    valid = smoothed_wgt > 1e-12
    result[valid] = smoothed_arr[valid] / smoothed_wgt[valid]
    # Re-NaN any bin that was originally NaN
    result[nan_mask] = np.nan
    return result


def mean_vector_length(
    tuning_curve: np.ndarray,
    bin_centers_deg: np.ndarray,
) -> float:
    """Compute the mean vector length (MVL) of a tuning curve.

    MVL = |sum(r_i * exp(j * theta_i))| / sum(r_i)

    Parameters
    ----------
    tuning_curve : (n_bins,) float
        Tuning curve values (may contain NaN).
    bin_centers_deg : (n_bins,) float
        Bin centres in degrees.

    Returns
    -------
    float
        MVL in [0, 1].  Returns 0.0 if the sum of rates is zero.
    """
    tc = np.where(np.isnan(tuning_curve), 0.0, tuning_curve)
    theta = np.deg2rad(bin_centers_deg)
    r_sum = np.sum(tc)
    if r_sum == 0.0:
        return 0.0
    z = np.sum(tc * np.exp(1j * theta))
    return float(np.abs(z) / r_sum)


def preferred_direction(
    tuning_curve: np.ndarray,
    bin_centers_deg: np.ndarray,
) -> float:
    """Angle of the circular mean vector in degrees [0, 360).

    Parameters
    ----------
    tuning_curve : (n_bins,) float
    bin_centers_deg : (n_bins,) float

    Returns
    -------
    float
        Preferred direction in degrees.
    """
    tc = np.where(np.isnan(tuning_curve), 0.0, tuning_curve)
    theta = np.deg2rad(bin_centers_deg)
    z = np.sum(tc * np.exp(1j * theta))
    angle_deg = float(np.rad2deg(np.angle(z))) % 360.0
    return angle_deg


def tuning_width_fwhm(
    tuning_curve: np.ndarray,
    bin_centers_deg: np.ndarray,
) -> float:
    """Full width at half maximum of the largest peak, in degrees.

    Parameters
    ----------
    tuning_curve : (n_bins,) float
    bin_centers_deg : (n_bins,) float

    Returns
    -------
    float
        FWHM in degrees.  Returns NaN if the tuning curve is all NaN.
    """
    tc = tuning_curve.copy()
    if np.all(np.isnan(tc)):
        return float("nan")

    # Replace NaN with the minimum valid value for peak-finding purposes
    valid = ~np.isnan(tc)
    min_val = np.nanmin(tc)
    tc_filled = np.where(valid, tc, min_val)

    peak_val = np.nanmax(tc_filled)
    trough_val = np.nanmin(tc_filled)
    half_height = trough_val + 0.5 * (peak_val - trough_val)

    # Find the peak bin
    peak_idx = int(np.argmax(tc_filled))
    n = len(tc_filled)
    bin_width = 360.0 / n

    # Walk outwards from peak in both directions (circular) to find
    # where the curve crosses the half-height level.
    def _walk(direction: int) -> float:
        """Return distance in degrees from peak to half-height crossing."""
        for step in range(1, n):
            idx = (peak_idx + direction * step) % n
            if tc_filled[idx] <= half_height:
                # Linear interpolation between this bin and the previous
                prev_idx = (idx - direction) % n
                v_prev = tc_filled[prev_idx]
                v_curr = tc_filled[idx]
                dv = v_prev - v_curr
                if dv == 0:
                    frac = 0.5
                else:
                    frac = (v_prev - half_height) / dv
                return (step - 1 + frac) * bin_width
        return n / 2.0 * bin_width  # full half circle

    left = _walk(-1)
    right = _walk(+1)
    return float(left + right)


def peak_to_trough_ratio(tuning_curve: np.ndarray) -> float:
    """Ratio of maximum to minimum tuning curve value (excluding NaN).

    Returns NaN if the minimum is zero.
    """
    tc = tuning_curve[~np.isnan(tuning_curve)]
    if len(tc) == 0:
        return float("nan")
    mn = float(np.min(tc))
    if mn == 0.0:
        return float("nan")
    return float(np.max(tc) / mn)


# ---------------------------------------------------------------------------
# Place tuning
# ---------------------------------------------------------------------------


def compute_place_rate_map(
    signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    bin_size: float = 2.5,
    smoothing_sigma: float = 3.0,
    min_occupancy_s: float = 0.5,
    fps: float = 9.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D occupancy-normalised place rate map.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal.
    x, y : (n_frames,) float
        Position in cm.
    mask : (n_frames,) bool
        Valid frames.
    bin_size : float
        Spatial bin width in cm.
    smoothing_sigma : float
        Gaussian smoothing sigma in bins.  0 disables smoothing.
    min_occupancy_s : float
        Minimum occupancy in seconds; bins below this are set to NaN.
    fps : float
        Sampling rate (frames per second), used to convert frame counts
        to seconds for the occupancy threshold.

    Returns
    -------
    rate_map : (ny, nx) float
        Mean signal per spatial bin.
    occupancy_map : (ny, nx) float
        Seconds spent in each bin.
    bin_edges_x : (nx+1,) float
    bin_edges_y : (ny+1,) float
    """
    signal = np.asarray(signal, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    xm, ym, sm = x[mask], y[mask], signal[mask]

    # Build bin edges — ensure at least one bin even for degenerate ranges
    x_min, x_max = np.min(xm), np.max(xm)
    y_min, y_max = np.min(ym), np.max(ym)
    # Pad by half a bin if range is zero so we get exactly one bin
    if x_max - x_min < bin_size:
        x_max = x_min + bin_size
    if y_max - y_min < bin_size:
        y_max = y_min + bin_size
    bin_edges_x = np.arange(x_min, x_max + bin_size, bin_size)
    bin_edges_y = np.arange(y_min, y_max + bin_size, bin_size)

    nx = len(bin_edges_x) - 1
    ny = len(bin_edges_y) - 1

    # Digitise positions
    xi = np.clip(np.digitize(xm, bin_edges_x) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(ym, bin_edges_y) - 1, 0, ny - 1)

    signal_sum = np.zeros((ny, nx), dtype=np.float64)
    occ_frames = np.zeros((ny, nx), dtype=np.float64)

    np.add.at(signal_sum, (yi, xi), sm)
    np.add.at(occ_frames, (yi, xi), 1.0)

    occupancy_map = occ_frames / fps  # seconds

    rate_map = np.full((ny, nx), np.nan, dtype=np.float64)
    occ_ok = occupancy_map >= min_occupancy_s
    rate_map[occ_ok] = signal_sum[occ_ok] / occ_frames[occ_ok]

    if smoothing_sigma > 0:
        rate_map = _smooth_rate_map(rate_map, smoothing_sigma)

    return rate_map, occupancy_map, bin_edges_x, bin_edges_y


def _smooth_rate_map(rate_map: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a 2-D rate map, handling NaN bins correctly.

    Uses Nadaraya-Watson normalisation so NaN neighbours do not bias
    the estimate.
    """
    nan_mask = np.isnan(rate_map)
    filled = np.where(nan_mask, 0.0, rate_map)
    weight = np.where(nan_mask, 0.0, 1.0)

    smoothed = gaussian_filter(filled, sigma=sigma, mode="constant", cval=0.0)
    smoothed_w = gaussian_filter(weight, sigma=sigma, mode="constant", cval=0.0)

    result = np.full_like(rate_map, np.nan)
    valid = smoothed_w > 1e-12
    result[valid] = smoothed[valid] / smoothed_w[valid]
    result[nan_mask] = np.nan
    return result


def spatial_information(
    rate_map: np.ndarray,
    occupancy_map: np.ndarray,
) -> float:
    """Skaggs spatial information (bits per event).

    SI = sum_i  p_i * (r_i / r_mean) * log2(r_i / r_mean)

    where p_i = occupancy_i / total_occupancy, r_i = rate in bin i.

    Parameters
    ----------
    rate_map : (ny, nx) float
    occupancy_map : (ny, nx) float — seconds in each bin.

    Returns
    -------
    float
        Spatial information in bits/event.
    """
    valid = (~np.isnan(rate_map)) & (~np.isnan(occupancy_map)) & (occupancy_map > 0)
    r = rate_map[valid]
    occ = occupancy_map[valid]

    total_occ = np.sum(occ)
    if total_occ == 0:
        return 0.0

    p = occ / total_occ
    r_mean = np.sum(p * r)
    if r_mean <= 0:
        return 0.0

    ratio = r / r_mean
    # Avoid log2(0); bins with zero rate contribute 0 to SI
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p * ratio * np.log2(ratio)
    terms = np.where(np.isfinite(terms), terms, 0.0)
    return float(np.sum(terms))


def spatial_coherence(rate_map: np.ndarray) -> float:
    """Spatial coherence: Pearson r between each bin and its 8-neighbour mean.

    Parameters
    ----------
    rate_map : (ny, nx) float

    Returns
    -------
    float
        Pearson correlation coefficient.  Returns NaN if fewer than 3
        valid bins exist.
    """
    ny, nx = rate_map.shape
    values = []
    neighbour_means = []

    for iy in range(ny):
        for ix in range(nx):
            if np.isnan(rate_map[iy, ix]):
                continue
            neighbours = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    jy, jx = iy + dy, ix + dx
                    if 0 <= jy < ny and 0 <= jx < nx and not np.isnan(rate_map[jy, jx]):
                        neighbours.append(rate_map[jy, jx])
            if len(neighbours) == 0:
                continue
            values.append(rate_map[iy, ix])
            neighbour_means.append(np.mean(neighbours))

    if len(values) < 3:
        return float("nan")

    return float(np.corrcoef(values, neighbour_means)[0, 1])


def spatial_sparsity(
    rate_map: np.ndarray,
    occupancy_map: np.ndarray,
) -> float:
    """Spatial sparsity of a rate map.

    sparsity = (sum(p_i * r_i))^2 / sum(p_i * r_i^2)

    Low values indicate sparse (place-like) firing.

    Parameters
    ----------
    rate_map : (ny, nx) float
    occupancy_map : (ny, nx) float

    Returns
    -------
    float
        Sparsity in [0, 1].  Returns NaN if denominator is zero.
    """
    valid = (~np.isnan(rate_map)) & (~np.isnan(occupancy_map)) & (occupancy_map > 0)
    r = rate_map[valid]
    occ = occupancy_map[valid]
    total_occ = np.sum(occ)
    if total_occ == 0:
        return float("nan")

    p = occ / total_occ
    numerator = np.sum(p * r) ** 2
    denominator = np.sum(p * r**2)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)
