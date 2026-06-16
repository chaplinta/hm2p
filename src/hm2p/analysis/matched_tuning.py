"""Occupancy- and kinematics-matched HD tuning with circular-shuffle debiasing.

This module supports the dark-enhancement confound gauntlet (hypotheses A1, A2,
C1). The "RSP HD MVL higher in dark" headline could be an artefact of how the
animal samples head direction in the dark: if dark epochs have narrower or more
peaked HD-occupancy distributions, or slower / steadier movement, the mean
vector length (MVL) of a tuning curve can rise without any change in the
underlying neural code. These functions equalise the *behavioural sampling*
between two conditions before recomputing MVL, so a surviving MVL difference
cannot be attributed to sampling differences.

Two corrections are provided and can be combined:

1. **Frame matching** (occupancy or kinematics): subsample frames so that the
   joint distribution of the matched variable(s) is equal between the two
   conditions. Done by histogram binning + per-bin subsampling to the minimum
   count, repeated over bootstrap draws and averaged.

2. **Circular-shuffle debiasing**: MVL computed from a finite, unevenly sampled
   tuning curve is positively biased — even a non-tuned cell yields MVL > 0.
   We estimate that bias by circularly shifting the neural signal relative to
   head direction (which destroys the true tuning but preserves the occupancy
   distribution and temporal autocorrelation) and subtract the mean shuffle MVL.
   ``MVL_debiased = MVL_observed - mean(MVL_shuffle)``.

The circular-shift null follows Muller & Kubie (1987); occupancy/stratified
subsampling for sampling-bias control follows the matched-sampling logic in
Hardcastle et al. (2017).

References
----------
Muller & Kubie 1987. "The effects of changes in the environment on the spatial
    firing of hippocampal complex-spike cells." J Neurosci 7(7):1951-1968.
    doi:10.1523/JNEUROSCI.07-07-01951.1987
Hardcastle et al. 2017. "A multiplexed, heterogeneous, and adaptive code for
    navigation in medial entorhinal cortex." Neuron 94(2):375-387.
    doi:10.1016/j.neuron.2017.03.025
Taube et al. 1990. "Head-direction cells recorded from the postsubiculum in
    freely moving rats." J Neurosci 10(2):420-435.
    doi:10.1523/JNEUROSCI.10-02-00420.1990
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

__all__ = [
    "occupancy_histogram",
    "match_indices_1d",
    "match_indices_2d",
    "shuffle_debiased_mvl",
    "matched_condition_mvl",
]


# ---------------------------------------------------------------------------
# Occupancy helpers
# ---------------------------------------------------------------------------


def occupancy_histogram(
    hd_deg: npt.NDArray[np.floating],
    n_bins: int = 36,
) -> npt.NDArray[np.float64]:
    """Normalised HD-occupancy histogram over [0, 360).

    Parameters
    ----------
    hd_deg : (n,) float
        Head direction in degrees (mod 360 applied internally).
    n_bins : int
        Number of angular bins.

    Returns
    -------
    (n_bins,) float
        Fraction of samples in each bin (sums to 1, or all-zero if empty).
    """
    hd = np.mod(np.asarray(hd_deg, dtype=np.float64), 360.0)
    hd = hd[np.isfinite(hd)]
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    counts, _ = np.histogram(hd, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(n_bins, dtype=np.float64)
    return counts.astype(np.float64) / total


def _digitize_circular(hd_deg: npt.NDArray[np.floating], n_bins: int) -> npt.NDArray[np.int_]:
    """Assign each HD sample to an angular bin index in [0, n_bins)."""
    hd = np.mod(np.asarray(hd_deg, dtype=np.float64), 360.0)
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    idx = np.digitize(hd, edges) - 1
    return np.clip(idx, 0, n_bins - 1)


# ---------------------------------------------------------------------------
# Stratified subsampling to equalise distributions
# ---------------------------------------------------------------------------


def match_indices_1d(
    values_a: npt.NDArray[np.floating],
    values_b: npt.NDArray[np.floating],
    n_bins: int = 36,
    circular: bool = True,
    value_range: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Subsample two index sets so a 1-D variable has a matched distribution.

    For each bin of the chosen variable, keeps ``min(count_a, count_b)`` samples
    from each condition (randomly chosen without replacement). The result is two
    index arrays (into the *original* per-condition arrays) whose binned
    distributions of the variable are identical.

    Parameters
    ----------
    values_a, values_b : 1-D float
        The variable to match (e.g. head direction) for condition A and B.
    n_bins : int
        Number of bins.
    circular : bool
        If True, bin over [0, 360) (head direction). If False, bin over
        ``value_range`` (or the pooled min/max).
    value_range : (lo, hi) or None
        Range for linear binning (ignored if ``circular``).
    rng : Generator or None
        RNG for reproducible subsampling.

    Returns
    -------
    (idx_a, idx_b)
        Positional indices into ``values_a`` / ``values_b`` to keep.
    """
    if rng is None:
        rng = np.random.default_rng()
    va = np.asarray(values_a, dtype=np.float64)
    vb = np.asarray(values_b, dtype=np.float64)

    if circular:
        bin_a = _digitize_circular(va, n_bins)
        bin_b = _digitize_circular(vb, n_bins)
    else:
        if value_range is None:
            lo = float(np.nanmin(np.concatenate([va, vb])))
            hi = float(np.nanmax(np.concatenate([va, vb])))
        else:
            lo, hi = value_range
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        bin_a = np.clip(np.digitize(va, edges) - 1, 0, n_bins - 1)
        bin_b = np.clip(np.digitize(vb, edges) - 1, 0, n_bins - 1)

    keep_a: list[int] = []
    keep_b: list[int] = []
    for b in range(n_bins):
        ia = np.where(bin_a == b)[0]
        ib = np.where(bin_b == b)[0]
        k = min(len(ia), len(ib))
        if k == 0:
            continue
        keep_a.extend(rng.choice(ia, size=k, replace=False).tolist())
        keep_b.extend(rng.choice(ib, size=k, replace=False).tolist())

    return np.asarray(sorted(keep_a), dtype=int), np.asarray(sorted(keep_b), dtype=int)


def match_indices_2d(
    xa: npt.NDArray[np.floating],
    ya: npt.NDArray[np.floating],
    xb: npt.NDArray[np.floating],
    yb: npt.NDArray[np.floating],
    n_bins: tuple[int, int] = (10, 10),
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Subsample so a joint 2-D variable (e.g. speed x |AHV|) is matched.

    Same stratified logic as :func:`match_indices_1d` but over a 2-D grid.

    Parameters
    ----------
    xa, ya : 1-D float
        Joint variable for condition A (e.g. speed, |AHV|).
    xb, yb : 1-D float
        Joint variable for condition B.
    n_bins : (nx, ny)
        Grid resolution.
    x_range, y_range : (lo, hi) or None
        Binning ranges; pooled percentiles used if None.
    rng : Generator or None

    Returns
    -------
    (idx_a, idx_b)
        Positional indices into condition A / B arrays to keep.
    """
    if rng is None:
        rng = np.random.default_rng()
    xa = np.asarray(xa, float)
    ya = np.asarray(ya, float)
    xb = np.asarray(xb, float)
    yb = np.asarray(yb, float)
    nx, ny = n_bins

    pooled_x = np.concatenate([xa, xb])
    pooled_y = np.concatenate([ya, yb])
    if x_range is None:
        x_range = (float(np.nanmin(pooled_x)), float(np.nanmax(pooled_x)))
    if y_range is None:
        y_range = (float(np.nanmin(pooled_y)), float(np.nanmax(pooled_y)))
    xe = np.linspace(
        x_range[0], x_range[1] if x_range[1] > x_range[0] else x_range[0] + 1.0, nx + 1
    )
    ye = np.linspace(
        y_range[0], y_range[1] if y_range[1] > y_range[0] else y_range[0] + 1.0, ny + 1
    )

    def _grid(x, y):
        bx = np.clip(np.digitize(x, xe) - 1, 0, nx - 1)
        by = np.clip(np.digitize(y, ye) - 1, 0, ny - 1)
        return bx * ny + by

    cell_a = _grid(xa, ya)
    cell_b = _grid(xb, yb)

    keep_a: list[int] = []
    keep_b: list[int] = []
    for c in range(nx * ny):
        ia = np.where(cell_a == c)[0]
        ib = np.where(cell_b == c)[0]
        k = min(len(ia), len(ib))
        if k == 0:
            continue
        keep_a.extend(rng.choice(ia, size=k, replace=False).tolist())
        keep_b.extend(rng.choice(ib, size=k, replace=False).tolist())

    return np.asarray(sorted(keep_a), dtype=int), np.asarray(sorted(keep_b), dtype=int)


# ---------------------------------------------------------------------------
# Shuffle-debiased MVL
# ---------------------------------------------------------------------------


def shuffle_debiased_mvl(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_] | None = None,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
    n_shuffles: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """MVL corrected for finite-sample / occupancy-shape bias.

    Computes the observed MVL of the HD tuning curve and an empirical bias from
    circularly shifting the *signal* relative to head direction. Shifting
    preserves the occupancy distribution and the signal's autocorrelation but
    destroys the true HD relationship, so the mean shuffle MVL is the expected
    MVL of an untuned cell sampled the same way.

    Returns
    -------
    dict with keys:
        ``mvl_raw`` — observed MVL.
        ``mvl_bias`` — mean shuffle MVL.
        ``mvl_debiased`` — ``mvl_raw - mvl_bias``.
        ``shuffle_dist`` — (n_shuffles,) shuffle MVLs.
    """
    if rng is None:
        rng = np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float64)
    hd_deg = np.asarray(hd_deg, dtype=np.float64)
    if mask is None:
        mask = np.ones(signal.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    tc, bc = compute_hd_tuning_curve(
        signal, hd_deg, mask, n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg
    )
    mvl_raw = mean_vector_length(tc, bc)

    n = len(signal)
    min_shift = min(200, n // 4)
    max_shift = n - min_shift
    if max_shift <= min_shift:
        min_shift, max_shift = 1, max(2, n - 1)

    shuf = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        off = int(rng.integers(min_shift, max_shift))
        rolled = np.roll(signal, off)
        tcs, bcs = compute_hd_tuning_curve(
            rolled, hd_deg, mask, n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg
        )
        shuf[i] = mean_vector_length(tcs, bcs)

    bias = float(np.mean(shuf))
    return {
        "mvl_raw": float(mvl_raw),
        "mvl_bias": bias,
        "mvl_debiased": float(mvl_raw - bias),
        "shuffle_dist": shuf,
    }


# ---------------------------------------------------------------------------
# Matched-condition MVL (the A1/A2/C1 workhorse)
# ---------------------------------------------------------------------------


def matched_condition_mvl(
    signal_a: npt.NDArray[np.floating],
    hd_a: npt.NDArray[np.floating],
    signal_b: npt.NDArray[np.floating],
    hd_b: npt.NDArray[np.floating],
    match_vars_a: tuple[npt.NDArray, ...] | None = None,
    match_vars_b: tuple[npt.NDArray, ...] | None = None,
    match: str = "none",
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
    n_boot: int = 30,
    n_shuffles: int = 100,
    debias: bool = True,
    match_n_bins=None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute MVL for two conditions under matched behavioural sampling.

    For each bootstrap draw, frames are subsampled so the matched variable(s)
    have an identical distribution across conditions A and B, then MVL is
    computed (optionally shuffle-debiased) for each condition on the *same*
    subsampled frames. Results are averaged over bootstraps.

    When ``match="none"`` no subsampling is performed and the function reduces
    to the raw (optionally debiased) per-condition MVL — used as a self-check
    that matching reproduces the raw result when disabled.

    Parameters
    ----------
    signal_a, signal_b : (na,), (nb,) float
        Neural signal for the frames of condition A and B (already masked to
        valid frames upstream).
    hd_a, hd_b : float
        Head direction for the same frames.
    match_vars_a, match_vars_b : tuple of arrays or None
        Variables to match on. For ``match="occupancy"`` this is ignored (HD is
        used). For ``match="kinematics"`` pass e.g. ``(speed, abs_ahv)``.
    match : {"none", "occupancy", "kinematics"}
    n_bins : int
        HD bins for the tuning curve.
    smoothing_sigma_deg : float
    n_boot : int
        Bootstrap subsample repetitions (1 if ``match="none"``).
    n_shuffles : int
        Circular shuffles per draw for debiasing.
    debias : bool
        Subtract the circular-shuffle bias from each MVL.
    match_n_bins : int or (nx, ny) or None
        Bin resolution for the matching variable(s). Defaults: 36 for
        occupancy, (10, 10) for kinematics.
    rng : Generator or None

    Returns
    -------
    dict with keys ``mvl_a``, ``mvl_b`` (bootstrap-mean debiased or raw MVL),
    ``mvl_a_raw``, ``mvl_b_raw`` (without debiasing), ``n_matched``
    (mean matched frame count), ``n_boot``.
    """
    if rng is None:
        rng = np.random.default_rng()
    signal_a = np.asarray(signal_a, float)
    signal_b = np.asarray(signal_b, float)
    hd_a = np.asarray(hd_a, float)
    hd_b = np.asarray(hd_b, float)

    if match == "none":
        n_boot = 1

    a_vals, b_vals = [], []
    a_raw, b_raw = [], []
    n_matched = []

    for _ in range(n_boot):
        if match == "none":
            ia = np.arange(len(signal_a))
            ib = np.arange(len(signal_b))
        elif match == "occupancy":
            nb = match_n_bins if match_n_bins is not None else n_bins
            ia, ib = match_indices_1d(hd_a, hd_b, n_bins=nb, circular=True, rng=rng)
        elif match == "kinematics":
            if match_vars_a is None or match_vars_b is None:
                raise ValueError("kinematics matching requires match_vars_a/b")
            nb = match_n_bins if match_n_bins is not None else (10, 10)
            ia, ib = match_indices_2d(
                match_vars_a[0],
                match_vars_a[1],
                match_vars_b[0],
                match_vars_b[1],
                n_bins=nb,
                rng=rng,
            )
        else:
            raise ValueError(f"unknown match mode {match!r}")

        if len(ia) < 5 or len(ib) < 5:
            continue
        n_matched.append(min(len(ia), len(ib)))

        sa, ha = signal_a[ia], hd_a[ia]
        sb, hb = signal_b[ib], hd_b[ib]

        if debias:
            ra = shuffle_debiased_mvl(
                sa,
                ha,
                n_bins=n_bins,
                smoothing_sigma_deg=smoothing_sigma_deg,
                n_shuffles=n_shuffles,
                rng=rng,
            )
            rb = shuffle_debiased_mvl(
                sb,
                hb,
                n_bins=n_bins,
                smoothing_sigma_deg=smoothing_sigma_deg,
                n_shuffles=n_shuffles,
                rng=rng,
            )
            a_vals.append(ra["mvl_debiased"])
            a_raw.append(ra["mvl_raw"])
            b_vals.append(rb["mvl_debiased"])
            b_raw.append(rb["mvl_raw"])
        else:
            tca, bca = compute_hd_tuning_curve(
                sa,
                ha,
                np.ones(len(sa), bool),
                n_bins=n_bins,
                smoothing_sigma_deg=smoothing_sigma_deg,
            )
            tcb, bcb = compute_hd_tuning_curve(
                sb,
                hb,
                np.ones(len(sb), bool),
                n_bins=n_bins,
                smoothing_sigma_deg=smoothing_sigma_deg,
            )
            ma = mean_vector_length(tca, bca)
            mb = mean_vector_length(tcb, bcb)
            a_vals.append(ma)
            a_raw.append(ma)
            b_vals.append(mb)
            b_raw.append(mb)

    if not a_vals:
        return {
            "mvl_a": np.nan,
            "mvl_b": np.nan,
            "mvl_a_raw": np.nan,
            "mvl_b_raw": np.nan,
            "n_matched": 0,
            "n_boot": 0,
        }

    return {
        "mvl_a": float(np.mean(a_vals)),
        "mvl_b": float(np.mean(b_vals)),
        "mvl_a_raw": float(np.mean(a_raw)),
        "mvl_b_raw": float(np.mean(b_raw)),
        "n_matched": float(np.mean(n_matched)) if n_matched else 0,
        "n_boot": len(a_vals),
    }
