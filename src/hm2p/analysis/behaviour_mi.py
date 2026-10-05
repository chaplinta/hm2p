"""Single-cell mutual information between activity and discrete maze behaviours.

For a cell, activity is binned per frame into quantile bins (computed over the
frames used); the plug-in mutual information I(activity; label) in bits is
computed over the labelled frames. Because the plug-in estimate is biased
upward and frames are autocorrelated, each cell's MI is compared with a
circular-shift null (activity shifted relative to the labels by at least a
minimum offset): debiased MI = observed - null mean; z and one-sided p.

Conditional MI, I(activity; label | condition bin), is the frame-weighted mean
of the MI within each bin of a conditioning variable (e.g. head direction in
8 sectors). It removes the information the label shares with that variable.

Labels used by the runner: inbound vs outbound run (frames inside matched runs
into vs out of dead ends), run type, maze cell during running, and light vs
dark.

Mutual information for neural tuning: Skaggs WE, McNaughton BL, Gothard KM,
Markus EJ. 1993. "An information-theoretic approach to deciphering the
hippocampal code." Advances in Neural Information Processing Systems 5:1030-1037.
Bias of plug-in estimates and shuffle correction: Panzeri S, Senatore R,
Montemurro MA, Petersen RS. 2007. "Correcting for the sampling bias problem in
spike train information measures." J Neurophysiol 98:1064-1072.
doi:10.1152/jn.00559.2007
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def quantile_bins(x: npt.ArrayLike, n_bins: int = 4) -> np.ndarray:
    """Integer bins 0..n_bins-1 by quantiles of ``x`` (ties share a bin; NaN -> -1).

    Zero-heavy signals (sparse spikes) collapse lower quantiles into one bin.
    """
    v = np.asarray(x, dtype=float)
    out = np.full(v.shape, -1, dtype=np.int64)
    ok = np.isfinite(v)
    if not ok.any():
        return out
    edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, n_bins + 1)[1:-1]))
    out[ok] = np.searchsorted(edges, v[ok], side="right")
    return out


def plugin_mi(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Plug-in mutual information in bits between two non-negative integer label arrays.

    Pairs where either label is negative are ignored. Returns 0 for < 2 samples.
    """
    x = np.asarray(a, dtype=np.int64)
    y = np.asarray(b, dtype=np.int64)
    ok = (x >= 0) & (y >= 0)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return 0.0
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(joint, (xi, yi), 1.0)
    p = joint / joint.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / (px @ py)[nz])))


def conditional_mi(a: npt.ArrayLike, b: npt.ArrayLike, cond: npt.ArrayLike) -> float:
    """I(a; b | cond): frame-weighted mean of the MI within each cond bin (cond < 0 ignored)."""
    x, y, c = (np.asarray(v, dtype=np.int64) for v in (a, b, cond))
    ok = (x >= 0) & (y >= 0) & (c >= 0)
    total = int(ok.sum())
    if total == 0:
        return 0.0
    mi = 0.0
    for k in np.unique(c[ok]):
        sel = ok & (c == k)
        mi += sel.sum() / total * plugin_mi(x[sel], y[sel])
    return float(mi)


def cell_label_mi(
    signal: npt.ArrayLike,
    labels: npt.ArrayLike,
    shifts: npt.ArrayLike,
    cond: npt.ArrayLike | None = None,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Per-cell MI (or conditional MI) between binned activity and frame labels, with shift null.

    ``labels`` is per frame (negative = frame not used). Activity bins are
    computed per cell over the labelled frames. Returns roi, mi, null_mean,
    null_sd, mi_debiased, z, p (one-sided, observed >= null).
    """
    sig = np.atleast_2d(np.asarray(signal, dtype=float))
    lab = np.asarray(labels, dtype=np.int64)
    used = lab >= 0
    c = None if cond is None else np.asarray(cond, dtype=np.int64)
    sh = np.asarray(shifts, dtype=np.int64)
    rows: list[dict[str, Any]] = []

    def mi_of(bins: np.ndarray) -> float:
        return plugin_mi(bins, lab) if c is None else conditional_mi(bins, lab, c)

    for j in range(sig.shape[0]):
        x = sig[j]

        def binned(arr: np.ndarray) -> np.ndarray:
            b = np.full(arr.shape, -1, dtype=np.int64)
            b[used] = quantile_bins(arr[used], n_bins)
            return b

        obs = mi_of(binned(x))
        null = np.array([mi_of(binned(np.roll(x, int(s)))) for s in sh])
        sd = float(null.std()) if null.size else float("nan")
        mean = float(null.mean()) if null.size else float("nan")
        rows.append(
            {
                "roi": j,
                "mi": obs,
                "null_mean": mean,
                "null_sd": sd,
                "mi_debiased": obs - mean if null.size else float("nan"),
                "z": (obs - mean) / sd if null.size and sd > 0 else float("nan"),
                "p": float((1 + np.sum(null >= obs)) / (1 + null.size))
                if null.size
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def frames_from_windows(
    n_frames: int, starts: npt.ArrayLike, ends: npt.ArrayLike, labels: npt.ArrayLike
) -> np.ndarray:
    """Per-frame labels from labelled windows ``[start, end)``; frames outside windows are -1."""
    out = np.full(n_frames, -1, dtype=np.int64)
    for s, e, lab in zip(
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        strict=True,
    ):
        out[max(0, s) : min(n_frames, e)] = lab
    return out


def hd_sectors(hd_deg: npt.ArrayLike, n: int = 8) -> np.ndarray:
    """Head direction in ``n`` equal sectors (NaN -> -1)."""
    h = np.asarray(hd_deg, dtype=float)
    out = np.full(h.shape, -1, dtype=np.int64)
    ok = np.isfinite(h)
    out[ok] = (np.mod(h[ok], 360.0) // (360.0 / n)).astype(np.int64) % n
    return out
