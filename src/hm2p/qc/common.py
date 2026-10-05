"""Small helpers shared by the QC summarisers.

Everything returned here is JSON-serialisable (lists, floats, ints, None),
so the summaries can be embedded in a standalone HTML page.
"""

from __future__ import annotations

import base64
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

I16_NAN = -32768


def finite(x: npt.ArrayLike) -> np.ndarray:
    """Finite values of *x* as a flat float64 array."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def fnum(v: Any, nd: int = 4) -> float | None:
    """Round a scalar to *nd* significant decimals; NaN/inf/None become None."""
    if v is None:
        return None
    v = float(v)
    if not np.isfinite(v):
        return None
    return round(v, nd)


def rounded(x: npt.ArrayLike, nd: int = 3) -> list[float | None]:
    """List of rounded values, NaN as None."""
    a = np.asarray(x, dtype=np.float64).ravel()
    out = np.round(a, nd)
    return [None if not np.isfinite(v) else float(v) for v in out]


def hist(x: npt.ArrayLike, lo: float, hi: float, n: int) -> dict:
    """Fixed-range histogram of the finite values of *x*.

    Values outside ``[lo, hi]`` are counted in ``below`` / ``above`` rather
    than dropped silently, so a report can show how much was off-scale.
    """
    a = finite(x)
    edges = np.linspace(lo, hi, n + 1)
    counts, _ = np.histogram(a[(a >= lo) & (a <= hi)], bins=edges)
    return {
        "lo": float(lo),
        "hi": float(hi),
        "counts": counts.astype(int).tolist(),
        "below": int((a < lo).sum()),
        "above": int((a > hi).sum()),
        "n": int(a.size),
    }


def quantiles(x: npt.ArrayLike, qs: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    """Named quantiles (``q05``, ``q50`` ...) of the finite values of *x*."""
    a = finite(x)
    keys = [f"q{round(q * 100):02d}" for q in qs]
    if a.size == 0:
        return dict.fromkeys(keys)
    vals = np.quantile(a, qs)
    return {k: fnum(v) for k, v in zip(keys, vals, strict=True)}


def circ_diff_deg(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Signed circular difference ``a - b`` in degrees, wrapped to [-180, 180)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (d + 180.0) % 360.0 - 180.0


def bin_reduce(x: npt.ArrayLike, n_per_bin: int, how: str = "mean") -> np.ndarray:
    """Reduce *x* in consecutive bins of ``n_per_bin`` samples (last partial bin kept).

    ``how`` is ``"mean"`` or ``"median"``; all-NaN bins give NaN.
    """
    if n_per_bin < 1:
        raise ValueError("n_per_bin must be >= 1")
    a = np.asarray(x, dtype=np.float64).ravel()
    n_bins = int(np.ceil(a.size / n_per_bin))
    pad = n_bins * n_per_bin - a.size
    if pad:
        a = np.concatenate([a, np.full(pad, np.nan)])
    m = a.reshape(n_bins, n_per_bin) if n_bins else a.reshape(0, n_per_bin)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if how == "mean":
            return np.nanmean(m, axis=1)
        if how == "median":
            return np.nanmedian(m, axis=1)
    raise ValueError(f"unknown reduction {how!r}")


def encode_i16(x: npt.ArrayLike) -> dict:
    """Quantise *x* to int16 and base64-encode it (compact trace storage).

    Decoding: ``value = offset + scale * int16``; the int16 value ``-32768``
    marks NaN. Precision is ``(max - min) / 65534`` of the trace range.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    ok = np.isfinite(a)
    if ok.any():
        lo, hi = float(a[ok].min()), float(a[ok].max())
    else:
        lo, hi = 0.0, 0.0
    scale = (hi - lo) / 65534.0 if hi > lo else 1.0
    q = np.full(a.size, I16_NAN, dtype="<i2")
    q[ok] = np.round((a[ok] - lo) / scale - 32767.0).astype("<i2")
    return {
        "b64": base64.b64encode(q.tobytes()).decode("ascii"),
        "offset": lo + 32767.0 * scale,
        "scale": scale,
        "n": int(a.size),
    }


def decode_i16(enc: dict) -> np.ndarray:
    """Inverse of :func:`encode_i16` (used in tests)."""
    q = np.frombuffer(base64.b64decode(enc["b64"]), dtype="<i2").astype(np.float64)
    out = enc["offset"] + enc["scale"] * q
    out[q == I16_NAN] = np.nan
    return out


def run_lengths(labels: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length encode a 1-D label sequence.

    Returns ``(values, starts, lengths)`` for each maximal run of equal values.
    """
    a = np.asarray(labels).ravel()
    if a.size == 0:
        e = np.array([], dtype=np.int64)
        return a[:0], e, e
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [a.size]]))
    return a[starts], starts.astype(np.int64), lengths.astype(np.int64)


def mask_intervals(mask: npt.ArrayLike) -> list[list[int]]:
    """``[start, stop)`` frame intervals where a boolean mask is True."""
    vals, starts, lengths = run_lengths(np.asarray(mask, dtype=bool))
    return [[int(s), int(s + n)] for v, s, n in zip(vals, starts, lengths, strict=True) if v]


def resample_bool(mask: npt.ArrayLike, n_out: int) -> np.ndarray:
    """Nearest-index resampling of a boolean mask to length *n_out*.

    Used when a per-frame flag (e.g. ``light_on`` from kinematics) has to be
    matched to a pose array of a slightly different length.
    """
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == n_out:
        return m
    if m.size == 0 or n_out == 0:
        return np.zeros(n_out, dtype=bool)
    idx = np.minimum((np.arange(n_out) * m.size / n_out).astype(np.int64), m.size - 1)
    return m[idx]
