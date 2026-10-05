"""Movement-metric (kinematics.h5) quality summary for one session.

Checks the Stage 3 outputs rather than the tracker: frame timing, gaps left
after confidence filtering and interpolation, physically implausible speed
and angular head velocity (AHV), agreement between the separate
head-direction estimators stored for QC (ears, nose-head, nose-neck,
head-neck), head vs body speed, and how far filtering moved the position
away from the raw tracker output.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

from hm2p.qc.common import (
    bin_reduce,
    circ_diff_deg,
    encode_i16,
    finite,
    fnum,
    hist,
    mask_intervals,
    quantiles,
    rounded,
)

HD_ESTIMATORS = ("hd_nose_head", "hd_nose_neck", "hd_head_neck")
MAX_PLAUSIBLE_SPEED_CM_S = 100.0
MAX_PLAUSIBLE_AHV_DEG_S = 1500.0
GAP_FACTOR = 2.0
EXCERPT_S = 120.0
EXCERPT_HZ = 15.0
TIMELINE_BIN_S = 10.0
OCC_BINS = 40


def clean_attrs(attrs: dict) -> dict:
    """Scalar HDF5 attributes as JSON-friendly Python values (others dropped)."""
    out: dict = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            out[k] = v.decode(errors="replace")
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = fnum(v, 6)
    return out


def _f(d: dict, k: str) -> np.ndarray | None:
    return None if k not in d else np.asarray(d[k], dtype=np.float64).ravel()


def _b(d: dict, k: str, n: int) -> np.ndarray:
    if k not in d:
        return np.zeros(n, dtype=bool)
    return np.asarray(d[k]).astype(bool).ravel()[:n]


def _split(
    x: np.ndarray, light: np.ndarray, good: np.ndarray, lo: float, hi: float, nb: int
) -> dict:
    return {
        "light": hist(x[good & light], lo, hi, nb),
        "dark": hist(x[good & ~light], lo, hi, nb),
    }


def summarise_movement(d: dict[str, npt.ArrayLike], attrs: dict | None = None) -> dict:
    """JSON-ready kinematics-quality summary of one session.

    Parameters
    ----------
    d
        Datasets read from ``kinematics.h5`` (``frame_times``, ``hd_deg``,
        ``speed_cm_s``, ``ahv_deg_s`` ... any missing optional key is skipped).
    attrs
        Root attributes of ``kinematics.h5``.
    """
    attrs = attrs or {}
    t = _f(d, "frame_times")
    if t is None or t.size < 3:
        raise ValueError("frame_times missing or too short")
    n = t.size
    dt = np.diff(t)
    med_dt = float(np.median(dt))
    fps = 1.0 / med_dt if med_dt > 0 else float("nan")
    light = _b(d, "light_on", n)
    bad = _b(d, "bad_behav", n)
    good = ~bad
    out: dict = {
        "n_frames": int(n),
        "fps": fnum(fps, 3),
        "duration_min": fnum((t[-1] - t[0]) / 60.0, 2),
        "timing": {
            "median_dt_ms": fnum(med_dt * 1e3, 3),
            "max_dt_ms": fnum(dt.max() * 1e3, 2),
            "n_gaps": int((dt > GAP_FACTOR * med_dt).sum()),
            "non_monotonic": int((dt <= 0).sum()),
        },
        "frac_light": fnum(light.mean()),
        "frac_bad_behav": fnum(bad.mean()),
        "bad_intervals_s": [
            [fnum(t[a] - t[0], 1), fnum(t[b - 1] - t[0], 1)] for a, b in mask_intervals(bad)
        ],
        "attrs": clean_attrs(attrs),
    }

    nan_keys = ("hd_deg", "x_mm", "y_mm", "speed_cm_s", "ahv_deg_s", "x_head_mm", "x_maze")
    out["nan_frac"] = {
        k: fnum(np.mean(~np.isfinite(np.asarray(d[k], dtype=np.float64))))
        for k in nan_keys
        if k in d
    }

    speed = _f(d, "speed_cm_s")
    if speed is not None:
        out["speed"] = {
            "hist": _split(speed, light, good, 0.0, 60.0, 60),
            "q": quantiles(speed[good]),
            "frac_implausible": fnum(np.mean(finite(speed) > MAX_PLAUSIBLE_SPEED_CM_S)),
            "max": fnum(np.nanmax(speed) if np.isfinite(speed).any() else np.nan, 2),
        }
        active = _b(d, "active", n)
        out["active"] = {
            "light": fnum(active[good & light].mean()) if (good & light).any() else None,
            "dark": fnum(active[good & ~light].mean()) if (good & ~light).any() else None,
            "threshold_cm_s": out["attrs"].get("speed_active_threshold_cm_s"),
        }
    sh = _f(d, "speed_head_cm_s")
    sb = _f(d, "speed_body_cm_s")
    if sh is not None and sb is not None:
        ok = np.isfinite(sh) & np.isfinite(sb) & good
        rho = stats.spearmanr(sh[ok], sb[ok]).statistic if ok.sum() > 10 else np.nan
        out["head_vs_body_speed"] = {
            "spearman": fnum(rho),
            "ratio_q": quantiles(sh[ok] / np.maximum(sb[ok], 0.5)),
        }

    ahv = _f(d, "ahv_deg_s")
    if ahv is not None:
        out["ahv"] = {
            "hist": _split(ahv, light, good, -800.0, 800.0, 80),
            "abs_q": quantiles(np.abs(ahv[good])),
            "frac_implausible": fnum(np.mean(np.abs(finite(ahv)) > MAX_PLAUSIBLE_AHV_DEG_S)),
        }

    hd = _f(d, "hd_deg")
    if hd is not None:
        wrapped = np.mod(hd, 360.0)
        out["hd_occupancy"] = _split(wrapped, light, good, 0.0, 360.0, 36)
        out["hd_agreement"] = _hd_agreement(d, light, good)
        conf = _f(d, "hd_confidence")
        if conf is not None:
            out["hd_confidence"] = {
                "hist": _split(conf, light, good, 0.0, 1.0, 40),
                "median_light": fnum(np.nanmedian(conf[good & light]))
                if (good & light).any()
                else None,
                "median_dark": fnum(np.nanmedian(conf[good & ~light]))
                if (good & ~light).any()
                else None,
            }

    out["raw_vs_filtered"] = _raw_vs_filtered(d, good)
    out["occupancy"] = _occupancy(d, good)
    out["timeline"] = _timeline(d, t, fps, light)
    out["excerpt"] = _excerpt(d, t, fps, light)
    return out


def _hd_agreement(d: dict, light: np.ndarray, good: np.ndarray) -> dict:
    """Circular difference of each alternative HD estimator from the ear-based HD."""
    ref = _f(d, "hd_ears")
    res: dict = {}
    if ref is None:
        return res
    for k in HD_ESTIMATORS:
        other = _f(d, k)
        if other is None:
            continue
        diff = circ_diff_deg(other, ref)
        ok = np.isfinite(diff) & good
        a = np.abs(diff[ok])
        res[k] = {
            "hist": hist(diff[ok], -180.0, 180.0, 72),
            "median_abs": fnum(np.median(a), 2) if a.size else None,
            "frac_gt45": fnum(np.mean(a > 45.0)) if a.size else None,
            "frac_gt90": fnum(np.mean(a > 90.0)) if a.size else None,
            "median_abs_light": fnum(np.median(np.abs(diff[ok & light])), 2)
            if (ok & light).any()
            else None,
            "median_abs_dark": fnum(np.median(np.abs(diff[ok & ~light])), 2)
            if (ok & ~light).any()
            else None,
        }
    fused = _f(d, "hd_deg")
    if fused is not None:
        diff = circ_diff_deg(fused, ref)
        ok = np.isfinite(diff) & good
        res["hd_deg"] = {
            "hist": hist(diff[ok], -180.0, 180.0, 72),
            "median_abs": fnum(np.median(np.abs(diff[ok])), 2) if ok.any() else None,
            "frac_gt45": fnum(np.mean(np.abs(diff[ok]) > 45.0)) if ok.any() else None,
        }
    return res


def _raw_vs_filtered(d: dict, good: np.ndarray) -> dict:
    """Distance between filtered and raw maze-coordinate body position."""
    keys = ("x_maze", "y_maze", "x_maze_raw", "y_maze_raw")
    if any(k not in d for k in keys):
        return {}
    x, y, xr, yr = (np.asarray(d[k], dtype=np.float64).ravel() for k in keys)
    dist = np.hypot(x - xr, y - yr)
    ok = np.isfinite(dist) & good
    return {
        "q": quantiles(dist[ok]),
        "frac_raw_nan": fnum(np.mean(~np.isfinite(xr))),
        "frac_filtered_nan": fnum(np.mean(~np.isfinite(x))),
        "units": "maze cells",
    }


def _occupancy(d: dict, good: np.ndarray) -> dict:
    x = _f(d, "x_mm")
    y = _f(d, "y_mm")
    if x is None or y is None:
        return {}
    ok = np.isfinite(x) & np.isfinite(y) & good
    if ok.sum() < 10:
        return {}
    xe = np.linspace(np.quantile(x[ok], 0.001), np.quantile(x[ok], 0.999), OCC_BINS + 1)
    ye = np.linspace(np.quantile(y[ok], 0.001), np.quantile(y[ok], 0.999), OCC_BINS + 1)
    h, _, _ = np.histogram2d(x[ok], y[ok], bins=[xe, ye])
    return {
        "x_edges": rounded(xe, 1),
        "y_edges": rounded(ye, 1),
        "counts": h.T.astype(int).tolist(),  # rows = y
    }


def _timeline(d: dict, t: np.ndarray, fps: float, light: np.ndarray) -> dict:
    per_bin = max(1, round(TIMELINE_BIN_S * fps))
    tl: dict = {
        "bin_s": TIMELINE_BIN_S,
        "t_min": rounded(bin_reduce((t - t[0]) / 60.0, per_bin), 3),
        "light": rounded(bin_reduce(light.astype(float), per_bin), 2),
    }
    for k, how in (("speed_cm_s", "median"), ("hd_confidence", "median")):
        v = _f(d, k)
        if v is not None:
            tl[k] = rounded(bin_reduce(v, per_bin, how), 3)
    hd = _f(d, "hd_deg")
    if hd is not None:
        tl["nan_frac"] = rounded(bin_reduce((~np.isfinite(hd)).astype(float), per_bin), 3)
    ahv = _f(d, "ahv_deg_s")
    if ahv is not None:
        tl["abs_ahv"] = rounded(bin_reduce(np.abs(ahv), per_bin, "median"), 1)
    return tl


def _excerpt(d: dict, t: np.ndarray, fps: float, light: np.ndarray) -> dict:
    """Two minutes from the middle of the session, downsampled, int16-encoded.

    The window is placed to include a light transition when one exists near
    the middle, so both conditions appear in the excerpt.
    """
    n = t.size
    win = min(n, round(EXCERPT_S * fps))
    step = max(1, round(fps / EXCERPT_HZ))
    mid = n // 2
    switches = np.flatnonzero(np.diff(light.astype(np.int8)) != 0)
    if switches.size:
        mid = int(switches[np.argmin(np.abs(switches - n // 2))])
    a = int(np.clip(mid - win // 2, 0, n - win))
    sl = slice(a, a + win, step)
    ex: dict = {"t0_s": fnum(t[a] - t[0], 2), "dt_s": fnum(step / fps, 4)}
    for k in (
        "hd_deg",
        "hd_ears",
        "hd_nose_neck",
        "speed_cm_s",
        "speed_head_cm_s",
        "ahv_deg_s",
        "hd_confidence",
    ):
        v = _f(d, k)
        if v is None:
            continue
        if k.startswith("hd_") and k != "hd_confidence":
            v = np.mod(v, 360.0)
        ex[k] = encode_i16(v[sl])
    ex["light"] = encode_i16(light[sl].astype(float))
    return ex


def overview_row(s: dict) -> dict:
    """The few numbers shown per session in the cross-session table."""
    agr = s.get("hd_agreement", {})
    return {
        "fps": s["fps"],
        "n_gaps": s["timing"]["n_gaps"],
        "frac_bad_behav": s["frac_bad_behav"],
        "hd_nan": s["nan_frac"].get("hd_deg"),
        "pos_nan": s["nan_frac"].get("x_mm"),
        "speed_q95": (s.get("speed") or {}).get("q", {}).get("q95"),
        "speed_implausible": (s.get("speed") or {}).get("frac_implausible"),
        "ahv_implausible": (s.get("ahv") or {}).get("frac_implausible"),
        "hd_nose_neck_mad": (agr.get("hd_nose_neck") or {}).get("median_abs"),
        "hd_nose_neck_gt45": (agr.get("hd_nose_neck") or {}).get("frac_gt45"),
        "active_light": (s.get("active") or {}).get("light"),
        "active_dark": (s.get("active") or {}).get("dark"),
    }
