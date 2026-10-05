"""Pose-tracking quality summary for one session.

Works on raw tracker output (pixel coordinates and per-keypoint likelihood,
before any filtering or interpolation), so it measures the tracker, not the
kinematics stage. Anatomical checks reuse :mod:`hm2p.pose.quality`.

DLC 3.x (PyTorch) likelihoods are not calibrated probabilities and typically
sit around 0.1-0.5, so no fixed likelihood cut-off is used to score a
session; the report shows distributions and the per-keypoint 25th
percentile, which is the cut the kinematics stage applies by default
(``confidence_threshold = "quantile:0.25"``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from hm2p.pose.quality import (
    body_length_consistency,
    detect_anterior_posterior_violations,
    detect_ear_distance_outliers,
    detect_ear_swaps,
    detect_jumps,
)
from hm2p.qc.common import bin_reduce, fnum, hist, quantiles, resample_bool, rounded

BODYPARTS = (
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
)
HEAD_KEYPOINTS = ("nose_tip", "left_ear", "right_ear", "head_midpoint")
ALIASES = {"implant_base_rear": "head_midpoint", "nose": "nose_tip"}
MAX_SPEED_MM_S = 1500.0  # 150 cm/s; faster frame-to-frame motion is treated as a jump
DEFAULT_JUMP_PX = 50.0
TIMELINE_BIN_S = 10.0
CUT_QUANTILE = 0.25

KeypointData = dict[str, dict[str, npt.NDArray[np.floating]]]


def normalise_names(kp: KeypointData) -> KeypointData:
    """Map legacy keypoint names (``implant_base_rear``, ``nose``) to current ones."""
    out: KeypointData = {}
    for name, d in kp.items():
        new = ALIASES.get(name, name)
        if new not in out:
            out[new] = d
    return out


def keypoints_from_dataset(ds: object) -> KeypointData:
    """Keypoint dict from a movement pose Dataset (dims time, space, keypoints, individuals).

    With several individuals (multi-animal DLC output), each frame takes the
    individual with the highest mean confidence across keypoints.
    """
    pos = ds["position"].transpose("time", "individuals", "keypoints", "space").values  # type: ignore[index]
    conf = ds["confidence"].transpose("time", "individuals", "keypoints").values  # type: ignore[index]
    names = [str(k) for k in ds["keypoints"].values]  # type: ignore[index]
    if pos.shape[1] > 1:
        with np.errstate(all="ignore"):
            best = np.nanargmax(np.nan_to_num(np.nanmean(conf, axis=2), nan=-1.0), axis=1)
    else:
        best = np.zeros(pos.shape[0], dtype=np.int64)
    t = np.arange(pos.shape[0])
    p = pos[t, best]  # (time, keypoints, space)
    c = conf[t, best]  # (time, keypoints)
    return {
        n: {
            "x": p[:, k, 0].astype(np.float64),
            "y": p[:, k, 1].astype(np.float64),
            "likelihood": c[:, k].astype(np.float64),
        }
        for k, n in enumerate(names)
    }


def jump_threshold_px(fps: float, mm_per_px: float | None) -> float:
    """Per-frame displacement (px) above which a keypoint move counts as a jump."""
    if mm_per_px is None or mm_per_px <= 0 or fps <= 0:
        return DEFAULT_JUMP_PX
    return MAX_SPEED_MM_S / fps / mm_per_px


def frozen_mask(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    window: int,
    max_range_px: float = 0.5,
) -> np.ndarray:
    """Frames inside a window of *window* frames where the keypoint spans < *max_range_px*.

    Vectorised equivalent of :func:`hm2p.pose.quality.detect_frozen_keypoint`
    (that function loops over every window, which is slow at 30-100 fps for a
    full session). Windows containing NaN are not counted as frozen.
    """
    n = len(x)
    out = np.zeros(n, dtype=bool)
    if n < window or window < 2:
        return out
    wx = sliding_window_view(np.asarray(x, dtype=np.float64), window)
    wy = sliding_window_view(np.asarray(y, dtype=np.float64), window)
    with np.errstate(invalid="ignore"):
        span = np.hypot(wx.max(axis=1) - wx.min(axis=1), wy.max(axis=1) - wy.min(axis=1))
    starts = np.flatnonzero(span < max_range_px)
    if starts.size:
        # mark [start, start + window) for each frozen window via a difference array
        diff = np.zeros(n + 1, dtype=np.int64)
        np.add.at(diff, starts, 1)
        np.add.at(diff, starts + window, -1)
        out = np.cumsum(diff[:n]) > 0
    return out


def _bp_summary(
    d: dict[str, np.ndarray], light: np.ndarray | None, jump_px: float, frozen_win: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    x = np.asarray(d["x"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.float64)
    lik = np.asarray(d["likelihood"], dtype=np.float64)
    jumps = detect_jumps(x, y, threshold_px=jump_px)
    frozen = frozen_mask(x, y, frozen_win)
    cut = float(np.nanquantile(lik, CUT_QUANTILE)) if np.isfinite(lik).any() else np.nan
    s = {
        "lik_hist": hist(lik, 0.0, 1.0, 50),
        "lik_q": quantiles(lik),
        "cut_q25": fnum(cut),
        "frac_nan": fnum(np.mean(~np.isfinite(x) | ~np.isfinite(y))),
        "jump_frac": fnum(jumps.mean()) if jumps.size else None,
        "frozen_frac": fnum(frozen.mean()) if frozen.size else None,
    }
    if light is not None and light.any() and (~light).any():
        s["lik_median_light"] = fnum(np.nanmedian(lik[light]))
        s["lik_median_dark"] = fnum(np.nanmedian(lik[~light]))
    return s, jumps, lik


def summarise_tracking(
    kp: KeypointData,
    fps: float,
    mm_per_px: float | None = None,
    light_on: npt.ArrayLike | None = None,
    bin_s: float = TIMELINE_BIN_S,
) -> dict:
    """JSON-ready tracking-quality summary of one session.

    Parameters
    ----------
    kp
        ``{bodypart: {"x", "y", "likelihood"}}`` raw tracker arrays (pixels).
    fps
        Tracking frame rate (Hz).
    mm_per_px
        Video scale; when None, distances stay in pixels.
    light_on
        Per-frame room-light flag; resampled by index if its length differs.
    bin_s
        Timeline bin width (s).
    """
    kp = normalise_names(kp)
    parts = [b for b in BODYPARTS if b in kp] + sorted(set(kp) - set(BODYPARTS))
    if not parts:
        raise ValueError("no keypoints")
    n = len(np.asarray(kp[parts[0]]["x"]))
    light = None if light_on is None else resample_bool(light_on, n)
    scale = mm_per_px if (mm_per_px and mm_per_px > 0) else 1.0
    unit = "mm" if (mm_per_px and mm_per_px > 0) else "px"
    jump_px = jump_threshold_px(fps, mm_per_px)
    frozen_win = max(2, round(fps))  # 1 s
    per_bp: dict[str, dict] = {}
    any_jump = np.zeros(n, dtype=bool)
    lik_rank = np.zeros(n, dtype=np.float64)
    n_head_low = np.zeros(n, dtype=np.int64)
    lik_series: dict[str, np.ndarray] = {}
    for b in parts:
        s, jumps, lik = _bp_summary(kp[b], light, jump_px, frozen_win)
        per_bp[b] = s
        any_jump |= jumps
        lik_series[b] = lik
        # per-keypoint percentile rank, so keypoints with different likelihood
        # scales contribute equally to the "worst stretches" ranking
        order = np.argsort(np.argsort(np.nan_to_num(lik, nan=-1.0)))
        lik_rank += order / max(n - 1, 1)
        if b in HEAD_KEYPOINTS and s["cut_q25"] is not None:
            n_head_low += (np.nan_to_num(lik, nan=-1.0) < s["cut_q25"]).astype(np.int64)
    lik_rank /= len(parts)

    per_bin = max(1, round(bin_s * fps))
    timeline = {
        "bin_s": bin_s,
        "t_min": rounded(np.arange(int(np.ceil(n / per_bin))) * per_bin / fps / 60.0, 3),
        "lik": {b: rounded(bin_reduce(lik_series[b], per_bin, "median"), 3) for b in parts},
        "jump_frac": rounded(bin_reduce(any_jump.astype(float), per_bin), 4),
        "head_low2_frac": rounded(bin_reduce((n_head_low >= 2).astype(float), per_bin), 4),
    }
    if light is not None:
        timeline["light"] = rounded(bin_reduce(light.astype(float), per_bin), 2)

    rank_bins = bin_reduce(lik_rank, per_bin)
    worst_idx = np.argsort(np.nan_to_num(rank_bins, nan=np.inf))[:8]
    worst = [
        {"t_min": fnum(i * per_bin / fps / 60.0, 2), "rank": fnum(rank_bins[i], 3)}
        for i in sorted(worst_idx.tolist())
    ]

    out: dict = {
        "n_frames": int(n),
        "fps": fnum(fps, 3),
        "duration_min": fnum(n / fps / 60.0, 2),
        "unit": unit,
        "mm_per_px": fnum(mm_per_px, 5),
        "jump_threshold": fnum(jump_px * scale, 2),
        "bodyparts": parts,
        "per_bp": per_bp,
        "timeline": timeline,
        "worst_bins": worst,
        "any_jump_frac": fnum(any_jump.mean()),
        "head_low2_frac": fnum((n_head_low >= 2).mean()),
    }
    out.update(_anatomy(kp, scale, light))
    return out


def _anatomy(kp: KeypointData, scale: float, light: np.ndarray | None) -> dict:
    """Ear distance, ear swaps, body length and anterior-posterior order checks."""
    res: dict = {}

    def xy(b: str) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(kp[b]["x"], dtype=np.float64), np.asarray(kp[b]["y"], dtype=np.float64)

    if "left_ear" in kp and "right_ear" in kp:
        lx, ly = xy("left_ear")
        rx, ry = xy("right_ear")
        e = detect_ear_distance_outliers(lx, ly, rx, ry)
        dist = e["distance"] * scale
        res["ear"] = {
            "hist": hist(dist, 0.0, 30.0 if scale != 1.0 else 120.0, 60),
            "median": fnum(e["median"] * scale, 3),
            "mad": fnum(e["mad"] * scale, 3),
            "outlier_frac": fnum(np.mean(e["is_outlier"])),
        }
        for a, p in (
            ("nose_tip", "neck"),
            ("nose_tip", "head_midpoint"),
            ("head_midpoint", "neck"),
        ):
            if a in kp and p in kp:
                ax, ay = xy(a)
                px, py = xy(p)
                sw = detect_ear_swaps(lx, ly, rx, ry, ax, ay, px, py)
                res["ear"]["swap_frac"] = fnum(sw["pct_swapped"])
                res["ear"]["swap_axis"] = f"{a} → {p}"
                break
    for front in ("mouse_center", "mid_back"):
        if front in kp and "tail_base" in kp:
            fx, fy = xy(front)
            tx, ty = xy("tail_base")
            bl = body_length_consistency(fx, fy, tx, ty)
            res["body_len"] = {
                "from": front,
                "hist": hist(bl["length"] * scale, 0.0, 80.0 if scale != 1.0 else 300.0, 60),
                "median": fnum(bl["median"] * scale, 3),
                "mad": fnum(bl["mad"] * scale, 3),
                "outlier_frac": fnum(np.mean(bl["is_outlier"])),
            }
            break
    order = [b for b in ("nose_tip", "head_midpoint", "neck", "mid_back", "tail_base") if b in kp]
    if len(order) >= 3:
        ap = detect_anterior_posterior_violations({b: xy(b) for b in order}, order=order)
        res["ap_order"] = {
            "order": order,
            "frac": fnum(ap["pct_violated"]),
            "per_pair": {k: int(v) for k, v in ap["violations_per_pair"].items()},
        }
    return res


def overview_row(s: dict) -> dict:
    """The few numbers shown per session in the cross-session table."""
    meds = [v["lik_q"]["q50"] for v in s["per_bp"].values() if v["lik_q"]["q50"] is not None]
    lt = [
        (v.get("lik_median_light"), v.get("lik_median_dark"))
        for v in s["per_bp"].values()
        if v.get("lik_median_light") is not None
    ]
    return {
        "median_lik": fnum(np.median(meds)) if meds else None,
        "any_jump_frac": s["any_jump_frac"],
        "head_low2_frac": s["head_low2_frac"],
        "ear_outlier_frac": s.get("ear", {}).get("outlier_frac"),
        "ear_swap_frac": s.get("ear", {}).get("swap_frac"),
        "body_outlier_frac": s.get("body_len", {}).get("outlier_frac"),
        "ap_frac": s.get("ap_order", {}).get("frac"),
        "dark_minus_light_lik": fnum(np.median([d - li for li, d in lt])) if lt else None,
    }
