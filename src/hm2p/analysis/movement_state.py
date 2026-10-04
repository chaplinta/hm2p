"""Movement-state and syllable coding of single cells: translation vs rotation.

Two per-cell descriptions of how activity relates to the kind of movement:

**Part 1 - movement states.** Running and turning co-occur, so they are
separated by joint behavioural state. Speed and angular head velocity (AHV)
are smoothed at imaging rate and every valid frame is assigned to at most one
state:

==============  =====================================  =====
state           definition                             code
==============  =====================================  =====
still           speed < s_lo  and |AHV| < a_lo         0
straight        speed >= s_hi and |AHV| < a_lo         1
turn            speed < s_lo  and |AHV| >= a_hi        2
runturn         speed >= s_hi and |AHV| >= a_hi        3
==============  =====================================  =====

Frames between the thresholds are unassigned (-1), as are frames in state
bouts shorter than ``min_bout_s``. Per cell, mean activity per state gives
normalised indices ``(a - b) / (a + b)``: ``run_index`` (straight vs still),
``turn_index`` (turn in place vs still), ``run_vs_turn_bias`` (straight vs
turn in place) and ``interaction_index`` (running+turning vs straight). The
same contrasts divided by the cell's standard deviation over valid frames
(``*_d``) are reported because the normalised index is unstable when mean
activity is near zero (dF/F). A graded rotation measure, ``ahv_rho_matched``,
is the Spearman correlation between |AHV| bin and mean activity within
running frames after removing the mean activity of each speed stratum, so
that the speed-AHV covariation within running does not drive it.

**Part 2 - syllables.** For keypoint-MoSeq syllables with enough frames, the
syllable's mean speed and mean |AHV| are computed. Per cell: Spearman across
syllables between mean activity and syllable mean speed (and |AHV|); mutual
information (bits) between the quantile-binned signal and syllable identity;
the same after collapsing syllables into three speed classes (syllable mean
speed tertiles: still, slow, fast); and the information about syllable
identity within the fast class only. Because the speed class is a function of
the syllable, ``I(signal; class) <= I(signal; syllable)`` for the plug-in
estimator, and ``I(signal; syllable) - I(signal; class)`` is the conditional
information about syllable identity given speed class (chain rule).

All statistics are tested against a circular shift of the signal relative to
behaviour (shifts of at least ``min_shift_s``). Mutual information is
reported raw, shuffle-debiased (observed minus null mean) and with a
one-sided p; the other statistics with a two-sided p.

References
----------
Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior
by linking point tracking to pose dynamics." Nature Methods 21:1329-1339.
doi:10.1038/s41592-024-02318-2. https://github.com/dattalab/keypoint-moseq

Skaggs WE, McNaughton BL, Gothard KM, Markus EJ. 1993. "An
information-theoretic approach to deciphering the hippocampal code."
NeurIPS 5:1030-1037. (Binned mutual-information estimator, as in
``hm2p.analysis.state_dynamics.syllable_information``.)

Panzeri S, Senatore R, Montemurro MA, Petersen RS. 2007. "Correcting for the
sampling bias problem in spike train information measures." Journal of
Neurophysiology 98:1064-1072. doi:10.1152/jn.00559.2007 (shuffle
subtraction of the plug-in bias).

Saleem AB, Ayaz A, Jeffery KJ, Harris KD, Carandini M. 2013. "Integration of
visual motion and locomotion in mouse visual cortex." Nature Neuroscience
16:1864-1869. doi:10.1038/nn.3567 (running-state modulation of cortex).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from hm2p.analysis.state_dynamics import segment_bouts

STATE_NAMES: tuple[str, ...] = ("still", "straight", "turn", "runturn")
UNASSIGNED = -1

# Session-independent defaults in physical units. The speed thresholds
# bracket the runshape running threshold (2.5 cm/s) with a dead zone so
# that "still" and "running" frames are well separated. The AHV thresholds
# are applied to AHV smoothed over ``DEFAULT_SMOOTH_S`` (signed AHV smoothed,
# then rectified); a_lo = 30 deg/s is below the slowest deliberate head turns
# and a_hi = 90 deg/s is a clear turn (a quarter revolution per second).
DEFAULT_SPEED_LO = 1.0  # cm/s
DEFAULT_SPEED_HI = 3.0  # cm/s
DEFAULT_AHV_LO = 30.0  # deg/s
DEFAULT_AHV_HI = 90.0  # deg/s
DEFAULT_SMOOTH_S = 0.5  # s, centred moving average of speed and signed AHV
DEFAULT_MIN_BOUT_S = 0.3  # s, minimum state bout length
DEFAULT_MIN_STATE_FRAMES = 30  # frames per state for a mean
DEFAULT_AHV_EDGES: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0, 135.0, 180.0, 270.0)
DEFAULT_MIN_AHV_BIN_FRAMES = 20
DEFAULT_N_SPEED_STRATA = 3
DEFAULT_MIN_SYLLABLE_FRAMES = 50
DEFAULT_N_SIGNAL_BINS = 10
DEFAULT_MIN_SHIFT_S = 10.0

SPEED_CLASS_NAMES: tuple[str, ...] = ("still", "slow", "fast")

TWO_SIDED_STATS: tuple[str, ...] = (
    "run_index",
    "turn_index",
    "run_vs_turn_bias",
    "interaction_index",
    "run_d",
    "turn_d",
    "bias_d",
    "interaction_d",
    "interaction_contrast_d",
    "ahv_rho_matched",
    "syl_speed_rho",
    "syl_ahv_rho",
)
MI_STATS: tuple[str, ...] = ("syl_mi", "class_mi", "within_fast_mi")


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def nan_smooth(x: npt.ArrayLike, window: int) -> np.ndarray:
    """Centred moving average that ignores NaN; all-NaN windows stay NaN.

    Parameters
    ----------
    x : (n,) float
    window : int
        Window length in frames; even values are increased by one so the
        window is centred. ``window <= 1`` returns a copy.

    Returns
    -------
    (n,) float
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    w = int(window)
    if w <= 1 or v.size == 0:
        return v.copy()
    if w % 2 == 0:
        w += 1
    ok = np.isfinite(v)
    kernel = np.ones(w)
    # full convolution sliced to the centre: also correct when w > len(v)
    sl = slice(w // 2, w // 2 + v.size)
    num = np.convolve(np.where(ok, v, 0.0), kernel, mode="full")[sl]
    den = np.convolve(ok.astype(np.float64), kernel, mode="full")[sl]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def smooth_kinematics(
    speed: npt.ArrayLike,
    ahv: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    smooth_s: float = DEFAULT_SMOOTH_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed speed and |AHV| at imaging rate, NaN outside valid frames.

    Signed AHV is smoothed before rectification so that alternating
    left-right jitter averages out rather than accumulating. Invalid frames
    are excluded from the moving average.

    Parameters
    ----------
    speed : (n,) float
        Speed in cm/s (``speed_cm_s`` in sync.h5).
    ahv : (n,) float
        Signed angular head velocity in deg/s (``ahv_deg_s``).
    valid : (n,) bool
        Usable behaviour frames (typically ``~bad_behav``).
    fps : float
        Imaging frame rate.
    smooth_s : float
        Moving-average window in seconds.

    Returns
    -------
    speed_s, abs_ahv_s : (n,) float
        Smoothed speed and rectified smoothed AHV; NaN where invalid.
    valid_out : (n,) bool
        ``valid`` and both smoothed values finite.
    """
    s = np.asarray(speed, dtype=np.float64).ravel()
    a = np.asarray(ahv, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel()
    if not (s.shape == a.shape == v.shape):
        raise ValueError(f"speed, ahv and valid differ in shape: {s.shape}, {a.shape}, {v.shape}")
    w = max(1, int(round(smooth_s * fps)))
    s_s = nan_smooth(np.where(v, s, np.nan), w)
    a_s = np.abs(nan_smooth(np.where(v, a, np.nan), w))
    out_valid = v & np.isfinite(s_s) & np.isfinite(a_s)
    s_s[~out_valid] = np.nan
    a_s[~out_valid] = np.nan
    return s_s, a_s, out_valid


def classify_states(
    speed_s: npt.ArrayLike,
    abs_ahv_s: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    speed_lo: float = DEFAULT_SPEED_LO,
    speed_hi: float = DEFAULT_SPEED_HI,
    ahv_lo: float = DEFAULT_AHV_LO,
    ahv_hi: float = DEFAULT_AHV_HI,
    min_bout_s: float = DEFAULT_MIN_BOUT_S,
) -> np.ndarray:
    """Movement-state code per frame (see module docstring); -1 = unassigned.

    Raises
    ------
    ValueError
        If ``speed_lo > speed_hi`` or ``ahv_lo > ahv_hi``.
    """
    if speed_lo > speed_hi:
        raise ValueError(f"speed_lo ({speed_lo}) must be <= speed_hi ({speed_hi})")
    if ahv_lo > ahv_hi:
        raise ValueError(f"ahv_lo ({ahv_lo}) must be <= ahv_hi ({ahv_hi})")
    s = np.asarray(speed_s, dtype=np.float64).ravel()
    a = np.asarray(abs_ahv_s, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel() & np.isfinite(s) & np.isfinite(a)
    slow, fast = s < speed_lo, s >= speed_hi
    straight_ahv, turning = a < ahv_lo, a >= ahv_hi
    masks = (
        v & slow & straight_ahv,
        v & fast & straight_ahv,
        v & slow & turning,
        v & fast & turning,
    )
    min_frames = max(1, int(round(min_bout_s * fps)))
    code = np.full(s.size, UNASSIGNED, dtype=np.int8)
    for k, m in enumerate(masks):
        on, off = segment_bouts(m, min_frames=min_frames)
        for a0, b0 in zip(on, off, strict=True):
            code[a0:b0] = k
    return code


def state_summary(
    code: npt.ArrayLike,
    speed_s: npt.ArrayLike,
    abs_ahv_s: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    light_on: npt.ArrayLike | None = None,
    abs_ahv_raw: npt.ArrayLike | None = None,
    speed_hi: float = DEFAULT_SPEED_HI,
    min_state_frames: int = DEFAULT_MIN_STATE_FRAMES,
) -> dict[str, Any]:
    """Per-session behaviour table: frames, bouts and kinematics per state.

    Returns frame counts, seconds, fraction of valid frames, bout count and
    median bout duration per state (also split by light condition when
    *light_on* is given), quantiles of smoothed speed and |AHV| (and of the
    raw |AHV| if given), the Spearman correlation between speed and |AHV|
    within running frames, and ``rare_states``: states with fewer than
    *min_state_frames* frames (per-cell means are NaN for these).
    """
    c = np.asarray(code).ravel()
    s = np.asarray(speed_s, dtype=np.float64).ravel()
    a = np.asarray(abs_ahv_s, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel()
    n_valid = int(v.sum())
    out: dict[str, Any] = {
        "n_frames": int(c.size),
        "n_valid_frames": n_valid,
        "frac_unassigned": float(np.mean(c[v] == UNASSIGNED)) if n_valid else np.nan,
    }
    light = None if light_on is None else np.asarray(light_on, dtype=bool).ravel()
    rare = []
    for k, name in enumerate(STATE_NAMES):
        m = c == k
        nk = int(m.sum())
        on, off = segment_bouts(m, min_frames=1)
        out[f"n_{name}"] = nk
        out[f"s_{name}"] = nk / fps
        out[f"frac_{name}"] = nk / n_valid if n_valid else np.nan
        out[f"n_bouts_{name}"] = int(on.size)
        out[f"median_bout_s_{name}"] = float(np.median(off - on) / fps) if on.size else np.nan
        if light is not None:
            out[f"n_{name}_light"] = int((m & light).sum())
            out[f"n_{name}_dark"] = int((m & ~light).sum())
        if nk < min_state_frames:
            rare.append(name)
    out["rare_states"] = ";".join(rare)
    q = (10, 25, 50, 75, 90)
    sv, av = s[v & np.isfinite(s)], a[v & np.isfinite(a)]
    for p in (50, 90):
        out[f"speed_p{p}"] = float(np.percentile(sv, p)) if sv.size else np.nan
    for p in q:
        out[f"abs_ahv_p{p}"] = float(np.percentile(av, p)) if av.size else np.nan
    if abs_ahv_raw is not None:
        r = np.asarray(abs_ahv_raw, dtype=np.float64).ravel()
        rv = r[v & np.isfinite(r)]
        for p in q:
            out[f"abs_ahv_raw_p{p}"] = float(np.percentile(rv, p)) if rv.size else np.nan
    run = v & (s >= speed_hi)
    out["rho_speed_ahv_running"] = _spearman(s[run], a[run])
    return out


# ---------------------------------------------------------------------------
# Syllables
# ---------------------------------------------------------------------------


def syllable_table(
    syllable_id: npt.ArrayLike,
    speed_s: npt.ArrayLike,
    abs_ahv_s: npt.ArrayLike,
    valid: npt.ArrayLike,
    min_frames: int = DEFAULT_MIN_SYLLABLE_FRAMES,
) -> pd.DataFrame:
    """Per-syllable frame count, mean speed, mean |AHV| and speed class.

    Syllables with at least *min_frames* valid frames are ``included``.
    Included syllables are split into three speed classes (0 still, 1 slow,
    2 fast) at the tertiles of their mean speeds (unweighted across
    syllables); with fewer than three included syllables the class is -1.
    Negative or non-finite syllable ids are missing.

    Returns
    -------
    pandas.DataFrame
        Columns ``syllable, n_frames, mean_speed, mean_abs_ahv, included,
        speed_class, speed_class_name``, sorted by syllable id.
    """
    cols = [
        "syllable",
        "n_frames",
        "mean_speed",
        "mean_abs_ahv",
        "included",
        "speed_class",
        "speed_class_name",
    ]
    syl = _syllable_codes(syllable_id)
    s = np.asarray(speed_s, dtype=np.float64).ravel()
    a = np.asarray(abs_ahv_s, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel() & (syl >= 0) & np.isfinite(s) & np.isfinite(a)
    labels = np.unique(syl[v])
    if labels.size == 0:
        return pd.DataFrame(columns=cols)
    idx = np.searchsorted(labels, syl[v])
    n = np.bincount(idx, minlength=labels.size)
    ms = np.bincount(idx, weights=s[v], minlength=labels.size) / n
    ma = np.bincount(idx, weights=a[v], minlength=labels.size) / n
    inc = n >= int(min_frames)
    cls = np.full(labels.size, -1, dtype=np.int64)
    if inc.sum() >= 3:
        t1, t2 = np.quantile(ms[inc], [1 / 3, 2 / 3])
        cls[inc] = np.digitize(ms[inc], [t1, t2], right=True)
    df = pd.DataFrame(
        {
            "syllable": labels.astype(np.int64),
            "n_frames": n.astype(np.int64),
            "mean_speed": ms,
            "mean_abs_ahv": ma,
            "included": inc,
            "speed_class": cls,
        }
    )
    df["speed_class_name"] = [SPEED_CLASS_NAMES[k] if k >= 0 else "" for k in cls]
    return df[cols]


def _syllable_codes(syllable_id: npt.ArrayLike) -> np.ndarray:
    raw = np.asarray(syllable_id, dtype=np.float64).ravel()
    return np.where(np.isfinite(raw), raw, -1).astype(np.int64)


# ---------------------------------------------------------------------------
# Information
# ---------------------------------------------------------------------------


def quantile_codes(signal: npt.ArrayLike, n_bins: int = DEFAULT_N_SIGNAL_BINS) -> np.ndarray:
    """Equal-occupancy bin index per frame (-1 for non-finite frames).

    Uses the same edges as ``state_dynamics.syllable_information`` (unique
    quantiles, so tied values such as zeros share one bin).

    Raises
    ------
    ValueError
        If ``n_bins < 2``.
    """
    if int(n_bins) < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    x = np.asarray(signal, dtype=np.float64).ravel()
    ok = np.isfinite(x)
    code = np.full(x.size, -1, dtype=np.int64)
    if not ok.any():
        return code
    edges = np.unique(np.quantile(x[ok], np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 3:
        code[ok] = 0
        return code
    code[ok] = np.clip(np.digitize(x[ok], edges[1:-1], right=False), 0, edges.size - 2)
    return code


def mutual_information_codes(
    x_code: npt.ArrayLike, y_code: npt.ArrayLike, nx: int | None = None, ny: int | None = None
) -> float:
    """Plug-in mutual information (bits) between two integer codes.

    Frames where either code is negative are ignored. Returns 0.0 when
    fewer than two frames remain.
    """
    x = np.asarray(x_code, dtype=np.int64).ravel()
    y = np.asarray(y_code, dtype=np.int64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x_code and y_code differ in shape: {x.shape}, {y.shape}")
    ok = (x >= 0) & (y >= 0)
    if ok.sum() < 2:
        return 0.0
    xs, ys = x[ok], y[ok]
    nx = int(xs.max()) + 1 if nx is None else int(nx)
    ny = int(ys.max()) + 1 if ny is None else int(ny)
    joint = np.bincount(xs * ny + ys, minlength=nx * ny).reshape(nx, ny).astype(np.float64)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px * py
    nz = joint > 0
    return max(float(np.sum(joint[nz] * np.log2(joint[nz] / denom[nz]))), 0.0)


# ---------------------------------------------------------------------------
# Session design and per-cell statistics
# ---------------------------------------------------------------------------


@dataclass
class MovementDesign:
    """Behaviour-only quantities shared by all cells of a session."""

    n: int
    fps: float
    state: np.ndarray  # (n,) int8 state code
    valid: np.ndarray  # (n,) bool behaviour-valid frames
    run_idx: np.ndarray  # frames with speed >= speed_hi (any AHV)
    run_stratum: np.ndarray  # speed stratum per running frame
    run_ahv_bin: np.ndarray  # |AHV| bin per running frame
    n_strata: int
    n_ahv_bins: int
    syl_code: np.ndarray  # (n,) index into included syllables, -1 otherwise
    syl_speed: np.ndarray  # (n_syl,)
    syl_ahv: np.ndarray  # (n_syl,)
    class_code: np.ndarray  # (n,) speed class, -1 otherwise
    fast_code: np.ndarray  # (n,) index among fast-class syllables, -1 otherwise
    n_syl: int
    n_fast: int
    min_state_frames: int = DEFAULT_MIN_STATE_FRAMES
    min_ahv_bin_frames: int = DEFAULT_MIN_AHV_BIN_FRAMES
    n_signal_bins: int = DEFAULT_N_SIGNAL_BINS


def build_design(
    speed: npt.ArrayLike,
    ahv: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    syllable_id: npt.ArrayLike | None = None,
    light_on: npt.ArrayLike | None = None,
    speed_lo: float = DEFAULT_SPEED_LO,
    speed_hi: float = DEFAULT_SPEED_HI,
    ahv_lo: float = DEFAULT_AHV_LO,
    ahv_hi: float = DEFAULT_AHV_HI,
    smooth_s: float = DEFAULT_SMOOTH_S,
    min_bout_s: float = DEFAULT_MIN_BOUT_S,
    min_state_frames: int = DEFAULT_MIN_STATE_FRAMES,
    ahv_edges: tuple[float, ...] = DEFAULT_AHV_EDGES,
    n_speed_strata: int = DEFAULT_N_SPEED_STRATA,
    min_ahv_bin_frames: int = DEFAULT_MIN_AHV_BIN_FRAMES,
    min_syllable_frames: int = DEFAULT_MIN_SYLLABLE_FRAMES,
    n_signal_bins: int = DEFAULT_N_SIGNAL_BINS,
) -> tuple[MovementDesign, dict[str, Any], pd.DataFrame]:
    """Smooth kinematics, classify states and index syllables for one session.

    *syllable_id* (keypoint-MoSeq, -1 missing) and *light_on* (frame counts
    per state split by light condition) are optional.

    Returns
    -------
    design : MovementDesign
    summary : dict
        :func:`state_summary` output plus syllable-level counts.
    syllables : pandas.DataFrame
        :func:`syllable_table` output (empty without syllables).
    """
    speed_s, ahv_s, v = smooth_kinematics(speed, ahv, valid, fps, smooth_s)
    n = speed_s.size
    state = classify_states(
        speed_s, ahv_s, v, fps, speed_lo, speed_hi, ahv_lo, ahv_hi, min_bout_s=min_bout_s
    )
    summary = state_summary(
        state,
        speed_s,
        ahv_s,
        v,
        fps,
        light_on=None if light_on is None else np.asarray(light_on, dtype=bool).ravel()[:n],
        abs_ahv_raw=np.abs(np.asarray(ahv, dtype=np.float64).ravel()),
        speed_hi=speed_hi,
        min_state_frames=min_state_frames,
    )

    run_idx = np.flatnonzero(v & (speed_s >= speed_hi))
    n_strata = max(1, int(n_speed_strata))
    if run_idx.size:
        q = np.quantile(speed_s[run_idx], np.linspace(0, 1, n_strata + 1)[1:-1])
        stratum = np.searchsorted(np.unique(q), speed_s[run_idx], side="right")
    else:
        stratum = np.zeros(0, dtype=np.int64)
    e = np.asarray(ahv_edges, dtype=np.float64)
    ahv_bin = np.clip(np.searchsorted(e, ahv_s[run_idx], side="right") - 1, 0, e.size - 1)

    syl_code = np.full(n, -1, dtype=np.int64)
    class_code = np.full(n, -1, dtype=np.int64)
    fast_code = np.full(n, -1, dtype=np.int64)
    syl_speed = syl_ahv = np.zeros(0)
    n_fast = 0
    if syllable_id is None:
        syl_df = syllable_table(np.full(n, -1), speed_s, ahv_s, v)
    else:
        raw = _syllable_codes(syllable_id)[:n]
        syl_df = syllable_table(raw, speed_s, ahv_s, v, min_frames=min_syllable_frames)
        inc = syl_df[syl_df["included"]].reset_index(drop=True)
        if len(inc):
            lab = inc["syllable"].to_numpy()
            pos = np.searchsorted(lab, raw)
            pos_c = np.clip(pos, 0, lab.size - 1)
            hit = v & (raw >= 0) & (lab[pos_c] == raw)
            syl_code[hit] = pos_c[hit]
            syl_speed = inc["mean_speed"].to_numpy(dtype=np.float64)
            syl_ahv = inc["mean_abs_ahv"].to_numpy(dtype=np.float64)
            cls = inc["speed_class"].to_numpy()
            class_code[hit] = cls[syl_code[hit]]
            fast = np.flatnonzero(cls == 2)
            n_fast = int(fast.size)
            if n_fast:
                remap = np.full(lab.size, -1, dtype=np.int64)
                remap[fast] = np.arange(n_fast)
                fast_code[hit] = remap[syl_code[hit]]
    n_syl = int(syl_speed.size)
    summary["n_syllables_total"] = int(len(syl_df))
    summary["n_syllables_included"] = n_syl
    summary["n_syllables_fast"] = n_fast
    summary["frac_frames_in_included_syllables"] = (
        float(np.mean(syl_code[v] >= 0)) if v.any() and n_syl else np.nan
    )
    summary["rho_syllable_speed_ahv"] = _spearman(syl_speed, syl_ahv)

    design = MovementDesign(
        n=n,
        fps=float(fps),
        state=state,
        valid=v,
        run_idx=run_idx,
        run_stratum=np.asarray(stratum, dtype=np.int64),
        run_ahv_bin=np.asarray(ahv_bin, dtype=np.int64),
        n_strata=n_strata,
        n_ahv_bins=int(e.size),
        syl_code=syl_code,
        syl_speed=syl_speed,
        syl_ahv=syl_ahv,
        class_code=class_code,
        fast_code=fast_code,
        n_syl=n_syl,
        n_fast=n_fast,
        min_state_frames=int(min_state_frames),
        min_ahv_bin_frames=int(min_ahv_bin_frames),
        n_signal_bins=int(n_signal_bins),
    )
    return design, summary, syl_df


def _spearman(a: npt.ArrayLike, b: npt.ArrayLike, min_n: int = 3) -> float:
    """Spearman rho on pairs finite in both; NaN when undefined."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    rx = stats.rankdata(x[ok])
    ry = stats.rankdata(y[ok])
    return float(np.corrcoef(rx, ry)[0, 1])


def _norm_index(a: float, b: float) -> float:
    """(a - b) / (a + b); NaN unless both finite and a + b > 0."""
    s = a + b
    if not (np.isfinite(a) and np.isfinite(b)) or s <= 0:
        return float("nan")
    return float((a - b) / s)


def _group_means(x: np.ndarray, code: np.ndarray, n_groups: int, min_n: int) -> np.ndarray:
    """Mean of *x* per non-negative code; NaN for groups with < *min_n* frames."""
    ok = (code >= 0) & np.isfinite(x)
    cnt = np.bincount(code[ok], minlength=n_groups)[:n_groups]
    tot = np.bincount(code[ok], weights=x[ok], minlength=n_groups)[:n_groups]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = tot / cnt
    means[cnt < max(1, min_n)] = np.nan
    return means


def observed_statistics(
    signal: npt.ArrayLike, design: MovementDesign, x_code: npt.ArrayLike | None = None
) -> dict[str, float]:
    """All per-cell statistics for one signal (no null).

    Parameters
    ----------
    signal : (n,) float
    design : MovementDesign
    x_code : (n,) int or None
        Precomputed :func:`quantile_codes` of *signal* (computed if None).
    """
    x = np.asarray(signal, dtype=np.float64).ravel()[: design.n]
    xc = quantile_codes(x, design.n_signal_bins) if x_code is None else np.asarray(x_code)
    nan = float("nan")
    m = _group_means(x, design.state, len(STATE_NAMES), design.min_state_frames)
    still, straight, turn, runturn = (float(v) for v in m)
    xv = x[design.valid & np.isfinite(x)]
    sd = float(np.std(xv)) if xv.size > 1 else nan
    sd = sd if sd > 0 else nan
    out: dict[str, float] = {
        "mean_still": still,
        "mean_straight": straight,
        "mean_turn": turn,
        "mean_runturn": runturn,
        "run_index": _norm_index(straight, still),
        "turn_index": _norm_index(turn, still),
        "run_vs_turn_bias": _norm_index(straight, turn),
        "interaction_index": _norm_index(runturn, straight),
        "run_d": (straight - still) / sd,
        "turn_d": (turn - still) / sd,
        "bias_d": (straight - turn) / sd,
        "interaction_d": (runturn - straight) / sd,
        "interaction_contrast_d": ((runturn - straight) - (turn - still)) / sd,
    }

    # graded |AHV| within running frames, speed stratum mean removed
    rho = nan
    if design.run_idx.size:
        xr = x[design.run_idx]
        smeans = _group_means(xr, design.run_stratum, design.n_strata, 1)
        resid = xr - smeans[design.run_stratum]
        bm = _group_means(resid, design.run_ahv_bin, design.n_ahv_bins, design.min_ahv_bin_frames)
        rho = _spearman(np.arange(bm.size, dtype=np.float64), bm)
    out["ahv_rho_matched"] = rho

    # syllables
    if design.n_syl:
        sm = _group_means(x, design.syl_code, design.n_syl, 1)
        out["syl_speed_rho"] = _spearman(sm, design.syl_speed)
        out["syl_ahv_rho"] = _spearman(sm, design.syl_ahv)
        nb = int(design.n_signal_bins)
        out["syl_mi"] = mutual_information_codes(xc, design.syl_code, nb, design.n_syl)
        out["class_mi"] = mutual_information_codes(xc, design.class_code, nb, 3)
        out["within_fast_mi"] = (
            mutual_information_codes(xc, design.fast_code, nb, design.n_fast)
            if design.n_fast >= 2
            else nan
        )
    else:
        for k in ("syl_speed_rho", "syl_ahv_rho", *MI_STATS):
            out[k] = nan
    return out


def _null_summary(obs: float, null: np.ndarray, two_sided: bool) -> tuple[float, float, float]:
    """(null mean, z, p) for one statistic; NaN with < 10 finite null values."""
    nk = null[np.isfinite(null)]
    nan = float("nan")
    if not np.isfinite(obs) or nk.size < 10:
        return nan, nan, nan
    mu, sd = float(np.mean(nk)), float(np.std(nk))
    z = float((obs - mu) / sd) if sd > 0 else nan
    if two_sided:
        p = float((np.sum(np.abs(nk - mu) >= abs(obs - mu)) + 1) / (nk.size + 1))
    else:
        p = float((np.sum(nk >= obs) + 1) / (nk.size + 1))
    return mu, z, p


def cell_statistics(
    signal: npt.ArrayLike,
    design: MovementDesign,
    n_shuffles: int = 200,
    min_shift_s: float = DEFAULT_MIN_SHIFT_S,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Observed per-cell statistics with circular-shift z and p.

    The signal (and its bin codes) is rolled by a random shift of at least
    *min_shift_s* relative to all behaviour. Two-sided p for indices and
    correlations; one-sided (greater) p for mutual information, which is
    also reported shuffle-debiased (``*_debiased`` = observed - null mean).
    Derived: ``class_frac_of_syl_mi`` = debiased class MI / debiased
    syllable MI (NaN unless the latter is positive) and
    ``cond_syl_mi_given_class`` = debiased syllable MI - debiased class MI.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()[: design.n]
    xc = quantile_codes(x, design.n_signal_bins)
    obs = observed_statistics(x, design, xc)
    out: dict[str, Any] = dict(obs)
    n = x.size
    min_shift = int(round(min_shift_s * design.fps))
    keys = TWO_SIDED_STATS + MI_STATS
    nulls = {k: np.full(0, np.nan) for k in keys}
    if n > 2 * min_shift and n_shuffles >= 1:
        rng = rng if rng is not None else np.random.default_rng(0)
        shifts = rng.integers(min_shift, n - min_shift, size=int(n_shuffles))
        acc: dict[str, list[float]] = {k: [] for k in keys}
        for s in shifts:
            res = observed_statistics(np.roll(x, int(s)), design, np.roll(xc, int(s)))
            for k in keys:
                acc[k].append(res[k])
        nulls = {k: np.asarray(v, dtype=np.float64) for k, v in acc.items()}
    for k in TWO_SIDED_STATS:
        _, out[f"{k}_z"], out[f"{k}_p"] = _null_summary(obs[k], nulls[k], two_sided=True)
    for k in MI_STATS:
        mu, out[f"{k}_z"], out[f"{k}_p"] = _null_summary(obs[k], nulls[k], two_sided=False)
        out[f"{k}_debiased"] = obs[k] - mu if np.isfinite(mu) else float("nan")
    syl_d, cls_d = out["syl_mi_debiased"], out["class_mi_debiased"]
    out["class_frac_of_syl_mi"] = (
        cls_d / syl_d if np.isfinite(syl_d) and np.isfinite(cls_d) and syl_d > 0 else float("nan")
    )
    out["cond_syl_mi_given_class"] = syl_d - cls_d
    out["n_syllables_used"] = int(design.n_syl)
    return out
