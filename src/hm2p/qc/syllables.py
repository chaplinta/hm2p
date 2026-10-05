"""Behavioural-syllable (keypoint-MoSeq) quality summary for one session.

Scores the syllable sequence against the validation criteria written down in
``docs/kpms-improvement-report.md`` (median bout 300-500 ms, 20-40 syllables
covering 80 % of frames, usage entropy ratio 0.6-0.8, < 5 % single-frame
bouts) and adds per-syllable kinematic signatures, light vs dark usage and
bout-level (not frame-level) transitions.

keypoint-MoSeq: Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq:
parsing behavior by linking point tracking to pose dynamics." Nature
Methods 21:1329-1339. doi:10.1038/s41592-024-02318-2.
https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

import base64

import numpy as np
import numpy.typing as npt

from hm2p.qc.common import fnum, hist, quantiles, run_lengths

BOUT_MS_RANGE = (300.0, 500.0)
N_SYLL_80PCT_RANGE = (20, 40)
ENTROPY_RATIO_RANGE = (0.6, 0.8)
SINGLE_FRAME_FRAC_MAX = 0.05
MIN_FRAMES_FOR_STATS = 50


def usage_entropy_ratio(counts: npt.ArrayLike) -> float:
    """Shannon entropy of the usage distribution divided by its maximum (log n used)."""
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    if c.size < 2:
        return float("nan")
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(c.size))


def n_to_cover(counts: npt.ArrayLike, frac: float = 0.8) -> int:
    """Smallest number of syllables whose usage sums to at least *frac* of frames."""
    c = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    if c.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(c) / c.sum(), frac - 1e-12) + 1)


def bout_transitions(values: npt.ArrayLike) -> list[list[int]]:
    """Sparse bout-to-bout transition counts ``[[from, to, count], ...]``.

    Consecutive bouts always differ, so there is no diagonal: this avoids the
    frame-level matrix in which self-transitions dominate.
    """
    v = np.asarray(values, dtype=np.int64)
    if v.size < 2:
        return []
    pairs, counts = np.unique(np.stack([v[:-1], v[1:]], axis=1), axis=0, return_counts=True)
    return [[int(a), int(b), int(c)] for (a, b), c in zip(pairs, counts, strict=True)]


def criteria_check(s: dict) -> dict:
    """Pass/fail of each documented criterion for a session summary."""
    lo, hi = BOUT_MS_RANGE
    nlo, nhi = N_SYLL_80PCT_RANGE
    elo, ehi = ENTROPY_RATIO_RANGE

    def within(v: float | None, a: float, b: float) -> bool | None:
        return None if v is None else bool(a <= v <= b)

    return {
        "median_bout_ms": within(s["median_bout_ms"], lo, hi),
        "n_syll_80pct": within(s["n_syll_80pct"], nlo, nhi),
        "entropy_ratio": within(s["entropy_ratio"], elo, ehi),
        "single_frame_frac": None
        if s["single_frame_frac"] is None
        else bool(s["single_frame_frac"] < SINGLE_FRAME_FRAC_MAX),
    }


def summarise_syllables(
    syllable_id: npt.ArrayLike,
    fps: float,
    speed_cm_s: npt.ArrayLike | None = None,
    ahv_deg_s: npt.ArrayLike | None = None,
    light_on: npt.ArrayLike | None = None,
    bad_behav: npt.ArrayLike | None = None,
) -> dict:
    """JSON-ready syllable-quality summary of one session.

    Kinematic arrays must be frame-aligned with *syllable_id* (keypoint-MoSeq
    runs on the same pose frames as the kinematics stage); arrays of another
    length are ignored. Negative ids are treated as unassigned.
    """
    sid = np.asarray(syllable_id).astype(np.int64).ravel()
    n = sid.size
    if n == 0:
        raise ValueError("empty syllable sequence")
    assigned = sid >= 0
    ids, counts = np.unique(sid[assigned], return_counts=True)
    vals, _starts, lens = run_lengths(sid)
    keep = vals >= 0
    bvals, blens = vals[keep], lens[keep]
    bout_s = blens / fps

    def aligned(a: npt.ArrayLike | None) -> np.ndarray | None:
        if a is None:
            return None
        arr = np.asarray(a).ravel()
        return arr if arr.size == n else None

    speed = aligned(speed_cm_s)
    ahv = aligned(ahv_deg_s)
    light = aligned(light_on)
    bad = aligned(bad_behav)
    good = np.ones(n, dtype=bool) if bad is None else ~bad.astype(bool)
    lmask = None if light is None else light.astype(bool)

    per: dict[str, dict] = {}
    for k, c in zip(ids.tolist(), counts.tolist(), strict=True):
        m = (sid == k) & good
        row: dict = {
            "frac": fnum(c / n, 5),
            "n_bouts": int((bvals == k).sum()),
            "median_bout_ms": fnum(np.median(blens[bvals == k]) / fps * 1e3, 1),
        }
        if m.sum() >= MIN_FRAMES_FOR_STATS:
            if speed is not None:
                row["speed_med"] = fnum(np.nanmedian(speed[m].astype(float)), 2)
            if ahv is not None:
                row["abs_ahv_med"] = fnum(np.nanmedian(np.abs(ahv[m].astype(float))), 1)
        if lmask is not None:
            nl, nd = (good & lmask).sum(), (good & ~lmask).sum()
            row["frac_light"] = fnum(((sid == k) & good & lmask).sum() / nl, 5) if nl else None
            row["frac_dark"] = fnum(((sid == k) & good & ~lmask).sum() / nd, 5) if nd else None
        per[str(k)] = row

    order = np.argsort(counts)[::-1]
    s: dict = {
        "n_frames": int(n),
        "fps": fnum(fps, 3),
        "frac_unassigned": fnum(np.mean(~assigned)),
        "n_unique": int(ids.size),
        "n_syll_80pct": n_to_cover(counts, 0.8),
        "top5_frac": fnum(counts[order[:5]].sum() / n),
        "entropy_ratio": fnum(usage_entropy_ratio(counts), 3),
        "n_bouts": int(blens.size),
        "median_bout_ms": fnum(np.median(bout_s) * 1e3, 1) if blens.size else None,
        "bout_q": {
            k: fnum(v * 1e3 if v is not None else None, 1) for k, v in quantiles(bout_s).items()
        },
        "single_frame_frac": fnum(np.mean(blens == 1)) if blens.size else None,
        "bout_hist_log10_ms": hist(np.log10(bout_s * 1e3), 1.0, 4.5, 50),
        "per_syllable": per,
        "transitions": bout_transitions(bvals),
        "ethogram": _encode_runs(vals, lens),
    }
    s["criteria"] = criteria_check(s)
    return s


def _encode_runs(vals: np.ndarray, lens: np.ndarray) -> dict:
    """Run-length ethogram as base64 int16 ids and uint16 lengths (long runs split)."""
    v_out: list[int] = []
    l_out: list[int] = []
    for v, ln in zip(vals.tolist(), lens.tolist(), strict=True):
        while ln > 65535:
            v_out.append(v)
            l_out.append(65535)
            ln -= 65535
        v_out.append(v)
        l_out.append(ln)
    return {
        "ids": base64.b64encode(np.asarray(v_out, dtype="<i2").tobytes()).decode("ascii"),
        "lens": base64.b64encode(np.asarray(l_out, dtype="<u2").tobytes()).decode("ascii"),
        "n_runs": len(v_out),
    }


def overview_row(s: dict) -> dict:
    """The few numbers shown per session in the cross-session table."""
    return {
        "n_unique": s["n_unique"],
        "n_syll_80pct": s["n_syll_80pct"],
        "top5_frac": s["top5_frac"],
        "entropy_ratio": s["entropy_ratio"],
        "median_bout_ms": s["median_bout_ms"],
        "single_frame_frac": s["single_frame_frac"],
        "n_pass": sum(1 for v in s["criteria"].values() if v),
    }
