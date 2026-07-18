"""Extra tests for hm2p.analysis.anchoring.

Exercises the dark→light re-anchoring time course, including the
out-of-bounds window, sparse-mask, and reference-PD fallback branches.
"""

from __future__ import annotations

import numpy as np

from hm2p.analysis.anchoring import anchoring_time_course, find_transitions


def _tuned_signal(n: int, pd_deg: float = 90.0):
    """A cosine-tuned signal over a repeatedly-swept head direction."""
    hd = (np.arange(n) * 7.0) % 360.0  # sweeps the circle many times
    signal = 1.0 + np.cos(np.deg2rad(hd - pd_deg))
    return signal, hd


# ── find_transitions ────────────────────────────────────────────────


def test_find_transitions_basic() -> None:
    light = np.array([0, 0, 1, 1, 0, 0, 1], dtype=bool)
    tr = find_transitions(light)
    assert tr["dark_to_light"].tolist() == [2, 6]
    assert tr["light_to_dark"].tolist() == [4]


def test_find_transitions_none() -> None:
    light = np.ones(10, dtype=bool)
    tr = find_transitions(light)
    assert tr["dark_to_light"].size == 0
    assert tr["light_to_dark"].size == 0


# ── anchoring_time_course ───────────────────────────────────────────


def test_anchoring_no_transitions_empty() -> None:
    """All-dark → no dark→light transition → empty result."""
    n = 500
    signal, hd = _tuned_signal(n)
    mask = np.ones(n, dtype=bool)
    light = np.zeros(n, dtype=bool)
    out = anchoring_time_course(signal, hd, mask, light, fps=9.8)
    assert out["n_transitions"] == 0
    assert out["time_offsets_s"].size == 0


def test_anchoring_time_course_main_path() -> None:
    """A single dark→light transition yields a populated time course.

    The transition sits early enough that pre-transition windows run off
    the start of the recording (out-of-bounds branch), while later windows
    are valid and produce PD deviations.
    """
    n = 2000
    signal, hd = _tuned_signal(n)
    mask = np.ones(n, dtype=bool)
    light = np.zeros(n, dtype=bool)
    light[150:] = True  # one dark→light transition at frame 150
    out = anchoring_time_course(
        signal,
        hd,
        mask,
        light,
        window_frames=200,
        step_frames=40,
        pre_transition_s=10.0,
        post_transition_s=30.0,
        fps=9.8,
        n_bins=18,
    )
    assert out["n_transitions"] == 1
    assert out["time_offsets_s"].size == out["pd_deviations"].size
    # Some post-transition windows are valid → at least one finite deviation.
    assert np.isfinite(out["pd_deviations"]).any()
    # Reference PD inferred from the first light epoch.
    assert np.isfinite(out["reference_pd"])


def test_anchoring_reference_pd_fallback_zero() -> None:
    """Too few light frames for a tuning curve → reference_pd falls back to 0."""
    n = 800
    signal, hd = _tuned_signal(n)
    mask = np.ones(n, dtype=bool)
    light = np.zeros(n, dtype=bool)
    # Only a handful of light frames right at a late transition — fewer than
    # n_bins → the reference-PD tuning curve is skipped (fallback to 0.0).
    light[795:] = True
    out = anchoring_time_course(signal, hd, mask, light, fps=9.8, n_bins=36, window_frames=100)
    assert out["reference_pd"] == 0.0


def test_anchoring_sparse_mask_all_nan() -> None:
    """A mostly-False mask → every window has too few valid frames → NaN."""
    n = 2000
    signal, hd = _tuned_signal(n)
    mask = np.zeros(n, dtype=bool)
    mask[::500] = True  # far fewer than n_bins per window
    light = np.zeros(n, dtype=bool)
    light[150:] = True
    out = anchoring_time_course(
        signal,
        hd,
        mask,
        light,
        window_frames=200,
        step_frames=40,
        fps=9.8,
        n_bins=18,
        reference_pd=0.0,
    )
    assert out["n_transitions"] == 1
    assert np.all(np.isnan(out["pd_deviations"]))
