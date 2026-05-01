"""SuperAnimal fine-tune comparison helpers (pure compute, no I/O).

Method: Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A,
Mathis MW. 2024. "SuperAnimal pretrained pose estimation models for
behavioral analysis." *Nature Communications* 15:5165.
doi:10.1038/s41467-024-48792-2.
Code: https://github.com/DeepLabCut/DeepLabCut.
Weights:
https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse.

These helpers implement the v2 plan §4.5 paired non-parametric comparison
and §4.6 promotion gate. All hypothesis tests are non-parametric per
CLAUDE.md (no t-tests, no ANOVA, no Pearson correlation). Module is
intentionally I/O-free: no boto3, no h5py, no logging side-effects.
Logging belongs in the wrapper layer (``scripts/compare_models.py``).

The matched-pair rank-biserial effect size cites:

    Kerby DS. 2014. "The simple difference formula: an approach to teaching
    nonparametric correlation." Comprehensive Psychology 3:1.
    doi:10.2466/11.IT.3.1.

Verdict-JSON contract:
    ``schema_version`` is pinned to ``"1.0"``. The JSON layer is the stable
    inter-process contract; the dataclasses are an in-memory convenience.
    The frontend reads the JSON via ``verdict_from_json`` which validates
    against the JSON-Schema bundled in
    ``tests/pose/fixtures/verdict.schema.json`` before returning.

Gate semantics (per design §3.1 and the lead-dev pre-resolutions):
    - ``zero_method="wilcox"`` for the paired Wilcoxon (drop tied pairs).
    - NaN p-value (insufficient data) is treated as a fail for that
      keypoint. ``overall_pass`` becomes False with code
      ``"insufficient_data_<keypoint>"`` in ``fail_reasons``.
    - ``no_regression`` boundary is strict ``>``: a -10% change exactly
      fails the no-regression check.
    - HD circular-error is descriptive only — it is reported in the
      verdict but does NOT factor into ``overall_pass``.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import scipy.stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Schema version for the verdict JSON contract. Bump only on breaking
#: changes (see module docstring).
VERDICT_SCHEMA_VERSION = "1.0"

#: Minimum number of non-zero paired differences before
#: :func:`paired_wilcoxon_per_keypoint` will run the test. Below this it
#: returns NaN, which the gate treats as a failure for that keypoint.
DEFAULT_MIN_PAIRS = 10

#: Project bodyparts in the canonical order. The gate predicates reference
#: these names directly. Kept here so callers can validate input arrays.
HM2P_BODYPARTS: tuple[str, ...] = (
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
)

#: Default detector bbox-rate threshold for :func:`probe_sa_detector_bbox_rate`.
#: Per design §6 pitfall #3.
DEFAULT_BBOX_RATE_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Per-keypoint paired Wilcoxon
# ---------------------------------------------------------------------------


def paired_wilcoxon_per_keypoint(
    e_baseline: np.ndarray,
    e_candidate: np.ndarray,
    *,
    alternative: str = "greater",
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> np.ndarray:
    """Per-keypoint paired Wilcoxon signed-rank test.

    H1 (``alternative="greater"``): the baseline error is greater than the
    candidate error (i.e. the candidate is the better model). For each
    keypoint column, paired differences with NaN on either side are dropped.
    The test uses ``zero_method="wilcox"`` (drop zero-diff pairs), pinned
    here to make the contract explicit and stable across SciPy versions.

    Parameters
    ----------
    e_baseline, e_candidate
        Per-frame Euclidean errors of shape ``(n_frames, n_keypoints)``.
        NaN cells are treated as missing.
    alternative
        Passed to :func:`scipy.stats.wilcoxon`. Defaults to ``"greater"``.
    min_pairs
        Minimum number of non-zero paired differences required to run the
        test. Below this threshold the function returns NaN for that
        keypoint. The gate treats NaN as a failure for that keypoint.

    Returns
    -------
    np.ndarray
        One-dimensional float64 array of length ``n_keypoints``. Entries
        are p-values in [0, 1] or NaN where ``n_pairs < min_pairs``.

    Raises
    ------
    ValueError
        If ``e_baseline`` and ``e_candidate`` are not the same shape.
    TypeError
        If either input is not a 2-D array.
    """
    if e_baseline.ndim != 2 or e_candidate.ndim != 2:
        raise TypeError(
            "paired_wilcoxon_per_keypoint requires 2-D arrays "
            f"(got shapes {e_baseline.shape} and {e_candidate.shape})"
        )
    if e_baseline.shape != e_candidate.shape:
        raise ValueError(
            f"shape mismatch: baseline {e_baseline.shape} vs candidate {e_candidate.shape}"
        )
    n_keypoints = e_baseline.shape[1]
    p_values = np.full(n_keypoints, np.nan, dtype=np.float64)
    for k in range(n_keypoints):
        b_col = e_baseline[:, k]
        c_col = e_candidate[:, k]
        valid = np.isfinite(b_col) & np.isfinite(c_col)
        b_valid = b_col[valid]
        c_valid = c_col[valid]
        # Drop zero-diff pairs to count "real" pairs the test will use.
        diff = b_valid - c_valid
        n_nonzero = int(np.count_nonzero(diff))
        if n_nonzero < min_pairs:
            continue
        try:
            res = scipy.stats.wilcoxon(
                b_valid, c_valid, alternative=alternative, zero_method="wilcox"
            )
            p_values[k] = float(res.pvalue)
        except ValueError:
            # SciPy raises on degenerate cases (all-zero diffs after
            # dropping). Treat as NaN — the gate will fail-closed.
            continue
    return p_values


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def rank_biserial_paired(e_baseline: np.ndarray, e_candidate: np.ndarray) -> float:
    """Matched-pair rank-biserial *r* (Kerby 2014).

    For paired errors ``(b_i, c_i)``, computes the differences
    ``d_i = b_i - c_i``, ranks the absolute non-zero differences, and
    returns the simple-difference formula
    ``r = (sum_pos_ranks - sum_neg_ranks) / sum_all_ranks``. The result
    lies in ``[-1, 1]``. Positive values mean the candidate is better
    (smaller error) than the baseline on average.

    NaN-cells in either array are dropped pairwise.

    References
    ----------
    Kerby DS. 2014. "The simple difference formula: an approach to teaching
    nonparametric correlation." *Comprehensive Psychology* 3:1.
    doi:10.2466/11.IT.3.1.

    Parameters
    ----------
    e_baseline, e_candidate
        One-dimensional paired arrays, or columns of a 2-D array. Must have
        the same length.

    Returns
    -------
    float
        Rank-biserial *r* in ``[-1, 1]``. Returns 0.0 when all paired
        differences are zero (no rankable pairs).

    Raises
    ------
    ValueError
        If the inputs do not have the same shape.
    """
    e_baseline = np.asarray(e_baseline, dtype=np.float64).ravel()
    e_candidate = np.asarray(e_candidate, dtype=np.float64).ravel()
    if e_baseline.shape != e_candidate.shape:
        raise ValueError(
            f"shape mismatch: baseline {e_baseline.shape} vs candidate {e_candidate.shape}"
        )
    valid = np.isfinite(e_baseline) & np.isfinite(e_candidate)
    diff = e_baseline[valid] - e_candidate[valid]
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0
    abs_ranks = scipy.stats.rankdata(np.abs(nonzero))
    pos_sum = float(abs_ranks[nonzero > 0].sum())
    neg_sum = float(abs_ranks[nonzero < 0].sum())
    total = pos_sum + neg_sum
    if total == 0.0:
        return 0.0
    return (pos_sum - neg_sum) / total


# ---------------------------------------------------------------------------
# Bootstrap median CI (percentile method)
# ---------------------------------------------------------------------------


def bootstrap_median_ci(
    x: np.ndarray,
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI on the median (no parametric assumption).

    Implements the percentile method (NOT BCa) per v2 plan §4.5. Samples
    with replacement, computes the median of each bootstrap resample, and
    reports the empirical α/2 and 1-α/2 quantiles. ``n_resamples=1``
    collapses to ``(median, median, median)``.

    Parameters
    ----------
    x
        One-dimensional array. NaN values are dropped before resampling.
    n_resamples
        Number of bootstrap resamples (default 10 000).
    ci
        Confidence level in ``(0, 1)`` (default 0.95).
    rng
        Optional ``numpy.random.Generator`` for reproducibility. When
        ``None``, a fresh default-RNG is used.

    Returns
    -------
    (median, low, high)
        Three floats: the sample median and the two percentile-CI bounds.

    Raises
    ------
    ValueError
        If ``x`` is empty after dropping NaN, or ``ci`` is outside ``(0, 1)``.
    """
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1) (got {ci})")
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("bootstrap_median_ci requires at least one finite value")
    if rng is None:
        rng = np.random.default_rng()
    sample_median = float(np.median(arr))
    if n_resamples <= 1:
        return (sample_median, sample_median, sample_median)
    n = arr.size
    # Sampling indices once is faster than per-resample sampling.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_medians = np.median(arr[idx], axis=1)
    alpha = 1.0 - ci
    lo = float(np.quantile(resampled_medians, alpha / 2.0))
    hi = float(np.quantile(resampled_medians, 1.0 - alpha / 2.0))
    return (sample_median, lo, hi)


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Return the Bonferroni-corrected α threshold for ``n_tests`` tests.

    Parameters
    ----------
    alpha
        Family-wise error rate (e.g. 0.05).
    n_tests
        Number of tests in the family.

    Returns
    -------
    float
        ``alpha / n_tests``.

    Raises
    ------
    ValueError
        If ``n_tests`` is not strictly positive.
    """
    if n_tests <= 0:
        raise ValueError(f"n_tests must be > 0 (got {n_tests})")
    return alpha / n_tests


# ---------------------------------------------------------------------------
# Per-frame Euclidean error
# ---------------------------------------------------------------------------


def per_frame_euclidean_error(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
) -> np.ndarray:
    """Per-frame Euclidean distance between predicted and ground-truth keypoints.

    Parameters
    ----------
    pred_xy, gt_xy
        Arrays of shape ``(n_frames, n_keypoints, 2)``. NaN coordinates
        propagate (the corresponding error cell is NaN).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_frames, n_keypoints)`` and dtype float64.

    Raises
    ------
    ValueError
        If the input shapes do not match or the trailing dim is not 2.
    """
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"expected (n_frames, n_keypoints, 2), got {pred.shape}")
    diff = pred - gt
    # ``np.linalg.norm`` propagates NaN naturally.
    return np.linalg.norm(diff, axis=-1)


# ---------------------------------------------------------------------------
# PCK
# ---------------------------------------------------------------------------


def pck_at(errors: np.ndarray, threshold_px: float) -> float:
    """Percentage of frames with error <= threshold (PCK@k).

    Parameters
    ----------
    errors
        One-dimensional array of per-frame errors. NaN values are excluded
        from both numerator and denominator.
    threshold_px
        PCK threshold in pixels.

    Returns
    -------
    float
        Fraction in ``[0, 1]``. Returns NaN when no finite errors are
        available (avoids division-by-zero).
    """
    arr = np.asarray(errors, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float((finite <= threshold_px).mean())


# ---------------------------------------------------------------------------
# Head-direction from ear vector
# ---------------------------------------------------------------------------


def hd_from_ear_vector(
    left_ear: np.ndarray,
    right_ear: np.ndarray,
) -> np.ndarray:
    """Head-direction angle (radians) from the perpendicular to the ear axis.

    Convention: the ear axis runs from ``left_ear`` to ``right_ear``. Head
    direction is the perpendicular pointing forward, computed as
    ``atan2(-(rx - lx), ry - ly)`` in image coordinates (y-axis pointing
    down). The result is wrapped to ``(-π, π]``. NaN coordinates propagate
    per-frame.

    This matches the existing project convention in
    :mod:`hm2p.kinematics.compute` (the ear-vector HD is reused for the
    descriptive HD-error comparison in :func:`evaluate_promotion_gate`).

    Parameters
    ----------
    left_ear, right_ear
        Arrays of shape ``(n_frames, 2)``: x in column 0, y in column 1.

    Returns
    -------
    np.ndarray
        One-dimensional float64 array of length ``n_frames``, radians.

    Raises
    ------
    ValueError
        If the inputs do not have shape ``(n, 2)`` or do not match.
    """
    le = np.asarray(left_ear, dtype=np.float64)
    re = np.asarray(right_ear, dtype=np.float64)
    if le.shape != re.shape or le.ndim != 2 or le.shape[1] != 2:
        raise ValueError(f"expected (n, 2) arrays of equal shape (got {le.shape}, {re.shape})")
    dx = re[:, 0] - le[:, 0]
    dy = re[:, 1] - le[:, 1]
    # Forward perpendicular in image coords.
    theta = np.arctan2(-dx, dy)
    # arctan2 already returns values in (-pi, pi].
    return theta


def circular_abs_error(
    theta_pred: np.ndarray,
    theta_gt: np.ndarray,
) -> np.ndarray:
    """Absolute circular error wrapped to ``[0, π]``.

    Parameters
    ----------
    theta_pred, theta_gt
        One-dimensional arrays of angles in radians (any range). NaN
        propagates.

    Returns
    -------
    np.ndarray
        Per-frame absolute angular distance in ``[0, π]``.

    Raises
    ------
    ValueError
        If the inputs do not have the same shape.
    """
    p = np.asarray(theta_pred, dtype=np.float64)
    g = np.asarray(theta_gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")
    diff = p - g
    # Wrap to (-pi, pi] then take abs, capping at pi.
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(wrapped)


# ---------------------------------------------------------------------------
# SA detector probe (pure-function predicate; the DLC call lives elsewhere)
# ---------------------------------------------------------------------------


def probe_sa_detector_bbox_rate(
    bbox_results: list[bool] | np.ndarray,
    *,
    threshold: float = DEFAULT_BBOX_RATE_THRESHOLD,
) -> tuple[bool, str]:
    """Predicate on the SA detector's bbox detection rate.

    The wrapper layer in ``scripts/run_dlc_retrain.py`` calls the DLC
    detector on N test frames and feeds a list of booleans (one per
    frame, ``True`` if the detector returned at least one bbox) to this
    function. Keeping the predicate pure makes it unit-testable without
    DLC. Per design §6 pitfall #3.

    Parameters
    ----------
    bbox_results
        Sequence of booleans or 0/1 ints, one per probed frame.
    threshold
        Minimum acceptable bbox-detection rate. Defaults to 0.90.

    Returns
    -------
    (passed, message)
        ``passed`` is True when the rate is at least ``threshold``.
        ``message`` carries a human-readable error when the probe fails;
        empty string on success.
    """
    arr = np.asarray(bbox_results, dtype=bool).ravel()
    n_frames = int(arr.size)
    if n_frames == 0:
        return (False, "no frames probed")
    n_with_bbox = int(arr.sum())
    rate = n_with_bbox / n_frames
    if rate >= threshold:
        return (True, "")
    return (
        False,
        (
            f"SA detector returned bboxes on {n_with_bbox}/{n_frames} frames "
            f"(rate={rate:.2f}); threshold={threshold:.2f}. "
            "Re-train just the detector before launching SA fine-tune."
        ),
    )


# ---------------------------------------------------------------------------
# Verdict dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeypointVerdict:
    """Per-keypoint statistics row of the verdict.

    All fields are JSON-serialisable. NaN is allowed (preserved through
    JSON via ``allow_nan=True``).
    """

    keypoint: str
    n_pairs: int
    median_baseline_px: float
    median_candidate_px: float
    pct_change_median: float
    p_value_wilcoxon: float
    rank_biserial_r: float
    bootstrap_ci_baseline: tuple[float, float, float]
    bootstrap_ci_candidate: tuple[float, float, float]
    pck_5_baseline: float
    pck_10_baseline: float
    pck_20_baseline: float
    pck_5_candidate: float
    pck_10_candidate: float
    pck_20_candidate: float
    p90_baseline: float
    p90_candidate: float
    pct_change_p90: float


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the six-conjunction promotion gate (design §3.1)."""

    alpha: float = 6.25e-3  # = 0.05 / 8
    nose_required_pct_reduction: float = 0.30
    tail_required_pct_reduction: float = 0.40
    head_p90_required_pct_reduction: float = 0.20
    rank_biserial_min: float = 0.30
    other_keypoint_max_regression_pct: float = 0.10
    other_keypoint_regression_p_max: float = 0.05


@dataclass(frozen=True)
class Verdict:
    """Full verdict — single-source-of-truth for the gate outcome."""

    schema_version: str
    baseline_id: str
    candidate_id: str
    n_frames_compared: int
    keypoints: tuple[KeypointVerdict, ...]
    hd: dict[str, float | int | None]
    gate: GateConfig
    gate_pass_per_keypoint: dict[str, dict[str, Any]]
    overall_pass: bool
    fail_reasons: tuple[str, ...]
    generated_at: str
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Verdict <-> JSON
# ---------------------------------------------------------------------------


def _verdict_to_dict(v: Verdict) -> dict[str, Any]:
    d = asdict(v)
    # asdict turns dataclass tuples into lists, but tuple semantics are
    # preserved on round-trip via verdict_from_json.
    return d


def verdict_to_json(v: Verdict, *, indent: int | None = 2) -> str:
    """Serialise a :class:`Verdict` to a JSON string.

    NaN, +Inf, -Inf are preserved (``allow_nan=True``). The contract pins
    ``schema_version`` to ``"1.0"`` (set this on the dataclass; the
    serialiser does not overwrite).

    Parameters
    ----------
    v
        Verdict to serialise.
    indent
        Pretty-print indent. ``None`` produces compact JSON.

    Returns
    -------
    str
        JSON text.
    """
    return json.dumps(_verdict_to_dict(v), indent=indent, allow_nan=True)


def _build_keypoint(d: dict[str, Any]) -> KeypointVerdict:
    required = {
        "keypoint",
        "n_pairs",
        "median_baseline_px",
        "median_candidate_px",
        "pct_change_median",
        "p_value_wilcoxon",
        "rank_biserial_r",
        "bootstrap_ci_baseline",
        "bootstrap_ci_candidate",
        "pck_5_baseline",
        "pck_10_baseline",
        "pck_20_baseline",
        "pck_5_candidate",
        "pck_10_candidate",
        "pck_20_candidate",
        "p90_baseline",
        "p90_candidate",
        "pct_change_p90",
    }
    missing = required - d.keys()
    if missing:
        raise ValueError(f"keypoint verdict missing field(s): {sorted(missing)}")
    return KeypointVerdict(
        keypoint=str(d["keypoint"]),
        n_pairs=int(d["n_pairs"]),
        median_baseline_px=float(d["median_baseline_px"]),
        median_candidate_px=float(d["median_candidate_px"]),
        pct_change_median=float(d["pct_change_median"]),
        p_value_wilcoxon=float(d["p_value_wilcoxon"]),
        rank_biserial_r=float(d["rank_biserial_r"]),
        bootstrap_ci_baseline=tuple(float(x) for x in d["bootstrap_ci_baseline"]),  # type: ignore[arg-type]
        bootstrap_ci_candidate=tuple(float(x) for x in d["bootstrap_ci_candidate"]),  # type: ignore[arg-type]
        pck_5_baseline=float(d["pck_5_baseline"]),
        pck_10_baseline=float(d["pck_10_baseline"]),
        pck_20_baseline=float(d["pck_20_baseline"]),
        pck_5_candidate=float(d["pck_5_candidate"]),
        pck_10_candidate=float(d["pck_10_candidate"]),
        pck_20_candidate=float(d["pck_20_candidate"]),
        p90_baseline=float(d["p90_baseline"]),
        p90_candidate=float(d["p90_candidate"]),
        pct_change_p90=float(d["pct_change_p90"]),
    )


def verdict_from_json(s: str) -> Verdict:
    """Deserialise a JSON string into a :class:`Verdict`.

    Validates ``schema_version`` against the supported value and raises
    on missing required fields. Future schema versions are rejected (we
    do not silently accept forward-compatible JSON).

    Parameters
    ----------
    s
        JSON text produced by :func:`verdict_to_json`.

    Returns
    -------
    Verdict

    Raises
    ------
    ValueError
        On schema mismatch or missing required field.
    """
    d = json.loads(s)
    sv = d.get("schema_version")
    if sv != VERDICT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported verdict schema_version: {sv!r} (expected {VERDICT_SCHEMA_VERSION!r})"
        )
    required = {
        "schema_version",
        "baseline_id",
        "candidate_id",
        "n_frames_compared",
        "keypoints",
        "hd",
        "gate",
        "gate_pass_per_keypoint",
        "overall_pass",
        "fail_reasons",
        "generated_at",
    }
    missing = required - d.keys()
    if missing:
        raise ValueError(f"verdict JSON missing field(s): {sorted(missing)}")
    gate_d = d["gate"]
    gate = GateConfig(
        alpha=float(gate_d.get("alpha", 6.25e-3)),
        nose_required_pct_reduction=float(gate_d.get("nose_required_pct_reduction", 0.30)),
        tail_required_pct_reduction=float(gate_d.get("tail_required_pct_reduction", 0.40)),
        head_p90_required_pct_reduction=float(gate_d.get("head_p90_required_pct_reduction", 0.20)),
        rank_biserial_min=float(gate_d.get("rank_biserial_min", 0.30)),
        other_keypoint_max_regression_pct=float(
            gate_d.get("other_keypoint_max_regression_pct", 0.10)
        ),
        other_keypoint_regression_p_max=float(gate_d.get("other_keypoint_regression_p_max", 0.05)),
    )
    keypoints = tuple(_build_keypoint(k) for k in d["keypoints"])
    return Verdict(
        schema_version=str(d["schema_version"]),
        baseline_id=str(d["baseline_id"]),
        candidate_id=str(d["candidate_id"]),
        n_frames_compared=int(d["n_frames_compared"]),
        keypoints=keypoints,
        hd=dict(d["hd"]),
        gate=gate,
        gate_pass_per_keypoint=dict(d["gate_pass_per_keypoint"]),
        overall_pass=bool(d["overall_pass"]),
        fail_reasons=tuple(str(r) for r in d["fail_reasons"]),
        generated_at=str(d["generated_at"]),
        meta=dict(d.get("meta", {})),
    )


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _pct_change(baseline: float, candidate: float) -> float:
    """Relative reduction: (baseline - candidate) / baseline.

    Returns NaN when baseline is zero or NaN.
    """
    if not math.isfinite(baseline) or baseline == 0.0:
        return float("nan")
    return (baseline - candidate) / baseline


def _build_keypoint_verdict(
    name: str,
    e_b_col: np.ndarray,
    e_c_col: np.ndarray,
    p_value: float,
    rng: np.random.Generator,
) -> KeypointVerdict:
    """Compute per-keypoint descriptive stats + bootstrap CIs."""
    valid = np.isfinite(e_b_col) & np.isfinite(e_c_col)
    b = e_b_col[valid]
    c = e_c_col[valid]
    n_pairs = int(b.size)
    if n_pairs == 0:
        return KeypointVerdict(
            keypoint=name,
            n_pairs=0,
            median_baseline_px=float("nan"),
            median_candidate_px=float("nan"),
            pct_change_median=float("nan"),
            p_value_wilcoxon=p_value,
            rank_biserial_r=0.0,
            bootstrap_ci_baseline=(float("nan"),) * 3,
            bootstrap_ci_candidate=(float("nan"),) * 3,
            pck_5_baseline=float("nan"),
            pck_10_baseline=float("nan"),
            pck_20_baseline=float("nan"),
            pck_5_candidate=float("nan"),
            pck_10_candidate=float("nan"),
            pck_20_candidate=float("nan"),
            p90_baseline=float("nan"),
            p90_candidate=float("nan"),
            pct_change_p90=float("nan"),
        )
    median_b = float(np.median(b))
    median_c = float(np.median(c))
    p90_b = float(np.quantile(b, 0.90))
    p90_c = float(np.quantile(c, 0.90))
    ci_b = bootstrap_median_ci(b, rng=rng)
    ci_c = bootstrap_median_ci(c, rng=rng)
    return KeypointVerdict(
        keypoint=name,
        n_pairs=n_pairs,
        median_baseline_px=median_b,
        median_candidate_px=median_c,
        pct_change_median=_pct_change(median_b, median_c),
        p_value_wilcoxon=p_value,
        rank_biserial_r=rank_biserial_paired(b, c),
        bootstrap_ci_baseline=ci_b,
        bootstrap_ci_candidate=ci_c,
        pck_5_baseline=pck_at(b, 5.0),
        pck_10_baseline=pck_at(b, 10.0),
        pck_20_baseline=pck_at(b, 20.0),
        pck_5_candidate=pck_at(c, 5.0),
        pck_10_candidate=pck_at(c, 10.0),
        pck_20_candidate=pck_at(c, 20.0),
        p90_baseline=p90_b,
        p90_candidate=p90_c,
        pct_change_p90=_pct_change(p90_b, p90_c),
    )


def evaluate_gate(
    keypoints: list[KeypointVerdict],
    gate: GateConfig,
) -> tuple[bool, list[str], dict[str, dict[str, Any]]]:
    """Evaluate the six-conjunction promotion gate (design §3.1).

    The gate is fail-closed on NaN p-values: a keypoint with insufficient
    data triggers an ``"insufficient_data_<keypoint>"`` failure code and
    sets ``overall_pass=False``.

    Parameters
    ----------
    keypoints
        Per-keypoint verdict rows. Must include ``nose_tip``, ``tail_base``,
        and ``head_midpoint`` for the gate to make sense.
    gate
        Threshold configuration.

    Returns
    -------
    (overall_pass, fail_reasons, gate_pass_per_keypoint)
        ``fail_reasons`` is a list of short codes (one per failed
        predicate); ``gate_pass_per_keypoint`` maps keypoint -> dict of
        per-predicate booleans for traceability.
    """
    by_name = {k.keypoint: k for k in keypoints}
    fail_reasons: list[str] = []
    per_kp: dict[str, dict[str, Any]] = {}

    nose = by_name.get("nose_tip")
    tail = by_name.get("tail_base")
    head = by_name.get("head_midpoint")

    # ---- nose_tip --------------------------------------------------------
    if nose is None:
        fail_reasons.append("missing_nose_tip")
    else:
        checks: dict[str, Any] = {}
        if math.isnan(nose.p_value_wilcoxon):
            fail_reasons.append("insufficient_data_nose_tip")
            checks["pct_reduction"] = False
            checks["p_value"] = False
            checks["rank_biserial"] = False
        else:
            pct_ok = nose.pct_change_median >= gate.nose_required_pct_reduction
            p_ok = nose.p_value_wilcoxon < gate.alpha
            r_ok = nose.rank_biserial_r >= gate.rank_biserial_min
            checks = {
                "pct_reduction": bool(pct_ok),
                "p_value": bool(p_ok),
                "rank_biserial": bool(r_ok),
            }
            if not pct_ok:
                fail_reasons.append("nose_pct_reduction")
            if not p_ok:
                fail_reasons.append("nose_significance")
            if not r_ok:
                fail_reasons.append("nose_effect_size")
        per_kp["nose_tip"] = {"pass": all(checks.values()), "checks": checks}

    # ---- tail_base -------------------------------------------------------
    if tail is None:
        fail_reasons.append("missing_tail_base")
    else:
        checks = {}
        if math.isnan(tail.p_value_wilcoxon):
            fail_reasons.append("insufficient_data_tail_base")
            checks["pct_reduction"] = False
            checks["p_value"] = False
            checks["rank_biserial"] = False
        else:
            pct_ok = tail.pct_change_median >= gate.tail_required_pct_reduction
            p_ok = tail.p_value_wilcoxon < gate.alpha
            r_ok = tail.rank_biserial_r >= gate.rank_biserial_min
            checks = {
                "pct_reduction": bool(pct_ok),
                "p_value": bool(p_ok),
                "rank_biserial": bool(r_ok),
            }
            if not pct_ok:
                fail_reasons.append("tail_pct_reduction")
            if not p_ok:
                fail_reasons.append("tail_significance")
            if not r_ok:
                fail_reasons.append("tail_effect_size")
        per_kp["tail_base"] = {"pass": all(checks.values()), "checks": checks}

    # ---- head_midpoint p90 ----------------------------------------------
    if head is None:
        fail_reasons.append("missing_head_midpoint")
    else:
        if math.isnan(head.pct_change_p90):
            fail_reasons.append("insufficient_data_head_midpoint")
            checks = {"p90_reduction": False}
        else:
            p90_ok = head.pct_change_p90 >= gate.head_p90_required_pct_reduction
            checks = {"p90_reduction": bool(p90_ok)}
            if not p90_ok:
                fail_reasons.append("head_p90_reduction")
        per_kp["head_midpoint"] = {"pass": all(checks.values()), "checks": checks}

    # ---- other keypoints — no_regression --------------------------------
    other_names = {"left_ear", "right_ear", "neck", "mid_back", "mouse_center"}
    for name in sorted(other_names):
        kp = by_name.get(name)
        if kp is None:
            # Missing isn't fatal here (the gate predicates target the
            # three named keypoints). But record it for traceability.
            per_kp[name] = {"pass": True, "checks": {"no_regression": True}}
            continue
        if math.isnan(kp.pct_change_median):
            fail_reasons.append(f"insufficient_data_{name}")
            per_kp[name] = {"pass": False, "checks": {"no_regression": False}}
            continue
        # Strict ``>`` on the no-regression boundary (design §3.1, lead-dev
        # pre-resolution #3): a -10% change exactly fails.
        within_band = kp.pct_change_median > -gate.other_keypoint_max_regression_pct
        # Significance test on the regression: only a fail if the
        # regression is also significant (p < 0.05) AND r < 0 (candidate
        # actually worse).
        if not within_band:
            # Outright regression beyond the band is a fail.
            fail_reasons.append(f"regression_{name}")
            per_kp[name] = {"pass": False, "checks": {"no_regression": False}}
            continue
        # Inside band — additionally fail if regression is significant.
        if (
            kp.pct_change_median < 0
            and not math.isnan(kp.p_value_wilcoxon)
            and kp.p_value_wilcoxon < gate.other_keypoint_regression_p_max
            and kp.rank_biserial_r < 0
        ):
            fail_reasons.append(f"regression_{name}")
            per_kp[name] = {"pass": False, "checks": {"no_regression": False}}
        else:
            per_kp[name] = {"pass": True, "checks": {"no_regression": True}}

    overall = len(fail_reasons) == 0
    return overall, fail_reasons, per_kp


def evaluate_promotion_gate(
    e_baseline: np.ndarray,
    e_candidate: np.ndarray,
    keypoint_names: list[str],
    hd_baseline_rad: np.ndarray | None,
    hd_candidate_rad: np.ndarray | None,
    hd_gt_rad: np.ndarray | None,
    *,
    baseline_id: str,
    candidate_id: str,
    gate: GateConfig | None = None,
    rng: np.random.Generator | None = None,
    meta: dict[str, Any] | None = None,
) -> Verdict:
    """Run the v2 §4.6 promotion gate end-to-end on per-frame errors.

    Parameters
    ----------
    e_baseline, e_candidate
        Per-frame Euclidean errors of shape ``(n_frames, n_keypoints)``.
    keypoint_names
        Bodypart names corresponding to the columns of ``e_baseline``.
    hd_baseline_rad, hd_candidate_rad, hd_gt_rad
        Optional per-frame HD predictions and GT (radians). When all three
        are provided the verdict's ``hd`` field is populated; otherwise it
        carries None placeholders. HD is descriptive only (does NOT
        factor into ``overall_pass``).
    baseline_id, candidate_id
        Champion-id strings for traceability.
    gate
        Threshold configuration (defaults to v2 §4.6 thresholds).
    rng
        Optional ``numpy.random.Generator`` for reproducible bootstrap CIs.
    meta
        Optional metadata dict (skipped sessions, prefixes, RNG seed,
        etc.). Round-trips through JSON unchanged.

    Returns
    -------
    Verdict
    """
    if gate is None:
        gate = GateConfig()
    if rng is None:
        rng = np.random.default_rng()
    if e_baseline.shape != e_candidate.shape:
        raise ValueError(
            f"shape mismatch: baseline {e_baseline.shape} vs candidate {e_candidate.shape}"
        )
    if e_baseline.ndim != 2:
        raise ValueError(f"expected (n_frames, n_keypoints), got {e_baseline.shape}")
    if len(keypoint_names) != e_baseline.shape[1]:
        raise ValueError(
            f"keypoint_names length {len(keypoint_names)} != n_keypoints {e_baseline.shape[1]}"
        )

    p_values = paired_wilcoxon_per_keypoint(e_baseline, e_candidate)
    keypoint_rows: list[KeypointVerdict] = []
    for k, name in enumerate(keypoint_names):
        keypoint_rows.append(
            _build_keypoint_verdict(
                name,
                e_baseline[:, k],
                e_candidate[:, k],
                float(p_values[k]),
                rng,
            )
        )

    # HD descriptive panel.
    if hd_baseline_rad is not None and hd_candidate_rad is not None and hd_gt_rad is not None:
        hd_b = circular_abs_error(hd_baseline_rad, hd_gt_rad)
        hd_c = circular_abs_error(hd_candidate_rad, hd_gt_rad)
        valid = np.isfinite(hd_b) & np.isfinite(hd_c)
        hd_b_v = hd_b[valid]
        hd_c_v = hd_c[valid]
        n_hd = int(hd_b_v.size)
        if n_hd >= DEFAULT_MIN_PAIRS:
            try:
                wp = float(
                    scipy.stats.wilcoxon(
                        hd_b_v, hd_c_v, alternative="greater", zero_method="wilcox"
                    ).pvalue
                )
            except ValueError:
                wp = float("nan")
            r = rank_biserial_paired(hd_b_v, hd_c_v)
        else:
            wp = float("nan")
            r = 0.0
        hd_panel: dict[str, float | int | None] = {
            "median_abs_error_baseline_rad": float(np.median(hd_b_v))
            if n_hd > 0
            else float("nan"),
            "median_abs_error_candidate_rad": float(np.median(hd_c_v))
            if n_hd > 0
            else float("nan"),
            "p_value_wilcoxon": wp,
            "rank_biserial_r": r,
            "n_frames": n_hd,
        }
    else:
        hd_panel = {
            "median_abs_error_baseline_rad": None,
            "median_abs_error_candidate_rad": None,
            "p_value_wilcoxon": None,
            "rank_biserial_r": None,
            "n_frames": 0,
        }

    overall, fail_reasons, per_kp = evaluate_gate(keypoint_rows, gate)

    # n_frames_compared = the number of frames available on at least one
    # keypoint (not strictly the intersection). For a clean signal, take
    # the modal n_pairs across keypoints.
    n_frames_compared = max((kp.n_pairs for kp in keypoint_rows), default=0)

    return Verdict(
        schema_version=VERDICT_SCHEMA_VERSION,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        n_frames_compared=int(n_frames_compared),
        keypoints=tuple(keypoint_rows),
        hd=hd_panel,
        gate=gate,
        gate_pass_per_keypoint=per_kp,
        overall_pass=overall,
        fail_reasons=tuple(fail_reasons),
        generated_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        meta=dict(meta or {}),
    )
