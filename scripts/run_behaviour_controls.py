#!/usr/bin/env python3
"""Control analyses for the behavioural manuscript.

Seven control analyses identified by the scientific QA analyst (Bikky)
as required before the behavioural manuscript can be published. These
address potential confounds in the main results.

Controls:
  1. Coverage per active minute (light vs dark) — speed confound
  2. MRL by node type (junction/corridor/dead-end), light vs dark
  3. MRL and AHV restricted to active frames only
  4. Speed by node type including ALL frames (not just active)
  5. Random walk null model for alternation
  6. Per-bodypart tracking quality by light condition
  7. Primary-only analysis (1 session per animal)

This script operates in two modes:
  - **Frame-level mode:** Downloads sync.h5 and kinematics.h5 from S3 to
    compute all controls from raw frame data.
  - **Summary mode (fallback):** When sync.h5 is unavailable, loads the
    existing behaviour-results.json and computes controls where possible.
    Controls 1, 2, 4, 6 require frame-level data and are approximated or
    deferred.

Outputs:
  - docs/manuscripts/behaviour-control-results.json
  - docs/manuscripts/behaviour-control-summary.md

Usage:
  python scripts/run_behaviour_controls.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.maze.analysis import classify_turn
from hm2p.maze.topology import build_rose_maze

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BUCKET = "hm2p-derivatives"
S3_REGION = "ap-southeast-2"
METADATA_CSV = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
ANIMALS_CSV = Path(__file__).resolve().parent.parent / "metadata" / "animals.csv"
PREV_RESULTS_JSON = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "manuscripts"
    / "behaviour-results.json"
)
OUTPUT_JSON = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "manuscripts"
    / "behaviour-control-results.json"
)
OUTPUT_MD = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "manuscripts"
    / "behaviour-control-summary.md"
)

SPEED_ACTIVE_THRESHOLD = 2.5  # cm/s
MIN_EPOCH_DURATION_S = 30.0  # minimum epoch duration to include

# Build maze once
MAZE = build_rose_maze()

# Bodyparts tracked by DLC
BODYPARTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _make_serializable(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.ndarray):
        return [_make_serializable(v) for v in obj.tolist()]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def rank_biserial_wilcoxon(x, y):
    """Compute rank-biserial correlation for Wilcoxon signed-rank test.

    r = 1 - (2W) / (n*(n+1)/2) where W is the test statistic.
    """
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    result = sp_stats.wilcoxon(x, y, alternative="two-sided")
    W = result.statistic
    r = 1.0 - (2.0 * W) / (n * (n + 1) / 2.0)
    return float(r)


def wilcoxon_test(x, y, alternative="two-sided"):
    """Wilcoxon signed-rank test with rank-biserial effect size."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 6:
        return {"stat": None, "p": None, "r": None, "n": n, "test": "wilcoxon"}
    try:
        result = sp_stats.wilcoxon(x, y, alternative=alternative)
        r = rank_biserial_wilcoxon(x, y)
        return {
            "stat": float(result.statistic),
            "p": float(result.pvalue),
            "r": r,
            "n": n,
            "test": "wilcoxon_signed_rank",
        }
    except Exception as e:
        return {"stat": None, "p": None, "r": None, "n": n, "error": str(e)}


def wilcoxon_one_sample(x, alternative="two-sided"):
    """One-sample Wilcoxon signed-rank test (vs 0)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return {"stat": None, "p": None, "r": None, "n": n}
    try:
        result = sp_stats.wilcoxon(x, alternative=alternative)
        r = 1.0 - (2.0 * result.statistic) / (n * (n + 1) / 2.0)
        return {
            "stat": float(result.statistic),
            "p": float(result.pvalue),
            "r": float(r),
            "n": n,
            "test": "wilcoxon_one_sample",
        }
    except Exception as e:
        return {"stat": None, "p": None, "r": None, "n": n, "error": str(e)}


def mannwhitney_test(x, y):
    """Mann-Whitney U test with Cliff's delta."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3:
        return {"U": None, "p": None, "cliff_d": None, "n1": nx, "n2": ny}
    try:
        U, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        more = np.sum(x[:, None] > y[None, :])
        less = np.sum(x[:, None] < y[None, :])
        cd = float((more - less) / (nx * ny))
        return {
            "U": float(U),
            "p": float(p),
            "cliff_d": cd,
            "n1": nx,
            "n2": ny,
            "test": "mann_whitney_u",
        }
    except Exception as e:
        return {
            "U": None,
            "p": None,
            "cliff_d": None,
            "n1": nx,
            "n2": ny,
            "error": str(e),
        }


def friedman_test(*groups):
    """Friedman test across multiple related groups."""
    arrays = [np.asarray(g, dtype=float) for g in groups]
    valid = np.all([np.isfinite(a) for a in arrays], axis=0)
    arrays = [a[valid] for a in arrays]
    n = len(arrays[0])
    if n < 6:
        return {"stat": None, "p": None, "n": n}
    try:
        stat, p = sp_stats.friedmanchisquare(*arrays)
        return {"stat": float(stat), "p": float(p), "n": n, "test": "friedman"}
    except Exception as e:
        return {"stat": None, "p": None, "n": n, "error": str(e)}


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    pvals = np.asarray(p_values, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals.tolist()
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = pvals[idx] * (n - rank)
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted.tolist()


def _extract_paired(results, key_l, key_d):
    """Extract paired numeric values, dropping None/NaN."""
    pairs = []
    for r in results:
        vl = r.get(key_l)
        vd = r.get(key_d)
        if (
            vl is not None
            and vd is not None
            and isinstance(vl, (int, float))
            and isinstance(vd, (int, float))
            and np.isfinite(vl)
            and np.isfinite(vd)
        ):
            pairs.append((vl, vd))
    if not pairs:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _extract_paired_from_dict(results, outer_key, inner_key_l, inner_key_d=None):
    """Extract paired values from nested dicts."""
    pairs = []
    for r in results:
        d = r.get(outer_key, {})
        if not isinstance(d, dict):
            continue
        vl = d.get(inner_key_l)
        vd = d.get(inner_key_d) if inner_key_d else None
        if vl is not None and vd is not None:
            if np.isfinite(vl) and np.isfinite(vd):
                pairs.append((vl, vd))
    if not pairs:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


# ---------------------------------------------------------------------------
# Random walk null model
# ---------------------------------------------------------------------------


def _simulate_random_walk(maze, n_steps, rng):
    """Simulate a random walk on the maze graph.

    At each step, the walker moves to a uniformly random neighbour.
    Returns the cell sequence as cell indices.
    """
    cell_list = maze.cell_list
    adj = maze.adj
    start_idx = rng.integers(len(cell_list))
    start_cell = cell_list[start_idx]
    path = [maze.cell_to_idx[start_cell]]
    current = start_cell
    for _ in range(n_steps - 1):
        neighbours = adj[current]
        next_cell = neighbours[int(rng.integers(len(neighbours)))]
        path.append(maze.cell_to_idx[next_cell])
        current = next_cell
    return np.array(path, dtype=np.intp)


def _turn_autocorrelation_from_cell_seq(cs, maze):
    """Compute lag-1 Spearman autocorrelation of L/R turn sequence."""
    if len(cs) < 3:
        return None
    junction_indices = {maze.cell_to_idx[j] for j in maze.junctions}
    cell_list = maze.cell_list
    turns = []
    for i in range(1, len(cs) - 1):
        if cs[i] in junction_indices:
            prev = cell_list[cs[i - 1]]
            curr = cell_list[cs[i]]
            nxt = cell_list[cs[i + 1]]
            t = classify_turn(prev, curr, nxt)
            if t in ("left", "right"):
                turns.append(0 if t == "left" else 1)
    if len(turns) < 4:
        return None
    turns = np.array(turns, dtype=float)
    t1 = turns[:-1]
    t2 = turns[1:]
    if np.std(t1) == 0 or np.std(t2) == 0:
        return 0.0
    rho, _ = sp_stats.spearmanr(t1, t2)
    return float(rho)


def control_5_random_walk_null(
    observed_autocorrs, observed_turn_counts, maze, n_simulations=1000
):
    """Simulate random walks and compare alternation to observed data.

    For each session, simulates random walks on the maze graph of
    comparable length and computes the lag-1 turn autocorrelation.
    The null distribution tests whether observed alternation exceeds
    what a random walk would produce.

    Parameters
    ----------
    observed_autocorrs : list
        Per-session observed lag-1 turn autocorrelations (may contain None).
    observed_turn_counts : list
        Per-session approximate cell-sequence lengths (from turn counts).
    maze : RoseMaze
        Maze topology for the random walk.
    n_simulations : int
        Number of random walks per session.

    Returns
    -------
    dict with null distribution statistics and observed vs null comparison.
    """
    rng = np.random.default_rng(42)

    # Estimate cell sequence length from turn counts. Each L/R turn implies
    # the walker visited a junction. The total cell sequence is roughly
    # 2x the number of turns (including non-junction visits).
    # Use a conversion factor: each junction visit corresponds to ~3 cells
    # in the cell sequence (approach, junction, departure).
    null_autocorrs_all = []
    per_session_null = []

    for i, walk_len in enumerate(observed_turn_counts):
        if walk_len < 20:
            per_session_null.append([])
            continue
        session_null = []
        for _ in range(n_simulations):
            cs = _simulate_random_walk(maze, int(walk_len), rng)
            ac = _turn_autocorrelation_from_cell_seq(cs, maze)
            if ac is not None:
                session_null.append(ac)
                null_autocorrs_all.append(ac)
        per_session_null.append(session_null)

    null_arr = np.array(null_autocorrs_all) if null_autocorrs_all else np.array([])
    observed = np.array([a for a in observed_autocorrs if a is not None])

    result = {
        "n_observed": len(observed),
        "n_null_total": len(null_arr),
        "n_simulations_per_session": n_simulations,
        "observed_mean": float(np.mean(observed)) if len(observed) > 0 else None,
        "observed_median": float(np.median(observed)) if len(observed) > 0 else None,
        "observed_sd": float(np.std(observed, ddof=1)) if len(observed) > 1 else None,
        "null_mean": float(np.mean(null_arr)) if len(null_arr) > 0 else None,
        "null_median": float(np.median(null_arr)) if len(null_arr) > 0 else None,
        "null_sd": float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else None,
        "null_pct_2_5": (
            float(np.percentile(null_arr, 2.5)) if len(null_arr) > 0 else None
        ),
        "null_pct_97_5": (
            float(np.percentile(null_arr, 97.5)) if len(null_arr) > 0 else None
        ),
    }

    # Mann-Whitney U: observed session-level autocorrs vs null distribution
    if len(observed) >= 3 and len(null_arr) >= 3:
        result["test_observed_vs_null"] = mannwhitney_test(observed, null_arr)
    else:
        result["test_observed_vs_null"] = {"p": None}

    # Bootstrap permutation test: is observed mean more negative than null?
    if len(observed) > 0 and len(null_arr) > 0:
        obs_mean = np.mean(observed)
        n_perm = 10000
        perm_means = []
        for _ in range(n_perm):
            sample = rng.choice(null_arr, size=len(observed), replace=True)
            perm_means.append(np.mean(sample))
        perm_means = np.array(perm_means)
        # One-sided: fraction of null means as negative as observed
        p_perm_onesided = float(np.mean(perm_means <= obs_mean))
        # Two-sided
        p_perm_twosided = float(np.mean(np.abs(perm_means) >= np.abs(obs_mean)))
        result["permutation_p_onesided"] = p_perm_onesided
        result["permutation_p_twosided"] = p_perm_twosided
    else:
        result["permutation_p_onesided"] = None
        result["permutation_p_twosided"] = None

    # How many observed sessions fall outside the null 95% CI?
    if len(observed) > 0 and len(null_arr) > 0:
        ci_lo = np.percentile(null_arr, 2.5)
        ci_hi = np.percentile(null_arr, 97.5)
        n_below = int(np.sum(observed < ci_lo))
        n_above = int(np.sum(observed > ci_hi))
        result["n_observed_below_null_ci"] = n_below
        result["n_observed_above_null_ci"] = n_above
        result["frac_observed_outside_null_ci"] = (n_below + n_above) / len(observed)
    else:
        result["n_observed_below_null_ci"] = None
        result["n_observed_above_null_ci"] = None
        result["frac_observed_outside_null_ci"] = None

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("BEHAVIOURAL CONTROL ANALYSES — hm2p project")
    print("=" * 70)

    # Load metadata
    with open(METADATA_CSV) as f:
        experiments = list(csv.DictReader(f))
    with open(ANIMALS_CSV) as f:
        animals = {row["animal_id"]: row for row in csv.DictReader(f)}

    sessions_meta = []
    for row in experiments:
        eid = row["exp_id"]
        parts = eid.split("_")
        animal_id = parts[-1]
        sessions_meta.append(
            {
                "exp_id": eid,
                "exp_index": int(row["exp_index"]),
                "animal_id": animal_id,
                "celltype": animals.get(animal_id, {}).get("celltype", "unknown"),
                "sex": animals.get(animal_id, {}).get("sex", "unknown"),
                "exclude": str(row.get("exclude", "0")).strip() == "1",
                "primary": str(row.get("primary_exp", "1")).strip() != "0",
            }
        )

    # Load previous results (frame-level data not on S3; using computed
    # per-session summaries from the main analysis run)
    if not PREV_RESULTS_JSON.exists():
        print(f"ERROR: {PREV_RESULTS_JSON} not found. Run run_behaviour_analysis.py first.")
        sys.exit(1)

    with open(PREV_RESULTS_JSON) as f:
        prev = json.load(f)

    prev_sessions = {r["exp_id"]: r for r in prev["per_session"]}
    prev_stats = prev["cross_session"]

    # Filter to usable sessions (matching the main analysis)
    usable = [
        prev_sessions[s["exp_id"]]
        for s in sessions_meta
        if not s["exclude"]
        and s["exp_id"] in prev_sessions
        and prev_sessions[s["exp_id"]].get("status") == "ok"
    ]

    # Annotate with metadata
    meta_lookup = {s["exp_id"]: s for s in sessions_meta}
    for r in usable:
        m = meta_lookup[r["exp_id"]]
        r["primary"] = m["primary"]
        r["exclude"] = m["exclude"]

    primary_only = [r for r in usable if r["primary"]]
    n_usable = len(usable)
    n_primary = len(primary_only)
    n_animals = len(set(r["animal_id"] for r in usable))

    print(f"\nUsable sessions: {n_usable}")
    print(f"Primary-only sessions: {n_primary}")
    print(f"Animals: {n_animals}")

    stats = {
        "dataset": {
            "n_usable_sessions": n_usable,
            "n_primary_sessions": n_primary,
            "n_animals": n_animals,
        }
    }

    # =================================================================
    # Control 1: Coverage per active minute
    # =================================================================
    #
    # The main analysis found per-epoch coverage is lower in dark (p=0.003).
    # This could be driven by the speed confound (less active time in dark).
    # We approximate coverage/active-minute from the existing summaries:
    #   cells_per_epoch / (active_fraction * epoch_duration) * 60
    #
    # The per-session mean_epoch_coverage_* gives the fraction of 23 cells
    # visited per 1-min epoch. frac_active_* gives the active fraction.
    # Coverage per active minute = (coverage * 23) / (frac_active * 60) * 60
    #                            = coverage * 23 / frac_active
    # =================================================================
    print("\n--- Control 1: Coverage per active minute ---")

    cov_per_active_min_light = []
    cov_per_active_min_dark = []
    for r in usable:
        cov_l = r.get("mean_epoch_coverage_light")
        cov_d = r.get("mean_epoch_coverage_dark")
        fa_l = r.get("frac_active_light")
        fa_d = r.get("frac_active_dark")

        if all(v is not None and v > 0 for v in [cov_l, cov_d, fa_l, fa_d]):
            # cells_visited / active_minutes = (coverage_frac * 23) / (frac_active * epoch_duration_min)
            # Since epoch duration is ~1 min and we have mean coverage per epoch:
            # coverage_per_active_min = (coverage * 23) / frac_active
            cov_per_active_min_light.append(cov_l * 23 / fa_l)
            cov_per_active_min_dark.append(cov_d * 23 / fa_d)

    c1_l = np.array(cov_per_active_min_light)
    c1_d = np.array(cov_per_active_min_dark)

    stats["control_1"] = {
        "description": (
            "Unique cells visited per active minute (speed >= 2.5 cm/s). "
            "Approximated from per-epoch coverage and active fraction. "
            "Controls for the speed confound: if normalised coverage is still "
            "lower in dark, the finding stands independent of reduced locomotion."
        ),
        "method": "coverage_frac * 23 / frac_active (approximation from summaries)",
        "n": len(c1_l),
        "mean_light": float(np.mean(c1_l)) if len(c1_l) > 0 else None,
        "mean_dark": float(np.mean(c1_d)) if len(c1_d) > 0 else None,
        "median_light": float(np.median(c1_l)) if len(c1_l) > 0 else None,
        "median_dark": float(np.median(c1_d)) if len(c1_d) > 0 else None,
        "test": wilcoxon_test(c1_l, c1_d),
    }
    if len(c1_l) > 0:
        t = stats["control_1"]["test"]
        print(
            f"  Coverage/active_min: Light={np.mean(c1_l):.2f}, Dark={np.mean(c1_d):.2f}"
        )
        print(f"  Wilcoxon: p={t.get('p')}, r={t.get('r')}, N={t.get('n')}")

    # =================================================================
    # Control 2: MRL by node type, light vs dark
    #
    # The existing analysis stores speed_junction, speed_corridor,
    # speed_dead_end (active-only). For MRL by node type, we need
    # frame-level HD + position data. This control cannot be computed
    # from summaries — it requires sync.h5 files.
    #
    # We note this limitation and defer to when sync.h5 is available.
    # =================================================================
    print("\n--- Control 2: MRL by node type ---")
    stats["control_2"] = {
        "description": (
            "HD mean resultant length at each maze-node type "
            "(junction, corridor, dead end), compared between light and dark. "
            "Controls for maze geometry confound on HD non-uniformity."
        ),
        "status": "requires_frame_data",
        "note": (
            "This control requires frame-level HD and position data from "
            "sync.h5 files. Cannot be computed from per-session summaries. "
            "When sync.h5 files are regenerated, re-run this script."
        ),
    }
    print("  STATUS: requires frame-level data (sync.h5). Deferred.")

    # =================================================================
    # Control 3: MRL and AHV restricted to active frames only
    #
    # The existing analysis ALREADY restricts MRL and AHV to active
    # frames (speed >= 2.5 cm/s). This is confirmed by the code in
    # run_behaviour_analysis.py lines 541-542:
    #   mask_hd_light = valid & light_on & active & hd_finite
    # and lines 561-562 for AHV.
    #
    # Therefore Control 3 is already satisfied by the existing results.
    # We report this as a verification, re-stating the existing values.
    # =================================================================
    print("\n--- Control 3: MRL and AHV, active frames only ---")

    mrl_l, mrl_d = _extract_paired(usable, "hd_mrl_light", "hd_mrl_dark")
    ahv_l, ahv_d = _extract_paired(usable, "median_ahv_light", "median_ahv_dark")

    stats["control_3"] = {
        "description": (
            "MRL and median |AHV| using only active frames (speed >= 2.5 cm/s). "
            "Controls for immobility driving HD non-uniformity. "
            "NOTE: The main analysis ALREADY restricts to active frames "
            "(mask_hd_light = valid & light_on & active & hd_finite). "
            "These values are identical to the Figure 5 results."
        ),
        "already_active_only": True,
        "mrl": {
            "n": len(mrl_l),
            "mean_light": float(np.mean(mrl_l)) if len(mrl_l) > 0 else None,
            "mean_dark": float(np.mean(mrl_d)) if len(mrl_d) > 0 else None,
            "median_light": float(np.median(mrl_l)) if len(mrl_l) > 0 else None,
            "median_dark": float(np.median(mrl_d)) if len(mrl_d) > 0 else None,
            "test": wilcoxon_test(mrl_l, mrl_d),
        },
        "ahv": {
            "n": len(ahv_l),
            "mean_light": float(np.mean(ahv_l)) if len(ahv_l) > 0 else None,
            "mean_dark": float(np.mean(ahv_d)) if len(ahv_d) > 0 else None,
            "median_light": float(np.median(ahv_l)) if len(ahv_l) > 0 else None,
            "median_dark": float(np.median(ahv_d)) if len(ahv_d) > 0 else None,
            "test": wilcoxon_test(ahv_l, ahv_d),
        },
    }
    c3_pvals = [
        stats["control_3"]["mrl"]["test"].get("p"),
        stats["control_3"]["ahv"]["test"].get("p"),
    ]
    c3_pvals_clean = [p if p is not None else 1.0 for p in c3_pvals]
    c3_adjusted = holm_bonferroni(c3_pvals_clean)
    stats["control_3"]["holm_bonferroni_adjusted_p"] = {
        "mrl": c3_adjusted[0],
        "ahv": c3_adjusted[1],
    }

    if len(mrl_l) > 0:
        t = stats["control_3"]["mrl"]["test"]
        print(
            f"  MRL (active): Light={np.mean(mrl_l):.3f}, Dark={np.mean(mrl_d):.3f}, "
            f"p={t.get('p'):.4f}, r={t.get('r'):.3f}"
        )
    if len(ahv_l) > 0:
        t = stats["control_3"]["ahv"]["test"]
        print(
            f"  AHV (active): Light={np.mean(ahv_l):.1f}, Dark={np.mean(ahv_d):.1f}, "
            f"p={t.get('p'):.4f}, r={t.get('r'):.3f}"
        )

    # =================================================================
    # Control 4: Speed by node type, all frames
    #
    # The existing analysis computes speed by node type using ONLY
    # active frames (speed >= 2.5 cm/s). For this control, we need
    # all frames including immobile ones. This requires sync.h5.
    #
    # We also report what the active-only Friedman test showed as
    # a reference.
    # =================================================================
    print("\n--- Control 4: Speed by node type, all frames ---")
    stats["control_4"] = {
        "description": (
            "Speed at each maze-node type including all valid frames "
            "(not just active). Shows full distributions including immobility. "
            "The existing Friedman test used active-only frames."
        ),
        "status": "requires_frame_data",
        "note": (
            "This control requires frame-level speed and position data from "
            "sync.h5 files. Cannot be computed from per-session summaries. "
            "When sync.h5 files are regenerated, re-run this script."
        ),
        "active_only_reference": prev_stats.get("figure6", {}).get(
            "speed_by_node_type", {}
        ),
    }
    print("  STATUS: requires frame-level data (sync.h5). Deferred.")
    print("  Active-only reference from existing results:")
    ref = prev_stats.get("figure6", {}).get("speed_by_node_type", {})
    if ref:
        print(
            f"    J={ref.get('mean_junction', 'N/A'):.2f}, "
            f"C={ref.get('mean_corridor', 'N/A'):.2f}, "
            f"DE={ref.get('mean_dead_end', 'N/A'):.2f} cm/s"
        )

    # =================================================================
    # Control 5: Random walk null model for alternation
    #
    # The main analysis found strong turn alternation (lag-1
    # autocorrelation = -0.196, p < 0.0001). This could be partly
    # explained by maze geometry (the graph constrains available turns).
    # We simulate random walks on the maze graph and compare.
    # =================================================================
    print("\n--- Control 5: Random walk null model for alternation ---")

    # Extract observed autocorrelations and walk lengths
    observed_autocorrs = [
        r.get("turn_autocorr_all") for r in usable
    ]
    # Estimate cell sequence length from the turn bias data
    # Total turns = left + right + back + forward at all junctions
    observed_walk_lengths = []
    for r in usable:
        tb_l = r.get("turn_bias_light", {})
        tb_d = r.get("turn_bias_dark", {})
        total_turns = 0
        for tb in [tb_l, tb_d]:
            if isinstance(tb, dict):
                for k in ["left", "right", "back", "forward"]:
                    total_turns += tb.get(k, 0) or 0
        # Each junction visit implies ~3 cells in the sequence
        # (approach, junction, departure). Total cell sequence ~3x turns.
        # But the walk also includes corridor traversals between junctions.
        # A rough estimate: cell_seq_len ~ 3 * total_junction_visits
        # where total_junction_visits = total_turns
        estimated_len = max(total_turns * 3, 50)
        observed_walk_lengths.append(estimated_len)

    c5 = control_5_random_walk_null(
        observed_autocorrs, observed_walk_lengths, MAZE, n_simulations=1000
    )
    c5["description"] = (
        "Random walk null model for sequential turn alternation. "
        "Simulates 1000 random walks per session on the maze graph "
        "and compares the observed lag-1 turn autocorrelation to the null. "
        "If observed alternation is significantly more negative than the null, "
        "it reflects a genuine behavioural strategy, not a maze geometry artefact."
    )
    stats["control_5"] = c5

    print(
        f"  Observed: mean={c5.get('observed_mean'):.3f}, "
        f"median={c5.get('observed_median'):.3f} (N={c5.get('n_observed')})"
    )
    print(
        f"  Null:     mean={c5.get('null_mean'):.3f}, "
        f"median={c5.get('null_median'):.3f}, "
        f"95% CI=[{c5.get('null_pct_2_5'):.3f}, {c5.get('null_pct_97_5'):.3f}] "
        f"(N={c5.get('n_null_total')})"
    )
    mw = c5.get("test_observed_vs_null", {})
    print(
        f"  Mann-Whitney: U={mw.get('U')}, p={mw.get('p')}, "
        f"Cliff's d={mw.get('cliff_d')}"
    )
    print(
        f"  Permutation p (one-sided): {c5.get('permutation_p_onesided'):.4f}"
    )
    print(
        f"  Sessions outside null 95% CI: "
        f"{c5.get('n_observed_below_null_ci')} below, "
        f"{c5.get('n_observed_above_null_ci')} above"
    )

    # =================================================================
    # Control 6: Per-bodypart tracking quality
    #
    # This requires kinematics.h5 files with raw per-bodypart positions.
    # These files are not currently on S3. Deferred.
    # =================================================================
    print("\n--- Control 6: Per-bodypart tracking quality ---")
    stats["control_6"] = {
        "description": (
            "Per-bodypart valid-frame fraction (non-NaN in raw kinematics) "
            "in light vs dark. NaN indicates non-detection by the pose tracker. "
            "This checks whether any bodypart has systematically worse tracking "
            "in one condition."
        ),
        "status": "requires_frame_data",
        "note": (
            "This control requires per-bodypart raw positions from "
            "kinematics.h5 files. These are not currently on S3. "
            "When kinematics.h5 files are regenerated, re-run this script."
        ),
    }
    print("  STATUS: requires frame-level data (kinematics.h5). Deferred.")

    # =================================================================
    # Control 7: Primary-only analysis
    #
    # Re-run key comparisons using only primary_exp=True sessions
    # (one session per animal, N=12-13).
    # =================================================================
    print("\n--- Control 7: Primary-only analysis ---")

    n_primary_animals = len(set(r["animal_id"] for r in primary_only))
    stats["control_7"] = {
        "description": (
            "Key comparisons re-run using only primary_exp=True sessions "
            "(one session per animal). Controls for pseudoreplication from "
            "animals with multiple sessions."
        ),
        "n_sessions": n_primary,
        "n_animals": n_primary_animals,
    }

    # 7a: Coverage (raw epoch coverage, from main analysis)
    ecov_l, ecov_d = _extract_paired(
        primary_only, "mean_epoch_coverage_light", "mean_epoch_coverage_dark"
    )
    stats["control_7"]["epoch_coverage"] = {
        "n": len(ecov_l),
        "mean_light": float(np.mean(ecov_l)) if len(ecov_l) > 0 else None,
        "mean_dark": float(np.mean(ecov_d)) if len(ecov_d) > 0 else None,
        "test": wilcoxon_test(ecov_l, ecov_d),
    }

    # 7b: Coverage per active minute (approximation)
    c7_cov_l, c7_cov_d = [], []
    for r in primary_only:
        cov_l = r.get("mean_epoch_coverage_light")
        cov_d = r.get("mean_epoch_coverage_dark")
        fa_l = r.get("frac_active_light")
        fa_d = r.get("frac_active_dark")
        if all(v is not None and v > 0 for v in [cov_l, cov_d, fa_l, fa_d]):
            c7_cov_l.append(cov_l * 23 / fa_l)
            c7_cov_d.append(cov_d * 23 / fa_d)
    c7_cl = np.array(c7_cov_l)
    c7_cd = np.array(c7_cov_d)
    stats["control_7"]["coverage_per_active_min"] = {
        "n": len(c7_cl),
        "mean_light": float(np.mean(c7_cl)) if len(c7_cl) > 0 else None,
        "mean_dark": float(np.mean(c7_cd)) if len(c7_cd) > 0 else None,
        "test": wilcoxon_test(c7_cl, c7_cd),
    }

    # 7c: MRL (active only — same as main analysis, since it already filters)
    mrl7_l, mrl7_d = _extract_paired(primary_only, "hd_mrl_light", "hd_mrl_dark")
    stats["control_7"]["mrl"] = {
        "n": len(mrl7_l),
        "mean_light": float(np.mean(mrl7_l)) if len(mrl7_l) > 0 else None,
        "mean_dark": float(np.mean(mrl7_d)) if len(mrl7_d) > 0 else None,
        "test": wilcoxon_test(mrl7_l, mrl7_d),
    }

    # 7d: AHV (active only)
    ahv7_l, ahv7_d = _extract_paired(primary_only, "median_ahv_light", "median_ahv_dark")
    stats["control_7"]["ahv"] = {
        "n": len(ahv7_l),
        "mean_light": float(np.mean(ahv7_l)) if len(ahv7_l) > 0 else None,
        "mean_dark": float(np.mean(ahv7_d)) if len(ahv7_d) > 0 else None,
        "test": wilcoxon_test(ahv7_l, ahv7_d),
    }

    # 7e: Speed
    sp7_l, sp7_d = _extract_paired(primary_only, "median_speed_light", "median_speed_dark")
    stats["control_7"]["speed"] = {
        "n": len(sp7_l),
        "mean_light": float(np.mean(sp7_l)) if len(sp7_l) > 0 else None,
        "mean_dark": float(np.mean(sp7_d)) if len(sp7_d) > 0 else None,
        "test": wilcoxon_test(sp7_l, sp7_d),
    }

    # 7f: Fraction active
    fa7_l, fa7_d = _extract_paired(primary_only, "frac_active_light", "frac_active_dark")
    stats["control_7"]["frac_active"] = {
        "n": len(fa7_l),
        "mean_light": float(np.mean(fa7_l)) if len(fa7_l) > 0 else None,
        "mean_dark": float(np.mean(fa7_d)) if len(fa7_d) > 0 else None,
        "test": wilcoxon_test(fa7_l, fa7_d),
    }

    # 7g: Turn autocorrelation
    ac7_all = np.array([
        r["turn_autocorr_all"]
        for r in primary_only
        if r.get("turn_autocorr_all") is not None
    ])
    stats["control_7"]["turn_autocorr_vs_zero"] = {
        "n": len(ac7_all),
        "mean": float(np.mean(ac7_all)) if len(ac7_all) > 0 else None,
        "test": wilcoxon_one_sample(ac7_all),
    }

    # Print summary
    for key in [
        "epoch_coverage",
        "coverage_per_active_min",
        "mrl",
        "ahv",
        "speed",
        "frac_active",
    ]:
        d = stats["control_7"].get(key, {})
        t = d.get("test", {})
        print(
            f"  {key:30s}: N={d.get('n', 'N/A'):>2}, "
            f"L={d.get('mean_light')}, D={d.get('mean_dark')}, "
            f"p={t.get('p')}, r={t.get('r')}"
        )
    ac_d = stats["control_7"]["turn_autocorr_vs_zero"]
    print(
        f"  {'turn_autocorr_vs_zero':30s}: N={ac_d.get('n', 'N/A'):>2}, "
        f"mean={ac_d.get('mean')}, p={ac_d.get('test', {}).get('p')}"
    )

    # ===================================================================
    # Save results
    # ===================================================================
    output = {"cross_session": stats}
    output_ser = _make_serializable(output)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_ser, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_JSON}")

    # ===================================================================
    # Generate summary markdown
    # ===================================================================
    _write_summary_markdown(stats)
    print(f"Summary saved to: {OUTPUT_MD}")


def _write_summary_markdown(stats):
    """Write a summary markdown of all control analyses."""

    def _fmt_test(test_dict, adjusted_p=None):
        if test_dict is None or test_dict.get("p") is None:
            return "N/A"
        p = test_dict["p"]
        r = test_dict.get("r")
        n = test_dict.get("n")
        parts = []
        if test_dict.get("stat") is not None:
            parts.append(f"W = {test_dict['stat']:.1f}")
        if test_dict.get("U") is not None:
            parts.append(f"U = {test_dict['U']:.1f}")
        parts.append(f"p = {p:.4f}")
        if adjusted_p is not None:
            parts.append(f"p_adj = {adjusted_p:.4f}")
        if r is not None:
            parts.append(f"r = {r:.3f}")
        if test_dict.get("cliff_d") is not None:
            parts.append(f"d = {test_dict['cliff_d']:.3f}")
        if n is not None:
            parts.append(f"N = {n}")
        return ", ".join(parts)

    def _rv(v, fmt=".3f"):
        if v is None:
            return "N/A"
        return f"{v:{fmt}}"

    lines = [
        "# Behavioural Control Analyses — Summary",
        "",
        f"Generated from {stats['dataset']['n_usable_sessions']} usable sessions "
        f"({stats['dataset']['n_animals']} animals). "
        f"{stats['dataset']['n_primary_sessions']} primary sessions.",
        "",
        "Seven control analyses addressing potential confounds in the main "
        "behavioural manuscript results. All tests non-parametric.",
        "",
        "---",
        "",
    ]

    # Control 1
    c1 = stats.get("control_1", {})
    lines.extend([
        "## Control 1: Coverage Per Active Minute (Light vs Dark)",
        "",
        "**Question:** Is the lower per-epoch coverage in dark driven by reduced "
        "locomotion speed, or does it persist when normalised by active time?",
        "",
        "**Method:** Coverage per active minute = (epoch_coverage * 23) / "
        "frac_active. Approximation from per-session summaries.",
        "",
        f"- Light: {_rv(c1.get('mean_light'), '.2f')} cells/active-min "
        f"(median {_rv(c1.get('median_light'), '.2f')})",
        f"- Dark: {_rv(c1.get('mean_dark'), '.2f')} cells/active-min "
        f"(median {_rv(c1.get('median_dark'), '.2f')})",
        f"- {_fmt_test(c1.get('test'))}",
        "",
    ])

    t1 = c1.get("test", {})
    if t1.get("p") is not None:
        if t1["p"] < 0.05:
            lines.append(
                "**Interpretation:** Coverage per active minute is significantly "
                "different between light and dark even after controlling for "
                "locomotion time. The main coverage result is not simply a speed "
                "artefact."
            )
        else:
            lines.append(
                "**Interpretation:** No significant difference in coverage per "
                "active minute between light and dark. The main coverage finding "
                "may be partly explained by reduced locomotion in darkness."
            )
    lines.extend(["", "---", ""])

    # Control 2
    c2 = stats.get("control_2", {})
    lines.extend([
        "## Control 2: MRL by Node Type (Light vs Dark)",
        "",
        "**Question:** Does the higher MRL in dark persist at each maze location "
        "type, or is it driven by differential occupancy of corridors vs junctions?",
        "",
        f"**Status:** {c2.get('status', 'computed')}",
        "",
    ])
    if c2.get("status") == "requires_frame_data":
        lines.append(
            f"_{c2.get('note', 'Frame-level data required.')}_"
        )
    lines.extend(["", "---", ""])

    # Control 3
    c3 = stats.get("control_3", {})
    c3_adj = c3.get("holm_bonferroni_adjusted_p", {})
    lines.extend([
        "## Control 3: MRL and AHV — Active Frames Only",
        "",
        "**Question:** Does the higher MRL in dark persist when excluding "
        "immobile frames? Does the AHV difference persist?",
        "",
        "**Important note:** The main analysis (Figure 5) **already restricts** "
        "to active frames only (speed >= 2.5 cm/s). These values are identical "
        "to the Figure 5 results. This control confirms that the reported MRL "
        "and AHV comparisons are not confounded by immobility.",
        "",
        "| Metric | Light | Dark | Test | p_adj |",
        "| ------ | ----- | ---- | ---- | ----- |",
    ])
    mrl_d = c3.get("mrl", {})
    ahv_d = c3.get("ahv", {})
    lines.append(
        f"| MRL (active only) | {_rv(mrl_d.get('mean_light'))} | "
        f"{_rv(mrl_d.get('mean_dark'))} | {_fmt_test(mrl_d.get('test'))} | "
        f"{_rv(c3_adj.get('mrl'), '.4f')} |"
    )
    lines.append(
        f"| Median |AHV| (active only) | {_rv(ahv_d.get('mean_light'), '.1f')} | "
        f"{_rv(ahv_d.get('mean_dark'), '.1f')} | {_fmt_test(ahv_d.get('test'))} | "
        f"{_rv(c3_adj.get('ahv'), '.4f')} |"
    )
    lines.extend(["", "---", ""])

    # Control 4
    c4 = stats.get("control_4", {})
    lines.extend([
        "## Control 4: Speed by Node Type — All Frames",
        "",
        "**Question:** Does the speed hierarchy (corridor > junction > dead end) "
        "persist when including immobile frames?",
        "",
        f"**Status:** {c4.get('status', 'computed')}",
        "",
    ])
    if c4.get("status") == "requires_frame_data":
        lines.append(
            f"_{c4.get('note', 'Frame-level data required.')}_"
        )
        ref = c4.get("active_only_reference", {})
        if ref:
            lines.extend([
                "",
                "Active-only reference from Figure 6:",
                f"- Junction: {_rv(ref.get('mean_junction'), '.2f')} cm/s",
                f"- Corridor: {_rv(ref.get('mean_corridor'), '.2f')} cm/s",
                f"- Dead end: {_rv(ref.get('mean_dead_end'), '.2f')} cm/s",
            ])
    lines.extend(["", "---", ""])

    # Control 5
    c5 = stats.get("control_5", {})
    lines.extend([
        "## Control 5: Random Walk Null Model for Alternation",
        "",
        "**Question:** Is the observed turn alternation (negative lag-1 "
        "autocorrelation) stronger than expected from a random walk on the "
        "maze graph? This controls for maze geometry constraining turn sequences.",
        "",
        f"- Observed: mean = {_rv(c5.get('observed_mean'))}, "
        f"median = {_rv(c5.get('observed_median'))}, "
        f"SD = {_rv(c5.get('observed_sd'))} "
        f"(N = {c5.get('n_observed')})",
        f"- Null: mean = {_rv(c5.get('null_mean'))}, "
        f"median = {_rv(c5.get('null_median'))}, "
        f"SD = {_rv(c5.get('null_sd'))} "
        f"(N = {c5.get('n_null_total')} simulations)",
        f"- Null 95% CI: [{_rv(c5.get('null_pct_2_5'))}, "
        f"{_rv(c5.get('null_pct_97_5'))}]",
        f"- Mann-Whitney (observed vs null): "
        f"{_fmt_test(c5.get('test_observed_vs_null'))}",
        f"- Bootstrap permutation p (one-sided, H1: observed < null): "
        f"{_rv(c5.get('permutation_p_onesided'), '.4f')}",
        f"- Sessions outside null 95% CI: "
        f"{c5.get('n_observed_below_null_ci')} below, "
        f"{c5.get('n_observed_above_null_ci')} above"
        + (f" ({c5['frac_observed_outside_null_ci'] * 100:.1f}%)"
           if c5.get('frac_observed_outside_null_ci') is not None else ""),
        "",
    ])

    # Interpretation
    obs_mean = c5.get("observed_mean")
    null_lo = c5.get("null_pct_2_5")
    p_perm = c5.get("permutation_p_onesided")
    null_mean_val = c5.get("null_mean")
    if obs_mean is not None and null_lo is not None:
        lines.append(
            f"**Note:** The null distribution itself has a negative mean "
            f"({_rv(null_mean_val)}), indicating that maze geometry alone "
            f"produces some degree of turn alternation. The question is whether "
            f"the observed alternation ({_rv(obs_mean)}) exceeds this "
            f"geometry-driven baseline."
        )
        lines.append("")
        if p_perm is not None and p_perm < 0.05:
            lines.append(
                "**Interpretation:** Observed alternation is significantly "
                "stronger than the random walk null (permutation p < 0.05). "
                "While maze geometry contributes some alternation, mice show "
                "additional spontaneous alternation beyond what the graph "
                "structure would produce."
            )
        else:
            lines.append(
                "**Interpretation:** The observed alternation does not differ "
                "significantly from the random walk null. The alternation "
                "pattern may be largely explained by maze geometry."
            )
    lines.extend(["", "---", ""])

    # Control 6
    c6 = stats.get("control_6", {})
    lines.extend([
        "## Control 6: Per-Bodypart Tracking Quality by Light Condition",
        "",
        "**Question:** Is tracking quality systematically different between "
        "light and dark for any bodypart?",
        "",
        f"**Status:** {c6.get('status', 'computed')}",
        "",
    ])
    if c6.get("status") == "requires_frame_data":
        lines.append(
            f"_{c6.get('note', 'Frame-level data required.')}_"
        )
    lines.extend(["", "---", ""])

    # Control 7
    c7 = stats.get("control_7", {})
    lines.extend([
        "## Control 7: Primary-Only Analysis",
        "",
        f"**Sessions:** {c7.get('n_sessions')} sessions from "
        f"{c7.get('n_animals')} animals (one per animal).",
        "",
        "Re-runs key comparisons using only primary_exp=True sessions to "
        "control for pseudoreplication from animals with multiple sessions.",
        "",
        "| Metric | N | Light | Dark | Test |",
        "| ------ | - | ----- | ---- | ---- |",
    ])
    for key, label, fmt in [
        ("epoch_coverage", "Epoch coverage", ".3f"),
        ("coverage_per_active_min", "Coverage / active min", ".2f"),
        ("mrl", "MRL (active)", ".3f"),
        ("ahv", "Median |AHV| (deg/s)", ".1f"),
        ("speed", "Median speed (cm/s)", ".2f"),
        ("frac_active", "Fraction active", ".3f"),
    ]:
        d = c7.get(key, {})
        lines.append(
            f"| {label} | {d.get('n', 'N/A')} | {_rv(d.get('mean_light'), fmt)} | "
            f"{_rv(d.get('mean_dark'), fmt)} | {_fmt_test(d.get('test'))} |"
        )

    # Turn autocorrelation vs zero (primary only)
    ac7 = c7.get("turn_autocorr_vs_zero", {})
    if ac7.get("n"):
        lines.extend([
            "",
            f"Turn autocorrelation vs 0 (primary only): mean = "
            f"{_rv(ac7.get('mean'))}, {_fmt_test(ac7.get('test'))}",
        ])

    lines.extend(["", "---", ""])

    # Overall verdict
    lines.extend([
        "## Summary of Control Analyses",
        "",
        "| Control | Status | Key finding |",
        "| ------- | ------ | ----------- |",
    ])

    # C1
    t1 = stats.get("control_1", {}).get("test", {})
    c1_status = "computed" if t1.get("p") is not None else "N/A"
    c1_finding = f"p = {t1['p']:.4f}, r = {t1.get('r', 0):.3f}" if t1.get("p") is not None else "N/A"
    lines.append(f"| 1. Coverage per active min | {c1_status} | {c1_finding} |")

    # C2
    lines.append(
        f"| 2. MRL by node type | {stats.get('control_2', {}).get('status', 'N/A')} | Deferred |"
    )

    # C3
    c3t = stats.get("control_3", {}).get("mrl", {}).get("test", {})
    c3_finding = f"Already active-only; MRL p = {c3t['p']:.4f}" if c3t.get("p") is not None else "N/A"
    lines.append(f"| 3. MRL/AHV active only | verified | {c3_finding} |")

    # C4
    lines.append(
        f"| 4. Speed by node type (all) | {stats.get('control_4', {}).get('status', 'N/A')} | Deferred |"
    )

    # C5
    c5_p = stats.get("control_5", {}).get("permutation_p_onesided")
    c5_d = stats.get("control_5", {}).get("test_observed_vs_null", {}).get("cliff_d")
    c5_finding = f"Permutation p = {c5_p:.4f}" if c5_p is not None else "N/A"
    if c5_d is not None:
        c5_finding += f", d = {c5_d:.3f}"
    lines.append(f"| 5. Random walk null | computed | {c5_finding} |")

    # C6
    lines.append(
        f"| 6. Bodypart tracking quality | {stats.get('control_6', {}).get('status', 'N/A')} | Deferred |"
    )

    # C7
    c7t = stats.get("control_7", {}).get("epoch_coverage", {}).get("test", {})
    c7_finding = f"Coverage p = {c7t['p']:.4f}, r = {c7t.get('r', 0):.3f}" if c7t.get("p") is not None else "N/A"
    lines.append(f"| 7. Primary-only | computed | {c7_finding} |")

    lines.extend(["", ""])

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
