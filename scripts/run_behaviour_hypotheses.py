#!/usr/bin/env python3
"""Run behaviour hypotheses from plan-behaviour-science.md.

Tier-1 (H1-H4): Tests how darkness affects maze navigation:
  H1: Speed-coverage partial correlation
  H2: Transition matrix changes (JSD + per-edge tests)
  H3: Spatial range contraction (per-cell-type coverage + diameter)
  H4: Increased revisitation (revisitation index + discovery AUC)

Tier-2 (H5, H6, H8-H10): Deeper mechanistic and spatial analyses:
  H5:  Within-epoch temporal dynamics (gradual vs immediate coverage loss)
  H6:  Corridor heatmap (per-cell coverage changes)
  H8:  Epoch-number adaptation (learning vs obligatory response)
  H9:  Individual differences (per-animal darkness sensitivity)
  H10: Cell-type Markov (3-state J/C/D second-order model)

Outputs:
  - docs/manuscripts/behaviour-hypotheses-results.json        (Tier-1)
  - docs/manuscripts/behaviour-hypotheses-tier2-results.json  (Tier-2)
  - Human-readable summary to stdout

Usage:
  python scripts/run_behaviour_hypotheses.py           # Tier-1 only
  python scripts/run_behaviour_hypotheses.py --tier2   # Tier-2 only
  python scripts/run_behaviour_hypotheses.py --all     # Both tiers
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
import h5py
import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import jensenshannon

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.maze.analysis import transition_matrix
from hm2p.maze.discretize import cell_sequence, discretize_position_fast
from hm2p.maze.topology import RoseMaze, build_rose_maze

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BUCKET = "hm2p-derivatives"
S3_REGION = "ap-southeast-2"
METADATA_CSV = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
ANIMALS_CSV = Path(__file__).resolve().parent.parent / "metadata" / "animals.csv"
OUTPUT_JSON = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "manuscripts"
    / "behaviour-hypotheses-results.json"
)

SPEED_ACTIVE_THRESHOLD = 2.5  # cm/s
MIN_EPOCH_DURATION_S = 30.0  # minimum epoch duration to include

# Tier-2 constants
HALF_EPOCH_S = 30.0  # split point within each epoch (seconds)
CUMULATIVE_BIN_S = 5.0  # bin width for cumulative coverage curves (seconds)
OUTPUT_JSON_T2 = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "manuscripts"
    / "behaviour-hypotheses-tier2-results.json"
)

# Cell-type indices for H10 Markov model
TYPE_J = 0  # junction
TYPE_C = 1  # corridor
TYPE_D = 2  # dead-end
N_TYPES = 3
TYPE_NAMES = ["J", "C", "D"]

MAZE = build_rose_maze()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_serializable(obj: object) -> object:
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
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def holm_bonferroni(p_values: list[float | None]) -> list[float]:
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    pvals = np.asarray([p if p is not None else 1.0 for p in p_values], dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    sorted_adj = np.array([min(pvals[order[rank]] * (n - rank), 1.0) for rank in range(n)])
    sorted_adj = np.maximum.accumulate(sorted_adj)
    adjusted = np.empty(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = sorted_adj[rank]
    return adjusted.tolist()


def rank_biserial_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank-biserial correlation for Wilcoxon signed-rank test."""
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    result = sp_stats.wilcoxon(x, y, alternative="two-sided")
    W = result.statistic
    r = 1.0 - (2.0 * W) / (n * (n + 1) / 2.0)
    return float(r)


def wilcoxon_test(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test with effect size."""
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


def spearman_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rank correlation with NaN filtering.

    Returns dict with keys: rho, p, n, test.
    Returns None values when n < 6.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 6:
        return {"rho": None, "p": None, "n": n, "test": "spearman_rank"}
    try:
        rho, p = sp_stats.spearmanr(x, y)
        return {
            "rho": float(rho),
            "p": float(p),
            "n": n,
            "test": "spearman_rank",
        }
    except Exception as e:
        return {"rho": None, "p": None, "n": n, "test": "spearman_rank", "error": str(e)}


def one_sample_wilcoxon(x: np.ndarray, alternative: str = "two-sided") -> dict:
    """One-sample Wilcoxon signed-rank test (test median differs from 0).

    Returns dict with keys: stat, p, r, n, median, mean, test.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return {
            "stat": None,
            "p": None,
            "r": None,
            "n": n,
            "median": float(np.median(x)) if n > 0 else None,
            "mean": float(np.mean(x)) if n > 0 else None,
            "test": "wilcoxon_one_sample",
        }
    try:
        result = sp_stats.wilcoxon(x, alternative=alternative)
        # Rank-biserial for one-sample: r = 1 - 2W / (n(n+1)/2)
        nonzero = x[x != 0]
        nn = len(nonzero)
        W = result.statistic
        r = 1.0 - (2.0 * W) / (nn * (nn + 1) / 2.0) if nn > 0 else 0.0
        return {
            "stat": float(result.statistic),
            "p": float(result.pvalue),
            "r": float(r),
            "n": n,
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "test": "wilcoxon_one_sample",
        }
    except Exception as e:
        return {
            "stat": None,
            "p": None,
            "r": None,
            "n": n,
            "median": float(np.median(x)) if n > 0 else None,
            "mean": float(np.mean(x)) if n > 0 else None,
            "error": str(e),
            "test": "wilcoxon_one_sample",
        }


def detect_epochs(light_on: np.ndarray, fps: float) -> list[dict]:
    """Detect contiguous light/dark epochs.

    Returns list of dicts with keys: start, end, condition, duration_s.
    """
    light = np.asarray(light_on, dtype=bool)
    n = len(light)
    epochs: list[dict] = []
    i = 0
    while i < n:
        condition = "light" if light[i] else "dark"
        start = i
        while i < n and light[i] == light[start]:
            i += 1
        dur = (i - start) / fps
        epochs.append({"start": start, "end": i, "condition": condition, "duration_s": dur})
    return epochs


# ---------------------------------------------------------------------------
# Load session data (same pattern as run_behaviour_analysis.py)
# ---------------------------------------------------------------------------


def load_session_data(s3_client: object, exp_id: str) -> dict | None:
    """Download sync.h5 and extract behavioural fields."""
    parts = exp_id.split("_")
    animal_id = parts[-1]
    ts = f"{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    sub = f"sub-{animal_id}"
    ses = f"ses-{ts}"
    key = f"sync/{sub}/{ses}/sync.h5"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmppath = tmp.name

    try:
        s3_client.download_file(S3_BUCKET, key, tmppath)
        with h5py.File(tmppath, "r") as f:
            if len(f.keys()) == 0:
                print(f"  STUB: {exp_id} -- empty sync.h5")
                return None

            fps = float(f.attrs.get("fps_imaging", 9.6))
            data: dict = {
                "exp_id": exp_id,
                "animal_id": animal_id,
                "sub": sub,
                "ses": ses,
                "fps": fps,
            }

            for field in [
                "x_mm",
                "y_mm",
                "x_maze",
                "y_maze",
                "speed_cm_s",
                "hd_deg",
                "light_on",
                "bad_behav",
                "frame_times",
            ]:
                if field in f:
                    data[field] = f[field][:]
                else:
                    data[field] = None

            return data
    except Exception as e:
        print(f"  ERROR loading {exp_id}: {e}")
        return None
    finally:
        if os.path.exists(tmppath):
            os.unlink(tmppath)


# ---------------------------------------------------------------------------
# H1: Speed-coverage partial correlation
# ---------------------------------------------------------------------------


def compute_h1_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute per-session speed and coverage for light/dark epochs.

    Returns dict with per-epoch speed/coverage lists and session-level
    differences (dark - light).
    """
    fps = data["fps"]
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    speed = data["speed_cm_s"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    epochs = detect_epochs(light_on, fps)

    # Per-epoch metrics
    epoch_speed_light: list[float] = []
    epoch_speed_dark: list[float] = []
    epoch_cov_light: list[float] = []
    epoch_cov_dark: list[float] = []

    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue
        sl = slice(ep["start"], ep["end"])
        ep_valid = valid[sl]
        ep_ci = cell_indices[sl].copy()
        ep_ci[~ep_valid] = -1
        ep_speed = speed[sl]

        # Coverage
        unique_in_epoch = len(set(int(c) for c in ep_ci if c >= 0))
        cov = unique_in_epoch / maze.n_cells

        # Mean speed (all valid frames, not just active)
        speed_mask = ep_valid & np.isfinite(ep_speed)
        mean_spd = float(np.nanmean(ep_speed[speed_mask])) if speed_mask.any() else np.nan

        if ep["condition"] == "light":
            epoch_speed_light.append(mean_spd)
            epoch_cov_light.append(cov)
        else:
            epoch_speed_dark.append(mean_spd)
            epoch_cov_dark.append(cov)

    # Session-level means
    mean_speed_light = float(np.nanmean(epoch_speed_light)) if epoch_speed_light else np.nan
    mean_speed_dark = float(np.nanmean(epoch_speed_dark)) if epoch_speed_dark else np.nan
    mean_cov_light = float(np.nanmean(epoch_cov_light)) if epoch_cov_light else np.nan
    mean_cov_dark = float(np.nanmean(epoch_cov_dark)) if epoch_cov_dark else np.nan

    # Coverage per transition: unique cells / total cell transitions per epoch.
    # This normalises for locomotion — a mouse making the same number of
    # cell transitions should discover the same number of unique cells if
    # its exploration efficiency is unchanged.
    epoch_cpt_light: list[float] = []
    epoch_cpt_dark: list[float] = []

    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue
        sl = slice(ep["start"], ep["end"])
        ep_valid = valid[sl]
        ep_ci = cell_indices[sl].copy()
        ep_ci[~ep_valid] = -1

        # Count cell transitions (consecutive frames in different cells)
        valid_cells = ep_ci[ep_ci >= 0]
        if len(valid_cells) < 2:
            continue
        transitions = int(np.sum(valid_cells[1:] != valid_cells[:-1]))
        if transitions == 0:
            continue
        unique = len(set(int(c) for c in valid_cells))
        cpt = unique / transitions

        if ep["condition"] == "light":
            epoch_cpt_light.append(cpt)
        else:
            epoch_cpt_dark.append(cpt)

    mean_cpt_light = float(np.nanmean(epoch_cpt_light)) if epoch_cpt_light else np.nan
    mean_cpt_dark = float(np.nanmean(epoch_cpt_dark)) if epoch_cpt_dark else np.nan

    return {
        "mean_speed_light": mean_speed_light,
        "mean_speed_dark": mean_speed_dark,
        "mean_cov_light": mean_cov_light,
        "mean_cov_dark": mean_cov_dark,
        "speed_diff": mean_speed_dark - mean_speed_light,
        "cov_diff": mean_cov_dark - mean_cov_light,
        "mean_cpt_light": mean_cpt_light,
        "mean_cpt_dark": mean_cpt_dark,
    }


def test_h1(session_results: list[dict]) -> dict:
    """Run H1: speed-coverage partial correlation across sessions."""
    speed_diffs = np.array([r["speed_diff"] for r in session_results])
    cov_diffs = np.array([r["cov_diff"] for r in session_results])

    # Filter out NaN
    valid = np.isfinite(speed_diffs) & np.isfinite(cov_diffs)
    speed_diffs_v = speed_diffs[valid]
    cov_diffs_v = cov_diffs[valid]
    n = len(speed_diffs_v)

    # Spearman correlation between speed-difference and coverage-difference
    if n >= 6:
        rho, p = sp_stats.spearmanr(speed_diffs_v, cov_diffs_v)
    else:
        rho, p = np.nan, np.nan

    # Coverage per transition: unique cells / total transitions.
    # Normalises for locomotion — tests whether exploration efficiency
    # (not just locomotor activity) differs between light and dark.
    cpt_light = np.array([r["mean_cpt_light"] for r in session_results])
    cpt_dark = np.array([r["mean_cpt_dark"] for r in session_results])
    cpt_valid = np.isfinite(cpt_light) & np.isfinite(cpt_dark)
    cpt_test = wilcoxon_test(cpt_light[cpt_valid], cpt_dark[cpt_valid])

    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "spearman_p": float(p) if np.isfinite(p) else None,
        "n_sessions": n,
        "coverage_per_transition": {
            "mean_light": float(np.nanmean(cpt_light[cpt_valid])),
            "mean_dark": float(np.nanmean(cpt_dark[cpt_valid])),
            "p": cpt_test.get("p"),
            "r": cpt_test.get("r"),
            "n_sessions": int(cpt_valid.sum()),
        },
    }


# ---------------------------------------------------------------------------
# H2: Transition matrix changes
# ---------------------------------------------------------------------------


def compute_h2_per_session(data: dict, maze: RoseMaze) -> dict:
    """Build light/dark transition matrices and compute JSD for one session."""
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    n_cells = maze.n_cells

    # Build cell sequences for light and dark
    ci_light = cell_indices.copy()
    ci_light[~(valid & light_on)] = -1
    cs_light, _ = cell_sequence(ci_light)

    ci_dark = cell_indices.copy()
    ci_dark[~(valid & ~light_on)] = -1
    cs_dark, _ = cell_sequence(ci_dark)

    if len(cs_light) < 10 or len(cs_dark) < 10:
        return {"jsd": np.nan, "tm_light": None, "tm_dark": None}

    tm_light = transition_matrix(cs_light, n_cells)
    tm_dark = transition_matrix(cs_dark, n_cells)

    # Jensen-Shannon divergence between the two transition matrices.
    # Compute JSD per row (per source cell) and take the weighted average
    # based on the empirical source-cell frequency (pooled light + dark).
    source_counts = np.zeros(n_cells)
    for c in cs_light[:-1]:
        if 0 <= c < n_cells:
            source_counts[c] += 1
    for c in cs_dark[:-1]:
        if 0 <= c < n_cells:
            source_counts[c] += 1
    total = source_counts.sum()
    if total == 0:
        return {"jsd": np.nan, "tm_light": tm_light, "tm_dark": tm_dark}

    weights = source_counts / total

    jsd_per_row = np.zeros(n_cells)
    for i in range(n_cells):
        row_l = tm_light[i]
        row_d = tm_dark[i]
        # Only compute JSD for rows with actual transitions
        if row_l.sum() > 0 and row_d.sum() > 0:
            # scipy jensenshannon returns the distance (sqrt of divergence)
            # We want the divergence itself
            js_dist = jensenshannon(row_l, row_d, base=2)
            jsd_per_row[i] = js_dist**2  # JSD = distance^2
        else:
            jsd_per_row[i] = 0.0

    weighted_jsd = float(np.sum(weights * jsd_per_row))

    # Per-edge transition probabilities (for H2 per-edge analysis)
    edge_probs: dict[str, dict] = {}
    for i in range(n_cells):
        for j in range(n_cells):
            # Only track edges that have transitions in at least one condition
            if tm_light[i, j] > 0 or tm_dark[i, j] > 0:
                cell_i = maze.cell_list[i]
                cell_j = maze.cell_list[j]
                # Only include edges between adjacent cells
                if cell_j in maze.adj.get(cell_i, []):
                    edge_key = f"{cell_i}->{cell_j}"
                    edge_probs[edge_key] = {
                        "from_cell": list(cell_i),
                        "to_cell": list(cell_j),
                        "p_light": float(tm_light[i, j]),
                        "p_dark": float(tm_dark[i, j]),
                    }

    # Store raw transition counts for permutation null in test_h2
    tc_light = np.zeros((n_cells, n_cells), dtype=np.float64)
    for k in range(len(cs_light) - 1):
        a, b = cs_light[k], cs_light[k + 1]
        if 0 <= a < n_cells and 0 <= b < n_cells:
            tc_light[a, b] += 1
    tc_dark = np.zeros((n_cells, n_cells), dtype=np.float64)
    for k in range(len(cs_dark) - 1):
        a, b = cs_dark[k], cs_dark[k + 1]
        if 0 <= a < n_cells and 0 <= b < n_cells:
            tc_dark[a, b] += 1

    return {
        "jsd": weighted_jsd,
        "tm_light": tm_light,
        "tm_dark": tm_dark,
        "edge_probs": edge_probs,
        "transition_counts_light": tc_light.tolist(),
        "transition_counts_dark": tc_dark.tolist(),
    }


def test_h2(session_results: list[dict], n_permutations: int = 1000) -> dict:
    """Run H2: transition matrix changes across sessions.

    Uses a permutation null: for each permutation, shuffle light/dark
    epoch labels within each session and recompute JSD. The observed
    mean JSD is compared to this null distribution.
    """
    jsd_values = np.array([r["jsd"] for r in session_results])
    valid_jsd = jsd_values[np.isfinite(jsd_values)]
    n = len(valid_jsd)

    observed_mean_jsd = float(np.mean(valid_jsd)) if n > 0 else np.nan

    # Permutation null: shuffle light/dark epoch labels within each session.
    # Each session stores its per-epoch cell sequences; we recompute JSD
    # after shuffling which epochs are called "light" vs "dark".
    # Since we don't store raw epoch sequences, use the per-session
    # light and dark transition counts to create a pooled matrix, then
    # randomly split the pooled counts into two halves and compute JSD.
    rng = np.random.default_rng(42)
    null_mean_jsds: list[float] = []

    for _ in range(n_permutations):
        perm_jsds: list[float] = []
        for r in session_results:
            # Pool light and dark transition counts
            tl = r.get("transition_counts_light")
            td = r.get("transition_counts_dark")
            if tl is None or td is None:
                continue
            tl_arr = np.array(tl, dtype=np.float64)
            td_arr = np.array(td, dtype=np.float64)
            pooled = tl_arr + td_arr
            total = pooled.sum()
            if total < 2:
                continue
            # Randomly split pooled counts into two halves
            flat = pooled.ravel()
            half = int(total // 2)
            perm_flat = np.zeros_like(flat)
            indices = []
            for idx, count in enumerate(flat):
                indices.extend([idx] * int(count))
            rng.shuffle(indices)
            for idx in indices[:half]:
                perm_flat[idx] += 1
            perm_a = perm_flat.reshape(pooled.shape)
            perm_b = pooled - perm_a
            # Compute JSD
            from scipy.spatial.distance import jensenshannon

            jsd_vals = []
            for row_idx in range(perm_a.shape[0]):
                row_a = perm_a[row_idx]
                row_b = perm_b[row_idx]
                s = row_a.sum() + row_b.sum()
                if s == 0:
                    continue
                p_a = row_a / max(row_a.sum(), 1)
                p_b = row_b / max(row_b.sum(), 1)
                js = jensenshannon(p_a, p_b) ** 2  # squared to match JSD convention
                if np.isfinite(js):
                    jsd_vals.append(js * row_a.sum() / max(total, 1))
            perm_jsds.append(sum(jsd_vals) if jsd_vals else 0.0)
        if perm_jsds:
            null_mean_jsds.append(float(np.mean(perm_jsds)))

    # Permutation p-value: fraction of null >= observed
    if null_mean_jsds:
        null_arr = np.array(null_mean_jsds)
        perm_p = float((null_arr >= observed_mean_jsd).mean())
        null_mean = float(np.mean(null_arr))
        null_95 = float(np.percentile(null_arr, 95))
    else:
        perm_p = np.nan
        null_mean = np.nan
        null_95 = np.nan

    jsd_test = {
        "observed_mean_jsd": observed_mean_jsd,
        "permutation_p": perm_p,
        "null_mean": null_mean,
        "null_95_pct": null_95,
        "n_permutations": n_permutations,
        "n": n,
        "test": "permutation",
    }

    # Per-edge comparison: collect light/dark probabilities across sessions
    # for each edge and run Wilcoxon signed-rank
    edge_data: dict[str, dict] = {}
    for r in session_results:
        ep = r.get("edge_probs", {})
        for edge_key, vals in ep.items():
            if edge_key not in edge_data:
                edge_data[edge_key] = {
                    "from_cell": vals["from_cell"],
                    "to_cell": vals["to_cell"],
                    "p_light_list": [],
                    "p_dark_list": [],
                }
            edge_data[edge_key]["p_light_list"].append(vals["p_light"])
            edge_data[edge_key]["p_dark_list"].append(vals["p_dark"])

    # Test each edge with sufficient data
    edge_results: list[dict] = []
    edge_p_values: list[float] = []

    for edge_key, ed in edge_data.items():
        pl = np.array(ed["p_light_list"])
        pd_arr = np.array(ed["p_dark_list"])
        n_edge = len(pl)
        if n_edge < 6:
            continue
        # Only test if there is non-zero variance in the difference
        diff = pd_arr - pl
        if np.all(diff == 0):
            continue
        try:
            w_res = sp_stats.wilcoxon(pl, pd_arr, alternative="two-sided")
            r_val = rank_biserial_wilcoxon(pl, pd_arr)
            direction = "dark>light" if np.mean(diff) > 0 else "light>dark"
            edge_results.append(
                {
                    "edge": edge_key,
                    "from": ed["from_cell"],
                    "to": ed["to_cell"],
                    "p_raw": float(w_res.pvalue),
                    "r": r_val,
                    "n": n_edge,
                    "mean_light": float(np.mean(pl)),
                    "mean_dark": float(np.mean(pd_arr)),
                    "direction": direction,
                }
            )
            edge_p_values.append(float(w_res.pvalue))
        except Exception:
            continue

    # Holm-Bonferroni correction across edges
    if edge_p_values:
        adjusted = holm_bonferroni(edge_p_values)
        for i, er in enumerate(edge_results):
            er["p_adj"] = adjusted[i]
    significant_edges = [
        er for er in edge_results if er.get("p_adj") is not None and er["p_adj"] < 0.05
    ]

    return {
        "mean_jsd": float(np.mean(valid_jsd)) if len(valid_jsd) > 0 else None,
        "median_jsd": float(np.median(valid_jsd)) if len(valid_jsd) > 0 else None,
        "jsd_test": jsd_test,
        "n_edges_tested": len(edge_results),
        "n_significant_edges": len(significant_edges),
        "significant_edges": significant_edges,
        "per_session_jsd": [float(j) if np.isfinite(j) else None for j in jsd_values],
    }


# ---------------------------------------------------------------------------
# H3: Spatial range contraction
# ---------------------------------------------------------------------------


def _classify_cell_type(cell: tuple[int, int], maze: RoseMaze) -> str:
    """Classify a cell as junction, corridor, or dead_end."""
    nt = maze.node_types.get(cell, "unknown")
    if nt in ("t_junction", "crossroads"):
        return "junction"
    elif nt == "corridor":
        return "corridor"
    elif nt == "dead_end":
        return "dead_end"
    return "unknown"


def _visited_subgraph_diameter(visited_cell_indices: set[int], maze: RoseMaze) -> int:
    """Compute diameter of the subgraph induced by visited cells.

    Diameter = longest shortest path between any two visited cells
    in the full maze graph. This uses the precomputed distance matrix.
    """
    if len(visited_cell_indices) < 2:
        return 0
    idx_list = sorted(visited_cell_indices)
    max_dist = 0
    for i in range(len(idx_list)):
        for j in range(i + 1, len(idx_list)):
            d = maze.dist[idx_list[i], idx_list[j]]
            if d > max_dist:
                max_dist = d
    return int(max_dist)


def compute_h3_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute per-cell-type coverage and diameter for light/dark epochs."""
    fps = data["fps"]
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    # Classify cells by type
    junction_indices = {maze.cell_to_idx[c] for c in maze.junctions}
    corridor_indices = {maze.cell_to_idx[c] for c in maze.corridors}
    dead_end_indices = {maze.cell_to_idx[c] for c in maze.dead_ends}

    n_junctions = len(junction_indices)
    n_corridors = len(corridor_indices)
    n_dead_ends = len(dead_end_indices)

    epochs = detect_epochs(light_on, fps)

    # Per-epoch per-cell-type coverage and diameter
    junc_cov_light: list[float] = []
    junc_cov_dark: list[float] = []
    corr_cov_light: list[float] = []
    corr_cov_dark: list[float] = []
    de_cov_light: list[float] = []
    de_cov_dark: list[float] = []
    diam_light: list[int] = []
    diam_dark: list[int] = []

    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue
        sl = slice(ep["start"], ep["end"])
        ep_valid = valid[sl]
        ep_ci = cell_indices[sl].copy()
        ep_ci[~ep_valid] = -1

        visited = set(int(c) for c in ep_ci if c >= 0)

        junc_visited = visited & junction_indices
        corr_visited = visited & corridor_indices
        de_visited = visited & dead_end_indices

        jc = len(junc_visited) / n_junctions if n_junctions > 0 else 0.0
        cc = len(corr_visited) / n_corridors if n_corridors > 0 else 0.0
        dc = len(de_visited) / n_dead_ends if n_dead_ends > 0 else 0.0

        diameter = _visited_subgraph_diameter(visited, maze)

        if ep["condition"] == "light":
            junc_cov_light.append(jc)
            corr_cov_light.append(cc)
            de_cov_light.append(dc)
            diam_light.append(diameter)
        else:
            junc_cov_dark.append(jc)
            corr_cov_dark.append(cc)
            de_cov_dark.append(dc)
            diam_dark.append(diameter)

    return {
        "mean_junc_cov_light": float(np.mean(junc_cov_light)) if junc_cov_light else np.nan,
        "mean_junc_cov_dark": float(np.mean(junc_cov_dark)) if junc_cov_dark else np.nan,
        "mean_corr_cov_light": float(np.mean(corr_cov_light)) if corr_cov_light else np.nan,
        "mean_corr_cov_dark": float(np.mean(corr_cov_dark)) if corr_cov_dark else np.nan,
        "mean_de_cov_light": float(np.mean(de_cov_light)) if de_cov_light else np.nan,
        "mean_de_cov_dark": float(np.mean(de_cov_dark)) if de_cov_dark else np.nan,
        "mean_diam_light": float(np.mean(diam_light)) if diam_light else np.nan,
        "mean_diam_dark": float(np.mean(diam_dark)) if diam_dark else np.nan,
    }


def test_h3(session_results: list[dict]) -> dict:
    """Run H3: spatial range contraction across sessions."""
    # Per-cell-type coverage light vs dark
    jl = np.array([r["mean_junc_cov_light"] for r in session_results])
    jd = np.array([r["mean_junc_cov_dark"] for r in session_results])
    cl = np.array([r["mean_corr_cov_light"] for r in session_results])
    cd = np.array([r["mean_corr_cov_dark"] for r in session_results])
    dl = np.array([r["mean_de_cov_light"] for r in session_results])
    dd = np.array([r["mean_de_cov_dark"] for r in session_results])

    junc_test = wilcoxon_test(jl, jd)
    corr_test = wilcoxon_test(cl, cd)
    de_test = wilcoxon_test(dl, dd)

    # Holm-Bonferroni across 3 cell-type tests
    cell_type_pvals = [junc_test.get("p"), corr_test.get("p"), de_test.get("p")]
    adjusted = holm_bonferroni(cell_type_pvals)

    # Test whether dead-end coverage drop is larger than junction coverage drop
    # Compute per-session coverage drop (light - dark, positive = drop)
    junc_drop = jl - jd  # positive = light > dark = drop in dark
    de_drop = dl - dd
    # Test: is dead-end drop > junction drop?
    valid_interaction = np.isfinite(junc_drop) & np.isfinite(de_drop)
    if valid_interaction.sum() >= 6:
        interaction_test = wilcoxon_test(de_drop[valid_interaction], junc_drop[valid_interaction])
    else:
        interaction_test = {"stat": None, "p": None, "r": None, "n": 0}

    # Diameter
    diam_l = np.array([r["mean_diam_light"] for r in session_results])
    diam_d = np.array([r["mean_diam_dark"] for r in session_results])
    diam_test = wilcoxon_test(diam_l, diam_d)

    # Compute summary values for output
    def _safe_mean(arr: np.ndarray) -> float | None:
        v = arr[np.isfinite(arr)]
        return float(np.mean(v)) if len(v) > 0 else None

    return {
        "junction_coverage": {
            "light": _safe_mean(jl),
            "dark": _safe_mean(jd),
            "p": junc_test.get("p"),
            "p_adj": adjusted[0],
            "r": junc_test.get("r"),
            "n": junc_test.get("n"),
        },
        "corridor_coverage": {
            "light": _safe_mean(cl),
            "dark": _safe_mean(cd),
            "p": corr_test.get("p"),
            "p_adj": adjusted[1],
            "r": corr_test.get("r"),
            "n": corr_test.get("n"),
        },
        "dead_end_coverage": {
            "light": _safe_mean(dl),
            "dark": _safe_mean(dd),
            "p": de_test.get("p"),
            "p_adj": adjusted[2],
            "r": de_test.get("r"),
            "n": de_test.get("n"),
        },
        "de_vs_junction_drop_interaction": {
            "mean_de_drop": _safe_mean(de_drop),
            "mean_junc_drop": _safe_mean(junc_drop),
            "p": interaction_test.get("p"),
            "r": interaction_test.get("r"),
            "n": interaction_test.get("n"),
        },
        "diameter": {
            "light": _safe_mean(diam_l),
            "dark": _safe_mean(diam_d),
            "p": diam_test.get("p"),
            "r": diam_test.get("r"),
            "n": diam_test.get("n"),
        },
    }


# ---------------------------------------------------------------------------
# H4: Increased revisitation
# ---------------------------------------------------------------------------


def compute_h4_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute revisitation index and discovery AUC for light/dark epochs."""
    fps = data["fps"]
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    epochs = detect_epochs(light_on, fps)

    revis_light: list[float] = []
    revis_dark: list[float] = []
    auc_light: list[float] = []
    auc_dark: list[float] = []

    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue
        sl = slice(ep["start"], ep["end"])
        ep_valid = valid[sl]
        ep_ci = cell_indices[sl].copy()
        ep_ci[~ep_valid] = -1

        # Get cell transition sequence (no consecutive duplicates)
        cs, _ = cell_sequence(ep_ci)
        if len(cs) < 2:
            continue

        # Revisitation index = total transitions / unique cells visited
        n_transitions = len(cs) - 1
        unique_cells = len(set(int(c) for c in cs))
        if unique_cells == 0:
            continue
        revis_idx = n_transitions / unique_cells

        # Discovery curve: cumulative unique cells as function of transition number
        seen: set[int] = set()
        cum_unique: list[int] = []
        for c in cs:
            seen.add(int(c))
            cum_unique.append(len(seen))

        # AUC of the discovery curve, normalised by the maximum possible AUC
        # Max AUC = n_transitions * maze.n_cells (if all cells found at step 0)
        # Actual AUC = sum of cumulative unique counts at each step
        # We normalise by n_steps * n_cells so AUC is in [0, 1]
        n_steps = len(cum_unique)
        raw_auc = float(np.sum(cum_unique))
        normalised_auc = raw_auc / (n_steps * maze.n_cells) if n_steps > 0 else 0.0

        if ep["condition"] == "light":
            revis_light.append(revis_idx)
            auc_light.append(normalised_auc)
        else:
            revis_dark.append(revis_idx)
            auc_dark.append(normalised_auc)

    return {
        "mean_revis_light": float(np.mean(revis_light)) if revis_light else np.nan,
        "mean_revis_dark": float(np.mean(revis_dark)) if revis_dark else np.nan,
        "mean_auc_light": float(np.mean(auc_light)) if auc_light else np.nan,
        "mean_auc_dark": float(np.mean(auc_dark)) if auc_dark else np.nan,
    }


def test_h4(session_results: list[dict]) -> dict:
    """Run H4: revisitation index and discovery AUC across sessions."""
    rl = np.array([r["mean_revis_light"] for r in session_results])
    rd = np.array([r["mean_revis_dark"] for r in session_results])
    al = np.array([r["mean_auc_light"] for r in session_results])
    ad = np.array([r["mean_auc_dark"] for r in session_results])

    revis_test = wilcoxon_test(rl, rd)
    auc_test = wilcoxon_test(al, ad)

    def _safe_mean(arr: np.ndarray) -> float | None:
        v = arr[np.isfinite(arr)]
        return float(np.mean(v)) if len(v) > 0 else None

    return {
        "revisitation_index": {
            "light": _safe_mean(rl),
            "dark": _safe_mean(rd),
            "p": revis_test.get("p"),
            "r": revis_test.get("r"),
            "n": revis_test.get("n"),
        },
        "discovery_auc": {
            "light": _safe_mean(al),
            "dark": _safe_mean(ad),
            "p": auc_test.get("p"),
            "r": auc_test.get("r"),
            "n": auc_test.get("n"),
        },
    }


# ===========================================================================
# TIER-2 HYPOTHESES (H5, H6, H8, H9, H10)
# ===========================================================================


# ---------------------------------------------------------------------------
# H5: Within-epoch temporal dynamics
# ---------------------------------------------------------------------------


def compute_h5_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute within-epoch temporal dynamics for one session.

    Splits each epoch at 30 s into first/second halves. Measures new
    unique cells discovered in each half, cumulative coverage curves
    in 5-second bins, and lights-on recovery.
    """
    fps = data["fps"]
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    speed = data["speed_cm_s"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    epochs = detect_epochs(light_on, fps)

    coverage_ratio_light: list[float] = []
    coverage_ratio_dark: list[float] = []
    speed_ratio_light: list[float] = []
    speed_ratio_dark: list[float] = []
    unique_1st_light: list[float] = []
    unique_1st_dark: list[float] = []
    new_2nd_light: list[float] = []
    new_2nd_dark: list[float] = []

    # Cumulative coverage curves (per condition, variable length)
    cum_curves_light: list[list[float]] = []
    cum_curves_dark: list[list[float]] = []

    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue
        sl = slice(ep["start"], ep["end"])
        ep_valid = valid[sl]
        ep_ci = cell_indices[sl].copy()
        ep_ci[~ep_valid] = -1
        ep_speed = speed[sl]

        # Midpoint frame
        mid_offset = int(round(HALF_EPOCH_S * fps))
        ep_len = ep["end"] - ep["start"]
        mid_offset = min(mid_offset, ep_len)

        # First half unique cells
        first_half_ci = ep_ci[:mid_offset]
        first_half_valid = ep_valid[:mid_offset]
        first_cells = set(int(c) for c in first_half_ci if c >= 0)

        # Second half NEW cells (not seen in first half)
        second_half_ci = ep_ci[mid_offset:]
        second_cells = set(int(c) for c in second_half_ci if c >= 0)
        new_cells = second_cells - first_cells

        n_first = len(first_cells)
        n_new = len(new_cells)

        # Coverage ratio
        cov_ratio = n_new / n_first if n_first > 0 else np.nan

        # Speed in each half
        sp_1st = ep_speed[:mid_offset]
        sp_1st_valid = first_half_valid & np.isfinite(sp_1st)
        sp_2nd = ep_speed[mid_offset:]
        sp_2nd_valid = ep_valid[mid_offset:] & np.isfinite(sp_2nd)
        mean_sp_1st = float(np.nanmean(sp_1st[sp_1st_valid])) if sp_1st_valid.any() else np.nan
        mean_sp_2nd = float(np.nanmean(sp_2nd[sp_2nd_valid])) if sp_2nd_valid.any() else np.nan
        sp_ratio = mean_sp_2nd / mean_sp_1st if mean_sp_1st > 0 else np.nan

        # Cumulative coverage curve in CUMULATIVE_BIN_S bins
        bin_frames = int(round(CUMULATIVE_BIN_S * fps))
        n_bins = max(1, ep_len // bin_frames)
        seen: set[int] = set()
        curve: list[float] = []
        for b in range(n_bins):
            b_start = b * bin_frames
            b_end = min((b + 1) * bin_frames, ep_len)
            for fi in range(b_start, b_end):
                c = ep_ci[fi]
                if c >= 0:
                    seen.add(int(c))
            curve.append(len(seen) / maze.n_cells)

        if ep["condition"] == "light":
            coverage_ratio_light.append(cov_ratio)
            speed_ratio_light.append(sp_ratio)
            unique_1st_light.append(float(n_first))
            new_2nd_light.append(float(n_new))
            cum_curves_light.append(curve)
        else:
            coverage_ratio_dark.append(cov_ratio)
            speed_ratio_dark.append(sp_ratio)
            unique_1st_dark.append(float(n_first))
            new_2nd_dark.append(float(n_new))
            cum_curves_dark.append(curve)

    # Aggregate cumulative curves to fixed length (min bins across epochs)
    def _aggregate_curves(curves: list[list[float]]) -> tuple[list[float], list[float], int]:
        if not curves:
            return [], [], 0
        min_len = min(len(c) for c in curves)
        if min_len == 0:
            return [], [], 0
        arr = np.array([c[:min_len] for c in curves])
        mean_curve = np.mean(arr, axis=0).tolist()
        sem_curve = (
            (np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])).tolist()
            if arr.shape[0] > 1
            else [0.0] * min_len
        )
        return mean_curve, sem_curve, min_len

    cum_mean_light, cum_sem_light, n_bins_light = _aggregate_curves(cum_curves_light)
    cum_mean_dark, cum_sem_dark, n_bins_dark = _aggregate_curves(cum_curves_dark)
    n_bins_out = (
        min(n_bins_light, n_bins_dark)
        if (n_bins_light > 0 and n_bins_dark > 0)
        else max(n_bins_light, n_bins_dark)
    )

    # Lights-on recovery: compare first 30s of recovery light epochs
    # (light epochs immediately following dark) vs initial light epoch
    filtered_epochs = [ep for ep in epochs if ep["duration_s"] >= MIN_EPOCH_DURATION_S]
    recovery_covs: list[float] = []
    initial_cov: float = np.nan

    for i, ep in enumerate(filtered_epochs):
        if ep["condition"] == "light":
            sl = slice(ep["start"], ep["end"])
            ep_ci = cell_indices[sl].copy()
            ep_valid_mask = valid[sl]
            ep_ci[~ep_valid_mask] = -1
            mid_offset_r = min(int(round(HALF_EPOCH_S * fps)), ep["end"] - ep["start"])
            first_half = ep_ci[:mid_offset_r]
            n_unique = len(set(int(c) for c in first_half if c >= 0))
            cov_frac = n_unique / maze.n_cells

            if i == 0:
                initial_cov = cov_frac
            elif i > 0 and filtered_epochs[i - 1]["condition"] == "dark":
                recovery_covs.append(cov_frac)

    recovery_cov = float(np.median(recovery_covs)) if recovery_covs else np.nan

    return {
        "median_coverage_ratio_light": float(np.nanmedian(coverage_ratio_light))
        if coverage_ratio_light
        else np.nan,
        "median_coverage_ratio_dark": float(np.nanmedian(coverage_ratio_dark))
        if coverage_ratio_dark
        else np.nan,
        "median_speed_ratio_light": float(np.nanmedian(speed_ratio_light))
        if speed_ratio_light
        else np.nan,
        "median_speed_ratio_dark": float(np.nanmedian(speed_ratio_dark))
        if speed_ratio_dark
        else np.nan,
        "mean_unique_1st_light": float(np.mean(unique_1st_light)) if unique_1st_light else np.nan,
        "mean_unique_1st_dark": float(np.mean(unique_1st_dark)) if unique_1st_dark else np.nan,
        "mean_new_2nd_light": float(np.mean(new_2nd_light)) if new_2nd_light else np.nan,
        "mean_new_2nd_dark": float(np.mean(new_2nd_dark)) if new_2nd_dark else np.nan,
        "cumulative_curve_light": cum_mean_light[:n_bins_out] if n_bins_out > 0 else [],
        "cumulative_curve_dark": cum_mean_dark[:n_bins_out] if n_bins_out > 0 else [],
        "cumulative_sem_light": cum_sem_light[:n_bins_out] if n_bins_out > 0 else [],
        "cumulative_sem_dark": cum_sem_dark[:n_bins_out] if n_bins_out > 0 else [],
        "n_bins": n_bins_out,
        "recovery_cov_first_half": recovery_cov,
        "initial_cov_first_half": initial_cov,
        "n_light_epochs": len(coverage_ratio_light),
        "n_dark_epochs": len(coverage_ratio_dark),
    }


def test_h5(session_results: list[dict]) -> dict:
    """Run H5: within-epoch temporal dynamics across sessions."""
    cov_ratio_light = np.array([r["median_coverage_ratio_light"] for r in session_results])
    cov_ratio_dark = np.array([r["median_coverage_ratio_dark"] for r in session_results])
    sp_ratio_light = np.array([r["median_speed_ratio_light"] for r in session_results])
    sp_ratio_dark = np.array([r["median_speed_ratio_dark"] for r in session_results])

    # Primary test: coverage ratio dark vs light
    cov_ratio_test = wilcoxon_test(cov_ratio_light, cov_ratio_dark)

    # Speed ratio control
    speed_ratio_test = wilcoxon_test(sp_ratio_light, sp_ratio_dark)

    # Partial correlation control: coverage_ratio_diff vs speed_ratio_diff
    cov_diff = cov_ratio_dark - cov_ratio_light
    sp_diff = sp_ratio_dark - sp_ratio_light
    speed_cov_corr = spearman_test(cov_diff, sp_diff)

    # Lights-on recovery
    recovery = np.array([r["recovery_cov_first_half"] for r in session_results])
    initial = np.array([r["initial_cov_first_half"] for r in session_results])
    recovery_test = wilcoxon_test(recovery, initial)

    # Holm-Bonferroni across 3 primary tests
    raw_pvals = [cov_ratio_test.get("p"), speed_ratio_test.get("p"), recovery_test.get("p")]
    adjusted = holm_bonferroni(raw_pvals)

    # Grand-mean cumulative curves across sessions
    def _grand_mean_curve(
        key_mean: str, key_sem: str, results: list[dict]
    ) -> tuple[list[float], list[float], int]:
        curves = [r[key_mean] for r in results if r[key_mean]]
        if not curves:
            return [], [], 0
        min_len = min(len(c) for c in curves)
        if min_len == 0:
            return [], [], 0
        arr = np.array([c[:min_len] for c in curves])
        grand_mean = np.mean(arr, axis=0).tolist()
        grand_sem = (
            (np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])).tolist()
            if arr.shape[0] > 1
            else [0.0] * min_len
        )
        return grand_mean, grand_sem, min_len

    grand_light, grand_sem_light, n_bins_l = _grand_mean_curve(
        "cumulative_curve_light", "cumulative_sem_light", session_results
    )
    grand_dark, grand_sem_dark, n_bins_d = _grand_mean_curve(
        "cumulative_curve_dark", "cumulative_sem_dark", session_results
    )
    n_bins = (
        min(n_bins_l, n_bins_d) if (n_bins_l > 0 and n_bins_d > 0) else max(n_bins_l, n_bins_d)
    )

    def _safe_mean(arr: np.ndarray) -> float | None:
        v = arr[np.isfinite(arr)]
        return float(np.mean(v)) if len(v) > 0 else None

    # Interpretation
    p_adj_cov = adjusted[0] if len(adjusted) > 0 else None
    if p_adj_cov is not None and p_adj_cov < 0.05:
        mean_dark = _safe_mean(cov_ratio_dark)
        mean_light = _safe_mean(cov_ratio_light)
        if mean_dark is not None and mean_light is not None and mean_dark < mean_light:
            interpretation = "gradual"
        else:
            interpretation = "inconclusive"
    else:
        interpretation = "inconclusive"

    return {
        "coverage_ratio_test": {
            "mean_light": _safe_mean(cov_ratio_light),
            "mean_dark": _safe_mean(cov_ratio_dark),
            "p": cov_ratio_test.get("p"),
            "p_adj": adjusted[0] if len(adjusted) > 0 else None,
            "r": cov_ratio_test.get("r"),
            "n": cov_ratio_test.get("n"),
            "test": "wilcoxon_signed_rank",
        },
        "speed_ratio_test": {
            "mean_light": _safe_mean(sp_ratio_light),
            "mean_dark": _safe_mean(sp_ratio_dark),
            "p": speed_ratio_test.get("p"),
            "p_adj": adjusted[1] if len(adjusted) > 1 else None,
            "r": speed_ratio_test.get("r"),
            "n": speed_ratio_test.get("n"),
        },
        "speed_coverage_correlation": speed_cov_corr,
        "recovery_test": {
            "mean_recovery": _safe_mean(recovery),
            "mean_initial": _safe_mean(initial),
            "p": recovery_test.get("p"),
            "p_adj": adjusted[2] if len(adjusted) > 2 else None,
            "r": recovery_test.get("r"),
            "n": recovery_test.get("n"),
        },
        "grand_cumulative_light": grand_light[:n_bins],
        "grand_cumulative_dark": grand_dark[:n_bins],
        "grand_cumulative_sem_light": grand_sem_light[:n_bins],
        "grand_cumulative_sem_dark": grand_sem_dark[:n_bins],
        "n_bins": n_bins,
        "bin_width_s": CUMULATIVE_BIN_S,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# H6: Corridor-specific coverage heatmap
# ---------------------------------------------------------------------------


def compute_h6_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute per-cell visit fraction in light/dark for one session."""
    fps = data["fps"]
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    epochs = detect_epochs(light_on, fps)
    n_cells = maze.n_cells

    # Per-cell: fraction of epochs in which cell was visited
    visit_count_light = np.zeros(n_cells, dtype=np.float64)
    visit_count_dark = np.zeros(n_cells, dtype=np.float64)
    n_light_epochs = 0
    n_dark_epochs = 0

    # Per-cell: entries per minute (visit rate)
    entries_light = np.zeros(n_cells, dtype=np.float64)
    entries_dark = np.zeros(n_cells, dtype=np.float64)
    total_duration_light_min = 0.0
    total_duration_dark_min = 0.0

    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue

        sl = slice(ep["start"], ep["end"])
        ep_valid = valid[sl]
        ep_ci = cell_indices[sl].copy()
        ep_ci[~ep_valid] = -1

        # Which cells were visited in this epoch?
        visited = set(int(c) for c in ep_ci if c >= 0)

        # Cell sequence for entry counting
        cs, _ = cell_sequence(ep_ci)

        # Count entries per cell (appearances in the cell sequence)
        entry_counts = np.zeros(n_cells, dtype=np.float64)
        for c in cs:
            if 0 <= c < n_cells:
                entry_counts[c] += 1

        if ep["condition"] == "light":
            n_light_epochs += 1
            total_duration_light_min += ep["duration_s"] / 60.0
            for c_idx in visited:
                visit_count_light[c_idx] += 1
            entries_light += entry_counts
        else:
            n_dark_epochs += 1
            total_duration_dark_min += ep["duration_s"] / 60.0
            for c_idx in visited:
                visit_count_dark[c_idx] += 1
            entries_dark += entry_counts

    # Convert to fractions
    visit_frac_light = (
        (visit_count_light / n_light_epochs).tolist() if n_light_epochs > 0 else [0.0] * n_cells
    )
    visit_frac_dark = (
        (visit_count_dark / n_dark_epochs).tolist() if n_dark_epochs > 0 else [0.0] * n_cells
    )

    # Visit rates (entries/min)
    visit_rate_light = (
        (entries_light / total_duration_light_min).tolist()
        if total_duration_light_min > 0
        else [0.0] * n_cells
    )
    visit_rate_dark = (
        (entries_dark / total_duration_dark_min).tolist()
        if total_duration_dark_min > 0
        else [0.0] * n_cells
    )

    return {
        "visit_frac_light": visit_frac_light,
        "visit_frac_dark": visit_frac_dark,
        "visit_rate_light": visit_rate_light,
        "visit_rate_dark": visit_rate_dark,
    }


def test_h6(session_results: list[dict], maze: RoseMaze) -> dict:
    """Run H6: per-cell coverage tests and eccentricity correlation."""
    n_cells = maze.n_cells

    # Collect per-cell visit fractions across sessions
    frac_light = np.array([r["visit_frac_light"] for r in session_results])  # (S, C)
    frac_dark = np.array([r["visit_frac_dark"] for r in session_results])

    # Per-cell Wilcoxon across sessions
    per_cell_results: list[dict] = []
    raw_p_values: list[float | None] = []

    for c_idx in range(n_cells):
        cell = maze.cell_list[c_idx]
        node_type = maze.node_types.get(cell, "unknown")
        fl = frac_light[:, c_idx]
        fd = frac_dark[:, c_idx]
        delta = fd - fl  # dark - light (negative = less visited in dark)

        mean_fl = float(np.nanmean(fl))
        mean_fd = float(np.nanmean(fd))
        mean_delta = float(np.nanmean(delta))

        wt = wilcoxon_test(fl, fd)

        per_cell_results.append(
            {
                "cell": list(cell),
                "cell_idx": c_idx,
                "node_type": node_type,
                "mean_visit_frac_light": mean_fl,
                "mean_visit_frac_dark": mean_fd,
                "mean_delta": mean_delta,
                "p_raw": wt.get("p"),
                "r": wt.get("r"),
                "n": wt.get("n"),
            }
        )
        raw_p_values.append(wt.get("p"))

    # Holm-Bonferroni across 23 cells
    adjusted = holm_bonferroni(raw_p_values)
    for i, pcr in enumerate(per_cell_results):
        pcr["p_adj"] = adjusted[i] if i < len(adjusted) else None

    # Eccentricity correlation (corridors only, descriptive)
    # Eccentricity of cell c = max(maze.dist[c, :])
    eccentricities = np.array([int(np.max(maze.dist[c_idx])) for c_idx in range(n_cells)])
    corridor_indices = [maze.cell_to_idx[c] for c in maze.corridors]
    corridor_eccentricities = eccentricities[corridor_indices].astype(float)
    corridor_deltas = np.array([per_cell_results[ci]["mean_delta"] for ci in corridor_indices])

    ecc_corr = spearman_test(corridor_eccentricities, corridor_deltas)

    # Distance-from-center correlation (all cells)
    # Center = cell with minimum eccentricity
    center_idx = int(np.argmin(eccentricities))
    distances_from_center = maze.dist[center_idx].astype(float)
    all_deltas = np.array([pcr["mean_delta"] for pcr in per_cell_results])

    dist_corr = spearman_test(distances_from_center, all_deltas)

    # Heatmap data
    heatmap_cells = [list(maze.cell_list[i]) for i in range(n_cells)]
    heatmap_deltas = [pcr["mean_delta"] for pcr in per_cell_results]
    heatmap_types = [pcr["node_type"] for pcr in per_cell_results]

    return {
        "per_cell": per_cell_results,
        "eccentricity_correlation_corridors": {
            "rho": ecc_corr.get("rho"),
            "p": ecc_corr.get("p"),
            "n": len(corridor_indices),
            "test": "spearman_rank",
            "note": f"N={len(corridor_indices)}, descriptive only",
        },
        "distance_from_center_correlation": {
            "rho": dist_corr.get("rho"),
            "p": dist_corr.get("p"),
            "n": n_cells,
            "test": "spearman_rank",
            "center_cell": list(maze.cell_list[center_idx]),
        },
        "heatmap_data": {
            "cells": heatmap_cells,
            "deltas": heatmap_deltas,
            "node_types": heatmap_types,
        },
    }


# ---------------------------------------------------------------------------
# H8: Epoch-number adaptation
# ---------------------------------------------------------------------------


def compute_h8_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute epoch-number adaptation metrics for one session."""
    fps = data["fps"]
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    speed = data["speed_cm_s"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    epochs = detect_epochs(light_on, fps)
    filtered_epochs = [ep for ep in epochs if ep["duration_s"] >= MIN_EPOCH_DURATION_S]

    # Pair dark epochs with their preceding light epochs
    epoch_numbers: list[int] = []
    coverage_deltas: list[float] = []
    dark_coverages: list[float] = []
    light_coverages: list[float] = []
    speed_deltas: list[float] = []
    dark_speeds: list[float] = []

    pair_index = 0
    last_light_ep: dict | None = None

    for ep in filtered_epochs:
        if ep["condition"] == "light":
            last_light_ep = ep
        elif ep["condition"] == "dark" and last_light_ep is not None:
            pair_index += 1

            # Light epoch metrics
            sl_l = slice(last_light_ep["start"], last_light_ep["end"])
            ci_l = cell_indices[sl_l].copy()
            ci_l[~valid[sl_l]] = -1
            sp_l = speed[sl_l]
            valid_l = valid[sl_l] & np.isfinite(sp_l)
            light_unique = len(set(int(c) for c in ci_l if c >= 0))
            light_cov = light_unique / maze.n_cells
            light_speed = float(np.nanmean(sp_l[valid_l])) if valid_l.any() else np.nan

            # Dark epoch metrics
            sl_d = slice(ep["start"], ep["end"])
            ci_d = cell_indices[sl_d].copy()
            ci_d[~valid[sl_d]] = -1
            sp_d = speed[sl_d]
            valid_d = valid[sl_d] & np.isfinite(sp_d)
            dark_unique = len(set(int(c) for c in ci_d if c >= 0))
            dark_cov = dark_unique / maze.n_cells
            dark_speed = float(np.nanmean(sp_d[valid_d])) if valid_d.any() else np.nan

            epoch_numbers.append(pair_index)
            coverage_deltas.append(light_cov - dark_cov)  # positive = drop in dark
            dark_coverages.append(dark_cov)
            light_coverages.append(light_cov)
            speed_deltas.append(light_speed - dark_speed)
            dark_speeds.append(dark_speed)

            last_light_ep = None  # consume the light epoch

    n_pairs = len(epoch_numbers)

    # Within-session Spearman: epoch_number vs coverage_delta
    if n_pairs >= 6:
        rho_cov, p_cov = sp_stats.spearmanr(epoch_numbers, coverage_deltas)
        within_rho: float | None = float(rho_cov)
        within_p: float | None = float(p_cov)
    elif n_pairs >= 3:
        # Report but note low power
        try:
            rho_cov, p_cov = sp_stats.spearmanr(epoch_numbers, coverage_deltas)
            within_rho = float(rho_cov)
            within_p = float(p_cov)
        except Exception:
            within_rho = None
            within_p = None
    else:
        within_rho = None
        within_p = None

    # Within-session Spearman: epoch_number vs speed_delta
    if n_pairs >= 3:
        try:
            rho_sp, p_sp = sp_stats.spearmanr(epoch_numbers, speed_deltas)
            within_speed_rho: float | None = float(rho_sp)
        except Exception:
            within_speed_rho = None
    else:
        within_speed_rho = None

    # Within-session Spearman: epoch_number vs light_coverage
    if n_pairs >= 3:
        try:
            rho_lc, p_lc = sp_stats.spearmanr(epoch_numbers, light_coverages)
            within_light_cov_rho: float | None = float(rho_lc)
        except Exception:
            within_light_cov_rho = None
    else:
        within_light_cov_rho = None

    # Early vs late classification
    third = max(1, n_pairs // 3)
    early_delta = float(np.mean(coverage_deltas[:third])) if third > 0 and n_pairs > 0 else np.nan
    late_delta = float(np.mean(coverage_deltas[-third:])) if third > 0 and n_pairs > 0 else np.nan
    early_dark_cov = (
        float(np.mean(dark_coverages[:third])) if third > 0 and n_pairs > 0 else np.nan
    )
    late_dark_cov = (
        float(np.mean(dark_coverages[-third:])) if third > 0 and n_pairs > 0 else np.nan
    )

    # First dark epoch vs rest
    first_dark_cov = dark_coverages[0] if n_pairs > 0 else np.nan
    rest_dark_cov = float(np.mean(dark_coverages[1:])) if n_pairs > 1 else np.nan
    first_dark_speed = dark_speeds[0] if n_pairs > 0 else np.nan
    rest_dark_speed = float(np.mean(dark_speeds[1:])) if n_pairs > 1 else np.nan

    return {
        "n_epoch_pairs": n_pairs,
        "epoch_numbers": epoch_numbers,
        "coverage_deltas": coverage_deltas,
        "dark_coverages": dark_coverages,
        "light_coverages": light_coverages,
        "speed_deltas": speed_deltas,
        "dark_speeds": dark_speeds,
        "within_session_rho": within_rho,
        "within_session_p": within_p,
        "within_speed_rho": within_speed_rho,
        "within_light_cov_rho": within_light_cov_rho,
        "early_mean_delta": early_delta,
        "late_mean_delta": late_delta,
        "early_mean_dark_cov": early_dark_cov,
        "late_mean_dark_cov": late_dark_cov,
        "first_dark_coverage": first_dark_cov,
        "rest_mean_dark_coverage": rest_dark_cov,
        "first_dark_speed": first_dark_speed,
        "rest_mean_dark_speed": rest_dark_speed,
    }


def test_h8(session_results: list[dict]) -> dict:
    """Run H8: epoch-number adaptation across sessions."""
    # Session-level rho values
    rhos = np.array(
        [r["within_session_rho"] for r in session_results if r["within_session_rho"] is not None],
        dtype=float,
    )

    # Primary: one-sample Wilcoxon on session-level rho values
    slope_test = one_sample_wilcoxon(rhos)

    # Early vs late coverage delta (paired)
    early = np.array([r["early_mean_delta"] for r in session_results])
    late = np.array([r["late_mean_delta"] for r in session_results])
    early_late_test = wilcoxon_test(early, late)

    # First dark epoch vs rest (paired)
    first_cov = np.array([r["first_dark_coverage"] for r in session_results])
    rest_cov = np.array([r["rest_mean_dark_coverage"] for r in session_results])
    first_rest_test = wilcoxon_test(first_cov, rest_cov)

    # Holm-Bonferroni across 3 primary tests
    raw_pvals = [slope_test.get("p"), early_late_test.get("p"), first_rest_test.get("p")]
    adjusted = holm_bonferroni(raw_pvals)

    # Speed control: one-sample Wilcoxon on speed rho values
    speed_rhos = np.array(
        [r["within_speed_rho"] for r in session_results if r["within_speed_rho"] is not None],
        dtype=float,
    )
    speed_slope_test = one_sample_wilcoxon(speed_rhos)

    # Light coverage control: one-sample Wilcoxon on light-cov rho values
    light_cov_rhos = np.array(
        [
            r["within_light_cov_rho"]
            for r in session_results
            if r["within_light_cov_rho"] is not None
        ],
        dtype=float,
    )
    light_cov_slope_test = one_sample_wilcoxon(light_cov_rhos)

    def _safe_mean(arr: np.ndarray) -> float | None:
        v = arr[np.isfinite(arr)]
        return float(np.mean(v)) if len(v) > 0 else None

    # Interpretation
    p_adj_slope = adjusted[0] if len(adjusted) > 0 else None
    if p_adj_slope is not None and p_adj_slope < 0.05:
        median_rho = float(np.median(rhos)) if len(rhos) > 0 else 0.0
        if median_rho < 0:
            interpretation = "adaptation"
        else:
            interpretation = "worsening"
    else:
        interpretation = "constant"

    return {
        "slope_direction_test": {
            "median_rho": slope_test.get("median"),
            "mean_rho": slope_test.get("mean"),
            "p": slope_test.get("p"),
            "p_adj": adjusted[0] if len(adjusted) > 0 else None,
            "r": slope_test.get("r"),
            "n": slope_test.get("n"),
            "test": "wilcoxon_one_sample",
        },
        "early_vs_late_test": {
            "mean_early_delta": _safe_mean(early),
            "mean_late_delta": _safe_mean(late),
            "p": early_late_test.get("p"),
            "p_adj": adjusted[1] if len(adjusted) > 1 else None,
            "r": early_late_test.get("r"),
            "n": early_late_test.get("n"),
        },
        "first_vs_rest_test": {
            "mean_first_cov": _safe_mean(first_cov),
            "mean_rest_cov": _safe_mean(rest_cov),
            "p": first_rest_test.get("p"),
            "p_adj": adjusted[2] if len(adjusted) > 2 else None,
            "r": first_rest_test.get("r"),
            "n": first_rest_test.get("n"),
        },
        "speed_slope_control": {
            "median_rho": speed_slope_test.get("median"),
            "p": speed_slope_test.get("p"),
            "n": speed_slope_test.get("n"),
            "test": "wilcoxon_one_sample",
        },
        "light_coverage_slope_control": {
            "median_rho": light_cov_slope_test.get("median"),
            "p": light_cov_slope_test.get("p"),
            "n": light_cov_slope_test.get("n"),
            "test": "wilcoxon_one_sample",
        },
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# H9: Individual differences (per-animal darkness sensitivity)
# ---------------------------------------------------------------------------


def compute_h9_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute per-session coverage and speed differences for H9."""
    fps = data["fps"]
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    speed = data["speed_cm_s"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    epochs = detect_epochs(light_on, fps)

    cov_light_epochs: list[float] = []
    cov_dark_epochs: list[float] = []
    speed_light_epochs: list[float] = []
    speed_dark_epochs: list[float] = []

    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue
        sl = slice(ep["start"], ep["end"])
        ep_valid = valid[sl]
        ep_ci = cell_indices[sl].copy()
        ep_ci[~ep_valid] = -1
        ep_speed = speed[sl]
        sp_mask = ep_valid & np.isfinite(ep_speed)

        unique = len(set(int(c) for c in ep_ci if c >= 0))
        cov = unique / maze.n_cells
        mean_sp = float(np.nanmean(ep_speed[sp_mask])) if sp_mask.any() else np.nan

        if ep["condition"] == "light":
            cov_light_epochs.append(cov)
            speed_light_epochs.append(mean_sp)
        else:
            cov_dark_epochs.append(cov)
            speed_dark_epochs.append(mean_sp)

    mean_cov_light = float(np.nanmean(cov_light_epochs)) if cov_light_epochs else np.nan
    mean_cov_dark = float(np.nanmean(cov_dark_epochs)) if cov_dark_epochs else np.nan
    mean_speed_light = float(np.nanmean(speed_light_epochs)) if speed_light_epochs else np.nan
    mean_speed_dark = float(np.nanmean(speed_dark_epochs)) if speed_dark_epochs else np.nan

    return {
        "animal_id": data["animal_id"],
        "coverage_light": mean_cov_light,
        "coverage_dark": mean_cov_dark,
        "coverage_diff": mean_cov_dark - mean_cov_light,  # negative = less coverage in dark
        "speed_light": mean_speed_light,
        "speed_dark": mean_speed_dark,
        "speed_diff": mean_speed_dark - mean_speed_light,
    }


def test_h9(session_results: list[dict], animals_meta: dict) -> dict:
    """Run H9: individual differences in darkness sensitivity.

    Aggregates per-session results to per-animal means, then tests
    whether coverage sensitivity correlates with speed sensitivity
    across animals.
    """
    # Aggregate to per-animal
    animal_data: dict[str, dict] = {}
    for r in session_results:
        aid = r["animal_id"]
        if aid not in animal_data:
            animal_data[aid] = {
                "animal_id": aid,
                "celltype": animals_meta.get(aid, {}).get("celltype", "unknown"),
                "cov_diffs": [],
                "speed_diffs": [],
                "cov_lights": [],
                "cov_darks": [],
            }
        if np.isfinite(r["coverage_diff"]):
            animal_data[aid]["cov_diffs"].append(r["coverage_diff"])
        if np.isfinite(r["speed_diff"]):
            animal_data[aid]["speed_diffs"].append(r["speed_diff"])
        if np.isfinite(r["coverage_light"]):
            animal_data[aid]["cov_lights"].append(r["coverage_light"])
        if np.isfinite(r["coverage_dark"]):
            animal_data[aid]["cov_darks"].append(r["coverage_dark"])

    # Per-animal summary
    per_animal: list[dict] = []
    cov_sensitivities: list[float] = []
    speed_sensitivities: list[float] = []

    for aid, ad in sorted(animal_data.items()):
        mean_cov_diff = float(np.mean(ad["cov_diffs"])) if ad["cov_diffs"] else np.nan
        mean_speed_diff = float(np.mean(ad["speed_diffs"])) if ad["speed_diffs"] else np.nan
        mean_cov_light = float(np.mean(ad["cov_lights"])) if ad["cov_lights"] else np.nan
        mean_cov_dark = float(np.mean(ad["cov_darks"])) if ad["cov_darks"] else np.nan

        # Coverage sensitivity: coverage_dark - coverage_light
        # More negative = more darkness-sensitive
        cov_sensitivity = mean_cov_diff  # dark - light
        # Coverage drop in cells (absolute)
        cov_drop_cells = -(mean_cov_diff * MAZE.n_cells) if np.isfinite(mean_cov_diff) else np.nan

        # Classify: darkness-resistant (drop < 1 cell), darkness-sensitive (drop > 3 cells)
        if np.isfinite(cov_drop_cells):
            if cov_drop_cells < 1:
                sensitivity_class = "darkness-resistant"
            elif cov_drop_cells > 3:
                sensitivity_class = "darkness-sensitive"
            else:
                sensitivity_class = "intermediate"
        else:
            sensitivity_class = "unknown"

        per_animal.append(
            {
                "animal_id": aid,
                "celltype": ad["celltype"],
                "n_sessions": len(ad["cov_diffs"]),
                "mean_coverage_light": mean_cov_light,
                "mean_coverage_dark": mean_cov_dark,
                "coverage_sensitivity": cov_sensitivity,
                "coverage_drop_cells": cov_drop_cells,
                "speed_sensitivity": mean_speed_diff,
                "sensitivity_class": sensitivity_class,
            }
        )

        if np.isfinite(cov_sensitivity):
            cov_sensitivities.append(cov_sensitivity)
        if np.isfinite(mean_speed_diff):
            speed_sensitivities.append(mean_speed_diff)

    # Spearman: coverage sensitivity vs speed sensitivity across animals
    cov_sens_arr = np.array([pa["coverage_sensitivity"] for pa in per_animal])
    speed_sens_arr = np.array([pa["speed_sensitivity"] for pa in per_animal])
    cov_speed_corr = spearman_test(cov_sens_arr, speed_sens_arr)

    # Descriptive counts
    n_resistant = sum(1 for pa in per_animal if pa["sensitivity_class"] == "darkness-resistant")
    n_sensitive = sum(1 for pa in per_animal if pa["sensitivity_class"] == "darkness-sensitive")
    n_intermediate = sum(1 for pa in per_animal if pa["sensitivity_class"] == "intermediate")

    return {
        "per_animal": per_animal,
        "coverage_speed_correlation": cov_speed_corr,
        "n_animals": len(per_animal),
        "n_darkness_resistant": n_resistant,
        "n_darkness_sensitive": n_sensitive,
        "n_intermediate": n_intermediate,
        "mean_coverage_sensitivity": float(np.nanmean(cov_sens_arr)),
        "std_coverage_sensitivity": float(np.nanstd(cov_sens_arr)),
    }


# ---------------------------------------------------------------------------
# H10: Cell-type Markov model (J/C/D 3-state)
# ---------------------------------------------------------------------------


def _build_type_map(maze: RoseMaze) -> dict[int, int]:
    """Build mapping from cell index to type index (J=0, C=1, D=2)."""
    type_map: dict[int, int] = {}
    for c in maze.junctions:
        type_map[maze.cell_to_idx[c]] = TYPE_J
    for c in maze.corridors:
        type_map[maze.cell_to_idx[c]] = TYPE_C
    for c in maze.dead_ends:
        type_map[maze.cell_to_idx[c]] = TYPE_D
    return type_map


def _cell_seq_to_type_seq(cs: np.ndarray, type_map: dict[int, int]) -> np.ndarray:
    """Convert cell sequence to type sequence, removing consecutive duplicates."""
    if len(cs) == 0:
        return np.array([], dtype=np.int32)
    ts = np.array([type_map.get(int(c), -1) for c in cs], dtype=np.int32)
    # Remove invalid
    ts = ts[ts >= 0]
    if len(ts) == 0:
        return np.array([], dtype=np.int32)
    # Remove consecutive duplicates
    mask = np.concatenate([[True], ts[1:] != ts[:-1]])
    return ts[mask]


def _type_transition_matrix_1st(type_seq: np.ndarray) -> np.ndarray:
    """Compute 3x3 first-order type transition matrix."""
    counts = np.zeros((N_TYPES, N_TYPES), dtype=np.float64)
    for i in range(len(type_seq) - 1):
        a, b = type_seq[i], type_seq[i + 1]
        if 0 <= a < N_TYPES and 0 <= b < N_TYPES:
            counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def _type_transition_matrix_2nd(type_seq: np.ndarray) -> np.ndarray:
    """Compute 3x3x3 second-order type transition matrix.

    tm[i, j, k] = P(next=k | prev=i, curr=j).
    """
    counts = np.zeros((N_TYPES, N_TYPES, N_TYPES), dtype=np.float64)
    for t in range(len(type_seq) - 2):
        a, b, c = type_seq[t], type_seq[t + 1], type_seq[t + 2]
        if 0 <= a < N_TYPES and 0 <= b < N_TYPES and 0 <= c < N_TYPES:
            counts[a, b, c] += 1
    # Normalize each [i, j, :] slice
    for i in range(N_TYPES):
        for j in range(N_TYPES):
            row_sum = counts[i, j, :].sum()
            if row_sum > 0:
                counts[i, j, :] /= row_sum
    return counts


def _type_level_jsd(tm_a: np.ndarray, tm_b: np.ndarray, type_seq_combined: np.ndarray) -> float:
    """Compute weighted JSD between two 3x3 type transition matrices."""
    source_counts = np.zeros(N_TYPES, dtype=np.float64)
    for t in range(len(type_seq_combined) - 1):
        c = type_seq_combined[t]
        if 0 <= c < N_TYPES:
            source_counts[c] += 1
    total = source_counts.sum()
    if total == 0:
        return np.nan
    weights = source_counts / total

    jsd_per_row = np.zeros(N_TYPES)
    for i in range(N_TYPES):
        row_a = tm_a[i]
        row_b = tm_b[i]
        if row_a.sum() > 0 and row_b.sum() > 0:
            js_dist = jensenshannon(row_a, row_b, base=2)
            jsd_per_row[i] = js_dist**2
    return float(np.sum(weights * jsd_per_row))


def _type_markov_order_comparison(type_seq: np.ndarray) -> dict:
    """BIC comparison of 1st vs 2nd order type-level Markov model."""
    n = len(type_seq)
    if n < 4:
        return {"delta_bic": 0.0, "preferred_order": 1}

    # 1st order
    tm1 = _type_transition_matrix_1st(type_seq)
    n_trans_1 = max(1, n - 1)
    # Cross-entropy
    ce1 = 0.0
    for t in range(n - 1):
        a, b = type_seq[t], type_seq[t + 1]
        p = tm1[a, b]
        ce1 -= np.log(max(p, 1e-12))
    ce1 /= n_trans_1

    # 2nd order
    tm2 = _type_transition_matrix_2nd(type_seq)
    n_trans_2 = max(1, n - 2)
    ce2 = 0.0
    for t in range(n - 2):
        a, b, c = type_seq[t], type_seq[t + 1], type_seq[t + 2]
        p = tm2[a, b, c]
        ce2 -= np.log(max(p, 1e-12))
    ce2 /= n_trans_2

    # Free parameters: count observed source states/pairs
    observed_states = len(set(int(s) for s in type_seq))
    observed_pairs = len(set((int(type_seq[t]), int(type_seq[t + 1])) for t in range(n - 1)))
    k1 = observed_states * (N_TYPES - 1)
    k2 = observed_pairs * (N_TYPES - 1)

    # Log-likelihood
    ll1 = -n_trans_1 * ce1
    ll2 = -n_trans_2 * ce2

    bic1 = k1 * np.log(n_trans_1) - 2 * ll1
    bic2 = k2 * np.log(n_trans_2) - 2 * ll2
    delta_bic = bic1 - bic2  # positive = 2nd order preferred

    return {
        "delta_bic": float(delta_bic),
        "preferred_order": 2 if delta_bic > 0 else 1,
    }


def compute_h10_per_session(data: dict, maze: RoseMaze) -> dict:
    """Compute cell-type Markov model metrics for one session."""
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)

    valid = ~bad_behav & np.isfinite(x_maze) & np.isfinite(y_maze)
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1

    type_map = _build_type_map(maze)

    # Cell sequences for light and dark
    ci_light = cell_indices.copy()
    ci_light[~(valid & light_on)] = -1
    cs_light, _ = cell_sequence(ci_light)

    ci_dark = cell_indices.copy()
    ci_dark[~(valid & ~light_on)] = -1
    cs_dark, _ = cell_sequence(ci_dark)

    # Convert to type sequences
    ts_light = _cell_seq_to_type_seq(cs_light, type_map)
    ts_dark = _cell_seq_to_type_seq(cs_dark, type_map)

    # Combined type sequence for BIC comparison
    cs_all, _ = cell_sequence(cell_indices)
    ts_all = _cell_seq_to_type_seq(cs_all, type_map)

    # 1st and 2nd order transition matrices
    if len(ts_light) < 3 or len(ts_dark) < 3:
        # Not enough data
        return {
            "tm_1st_light": np.zeros((N_TYPES, N_TYPES)).tolist(),
            "tm_1st_dark": np.zeros((N_TYPES, N_TYPES)).tolist(),
            "tm_2nd_light": np.zeros((N_TYPES, N_TYPES, N_TYPES)).tolist(),
            "tm_2nd_dark": np.zeros((N_TYPES, N_TYPES, N_TYPES)).tolist(),
            "jsd_type_level": np.nan,
            "n_type_transitions_light": len(ts_light) - 1 if len(ts_light) > 0 else 0,
            "n_type_transitions_dark": len(ts_dark) - 1 if len(ts_dark) > 0 else 0,
            "p_D_given_JC_light": np.nan,
            "p_D_given_JC_dark": np.nan,
            "p_J_given_JC_light": np.nan,
            "p_J_given_JC_dark": np.nan,
            "p_J_given_DC_light": np.nan,
            "p_J_given_DC_dark": np.nan,
            "p_C_given_CJ_light": np.nan,
            "p_C_given_CJ_dark": np.nan,
            "type_markov_order": {"delta_bic": 0.0, "preferred_order": 1},
        }

    tm1_light = _type_transition_matrix_1st(ts_light)
    tm1_dark = _type_transition_matrix_1st(ts_dark)
    tm2_light = _type_transition_matrix_2nd(ts_light)
    tm2_dark = _type_transition_matrix_2nd(ts_dark)

    # JSD between light and dark 1st-order type matrices
    ts_combined = np.concatenate([ts_light, ts_dark])
    jsd = _type_level_jsd(tm1_light, tm1_dark, ts_combined)

    # Key second-order transition probabilities
    # P(D | J->C) = tm2[J, C, D] = tm2[0, 1, 2]
    p_D_JC_light = float(tm2_light[TYPE_J, TYPE_C, TYPE_D])
    p_D_JC_dark = float(tm2_dark[TYPE_J, TYPE_C, TYPE_D])
    # P(J | J->C) = tm2[J, C, J] = tm2[0, 1, 0]
    p_J_JC_light = float(tm2_light[TYPE_J, TYPE_C, TYPE_J])
    p_J_JC_dark = float(tm2_dark[TYPE_J, TYPE_C, TYPE_J])
    # P(J | D->C) = tm2[D, C, J] = tm2[2, 1, 0]
    p_J_DC_light = float(tm2_light[TYPE_D, TYPE_C, TYPE_J])
    p_J_DC_dark = float(tm2_dark[TYPE_D, TYPE_C, TYPE_J])
    # P(C | C->J) = tm2[C, J, C] = tm2[1, 0, 1]
    p_C_CJ_light = float(tm2_light[TYPE_C, TYPE_J, TYPE_C])
    p_C_CJ_dark = float(tm2_dark[TYPE_C, TYPE_J, TYPE_C])

    # BIC comparison on full session
    order_comp = _type_markov_order_comparison(ts_all)

    # Store transition counts for permutation null
    tc_light = np.zeros((N_TYPES, N_TYPES), dtype=np.float64)
    for t in range(len(ts_light) - 1):
        tc_light[ts_light[t], ts_light[t + 1]] += 1
    tc_dark = np.zeros((N_TYPES, N_TYPES), dtype=np.float64)
    for t in range(len(ts_dark) - 1):
        tc_dark[ts_dark[t], ts_dark[t + 1]] += 1

    return {
        "tm_1st_light": tm1_light.tolist(),
        "tm_1st_dark": tm1_dark.tolist(),
        "tm_2nd_light": tm2_light.tolist(),
        "tm_2nd_dark": tm2_dark.tolist(),
        "jsd_type_level": jsd,
        "n_type_transitions_light": len(ts_light) - 1,
        "n_type_transitions_dark": len(ts_dark) - 1,
        "p_D_given_JC_light": p_D_JC_light,
        "p_D_given_JC_dark": p_D_JC_dark,
        "p_J_given_JC_light": p_J_JC_light,
        "p_J_given_JC_dark": p_J_JC_dark,
        "p_J_given_DC_light": p_J_DC_light,
        "p_J_given_DC_dark": p_J_DC_dark,
        "p_C_given_CJ_light": p_C_CJ_light,
        "p_C_given_CJ_dark": p_C_CJ_dark,
        "type_transition_counts_light": tc_light.tolist(),
        "type_transition_counts_dark": tc_dark.tolist(),
        "type_markov_order": order_comp,
    }


def test_h10(session_results: list[dict], n_permutations: int = 1000) -> dict:
    """Run H10: cell-type Markov model tests across sessions."""
    # JSD permutation test
    jsd_values = np.array([r["jsd_type_level"] for r in session_results])
    valid_jsd = jsd_values[np.isfinite(jsd_values)]
    n_jsd = len(valid_jsd)
    observed_mean_jsd = float(np.mean(valid_jsd)) if n_jsd > 0 else np.nan

    rng = np.random.default_rng(42)
    null_mean_jsds: list[float] = []

    for _ in range(n_permutations):
        perm_jsds: list[float] = []
        for r in session_results:
            tl = r.get("type_transition_counts_light")
            td = r.get("type_transition_counts_dark")
            if tl is None or td is None:
                continue
            tl_arr = np.array(tl, dtype=np.float64)
            td_arr = np.array(td, dtype=np.float64)
            pooled = tl_arr + td_arr
            total = pooled.sum()
            if total < 2:
                continue
            flat = pooled.ravel()
            half = int(total // 2)
            perm_flat = np.zeros_like(flat)
            indices = []
            for idx, count in enumerate(flat):
                indices.extend([idx] * int(count))
            rng.shuffle(indices)
            for idx in indices[:half]:
                perm_flat[idx] += 1
            perm_a = perm_flat.reshape(pooled.shape)
            perm_b = pooled - perm_a
            # Compute JSD
            jsd_vals_perm: list[float] = []
            for row_idx in range(perm_a.shape[0]):
                row_a = perm_a[row_idx]
                row_b = perm_b[row_idx]
                if row_a.sum() == 0 or row_b.sum() == 0:
                    continue
                p_a = row_a / row_a.sum()
                p_b = row_b / row_b.sum()
                js = jensenshannon(p_a, p_b, base=2) ** 2
                if np.isfinite(js):
                    jsd_vals_perm.append(js * (row_a.sum() + row_b.sum()) / max(total, 1))
            perm_jsds.append(sum(jsd_vals_perm) if jsd_vals_perm else 0.0)
        if perm_jsds:
            null_mean_jsds.append(float(np.mean(perm_jsds)))

    if null_mean_jsds:
        null_arr = np.array(null_mean_jsds)
        perm_p = float((null_arr >= observed_mean_jsd).mean())
        null_mean = float(np.mean(null_arr))
        null_95 = float(np.percentile(null_arr, 95))
    else:
        perm_p, null_mean, null_95 = np.nan, np.nan, np.nan

    jsd_type_test = {
        "observed_mean_jsd": observed_mean_jsd,
        "permutation_p": perm_p,
        "null_mean": null_mean,
        "null_95_pct": null_95,
        "n": n_jsd,
    }

    # Per-transition Wilcoxon tests
    def _safe_mean(arr: np.ndarray) -> float | None:
        v = arr[np.isfinite(arr)]
        return float(np.mean(v)) if len(v) > 0 else None

    def _transition_test(key_light: str, key_dark: str) -> dict:
        vals_l = np.array([r[key_light] for r in session_results])
        vals_d = np.array([r[key_dark] for r in session_results])
        wt = wilcoxon_test(vals_l, vals_d)
        return {
            "mean_light": _safe_mean(vals_l),
            "mean_dark": _safe_mean(vals_d),
            "p": wt.get("p"),
            "r": wt.get("r"),
            "n": wt.get("n"),
        }

    t_D_JC = _transition_test("p_D_given_JC_light", "p_D_given_JC_dark")
    t_J_JC = _transition_test("p_J_given_JC_light", "p_J_given_JC_dark")
    t_C_CJ = _transition_test("p_C_given_CJ_light", "p_C_given_CJ_dark")
    t_J_DC = _transition_test("p_J_given_DC_light", "p_J_given_DC_dark")

    # Holm-Bonferroni across 4 transition tests
    trans_pvals = [t_D_JC.get("p"), t_J_JC.get("p"), t_C_CJ.get("p"), t_J_DC.get("p")]
    trans_adjusted = holm_bonferroni(trans_pvals)
    t_D_JC["p_adj"] = trans_adjusted[0] if len(trans_adjusted) > 0 else None
    t_J_JC["p_adj"] = trans_adjusted[1] if len(trans_adjusted) > 1 else None
    t_C_CJ["p_adj"] = trans_adjusted[2] if len(trans_adjusted) > 2 else None
    t_J_DC["p_adj"] = trans_adjusted[3] if len(trans_adjusted) > 3 else None

    # Model order: one-sample Wilcoxon on delta_bic
    delta_bics = np.array([r["type_markov_order"]["delta_bic"] for r in session_results])
    n_prefer_2nd = int(np.sum(delta_bics > 0))
    model_order_test = one_sample_wilcoxon(delta_bics)

    return {
        "jsd_type_test": jsd_type_test,
        "p_D_given_JC": t_D_JC,
        "p_J_given_JC": t_J_JC,
        "p_C_given_CJ": t_C_CJ,
        "p_J_given_DC": t_J_DC,
        "model_order": {
            "mean_delta_bic": float(np.mean(delta_bics)) if len(delta_bics) > 0 else None,
            "n_prefer_2nd": n_prefer_2nd,
            "n_total": len(delta_bics),
            "p": model_order_test.get("p"),
            "test": "wilcoxon_one_sample",
        },
    }


# ---------------------------------------------------------------------------
# Tier-2 print summary
# ---------------------------------------------------------------------------


def _print_h5_summary(stats: dict) -> None:
    """Print H5 results."""
    print("\n--- H5: Within-epoch temporal dynamics ---")
    ct = stats["coverage_ratio_test"]
    print("  Coverage ratio (2nd/1st half):")
    print(f"    Light: {ct['mean_light']:.3f}, Dark: {ct['mean_dark']:.3f}")
    print(f"    p = {ct['p']:.4f}, p_adj = {ct['p_adj']:.4f}, r = {ct['r']:.3f}, N = {ct['n']}")
    st = stats["speed_ratio_test"]
    print("  Speed ratio (2nd/1st half):")
    print(f"    Light: {st['mean_light']:.3f}, Dark: {st['mean_dark']:.3f}")
    print(f"    p = {st['p']:.4f}, p_adj = {st['p_adj']:.4f}, r = {st['r']:.3f}, N = {st['n']}")
    sc = stats["speed_coverage_correlation"]
    print(f"  Speed-coverage correlation: rho = {sc['rho']}, p = {sc['p']}")
    rt = stats["recovery_test"]
    print("  Lights-on recovery:")
    mr = rt.get('mean_recovery')
    mi = rt.get('mean_initial')
    print(f"    Recovery: {f'{mr:.3f}' if mr is not None else 'N/A'}, Initial: {f'{mi:.3f}' if mi is not None else 'N/A'}")
    rp = rt.get('p')
    ra = rt.get('p_adj')
    rr = rt.get('r')
    print(f"    p = {f'{rp:.4f}' if rp is not None else 'N/A'}, p_adj = {f'{ra:.4f}' if ra is not None else 'N/A'}, r = {f'{rr:.3f}' if rr is not None else 'N/A'}, N = {rt.get('n', '?')}")
    print(f"  Interpretation: {stats['interpretation']}")


def _print_h6_summary(stats: dict) -> None:
    """Print H6 results."""
    print("\n--- H6: Corridor-specific coverage ---")
    sig_cells = [
        pc for pc in stats["per_cell"] if pc.get("p_adj") is not None and pc["p_adj"] < 0.05
    ]
    sig_names = [f"({pc['cell'][0]},{pc['cell'][1]})" for pc in sig_cells]
    print(f"  Cells with significant delta (p_adj < 0.05): {sig_names if sig_names else 'none'}")

    # Top 5 by |delta|
    sorted_cells = sorted(stats["per_cell"], key=lambda x: abs(x["mean_delta"]), reverse=True)
    print("  Top 5 cells by |delta|:")
    for pc in sorted_cells[:5]:
        p_adj_str = f"{pc['p_adj']:.4f}" if pc["p_adj"] is not None else "N/A"
        cell_str = f"({pc['cell'][0]},{pc['cell'][1]})"
        print(
            f"    {cell_str}: delta={pc['mean_delta']:.3f}, "
            f"type={pc['node_type']}, p_adj={p_adj_str}"
        )

    ec = stats["eccentricity_correlation_corridors"]
    print(f"  Eccentricity correlation (corridors, N={ec['n']}): rho={ec['rho']}, p={ec['p']}")
    dc = stats["distance_from_center_correlation"]
    print(f"  Distance-from-center correlation (all, N={dc['n']}): rho={dc['rho']}, p={dc['p']}")


def _print_h8_summary(stats: dict) -> None:
    """Print H8 results."""
    print("\n--- H8: Epoch-number adaptation ---")
    sd = stats["slope_direction_test"]
    print("  Within-session slope (epoch# vs coverage delta):")
    print(
        f"    Median rho = {sd['median_rho']}, p = {sd['p']}, p_adj = {sd['p_adj']}, N = {sd['n']}"
    )
    el = stats["early_vs_late_test"]
    print("  Early vs late coverage delta:")
    print(f"    Early: {el['mean_early_delta']:.3f}, Late: {el['mean_late_delta']:.3f}")
    print(f"    p = {el['p']}, p_adj = {el['p_adj']}, r = {el['r']}, N = {el['n']}")
    fr = stats["first_vs_rest_test"]
    print("  First dark epoch vs rest:")
    print(f"    First: {fr['mean_first_cov']:.3f}, Rest: {fr['mean_rest_cov']:.3f}")
    print(f"    p = {fr['p']}, p_adj = {fr['p_adj']}, r = {fr['r']}, N = {fr['n']}")
    ssc = stats["speed_slope_control"]
    print(f"  Speed slope control: median rho = {ssc['median_rho']}, p = {ssc['p']}")
    lsc = stats["light_coverage_slope_control"]
    print(f"  Light coverage slope control: median rho = {lsc['median_rho']}, p = {lsc['p']}")
    print(f"  Interpretation: {stats['interpretation']}")


def _print_h9_summary(stats: dict) -> None:
    """Print H9 results."""
    print("\n--- H9: Individual differences ---")
    print(f"  N animals: {stats['n_animals']}")
    print(
        f"  Darkness-resistant: {stats['n_darkness_resistant']}, "
        f"Intermediate: {stats['n_intermediate']}, "
        f"Darkness-sensitive: {stats['n_darkness_sensitive']}"
    )
    print(
        f"  Mean coverage sensitivity: {stats['mean_coverage_sensitivity']:.4f} "
        f"(SD = {stats['std_coverage_sensitivity']:.4f})"
    )
    csc = stats["coverage_speed_correlation"]
    print(f"  Coverage-speed correlation: rho = {csc['rho']}, p = {csc['p']}, N = {csc['n']}")
    print("  Per-animal summary:")
    for pa in stats["per_animal"]:
        print(
            f"    {pa['animal_id']} ({pa['celltype']}): "
            f"cov_drop={pa['coverage_drop_cells']:.1f} cells, "
            f"speed_diff={pa['speed_sensitivity']:.2f} cm/s, "
            f"class={pa['sensitivity_class']}"
        )


def _print_h10_summary(stats: dict) -> None:
    """Print H10 results."""
    print("\n--- H10: Cell-type Markov model ---")
    jt = stats["jsd_type_test"]
    print("  Type-level JSD (light vs dark):")
    print(
        f"    Observed: {jt['observed_mean_jsd']:.4f}, Null: {jt['null_mean']:.4f}, "
        f"p_perm = {jt['permutation_p']:.4f}, N = {jt['n']}"
    )
    print("  Key transitions (2nd-order, light vs dark):")
    for key, label in [
        ("p_D_given_JC", "P(D|J,C)"),
        ("p_J_given_JC", "P(J|J,C)"),
        ("p_C_given_CJ", "P(C|C,J)"),
        ("p_J_given_DC", "P(J|D,C)"),
    ]:
        t = stats[key]
        p_adj_str = f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "N/A"
        r_str = f"{t['r']:.3f}" if t.get("r") is not None else "N/A"
        p_str = f"{t['p']:.4f}" if t.get("p") is not None else "N/A"
        ml = f"{t['mean_light']:.3f}" if t.get("mean_light") is not None else "N/A"
        md = f"{t['mean_dark']:.3f}" if t.get("mean_dark") is not None else "N/A"
        print(f"    {label}: light={ml}, dark={md}, p={p_str}, p_adj={p_adj_str}, r={r_str}")
    mo = stats["model_order"]
    print(
        f"  Model order (type-level): {mo['n_prefer_2nd']}/{mo['n_total']} prefer 2nd order, "
        f"delta_BIC = {mo['mean_delta_bic']:.1f}, p = {mo['p']}"
    )


# ---------------------------------------------------------------------------
# Tier-2 main
# ---------------------------------------------------------------------------


def main_tier2() -> None:
    """Run Tier-2 behaviour hypotheses (H5, H6, H8, H9, H10)."""
    print("=" * 70)
    print("BEHAVIOUR HYPOTHESES -- Tier-2 Tests (H5, H6, H8, H9, H10)")
    print("=" * 70)

    # Load metadata
    with open(METADATA_CSV) as f:
        experiments = list(csv.DictReader(f))
    with open(ANIMALS_CSV) as f:
        animals = {row["animal_id"]: row for row in csv.DictReader(f)}

    sessions: list[dict] = []
    for row in experiments:
        eid = row["exp_id"]
        parts = eid.split("_")
        animal_id = parts[-1]
        sessions.append(
            {
                "exp_id": eid,
                "exp_index": int(row["exp_index"]),
                "animal_id": animal_id,
                "celltype": animals.get(animal_id, {}).get("celltype", "unknown"),
                "exclude": str(row.get("exclude", "0")).strip() == "1",
                "primary": str(row.get("primary_exp", "1")).strip() != "0",
            }
        )

    usable_sessions = [s for s in sessions if not s["exclude"]]
    print(f"\nTotal sessions: {len(sessions)}")
    print(f"Excluded: {sum(1 for s in sessions if s['exclude'])}")
    print(f"Usable: {len(usable_sessions)}")

    # Download and analyse
    s3 = boto3.client("s3", region_name=S3_REGION)

    h5_results: list[dict] = []
    h6_results: list[dict] = []
    h8_results: list[dict] = []
    h9_results: list[dict] = []
    h10_results: list[dict] = []
    session_ids: list[str] = []

    for sess in usable_sessions:
        eid = sess["exp_id"]
        print(f"\n--- {eid} (#{sess['exp_index']}) ---")

        data = load_session_data(s3, eid)
        if data is None:
            print("    SKIPPED (no data)")
            continue

        session_ids.append(eid)

        # H5
        r5 = compute_h5_per_session(data, MAZE)
        h5_results.append(r5)
        crl = r5["median_coverage_ratio_light"]
        crd = r5["median_coverage_ratio_dark"]
        print(f"  H5: cov_ratio L/D={crl:.3f}/{crd:.3f}")

        # H6
        r6 = compute_h6_per_session(data, MAZE)
        h6_results.append(r6)
        print("  H6: visit fracs computed (23 cells)")

        # H8
        r8 = compute_h8_per_session(data, MAZE)
        h8_results.append(r8)
        rho_str = (
            f"{r8['within_session_rho']:.3f}" if r8["within_session_rho"] is not None else "N/A"
        )
        print(f"  H8: {r8['n_epoch_pairs']} epoch pairs, slope rho={rho_str}")

        # H9
        r9 = compute_h9_per_session(data, MAZE)
        h9_results.append(r9)
        print(f"  H9: cov_diff={r9['coverage_diff']:.3f}, speed_diff={r9['speed_diff']:.2f}")

        # H10
        r10 = compute_h10_per_session(data, MAZE)
        h10_results.append(r10)
        pdl = r10["p_D_given_JC_light"]
        pdd = r10["p_D_given_JC_dark"]
        if np.isfinite(pdl) and np.isfinite(pdd):
            print(f"  H10: P(D|J,C) L/D={pdl:.3f}/{pdd:.3f}, JSD={r10['jsd_type_level']:.4f}")
        else:
            print("  H10: insufficient type transitions")

    # ===================================================================
    # Cross-session hypothesis tests
    # ===================================================================
    print("\n" + "=" * 70)
    print("CROSS-SESSION HYPOTHESIS TESTS -- Tier-2")
    print("=" * 70)

    h5_stats = test_h5(h5_results)
    h6_stats = test_h6(h6_results, MAZE)
    h8_stats = test_h8(h8_results)
    h9_stats = test_h9(h9_results, animals)
    h10_stats = test_h10(h10_results)

    # Print summaries
    _print_h5_summary(h5_stats)
    _print_h6_summary(h6_stats)
    _print_h8_summary(h8_stats)
    _print_h9_summary(h9_stats)
    _print_h10_summary(h10_stats)

    # ===================================================================
    # Save results
    # ===================================================================
    output = {
        "metadata": {
            "n_sessions": len(session_ids),
            "session_ids": session_ids,
            "description": (
                "Tier-2 behaviour hypotheses from plan-behaviour-science-tier2.md. "
                "Deeper analyses of within-epoch dynamics, spatial specificity, "
                "adaptation, individual differences, and cell-type Markov structure."
            ),
        },
        "h5_temporal_dynamics": h5_stats,
        "h6_corridor_heatmap": h6_stats,
        "h8_epoch_adaptation": h8_stats,
        "h9_individual_differences": h9_stats,
        "h10_cell_type_markov": h10_stats,
    }

    output_ser = _make_serializable(output)
    OUTPUT_JSON_T2.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_T2, "w") as f:
        json.dump(output_ser, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_JSON_T2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("BEHAVIOUR HYPOTHESES — Tier-1 Tests (H1-H4)")
    print("=" * 70)

    # Load metadata
    with open(METADATA_CSV) as f:
        experiments = list(csv.DictReader(f))
    with open(ANIMALS_CSV) as f:
        animals = {row["animal_id"]: row for row in csv.DictReader(f)}

    sessions: list[dict] = []
    for row in experiments:
        eid = row["exp_id"]
        parts = eid.split("_")
        animal_id = parts[-1]
        sessions.append(
            {
                "exp_id": eid,
                "exp_index": int(row["exp_index"]),
                "animal_id": animal_id,
                "celltype": animals.get(animal_id, {}).get("celltype", "unknown"),
                "exclude": str(row.get("exclude", "0")).strip() == "1",
                "primary": str(row.get("primary_exp", "1")).strip() != "0",
            }
        )

    usable_sessions = [s for s in sessions if not s["exclude"]]
    print(f"\nTotal sessions: {len(sessions)}")
    print(f"Excluded: {sum(1 for s in sessions if s['exclude'])}")
    print(f"Usable: {len(usable_sessions)}")

    # Download and analyse
    s3 = boto3.client("s3", region_name=S3_REGION)

    h1_results: list[dict] = []
    h2_results: list[dict] = []
    h3_results: list[dict] = []
    h4_results: list[dict] = []
    session_ids: list[str] = []

    for sess in usable_sessions:
        eid = sess["exp_id"]
        print(f"\n--- {eid} (#{sess['exp_index']}) ---")

        data = load_session_data(s3, eid)
        if data is None:
            print("    SKIPPED (no data)")
            continue

        session_ids.append(eid)

        # H1
        r1 = compute_h1_per_session(data, MAZE)
        h1_results.append(r1)
        print(f"  H1: speed diff={r1['speed_diff']:.2f}, cov diff={r1['cov_diff']:.3f}")

        # H2
        r2 = compute_h2_per_session(data, MAZE)
        h2_results.append(r2)
        print(f"  H2: JSD={r2['jsd']:.4f}" if np.isfinite(r2["jsd"]) else "  H2: JSD=N/A")

        # H3
        r3 = compute_h3_per_session(data, MAZE)
        h3_results.append(r3)
        print(
            f"  H3: junc cov L/D={r3['mean_junc_cov_light']:.2f}/{r3['mean_junc_cov_dark']:.2f}, "
            f"DE cov L/D={r3['mean_de_cov_light']:.2f}/{r3['mean_de_cov_dark']:.2f}, "
            f"diam L/D={r3['mean_diam_light']:.1f}/{r3['mean_diam_dark']:.1f}"
        )

        # H4
        r4 = compute_h4_per_session(data, MAZE)
        h4_results.append(r4)
        print(
            f"  H4: revis L/D={r4['mean_revis_light']:.2f}/{r4['mean_revis_dark']:.2f}, "
            f"AUC L/D={r4['mean_auc_light']:.3f}/{r4['mean_auc_dark']:.3f}"
        )

    # ===================================================================
    # Cross-session hypothesis tests
    # ===================================================================
    print("\n" + "=" * 70)
    print("CROSS-SESSION HYPOTHESIS TESTS")
    print("=" * 70)

    h1_stats = test_h1(h1_results)
    h2_stats = test_h2(h2_results)
    h3_stats = test_h3(h3_results)
    h4_stats = test_h4(h4_results)

    # ---- Print human-readable summary ----

    # H1
    print("\n--- H1: Speed-coverage partial correlation ---")
    print(f"  N sessions: {h1_stats['n_sessions']}")
    print(f"  Spearman rho = {h1_stats['spearman_rho']}, p = {h1_stats['spearman_p']}")
    cpt = h1_stats["coverage_per_transition"]
    print("  Coverage per transition (speed-normalised):")
    print(f"    Light: {cpt['mean_light']:.4f}, Dark: {cpt['mean_dark']:.4f}")
    print(f"    p = {cpt['p']}, r = {cpt['r']}, N = {cpt['n_sessions']}")

    # H2
    print("\n--- H2: Transition matrix changes ---")
    jt = h2_stats["jsd_test"]
    print(f"  Observed mean JSD: {jt.get('observed_mean_jsd'):.4f}")
    print(f"  Permutation null: mean={jt.get('null_mean'):.4f}, 95th={jt.get('null_95_pct'):.4f}")
    print(f"  Permutation p = {jt.get('permutation_p')}, N = {jt.get('n')}")
    print(
        f"  Edges tested: {h2_stats['n_edges_tested']}, "
        f"significant: {h2_stats['n_significant_edges']}"
    )
    for se in h2_stats["significant_edges"]:
        print(
            f"    {se['edge']}: p_adj={se['p_adj']:.4f}, "
            f"r={se['r']:.3f}, direction={se['direction']}"
        )

    # H3
    print("\n--- H3: Spatial range contraction ---")
    for ctype in ["junction_coverage", "corridor_coverage", "dead_end_coverage"]:
        ct = h3_stats[ctype]
        print(
            f"  {ctype}: light={ct['light']}, dark={ct['dark']}, "
            f"p={ct['p']}, p_adj={ct.get('p_adj')}, r={ct['r']}, N={ct['n']}"
        )
    inter = h3_stats["de_vs_junction_drop_interaction"]
    print(
        f"  DE drop vs junction drop: "
        f"mean DE drop={inter['mean_de_drop']}, "
        f"mean junc drop={inter['mean_junc_drop']}, "
        f"p={inter['p']}, r={inter['r']}, N={inter['n']}"
    )
    dm = h3_stats["diameter"]
    print(
        f"  Diameter: light={dm['light']}, dark={dm['dark']}, "
        f"p={dm['p']}, r={dm['r']}, N={dm['n']}"
    )

    # H4
    print("\n--- H4: Increased revisitation ---")
    ri = h4_stats["revisitation_index"]
    print(
        f"  Revisitation index: light={ri['light']}, dark={ri['dark']}, "
        f"p={ri['p']}, r={ri['r']}, N={ri['n']}"
    )
    da = h4_stats["discovery_auc"]
    print(
        f"  Discovery AUC: light={da['light']}, dark={da['dark']}, "
        f"p={da['p']}, r={da['r']}, N={da['n']}"
    )

    # ===================================================================
    # Save results
    # ===================================================================
    output = {
        "metadata": {
            "n_sessions": len(session_ids),
            "session_ids": session_ids,
            "description": (
                "Tier-1 behaviour hypotheses from plan-behaviour-science.md. "
                "Tests how darkness affects maze navigation beyond simple "
                "speed reduction."
            ),
        },
        "h1_speed_coverage": h1_stats,
        "h2_transition_matrix": h2_stats,
        "h3_spatial_contraction": h3_stats,
        "h4_revisitation": h4_stats,
    }

    output_ser = _make_serializable(output)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_ser, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run behaviour hypotheses (Tier-1 and/or Tier-2)."
    )
    parser.add_argument(
        "--tier2",
        action="store_true",
        help="Run Tier-2 hypotheses (H5, H6, H8, H9, H10) only.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both Tier-1 and Tier-2 hypotheses.",
    )
    args = parser.parse_args()

    if args.all:
        main()
        main_tier2()
    elif args.tier2:
        main_tier2()
    else:
        main()
