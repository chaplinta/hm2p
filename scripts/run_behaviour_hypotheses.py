#!/usr/bin/env python3
"""Run Tier-1 behaviour hypotheses (H1-H4) from plan-behaviour-science.md.

Downloads sync.h5 files from S3 and tests four hypotheses about how
darkness affects maze navigation:

  H1: Speed-coverage partial correlation
  H2: Transition matrix changes (JSD + per-edge tests)
  H3: Spatial range contraction (per-cell-type coverage + diameter)
  H4: Increased revisitation (revisitation index + discovery AUC)

Outputs:
  - docs/manuscripts/behaviour-hypotheses-results.json
  - Human-readable summary to stdout

Usage:
  python scripts/run_behaviour_hypotheses.py
"""

from __future__ import annotations

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
    main()
