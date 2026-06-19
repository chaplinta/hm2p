#!/usr/bin/env python3
"""Run behavioural analyses for the behaviour manuscript.

Downloads sync.h5 files from S3 and computes summary statistics,
light/dark comparisons, exploration, turn bias, speed, and HD analyses.

Outputs:
  - docs/manuscripts/behaviour-results.json  (all statistics)
  - docs/manuscripts/behaviour-results-summary.md  (markdown table)

Usage:
  python scripts/run_behaviour_analysis.py
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.analysis.ahv import compute_ahv
from hm2p.maze.analysis import (
    dead_end_visits,
    exploration_efficiency,
    markov_order_comparison,
    maze_exploration_summary,
    per_junction_turn_bias,
    transition_entropy,
    transition_matrix,
    turn_bias,
)
from hm2p.maze.discretize import cell_sequence, discretize_position_fast, node_sequence
from hm2p.maze.exploration_complexity import (
    build_adjacency_indices,
    coverage_z_vs_null,
    normalized_lz76,
    occupancy_entropy,
    random_walk_coverage_null,
)
from hm2p.maze.neural import classify_frames_by_node_type, extract_junction_events
from hm2p.maze.topology import build_rose_maze

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BUCKET = "hm2p-derivatives"
S3_REGION = "ap-southeast-2"
METADATA_CSV = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
ANIMALS_CSV = Path(__file__).resolve().parent.parent / "metadata" / "animals.csv"
OUTPUT_JSON = Path(__file__).resolve().parent.parent / "docs" / "manuscripts" / "behaviour-results.json"
OUTPUT_MD = Path(__file__).resolve().parent.parent / "docs" / "manuscripts" / "behaviour-results-summary.md"

SPEED_ACTIVE_THRESHOLD = 2.5  # cm/s
SPEED_NOISE_FLOOR = 0.5  # cm/s
MIN_EPOCH_DURATION_S = 30.0  # minimum epoch duration to include
IMMOBILITY_MIN_DURATION_S = 0.5  # minimum immobility bout duration

# Build maze once
MAZE = build_rose_maze()
MAZE_ADJ_IDX = build_adjacency_indices(MAZE)
N_NULL_WALKS = 200  # random walks per epoch for the coverage null model


# ---------------------------------------------------------------------------
# Helpers
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

    r = 1 - (2W) / (n*(n+1)/2)
    where W is the smaller of the two rank sums (the test statistic
    returned by scipy with method='exact' or 'approx').
    """
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    result = sp_stats.wilcoxon(x, y, alternative="two-sided")
    W = result.statistic
    # rank-biserial: r = 1 - (2W / (n(n+1)/2))
    r = 1.0 - (2.0 * W) / (n * (n + 1) / 2.0)
    return float(r)


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size for Mann-Whitney U."""
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    # Count dominance pairs
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (nx * ny))


def wilcoxon_test(x, y, alternative="two-sided"):
    """Wilcoxon signed-rank test with effect size."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    # Remove pairs where both are NaN
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


def mannwhitney_test(x, y):
    """Mann-Whitney U test with Cliff's delta."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3:
        return {"U": None, "p": None, "cliff_d": None, "n1": nx, "n2": ny}
    try:
        U, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        cd = cliffs_delta(x, y)
        return {"U": float(U), "p": float(p), "cliff_d": cd, "n1": nx, "n2": ny}
    except Exception as e:
        return {"U": None, "p": None, "cliff_d": None, "n1": nx, "n2": ny, "error": str(e)}


def friedman_test(*groups):
    """Friedman test across multiple related groups."""
    # Align observations — only include where all groups have data
    arrays = [np.asarray(g, dtype=float) for g in groups]
    valid = np.all([np.isfinite(a) for a in arrays], axis=0)
    arrays = [a[valid] for a in arrays]
    n = len(arrays[0])
    if n < 6:
        return {"stat": None, "p": None, "n": n}
    try:
        stat, p = sp_stats.friedmanchisquare(*arrays)
        return {"stat": float(stat), "p": float(p), "n": n}
    except Exception as e:
        return {"stat": None, "p": None, "n": n, "error": str(e)}


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction. Returns adjusted p-values.

    The adjustment is: p_adj[i] = p_sorted[i] * (n - rank_i), where
    rank_i is the 0-based rank in ascending order.  A cumulative maximum
    is then applied so that adjusted p-values are non-decreasing when
    sorted by raw p-value (Holm 1979, requirement for step-down control).
    """
    pvals = np.asarray(p_values, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    # Sort and correct
    order = np.argsort(pvals)
    sorted_adj = np.array(
        [min(pvals[order[rank]] * (n - rank), 1.0) for rank in range(n)]
    )
    # Enforce monotonicity: adjusted p-values must be non-decreasing
    sorted_adj = np.maximum.accumulate(sorted_adj)
    # Map back to original order
    adjusted = np.empty(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = sorted_adj[rank]
    return adjusted.tolist()


def detect_epochs(light_on, fps):
    """Detect contiguous light/dark epochs.

    Returns list of dicts with keys: start, end, condition ('light'/'dark'),
    duration_s.
    """
    light = np.asarray(light_on, dtype=bool)
    n = len(light)
    epochs = []
    i = 0
    while i < n:
        condition = "light" if light[i] else "dark"
        start = i
        while i < n and light[i] == light[start]:
            i += 1
        dur = (i - start) / fps
        epochs.append({
            "start": start,
            "end": i,
            "condition": condition,
            "duration_s": dur,
        })
    return epochs


def detect_immobility_bouts(speed, bad_behav, fps, threshold=2.5, min_dur_s=0.5):
    """Detect contiguous immobility bouts.

    Returns list of dicts: start, end, duration_s.
    """
    immobile = (speed < threshold) & ~bad_behav
    bouts = []
    i = 0
    n = len(immobile)
    while i < n:
        if immobile[i]:
            start = i
            while i < n and immobile[i]:
                i += 1
            dur = (i - start) / fps
            if dur >= min_dur_s:
                bouts.append({"start": start, "end": i, "duration_s": dur})
        else:
            i += 1
    return bouts


# ---------------------------------------------------------------------------
# Load session data
# ---------------------------------------------------------------------------


def load_session_data(s3_client, exp_id):
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
                print(f"  STUB: {exp_id} — empty sync.h5")
                return None

            fps = float(f.attrs.get("fps_imaging", 9.6))
            data = {
                "exp_id": exp_id,
                "animal_id": animal_id,
                "sub": sub,
                "ses": ses,
                "fps": fps,
            }

            # Load behavioural arrays
            for field in [
                "x_mm", "y_mm", "x_maze", "y_maze",
                "speed_cm_s", "hd_deg", "light_on", "bad_behav",
                "frame_times", "active", "ahv_deg_s",
                "speed_head_cm_s", "speed_body_cm_s",
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
# Per-session analysis
# ---------------------------------------------------------------------------


def analyze_session(data, maze):
    """Compute all behavioural metrics for one session."""
    result = {
        "exp_id": data["exp_id"],
        "animal_id": data["animal_id"],
        "fps": data["fps"],
    }

    fps = data["fps"]
    x_mm = data["x_mm"].astype(np.float64)
    y_mm = data["y_mm"].astype(np.float64)
    x_maze = data["x_maze"].astype(np.float64)
    y_maze = data["y_maze"].astype(np.float64)
    speed = data["speed_cm_s"].astype(np.float64)
    hd_deg = data["hd_deg"].astype(np.float64)
    light_on = data["light_on"].astype(bool)
    bad_behav = data["bad_behav"].astype(bool)
    frame_times = data["frame_times"].astype(np.float64)
    ahv = data["ahv_deg_s"].astype(np.float64) if data["ahv_deg_s"] is not None else compute_ahv(hd_deg, fps)

    n_frames = len(x_mm)
    valid = ~bad_behav & np.isfinite(x_mm) & np.isfinite(y_mm)
    active = speed >= SPEED_ACTIVE_THRESHOLD

    # ---- 1. Session duration ----
    duration_s = float(frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
    usable_duration_s = float(np.sum(~bad_behav)) / fps
    result["duration_s"] = duration_s
    result["usable_duration_s"] = usable_duration_s
    result["n_frames"] = n_frames
    result["n_valid_frames"] = int(valid.sum())

    # ---- 2. Total distance ----
    valid_pos = valid.copy()
    dx = np.diff(x_mm)
    dy = np.diff(y_mm)
    # Mask transitions where either frame is bad
    valid_trans = valid_pos[:-1] & valid_pos[1:]
    dist_mm = np.sqrt(dx**2 + dy**2)
    dist_mm[~valid_trans] = 0.0
    total_distance_m = float(np.sum(dist_mm)) / 1000.0
    result["total_distance_m"] = total_distance_m

    # ---- 3. Speed: light vs dark ----
    mask_light = valid & light_on
    mask_dark = valid & ~light_on

    # All speeds (no floor)
    result["mean_speed_all"] = float(np.nanmean(speed[valid])) if valid.any() else None

    # With speed floor
    above_floor = speed >= SPEED_NOISE_FLOOR
    result["mean_speed_light_nofloor"] = float(np.nanmean(speed[mask_light])) if mask_light.any() else None
    result["mean_speed_dark_nofloor"] = float(np.nanmean(speed[mask_dark])) if mask_dark.any() else None
    result["mean_speed_light"] = float(np.nanmean(speed[mask_light & above_floor])) if (mask_light & above_floor).any() else None
    result["mean_speed_dark"] = float(np.nanmean(speed[mask_dark & above_floor])) if (mask_dark & above_floor).any() else None
    result["median_speed_light"] = float(np.nanmedian(speed[mask_light])) if mask_light.any() else None
    result["median_speed_dark"] = float(np.nanmedian(speed[mask_dark])) if mask_dark.any() else None

    # Speed percentiles
    if mask_light.any():
        sl = speed[mask_light]
        result["speed_pct_25_light"] = float(np.percentile(sl, 25))
        result["speed_pct_75_light"] = float(np.percentile(sl, 75))
        result["speed_pct_95_light"] = float(np.percentile(sl, 95))
        result["frac_high_speed_light"] = float(np.mean(sl > 10))
    if mask_dark.any():
        sd = speed[mask_dark]
        result["speed_pct_25_dark"] = float(np.percentile(sd, 25))
        result["speed_pct_75_dark"] = float(np.percentile(sd, 75))
        result["speed_pct_95_dark"] = float(np.percentile(sd, 95))
        result["frac_high_speed_dark"] = float(np.mean(sd > 10))

    # ---- 4. Fraction active ----
    result["frac_active_light"] = float(np.mean(active[mask_light])) if mask_light.any() else None
    result["frac_active_dark"] = float(np.mean(active[mask_dark])) if mask_dark.any() else None
    result["frac_active_all"] = float(np.mean(active[valid])) if valid.any() else None

    # ---- 5. Maze cell coverage ----
    cell_indices = discretize_position_fast(x_maze, y_maze, maze)
    cell_indices[~valid] = -1
    unique_cells = len(set(int(c) for c in cell_indices if c >= 0))
    result["unique_cells_visited"] = unique_cells
    result["coverage_frac"] = unique_cells / maze.n_cells

    # ---- 6. Exploration summary ----
    cells_visited, cell_times = cell_sequence(cell_indices)
    nodes_visited, node_times = node_sequence(cell_indices, maze)

    # ---- 7. Per-epoch coverage (Figure 3B) ----
    epochs = detect_epochs(light_on, fps)
    epoch_coverages_light = []
    epoch_coverages_dark = []
    # Per-epoch locomotion distance (body centroid path length, metres). Parallel
    # to coverage: total distance walked within each ~1-min epoch, so it is time-
    # controlled and uses the body (x_mm/y_mm = mid_back+mouse_center+tail_base
    # centroid), not the head.
    epoch_dist_light = []
    epoch_dist_dark = []
    # Per-epoch exploration efficiency, controlling for how much they moved:
    #   cells_per_m = unique cells visited / distance walked in the epoch
    #   revisit     = total cell-entries / unique cells (1 = never retraced)
    # These separate "explores new ground" from "just moves more". Only computed
    # for epochs with real movement (>= MIN_EPOCH_DIST_M) to avoid blow-up.
    MIN_EPOCH_DIST_M = 0.5
    epoch_cpm_light, epoch_cpm_dark = [], []
    epoch_revisit_light, epoch_revisit_dark = [], []
    # Supplementary coverage methods (see docs/plan-coverage-supplementary-methods.md):
    #   occupancy entropy   — time-weighted spread over cells (bits)
    #   normalised LZ76     — route-sequence compressibility (lower = stereotyped)
    #   coverage z vs null  — coverage relative to a random walk of equal steps
    epoch_ent_light, epoch_ent_dark = [], []
    epoch_lz_light, epoch_lz_dark = [], []
    epoch_zcov_light, epoch_zcov_dark = [], []
    null_rng = np.random.default_rng(0)
    for ep in epochs:
        if ep["duration_s"] < MIN_EPOCH_DURATION_S:
            continue
        ep_valid = valid[ep["start"]:ep["end"]]
        ep_ci = cell_indices[ep["start"]:ep["end"]].copy()
        ep_ci[~ep_valid] = -1
        unique_in_epoch = len(set(int(c) for c in ep_ci if c >= 0))
        cov = unique_in_epoch / maze.n_cells
        # Distance: sum body-centroid step lengths over valid transitions whose
        # start frame falls in this epoch (dist_mm has length n_frames-1).
        t0, t1 = ep["start"], min(ep["end"], len(dist_mm))
        ep_dist_m = float(np.sum(dist_mm[t0:t1])) / 1000.0
        ep_entropy = occupancy_entropy(ep_ci)
        cpm = revisit = lz = zcov = None
        if ep_dist_m >= MIN_EPOCH_DIST_M and unique_in_epoch > 0:
            seq, _ = cell_sequence(ep_ci)
            cpm = unique_in_epoch / ep_dist_m
            revisit = len(seq) / unique_in_epoch
            if len(seq) >= 2:
                lz = normalized_lz76(seq.tolist())
                null = random_walk_coverage_null(
                    MAZE_ADJ_IDX, int(seq[0]), len(seq) - 1, N_NULL_WALKS, null_rng
                )
                zcov = coverage_z_vs_null(unique_in_epoch, null)
        if ep["condition"] == "light":
            epoch_coverages_light.append(cov)
            epoch_dist_light.append(ep_dist_m)
            epoch_ent_light.append(ep_entropy)
            if cpm is not None:
                epoch_cpm_light.append(cpm)
                epoch_revisit_light.append(revisit)
            if lz is not None:
                epoch_lz_light.append(lz)
                if not np.isnan(zcov):
                    epoch_zcov_light.append(zcov)
        else:
            epoch_coverages_dark.append(cov)
            epoch_dist_dark.append(ep_dist_m)
            epoch_ent_dark.append(ep_entropy)
            if cpm is not None:
                epoch_cpm_dark.append(cpm)
                epoch_revisit_dark.append(revisit)
            if lz is not None:
                epoch_lz_dark.append(lz)
                if not np.isnan(zcov):
                    epoch_zcov_dark.append(zcov)

    result["mean_epoch_coverage_light"] = float(np.mean(epoch_coverages_light)) if epoch_coverages_light else None
    result["mean_epoch_coverage_dark"] = float(np.mean(epoch_coverages_dark)) if epoch_coverages_dark else None
    result["mean_epoch_distance_light_m"] = float(np.mean(epoch_dist_light)) if epoch_dist_light else None
    result["mean_epoch_distance_dark_m"] = float(np.mean(epoch_dist_dark)) if epoch_dist_dark else None
    result["mean_epoch_cells_per_m_light"] = float(np.mean(epoch_cpm_light)) if epoch_cpm_light else None
    result["mean_epoch_cells_per_m_dark"] = float(np.mean(epoch_cpm_dark)) if epoch_cpm_dark else None
    result["mean_epoch_revisit_light"] = float(np.mean(epoch_revisit_light)) if epoch_revisit_light else None
    result["mean_epoch_revisit_dark"] = float(np.mean(epoch_revisit_dark)) if epoch_revisit_dark else None
    result["mean_epoch_entropy_light"] = float(np.mean(epoch_ent_light)) if epoch_ent_light else None
    result["mean_epoch_entropy_dark"] = float(np.mean(epoch_ent_dark)) if epoch_ent_dark else None
    result["mean_epoch_lz_light"] = float(np.mean(epoch_lz_light)) if epoch_lz_light else None
    result["mean_epoch_lz_dark"] = float(np.mean(epoch_lz_dark)) if epoch_lz_dark else None
    result["mean_epoch_zcov_light"] = float(np.mean(epoch_zcov_light)) if epoch_zcov_light else None
    result["mean_epoch_zcov_dark"] = float(np.mean(epoch_zcov_dark)) if epoch_zcov_dark else None
    result["n_light_epochs"] = len(epoch_coverages_light)
    result["n_dark_epochs"] = len(epoch_coverages_dark)

    # ---- 8. Dead-end visits (Figure 3C) ----
    # Light subset
    ci_light = cell_indices.copy()
    ci_light[~(valid & light_on)] = -1
    cs_light, _ = cell_sequence(ci_light)
    de_light = dead_end_visits(cs_light, maze) if len(cs_light) > 0 else {}
    total_de_visits_light = sum(v["visits"] for v in de_light.values())
    light_duration_min = float(np.sum(valid & light_on)) / fps / 60.0

    ci_dark = cell_indices.copy()
    ci_dark[~(valid & ~light_on)] = -1
    cs_dark, _ = cell_sequence(ci_dark)
    de_dark = dead_end_visits(cs_dark, maze) if len(cs_dark) > 0 else {}
    total_de_visits_dark = sum(v["visits"] for v in de_dark.values())
    dark_duration_min = float(np.sum(valid & ~light_on)) / fps / 60.0

    result["dead_end_rate_light"] = total_de_visits_light / light_duration_min if light_duration_min > 0 else None
    result["dead_end_rate_dark"] = total_de_visits_dark / dark_duration_min if dark_duration_min > 0 else None

    # ---- 9. Exploration efficiency (Figure 3D) ----
    ns_light, _ = node_sequence(ci_light, maze)
    ns_dark, _ = node_sequence(ci_dark, maze)
    window_sizes = np.array([2, 3, 5, 8, 13, 21])
    eff_light_ws, eff_light = exploration_efficiency(ns_light, window_sizes) if len(ns_light) > 2 else (np.array([]), np.array([]))
    eff_dark_ws, eff_dark = exploration_efficiency(ns_dark, window_sizes) if len(ns_dark) > 2 else (np.array([]), np.array([]))
    result["exploration_efficiency_light"] = dict(zip(eff_light_ws.tolist(), eff_light.tolist())) if len(eff_light) > 0 else {}
    result["exploration_efficiency_dark"] = dict(zip(eff_dark_ws.tolist(), eff_dark.tolist())) if len(eff_dark) > 0 else {}

    # ---- 10. Turn bias (Figure 4) ----
    # Global turn bias - light
    tb_light = turn_bias(cs_light, maze) if len(cs_light) > 2 else {"left_frac": None}
    tb_dark = turn_bias(cs_dark, maze) if len(cs_dark) > 2 else {"left_frac": None}
    result["turn_bias_light"] = tb_light
    result["turn_bias_dark"] = tb_dark

    # Per-junction turn bias
    pjt_light = per_junction_turn_bias(cs_light, maze) if len(cs_light) > 2 else {}
    pjt_dark = per_junction_turn_bias(cs_dark, maze) if len(cs_dark) > 2 else {}
    # Convert tuple keys to strings for JSON
    result["per_junction_light"] = {str(k): v for k, v in pjt_light.items()}
    result["per_junction_dark"] = {str(k): v for k, v in pjt_dark.items()}

    # Back-tracking rate (Figure 4D)
    tb_all_light = tb_light
    total_junc_light = sum(tb_all_light.get(k, 0) for k in ["left", "right", "back", "forward"])
    result["back_rate_light"] = tb_all_light.get("back", 0) / total_junc_light if total_junc_light > 0 else None
    tb_all_dark = tb_dark
    total_junc_dark = sum(tb_all_dark.get(k, 0) for k in ["left", "right", "back", "forward"])
    result["back_rate_dark"] = tb_all_dark.get("back", 0) / total_junc_dark if total_junc_dark > 0 else None

    # ---- 11. Sequential turn autocorrelation (Figure 4C) ----
    def _turn_autocorrelation(cs, maze_obj):
        """Compute lag-1 autocorrelation of L/R turn sequence."""
        if len(cs) < 3:
            return None
        junction_indices = {maze_obj.cell_to_idx[j] for j in maze_obj.junctions}
        cell_list = maze_obj.cell_list
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
        # Lag-1 Spearman correlation
        t1 = turns[:-1]
        t2 = turns[1:]
        if np.std(t1) == 0 or np.std(t2) == 0:
            return 0.0
        rho, _ = sp_stats.spearmanr(t1, t2)
        return float(rho)

    from hm2p.maze.analysis import classify_turn  # already imported at top via turn_bias internals

    result["turn_autocorr_light"] = _turn_autocorrelation(cs_light, maze)
    result["turn_autocorr_dark"] = _turn_autocorrelation(cs_dark, maze)
    result["turn_autocorr_all"] = _turn_autocorrelation(cells_visited, maze)

    # ---- 12. Speed by node type (Figure 6B) ----
    node_masks = classify_frames_by_node_type(cell_indices, maze)
    speed_junction = float(np.mean(speed[valid & active & node_masks["junction"]])) if (valid & active & node_masks["junction"]).any() else None
    speed_corridor = float(np.mean(speed[valid & active & node_masks["corridor"]])) if (valid & active & node_masks["corridor"]).any() else None
    speed_dead_end = float(np.mean(speed[valid & active & node_masks["dead_end"]])) if (valid & active & node_masks["dead_end"]).any() else None
    result["speed_junction"] = speed_junction
    result["speed_corridor"] = speed_corridor
    result["speed_dead_end"] = speed_dead_end

    # ---- 13. Immobility bouts (Figure 2D) ----
    bouts_light = detect_immobility_bouts(
        speed[light_on & ~bad_behav].copy() if (light_on & ~bad_behav).any() else np.array([]),
        np.zeros_like(speed[light_on & ~bad_behav], dtype=bool) if (light_on & ~bad_behav).any() else np.array([], dtype=bool),
        fps,
    )
    bouts_dark = detect_immobility_bouts(
        speed[~light_on & ~bad_behav].copy() if (~light_on & ~bad_behav).any() else np.array([]),
        np.zeros_like(speed[~light_on & ~bad_behav], dtype=bool) if (~light_on & ~bad_behav).any() else np.array([], dtype=bool),
        fps,
    )
    # Actually, we should detect bouts on the full time series with proper masking
    all_bouts_light = []
    all_bouts_dark = []
    # Re-detect on full series with condition masks
    immobile = (speed < SPEED_ACTIVE_THRESHOLD) & ~bad_behav
    i = 0
    while i < n_frames:
        if immobile[i]:
            start = i
            while i < n_frames and immobile[i]:
                i += 1
            dur = (i - start) / fps
            if dur >= IMMOBILITY_MIN_DURATION_S:
                # Check if majority of bout is in light or dark
                bout_light_frac = np.mean(light_on[start:i])
                if bout_light_frac > 0.5:
                    all_bouts_light.append(dur)
                else:
                    all_bouts_dark.append(dur)
        else:
            i += 1

    result["median_immobility_bout_light"] = float(np.median(all_bouts_light)) if all_bouts_light else None
    result["median_immobility_bout_dark"] = float(np.median(all_bouts_dark)) if all_bouts_dark else None
    result["n_immobility_bouts_light"] = len(all_bouts_light)
    result["n_immobility_bouts_dark"] = len(all_bouts_dark)

    # ---- 14. HD distribution (Figure 5) ----
    hd_finite = np.isfinite(hd_deg)
    hd_wrapped = np.where(hd_finite, hd_deg % 360.0, np.nan)
    mask_hd_light = valid & light_on & active & hd_finite
    mask_hd_dark = valid & ~light_on & active & hd_finite

    def _hd_resultant_length(hd_w, mask):
        """Mean resultant length of circular distribution."""
        hd_vals = hd_w[mask]
        hd_vals = hd_vals[np.isfinite(hd_vals)]
        if len(hd_vals) < 10:
            return None
        hd_r = np.deg2rad(hd_vals)
        C = np.mean(np.cos(hd_r))
        S = np.mean(np.sin(hd_r))
        return float(np.sqrt(C**2 + S**2))

    result["hd_mrl_light"] = _hd_resultant_length(hd_wrapped, mask_hd_light)
    result["hd_mrl_dark"] = _hd_resultant_length(hd_wrapped, mask_hd_dark)

    # ---- 15. AHV (Figure 5C, 5D) ----
    ahv_abs = np.abs(ahv)
    ahv_finite = np.isfinite(ahv_abs)
    mask_ahv_light = mask_hd_light & ahv_finite
    mask_ahv_dark = mask_hd_dark & ahv_finite
    result["median_ahv_light"] = float(np.median(ahv_abs[mask_ahv_light])) if mask_ahv_light.any() else None
    result["median_ahv_dark"] = float(np.median(ahv_abs[mask_ahv_dark])) if mask_ahv_dark.any() else None
    result["pct95_ahv_light"] = float(np.percentile(ahv_abs[mask_ahv_light], 95)) if mask_ahv_light.any() else None
    result["pct95_ahv_dark"] = float(np.percentile(ahv_abs[mask_ahv_dark], 95)) if mask_ahv_dark.any() else None

    # ---- 16. Transition entropy (Supp Fig S1) ----
    n_cells = maze.n_cells
    if len(cs_light) > 10:
        tm_light = transition_matrix(cs_light, n_cells)
        te_light = transition_entropy(tm_light, cs_light)
    else:
        te_light = None
    if len(cs_dark) > 10:
        tm_dark = transition_matrix(cs_dark, n_cells)
        te_dark = transition_entropy(tm_dark, cs_dark)
    else:
        te_dark = None
    result["transition_entropy_light"] = te_light
    result["transition_entropy_dark"] = te_dark

    # ---- 17. Markov order comparison (Supp Fig S1C) ----
    if len(cells_visited) > 20:
        moc = markov_order_comparison(cells_visited, n_cells)
        result["markov_delta_bic"] = moc["delta_bic"]
        result["markov_preferred_order"] = moc["preferred_order"]
    else:
        result["markov_delta_bic"] = None
        result["markov_preferred_order"] = None

    # ---- 18. Junction approach speed (Figure 6C) ----
    events = extract_junction_events(cell_indices, maze)
    pre_speeds = []
    at_speeds = []
    post_speeds = []
    window_frames = int(round(1.0 * fps))  # 1 second window
    for ev in events:
        jf = ev["junction_frame"]
        # Find how many frames at junction
        j_end = jf
        while j_end < n_frames and cell_indices[j_end] == ev["junction"]:
            j_end += 1
        # Pre-junction: frames before junction entry
        pre_start = max(0, jf - window_frames)
        pre_end = jf
        # Post-junction: frames after junction exit
        post_start = j_end
        post_end = min(n_frames, j_end + window_frames)

        if pre_end - pre_start >= 3 and post_end - post_start >= 3:
            pre_mask = valid[pre_start:pre_end]
            at_mask = valid[jf:j_end]
            post_mask = valid[post_start:post_end]
            if pre_mask.any() and at_mask.any() and post_mask.any():
                pre_speeds.append(float(np.nanmean(speed[pre_start:pre_end][pre_mask])))
                at_speeds.append(float(np.nanmean(speed[jf:j_end][at_mask])))
                post_speeds.append(float(np.nanmean(speed[post_start:post_end][post_mask])))

    # Filter out any NaN values from junction approach speeds
    pre_speeds_clean = [s for s in pre_speeds if np.isfinite(s)]
    at_speeds_clean = [s for s in at_speeds if np.isfinite(s)]
    post_speeds_clean = [s for s in post_speeds if np.isfinite(s)]
    result["mean_pre_junction_speed"] = float(np.mean(pre_speeds_clean)) if pre_speeds_clean else None
    result["mean_at_junction_speed"] = float(np.mean(at_speeds_clean)) if at_speeds_clean else None
    result["mean_post_junction_speed"] = float(np.mean(post_speeds_clean)) if post_speeds_clean else None
    result["n_junction_events"] = len(events)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("BEHAVIOURAL ANALYSIS — hm2p project")
    print("=" * 70)

    # Load metadata
    with open(METADATA_CSV) as f:
        experiments = list(csv.DictReader(f))
    with open(ANIMALS_CSV) as f:
        animals = {row["animal_id"]: row for row in csv.DictReader(f)}

    sessions = []
    for row in experiments:
        eid = row["exp_id"]
        parts = eid.split("_")
        animal_id = parts[-1]
        sessions.append({
            "exp_id": eid,
            "exp_index": int(row["exp_index"]),
            "animal_id": animal_id,
            "celltype": animals.get(animal_id, {}).get("celltype", "unknown"),
            "sex": animals.get(animal_id, {}).get("sex", "unknown"),
            "exclude": str(row.get("exclude", "0")).strip() == "1",
            "primary": str(row.get("primary_exp", "1")).strip() != "0",
        })

    print(f"\nTotal sessions: {len(sessions)}")
    print(f"Excluded: {sum(1 for s in sessions if s['exclude'])}")
    print(f"Usable: {sum(1 for s in sessions if not s['exclude'])}")

    # Download and analyze all sessions
    s3 = boto3.client("s3", region_name=S3_REGION)
    all_results = []

    for sess in sessions:
        eid = sess["exp_id"]
        print(f"\n--- Session {sess['exp_index']}: {eid} ---")
        print(f"    Animal: {sess['animal_id']}, Celltype: {sess['celltype']}, "
              f"Exclude: {sess['exclude']}, Primary: {sess['primary']}")

        data = load_session_data(s3, eid)
        if data is None:
            print(f"    SKIPPED (no data)")
            all_results.append({
                "exp_id": eid,
                "exp_index": sess["exp_index"],
                "animal_id": sess["animal_id"],
                "celltype": sess["celltype"],
                "sex": sess["sex"],
                "exclude": sess["exclude"],
                "primary": sess["primary"],
                "status": "no_data",
            })
            continue

        result = analyze_session(data, MAZE)
        result["exp_index"] = sess["exp_index"]
        result["celltype"] = sess["celltype"]
        result["sex"] = sess["sex"]
        result["exclude"] = sess["exclude"]
        result["primary"] = sess["primary"]
        result["status"] = "ok"

        print(f"    Duration: {result['duration_s']:.0f}s, Distance: {result['total_distance_m']:.1f}m, "
              f"Speed L/D: {result.get('mean_speed_light', 'N/A')}/{result.get('mean_speed_dark', 'N/A')} cm/s, "
              f"Coverage: {result['coverage_frac']:.2f}")

        all_results.append(result)

    # ===================================================================
    # Cross-session statistics
    # ===================================================================
    print("\n" + "=" * 70)
    print("CROSS-SESSION ANALYSES")
    print("=" * 70)

    # Filter to usable sessions
    usable = [r for r in all_results if not r["exclude"] and r["status"] == "ok"]
    primary_only = [r for r in usable if r["primary"]]
    penk = [r for r in usable if r["celltype"] == "penk"]
    nonpenk = [r for r in usable if r["celltype"] == "nonpenk"]

    print(f"\nUsable sessions: {len(usable)}")
    print(f"Primary-only sessions: {len(primary_only)}")
    print(f"Penk+ animals sessions: {len(penk)}")
    print(f"Penk-CamKII+ animals sessions: {len(nonpenk)}")

    stats = {"dataset": {
        "n_usable_sessions": len(usable),
        "n_primary_sessions": len(primary_only),
        "n_penk_sessions": len(penk),
        "n_nonpenk_sessions": len(nonpenk),
        "n_animals": len(set(r["animal_id"] for r in usable)),
        "n_penk_animals": len(set(r["animal_id"] for r in penk)),
        "n_nonpenk_animals": len(set(r["animal_id"] for r in nonpenk)),
    }}

    # ---- Summary statistics (Analysis 1) ----
    stats["summary"] = {
        "total_distance_m": {
            "mean": float(np.mean([r["total_distance_m"] for r in usable])),
            "sd": float(np.std([r["total_distance_m"] for r in usable], ddof=1)),
            "median": float(np.median([r["total_distance_m"] for r in usable])),
            "min": float(np.min([r["total_distance_m"] for r in usable])),
            "max": float(np.max([r["total_distance_m"] for r in usable])),
        },
        "duration_s": {
            "mean": float(np.mean([r["duration_s"] for r in usable])),
            "sd": float(np.std([r["duration_s"] for r in usable], ddof=1)),
            "median": float(np.median([r["duration_s"] for r in usable])),
        },
        "usable_duration_s": {
            "mean": float(np.mean([r["usable_duration_s"] for r in usable])),
            "sd": float(np.std([r["usable_duration_s"] for r in usable], ddof=1)),
        },
        "mean_speed_all": {
            "mean": float(np.mean([r["mean_speed_all"] for r in usable if r["mean_speed_all"] is not None])),
            "sd": float(np.std([r["mean_speed_all"] for r in usable if r["mean_speed_all"] is not None], ddof=1)),
        },
        "coverage_frac": {
            "mean": float(np.mean([r["coverage_frac"] for r in usable])),
            "sd": float(np.std([r["coverage_frac"] for r in usable], ddof=1)),
        },
        "unique_cells_visited": {
            "mean": float(np.mean([r["unique_cells_visited"] for r in usable])),
            "sd": float(np.std([r["unique_cells_visited"] for r in usable], ddof=1)),
            "median": float(np.median([r["unique_cells_visited"] for r in usable])),
        },
        "frac_active_all": {
            "mean": float(np.mean([r["frac_active_all"] for r in usable if r["frac_active_all"] is not None])),
            "sd": float(np.std([r["frac_active_all"] for r in usable if r["frac_active_all"] is not None], ddof=1)),
        },
    }

    # ---- Figure 2: Speed and locomotion ----
    def _extract_paired(results, key_light, key_dark):
        """Extract paired light/dark values, dropping missing or NaN."""
        pairs = []
        for r in results:
            vl = r.get(key_light)
            vd = r.get(key_dark)
            if vl is not None and vd is not None:
                # Handle nested dicts (e.g., turn_bias_light → left_frac)
                if isinstance(vl, (int, float)) and isinstance(vd, (int, float)):
                    if np.isfinite(vl) and np.isfinite(vd):
                        pairs.append((vl, vd))
        if not pairs:
            return np.array([]), np.array([])
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

    # Speed light vs dark
    sp_l, sp_d = _extract_paired(usable, "median_speed_light", "median_speed_dark")
    stats["figure2"] = {}
    stats["figure2"]["speed_light_vs_dark"] = {
        "n": len(sp_l),
        "median_light": float(np.median(sp_l)) if len(sp_l) > 0 else None,
        "median_dark": float(np.median(sp_d)) if len(sp_d) > 0 else None,
        "mean_light": float(np.mean(sp_l)) if len(sp_l) > 0 else None,
        "mean_dark": float(np.mean(sp_d)) if len(sp_d) > 0 else None,
        "test": wilcoxon_test(sp_l, sp_d),
    }
    print(f"\nSpeed L vs D: median {np.median(sp_l):.2f} vs {np.median(sp_d):.2f} cm/s")

    # Fraction active light vs dark
    fa_l, fa_d = _extract_paired(usable, "frac_active_light", "frac_active_dark")
    stats["figure2"]["frac_active_light_vs_dark"] = {
        "n": len(fa_l),
        "mean_light": float(np.mean(fa_l)) if len(fa_l) > 0 else None,
        "mean_dark": float(np.mean(fa_d)) if len(fa_d) > 0 else None,
        "test": wilcoxon_test(fa_l, fa_d),
    }
    print(f"Frac active L vs D: {np.mean(fa_l):.3f} vs {np.mean(fa_d):.3f}")

    # Immobility bout duration
    ib_l, ib_d = _extract_paired(usable, "median_immobility_bout_light", "median_immobility_bout_dark")
    stats["figure2"]["immobility_bout_duration"] = {
        "n": len(ib_l),
        "median_light": float(np.median(ib_l)) if len(ib_l) > 0 else None,
        "median_dark": float(np.median(ib_d)) if len(ib_d) > 0 else None,
        "test": wilcoxon_test(ib_l, ib_d),
    }

    # Per-epoch locomotion distance (body centroid) light vs dark
    di_l, di_d = _extract_paired(
        usable, "mean_epoch_distance_light_m", "mean_epoch_distance_dark_m"
    )
    stats["figure2"]["epoch_distance_light_vs_dark"] = {
        "n": len(di_l),
        "median_light": float(np.median(di_l)) if len(di_l) > 0 else None,
        "median_dark": float(np.median(di_d)) if len(di_d) > 0 else None,
        "mean_light": float(np.mean(di_l)) if len(di_l) > 0 else None,
        "mean_dark": float(np.mean(di_d)) if len(di_d) > 0 else None,
        "test": wilcoxon_test(di_l, di_d),
    }
    if len(di_l) > 0:
        print(
            f"Distance/epoch L vs D: {np.median(di_l):.2f} vs "
            f"{np.median(di_d):.2f} m (body)"
        )

    # Holm-Bonferroni for Figure 2 (4 tests)
    fig2_pvals = [
        stats["figure2"]["speed_light_vs_dark"]["test"].get("p"),
        stats["figure2"]["frac_active_light_vs_dark"]["test"].get("p"),
        stats["figure2"]["immobility_bout_duration"]["test"].get("p"),
        stats["figure2"]["epoch_distance_light_vs_dark"]["test"].get("p"),
    ]
    fig2_pvals_clean = [p if p is not None else 1.0 for p in fig2_pvals]
    fig2_adjusted = holm_bonferroni(fig2_pvals_clean)
    stats["figure2"]["holm_bonferroni_adjusted_p"] = {
        "speed": fig2_adjusted[0],
        "frac_active": fig2_adjusted[1],
        "immobility_bout": fig2_adjusted[2],
        "epoch_distance": fig2_adjusted[3],
    }

    # ---- Figure 3: Exploration and coverage (PRIORITY) ----
    stats["figure3"] = {}

    # Per-epoch coverage
    cov_l, cov_d = _extract_paired(usable, "mean_epoch_coverage_light", "mean_epoch_coverage_dark")
    stats["figure3"]["epoch_coverage_light_vs_dark"] = {
        "n": len(cov_l),
        "mean_light": float(np.mean(cov_l)) if len(cov_l) > 0 else None,
        "mean_dark": float(np.mean(cov_d)) if len(cov_d) > 0 else None,
        "test": wilcoxon_test(cov_l, cov_d),
    }
    print(f"\nEpoch coverage L vs D: {np.mean(cov_l):.3f} vs {np.mean(cov_d):.3f}")

    # Dead-end visit rate
    de_l, de_d = _extract_paired(usable, "dead_end_rate_light", "dead_end_rate_dark")
    stats["figure3"]["dead_end_rate_light_vs_dark"] = {
        "n": len(de_l),
        "mean_light": float(np.mean(de_l)) if len(de_l) > 0 else None,
        "mean_dark": float(np.mean(de_d)) if len(de_d) > 0 else None,
        "test": wilcoxon_test(de_l, de_d),
    }
    print(f"Dead-end rate L vs D: {np.mean(de_l):.2f} vs {np.mean(de_d):.2f} visits/min")

    # Exploration efficiency at matched window sizes
    # Pool across sessions at window=5 (common size)
    eff5_l = [r["exploration_efficiency_light"].get(5, r["exploration_efficiency_light"].get(5.0)) for r in usable if r.get("exploration_efficiency_light")]
    eff5_d = [r["exploration_efficiency_dark"].get(5, r["exploration_efficiency_dark"].get(5.0)) for r in usable if r.get("exploration_efficiency_dark")]
    # Align — only sessions with both
    eff_paired = []
    for r in usable:
        el = r.get("exploration_efficiency_light", {})
        ed = r.get("exploration_efficiency_dark", {})
        vl = el.get(5) or el.get(5.0)
        vd = ed.get(5) or ed.get(5.0)
        if vl is not None and vd is not None:
            eff_paired.append((vl, vd))
    if eff_paired:
        eff_l_arr = np.array([p[0] for p in eff_paired])
        eff_d_arr = np.array([p[1] for p in eff_paired])
        stats["figure3"]["exploration_efficiency_w5"] = {
            "n": len(eff_l_arr),
            "mean_light": float(np.mean(eff_l_arr)),
            "mean_dark": float(np.mean(eff_d_arr)),
            "test": wilcoxon_test(eff_l_arr, eff_d_arr),
        }
    else:
        stats["figure3"]["exploration_efficiency_w5"] = {"n": 0, "test": {"p": None}}

    # New cells per metre — the direct test of whether reduced dark coverage is
    # more than just reduced locomotion.
    cpm_l, cpm_d = _extract_paired(usable, "mean_epoch_cells_per_m_light", "mean_epoch_cells_per_m_dark")
    stats["figure3"]["cells_per_m_light_vs_dark"] = {
        "n": len(cpm_l),
        "mean_light": float(np.mean(cpm_l)) if len(cpm_l) > 0 else None,
        "mean_dark": float(np.mean(cpm_d)) if len(cpm_d) > 0 else None,
        "median_light": float(np.median(cpm_l)) if len(cpm_l) > 0 else None,
        "median_dark": float(np.median(cpm_d)) if len(cpm_d) > 0 else None,
        "test": wilcoxon_test(cpm_l, cpm_d),
    }
    if len(cpm_l) > 0:
        print(f"Cells per metre L vs D: {np.mean(cpm_l):.3f} vs {np.mean(cpm_d):.3f}")

    # Revisitation (entries per unique cell).
    rev_l, rev_d = _extract_paired(usable, "mean_epoch_revisit_light", "mean_epoch_revisit_dark")
    stats["figure3"]["revisit_light_vs_dark"] = {
        "n": len(rev_l),
        "mean_light": float(np.mean(rev_l)) if len(rev_l) > 0 else None,
        "mean_dark": float(np.mean(rev_d)) if len(rev_d) > 0 else None,
        "test": wilcoxon_test(rev_l, rev_d),
    }
    if len(rev_l) > 0:
        print(f"Revisitation L vs D: {np.mean(rev_l):.3f} vs {np.mean(rev_d):.3f}")

    # Supplementary coverage methods: occupancy entropy, LZ compressibility,
    # coverage vs random-walk null.
    for key, (kl, kd) in {
        "occupancy_entropy_light_vs_dark": ("mean_epoch_entropy_light", "mean_epoch_entropy_dark"),
        "lz_compressibility_light_vs_dark": ("mean_epoch_lz_light", "mean_epoch_lz_dark"),
        "coverage_vs_null_light_vs_dark": ("mean_epoch_zcov_light", "mean_epoch_zcov_dark"),
    }.items():
        xl, xd = _extract_paired(usable, kl, kd)
        stats["figure3"][key] = {
            "n": len(xl),
            "mean_light": float(np.mean(xl)) if len(xl) > 0 else None,
            "mean_dark": float(np.mean(xd)) if len(xd) > 0 else None,
            "test": wilcoxon_test(xl, xd),
        }
        if len(xl) > 0:
            print(f"{key}: {np.mean(xl):.3f} vs {np.mean(xd):.3f}")

    # Holm-Bonferroni for Figure 3
    fig3_keys = [
        "epoch_coverage_light_vs_dark",
        "dead_end_rate_light_vs_dark",
        "exploration_efficiency_w5",
        "cells_per_m_light_vs_dark",
        "revisit_light_vs_dark",
        "occupancy_entropy_light_vs_dark",
        "lz_compressibility_light_vs_dark",
        "coverage_vs_null_light_vs_dark",
    ]
    fig3_pvals_clean = [
        (stats["figure3"].get(k, {}).get("test", {}).get("p") or 1.0) for k in fig3_keys
    ]
    fig3_adjusted = holm_bonferroni(fig3_pvals_clean)
    stats["figure3"]["holm_bonferroni_adjusted_p"] = {
        "epoch_coverage": fig3_adjusted[0],
        "dead_end_rate": fig3_adjusted[1],
        "exploration_efficiency": fig3_adjusted[2],
        "cells_per_m": fig3_adjusted[3],
        "revisit": fig3_adjusted[4],
        "occupancy_entropy": fig3_adjusted[5],
        "lz_compressibility": fig3_adjusted[6],
        "coverage_vs_null": fig3_adjusted[7],
    }

    # ---- Figure 4: Turn behaviour (PRIORITY) ----
    stats["figure4"] = {}

    # Left fraction light vs dark (4B)
    lf_l, lf_d = _extract_paired(usable, "turn_bias_light", "turn_bias_dark")
    # Extract left_frac from dicts
    lf_l_vals = np.array([r["turn_bias_light"]["left_frac"] for r in usable
                          if r.get("turn_bias_light", {}).get("left_frac") is not None
                          and r.get("turn_bias_dark", {}).get("left_frac") is not None])
    lf_d_vals = np.array([r["turn_bias_dark"]["left_frac"] for r in usable
                          if r.get("turn_bias_light", {}).get("left_frac") is not None
                          and r.get("turn_bias_dark", {}).get("left_frac") is not None])
    stats["figure4"]["left_frac_light_vs_dark"] = {
        "n": len(lf_l_vals),
        "mean_light": float(np.mean(lf_l_vals)) if len(lf_l_vals) > 0 else None,
        "mean_dark": float(np.mean(lf_d_vals)) if len(lf_d_vals) > 0 else None,
        "test": wilcoxon_test(lf_l_vals, lf_d_vals),
    }
    print(f"\nLeft fraction L vs D: {np.mean(lf_l_vals):.3f} vs {np.mean(lf_d_vals):.3f}")

    # Sequential turn autocorrelation (4C)
    ac_l, ac_d = _extract_paired(usable, "turn_autocorr_light", "turn_autocorr_dark")
    stats["figure4"]["turn_autocorr_light_vs_dark"] = {
        "n": len(ac_l),
        "mean_light": float(np.mean(ac_l)) if len(ac_l) > 0 else None,
        "mean_dark": float(np.mean(ac_d)) if len(ac_d) > 0 else None,
        "test_light_vs_dark": wilcoxon_test(ac_l, ac_d),
    }
    # One-sample test: does autocorrelation differ from 0?
    ac_all = np.array([r["turn_autocorr_all"] for r in usable if r.get("turn_autocorr_all") is not None])
    if len(ac_all) >= 6:
        try:
            w_res = sp_stats.wilcoxon(ac_all, alternative="two-sided")
            n_ac = len(ac_all)
            r_rb = 1.0 - (2.0 * w_res.statistic) / (n_ac * (n_ac + 1) / 2.0)
            stats["figure4"]["turn_autocorr_vs_zero"] = {
                "n": n_ac,
                "mean": float(np.mean(ac_all)),
                "median": float(np.median(ac_all)),
                "test": {
                    "stat": float(w_res.statistic),
                    "p": float(w_res.pvalue),
                    "r": float(r_rb),
                    "n": n_ac,
                    "test": "wilcoxon_one_sample",
                },
            }
        except Exception:
            stats["figure4"]["turn_autocorr_vs_zero"] = {"n": len(ac_all), "test": {"p": None}}
    else:
        stats["figure4"]["turn_autocorr_vs_zero"] = {"n": len(ac_all), "test": {"p": None}}

    # One-sample tests for light and dark separately
    ac_l_all = np.array([r["turn_autocorr_light"] for r in usable if r.get("turn_autocorr_light") is not None])
    ac_d_all = np.array([r["turn_autocorr_dark"] for r in usable if r.get("turn_autocorr_dark") is not None])
    if len(ac_l_all) >= 6:
        try:
            w_l = sp_stats.wilcoxon(ac_l_all, alternative="two-sided")
            n_l = len(ac_l_all)
            r_l = 1.0 - (2.0 * w_l.statistic) / (n_l * (n_l + 1) / 2.0)
            stats["figure4"]["turn_autocorr_light_vs_zero"] = {
                "n": n_l, "mean": float(np.mean(ac_l_all)),
                "test": {"stat": float(w_l.statistic), "p": float(w_l.pvalue), "r": float(r_l), "n": n_l},
            }
        except Exception:
            stats["figure4"]["turn_autocorr_light_vs_zero"] = {"n": len(ac_l_all), "test": {"p": None}}
    if len(ac_d_all) >= 6:
        try:
            w_d = sp_stats.wilcoxon(ac_d_all, alternative="two-sided")
            n_d = len(ac_d_all)
            r_d = 1.0 - (2.0 * w_d.statistic) / (n_d * (n_d + 1) / 2.0)
            stats["figure4"]["turn_autocorr_dark_vs_zero"] = {
                "n": n_d, "mean": float(np.mean(ac_d_all)),
                "test": {"stat": float(w_d.statistic), "p": float(w_d.pvalue), "r": float(r_d), "n": n_d},
            }
        except Exception:
            stats["figure4"]["turn_autocorr_dark_vs_zero"] = {"n": len(ac_d_all), "test": {"p": None}}

    # Back-tracking rate (4D)
    br_l, br_d = _extract_paired(usable, "back_rate_light", "back_rate_dark")
    stats["figure4"]["back_rate_light_vs_dark"] = {
        "n": len(br_l),
        "mean_light": float(np.mean(br_l)) if len(br_l) > 0 else None,
        "mean_dark": float(np.mean(br_d)) if len(br_d) > 0 else None,
        "test": wilcoxon_test(br_l, br_d),
    }
    print(f"Back-tracking rate L vs D: {np.mean(br_l):.3f} vs {np.mean(br_d):.3f}")

    # Per-junction binomial tests (4A)
    # Pool turns across all sessions for each junction
    junction_stats = {}
    for junc in MAZE.junctions:
        jkey = str(junc)
        total_left = 0
        total_right = 0
        for r in usable:
            pj = r.get("per_junction_light", {}).get(jkey, {})
            total_left += pj.get("left", 0)
            total_right += pj.get("right", 0)
            pj_d = r.get("per_junction_dark", {}).get(jkey, {})
            total_left += pj_d.get("left", 0)
            total_right += pj_d.get("right", 0)

        total = total_left + total_right
        if total >= 5:
            binom_p = float(sp_stats.binomtest(total_left, total, 0.5).pvalue)
            left_frac = total_left / total
        else:
            binom_p = None
            left_frac = None
        junction_stats[jkey] = {
            "total_left": total_left,
            "total_right": total_right,
            "total": total,
            "left_frac": left_frac,
            "binomial_p": binom_p,
        }

    # Holm-Bonferroni across 7 junctions
    junc_pvals = [junction_stats[str(j)].get("binomial_p") for j in MAZE.junctions]
    junc_pvals_clean = [p if p is not None else 1.0 for p in junc_pvals]
    junc_adjusted = holm_bonferroni(junc_pvals_clean)
    for i, j in enumerate(MAZE.junctions):
        junction_stats[str(j)]["binomial_p_adjusted"] = junc_adjusted[i]

    stats["figure4"]["per_junction_bias"] = junction_stats

    # Holm-Bonferroni for Figure 4 (4 main tests)
    fig4_pvals = [
        stats["figure4"]["left_frac_light_vs_dark"]["test"].get("p"),
        stats["figure4"].get("turn_autocorr_vs_zero", {}).get("test", {}).get("p"),
        stats["figure4"]["turn_autocorr_light_vs_dark"].get("test_light_vs_dark", {}).get("p"),
        stats["figure4"]["back_rate_light_vs_dark"]["test"].get("p"),
    ]
    fig4_pvals_clean = [p if p is not None else 1.0 for p in fig4_pvals]
    fig4_adjusted = holm_bonferroni(fig4_pvals_clean)
    stats["figure4"]["holm_bonferroni_adjusted_p"] = {
        "left_frac": fig4_adjusted[0],
        "autocorr_vs_zero": fig4_adjusted[1],
        "autocorr_light_vs_dark": fig4_adjusted[2],
        "back_rate": fig4_adjusted[3],
    }

    # ---- Figure 5: HD and AHV ----
    stats["figure5"] = {}

    hd_l, hd_d = _extract_paired(usable, "hd_mrl_light", "hd_mrl_dark")
    stats["figure5"]["hd_mrl_light_vs_dark"] = {
        "n": len(hd_l),
        "mean_light": float(np.mean(hd_l)) if len(hd_l) > 0 else None,
        "mean_dark": float(np.mean(hd_d)) if len(hd_d) > 0 else None,
        "test": wilcoxon_test(hd_l, hd_d),
    }

    ahv_l, ahv_d = _extract_paired(usable, "median_ahv_light", "median_ahv_dark")
    stats["figure5"]["ahv_light_vs_dark"] = {
        "n": len(ahv_l),
        "mean_light": float(np.mean(ahv_l)) if len(ahv_l) > 0 else None,
        "mean_dark": float(np.mean(ahv_d)) if len(ahv_d) > 0 else None,
        "test": wilcoxon_test(ahv_l, ahv_d),
    }

    # Holm-Bonferroni for Figure 5 (2 tests)
    fig5_pvals = [
        stats["figure5"]["hd_mrl_light_vs_dark"]["test"].get("p"),
        stats["figure5"]["ahv_light_vs_dark"]["test"].get("p"),
    ]
    fig5_pvals_clean = [p if p is not None else 1.0 for p in fig5_pvals]
    fig5_adjusted = holm_bonferroni(fig5_pvals_clean)
    stats["figure5"]["holm_bonferroni_adjusted_p"] = {
        "hd_mrl": fig5_adjusted[0],
        "ahv": fig5_adjusted[1],
    }

    # ---- Figure 6: Speed at maze locations ----
    stats["figure6"] = {}

    spd_j = np.array([r["speed_junction"] for r in usable if r.get("speed_junction") is not None])
    spd_c = np.array([r["speed_corridor"] for r in usable if r.get("speed_corridor") is not None])
    spd_de = np.array([r["speed_dead_end"] for r in usable if r.get("speed_dead_end") is not None])
    # Align — only sessions with all 3
    aligned = [(r["speed_junction"], r["speed_corridor"], r["speed_dead_end"])
               for r in usable
               if r.get("speed_junction") is not None
               and r.get("speed_corridor") is not None
               and r.get("speed_dead_end") is not None]
    if aligned:
        sj, sc, sde = zip(*aligned)
        sj, sc, sde = np.array(sj), np.array(sc), np.array(sde)
        stats["figure6"]["speed_by_node_type"] = {
            "mean_junction": float(np.mean(sj)),
            "mean_corridor": float(np.mean(sc)),
            "mean_dead_end": float(np.mean(sde)),
            "friedman": friedman_test(sj, sc, sde),
            "posthoc_junc_vs_corr": wilcoxon_test(sj, sc),
            "posthoc_junc_vs_de": wilcoxon_test(sj, sde),
            "posthoc_corr_vs_de": wilcoxon_test(sc, sde),
        }
        # Holm-Bonferroni for post-hoc
        ph_pvals = [
            stats["figure6"]["speed_by_node_type"]["posthoc_junc_vs_corr"].get("p"),
            stats["figure6"]["speed_by_node_type"]["posthoc_junc_vs_de"].get("p"),
            stats["figure6"]["speed_by_node_type"]["posthoc_corr_vs_de"].get("p"),
        ]
        ph_pvals_clean = [p if p is not None else 1.0 for p in ph_pvals]
        ph_adjusted = holm_bonferroni(ph_pvals_clean)
        stats["figure6"]["speed_by_node_type"]["posthoc_adjusted_p"] = {
            "junc_vs_corr": ph_adjusted[0],
            "junc_vs_de": ph_adjusted[1],
            "corr_vs_de": ph_adjusted[2],
        }

    # Junction approach speed
    aligned_j = [(r["mean_pre_junction_speed"], r["mean_at_junction_speed"])
                  for r in usable
                  if r.get("mean_pre_junction_speed") is not None
                  and r.get("mean_at_junction_speed") is not None
                  and np.isfinite(r["mean_pre_junction_speed"])
                  and np.isfinite(r["mean_at_junction_speed"])]
    if aligned_j:
        pj_arr, aj_arr = zip(*aligned_j)
        pj_arr, aj_arr = np.array(pj_arr), np.array(aj_arr)
        stats["figure6"]["junction_approach"] = {
            "mean_pre": float(np.mean(pj_arr)),
            "mean_at": float(np.mean(aj_arr)),
            "test_pre_vs_at": wilcoxon_test(pj_arr, aj_arr),
        }

    # ---- Supp S1: Transition entropy ----
    stats["supp_s1"] = {}
    te_l, te_d = _extract_paired(usable, "transition_entropy_light", "transition_entropy_dark")
    stats["supp_s1"]["transition_entropy_light_vs_dark"] = {
        "n": len(te_l),
        "mean_light": float(np.mean(te_l)) if len(te_l) > 0 else None,
        "mean_dark": float(np.mean(te_d)) if len(te_d) > 0 else None,
        "test": wilcoxon_test(te_l, te_d),
    }
    print(f"\nTransition entropy L vs D: {np.mean(te_l):.3f} vs {np.mean(te_d):.3f}")

    # Markov order comparison
    dbic = np.array([r["markov_delta_bic"] for r in usable if r.get("markov_delta_bic") is not None])
    if len(dbic) >= 6:
        try:
            w_bic = sp_stats.wilcoxon(dbic, alternative="greater")  # test delta_BIC > 0
            n_bic = len(dbic)
            r_bic = 1.0 - (2.0 * w_bic.statistic) / (n_bic * (n_bic + 1) / 2.0)
            stats["supp_s1"]["markov_order"] = {
                "n": n_bic,
                "mean_delta_bic": float(np.mean(dbic)),
                "median_delta_bic": float(np.median(dbic)),
                "n_prefer_2nd_order": int(np.sum(dbic > 0)),
                "test": {"stat": float(w_bic.statistic), "p": float(w_bic.pvalue), "r": float(r_bic), "n": n_bic},
            }
        except Exception:
            stats["supp_s1"]["markov_order"] = {"n": len(dbic), "test": {"p": None}}

    # ---- Supp S2: Genotype comparison (exploratory) ----
    stats["supp_s2"] = {}

    # Collapse to animal means for between-genotype comparisons
    animal_means = {}
    for r in usable:
        aid = r["animal_id"]
        if aid not in animal_means:
            animal_means[aid] = {"celltype": r["celltype"], "values": []}
        animal_means[aid]["values"].append(r)

    def _animal_mean(key):
        penk_vals = []
        nonpenk_vals = []
        for aid, info in animal_means.items():
            vals = [v[key] for v in info["values"] if v.get(key) is not None]
            if vals:
                m = float(np.mean(vals))
                if info["celltype"] == "penk":
                    penk_vals.append(m)
                else:
                    nonpenk_vals.append(m)
        return np.array(penk_vals), np.array(nonpenk_vals)

    for metric, key in [
        ("median_speed", "mean_speed_all"),
        ("frac_active", "frac_active_all"),
        ("coverage", "coverage_frac"),
    ]:
        p_vals, np_vals = _animal_mean(key)
        stats["supp_s2"][metric] = {
            "n_penk": len(p_vals),
            "n_nonpenk": len(np_vals),
            "mean_penk": float(np.mean(p_vals)) if len(p_vals) > 0 else None,
            "mean_nonpenk": float(np.mean(np_vals)) if len(np_vals) > 0 else None,
            "test": mannwhitney_test(p_vals, np_vals),
        }

    # Left_frac by genotype
    def _animal_mean_from_dict(outer_key, inner_key):
        penk_vals = []
        nonpenk_vals = []
        for aid, info in animal_means.items():
            vals = []
            for v in info["values"]:
                d = v.get(outer_key, {})
                if isinstance(d, dict) and d.get(inner_key) is not None:
                    vals.append(d[inner_key])
            if vals:
                m = float(np.mean(vals))
                if info["celltype"] == "penk":
                    penk_vals.append(m)
                else:
                    nonpenk_vals.append(m)
        return np.array(penk_vals), np.array(nonpenk_vals)

    # Combine light and dark turn bias for overall left_frac per session
    lf_combined_penk = []
    lf_combined_nonpenk = []
    for aid, info in animal_means.items():
        sess_lfs = []
        for v in info["values"]:
            tbl = v.get("turn_bias_light", {})
            tbd = v.get("turn_bias_dark", {})
            total_l = (tbl.get("left", 0) or 0) + (tbd.get("left", 0) or 0)
            total_r = (tbl.get("right", 0) or 0) + (tbd.get("right", 0) or 0)
            if total_l + total_r > 0:
                sess_lfs.append(total_l / (total_l + total_r))
        if sess_lfs:
            m = float(np.mean(sess_lfs))
            if info["celltype"] == "penk":
                lf_combined_penk.append(m)
            else:
                lf_combined_nonpenk.append(m)

    stats["supp_s2"]["left_frac"] = {
        "n_penk": len(lf_combined_penk),
        "n_nonpenk": len(lf_combined_nonpenk),
        "mean_penk": float(np.mean(lf_combined_penk)) if lf_combined_penk else None,
        "mean_nonpenk": float(np.mean(lf_combined_nonpenk)) if lf_combined_nonpenk else None,
        "test": mannwhitney_test(np.array(lf_combined_penk), np.array(lf_combined_nonpenk)),
    }

    # Light-dark speed difference by genotype
    def _animal_light_dark_diff(key_light, key_dark):
        penk_diffs = []
        nonpenk_diffs = []
        for aid, info in animal_means.items():
            diffs = []
            for v in info["values"]:
                vl = v.get(key_light)
                vd = v.get(key_dark)
                if vl is not None and vd is not None:
                    diffs.append(vl - vd)
            if diffs:
                m = float(np.mean(diffs))
                if info["celltype"] == "penk":
                    penk_diffs.append(m)
                else:
                    nonpenk_diffs.append(m)
        return np.array(penk_diffs), np.array(nonpenk_diffs)

    p_diff, np_diff = _animal_light_dark_diff("median_speed_light", "median_speed_dark")
    stats["supp_s2"]["light_dark_speed_diff"] = {
        "n_penk": len(p_diff),
        "n_nonpenk": len(np_diff),
        "mean_penk": float(np.mean(p_diff)) if len(p_diff) > 0 else None,
        "mean_nonpenk": float(np.mean(np_diff)) if len(np_diff) > 0 else None,
        "test": mannwhitney_test(p_diff, np_diff),
    }

    # Holm-Bonferroni for Supp S2 (5 tests)
    s2_pvals = [
        stats["supp_s2"]["median_speed"]["test"].get("p"),
        stats["supp_s2"]["frac_active"]["test"].get("p"),
        stats["supp_s2"]["coverage"]["test"].get("p"),
        stats["supp_s2"]["left_frac"]["test"].get("p"),
        stats["supp_s2"]["light_dark_speed_diff"]["test"].get("p"),
    ]
    s2_pvals_clean = [p if p is not None else 1.0 for p in s2_pvals]
    s2_adjusted = holm_bonferroni(s2_pvals_clean)
    stats["supp_s2"]["holm_bonferroni_adjusted_p"] = {
        "median_speed": s2_adjusted[0],
        "frac_active": s2_adjusted[1],
        "coverage": s2_adjusted[2],
        "left_frac": s2_adjusted[3],
        "light_dark_speed_diff": s2_adjusted[4],
    }

    # ---- Robustness: primary-only sessions ----
    stats["robustness_primary_only"] = {}
    sp_l_p, sp_d_p = _extract_paired(primary_only, "median_speed_light", "median_speed_dark")
    stats["robustness_primary_only"]["speed_light_vs_dark"] = {
        "n": len(sp_l_p),
        "test": wilcoxon_test(sp_l_p, sp_d_p),
    }
    fa_l_p, fa_d_p = _extract_paired(primary_only, "frac_active_light", "frac_active_dark")
    stats["robustness_primary_only"]["frac_active_light_vs_dark"] = {
        "n": len(fa_l_p),
        "test": wilcoxon_test(fa_l_p, fa_d_p),
    }
    cov_l_p, cov_d_p = _extract_paired(primary_only, "mean_epoch_coverage_light", "mean_epoch_coverage_dark")
    stats["robustness_primary_only"]["epoch_coverage_light_vs_dark"] = {
        "n": len(cov_l_p),
        "test": wilcoxon_test(cov_l_p, cov_d_p),
    }

    # ===================================================================
    # Save results
    # ===================================================================

    output = {
        "per_session": all_results,
        "cross_session": stats,
    }

    # Serialize
    output_ser = _make_serializable(output)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_ser, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_JSON}")

    # ===================================================================
    # Generate summary markdown
    # ===================================================================
    _write_summary_markdown(stats, usable, output_ser)
    print(f"Summary saved to: {OUTPUT_MD}")


def _write_summary_markdown(stats, usable, output):
    """Write a summary markdown table of results."""

    def _fmt_test(test_dict, adjusted_p=None):
        """Format a test result."""
        if test_dict is None or test_dict.get("p") is None:
            return "N/A"
        p = test_dict["p"]
        r = test_dict.get("r")
        n = test_dict.get("n")
        parts = []
        if test_dict.get("stat") is not None:
            parts.append(f"W = {test_dict['stat']:.1f}")
        parts.append(f"p = {p:.4f}")
        if adjusted_p is not None:
            parts.append(f"p_adj = {adjusted_p:.4f}")
        if r is not None:
            parts.append(f"r = {r:.3f}")
        if n is not None:
            parts.append(f"N = {n}")
        return ", ".join(parts)

    lines = [
        "# Behavioural Analysis Results — Summary",
        "",
        f"Generated from {stats['dataset']['n_usable_sessions']} usable sessions "
        f"({stats['dataset']['n_animals']} animals: "
        f"{stats['dataset']['n_penk_animals']} Penk+, "
        f"{stats['dataset']['n_nonpenk_animals']} Penk-CamKII+).",
        "",
        "All tests are non-parametric. Effect sizes: rank-biserial r (Wilcoxon), "
        "Cliff's delta (Mann-Whitney). Multiple comparisons: Holm-Bonferroni within each figure.",
        "",
        "---",
        "",
        "## Session Summary",
        "",
        "| Metric | Mean +/- SD | Median | Range |",
        "| ------ | ----------- | ------ | ----- |",
    ]

    s = stats["summary"]
    lines.append(f"| Total distance (m) | {s['total_distance_m']['mean']:.1f} +/- {s['total_distance_m']['sd']:.1f} | {s['total_distance_m']['median']:.1f} | {s['total_distance_m']['min']:.1f} - {s['total_distance_m']['max']:.1f} |")
    lines.append(f"| Duration (s) | {s['duration_s']['mean']:.0f} +/- {s['duration_s']['sd']:.0f} | {s['duration_s']['median']:.0f} | |")
    lines.append(f"| Usable duration (s) | {s['usable_duration_s']['mean']:.0f} +/- {s['usable_duration_s']['sd']:.0f} | | |")
    lines.append(f"| Mean speed (cm/s) | {s['mean_speed_all']['mean']:.2f} +/- {s['mean_speed_all']['sd']:.2f} | | |")
    lines.append(f"| Fraction active | {s['frac_active_all']['mean']:.3f} +/- {s['frac_active_all']['sd']:.3f} | | |")
    lines.append(f"| Cells visited | {s['unique_cells_visited']['mean']:.1f} +/- {s['unique_cells_visited']['sd']:.1f} | {s['unique_cells_visited']['median']:.0f} | |")
    lines.append(f"| Coverage fraction | {s['coverage_frac']['mean']:.3f} +/- {s['coverage_frac']['sd']:.3f} | | |")

    # Figure 2
    lines.extend([
        "",
        "---",
        "",
        "## Figure 2: Speed and Locomotion — Light vs Dark",
        "",
        "| Comparison | Light | Dark | Test |",
        "| ---------- | ----- | ---- | ---- |",
    ])

    f2 = stats["figure2"]
    adj2 = f2.get("holm_bonferroni_adjusted_p", {})
    def _rv(v, fmt=".2f"):
        """Round a value for display."""
        if v is None or v == "N/A":
            return "N/A"
        return f"{v:{fmt}}"

    lines.append(f"| Median speed (cm/s) | {_rv(f2['speed_light_vs_dark'].get('median_light'))} | {_rv(f2['speed_light_vs_dark'].get('median_dark'))} | {_fmt_test(f2['speed_light_vs_dark']['test'], adj2.get('speed'))} |")
    lines.append(f"| Fraction active | {_rv(f2['frac_active_light_vs_dark'].get('mean_light'), '.3f')} | {_rv(f2['frac_active_light_vs_dark'].get('mean_dark'), '.3f')} | {_fmt_test(f2['frac_active_light_vs_dark']['test'], adj2.get('frac_active'))} |")
    lines.append(f"| Median immobility bout (s) | {_rv(f2['immobility_bout_duration'].get('median_light'))} | {_rv(f2['immobility_bout_duration'].get('median_dark'))} | {_fmt_test(f2['immobility_bout_duration']['test'], adj2.get('immobility_bout'))} |")
    if "epoch_distance_light_vs_dark" in f2:
        ed = f2["epoch_distance_light_vs_dark"]
        lines.append(f"| Distance per epoch (m, body) | {_rv(ed.get('median_light'))} | {_rv(ed.get('median_dark'))} | {_fmt_test(ed['test'], adj2.get('epoch_distance'))} |")

    # Figure 3
    lines.extend([
        "",
        "---",
        "",
        "## Figure 3: Maze Exploration — Light vs Dark (PRIORITY)",
        "",
        "| Comparison | Light | Dark | Test |",
        "| ---------- | ----- | ---- | ---- |",
    ])

    f3 = stats["figure3"]
    adj3 = f3.get("holm_bonferroni_adjusted_p", {})
    ec = f3["epoch_coverage_light_vs_dark"]
    lines.append(f"| Per-epoch coverage | {ec.get('mean_light', 'N/A'):.3f} | {ec.get('mean_dark', 'N/A'):.3f} | {_fmt_test(ec['test'], adj3.get('epoch_coverage'))} |" if ec.get('mean_light') is not None else "| Per-epoch coverage | N/A | N/A | N/A |")
    de = f3["dead_end_rate_light_vs_dark"]
    lines.append(f"| Dead-end rate (visits/min) | {de.get('mean_light', 'N/A'):.2f} | {de.get('mean_dark', 'N/A'):.2f} | {_fmt_test(de['test'], adj3.get('dead_end_rate'))} |" if de.get('mean_light') is not None else "| Dead-end rate | N/A | N/A | N/A |")
    ee = f3.get("exploration_efficiency_w5", {})
    lines.append(f"| Exploration efficiency (w=5) | {ee.get('mean_light', 'N/A'):.2f} | {ee.get('mean_dark', 'N/A'):.2f} | {_fmt_test(ee.get('test'), adj3.get('exploration_efficiency'))} |" if ee.get('mean_light') is not None else "| Exploration efficiency | N/A | N/A | N/A |")
    cpm = f3.get("cells_per_m_light_vs_dark", {})
    if cpm.get("mean_light") is not None:
        lines.append(f"| New cells per metre (body) | {cpm['mean_light']:.3f} | {cpm['mean_dark']:.3f} | {_fmt_test(cpm['test'], adj3.get('cells_per_m'))} |")
    rev = f3.get("revisit_light_vs_dark", {})
    if rev.get("mean_light") is not None:
        lines.append(f"| Revisitation (entries/cell) | {rev['mean_light']:.3f} | {rev['mean_dark']:.3f} | {_fmt_test(rev['test'], adj3.get('revisit'))} |")
    ent = f3.get("occupancy_entropy_light_vs_dark", {})
    if ent.get("mean_light") is not None:
        lines.append(f"| Occupancy entropy (bits) | {ent['mean_light']:.3f} | {ent['mean_dark']:.3f} | {_fmt_test(ent['test'], adj3.get('occupancy_entropy'))} |")
    lz = f3.get("lz_compressibility_light_vs_dark", {})
    if lz.get("mean_light") is not None:
        lines.append(f"| Normalised LZ complexity | {lz['mean_light']:.3f} | {lz['mean_dark']:.3f} | {_fmt_test(lz['test'], adj3.get('lz_compressibility'))} |")
    zc = f3.get("coverage_vs_null_light_vs_dark", {})
    if zc.get("mean_light") is not None:
        lines.append(f"| Coverage z vs random-walk null | {zc['mean_light']:.3f} | {zc['mean_dark']:.3f} | {_fmt_test(zc['test'], adj3.get('coverage_vs_null'))} |")

    # Figure 4
    lines.extend([
        "",
        "---",
        "",
        "## Figure 4: Turn Behaviour — Light vs Dark (PRIORITY)",
        "",
        "| Comparison | Light | Dark | Test |",
        "| ---------- | ----- | ---- | ---- |",
    ])

    f4 = stats["figure4"]
    adj4 = f4.get("holm_bonferroni_adjusted_p", {})
    lf = f4["left_frac_light_vs_dark"]
    lines.append(f"| Left fraction | {lf.get('mean_light', 'N/A'):.3f} | {lf.get('mean_dark', 'N/A'):.3f} | {_fmt_test(lf['test'], adj4.get('left_frac'))} |" if lf.get('mean_light') is not None else "| Left fraction | N/A | N/A | N/A |")
    br = f4["back_rate_light_vs_dark"]
    lines.append(f"| Back-tracking rate | {br.get('mean_light', 'N/A'):.3f} | {br.get('mean_dark', 'N/A'):.3f} | {_fmt_test(br['test'], adj4.get('back_rate'))} |" if br.get('mean_light') is not None else "| Back-tracking rate | N/A | N/A | N/A |")

    lines.extend([
        "",
        "### Sequential turn autocorrelation",
        "",
    ])

    ac_z = f4.get("turn_autocorr_vs_zero", {})
    if ac_z.get("test", {}).get("p") is not None:
        lines.append(f"- **Overall autocorrelation vs 0:** mean = {ac_z.get('mean', 'N/A'):.3f}, {_fmt_test(ac_z['test'], adj4.get('autocorr_vs_zero'))}")
    ac_ld = f4.get("turn_autocorr_light_vs_dark", {})
    if ac_ld.get("test_light_vs_dark", {}).get("p") is not None:
        lines.append(f"- **Light vs dark autocorrelation:** light mean = {ac_ld.get('mean_light', 'N/A'):.3f}, dark mean = {ac_ld.get('mean_dark', 'N/A'):.3f}, {_fmt_test(ac_ld['test_light_vs_dark'], adj4.get('autocorr_light_vs_dark'))}")

    # Per-junction
    lines.extend([
        "",
        "### Per-junction turn bias (pooled across sessions)",
        "",
        "| Junction | Left | Right | Total | Left frac | Binomial p | p_adj |",
        "| -------- | ---- | ----- | ----- | --------- | ---------- | ----- |",
    ])
    pjb = f4.get("per_junction_bias", {})
    for jkey in sorted(pjb.keys()):
        j = pjb[jkey]
        lf_val = f"{j['left_frac']:.3f}" if j.get("left_frac") is not None else "N/A"
        bp_val = f"{j['binomial_p']:.4f}" if j.get("binomial_p") is not None else "N/A"
        bp_adj = f"{j['binomial_p_adjusted']:.4f}" if j.get("binomial_p_adjusted") is not None else "N/A"
        lines.append(f"| {jkey} | {j['total_left']} | {j['total_right']} | {j['total']} | {lf_val} | {bp_val} | {bp_adj} |")

    # Figure 5
    lines.extend([
        "",
        "---",
        "",
        "## Figure 5: Head Direction and AHV",
        "",
        "| Comparison | Light | Dark | Test |",
        "| ---------- | ----- | ---- | ---- |",
    ])

    f5 = stats["figure5"]
    adj5 = f5.get("holm_bonferroni_adjusted_p", {})
    hd = f5["hd_mrl_light_vs_dark"]
    lines.append(f"| HD mean resultant length | {hd.get('mean_light', 'N/A'):.3f} | {hd.get('mean_dark', 'N/A'):.3f} | {_fmt_test(hd['test'], adj5.get('hd_mrl'))} |" if hd.get('mean_light') is not None else "| HD MRL | N/A | N/A | N/A |")
    ahv = f5["ahv_light_vs_dark"]
    lines.append(f"| Median |AHV| (deg/s) | {ahv.get('mean_light', 'N/A'):.1f} | {ahv.get('mean_dark', 'N/A'):.1f} | {_fmt_test(ahv['test'], adj5.get('ahv'))} |" if ahv.get('mean_light') is not None else "| Median |AHV| | N/A | N/A | N/A |")

    # Figure 6
    lines.extend([
        "",
        "---",
        "",
        "## Figure 6: Speed at Maze Locations",
        "",
    ])
    f6 = stats.get("figure6", {})
    snt = f6.get("speed_by_node_type", {})
    if snt:
        lines.extend([
            f"- **Mean speed (cm/s):** Junction = {snt.get('mean_junction', 'N/A'):.2f}, "
            f"Corridor = {snt.get('mean_corridor', 'N/A'):.2f}, "
            f"Dead end = {snt.get('mean_dead_end', 'N/A'):.2f}",
            f"- **Friedman test:** {_fmt_test(snt.get('friedman', {}))}",
        ])
        padj = snt.get("posthoc_adjusted_p", {})
        lines.append(f"- **Post-hoc (Holm-Bonferroni):** J vs C p_adj = {padj.get('junc_vs_corr', 'N/A')}, "
                     f"J vs DE p_adj = {padj.get('junc_vs_de', 'N/A')}, "
                     f"C vs DE p_adj = {padj.get('corr_vs_de', 'N/A')}")

    ja = f6.get("junction_approach", {})
    if ja:
        lines.append(f"- **Junction approach:** pre = {ja.get('mean_pre', 'N/A'):.2f}, "
                     f"at = {ja.get('mean_at', 'N/A'):.2f} cm/s, {_fmt_test(ja.get('test_pre_vs_at', {}))}")

    # Supp S1
    lines.extend([
        "",
        "---",
        "",
        "## Supplementary S1: Markov Models",
        "",
    ])
    s1 = stats.get("supp_s1", {})
    te = s1.get("transition_entropy_light_vs_dark", {})
    if te.get("mean_light") is not None:
        lines.append(f"- **Transition entropy:** Light = {te['mean_light']:.3f}, Dark = {te['mean_dark']:.3f}, {_fmt_test(te.get('test', {}))}")
    mo = s1.get("markov_order", {})
    if mo.get("mean_delta_bic") is not None:
        lines.append(f"- **Markov order:** mean delta_BIC = {mo['mean_delta_bic']:.1f}, "
                     f"{mo['n_prefer_2nd_order']}/{mo['n']} sessions prefer 2nd order, "
                     f"{_fmt_test(mo.get('test', {}))}")

    # Robustness
    lines.extend([
        "",
        "---",
        "",
        "## Robustness: Primary-Only Sessions",
        "",
        "| Comparison | N | p | r |",
        "| ---------- | - | - | - |",
    ])
    rob = stats.get("robustness_primary_only", {})
    for key, label in [
        ("speed_light_vs_dark", "Speed L vs D"),
        ("frac_active_light_vs_dark", "Frac active L vs D"),
        ("epoch_coverage_light_vs_dark", "Epoch coverage L vs D"),
    ]:
        t = rob.get(key, {}).get("test", {})
        n = rob.get(key, {}).get("n", "N/A")
        p = t.get("p")
        r = t.get("r")
        p_str = f"{p:.4f}" if p is not None else "N/A"
        r_str = f"{r:.3f}" if r is not None else "N/A"
        lines.append(f"| {label} | {n} | {p_str} | {r_str} |")

    # Per-session table
    lines.extend([
        "",
        "---",
        "",
        "## Per-Session Data",
        "",
        "| Exp | Animal | Type | Excl | Dur(s) | Dist(m) | Speed L | Speed D | Frac Act L | Frac Act D | Cells | Cov |",
        "| --- | ------ | ---- | ---- | ------ | ------- | ------- | ------- | ---------- | ---------- | ----- | --- |",
    ])

    for r in sorted(output["per_session"], key=lambda x: x.get("exp_index", 0)):
        if r.get("status") != "ok":
            lines.append(f"| {r.get('exp_index', '?')} | {r.get('animal_id', '?')} | {r.get('celltype', '?')} | {'Y' if r.get('exclude') else ''} | - | - | - | - | - | - | - | - |")
            continue
        def _f(v, fmt=".1f"):
            if v is None:
                return "-"
            return f"{v:{fmt}}"
        lines.append(
            f"| {r.get('exp_index', '?')} | {r['animal_id']} | {r['celltype']} | "
            f"{'Y' if r.get('exclude') else ''} | "
            f"{_f(r.get('duration_s'), '.0f')} | {_f(r.get('total_distance_m'))} | "
            f"{_f(r.get('median_speed_light'), '.2f')} | {_f(r.get('median_speed_dark'), '.2f')} | "
            f"{_f(r.get('frac_active_light'), '.3f')} | {_f(r.get('frac_active_dark'), '.3f')} | "
            f"{r.get('unique_cells_visited', '-')} | {_f(r.get('coverage_frac'), '.2f')} |"
        )

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
