#!/usr/bin/env python3
"""Occupancy- and kinematics-matched controls for the two significant new
neural hypotheses.

Two light/dark contrasts are significant at the cell level in Stage 6 /
``test_hypotheses.py``:

- **H6.1** (gain modulation, H-N12): HD tuning *peak* amplitude is higher in
  light than dark.
- **H7.3** (junction coding, H-N13): junction-restricted HD MVL is higher in
  dark than light.

Both are raw, unmatched contrasts — the same class of effect that the H-N3
dark-enhancement gauntlet showed to be driven by differential HD/occupancy
sampling rather than a coding change. This runner subjects them to the same
controls: subsample frames so the head-direction occupancy (A1) or the
speed x |AHV| distribution (A2) is matched between light and dark, recompute
the statistic with circular-shuffle debiasing, and test the light-vs-dark
difference at the session level (paired Wilcoxon, the proper experimental
unit) with cell-level as a secondary read.

The peak statistic is added to ``hm2p.analysis.matched_tuning`` so it flows
through the same matching + debiasing pipeline as MVL and Skaggs info.

Reuses the loaders, matched-session engine, and paired test from
``run_dark_hypotheses.py``.

Usage:
    python scripts/run_gain_junction_controls.py [--signal dff] [--smoke]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dark_hypotheses import (  # noqa: E402
    _matched_session,
    _session_summary,
    assemble_sessions,
    load_metadata,
    paired_test,
)

from hm2p.maze.discretize import discretize_position_fast  # noqa: E402
from hm2p.maze.neural import classify_frames_by_node_type  # noqa: E402
from hm2p.maze.topology import build_rose_maze  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gain-junction-controls")


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values; NaNs pass through."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    adj = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return adj.tolist()
    sub = p[idx]
    order = np.argsort(sub)
    ranked = sub[order]
    m = len(sub)
    adj_sorted = ranked * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    out = np.empty(m)
    out[order] = adj_sorted
    adj[idx] = out
    return adj.tolist()


def junction_frame_mask(arrays: dict) -> np.ndarray | None:
    """Boolean over frames: True where the animal is at a junction node.

    Uses the maze-registered coordinates (x_maze/y_maze) and the same
    node-type classification as the Stage 6 junction analysis. Off-maze /
    unclassified frames are False. Returns None when maze coordinates are
    absent.
    """
    if "x_maze" not in arrays or "y_maze" not in arrays:
        return None
    maze = build_rose_maze()
    cell_indices = discretize_position_fast(
        np.asarray(arrays["x_maze"]), np.asarray(arrays["y_maze"]), maze
    )
    masks = classify_frames_by_node_type(cell_indices, maze)
    return np.asarray(masks["junction"], dtype=bool)


def run_peak_control(sessions, match, n_boot, n_shuffles, seed):
    """H6.1 control: occupancy/kinematics-matched HD tuning PEAK, light vs dark."""
    light_summ, dark_summ, rows = [], [], []
    for s in sessions:
        arrays = s["arrays"]
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        res = _matched_session(
            arrays, match, n_boot, n_shuffles, rng, statistic="peak"
        )
        if res is None:
            continue
        sig_mask = s.get("hd_sig_mask")
        l, d, ncell = _session_summary(res["mvl_light"], res["mvl_dark"], sig_mask)
        if not (np.isfinite(l) and np.isfinite(d)):
            continue
        light_summ.append(l)
        dark_summ.append(d)
        rows.append(
            {
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
                "n_cells": ncell,
                "peak_light_matched": l,
                "peak_dark_matched": d,
                "n_matched_frames": res["n_matched"],
            }
        )
    return paired_test(light_summ, dark_summ, label=f"H6.1-peak-{match}"), pd.DataFrame(rows)


def run_junction_mvl_control(sessions, match, n_boot, n_shuffles, seed):
    """H7.3 control: occupancy/kinematics-matched junction-restricted MVL, light vs dark."""
    light_summ, dark_summ, rows = [], [], []
    skipped = 0
    for s in sessions:
        arrays = s["arrays"]
        jm = junction_frame_mask(arrays)
        if jm is None:
            skipped += 1
            continue
        n = arrays["moving"].shape[0]
        jm = jm[:n]
        ml = arrays["moving"] & arrays["light_on"] & jm
        md = arrays["moving"] & ~arrays["light_on"] & jm
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        res = _matched_session(
            arrays, match, n_boot, n_shuffles, rng,
            mask_light=ml, mask_dark=md, statistic="mvl",
        )
        if res is None:
            continue
        sig_mask = s.get("hd_sig_mask")
        l, d, ncell = _session_summary(res["mvl_light"], res["mvl_dark"], sig_mask)
        if not (np.isfinite(l) and np.isfinite(d)):
            continue
        light_summ.append(l)
        dark_summ.append(d)
        rows.append(
            {
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
                "n_cells": ncell,
                "mvl_junction_light_matched": l,
                "mvl_junction_dark_matched": d,
                "n_matched_frames": res["n_matched"],
            }
        )
    if skipped:
        log.warning("junction control: %d sessions skipped (no maze coords)", skipped)
    return paired_test(light_summ, dark_summ, label=f"H7.3-junction-mvl-{match}"), pd.DataFrame(rows)


def _fmt(test: dict) -> str:
    return (
        f"n={test['n']:>2}  median(dark-light)={test['median_diff']:+.4f}  "
        f"Wilcoxon p={test['p_value']:.4f}  rank-biserial={test['rank_biserial']:+.3f}  "
        f"dark>light in {test['n_dark_gt_light']}/{test['n']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", default="dff", choices=["dff", "events", "deconv"])
    parser.add_argument("--n-boot", type=int, default=30)
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="2 sessions, fast")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    limit = 2 if args.smoke else None
    n_boot = 5 if args.smoke else args.n_boot
    n_shuffles = 50 if args.smoke else args.n_shuffles

    animals, exps = load_metadata()
    sessions = assemble_sessions(animals, exps, args.signal, limit, need_analysis=True)
    log.info("Loaded %d sessions", len(sessions))

    results = {}
    # match="none" reproduces the raw contrast (self-check); occupancy=A1,
    # kinematics=A2 are the controls that go in the FDR family.
    for match in ("none", "occupancy", "kinematics"):
        log.info("=== H6.1 gain-peak control, match=%s ===", match)
        t6, df6 = run_peak_control(sessions, match, n_boot, n_shuffles, args.seed)
        log.info("  %s", _fmt(t6))
        log.info("=== H7.3 junction-MVL control, match=%s ===", match)
        t7, df7 = run_junction_mvl_control(sessions, match, n_boot, n_shuffles, args.seed)
        log.info("  %s", _fmt(t7))
        results[match] = {"H6.1": (t6, df6), "H7.3": (t7, df7)}

    # BH-FDR across the four matched confirmatory tests (occupancy+kinematics
    # for both hypotheses); match=none is the raw self-check, excluded.
    fam = []
    for match in ("occupancy", "kinematics"):
        for hid in ("H6.1", "H7.3"):
            fam.append((match, hid, results[match][hid][0]))
    fdr = bh_fdr([t["p_value"] for _, _, t in fam])

    print("\n" + "=" * 78)
    print(f"MATCHED CONTROLS — signal={args.signal}  "
          f"(n_boot={n_boot}, n_shuffles={n_shuffles})")
    print("=" * 78)
    print("\nRaw self-check (match=none — should reproduce the significant raw contrast):")
    for hid in ("H6.1", "H7.3"):
        t = results["none"][hid][0]
        print(f"  {hid} (raw): {_fmt(t)}")
    print("\nControls (FDR-corrected family):")
    for (match, hid, t), q in zip(fam, fdr):
        survives = "SURVIVES" if (np.isfinite(q) and q < 0.05) else "does NOT survive"
        print(f"  {hid} match={match:<11} {_fmt(t)}  FDR q={q:.4f}  [{survives}]")
    print("\nDirection note: H6.1 peak expected light>dark (negative dark-light);")
    print("H7.3 junction MVL expected dark>light (positive dark-light).")

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        for match in results:
            for hid in ("H6.1", "H7.3"):
                results[match][hid][1].to_csv(
                    args.outdir / f"{hid}_{match}.csv", index=False
                )
        log.info("per-session tables written to %s", args.outdir)


if __name__ == "__main__":
    main()
